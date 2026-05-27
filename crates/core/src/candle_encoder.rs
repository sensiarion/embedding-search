//! Apple-Silicon Metal backend for CodeRankEmbed (NomicBert), via
//! candle, loaded directly from the base repo's f32 safetensors.
//!
//! Why this exists: the ORT CoreML EP cannot accelerate this rotary
//! architecture (int8 QDQ falls back to CPU and is *slower*; MLProgram
//! miscompiles the rotary — see `embedder::accel_active`). candle on
//! the Metal GPU runs the same f32 weights ~1.8x faster than the int8
//! ONNX CPU path, with identical embeddings. Apple-Silicon only;
//! everything here is `cfg`-gated so other targets never see candle.

use crate::embedder::load_tokenizer;
use crate::error::{Error, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor, WithDType};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config as NomicConfig, NomicBertModel};
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

/// Length-bucketed forward driver shared by `CandleEncoder` and
/// `CandleReranker`. Both models pad each sub-batch row to the
/// window's longest, and both pay O(seq²) attention, so the input
/// indices are sorted by token length before being fed in
/// `candle_batch()`-sized windows; per-row results scatter back to
/// the caller's input order. Generic over the per-row result `R` (an
/// embedding `Vec<f32>` for the encoder, a single logit `f32` for the
/// reranker) so the only thing that differs at the call site is the
/// `forward` closure body (model + pooling/head).
pub(crate) fn length_batched<R: Default + Clone>(
    encs: &[Encoding],
    mut forward: impl FnMut(&[&Encoding]) -> Result<Vec<R>>,
) -> Result<Vec<R>> {
    let mut order: Vec<usize> = (0..encs.len()).collect();
    order.sort_unstable_by_key(|&i| encs[i].get_ids().len());
    let mut out = vec![R::default(); encs.len()];
    for window in order.chunks(candle_batch()) {
        let batch: Vec<&Encoding> = window.iter().map(|&i| &encs[i]).collect();
        for (&slot, item) in window.iter().zip(forward(&batch)?) {
            out[slot] = item;
        }
    }
    Ok(out)
}

/// Stage one length-homogeneous sub-batch into `(ids, mask)` candle
/// tensors of shape `[b, seq]`. The mask scalar `M` is parameterized
/// because the two models need different dtypes: NomicBert's
/// `where_cond` needs an integer mask (`u8`); ModernBERT's
/// `prepare_4d_attention_mask` adds the mask to f32 scores, so it
/// must arrive as `f32`. Host buffers are reused across windows
/// (resize+zero, never a per-window `vec!`).
pub(crate) fn pack_ids_mask<M>(
    encs: &[&Encoding],
    ids_buf: &mut Vec<u32>,
    mask_buf: &mut Vec<M>,
    mask_cvt: impl Fn(u32) -> M,
    device: &Device,
) -> Result<(Tensor, Tensor)>
where
    M: WithDType + Copy + Default,
{
    let b = encs.len();
    let seq = encs
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(1)
        .max(1);
    ids_buf.clear();
    ids_buf.resize(b * seq, 0);
    mask_buf.clear();
    mask_buf.resize(b * seq, M::default());
    for (r, e) in encs.iter().enumerate() {
        for (j, (&id, &m)) in e.get_ids().iter().zip(e.get_attention_mask()).enumerate() {
            ids_buf[r * seq + j] = id;
            mask_buf[r * seq + j] = mask_cvt(m);
        }
    }
    let ids = Tensor::from_slice(&ids_buf[..b * seq], (b, seq), device)?;
    let mask = Tensor::from_slice(&mask_buf[..b * seq], (b, seq), device)?;
    Ok((ids, mask))
}

/// candle tensor-op errors are uniform ("a GPU op failed") on the hot
/// path — one `From` lets `run()` use `?`. `build()` keeps explicit
/// per-step messages (which file/stage failed is worth diagnosing).
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Embed(format!("candle: {e}"))
    }
}

/// Inner GPU forward batch size — the length-bucket window. Decoupled
/// from the OUTER `[sync] embed_batch_size` / `rec_batch` which feeds
/// `embed_documents` (that one bounds the ONNX CoreML/CUDA per-shape
/// graph cache; candle has no such constraint). Forward attention is
/// `O(b · h · t²)` per layer; with seq=512 and a transformer of this
/// class (~300 M f32 backbone), the unified-memory working set starts
/// paging past b=16 on a 16 GB M-series. Bench peak across Gemma3 f32
/// and CodeRankEmbed-f16 lands at b=8 — smaller windows lose Metal
/// occupancy and bigger ones thrash. Override via the env var
/// `EMBEDDING_SEARCH_CANDLE_BATCH` for hardware tuning.
pub(crate) fn candle_batch() -> usize {
    use std::sync::OnceLock;
    static BATCH: OnceLock<usize> = OnceLock::new();
    *BATCH.get_or_init(|| {
        std::env::var("EMBEDDING_SEARCH_CANDLE_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(8)
    })
}

/// Common surface every candle-Metal embedding model exposes — so the
/// `Backend::Candle` dispatch stays one boxed trait object regardless
/// of which architecture (NomicBert today, Gemma3 next) is loaded.
/// Object-safe by construction (only `&self` methods, no generics in
/// the signatures). The concrete embedder owns the tokenizer, model,
/// and device, applying per-architecture pooling / projection / L2
/// internally — callers only see f32 vectors out.
pub(crate) trait CandleEmbed: Send + Sync {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
    /// Index-identity suffix that flips when the underlying weight
    /// precision changes (so f16 ↔ f32 ↔ bf16 re-embed cleanly).
    fn variant(&self) -> &'static str;
}

impl CandleEmbed for CandleEncoder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        CandleEncoder::embed(self, texts)
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn variant(&self) -> &'static str {
        CandleEncoder::variant(self)
    }
}

/// CodeRankEmbed on Metal. CLS-pooled + L2-normalized; the query/doc
/// prefix is applied by the caller (`Contract`), exactly as for the
/// ONNX path, so embeddings are interchangeable across backends.
pub(crate) struct CandleEncoder {
    model: NomicBertModel,
    tok: Tokenizer,
    device: Device,
    /// Resolved weight precision — folded into the index identity so a
    /// repo precision change (f32 base ↔ f16 export) re-embeds.
    dtype: DType,
    pub dim: usize,
}

/// Metal device without aborting the process. candle's
/// `Device::new_metal` panics (it indexes an empty device list) when
/// no GPU is reachable — e.g. a headless/CI context. Convert that into
/// an `Err` so the caller can fall back to the ONNX path.
pub(crate) fn metal_device() -> Result<Device> {
    std::panic::catch_unwind(|| Device::new_metal(0))
        .map_err(|_| Error::Embed("candle: Metal device unavailable (headless?)".into()))?
        .map_err(|e| Error::Embed(format!("candle: Metal init: {e}")))
}

impl CandleEncoder {
    /// Build from already-fetched base-repo files. Any failure (no
    /// Metal, bad weights) is an `Err` so the embedder can fall back to
    /// the ONNX encoder rather than hard-fail.
    pub fn build(
        safetensors: &Path,
        config_json: &[u8],
        tokenizer_json: &[u8],
        max_length: usize,
    ) -> Result<Self> {
        let device = metal_device()?;
        let cfg: NomicConfig = serde_json::from_slice(config_json)
            .map_err(|e| Error::Embed(format!("candle: nomic config: {e}")))?;
        // Load weights at their native precision: an f16/bf16 export
        // runs half-precision (matmul-bandwidth win) for free, an f32
        // export stays f32 (no silent lossy downcast). Read the dtype
        // off the safetensors header rather than forcing one. The L2
        // norm in `run` is upcast to f32 regardless.
        // SAFETY: same contract as `from_mmaped_safetensors` below —
        // the file must not be mutated for the mmap's lifetime, which
        // holds for our immutable HF cache.
        let native = unsafe { MmapedSafetensors::new(safetensors) }
            .map_err(|e| Error::Embed(format!("candle: safetensors open: {e}")))?
            .tensors()
            .first()
            .map(|(_, v)| v.dtype())
            .ok_or_else(|| Error::Embed("candle: empty safetensors".into()))?;
        let dtype = DType::try_from(native).map_err(|e| {
            Error::Embed(format!(
                "candle: unsupported safetensors dtype {native:?}: {e}"
            ))
        })?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors], dtype, &device)
                .map_err(|e| Error::Embed(format!("candle: safetensors: {e}")))?
        };
        let model = NomicBertModel::load(vb, &cfg)
            .map_err(|e| Error::Embed(format!("candle: nomic load: {e}")))?;
        let tok = load_tokenizer(tokenizer_json, max_length)?;

        let mut enc = Self {
            model,
            tok,
            device,
            dtype,
            dim: 0,
        };
        enc.dim = enc
            .embed(&["probe"])?
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Embed("candle: empty probe".into()))?;
        Ok(enc)
    }

    /// Index-identity suffix: the real loaded weight dtype, so
    /// switching the candle repo's dtype (f32 base ↔ f16 export) busts
    /// the index. Every distinct dtype must yield a distinct tag — f16
    /// and bf16 are deliberately NOT merged (different embeddings ⇒
    /// must re-embed separately). Only F16/BF16/F32/F64 are reachable
    /// (the float dtypes `DType::try_from` yields for a safetensors
    /// model); `candle-other` is an exhaustiveness sentinel that is
    /// never taken in practice.
    pub fn variant(&self) -> &'static str {
        match self.dtype {
            DType::F16 => "candle-f16",
            DType::BF16 => "candle-bf16",
            DType::F32 => "candle-f32",
            DType::F64 => "candle-f64",
            _ => "candle-other",
        }
    }

    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encs = self
            .tok
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Embed(format!("candle: tokenize: {e}")))?;
        let mut ids_buf: Vec<u32> = Vec::new();
        let mut mask_buf: Vec<u8> = Vec::new();
        length_batched(&encs, |batch| {
            let (ids, mask) = pack_ids_mask(
                batch,
                &mut ids_buf,
                &mut mask_buf,
                |m| m as u8,
                &self.device,
            )?;
            let hidden = self.model.forward(&ids, None, Some(&mask))?; // [b, seq, n_embd]
                                                                       // CLS upcast to f32 so the L2 norm + readback stay
                                                                       // full-precision when the model loaded at f16/bf16.
            let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
            let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
            Ok(cls.broadcast_div(&norm)?.to_vec2::<f32>()?)
        })
    }
}
