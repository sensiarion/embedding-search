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
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config as NomicConfig, NomicBertModel};
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

/// candle tensor-op errors are uniform ("a GPU op failed") on the hot
/// path — one `From` lets `run()` use `?`. `build()` keeps explicit
/// per-step messages (which file/stage failed is worth diagnosing).
impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Embed(format!("candle: {e}"))
    }
}

/// GPU sub-batch. Decoupled from `[sync] embed_batch` / `rec_batch`
/// (which is small on purpose to bound the ONNX CoreML/CUDA per-shape
/// graph cache — a constraint candle does NOT have). A wider batch
/// amortizes tokenize/upload/readback and raises GPU occupancy;
/// working set is `BATCH * seq * n_embd * 4 B` ≈ 50 MB at seq 512 per
/// 32. Overridable via `EMBEDDING_SEARCH_CANDLE_BATCH` for tuning
/// without a rebuild (default 32; the GPU-occupancy sweet spot is
/// hardware-dependent).
fn candle_batch() -> usize {
    use std::sync::OnceLock;
    static BATCH: OnceLock<usize> = OnceLock::new();
    *BATCH.get_or_init(|| {
        std::env::var("EMBEDDING_SEARCH_CANDLE_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(32)
    })
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
fn metal_device() -> Result<Device> {
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
            Error::Embed(format!("candle: unsupported safetensors dtype {native:?}: {e}"))
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

    /// Index-identity suffix: the real loaded precision, so switching
    /// the candle repo's dtype (f32 base ↔ f16 export) busts the index.
    /// Every distinct weight precision must yield a distinct tag — f16
    /// and bf16 are deliberately NOT merged (different embeddings ⇒
    /// must re-embed separately), which is why this does not route
    /// through `Precision::label` (it has no bf16 and would collide
    /// the two). Only F16/BF16/F32/F64 are reachable (the float dtypes
    /// `DType::try_from` yields for a safetensors model); `candle-other`
    /// is an exhaustiveness sentinel that is never taken in practice
    /// and still cannot collide with a real precision tag.
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
        // Tokenize once, then sub-batch by ascending token length:
        // NomicBert attention is O(seq^2) and every row in a sub-batch
        // pads up to its longest, so a single long chunk mixed with
        // short ones makes the short ones pay the long seq. Sorting
        // groups similar lengths; results are scattered back to the
        // caller's order so this stays transparent to the collector.
        let encs = self
            .tok
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Embed(format!("candle: tokenize: {e}")))?;
        let mut order: Vec<usize> = (0..encs.len()).collect();
        order.sort_unstable_by_key(|&i| encs[i].get_ids().len());

        // ids/mask host staging reused across sub-batches (one alloc,
        // grown to the largest window, instead of two `vec!`s per
        // forward).
        let mut ids_buf: Vec<u32> = Vec::new();
        let mut mask_buf: Vec<u8> = Vec::new();
        let mut out = vec![Vec::new(); texts.len()];
        for window in order.chunks(candle_batch()) {
            let batch: Vec<&Encoding> = window.iter().map(|&i| &encs[i]).collect();
            for (&slot, vec) in window
                .iter()
                .zip(self.forward(&batch, &mut ids_buf, &mut mask_buf)?)
            {
                out[slot] = vec;
            }
        }
        Ok(out)
    }

    /// One forward on a length-homogeneous sub-batch (`seq` = its
    /// longest, already truncated to `max_length`). candle is not the
    /// ONNX CoreML EP, so a per-call dynamic shape is fine — no
    /// graph-per-shape blowup.
    fn forward(
        &self,
        encs: &[&Encoding],
        ids_buf: &mut Vec<u32>,
        mask_buf: &mut Vec<u8>,
    ) -> Result<Vec<Vec<f32>>> {
        let b = encs.len();
        let seq = encs
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .max(1);
        // resize(.., 0) reuses the existing capacity after the first
        // window and re-zeros it (pad positions must be 0).
        ids_buf.clear();
        ids_buf.resize(b * seq, 0);
        mask_buf.clear();
        mask_buf.resize(b * seq, 0);
        for (r, e) in encs.iter().enumerate() {
            for (j, (&id, &m)) in e.get_ids().iter().zip(e.get_attention_mask()).enumerate() {
                ids_buf[r * seq + j] = id;
                mask_buf[r * seq + j] = m as u8;
            }
        }
        let ids = Tensor::from_slice(&ids_buf[..b * seq], (b, seq), &self.device)?;
        // U8 mask: candle's NomicBert masking uses `where_cond`, whose
        // predicate must be an integer dtype, not f32.
        let mask = Tensor::from_slice(&mask_buf[..b * seq], (b, seq), &self.device)?;

        let hidden = self.model.forward(&ids, None, Some(&mask))?; // [b, seq, n_embd]
        // Upcast CLS to f32: L2 norm + readback stay full-precision
        // regardless of the model dtype (no-op when weights are f32).
        let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?.to_dtype(DType::F32)?;
        let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
        Ok(cls.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }
}
