//! Apple-Silicon Metal backend for the optional cross-encoder
//! re-rank stage: `cross-encoder/ettin-reranker-68m-v1`, a
//! sentence-transformers **CrossEncoder** over the Ettin
//! (ModernBERT-recipe) 68M encoder, via candle.
//!
//! Why: the int8 ONNX rerankers get no GPU EP on Apple Silicon (the
//! same QDQ-graph limitation as the embedder) — CPU-only and a
//! 50-candidate re-rank is pathologically slow there. candle on the
//! Metal GPU runs the f32 weights far faster, and Ettin-68M is the
//! smallest strong code-capable reranker (vs 278M `bge-reranker-base`).
//! Mirrors `candle_encoder` (device, length sub-batching).
//!
//! Loaded **as-is** at the export's native f32 precision (no cast):
//! ModernBERT is numerically unstable in f16 (activation overflow →
//! NaN), and at 68M the f32 weights are only ~0.27 GB.
//!
//! ettin is a sentence-transformers CrossEncoder, **not** a HF
//! `*ForSequenceClassification`: a bare `ModernBertModel` (root
//! `model.safetensors`, no `model.` key prefix) plus a 3-module head
//! saved in its own sub-dirs — CLS pooling, then
//! `Dense(512→512, GELU)` → `LayerNorm(512)` → `Dense(512→1)`, **no
//! softmax** (the raw logit is the relevance score). candle's
//! `ModernBertForSequenceClassification` is unusable here (it applies
//! `softmax` over `num_labels`, constant 1.0 for a single logit, and
//! its head/classifier loaders are private), so the encoder is loaded
//! via the public `ModernBert` and the head is rebuilt by hand.

use crate::embedder::load_tokenizer;
use crate::error::{Error, Result};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::{Encoding, Tokenizer};

/// `ettin-reranker-68m-v1` on Metal. One relevance logit per
/// `(query, passage)` pair — same contract as the ONNX reranker, so
/// `Reranker` can dispatch to either transparently.
pub(crate) struct CandleReranker {
    encoder: ModernBert,
    dense1: Linear,
    norm: LayerNorm,
    dense2: Linear,
    tok: Tokenizer,
    device: candle_core::Device,
}

/// Compute precision for the reranker on Metal. **f32, not f16.**
/// f16 was attempted (the rerank model is fixed, so a hardcoded dtype
/// would be fine and ~halve GPU bandwidth) but candle 0.10's
/// `modernbert` builds the additive attention mask hardcoded to F32
/// (`prepare_4d_attention_mask(.., DType::F32, ..)`) and adds it to
/// the f16 attention scores → `dtype mismatch in add`. The mask is
/// constructed *inside* candle, so f16 is impossible here without
/// vendoring candle. At 68M the f32 weights are only ~0.27 GB anyway.
const RERANK_DTYPE: DType = DType::F32;

/// One named tensor out of an already-loaded safetensors map, cast to
/// the reranker compute dtype so the hand-built head matches the
/// encoder output (mixed-dtype matmul would otherwise error).
fn take(map: &HashMap<String, Tensor>, name: &str, file: &str) -> Result<Tensor> {
    map.get(name)
        .ok_or_else(|| Error::Embed(format!("candle: {file} missing tensor {name}")))?
        .to_dtype(RERANK_DTYPE)
        .map_err(|e| Error::Embed(format!("candle: {file}/{name} cast: {e}")))
}

impl CandleReranker {
    /// Build from already-fetched repo files (root encoder + the three
    /// ST head modules). Any failure (no Metal, bad weights) is an
    /// `Err` so `Reranker::load` can fall back rather than hard-fail.
    pub fn build(
        encoder_st: &Path,
        dense1_st: &Path,
        norm_st: &Path,
        dense2_st: &Path,
        config_json: &[u8],
        tokenizer_json: &[u8],
        max_length: usize,
    ) -> Result<Self> {
        let device = crate::candle_encoder::metal_device()?;

        // ettin ships the new transformers-5.x ModernBERT config
        // (`rope_parameters` / `layer_types`); candle 0.10's `Config`
        // predates it and wants the flat `global_rope_theta` /
        // `local_rope_theta`. Backfill them from `rope_parameters`
        // (unknown extra keys are ignored by serde) so the otherwise
        // identical architecture deserializes.
        let mut raw: serde_json::Value = serde_json::from_slice(config_json)
            .map_err(|e| Error::Embed(format!("candle: ettin config: {e}")))?;
        if let Some(theta) = raw
            .get("rope_parameters")
            .and_then(|r| r.get("full_attention"))
            .and_then(|f| f.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
        {
            let obj = raw
                .as_object_mut()
                .ok_or_else(|| Error::Embed("candle: ettin config not an object".into()))?;
            obj.entry("global_rope_theta").or_insert(theta.into());
            obj.entry("local_rope_theta").or_insert(theta.into());
        }
        let cfg: ModernBertConfig = serde_json::from_value(raw)
            .map_err(|e| Error::Embed(format!("candle: ettin config: {e}")))?;

        // Encoder: ettin's root safetensors is a bare `ModernBertModel`
        // (`embeddings.*`, `layers.*`, `final_norm.*`). candle's
        // `ModernBert::load` hardcodes the `model.` submodule prefix
        // (it expects a `*ForSequenceClassification` layout), so every
        // tensor is re-keyed under `model.` before binding.
        let flat = candle_core::safetensors::load(encoder_st, &device)
            .map_err(|e| Error::Embed(format!("candle: ettin encoder: {e}")))?;
        let prefixed: HashMap<String, Tensor> =
            flat.into_iter().map(|(k, v)| (format!("model.{k}"), v)).collect();
        let vb = VarBuilder::from_tensors(prefixed, RERANK_DTYPE, &device);
        let encoder = ModernBert::load(vb, &cfg)
            .map_err(|e| Error::Embed(format!("candle: ettin encoder load: {e}")))?;

        // ST head: Dense(512→512, no bias) · GELU → LayerNorm(512) →
        // Dense(512→1, bias). No activation/softmax after the last
        // Dense — its raw output is the relevance logit.
        let h1 = candle_core::safetensors::load(dense1_st, &device)
            .map_err(|e| Error::Embed(format!("candle: 2_Dense: {e}")))?;
        let dense1 = Linear::new(take(&h1, "linear.weight", "2_Dense")?, None);
        let hn = candle_core::safetensors::load(norm_st, &device)
            .map_err(|e| Error::Embed(format!("candle: 3_LayerNorm: {e}")))?;
        let norm = LayerNorm::new(
            take(&hn, "norm.weight", "3_LayerNorm")?,
            take(&hn, "norm.bias", "3_LayerNorm")?,
            cfg.layer_norm_eps,
        );
        let h2 = candle_core::safetensors::load(dense2_st, &device)
            .map_err(|e| Error::Embed(format!("candle: 4_Dense: {e}")))?;
        let dense2 = Linear::new(
            take(&h2, "linear.weight", "4_Dense")?,
            Some(take(&h2, "linear.bias", "4_Dense")?),
        );

        let tok = load_tokenizer(tokenizer_json, max_length)?;
        Ok(Self {
            encoder,
            dense1,
            norm,
            dense2,
            tok,
            device,
        })
    }

    /// Relevance logit of `query` against each passage, in input
    /// order. Pairs are length sub-batched (attention is O(seq²) and
    /// every row pads to its sub-batch's longest), results scattered
    /// back so this stays transparent to the caller.
    pub fn score(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        // Borrowed pairs — no per-(query,passage) String allocation on
        // this hot path (mirrors the ONNX reranker).
        let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();
        let encs = self
            .tok
            .encode_batch(pairs, true)
            .map_err(|e| Error::Embed(format!("candle: rerank tokenize: {e}")))?;
        let mut order: Vec<usize> = (0..encs.len()).collect();
        order.sort_unstable_by_key(|&i| encs[i].get_ids().len());

        let mut ids_buf: Vec<u32> = Vec::new();
        let mut mask_buf: Vec<f32> = Vec::new();
        let mut out = vec![0.0f32; passages.len()];
        for window in order.chunks(crate::candle_encoder::candle_batch()) {
            let batch: Vec<&Encoding> = window.iter().map(|&i| &encs[i]).collect();
            for (&slot, logit) in window
                .iter()
                .zip(self.forward(&batch, &mut ids_buf, &mut mask_buf)?)
            {
                out[slot] = logit;
            }
        }
        Ok(out)
    }

    /// One forward on a length-homogeneous sub-batch → one logit per
    /// row. ModernBERT has no segment embeddings, so there are no
    /// `token_type_ids`: just `(input_ids, attention_mask)`.
    fn forward(
        &self,
        encs: &[&Encoding],
        ids_buf: &mut Vec<u32>,
        mask_buf: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
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
        mask_buf.resize(b * seq, 0.0);
        for (r, e) in encs.iter().enumerate() {
            for (j, (&id, &m)) in e.get_ids().iter().zip(e.get_attention_mask()).enumerate() {
                ids_buf[r * seq + j] = id;
                mask_buf[r * seq + j] = m as f32;
            }
        }
        let ids = Tensor::from_slice(&ids_buf[..b * seq], (b, seq), &self.device)?;
        // candle's modernbert resolves the 4-D mask itself via
        // `prepare_4d_attention_mask(.., DType::F32, ..)`; F32 in is
        // what it expects.
        let mask = Tensor::from_slice(&mask_buf[..b * seq], (b, seq), &self.device)?;
        let hidden = self.encoder.forward(&ids, &mask)?; // [b, seq, h]
        // ST CrossEncoder head: CLS pooling (token 0) → Dense·GELU →
        // LayerNorm → Dense → the single relevance logit.
        let cls = hidden.i((.., 0, ..))?.contiguous()?; // [b, h]
        let x = self.dense1.forward(&cls)?.gelu_erf()?;
        let x = self.norm.forward(&x)?;
        let logits = self.dense2.forward(&x)?; // [b, 1]
        Ok(logits
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?
            .into_iter()
            .map(|row| row.first().copied().unwrap_or(0.0))
            .collect())
    }
}
