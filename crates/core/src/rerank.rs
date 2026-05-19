//! Optional cross-encoder re-rank stage. A bi-encoder (the embedder)
//! scores query and passage independently; a cross-encoder reads the
//! `(query, passage)` pair jointly and is markedly more precise at
//! ordering an already-recalled candidate set — the dominant lever once
//! recall is adequate (SWE Context Bench). Gated behind
//! `[rerank] enabled` (default off): when disabled this module is never
//! constructed, so default users pay no download, memory or latency.
//!
//! Reuses the embedder's fixed-shape ORT discipline (`providers`,
//! `seq_buckets`, pinned batch width) so the CoreML/CUDA partitioner
//! sees a tiny bounded set of input shapes instead of one per
//! (batch × seq) — the same memory-bounding fix as `OnnxEncoder`.
#![cfg(not(feature = "bench-stub"))]

use crate::config::{Config, Precision};
use crate::embedder::{build_onnx_session, fill_fixed_batch, load_user_defined, seq_buckets};
use crate::error::{Error, Result};
use ort::{session::Session, value::Tensor};
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// ONNX cross-encoder reranker: tokenizes each `(query, passage)` pair,
/// runs one fixed-shape `[batch, seq]` session call, and reads the
/// single relevance logit (`*ForSequenceClassification`, `num_labels =
/// 1`). Higher logit = more relevant; only the ordering is used, so no
/// sigmoid is applied.
pub struct Reranker {
    session: Mutex<Session>,
    tok: Tokenizer,
    pad_id: i64,
    need_type_ids: bool,
    buckets: Vec<usize>,
    /// Fixed batch width every run is padded to (with filler pairs),
    /// mirroring `OnnxEncoder` so the EP compiles `buckets.len()`
    /// graphs total, not one per partial batch.
    batch: usize,
    /// How many top fused candidates the caller should re-score.
    top_n: usize,
}

impl Reranker {
    /// Download (cached) the configured cross-encoder and build the
    /// session. Only called when `[rerank] enabled` — the LM-head /
    /// KV-cache guards run inside `load_user_defined`.
    pub fn load(cfg: &Config) -> Result<Self> {
        let cache_dir = cfg.model_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)?;
        // Pinned to the int8 export: a reranker runs alongside the
        // embedder, so its footprint must stay small (combined RSS is
        // bounded by `[sync] max_rss_mb`).
        let udm = load_user_defined(
            &cfg.rerank.model,
            Precision::Int8,
            &cache_dir,
            Some("onnx/model_quantized.onnx"),
        )?;

        // Deliberately shares the embedder's `max_length`: the
        // reranker truncates each (query, passage) pair to the same
        // token budget so one knob bounds BOTH models' sequence memory
        // (they run in the same process). `build_onnx_session`'s
        // default truncation strategy trims the longer side first,
        // keeping the short query intact.
        let max_length = cfg.model.max_length.max(1);
        let (session, tok, pad_id, need_type_ids) =
            build_onnx_session(udm, &cfg.backend, false, max_length, &cache_dir)?;

        Ok(Self {
            session: Mutex::new(session),
            tok,
            pad_id,
            need_type_ids,
            buckets: seq_buckets(max_length),
            batch: cfg.embed_batch().max(1),
            top_n: cfg.rerank.top_n.max(1),
        })
    }

    /// How many top fused candidates `run` should hand to `score`.
    pub fn top_n(&self) -> usize {
        self.top_n
    }

    /// Relevance logit of `query` against each passage, in input order.
    /// One fixed-shape ORT call per `self.batch` slice.
    pub fn score(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        if passages.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(passages.len());
        for chunk in passages.chunks(self.batch) {
            out.extend(self.run_batch(query, chunk)?);
        }
        Ok(out)
    }

    fn run_batch(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>> {
        let real = passages.len();
        let pairs: Vec<(&str, &str)> = passages.iter().map(|p| (query, *p)).collect();
        let encs = self
            .tok
            .encode_batch(pairs, true)
            .map_err(|e| Error::Embed(format!("reranker tokenize: {e}")))?;
        let bsz = self.batch;
        let (ids, mask, seq) = fill_fixed_batch(&encs, self.pad_id, bsz, &self.buckets);

        let shape = vec![bsz as i64, seq as i64];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), ids))
                .map_err(|e| Error::Embed(e.to_string()))?,
            "attention_mask" => Tensor::from_array((shape.clone(), mask))
                .map_err(|e| Error::Embed(e.to_string()))?,
        ];
        if self.need_type_ids {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, vec![0i64; bsz * seq]))
                    .map_err(|e| Error::Embed(e.to_string()))?
                    .into(),
            ));
        }

        let (cols, data) = {
            let mut sess = self
                .session
                .lock()
                .map_err(|_| Error::Embed("reranker session lock poisoned".into()))?;
            let outputs = sess.run(inputs).map_err(|e| Error::Embed(e.to_string()))?;
            // A `*ForSequenceClassification` reranker has one float
            // output, `logits` [batch, num_labels] (num_labels = 1).
            let mut found: Option<(usize, Vec<f32>)> = None;
            for (_name, val) in outputs.iter() {
                if let Ok((sh, d)) = val.try_extract_tensor::<f32>() {
                    if sh.len() == 2 {
                        found = Some((sh[1] as usize, d.to_vec()));
                        break;
                    }
                }
            }
            found.ok_or_else(|| {
                Error::Embed("reranker: model produced no [batch, n] logit tensor".into())
            })?
        };

        // Column 0 is the relevance logit; filler rows that only pinned
        // the batch axis are dropped.
        Ok((0..real).map(|r| data[r * cols]).collect())
    }
}
