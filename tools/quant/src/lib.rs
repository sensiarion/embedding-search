//! Shared NomicBert (CodeRankEmbed-family) candle/Metal encoder +
//! retrieval metrics, used by the `bench` binary. The `cast` binary is
//! architecture-agnostic and does not use this module.
//!
//! Reuse for another model: a NomicBert-arch safetensors model works
//! as-is (point `Enc::load` at its dir). A different architecture needs
//! its own `candle_transformers` model in `Enc::load`; everything else
//! (CSN loader, cosine/MRR) is generic.
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::nomic_bert::{Config as NomicConfig, NomicBertModel};
use tokenizers::Tokenizer;

/// CodeRankEmbed's query instruction; corpus/code side gets no prefix.
pub const QPREFIX: &str = "Represent this query for searching relevant code: ";
pub const MAXLEN: usize = 512;
pub const BATCH: usize = 32;

pub struct Enc {
    model: NomicBertModel,
    tok: Tokenizer,
    dev: Device,
    pub dtype: DType,
}

impl Enc {
    /// Load a NomicBert safetensors model dir (`model.safetensors`,
    /// `config.json`, `tokenizer.json`) at its native precision.
    pub fn load(dir: &str, dev: &Device) -> Enc {
        let st_path = format!("{dir}/model.safetensors");
        let cfg: NomicConfig =
            serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).unwrap()).unwrap();
        // SAFETY: immutable model file, not mutated for the mmap's life.
        let native = unsafe { MmapedSafetensors::new(&st_path) }
            .unwrap()
            .tensors()
            .first()
            .map(|(_, v)| v.dtype())
            .unwrap();
        let dtype = DType::try_from(native).unwrap();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&st_path], dtype, dev).unwrap() };
        let model = NomicBertModel::load(vb, &cfg).unwrap();
        let mut tok = Tokenizer::from_file(format!("{dir}/tokenizer.json")).unwrap();
        tok.with_truncation(Some(tokenizers::TruncationParams {
            max_length: MAXLEN,
            ..Default::default()
        }))
        .unwrap();
        Enc {
            model,
            tok,
            dev: dev.clone(),
            dtype,
        }
    }

    /// CLS-pooled, L2-normalized embeddings. Length-sorted sub-batching
    /// (same as the main crate) so a long row never pads short ones.
    pub fn embed(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::new(); texts.len()];
        let encs = self.tok.encode_batch(texts.to_vec(), true).unwrap();
        let mut order: Vec<usize> = (0..encs.len()).collect();
        order.sort_unstable_by_key(|&i| encs[i].get_ids().len());
        for win in order.chunks(BATCH) {
            let b = win.len();
            let seq = win
                .iter()
                .map(|&i| encs[i].get_ids().len())
                .max()
                .unwrap()
                .max(1);
            let mut ids = vec![0u32; b * seq];
            let mut mask = vec![0u8; b * seq];
            for (r, &i) in win.iter().enumerate() {
                let e = &encs[i];
                for (j, (&id, &m)) in e.get_ids().iter().zip(e.get_attention_mask()).enumerate() {
                    ids[r * seq + j] = id;
                    mask[r * seq + j] = m as u8;
                }
            }
            let ids = Tensor::from_vec(ids, (b, seq), &self.dev).unwrap();
            let mask = Tensor::from_vec(mask, (b, seq), &self.dev).unwrap();
            let hidden = self.model.forward(&ids, None, Some(&mask)).unwrap();
            let cls = hidden
                .narrow(1, 0, 1)
                .unwrap()
                .squeeze(1)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();
            let norm = cls.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap();
            let v = cls.broadcast_div(&norm).unwrap().to_vec2::<f32>().unwrap();
            for (k, &i) in win.iter().enumerate() {
                out[i] = v[k].clone();
            }
        }
        out
    }
}

/// Dot product of two L2-normalized vectors == cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// CodeSearchNet-style retrieval: query `i`'s positive is doc `i`,
/// every doc is a distractor. Returns `(MRR@10, Recall@1)`.
pub fn retrieval(q: &[Vec<f32>], d: &[Vec<f32>]) -> (f64, f64) {
    let n = q.len();
    let mut mrr = 0.0;
    let mut r1 = 0.0;
    for i in 0..n {
        let mut scored: Vec<(usize, f32)> = (0..n).map(|j| (j, dot(&q[i], &d[j]))).collect();
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let rank = scored.iter().position(|&(j, _)| j == i).unwrap() + 1;
        if rank <= 10 {
            mrr += 1.0 / rank as f64;
        }
        if rank == 1 {
            r1 += 1.0;
        }
    }
    (mrr / n as f64, r1 / n as f64)
}

/// Load `n` `{doc, code}` pairs from a prefetched CSN JSON array.
/// Queries get [`QPREFIX`]; docs are raw code.
pub fn load_csn(path: &str, n: usize) -> (Vec<String>, Vec<String>) {
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let arr = v.as_array().unwrap();
    let take = n.min(arr.len());
    let mut q = Vec::with_capacity(take);
    let mut d = Vec::with_capacity(take);
    for r in &arr[..take] {
        q.push(format!("{QPREFIX}{}", r["doc"].as_str().unwrap()));
        d.push(r["code"].as_str().unwrap().to_string());
    }
    (q, d)
}
