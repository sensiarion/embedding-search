//! EmbeddingGemma 300M on Metal via candle. Architecturally a fork of
//! `candle-transformers::models::gemma3` adapted for encoder use:
//!
//!   1. **Bidirectional attention** — the EmbeddingGemma config sets
//!      `use_bidirectional_attention: true`. Every layer attends to
//!      every (non-padding) position; the causal triangle is gone.
//!   2. **Per-layer rope + sliding-window** — gemma3 alternates
//!      `sliding_attention` and `full_attention` layers (pattern: 1
//!      full every `sliding_window_pattern` layers; for the 300 M
//!      model that's `[s,s,s,s,s,f] × 4`). Sliding layers rotate q/k
//!      with `rope_local_base_freq` and restrict attention to a
//!      `sliding_window=512` band; full layers rotate with
//!      `rope_theta` and attend to everything (within padding).
//!   3. **Stateless forward** — no KV cache. Every batch is fresh.
//!   4. **Hidden states out** — `lm_head` is gone; the backbone
//!      returns `[B, T, H]` for the pooling head to consume.
//!
//! On top of that the sentence-transformers head (`modules.json`):
//!
//!   Backbone → `1_Pooling` (mean, include_prompt=true) →
//!   `2_Dense` Linear(768→3072, bias=false, identity) →
//!   `3_Dense` Linear(3072→768, bias=false, identity) →
//!   `4_Normalize` (L2).
//!
//! All four head safetensors keys are `linear.weight`; the dense
//! weights are stored as **f32** in the official repo, distinct from
//! the bf16/f32 backbone — they are loaded with a dedicated `f32`
//! VarBuilder so candle does not refuse them.
//!
//! Apple-Silicon only; everything here is `cfg`-gated through
//! `candle_backend`.

use crate::candle_encoder::{length_batched, metal_device, pack_ids_mask, CandleEmbed};
use crate::embedder::load_tokenizer;
use crate::error::{Error, Result};
use candle_core::safetensors::MmapedSafetensors;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{linear_no_bias, Linear, VarBuilder};
use candle_transformers::models::gemma3::Config as Gemma3Config;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Width of the sentence-transformers dense bottleneck. Fixed at
/// 3072 in `google/embeddinggemma-300m`'s `2_Dense/config.json`
/// (= 4 × hidden_size). NOT `cfg.intermediate_size` — that's the
/// gemma3 MLP inner size (1152), a different number.
const DENSE_INNER: usize = 3072;

// ---------- backbone (encoder-style fork of candle gemma3) ----------

/// Sin/cos tables for one rotary embedding. We instantiate TWO of
/// these per model — one with `sliding_window=None` (uses
/// `rope_theta` ≈ 1e6 for full-attention layers) and one with
/// `sliding_window=Some(cfg.sliding_window)` (uses
/// `rope_local_base_freq` ≈ 1e4 for sliding layers). Layers pick the
/// right one at construction.
#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        cfg: &Gemma3Config,
        dev: &Device,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let rope_freq = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor) -> Result<(Tensor, Tensor)> {
        let (_b, _h, seq_len, _d) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q, k))
    }
}

#[derive(Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl MLP {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(i, h, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        Ok((lhs * rhs)?.apply(&self.down_proj)?)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    /// Per-layer rotary — local-rope for sliding layers, global-rope
    /// for full layers. Wrong assignment here is a silent quality bug.
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(rotary_emb: Arc<RotaryEmbedding>, cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let groups = nh / nkv;
        let hd = cfg.head_dim;
        let bias = cfg.attention_bias;
        let lin = |i, o, b, name| {
            candle_nn::linear_b(i, o, b, vb.pp(name))
                .map_err(|e| Error::Embed(format!("candle: linear {name}: {e}")))
        };
        Ok(Self {
            q_proj: lin(h, nh * hd, bias, "q_proj")?,
            k_proj: lin(h, nkv * hd, bias, "k_proj")?,
            v_proj: lin(h, nkv * hd, bias, "v_proj")?,
            o_proj: lin(nh * hd, h, bias, "o_proj")?,
            q_norm: GemmaRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: GemmaRmsNorm::new(hd, cfg.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: nh,
            num_kv_heads: nkv,
            num_kv_groups: groups,
            head_dim: hd,
            rotary_emb,
        })
    }

    /// `attn_mask` is `[b, 1, 1, t]` (broadcast over heads + query
    /// positions) for a full-attention layer, or `[b, 1, t, t]`
    /// (per-query masking) for a sliding-window layer.
    fn forward(&self, xs: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let (q, k) = self.rotary_emb.apply(&q, &k)?;
        // GQA: tile kv to query head count.
        let k = candle_transformers::utils::repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = candle_transformers::utils::repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        // EmbeddingGemma uses `query_pre_attn_scalar` (256, same as
        // head_dim here) as the score divisor, not the classic
        // sqrt(head_dim). For head_dim=256 both happen to evaluate
        // to 1/sqrt(256) = 1/16, but make it explicit.
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = scores.broadcast_add(attn_mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;
        Ok(out
            .transpose(1, 2)?
            .reshape((b, t, ()))?
            .apply(&self.o_proj)?)
    }
}

#[derive(Debug, Clone)]
struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb
            .get(dim, "weight")
            .map_err(|e| Error::Embed(format!("candle: rmsnorm weight: {e}")))?;
        Ok(Self { weight, eps })
    }

    /// Exposed for the fused `residual_add + rmsnorm` Metal kernel in
    /// `candle_gemma_kernels`. The kernel reads `weight` directly
    /// into its `(1 + w) * x * inv_rms` step. Currently dead pending
    /// the packed-output redesign (see
    /// `docs/OPT4-METAL-KERNELS-PLAN.md` Phase A footnote).
    #[allow(dead_code)]
    fn weight(&self) -> &Tensor {
        &self.weight
    }

    #[allow(dead_code)]
    fn eps(&self) -> f64 {
        self.eps
    }
}

impl Module for GemmaRmsNorm {
    // Gemma RMSNorm: `x * (1 + weight)`, normalize in f32 for stability
    // when the activations live at bf16/f16.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let xdt = x.dtype();
        let inner = match xdt {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let h = x.dim(D::Minus1)?;
        let x32 = x.to_dtype(inner)?;
        let norm = (x32.sqr()?.sum_keepdim(D::Minus1)? / h as f64)?;
        let normed = x32.broadcast_div(&(norm + self.eps)?.sqrt()?)?;
        normed.to_dtype(xdt)?.broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct Layer {
    attn: Attention,
    mlp: MLP,
    input_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
    /// `Some(window)` for sliding layers, `None` for full. The
    /// backbone picks the matching mask tensor per forward.
    sliding_window: Option<usize>,
}

impl Layer {
    fn new(
        cfg: &Gemma3Config,
        vb: VarBuilder,
        rotary: Arc<RotaryEmbedding>,
        sliding_window: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(rotary, cfg, vb.pp("self_attn"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
            input_layernorm: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            pre_feedforward_layernorm: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("pre_feedforward_layernorm"),
            )?,
            post_feedforward_layernorm: GemmaRmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_feedforward_layernorm"),
            )?,
            sliding_window,
        })
    }

    fn forward(&self, xs: &Tensor, attn_mask: &Tensor) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.attn.forward(&xs, attn_mask)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = xs.clone();
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        Ok((residual + xs)?)
    }
}

// Backbone with no `model.` prefix (sentence-transformers package
// layout). Stateless forward, bidirectional attention with a per-layer
// full / sliding-window choice.
struct Backbone {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Layer>,
    norm: GemmaRmsNorm,
    hidden_size: usize,
    dtype: DType,
    device: Device,
}

impl Backbone {
    fn new(cfg: &Gemma3Config, vb: VarBuilder) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                .map_err(|e| Error::Embed(format!("candle: embed_tokens: {e}")))?;
        let dtype = vb.dtype();
        let dev = vb.device().clone();
        // Two shared rotary tables — local (sliding layers) + global
        // (full layers).
        let rot_global = Arc::new(RotaryEmbedding::new(dtype, cfg, &dev, None)?);
        let rot_local = Arc::new(RotaryEmbedding::new(
            dtype,
            cfg,
            &dev,
            Some(cfg.sliding_window),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            // Matches candle's gemma3 layer-kind rule:
            //   sliding iff (i + 1) % sliding_window_pattern != 0.
            // For the 300 M config (`sliding_window_pattern=6`,
            // 24 layers) this picks layers 5/11/17/23 as full —
            // identical to the published `layer_types`.
            let is_sliding = (i + 1) % cfg.sliding_window_pattern != 0;
            let (rot, sw) = if is_sliding {
                (rot_local.clone(), Some(cfg.sliding_window))
            } else {
                (rot_global.clone(), None)
            };
            layers.push(Layer::new(cfg, vb.pp(format!("layers.{i}")), rot, sw)?);
        }
        let norm = GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: cfg.hidden_size,
            dtype,
            device: dev,
        })
    }

    fn forward(&self, ids: &Tensor, full_mask: &Tensor, sliding_mask: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(ids)?;
        // gemma3 scales embeddings by sqrt(d) before the layer stack.
        let mut xs = (xs * (self.hidden_size as f64).sqrt())?;
        for layer in &self.layers {
            let mask = if layer.sliding_window.is_some() {
                sliding_mask
            } else {
                full_mask
            };
            xs = layer.forward(&xs, mask)?;
        }
        Ok(xs.apply(&self.norm)?)
    }
}

// ---------- mask helpers ----------

/// Full-attention bidirectional padding mask `[b, 1, 1, t]`: 0 on
/// real tokens, `-inf` on padding. Broadcasts over heads and over
/// query positions.
fn full_mask(
    mask_u32: &[u32],
    b: usize,
    t: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; b * t];
    for (i, &m) in mask_u32.iter().enumerate() {
        if m == 0 {
            data[i] = f32::NEG_INFINITY;
        }
    }
    let m = Tensor::from_vec(data, (b, 1, 1, t), device)?;
    Ok(m.to_dtype(dtype)?)
}

/// Sliding-window bidirectional padding mask `[b, 1, t, t]`: 0 when
/// query `i` is allowed to attend to key `j`, `-inf` otherwise.
/// Allowed iff `|i - j| < sliding_window` AND `j` is not padding.
fn sliding_mask_tensor(
    mask_u32: &[u32],
    b: usize,
    t: usize,
    sliding_window: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let mut data = vec![0f32; b * t * t];
    for row in 0..b {
        let pad_off = row * t;
        for i in 0..t {
            for j in 0..t {
                let blocked = mask_u32[pad_off + j] == 0 || i.abs_diff(j) >= sliding_window;
                if blocked {
                    data[row * t * t + i * t + j] = f32::NEG_INFINITY;
                }
            }
        }
    }
    let m = Tensor::from_vec(data, (b, 1, t, t), device)?;
    Ok(m.to_dtype(dtype)?)
}

// ---------- the public embedder ----------

pub(crate) struct CandleGemmaEncoder {
    backbone: Backbone,
    /// `Linear(768 → 3072, no bias)`. Identity activation.
    dense2: Linear,
    /// `Linear(3072 → 768, no bias)`. Identity activation.
    dense3: Linear,
    tok: Tokenizer,
    dim: usize,
    dtype: DType,
    sliding_window: usize,
}

impl CandleGemmaEncoder {
    /// Build the EmbeddingGemma encoder from three on-disk files:
    /// the backbone safetensors (top-level `model.safetensors` in the
    /// HF repo) and the two sentence-transformers dense heads
    /// (`2_Dense/model.safetensors` and `3_Dense/model.safetensors`).
    pub fn build(
        backbone_safetensors: &Path,
        dense2_safetensors: &Path,
        dense3_safetensors: &Path,
        config_json: &[u8],
        tokenizer_json: &[u8],
        max_length: usize,
    ) -> Result<Self> {
        let device = metal_device()?;
        // EmbeddingGemma's config.json names the layer-kind cycle
        // `_sliding_window_pattern` (leading underscore); candle's
        // `Gemma3Config` expects `sliding_window_pattern`. Patch the
        // JSON on the fly so the rest of the deserialize succeeds.
        let cfg: Gemma3Config = {
            let mut v: serde_json::Value = serde_json::from_slice(config_json)
                .map_err(|e| Error::Embed(format!("candle: gemma3 config parse: {e}")))?;
            if let Some(obj) = v.as_object_mut() {
                if let Some(p) = obj.remove("_sliding_window_pattern") {
                    obj.entry("sliding_window_pattern").or_insert(p);
                }
            }
            serde_json::from_value(v)
                .map_err(|e| Error::Embed(format!("candle: gemma3 config: {e}")))?
        };
        // Match the existing CandleEncoder behavior: read native dtype
        // off the safetensors header (so f32/bf16 exports run at their
        // packaged precision). The official Google export ships f32;
        // Metal handles it fine, just larger working set than bf16.
        // SAFETY: same immutable-mmap contract as `CandleEncoder::build`.
        let native = unsafe { MmapedSafetensors::new(backbone_safetensors) }
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
            VarBuilder::from_mmaped_safetensors(&[backbone_safetensors], dtype, &device)
                .map_err(|e| Error::Embed(format!("candle: safetensors: {e}")))?
        };
        let backbone = Backbone::new(&cfg, vb)?;
        // 2_Dense / 3_Dense — both Linear(no bias, Identity). Stored
        // as f32 in the official repo (verified via the safetensors
        // header), distinct from the backbone dtype; load with an f32
        // VarBuilder. Each head's tensor is named `linear.weight`.
        let dense_vb2 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dense2_safetensors], DType::F32, &device)
                .map_err(|e| Error::Embed(format!("candle: 2_Dense safetensors: {e}")))?
        };
        let dense_vb3 = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dense3_safetensors], DType::F32, &device)
                .map_err(|e| Error::Embed(format!("candle: 3_Dense safetensors: {e}")))?
        };
        let dense2 = linear_no_bias(cfg.hidden_size, DENSE_INNER, dense_vb2.pp("linear"))
            .map_err(|e| Error::Embed(format!("candle: 2_Dense load: {e}")))?;
        let dense3 = linear_no_bias(DENSE_INNER, cfg.hidden_size, dense_vb3.pp("linear"))
            .map_err(|e| Error::Embed(format!("candle: 3_Dense load: {e}")))?;
        let tok = load_tokenizer(tokenizer_json, max_length.min(cfg.max_position_embeddings))?;
        let mut enc = Self {
            backbone,
            dense2,
            dense3,
            tok,
            dim: 0,
            dtype,
            sliding_window: cfg.sliding_window,
        };
        // Probe inside catch_unwind: a candle Metal kernel that
        // panics on an unsupported shape would otherwise abort the
        // whole CLI/MCP server. Convert into a fallback-eligible Err.
        let probed =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| enc.embed(&["probe"])))
                .map_err(|_| Error::Embed("candle: probe panicked".into()))??;
        enc.dim = probed
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Embed("candle: empty probe".into()))?;
        Ok(enc)
    }
}

impl CandleEmbed for CandleGemmaEncoder {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encs = self
            .tok
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Embed(format!("candle: tokenize: {e}")))?;
        let mut ids_buf: Vec<u32> = Vec::new();
        let mut mask_buf: Vec<u32> = Vec::new();
        length_batched(&encs, |batch| {
            // pack_ids_mask writes `mask_buf` and an unused `_padded`
            // tensor. We only consume `mask_buf` (the f32 attn mask
            // is built directly from it).
            let (ids, _padded) = pack_ids_mask(
                batch,
                &mut ids_buf,
                &mut mask_buf,
                |m| m,
                &self.backbone.device,
            )?;
            let (b, t) = ids.dims2()?;
            let mask_slice = &mask_buf[..b * t];
            let full = full_mask(mask_slice, b, t, self.backbone.dtype, &self.backbone.device)?;
            let sliding = sliding_mask_tensor(
                mask_slice,
                b,
                t,
                self.sliding_window,
                self.backbone.dtype,
                &self.backbone.device,
            )?;
            let hidden = self.backbone.forward(&ids, &full, &sliding)?; // [b, t, h]

            // Attention-masked mean pool over the time axis. Multiply
            // hidden by the binary mask (broadcast to [b,t,1]), sum
            // along time, divide by per-row token count.
            //
            // Stage 2 micro-opt: pool in the BACKBONE's dtype, then
            // up-convert the pooled `[b, h]` once. Avoids materializing
            // a `[b, t, h]` f32 promotion (b·t·h = up to ~3 M elements
            // at b=8 t=512 h=768) when the backbone runs at half
            // precision. The intermediate mask host vec is built
            // directly with extend (no extra collect allocation).
            let mut mask_host: Vec<f32> = Vec::with_capacity(b * t);
            mask_host.extend(mask_slice.iter().map(|&m| m as f32));
            let mask_f = Tensor::from_vec(mask_host, (b, t, 1), &self.backbone.device)?
                .to_dtype(self.backbone.dtype)?;
            let summed = hidden.broadcast_mul(&mask_f)?.sum(1)?; // [b, h]
            let counts = mask_f.sum(1)?.clamp(1f32, f32::INFINITY)?; // [b, 1]
            let pooled = summed.broadcast_div(&counts)?.to_dtype(DType::F32)?; // [b, h] @ f32 for the dense head

            // 2_Dense → 3_Dense (both f32, Identity activation, no
            // bias). Composed effective `Linear(768→768)` via a 3072
            // intermediate — kept as two layers to mirror Matryoshka
            // truncation semantics in case we expose dim downscaling
            // later.
            let proj = self.dense3.forward(&self.dense2.forward(&pooled)?)?; // [b, 768]

            // L2 normalize along dim 1. Replace NaN/empty-row collapse
            // (all-padding row from a degenerate input) with a zero
            // vector rather than NaN, so the index never receives a
            // poisoned row.
            let norm = proj
                .sqr()?
                .sum_keepdim(1)?
                .sqrt()?
                .clamp(1e-12f32, f32::INFINITY)?;
            let out = proj.broadcast_div(&norm)?;
            // Stage 2 micro-opt: single batched readback (one Metal →
            // CPU sync) instead of a per-row `to_vec1` (b separate
            // syncs). The candle Metal storage layout for a
            // contiguous `[b, h]` tensor is row-major so `to_vec2`
            // hands back exactly the per-row vectors we want.
            let mut rows: Vec<Vec<f32>> = out.to_vec2::<f32>()?;
            for row in &mut rows {
                if row.iter().any(|v| v.is_nan()) {
                    row.iter_mut().for_each(|v| *v = 0.0);
                }
            }
            Ok(rows)
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn variant(&self) -> &'static str {
        match self.dtype {
            DType::F16 => "candle-gemma-f16",
            DType::BF16 => "candle-gemma-bf16",
            DType::F32 => "candle-gemma-f32",
            DType::F64 => "candle-gemma-f64",
            _ => "candle-gemma-other",
        }
    }
}
