# Plan: EmbeddingGemma on Metal via Candle fork

Current path is ONNX int8 + CoreML EP → silently CPU (dynamic seqlen
fragments the CoreML partitioner, same blocker as NomicBert in
this repo). This plan moves Gemma to the same candle Metal path
`CodeRankEmbed-f16` already uses, with bf16 (not fp16) to respect the
model card.

## Why a candle fork

- `candle-transformers::models::gemma3::Model` is implemented and
  works on Metal — but `forward()` returns causal-LM logits, not
  hidden states. No mean-pool, no dense head. Both are required for
  the EmbeddingGemma contract.
- `sentence-transformers-rs` and `glowrs` don't cover Gemma3 — no
  drop-in.
- `mlx-rs` has no Gemma3 layer (only mistral example). Would mean
  porting the full backbone — comparable effort to the fork, and we
  lose candle's tokenizer + registry + cache plumbing.
- CoreML static-shape ANE path is a strong #2 but requires owning a
  HF export upload pipeline + pins seqlen.

## Implementation sketch (~250 LOC)

New file: `crates/core/src/candle_gemma_embed.rs`, plus a `ModelArch`
case to dispatch to it.

1. **Copy `gemma3.rs` body in** (or vendor as submodule under
   `vendor/`). Drop `lm_head` from `Model::new`. Forward returns
   `[B, T, H]`.

2. **Attention-mask mean pool** (pattern from candle's `bert.rs`):
   ```rust
   let mask = attention_mask.unsqueeze(2)?.to_dtype(h.dtype())?;
   let sum = (h * &mask)?.sum(1)?;
   let n   = mask.sum(1)?.clamp(1e-9, f32::INFINITY)?;
   let pooled = (sum / n)?;
   ```

3. **Load 2_Dense head** (`2_Dense/model.safetensors`, ~9.4 MB,
   Linear 768→768 no bias):
   ```rust
   let dense_vb = vb.pp("2_Dense");
   let dense = candle_nn::linear_no_bias(768, 768, dense_vb.pp("linear"))?;
   let out = dense.forward(&pooled)?;
   ```

4. **L2 normalize** (`out / out.norm_keepdim()`).

5. **Dtype**: bf16 throughout. Apple Silicon Metal supports bf16;
   EmbeddingGemma activations support bf16 (not fp16 — model card).
   Load safetensors as bf16 via `VarBuilder::from_mmaped_safetensors`
   with `DType::BF16`.

6. **Wire into the registry.** In `crates/core/src/config.rs`
   `SUPPORTED_MODELS`, set `candle_repo: Some("google/embeddinggemma-300m")`
   on the existing entry. Keep `hf_repo` pointing at the onnx-community
   repo as a non-Mac / fallback path.

7. **ram_mb override.** Today `candle_bytes_per_param` keys off the
   `-f16` suffix on the candle repo. Generalize to recognize bf16 by
   spec, not by name (add a `candle_dtype: Option<DType>` field, or
   the simpler `bytes_per_param` override).

## Verification

- **Numerical**: a unit test that embeds 5 strings via the candle
  path and a Python `SentenceTransformer.encode(...)` reference, asserts
  cosine ≥ 0.999 element-wise. Mirror `tools/quant` pattern.
- **Quality regression**: re-run `cargo xtask eval --models
  google/embeddinggemma-300m --rerank` on CodeSearchNet 5000/200 and
  `cargo xtask golden --models google/embeddinggemma-300m`. Numbers
  must match the int8 ONNX baseline to within noise (~0.005 MRR).
- **Latency**: time 200 queries on a real repo. Expectation: ~2-4×
  faster than the current CPU-pinned ONNX path based on the
  CodeRankEmbed Metal-vs-int8 ratio (~1.8× there, Gemma is bigger
  so a wider gap is plausible).
- **RSS**: peak ≤ 700 MB on Metal (bf16 weights ≈ 616 MB + working
  set). The current int8 CPU path uses ~1.2 GB — Metal should win on
  RSS too because bf16 + no ORT runtime arena.

## Out of scope (explicitly deferred)

- **Flash Attention.** candle's `use_flash_attn` is CUDA-only; the
  Metal kernel from `philipturner/metal-flash-attention` is a separate
  Swift/Metal artifact, not wired into candle. At T=256-512 the
  FA win is noise anyway (crossover ≥ T=1024). Ignore.
- **QAT q4_0 / q8_0** (`google/embeddinggemma-300m-qat-q4_0-unquantized`).
  Google reports <1% MTEB degradation but candle's quantized loader
  expects GGUF; the QAT exports are safetensors. Revisit once a
  candle int8 weights loader exists for Gemma3.
- **ANE / static-shape CoreML export.** Strong #2 path but requires
  building + uploading a `*-mlpackage` to HF and pinning seqlen.
  Defer until the Metal-bf16 numbers are landed and we have a real
  reason to chase ANE residency.

## Cost estimate

- 1 day to write `candle_gemma_embed.rs` + tests.
- 0.5 day to wire `ModelArch` dispatch + variant tagging (so the
  index fingerprint flips between ONNX and candle paths, mirroring
  the existing `OnnxFiles::AccelCpu` pattern).
- 0.5 day to land + commit + CHANGELOG.

Total: ~2 days. Open question for ahead-of-time decision: do we want
the candle path on by default on Apple Silicon (mirroring CodeRank),
or opt-in via a config flag for the first release? Mirroring keeps
the contract uniform — recommend default-on with a clear escape hatch
(`[backend] candle = false` already exists in `BackendConfig`).
