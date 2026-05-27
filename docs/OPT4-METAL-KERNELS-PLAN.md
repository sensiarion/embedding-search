# Opt 4 — Custom fused Metal kernels for Gemma3 backbone

**Status:** parked. Profile says backbone forward = 99% of wall time
(see `benchmarks/results/PERF-EXPERIMENTS.md`). This is the only
remaining lever that targets the actual bottleneck. Multi-day work
for predicted +5-15%, with real risk of zero.

Pick this back up when (a) we genuinely need the throughput and other
ideas exhausted, or (b) candle 0.11+ exposes Metal SDPA / MPSGraph
binding upstream, making this cheaper.

## Current architecture — `google/embeddinggemma-300m`

### Model config (canonical)

```
vocab:                 256 000          (multilingual SentencePiece)
hidden_size:           768
intermediate_size:     1152             (MLP inner; SwiGLU)
num_hidden_layers:     24
num_attention_heads:   3                (query heads)
num_key_value_heads:   1                (GQA 3:1)
head_dim:              256              (unusual — most models 64/128)
max_position_embed:    2048
sliding_window:        512
sliding_window_pattern: 6               (every 6th layer = full; rest sliding)
use_bidirectional_attention: true
rope_theta:            1_000_000        (global rope)
rope_local_base_freq:  10_000           (local rope, sliding layers)
rms_norm_eps:          1e-6
hidden_activation:     gelu_pytorch_tanh
```

Backbone ≈ 302 M params. ST head adds ~4.7 M:

```
1_Pooling:    mean over time (include_prompt=true, no special-token mask)
2_Dense:      Linear(768 → 3072, bias=false, identity)
3_Dense:      Linear(3072 → 768, bias=false, identity)
4_Normalize:  L2
```

### Per-layer compute (gemma3 decoder block, encoder-ified)

```
x_in
 ↓ input_layernorm                 (RmsNorm eps=1e-6, weight (768,))
 ↓ Attention                       (see below)
 ↓ post_attention_layernorm
 ↓ + x_in                          (residual)
 ↓ pre_feedforward_layernorm
 ↓ MLP (SwiGLU)                    (see below)
 ↓ post_feedforward_layernorm
 ↓ + (above_residual)              (residual)
x_out
```

Attention (GQA 3:1):

```
q = q_proj(x)     [b,t,768]  reshape [b,3,t,256]
k = k_proj(x)     [b,t,256]  reshape [b,1,t,256]
v = v_proj(x)    "
q = q_norm(q)     RmsNorm over head_dim
k = k_norm(k)
(q,k) = RoPE(q,k) per-layer table (local for sliding layers, global for full)
k = repeat_kv(k, groups=3)
v = repeat_kv(v, groups=3)
scores = q @ k.T * (1 / sqrt(256))
scores += attn_mask    [b,1,1,t] full / [b,1,t,t] sliding
probs = softmax(scores, dim=-1)
out = probs @ v          → reshape [b,t,768] → o_proj
```

MLP (SwiGLU):

```
down_proj( gelu_pytorch_tanh(gate_proj(x)) * up_proj(x) )
gate_proj, up_proj: 768 → 1152
down_proj:          1152 → 768
```

Layer kind: `(i + 1) % 6 != 0` → sliding (uses local rope + sliding
mask). Layers 5/11/17/23 are full attention.

### Our implementation (`crates/core/src/candle_gemma_embed.rs`)

Fork of `candle_transformers::models::gemma3` adapted for encoder use:

1. `Backbone::forward` returns full `[b, t, h]` (no `lm_head`).
2. Two `RotaryEmbedding` instances pre-built (local + global), assigned
   per layer by `(i+1) % sliding_window_pattern`.
3. `Layer` carries `sliding_window: Option<usize>`; backbone picks
   `sliding_mask` vs `full_mask` per layer.
4. `Attention::forward` bidirectional, no KV cache,
   `scale = 1/sqrt(head_dim)`.
5. `GemmaRmsNorm` upcasts to f32, applies `x * (1 + weight)` (Gemma's
   `weight + 1` convention).
6. Backbone runs **native f32 on Metal**. f16 cast broken per Google's
   model card warning (bench: MRR 0.45 → 0.12). Dense heads always f32.

### Per-forward dispatch budget (b=8, t=512)

Per layer:
- 4 outer RmsNorm (input, post_attn, pre_ff, post_ff)
- 2 inner RmsNorm (q_norm, k_norm) — per-head, narrower
- 7 Linear (q, k, v, o, gate, up, down)
- 1 softmax + 2 matmul in attention
- 2 elementwise (residual adds, SwiGLU mul, activation)

≈ **20 dispatches × 24 layers = ~480 GPU dispatches per backbone
forward**, plus rotary apply / reshape / transpose ops.

## Fusion targets (ranked by ROI)

### T1 — Fused RmsNorm + Linear (matmul)

4 RmsNorm→Linear pairs per layer (most fusable: `input_norm → q/k/v`
all share input; `pre_ff_norm → gate/up` share input).

- Norm tiled in threadgroup memory; output streams into simdgroup
  matmul accumulator without a roundtrip to VRAM.
- Saves ~3 norm roundtrips per layer × 24 = 72.

**Effort:** 1 MSL kernel with norm prologue + tiled matmul. Reference:
Apple's MPS samples for tile-based GEMM.

### T2 — Fused SwiGLU MLP

`down(gelu(gate(x)) * up(x))` — 3 Linears + activation + elementwise
mul in one kernel.

- `gate` and `up` share input → tile-once-read-twice.
- Activation + mul + `down` fused.
- Saves ~3 dispatches per layer × 24 = 72.

**Effort:** complex tiled kernel — three matmuls in sequence with
shared inputs. Apple's MPSGraph has fusable matmul→matmul chains but
candle doesn't bind it.

### T3 — Fused attention score block (QK^T → scale → mask_add → softmax → V)

NOT flash-attention (verdicted slower at S=512 in
`FLASH_ATTENTION_VERDICT.md` on `flash-attention-attempt` branch).
Just collapse the 3 score-side ops into one kernel:

- score scale done in-register after qk matmul
- mask add fused
- softmax in same kernel
- AV matmul stays separate (no online normalization tile)

Saves 2-3 dispatches per layer × 24 = 48-72.

**Effort:** medium. Pattern similar to existing
`candle_metal_kernels::call_softmax`.

### T4 — Fused residual_add + RmsNorm

`x = (x_in + sublayer_out); x = next_norm(x)` → one kernel.

- 2 residuals per layer × 24 = 48 add dispatches saved
- Each followed by a norm — fuse the read into the norm's first pass

**Effort:** low. Simplest kernel. Best for de-risking the integration
path early.

## Integration path (candle-metal-kernels)

`candle-metal-kernels` crate provides:
- `Kernels` struct: compiled MSL libraries
- `call_*` functions: take command encoder + buffers + dispatch dims
- MSL sources compiled at crate build time

To add a new kernel:
1. Write MSL source in a new `.metal` file
2. Reference from `Kernels::load_library` map
3. Add `call_<name>` thin Rust wrapper that grabs pipeline state +
   dispatches
4. In `candle_gemma_embed.rs`, hook via `Tensor::custom_op*`:
   - Implement `CustomOp1/2/3` with `cpu_fwd` + `metal_fwd`
   - `metal_fwd` grabs `MetalStorage`, command encoder, calls
     `candle_metal_kernels::call_<name>`

**Decision: in-tree vs vendored fork.** Adding kernels to upstream
`candle-metal-kernels` means forking. Alternative: ship our own
`embedding-search-metal-kernels` crate or inline module — cleaner
upgrade path. Recommend the latter.

## Staged delivery

### Phase A — scaffolding (4-6 hours)

1. New module `crates/core/src/candle_gemma_kernels.rs` (or sibling
   workspace member). Builds MSL via runtime
   `Device::new_library_with_source` (no build.rs needed if we embed
   MSL as `include_str!`).
2. **First kernel: T4 (fused `residual_add + rmsnorm`).** Simplest —
   pure elementwise + reduction. Validates candle CustomOp plumbing
   end-to-end.
3. Unit test: `MetalStorage` round-trip matches CPU reference within
   1e-5.
4. Wire into `Layer::forward`; A/B bench on a synthetic batch.

**Decision gate:** if T4 alone shows ≥3% perf gain → continue. If 0%
→ abandon. Means we are NOT dispatch-overhead-bound; we are
matmul-compute-bound and fusions can't help.

### Phase B — bigger fusions (6-10 hours)

5. T1: fused `RmsNorm + Linear`. Norm in threadgroup memory, output
   into simdgroup matmul accumulator.
6. T2: SwiGLU MLP fuse. Three matmuls + activation + mul.

**Decision gate:** if Phase A + T1 ≥ +10% combined, ship; skip T2.

### Phase C — attention fuse (8-12 hours, optional)

7. T3: score+softmax fuse.

Skip if Phase A+B already ≥ +15%.

## Risks

1. **candle `MetalStorage` API access.** `buffer()` may be `pub(crate)`
   in 0.10.2. Verify before coding — may need `unsafe` shim or a
   minor candle fork.
2. **MSL compile path.** candle-metal-kernels uses runtime compilation
   from embedded source — replicable.
3. **Numerical drift.** Gemma's RmsNorm specifically does `x * (1 +
   w)` AND upcasts to f32. Fused kernel must match exactly or quality
   degrades silently. Reference-compare every kernel.
4. **Effort > predicted gain.** Profile shows backbone = 99% but does
   NOT decompose forward into per-op time. Possible 80% of forward is
   pure matmul compute (not dispatch overhead), in which case Phase A
   gates fail at 0% gain and we abandon. **Cheap to test, expensive
   to commit.**

## Predicted realistic outcome

- Phase A (T4 only): +1-3%
- Phase A + T1: +5-10%
- Phase A + T1 + T2: +8-15%
- All four: +10-20% (theoretical ceiling on this architecture)

Multi-day investment for a fraction of what Exp B (max_length=256)
delivers in 30 minutes (−42% sync at −7% MRR). Opt 4 buys quality
preservation; Exp B trades quality.

## Resume checklist

When picking this back up:

- [ ] Verify candle version (`Cargo.toml`) still 0.10.2 or compatible.
      API changes around `MetalStorage` would shift effort estimate.
- [ ] Re-run baseline bench. If sync_ms < 30 s on Gemma, Opt 4 has
      less headroom than today.
- [ ] Re-check Apple Silicon generation. M3+ has bf16 native + better
      matmul pipelines; some fusions may not pay back on M3+.
- [ ] Check whether candle 0.11+ exposes Metal SDPA / MPSGraph
      binding. If so, T3 becomes free — skip the kernel.
- [ ] Re-read `FLASH_ATTENTION_VERDICT.md` (commit 0ddc448) — same
      hard data on what doesn't work.

## Files touched in this plan

```
crates/core/src/candle_gemma_embed.rs   (Backbone, Layer, Attention, MLP)
crates/core/src/candle_gemma_kernels.rs (NEW — MSL + CustomOp wrappers)
crates/core/Cargo.toml                  (objc2-metal, metal feature)
benchmarks/results/PERF-EXPERIMENTS.md  (final row)
```
