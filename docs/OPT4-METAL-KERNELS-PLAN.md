# Opt 4 — Custom fused Metal kernels for Gemma3 backbone

**Status:** parked. Profile says backbone forward = 99% of wall time
(see `benchmarks/results/PERF-EXPERIMENTS.md`). Only remaining lever
on the actual bottleneck. Multi-day work for predicted +5-15%, real
risk of zero. Pick back up when (a) we need it and others exhausted,
or (b) candle 0.11+ exposes Metal SDPA / MPSGraph upstream.

## Arch — `google/embeddinggemma-300m`

```
vocab               256 000     SentencePiece (multilingual)
hidden_size         768
intermediate_size   1152        SwiGLU MLP inner
num_hidden_layers   24
num_attention_heads 3           query heads
num_kv_heads        1           GQA 3:1
head_dim            256         unusual (most are 64/128)
sliding_window      512
sliding_window_pat  6           every 6th = full; rest sliding
bidirectional       true
rope_theta          1e6         global (full layers)
rope_local_base_freq 1e4        local (sliding layers)
rms_norm_eps        1e-6
activation          gelu_pytorch_tanh
```

Backbone ≈302 M params. ST head ~4.7 M:
`Pooling(mean) → Dense(768→3072) → Dense(3072→768) → L2`.

### Per-layer block

```
x → input_norm → Attn → post_attn_norm → +x → pre_ff_norm
  → MLP(SwiGLU) → post_ff_norm → +above_resid → x_out
```

**Attention** (GQA 3:1, no KV cache, bidirectional):
```
q,k,v = projs(x); reshape [b,heads,t,256]
q,k = q_norm(q), k_norm(k)          RmsNorm over head_dim
q,k = RoPE(q,k)                     per-layer table (local|global)
k,v = repeat_kv(_, groups=3)
scores = q @ k.T * (1/sqrt(256)) + attn_mask
probs  = softmax(scores)
out    = probs @ v → reshape → o_proj
```
Mask: `[b,1,1,t]` (full) or `[b,1,t,t]` (sliding).

**MLP**: `down(gelu(gate(x)) * up(x))`. `gate`,`up`: 768→1152;
`down`: 1152→768.

Layer kind: `(i+1) % 6 != 0` → sliding. Full layers: 5/11/17/23.

### Our impl (`crates/core/src/candle_gemma_embed.rs`)

Fork of `candle_transformers::models::gemma3`:
1. `Backbone::forward` returns `[b,t,h]`, no `lm_head`.
2. Two `RotaryEmbedding` pre-built (local + global); layers pick by
   `(i+1) % sliding_window_pattern`.
3. `Layer.sliding_window: Option<usize>` drives mask choice.
4. Bidirectional attention, no KV cache, `scale = 1/sqrt(head_dim)`.
5. `GemmaRmsNorm` upcasts to f32, applies `x * (1 + weight)`.
6. Backbone native f32 on Metal. f16 collapses MRR 0.45 → 0.12
   (Google's card warns; bench confirms). Dense heads always f32.

### Dispatch budget (b=8, t=512)

Per layer:
- 4 outer RmsNorm + 2 inner (q_norm, k_norm)
- 7 Linear (q/k/v/o/gate/up/down)
- 1 softmax + 2 matmul (attn)
- 2 elementwise (resid adds, SwiGLU mul, activation)

≈ **20 dispatches × 24 layers = ~480 GPU dispatches per forward**,
plus rotary/reshape/transpose.

## Fusion targets

### T1 — RmsNorm + Linear fuse

Most fusable: `input_norm → q/k/v` (shared input);
`pre_ff_norm → gate/up` (shared input). Norm in threadgroup memory,
output streams into simdgroup matmul accumulator. Saves ~3
roundtrips per layer × 24 = 72.
**Effort:** 1 MSL kernel with norm prologue + tiled matmul.

### T2 — SwiGLU MLP fuse

`down(gelu(gate(x)) * up(x))` as one kernel. `gate`,`up` share
input → tile-once-read-twice. Saves ~3 dispatches per layer × 24
= 72. **Effort:** complex tiled three-matmul kernel.

### T3 — Attention score fuse (`QK^T → scale → mask_add → softmax`)

NOT flash-attention (verdicted slower at S=512 in
`FLASH_ATTENTION_VERDICT.md` on `flash-attention-attempt`). Collapse
3 score-side ops into one; AV matmul stays separate. Saves 2-3
dispatches per layer × 24 = 48-72.
**Effort:** medium. Pattern follows `candle_metal_kernels::call_softmax`.

### T4 — Residual_add + RmsNorm fuse — **SHIPPED**

Phase A delivered. Dual-output MSL kernel (residual sum + Gemma
RmsNorm in one dispatch) wired into `Layer::forward`. Bench:
**~14% sync gain on Gemma f32 golden** (baseline ~58 s → fused
~50 s averaged across 2 runs each), MRR / R@1 / NDCG bit-identical
across runs. Well above Phase A's 3% decision gate.

Code: `crates/core/src/candle_gemma_kernels.{rs,metal}` +
`Layer::forward` in `candle_gemma_embed.rs`.

**Design block uncovered in Phase A:** Gemma3 normalizes BEFORE the
residual add, not after. The fusable pattern is `(post_attn_norm_out
+ residual) → pre_ff_norm`, BUT the `(post_attn_norm_out + residual)`
value is **also saved as the next residual** (used at end-of-layer).
A single-output fused kernel forces recompute, killing the savings.

**Fix: dual-output kernel.** Output buffer `[2, N, h]`: index 0 =
`y` (normed), index 1 = `residual_sum`. Caller narrows both for ~0
cost. CustomOp3 returns one Tensor; we adopt the `[2, N, h]` shape
and `i(0)` / `i(1)` to split.

`x = (x_in + sublayer_out); x = next_norm(x)` → one kernel, two
outputs. 1 fuse per layer × 24 = 24 dispatches saved (within-layer
only; cross-layer fuse would need to restructure `Backbone::forward`
to absorb the next layer's `input_layernorm`).

**Effort:** medium (was: low). Most plumbing is done. The MSL change
is small: write `(x_in + sub)` to a second output slot before
applying the norm + weight multiply.

## Integration — candle-metal-kernels

`candle-metal-kernels` provides `Kernels` struct (compiled MSL libs)
+ `call_*` functions taking command encoder + buffers + dispatch dims.

To add a kernel:
1. MSL source in `.metal` file
2. Reference from `Kernels::load_library`
3. Add `call_<name>` Rust wrapper grabbing pipeline state + dispatch
4. Hook via `Tensor::custom_op*`: impl `CustomOp1/2/3` with
   `cpu_fwd` + `metal_fwd`. `metal_fwd` grabs `MetalStorage`,
   command encoder, calls our `call_<name>`.

**Recommend:** ship kernels in own crate (`crates/core/src/candle_gemma_kernels.rs`
or sibling). Cleaner upgrade path than forking
`candle-metal-kernels`.

## Phased delivery

### Phase A — scaffolding + T4 (4-6 h)

1. New module `crates/core/src/candle_gemma_kernels.rs`. Embed MSL
   via `include_str!`. Build with `Device::new_library_with_source`
   at runtime.
2. **First kernel: T4** (residual_add + rmsnorm). Pure elementwise
   + reduction. Validates CustomOp plumbing end-to-end.
3. Unit test: MetalStorage round-trip vs CPU reference within 1e-5.
4. Wire into `Layer::forward`. A/B bench synthetic batch.

**Gate:** T4 ≥ 3% gain → continue. 0% → abandon (we're matmul-
compute-bound, not dispatch-bound; fusions can't help).

### Phase B — T1, T2 (6-10 h)

5. T1: RmsNorm + Linear. Norm in threadgroup memory, output into
   simdgroup matmul accumulator.
6. T2: SwiGLU. Three matmuls + activation + mul.

**Gate:** Phase A + T1 ≥ 10% combined → ship; skip T2.

### Phase C — T3 (8-12 h, optional)

7. Attention score fuse. Skip if A+B already ≥ 15%.

## Risks

1. **`MetalStorage` API access.** `buffer()` may be `pub(crate)` in
   0.10.2. Verify early — may need `unsafe` shim or candle fork.
2. **MSL compile.** Runtime compilation from embedded source is the
   pattern. Replicable.
3. **Numerical drift.** Gemma's RmsNorm specifically does `x * (1 +
   w)` AND upcasts to f32. Fused kernel must match exactly or
   silently degrades quality. Reference-compare every kernel.
4. **Effort > gain.** Profile shows backbone = 99% but does NOT
   decompose into per-op time. If 80% of forward is pure matmul
   compute (not dispatch overhead), Phase A gate fails at 0% and
   we abandon. **Cheap to test, expensive to commit.**

## Predicted outcome

- Phase A (T4 only): +1-3%
- Phase A + T1: +5-10%
- Phase A + T1 + T2: +8-15%
- All four: +10-20% (ceiling on this arch)

vs `max_length` knob (already shipped) at +9% incremental for 30
min work — Opt 4 buys quality preservation that the knob trades
away. Multi-day investment justified only when quality matters
more than dev time.

## Resume checklist

- [ ] Candle version still 0.10.2-compatible? `MetalStorage` API
      shifts effort.
- [ ] Re-run baseline. If sync_ms < 30 s on Gemma, less headroom.
- [ ] Apple Silicon generation. M3+ has native bf16 + better matmul
      pipelines; some fusions may not pay back on M3+.
- [ ] Candle 0.11+ Metal SDPA / MPSGraph binding? T3 becomes free.
- [ ] Re-read `FLASH_ATTENTION_VERDICT.md` (commit 0ddc448).

## Files

```
crates/core/src/candle_gemma_embed.rs    Backbone, Layer, Attention, MLP
crates/core/src/candle_gemma_kernels.rs  NEW — MSL + CustomOp wrappers
benchmarks/results/PERF-EXPERIMENTS.md   final row
```
