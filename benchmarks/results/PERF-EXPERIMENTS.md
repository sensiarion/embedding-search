# Gemma3 Metal throughput experiments

Golden corpus, 130 queries, this repo. Macbook M-series, 16 GB unified
memory. All numbers per-model; `Δsync` and `ΔMRR` are vs the f32
baseline at the row's model.

## Stage 1+2 (merged into `main` as 362a80a)

**Stage 1** (`crates/core/src/embedder.rs`,
`crates/core/src/candle_encoder.rs`): per-backend `recommended_batch`
returns 32 for Candle (was 4 — an ONNX-CoreML legacy constraint).
Inner `candle_batch()` window drops 32 → 8 (Metal sweet spot — bigger
pages working set on 16 GB M-series). Override via
`EMBEDDING_SEARCH_BATCH_CANDLE` / `EMBEDDING_SEARCH_CANDLE_BATCH`.

**Stage 2** (`crates/core/src/candle_gemma_embed.rs`): pool in
backbone dtype before final f32 cast; single batched `to_vec2`
readback instead of per-row `to_vec1`; mask host vec built via
`extend` (no extra collect alloc).

| model | sync_ms baseline | sync_ms S1+S2 | Δsync | MRR baseline | MRR S1+S2 | ΔMRR |
|-------|-----------------:|--------------:|------:|-------------:|----------:|-----:|
| google/embeddinggemma-300m (f32 Metal) | 60818 | **43432** | **−28.5%** | 0.4452 | 0.4463 | +0.001 |
| sensiarion/CodeRankEmbed-f16 | 44328 (older) | **28950** | **−34.7%** | 0.327 | 0.3324 | +0.005 |
| nomic-ai/CodeRankEmbed (f32 Metal) | unmeasured | 32674 | — | — | 0.3324 | — |

Quality deltas are within bench noise on three repeats. Search latency
(`search_ms`) drops ~13% on Gemma from the batched readback.

## Experiments (not merged — each lives in `git stash`)

| exp | stash | code change | sync ms (Gemma) | ΔMRR (Gemma) | verdict |
|----:|-------|-------------|----------------:|-------------:|---------|
| A   | —     | `EMBEDDING_SEARCH_BATCH_CANDLE=64` (env only) | 43155 | +0.000 | **no win** — outer batch already saturated at 32 |
| B   | `stash@{2}` | `[model] max_length: 512 → 256` (`config.rs:365`) | **25393 (−42%)** | **−0.032 (−7.2%)** | **tunable** — keep behind a knob; ship only if user accepts the quality drop |
| C   | `stash@{0}` | drop last 4 of 24 Gemma3 layers (`candle_gemma_embed.rs:336`) | 38048 (−12%) | **−0.230 (−51%)** | **reject** — quality destroyed |
| D   | `stash@{1}` | fuse 2_Dense·3_Dense into one Linear(768→768) | 43792 (≈0) | −0.0002 (noise) | **no win** — dense head too small (4.7 M params) to matter on a 300 M backbone |

Predicted but not run (high effort, low expected ROI per the
candle-quant analysis below):

| exp | reason skipped |
|----:|----------------|
| E (Q8_0 gate_proj only)   | candle Metal qmatmul requires F32 activations (`quantized/metal.rs:288`) — every quantized linear forces an `f16 → f32 → qmatmul → f32 → f16` round-trip per layer. At b=8 the cast cost dominates the bandwidth win for a 300 M model. |
| F (Q8_0 full backbone)    | same cast issue × 168 layers. Memory −300 MB but throughput likely worse, MRR −0.004 ± noise. Worth it ONLY on memory-constrained hosts (<8 GB unified) — not on this machine. |
| G (CPU/GPU pipeline)      | producer/consumer threads inside `embed()`: tokenize batch N+1 while forward(N) runs on Metal. Worth it only if tokenize is >20% of wall-time — not profiled. Non-trivial restructure of `CandleGemmaEncoder::embed`. |

## Deeper experiments (post Stage 1+2)

### Profile (golden, Gemma f32, EMBEDDING_SEARCH_PROFILE=1)

| stage | sum | % of measured |
|-------|----:|--------------:|
| tokenize | 0.16 s | 0.3% |
| pack ids | 0.01 s | 0.02% |
| mask build | 0.17 s | 0.3% |
| **backbone forward** | **52.1 s** | **99.0%** |
| pool | 0.21 s | 0.4% |
| dense | 0.17 s | 0.3% |
| readback | 0.05 s | 0.1% |

Profile killed two planned optimizations dead by Amdahl:

- **Opt 1 (pipelined readback)** — readback is 0.05 s → max saving
  0.1%. Abandoned.
- **Opt 2 (tokenize-ahead)** — tokenize is 0.16 s → max saving 0.3%.
  Abandoned.

### Opt 5 (Flash Attention) — verdicted on `flash-attention-attempt` branch

`FLASH_ATTENTION_VERDICT.md` (commit 0ddc448, branch-only) shows
UMFA FA is **2.3–4.2× slower** than naive at seq ≤ 512. Crossover is
at seq ≈ 4 k. We cap chunks at 512. Hard skip.

### max_length sweep (the shipped knob)

Gemma f32:

| max_length | sync ms | Δsync | MRR | ΔMRR | R@1 |
|-----------:|--------:|------:|----:|-----:|----:|
| 256 | 28262 | **−43%** | 0.413 | −0.030 | 0.139 |
| 320 | 34152 | −31% | 0.422 | −0.021 | 0.139 |
| 384 | 39766 | −19% | 0.430 | −0.013 | 0.162 |
| 448 | 45344 | −8% | 0.452 | **+0.009** (noise?) | 0.185 |
| **512 (default)** | 49212 | 0 | 0.443 | 0 | 0.169 |

CodeRankEmbed-f16 — monotonic (no sweet spot):

| max_length | sync ms | Δsync | MRR | ΔMRR |
|-----------:|--------:|------:|----:|-----:|
| 256 | 16661 | **−48%** | 0.305 | −0.024 |
| 320 | 20621 | −36% | 0.317 | −0.012 |
| 384 | 24584 | −23% | 0.321 | −0.007 |
| 448 | 28729 | −10% | 0.325 | −0.004 |
| **512 (default)** | 32026 | 0 | 0.328 | 0 |

Shipped as `EMBEDDING_SEARCH_MAX_LENGTH=N` env var + per-project
`[model] max_length` config.

**Default changed 512 → 448** after CSN confirmation (5000
distractor pool, 200 queries, --rerank):

| model | metric | @448 | README old @512 | Δ |
|-------|--------|-----:|----------------:|---|
| Gemma f32 | MRR base | 0.9395 | 0.940 | −0.0005 |
|  | R@1 | 0.915 | 0.915 | 0 |
|  | NDCG | 0.9495 | 0.951 | −0.0015 |
|  | MRR +rerank | 0.9443 | 0.944 | +0.0003 |
|  | embed docs/s | **16.6** | ~12 | **+38%** (combined with Stage 1+2) |
| CodeRankEmbed-f16 | MRR base | 0.9288 | 0.929 | −0.0002 |
|  | R@1 | 0.91 | 0.910 | 0 |
|  | NDCG | 0.9354 | 0.937 | −0.0016 |
|  | MRR +rerank | 0.9275 | 0.928 | −0.0005 |
|  | embed docs/s | **26.4** | ~21 | **+26%** |

All quality deltas within bench noise. Throughput gain shipped.

### Opt 4 Phase A — shipped (`bf52f70` + follow-up)

Fused `residual_add + Gemma RmsNorm` MSL kernel. Dual-output
(`[2, N, h]`) so Gemma's residual-summed value is reused both as
norm input and end-of-layer residual without recompute. Wired into
`Layer::forward`; one fewer pair of dispatches per Gemma3 layer ×
24 = 24 dispatches removed per forward.

Head-to-head golden 130q (this repo, same indexing state):

| variant | run 1 sync | run 2 sync | mid | MRR | R@1 |
|---------|-----------:|-----------:|----:|----:|----:|
| baseline (S1+S2 + max_length=448) | 65 217 | 51 264 | ~58 000 | 0.446 | 0.1846 |
| **+ T4 fused** | **47 476** | **53 000** | **~50 000** | **0.446** | **0.1846** |

**~14% extra sync gain on Gemma f32, quality bit-identical.** Above
Phase A's 3% decision gate.

### Opt 3 / Opt 4 Phase B+ — parked

- **Opt 3 (sequence packing):** profile shows most buckets pad to
  512 anyway because long chunks dominate length-sorted buckets.
  Predicted gain shrinks to ~3-10%. Not run.
- **Opt 4 (custom fused Metal kernels):** the only remaining lever
  that targets the actual bottleneck. Multi-day work for predicted
  +5-15% with real risk of zero (matmul-compute-bound vs
  dispatch-bound is unknown). Full plan + arch description parked at
  `docs/OPT4-METAL-KERNELS-PLAN.md` for future pickup.

## What to merge next

Already shipped:
- Stage 1+2 (commit 362a80a): −28% Gemma sync, −35% CodeRank sync
- `max_length` knob + docs

Nothing else worth shipping right now without Opt 4. Profile says
we are at the floor for this model on candle Metal absent custom
MSL kernels.

## Reproduce

```
# Baseline (commit `f5f7f1f`):
git checkout f5f7f1f -- crates/core/
cargo build --release
rm -rf .embedding-search && ./target/release/xtask golden \
  --models google/embeddinggemma-300m,sensiarion/CodeRankEmbed-f16 \
  --output benchmarks/results/baseline

# Stage 1+2 (commit `362a80a`):
git checkout 362a80a -- crates/core/
cargo build --release
rm -rf .embedding-search && ./target/release/xtask golden \
  --models google/embeddinggemma-300m,sensiarion/CodeRankEmbed-f16 \
  --output benchmarks/results/s1s2

# Replay any experiment:
git stash apply stash@{n}
cargo build --release
rm -rf .embedding-search && ./target/release/xtask golden ...
```
