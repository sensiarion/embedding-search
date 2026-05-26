# Golden retrieval baseline

Captured: `2026-05-26T11:24:06Z`, commit `903674a`, host `macos-aarch64`.
Corpus: `benchmarks/golden/this-repo.toml` — 130 NL queries over the
indexed source tree (~53 files, ~501 chunks).

Reproduce:

```sh
cargo xtask golden --src . --models all -k 10
# or pick a subset:
cargo xtask golden --models "google/embeddinggemma-300m,sensiarion/CodeRankEmbed-f16"
```

## Headline

| model | MRR@10 | R@1 | R@5 | NDCG@10 | sync_ms | search_ms |
|---|---:|---:|---:|---:|---:|---:|
| `google/embeddinggemma-300m` | **0.4365** | **0.1615** | **0.8154** | **0.5511** | 167616 | **33606** |
| `minishlab/potion-base-32M` | 0.3394 | 0.0000 | 0.8000 | 0.4836 | **128** | 1015972 |
| `minishlab/potion-multilingual-128M` | 0.3357 | 0.0000 | 0.7923 | 0.4774 | 162 | 1097075 |
| `sensiarion/CodeRankEmbed-f16` | 0.3270 | 0.0000 | 0.7769 | 0.4645 | 44284 | **2392** |

`search_ms` covers all 130 queries. The static models drive their
default-on cross-encoder rerank (~8 s/query) — the rerank dominates
the wall-clock, not the embedder. CodeRankEmbed + Gemma run rerank-off
by default; their `search_ms` is the unreranked baseline.

## What changed vs the 18-query baseline

The old `2c1429c` baseline ranked `CodeRankEmbed-f16` first on 18
queries (MRR 0.4173). Expanding the corpus to 130 hand-curated NL
queries collapsed that lead — `CodeRankEmbed-f16` falls to last on
MRR, and **`google/embeddinggemma-300m` takes a +0.10 MRR / +0.16 R@1
lead over the next-best candidate.** The smaller corpus did not have
the statistical power to surface this: 1/18 vs 0/18 R@1 looked like
noise; 21/130 vs 0/130 is not.

## What to read

- **Recall@1** — Gemma is the only model that lands the right file at
  rank #1 at all (21/130 = 16.2 %). Every other candidate is 0/130.
  This is the headline result.
- **MRR@10 / NDCG@10** — Gemma wins by ~0.10 / ~0.07 absolute over the
  rerank-boosted static models, and ~0.11 / ~0.09 over CodeRankEmbed.
- **Recall@5** — Spread compressed (0.78 → 0.82). The differentiator
  is not "is it in the top-5", it's "is it ranked #1 vs #3".
- **sync_ms** — Gemma indexes the tree in ~168 s (308 M params, int8
  ONNX); CodeRankEmbed in ~44 s (137 M params); potion in <0.5 s.
  Gemma's index time is the real cost of the quality win.
- **search_ms** — The rerank-on potion configurations are ~30× slower
  end-to-end than Gemma at this corpus size. The rerank lift is real
  on quality (~+0.01 MRR over rerank-off) but it doesn't close the gap.

## Caveats

- 130 queries is enough for ordering to be stable on MRR/NDCG and for
  R@1 differences to clear noise (1/130 ≈ 0.77 % granularity), but
  still small for absolute confidence intervals — treat the deltas as
  directional. The next expansion should aim for ≥200 queries or a
  second eval repo to confirm Gemma's lead generalizes.
- Hand-curated for this Rust workspace. A model that wins here will
  not necessarily win on a Python or TypeScript repo.
- Rerank ON-by-default for the static models (`rerank_default: true`)
  and OFF for CodeRankEmbed + Gemma. The `search_ms` asymmetry
  reflects that. Rerank-off potion numbers are not reported here — a
  separate ablation row is planned (B6 already adds the knob).
- Gemma is the int8 ONNX export pinned to the CPU EP (no CoreML
  acceleration — same dynamic-shape blocker as NomicBert). A future
  candle Metal port would only affect latency, not quality.

## Adding candidates

Drop a new model into `cargo xtask golden --models` once registered via
`models add` or added to `SUPPORTED_MODELS`. Pair the run with the
RSS smoke check (see the 0.2.9 CHANGELOG note) before counting numbers
as trustworthy.
