# Golden retrieval baseline

Captured: `2026-05-26T03:21:17Z`, commit `2c1429c`, host `macos-aarch64`.
Corpus: `benchmarks/golden/this-repo.toml` — 18 NL queries over the
indexed source tree (~52 files, ~432 chunks).

Reproduce:

```sh
cargo xtask golden --src . --models all -k 10
# or pick a subset:
cargo xtask golden --models "sensiarion/CodeRankEmbed-f16,minishlab/potion-base-32M"
```

## Headline

| model | MRR@10 | R@1 | R@5 | NDCG@10 | sync_ms | search_ms |
|---|---:|---:|---:|---:|---:|---:|
| `sensiarion/CodeRankEmbed-f16` | **0.4173** | **0.0556** | 0.8889 | **0.5616** | 39006 | **350** |
| `minishlab/potion-multilingual-128M` | 0.3690 | 0.0000 | 0.8333 | 0.5253 | 188 | 149798 |
| `minishlab/potion-base-32M` | 0.3663 | 0.0000 | 0.8889 | 0.5231 | **128** | 149602 |

`search_ms` covers all 18 queries (~150s total ≈ 8s/query) when the
static models drive their default-on cross-encoder rerank — the heavy
component is the reranker, not the embedder. CodeRankEmbed runs with
rerank-off by default so its 350ms is the unreranked baseline.

## What to read

- **MRR@10 / NDCG@10** — overall ranking quality across the 18 queries.
  CodeRankEmbed is +0.05 MRR over potion baselines on this corpus.
- **Recall@1** — only CodeRankEmbed gets any top-1 hit (1/18 = 5.6%);
  potion never lands the right file at #1 on this corpus.
- **Recall@5** — coverage in top-5 is ~equivalent (~83-89%); the
  differentiator is top-ranking precision.
- **sync_ms** — CodeRankEmbed costs ~200× more to index than potion.
  That asymmetry is the trade the project was built for: static models
  power large-repo indexing, the SOTA bi-encoder powers default quality.

## Caveats

- 18 queries is far too few to be statistically conclusive; treat the
  numbers as a regression anchor for future model comparisons on this
  specific tree, not as a model-ranking verdict.
- The golden pairs are hand-curated for this repo. A model that wins
  here won't necessarily win on a Python service repo or a TypeScript
  frontend.
- Rerank is ON-by-default for the static models (`rerank_default: true`
  in `SUPPORTED_MODELS`) and OFF for CodeRankEmbed. The `search_ms`
  asymmetry reflects that.

## Adding candidates

Drop a new model into `cargo xtask golden --models` once registered via
`models add` or added to `SUPPORTED_MODELS`. Pair the run with the
RSS smoke check (see the 0.2.9 CHANGELOG note) before counting numbers
as trustworthy.
