# Changelog

## Unreleased

- Opt 4 Phase A — fused `residual_add + Gemma RmsNorm` Metal kernel
  shipped. One MSL dispatch replaces a pair (residual sum + norm)
  per Gemma3 layer × 24 layers = 24 dispatches removed per forward.
  Kernel is dual-output (`[2, N, h]`): `i(0)` = residual sum (saved
  for end-of-layer add), `i(1)` = normed (fed to MLP). Numerical
  match with CPU/reference ≤5e-5. Bench on golden 130q: ~14% sync
  gain on Gemma f32, quality bit-identical (MRR / R@1 / NDCG
  unchanged across A/B runs). Apple Silicon Metal only; CPU
  fallback in the CustomOp3 path keeps tests working off-Metal.
- **Default `[model] max_length` is now 448** (was 512). Confirmed on
  the CSN eval (5000 distractor pool, 200 queries) to be the Pareto
  sweet spot for the candle Metal path: quality identical to 512
  within bench noise (Gemma MRR Δ −0.0005, CodeRankEmbed MRR Δ
  −0.0002), indexing throughput +26–38%. First sync after upgrade
  rebuilds the index (the fingerprint includes `max_length`).
  Override per-project via `[model] max_length = N`, or ad-hoc via
  `EMBEDDING_SEARCH_MAX_LENGTH=N`. Pareto frontier (relative to old
  512): 256 → −43%/−0.030 MRR · 384 → −19%/−0.013 · 448 → −8%/~0.
- EmbeddingGemma candle Metal path stays at native f32. An attempt to
  cast to f16 on Metal collapsed retrieval quality (Google's model
  card explicitly warns activations don't survive fp16 — confirmed by
  bench: MRR 0.445 → 0.117). bf16 fared no better — emulated below M3
  and ~3× slower. The 308 M backbone at f32 stays well under typical
  GPU memory limits, so the cost is acceptable for the quality.
- Sharpen `search_code` MCP tool description: state "ONE call usually
  beats several greps", clarify when `path`/`limit` help, and trim
  self-referential examples so the description applies to any
  project (not just embedding-search itself).
- Tighten loading guidance in the `SessionStart` /
  `UserPromptSubmit` hooks and SKILL.md: load via `ToolSearch` only
  when the tool is listed as deferred; if it is absent from both the
  available and deferred lists, the MCP is not wired up — don't call
  `ToolSearch` (it returns `No matching deferred tools found`) and
  don't retry within the session. Cuts wasted tool calls.

## 0.2.9

- **BREAKING (safety):** the MCP server and CLI now refuse to index
  `$HOME`, `/`, or any ancestor of `$HOME`. Previously a launcher
  spawning the MCP server from `$HOME` would walk the entire home
  tree and write `~/.embedding-search/` artifacts; this is the bug
  that motivated the release. When `HOME` is unset (sandboxed
  launcher, CI), the resolved path must be inside a git repository —
  otherwise the indexer refuses to start. To explicitly target a path
  set `EMBEDDING_SEARCH_PROJECT_DIR` to a git checkout subdir.
- **BREAKING:** CLI subcommands (`init`, `sync`, `search`, `status`,
  `clear`, `debug ...`) no longer default `path` to `.`. When the
  path is omitted, the project root is resolved via
  `EMBEDDING_SEARCH_PROJECT_DIR` → `git rev-parse --show-toplevel` →
  CWD. Pass an explicit path to keep the old "use CWD" behavior:
  `embedding-search sync .` still works.
- MCP `search_code` tool gets an intent-shaped description with
  concrete examples and inline `tool_use_examples` in the JSON
  Schema so Claude prefers it over `Grep` for conceptual queries.
- `models list` RAM~MB is now platform-aware: the candle Metal path
  on Apple Silicon reports 2 B/param for `*-f16` repos and 4 B/param
  for native-f32 repos (`sensiarion/CodeRankEmbed-f16` → ~424 MB,
  `nomic-ai/CodeRankEmbed` → ~698 MB). Off Apple Silicon both still
  report the int8 ONNX footprint.
- Incremental sync hardened:
  - `git checkout` no longer triggers an endless re-read loop: a
    file whose bytes match but whose mtime/size moved now refreshes
    the DB row's stat so the next sync's cheap (mtime, size)
    short-circuit fires.
  - Cross-file chunk reuse: a chunk whose body matches one already
    embedded somewhere (rename, branch switch, copy-paste) now
    copies the existing vector instead of re-running the model. The
    source vector encoded the original path's header, so a small
    embedding drift is accepted in exchange for skipping the embed.
  - All content/chunk hashing audited to use blake3 end to end.
- `cargo xtask eval` driver:
  - `--models a,b,c` or `--models all` selects the model set
    (defaults preserved).
  - `--output DIR` overrides the per-run results dir; default
    layout is `benchmarks/results/<UTC>-<commit>/results.jsonl` +
    `REPORT.md` (ranked Markdown summary). The legacy
    `effectiveness.jsonl` history file is still appended for any
    tooling that reads it.
  - A bad model no longer aborts the sweep; its failure shows up in
    REPORT.md under an `## errors` section.
  - `--rerank-model REPO` swaps the cross-encoder for ablation
    (default: ettin; alternatives: `mixedbread-ai/mxbai-rerank-base-v2`,
    `BAAI/bge-reranker-v2-m3`).
- New `cargo xtask golden` subcommand: runs hand-curated
  `(query → expected_file)` pairs through a real `SyncEngine` over a
  filtered snapshot of the source root (default
  `benchmarks/golden/this-repo.toml`). Reports MRR@k / Recall@1 / R@5 /
  NDCG@k + sync/search latency per model. Use to compare candidates on
  this repo's own retrieval before picking a default.
- New built-in model: `google/embeddinggemma-300m` (308 M, Matryoshka
  128–768 dims, multilingual incl. Russian, Gemma license). On
  **Apple Silicon** it runs on the **Metal GPU via candle (f32)** —
  ~2.8× faster indexing than the int8 ONNX CPU fallback used on
  other platforms (or when the candle path is unavailable). The HF
  repo is license-gated: the first sync needs a token with the
  "Read access to public gated repos" scope (`canReadGatedRepos`),
  otherwise the loader falls back to int8 ONNX with a single warn
  line. Task-shaped prompts wired automatically
  (`task: code retrieval | query: …` / `title: none | text: …`).
  Pick with `models set-default google/embeddinggemma-300m` (and
  re-index). On CodeSearchNet 5000/200 it edges past the default on
  every quality metric (MRR 0.938 vs 0.929, NDCG 0.949 vs 0.937,
  R@1 tied at 0.91); on the in-repo 130-query golden corpus it is
  the only built-in that lands top-1 hits (R@1 0.16 vs 0.00 for
  every other candidate). The previous `models add` example for the
  same repo was dropped from the README — it's a built-in now.

## 0.2.8

- Skill description rewritten to trigger in subagent (Explore /
  general-purpose / Task) contexts where the SessionStart and
  UserPromptSubmit hooks don't propagate. New "Subagent note" and
  "Loading the tool when deferred" sections in `SKILL.md` make the
  skill self-sufficient inside nested agents and explain the one
  `ToolSearch select:` step needed when the tool schema is deferred.

## 0.2.7

- **BREAKING:** per-project model override moved from the top-level
  `embedding-search set <model> [path]` to `embedding-search models set
  <model>` (next to `models set-default`). The `[path]` argument is
  gone — the command now always targets the current working directory
  (`cd` to the project first). The behavior is otherwise unchanged.
- **BREAKING:** `--e5_prefix` (alias of
  `--query-prefix "query: " --doc-prefix "passage: "`) is gone from
  `models add` and `models add-remote`. The built-in models that needed
  it were removed in 0.2.6 — pass the explicit prefixes if you still
  need them on a custom model.

## 0.2.6

- **BREAKING:** the predefined model list is trimmed to five:
  `sensiarion/CodeRankEmbed-f16` (default), `nomic-ai/CodeRankEmbed`,
  `jinaai/jina-embeddings-v2-base-code`, `minishlab/potion-base-32M`,
  `minishlab/potion-multilingual-128M`. The e5 / arctic / nomic-embed
  / jamie8johnson built-ins are gone — if your config `[model] default`
  named one, set a remaining built-in (or register it yourself with
  `models add --repo …`). Any other HF model still works via
  `models add`.
- **BREAKING:** the `fp16 | int8 | full` precision knob is removed —
  `[model] precision`, the `models add --precision` flag and per-model
  `precision` no longer exist. A model is identified solely by its
  concrete `.onnx` file: built-ins pin one; for `models add` use
  `--onnx-file <name>` (default `onnx/model.onnx`). A stale
  `precision = …` line in an old config is ignored. The index
  fingerprint changed, so the next sync re-embeds once automatically.
- **BREAKING:** cross-encoder re-rank now defaults **on for the static
  potion models** (a large measured quality rescue) and **off for the
  SOTA CodeRank/jina bi-encoders** (≈neutral there). Previously it was
  off for everyone. `[rerank] enabled` is now unspecified by default
  (model-driven); set `enabled = true | false` to force it either way.
- Better search results: adjacent small declarations are now coalesced
  into larger chunks (was one chunk per function/struct), so each hit
  carries more surrounding context and ranking improves. The
  per-project index re-embeds itself once automatically on upgrade.
- Fix: files matched by `.gitignore` were still indexed when the
  project is **not a git repository** (or before its first commit) —
  e.g. secrets/build output in a plain directory. They are now
  excluded correctly; the next sync drops any that had slipped in.
- Per-project model: `embedding-search models set <model>` pins a
  model for the current repo only — it writes
  `<project>/.embedding-search/config.toml`, which overrides the
  global default without affecting other projects. `status` shows
  when a project override is active. (Renamed/moved from the previous
  top-level `embedding-search set` — now under `models` next to
  `models set-default`.)
- The CLI now prints the previous/next chunk around each hit for
  immediate context. The MCP/agent output is unchanged (still
  refs-only, so agent searches stay token-lean).
- Large repos: the CLI suggests (and the MCP server exposes a
  `set_model` tool plus a `hint` on status/sync) switching to a fast
  static model (`minishlab/potion-base-32M`) when a heavy model is
  slow to index — an agent can do this itself on the first sync of a
  big repo, no restart needed.
- The optional `[rerank]` cross-encoder is now
  `cross-encoder/ettin-reranker-68m-v1` — at ~68M params the smallest
  strong code-capable reranker (4× smaller than the previous 278M
  `bge-reranker-base`), and stronger on code. On Apple Silicon it runs
  on the **Metal GPU via candle** as-is (native f32, no cast). Off
  Apple Silicon re-rank is a no-op for now (the int8 ONNX export is
  encoder-only) — search returns the fused ranking unchanged, as
  before. Measured lift (`cargo xtask eval --rerank`, 5000-doc pool /
  200 queries, base and rerank on the identical set): a major rescue
  for fast static retrievers — potion-32M 0.730→**0.849** MRR@10,
  Recall@1 0.660→**0.830** (now ≈ a transformer's top-1) — and
  ~neutral on the already-SOTA default (CodeRankEmbed 0.929→0.928), so
  it is the lever that makes a static model viable on a big repo —
  hence the new model-driven default (on for potion, off for
  CodeRank/jina; see the BREAKING note above).
- New `cargo xtask eval`: CodeSearchNet retrieval quality
  (MRR@10 / Recall@1 / NDCG@10) over a large distractor pool —
  `--corpus N` (default 5000) sets the pool, `--queries N` (default
  200) the evaluated queries; `--rerank` adds an opt-in cross-encoder
  pass on the same set so `base` vs `rerank` is a like-for-like delta
  (dev-only).

## 0.2.5

- **The default model is now `sensiarion/CodeRankEmbed-f16`** — an f16
  cast of the official `nomic-ai/CodeRankEmbed`, validated equivalent
  (cosine 0.999998, identical CodeSearchNet MRR@10 / Recall@1; search
  quality unchanged). On **Apple Silicon** it runs on the **Metal GPU**
  via candle at about **half the RAM** (peak ~0.57 GB vs ~1.1 GB f32)
  and faster than the CPU path — CoreML can't accelerate this
  architecture so the GPU is driven directly. CUDA runs the f32 ONNX,
  everything else the int8 ONNX on CPU, with automatic int8-CPU
  fallback if the GPU is unreachable. Both CodeRankEmbed builtins now
  show in `models list`; pick the official upstream f32 weights with
  `models set-default nomic-ai/CodeRankEmbed` (~2x Metal RAM, same
  embeddings, identical off Apple-Silicon). The per-project index
  re-embeds itself once automatically on upgrade (nothing to do).
- Faster indexing: parallel file walk and length-sorted GPU
  sub-batching. The candle sub-batch size is tunable via the
  `EMBEDDING_SEARCH_CANDLE_BATCH` env var (default 32).
- CoreML now reuses a persistent compiled-model cache across runs, so
  ONNX models that do run on CoreML no longer recompile on every
  startup and no longer emit the macOS "Context leak detected" log
  spam during sync.
- Clearer sync status output.
- Vendored code under `/vendor/` is git-ignored and never indexed.

## 0.2.4

- Much better search results. The default model is now
  `nomic-ai/CodeRankEmbed` (state-of-the-art code retrieval); chunk
  enrichment bridges plain-English queries to code so questions land
  on the right function far more often. The per-project index rebuilds
  itself automatically on the first sync after upgrade — nothing to do.
- Lighter model options if the default (~2.4 GB peak) is too heavy:
  `models set-default jinaai/jina-embeddings-v2-base-code` (~0.76 GB,
  code), or `minishlab/potion-multilingual-128M` / an e5 model for
  multilingual incl. Russian.
- New optional cross-encoder re-rank for sharper top results. Off by
  default; enable with `[rerank] enabled = true` (downloads a second
  ~280 MB model, adds some latency). Tunables: `[rerank] model`,
  `[rerank] top_n`.
- More built-in models: `intfloat/multilingual-e5-small/-base/-large`
  and `Snowflake/snowflake-arctic-embed-m-v2.0` (multilingual incl.
  Russian; arctic also code), plus `jamie8johnson/e5-base-v2-code-search`
  and `jinaai/jina-embeddings-v2-base-code` (int8, ~0.76 GB). Each
  built-in applies its required query/document prefixes and pooling
  automatically.
- Much lower memory: syncing a real repo with a transformer model no
  longer spikes to 6–10 GB or wedges the machine — embedding memory is
  now bounded (measured well under 1 GB). `[model] precision =
  fp16|int8` is honored to further cut RAM.
- Hybrid (semantic + keyword) search is always on — remove `[search]
  hybrid` from config if present (ignored otherwise).
- `models add`: query/document prefix and pooling are set automatically
  for built-ins; new `--query-prefix`, `--doc-prefix`, `--pooling`
  flags for custom models (and `--e5_prefix` shortcut, also on
  `models add-remote`). An unsupported model export is now rejected
  immediately with a clear message instead of failing or eating memory
  later.
- Config: the old `e5_prefix` toggle is replaced by explicit
  `query_prefix` / `doc_prefix` (and `onnx_query_prefix` /
  `onnx_doc_prefix`, `[remote]` query/doc prefix). Set
  `query_prefix = "query: "` + `doc_prefix = "passage: "` for e5-style
  models, or use the built-ins (no config needed).

## 0.2.3

- Fix: decoder-LLM embedding repos exported with a KV-cache graph
  (e.g. `onnx-community/Qwen3-Embedding-*`) failed late with a cryptic
  ONNX "Missing Input: past_key_values" error after a multi-GB
  download. They are now detected from the small graph file and
  rejected up front with an actionable message (use an
  encoder/embedding export, a built-in, or a remote endpoint).

## 0.2.2

- Fix: a single transient HTTP timeout while downloading a model
  (large weights/tokenizer on a slow link) aborted the whole load.
  Model fetches now retry with backoff.

## 0.2.1

- Fix: indexing panicked ("byte index N is not a char boundary") on
  files with CRLF line endings and multibyte (e.g. Cyrillic) content
  — the line-window chunker miscounted `\r\n` as one byte. Such files
  (common in frontend repos) now index correctly.

## 0.2.0

- **BREAKING:** the default model is now `minishlab/potion-multilingual-128M`
  (a [Model2Vec](https://github.com/MinishLab/model2vec) static model:
  256-dim, multilingual incl. Russian, tiny RAM, very fast). The index
  fingerprint changes, so the next `sync` rebuilds automatically. To
  keep the previous model run
  `embedding-search models set-default intfloat/multilingual-e5-base`.
- **Hybrid search** (on by default): semantic results are re-ranked by
  fusing the cosine ranking with a BM25 lexical ranking — exact
  identifier queries now surface the literal match. Disable with
  `[search] hybrid = false`.
- **Model2Vec / `StaticModel`** repos are supported (e.g.
  `minishlab/potion-*`) via a built-in static backend — no ONNX.
  `minishlab/potion-base-32M` (English, smallest) is also a built-in.
- `models add`:
  - downloads the model and runs a test embed **before** saving — a
    bad repo/URL fails immediately and nothing is written to config;
  - `--repo` accepts a Hugging Face id **or** a full `huggingface.co`
    URL (browser paste);
  - `--precision fp16|int8|full` and `--onnx-file <name>` to pick an
    exact quantization;
  - external-weights `.onnx_data` sidecars (onnx-community / >2 GB
    models) are fetched automatically;
  - language-model-head and other non-embedding exports are rejected
    up front with a clear message (instead of a multi-GB OOM).
- `models remove <name>` (alias `rm`): unregister a custom/remote
  model and delete its cached weights.
- `clear [path]`: delete a project's index (rebuilds on next sync).
- **Install:** use the bare spec `sensiarion/embedding-search` (not
  `@latest`, which 404s). The Claude Code plugin now also installs the
  MCP server — one install gives the tool + the auto-use nudge.
- Git worktrees nested in a project are no longer indexed (only the
  main working tree).
- ONNX Runtime log spam is silenced (set `RUST_LOG=ort=info` to see it).
- Single self-contained binary; `embedding-search --mcp` runs the
  stdio MCP server.

## 0.1.0

- Initial release: semantic code search CLI + MCP server.
