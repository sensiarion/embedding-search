# Changelog

## 0.2.5

- Apple Silicon now runs the default `nomic-ai/CodeRankEmbed` on the
  **Metal GPU** via an **f16** weight build — about **half the RAM**
  (peak ~0.57 GB vs ~1.1 GB f32) and faster than the CPU path. CoreML
  can't accelerate this architecture, so the GPU is driven directly;
  CUDA machines run the accelerated f32 ONNX, everything else the int8
  ONNX on CPU, with automatic int8-CPU fallback if the GPU is
  unreachable. The f16 build is validated equivalent to f32 (cosine
  0.999998, identical CodeSearchNet MRR@10 / Recall@1) — search quality
  is unchanged. The index records the backend/precision, so it
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
