# Changelog

## 0.2.4

- Search recall improved by **chunk enrichment**: each chunk is now
  embedded with a short `path::symbol (kind) — signature` header that
  bridges natural-language queries to code, so plain-English questions
  land on the right function more often. The stored/returned snippet
  stays raw code — only retrieval changes. Triggers one automatic
  re-index on first sync after upgrade (no config change needed).
- New optional **cross-encoder re-rank** stage, **off by default**.
  Enable with `[rerank] enabled = true` to re-score the top fused
  candidates with a small int8 reranker for sharper top results (it
  downloads a second ~280 MB model and adds some latency; disabled =
  exactly the previous behavior, no extra download). Tunables:
  `[rerank] model`, `[rerank] top_n`.
- **BREAKING: the default model is now `nomic-ai/CodeRankEmbed`** (SOTA
  code retrieval, int8 ONNX, ~0.7 GB peak) instead of the static
  `potion-multilingual-128M`. The per-project index re-embeds itself
  once automatically on first search/sync after upgrade. Lighter
  alternatives: `models set-default jinaai/jina-embeddings-v2-base-code`
  (~0.76 GB) for code, `minishlab/potion-multilingual-128M` or an e5
  model for multilingual/Russian.
- New built-in `nomic-ai/CodeRankEmbed` (the default), with its
  required CLS pooling and `Represent this query for searching relevant
  code: ` query prefix applied automatically. The right backend is
  chosen for your hardware, no flags or downloads to manage:
  **Apple Silicon** runs it on the **Metal GPU** (~1.7x faster than
  CPU; the CoreML runtime can't accelerate this architecture so the GPU
  path is used directly), **CUDA** machines run the accelerated f32
  ONNX, and everything else runs the small **int8** ONNX on CPU. If the
  GPU is unreachable it falls back to int8 CPU automatically.
- CoreML now reuses a persistent compiled-model cache across runs, so
  ONNX models that do run on CoreML no longer recompile on every
  startup and no longer emit the macOS "Context leak detected" log
  spam during sync.
- Each model now has a per-model query/document prefix and pooling
  (mean/cls/last-token), applied automatically — no manual flags
  needed for built-ins. `models add` gained `--query-prefix`,
  `--doc-prefix`, `--pooling`; `--e5_prefix` still works as a shortcut
  (now also on `models add-remote`).
- **BREAKING**: the `e5_prefix` bool is gone — `[remote] e5_prefix` and
  `[model] onnx_e5_prefix` are replaced by explicit
  `query_prefix`/`doc_prefix` (and `onnx_query_prefix`/`onnx_doc_prefix`)
  string options. Old `e5_prefix = true` is ignored; set
  `query_prefix = "query: "` + `doc_prefix = "passage: "` (or re-run
  `models add-remote --e5_prefix`).
- Fix: `nomic-ai/nomic-embed-text-v1.5` now applies its required
  `search_query: ` / `search_document: ` prefixes (it previously ran
  with no prefix — degraded results).
- New built-in models: `intfloat/multilingual-e5-base`,
  `intfloat/multilingual-e5-large` (multilingual incl. Russian),
  `Snowflake/snowflake-arctic-embed-m-v2.0` (multilingual + code,
  CLS-pooled).
- Changing a model's prefix/pooling now invalidates the index
  (automatic one-time re-embed), so a contract fix can't leave stale
  vectors behind.
- **Fix (critical): search returned irrelevant results for every query
  and model.** Upgrade to fix. The per-project index rebuilds itself
  automatically on the first search after upgrade (one slower sync);
  no config change needed.
- Hybrid search is now always on; the `[search] hybrid` setting is
  ignored (remove it if present — nothing else to do).
- Fix: `nomic-ai/nomic-embed-text-v1.5` now loads as a quantized ONNX
  model — it (and the e5 models) honor `[model] precision = fp16|int8`,
  roughly halving/quartering RAM instead of a fixed heavy f32 load.
- Fix: a transformer model (e5 / nomic) on a real repo could spike to
  ~10 GB RAM and wedge the machine. The embed batch size is now
  **per-model** (auto): heavy ONNX models use a small batch, the static
  models a large one — bounding memory without slowing the light
  models. `[sync] embed_batch_size` now defaults to `0` (auto); set a
  non-zero value to override. For large repos the **default static
  model** (`minishlab/potion-multilingual-128M`) is still best — no
  attention, full sync in seconds at ~1 GB.
- `jinaai/jina-embeddings-v2-base-code` works again and now defaults to
  its **int8** ONNX (~0.76 GB vs ~2.5 GB f32). Adding it (or any
  jina-code repo) via `models add` also works now.
- Each model now declares an architecture (static / onnx / fastembed);
  a model added with `models add` has its `config.json` architecture
  checked at add time, so an unsupported export (LM-head / KV-cache
  decoder) is rejected up front instead of failing or eating memory
  later.
- Built-in list trimmed: `intfloat/multilingual-e5-base` and
  `-e5-large` removed (too slow/heavy to be usable on real repos —
  `models add` them manually if you want them). Built-ins now:
  potion-multilingual-128M (default), potion-base-32M,
  multilingual-e5-small, jina-embeddings-v2-base-code,
  nomic-embed-text-v1.5,
  jamie8johnson/e5-base-v2-code-search. See the README table for
  measured RAM/speed.
- New built-in `jamie8johnson/e5-base-v2-code-search` — e5-base-v2 tuned
  for code search. Select with
  `models set-default jamie8johnson/e5-base-v2-code-search`. (If you had
  added it via `models add`, drop that custom entry and use the
  built-in: it applies the required `query:`/`passage:` prefixes and the
  CPU pin below automatically.)
- Fix: an ONNX model on a real repo could blow past 6 GB RAM (and crash
  the sync) on macOS, especially with a model added via `models add`.
  The accelerator (CoreML/CUDA) recompiled and kept a copy of the graph
  for every distinct batch size; embedding now runs a single fixed batch
  width so memory stays bounded (measured ~0.6 GB vs 6+ GB) with no
  config change. A model whose raw ONNX export the accelerator handles
  too slowly can now also pin CPU automatically (built-ins only).

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
