# embedding-search

Fast semantic code search for Claude Code and the terminal. A persistent MCP
server indexes your codebase into embeddings, keeps the index fresh via a file
watcher, and exposes a `search_code` tool that Claude uses instead of grep for
conceptual lookups. Also usable directly as a CLI.

## Install

One self-contained binary: `embedding-search` is the CLI, and
`embedding-search --mcp` runs the stdio MCP server (no separate
binary). It ships through [mcp-bin](https://github.com/sensiarion/mcp-bin):
`npx -y mcp-bin sensiarion/embedding-search --mcp` resolves the right
binary from the latest GitHub release on first launch — no pre-install.
The bare `owner/repo` spec tracks the newest release; mcp-bin caches it
and reuses it (run `npx mcp-bin expire sensiarion/embedding-search` to
move to a newer release, or pin one with `@vX.Y.Z`). Editors spawn it
with the workspace as cwd, so it indexes the open project
automatically. (The `--mcp` trailing arg flips the binary into server
mode.)

Prebuilt release binaries: **macOS arm64, Linux x86_64, Linux arm64**.
Intel macOS and Windows are not prebuilt (their onnxruntime toolchain
paths are unreliable) — build from source there: `cargo build
--release` (see *Build from source*).

### Claude Code

One install — registers the `search_code` tool and makes Claude prefer
it over grep/find for conceptual lookups:

```bash
claude plugin marketplace add sensiarion/embedding-search \
&& claude plugin install embedding-search-autouse@embedding-search
```

Restart the session for it to take effect. Disable any time: tell Claude
"stop using embedding search", or `claude plugin uninstall
embedding-search-autouse` (removes the tool and the nudge together).

Just want the `search_code` tool without the auto-use nudge:

```bash
claude mcp add --scope user --env RUST_LOG=warn embedding-search \
  -- npx -y mcp-bin sensiarion/embedding-search --mcp
```

### OpenCode

Edit `~/.config/opencode/opencode.json` (global) or `opencode.json` (per
repo) — note OpenCode uses `mcp`/`type: local`/`environment`, not the
`mcpServers` shape:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "embedding-search": {
      "type": "local",
      "command": ["npx", "-y", "mcp-bin", "sensiarion/embedding-search", "--mcp"],
      "enabled": true,
      "environment": { "RUST_LOG": "warn" }
    }
  }
}
```

Optional auto-use nudge (so OpenCode prefers `search_code` over
grep/find) — pick one:

- Copy [`opencode/prefer-search.js`](opencode/prefer-search.js) to
  `~/.config/opencode/plugin/` (global) or `.opencode/plugin/` (per repo),
  restart OpenCode.
- Or add to `AGENTS.md`: *"For code exploration prefer the
  embedding-search semantic search tool over grep/find; use grep/find only
  for exact known strings."*

## CLI

```
embedding-search init [path]                     create the index, first sync
embedding-search sync [path] [--force]           re-index (progress bar)
embedding-search search <query> [-n N] [--json] [--in DIR|FILE] [--no-sync]
embedding-search status [path]                   index/sync health
embedding-search clear [path]                     delete the index (rebuild on next sync)
embedding-search serve | --mcp                   run MCP server on stdio
embedding-search debug files [path]              list indexed files
embedding-search debug chunks <file> [--path .]  show a file's chunks
embedding-search models list                     built-in + custom models
embedding-search models set <name>                per-project (CWD) model override (re-index)
embedding-search models set-default <name>       switch the global default (re-index)
embedding-search models add --name X --repo ORG/M [--onnx-file F] [--query-prefix .. --doc-prefix .. --pooling mean|cls|last-token] | --url URL
embedding-search models remove <name>            unregister + delete cached weights (alias: rm)
embedding-search models add-remote --name N --base-url U --model M   add + select a remote
```

## Config

`~/.embedding-search/config.toml` — one home directory holds all
global state (config + model cache), same path on every OS. Run
`embedding-search status` — it prints the resolved `config:` path.
**Auto-created with the defaults below on first run** of any
command (CLI or MCP), so it's there to edit. Delete it to reset.

```toml
[model]
default   = "sensiarion/CodeRankEmbed-f16"    # SOTA code (f16 Metal / int8 CPU)
max_length = 512                              # token cap; keep ≈ max_chunk_bytes/4
# onnx_path = "/path/to/model-dir"            # custom local ONNX (see below)
# onnx_query_prefix = "search_query: "        # e.g. nomic; omit for none
# onnx_doc_prefix   = "search_document: "     # e.g. nomic; omit for none

[backend]
execution_provider = "auto"   # auto | coreml | cuda | cpu
disable_mem_arena  = true     # keep ORT arena off (memory)

[sync]
max_chunk_bytes  = 2048       # hard cap ≈ max_length*4; large files split, never skipped
embed_batch_size = 0          # 0 = auto (per-model: small for heavy ONNX, large for static)
embed_batch_bytes = 262144    # flush a batch at this many bytes
sync_threads     = 0          # scan/hash/parse worker cap; 0 = all cores but one
resync_interval_minutes = 10  # background catch-up cadence (CLI + MCP)
exclude = []

[search]
exact_below = 50000           # < this many vectors → exact (brute) search, not HNSW

[rerank]
# enabled unspecified ⇒ model-driven: ON for the static potion
# models (big quality rescue), OFF for CodeRank/jina (≈neutral).
# Force it: enabled = true | false
# model = "cross-encoder/ettin-reranker-68m-v1"  # default; Apple Silicon → Metal GPU via candle (as-is f32)
# top_n = 50                            # how many fused candidates to re-score
```

**Hybrid search** (always on): the embedding neighborhood is
over-fetched, then re-ranked by Reciprocal Rank Fusion of the cosine
ranking and a BM25 lexical ranking — so an exact-identifier query
(`validate_bearer_token`) surfaces the literal match pure-vector
ranking buries, while conceptual queries still ride the embeddings.
`score` reports the fused relevance.

**Chunk enrichment** (always on): each chunk is embedded with a short
`path::symbol (kind) — signature` header so natural-language queries
bridge to code; the stored/returned snippet stays raw.

**Cross-encoder re-rank**: the top `top_n` fused candidates are
re-scored jointly `(query, passage)` by
`cross-encoder/ettin-reranker-68m-v1` (a sentence-transformers
CrossEncoder over the Ettin/ModernBERT 68M encoder) and reordered
before truncation — sharper top results when recall is already good.
**Default is model-driven** (override with `[rerank] enabled`): ON
for the fast static potion models (a large measured quality rescue —
they reach ≈ a transformer's top-1), OFF for the SOTA CodeRank/jina
bi-encoders (≈neutral there, not worth the second model + latency)
and for custom/remote models. It does not affect the index (ordering
only), so toggling it never triggers a rebuild. At ~68M params it is
the smallest strong code-capable reranker (vs 278M
`bge-reranker-base`). On **Apple Silicon** it runs **as-is** (native
f32 safetensors, no cast — ModernBERT is unstable in f16) on the
**Metal GPU via candle**. Off Apple Silicon the candle path is
unavailable and the int8 ONNX export is encoder-only (the ST head
ships as separate modules), so re-rank is a no-op there for now —
search still returns the fused ranking unchanged. A code-blind
web-text cross-encoder (e.g. ms-marco MiniLM) badly degrades code
ranking — measured by `cargo xtask eval --rerank` — so the default is
this code-strong model.

Every sync is hash-incremental: a file whose mtime+size are unchanged
is skipped before any read/hash/parse (the cheap node of a blake3 file
hash-tree; content hash is the tiebreaker on mtime-only changes), so a
clean resync is near-instant. **There is no file watcher** — a file an
agent just edited is already in that agent's context, so the index
only needs to catch up for the *next* agent run. The **MCP server**
does a startup sync, then a periodic background resync every
`resync_interval_minutes`; CPU stays idle between ticks (a no-edit
tick costs ~nothing). The **CLI** `search` resyncs at most once per
that same interval (~10 min, Cursor-style; `--no-sync` skips,
`embedding-search status` shows when a resync is due). `--in DIR|FILE`
scopes a search to a subtree/file.
Small indexes (< `[search] exact_below`) use exact brute-force search
instead of the approximate HNSW graph — more accurate, just as fast at
that scale.

Models cached in `~/.embedding-search/models` (override:
`[paths] cache_dir`). The per-project index lives in
`<project>/.embedding-search/` (git-ignored automatically).

### External embeddings service (OpenAI-compatible)

Instead of the in-process ONNX model, point at any OpenAI-compatible
`/embeddings` endpoint — DeepSeek, a local LiteLLM proxy, OpenAI. The
`[model]`/`[backend]` ONNX settings are then ignored.

```toml
[model]
provider = "openai"        # local (default) | openai

[remote]
base_url   = "http://localhost:4000/v1"   # /embeddings is appended
api_key    = "$OPENAI_API_KEY"            # $VAR / ${VAR} → env; "" = no auth
model      = "text-embedding-3-small"
dimensions = 0                            # omit/0 → probed at startup
batch_size  = 64          # texts per request (OpenAI `input` array)
concurrency = 4           # max parallel in-flight requests
timeout_seconds = 60
# query_prefix = "search_query: "    # e.g. nomic remote; omit for OpenAI/DeepSeek
# doc_prefix   = "search_document: "  # e.g. nomic remote; omit for OpenAI/DeepSeek
```

Batching uses the OpenAI `input` array (`batch_size` texts/request);
requests run on a bounded pool of `concurrency` workers, order
preserved. On startup one probe request validates connectivity + auth
and resolves `dimensions` (a mismatch vs. a configured value is a hard
error). Changing remote model/endpoint triggers a one-time re-index.

## Models

Quality: ★ = code relevance / multilingual. RAM≈ = resident memory
(weights + ~350MB ONNX Runtime/tokenizer overhead; ~30MB for the
static models). Large files are never skipped — chunks are hard-capped
and streamed in byte-bounded batches.

`ml` rating covers non-English (incl. Russian): ★★★★★ = full, ★★ = English-centric.

Built-in models are pulled (and cached) from **Hugging Face**. A model
is identified by its concrete `.onnx` file — there is **no precision
knob**. Each declares an *architecture* that decides how it is loaded:

- **static** — [Model2Vec](https://github.com/MinishLab/model2vec)
  token-embedding matrix (no transformer/ONNX): tiny RAM, very fast.
  The two `potion-*` models.
- **onnx** — transformer encoder from an HF ONNX repo, mean/CLS
  pooled, weights from the pinned `.onnx` file. CodeRank (int8 on CPU,
  f32 on CUDA, candle on Metal) and jina-code (pinned int8).
- **fastembed** — encoder served by fastembed's bundled ONNX (a
  fallback arch; no built-in currently needs it).

Each model has a fixed input/output **contract** (query/doc prefix +
pooling) — applied automatically; `pool` = mean / cls. `peak RSS` is
measured indexing a real ~12k-chunk repo (auto per-model batch).

| model | arch | dim | code | ml | pool | peak RSS |
|-------|------|-----|------|----|------|----------|
| sensiarion/CodeRankEmbed-f16 **(default)** | candle f16 Metal / onnx int8 CPU | 768 | ★★★★★ | ★★ | cls | ~0.57 GB |
| nomic-ai/CodeRankEmbed | candle f32 Metal / onnx int8 CPU | 768 | ★★★★★ | ★★ | cls | ~1.1 GB |
| jinaai/jina-embeddings-v2-base-code | onnx (int8) | 768 | ★★★★★ | ★★ | mean | ~0.76 GB |
| google/embeddinggemma-300m | candle f32 Metal / onnx int8 CPU | 768 | ★★★★ | ★★★★★ | mean | ~1.6 GB |
| minishlab/potion-multilingual-128M | static | 256 | ★★★ | ★★★★★ | mean | ~0.85 GB |
| minishlab/potion-base-32M | static | 512 | ★★★ | ★★ | mean | ~0.36 GB |

**The default `sensiarion/CodeRankEmbed-f16`** is SOTA code retrieval
(CLS-pooled). It is a pure **f16 cast** of the official
`nomic-ai/CodeRankEmbed` safetensors, validated equivalent (cosine
0.999998, identical CodeSearchNet MRR@10/Recall@1 — see `tools/quant`).
On **Apple Silicon** it runs on the **Metal GPU** via candle at ~0.57 GB
(the ORT CoreML EP can't accelerate NomicBert); on CPU it uses the int8
ONNX, on CUDA the f32 ONNX. For exact upstream provenance pick the
official f32 weights: `models set-default nomic-ai/CodeRankEmbed`
(~2x RAM on Metal, same embeddings; identical to the default off
Apple-Silicon). `jinaai/jina-embeddings-v2-base-code` is a lighter
code alternative (~0.76 GB). `google/embeddinggemma-300m` (308 M,
int8 ONNX, Matryoshka 128–768, multilingual incl. Russian) is the
strongest non-default on the in-repo 130-query golden corpus and the
only model that lands a top-1 hit there — pick it when the queries
are NL-rich ("how does X work") or non-English; the trade is ~4×
the per-query latency of CodeRankEmbed (rerank off, both). For
multilingual / Russian (or the fastest possible sync of a big repo,
in seconds at <1 GB) use the static `potion-multilingual-128M` — bag-of-token-means (weaker raw
code relevance, but re-rank is **on by default** for it and largely
closes the gap); the onnx encoders are slower on the first sync of a
large repo (incremental after). `models add` registers any other HF
model — set `--query-prefix`/`--doc-prefix`/`--pooling` for its
contract, and `--onnx-file` to pick a specific quantization; its `config.json` architecture is checked at add time
and an unsupported one (LM-head / KV-cache decoder) is rejected up
front.

### Retrieval quality (CodeSearchNet)

One run, `cargo xtask eval --rerank` on CodeSearchNet python/test: a
**5000-function distractor pool**, 200 real docstrings as queries,
each retrieving its own function out of the 5000 (higher = better) — a
large pool is the discriminating setup (a tiny pool saturates the
metric). `base` and the `+rerank` column are the **same 200 queries /
5000-doc pool** (every query reranked) so the delta is honest. `index`
= corpus embed throughput (the dominant sync cost). macOS aarch64:

| model | MRR@10 | R@1 | NDCG@10 | MRR@10 +rerank | index (docs/s) | rerank default |
|-------|-------|-----|---------|----------------|----------------|----------------|
| google/embeddinggemma-300m | **0.938** | **0.910** | **0.949** | **0.942** | ~2 | off |
| sensiarion/CodeRankEmbed-f16 **(default)** | 0.929 | 0.910 | 0.937 | 0.928 | ~21 | off |
| minishlab/potion-base-32M | 0.730 | 0.660 | 0.759 | **0.849** | ~11 000 | on |
| minishlab/potion-multilingual-128M | 0.716 | 0.635 | 0.749 | **0.858** | ~7 600 | on |

`google/embeddinggemma-300m` edges past the default on every quality
metric (+0.009 MRR base, +0.014 MRR with re-rank, +0.012 NDCG) and
matches it on Recall@1 (0.910). On Apple Silicon it runs on the
**Metal GPU via candle (f32)** — about 2.8× the indexing throughput
of the int8 ONNX CPU fallback, and on this repo's 130-query golden
set it also beats the ONNX path on MRR (0.444 vs 0.437). The HF repo
is license-gated; the first sync needs a token with `canReadGatedRepos`
or it falls back to int8 ONNX with a single warn line. Pick
CodeRankEmbed as the speed-balanced default; switch to EmbeddingGemma
when query language is multilingual or NL ("how does X work") rather
than identifier-shaped. The default transformer is markedly more
accurate than the static models; the static models trade ~0.20 MRR
for a **~500× faster index** (the right pick on a large repo or when
sync speed matters). Cross-encoder re-rank
(`ettin-reranker-68m-v1`, ~68M, candle Metal, top-20) is a **major
rescue for the static retrievers** (+0.12–0.14 MRR, +0.17–0.21 R@1 —
they reach ≈ a transformer's top-1) but **~neutral on the SOTA
default** (−0.001 MRR: the bi-encoder already has the gold in the
top-20, the cross-encoder can only reshuffle). So re-rank defaults
**on for the static models, off for CodeRank/jina** — auto, per the
active model; an explicit `[rerank] enabled` overrides. A code-blind
web reranker (ms-marco MiniLM) instead *collapses* this to ~0.56 MRR
— why the default is a code-strong cross-encoder. Reproduce / track:
`cargo xtask eval [--corpus N] [--queries N] [--rerank]`;
`benchmarks/results/effectiveness.jsonl` holds exactly this one
matched run so every comparison in it is honest.

### Selecting a model

Pick by name from the table — one name per model, identified by its
concrete `.onnx` file (no precision knob). Two ways:

```
embedding-search models set-default minishlab/potion-base-32M
embedding-search models list           # shows the table + RAM estimate
```

or edit `~/.embedding-search/config.toml` directly:

```toml
[model]
default = "minishlab/potion-base-32M"
```

Changing the model shifts the index fingerprint → the index
auto-rebuilds on next run (or force it: `embedding-search sync --force`).

#### Per-project override

`models set-default` is global. To pin a model for **one repo only**
(the current working directory):

```
cd /path/to/project
embedding-search models set minishlab/potion-base-32M
```

This writes a minimal `<project>/.embedding-search/config.toml` that is
**deep-merged over** the global `~/.embedding-search/config.toml`
(project keys win; everything you didn't override still comes from
global). Other projects are unaffected; `embedding-search status` shows
`project config:` when an override is active. The model is verified
(download + test embed) before anything is written, and the project
re-indexes on the next sync. Useful on a large repo where the default
heavy transformer is slow to index — a static model (`potion-base-32M`,
or `potion-multilingual-128M` for non-English) syncs the same tree
~150× faster. The MCP server exposes the same switch as a `set_model`
tool (and hints toward it on a large repo), so an agent can do this
itself on the first sync.

### Using any Hugging Face model

**Yes — any HF embedding model works, as long as you give it the
model's *own* ONNX file and its *own* tokenizer files.** The vocab is
not a problem: it travels with `tokenizer.json` (the full vocabulary +
merges). Point at a model and its matching tokenizer and the right
vocab is loaded automatically — there is no shared/global vocab to
clash. The only requirement is an **ONNX export** (this tool runs ONNX
Runtime, not PyTorch).

Most popular embedding models already have an ONNX build on Hugging
Face (look for a `model.onnx` / `onnx/` folder, or a `Xenova/<name>`
mirror).

**Easiest — register it by one command.** `models add` **downloads
the model and runs a test embed right then** — if the repo/URL is bad
or missing files the command fails and **nothing is written to
config** (no broken state to clean up later). On success it's cached
like a built-in, listed in `models list`, **and selected as the active
model** (marked `*`). `--repo` takes a Hugging Face repo id **or** a
full `huggingface.co` URL (paste straight from the browser — it's
canonicalized to the `org/name` id):

```bash
# HF repo id …
embedding-search models add --name bge-small --repo Xenova/bge-small-en-v1.5
# … or the full page URL (with or without /tree/main) — both work
embedding-search models add --name mxbai \
  --repo https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

# onnx-community / large models: weights split into a .onnx_data
# sidecar — fetched automatically. --onnx-file picks an EXACT
# quantization (q4 / q4f16 / bnb4 / uint8 …) by its concrete filename.
# (EmbeddingGemma is now built-in — this pattern is for everything else.)
embedding-search models add --name qwen3 \
  --repo onnx-community/Qwen3-Embedding-0.6B-ONNX --onnx-file model_q4f16.onnx

# …or a direct .onnx URL (the 4 tokenizer files must sit next to it)
embedding-search models add --name my-model \
  --url https://huggingface.co/Org/Model/resolve/main/onnx/model.onnx

embedding-search sync --force   # re-index with the new model
```

Pass `--query-prefix` / `--doc-prefix` if the model needs them
(e.g. `--query-prefix "search_query: " --doc-prefix "search_document: "`
for a nomic-style model). `--onnx-file
model_q4f16.onnx` (or `onnx/model_q4.onnx`) pulls that exact file —
the **sole weight selector** (q4, q4f16, bnb4, uint8, quantized…);
omit it and the loader uses `onnx/model.onnx` with a flat
`model.onnx` fallback, so a repo that ships only one ONNX still works.
Required in the repo: ONNX weights **plus all four tokenizer files**
(`tokenizer.json`, `config.json`, `special_tokens_map.json`,
`tokenizer_config.json`). The `.onnx` graph plus any `.onnx_data`
external-weights sidecar (onnx-community / models >2 GB) are both
fetched. A PyTorch-only repo (no ONNX) or a missing tokenizer file
fails with an explicit message naming the repo and every path tried. A language-model-head export
(`*ForMaskedLM`/`CausalLM`…) is rejected up front (it's not an
embedding model and ALiBi long-context ones OOM at tens of GB).
Stored under `[[custom_model]]`; output dimensions probed at load.

**Model2Vec / `StaticModel`** repos (e.g. `minishlab/potion-*`) are
supported via a built-in static backend — no ONNX/transformer: the
`model.safetensors` token-embedding matrix + tokenizer are loaded and
mean-pooled (+ L2 per the model's `Normalize`). Tiny and fast, lowest
RAM. Just `models add --name potion --repo minishlab/potion-multilingual-128M`.

**Offline / local files — `[model] onnx_path`** (no download):

```toml
[model]
onnx_path        = "/models/bge-small-en" # dir or direct .onnx file
# onnx_query_prefix = "search_query: "    # e.g. nomic; omit for none
# onnx_doc_prefix   = "search_document: " # e.g. nomic; omit for none
```

Same file requirements as above (a directory with `onnx/model.onnx`
or `model.onnx`, plus the four tokenizer files; or the `.onnx` with
them alongside).
`[model] default` is then ignored; swapping the file busts the index
fingerprint → auto-rebuild. Fetch the files yourself with e.g.
`pip install -U "huggingface_hub[cli]" && hf download
Xenova/bge-small-en-v1.5 --local-dir /models/bge-small-en`.

If a model has **only** PyTorch weights, export it once with
[🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model):

```bash
pip install "optimum[exporters]"
optimum-cli export onnx -m mixedbread-ai/mxbai-embed-large-v1 /models/mxbai
# → writes model.onnx + the tokenizer files into /models/mxbai
```

### Using a remote API instead (OpenAI / DeepSeek / LiteLLM)

One command registers **and selects** any OpenAI-compatible
`/embeddings` endpoint — no local model loaded. Each remote is saved
under a `--name`, so you can register several and switch between them
later **by name alone** (no need to re-enter base-url/key/model):

```bash
embedding-search models add-remote --name deepseek \
  --base-url https://api.deepseek.com/v1 \
  --model deepseek-embedding \
  --api-key '$DEEPSEEK_API_KEY'

embedding-search models add-remote --name openai-small \
  --base-url https://api.openai.com/v1 \
  --model text-embedding-3-small --api-key '$OPENAI_API_KEY'

embedding-search models list                 # both listed; active marked *
embedding-search models set-default deepseek  # re-select by name only
embedding-search sync --force
```

`models add-remote` appends a `[[remote_model]]` entry, copies it into
the active `[remote]` block, and sets `provider = "openai"`.
`models set-default <name>` re-selects any registered model — built-in,
`[[custom_model]]`, or `[[remote_model]]` — flipping the provider as
needed. `--dimensions` is optional (probed from a live request if
omitted); add `--query-prefix`/`--doc-prefix` only if the remote
needs them (e.g. a self-hosted nomic-style model).
Works the same for OpenAI, a self-hosted
[LiteLLM](https://github.com/BerriAI/litellm) proxy, TEI, or Ollama —
just change `--base-url`/`--model`. See *External embeddings service*
above for the equivalent hand-edited config.

## How it works

- **Chunking** — tree-sitter AST splitting for Rust, Python, TS/JS, Go, Java,
  C/C++ (functions, classes, impls); header splitting for Markdown; top-level
  key splitting for YAML/TOML; line windows otherwise.
- **Embeddings** — `fastembed` (ONNX Runtime), CoreML on Apple Silicon, CUDA
  with the `cuda` feature, CPU fallback. Chunks embedded in batches.
- **Storage** — `usearch` HNSW index for vectors + SQLite for metadata, both
  file-backed in `.embedding-search/`.
- **Incremental** — a blake3 file hash-tree skips unchanged files
  (mtime+size fast path); within a changed file, each chunk is hashed
  too, so only chunks whose content actually changed are re-embedded —
  unchanged chunks keep their vector, moved ones just get their line
  range refreshed. No file watcher: the MCP server catches the index
  up with a periodic background resync (and one at startup).

### Search result shape

Each result is built for an LLM agent (Module–Class–Function hierarchy,
line-addressable, token-lean — per repo-level RACG research):

```json
{
  "file_path": "crates/core/src/sync.rs",
  "language": "rust",
  "node_type": "function",
  "signature": "pub fn sync<F: FnMut(SyncEvent<'_>)>(",
  "start_line": 313, "end_line": 391,
  "parent":  { "node_type": "impl",     "signature": "impl SyncEngine {", "start_line": 70, "end_line": 480 },
  "prev":    { "node_type": "function", "signature": "fn flush_group(", "start_line": 289, "end_line": 311 },
  "next":    { "node_type": "function", "signature": "pub fn apply_events(", "start_line": 394, "end_line": 420 },
  "content": "…the matched chunk…",
  "score": 0.83
}
```

`parent` / `prev` / `next` are bodyless refs: enough to place the hit in
the architecture and decide whether to open the file or expand, without
spending context on code the agent may not need. Line numbers are
1-based inclusive; byte offsets are not exposed.

## Build from source

```
cargo build --release
# one binary: target/release/embedding-search (CLI; `--mcp` runs the server)
```

## Releasing (maintainers / agents)

The version is hand-duplicated in several places; **never edit them
individually** — run one command, which keeps them in lockstep:

```
cargo xtask bump 0.3.0     # Cargo.toml + mcp.rs handshake +
                           # .claude-plugin/plugin.json "version"
                           # AND its pinned mcp-bin tag
# then:
#  1. add a CHANGELOG.md entry (BREAKING: prefix if applicable)
#  2. cargo build            # refreshes Cargo.lock
#  3. commit, git tag -a v0.3.0 -m v0.3.0
#  4. git push origin main && git push origin v0.3.0
```

Pushing a `v*` tag triggers `.github/workflows/release.yml`
(cargo-dist): builds macOS arm64 + Linux x64/arm64 tarballs and
publishes the GitHub release. mcp-bin downloads from there.

**Why the plugin pins a tag — do not change back to a bare/`@latest`
spec.** mcp-bin caches a bare `owner/repo` (or `@latest`, which 404s)
as the resolved release *forever*, so a `claude plugin update` would
keep running the old MCP binary. `.claude-plugin/plugin.json` therefore
pins `sensiarion/embedding-search@v<version>`; `cargo xtask bump`
advances that tag with the crate version so an updated plugin pulls
the matching server. (Standalone `claude mcp add` / OpenCode users on
the bare spec refresh with `npx mcp-bin expire sensiarion/embedding-search`.)

