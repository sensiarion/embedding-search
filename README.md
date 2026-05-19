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
embedding-search models set-default <name>       switch model (re-index)
embedding-search models add --name X --repo ORG/M [--precision P|--onnx-file F] [--query-prefix .. --doc-prefix .. --pooling mean|cls|last-token | --e5_prefix] | --url URL
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
default   = "nomic-ai/CodeRankEmbed"          # SOTA code encoder (int8)
precision = "fp16"                            # fp16 | int8 | full (ONNX models only)
max_length = 512                              # token cap; keep ≈ max_chunk_bytes/4
# onnx_path = "/path/to/model-dir"            # custom local ONNX (see below)
# onnx_query_prefix = "query: "               # e.g. e5; omit for none
# onnx_doc_prefix   = "passage: "             # e.g. e5; omit for none

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
enabled = false               # opt-in cross-encoder re-rank (off = previous behavior)
# model = "Xenova/bge-reranker-base"   # int8 ONNX cross-encoder
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

**Cross-encoder re-rank** (opt-in, `[rerank] enabled = true`): the top
`top_n` fused candidates are re-scored jointly `(query, passage)` by a
small int8 reranker and reordered before truncation — sharper top
results when recall is already good. Off by default: no second model
download, no added latency, and results are exactly the pre-rerank
behavior. It does not affect the index (ordering only), so toggling it
never triggers a rebuild.

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
# query_prefix = "query: "    # e.g. e5 remote; omit for OpenAI/DeepSeek
# doc_prefix   = "passage: "  # e.g. e5 remote; omit for OpenAI/DeepSeek
```

Batching uses the OpenAI `input` array (`batch_size` texts/request);
requests run on a bounded pool of `concurrency` workers, order
preserved. On startup one probe request validates connectivity + auth
and resolves `dimensions` (a mismatch vs. a configured value is a hard
error). Changing remote model/endpoint triggers a one-time re-index.

## Models

Quality: ★ = code relevance / multilingual. RAM≈ = resident memory
(weights at the chosen precision + ~350MB ONNX Runtime/tokenizer
overhead). Large files are never skipped — chunks are hard-capped and
streamed in byte-bounded batches.

`ml` rating covers non-English (incl. Russian): ★★★★★ = full, ★★ = English-centric.

Built-in models are pulled (and cached) from **Hugging Face**. Each
declares an *architecture* that decides how it is loaded:

- **static** — [Model2Vec](https://github.com/MinishLab/model2vec)
  token-embedding matrix (no transformer/ONNX): tiny RAM, very fast,
  no precision knob. The default + `potion-base-32M`.
- **onnx** — transformer encoder from an HF ONNX repo, mean-pooled;
  the `fp16` / `int8` precision knob applies. The e5 / nomic models,
  and jina-code (pinned to its int8 export).
- **fastembed** — encoder served by fastembed's bundled ONNX (a
  fallback arch; no built-in currently needs it).

`peak RSS` below is measured indexing a real ~12k-chunk repo (auto
per-model batch); `sync` is that first full sync (incremental after).

Each model has a fixed input/output **contract** (query/doc prefix +
pooling) — applied automatically; `pool` = mean / cls. `peak RSS` is
measured indexing a real ~12k-chunk repo (auto per-model batch).

| model | arch | dim | code | ml | pool | peak RSS |
|-------|------|-----|------|----|------|----------|
| nomic-ai/CodeRankEmbed **(default)** | onnx (int8, CPU) | 768 | ★★★★★ | ★★ | cls | ~0.7 GB |
| jinaai/jina-embeddings-v2-base-code | onnx (int8) | 768 | ★★★★★ | ★★ | mean | ~0.76 GB |
| minishlab/potion-multilingual-128M | static | 256 | ★★★ | ★★★★★ | mean | ~0.85 GB |
| minishlab/potion-base-32M | static | 512 | ★★★ | ★★ | mean | ~0.36 GB |
| intfloat/multilingual-e5-small | onnx | 384 | ★★★ | ★★★★★ | mean | ~1.1 GB |
| intfloat/multilingual-e5-base | onnx | 768 | ★★★ | ★★★★★ | mean | ~1.4 GB |
| intfloat/multilingual-e5-large | onnx | 1024 | ★★★ | ★★★★★ | mean | ~2.2 GB |
| nomic-ai/nomic-embed-text-v1.5 | onnx | 768 | ★★★★ | ★★★ | mean | ~0.8 GB |
| Snowflake/snowflake-arctic-embed-m-v2.0 | onnx | 768 | ★★★ | ★★★★★ | cls | ~1.2 GB |
| jamie8johnson/e5-base-v2-code-search | onnx (f32, CPU) | 768 | ★★★★ | ★★ | mean | ~1.1 GB |

**The default `nomic-ai/CodeRankEmbed`** is SOTA code retrieval (int8
ONNX, CLS-pooled, CPU-pinned — its NomicBert export is slow under
CoreML/CUDA). `jinaai/jina-embeddings-v2-base-code`
is a lighter code alternative (~0.76 GB). For multilingual / Russian
use an e5 model or the static `potion-multilingual-128M` (fastest, full
sync of a big repo in seconds at <1 GB). The static models are
bag-of-token-means (weaker code relevance) but cheap; the onnx encoders
are slower on the first sync of a large repo (incremental after).
`models add` registers any other HF model — set
`--query-prefix`/`--doc-prefix`/`--pooling` (or `--e5_prefix`) for its
contract; its `config.json` architecture is checked at add time and an
unsupported one (LM-head / KV-cache decoder) is rejected up front.

(Qwen3-Embedding decoder models need a candle backend that does not
exist yet.)

### Selecting a model

Pick by name from the table — there is one name per model; precision
is **not** encoded in the name, it's a separate config knob (same name,
different ONNX weights). Two ways:

```
embedding-search models set-default intfloat/multilingual-e5-small
embedding-search models list           # shows table + active precision/RAM
```

or edit `~/.embedding-search/config.toml` directly:

```toml
[model]
default   = "intfloat/multilingual-e5-small"
precision = "int8"
```

Changing model or precision shifts the index fingerprint → the index
auto-rebuilds on next run (or force it: `embedding-search sync --force`).

### Precision

`[model] precision` — e5 family only (loaded as user-defined ONNX from
the Hugging Face `Xenova/*` repos, which ship `model.onnx` /
`model_fp16.onnx` / `model_quantized.onnx`):

- `fp16` — **default**. ≈½ the RAM of f32, quality loss negligible
  (<0.5% on retrieval). Recommended for every codebase.
- `int8` — ≈¼ RAM. Small quality loss (~1–3% recall); fine for large
  repos / low-RAM machines.
- `full` — f32 reference quality, highest RAM.

The static Model2Vec models (the default + `potion-base-32M`) have no
precision knob — a single f32 matrix; `models list` shows `f32` and
`precision` is ignored for them.

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
embedding-search models add --name e5 --repo https://huggingface.co/Xenova/multilingual-e5-base

# onnx-community / large models: weights split into a .onnx_data
# sidecar — fetched automatically. --precision picks the variant.
embedding-search models add --name gemma \
  --repo onnx-community/embeddinggemma-300m-ONNX --precision int8

# pick an EXACT quantization the precision mapping can't reach
# (q4 / q4f16 / bnb4 / uint8 …) with --onnx-file
embedding-search models add --name qwen3 \
  --repo onnx-community/Qwen3-Embedding-0.6B-ONNX --onnx-file model_q4f16.onnx

# …or a direct .onnx URL (the 4 tokenizer files must sit next to it)
embedding-search models add --name my-model \
  --url https://huggingface.co/Org/Model/resolve/main/onnx/model.onnx

embedding-search sync --force   # re-index with the new model
```

Add `--e5-prefix` if it's an e5-style model. `--precision fp16 | int8
| full` selects which ONNX variant to pull, per model (a big model can
be `int8` without changing the others); omit it to use the global
`[model] precision`. `--onnx-file model_q4f16.onnx` (or
`onnx/model_q4.onnx`) pulls that exact file instead — for repos whose
quantizations (q4, q4f16, bnb4, uint8…) the precision mapping doesn't
cover; it overrides `--precision`.
Required in the repo: ONNX weights **plus all four tokenizer files**
(`tokenizer.json`, `config.json`, `special_tokens_map.json`,
`tokenizer_config.json`). The `.onnx` graph plus any `.onnx_data`
external-weights sidecar (onnx-community / models >2 GB) are both
fetched. For `--repo` the loader tries the requested precision then
falls back to `onnx/model.onnx` / `model.onnx`, so a repo that ships
only full-precision ONNX still works. A PyTorch-only repo (no ONNX) or
a missing tokenizer file fails with an explicit message naming the
repo and every path tried. A language-model-head export
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
# onnx_query_prefix = "query: "           # e.g. e5; omit for none
# onnx_doc_prefix   = "passage: "         # e.g. e5; omit for none
precision        = "fp16"                 # picks onnx/model_fp16.onnx if present
```

Same file requirements as above (a directory with `model.onnx` or
`onnx/model_fp16.onnx` / `model_quantized.onnx` matching `precision`,
plus the four tokenizer files; or the `.onnx` with them alongside).
`[model] default` is then ignored; swapping the file busts the index
fingerprint → auto-rebuild. Fetch the files yourself with e.g.
`pip install -U "huggingface_hub[cli]" && hf download
Xenova/bge-small-en-v1.5 --local-dir /models/bge-small-en`.

If a model has **only** PyTorch weights, export it once with
[🤗 Optimum](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model):

```bash
pip install "optimum[exporters]"
optimum-cli export onnx -m intfloat/e5-small-v2 /models/e5-small-v2
# → writes model.onnx + the tokenizer files into /models/e5-small-v2
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
omitted); add `--e5-prefix` only if the remote serves an e5 model.
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


## TODO

1. [ ] skill/tool for usage
2. [x] installation scripts for claude/cursor
3. [ ] test usage for big model, like https://huggingface.co/majentik/Qwen3-Embedding-8B-ONNX-INT8/tree/main
4. [ ] restrict concurrency to N threads
5. [ ] match with cursor approach https://www.digitalapplied.com/blog/cursor-semantic-search-coding-ai-guide
  - sync every 10 min
  - check latencies and indexes for vector search (for small codebase it's better to not use indexes, cause it's heuristics either way)
  - merkle tree hash tree for file states
  - allow to specify search to be more concrete (args to search concrete directory or concrete file)
  - add guidelines to use semantic search
    - Conceptual questions ("how does auth work?")
    - Finding related code across files
    - Exploring unfamiliar codebases
    - Identifying patterns and relationships
    - Queries with varying terminology