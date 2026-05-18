# embedding-search

Fast semantic code search for Claude Code and the terminal. A persistent MCP
server indexes your codebase into embeddings, keeps the index fresh via a file
watcher, and exposes a `search_code` tool that Claude uses instead of grep for
conceptual lookups. Also usable directly as a CLI.

## Install

One self-contained binary: `embedding-search` is the CLI, and
`embedding-search --mcp` runs the stdio MCP server (no separate
binary). It ships through [mcp-bin](https://github.com/sensiarion/mcp-bin):
`npx -y mcp-bin sensiarion/embedding-search@latest --mcp` auto-resolves
the right binary from GitHub releases on first launch — no pre-install.
Editors spawn it with the workspace as cwd, so it indexes the open
project automatically. (The `--mcp` trailing arg is what flips the
binary into server mode.)

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
  -- npx -y mcp-bin sensiarion/embedding-search@latest --mcp
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
      "command": ["npx", "-y", "mcp-bin", "sensiarion/embedding-search@latest", "--mcp"],
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
embedding-search serve | --mcp                   run MCP server on stdio
embedding-search debug files [path]              list indexed files
embedding-search debug chunks <file> [--path .]  show a file's chunks
embedding-search models list                     built-in + custom models
embedding-search models set-default <name>       switch model (re-index)
embedding-search models add --name X --repo ORG/M | --url URL   add + select a model
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
default   = "intfloat/multilingual-e5-base"  # multilingual + code
precision = "fp16"                            # fp16 | int8 | full
max_length = 512                              # token cap; keep ≈ max_chunk_bytes/4
# onnx_path = "/path/to/model-dir"            # custom local ONNX (see below)
# onnx_e5_prefix = false                      # true if custom model is e5

[backend]
execution_provider = "auto"   # auto | coreml | cuda | cpu
disable_mem_arena  = true     # keep ORT arena off (memory)

[sync]
max_chunk_bytes  = 2048       # hard cap ≈ max_length*4; large files split, never skipped
embed_batch_size = 16
embed_batch_bytes = 262144    # flush a batch at this many bytes
sync_threads     = 0          # scan/hash/parse worker cap; 0 = all cores but one
resync_interval_minutes = 10  # background catch-up cadence (CLI + MCP)
exclude = []

[search]
exact_below = 50000           # < this many vectors → exact (brute) search, not HNSW
```

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
e5_prefix = false         # true only if the remote serves an e5 model
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

All built-in models are pulled (and cached) from **Hugging Face** as
ONNX. The e5 family loads from the [`Xenova`](https://huggingface.co/Xenova)
org — a community account on huggingface.co that re-publishes popular
models exported to ONNX (e.g.
[`Xenova/multilingual-e5-base`](https://huggingface.co/Xenova/multilingual-e5-base)),
which is what makes the fp16 / int8 variants available. Not a GitHub
repo — it's a Hugging Face namespace.

| model | dim | code | ml | RAM≈ f32 / fp16 / int8 |
|-------|-----|------|----|------------------------|
| intfloat/multilingual-e5-base **(default, fp16)** | 768 | ★★★★ | ★★★★★ | ~1462 / ~906 / ~628 MB |
| intfloat/multilingual-e5-small | 384 | ★★★ | ★★★★★ | ~822 / ~586 / ~468 MB |
| intfloat/multilingual-e5-large | 1024 | ★★★ | ★★★★★ | ~2590 / ~1470 / ~910 MB |
| jinaai/jina-embeddings-v2-base-code | 768 | ★★★★★ | ★★ | ~994 MB (f32 only) |
| nomic-ai/nomic-embed-text-v1.5 | 768 | ★★★★ | ★★★ | ~898 MB (f32 only) |

### Selecting a model

Pick by name from the table — there is one name per model; precision
is **not** encoded in the name, it's a separate config knob (same name,
different ONNX weights). Two ways:

```
embedding-search models set-default intfloat/multilingual-e5-large
embedding-search models list           # shows table + active precision/RAM
```

or edit `~/.embedding-search/config.toml` directly:

```toml
[model]
default   = "intfloat/multilingual-e5-large"
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

`jina`/`nomic` have no quantized ONNX in fastembed → always f32
(`precision` ignored, shows `f32` in `models list`).

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

**Easiest — register it by one command.** `models add` downloads and
caches it like a built-in, lists it in `models list`, **and selects it
as the active model** (marked `*`). Pass a Hugging Face repo id **or**
a direct `.onnx` URL:

```bash
# HF repo id (precision-specific ONNX + tokenizer pulled from the repo)
embedding-search models add --name bge-small --repo Xenova/bge-small-en-v1.5

# …or a direct .onnx URL (the 4 tokenizer files must sit next to it)
embedding-search models add --name my-model \
  --url https://huggingface.co/Org/Model/resolve/main/onnx/model.onnx

embedding-search sync --force   # re-index with the new model
```

Add `--e5-prefix` if it's an e5-style model. The required files are
`model.onnx` (precision variant for `--repo`) **plus all four
tokenizer files**: `tokenizer.json`, `config.json`,
`special_tokens_map.json`, `tokenizer_config.json`. If any is missing
the loader fails with an explicit message naming the exact repo/URL
and the missing file. Stored under `[[custom_model]]` in
`config.toml`; output dimensions are probed at load.

**Offline / local files — `[model] onnx_path`** (no download):

```toml
[model]
onnx_path      = "/models/bge-small-en"   # dir or direct .onnx file
onnx_e5_prefix = false                     # true only for e5-style models
precision      = "fp16"                    # picks onnx/model_fp16.onnx if present
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