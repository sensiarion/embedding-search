# embedding-search

Fast semantic code search for Claude Code and the terminal. A persistent MCP
server indexes your codebase into embeddings, keeps the index fresh with a
periodic background resync, and exposes a `search_code` tool that Claude uses
instead of grep for conceptual lookups. Also usable directly as a CLI.

## Install

One self-contained binary — `embedding-search` is the CLI, and
`embedding-search --mcp` runs the stdio MCP server. Distributed via
[mcp-bin](https://github.com/sensiarion/mcp-bin):
`npx -y mcp-bin sensiarion/embedding-search --mcp` fetches the latest
GitHub release on first launch (cached for reuse — `npx mcp-bin expire
sensiarion/embedding-search` to refresh, or pin with `@vX.Y.Z`).
Editors spawn it with the workspace as cwd, so it indexes the open
project automatically.

Prebuilt releases: **macOS arm64, Linux x86_64, Linux arm64**. Intel
macOS and Windows: build from source (`cargo build --release` — see
*Build from source*).

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

**Cross-encoder re-rank** (top `top_n` fused candidates re-scored
jointly `(query, passage)`, then reordered): default is model-driven —
ON for the static potion models (big quality rescue), OFF for the
SOTA bi-encoders (neutral, not worth the latency). Override with
`[rerank] enabled`. Ordering-only, so toggling never triggers a
rebuild. The default `cross-encoder/ettin-reranker-68m-v1` (~68M, the
smallest strong code-capable reranker) runs on the Metal GPU via
candle on Apple Silicon; off Apple Silicon the candle path is
unavailable and the int8 ONNX export is encoder-only, so re-rank is a
no-op there for now. Quality measured via `cargo xtask eval --rerank`
— see *Retrieval quality* below.

**Hash-incremental sync** (no file watcher): a file whose mtime+size
are unchanged is skipped before any read/hash/parse (blake3 file
hash-tree, content hash as the tiebreaker on mtime-only changes), so a
clean resync is near-instant. The **MCP server** does a startup sync
then a periodic background resync every `resync_interval_minutes`; the
**CLI** `search` resyncs at most once per that interval (`--no-sync`
skips, `embedding-search status` shows when a resync is due). `--in
DIR|FILE` scopes a search. Small indexes (< `[search] exact_below`)
use exact brute-force search instead of the approximate HNSW graph.

Model cache: `~/.embedding-search/models` (override: `[paths]
cache_dir`). Per-project index: `<project>/.embedding-search/`
(git-ignored automatically).

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

Built-in models are pulled from **Hugging Face** and cached locally.
Each declares an *architecture* and a fixed query/doc-prefix + pooling
contract (applied automatically):

- **static** — [Model2Vec](https://github.com/MinishLab/model2vec)
  token-embedding matrix (no transformer): tiny RAM, very fast. The
  two `potion-*` models.
- **onnx** — transformer encoder, weights from a pinned `.onnx` file.
  On **Apple Silicon** supported models swap to a candle Metal backend
  (CodeRankEmbed, EmbeddingGemma); elsewhere the ONNX export is used.

Columns below: `code` / `ml` = code-retrieval / multilingual rating
(★★★★★ = strong, ★★ = English-centric). `peak RSS` = resident memory
indexing a ~12k-chunk repo (weights + tokenizer + runtime overhead).
Large files are never skipped — chunks are hard-capped and streamed in
byte-bounded batches.

| model | arch | dim | code | ml | pool | peak RSS |
|-------|------|-----|------|----|------|----------|
| sensiarion/CodeRankEmbed-f16 **(default)** | candle f16 Metal / onnx int8 CPU | 768 | ★★★★★ | ★★ | cls | ~0.57 GB |
| nomic-ai/CodeRankEmbed | candle f32 Metal / onnx int8 CPU | 768 | ★★★★★ | ★★ | cls | ~1.1 GB |
| jinaai/jina-embeddings-v2-base-code | onnx (int8) | 768 | ★★★★★ | ★★ | mean | ~0.76 GB |
| google/embeddinggemma-300m | candle f32 Metal / onnx int8 CPU | 768 | ★★★★ | ★★★★★ | mean | ~1.6 GB |
| minishlab/potion-multilingual-128M | static | 256 | ★★★ | ★★★★★ | mean | ~0.85 GB |
| minishlab/potion-base-32M | static | 512 | ★★★ | ★★ | mean | ~0.36 GB |

**When to pick what:**

- `sensiarion/CodeRankEmbed-f16` **(default)** — SOTA code retrieval,
  CLS-pooled. f16 cast of `nomic-ai/CodeRankEmbed`, validated
  equivalent (cosine 0.999998, identical CodeSearchNet metrics — see
  `tools/quant`). Apple Silicon runs Metal/candle; CPU uses int8 ONNX,
  CUDA uses f32 ONNX.
- `nomic-ai/CodeRankEmbed` — official f32 weights (~2× RAM on Metal,
  same embeddings). Pick for exact upstream provenance.
- `google/embeddinggemma-300m` — strongest on NL-rich or non-English
  queries; multilingual (incl. Russian), Matryoshka 128–768. Trade:
  ~4× per-query latency vs CodeRankEmbed.
- `jinaai/jina-embeddings-v2-base-code` — lighter code alternative.
- `potion-multilingual-128M` / `potion-base-32M` — bag-of-token-means
  static models. Weak raw code relevance but rerank **defaults on** to
  largely close the gap. Pick when sync speed matters (~500× faster
  index) or for multilingual on a large repo.

Adding more models: `models add --repo ORG/M [--onnx-file F]
[--query-prefix .. --doc-prefix .. --pooling mean|cls|last-token]`
— see *Using any Hugging Face model* below. An LM-head /
KV-cache-decoder export is rejected at add time.

### Retrieval quality (CodeSearchNet)

`cargo xtask eval --rerank` on CodeSearchNet python/test: 200 real
docstrings as queries against a 5000-function distractor pool, each
retrieving its own function. `index` = corpus embed throughput on
Apple Silicon (the dominant sync cost):

| model | MRR@10 | R@1 | NDCG@10 | MRR@10 +rerank | index (docs/s) | rerank default |
|-------|-------|-----|---------|----------------|----------------|----------------|
| google/embeddinggemma-300m | **0.940** | **0.915** | **0.951** | **0.944** | ~12 | off |
| sensiarion/CodeRankEmbed-f16 **(default)** | 0.929 | 0.910 | 0.937 | 0.928 | ~21 | off |
| minishlab/potion-base-32M | 0.730 | 0.660 | 0.759 | **0.849** | ~11 000 | on |
| minishlab/potion-multilingual-128M | 0.716 | 0.635 | 0.749 | **0.858** | ~7 600 | on |

Takeaways:

- EmbeddingGemma edges past the default on every metric (+0.011 MRR
  base, +0.014 NDCG, +0.005 R@1) at ~half the indexing throughput.
  Numbers are the **candle Metal f32 path** (Google's model card
  warns activations don't survive fp16; benching confirmed the cast
  collapses MRR ~75%, so the backbone stays at native f32); the int8
  ONNX CPU fallback (off-Mac, headless host, or gated-download
  failure) scores ~0.002 MRR / ~0.005 R@1 lower and indexes ~6×
  slower. The HF repo
  is license-gated — first sync needs a token with `canReadGatedRepos`
  or it falls back to int8 ONNX with one warn line.
- The static `potion-*` models trade ~0.20 MRR for a **~500× faster
  index** — the right pick on a very large repo. Cross-encoder rerank
  closes most of the quality gap (+0.12–0.14 MRR, reaching ≈ a
  transformer's top-1), which is why rerank defaults **on** for them.
- Rerank is **~neutral** on the SOTA bi-encoders (−0.001 MRR) — the
  gold is already in the top-20, so it can only reshuffle. Default
  **off** for CodeRank/jina. A code-blind web reranker (e.g. ms-marco
  MiniLM) collapses these to ~0.56 MRR, hence the code-strong
  `ettin-reranker-68m-v1` default.

Reproduce: `cargo xtask eval [--corpus N] [--queries N] [--rerank]`;
`benchmarks/results/effectiveness.jsonl` holds this matched run.

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

Any HF embedding model works as long as it ships an **ONNX export**
plus its four tokenizer files (this tool runs ONNX Runtime, not
PyTorch). Look for a `model.onnx` / `onnx/` folder, or a `Xenova/<name>`
mirror.

`models add` downloads the model and runs a test embed before writing
anything to config — a bad repo/URL fails cleanly. On success it's
cached like a built-in, listed in `models list`, and selected as
active. `--repo` takes a HF repo id **or** a full `huggingface.co`
URL (canonicalized to `org/name`):

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
(e.g. `"search_query: "` / `"search_document: "` for nomic-style).
`--onnx-file` is the **sole weight selector** (q4, q4f16, bnb4,
uint8, quantized…) — omit it and the loader uses `onnx/model.onnx`
with a flat `model.onnx` fallback. The repo must ship the four
tokenizer files (`tokenizer.json`, `config.json`,
`special_tokens_map.json`, `tokenizer_config.json`); `.onnx_data`
external-weights sidecars (>2 GB models) are fetched automatically.
A PyTorch-only repo or a missing tokenizer fails with an explicit
message. LM-head exports (`*ForMaskedLM` / `CausalLM` / …) are
rejected up front. Stored under `[[custom_model]]`; output dimensions
probed at load.

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
- **Embeddings** — `fastembed` (ONNX Runtime) with CoreML on Apple Silicon,
  CUDA via the `cuda` feature, CPU fallback. Apple Silicon swaps to a candle
  Metal backend for supported encoders (see *Models*).
- **Storage** — `usearch` HNSW index for vectors + SQLite for metadata,
  file-backed in `.embedding-search/`. See *Config* for the incremental sync
  details.

### Search result shape

Each result is built for an LLM agent — Module–Class–Function
hierarchy, line-addressable, token-lean:

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

`parent` / `prev` / `next` are bodyless refs: enough to place the hit
in the architecture and decide whether to open the file or expand,
without spending context on code the agent may not need. Line numbers
are 1-based inclusive.

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

