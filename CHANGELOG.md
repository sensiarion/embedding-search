# Changelog

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
