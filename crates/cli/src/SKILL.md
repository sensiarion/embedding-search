---
name: embedding-search
description: Semantic (meaning-based) code search over the current project via the embedding-search MCP. Use for conceptual/relationship code questions ("how does auth work", "where is X enforced") instead of grep/find; grep only for exact known strings.
---

# embedding-search — semantic code search

This server gives you **meaning-based** code search over the current
project. Prefer it over grep whenever the *concept* matters more than
an exact string.

## Use `search_code` for

- Conceptual questions — "how does auth work?", "where is rate limiting
  enforced?"
- Finding related code spread across multiple files
- Exploring an unfamiliar codebase ("what handles billing?")
- Identifying patterns / relationships between components
- Queries where you don't know the exact identifier or wording

## Keep using grep for

Exact strings: a known function/type name, an error message, import
lines, `TODO`/`FIXME`, regex. Semantic search and grep are
complementary — semantic to understand, grep for precision.

## Scoping

Pass `path` to restrict results to a directory or file
(project-relative), e.g. `path: "crates/core/src"` or
`path: "src/auth.rs"`. Omit it to search the whole codebase. Scope
when you already know the relevant area — it sharpens results.

## Reading results

Each result is ranked by `score` and carries:

- `file_path`, `start_line`–`end_line` (1-based) — open exactly here
- `signature` — the def line; often enough to answer without opening
- `node_type`, `language`
- `parent` — enclosing impl/class/module (architectural context)
- `prev` / `next` — adjacent chunk refs (no body)

Workflow: read `signature` + `parent` first; open the file at
`start_line` only if you need the body; follow `prev`/`next` to widen
context instead of re-searching.

## Index freshness (you don't manage this)

The index auto-syncs on startup and re-syncs periodically in the
background (hash-incremental) — results reflect the recent tree.
`sync_codebase` forces an immediate catch-up after large external
changes (e.g. a branch switch); `get_index_status` reports health.
You normally never need either — just call `search_code`.
