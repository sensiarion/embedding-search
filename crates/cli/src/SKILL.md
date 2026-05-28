---
name: embedding-search
description: Semantic (meaning-based) code search over the current project. Use it for any code exploration or where/how/what questions about an unfamiliar codebase — prefer over grep/find. Grep only for exact known strings. Also use inside subagents (Explore, general-purpose, Task), where the plugin's hook-injected reminders don't reach.
---

# embedding-search — semantic code search

This server gives you **meaning-based** code search over the current
project. Prefer it over grep whenever the *concept* matters more than
an exact string.

## Subagent note

The plugin's SessionStart and UserPromptSubmit hooks inject reminders
into the **top-level** session only — they do not reach subagent
contexts (Explore, general-purpose, Task). Inside a subagent this
skill is your standing directive: reach for `search_code` first for
any code exploration.

## Loading the tool (at most once per session)

`mcp__embedding-search__search_code` may appear under "deferred tools"
with its schema not loaded — direct calls then fail with
`InputValidationError`. Procedure:

1. Scan the available-tools / deferred-tools list in the system
   reminders for the tool name.
2. **Listed as deferred** → load once with
   `ToolSearch select:mcp__embedding-search__search_code`, then call
   the tool normally. Schema persists for the session.
3. **Already directly callable** → just call it. No `ToolSearch`.
4. **Not in either list** → the MCP is not wired up in this project.
   Do NOT call `ToolSearch` — it returns
   `No matching deferred tools found` and the retry is wasted. Use
   grep/find for this session.

Never call `ToolSearch` for this tool more than once per session: if
the first attempt did not surface it, neither will the second.

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
- `prev` / `next` — adjacent chunk refs: `node_type`, `signature`,
  and `start_line`/`end_line`. **No body** (kept token-lean on
  purpose). The neighbor is in the **same file** (`file_path`) — to
  read it, open that file at the neighbor's `start_line..end_line`
  with your file-read tool. Do this only when the hit alone is
  insufficient; usually the `signature` already tells you whether the
  neighbor is relevant.

Workflow: read `signature` + `parent` first; open `file_path` at the
hit's `start_line` only if you need its body; to widen context, read
the same file at `prev`/`next`'s line range instead of re-searching.

## Index freshness (you don't manage this)

The index auto-syncs on startup and re-syncs periodically in the
background (hash-incremental) — results reflect the recent tree.
`sync_codebase` forces an immediate catch-up after large external
changes (e.g. a branch switch); `get_index_status` reports health.
You normally never need either — just call `search_code`.

## Large repos: `set_model`

The default model is a heavy transformer (best results). On a large
repo its first index is slow. When `get_index_status` or
`sync_codebase` returns a `hint`, call `set_model` with a static model
(`minishlab/potion-base-32M`, English, or
`minishlab/potion-multilingual-128M` for non-English incl. Russian) —
it indexes the same tree in a fraction of the time. This is a
per-project override (other projects keep the default); the index
rebuilds in the background and search keeps working on the old index
meanwhile. Only switch on a hint or explicit user request — don't
downgrade a small repo that already indexes fast.
