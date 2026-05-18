#!/usr/bin/env bash
# embedding-search — SessionStart hook.
#
# Injects a standing directive so Claude reaches for the semantic search
# MCP (mcp__embedding-search__search_code) instead of grep/find for
# conceptual code exploration. Mirrors how the caveman plugin "heats"
# behavior: full ruleset once at session start, reinforced per-turn by
# prefer-search-turn.sh (SessionStart context drifts after compaction).
#
# Plain stdout is injected as SessionStart context by Claude Code.
set -uo pipefail

cat <<'EOF'
EMBEDDING-SEARCH ACTIVE — semantic code search is the default exploration tool.

Tool: mcp__embedding-search__search_code (meaning-based search over this codebase).
If it appears under deferred tools (schema not loaded), FIRST call ToolSearch
with query "select:mcp__embedding-search__search_code", then call it. Do NOT
fall back to grep/find merely because the schema is not loaded yet — loading
it is one cheap step.

## Use search_code (default for exploration)
- "how does X work", "where is Y", "what handles Z"
- finding related code spread across files / unfamiliar areas
- any conceptual or relationship question where exact wording is unknown

## Keep grep/find only for
- exact known strings: a symbol/type name, an error message, an import line,
  TODO/FIXME, a regex.

## Persistence
ACTIVE EVERY RESPONSE. Do not drift back to grep-first after many turns or
after context compaction. For exploratory questions, search_code goes before
grep/find. Off only if the user says "stop using embedding search".
EOF
