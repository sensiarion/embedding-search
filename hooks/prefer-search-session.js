#!/usr/bin/env node
// embedding-search — SessionStart hook (cross-platform: Node, not bash,
// so Windows works; Node is already required by the npx MCP launcher).
//
// Injects a standing directive so Claude reaches for the semantic search
// MCP (mcp__embedding-search__search_code) instead of grep/find for
// conceptual exploration. Mirrors how the caveman plugin "heats"
// behavior: full ruleset once at session start, reinforced per-turn by
// prefer-search-turn.js (SessionStart context drifts after compaction).
//
// Plain stdout is injected as SessionStart context by Claude Code.

process.stdout.write(
  "EMBEDDING-SEARCH ACTIVE — semantic code search is the default exploration tool.\n\n" +
    "Tool: mcp__embedding-search__search_code (meaning-based search over this codebase).\n" +
    "If it appears under deferred tools (schema not loaded), FIRST call ToolSearch\n" +
    'with query "select:mcp__embedding-search__search_code", then call it. Do NOT\n' +
    "fall back to grep/find merely because the schema is not loaded yet — loading\n" +
    "it is one cheap step.\n\n" +
    "## Use search_code (default for exploration)\n" +
    '- "how does X work", "where is Y", "what handles Z"\n' +
    "- finding related code spread across files / unfamiliar areas\n" +
    "- any conceptual or relationship question where exact wording is unknown\n\n" +
    "## Keep grep/find only for\n" +
    "- exact known strings: a symbol/type name, an error message, an import line,\n" +
    "  TODO/FIXME, a regex.\n\n" +
    "## Persistence\n" +
    "ACTIVE EVERY RESPONSE. Do not drift back to grep-first after many turns or\n" +
    "after context compaction. For exploratory questions, search_code goes before\n" +
    'grep/find. Off only if the user says "stop using embedding search".'
);
