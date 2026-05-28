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
    "Tool: mcp__embedding-search__search_code (meaning-based search over this codebase).\n\n" +
    "## Loading the tool (do this AT MOST ONCE per session)\n" +
    "1. Look at the available-tools / deferred-tools list in your system reminders.\n" +
    "2. If `mcp__embedding-search__search_code` is listed as deferred → call\n" +
    '   ToolSearch ONCE with `select:mcp__embedding-search__search_code`, then\n' +
    "   call the tool. The schema stays loaded for the rest of the session.\n" +
    "3. If the tool name is NOT in either list → the MCP is not installed in this\n" +
    "   project. DO NOT call ToolSearch — it will return `No matching deferred\n" +
    "   tools found` and the retry is pure waste. Fall back to grep/find for\n" +
    "   this session and do not try to load it again.\n" +
    "4. If the tool is already directly callable → just call it. No ToolSearch.\n\n" +
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
