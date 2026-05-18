#!/usr/bin/env node
// embedding-search — UserPromptSubmit hook (cross-platform Node).
//
// Per-turn reinforcement. The SessionStart directive is injected once and
// drifts out of attention after many turns / context compaction (same
// failure caveman hit with SessionStart-only injection). One short line
// every prompt keeps the preference live without meaningful token cost.
//
// The hook receives JSON on stdin; we don't need it (the reminder is
// unconditional). Emit the additionalContext envelope Claude Code injects
// before the model sees the user's prompt.

let input = "";
process.stdin.on("data", (chunk) => {
  input += chunk;
});
process.stdin.on("end", () => {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "UserPromptSubmit",
        additionalContext:
          "EMBEDDING-SEARCH ACTIVE: for code exploration prefer " +
          "mcp__embedding-search__search_code over grep/find. If it is " +
          "deferred, load it first via ToolSearch " +
          "select:mcp__embedding-search__search_code. grep/find only for " +
          "exact strings.",
      },
    })
  );
});
