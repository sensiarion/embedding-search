/**
 * OpenCode plugin — embedding-search auto-use.
 *
 * OpenCode has no SessionStart / UserPromptSubmit hook like Claude Code.
 * The closest equivalent is `chat.message`, which fires for each user
 * message before it goes to the model. We prepend a short directive so
 * the model prefers the embedding-search semantic-search tool over
 * grep/find — the OpenCode counterpart to the Claude Code plugin hooks.
 *
 * Install (pick one):
 *   - global:  copy this file to  ~/.config/opencode/plugin/
 *   - per-repo: copy to  .opencode/plugin/  in your project
 * Then restart OpenCode.
 */
export const PreferEmbeddingSearch = async () => {
  const REMINDER =
    "EMBEDDING-SEARCH ACTIVE: for code exploration (how does X work, " +
    "where is Y, what handles Z, cross-file/unfamiliar areas) prefer the " +
    "embedding-search semantic search tool over grep/find. Use grep/find " +
    "only for exact known strings (symbol name, error message, import, regex)."

  return {
    "chat.message": async (_input, { parts }) => {
      parts.unshift({ type: "text", text: REMINDER })
    },
  }
}
