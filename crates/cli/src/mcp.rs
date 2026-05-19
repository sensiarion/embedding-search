//! In-process MCP server (`embedding-search --mcp` / `serve`). stdio
//! transport: stdout is the JSON-RPC channel, logs go to stderr.

use anyhow::{Context, Result};
use embedding_search_core::config::LARGE_REPO_FILES;
use embedding_search_core::embedder::Embedder;
use embedding_search_core::{Config, SyncEngine};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use tokio::sync::Semaphore;

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchCodeArgs {
    /// Natural-language query describing the code you want to find.
    pub query: String,
    /// Max results (default 10, max 50).
    #[serde(default)]
    pub limit: Option<u8>,
    /// Optional project-relative directory or file to restrict the
    /// search to (e.g. "crates/core/src" or "src/auth.rs"). Like
    /// Cursor's @folder / @file. Omit to search the whole codebase.
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetModelArgs {
    /// Embedding model to use for THIS project: a built-in or a
    /// globally-registered custom/remote model name. On a large repo a
    /// static model (e.g. `minishlab/potion-base-32M`) indexes in a
    /// fraction of the time of a heavy transformer.
    pub model: String,
}

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

/// The live engine, swappable so `set_model` can switch the project's
/// model without restarting the server. Tools take a cheap `Arc`
/// snapshot under a short read lock; `set_model` replaces it under the
/// write lock. The background resync loop re-reads it every tick, so a
/// swap takes effect there too.
type SharedEngine = Arc<RwLock<Arc<SyncEngine>>>;

/// Single-flight gate for sync. The server fires a resync from four
/// independent places (startup, the periodic loop, the `sync_codebase`
/// tool, `set_model`); `SyncEngine::sync` has no internal mutual
/// exclusion, so without this a slow first index on a big repo would
/// overlap the periodic tick (or a tool call) and run several full
/// sync pipelines at once — each with its own scan window + embed
/// tensors + model forward buffers, i.e. N× the bounded memory the
/// CLI (one sync, then process exit) ever uses. One permit ⇒ at most
/// one sync at a time, same memory budget as the CLI.
type SyncGate = Arc<Semaphore>;

#[derive(Clone)]
struct Server {
    engine: SharedEngine,
    project_dir: PathBuf,
    gate: SyncGate,
}

fn internal(msg: String) -> ErrorData {
    ErrorData::internal_error(msg, None)
}

/// One-line nudge when a heavy transformer model is indexing a big
/// repo: a static model is far faster for the same tree. `None` when
/// already static or the repo is small enough not to matter.
fn large_repo_hint(eng: &SyncEngine, files: i64) -> Option<String> {
    (files > LARGE_REPO_FILES && !eng.model_is_static()).then(|| {
        format!(
            "Large repo ({files} files) on heavy model '{}'. For much \
             faster (re)indexing call set_model with \
             'minishlab/potion-base-32M' (static; re-indexes once).",
            eng.configured_model()
        )
    })
}

/// Attach `hint` to a serializable payload without a wrapper type.
fn with_hint<T: serde::Serialize>(payload: &T, hint: Option<String>) -> String {
    let mut v = serde_json::to_value(payload).unwrap_or(serde_json::Value::Null);
    if let (Some(h), Some(obj)) = (hint, v.as_object_mut()) {
        obj.insert("hint".into(), serde_json::Value::String(h));
    }
    serde_json::to_string(&v).unwrap_or_else(|_| "{}".into())
}

#[tool_router]
impl Server {
    fn new(engine: Arc<SyncEngine>, project_dir: PathBuf) -> Self {
        Self {
            engine: Arc::new(RwLock::new(engine)),
            project_dir,
            gate: Arc::new(Semaphore::new(1)),
        }
    }

    /// Cheap snapshot of the current engine for a tool call.
    fn current(&self) -> Arc<SyncEngine> {
        self.engine.read().unwrap().clone()
    }

    #[tool(
        description = "Semantic code search across the current project. Use this FIRST instead of grep for: conceptual questions (\"how does auth work?\"), finding related code across files, exploring an unfamiliar codebase, identifying patterns/relationships, and queries where the exact terminology is unknown. (Keep using grep for exact strings: error messages, a known symbol name, imports, TODO/FIXME.) Optional `path` scopes the search to a directory or file. Returns ranked chunks; each has file_path, language, node_type, signature (def line), 1-based start_line/end_line, content, score, an optional parent (enclosing impl/class/module) and prev/next sibling refs (no body). Use start_line/parent to locate code and decide whether to open the file; expand via prev/next only when needed."
    )]
    async fn search_code(
        &self,
        Parameters(a): Parameters<SearchCodeArgs>,
    ) -> Result<String, ErrorData> {
        let engine = self.current();
        let limit = a.limit.unwrap_or(10).clamp(1, 50) as usize;
        let query = a.query;
        let scope = a.path;
        let res =
            tokio::task::spawn_blocking(move || engine.search(&query, limit, scope.as_deref()))
                .await
                .map_err(|e| internal(e.to_string()))?
                .map_err(|e| internal(e.to_string()))?;
        Ok(serde_json::to_string(&res).unwrap_or_else(|_| "[]".into()))
    }

    #[tool(
        description = "Force a full incremental re-index of the codebase. The index is normally kept fresh automatically; call this only after large external changes."
    )]
    async fn sync_codebase(&self, Parameters(_): Parameters<NoArgs>) -> Result<String, ErrorData> {
        let engine = self.current();
        // Explicit client request → wait for the gate (don't skip) so
        // it can't run alongside a background resync.
        let _permit = self
            .gate
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| internal(e.to_string()))?;
        let stats = tokio::task::spawn_blocking({
            let e = engine.clone();
            move || e.sync(false, |_| {})
        })
        .await
        .map_err(|e| internal(e.to_string()))?
        .map_err(|e| internal(e.to_string()))?;
        let files = engine.status().map(|s| s.files).unwrap_or(0);
        Ok(with_hint(&stats, large_repo_hint(&engine, files)))
    }

    #[tool(
        description = "Index status: indexed file/chunk counts, vector count, active model, last sync time and whether the index is stale. May include a `hint` when a faster model is advisable for this repo."
    )]
    async fn get_index_status(
        &self,
        Parameters(_): Parameters<NoArgs>,
    ) -> Result<String, ErrorData> {
        let engine = self.current();
        let status = engine.status().map_err(|e| internal(e.to_string()))?;
        let hint = large_repo_hint(&engine, status.files);
        Ok(with_hint(&status, hint))
    }

    #[tool(
        description = "Switch the embedding model for THIS project only (writes <project>/.embedding-search/config.toml, overriding the global default; other projects are unaffected). Use this on a large repo before the first index when get_index_status / sync_codebase returns a hint: a static model like 'minishlab/potion-base-32M' indexes far faster than the default heavy transformer. The model is verified, then the index is rebuilt in the background (search stays available on the old index until it finishes)."
    )]
    async fn set_model(
        &self,
        Parameters(a): Parameters<SetModelArgs>,
    ) -> Result<String, ErrorData> {
        let project_dir = self.project_dir.clone();
        let model = a.model.clone();
        // Validate + persist + rebuild on a blocking thread (model
        // download / embed probe + index open). Nothing is written
        // until the model verifies, mirroring the CLI `set`.
        let new_engine = tokio::task::spawn_blocking(move || -> Result<Arc<SyncEngine>> {
            let mut cfg = Config::load_for_project(&project_dir).context("load config")?;
            cfg.select_model(&model)?;
            Embedder::new(&cfg)
                .with_context(|| format!("'{model}' failed verification — nothing saved"))?;
            cfg.save_project_override(&project_dir)
                .context("save project override")?;
            // Reuse the verified, model-selected cfg (a fresh
            // load_for_project would deserialize to the same thing —
            // global merged with the override just written). The
            // fingerprint shifts, so SyncEngine::new wipes the stale
            // index; the resync below refills it.
            Ok(Arc::new(
                SyncEngine::new(project_dir.clone(), cfg).context("open engine")?,
            ))
        })
        .await
        .map_err(|e| internal(e.to_string()))?
        .map_err(|e| internal(e.to_string()))?;

        let model_name = new_engine.configured_model().to_string();
        *self.engine.write().unwrap() = new_engine.clone();
        // Rebuild the index now so search reflects the new model soon
        // (runs detached; the stale vectors were wiped on open). Gated:
        // it waits out any in-flight resync instead of doubling memory.
        tokio::spawn(run_resync(new_engine, "set_model", self.gate.clone(), true));
        Ok(serde_json::json!({
            "ok": true,
            "model": model_name,
            "note": "model set for this project; index rebuilding in \
                     the background — search uses the previous index \
                     until it completes",
        })
        .to_string())
    }
}

/// Tool-usage guide, embedded at compile time. This is the SINGLE
/// source: the same file is symlinked as the plugin's Skill
/// (`skills/embedding-search/SKILL.md`), which requires YAML
/// frontmatter — stripped here so the MCP client sees only the guide.
const SKILL: &str = include_str!("SKILL.md");

fn skill_instructions() -> &'static str {
    match SKILL
        .strip_prefix("---\n")
        .and_then(|r| r.split_once("\n---\n"))
    {
        Some((_front, body)) => body.trim_start(),
        None => SKILL,
    }
}

#[tool_handler(name = "embedding-search", version = "0.2.6")]
impl ServerHandler for Server {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::new(
            "embedding-search",
            env!("CARGO_PKG_VERSION"),
        ))
        .with_instructions(skill_instructions().to_string())
    }
}

/// One hash-incremental resync on a blocking thread, behind the
/// single-flight `gate` (see [`SyncGate`]). Shared by the startup
/// sync, the periodic loop and `set_model` so their log format never
/// drifts. `wait`: a `set_model` rebuild must eventually run, so it
/// waits for the gate; startup/periodic just skip when one is already
/// in progress (the running/next sync covers the same tree).
async fn run_resync(eng: Arc<SyncEngine>, label: &'static str, gate: SyncGate, wait: bool) {
    let _permit = if wait {
        match gate.acquire_owned().await {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("{label} resync gate closed: {e}");
                return;
            }
        }
    } else {
        let Ok(p) = gate.try_acquire_owned() else {
            tracing::debug!("{label} resync skipped — another sync in progress");
            return;
        };
        p
    };
    match tokio::task::spawn_blocking(move || eng.sync(false, |_| {})).await {
        Ok(Ok(s)) if s.files_indexed + s.files_deleted > 0 => {
            tracing::info!(
                "{label} resync: {} indexed, {} deleted",
                s.files_indexed,
                s.files_deleted
            );
        }
        Ok(Ok(_)) => {}
        Ok(Err(e)) => tracing::warn!("{label} resync failed: {e}"),
        Err(e) => tracing::warn!("{label} resync join: {e}"),
    }
}

async fn serve_async() -> Result<()> {
    let project_dir = match std::env::var("EMBEDDING_SEARCH_PROJECT_DIR") {
        Ok(p) => PathBuf::from(p),
        Err(_) => std::env::current_dir().context("resolve current dir")?,
    };
    let project_dir = std::fs::canonicalize(&project_dir).unwrap_or(project_dir);

    let config = Config::load_for_project(&project_dir).context("load config")?;
    // Catch-up cadence for the background resync loop. Captured before
    // `config` moves into the engine.
    let resync_minutes = config.sync.resync_interval_minutes.max(1) as u64;

    tracing::info!("opening index for {}", project_dir.display());
    let dir = project_dir.clone();
    let engine = tokio::task::spawn_blocking(move || SyncEngine::new(dir, config))
        .await
        .context("join")?
        .context("open engine")?;
    let server = Server::new(Arc::new(engine), project_dir);

    // Hash-incremental startup sync in the background (near-instant on
    // a clean tree — no staleness timer, no full re-index).
    tokio::spawn(run_resync(
        server.current(),
        "startup",
        server.gate.clone(),
        false,
    ));

    // Periodic background resync (no file watcher: a file the agent
    // just edited is already in its context — the index only needs to
    // catch up for the *next* agent run). A no-edit tick is ~free.
    // Re-reads the shared engine each tick so a `set_model` swap is
    // honored without restarting the loop.
    {
        let shared = server.engine.clone();
        let gate = server.gate.clone();
        tokio::spawn(async move {
            let period = std::time::Duration::from_secs(resync_minutes * 60);
            loop {
                tokio::time::sleep(period).await;
                let eng = shared.read().unwrap().clone();
                run_resync(eng, "background", gate.clone(), false).await;
            }
        });
    }

    let service = server.serve(stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// Block on the stdio MCP server. The tracing subscriber is already
/// installed by `main` (stderr only — stdout is the JSON-RPC channel).
pub fn run() -> Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("tokio runtime")?;
    rt.block_on(serve_async())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skill_instructions_strips_plugin_frontmatter() {
        // SKILL.md is the single source; it carries plugin-Skill YAML
        // frontmatter that must not leak into MCP instructions.
        assert!(SKILL.starts_with("---\n"));
        let body = skill_instructions();
        assert!(body.starts_with("# embedding-search"));
        assert!(!body.contains("description:"));
    }
}
