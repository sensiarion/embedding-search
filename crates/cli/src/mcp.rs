//! In-process MCP server (`embedding-search --mcp` / `serve`). stdio
//! transport: stdout is the JSON-RPC channel, logs go to stderr.

use anyhow::{Context, Result};
use embedding_search_core::{Config, SyncEngine};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::transport::stdio;
use rmcp::{tool, tool_handler, tool_router, ErrorData, ServerHandler, ServiceExt};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;
use std::sync::Arc;

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

#[derive(Debug, Default, Deserialize, JsonSchema)]
pub struct NoArgs {}

#[derive(Clone)]
struct Server {
    engine: Arc<SyncEngine>,
}

fn internal(msg: String) -> ErrorData {
    ErrorData::internal_error(msg, None)
}

#[tool_router]
impl Server {
    fn new(engine: Arc<SyncEngine>) -> Self {
        Self { engine }
    }

    #[tool(
        description = "Semantic code search across the current project. Use this FIRST instead of grep for: conceptual questions (\"how does auth work?\"), finding related code across files, exploring an unfamiliar codebase, identifying patterns/relationships, and queries where the exact terminology is unknown. (Keep using grep for exact strings: error messages, a known symbol name, imports, TODO/FIXME.) Optional `path` scopes the search to a directory or file. Returns ranked chunks; each has file_path, language, node_type, signature (def line), 1-based start_line/end_line, content, score, an optional parent (enclosing impl/class/module) and prev/next sibling refs (no body). Use start_line/parent to locate code and decide whether to open the file; expand via prev/next only when needed."
    )]
    async fn search_code(
        &self,
        Parameters(a): Parameters<SearchCodeArgs>,
    ) -> Result<String, ErrorData> {
        let engine = self.engine.clone();
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
        let engine = self.engine.clone();
        let stats = tokio::task::spawn_blocking(move || engine.sync(false, |_| {}))
            .await
            .map_err(|e| internal(e.to_string()))?
            .map_err(|e| internal(e.to_string()))?;
        Ok(serde_json::to_string(&stats).unwrap_or_else(|_| "{}".into()))
    }

    #[tool(
        description = "Index status: indexed file/chunk counts, vector count, active model, last sync time and whether the index is stale."
    )]
    async fn get_index_status(
        &self,
        Parameters(_): Parameters<NoArgs>,
    ) -> Result<String, ErrorData> {
        let status = self.engine.status().map_err(|e| internal(e.to_string()))?;
        Ok(serde_json::to_string(&status).unwrap_or_else(|_| "{}".into()))
    }
}

/// Tool-usage guide handed to the client as MCP server instructions —
/// authored as Markdown, embedded at compile time so it stays in sync.
const SKILL: &str = include_str!("SKILL.md");

#[tool_handler(name = "embedding-search", version = "0.1.0")]
impl ServerHandler for Server {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_server_info(rmcp::model::Implementation::new(
            "embedding-search",
            "0.1.0",
        ))
        .with_instructions(SKILL.to_string())
    }
}

/// One hash-incremental resync on a blocking thread, with unified
/// logging. Shared by the startup sync and the periodic loop so their
/// log format never drifts.
async fn run_resync(eng: Arc<SyncEngine>, label: &'static str) {
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

    let config = Config::load_or_init().context("load config")?;
    // Catch-up cadence for the background resync loop. Captured before
    // `config` moves into the engine.
    let resync_minutes = config.sync.resync_interval_minutes.max(1) as u64;

    tracing::info!("opening index for {}", project_dir.display());
    let dir = project_dir.clone();
    let engine = tokio::task::spawn_blocking(move || SyncEngine::new(dir, config))
        .await
        .context("join")?
        .context("open engine")?;
    let engine = Arc::new(engine);

    // Hash-incremental startup sync in the background (near-instant on
    // a clean tree — no staleness timer, no full re-index).
    tokio::spawn(run_resync(engine.clone(), "startup"));

    // Periodic background resync (no file watcher: a file the agent
    // just edited is already in its context — the index only needs to
    // catch up for the *next* agent run). A no-edit tick is ~free.
    {
        let eng = engine.clone();
        tokio::spawn(async move {
            let period = std::time::Duration::from_secs(resync_minutes * 60);
            loop {
                tokio::time::sleep(period).await;
                run_resync(eng.clone(), "background").await;
            }
        });
    }

    let service = Server::new(engine).serve(stdio()).await?;
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
