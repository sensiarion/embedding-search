mod mcp;

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser, Subcommand};
use embedding_search_core::{
    config::{
        normalize_hf_repo, CustomModel, EmbeddingProvider, Pooling, Precision, RemoteConfig,
        DEFAULT_MODEL, SUPPORTED_MODELS,
    },
    embedder::custom_model_cache_dir,
    embedder::Embedder,
    Config, Inspector, SyncEngine, SyncEvent,
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "embedding-search",
    version,
    about = "Semantic code search with embeddings"
)]
struct Cli {
    /// Run the MCP server on stdio (same as the `serve` subcommand).
    /// Launcher-only arg (mcp-bin / Claude Code / OpenCode configs);
    /// hidden from help — humans use the `serve` subcommand.
    #[arg(long, global = true, hide = true)]
    mcp: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Create the index for a directory and run the first sync
    Init {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Re-index a directory
    Sync {
        #[arg(default_value = ".")]
        path: PathBuf,
        /// Ignore the hash cache and re-embed everything
        #[arg(long)]
        force: bool,
    },
    /// Semantic search
    Search {
        query: String,
        #[arg(short, long, default_value = "10")]
        n: usize,
        #[arg(long)]
        json: bool,
        #[arg(long, default_value = ".")]
        path: PathBuf,
        /// Skip the freshness sync and search the index as-is
        #[arg(long)]
        no_sync: bool,
        /// Restrict results to a project-relative dir or file
        /// (e.g. crates/core/src or src/auth.rs)
        #[arg(long = "in")]
        scope: Option<String>,
    },
    /// Index / sync status (model, counts, freshness, search backend)
    Status {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Delete a project's index (embeddings + metadata). The next
    /// init/sync rebuilds from scratch.
    Clear {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Start MCP server mode on stdio
    Serve,
    /// Inspection helpers
    Debug {
        #[command(subcommand)]
        cmd: DebugCmd,
    },
    /// Model management
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
}

#[derive(Subcommand)]
enum DebugCmd {
    /// List indexed files with chunk counts
    Files {
        #[arg(default_value = ".")]
        path: PathBuf,
    },
    /// Show all chunks for one file
    Chunks {
        file: String,
        #[arg(long, default_value = ".")]
        path: PathBuf,
    },
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List supported models
    List,
    /// Set the default model (requires re-index)
    SetDefault { model: String },
    /// Unregister a custom/remote model and delete its cached weights.
    /// Built-ins can't be removed. If it was active, the default
    /// resets to the built-in default (re-index after).
    #[command(alias = "rm")]
    Remove { name: String },
    /// Register a custom model (auto-downloaded + cached like a
    /// built-in, then listed/selectable). Pass exactly one of --repo
    /// (Hugging Face id) or --url (direct .onnx). Re-index after.
    #[command(group(ArgGroup::new("source").required(true).args(["repo", "url"])))]
    Add {
        /// Label to select it by (`models set-default <name>`)
        #[arg(long)]
        name: String,
        /// Hugging Face repo id, e.g. Xenova/bge-small-en-v1.5
        #[arg(long)]
        repo: Option<String>,
        /// Direct URL to a .onnx file (tokenizer files must sit
        /// in the same directory)
        #[arg(long)]
        url: Option<String>,
        /// Alias for `--query-prefix "query: " --doc-prefix
        /// "passage: "` (e5 models). Explicit prefixes below win.
        #[arg(long)]
        e5_prefix: bool,
        /// Text prepended to a search query before embedding, e.g.
        /// "query: " (e5), "search_query: " (nomic), or an instruction.
        #[arg(long)]
        query_prefix: Option<String>,
        /// Text prepended to each indexed chunk before embedding, e.g.
        /// "passage: " / "search_document: ". Omit for query-only
        /// (CLS code) models.
        #[arg(long)]
        doc_prefix: Option<String>,
        /// Encoder output pooling: mean (default) | cls | last-token.
        #[arg(long)]
        pooling: Option<Pooling>,
        /// ONNX precision to pull (HF --repo): fp16 | int8 | full.
        /// Omitted ⇒ the global `[model] precision`.
        #[arg(long)]
        precision: Option<Precision>,
        /// Exact ONNX file in the repo, e.g. model_q4f16.onnx or
        /// onnx/model_q4.onnx. Overrides --precision; for repos with
        /// quantizations the precision mapping doesn't cover.
        #[arg(long)]
        onnx_file: Option<String>,
    },
    /// Switch to an external OpenAI-compatible embeddings API
    /// (sets provider=openai + the [remote] section). Re-index after.
    AddRemote {
        /// Registry label to re-select it by later
        /// (`models set-default <name>`)
        #[arg(long)]
        name: String,
        /// API base, e.g. https://api.deepseek.com/v1
        #[arg(long)]
        base_url: String,
        /// Remote model id, e.g. deepseek-embedding
        #[arg(long)]
        model: String,
        /// Bearer token; "$NAME" / "${NAME}" reads env var NAME
        #[arg(long, default_value = "")]
        api_key: String,
        /// Output dimensions (probed from a live request if omitted)
        #[arg(long)]
        dimensions: Option<usize>,
        /// Alias for `--query-prefix "query: " --doc-prefix
        /// "passage: "` (e5 remote). Explicit prefixes below win.
        #[arg(long)]
        e5_prefix: bool,
        /// Text prepended to a query before sending it to the remote.
        #[arg(long)]
        query_prefix: Option<String>,
        /// Text prepended to each chunk before sending it to the remote.
        #[arg(long)]
        doc_prefix: Option<String>,
    },
}

/// Is the (normalized) HF repo still referenced by a remaining custom
/// model or a built-in? Guards `models remove` from deleting an
/// hf-hub cache dir shared via `models--{repo}` (one dir per repo, no
/// precision/file in the path) — call AFTER removing the target entry.
fn repo_still_used(cfg: &Config, repo_norm: &str) -> bool {
    cfg.custom_models
        .iter()
        .filter_map(|m| m.repo.as_deref())
        .any(|r| normalize_hf_repo(r) == repo_norm)
        || SUPPORTED_MODELS
            .iter()
            .any(|s| s.hf_repo == Some(repo_norm))
}

/// Verify a freshly selected model end-to-end *before* persisting it:
/// `Embedder::new` downloads the weights (HF progress on stderr) and
/// runs a probe embed, so a bad repo/URL/endpoint fails here with an
/// actionable error and the config is left untouched.
fn verify_and_save(cfg: &Config, name: &str) -> Result<()> {
    let remote = cfg.model.provider == EmbeddingProvider::Openai;
    let how = if remote {
        "endpoint reachability + test embed"
    } else {
        "model download + test embed"
    };
    eprintln!("Verifying '{name}' ({how})…");
    let emb = Embedder::new(cfg)
        .with_context(|| format!("'{name}' failed verification — nothing saved"))?;
    cfg.save().context("save config")?;
    let kind = if remote { " (remote)" } else { "" };
    println!(
        "\u{2713} '{name}'{kind} verified — {} dims. Run `embedding-search sync --force` to re-index.",
        emb.dimensions
    );
    Ok(())
}

fn engine(path: &Path) -> Result<SyncEngine> {
    let cfg = Config::load_or_init().context("load config")?;
    let dir = std::fs::canonicalize(path).context("resolve path")?;
    SyncEngine::new(dir, cfg).context("open engine")
}

fn inspector(path: &Path) -> Result<Inspector> {
    let cfg = Config::load_or_init().context("load config")?;
    let dir = std::fs::canonicalize(path).context("resolve path")?;
    Inspector::open(&dir, cfg).context("open inspector")
}

/// Whether a `run_sync` prints its trailing stats. `Quiet` is used
/// when the sync is just a freshness pass before another command.
#[derive(PartialEq)]
enum Report {
    Summary,
    Quiet,
}

/// Run a sync with two live bars: files scanned/processed and chunks
/// embedded (the chunk total grows as files are parsed, so big files
/// show real progress even while the file bar barely moves). Both bars
/// are cleared on completion.
fn run_sync(eng: &SyncEngine, force: bool, report: Report) -> Result<()> {
    let mp = MultiProgress::new();
    let files = mp.add(ProgressBar::new(0));
    files.set_style(ProgressStyle::with_template("files  {bar:28} {pos}/{len}  {msg}").unwrap());
    let chunks = mp.add(ProgressBar::new(0));
    chunks.set_style(ProgressStyle::with_template("chunks {bar:28} {pos}/{len} embedded").unwrap());
    let stats = eng.sync(force, |ev| match ev {
        SyncEvent::Scanned(n) => {
            files.set_length(n as u64);
        }
        SyncEvent::File {
            done,
            path,
            indexed,
            changed,
            ..
        } => {
            files.set_position(done as u64);
            // Pin the message to the last file actually re-embedded
            // (plus a running indexed count) instead of flickering
            // through every skipped file: on an incremental sync the
            // bar otherwise reads as a random mid-repo file.
            if changed {
                files.set_message(format!("{indexed} indexed \u{b7} {path}"));
            }
        }
        SyncEvent::Chunks {
            embedded,
            discovered,
        } => {
            chunks.set_length(discovered as u64);
            chunks.set_position(embedded as u64);
        }
    })?;
    files.finish_and_clear();
    chunks.finish_and_clear();
    if report == Report::Summary {
        println!(
            "  \u{2713} {} indexed  \u{21b7} {} skipped  \u{2717} {} deleted",
            stats.files_indexed, stats.files_skipped, stats.files_deleted
        );
        println!(
            "Done in {:.1}s — {} chunks total",
            stats.elapsed_ms as f64 / 1000.0,
            stats.chunks_total
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    // ort routes ONNX Runtime logs through `tracing` (target `ort`):
    // CoreML graph-partition / "iOS 17.4+ required" / node-assignment
    // WARNs are pure noise (and a real failure surfaces as our own
    // error anyway). Silence the `ort` target unless the user opted
    // into it explicitly via RUST_LOG.
    let base = std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".into());
    // Only skip the override if RUST_LOG carries an explicit `ort`
    // *directive* (target `ort` / `ort::…`) — a plain substring test
    // would also match unrelated targets like `report`/`support`.
    let has_ort_directive = base.split(',').any(|d| {
        let target = d.split('=').next().unwrap_or("").trim();
        target == "ort" || target.starts_with("ort::")
    });
    let filter = if has_ort_directive {
        base
    } else {
        format!("{base},ort=off")
    };
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::new(filter))
        .init();

    let cli = Cli::parse();
    // `--mcp`, or no subcommand at all (the mcp-bin launch shape), runs
    // the stdio MCP server. `serve` is the explicit human-facing alias,
    // routed through the match below.
    if cli.mcp {
        return mcp::run();
    }
    let Some(cmd) = cli.command else {
        return mcp::run();
    };
    match cmd {
        Command::Serve => return mcp::run(),
        Command::Clear { path } => {
            use embedding_search_core::{config::PROJECT_INDEX_DIR, sync::wipe_index};
            let dir = std::fs::canonicalize(&path)
                .context("resolve path")?
                .join(PROJECT_INDEX_DIR);
            if dir.is_dir() {
                wipe_index(&dir).context("clear index")?;
                println!("Cleared index at {}", dir.display());
            } else {
                println!("No index at {} (nothing to clear)", dir.display());
            }
        }
        Command::Init { path } => {
            let eng = engine(&path)?;
            println!("Index at {}", eng.index_dir().display());
            run_sync(&eng, true, Report::Summary)?;
        }
        Command::Sync { path, force } => {
            let eng = engine(&path)?;
            run_sync(&eng, force, Report::Summary)?;
        }
        Command::Search {
            query,
            n,
            json,
            path,
            no_sync,
            scope,
        } => {
            let eng = engine(&path)?;
            // Cursor-style throttle: only resync if the last one is
            // older than sync.resync_interval_minutes (~10 min), not on
            // every search. Hash-incremental, bars self-clear.
            if !no_sync && eng.is_due()? {
                run_sync(&eng, false, Report::Quiet)?;
            }
            let res = eng.search(&query, n, scope.as_deref())?;
            if json {
                println!("{}", serde_json::to_string_pretty(&res)?);
            } else if res.is_empty() {
                println!("No results.");
            } else {
                for r in res {
                    println!(
                        "\n\u{2022} {}:{}-{}  [{}/{}]  score={:.3}",
                        r.file_path, r.start_line, r.end_line, r.language, r.node_type, r.score
                    );
                    if let Some(p) = &r.parent {
                        println!("  in {} {}", p.node_type, p.signature);
                    }
                    let preview: String = r.content.chars().take(240).collect();
                    println!("{preview}");
                }
            }
        }
        Command::Status { path } => {
            let s = inspector(&path)?.status()?;
            match Config::config_path() {
                Ok(p) => println!("config:         {}", p.display()),
                Err(e) => println!("config:         <unavailable: {e}>"),
            }
            println!("model:          {}", s.model);
            println!("files:          {}", s.files);
            println!("chunks:         {}", s.chunks);
            println!("vectors:        {}", s.vector_count);
            println!("search backend: {}", s.search_backend.as_str());
            println!(
                "last sync:      {}",
                s.last_sync_at.as_deref().unwrap_or("never")
            );
            println!("resync due:     {}", s.resync_due);
            println!(
                "merkle root:    {}",
                s.merkle_root.as_deref().unwrap_or("-")
            );
            if s.is_stale {
                println!("(index never built — run `embedding-search init`)");
            }
        }
        Command::Debug { cmd } => match cmd {
            DebugCmd::Files { path } => {
                let ins = inspector(&path)?;
                let files = ins.list_files()?;
                println!("{} files indexed", files.len());
                for f in files {
                    println!("{:>5}  {}", f.chunk_count, f.path);
                }
            }
            DebugCmd::Chunks { file, path } => {
                let ins = inspector(&path)?;
                let chunks = ins.chunks_for_file(&file)?;
                if chunks.is_empty() {
                    println!("No chunks for {file}");
                }
                for c in chunks {
                    println!(
                        "\n#{} [{}] bytes {}..{}",
                        c.chunk_index, c.node_type, c.start_byte, c.end_byte
                    );
                    let preview: String = c.content.chars().take(200).collect();
                    println!("{preview}");
                }
            }
        },
        Command::Models { cmd } => match cmd {
            ModelsCmd::List => {
                let cfg = Config::load_or_init().context("load config")?;
                let remote_active = cfg.model.provider == EmbeddingProvider::Openai;
                // Active LOCAL selection (only meaningful when not
                // remote): an onnx_path override wins over the registry,
                // else `[model] default` (a built-in or custom name).
                let onnx = cfg.model.onnx_path.as_ref();
                let mark = |is_active: bool| if is_active { "*" } else { " " };

                println!(
                    " {:<37} {:>4} {:>4} {:>4} {:>6} {:>9}",
                    "model", "dim", "code", "ml", "prec", "RAM~MB"
                );
                for m in SUPPORTED_MODELS {
                    let active = !remote_active
                        && onnx.is_none()
                        && cfg.custom_model().is_none()
                        && m.name == cfg.model.default;
                    println!(
                        "{}{:<37} {:>4} {:>4} {:>4} {:>6} {:>9}",
                        mark(active),
                        m.name,
                        m.dimensions,
                        m.code,
                        m.multilingual,
                        m.effective_precision(cfg.model.precision).label(),
                        m.ram_mb(cfg.model.precision),
                    );
                }
                for cm in &cfg.custom_models {
                    let active = !remote_active && onnx.is_none() && cm.name == cfg.model.default;
                    let src = cm
                        .repo
                        .as_deref()
                        .or(cm.url.as_deref())
                        .unwrap_or("(no source)");
                    println!("{}{:<37} custom — {src}", mark(active), cm.name);
                }
                if let Some(p) = onnx {
                    println!(
                        "{}{:<37} custom — local onnx",
                        mark(!remote_active),
                        p.display()
                    );
                }
                for r in &cfg.remote_models {
                    let active = remote_active && r.name == cfg.model.default;
                    println!(
                        "{}{:<37} remote — {} @ {}",
                        mark(active),
                        r.name,
                        r.model,
                        r.base_url
                    );
                }
                println!(
                    "\n* = active model. [model] precision (f32/fp16/int8) \
                     applies to ONNX-encoder models only (static + \
                     fastembed ignore it). Add: `models add` / `models add-remote`."
                );
            }
            ModelsCmd::SetDefault { model } => {
                let mut cfg = Config::load_or_init().context("load config")?;
                cfg.select_model(&model)?;
                verify_and_save(&cfg, &model)?;
            }
            ModelsCmd::Remove { name } => {
                let mut cfg = Config::load_or_init().context("load config")?;
                let was_active = cfg.model.default == name;

                if let Some(i) = cfg.custom_models.iter().position(|m| m.name == name) {
                    let cm = cfg.custom_models.remove(i);
                    let root = cfg.model_cache_dir().context("cache dir")?;
                    if let Some(dir) = custom_model_cache_dir(&root, &cm) {
                        let still_used = cm
                            .repo
                            .as_deref()
                            .map(normalize_hf_repo)
                            .is_some_and(|r| repo_still_used(&cfg, &r));
                        if still_used {
                            println!("Kept shared cache {} (still in use).", dir.display());
                        } else if dir.is_dir() {
                            std::fs::remove_dir_all(&dir)
                                .with_context(|| format!("delete {}", dir.display()))?;
                            println!("Deleted cached weights {}", dir.display());
                        }
                    }
                    println!("Removed custom model '{name}'.");
                } else if let Some(i) = cfg.remote_models.iter().position(|r| r.name == name) {
                    cfg.remote_models.remove(i);
                    println!("Removed remote '{name}' (no local weights).");
                } else {
                    anyhow::bail!(
                        "no custom or remote model named '{name}' \
                         (built-ins can't be removed — see `models list`)"
                    );
                }

                if was_active {
                    cfg.model.default = DEFAULT_MODEL.to_string();
                    cfg.model.provider = EmbeddingProvider::Local;
                    println!(
                        "Was active — default reset to {DEFAULT_MODEL}. \
                         Run `embedding-search sync --force` to re-index."
                    );
                }
                cfg.save().context("save config")?;
            }
            ModelsCmd::Add {
                name,
                repo,
                url,
                e5_prefix,
                query_prefix,
                doc_prefix,
                pooling,
                precision,
                onnx_file,
            } => {
                // clap's `source` ArgGroup guarantees exactly one of
                // --repo / --url is set. A `--repo` value may be a full
                // HF URL (copied from the browser) — canonicalize to
                // the bare `org/name` id `hf-hub` expects.
                // `--e5_prefix` is sugar for the e5 prefix pair;
                // explicit `--query-prefix`/`--doc-prefix` win.
                let (qp, dp) = if e5_prefix {
                    (
                        query_prefix
                            .or_else(|| Some(embedding_search_core::config::E5_QUERY.to_string())),
                        doc_prefix.or_else(|| {
                            Some(embedding_search_core::config::E5_PASSAGE.to_string())
                        }),
                    )
                } else {
                    (query_prefix, doc_prefix)
                };
                let mut cfg = Config::load_or_init().context("load config")?;
                cfg.custom_models.retain(|m| m.name != name);
                cfg.custom_models.push(CustomModel {
                    name: name.clone(),
                    repo: repo.map(|r| normalize_hf_repo(&r)),
                    url,
                    query_prefix: qp,
                    doc_prefix: dp,
                    pooling: pooling.unwrap_or_default(),
                    precision,
                    onnx_file,
                });
                // Same activation path as add-remote / set-default.
                cfg.select_model(&name)?;
                verify_and_save(&cfg, &name)?;
            }
            ModelsCmd::AddRemote {
                name,
                base_url,
                model,
                api_key,
                dimensions,
                e5_prefix,
                query_prefix,
                doc_prefix,
            } => {
                // `--e5_prefix` is sugar for the e5 prefix pair;
                // explicit `--query-prefix`/`--doc-prefix` win.
                let (qp, dp) = if e5_prefix {
                    (
                        query_prefix
                            .or_else(|| Some(embedding_search_core::config::E5_QUERY.to_string())),
                        doc_prefix.or_else(|| {
                            Some(embedding_search_core::config::E5_PASSAGE.to_string())
                        }),
                    )
                } else {
                    (query_prefix, doc_prefix)
                };
                let mut cfg = Config::load_or_init().context("load config")?;
                let entry = RemoteConfig {
                    name: name.clone(),
                    base_url: base_url.clone(),
                    model: model.clone(),
                    api_key,
                    dimensions,
                    query_prefix: qp,
                    doc_prefix: dp,
                    ..RemoteConfig::default()
                };
                // Keep it in the registry so it can be re-selected
                // later by name without redefining it.
                cfg.remote_models.retain(|r| r.name != name);
                cfg.remote_models.push(entry);
                cfg.select_model(&name)?; // copies into [remote], provider=openai
                verify_and_save(&cfg, &name)?;
            }
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_flag_parses() {
        let cli = Cli::try_parse_from(["es", "--mcp"]).unwrap();
        assert!(cli.mcp);
        assert!(cli.command.is_none());
    }

    #[test]
    fn no_subcommand_is_server_shape() {
        let cli = Cli::try_parse_from(["es"]).unwrap();
        assert!(!cli.mcp);
        assert!(cli.command.is_none());
    }

    #[test]
    fn serve_subcommand_parses() {
        let cli = Cli::try_parse_from(["es", "serve"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Serve)));
    }

    #[test]
    fn regular_subcommand_is_not_server_shape() {
        let cli = Cli::try_parse_from(["es", "search", "q"]).unwrap();
        assert!(!cli.mcp);
        assert!(matches!(cli.command, Some(Command::Search { .. })));
    }
}
