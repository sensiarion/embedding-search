use crate::chunker::{Chunk, Chunker};
use crate::config::{Config, ModelSpec, PROJECT_INDEX_DIR};
use crate::db::{Db, NewChunk};
use crate::embedder::Embedder;
use crate::error::{Error, Result};
use crate::vector::VectorIndex;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct SyncStats {
    pub files_indexed: usize,
    pub files_skipped: usize,
    pub files_deleted: usize,
    pub chunks_total: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct IndexStatus {
    pub files: i64,
    pub chunks: i64,
    pub vector_count: usize,
    pub model: String,
    pub last_sync_at: Option<String>,
    /// Index was never built.
    pub is_stale: bool,
    /// A CLI search would trigger a resync now (never synced, or last
    /// sync older than `sync.resync_interval_minutes`).
    pub resync_due: bool,
    /// Search backend the next query will use.
    pub search_backend: SearchBackend,
    /// File hash-tree root (short) — changes iff indexed content did.
    pub merkle_root: Option<String>,
}

/// Progress callback events.
pub enum SyncEvent<'a> {
    Scanned(usize),
    File {
        done: usize,
        total: usize,
        path: &'a str,
        /// Files re-embedded so far (changed since last sync).
        indexed: usize,
        /// This file changed and was re-embedded (vs. skipped clean).
        changed: bool,
    },
    /// Chunk-level progress, emitted at each embed-batch flush: of the
    /// `discovered` chunks seen so far, `embedded` are written. The
    /// total rises as files are parsed, so this advances within a large
    /// file even while the file count barely moves.
    Chunks {
        embedded: usize,
        discovered: usize,
    },
}

/// Parse a stored rfc3339 `last_sync_at` to UTC, or `None` if absent /
/// unparseable. Single source for both freshness checks below.
fn parse_last(last: Option<&str>) -> Option<chrono::DateTime<chrono::Utc>> {
    let s = last?;
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|ts| ts.with_timezone(&chrono::Utc))
}

/// The index is "stale" only if it was never built (no recorded sync,
/// or an unparseable timestamp). There is no time-based expiry: every
/// sync is hash-incremental and near-instant when nothing changed, so
/// freshness is driven by content, not a clock.
pub(crate) fn never_synced(last: Option<&str>) -> bool {
    parse_last(last).is_none()
}

/// Whether a CLI search should resync first: never synced, an
/// unparseable stamp, or the last sync is older than the throttle.
/// Not a correctness expiry — just paces the walk+hash on a clean tree.
pub(crate) fn resync_due(last: Option<&str>, interval_minutes: i64) -> bool {
    match parse_last(last) {
        None => true,
        Some(ts) => chrono::Utc::now().signed_duration_since(ts).num_minutes() >= interval_minutes,
    }
}

/// Vector backend the next query uses: exact (brute force) under the
/// configured size, else the HNSW graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchBackend {
    Exact,
    Hnsw,
}

impl SearchBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            SearchBackend::Exact => "exact",
            SearchBackend::Hnsw => "hnsw",
        }
    }

    fn pick(vectors: usize, exact_below: usize) -> Self {
        if vectors < exact_below {
            SearchBackend::Exact
        } else {
            SearchBackend::Hnsw
        }
    }
}

impl IndexStatus {
    /// Single assembly point for the status DTO (shared by the live
    /// engine and the read-only Inspector) — keeps the derived fields
    /// and the short-root truncation from drifting.
    pub(crate) fn assemble(
        files: i64,
        chunks: i64,
        vector_count: usize,
        model: String,
        last_sync_at: Option<String>,
        merkle_root: Option<String>,
        cfg: &Config,
    ) -> Self {
        Self {
            files,
            chunks,
            vector_count,
            model,
            is_stale: never_synced(last_sync_at.as_deref()),
            resync_due: resync_due(last_sync_at.as_deref(), cfg.sync.resync_interval_minutes),
            search_backend: SearchBackend::pick(vector_count, cfg.search.exact_below),
            merkle_root: merkle_root.map(|r| r.chars().take(16).collect()),
            last_sync_at,
        }
    }
}

/// Outcome of `classify` for a single file.
enum Classified {
    /// DB row already matches disk (or file unreadable / has no chunks).
    Unchanged,
    /// Bytes match the indexed hash but `(mtime, size)` moved — refresh
    /// the file row so the next sync hits the cheap stat short-circuit
    /// instead of re-reading + re-hashing the same bytes.
    Touched { mtime: i64, size: i64 },
    /// File needs to be re-embedded.
    Changed(FileWork),
}

/// A changed file ready to embed: metadata + its chunks.
struct FileWork {
    rel: String,
    mtime: i64,
    size: i64,
    hash: String,
    lang: String,
    chunks: Vec<Chunk>,
}

/// How a planned chunk gets its embedding.
enum Reuse {
    /// Identical chunk already lived in THIS file at the prior index
    /// state — keep its vector_id verbatim (`vector_id` is `UNIQUE`
    /// per chunk row, so this avoids the schema fight).
    Same(u64),
    /// Identical content lives in the index under another file (rename,
    /// branch checkout, copy-paste). Allocate a fresh vector_id at
    /// apply time and COPY the source bytes into it — skips the
    /// embedding compute, the dominant cost.
    Copy { source: u64 },
    /// No prior vector matches — embed fresh.
    Fresh,
}

/// A chunk tagged with the text actually fed to the model (raw body +
/// the code↔NL header), its content hash **over that enriched text**,
/// and how its embedding is sourced.
struct Planned {
    chunk: Chunk,
    /// Small code↔NL header (path/symbol/kind/signature) prepended to
    /// the body *for embedding only*. Stored separately from the large
    /// body so the bounded scan window holds each body once, not twice
    /// — the enriched `header + body` string is materialized
    /// transiently only for the chunks actually being embedded.
    embed_header: String,
    /// blake3(header ++ body) — exact embed-text identity for intra-file
    /// `Same` reuse and for the DB `content_hash` column.
    hash: String,
    /// blake3(body) — header-independent identity used to find cross-file
    /// `Copy` sources (rename / branch switch lands the same body under a
    /// new path; the source vector is close enough — see `Reuse::Copy`).
    body_hash: String,
    reuse: Reuse,
}

/// Code↔NL bridge header prepended to a chunk **for embedding only**:
/// the relative path, symbol, node kind and signature carry the
/// natural-language cues raw code lacks (RANGER-style enrichment),
/// ending with the `\n` that separates it from the body. The stored DB
/// content and search-result content stay raw, so BM25 tokenization
/// and prev/next/parent refs are unchanged. The embedded text is
/// `header + body`; `plan_file` hashes both halves so the reuse
/// identity covers the header without a third large allocation. The
/// model's own `doc_prefix` is applied on top by `embed_documents`.
fn embed_header(rel: &str, c: &Chunk) -> String {
    let sig = crate::search::signature_of(&c.content);
    match &c.symbol {
        Some(sym) => format!("{rel}::{sym} ({}) — {sig}\n", c.node_type),
        None => format!("{rel} ({}) — {sig}\n", c.node_type),
    }
}

/// A changed file resolved against its prior index state: which chunks
/// reuse a vector, which need embedding, and which old vectors are now
/// orphaned.
struct FilePlan {
    rel: String,
    mtime: i64,
    size: i64,
    fhash: String,
    lang: String,
    chunks: Vec<Planned>,
    removed: Vec<u64>,
}

/// Resolve the sync worker-thread cap: the configured value, or (when
/// `0`) all cores but one so the workstation stays responsive instead
/// of every core saturating on a large reindex.
fn sync_thread_count(configured: usize) -> usize {
    if configured > 0 {
        return configured;
    }
    std::thread::available_parallelism()
        .map(|n| n.get().saturating_sub(1))
        .unwrap_or(1)
        .max(1)
}

/// Absolute roots of git linked worktrees registered under this repo
/// (`<project>/.git/worktrees/*/gitdir`), restricted to ones nested
/// inside `project_dir`. Git's own registry is authoritative and read
/// once — unlike a per-directory `.git` probe, which also misfires on
/// submodules (their work dir is a `.git` *file* too). If `project_dir`
/// is itself a linked worktree (`.git` is a file) it owns no registry,
/// so there is nothing nested to prune.
fn linked_worktree_roots(project_dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(project_dir.join(".git").join("worktrees")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|e| {
            // `gitdir` points at the worktree's own `.git` FILE; its
            // parent is that worktree's working-tree root.
            let content = std::fs::read_to_string(e.path().join("gitdir")).ok()?;
            let root = Path::new(content.trim()).parent()?.to_path_buf();
            let abs = std::fs::canonicalize(root).ok()?;
            (abs != project_dir && abs.starts_with(project_dir)).then_some(abs)
        })
        .collect()
}

/// Remove a project's on-disk index (SQLite + WAL/SHM + vector file).
/// Single source of the wipe set — shared by the fingerprint-mismatch
/// rebuild and the `clear` command. A missing file is fine, but a file
/// that *survives* removal is fatal: reopening on top of a half-wiped
/// index silently corrupts the rebuild (the
/// `UNIQUE constraint failed: chunks.vector_id` failure). Fail loudly
/// instead so the cause (e.g. another process holding the index) is
/// visible rather than producing garbage results.
pub fn wipe_index(index_dir: &Path) -> Result<()> {
    for f in ["meta.db", "meta.db-wal", "meta.db-shm", "vectors.usearch"] {
        let p = index_dir.join(f);
        match std::fs::remove_file(&p) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(Error::Index(format!("wipe {}: {e}", p.display()))),
        }
    }
    for f in ["meta.db", "vectors.usearch"] {
        let p = index_dir.join(f);
        if p.exists() {
            return Err(Error::Index(format!(
                "index wipe incomplete: {} still present — close other \
                 embedding-search processes on this project and retry",
                p.display()
            )));
        }
    }
    Ok(())
}

/// (mtime secs, size bytes) of a path, or (0, 0) if unstattable.
fn mtime_size(path: &Path) -> (i64, i64) {
    let Ok(m) = std::fs::metadata(path) else {
        return (0, 0);
    };
    let mtime = m
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    (mtime, m.len() as i64)
}

pub struct SyncEngine {
    project_dir: PathBuf,
    index_dir: PathBuf,
    config: Config,
    db: Mutex<Db>,
    vector: Mutex<VectorIndex>,
    embedder: Embedder,
    chunker: Chunker,
    /// Optional cross-encoder re-rank stage. `None` unless
    /// `[rerank] enabled` — default users never construct or pay for
    /// it. Absent entirely under `bench-stub` (ONNX-only feature).
    #[cfg(not(feature = "bench-stub"))]
    reranker: Option<crate::rerank::Reranker>,
    /// Private worker pool for the parallel scan/hash/parse phase,
    /// built once (not the global pool) so its thread count bounds CPU
    /// and repeated syncs don't pay pool spawn/teardown.
    pool: rayon::ThreadPool,
}

impl SyncEngine {
    pub fn new(project_dir: PathBuf, config: Config) -> Result<Self> {
        let index_dir = project_dir.join(PROJECT_INDEX_DIR);

        // Build the embedder first: it is the single source of the
        // active (model_name, dimensions) for both local ONNX and the
        // external OpenAI-compatible backend (no static spec for remote).
        let embedder = Embedder::new(&config)?;
        let model_name = embedder.model_name.as_str();
        let dims = embedder.dimensions;

        // Any change that makes existing chunks/vectors invalid (model
        // + resolved weight variant, dims, token cap, chunk byte cap,
        // chunker logic) shifts the fingerprint → wipe stale index
        // before reopening.
        let fingerprint = config.index_fingerprint(model_name, dims, &embedder.fingerprint_tag());
        if index_dir.exists() {
            let probe = Db::open_or_create(&index_dir)?;
            let stored = probe.get_meta("index_fingerprint")?;
            let (indexed_files, _) = probe.counts()?;
            drop(probe);
            // Wipe when the stored fingerprint differs OR is absent
            // (legacy index from before fingerprinting) — but only if
            // there is data to invalidate, so a freshly created empty
            // db is left alone.
            let mismatch = stored.as_deref() != Some(fingerprint.as_str());
            if mismatch && indexed_files > 0 {
                tracing::warn!(
                    "index fingerprint changed ({} -> {fingerprint}) — wiping index",
                    stored.as_deref().unwrap_or("none")
                );
                wipe_index(&index_dir)?;
            }
        }

        let db = Db::open_or_create(&index_dir)?;
        db.set_meta("model_name", model_name)?;
        db.set_meta("dimensions", &dims.to_string())?;
        db.set_meta("index_fingerprint", &fingerprint)?;
        let vector = VectorIndex::open_or_create(&index_dir, dims)?;
        let chunker = Chunker::new(config.sync.max_chunk_bytes);

        let gi = index_dir.join(".gitignore");
        if !gi.exists() {
            let _ = std::fs::write(&gi, "*\n");
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(sync_thread_count(config.sync.sync_threads))
            .build()
            .map_err(|e| Error::Index(format!("rayon pool: {e}")))?;

        // Load the reranker only when explicitly enabled — skipping the
        // download/session entirely is what makes the default path
        // zero-cost.
        #[cfg(not(feature = "bench-stub"))]
        let reranker = if config.rerank_enabled() {
            Some(crate::rerank::Reranker::load(&config)?)
        } else {
            None
        };

        Ok(Self {
            project_dir,
            index_dir,
            config,
            db: Mutex::new(db),
            vector: Mutex::new(vector),
            embedder,
            chunker,
            #[cfg(not(feature = "bench-stub"))]
            reranker,
            pool,
        })
    }

    pub fn index_dir(&self) -> &Path {
        &self.index_dir
    }

    /// The configured model name (`[model] default`, untagged) — the
    /// value a user passes to `set` / `models set-default`.
    pub fn configured_model(&self) -> &str {
        &self.config.model.default
    }

    /// Whether the active model is a static Model2Vec (no transformer
    /// /ONNX): cheap, no per-batch attention. A custom/remote model
    /// (no spec) is treated as heavy. Drives the large-repo hint.
    pub fn model_is_static(&self) -> bool {
        crate::config::model_spec(&self.config.model.default).is_some_and(ModelSpec::is_static)
    }

    pub fn project_dir(&self) -> &Path {
        &self.project_dir
    }

    fn is_excluded(&self, rel: &Path) -> bool {
        let dirs = &self.config.sync.exclude_dirs;
        if rel
            .components()
            .any(|c| dirs.iter().any(|d| d == &*c.as_os_str().to_string_lossy()))
        {
            return true;
        }
        let rel_s = rel.to_string_lossy();
        if rel_s.ends_with(".lock") {
            return true;
        }
        self.config
            .sync
            .exclude
            .iter()
            .any(|p| !p.is_empty() && rel_s.contains(p.as_str()))
    }

    /// Read a file as UTF-8 text, or `None` if it is binary. Sniffs an
    /// 8 KiB head for a NUL byte first so huge binaries (videos, model
    /// weights, …) are rejected without slurping the whole file into
    /// RAM; zip/pdf/image/office (.docx) containers all carry NUL in
    /// their header and are caught here.
    fn read_text(path: &Path) -> Option<String> {
        use std::io::Read;
        let mut f = std::fs::File::open(path).ok()?;
        let mut head = [0u8; 8192];
        let n = f.read(&mut head).ok()?;
        if head[..n].contains(&0) {
            return None;
        }
        let mut bytes = Vec::with_capacity(n);
        bytes.extend_from_slice(&head[..n]);
        f.read_to_end(&mut bytes).ok()?;
        // tail NUL (binary without one in the first 8 KiB) — cheap vs
        // chunk+embed; non-UTF-8 also rejects most remaining binaries.
        if bytes[n..].contains(&0) {
            return None;
        }
        String::from_utf8(bytes).ok()
    }

    fn collect_files(&self) -> Vec<PathBuf> {
        // Nested git linked worktrees are duplicate checkouts of the
        // same code — resolve their roots once and prune those subtrees
        // (only the working tree being synced is "the codebase").
        let worktrees = linked_worktree_roots(&self.project_dir);
        // Parallel walk (ripgrep's `ignore` is already the fastest
        // gitignore-aware walker; the win is using its thread pool, not
        // a different crate). Threads gather raw file paths via a
        // channel — only `project_dir`-relative filtering and the
        // deterministic sort happen on the collecting side, so the
        // visitor needs no `&self`. `is_file()` (not the walker's
        // dirent kind) so a symlinked source file is followed and
        // still indexed, matching the pre-parallel behavior.
        let (tx, rx) = std::sync::mpsc::channel();
        ignore::WalkBuilder::new(&self.project_dir)
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            // `ignore` defaults `require_git(true)`: .gitignore / global
            // git excludes are applied ONLY inside a git work tree, so a
            // plain directory (or a repo before its first commit) would
            // index files the user clearly meant to exclude. Honor
            // .gitignore regardless of a .git dir (ripgrep --no-require-git).
            .require_git(false)
            .parents(true)
            .filter_entry(move |e| !worktrees.iter().any(|w| e.path().starts_with(w)))
            .build_parallel()
            .run(|| {
                let tx = tx.clone();
                Box::new(move |res| {
                    if let Ok(entry) = res {
                        let path = entry.into_path();
                        if path.is_file() {
                            let _ = tx.send(path);
                        }
                    }
                    ignore::WalkState::Continue
                })
            });
        drop(tx);
        let mut out: Vec<PathBuf> = rx
            .into_iter()
            .filter(|path| {
                path.strip_prefix(&self.project_dir)
                    .is_ok_and(|rel| !self.is_excluded(rel))
            })
            .collect();
        // Parallel traversal yields a run-dependent order; sort so the
        // file list (and any downstream merkle/diff) is deterministic.
        out.sort();
        // A git-tracked symlink whose target is also walked (e.g.
        // `skills/x -> crates/.../x`) would embed the same content
        // twice. Dedup by canonical path, but never pay a per-file
        // `canonicalize` (O(path-depth) syscalls) on the symlink-free
        // common case: resolve `project_dir` once, key each real file
        // as `canon_root + rel` (no syscall, still matches when the
        // project has a symlinked ancestor), and `canonicalize` only
        // actual symlinks. A symlink is dropped only when its target is
        // itself a walked real file (out-of-tree targets are kept); the
        // real file's path is the one indexed, so search hits the
        // canonical location. Reals stay ahead of kept links and both
        // keep the sorted order, so vector-id allocation is stable.
        let canon_root =
            std::fs::canonicalize(&self.project_dir).unwrap_or_else(|_| self.project_dir.clone());
        let mut seen: HashSet<PathBuf> = HashSet::with_capacity(out.len());
        let mut reals: Vec<PathBuf> = Vec::with_capacity(out.len());
        let mut links: Vec<PathBuf> = Vec::new();
        for path in out {
            if path
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink())
            {
                links.push(path);
            } else {
                let rel = path
                    .strip_prefix(&self.project_dir)
                    .unwrap_or(path.as_path());
                seen.insert(canon_root.join(rel));
                reals.push(path);
            }
        }
        if links.is_empty() {
            return reals;
        }
        for link in links {
            let target = std::fs::canonicalize(&link).unwrap_or_else(|_| link.clone());
            if seen.insert(target) {
                reals.push(link);
            }
        }
        reals
    }

    fn rel_str(&self, path: &Path) -> String {
        path.strip_prefix(&self.project_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    }

    /// Drop a file's chunks+vectors from both stores.
    fn purge(&self, rel: &str) -> Result<()> {
        let old = self.db.lock().unwrap().delete_file_by_path(rel)?;
        if !old.is_empty() {
            self.vector.lock().unwrap().remove_many(&old)?;
        }
        Ok(())
    }

    /// Persist the vector index and stamp the sync time. The merkle
    /// root is rescanned only when `changed` (files added/removed):
    /// on a clean no-op resync the root cannot have moved, so the
    /// O(files) scan + hash is skipped.
    fn finalize(&self, changed: bool) -> Result<()> {
        self.vector.lock().unwrap().save()?;
        let now = chrono::Utc::now().to_rfc3339();
        let db = self.db.lock().unwrap();
        db.set_meta("last_sync_at", &now)?;
        if changed {
            let root = db.merkle_root()?;
            db.set_meta("merkle_root", &root)?;
        }
        Ok(())
    }

    /// Resolve which of a file's new chunks can keep their existing
    /// vector (content unchanged across the edit) vs. must be embedded,
    /// and which old vectors are now orphaned. Identity is the chunk's
    /// blake3 — robust to chunks moving (byte offsets / index shift)
    /// when the file is edited above them.
    ///
    /// Three reuse paths, in priority order: (1) intra-file Same —
    /// identical chunk already lived in this file at the prior index,
    /// keep its `vector_id` verbatim; (2) cross-file Copy — identical
    /// content lives in the index under another file (rename, branch
    /// switch, copy-paste), so allocate a fresh `vector_id` and copy
    /// the source embedding bytes at apply time, skipping the model
    /// call; (3) Fresh — no prior vector matches, embed normally.
    fn plan_file(&self, f: FileWork) -> Result<FilePlan> {
        let old = self
            .db
            .lock()
            .unwrap()
            .chunk_fingerprints_for_file(&f.rel)?;
        let mut avail: HashMap<String, Vec<u64>> = HashMap::new();
        for (h, vid) in old {
            avail.entry(h).or_default().push(vid);
        }
        let mut chunks = Vec::with_capacity(f.chunks.len());
        for c in f.chunks {
            // Identity = hash(header ++ body), the exact bytes embedded,
            // computed incrementally so the concatenation is never
            // allocated. A body unchanged but moved to a new
            // path/symbol gets a new header → new hash → re-embed
            // (correct: its vector would otherwise be stale).
            let header = embed_header(&f.rel, &c);
            // `hash` = full embed-text identity (header ++ body) for
            // intra-file Same reuse; `body_hash` strips the header so
            // cross-file Copy lookups can match renames / branch switches.
            let body_hash = blake3::hash(c.content.as_bytes()).to_hex().to_string();
            let mut hasher = blake3::Hasher::new();
            hasher.update(header.as_bytes());
            hasher.update(c.content.as_bytes());
            let hash = hasher.finalize().to_hex().to_string();
            let reuse = if let Some(v) = avail.get_mut(&hash).and_then(Vec::pop) {
                Reuse::Same(v)
            } else if let Some(src) = self
                .db
                .lock()
                .unwrap()
                .lookup_vector_id_by_body_hash(&body_hash)?
            {
                // Pre-flight: confirm the source vector actually exists in
                // the usearch store (the chunks ↔ vector stores are not
                // transactional — a partial-crash recovery can leave an
                // orphan chunk row whose vector_id was already removed,
                // and a prior file in this same `flush_group` may have
                // dropped the source as part of its own purge). If it's
                // gone, downgrade to Fresh here so the embed batch sized
                // by `flush_group` (Fresh-count) stays aligned with what
                // `apply_plan` consumes from the iterator.
                let dims = self.embedder.dimensions;
                let exists = self.vector.lock().unwrap().get(src, dims)?.is_some();
                if exists {
                    Reuse::Copy { source: src }
                } else {
                    Reuse::Fresh
                }
            } else {
                Reuse::Fresh
            };
            chunks.push(Planned {
                chunk: c,
                embed_header: header,
                hash,
                body_hash,
                reuse,
            });
        }
        // vectors left unclaimed = chunks that vanished from the file
        let removed = avail.into_values().flatten().collect();
        Ok(FilePlan {
            rel: f.rel,
            mtime: f.mtime,
            size: f.size,
            fhash: f.hash,
            lang: f.lang,
            chunks,
            removed,
        })
    }

    /// Persist one planned file: `Same` chunks keep their vector_id,
    /// `Copy` chunks get a fresh vector_id with the source bytes
    /// duplicated into it (no embedding compute), `Fresh` chunks
    /// consume freshly embedded vectors. Orphaned vectors from the
    /// prior file state are dropped (best-effort — a vector still
    /// referenced by another chunk row is left in place by
    /// `remove_many`'s tolerance). `embedded` yields vectors for
    /// `Fresh`-only chunks in plan order.
    fn apply_plan(
        &self,
        p: FilePlan,
        embedded: &mut impl Iterator<Item = Vec<f32>>,
    ) -> Result<usize> {
        let n = p.chunks.len();
        let need_new = p
            .chunks
            .iter()
            .filter(|c| !matches!(c.reuse, Reuse::Same(_)))
            .count();
        let (file_id, mut fresh) = {
            let db = self.db.lock().unwrap();
            let id = db.upsert_file(&p.rel, p.mtime, p.size, &p.fhash)?;
            (id, db.alloc_vector_ids(need_new)?.into_iter())
        };
        let dims = self.embedder.dimensions;
        let mut add_keys = Vec::with_capacity(need_new);
        let mut add_vecs = Vec::with_capacity(need_new);
        let mut new_chunks = Vec::with_capacity(n);
        let mut vids = Vec::with_capacity(n);
        for (i, pc) in p.chunks.into_iter().enumerate() {
            let vid = match pc.reuse {
                Reuse::Same(v) => v,
                Reuse::Copy { source } => {
                    let v = fresh.next().ok_or_else(|| {
                        Error::Index("internal: allocated vector ids < new chunks".into())
                    })?;
                    // Read the source embedding under the existing key
                    // and re-add under the freshly-allocated key. Cheap
                    // (memcpy + usearch add) vs. a full model forward
                    // pass. `plan_file` pre-flights existence and
                    // downgrades to Fresh when the source is gone, so
                    // `None` here means a concurrent purge raced past
                    // the pre-flight check. Treat it as an internal
                    // error rather than steal from the Fresh-sized
                    // embed iterator: the next sync will replan and
                    // self-heal.
                    let src_vec = self.vector.lock().unwrap().get(source, dims)?;
                    let vec = src_vec.ok_or_else(|| {
                        Error::Index(format!(
                            "internal: cross-file Copy source vector {source} vanished after \
                             plan_file pre-flight; abort sync (next pass will replan)"
                        ))
                    })?;
                    add_keys.push(v);
                    add_vecs.push(vec);
                    v
                }
                Reuse::Fresh => {
                    let v = fresh.next().ok_or_else(|| {
                        Error::Index("internal: allocated vector ids < new chunks".into())
                    })?;
                    let vec = embedded.next().ok_or_else(|| {
                        Error::Index("internal: embedded vectors < new chunks".into())
                    })?;
                    add_keys.push(v);
                    add_vecs.push(vec);
                    v
                }
            };
            new_chunks.push(NewChunk {
                chunk_index: i as i32,
                content: pc.chunk.content,
                start_byte: pc.chunk.start_byte,
                end_byte: pc.chunk.end_byte,
                node_type: pc.chunk.node_type,
                language: p.lang.clone(),
                content_hash: pc.hash,
                body_hash: pc.body_hash,
            });
            vids.push(vid);
        }
        // Two-store write: vectors first, then chunk rows. A crash
        // between them leaves an orphan/missing vector for this one
        // file; the next resync re-plans it from content hashes and
        // self-heals. Acceptable — no cross-store transaction exists.
        {
            let v = self.vector.lock().unwrap();
            if !p.removed.is_empty() {
                v.remove_many(&p.removed)?;
            }
            if !add_keys.is_empty() {
                v.add_batch(&add_keys, &add_vecs)?;
            }
        }
        self.db
            .lock()
            .unwrap()
            .replace_chunks(file_id, &new_chunks, &vids)?;
        Ok(n)
    }

    /// Plan a group of changed files, embed ONLY the chunks that
    /// actually changed (across all files, one model call), then store.
    fn flush_group(&self, group: Vec<FileWork>) -> Result<usize> {
        if group.is_empty() {
            return Ok(0);
        }
        let plans: Vec<FilePlan> = group
            .into_iter()
            .map(|f| self.plan_file(f))
            .collect::<Result<_>>()?;
        let vectors = {
            // Materialize `header + body` only for the chunks actually
            // being embedded, and only for this group — freed right
            // after the embed call, so the bounded window never holds
            // two copies of a body.
            // Only Fresh chunks consume the embed iterator. Copy
            // chunks also need a fresh vector_id but they reuse the
            // source vector's bytes at apply time (no model call), so
            // they are NOT in the embedded text list.
            let owned: Vec<String> = plans
                .iter()
                .flat_map(|p| p.chunks.iter().filter(|c| matches!(c.reuse, Reuse::Fresh)))
                .map(|c| format!("{}{}", c.embed_header, c.chunk.content))
                .collect();
            let texts: Vec<&str> = owned.iter().map(String::as_str).collect();
            self.embedder
                .embed_documents(&texts, self.config.embed_batch())?
        };
        let mut it = vectors.into_iter();
        let mut chunks_total = 0;
        for p in plans {
            chunks_total += self.apply_plan(p, &mut it)?;
        }
        Ok(chunks_total)
    }

    /// Classify a file. Returns the relative key (so the caller tracks
    /// `seen` for delete-detection) and one of three outcomes. The
    /// expensive tree-sitter parse is skipped entirely for unchanged
    /// files (hash compared first).
    fn classify(
        &self,
        path: &Path,
        existing: &HashMap<String, crate::db::FileState>,
        force: bool,
    ) -> (String, Classified) {
        let rel = self.rel_str(path);
        let (mtime, size) = mtime_size(path);
        let prior = existing.get(&rel);

        // Cheap hash-tree node check: same mtime + size ⇒ assume
        // unchanged and skip the file read, blake3 AND tree-sitter
        // parse entirely (the dominant cost on a clean resync).
        if !force && prior.is_some_and(|st| st.mtime == mtime && st.size == size) {
            return (rel, Classified::Unchanged);
        }

        let Some(content) = Self::read_text(path) else {
            return (rel, Classified::Unchanged);
        };
        let hash = blake3::hash(content.as_bytes()).to_hex().to_string();
        // mtime/size moved but bytes are identical (touch, `git
        // checkout`): nothing to embed, but refresh the DB row so the
        // fast (mtime,size) short-circuit hits next sync instead of
        // re-reading + re-hashing the same bytes forever. Use the
        // stat captured BEFORE the read: re-statting after the read
        // can pick up a concurrent writer's newer mtime (paired with a
        // possibly-torn-read hash) and lock that mismatched pair into
        // the DB. The pre-read stat may be slightly stale relative to
        // the disk if a touch raced the read, but the next sync's
        // mtime mismatch re-reads correctly and self-heals.
        if !force && prior.is_some_and(|st| st.hash == hash) {
            return (rel, Classified::Touched { mtime, size });
        }
        let (lang, chunks) = self.chunker.chunk_file(path, &content);
        if chunks.is_empty() {
            return (rel, Classified::Unchanged);
        }
        let rel2 = rel.clone();
        (
            rel,
            Classified::Changed(FileWork {
                rel: rel2,
                mtime,
                size,
                hash,
                lang,
                chunks,
            }),
        )
    }

    /// Full incremental sync. `progress` receives per-file events.
    pub fn sync<F: FnMut(SyncEvent<'_>)>(&self, force: bool, mut progress: F) -> Result<SyncStats> {
        let started = Instant::now();
        let mut stats = SyncStats::default();

        let files = self.collect_files();
        progress(SyncEvent::Scanned(files.len()));

        let existing = self.db.lock().unwrap().file_states()?;

        // Chunking (CPU: file read + tree-sitter parse) and embedding
        // (GPU, blocking) run as a bounded producer/consumer pipeline so
        // they overlap instead of idling each other's hardware: a
        // producer parallel-classifies one window at a time and forwards
        // results; the consumer (this thread) batches + embeds + writes.
        // The producer runs ahead while the consumer embeds, which also
        // keeps the encoder's length-sorter fed with a continuously
        // refilled candidate pool. `sync_channel(window)` bounds the
        // in-flight chunked files (backpressure) so peak memory stays
        // O(window), not O(repo). Each window is `collect`ed in order
        // and sent sequentially, and the consumer drains in receive
        // order, so vector-id allocation is identical to the serial
        // path (deterministic index). `self.pool` (private, not the
        // global pool) bounds chunking CPU to `sync_threads`.
        use rayon::prelude::*;
        let batch_chunks = self.config.embed_batch().max(1);
        let batch_bytes = self.config.sync.embed_batch_bytes.max(4096);
        let window = self.config.sync.scan_window.max(1);
        let mut group: Vec<FileWork> = Vec::new();
        let mut pending_chunks = 0usize;
        let mut pending_bytes = 0usize;

        let total = files.len();
        let mut seen: HashSet<String> = HashSet::with_capacity(total);
        let mut done = 0usize;
        let mut discovered = 0usize;

        let (tx, rx) = std::sync::mpsc::sync_channel::<(String, Classified)>(window);
        let files_ref = &files;
        let existing_ref = &existing;
        std::thread::scope(|s| -> Result<()> {
            let producer = s.spawn(move || {
                for win in files_ref.chunks(window) {
                    let classified: Vec<(String, Classified)> = self.pool.install(|| {
                        win.par_iter()
                            .map(|p| self.classify(p, existing_ref, force))
                            .collect()
                    });
                    for item in classified {
                        // Err only if the consumer stopped early (flush
                        // failed): nothing left to feed, so exit.
                        if tx.send(item).is_err() {
                            return;
                        }
                    }
                }
            });

            for (rel, outcome) in rx {
                done += 1;
                // Defensive dedup: two paths can produce the same `rel`
                // on case-insensitive filesystems or via a symlink-chain
                // miss in collect_files. Apply only the first outcome
                // for each rel; otherwise a Touched arriving after a
                // Changed would overwrite (mtime, size) while leaving
                // the content_hash from Changed — DB internally
                // inconsistent and the next sync's cheap short-circuit
                // would trust a stat that doesn't match the bytes.
                if seen.contains(&rel) {
                    tracing::warn!("duplicate rel during sync, skipping later outcome: {rel}");
                    continue;
                }
                let changed = matches!(outcome, Classified::Changed(_));
                match outcome {
                    Classified::Unchanged => stats.files_skipped += 1,
                    Classified::Touched { mtime, size } => {
                        // Refresh the stale stat so the cheap path hits
                        // on the next sync. No vectors to touch. If the
                        // row vanished between snapshot and write
                        // (concurrent Clear / external mutation), warn
                        // and drop it from `seen` so the next sync will
                        // re-classify it as Changed and re-embed from
                        // scratch.
                        let existed = self.db.lock().unwrap().touch_file_meta(&rel, mtime, size)?;
                        if existed {
                            stats.files_skipped += 1;
                        } else {
                            tracing::warn!(
                                "file row vanished mid-sync, will re-index next pass: {rel}"
                            );
                            // Skip marking `seen` so delete-detection
                            // doesn't think it was just dropped; next
                            // sync will pick it back up.
                            continue;
                        }
                    }
                    Classified::Changed(work) => {
                        pending_chunks += work.chunks.len();
                        pending_bytes += work.chunks.iter().map(|c| c.content.len()).sum::<usize>();
                        discovered += work.chunks.len();
                        group.push(work);
                        stats.files_indexed += 1;
                    }
                }
                // Emit after classifying so `changed`/`indexed` are
                // accurate: the bar can fix its message on the file
                // actually re-embedded, not flicker through skips.
                progress(SyncEvent::File {
                    done,
                    total,
                    path: &rel,
                    indexed: stats.files_indexed,
                    changed,
                });
                seen.insert(rel);

                if pending_chunks >= batch_chunks || pending_bytes >= batch_bytes {
                    stats.chunks_total += self.flush_group(std::mem::take(&mut group))?;
                    pending_chunks = 0;
                    pending_bytes = 0;
                }
                // Emit every classified file, not only at a flush: the
                // producer runs ahead of the blocking embed, so
                // `discovered` climbs while `embedded` (`chunks_total`)
                // steps only per flush — the buffered lead is visible
                // instead of the two always reading equal post-flush.
                if changed {
                    progress(SyncEvent::Chunks {
                        embedded: stats.chunks_total,
                        discovered,
                    });
                }
            }
            stats.chunks_total += self.flush_group(std::mem::take(&mut group))?;
            progress(SyncEvent::Chunks {
                embedded: stats.chunks_total,
                discovered,
            });
            producer
                .join()
                .map_err(|_| Error::Index("sync classify thread panicked".into()))?;
            Ok(())
        })?;

        let to_delete: Vec<String> = existing
            .keys()
            .filter(|p| !seen.contains(*p))
            .cloned()
            .collect();
        for rel in to_delete {
            self.purge(&rel)?;
            stats.files_deleted += 1;
        }

        let changed = stats.files_indexed > 0 || stats.files_deleted > 0;
        self.finalize(changed)?;
        stats.elapsed_ms = started.elapsed().as_millis() as u64;
        Ok(stats)
    }

    /// True only if the index was never built. A startup sync runs
    /// unconditionally regardless (it is hash-incremental and cheap);
    /// this just reports health.
    pub fn is_stale(&self) -> Result<bool> {
        let last = self.db.lock().unwrap().get_meta("last_sync_at")?;
        Ok(never_synced(last.as_deref()))
    }

    /// CLI-side throttle check (Cursor ~10 min): has it been longer than
    /// `resync_interval_minutes` since the last sync? The MCP server
    /// does not call this — it runs its own periodic background resync
    /// on the same interval plus a startup sync.
    pub fn is_due(&self) -> Result<bool> {
        let last = self.db.lock().unwrap().get_meta("last_sync_at")?;
        Ok(resync_due(
            last.as_deref(),
            self.config.sync.resync_interval_minutes,
        ))
    }

    pub fn status(&self) -> Result<IndexStatus> {
        let db = self.db.lock().unwrap();
        let (files, chunks) = db.counts()?;
        let last = db.get_meta("last_sync_at")?;
        let merkle_root = db.get_meta("merkle_root")?;
        drop(db);
        Ok(IndexStatus::assemble(
            files,
            chunks,
            self.vector.lock().unwrap().len(),
            self.embedder.model_name.clone(),
            last,
            merkle_root,
            &self.config,
        ))
    }

    /// `scope`: optional project-relative dir or file to restrict
    /// results to (Cursor `@folder` / `@file`).
    pub fn search(
        &self,
        query: &str,
        limit: usize,
        scope: Option<&str>,
    ) -> Result<Vec<crate::search::SearchResult>> {
        crate::search::run(self, query, limit, scope)
    }

    /// Small index ⇒ exact brute-force search (HNSW heuristic buys
    /// nothing and can miss the true nearest at this scale).
    pub(crate) fn use_exact(&self) -> bool {
        self.vector.lock().unwrap().len() < self.config.search.exact_below
    }

    pub(crate) fn embedder(&self) -> &Embedder {
        &self.embedder
    }

    /// The cross-encoder reranker, present only when `[rerank] enabled`
    /// (and not under `bench-stub`). `search::run` re-orders the fused
    /// candidate prefix with it when `Some`.
    #[cfg(not(feature = "bench-stub"))]
    pub(crate) fn reranker(&self) -> Option<&crate::rerank::Reranker> {
        self.reranker.as_ref()
    }

    pub(crate) fn with_vector<R>(&self, f: impl FnOnce(&VectorIndex) -> R) -> R {
        f(&self.vector.lock().unwrap())
    }

    pub(crate) fn with_db<R>(&self, f: impl FnOnce(&Db) -> Result<R>) -> Result<R> {
        f(&self.db.lock().unwrap())
    }

    pub fn list_files(&self) -> Result<Vec<crate::db::FileInfo>> {
        self.db.lock().unwrap().list_files()
    }

    pub fn chunks_for_file(&self, rel: &str) -> Result<Vec<crate::db::ChunkRow>> {
        self.db.lock().unwrap().chunks_for_file(rel)
    }
}
