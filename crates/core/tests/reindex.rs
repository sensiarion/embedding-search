//! Regression: changing the configured model on an EXISTING project
//! index must wipe + rebuild cleanly (it used to fail with
//! `UNIQUE constraint failed: chunks.vector_id`). Deterministic under
//! `bench-stub` (no model/network); the stub's dimensions come from the
//! model spec, so swapping models also changes dims → the index
//! fingerprint changes, exercising the wipe path.
//!
//!   cargo test -p embedding-search-core --features bench-stub --test reindex
#![cfg(feature = "bench-stub")]

use embedding_search_core::{Config, SyncEngine};
use std::fs;

fn cfg_for(model: &str) -> Config {
    let mut c = Config::default();
    c.model.default = model.to_string();
    c
}

#[test]
fn model_change_rebuilds_without_vector_id_collision() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path();
    fs::create_dir_all(p.join("src")).expect("mkdir");
    for i in 0..6 {
        fs::write(
            p.join(format!("src/m{i}.rs")),
            format!(
                "//! module {i}\npub fn alpha{i}() {{}}\npub fn beta{i}() {{}}\nstruct S{i};\n"
            ),
        )
        .expect("write");
    }

    // Model A: potion-multilingual-128M (256-dim).
    let e1 = SyncEngine::new(
        p.to_path_buf(),
        cfg_for("minishlab/potion-multilingual-128M"),
    )
    .expect("engine A");
    let s1 = e1.sync(true, |_| {}).expect("sync A");
    assert!(s1.chunks_total > 0, "model A indexed nothing");
    drop(e1);

    // Model B: different name AND dims (512) → fingerprint mismatch →
    // the on-disk index must be wiped and rebuilt, not appended to.
    let e2 = SyncEngine::new(p.to_path_buf(), cfg_for("minishlab/potion-base-32M"))
        .expect("engine B must rebuild on model change");
    let s2 = e2
        .sync(true, |_| {})
        .expect("re-sync after model swap must not hit a vector_id collision");
    assert!(s2.chunks_total > 0, "model B rebuilt nothing");

    let hits = e2
        .search("alpha1 function", 5, None)
        .expect("search post-swap");
    assert!(!hits.is_empty(), "no results after model swap rebuild");

    // Swapping back to model A must stay clean too.
    drop(e2);
    let e3 = SyncEngine::new(
        p.to_path_buf(),
        cfg_for("minishlab/potion-multilingual-128M"),
    )
    .expect("engine C");
    e3.sync(true, |_| {})
        .expect("third model swap must be clean");
}

/// Phase-2 reuse-hash correctness: the enriched embed text and the
/// `plan_file` identity hash must derive from the SAME string. If they
/// diverged, a forced re-sync (which always goes through `plan_file`,
/// bypassing the mtime fast-path) would corrupt the index or change
/// results. A no-op `sync(false)` must also skip every unchanged file.
#[test]
fn forced_resync_is_stable_with_enriched_hash() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path();
    fs::create_dir_all(p.join("src")).expect("mkdir");
    fs::write(
        p.join("src/auth.rs"),
        "/// Validate a JWT bearer token.\npub fn verify_jwt(t: &str) -> bool { !t.is_empty() }\n",
    )
    .expect("write");
    fs::write(
        p.join("src/cache.rs"),
        "/// LRU eviction.\npub fn evict_lru(&mut self) { self.q.pop_front(); }\n",
    )
    .expect("write");

    let eng =
        SyncEngine::new(p.to_path_buf(), cfg_for("minishlab/potion-base-32M")).expect("engine");
    let s1 = eng.sync(true, |_| {}).expect("initial sync");
    assert!(s1.chunks_total > 0, "indexed nothing");
    let want: Vec<String> = eng
        .search("verify jwt", 5, None)
        .expect("search")
        .into_iter()
        .map(|h| h.file_path)
        .collect();
    assert!(!want.is_empty(), "no results after initial sync");

    // No-op resync: nothing changed → every file skipped, none embedded.
    let s2 = eng.sync(false, |_| {}).expect("no-op resync");
    assert_eq!(s2.files_indexed, 0, "no-op resync re-embedded a file");
    assert_eq!(s2.files_skipped, 2, "no-op resync did not skip both files");

    // Forced resync: bypasses the mtime fast-path, so every chunk goes
    // through `plan_file` → blake3(enriched). The stored content_hash
    // (also blake3(enriched)) must match so reuse fires and results are
    // byte-for-byte identical — proves the two hashes share one string.
    let s3 = eng.sync(true, |_| {}).expect("forced resync");
    assert_eq!(
        s3.chunks_total, s1.chunks_total,
        "forced resync changed the chunk count"
    );
    let got: Vec<String> = eng
        .search("verify jwt", 5, None)
        .expect("search")
        .into_iter()
        .map(|h| h.file_path)
        .collect();
    assert_eq!(got, want, "forced resync changed search results");
}

/// `git checkout` rewrites file mtimes without changing content. The
/// fast `(mtime, size)` short-circuit in `classify` won't fire on a
/// touched file unless the DB row's `last_modified` is bumped to match
/// the new disk mtime. Without `touch_file_meta` we'd re-read +
/// re-hash the same bytes on every sync forever.
#[test]
fn touch_refreshes_db_meta_so_next_sync_short_circuits() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path();
    fs::create_dir_all(p.join("src")).expect("mkdir");
    let f = p.join("src/lib.rs");
    fs::write(&f, "/// alpha.\npub fn one() {}\n").expect("write");

    let eng =
        SyncEngine::new(p.to_path_buf(), cfg_for("minishlab/potion-base-32M")).expect("engine");
    eng.sync(true, |_| {}).expect("initial sync");

    // Bump mtime ~3s into the future without touching the bytes.
    let later = std::time::SystemTime::now() + std::time::Duration::from_secs(3);
    std::fs::OpenOptions::new()
        .write(true)
        .open(&f)
        .expect("open")
        .set_modified(later)
        .expect("set mtime");

    // Hash matches → Touched → DB row refreshed, no re-embed.
    let s1 = eng.sync(false, |_| {}).expect("touched sync");
    assert_eq!(s1.files_indexed, 0, "touched file was re-embedded");
    assert_eq!(s1.files_skipped, 1);

    // Disk and DB mtime must now agree, so the next sync hits the
    // cheap stat short-circuit instead of re-reading + re-hashing.
    let disk = f
        .metadata()
        .and_then(|m| m.modified())
        .expect("modified")
        .duration_since(std::time::UNIX_EPOCH)
        .expect("epoch")
        .as_secs() as i64;
    let infos = eng.list_files().expect("list_files");
    assert_eq!(infos.len(), 1);
    assert_eq!(
        infos[0].last_modified, disk,
        "touch_file_meta did not refresh the DB row"
    );
}

/// Cross-file chunk reuse (B7 phase 2). Renaming a file shifts the
/// embed-text header (path component) so intra-file `Same` reuse
/// misses, but the header-independent `body_hash` lookup still finds
/// the source vector → new file Copy-reuses the existing embedding
/// without re-running the model. Verifies (a) clean indexing across
/// the rename (no `vector_id UNIQUE` violation) and (b) search keeps
/// finding the renamed file.
#[test]
fn cross_file_copy_survives_rename() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path();
    fs::create_dir_all(p.join("src")).expect("mkdir");
    let body = "/// auth helper.\npub fn check_token(t: &str) -> bool { !t.is_empty() }\n";
    fs::write(p.join("src/a.rs"), body).expect("write a");

    let eng =
        SyncEngine::new(p.to_path_buf(), cfg_for("minishlab/potion-base-32M")).expect("engine");
    let s1 = eng.sync(true, |_| {}).expect("sync 1");
    assert!(s1.chunks_total > 0);

    fs::rename(p.join("src/a.rs"), p.join("src/b.rs")).expect("rename");
    let s2 = eng.sync(false, |_| {}).expect("sync 2");
    assert!(s2.files_indexed >= 1, "sync 2 indexed nothing new");
    assert_eq!(s2.files_deleted, 1, "old path was not retired");

    let hits: Vec<String> = eng
        .search("check token", 5, None)
        .expect("search")
        .into_iter()
        .map(|h| h.file_path)
        .collect();
    assert!(
        hits.iter().any(|p| p.ends_with("b.rs")),
        "lost b.rs after rename: {hits:?}"
    );
}

/// Two distinct files with identical bodies must index cleanly side-by-
/// side (no `vector_id UNIQUE` violation — each chunk row owns its
/// vector_id; Copy duplicates the source bytes into the new key).
#[test]
fn cross_file_duplicate_content_indexes_and_searches() {
    let dir = tempfile::tempdir().expect("tmp");
    let p = dir.path();
    fs::create_dir_all(p.join("src")).expect("mkdir");
    let body = "/// auth helper.\npub fn check_token(t: &str) -> bool { !t.is_empty() }\n";
    fs::write(p.join("src/a.rs"), body).expect("write a");

    let eng =
        SyncEngine::new(p.to_path_buf(), cfg_for("minishlab/potion-base-32M")).expect("engine");
    eng.sync(true, |_| {}).expect("sync 1");
    fs::write(p.join("src/b.rs"), body).expect("write b");
    eng.sync(false, |_| {}).expect("sync 2");

    let hits: Vec<String> = eng
        .search("check token", 5, None)
        .expect("search")
        .into_iter()
        .map(|h| h.file_path)
        .collect();
    assert!(
        hits.iter().any(|p| p.ends_with("a.rs")),
        "lost a.rs: {hits:?}"
    );
    assert!(
        hits.iter().any(|p| p.ends_with("b.rs")),
        "lost b.rs: {hits:?}"
    );
}
