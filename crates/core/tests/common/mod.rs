//! Shared integration-test helper. Not a test binary itself (it lives
//! in a subdir); each test file pulls it in with `mod common;`.
#![allow(dead_code)] // each test binary uses only the helpers it needs

use embedding_search_core::{Config, SyncEngine};
use std::fs;
use tempfile::TempDir;

/// Write `(relative path, contents)` files into a fresh temp dir
/// (creating parents), then build a `SyncEngine` with `cfg` and run a
/// full sync. The `TempDir` is returned so the caller keeps it alive.
pub fn build_repo_with(files: &[(&str, &str)], cfg: Config) -> (TempDir, SyncEngine) {
    let dir = tempfile::tempdir().expect("tmp");
    for (rel, body) in files {
        let path = dir.path().join(rel);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("mkdir");
        }
        fs::write(path, body).expect("write");
    }
    let eng = SyncEngine::new(dir.path().to_path_buf(), cfg).expect("engine");
    eng.sync(true, |_| {}).expect("sync");
    (dir, eng)
}

/// `build_repo_with` using the default config (the default model is
/// honored, or `ES_TEST_MODEL` if set — see `model_config`).
pub fn build_repo(files: &[(&str, &str)]) -> (TempDir, SyncEngine) {
    build_repo_with(files, model_config())
}

/// Default config, with the model overridden by `ES_TEST_MODEL` when
/// set — lets the quality suite run against a smaller/faster model.
pub fn model_config() -> Config {
    let mut cfg = Config::default();
    if let Ok(m) = std::env::var("ES_TEST_MODEL") {
        cfg.model.default = m;
    }
    cfg
}
