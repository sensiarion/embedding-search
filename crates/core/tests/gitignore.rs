//! Regression: `.gitignore` must be honored even when the indexed
//! directory is NOT a git work tree. `ignore::WalkBuilder` defaults
//! `require_git(true)`, so before the `require_git(false)` fix a plain
//! directory (or a repo before its first commit) indexed files the
//! user explicitly git-ignored. Deterministic under `bench-stub`
//! (no model / network).
//!
//!   cargo test -p embedding-search-core --features bench-stub --test gitignore
#![cfg(feature = "bench-stub")]

mod common;

use common::build_repo;

#[test]
fn gitignored_file_is_not_indexed_in_a_non_git_dir() {
    // tempfile dirs are NOT git repos — exactly the failing case.
    let (_dir, eng) = build_repo(&[
        (".gitignore", "secret.txt\nignored_dir/\n"),
        ("secret.txt", "TOKEN = sk-do-not-index\n"),
        ("ignored_dir/buried.rs", "fn buried() {}\n"),
        ("kept.rs", "fn kept() {}\n"),
    ]);

    let indexed: Vec<String> = eng
        .list_files()
        .expect("list")
        .into_iter()
        .map(|f| f.path)
        .collect();

    assert!(
        indexed.iter().any(|p| p == "kept.rs"),
        "non-ignored source must still be indexed: {indexed:?}"
    );
    assert!(
        !indexed.iter().any(|p| p == "secret.txt"),
        "git-ignored file must NOT be indexed: {indexed:?}"
    );
    assert!(
        !indexed.iter().any(|p| p.starts_with("ignored_dir/")),
        "git-ignored directory must NOT be indexed: {indexed:?}"
    );
}
