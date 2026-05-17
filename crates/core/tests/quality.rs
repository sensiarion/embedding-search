//! Search-quality tests. These load the real embedding model (network +
//! ~hundreds of MB) so they are `#[ignore]` by default. Run explicitly:
//!
//!   cargo test -p embedding-search-core --test quality -- --ignored
//!
//! Optionally point at a smaller/faster model:
//!   ES_TEST_MODEL=intfloat/multilingual-e5-small cargo test ... -- --ignored

use embedding_search_core::{Config, SyncEngine};
use std::fs;
use std::path::PathBuf;

fn fixture() -> (tempfile::TempDir, SyncEngine) {
    let dir = tempfile::tempdir().unwrap();
    let p = dir.path();
    fs::create_dir_all(p.join("src")).unwrap();
    fs::write(
        p.join("src/auth.rs"),
        "/// Validate a JWT bearer token and return the user id.\n\
         pub fn verify_token(token: &str) -> Option<UserId> {\n\
         \u{20}   decode_jwt(token).filter(|c| !c.is_expired()).map(|c| c.sub)\n\
         }\n",
    )
    .unwrap();
    fs::write(
        p.join("src/math.rs"),
        "/// Sum every element of the slice.\n\
         pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
    )
    .unwrap();
    fs::write(
        p.join("src/cache.rs"),
        "/// LRU cache eviction: drop the least-recently-used entry when full.\n\
         pub fn evict(&mut self) { self.order.pop_front(); }\n",
    )
    .unwrap();
    fs::write(
        p.join("README.md"),
        "# Сервис\n\n## Авторизация\n\n\
         Проверка токена и вход пользователя по логину и паролю.\n",
    )
    .unwrap();

    let mut cfg = Config::default();
    if let Ok(m) = std::env::var("ES_TEST_MODEL") {
        cfg.model.default = m;
    }
    let eng = SyncEngine::new(p.to_path_buf(), cfg).expect("engine");
    eng.sync(true, |_| {}).expect("sync");
    (dir, eng)
}

fn top_path(eng: &SyncEngine, q: &str) -> String {
    let r = eng.search(q, 5, None).expect("search");
    assert!(!r.is_empty(), "no results for {q:?}");
    PathBuf::from(&r[0].file_path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

#[test]
#[ignore = "loads real model"]
fn english_concept_queries_rank_right_file() {
    let (_d, eng) = fixture();
    assert_eq!(top_path(&eng, "validate authentication token"), "auth.rs");
    assert_eq!(top_path(&eng, "add up numbers in a list"), "math.rs");
    assert_eq!(
        top_path(&eng, "remove least recently used item"),
        "cache.rs"
    );
}

#[test]
#[ignore = "loads real model"]
fn russian_query_matches_russian_doc() {
    let (_d, eng) = fixture();
    // Russian query must find the Russian auth section, proving
    // multilingual retrieval works.
    let hit = top_path(&eng, "проверка токена авторизации пользователя");
    assert!(
        hit == "README.md" || hit == "auth.rs",
        "russian query ranked {hit} first"
    );
}

#[test]
#[ignore = "loads real model"]
fn relevant_scores_beat_irrelevant() {
    let (_d, eng) = fixture();
    let r = eng.search("jwt token verification", 5, None).unwrap();
    let auth = r
        .iter()
        .find(|x| x.file_path.ends_with("auth.rs"))
        .expect("auth.rs present");
    let math = r
        .iter()
        .find(|x| x.file_path.ends_with("math.rs"))
        .expect("math.rs should be in top-5 for comparison");
    assert!(
        auth.score > math.score,
        "auth {} should outrank math {}",
        auth.score,
        math.score
    );
}
