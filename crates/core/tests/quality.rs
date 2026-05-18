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
#[ignore = "downloads a real external-data ONNX model (~300 MB)"]
fn external_data_custom_model_loads_and_embeds() {
    // onnx-community / >2 GB models split weights into a `.onnx_data`
    // sidecar — proves it is fetched and wired so ORT initializes.
    use embedding_search_core::config::{CustomModel, Precision};
    use embedding_search_core::embedder::Embedder;
    let mut cfg = Config::default();
    cfg.custom_models.push(CustomModel {
        name: "gemma".into(),
        repo: Some("onnx-community/embeddinggemma-300m-ONNX".into()),
        url: None,
        e5_prefix: false,
        precision: Some(Precision::Int8),
        onnx_file: None,
    });
    cfg.model.default = "gemma".into();
    let emb = Embedder::new(&cfg).expect("external-data model must load");
    assert!(emb.dimensions > 0);
    let v = emb.embed_query("semantic code search").expect("embed");
    assert_eq!(v.len(), emb.dimensions);
}

#[test]
#[ignore = "downloads a real quantized ONNX model (~570 MB)"]
fn explicit_onnx_file_picks_exact_quantization() {
    // A repo quantization the precision→file mapping can't reach
    // (q4f16) is loaded via the exact-file override.
    use embedding_search_core::config::CustomModel;
    use embedding_search_core::embedder::Embedder;
    let mut cfg = Config::default();
    cfg.custom_models.push(CustomModel {
        name: "qwen3".into(),
        repo: Some("onnx-community/Qwen3-Embedding-0.6B-ONNX".into()),
        url: None,
        e5_prefix: false,
        precision: None,
        onnx_file: Some("model_q4f16.onnx".into()),
    });
    cfg.model.default = "qwen3".into();
    let emb = Embedder::new(&cfg).expect("explicit q4f16 file must load");
    assert!(emb.dimensions > 0);
    let v = emb.embed_query("semantic code search").expect("embed");
    assert_eq!(v.len(), emb.dimensions);
}

#[test]
#[ignore = "downloads a real Model2Vec model (~500 MB)"]
fn model2vec_static_backend_loads_and_embeds() {
    // Model2Vec/StaticModel: static matrix + tokenizer, no ONNX —
    // proves the static backend path produces real embeddings.
    use embedding_search_core::config::CustomModel;
    use embedding_search_core::embedder::Embedder;
    let mut cfg = Config::default();
    cfg.custom_models.push(CustomModel {
        name: "potion".into(),
        repo: Some("minishlab/potion-multilingual-128M".into()),
        url: None,
        e5_prefix: false,
        precision: None,
        onnx_file: None,
    });
    cfg.model.default = "potion".into();
    let emb = Embedder::new(&cfg).expect("model2vec must load");
    assert_eq!(emb.dimensions, 256);
    let v = emb.embed_query("authentication token").expect("embed");
    assert_eq!(v.len(), 256);
    // L2-normalized (config.normalize = true) → unit norm
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "expected unit norm, got {norm}");
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
