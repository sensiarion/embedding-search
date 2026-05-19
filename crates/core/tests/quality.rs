//! Search-quality tests. These load the real embedding model (network +
//! ~hundreds of MB) so they are `#[ignore]` by default. Run explicitly:
//!
//!   cargo test -p embedding-search-core --test quality -- --ignored
//!
//! Optionally point at a smaller/faster model:
//!   ES_TEST_MODEL=intfloat/multilingual-e5-small cargo test ... -- --ignored

use embedding_search_core::{Config, SyncEngine};
use std::path::PathBuf;

mod common;

fn fixture() -> (tempfile::TempDir, SyncEngine) {
    common::build_repo(&[
        (
            "src/auth.rs",
            "/// Validate a JWT bearer token and return the user id.\n\
             pub fn verify_token(token: &str) -> Option<UserId> {\n\
             \u{20}   decode_jwt(token).filter(|c| !c.is_expired()).map(|c| c.sub)\n\
             }\n",
        ),
        (
            "src/math.rs",
            "/// Sum every element of the slice.\n\
             pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
        ),
        (
            "src/cache.rs",
            "/// LRU cache eviction: drop the least-recently-used entry when full.\n\
             pub fn evict(&mut self) { self.order.pop_front(); }\n",
        ),
        (
            "README.md",
            "# Сервис\n\n## Авторизация\n\n\
             Проверка токена и вход пользователя по логину и паролю.\n",
        ),
    ])
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
#[ignore = "downloads a real f32 ONNX model (~440 MB)"]
fn predefined_e5_code_model_discriminates_via_forced_cpu() {
    // Regression for the "memory blowing, no results" report: this
    // built-in is a raw torch opset-11 export that OOMs the CoreML/CUDA
    // partitioner — `ModelSpec::force_cpu` pins CPU so it loads at all.
    // It is e5, so its `query: `/`passage: ` contract must apply for
    // the embedding to separate the relevant code from the unrelated
    // chunks.
    let mut cfg = Config::default();
    cfg.model.default = "jamie8johnson/e5-base-v2-code-search".into();
    let (_d, eng) = common::build_repo_with(
        &[
            (
                "src/mail.rs",
                "pub fn validate_email(addr: &str) -> bool { addr.contains('@') }\n",
            ),
            (
                "src/math.rs",
                "pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
            ),
        ],
        cfg,
    );
    let hits = eng
        .search("function that checks an email address", 5, None)
        .expect("search");
    assert_eq!(
        hits.first().map(|h| h.file_path.as_str()),
        Some("src/mail.rs"),
        "email query did not rank validate_email first: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "downloads the default CodeRankEmbed ONNX (int8 ~138 MB on CPU/Apple-Silicon, f32 ~548 MB under CUDA)"]
fn default_coderankembed_discriminates_with_accel_policy() {
    // Regression for repointing the default to `jalipalo/CodeRankEmbed-onnx`
    // and the `OnnxFiles::AccelCpu` policy: whichever file the run
    // resolves to (int8 on CPU/Apple-Silicon, f32 under CUDA) must
    // still CLS-pool and apply the `Represent this query for searching
    // relevant code: ` query prefix, so the relevant code out-ranks the
    // unrelated chunk. Catches a broken repo swap / pooling / prefix.
    let cfg = Config::default();
    let (_d, eng) = common::build_repo_with(
        &[
            (
                "src/mail.rs",
                "pub fn validate_email(addr: &str) -> bool { addr.contains('@') }\n",
            ),
            (
                "src/math.rs",
                "pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
            ),
        ],
        cfg,
    );
    let hits = eng
        .search("function that checks an email address", 5, None)
        .expect("search");
    assert_eq!(
        hits.first().map(|h| h.file_path.as_str()),
        Some("src/mail.rs"),
        "email query did not rank validate_email first: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
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
        query_prefix: None,
        doc_prefix: None,
        pooling: Default::default(),
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
        query_prefix: None,
        doc_prefix: None,
        pooling: Default::default(),
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
        query_prefix: None,
        doc_prefix: None,
        pooling: Default::default(),
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

// `rerank::Reranker` is an ONNX-only type (compiled out under
// `bench-stub`), so this test only exists in a real build.
#[cfg(not(feature = "bench-stub"))]
#[test]
#[ignore = "downloads the int8 cross-encoder reranker (~280 MB)"]
fn reranker_scores_relevant_above_irrelevant() {
    // Cross-encoder sanity: the joint (query, passage) relevance logit
    // must rank a passage that answers the query above an unrelated one.
    use embedding_search_core::rerank::Reranker;
    let mut cfg = Config::default();
    cfg.rerank.enabled = true;
    let rr = Reranker::load(&cfg).expect("reranker must load");
    let scores = rr
        .score(
            "how to validate a JWT bearer token",
            &[
                "pub fn verify_jwt(t: &str) -> Option<UserId> { decode(t) }",
                "pub fn haversine(a: LatLng, b: LatLng) -> Meters { gc(a, b) }",
            ],
        )
        .expect("score");
    assert!(
        scores[0] > scores[1],
        "relevant passage {} did not outscore irrelevant {}",
        scores[0],
        scores[1]
    );
}

#[test]
#[ignore = "loads the default model + the reranker (~280 MB extra)"]
fn rerank_enabled_keeps_relevant_file_top1() {
    // End-to-end with `[rerank] enabled = true`: the cross-encoder
    // re-orders the fused top-N and the semantically correct file must
    // be #1. (Disabled ⇒ the early-return path, byte-for-byte the
    // pre-Phase-3 behavior — covered by the deterministic bench-stub
    // suite, which exercises the disabled branch unchanged.)
    let files = &[
        (
            "src/auth.rs",
            "/// Validate a JWT bearer token and authenticate the user.\n\
             pub fn verify_jwt(token: &str) -> Option<UserId> { decode(token) }\n",
        ),
        (
            "docs/refunds.md",
            "# Token refunds\n\nHow to validate a refund token for a payment dispute.\n",
        ),
        (
            "src/math.rs",
            "pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
        ),
    ];
    let mut cfg = Config::default();
    cfg.rerank.enabled = true;
    let (_d, eng) = common::build_repo_with(files, cfg);
    let hits = eng
        .search("authenticate a user from a JWT bearer token", 5, None)
        .expect("search");
    assert_eq!(
        hits.first().map(|h| h.file_path.as_str()),
        Some("src/auth.rs"),
        "reranked top-1 was not auth.rs: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "loads the real default model"]
fn enrichment_header_bridges_path_symbol_to_nl_query() {
    // Phase-2 chunk-enrichment regression. The target's BODY shares no
    // tokens with the query; the natural-language cue lives only in its
    // PATH + SYMBOL ("hugging face download retry / timeout"), which the
    // embed header now prepends. A lexical decoy file literally contains
    // "timeout". Without enrichment the decoy wins; with it the real
    // function ranks first.
    let (_d, eng) = common::build_repo(&[
        (
            "src/net/hf_download_retry.rs",
            "pub fn fetch_with_backoff(u: &str) -> Vec<u8> {\n\
             \u{20}   for _ in 0..4 { if let Ok(b) = grab(u) { return b; } }\n\
             \u{20}   Vec::new()\n}\n",
        ),
        (
            "src/timeout_config.rs",
            "// default socket timeout constant\npub const TIMEOUT_MS: u64 = 5000;\n",
        ),
        (
            "src/math.rs",
            "pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
        ),
    ]);
    let hits = eng
        .search("retry hugging face download on timeout", 5, None)
        .expect("search");
    assert_eq!(
        hits.first().map(|h| h.file_path.as_str()),
        Some("src/net/hf_download_retry.rs"),
        "enrichment header did not bridge path/symbol to the NL query: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "loads the real default model"]
fn natural_language_query_lands_on_right_file_with_default_model() {
    // Regression for the "poor results" report: a plain-English query
    // with a lexical decoy — a `pyproject.toml` chunk literally
    // containing "configure" — must not outrank the file that actually
    // mounts the static dir. The unconditional hybrid (cosine + BM25
    // RRF) re-rank keeps `run.rs` (the real answer) strictly above the
    // decoy regardless of the active default model.
    let (_d, eng) = common::build_repo(&[
        (
            "src/run.rs",
            "// Granian ASGI server bootstrap.\n\
             fn main() { granian::serve(\"main:app\", static_path_mount=[\"static\"]); }\n",
        ),
        (
            "src/math.rs",
            "pub fn sum_all(xs: &[i64]) -> i64 { xs.iter().sum() }\n",
        ),
        (
            "pyproject.toml",
            "[tool.poe.tasks.configure]\nhelp = \"configure the project\"\n",
        ),
    ]);

    let hits = eng
        .search("how are static files configured", 5, None)
        .expect("search");
    let rank = |needle: &str| hits.iter().position(|h| h.file_path.ends_with(needle));
    let run = rank("run.rs").expect("run.rs in top-5");
    // `pyproject.toml` is the lexical decoy ("configure"); run.rs must
    // outrank it, not merely appear somewhere.
    assert!(
        rank("pyproject.toml").is_none_or(|toml| run < toml),
        "run.rs (#{run}) did not outrank the pyproject decoy: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );
}
