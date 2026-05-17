use embedding_search_core::chunker::Chunker;
use embedding_search_core::config::{
    model_spec, Config, EmbeddingProvider, ExecutionProvider, Precision, RemoteConfig,
    DEFAULT_MODEL, SUPPORTED_MODELS,
};
use embedding_search_core::db::{Db, NewChunk};
use std::path::Path;

#[test]
fn config_defaults_are_sane() {
    let c = Config::default();
    assert_eq!(c.model.default, DEFAULT_MODEL);
    assert_eq!(c.model.precision, Precision::Fp16);
    assert_eq!(c.sync.max_chunk_bytes, 2048);
    assert_eq!(c.sync.embed_batch_size, 16);
    // chunk byte cap aligned to token cap (~4 bytes/token) so chunks
    // are fully embedded, not truncated.
    assert_eq!(c.model.max_length, 512);
    assert!(c.sync.max_chunk_bytes <= c.model.max_length * 5);
    // memory-safety knobs
    assert!(c.model.max_length <= 512 && c.model.max_length > 0);
    assert!(c.backend.disable_mem_arena);
    assert_eq!(c.backend.execution_provider, ExecutionProvider::Auto);
    assert!(c.sync.embed_batch_bytes >= 4096);
    let spec = c.model_spec().expect("default model known");
    assert_eq!(spec.dimensions, 768);
    assert_eq!(spec.multilingual, 5);
    assert!(spec.supports_precision()); // default is a Xenova HF model
                                        // local in-process backend by default
    assert_eq!(c.model.provider, EmbeddingProvider::Local);
    // no custom ONNX override by default
    assert!(c.model.onnx_path.is_none());
    assert!(!c.model.onnx_e5_prefix);
}

#[test]
fn chunk_param_change_invalidates_index() {
    let mut c = Config::default();
    let base = c.index_fingerprint("m", 768);
    // same inputs → stable
    assert_eq!(base, c.index_fingerprint("m", 768));
    // each invalidating knob shifts the fingerprint
    c.sync.max_chunk_bytes += 1;
    assert_ne!(base, c.index_fingerprint("m", 768));
    let mut c = Config::default();
    c.model.max_length += 1;
    assert_ne!(base, c.index_fingerprint("m", 768));
    let mut c = Config::default();
    c.model.precision = Precision::Int8;
    assert_ne!(base, c.index_fingerprint("m", 768));
    // model name / dims also part of identity
    assert_ne!(base, Config::default().index_fingerprint("other", 768));
    assert_ne!(base, Config::default().index_fingerprint("m", 384));
}

#[test]
fn remote_config_defaults_and_endpoint() {
    let r = RemoteConfig::default();
    assert_eq!(r.batch_size, 64);
    assert_eq!(r.concurrency, 4);
    assert!(!r.e5_prefix);
    assert!(r.dimensions.is_none());
    assert_eq!(r.endpoint(), "http://localhost:4000/v1/embeddings");
}

#[test]
fn select_model_activates_registered_remote() {
    let mut c = Config::default();
    c.remote_models.push(RemoteConfig {
        name: "ds".into(),
        base_url: "http://x/v1".into(),
        model: "m".into(),
        ..RemoteConfig::default()
    });
    let p = c.select_model("ds").unwrap();
    assert_eq!(p, EmbeddingProvider::Openai);
    assert_eq!(c.model.provider, EmbeddingProvider::Openai);
    assert_eq!(c.model.default, "ds");
    // chosen entry copied into the active [remote] section
    assert_eq!(c.remote.base_url, "http://x/v1");
}

#[test]
fn select_model_builtin_is_local() {
    let mut c = Config::default();
    let p = c.select_model(DEFAULT_MODEL).unwrap();
    assert_eq!(p, EmbeddingProvider::Local);
    assert_eq!(c.model.provider, EmbeddingProvider::Local);
}

#[test]
fn select_model_rejects_unknown_name() {
    assert!(Config::default().select_model("nope/nope").is_err());
}

#[test]
fn remote_api_key_expands_env_var() {
    std::env::set_var("ES_TEST_KEY", "secret-123");
    let r = RemoteConfig {
        api_key: "$ES_TEST_KEY".into(),
        ..RemoteConfig::default()
    };
    assert_eq!(r.resolved_api_key(), "secret-123");

    let braced = RemoteConfig {
        api_key: "${ES_TEST_KEY}".into(),
        ..RemoteConfig::default()
    };
    assert_eq!(braced.resolved_api_key(), "secret-123");

    let literal = RemoteConfig {
        api_key: "plain".into(),
        ..RemoteConfig::default()
    };
    assert_eq!(literal.resolved_api_key(), "plain");
}

#[test]
fn ram_estimate_scales_with_precision() {
    let e5_base = model_spec("intfloat/multilingual-e5-base").unwrap();
    let f32 = e5_base.ram_mb(Precision::Full);
    let f16 = e5_base.ram_mb(Precision::Fp16);
    let i8 = e5_base.ram_mb(Precision::Int8);
    assert!(f32 > f16 && f16 > i8, "{f32} {f16} {i8}");
    // fp16 default must be well under the old 32GB / f32 blowup
    assert!(f16 < 1200, "fp16 RAM estimate {f16}MB too high");

    // models without an HF repo ignore precision (f32 only)
    let jina = model_spec("jinaai/jina-embeddings-v2-base-code").unwrap();
    assert!(!jina.supports_precision());
    assert_eq!(jina.ram_mb(Precision::Int8), jina.ram_mb(Precision::Full));
    assert!(SUPPORTED_MODELS.iter().any(|m| m.supports_precision()));
}

#[test]
fn chunker_caps_oversized_chunk() {
    // a single 200KB line must never become one chunk
    let big = format!("const X = \"{}\";", "a".repeat(200_000));
    let ck = Chunker::new(4096);
    let (_lang, chunks) = ck.chunk_file(std::path::Path::new("min.js"), &big);
    assert!(!chunks.is_empty());
    assert!(
        chunks.iter().all(|c| c.content.len() <= 4096),
        "max chunk = {}",
        chunks.iter().map(|c| c.content.len()).max().unwrap()
    );
}

#[test]
fn unknown_model_rejected() {
    assert!(model_spec("does/not-exist").is_none());
}

#[test]
fn chunker_extracts_rust_functions() {
    let code = r#"
fn alpha() -> i32 { 1 }

struct Point { x: i32, y: i32 }

impl Point {
    fn beta(&self) -> i32 { self.x }
}
"#;
    let ck = Chunker::new(4096);
    let (lang, chunks) = ck.chunk_file(Path::new("sample.rs"), code);
    assert_eq!(lang, "rust");
    assert!(!chunks.is_empty());
    let kinds: Vec<_> = chunks.iter().map(|c| c.node_type.as_str()).collect();
    assert!(kinds.contains(&"function"));
    assert!(kinds.iter().any(|k| *k == "struct" || *k == "impl"));
}

#[test]
fn chunker_markdown_splits_on_headers() {
    let md = "# A\nintro\n\n## B\nmore text\n\n## C\nlast\n";
    let ck = Chunker::new(4096);
    let (lang, chunks) = ck.chunk_file(Path::new("doc.md"), md);
    assert_eq!(lang, "markdown");
    assert!(chunks.len() >= 2);
    assert!(chunks.iter().all(|c| c.node_type == "heading"));
}

#[test]
fn chunker_unknown_ext_line_window() {
    let txt = "line\n".repeat(10);
    let ck = Chunker::new(4096);
    let (lang, chunks) = ck.chunk_file(Path::new("notes.xyz"), &txt);
    assert_eq!(lang, "text");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].node_type, "lines");
}

fn chunk(idx: i32, content: &str, hash: &str) -> NewChunk {
    NewChunk {
        chunk_index: idx,
        content: content.into(),
        start_byte: 0,
        end_byte: content.len() as i64,
        node_type: "function".into(),
        language: "rust".into(),
        content_hash: hash.into(),
    }
}

#[test]
fn db_crud_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::open_or_create(dir.path()).unwrap();

    let fid = db.upsert_file("src/x.rs", 123, 456, "hash1").unwrap();
    let vids = db.alloc_vector_ids(2).unwrap();
    let chunks = vec![chunk(0, "fn a() {}", "ha"), chunk(1, "fn b() {}", "hb")];
    db.replace_chunks(fid, &chunks, &vids).unwrap();

    let (files, chs) = db.counts().unwrap();
    assert_eq!((files, chs), (1, 2));

    let row = db.chunk_by_vector_id(vids[0]).unwrap().unwrap();
    assert_eq!(row.content, "fn a() {}");

    let removed = db.delete_file_by_path("src/x.rs").unwrap();
    assert_eq!(removed, vids);
}

#[test]
fn alloc_vector_ids_never_reuses() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open_or_create(dir.path()).unwrap();
    let first = db.alloc_vector_ids(2).unwrap();
    let next = db.alloc_vector_ids(1).unwrap();
    assert!(next[0] > first[1]);
}

#[test]
fn replace_chunks_keeps_file_row_for_vector_reuse() {
    let dir = tempfile::tempdir().unwrap();
    let mut db = Db::open_or_create(dir.path()).unwrap();
    let fid = db.upsert_file("src/x.rs", 1, 9, "h").unwrap();
    let vids = db.alloc_vector_ids(2).unwrap();
    db.replace_chunks(
        fid,
        &[chunk(0, "fn a() {}", "ha"), chunk(1, "fn b() {}", "hb")],
        &vids,
    )
    .unwrap();

    // fingerprints (hash -> vector_id) drive cross-edit reuse
    assert_eq!(
        db.chunk_fingerprints_for_file("src/x.rs").unwrap(),
        vec![("ha".into(), vids[0]), ("hb".into(), vids[1])]
    );

    // an edit that drops one chunk: file row + id survive
    db.replace_chunks(fid, &[chunk(0, "fn a() {}", "ha")], &vids[..1])
        .unwrap();
    let (files, chs) = db.counts().unwrap();
    assert_eq!((files, chs), (1, 1));
    assert_eq!(
        db.chunk_fingerprints_for_file("src/x.rs").unwrap(),
        vec![("ha".into(), vids[0])]
    );
}
