use embedding_search_core::chunker::Chunker;
use embedding_search_core::config::{
    model_spec, normalize_hf_repo, Config, EmbeddingProvider, ExecutionProvider, RemoteConfig,
    DEFAULT_MODEL, SUPPORTED_MODELS,
};
use embedding_search_core::db::{Db, NewChunk};
use std::path::Path;

#[test]
fn config_defaults_are_sane() {
    let c = Config::default();
    assert_eq!(c.model.default, DEFAULT_MODEL);
    assert_eq!(c.sync.max_chunk_bytes, 2048);
    // 0 = auto: resolves to the active model's per-model rec_batch.
    assert_eq!(c.sync.embed_batch_size, 0);
    // default is now jina-code (a transformer) → small auto batch.
    let rec = model_spec(DEFAULT_MODEL).unwrap().rec_batch as usize;
    assert_eq!(c.embed_batch(), rec);
    // transformer default: a too-large explicit value (e.g. a stale
    // pre-per-model config) is clamped down to the safe rec_batch.
    let mut heavy = Config::default();
    heavy.sync.embed_batch_size = 16;
    assert_eq!(heavy.embed_batch(), rec); // 16 → clamped to rec
    heavy.sync.embed_batch_size = 2;
    assert_eq!(heavy.embed_batch(), 2); // smaller-than-rec honored
                                        // static model: an explicit value is honored as-is (no attention
                                        // → no OOM risk, bigger = faster).
    let mut stat = Config::default();
    stat.model.default = "minishlab/potion-multilingual-128M".into();
    stat.sync.embed_batch_size = 7;
    assert_eq!(stat.embed_batch(), 7);
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
    // default is jina-embeddings-v2-base-code (ONNX encoder)
    assert_eq!(spec.dimensions, 768);
    assert_eq!(spec.code, 5);
    assert!(!spec.is_static());
    // local in-process backend by default
    assert_eq!(c.model.provider, EmbeddingProvider::Local);
    // no custom ONNX override by default
    assert!(c.model.onnx_path.is_none());
    assert!(c.model.onnx_query_prefix.is_none() && c.model.onnx_doc_prefix.is_none());
}

#[test]
fn chunk_param_change_invalidates_index() {
    let mut c = Config::default();
    let base = c.index_fingerprint("m", 768, "ct");
    // same inputs → stable
    assert_eq!(base, c.index_fingerprint("m", 768, "ct"));
    // each invalidating knob shifts the fingerprint
    c.sync.max_chunk_bytes += 1;
    assert_ne!(base, c.index_fingerprint("m", 768, "ct"));
    let mut c = Config::default();
    c.model.max_length += 1;
    assert_ne!(base, c.index_fingerprint("m", 768, "ct"));
    // model name / dims also part of identity
    assert_ne!(
        base,
        Config::default().index_fingerprint("other", 768, "ct")
    );
    assert_ne!(base, Config::default().index_fingerprint("m", 384, "ct"));
    // the resolved contract (prefix/pooling) is part of identity too —
    // changing a model's prefix or pooling must force a re-embed.
    assert_ne!(
        base,
        Config::default().index_fingerprint("m", 768, "other-contract")
    );
}

#[test]
fn remote_config_defaults_and_endpoint() {
    let r = RemoteConfig::default();
    assert_eq!(r.batch_size, 64);
    assert_eq!(r.concurrency, 4);
    assert!(r.query_prefix.is_none() && r.doc_prefix.is_none());
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
fn normalize_hf_repo_strips_url_forms_to_id() {
    let id = "minishlab/potion-base-32M";
    assert_eq!(normalize_hf_repo(id), id);
    assert_eq!(
        normalize_hf_repo("https://huggingface.co/minishlab/potion-base-32M"),
        id
    );
    assert_eq!(
        normalize_hf_repo("http://www.huggingface.co/minishlab/potion-base-32M/tree/main"),
        id
    );
    assert_eq!(
        normalize_hf_repo("  hf.co/minishlab/potion-base-32M/  "),
        id
    );
    assert_eq!(
        normalize_hf_repo("https://huggingface.co/minishlab/potion-base-32M/blob/main/onnx"),
        id
    );
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
fn ram_estimate_is_bounded_and_distinguishes_arch() {
    // ONNX-encoder built-in (int8 on CPU): weights + ORT overhead,
    // still well under the old multi-GB blowup.
    let jina = model_spec("jinaai/jina-embeddings-v2-base-code").unwrap();
    let onnx_ram = jina.ram_mb();
    assert!(onnx_ram < 1200, "ONNX RAM estimate {onnx_ram}MB too high");

    // a static (Model2Vec) model: tiny tokenizer overhead, no ORT.
    let potion = model_spec("minishlab/potion-base-32M").unwrap();
    assert!(potion.is_static());
    assert!(
        potion.ram_mb() < onnx_ram,
        "static {} should be lighter than ONNX {onnx_ram}",
        potion.ram_mb()
    );
    assert!(SUPPORTED_MODELS.iter().all(|m| m.ram_mb() > 0));
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
fn chunker_merges_adjacent_small_nodes() {
    // semble-style: small adjacent declarations are coalesced into one
    // bigger chunk (was one-chunk-per-leaf). Three tiny decls well
    // under the 4096 cap → a single merged chunk; a mixed run is typed
    // `block` and the contiguous span keeps the blank lines between.
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
    assert_eq!(chunks.len(), 1, "tiny decls must merge into one chunk");
    let m = &chunks[0];
    assert_eq!(m.node_type, "block"); // heterogeneous run
    assert!(m.symbol.is_none()); // no single decl owns a merged run
    assert!(m.content.contains("fn alpha"));
    assert!(m.content.contains("struct Point"));
    assert!(m.content.contains("fn beta"));
}

#[test]
fn chunker_captures_symbol_name_for_enrichment_header() {
    // The code↔NL embed header needs the declared name: `name` field
    // for fn/struct, first type_identifier for an `impl` (no name
    // field). Bodies are padded past half the cap so no two nodes
    // merge — each stays its own typed, symboled chunk.
    let pad = "x".repeat(3000);
    let fields = (0..400).map(|i| format!("f{i}: i64,")).collect::<String>();
    let code = format!(
        "fn alpha() -> i32 {{ let _p = \"{pad}\"; 1 }}\n\
         struct Point {{ {fields} }}\n\
         impl Point {{ fn beta(&self) -> i32 {{ let _p = \"{pad}\"; self.x }} }}\n"
    );
    let ck = Chunker::new(4096);
    let (_lang, chunks) = ck.chunk_file(Path::new("sample.rs"), &code);
    let sym = |nt: &str| {
        chunks
            .iter()
            .find(|c| c.node_type == nt)
            .and_then(|c| c.symbol.as_deref())
    };
    assert_eq!(sym("function"), Some("alpha"));
    assert_eq!(sym("struct"), Some("Point"));
    assert_eq!(sym("impl"), Some("Point")); // type_identifier fallback
}

#[test]
fn chunker_line_chunks_have_no_symbol() {
    let txt = (0..50).map(|i| format!("line {i}\n")).collect::<String>();
    let ck = Chunker::new(4096);
    let (_lang, chunks) = ck.chunk_file(Path::new("notes.xyz"), &txt);
    assert!(!chunks.is_empty());
    assert!(chunks.iter().all(|c| c.symbol.is_none()));
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

#[test]
fn chunker_crlf_multibyte_does_not_panic() {
    // CRLF + Cyrillic (2-byte) over the 100-line window: the old
    // `len()+1` offset drift sliced mid-char and panicked. >120 lines
    // so the windowing slice (not just one full-content chunk) runs.
    let txt = "импорт SPage из \"@apps/portal\";\r\n".repeat(150);
    let ck = Chunker::new(4096);
    let (_lang, chunks) = ck.chunk_file(Path::new("a.xyz"), &txt);
    assert!(chunks.len() >= 2, "expected windowed chunks");
    // every chunk's bytes are valid UTF-8 (no mid-char split)
    for c in &chunks {
        assert!(std::str::from_utf8(c.content.as_bytes()).is_ok());
    }
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
        body_hash: hash.into(),
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
