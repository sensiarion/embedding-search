//! End-to-end search test on a synthetic repo. Runs under the
//! `bench-stub` embedder so it is deterministic and needs no model /
//! network. The vector side is hash noise, but on a tiny corpus the
//! over-fetch neighborhood is the WHOLE index, so the hybrid BM25
//! re-rank is authoritative — this exercises the full pipeline
//! (sync → chunk → db → hybrid RRF search → scope filter) and proves a
//! distinctive keyword query lands on the right file.
//!
//!   cargo test -p embedding-search-core --features bench-stub --test search_synthetic
//!
//! Without the feature the file compiles to nothing (no real model is
//! ever downloaded in unit/integration CI).
#![cfg(feature = "bench-stub")]

use embedding_search_core::{Config, SyncEngine};

mod common;

/// Five single-domain files; each query's terms occur in exactly one of
/// them, so the lexical channel has an unambiguous correct answer.
fn repo() -> (tempfile::TempDir, SyncEngine) {
    // Pin a no-prefix model: this test asserts the lexical (BM25)
    // channel is authoritative on a tiny corpus, so the stub query
    // vector must be hash(raw query) — a model with a query prefix
    // would perturb it and is irrelevant to what's under test.
    let mut cfg = Config::default();
    cfg.model.default = "jinaai/jina-embeddings-v2-base-code".into();
    common::build_repo_with(
        &[
            (
                "src/auth.rs",
                "/// Validate a JWT bearer token and authenticate the user.\n\
             pub fn verify_jwt(token: &str) -> Option<UserId> { decode(token) }\n",
            ),
            (
                "src/cache.rs",
                "/// LRU cache eviction: drop the least recently used entry.\n\
             pub fn evict_lru(&mut self) { self.order.pop_front(); }\n",
            ),
            (
                "src/payment.rs",
                "/// Charge a Stripe payment and issue a refund on an invoice.\n\
             pub fn refund_invoice(id: InvoiceId) -> Result<Refund> { stripe_refund(id) }\n",
            ),
            (
                "src/email.rs",
                "/// SMTP email sender with MIME attachment support.\n\
             pub fn send_attachment(to: &str, file: &Path) -> Smtp { smtp_send(to, file) }\n",
            ),
            (
                "src/geo.rs",
                "/// Haversine great-circle distance between two latitude/longitude points.\n\
             pub fn haversine(a: LatLng, b: LatLng) -> Meters { gc_distance(a, b) }\n",
            ),
        ],
        cfg,
    )
}

#[test]
fn distinctive_query_surfaces_its_file_in_top_results() {
    let (_d, eng) = repo();
    // The lexical channel gives the keyword-bearing file the best BM25
    // rank, so the fused result must contain it well inside the top-5
    // even though the stub vector side is noise. (Top-3 is the
    // tolerance for the one possible stub-cosine tie ahead of it.)
    for (query, want) in [
        ("jwt bearer token authentication", "src/auth.rs"),
        ("lru cache eviction least recently used", "src/cache.rs"),
        ("stripe payment refund invoice", "src/payment.rs"),
        ("smtp email attachment sender", "src/email.rs"),
        ("haversine latitude longitude distance", "src/geo.rs"),
    ] {
        let hits = eng.search(query, 5, None).expect("search");
        let pos = hits
            .iter()
            .position(|h| h.file_path.ends_with(want))
            .unwrap_or_else(|| panic!("{want} missing from top-5 for {query:?}"));
        assert!(
            pos < 3,
            "{want} ranked #{} for {query:?} (expected top-3)",
            pos + 1
        );
    }
}

#[test]
fn scope_restricts_results_to_the_path() {
    let (_d, eng) = repo();
    // A scoped query must only ever return chunks under that path,
    // regardless of where the embedding/lexical signal points.
    let hits = eng
        .search("token payment distance", 5, Some("src/payment.rs"))
        .expect("search");
    assert!(!hits.is_empty(), "scoped query returned nothing");
    assert!(
        hits.iter().all(|h| h.file_path == "src/payment.rs"),
        "scope leaked: {:?}",
        hits.iter().map(|h| &h.file_path).collect::<Vec<_>>()
    );
}

#[test]
fn symbol_only_query_still_returns_without_lexical_terms() {
    let (_d, eng) = repo();
    // A query with no >=2-char alnum token (pure punctuation) skips the
    // lexical channel; the pure-vector path must still return, not panic.
    let hits = eng.search("::", 5, None).expect("search");
    assert!(!hits.is_empty(), "vector-only fallback returned nothing");
}
