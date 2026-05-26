//! Dev tasks: synthetic corpus generation + indexing benchmark.
//!
//!   cargo xtask gen-corpus [--files N] [--seed S] [--out DIR]
//!   cargo xtask bench       [--files N] [--seed S]
//!
//! The benchmark uses the `bench-stub` embedder (no model load / network)
//! so it measures chunking + db + usearch + pipeline cost only, and is
//! deterministic + CI-friendly.

use anyhow::{Context, Result};
use embedding_search_core::{Config, SyncEngine};
// `eval` runs only in a real (non-stub) build (`run_eval` re-execs out
// of the bench-stub alias). The `rerank` core module + a real Embedder
// don't exist under `bench-stub`, so the whole eval cluster is gated —
// otherwise the alias build (`cargo xtask bench`/`bump`) won't compile.
#[cfg(not(feature = "bench-stub"))]
use embedding_search_core::config::SUPPORTED_MODELS;
#[cfg(not(feature = "bench-stub"))]
use embedding_search_core::{embedder::Embedder, rerank::Reranker};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Default model set the eval compares when `--models` is omitted: the
/// two static Model2Vec builtins + the default f16 transformer. Pass
/// `--models a,b,c` (or `--models all` for every entry in
/// `SUPPORTED_MODELS`) to override.
#[cfg(not(feature = "bench-stub"))]
const DEFAULT_EVAL_MODELS: &[&str] = &[
    "minishlab/potion-multilingual-128M",
    "minishlab/potion-base-32M",
    "sensiarion/CodeRankEmbed-f16",
];

/// Resolve `--models` selector to a concrete list. `all` expands to
/// every built-in registered in `SUPPORTED_MODELS`; otherwise it's a
/// comma-separated explicit list (built-in or registered custom name).
/// `""` (no flag) falls back to `DEFAULT_EVAL_MODELS`.
#[cfg(not(feature = "bench-stub"))]
fn pick_models(selector: &str) -> Vec<String> {
    if selector.is_empty() {
        return DEFAULT_EVAL_MODELS.iter().map(|&s| s.to_string()).collect();
    }
    if selector == "all" {
        return SUPPORTED_MODELS
            .iter()
            .map(|m| m.name.to_string())
            .collect();
    }
    selector
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

fn arg(args: &[String], key: &str, default: &str) -> String {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| default.to_string())
}

/// Deterministic splitmix64 — no `rand` dependency, reproducible corpus.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

const LANGS: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("ts", "typescript"),
    ("md", "markdown"),
    ("json", "json"),
];

fn file_body(lang: &str, n: usize, rng: &mut Rng) -> String {
    let id = rng.next() % 1000;
    match lang {
        "rust" => format!(
            "//! module {n}\n\n/// Compute thing {id}.\npub fn compute_{n}(x: i64) -> i64 {{\n    let mut acc = 0;\n    for i in 0..x {{ acc += i * {id}; }}\n    acc\n}}\n\npub struct Widget{n} {{ pub id: u64, pub name: String }}\n\nimpl Widget{n} {{\n    pub fn new(id: u64) -> Self {{ Self {{ id, name: format!(\"w{{id}}\") }} }}\n    pub fn rank(&self) -> u64 {{ self.id ^ {id} }}\n}}\n"
        ),
        "python" => format!(
            "\"\"\"module {n}\"\"\"\n\nclass Service{n}:\n    def __init__(self, token: str):\n        self.token = token\n\n    def verify(self) -> bool:\n        return len(self.token) > {id}\n\n\ndef sum_all_{n}(xs):\n    return sum(x * {id} for x in xs)\n"
        ),
        "typescript" => format!(
            "export interface User{n} {{ id: number; name: string }}\n\nexport function authenticate{n}(token: string): boolean {{\n  return token.length > {id};\n}}\n\nexport class Cache{n} {{\n  private items = new Map<string, number>();\n  evict() {{ this.items.clear(); }}\n}}\n"
        ),
        "markdown" => format!(
            "# Component {n}\n\nOverview of component {n}.\n\n## Authentication\n\nToken validation logic, id {id}.\n\n## Usage\n\nCall `compute_{n}` with an integer.\n"
        ),
        _ => format!(
            "{{\n  \"id\": {id},\n  \"module\": {n},\n  \"enabled\": true,\n  \"tags\": [\"a\", \"b\", \"c\"]\n}}\n"
        ),
    }
}

fn gen_corpus(out: &Path, files: usize, seed: u64) -> Result<usize> {
    if out.exists() {
        std::fs::remove_dir_all(out).ok();
    }
    let mut rng = Rng(seed);
    let mut written = 0;
    for i in 0..files {
        let (ext, lang) = LANGS[i % LANGS.len()];
        let depth = (rng.next() % 4) as usize;
        let mut dir = out.to_path_buf();
        for d in 0..depth {
            dir = dir.join(format!("pkg{}", (rng.next() % 6) + d as u64));
        }
        std::fs::create_dir_all(&dir)?;
        let path = dir.join(format!("mod_{i}.{ext}"));
        let reps = 1 + (rng.next() % 8) as usize;
        let mut body = String::new();
        for _ in 0..reps {
            body.push_str(&file_body(lang, i, &mut rng));
        }
        std::fs::write(&path, body)?;
        written += 1;
    }
    Ok(written)
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn peak_rss_mb() -> u64 {
    // SAFETY: getrusage with a zeroed struct is always sound.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let max = ru.ru_maxrss as u64;
    if cfg!(target_os = "macos") {
        max / (1024 * 1024) // bytes -> MB
    } else {
        max / 1024 // KB -> MB
    }
}
#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn peak_rss_mb() -> u64 {
    0
}

fn git_short() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "nogit".into())
}

fn bench(files: usize, seed: u64) -> Result<()> {
    let root = workspace_root();
    let corpus = root.join("benchmarks/corpus");
    let n = gen_corpus(&corpus, files, seed).context("gen corpus")?;

    let work = tempfile::tempdir()?;
    copy_dir(&corpus, work.path())?;

    let cfg = Config::default();
    let engine = SyncEngine::new(work.path().to_path_buf(), cfg).context("engine")?;

    let t0 = Instant::now();
    let stats = engine.sync(true, |_| {}).context("full sync")?;
    let full_ms = t0.elapsed().as_millis() as u64;

    let t1 = Instant::now();
    engine.sync(false, |_| {}).context("incremental sync")?;
    let incr_ms = t1.elapsed().as_millis() as u64;

    let peak = peak_rss_mb();
    let rec = serde_json::json!({
        "commit": git_short(),
        "date": chrono::Utc::now().to_rfc3339(),
        "model": "stub",
        "files": n,
        "chunks": stats.chunks_total,
        "full_sync_ms": full_ms,
        "incr_ms": incr_ms,
        "peak_rss_mb": peak,
        "host": format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH),
        "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
    });

    let results = root.join("benchmarks/results");
    std::fs::create_dir_all(&results)?;
    let hist = results.join("history.jsonl");
    append_jsonl(&hist, &rec)?;

    println!("{}", serde_json::to_string_pretty(&rec)?);
    println!("appended -> {}", hist.display());
    Ok(())
}

/// Append one JSON record as a line to a `.jsonl` history file
/// (created if absent). Shared by `bench` and `eval`.
fn append_jsonl(path: &Path, rec: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let mut line = serde_json::to_string(rec)?;
    line.push('\n');
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?
        .write_all(line.as_bytes())?;
    Ok(())
}

fn workspace_root() -> PathBuf {
    // xtask runs from workspace root via the cargo alias.
    std::env::current_dir().expect("cwd")
}

fn copy_dir(src: &Path, dst: &Path) -> Result<()> {
    for entry in std::fs::read_dir(src)? {
        let e = entry?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            std::fs::create_dir_all(&to)?;
            copy_dir(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), &to)?;
        }
    }
    Ok(())
}

/// Set the release version everywhere it is hand-duplicated, so the
/// plugin's pinned mcp-bin tag never drifts from the crate version (a
/// bare/`@latest` spec is cached forever by mcp-bin → plugin updates
/// would keep the old binary). Single command: `cargo xtask bump X`.
fn bump(new: &str) -> Result<()> {
    // xtask's manifest dir is `<root>/xtask`; its parent is the root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("workspace root")?
        .to_path_buf();
    let cargo = std::fs::read_to_string(root.join("Cargo.toml"))?;
    // Anchor on a leading newline so `rust-version = "` (which
    // contains the substring `version = "`) cannot match.
    let old = cargo
        .split("[workspace.package]")
        .nth(1)
        .and_then(|s| s.split("\nversion = \"").nth(1))
        .and_then(|s| s.split('"').next())
        .context("workspace.package version not found")?
        .to_string();
    if old == new {
        println!("already {new}");
        return Ok(());
    }
    let edit = |rel: &str, from: String, to: String| -> Result<()> {
        let p = root.join(rel);
        let c = std::fs::read_to_string(&p)?;
        if !c.contains(&from) {
            anyhow::bail!("pattern not found in {rel}: {from}");
        }
        std::fs::write(&p, c.replace(&from, &to)).context(rel.to_string())
    };
    // workspace crate version (anchored by the following `license` line
    // so workspace.dependencies versions can't match)
    edit(
        "Cargo.toml",
        format!("version = \"{old}\"\nlicense"),
        format!("version = \"{new}\"\nlicense"),
    )?;
    // MCP server handshake version (attribute arg must be a literal)
    edit(
        "crates/cli/src/mcp.rs",
        format!("version = \"{old}\")]"),
        format!("version = \"{new}\")]"),
    )?;
    // plugin manifest version + its pinned mcp-bin release tag
    edit(
        ".claude-plugin/plugin.json",
        format!("\"version\": \"{old}\""),
        format!("\"version\": \"{new}\""),
    )?;
    edit(
        ".claude-plugin/plugin.json",
        format!("sensiarion/embedding-search@v{old}"),
        format!("sensiarion/embedding-search@v{new}"),
    )?;
    println!(
        "bumped {old} -> {new} (Cargo.toml, mcp.rs, plugin.json version + \
         mcp-bin tag).\nnext: add CHANGELOG.md entry, `cargo build` (lock), \
         commit, `git tag -a v{new}`"
    );
    Ok(())
}

/// A (natural-language query, code) retrieval pair from CodeSearchNet.
#[cfg(not(feature = "bench-stub"))]
struct Pair {
    query: String,
    code: String,
}

/// Local cache of the CodeSearchNet python test pairs (jsonl). Kept
/// out of git (see `.gitignore`); fetched once from the HF
/// datasets-server, then reused so `eval` is offline-repeatable.
#[cfg(not(feature = "bench-stub"))]
fn load_csn(root: &Path, n: usize) -> Result<Vec<Pair>> {
    let dir = root.join("benchmarks/csn");
    std::fs::create_dir_all(&dir)?;
    let cache = dir.join("python_test.jsonl");

    let mut pairs: Vec<Pair> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(&cache) {
        for line in text.lines() {
            let v: serde_json::Value = serde_json::from_str(line)?;
            pairs.push(Pair {
                query: v["q"].as_str().unwrap_or("").to_string(),
                code: v["c"].as_str().unwrap_or("").to_string(),
            });
        }
    }
    if pairs.len() >= n {
        pairs.truncate(n);
        return Ok(pairs);
    }

    // datasets-server caps `length` at 100 — page through until `n`.
    let agent = ureq::Agent::new_with_defaults();
    let mut out = String::new();
    pairs.clear();
    let mut offset = 0usize;
    while pairs.len() < n {
        let url = format!(
            "https://datasets-server.huggingface.co/rows?dataset=code-search-net%2Fcode_search_net\
             &config=python&split=test&offset={offset}&length=100"
        );
        let body = agent
            .get(&url)
            .call()
            .with_context(|| format!("fetch CSN rows @{offset}"))?
            .into_body()
            .read_to_string()?;
        let v: serde_json::Value = serde_json::from_str(&body)?;
        let rows = v["rows"].as_array().context("CSN: no rows")?;
        if rows.is_empty() {
            break;
        }
        for r in rows {
            let row = &r["row"];
            let q = row["func_documentation_string"].as_str().unwrap_or("");
            let c = row["func_code_string"].as_str().unwrap_or("");
            if q.trim().is_empty() || c.trim().is_empty() {
                continue;
            }
            out.push_str(&serde_json::json!({ "q": q, "c": c }).to_string());
            out.push('\n');
            pairs.push(Pair {
                query: q.to_string(),
                code: c.to_string(),
            });
            if pairs.len() >= n {
                break;
            }
        }
        offset += 100;
    }
    std::fs::write(&cache, &out).context("write CSN cache")?;
    Ok(pairs)
}

#[cfg(not(feature = "bench-stub"))]
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let d = (na.sqrt() * nb.sqrt()).max(1e-12);
    dot / d
}

/// Single-relevant retrieval metrics accumulator. The gold doc is the
/// code at the query's own index, so NDCG@10 reduces to `1/log2(rank+1)`
/// and MRR to `1/rank`.
#[cfg(not(feature = "bench-stub"))]
#[derive(Default)]
struct Metrics {
    n: usize,
    mrr: f64,
    r1: usize,
    r5: usize,
    ndcg: f64,
}

#[cfg(not(feature = "bench-stub"))]
impl Metrics {
    fn add(&mut self, rank: usize) {
        self.n += 1;
        if rank == 1 {
            self.r1 += 1;
        }
        if rank <= 5 {
            self.r5 += 1;
        }
        if rank <= 10 {
            self.mrr += 1.0 / rank as f64;
            self.ndcg += 1.0 / ((rank as f64) + 1.0).log2();
        }
    }

    fn record(&self, extra: serde_json::Value) -> serde_json::Value {
        let nf = self.n.max(1) as f64;
        let round4 = |x: f64| (x * 1e4).round() / 1e4;
        let mut v = serde_json::json!({
            "n": self.n,
            "mrr@10": round4(self.mrr / nf),
            "recall@1": round4(self.r1 as f64 / nf),
            "recall@5": round4(self.r5 as f64 / nf),
            "ndcg@10": round4(self.ndcg / nf),
        });
        let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) else {
            return v;
        };
        for (k, val) in e {
            o.insert(k.clone(), val.clone());
        }
        v
    }
}

/// Retrieval metrics for one model over the pairs. Always emits the
/// `base` (pure cosine ranking) record; when `reranker` is `Some`,
/// also emits a `rerank` record — the cross-encoder re-scores the
/// model's top-`top_n` cosine candidates (exactly the optional
/// `[rerank]` search stage), so the benchmark shows what enabling it
/// buys per model. Every evaluated query is reranked (the exact set
/// the `base` cosine record measures), so `base` vs `rerank` is a
/// like-for-like delta — no subsampled pseudo-baseline. Cost is
/// O(queries · top_n · seq²) (independent of corpus size); the fast
/// ModernBERT reranker (candle Metal / int8 CPU) keeps it tractable.
#[cfg(not(feature = "bench-stub"))]
fn eval_model(
    name: &str,
    pairs: &[Pair],
    queries_n: usize,
    reranker: Option<&Reranker>,
) -> Result<Vec<serde_json::Value>> {
    let mut cfg = Config::default();
    cfg.model.default = name.to_string();
    let emb = Embedder::new(&cfg).with_context(|| format!("load {name}"))?;

    // The corpus (distractor pool) is every loaded pair's code; only
    // the first `queries_n` docstrings are used as queries. A large
    // pool with far fewer queries is the realistic, discriminating
    // setup — ranking the gold among thousands, not among the handful
    // of queries.
    let t0 = Instant::now();
    let codes: Vec<&str> = pairs.iter().map(|p| p.code.as_str()).collect();
    let doc_vecs = emb
        .embed_documents(&codes, cfg.embed_batch())
        .context("embed corpus")?;
    let embed_ms = t0.elapsed().as_millis() as u64;

    let mut base = Metrics::default();
    let mut reranked = Metrics::default();
    let mut rerank_ms = 0u64;
    let top_n = reranker.map_or(0, Reranker::top_n);

    for (i, p) in pairs.iter().take(queries_n).enumerate() {
        let q = emb.embed_query(&p.query)?;
        // Full cosine ranking (stable: ties keep corpus order).
        let mut sims: Vec<(usize, f32)> = doc_vecs
            .iter()
            .enumerate()
            .map(|(j, d)| (j, cosine(&q, d)))
            .collect();
        sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let pos = |order: &[(usize, f32)]| {
            order
                .iter()
                .position(|(j, _)| *j == i)
                .map_or(order.len(), |r| r)
                + 1
        };
        base.add(pos(&sims));

        // Rerank every query — `base` and `rerank` then measure the
        // exact same set (honest delta, no subsample illusion).
        let Some(rr) = reranker else {
            continue;
        };
        // Re-score the top-n cosine candidates jointly (query, code),
        // reorder that prefix, keep the cosine tail — mirrors
        // search::rerank_fused.
        let n = top_n.min(sims.len());
        let cand: Vec<usize> = sims[..n].iter().map(|(j, _)| *j).collect();
        let passages: Vec<&str> = cand.iter().map(|&j| pairs[j].code.as_str()).collect();
        let t = Instant::now();
        let scores = rr.score(&p.query, &passages)?;
        rerank_ms += t.elapsed().as_millis() as u64;
        let mut prefix: Vec<usize> = (0..n).collect();
        prefix.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let rank = match prefix.iter().position(|&k| cand[k] == i) {
            Some(r) => r + 1,
            // Gold fell outside the reranked window: keep its cosine
            // position, shifted past the n reranked slots.
            None => {
                n + sims[n..]
                    .iter()
                    .position(|(j, _)| *j == i)
                    .map_or(0, |r| r + 1)
            }
        };
        reranked.add(rank);
    }

    let mut out = vec![base.record(serde_json::json!({
        "model": name,
        "variant": "base",
        "corpus": pairs.len(),
        "queries": queries_n,
        "embed_ms": embed_ms,
        "dims": emb.dimensions,
    }))];
    if reranker.is_some() {
        // Same query set as `base` above (every query reranked) — read
        // the two together for the cross-encoder's like-for-like delta.
        out.push(reranked.record(serde_json::json!({
            "model": name,
            "variant": "rerank",
            "corpus": pairs.len(),
            "queries": queries_n,
            "reranker": cfg.rerank.model,
            "top_n": top_n,
            "rerank_ms": rerank_ms,
            "dims": emb.dimensions,
        })));
    }
    Ok(out)
}

/// bench-stub builds never reach this: `run_eval` re-execs a real
/// build and `process::exit`s first. Present only so `main` compiles
/// under the alias feature.
#[cfg(feature = "bench-stub")]
fn eval(
    _cn: usize,
    _qn: usize,
    _rr: bool,
    _tn: usize,
    _ml: usize,
    _models: &str,
    _out: Option<&Path>,
) -> Result<()> {
    unreachable!("eval re-execs into a non-stub build via run_eval")
}

#[cfg(not(feature = "bench-stub"))]
#[allow(clippy::too_many_arguments)]
fn eval(
    corpus_n: usize,
    queries_n: usize,
    do_rerank: bool,
    rerank_top_n: usize,
    rerank_max_len: usize,
    models_selector: &str,
    out_override: Option<&Path>,
) -> Result<()> {
    let root = workspace_root();
    let pairs = load_csn(&root, corpus_n).context("load CodeSearchNet")?;
    anyhow::ensure!(!pairs.is_empty(), "no CSN pairs loaded");
    let queries_n = queries_n.min(pairs.len());
    let models = pick_models(models_selector);
    anyhow::ensure!(!models.is_empty(), "no models selected");
    println!(
        "CodeSearchNet python/test — {} corpus docs, {} queries, {} models\n",
        pairs.len(),
        queries_n,
        models.len(),
    );

    let commit = git_short();
    let date = chrono::Utc::now().to_rfc3339();
    let host = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);

    // Per-run output dir keeps each invocation's results isolated for
    // before/after comparison (effectiveness.jsonl accumulates across
    // runs and mixes commits). Default: benchmarks/results/<UTC-YYYY-
    // MM-DDTHH-MM-SS>-<commit>/; --output PATH overrides.
    let run_dir = match out_override {
        Some(p) => p.to_path_buf(),
        None => {
            let slug = date.replace(':', "-");
            root.join("benchmarks/results")
                .join(format!("{slug}-{commit}"))
        }
    };
    std::fs::create_dir_all(&run_dir).with_context(|| format!("mkdir {}", run_dir.display()))?;
    let results_jsonl = run_dir.join("results.jsonl");
    // Legacy single-file history kept for any tooling that reads it.
    let history_jsonl = root.join("benchmarks/results/effectiveness.jsonl");

    // Rerank is OFF unless `--rerank`. The cross-encoder is
    // model-independent (scores raw query↔code), so load once. Every
    // evaluated query is reranked → cost O(queries · top_n · seq²)
    // (independent of corpus size); the default ModernBERT reranker
    // (candle Metal / int8 CPU) keeps it tractable.
    // `Reranker`'s ORT batch is `cfg.embed_batch()`;
    // the default embed model (CodeRankEmbed) has rec_batch 4 → tiny
    // batches, and the reranker never touches the embedder, so point
    // the cfg at a static model (rec_batch 64) purely for a sane batch
    // and cap its seq.
    let reranker = if !do_rerank {
        None
    } else {
        eprintln!(
            "rerank ON: ~{} cross-encodes ({} models × {} queries × \
             top_n {}), seq≤{}.",
            models.len() * queries_n * rerank_top_n,
            models.len(),
            queries_n,
            rerank_top_n,
            rerank_max_len,
        );
        let mut rcfg = Config::default();
        rcfg.model.default = "minishlab/potion-base-32M".to_string();
        rcfg.model.max_length = rerank_max_len;
        rcfg.rerank.top_n = rerank_top_n;
        match Reranker::load(&rcfg) {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("reranker unavailable ({e}); base only");
                None
            }
        }
    };

    let mut all_records: Vec<serde_json::Value> = Vec::new();
    for name in &models {
        match eval_model(name, &pairs, queries_n, reranker.as_ref()) {
            Ok(recs) => {
                for mut rec in recs {
                    let obj = rec.as_object_mut().unwrap();
                    obj.insert("commit".into(), serde_json::json!(commit));
                    obj.insert("date".into(), serde_json::json!(date));
                    obj.insert("host".into(), serde_json::json!(host));
                    println!("{}", serde_json::to_string_pretty(&rec)?);
                    append_jsonl(&results_jsonl, &rec)?;
                    append_jsonl(&history_jsonl, &rec)?;
                    all_records.push(rec);
                }
            }
            Err(e) => {
                // One bad model shouldn't abort a 10-model sweep — record
                // the failure as a row so the summary makes the gap
                // visible, then keep going.
                eprintln!("model '{name}' failed: {e:#}");
                let rec = serde_json::json!({
                    "model": name,
                    "variant": "base",
                    "error": format!("{e:#}"),
                    "commit": commit,
                    "date": date,
                    "host": host,
                });
                append_jsonl(&results_jsonl, &rec)?;
                all_records.push(rec);
            }
        }
    }

    let report = render_report(&all_records, &date, &commit, &host, pairs.len(), queries_n);
    let report_path = run_dir.join("REPORT.md");
    std::fs::write(&report_path, &report).context("write REPORT.md")?;
    println!("\nresults  -> {}", results_jsonl.display());
    println!("report   -> {}", report_path.display());
    Ok(())
}

/// Build a ranked Markdown summary of one eval run — one row per
/// (model, variant), sorted by MRR@10 descending. Comparable across
/// runs because every column is derived from the same `results.jsonl`.
/// One row of the rendered eval report (sortable by `mrr`).
#[cfg(not(feature = "bench-stub"))]
struct ReportRow {
    model: String,
    variant: String,
    mrr: f64,
    r1: f64,
    r5: f64,
    ndcg: f64,
    embed_ms: u64,
    rerank_ms: Option<u64>,
    error: Option<String>,
}

#[cfg(not(feature = "bench-stub"))]
fn render_report(
    records: &[serde_json::Value],
    date: &str,
    commit: &str,
    host: &str,
    corpus: usize,
    queries: usize,
) -> String {
    use std::fmt::Write;
    let mut rows: Vec<ReportRow> = records
        .iter()
        .map(|r| ReportRow {
            model: r["model"].as_str().unwrap_or("?").to_string(),
            variant: r["variant"].as_str().unwrap_or("base").to_string(),
            mrr: r["mrr@10"].as_f64().unwrap_or(0.0),
            r1: r["recall@1"].as_f64().unwrap_or(0.0),
            r5: r["recall@5"].as_f64().unwrap_or(0.0),
            ndcg: r["ndcg@10"].as_f64().unwrap_or(0.0),
            embed_ms: r["embed_ms"].as_u64().unwrap_or(0),
            rerank_ms: r["rerank_ms"].as_u64(),
            error: r["error"].as_str().map(str::to_string),
        })
        .collect();
    rows.sort_by(|a, b| {
        b.mrr
            .partial_cmp(&a.mrr)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Sanitize a string for a single markdown table cell — strip
    // newlines (would break the row) and escape pipes (would split
    // into extra cells). Used for error messages, which can contain
    // both when anyhow stringifies a multi-line cause chain.
    fn cell(s: &str) -> String {
        s.replace('\n', " ").replace('|', "\\|")
    }

    let mut out = String::new();
    let _ = writeln!(out, "# eval REPORT");
    let _ = writeln!(out, "- date: `{date}`");
    let _ = writeln!(out, "- commit: `{commit}`");
    let _ = writeln!(out, "- host: `{host}`");
    let _ = writeln!(out, "- corpus: CodeSearchNet python/test, {corpus} docs");
    let _ = writeln!(out, "- queries: {queries}");
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| model | variant | MRR@10 | R@1 | R@5 | NDCG@10 | embed (ms) | rerank (ms) |"
    );
    let _ = writeln!(out, "|---|---|---:|---:|---:|---:|---:|---:|");
    // Collect any errored rows so they get their own section below the
    // metrics table — keeps the table well-formed (fixed 8 columns)
    // and gives the error message room to be readable.
    let mut errors: Vec<(&str, &str, &str)> = Vec::new();
    for row in &rows {
        let rr_cell = row
            .rerank_ms
            .map(|x| x.to_string())
            .unwrap_or_else(|| "—".into());
        if let Some(e) = &row.error {
            errors.push((&row.model, &row.variant, e));
            let _ = writeln!(
                out,
                "| `{}` | `{}` | ERR | — | — | — | — | — |",
                row.model, row.variant
            );
        } else {
            let _ = writeln!(
                out,
                "| `{}` | `{}` | {:.4} | {:.4} | {:.4} | {:.4} | {} | {rr_cell} |",
                row.model, row.variant, row.mrr, row.r1, row.r5, row.ndcg, row.embed_ms
            );
        }
    }
    if !errors.is_empty() {
        let _ = writeln!(out, "\n## errors\n");
        for (m, v, e) in errors {
            let _ = writeln!(out, "- `{}` ({}): {}", m, v, cell(e));
        }
    }
    out
}

/// Under the `cargo xtask` alias the build carries `bench-stub` (the
/// deterministic hash embedder) — meaningless for a retrieval metric.
/// Re-exec the eval through a real (non-stub) build once.
#[cfg(feature = "bench-stub")]
fn run_eval(args: &[String]) -> Result<()> {
    if std::env::var_os("ES_EVAL_REAL").is_some() {
        // Shouldn't happen (a real build has no bench-stub), but guard
        // against an infinite re-exec loop regardless.
        anyhow::bail!("eval re-exec still has bench-stub — check the cargo alias");
    }
    eprintln!("eval needs a real model; rebuilding xtask without bench-stub…");
    let status = std::process::Command::new(env!("CARGO"))
        .args(["run", "-p", "xtask", "--release", "--"])
        .args(args)
        .env("ES_EVAL_REAL", "1")
        .status()
        .context("re-exec cargo run for eval")?;
    std::process::exit(status.code().unwrap_or(1));
}
#[cfg(not(feature = "bench-stub"))]
fn run_eval(_args: &[String]) -> Result<()> {
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("");
    let files: usize = arg(&args, "--files", "500").parse().unwrap_or(500);
    let seed: u64 = arg(&args, "--seed", "42").parse().unwrap_or(42);
    match cmd {
        "gen-corpus" => {
            let out = arg(&args, "--out", "benchmarks/corpus");
            let n = gen_corpus(Path::new(&out), files, seed)?;
            println!("generated {n} files in {out}");
        }
        "bench" => bench(files, seed)?,
        "eval" => {
            // No-op on a real build; on the bench-stub alias build it
            // re-execs a real build and exits, so the line below only
            // runs with a genuine model.
            run_eval(&args)?;
            // Large distractor pool, far fewer queries — the
            // discriminating setup (rank the gold among thousands).
            let corpus = arg(&args, "--corpus", "5000").parse().unwrap_or(5000);
            let queries = arg(&args, "--queries", "200").parse().unwrap_or(200);
            // Rerank is opt-in and CPU-heavy: off unless `--rerank`,
            // then bounded by these (intentionally small) defaults.
            let do_rerank = args.iter().any(|a| a == "--rerank");
            let rr_top_n = arg(&args, "--rerank-top-n", "20").parse().unwrap_or(20);
            let rr_max_len = arg(&args, "--rerank-max-len", "256").parse().unwrap_or(256);
            // `--models a,b,c` (or `--models all`) overrides the default
            // 3-model set. `--output PATH` overrides the auto-generated
            // per-run dir under benchmarks/results/.
            let models = arg(&args, "--models", "");
            let out_raw = arg(&args, "--output", "");
            let out_path = (!out_raw.is_empty()).then(|| PathBuf::from(out_raw));
            eval(
                corpus,
                queries,
                do_rerank,
                rr_top_n,
                rr_max_len,
                &models,
                out_path.as_deref(),
            )?;
        }
        "bump" => {
            let v = args.get(1).context("usage: cargo xtask bump <version>")?;
            bump(v)?;
        }
        _ => {
            eprintln!(
                "usage: cargo xtask <gen-corpus|bench|eval|bump> \
                 [--files N] [--seed S] [--corpus N] [--queries N] \
                 [--models a,b,c|all] [--output DIR] \
                 [--rerank [--rerank-top-n N] [--rerank-max-len N]] \
                 [--out DIR]\n\n\
                 eval:\n  \
                 --models  comma-separated names, or `all` for every \
                 entry in SUPPORTED_MODELS (default: 3-model baseline).\n  \
                 --output  per-run results dir (default: \
                 benchmarks/results/<date>-<commit>/). Writes \
                 results.jsonl + REPORT.md."
            );
            std::process::exit(2);
        }
    }
    Ok(())
}
