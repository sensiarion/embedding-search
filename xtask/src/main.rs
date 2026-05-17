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
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    let mut line = serde_json::to_string(&rec)?;
    line.push('\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&hist)?
        .write_all(line.as_bytes())?;

    println!("{}", serde_json::to_string_pretty(&rec)?);
    println!("appended -> {}", hist.display());
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
        _ => {
            eprintln!("usage: cargo xtask <gen-corpus|bench> [--files N] [--seed S] [--out DIR]");
            std::process::exit(2);
        }
    }
    Ok(())
}
