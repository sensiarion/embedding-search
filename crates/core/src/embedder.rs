use crate::config::Config;
use crate::error::{Error, Result};
use fastembed::TextEmbedding;
use std::sync::Mutex;
#[cfg(not(feature = "bench-stub"))]
use {
    crate::config::{BackendConfig, EmbeddingProvider, ExecutionProvider, Precision, RemoteConfig},
    fastembed::{
        InitOptions, InitOptionsUserDefined, Pooling, TokenizerFiles, UserDefinedEmbeddingModel,
    },
    rayon::prelude::*,
    safetensors::SafeTensors,
    serde::Deserialize,
    std::path::Path,
    std::time::Duration,
    tokenizers::Tokenizer,
};

enum Backend {
    /// `Some` = loaded ONNX model. `None` only under `bench-stub`
    /// (deterministic hash embedder, no model load).
    Local(Option<Mutex<TextEmbedding>>),
    #[cfg(not(feature = "bench-stub"))]
    Remote(RemoteEmbedder),
    /// Model2Vec / `StaticModel`: a static token-embedding matrix +
    /// tokenizer, mean-pooled (+ optional L2). No transformer/ONNX —
    /// its own tiny inference path.
    #[cfg(not(feature = "bench-stub"))]
    Static(StaticModel2Vec),
}

pub struct Embedder {
    backend: Backend,
    pub dimensions: usize,
    pub model_name: String,
    /// e5 models require "query:" / "passage:" input prefixes.
    is_e5: bool,
}

#[cfg(feature = "bench-stub")]
fn stub_vector(text: &str, dims: usize) -> Vec<f32> {
    // deterministic pseudo-embedding from the text hash
    let h = blake3::hash(text.as_bytes());
    let seed = h.as_bytes();
    (0..dims)
        .map(|i| {
            let b = seed[i % 32] ^ (i as u8).wrapping_mul(31);
            (b as f32 / 127.5) - 1.0
        })
        .collect()
}

#[cfg(not(feature = "bench-stub"))]
fn providers(b: &BackendConfig) -> Vec<fastembed::ExecutionProviderDispatch> {
    use ort::ep::CPU;
    let mac_arm = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let mut v: Vec<fastembed::ExecutionProviderDispatch> = Vec::new();

    match b.execution_provider {
        ExecutionProvider::Coreml => {
            if mac_arm {
                v.push(ort::ep::CoreML::default().build());
            } else {
                tracing::warn!("coreml requested but not on Apple Silicon — CPU");
            }
        }
        ExecutionProvider::Cuda => {
            #[cfg(feature = "cuda")]
            v.push(ort::ep::CUDA::default().build());
        }
        ExecutionProvider::Auto => {
            if mac_arm {
                v.push(ort::ep::CoreML::default().build());
            }
            #[cfg(feature = "cuda")]
            v.push(ort::ep::CUDA::default().build());
        }
        ExecutionProvider::Cpu => {}
    }

    // CPU last as fallback; arena off by default (it extends by powers of
    // two and never frees — the multi-GB blowup with dynamic shapes).
    v.push(
        CPU::default()
            .with_arena_allocator(!b.disable_mem_arena)
            .build(),
    );
    v
}

#[cfg(not(feature = "bench-stub"))]
fn read(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(Error::Io)
}

/// Normalized repo id + a cached hf-hub handle. Single source of the
/// `ApiBuilder` setup shared by the ONNX (`load_user_defined`) and
/// Model2Vec routes. Callers build the trivial `fetch` closure from
/// `r` (it must borrow `r`, so it can't be returned here).
#[cfg(not(feature = "bench-stub"))]
fn hf_repo(repo: &str, cache_dir: &Path) -> Result<(String, hf_hub::api::sync::ApiRepo)> {
    // Defensive: a pre-existing config / copy-pasted browser URL may
    // hold a full HF URL where an id is expected.
    let repo = crate::config::normalize_hf_repo(repo);
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(true)
        .build()
        .map_err(|e| Error::Embed(format!("hf-hub: {e}")))?;
    let r = api.model(repo.clone());
    Ok((repo, r))
}

/// External-data sidecar candidates for an ONNX file `mf` (repo- or
/// dir-relative). Pairs of `(relative path to fetch, basename the
/// graph references)`. ONNX external data is keyed by basename, so a
/// `.onnx` in an `onnx/` subdir still points at `model.onnx_data`.
#[cfg(not(feature = "bench-stub"))]
fn onnx_data_sidecars(mf: &str) -> [(String, String); 2] {
    [format!("{mf}_data"), format!("{mf}.data")].map(|rel| {
        let name = rel.rsplit('/').next().unwrap_or(&rel).to_string();
        (rel, name)
    })
}

/// Assemble a user-defined model: `onnx` is the model bytes,
/// `fetch_tok` supplies the four tokenizer files by name (hf-hub cache
/// or a local dir). Single source of the tokenizer file set + pooling.
#[cfg(not(feature = "bench-stub"))]
fn user_defined_from(
    onnx: Vec<u8>,
    fetch_tok: impl Fn(&str) -> Result<Vec<u8>>,
    externals: Vec<(String, Vec<u8>)>,
) -> Result<UserDefinedEmbeddingModel> {
    let tok = TokenizerFiles {
        tokenizer_file: fetch_tok("tokenizer.json")?,
        config_file: fetch_tok("config.json")?,
        special_tokens_map_file: fetch_tok("special_tokens_map.json")?,
        tokenizer_config_file: fetch_tok("tokenizer_config.json")?,
    };
    let mut m = UserDefinedEmbeddingModel::new(onnx, tok).with_pooling(Pooling::Mean);
    // ONNX external-data: the `.onnx` is just the graph; weights live in
    // a sibling file the model references by name (onnx-community /
    // models >2 GB). fastembed loads it in-memory by matching this name.
    for (name, buf) in externals {
        m = m.with_external_initializer(name, buf);
    }
    Ok(m)
}

/// Repo-relative ONNX candidates, first match wins. `explicit` (an
/// exact file like `model_q4f16.onnx`) overrides the precision→file
/// mapping; a bare name is also tried under `onnx/`.
#[cfg(not(feature = "bench-stub"))]
fn onnx_candidates(precision: Precision, explicit: Option<&str>) -> Vec<String> {
    match explicit {
        Some(f) if f.contains('/') => vec![f.to_string()],
        Some(f) => vec![f.to_string(), format!("onnx/{f}")],
        None => vec![
            precision.onnx_file().to_string(),
            "onnx/model.onnx".to_string(),
            "model.onnx".to_string(),
            "onnx/model_quantized.onnx".to_string(),
        ],
    }
}

/// `Some(normalize)` if `config.json` is a Model2Vec / `StaticModel`
/// (routed to the static backend, not fastembed) — carrying its
/// `normalize` flag so the route parses config.json exactly once.
/// `None` ⇒ not Model2Vec. Defaults `normalize` to true (the
/// `potion-*` / `StaticEmbedding`+`Normalize` common case).
#[cfg(not(feature = "bench-stub"))]
fn model2vec_normalize(config_json: &[u8]) -> Option<bool> {
    let v = serde_json::from_slice::<serde_json::Value>(config_json).ok()?;
    let is_m2v = v.get("model_type").and_then(|m| m.as_str()) == Some("model2vec")
        || v.get("architectures")
            .and_then(|a| a.as_array())
            .is_some_and(|a| a.iter().any(|x| x.as_str() == Some("StaticModel")));
    is_m2v.then(|| v.get("normalize").and_then(|n| n.as_bool()).unwrap_or(true))
}

/// Reject LM-head ONNX exports (transformer path only), from
/// `config.json`, *before* downloading weights: `*ForMaskedLM/
/// CausalLM/PreTraining` / `*LMHead*` output vocab logits, not
/// embeddings, and ALiBi long-context ones OOM at tens of GB from a
/// baked `[heads, ctx, ctx]` bias. Unparseable/odd config → allow
/// (don't block on a heuristic). Model2Vec is handled separately
/// (`model2vec_normalize`), not rejected.
#[cfg(not(feature = "bench-stub"))]
fn assert_loadable_embedding(repo: &str, config_json: &[u8]) -> Result<()> {
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(config_json) else {
        return Ok(());
    };
    let names: Vec<&str> = v
        .get("architectures")
        .and_then(|a| a.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str()).collect())
        .unwrap_or_default();

    let bad = |n: &str| {
        n.ends_with("ForMaskedLM")
            || n.ends_with("ForCausalLM")
            || n.ends_with("ForPreTraining")
            || n.contains("LMHead")
    };
    if !names.is_empty() && names.iter().all(|n| bad(n)) {
        return Err(Error::Embed(format!(
            "HF repo '{repo}': its ONNX is a language-model head ({}), not a \
             sentence-embedding model — pooling its vocab logits is not an \
             embedding and these exports (ALiBi/long-context) consume tens \
             of GB of RAM. Use an embedding export, or the built-in \
             `jinaai/jina-embeddings-v2-base-code` (models set-default).",
            names.join(", ")
        )));
    }
    Ok(())
}

/// Model2Vec / `StaticModel`: a `[vocab, dim]` static token-embedding
/// matrix + tokenizer. `embed` = tokenize → mean of the token rows →
/// optional L2 (per the model's `Normalize` module / `config.json`).
/// No transformer, no ONNX — that is the whole model.
#[cfg(not(feature = "bench-stub"))]
struct StaticModel2Vec {
    emb: Vec<f32>,
    vocab: usize,
    dim: usize,
    tok: Tokenizer,
    normalize: bool,
}

#[cfg(not(feature = "bench-stub"))]
impl StaticModel2Vec {
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| Error::Embed(format!("model2vec tokenize: {e}")))?;
        let mut acc = vec![0.0f32; self.dim];
        let mut n = 0usize;
        for &id in enc.get_ids() {
            let i = id as usize;
            if i >= self.vocab {
                continue;
            }
            let row = &self.emb[i * self.dim..(i + 1) * self.dim];
            for (a, r) in acc.iter_mut().zip(row) {
                *a += r;
            }
            n += 1;
        }
        // No in-vocab tokens (empty/all-OOB chunk) → a zero vector: a
        // deliberate degenerate fallback (cosine-neutral) rather than
        // an error, so one odd chunk never fails a whole sync.
        if n > 0 {
            let inv = 1.0 / n as f32;
            for a in &mut acc {
                *a *= inv;
            }
        }
        if self.normalize {
            let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for a in &mut acc {
                    *a /= norm;
                }
            }
        }
        Ok(acc)
    }
}

/// Download (cached) the Model2Vec matrix + tokenizer from the HF repo
/// and build the static embedder. `fetch` is the shared hf-hub getter.
#[cfg(not(feature = "bench-stub"))]
fn load_model2vec(
    repo: &str,
    normalize: bool,
    fetch: impl Fn(&str) -> Result<Vec<u8>>,
) -> Result<StaticModel2Vec> {
    let st_bytes = fetch("model.safetensors")?;
    let st = SafeTensors::deserialize(&st_bytes)
        .map_err(|e| Error::Embed(format!("model2vec '{repo}': bad safetensors: {e}")))?;
    let t = st
        .tensor("embeddings")
        .map_err(|e| Error::Embed(format!("model2vec '{repo}': no `embeddings` tensor: {e}")))?;
    let [vocab, dim] = *t.shape() else {
        return Err(Error::Embed(format!(
            "model2vec '{repo}': embeddings must be 2-D [vocab, dim], got {:?}",
            t.shape()
        )));
    };
    if t.dtype() != safetensors::Dtype::F32 {
        return Err(Error::Embed(format!(
            "model2vec '{repo}': embeddings dtype {:?} unsupported (need F32)",
            t.dtype()
        )));
    }
    // safetensors slice is little-endian f32, contiguous [vocab, dim].
    let raw = t.data();
    let emb: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let tok = Tokenizer::from_bytes(fetch("tokenizer.json")?)
        .map_err(|e| Error::Embed(format!("model2vec '{repo}': tokenizer: {e}")))?;
    Ok(StaticModel2Vec {
        emb,
        vocab,
        dim,
        tok,
        normalize,
    })
}

/// Route an HF repo: `Some(static model)` if its `config.json` is
/// Model2Vec, else `None` (caller falls back to the ONNX path).
/// config.json is parsed exactly once here. Shared by the custom and
/// built-in repo paths so a Model2Vec built-in default works too.
#[cfg(not(feature = "bench-stub"))]
fn try_model2vec(repo: &str, cache_dir: &Path) -> Result<Option<StaticModel2Vec>> {
    let (repo_n, r) = hf_repo(repo, cache_dir)?;
    let fetch = |f: &str| -> Result<Vec<u8>> {
        let p = r
            .get(f)
            .map_err(|e| Error::Embed(format!("hf-hub get {f}: {e}")))?;
        read(&p)
    };
    match model2vec_normalize(&fetch("config.json")?) {
        Some(normalize) => Ok(Some(load_model2vec(&repo_n, normalize, fetch)?)),
        None => Ok(None),
    }
}

/// Download (cached) the chosen ONNX + tokenizer files from the HF
/// repo. `explicit` pins an exact file; else the precision mapping.
#[cfg(not(feature = "bench-stub"))]
fn load_user_defined(
    repo: &str,
    precision: Precision,
    cache_dir: &Path,
    explicit: Option<&str>,
) -> Result<UserDefinedEmbeddingModel> {
    let (repo, r) = hf_repo(repo, cache_dir)?;
    let fetch = |f: &str| -> Result<Vec<u8>> {
        let p = r
            .get(f)
            .map_err(|e| Error::Embed(format!("hf-hub get {f}: {e}")))?;
        read(&p)
    };
    // One repo file listing (single request) so we fetch only files
    // that exist — no speculative 404 round-trips for the common
    // single-file model on every process start. If the API is
    // unreachable (offline) `present` is permissive so a fully-cached
    // model still loads via the prior speculative behavior.
    let listing: Option<Vec<String>> = r
        .info()
        .ok()
        .map(|i| i.siblings.into_iter().map(|s| s.rfilename).collect());
    let present = |f: &str| listing.as_ref().is_none_or(|l| l.iter().any(|x| x == f));

    // Reject wrong-head exports up front — before downloading the
    // (often >600 MB) weights — since an MLM/LM-head repo is unusable
    // as an embedder and ALiBi-MLM ones OOM at ~15 GB. config.json is
    // tiny and a required tokenizer file anyway.
    if present("config.json") {
        assert_loadable_embedding(&repo, &fetch("config.json")?)?;
    }

    // Explicit file (if any) overrides the precision mapping; first
    // present candidate that fetches wins; all-miss → one actionable
    // error. Many repos ship only `onnx/model.onnx` or flat
    // `model.onnx`, so the precision path also falls back.
    let candidates = onnx_candidates(precision, explicit);
    let (model_file, onnx) = candidates
        .iter()
        .filter(|f| present(f))
        .find_map(|f| fetch(f).ok().map(|b| (f.clone(), b)))
        .ok_or_else(|| {
            // Surface the repo's actual `.onnx` files so a wrong
            // --onnx-file / unmapped precision is a pick-from-this list,
            // not a dead end. (Listing absent only when offline.)
            let avail = listing.as_ref().map(|l| {
                let mut v: Vec<&str> = l
                    .iter()
                    .filter(|f| f.ends_with(".onnx"))
                    .map(String::as_str)
                    .collect();
                v.sort_unstable();
                v.join(", ")
            });
            let hint = match avail.as_deref() {
                Some(a) if !a.is_empty() => format!(" Available in repo: {a}."),
                _ => String::new(),
            };
            Error::Embed(format!(
                "HF repo '{repo}': no ONNX weights found (looked for {}).{hint} \
                 The repo must also contain tokenizer.json, config.json, \
                 special_tokens_map.json, tokenizer_config.json.",
                candidates.join(", ")
            ))
        })?;
    // External-data sidecar (onnx-community / >2 GB models): the graph
    // references its weights by basename (`model_fp16.onnx_data`). Fetch
    // it next to the `.onnx`; absent ⇒ a self-contained model, fine.
    let externals = onnx_data_sidecars(&model_file)
        .into_iter()
        .filter(|(rel, _)| present(rel))
        .find_map(|(rel, name)| Some(vec![(name, fetch(&rel).ok()?)]))
        .unwrap_or_default();
    user_defined_from(onnx, fetch, externals)
}

/// Resolve a user `onnx_path` (file or directory) to the actual model
/// file + the directory holding its tokenizer files.
#[cfg(not(feature = "bench-stub"))]
fn resolve_onnx(
    path: &Path,
    precision: Precision,
) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    if path.is_file() {
        let dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::path::PathBuf::from("."));
        return Ok((path.to_path_buf(), dir));
    }
    if !path.is_dir() {
        return Err(Error::Embed(format!(
            "onnx_path does not exist: {}",
            path.display()
        )));
    }
    let pf = precision.onnx_file(); // e.g. "onnx/model_fp16.onnx"
    let base = Path::new(pf)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.onnx");
    let candidates = [
        path.join(pf),
        path.join(base),
        path.join("onnx/model.onnx"),
        path.join("model.onnx"),
    ];
    let onnx = candidates
        .iter()
        .find(|p| p.is_file())
        .cloned()
        .ok_or_else(|| {
            Error::Embed(format!(
                "no ONNX file under {} (looked for {pf} or model.onnx)",
                path.display()
            ))
        })?;
    Ok((onnx, path.to_path_buf()))
}

/// Filesystem-safe slug for a custom-model cache subdir.
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

/// On-disk cache directory holding a registered custom model's weights
/// (`models remove` deletes it). Same naming as where they are written:
/// hf-hub's `models--{repo}` for `--repo`, our `url/<slug>` for `--url`.
/// `None` for a repo-less/url-less (malformed) entry. The repo is
/// normalized so it matches what the loader actually fetched. Not
/// gated on `bench-stub`: the CLI `models remove` needs it regardless.
pub fn custom_model_cache_dir(
    cache_dir: &std::path::Path,
    cm: &crate::config::CustomModel,
) -> Option<std::path::PathBuf> {
    if let Some(repo) = &cm.repo {
        let repo = crate::config::normalize_hf_repo(repo);
        Some(cache_dir.join(format!("models--{repo}").replace('/', "--")))
    } else if cm.url.is_some() {
        Some(cache_dir.join("url").join(slug(&cm.name)))
    } else {
        None
    }
}

/// A ureq agent with a global timeout — the single place HTTP client
/// settings (timeout/proxy/retry) are configured.
#[cfg(not(feature = "bench-stub"))]
fn http_agent(timeout: Duration) -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .build(),
    )
}

#[cfg(not(feature = "bench-stub"))]
fn http_get_bytes(agent: &ureq::Agent, url: &str) -> Result<Vec<u8>> {
    agent
        .get(url)
        .call()
        .map_err(|e| Error::Embed(format!("GET {url}: {e}")))?
        .body_mut()
        .read_to_vec()
        .map_err(|e| Error::Embed(format!("read {url}: {e}")))
}

/// Download (and cache) a custom model from a direct `.onnx` URL: the
/// four tokenizer files are fetched from the same directory (same URL
/// with the filename swapped). A missing file yields an actionable
/// error naming the exact URL tried and the full required file set.
#[cfg(not(feature = "bench-stub"))]
fn load_url_user_defined(
    name: &str,
    url: &str,
    cache_dir: &Path,
) -> Result<(UserDefinedEmbeddingModel, String)> {
    let (base, file) = url
        .rsplit_once('/')
        .ok_or_else(|| Error::Embed(format!("custom model '{name}': invalid url {url}")))?;
    if !file.ends_with(".onnx") {
        return Err(Error::Embed(format!(
            "custom model '{name}': url must point to a .onnx file, got {url}"
        )));
    }
    let dir = cache_dir.join("url").join(slug(name));
    std::fs::create_dir_all(&dir)?;
    let agent = http_agent(Duration::from_secs(120));
    // Cache by local filename; fetch `{base}/{remote}` on a miss.
    let fetch = |remote: &str, local: &str| -> Result<Vec<u8>> {
        let cached = dir.join(local);
        if let Ok(b) = std::fs::read(&cached) {
            if !b.is_empty() {
                return Ok(b);
            }
        }
        let full = format!("{base}/{remote}");
        let bytes = http_get_bytes(&agent, &full).map_err(|e| {
            Error::Embed(format!(
                "custom model '{name}': could not download {local} from {full} ({e}). \
                 The .onnx URL's directory must also contain tokenizer.json, \
                 config.json, special_tokens_map.json, tokenizer_config.json."
            ))
        })?;
        std::fs::write(&cached, &bytes)?;
        Ok(bytes)
    };
    let onnx = fetch(file, "model.onnx")?;
    let digest = blake3::hash(&onnx).to_hex();
    let model_name = format!("custom:{name}#{}", &digest[..16]);
    let udm = user_defined_from(onnx, |f| fetch(f, f), Vec::new())?;
    Ok((udm, model_name))
}

/// Build a user-defined model from local files plus a content-hash
/// identity: swapping the model bytes changes it (so the index
/// fingerprint busts and re-embeds), while a touch / checkout with
/// identical bytes does not — unlike an mtime/size signature.
#[cfg(not(feature = "bench-stub"))]
fn load_local_user_defined(
    path: &Path,
    precision: Precision,
) -> Result<(UserDefinedEmbeddingModel, String)> {
    let (onnx_path, dir) = resolve_onnx(path, precision)?;
    let onnx = read(&onnx_path)?;
    let digest = blake3::hash(&onnx).to_hex();
    let name = format!("custom:{}#{}", onnx_path.display(), &digest[..16]);
    // External-data sidecar next to the `.onnx` (same split as the HF
    // path): a self-contained model simply has none.
    let parent = onnx_path.parent().unwrap_or(Path::new("."));
    let mf = onnx_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.onnx");
    let externals = onnx_data_sidecars(mf)
        .into_iter()
        .find_map(|(rel, name)| Some(vec![(name, read(&parent.join(&rel)).ok()?)]))
        .unwrap_or_default();
    let udm = user_defined_from(onnx, |f| read(&dir.join(f)), externals)?;
    Ok((udm, name))
}

/// Wrap a `UserDefinedEmbeddingModel` into a `TextEmbedding` with the
/// shared init options (max length + execution providers).
#[cfg(not(feature = "bench-stub"))]
fn build_user_defined(udm: UserDefinedEmbeddingModel, cfg: &Config) -> Result<TextEmbedding> {
    let opts = InitOptionsUserDefined::new()
        .with_max_length(cfg.model.max_length)
        .with_execution_providers(providers(&cfg.backend));
    TextEmbedding::try_new_from_user_defined(udm, opts).map_err(|e| Error::Embed(e.to_string()))
}

/// External OpenAI-compatible `/embeddings` client with batch + bounded
/// parallel requests.
#[cfg(not(feature = "bench-stub"))]
struct RemoteEmbedder {
    agent: ureq::Agent,
    endpoint: String,
    model: String,
    auth: Option<String>,
    batch_size: usize,
    pool: rayon::ThreadPool,
}

#[cfg(not(feature = "bench-stub"))]
#[derive(Deserialize)]
struct EmbItem {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

#[cfg(not(feature = "bench-stub"))]
#[derive(Deserialize)]
struct EmbResp {
    data: Vec<EmbItem>,
}

/// Order embeddings by the response `index` (OpenAI returns one item per
/// input but does not guarantee response order) and validate the count.
#[cfg(not(feature = "bench-stub"))]
fn parse_embeddings(mut resp: EmbResp, expected: usize) -> Result<Vec<Vec<f32>>> {
    if resp.data.len() != expected {
        return Err(Error::Embed(format!(
            "remote embeddings: expected {expected} vectors, got {}",
            resp.data.len()
        )));
    }
    resp.data.sort_by_key(|d| d.index);
    Ok(resp.data.into_iter().map(|d| d.embedding).collect())
}

#[cfg(not(feature = "bench-stub"))]
impl RemoteEmbedder {
    fn connect(cfg: &RemoteConfig) -> Result<Self> {
        let agent = http_agent(Duration::from_secs(cfg.timeout_seconds));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.concurrency.max(1))
            .build()
            .map_err(|e| Error::Embed(format!("remote pool: {e}")))?;
        let key = cfg.resolved_api_key();
        Ok(Self {
            agent,
            endpoint: cfg.endpoint(),
            model: cfg.model.clone(),
            auth: (!key.is_empty()).then(|| format!("Bearer {key}")),
            batch_size: cfg.batch_size.max(1),
            pool,
        })
    }

    fn embed_batch(&self, batch: &[&str]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": batch,
            "encoding_format": "float",
        });
        let mut req = self.agent.post(&self.endpoint);
        if let Some(a) = &self.auth {
            req = req.header("Authorization", a);
        }
        let mut resp = req
            .send_json(&body)
            .map_err(|e| Error::Embed(format!("remote embeddings POST {}: {e}", self.endpoint)))?;
        let parsed: EmbResp = resp.body_mut().read_json().map_err(|e| {
            Error::Embed(format!(
                "remote embeddings: bad response from {}: {e}",
                self.endpoint
            ))
        })?;
        parse_embeddings(parsed, batch.len())
    }

    /// Split into `batch_size` requests dispatched on the bounded pool;
    /// `par_chunks` is indexed so collected order matches input order.
    fn embed_documents(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let grouped: Vec<Vec<Vec<f32>>> = self.pool.install(|| {
            texts
                .par_chunks(self.batch_size)
                .map(|b| self.embed_batch(b))
                .collect::<Result<Vec<_>>>()
        })?;
        Ok(grouped.into_iter().flatten().collect())
    }
}

impl Embedder {
    /// Benchmark build: deterministic hash embedder, no model load.
    #[cfg(feature = "bench-stub")]
    pub fn new(cfg: &Config) -> Result<Self> {
        let spec = cfg.model_spec()?;
        Ok(Self {
            backend: Backend::Local(None),
            dimensions: spec.dimensions,
            model_name: format!("{}#stub", spec.name),
            is_e5: spec.needs_e5_prefix,
        })
    }

    #[cfg(not(feature = "bench-stub"))]
    pub fn new(cfg: &Config) -> Result<Self> {
        match cfg.model.provider {
            EmbeddingProvider::Local => Self::new_local(cfg),
            EmbeddingProvider::Openai => Self::new_remote(&cfg.remote),
        }
    }

    /// Load the real embedding model (user-defined ONNX or fastembed
    /// built-in) per the model registry.
    #[cfg(not(feature = "bench-stub"))]
    fn new_local(cfg: &Config) -> Result<Self> {
        let cache_dir = cfg.model_cache_dir()?;
        std::fs::create_dir_all(&cache_dir)?;

        // User-provided local ONNX model: bypass the registry entirely.
        if let Some(custom) = &cfg.model.onnx_path {
            let (udm, name) = load_local_user_defined(custom, cfg.model.precision)?;
            return Self::finish_user_defined(udm, cfg, name, cfg.model.onnx_e5_prefix);
        }

        // Registered custom model (HF repo id or direct .onnx URL),
        // downloaded/cached like a built-in.
        if let Some(cm) = cfg.custom_model() {
            let (udm, name) = match (&cm.repo, &cm.url) {
                (Some(repo), None) => {
                    // Model2Vec → static backend; else the ONNX path.
                    if let Some(sm) = try_model2vec(repo, &cache_dir)? {
                        return Ok(Self {
                            dimensions: sm.dim,
                            model_name: format!(
                                "model2vec:{}",
                                crate::config::normalize_hf_repo(repo)
                            ),
                            is_e5: false,
                            backend: Backend::Static(sm),
                        });
                    }
                    // Per-model precision overrides the global one (so a
                    // big model can be int8 without changing the rest);
                    // an explicit `onnx_file` overrides precision.
                    let precision = cm.precision.unwrap_or(cfg.model.precision);
                    let explicit = cm.onnx_file.as_deref();
                    // `load_user_defined`'s error already names the repo
                    // and every path tried; the CLI adds the model-name
                    // context — no extra wrap (avoids a nested prefix).
                    let udm = load_user_defined(repo, precision, &cache_dir, explicit)?;
                    // The identity (→ index fingerprint) tracks the
                    // actual weights pulled: the exact file, else the
                    // precision label.
                    let tag = explicit
                        .map(|f| format!("#{f}"))
                        .unwrap_or_else(|| format!("@{}", precision.label()));
                    (udm, format!("custom:{repo}{tag}"))
                }
                (None, Some(url)) => load_url_user_defined(&cm.name, url, &cache_dir)?,
                (Some(_), Some(_)) => {
                    return Err(Error::Embed(format!(
                        "custom model '{}': set only one of repo / url, not both",
                        cm.name
                    )))
                }
                (None, None) => {
                    return Err(Error::Embed(format!(
                        "custom model '{}': set one of repo / url",
                        cm.name
                    )))
                }
            };
            return Self::finish_user_defined(udm, cfg, name, cm.e5_prefix);
        }

        let spec = cfg.model_spec()?;

        // Built-in Model2Vec (e.g. the default potion-*): the static
        // backend, same routing as a custom repo.
        if spec.static_model {
            if let Some(repo) = spec.hf_repo {
                if let Some(sm) = try_model2vec(repo, &cache_dir)? {
                    return Ok(Self {
                        dimensions: sm.dim,
                        model_name: spec.name.to_string(),
                        is_e5: spec.needs_e5_prefix,
                        backend: Backend::Static(sm),
                    });
                }
            }
        }

        let model = match (&spec.hf_repo, &spec.fastembed) {
            (Some(repo), _) => {
                let precision = spec.effective_precision(cfg.model.precision);
                let udm = load_user_defined(repo, precision, &cache_dir, None)?;
                build_user_defined(udm, cfg)?
            }
            (None, Some(model)) => {
                let init = InitOptions::new(model.clone())
                    .with_cache_dir(cache_dir)
                    .with_show_download_progress(true)
                    .with_max_length(cfg.model.max_length)
                    .with_execution_providers(providers(&cfg.backend));
                TextEmbedding::try_new(init).map_err(|e| Error::Embed(e.to_string()))?
            }
            (None, None) => {
                return Err(Error::Embed(format!(
                    "model {} has neither hf_repo nor fastembed variant",
                    spec.name
                )))
            }
        };

        Ok(Self {
            backend: Backend::Local(Some(Mutex::new(model))),
            dimensions: spec.dimensions,
            model_name: spec.name.to_string(),
            is_e5: spec.needs_e5_prefix,
        })
    }

    /// Build a user-defined model and probe its output dimensions from
    /// a live embed (user-defined ONNX has no static dimension spec).
    /// Shared by the `onnx_path` and registered-custom-model paths.
    #[cfg(not(feature = "bench-stub"))]
    fn finish_user_defined(
        udm: UserDefinedEmbeddingModel,
        cfg: &Config,
        name: String,
        is_e5: bool,
    ) -> Result<Self> {
        let mut model = build_user_defined(udm, cfg)?;
        let dimensions = model
            .embed(vec!["probe"], None)
            .map_err(|e| Error::Embed(e.to_string()))?
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Embed(format!("custom model '{name}': empty probe")))?;
        Ok(Self {
            backend: Backend::Local(Some(Mutex::new(model))),
            dimensions,
            model_name: name,
            is_e5,
        })
    }

    /// Connect to the external service and probe once: this validates
    /// connectivity + auth and resolves output dimensions in one step.
    #[cfg(not(feature = "bench-stub"))]
    fn new_remote(cfg: &RemoteConfig) -> Result<Self> {
        let re = RemoteEmbedder::connect(cfg)?;
        let probe = re.embed_documents(&["probe"]).map_err(|e| {
            Error::Embed(format!(
                "cannot reach/configure embeddings backend at {} (model \
                 {}): {e}. Check [model].provider, [remote] \
                 base_url/api_key/model, and that the service is running.",
                cfg.endpoint(),
                cfg.model
            ))
        })?;
        let probed = probe
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Embed("remote: empty probe response".into()))?;
        let dimensions = match cfg.dimensions {
            Some(d) if d != probed => {
                return Err(Error::Embed(format!(
                    "remote embeddings: configured dimensions {d} != server \
                     {probed} for model {}",
                    cfg.model
                )))
            }
            _ => probed,
        };

        Ok(Self {
            backend: Backend::Remote(re),
            dimensions,
            model_name: format!("{} @ {}", cfg.model, cfg.base_url),
            is_e5: cfg.e5_prefix,
        })
    }

    fn embed_raw(&self, texts: Vec<&str>, batch_size: usize) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match &self.backend {
            Backend::Local(Some(m)) => m
                .lock()
                .map_err(|_| Error::Embed("model lock poisoned".into()))?
                .embed(texts, Some(batch_size))
                .map_err(|e| Error::Embed(e.to_string())),
            #[cfg(feature = "bench-stub")]
            Backend::Local(None) => Ok(texts
                .iter()
                .map(|t| stub_vector(t, self.dimensions))
                .collect()),
            #[cfg(not(feature = "bench-stub"))]
            Backend::Local(None) => Err(Error::Embed("embedder not initialized".into())),
            #[cfg(not(feature = "bench-stub"))]
            Backend::Remote(r) => r.embed_documents(&texts),
            #[cfg(not(feature = "bench-stub"))]
            // Pure CPU, no shared state, no internal batching — embed
            // the group in parallel (the ONNX/remote backends batch
            // internally; this is the equivalent for static models).
            Backend::Static(s) => texts.par_iter().map(|t| s.embed_one(t)).collect(),
        }
    }

    /// Embed indexed chunks. e5 needs a "passage: " prefix; other models
    /// take the text verbatim with no extra allocation.
    pub fn embed_documents(&self, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        if self.is_e5 {
            let owned: Vec<String> = texts.iter().map(|t| format!("passage: {t}")).collect();
            self.embed_raw(owned.iter().map(String::as_str).collect(), batch_size)
        } else {
            self.embed_raw(texts.to_vec(), batch_size)
        }
    }

    /// Embed a search query (e5: "query:" prefix).
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let owned;
        let q: &str = if self.is_e5 {
            owned = format!("query: {text}");
            &owned
        } else {
            text
        };
        let mut v = self.embed_raw(vec![q], 1)?;
        v.pop()
            .ok_or_else(|| Error::Embed("empty embedding".into()))
    }
}

#[cfg(all(test, not(feature = "bench-stub")))]
mod tests {
    use super::*;

    #[test]
    fn onnx_data_sidecars_keep_basename_for_subdir_models() {
        // `onnx/model_fp16.onnx` → fetch `onnx/model_fp16.onnx_data`
        // but the graph references the basename `model_fp16.onnx_data`.
        let s = onnx_data_sidecars("onnx/model_fp16.onnx");
        assert_eq!(
            s,
            [
                (
                    "onnx/model_fp16.onnx_data".to_string(),
                    "model_fp16.onnx_data".to_string()
                ),
                (
                    "onnx/model_fp16.onnx.data".to_string(),
                    "model_fp16.onnx.data".to_string()
                ),
            ]
        );
        // flat model: relative path == basename
        let (rel, name) = onnx_data_sidecars("model.onnx")[0].clone();
        assert_eq!(rel, "model.onnx_data");
        assert_eq!(name, "model.onnx_data");
    }

    #[test]
    fn onnx_candidates_explicit_overrides_precision() {
        // bare explicit name also tried under onnx/
        assert_eq!(
            onnx_candidates(Precision::Fp16, Some("model_q4f16.onnx")),
            vec!["model_q4f16.onnx", "onnx/model_q4f16.onnx"]
        );
        // explicit with a path is used verbatim
        assert_eq!(
            onnx_candidates(Precision::Full, Some("onnx/model_q4.onnx")),
            vec!["onnx/model_q4.onnx"]
        );
        // no explicit ⇒ precision mapping + fallbacks
        assert_eq!(
            onnx_candidates(Precision::Fp16, None)[0],
            "onnx/model_fp16.onnx"
        );
    }

    #[test]
    fn assert_loadable_embedding_rejects_lm_heads() {
        // LM head only → reject
        let mlm = br#"{"architectures":["JinaBertForMaskedLM"]}"#;
        assert!(assert_loadable_embedding("r", mlm).is_err());
        // encoder export, mixed, unparseable/absent → allow
        assert!(assert_loadable_embedding("r", br#"{"architectures":["JinaBertModel"]}"#).is_ok());
        let mixed = br#"{"architectures":["BertModel","BertForMaskedLM"]}"#;
        assert!(assert_loadable_embedding("r", mixed).is_ok());
        assert!(assert_loadable_embedding("r", b"not json").is_ok());
        assert!(assert_loadable_embedding("r", b"{}").is_ok());
    }

    #[test]
    fn model2vec_normalize_detects_static_and_carries_flag() {
        // model_type or StaticModel arch → Some(normalize); normalize
        // defaults to true, honored when explicitly false.
        assert_eq!(
            model2vec_normalize(br#"{"model_type":"model2vec"}"#),
            Some(true)
        );
        assert_eq!(
            model2vec_normalize(br#"{"architectures":["StaticModel"],"normalize":false}"#),
            Some(false)
        );
        // not model2vec → None; LM-head guard does NOT reject model2vec
        assert_eq!(
            model2vec_normalize(br#"{"architectures":["BertModel"]}"#),
            None
        );
        assert_eq!(model2vec_normalize(b"not json"), None);
        assert!(assert_loadable_embedding(
            "r",
            br#"{"model_type":"model2vec","architectures":["StaticModel"]}"#
        )
        .is_ok());
    }

    #[test]
    fn parse_embeddings_reorders_by_index() {
        let resp = EmbResp {
            data: vec![
                EmbItem {
                    embedding: vec![2.0],
                    index: 1,
                },
                EmbItem {
                    embedding: vec![0.0],
                    index: 0,
                },
                EmbItem {
                    embedding: vec![1.0],
                    index: 2,
                },
            ],
        };
        let out = parse_embeddings(resp, 3).unwrap();
        assert_eq!(out, vec![vec![0.0], vec![2.0], vec![1.0]]);
    }

    #[test]
    fn parse_embeddings_rejects_count_mismatch() {
        let resp = EmbResp {
            data: vec![EmbItem {
                embedding: vec![0.0],
                index: 0,
            }],
        };
        assert!(parse_embeddings(resp, 2).is_err());
    }
}
