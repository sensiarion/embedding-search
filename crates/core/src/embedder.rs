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
    serde::Deserialize,
    std::path::Path,
    std::time::Duration,
};

enum Backend {
    /// `Some` = loaded ONNX model. `None` only under `bench-stub`
    /// (deterministic hash embedder, no model load).
    Local(Option<Mutex<TextEmbedding>>),
    #[cfg(not(feature = "bench-stub"))]
    Remote(RemoteEmbedder),
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

/// Assemble a user-defined model: `onnx` is the model bytes,
/// `fetch_tok` supplies the four tokenizer files by name (hf-hub cache
/// or a local dir). Single source of the tokenizer file set + pooling.
#[cfg(not(feature = "bench-stub"))]
fn user_defined_from(
    onnx: Vec<u8>,
    fetch_tok: impl Fn(&str) -> Result<Vec<u8>>,
) -> Result<UserDefinedEmbeddingModel> {
    let tok = TokenizerFiles {
        tokenizer_file: fetch_tok("tokenizer.json")?,
        config_file: fetch_tok("config.json")?,
        special_tokens_map_file: fetch_tok("special_tokens_map.json")?,
        tokenizer_config_file: fetch_tok("tokenizer_config.json")?,
    };
    Ok(UserDefinedEmbeddingModel::new(onnx, tok).with_pooling(Pooling::Mean))
}

/// Download (cached) the precision-specific ONNX + tokenizer files from
/// the HF repo so fp16/int8 works.
#[cfg(not(feature = "bench-stub"))]
fn load_user_defined(
    repo: &str,
    precision: Precision,
    cache_dir: &Path,
) -> Result<UserDefinedEmbeddingModel> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(cache_dir.to_path_buf())
        .with_progress(true)
        .build()
        .map_err(|e| Error::Embed(format!("hf-hub: {e}")))?;
    let r = api.model(repo.to_string());
    let fetch = |f: &str| -> Result<Vec<u8>> {
        let p = r
            .get(f)
            .map_err(|e| Error::Embed(format!("hf-hub get {f}: {e}")))?;
        read(&p)
    };
    let onnx = fetch(precision.onnx_file())?;
    user_defined_from(onnx, fetch)
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
#[cfg(not(feature = "bench-stub"))]
fn slug(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
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
    let udm = user_defined_from(onnx, |f| fetch(f, f))?;
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
    let udm = user_defined_from(onnx, |f| read(&dir.join(f)))?;
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
                    let udm =
                        load_user_defined(repo, cfg.model.precision, &cache_dir).map_err(|e| {
                            Error::Embed(format!(
                                "custom model '{}' (repo {repo}): {e}. The HF repo must \
                                 contain the ONNX weights ({}) plus tokenizer.json, \
                                 config.json, special_tokens_map.json, \
                                 tokenizer_config.json.",
                                cm.name,
                                cfg.model.precision.onnx_file()
                            ))
                        })?;
                    (
                        udm,
                        format!("custom:{repo}@{}", cfg.model.precision.label()),
                    )
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

        let model = match (&spec.hf_repo, &spec.fastembed) {
            (Some(repo), _) => {
                let precision = spec.effective_precision(cfg.model.precision);
                let udm = load_user_defined(repo, precision, &cache_dir)?;
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
