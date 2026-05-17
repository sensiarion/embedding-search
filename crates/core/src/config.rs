use crate::error::{Error, Result};
use fastembed::EmbeddingModel;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "intfloat/multilingual-e5-base";

/// Bumped whenever chunking logic changes shape (AST kinds, line
/// window, structured split). Part of the index fingerprint so a
/// chunker change forces a full re-index.
pub const CHUNKER_VERSION: u32 = 1;

/// Per-project index directory name (`<project>/.embedding-search`).
/// Single owner of this literal — also the global state dir's basename
/// under `$HOME` (see `Config::app_dir`) and a default walk exclusion.
pub const PROJECT_INDEX_DIR: &str = ".embedding-search";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub backend: BackendConfig,
    pub paths: PathsConfig,
    pub sync: SyncConfig,
    pub search: SearchConfig,
    pub remote: RemoteConfig,
    /// User-registered models (HF repo id or direct ONNX URL), added
    /// via `embedding-search models add`. Select one by setting
    /// `[model] default` to its `name` (or `models set-default`).
    #[serde(default, rename = "custom_model")]
    pub custom_models: Vec<CustomModel>,
    /// Registered remote (OpenAI-compatible) backends, added via
    /// `models add-remote`. Re-select one by name with
    /// `models set-default <name>` — no need to redefine it. The
    /// chosen entry is copied into `[remote]` (the active resolved
    /// config the embedder reads).
    #[serde(default, rename = "remote_model")]
    pub remote_models: Vec<RemoteConfig>,
}

/// A user-registered embedding model resolved at load time: an ONNX
/// model + its own tokenizer files, fetched from a Hugging Face repo or
/// a direct URL and cached like the built-ins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomModel {
    /// Label used to select it (`[model] default = "<name>"`).
    pub name: String,
    /// Hugging Face repo id, e.g. `Xenova/bge-small-en-v1.5`. The
    /// precision-specific ONNX + the four tokenizer files are pulled
    /// from it (cached). Mutually exclusive with `url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Direct URL to a `.onnx` file. The four tokenizer files are
    /// fetched from the same directory (same URL, filename swapped).
    /// Mutually exclusive with `repo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// True if it is an e5-style model needing `query:`/`passage:`.
    #[serde(default)]
    pub e5_prefix: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// When the vector index holds fewer than this many vectors, use
    /// exact (brute-force) cosine instead of the HNSW graph. HNSW is an
    /// approximate heuristic; on a small codebase exact is both more
    /// accurate and effectively as fast, so the graph buys nothing.
    pub exact_below: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            exact_below: 50_000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub default: String,
    /// Local in-process ONNX (fastembed) vs an external OpenAI-compatible
    /// embeddings HTTP service (DeepSeek, LiteLLM, OpenAI). When `openai`,
    /// the `[remote]` section is used and `[model]`/`[backend]` ONNX
    /// settings are ignored.
    pub provider: EmbeddingProvider,
    /// Weight precision. fp16 ≈ half the RAM of f32 with negligible
    /// quality loss; int8 ≈ quarter RAM with small loss. Ignored for
    /// models that have no quantized ONNX (falls back to full).
    pub precision: Precision,
    /// Max token sequence fed to the model. Caps the O(batch·seq²)
    /// attention tensor (a memory lever) AND the amount of each chunk
    /// the model actually sees — text past this many tokens is silently
    /// truncated before embedding. Keep it aligned with
    /// `sync.max_chunk_bytes` (≈4 bytes/token for code) so whole chunks
    /// are embedded, not just their head. e5 native/hard max is 512.
    pub max_length: usize,
    /// Use a user-provided ONNX model instead of the built-in registry.
    /// Either a directory containing `model.onnx` (or
    /// `onnx/model*.onnx`) plus the four tokenizer files
    /// (`tokenizer.json`, `config.json`, `special_tokens_map.json`,
    /// `tokenizer_config.json`), or a direct path to the `.onnx` file
    /// (tokenizer files taken from its parent dir). Output dimensions
    /// are probed at load. `[model] default` is ignored when set.
    pub onnx_path: Option<PathBuf>,
    /// Set true if the custom ONNX model is an e5 variant that needs
    /// `query: ` / `passage: ` input prefixes.
    pub onnx_e5_prefix: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
    /// In-process fastembed ONNX model (default).
    #[default]
    Local,
    /// External OpenAI-compatible `/embeddings` HTTP endpoint.
    Openai,
}

/// External OpenAI-compatible embeddings service settings. Used only
/// when `[model] provider = "openai"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RemoteConfig {
    /// Registry label. Empty for the active resolved `[remote]`
    /// section; set on each `[[remote_model]]` registry entry so it
    /// can be re-selected by name later without redefining it.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub name: String,
    /// API base, e.g. `http://localhost:4000/v1` (LiteLLM) or
    /// `https://api.deepseek.com/v1`. `/embeddings` is appended.
    pub base_url: String,
    /// Bearer token. A value of `"$NAME"` or `"${NAME}"` is read from
    /// the `NAME` env var; empty string sends no Authorization header
    /// (local unauthenticated LiteLLM).
    pub api_key: String,
    /// Remote model id, e.g. `text-embedding-3-small`.
    pub model: String,
    /// Output dimensions. `None` → probed once at startup from a live
    /// request (also validates connectivity + auth).
    pub dimensions: Option<usize>,
    /// Texts per request — the OpenAI `input` array batch size.
    pub batch_size: usize,
    /// Max parallel in-flight requests (bounded worker pool).
    pub concurrency: usize,
    /// Per-request timeout.
    pub timeout_seconds: u64,
    /// Prefix `query: ` / `passage: ` (only if the remote serves an e5
    /// model). OpenAI / DeepSeek models: leave false.
    pub e5_prefix: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            base_url: "http://localhost:4000/v1".to_string(),
            api_key: String::new(),
            model: "text-embedding-3-small".to_string(),
            dimensions: None,
            batch_size: 64,
            concurrency: 4,
            timeout_seconds: 60,
            e5_prefix: false,
        }
    }
}

impl RemoteConfig {
    /// Resolve the bearer token, expanding a leading `$NAME` / `${NAME}`
    /// to the env var. Empty (or unset env) → no auth header.
    pub fn resolved_api_key(&self) -> String {
        let raw = self.api_key.trim();
        let var = raw
            .strip_prefix("${")
            .and_then(|s| s.strip_suffix('}'))
            .or_else(|| raw.strip_prefix('$'));
        match var {
            Some(name) => std::env::var(name).unwrap_or_default(),
            None => raw.to_string(),
        }
    }

    /// Full endpoint URL (`base_url` + `/embeddings`).
    pub fn endpoint(&self) -> String {
        format!("{}/embeddings", self.base_url.trim_end_matches('/'))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    Full,
    Fp16,
    Int8,
}

impl Precision {
    /// ONNX file inside the HF repo's `onnx/` dir.
    pub fn onnx_file(self) -> &'static str {
        match self {
            Precision::Full => "onnx/model.onnx",
            Precision::Fp16 => "onnx/model_fp16.onnx",
            Precision::Int8 => "onnx/model_quantized.onnx",
        }
    }
    /// Bytes per weight, for RAM estimation.
    pub fn bytes_per_param(self) -> f32 {
        match self {
            Precision::Full => 4.0,
            Precision::Fp16 => 2.0,
            Precision::Int8 => 1.0,
        }
    }
    /// Short display label (config uses lowercase serde names: full/fp16/int8).
    pub fn label(self) -> &'static str {
        match self {
            Precision::Full => "f32",
            Precision::Fp16 => "fp16",
            Precision::Int8 => "int8",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BackendConfig {
    pub execution_provider: ExecutionProvider,
    /// Disable the ONNX Runtime CPU memory arena. The arena extends by
    /// powers of two and never returns memory — the source of multi-GB
    /// blowup with dynamic input shapes.
    pub disable_mem_arena: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    Auto,
    Coreml,
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    pub cache_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncConfig {
    /// Directory names skipped during the walk. Defaults to common
    /// build/VCS/venv dirs; set this in config.toml to override the
    /// whole list.
    #[serde(default = "default_exclude_dirs")]
    pub exclude_dirs: Vec<String>,
    /// Extra substring patterns; a path containing any is skipped.
    pub exclude: Vec<String>,
    /// Hard ceiling on every emitted chunk (bytes). Enforced on all
    /// chunk paths, not just AST. Keep ≈ `model.max_length * 4` (code
    /// tokenizes at ~4 bytes/token): larger wastes embedding on text
    /// the model truncates away; much smaller fragments functions and
    /// loses semantic context. Changing this triggers a full re-index.
    pub max_chunk_bytes: usize,
    pub embed_batch_size: usize,
    /// Flush an embed group once accumulated chunk bytes reach this,
    /// regardless of count — bounds peak tokenizer/tensor memory. Large
    /// files are NOT skipped; they are chunk-capped and streamed.
    pub embed_batch_bytes: usize,
    /// Files are read+hashed+chunked in parallel windows of this size.
    /// Caps peak memory at O(window) chunked files.
    pub scan_window: usize,
    /// Max worker threads for the parallel scan/hash/parse phase.
    /// `0` = auto (all cores but one, so the machine stays responsive).
    /// Set lower to cap CPU during sync on a busy workstation.
    pub sync_threads: usize,
    /// Background resync cadence (minutes). Drives BOTH the CLI search
    /// throttle (`search` resyncs only if the last one is older than
    /// this) AND the MCP server's periodic background resync loop.
    /// There is no file watcher: a file the agent just edited is
    /// already in its context, so the index only needs to catch up for
    /// the *next* agent run — this interval is that catch-up period.
    /// Every resync is hash-incremental (unchanged files skipped before
    /// any parse/embed), so a tick with no edits costs ~nothing.
    pub resync_interval_minutes: i64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default: DEFAULT_MODEL.to_string(),
            provider: EmbeddingProvider::Local,
            precision: Precision::Fp16,
            max_length: 512,
            onnx_path: None,
            onnx_e5_prefix: false,
        }
    }
}

impl Default for BackendConfig {
    fn default() -> Self {
        // Auto: CoreML on Apple Silicon / CUDA when built with the
        // feature / else CPU. Quantized weights + arena-off + capped
        // chunks keep CoreML's graph-partition memory bounded.
        Self {
            execution_provider: ExecutionProvider::Auto,
            disable_mem_arena: true,
        }
    }
}

/// Default directory names excluded from the walk (build/VCS/venv).
pub fn default_exclude_dirs() -> Vec<String> {
    [
        PROJECT_INDEX_DIR,
        "target",
        "node_modules",
        ".venv",
        "venv",
        "env",
        ".env",
        "__pycache__",
        "dist",
        "build",
        ".git",
        ".svn",
        ".hg",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect()
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            exclude_dirs: default_exclude_dirs(),
            exclude: Vec::new(),
            // ≈ model.max_length(512) * 4 bytes/token: whole chunks fit
            // the model's context, none silently truncated.
            max_chunk_bytes: 2048,
            embed_batch_size: 16,
            embed_batch_bytes: 256 * 1024,
            scan_window: 128,
            sync_threads: 0,
            resync_interval_minutes: 10,
        }
    }
}

impl Config {
    /// Global app state directory `~/.embedding-search` (config + model
    /// cache). One fixed location on every OS — no platform
    /// config/cache split. (Distinct from `dirs::home_dir()`, the OS
    /// home; this is the app's subdir under it.)
    pub fn app_dir() -> Result<PathBuf> {
        let home = dirs::home_dir().ok_or_else(|| Error::Config("no home dir".into()))?;
        Ok(home.join(PROJECT_INDEX_DIR))
    }

    /// `~/.embedding-search/config.toml`. `embedding-search status`
    /// prints the resolved path.
    pub fn config_path() -> Result<PathBuf> {
        Ok(Self::app_dir()?.join("config.toml"))
    }

    /// Load from config file, falling back to defaults if missing.
    pub fn load() -> Result<Self> {
        let path = Self::config_path()?;
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)?;
        toml::from_str(&raw).map_err(|e| Error::Config(e.to_string()))
    }

    /// Load the config, materializing a default `config.toml` on first
    /// run so the user can see and edit it. A write failure (read-only
    /// HOME, etc.) is non-fatal — defaults are still returned.
    pub fn load_or_init() -> Result<Self> {
        let path = Self::config_path()?;
        if path.exists() {
            return Self::load();
        }
        let cfg = Self::default();
        if let Err(e) = cfg.save() {
            tracing::warn!("could not write default config to {path:?}: {e}");
        }
        Ok(cfg)
    }

    /// Write current config to the config file (creating parent dirs).
    pub fn save(&self) -> Result<()> {
        let path = Self::config_path()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let raw = toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))?;
        std::fs::write(&path, raw)?;
        Ok(())
    }

    /// Model cache dir: configured override or `~/.embedding-search/models`.
    pub fn model_cache_dir(&self) -> Result<PathBuf> {
        if let Some(dir) = &self.paths.cache_dir {
            return Ok(dir.clone());
        }
        Ok(Self::app_dir()?.join("models"))
    }

    /// The registered custom model selected by `[model] default`, if any.
    pub fn custom_model(&self) -> Option<&CustomModel> {
        self.custom_models
            .iter()
            .find(|m| m.name == self.model.default)
    }

    /// Find a registered remote backend by name.
    pub fn remote_model(&self, name: &str) -> Option<&RemoteConfig> {
        self.remote_models.iter().find(|r| r.name == name)
    }

    /// Select a previously registered model/remote by name: a built-in,
    /// a `[[custom_model]]`, or a `[[remote_model]]`. For a remote, the
    /// stored entry is copied into the active `[remote]` section.
    /// Returns the resolved provider; errors if the name is unknown.
    pub fn select_model(&mut self, name: &str) -> Result<EmbeddingProvider> {
        if let Some(remote) = self.remote_model(name).cloned() {
            self.remote = remote;
            self.model.provider = EmbeddingProvider::Openai;
            self.model.default = name.to_string();
            return Ok(EmbeddingProvider::Openai);
        }
        let local_known =
            model_spec(name).is_some() || self.custom_models.iter().any(|m| m.name == name);
        if !local_known {
            return Err(Error::Config(format!(
                "unknown model: {name} (not a built-in, custom, or remote — \
                 register it with `models add` / `models add-remote`)"
            )));
        }
        self.model.provider = EmbeddingProvider::Local;
        self.model.default = name.to_string();
        Ok(EmbeddingProvider::Local)
    }

    pub fn model_spec(&self) -> Result<&'static ModelSpec> {
        model_spec(&self.model.default)
            .ok_or_else(|| Error::Config(format!("unknown model: {}", self.model.default)))
    }

    /// Identity of everything that invalidates an existing index when
    /// changed: model, output dims, precision, token cap, chunk byte
    /// cap and chunker logic version. Stored in
    /// `meta.index_fingerprint`; a mismatch on startup wipes + rebuilds.
    pub fn index_fingerprint(&self, model_name: &str, dimensions: usize) -> String {
        format!(
            "v{}|{}|{}|{}|{}|{}",
            CHUNKER_VERSION,
            model_name,
            dimensions,
            self.model.precision.label(),
            self.model.max_length,
            self.sync.max_chunk_bytes,
        )
    }
}

/// Static metadata for a supported embedding model — the single source
/// of truth shared by config, embedder and CLI.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    /// Built-in fastembed model (f32 enum path). Mutually exclusive with
    /// `hf_repo` (user-defined ONNX path with fp16/int8).
    pub fastembed: Option<EmbeddingModel>,
    pub dimensions: usize,
    pub code: u8,
    /// 1..5 multilingual / non-English (incl. Russian) coverage rating.
    pub multilingual: u8,
    pub params_m: u32,
    /// e5 models require "query:" / "passage:" input prefixes.
    pub needs_e5_prefix: bool,
    /// HF repo with precision-specific ONNX (`onnx/model*.onnx`) +
    /// tokenizer. `Some` => loaded as a user-defined model so fp16/int8
    /// works. `None` => fastembed built-in enum, f32 only.
    pub hf_repo: Option<&'static str>,
    pub note: &'static str,
}

impl ModelSpec {
    /// Whether `precision` is honored (only user-defined HF models).
    pub fn supports_precision(&self) -> bool {
        self.hf_repo.is_some()
    }

    /// Effective precision: requested if supported, else Full.
    pub fn effective_precision(&self, want: Precision) -> Precision {
        if self.supports_precision() {
            want
        } else {
            Precision::Full
        }
    }

    /// Rough resident RAM in MB: weights at the precision + a fixed
    /// ONNX Runtime / tokenizer working-set overhead.
    pub fn ram_mb(&self, precision: Precision) -> u32 {
        const OVERHEAD_MB: f32 = 350.0;
        let p = self.effective_precision(precision);
        let weights = self.params_m as f32 * p.bytes_per_param();
        (weights + OVERHEAD_MB) as u32
    }
}

pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "intfloat/multilingual-e5-base",
        fastembed: None,
        dimensions: 768,
        code: 4,
        multilingual: 5,
        params_m: 278,
        needs_e5_prefix: true,
        hf_repo: Some("Xenova/multilingual-e5-base"),
        note: "DEFAULT: 100 langs incl. Russian, strong code",
    },
    ModelSpec {
        name: "intfloat/multilingual-e5-small",
        fastembed: None,
        dimensions: 384,
        code: 3,
        multilingual: 5,
        params_m: 118,
        needs_e5_prefix: true,
        hf_repo: Some("Xenova/multilingual-e5-small"),
        note: "Lightweight multilingual, lowest RAM",
    },
    ModelSpec {
        name: "intfloat/multilingual-e5-large",
        fastembed: None,
        dimensions: 1024,
        code: 3,
        multilingual: 5,
        params_m: 560,
        needs_e5_prefix: true,
        hf_repo: Some("Xenova/multilingual-e5-large"),
        note: "Max multilingual quality, heavier",
    },
    ModelSpec {
        name: "jinaai/jina-embeddings-v2-base-code",
        fastembed: Some(EmbeddingModel::JinaEmbeddingsV2BaseCode),
        dimensions: 768,
        code: 5,
        multilingual: 2,
        params_m: 161,
        needs_e5_prefix: false,
        hf_repo: None,
        note: "Best pure code, 30 prog langs, English only (f32)",
    },
    ModelSpec {
        name: "nomic-ai/nomic-embed-text-v1.5",
        fastembed: Some(EmbeddingModel::NomicEmbedTextV15),
        dimensions: 768,
        code: 4,
        multilingual: 3,
        params_m: 137,
        needs_e5_prefix: false,
        hf_repo: None,
        note: "Fast, matryoshka dims, mostly English (f32)",
    },
];

pub fn model_spec(name: &str) -> Option<&'static ModelSpec> {
    SUPPORTED_MODELS.iter().find(|m| m.name == name)
}
