use crate::error::{Error, Result};
use fastembed::EmbeddingModel;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "nomic-ai/CodeRankEmbed";

/// Bumped whenever chunking logic OR the embedded-text shape changes
/// (AST kinds, line window, structured split, the code↔NL embed
/// header). Part of the index fingerprint so the change forces a
/// one-time full re-index. v2: chunk-enrichment header prepended to
/// the embedded text.
pub const CHUNKER_VERSION: u32 = 2;

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
    pub rerank: RerankConfig,
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
    /// Prepended to a search query before embedding (`None`/absent =
    /// none). e.g. `"query: "` for e5, `"search_query: "` for nomic, a
    /// task instruction for CLS code models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    /// Prepended to an indexed chunk before embedding (`None`/absent =
    /// none — many code/CLS models prefix only the query side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_prefix: Option<String>,
    /// Encoder output pooling (`mean` default; `cls` / `last-token`).
    #[serde(default)]
    pub pooling: Pooling,
    /// ONNX precision to pull for this model (HF `--repo` only). `None`
    /// ⇒ the global `[model] precision`. Per-model so registering a
    /// big model at int8 doesn't change precision for the others.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precision: Option<Precision>,
    /// Exact ONNX file to pull from the repo, e.g. `model_q4f16.onnx`
    /// or `onnx/model_q4.onnx` — overrides the `precision`→file
    /// mapping for repos with quantizations it doesn't cover (q4,
    /// q4f16, bnb4, uint8…). A bare name is also tried under `onnx/`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_file: Option<String>,
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

/// Default ONNX cross-encoder for the optional re-rank stage:
/// XLM-RoBERTa `*ForSequenceClassification` (one relevance logit),
/// pinned to its int8 export.
pub const DEFAULT_RERANK_MODEL: &str = "Xenova/bge-reranker-base";

/// Optional cross-encoder re-rank of the fused candidate neighborhood
/// before truncating to the result limit (selection precision is the
/// dominant lever once recall is adequate). **Off by default**:
/// disabled ⇒ byte-for-byte the pre-rerank behavior, no second model
/// download, no added latency. It does NOT affect the index
/// fingerprint — re-rank changes ordering, not stored vectors, so
/// toggling it never triggers a rebuild.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    pub enabled: bool,
    /// HF repo of the ONNX cross-encoder. A
    /// `*ForSequenceClassification` reranker emitting a single
    /// relevance logit; loaded from its int8 `model_quantized.onnx`.
    pub model: String,
    /// How many top fused candidates to re-score. Beyond this the
    /// cross-encoder's marginal gain no longer pays its linear cost.
    pub top_n: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: DEFAULT_RERANK_MODEL.to_string(),
            top_n: 50,
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
    /// Prefix prepended to a query / document before embedding the
    /// `onnx_path` model (`None` = none). e5 ⇒ `"query: "` /
    /// `"passage: "`; nomic ⇒ `"search_query: "` / `"search_document: "`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_query_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub onnx_doc_prefix: Option<String>,
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
    /// Prefix prepended to a query / document before sending it to the
    /// remote (`None` = none — OpenAI / DeepSeek). e5 ⇒ `"query: "` /
    /// `"passage: "`; nomic ⇒ `"search_query: "` / `"search_document: "`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_prefix: Option<String>,
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
            query_prefix: None,
            doc_prefix: None,
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

/// Single source for string → `Precision` (CLI `--precision`, etc.):
/// the serde names plus `f32` as an alias for `full`. clap derives its
/// value parser from this automatically.
impl std::str::FromStr for Precision {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "fp16" => Ok(Precision::Fp16),
            "int8" => Ok(Precision::Int8),
            "full" | "f32" => Ok(Precision::Full),
            other => Err(Error::Config(format!(
                "invalid precision {other:?} (expected fp16 | int8 | full)"
            ))),
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
    /// Hard ceiling on the process's peak resident memory during a
    /// sync, in MB. Checked at every embed-batch flush; if exceeded the
    /// sync aborts with an actionable error instead of letting an ONNX
    /// transformer balloon to tens of GB and wedge the machine (already
    /// embedded files are committed — the index just stays partial
    /// until a lighter model / smaller knobs finish it). `0` disables
    /// the guard. The static default model peaks well under this.
    pub max_rss_mb: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            default: DEFAULT_MODEL.to_string(),
            provider: EmbeddingProvider::Local,
            precision: Precision::Fp16,
            max_length: 512,
            onnx_path: None,
            onnx_query_prefix: None,
            onnx_doc_prefix: None,
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
            // 0 = auto: use the active model's `rec_batch` (small for
            // heavy ONNX so it can't 10 GB-wedge a real repo, large for
            // the static models). Any explicit value overrides it.
            embed_batch_size: 0,
            embed_batch_bytes: 256 * 1024,
            scan_window: 128,
            sync_threads: 0,
            resync_interval_minutes: 10,
            // Abort a sync before it can wedge the machine. The static
            // default peaks <1 GB on a large repo; an ONNX transformer
            // can spike past this — by design it then aborts with guidance.
            max_rss_mb: 3072,
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
    /// cap, chunker logic version, and the resolved input/output
    /// `contract` (query/doc prefix + pooling — changing a registered
    /// model's prefix or pooling must re-embed, model name alone won't
    /// shift). Stored in `meta.index_fingerprint`; a mismatch on
    /// startup wipes + rebuilds.
    pub fn index_fingerprint(&self, model_name: &str, dimensions: usize, contract: &str) -> String {
        format!(
            "v{}|{}|{}|{}|{}|{}|{}",
            CHUNKER_VERSION,
            model_name,
            dimensions,
            self.model.precision.label(),
            self.model.max_length,
            self.sync.max_chunk_bytes,
            contract,
        )
    }

    /// Effective embed batch size.
    ///
    /// `0` = auto → the active model's `rec_batch` (or
    /// `AUTO_EMBED_BATCH_FALLBACK` for a custom/remote model with no
    /// spec). A non-zero `[sync] embed_batch_size` is an explicit
    /// request, but for a **transformer** model it is clamped down to
    /// `rec_batch`: attention memory is O(batch · seq²), so a large
    /// value (typically a stale pre-per-model config) would OOM a real
    /// repo. A **static** Model2Vec model has no attention, so its
    /// explicit value is honored as-is (bigger = faster, no risk).
    pub fn embed_batch(&self) -> usize {
        let spec = model_spec(&self.model.default);
        let rec = spec
            .map(|s| s.rec_batch as usize)
            .unwrap_or(AUTO_EMBED_BATCH_FALLBACK);
        let explicit = self.sync.embed_batch_size;
        if explicit == 0 {
            return rec;
        }
        // Static (no spec ⇒ assume transformer, the safe default).
        if spec.is_some_and(ModelSpec::is_static) {
            explicit
        } else {
            explicit.min(rec)
        }
    }
}

/// Auto-batch fallback for models with no static spec (custom ONNX /
/// remote): small enough that an unknown heavy model can't OOM.
pub const AUTO_EMBED_BATCH_FALLBACK: usize = 8;

/// Sentence-vector pooling over the encoder's token states. Part of a
/// model's input/output contract (alongside the query/doc prefixes):
/// the wrong one silently produces unusable vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pooling {
    /// Attention-masked mean of `last_hidden_state` — e5, nomic, jina,
    /// most sentence-transformers encoders.
    #[default]
    Mean,
    /// First token (`[CLS]`) of `last_hidden_state` — CLS-trained
    /// encoders (Snowflake arctic-embed-v2, nomic CodeRankEmbed).
    Cls,
    /// Last non-padding token — causal/decoder embedders
    /// (Qwen3-Embedding). Encoder ONNX never needs this; here so the
    /// abstraction is complete for the forthcoming candle backend.
    LastToken,
}

impl Pooling {
    /// Stable short label (kebab serde names) for the index fingerprint
    /// and `models list`.
    pub fn label(self) -> &'static str {
        match self {
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::LastToken => "last-token",
        }
    }
}

/// Single source for string → `Pooling` (CLI `--pooling`); clap derives
/// its value parser from this. Accepts `last_token`/`lasttoken` too.
impl std::str::FromStr for Pooling {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "mean" => Ok(Pooling::Mean),
            "cls" => Ok(Pooling::Cls),
            "last-token" | "last_token" | "lasttoken" => Ok(Pooling::LastToken),
            other => Err(Error::Config(format!(
                "invalid pooling {other:?} (expected mean | cls | last-token)"
            ))),
        }
    }
}

/// e5-family input prefixes (intfloat e5 / multilingual-e5 / e5
/// fine-tunes). One source for the registry entries and the
/// `--e5_prefix` CLI sugar (Add / AddRemote).
pub const E5_QUERY: &str = "query: ";
pub const E5_PASSAGE: &str = "passage: ";

/// How a model is loaded and run. Detected from a model's
/// `config.json` for user-added models (`models add`); declared up
/// front for every built-in. The single switch the embedder dispatches
/// on, so adding an architecture is one enum arm + one detector branch.
#[derive(Debug, Clone)]
pub enum ModelArch {
    /// Model2Vec / `StaticModel`: a static `[vocab, dim]` token-
    /// embedding matrix, mean-pooled. No transformer/ONNX, tiny RAM,
    /// precision N/A. (`config.json`: `model_type == "model2vec"` or
    /// `architectures` ⊇ `StaticModel`.)
    Static,
    /// Transformer encoder loaded from an HF ONNX repo as a
    /// user-defined model: mean-pooled `last_hidden_state`, precision
    /// (fp16/int8) honored. (`config.json`: a BERT/RoBERTa/XLM-R/MPNet/
    /// Nomic-BERT… encoder — NOT an `*ForMaskedLM/CausalLM` head.)
    OnnxEncoder,
    /// Transformer encoder served by fastembed's bundled registry —
    /// fastembed ships a known-good embedding ONNX + pooling. Used when
    /// the only public HF export is an LM head (e.g. jina-v2-base-code,
    /// whose HF ONNX is `JinaBertForMaskedLM`). Built-ins only; carries
    /// the fastembed model id.
    Fastembed(EmbeddingModel),
}

/// Which repo ONNX file to load, resolved against the active execution
/// provider. Replaces a blunt `force_cpu` bool: most models ship one
/// export used on any provider; CodeRankEmbed keeps the small **int8**
/// export for CPU/CoreML (the ORT CoreML EP can't accelerate it — and
/// MLProgram miscompiles its NomicBert rotary — so Apple Silicon stays
/// on int8/CPU) but switches to the **f32** export under CUDA, where
/// it is genuinely accelerated; a few raw exports have no accelerable
/// form and must always run on CPU.
#[derive(Debug, Clone)]
pub enum OnnxFiles {
    /// One export for every provider. `None` ⇒ derive the file from
    /// `[model] precision` (the fp16/int8 mapping); `Some` pins an
    /// exact repo file (a non-standard quantized name).
    Single(Option<&'static str>),
    /// Distinct files per provider: the `accel` (f32) file when an
    /// accelerator that actually speeds it up is active (CUDA — see
    /// `embedder::accel_active`), else the smaller `cpu` (int8) file
    /// pinned to the CPU provider (also the Apple-Silicon path: the
    /// ORT CoreML EP cannot accelerate these graphs).
    AccelCpu {
        accel: &'static str,
        cpu: &'static str,
    },
    /// No accelerable export — always pinned to the CPU provider
    /// (raw `torch.onnx.export`: low opset, hundreds of dynamic-shape
    /// nodes that balloon the CoreML/CUDA partitioner to multiple GB).
    /// `None` ⇒ precision mapping.
    CpuOnly(Option<&'static str>),
}

impl OnnxFiles {
    /// Resolve to `(explicit repo file, pin the CPU provider?)` given
    /// whether an accelerator that genuinely speeds these graphs up
    /// (CUDA) is active. The single switch the embedder dispatches on.
    pub fn resolve(&self, accel_active: bool) -> (Option<&'static str>, bool) {
        match self {
            OnnxFiles::Single(f) => (*f, false),
            OnnxFiles::CpuOnly(f) => (*f, true),
            OnnxFiles::AccelCpu { accel, cpu } => {
                if accel_active {
                    (Some(accel), false)
                } else {
                    (Some(cpu), true)
                }
            }
        }
    }

    /// `Some` short tag for the index fingerprint iff the resolved file
    /// can flip (`AccelCpu`: f32↔int8 = different vectors, must re-embed
    /// when it changes). `None` for `Single`/`CpuOnly` — they never
    /// flip, precision already covers them.
    pub fn variant_tag(&self, accel_active: bool) -> Option<&'static str> {
        match self {
            OnnxFiles::AccelCpu { .. } => Some(if accel_active { "accel" } else { "cpu" }),
            OnnxFiles::Single(_) | OnnxFiles::CpuOnly(_) => None,
        }
    }
}

/// Static metadata for a supported embedding model — the single source
/// of truth shared by config, embedder and CLI.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    pub dimensions: usize,
    pub code: u8,
    /// 1..5 multilingual / non-English (incl. Russian) coverage rating.
    pub multilingual: u8,
    pub params_m: u32,
    /// Prepended to a search query before embedding (`None` = none).
    /// e5: `"query: "`; nomic: `"search_query: "`; arctic-v2 / nomic
    /// CodeRankEmbed: a task instruction; Qwen3: the full
    /// `Instruct: …\nQuery:` template.
    pub query_prefix: Option<&'static str>,
    /// Prepended to an indexed document chunk before embedding (`None`
    /// = none — many code/CLS models prefix only the query side).
    pub doc_prefix: Option<&'static str>,
    /// Sentence-vector pooling for this model's encoder output.
    pub pooling: Pooling,
    /// Source HF repo for `Static`/`OnnxEncoder` (Model2Vec safetensors
    /// resp. precision-specific `onnx/model*.onnx` + tokenizer).
    /// `None` for `Fastembed` (fastembed fetches its own bundle).
    pub hf_repo: Option<&'static str>,
    /// Which repo ONNX file to load per resolved execution provider
    /// (`OnnxEncoder` only). Subsumes the old `onnx_file` + `force_cpu`:
    /// pins an exact file and/or the CPU provider for raw exports that
    /// fragment the CoreML/CUDA partitioner.
    pub onnx: OnnxFiles,
    /// Base HF repo with f32 `model.safetensors` for the candle Metal
    /// backend (Apple Silicon only). `Some` ⇒ on aarch64 macOS this
    /// model runs on the Metal GPU via candle (≈1.8x the int8 ONNX CPU
    /// path; the ORT CoreML EP can't accelerate it), falling back to
    /// the `onnx`/ONNX path if Metal is unreachable or off-Apple. Only
    /// NomicBert (CodeRankEmbed) is wired today.
    pub candle_repo: Option<&'static str>,
    /// How this model is loaded/run (the dispatch switch).
    pub arch: ModelArch,
    /// Recommended embed batch size for this model, used when
    /// `[sync] embed_batch_size = 0` (auto). Transformer memory is
    /// O(batch · seq²), so heavy ONNX models get a small batch (avoids
    /// the ~10 GB blow-up on a real repo) while the static Model2Vec
    /// models — no attention — get a large batch for throughput.
    pub rec_batch: u16,
    pub note: &'static str,
}

impl ModelSpec {
    /// Model2Vec static-matrix backend (no transformer / ORT).
    pub fn is_static(&self) -> bool {
        matches!(self.arch, ModelArch::Static)
    }

    /// Whether `[model] precision` is honored. Only the user-defined
    /// ONNX path has per-precision weight files; the static matrix is
    /// f32-only and fastembed picks its own bundled weights.
    pub fn supports_precision(&self) -> bool {
        matches!(self.arch, ModelArch::OnnxEncoder)
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
    /// working-set overhead. Model2Vec has no ONNX Runtime, so only a
    /// small tokenizer overhead vs. ORT's ~350 MB.
    pub fn ram_mb(&self, precision: Precision) -> u32 {
        let overhead = if self.is_static() { 30.0 } else { 350.0 };
        let p = self.effective_precision(precision);
        let weights = self.params_m as f32 * p.bytes_per_param();
        (weights + overhead) as u32
    }
}

pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    // DEFAULT: nomic CodeRankEmbed — SOTA code retrieval (NomicBertModel
    // encoder, CLS-pooled, query-only instruction prefix). The base
    // repo ships safetensors only; this community export carries both
    // the f32 `onnx/model.onnx` and the int8 `onnx/model_quantized.onnx`.
    // Apple Silicon / CPU stays on int8 (the ORT CoreML EP can't
    // accelerate NomicBert — int8 QDQ falls back to CPU and is *slower*
    // there, and MLProgram miscompiles its rotary); CUDA gets the f32
    // file where it is genuinely accelerated. See `OnnxFiles`.
    ModelSpec {
        name: "nomic-ai/CodeRankEmbed",
        dimensions: 768,
        code: 5,
        multilingual: 2,
        params_m: 137,
        query_prefix: Some("Represent this query for searching relevant code: "),
        doc_prefix: None,
        pooling: Pooling::Cls,
        hf_repo: Some("jalipalo/CodeRankEmbed-onnx"),
        // One repo, both files: f32 for CUDA (accelerated), int8 for
        // CPU/Apple-Silicon (fastest correct path there — ~0.7 s/10
        // batches vs the f32's CPU-equal CoreML run; MLProgram crashes
        // the rotary). `accel_active` is CUDA-only, so on a Mac this
        // resolves to int8/CPU automatically.
        onnx: OnnxFiles::AccelCpu {
            accel: "onnx/model.onnx",
            cpu: "onnx/model_quantized.onnx",
        },
        // Apple Silicon: run on the Metal GPU via candle. f16 export
        // of the base safetensors (candle reads the native dtype) —
        // half the weight bandwidth, embeddings ≈ f32, ~1.8x the int8
        // ONNX CPU fallback. The `candle-f16` variant tag busts the
        // index vs the old f32 repo.
        candle_repo: Some("sensiarion/CodeRankEmbed-f16"),
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        note: "DEFAULT: SOTA code retrieval, English, CLS (int8)",
    },
    // jina-v2-base-code's HF ONNX config is `JinaBertForMaskedLM`, but
    // the export exposes a poolable encoder output (fastembed mean-pools
    // this very repo) — so it loads via the ONNX encoder path pinned to
    // the int8 `model_quantized.onnx` (the f32 is ~2.5 GB). No prefixes.
    ModelSpec {
        name: "jinaai/jina-embeddings-v2-base-code",
        dimensions: 768,
        code: 5,
        multilingual: 2,
        params_m: 161,
        query_prefix: None,
        doc_prefix: None,
        pooling: Pooling::Mean,
        hf_repo: Some("jinaai/jina-embeddings-v2-base-code"),
        onnx: OnnxFiles::Single(Some("onnx/model_quantized.onnx")),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        note: "Pure code, 30 prog langs, English (int8)",
    },
    ModelSpec {
        name: "minishlab/potion-multilingual-128M",
        dimensions: 256,
        code: 3,
        multilingual: 5,
        // Model2Vec matrix params (vocab 500353 × 256 ≈ 128M).
        params_m: 128,
        query_prefix: None,
        doc_prefix: None,
        pooling: Pooling::Mean,
        hf_repo: Some("minishlab/potion-multilingual-128M"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::Static,
        rec_batch: 64,
        note: "Model2Vec static, multilingual incl. Russian, tiny+fast",
    },
    ModelSpec {
        name: "minishlab/potion-base-32M",
        dimensions: 512,
        code: 3,
        multilingual: 2,
        // vocab 63091 × 512 ≈ 32M.
        params_m: 32,
        query_prefix: None,
        doc_prefix: None,
        pooling: Pooling::Mean,
        hf_repo: Some("minishlab/potion-base-32M"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::Static,
        rec_batch: 64,
        note: "Model2Vec static, English, smallest+fastest, lowest RAM",
    },
    ModelSpec {
        name: "intfloat/multilingual-e5-small",
        dimensions: 384,
        code: 3,
        multilingual: 5,
        params_m: 118,
        query_prefix: Some(E5_QUERY),
        doc_prefix: Some(E5_PASSAGE),
        pooling: Pooling::Mean,
        hf_repo: Some("Xenova/multilingual-e5-small"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 8,
        note: "Multilingual incl. Russian, ONNX encoder (fp16/int8)",
    },
    ModelSpec {
        name: "intfloat/multilingual-e5-base",
        dimensions: 768,
        code: 3,
        multilingual: 5,
        params_m: 278,
        query_prefix: Some(E5_QUERY),
        doc_prefix: Some(E5_PASSAGE),
        pooling: Pooling::Mean,
        hf_repo: Some("Xenova/multilingual-e5-base"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        note: "Multilingual incl. Russian, stronger than -small (fp16/int8)",
    },
    ModelSpec {
        name: "intfloat/multilingual-e5-large",
        dimensions: 1024,
        code: 3,
        multilingual: 5,
        params_m: 560,
        query_prefix: Some(E5_QUERY),
        doc_prefix: Some(E5_PASSAGE),
        pooling: Pooling::Mean,
        hf_repo: Some("Xenova/multilingual-e5-large"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 2,
        note: "Strongest multilingual incl. Russian, heaviest (fp16/int8)",
    },
    // nomic-embed-text uses task-specific prefixes; without them
    // retrieval is degraded (the prior latent bug — it ran prefix-less).
    ModelSpec {
        name: "nomic-ai/nomic-embed-text-v1.5",
        dimensions: 768,
        code: 4,
        multilingual: 3,
        params_m: 137,
        query_prefix: Some("search_query: "),
        doc_prefix: Some("search_document: "),
        pooling: Pooling::Mean,
        // Official ONNX export (onnx/model{,_fp16,_int8,_q4}.onnx).
        hf_repo: Some("nomic-ai/nomic-embed-text-v1.5"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        note: "Fast, matryoshka dims, mostly English",
    },
    // Snowflake arctic-embed v2 (gte-multilingual-base): CLS-pooled, a
    // query-only `query: ` prompt, documents raw. `-l-v2.0` is the same
    // contract (add it identically if needed).
    ModelSpec {
        name: "Snowflake/snowflake-arctic-embed-m-v2.0",
        dimensions: 768,
        code: 3,
        multilingual: 5,
        params_m: 305,
        query_prefix: Some("query: "),
        doc_prefix: None,
        pooling: Pooling::Cls,
        hf_repo: Some("Snowflake/snowflake-arctic-embed-m-v2.0"),
        onnx: OnnxFiles::Single(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        note: "Multilingual incl. Russian + code, CLS (fp16/int8)",
    },
    // e5-base-v2 fine-tuned for code search. Its only HF artifact is a
    // raw `torch.onnx.export` (opset 11, ~1.2k dynamic-shape nodes,
    // f32-only — no quantized variant): the CoreML/CUDA partitioner
    // OOMs at 6+ GB on it, so it is pinned to the CPU provider (bounded
    // ~0.6 GB). e5 ⇒ needs the query:/passage: prefixes.
    ModelSpec {
        name: "jamie8johnson/e5-base-v2-code-search",
        dimensions: 768,
        code: 4,
        multilingual: 2,
        params_m: 110,
        query_prefix: Some(E5_QUERY),
        doc_prefix: Some(E5_PASSAGE),
        pooling: Pooling::Mean,
        hf_repo: Some("jamie8johnson/e5-base-v2-code-search"),
        onnx: OnnxFiles::CpuOnly(None),
        candle_repo: None,
        arch: ModelArch::OnnxEncoder,
        rec_batch: 8,
        note: "e5-base-v2 tuned for code search, English (f32, CPU-only)",
    },
    // TODO(candle-qwen3): `Qwen/Qwen3-Embedding-0.6B` — a Qwen3 *decoder*
    // embedder: `query_prefix` = "Instruct: Given a code search query,
    // retrieve relevant code that answers the query\nQuery:" (literal
    // \n, NO space after `Query:` — authored, tunable), `doc_prefix` =
    // None, `pooling = LastToken`. NOT registerable yet: every public
    // ONNX is a KV-cache decoder the ORT path rejects; enable as a
    // one-line entry once the candle `qwen3` backend (ModelArch) lands.
];

pub fn model_spec(name: &str) -> Option<&'static ModelSpec> {
    SUPPORTED_MODELS.iter().find(|m| m.name == name)
}

/// Canonicalize a Hugging Face repo reference to the bare `org/name`
/// id that `hf-hub` expects. Accepts a plain id, a full
/// `https://huggingface.co/org/name` URL, or one with a trailing
/// `/tree/<rev>` / `/blob/...` (what a user copies from the browser).
/// Single owner of this parsing — used by `models add` (stored clean)
/// and the embedder (defensive, so a pre-existing URL in config still
/// loads after upgrade).
pub fn normalize_hf_repo(input: &str) -> String {
    let s = input.trim();
    let s = s
        .strip_prefix("https://")
        .or_else(|| s.strip_prefix("http://"))
        .unwrap_or(s);
    let s = s.strip_prefix("www.").unwrap_or(s);
    let s = s
        .strip_prefix("huggingface.co/")
        .or_else(|| s.strip_prefix("hf.co/"))
        .unwrap_or(s);
    let s = s.trim_start_matches('@').trim_matches('/');
    let mut segs = Vec::with_capacity(2);
    for seg in s.split('/') {
        if matches!(seg, "tree" | "blob" | "resolve") {
            break;
        }
        if seg.is_empty() {
            continue;
        }
        segs.push(seg);
        if segs.len() == 2 {
            break;
        }
    }
    if segs.is_empty() {
        s.to_string()
    } else {
        segs.join("/")
    }
}
