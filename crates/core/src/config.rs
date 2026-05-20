use crate::error::{Error, Result};
use fastembed::EmbeddingModel;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

pub const DEFAULT_MODEL: &str = "sensiarion/CodeRankEmbed-f16";

/// Bumped whenever chunking logic OR the embedded-text shape changes
/// (AST kinds, line window, structured split, the code↔NL embed
/// header). Part of the index fingerprint so the change forces a
/// one-time full re-index. v2: chunk-enrichment header prepended to
/// the embedded text. v3: adjacent small AST nodes are merged up to
/// `max_chunk_bytes` (semble-style) instead of one chunk per leaf.
pub const CHUNKER_VERSION: u32 = 3;

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
    /// `onnx_file` (or `onnx/model.onnx`) + the four tokenizer files
    /// are pulled from it (cached). Mutually exclusive with `url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repo: Option<String>,
    /// Direct URL to a `.onnx` file. The four tokenizer files are
    /// fetched from the same directory (same URL, filename swapped).
    /// Mutually exclusive with `repo`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Prepended to a search query before embedding (`None`/absent =
    /// none). e.g. `"search_query: "` for nomic or a task instruction
    /// for CLS code models (CodeRankEmbed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_prefix: Option<String>,
    /// Prepended to an indexed chunk before embedding (`None`/absent =
    /// none — many code/CLS models prefix only the query side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_prefix: Option<String>,
    /// Encoder output pooling (`mean` default; `cls` / `last-token`).
    #[serde(default)]
    pub pooling: Pooling,
    /// Exact ONNX file to pull from the repo, e.g. `model_q4f16.onnx`
    /// or `onnx/model_q4.onnx`. Absent ⇒ `onnx/model.onnx` (with a
    /// flat `model.onnx` fallback). A bare name is also tried under
    /// `onnx/`. This filename is the sole weight selector.
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

/// Default cross-encoder for the optional re-rank stage:
/// `cross-encoder/ettin-reranker-68m-v1` — a sentence-transformers
/// CrossEncoder over the Ettin (ModernBERT-recipe) 68M encoder, at
/// only ~68M params the smallest strong code-capable reranker (vs
/// 278M `bge-reranker-base`). On Apple Silicon it runs **as-is**
/// (native f32 safetensors, no precision cast) on the **Metal GPU via
/// candle**: the bare `ModernBertModel` encoder + the model's
/// sentence-transformers head (CLS pool → Dense·GELU → LayerNorm →
/// Dense → one relevance logit, no softmax). Off Apple Silicon the
/// candle path is unavailable and the int8 ONNX export is encoder-only
/// (the ST head lives in separate modules), so re-rank is a no-op
/// there for now — search still returns the fused ranking unchanged.
/// A code-blind web-text cross-encoder (e.g. ms-marco MiniLM) badly
/// degrades code ranking — measured — so the default stays a
/// code-strong model.
pub const DEFAULT_RERANK_MODEL: &str = "cross-encoder/ettin-reranker-68m-v1";

/// Optional cross-encoder re-rank of the fused candidate neighborhood
/// before truncating to the result limit (selection precision is the
/// dominant lever once recall is adequate). It does NOT affect the
/// index fingerprint — re-rank changes ordering, not stored vectors,
/// so toggling it never triggers a rebuild.
///
/// `enabled` is **unspecified by default** (not written to a generated
/// config) and falls back per active model to
/// `ModelSpec::rerank_default`: ON for the fast static potion models
/// (re-rank is a large quality rescue for them — it lifts a static
/// retriever to ≈ a transformer's top-1, measured), OFF for the SOTA
/// CodeRank / jina bi-encoders (re-rank is ~neutral there and not
/// worth the extra model + latency). A custom/remote model with no
/// spec defaults OFF. An explicit `[rerank] enabled = true|false`
/// always wins. Resolve via `Config::rerank_enabled()`, never the
/// raw field.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RerankConfig {
    /// `None` ⇒ model-driven (see `ModelSpec::rerank_default`); `Some`
    /// is an explicit user override. Never read directly — go through
    /// `Config::rerank_enabled()`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// HF repo of the cross-encoder. The default
    /// (`DEFAULT_RERANK_MODEL`, ettin) runs via the candle backend on
    /// Apple Silicon; otherwise / for any other repo the **ONNX
    /// fallback** loads its int8 `onnx/model_quantized.onnx` (a
    /// `*ForSequenceClassification`-style single relevance logit).
    pub model: String,
    /// How many top fused candidates to re-score. Beyond this the
    /// cross-encoder's marginal gain no longer pays its linear cost.
    pub top_n: usize,
}

impl Default for RerankConfig {
    fn default() -> Self {
        Self {
            enabled: None,
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
    /// Max token sequence fed to the model. Caps the O(batch·seq²)
    /// attention tensor (a memory lever) AND the amount of each chunk
    /// the model actually sees — text past this many tokens is silently
    /// truncated before embedding. Keep it aligned with
    /// `sync.max_chunk_bytes` (≈4 bytes/token for code) so whole chunks
    /// are embedded, not just their head. 512 is the common BERT/jina
    /// native max.
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
    /// `onnx_path` model (`None` = none). nomic ⇒ `"search_query: "` /
    /// `"search_document: "`; CodeRankEmbed-style: a task instruction
    /// on the query side only.
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
    /// remote (`None` = none — OpenAI / DeepSeek). nomic ⇒
    /// `"search_query: "` / `"search_document: "`.
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

// (Removed `Precision`.) Model weights are no longer selected by a
// fp16/int8/full knob; a model is identified solely by its concrete
// `.onnx` filename — built-ins pin one in `ModelSpec::onnx`, a custom
// model uses its `onnx_file` (default `onnx/model.onnx`). The
// reranker pins `onnx/model_quantized.onnx` directly.

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

    /// Per-project override file: `<project>/.embedding-search/config.toml`.
    /// Deep-merged ON TOP of the global config (project keys win), so a
    /// repo can pin its own model without touching the global default
    /// or other projects.
    pub fn project_override_path(project_dir: &Path) -> PathBuf {
        project_dir.join(PROJECT_INDEX_DIR).join("config.toml")
    }

    /// Effective config for a project: the global `config.toml`
    /// (materialized on first run, as `load_or_init`) with the
    /// project's `.embedding-search/config.toml` deep-merged over it.
    /// Used by every project-scoped entry point (sync engine,
    /// inspector, MCP server) so a per-repo `set` is honored;
    /// `models list`/`set-default` stay global via `load_or_init`.
    pub fn load_for_project(project_dir: &Path) -> Result<Self> {
        let global_path = Self::config_path()?;
        if !global_path.exists() {
            // Preserve first-run UX: a default global file is written
            // so the user can see/edit it (best-effort, as load_or_init).
            if let Err(e) = Self::default().save() {
                tracing::warn!("could not write default config to {global_path:?}: {e}");
            }
        }
        let mut merged = read_toml_table(&global_path);
        // `read_toml_table` yields an empty table when the override is
        // absent (merging it is a no-op) — no exists() race/stat.
        merge_toml(&mut merged, read_toml_table(&Self::project_override_path(project_dir)));
        merged
            .try_into()
            .map_err(|e| Error::Config(format!("merge project config: {e}")))
    }

    /// Write a MINIMAL project override (just the model selection, plus
    /// the resolved `[remote]` block for a remote model) to
    /// `<project>/.embedding-search/config.toml`. Minimal — not a full
    /// snapshot — so unrelated global settings still flow through the
    /// merge after the user edits them. Returns the file path.
    pub fn save_project_override(&self, project_dir: &Path) -> Result<PathBuf> {
        let dir = project_dir.join(PROJECT_INDEX_DIR);
        std::fs::create_dir_all(&dir)?;
        let path = dir.join("config.toml");
        let mut model = toml::Table::new();
        model.insert(
            "default".into(),
            toml::Value::String(self.model.default.clone()),
        );
        let mut root = toml::Table::new();
        if self.model.provider == EmbeddingProvider::Openai {
            model.insert(
                "provider".into(),
                toml::Value::try_from(self.model.provider)
                    .map_err(|e| Error::Config(e.to_string()))?,
            );
            root.insert(
                "remote".into(),
                toml::Value::try_from(&self.remote)
                    .map_err(|e| Error::Config(e.to_string()))?,
            );
        }
        root.insert("model".into(), toml::Value::Table(model));
        let doc = toml::to_string_pretty(&toml::Value::Table(root))
            .map_err(|e| Error::Config(e.to_string()))?;
        std::fs::write(&path, doc)?;
        Ok(path)
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

    /// Whether the cross-encoder re-rank stage runs. An explicit
    /// `[rerank] enabled` wins; otherwise it is model-driven —
    /// `ModelSpec::rerank_default` for the active built-in (ON for the
    /// static potion models, OFF for the SOTA CodeRank/jina
    /// bi-encoders), and OFF for a custom/remote model with no spec.
    pub fn rerank_enabled(&self) -> bool {
        self.rerank.enabled.unwrap_or_else(|| {
            model_spec(&self.model.default).is_some_and(|s| s.rerank_default)
        })
    }

    /// Identity of everything that invalidates an existing index when
    /// changed: model (the resolved name carries the weight-file
    /// variant via `tagged_model_name`), output dims, token cap, chunk
    /// byte cap, chunker logic version, and the resolved input/output
    /// `contract` (query/doc prefix + pooling — changing a registered
    /// model's prefix or pooling must re-embed, model name alone won't
    /// shift). Stored in `meta.index_fingerprint`; a mismatch on
    /// startup wipes + rebuilds.
    pub fn index_fingerprint(&self, model_name: &str, dimensions: usize, contract: &str) -> String {
        format!(
            "v{}|{}|{}|{}|{}|{}",
            CHUNKER_VERSION,
            model_name,
            dimensions,
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

/// Parse a TOML file into a `Value::Table`, or an empty table if the
/// file is absent or unreadable/invalid (a broken project override
/// must not wedge the engine — it just contributes nothing to merge).
fn read_toml_table(path: &Path) -> toml::Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| toml::from_str::<toml::Value>(&s).ok())
        .filter(toml::Value::is_table)
        .unwrap_or_else(|| toml::Value::Table(toml::Table::new()))
}

/// Recursively merge `over` INTO `base` (project wins): tables are
/// merged key-by-key; any non-table value (scalar/array) replaces the
/// base wholesale. Single source for the global←project overlay.
fn merge_toml(base: &mut toml::Value, over: toml::Value) {
    match (base, over) {
        (toml::Value::Table(b), toml::Value::Table(o)) => {
            for (k, v) in o {
                match b.get_mut(&k) {
                    Some(slot) => merge_toml(slot, v),
                    None => {
                        b.insert(k, v);
                    }
                }
            }
        }
        (b, o) => *b = o,
    }
}

/// Auto-batch fallback for models with no static spec (custom ONNX /
/// remote): small enough that an unknown heavy model can't OOM.
pub const AUTO_EMBED_BATCH_FALLBACK: usize = 8;

/// File count above which a repo is "large" for the speed nudge: on a
/// heavy transformer model the first index is slow past roughly this
/// many files, where a static model is far faster for the same tree.
/// Single source for the CLI hint and the MCP `set_model` hint.
pub const LARGE_REPO_FILES: i64 = 1_500;

/// Sentence-vector pooling over the encoder's token states. Part of a
/// model's input/output contract (alongside the query/doc prefixes):
/// the wrong one silently produces unusable vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Pooling {
    /// Attention-masked mean of `last_hidden_state` — nomic, jina,
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

/// How a model is loaded and run. Detected from a model's
/// `config.json` for user-added models (`models add`); declared up
/// front for every built-in. The single switch the embedder dispatches
/// on, so adding an architecture is one enum arm + one detector branch.
#[derive(Debug, Clone)]
pub enum ModelArch {
    /// Model2Vec / `StaticModel`: a static `[vocab, dim]` token-
    /// embedding matrix, mean-pooled. No transformer/ONNX, tiny RAM.
    /// (`config.json`: `model_type == "model2vec"` or
    /// `architectures` ⊇ `StaticModel`.)
    Static,
    /// Transformer encoder loaded from an HF ONNX repo as a
    /// user-defined model: mean-pooled `last_hidden_state`, weights
    /// from the pinned `.onnx` file. (`config.json`: a
    /// BERT/RoBERTa/XLM-R/MPNet/Nomic-BERT… encoder — NOT an
    /// `*ForMaskedLM/CausalLM` head.)
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
    /// One export for every provider. `Some` pins an exact repo file;
    /// `None` ⇒ the default `onnx/model.onnx` (flat `model.onnx`
    /// fallback). Static models use `None` (no ONNX is loaded).
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
    /// `None` ⇒ the default `onnx/model.onnx`.
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
    /// when it changes). `None` for `Single`/`CpuOnly` — a fixed file,
    /// it never flips.
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
    /// nomic: `"search_query: "`; CodeRankEmbed: a task instruction.
    pub query_prefix: Option<&'static str>,
    /// Prepended to an indexed document chunk before embedding (`None`
    /// = none — many code/CLS models prefix only the query side).
    pub doc_prefix: Option<&'static str>,
    /// Sentence-vector pooling for this model's encoder output.
    pub pooling: Pooling,
    /// Source HF repo for `Static`/`OnnxEncoder` (Model2Vec safetensors
    /// resp. the pinned `onnx/model*.onnx` + tokenizer).
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
    /// Default for the optional cross-encoder re-rank when the user
    /// leaves `[rerank] enabled` unspecified. `true` for the fast
    /// static models (re-rank is a large measured quality rescue —
    /// lifts them to ≈ a transformer's top-1); `false` for the SOTA
    /// bi-encoders (re-rank is ~neutral there, not worth the latency).
    pub rerank_default: bool,
    pub note: &'static str,
}

impl ModelSpec {
    /// Model2Vec static-matrix backend (no transformer / ORT).
    pub fn is_static(&self) -> bool {
        matches!(self.arch, ModelArch::Static)
    }

    /// Rough resident RAM in MB: weights + a fixed working-set
    /// overhead. Model2Vec is a static f32 matrix with no ONNX Runtime
    /// (~30 MB tokenizer overhead, 4 B/param); an `OnnxEncoder`
    /// built-in ships its int8 export on CPU (~1 B/param) under ORT's
    /// ~350 MB. A coarse estimate for `models list` display only.
    pub fn ram_mb(&self) -> u32 {
        let (overhead, bytes_per_param) = if self.is_static() {
            (30.0, 4.0)
        } else {
            (350.0, 1.0)
        };
        (self.params_m as f32 * bytes_per_param + overhead) as u32
    }
}

pub const SUPPORTED_MODELS: &[ModelSpec] = &[
    // DEFAULT: CodeRankEmbed — SOTA code retrieval (NomicBert encoder,
    // CLS-pooled, query-only instruction prefix). Two selectable
    // builtins, identical EXCEPT the Apple-Silicon candle weights:
    //
    //  * `sensiarion/CodeRankEmbed-f16` (default) — a pure f16 cast of
    //    the official `nomic-ai/CodeRankEmbed` safetensors, validated
    //    equivalent (cosine 0.999998, identical CodeSearchNet
    //    MRR@10/Recall@1; see tools/quant) at ~half the RAM.
    //  * `nomic-ai/CodeRankEmbed` — the official upstream f32 weights
    //    (~2x RAM, same embeddings). `models set-default
    //    nomic-ai/CodeRankEmbed` to pick exact upstream provenance.
    //
    // Off Apple-Silicon both resolve to the SAME ONNX path
    // (`jalipalo/CodeRankEmbed-onnx`): int8 on CPU, f32 on CUDA (the
    // ORT CoreML EP can't accelerate NomicBert — int8 QDQ falls back
    // to CPU and is slower, MLProgram miscompiles the rotary). See
    // `OnnxFiles`.
    ModelSpec {
        name: "sensiarion/CodeRankEmbed-f16",
        dimensions: 768,
        code: 5,
        multilingual: 2,
        params_m: 137,
        query_prefix: Some("Represent this query for searching relevant code: "),
        doc_prefix: None,
        pooling: Pooling::Cls,
        hf_repo: Some("jalipalo/CodeRankEmbed-onnx"),
        onnx: OnnxFiles::AccelCpu {
            accel: "onnx/model.onnx",
            cpu: "onnx/model_quantized.onnx",
        },
        // candle reads the native safetensors dtype, so this f16 repo
        // runs half-precision on Metal; the `candle-f16` variant tag
        // keeps its index distinct from the f32 entry below.
        candle_repo: Some("sensiarion/CodeRankEmbed-f16"),
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        rerank_default: false,
        note: "DEFAULT: SOTA code retrieval, English, CLS (f16 Metal / int8 CPU)",
    },
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
        onnx: OnnxFiles::AccelCpu {
            accel: "onnx/model.onnx",
            cpu: "onnx/model_quantized.onnx",
        },
        // Official upstream f32 safetensors on Metal (~2x the f16
        // default's RAM, identical embeddings). `candle-f32` variant
        // tag → its own index.
        candle_repo: Some("nomic-ai/CodeRankEmbed"),
        arch: ModelArch::OnnxEncoder,
        rec_batch: 4,
        rerank_default: false,
        note: "Official upstream f32 weights (Metal); ~2x f16 default RAM",
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
        rerank_default: false,
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
        rerank_default: true,
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
        rerank_default: true,
        note: "Model2Vec static, English, smallest+fastest, lowest RAM",
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_toml_overlays_project_keys_and_keeps_the_rest() {
        let mut base: toml::Value = toml::from_str(
            "[model]\ndefault = \"global-model\"\nmax_length = 333\n\
             [sync]\nmax_chunk_bytes = 2048\n",
        )
        .unwrap();
        // Project overrides ONLY the model default; the other [model]
        // keys + [sync] survive.
        let over: toml::Value = toml::from_str("[model]\ndefault = \"proj-model\"\n").unwrap();
        merge_toml(&mut base, over);
        let cfg: Config = base.try_into().unwrap();
        assert_eq!(cfg.model.default, "proj-model"); // project wins
        assert_eq!(cfg.model.max_length, 333); // global [model] key kept
        assert_eq!(cfg.sync.max_chunk_bytes, 2048); // untouched section kept
    }

    #[test]
    fn save_project_override_writes_minimal_local_model_only() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = Config::default();
        cfg.model.default = "minishlab/potion-base-32M".into();
        let p = cfg.save_project_override(dir.path()).unwrap();
        assert_eq!(p, Config::project_override_path(dir.path()));
        let written = std::fs::read_to_string(&p).unwrap();
        assert!(written.contains("minishlab/potion-base-32M"));
        // Minimal: only the local model — no remote block, no full
        // config dump ([sync] etc.).
        assert!(!written.contains("[remote]"));
        assert!(!written.contains("[sync]"));
    }
}
