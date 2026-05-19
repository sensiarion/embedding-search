use crate::config::{Config, Pooling};
use crate::error::{Error, Result};
use fastembed::TextEmbedding;
use std::sync::Mutex;
#[cfg(not(feature = "bench-stub"))]
use {
    crate::config::{
        BackendConfig, EmbeddingProvider, ExecutionProvider, ModelArch, Precision, RemoteConfig,
    },
    fastembed::{InitOptions, TokenizerFiles, UserDefinedEmbeddingModel},
    ort::{
        session::{builder::GraphOptimizationLevel, Session},
        value::Tensor,
    },
    rayon::prelude::*,
    safetensors::SafeTensors,
    serde::Deserialize,
    std::path::Path,
    std::thread::available_parallelism,
    std::time::Duration,
    tokenizers::{Encoding, Tokenizer, TruncationParams},
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
    /// Transformer ONNX (e5 / jina-code / nomic / any user-defined
    /// encoder) run directly on ORT with bounded fixed-shape batches.
    #[cfg(not(feature = "bench-stub"))]
    Onnx(OnnxEncoder),
    /// CodeRankEmbed (NomicBert) on the Metal GPU via candle —
    /// Apple-Silicon only, ~1.8x the int8 ONNX CPU path.
    #[cfg(candle_backend)]
    Candle(Box<crate::candle_encoder::CandleEncoder>),
}

/// Resolved per-model input/output contract: the query/document
/// prefixes and the pooling. Built once at construction from a built-in
/// `ModelSpec`, a registered `CustomModel`, or an e5 bool (remote /
/// local-onnx-path), then carried on the `Embedder`.
#[derive(Clone)]
pub struct Contract {
    query_prefix: Option<String>,
    doc_prefix: Option<String>,
    pooling: Pooling,
}

impl Contract {
    /// Just a query/doc prefix pair, mean pooling — the remote and
    /// local-`onnx_path` paths (no per-model pooling knob there; e5 is
    /// simply `query: `/`passage: `, nomic `search_query: `/…).
    #[cfg(not(feature = "bench-stub"))]
    fn from_prefixes(query_prefix: Option<String>, doc_prefix: Option<String>) -> Self {
        Self {
            query_prefix,
            doc_prefix,
            pooling: Pooling::Mean,
        }
    }

    /// From a built-in registry entry (its `&'static str` prefixes).
    fn from_spec(s: &crate::config::ModelSpec) -> Self {
        Self {
            query_prefix: s.query_prefix.map(str::to_owned),
            doc_prefix: s.doc_prefix.map(str::to_owned),
            pooling: s.pooling,
        }
    }

    /// From a registered `[[custom_model]]` entry.
    #[cfg(not(feature = "bench-stub"))]
    fn from_custom(cm: &crate::config::CustomModel) -> Self {
        Self {
            query_prefix: cm.query_prefix.clone(),
            doc_prefix: cm.doc_prefix.clone(),
            pooling: cm.pooling,
        }
    }

    /// Stable identity fragment for the index fingerprint — changing a
    /// model's prefix or pooling must re-embed even if its name is
    /// unchanged.
    fn tag(&self) -> String {
        format!(
            "{}|{}|{}",
            self.query_prefix.as_deref().unwrap_or(""),
            self.doc_prefix.as_deref().unwrap_or(""),
            self.pooling.label()
        )
    }
}

pub struct Embedder {
    backend: Backend,
    pub dimensions: usize,
    pub model_name: String,
    /// Resolved per-model query/doc prefixes + pooling.
    contract: Contract,
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

/// Whether an accelerator that *actually speeds up* our ONNX encoders
/// is active for this backend — which empirically means **CUDA only**.
///
/// Measured on the CodeRankEmbed (NomicBert) export, batch `[4,64]`,
/// 10 runs: every quantized variant is *slower* on the ORT CoreML EP
/// than on CPU (it cannot run int8 QDQ, falls back to CPU and adds
/// copy/partition overhead — ~0.95 s vs ~0.72 s), and the f32 export
/// is only CPU-equal; the newer MLProgram format miscompiles the
/// rotary `Mul` and hard-fails. So the f32↔int8 `OnnxFiles` flip is
/// pure downside on Apple Silicon (4× the download for no speedup) and
/// only worth it under CUDA. (CoreML EP is still attached by
/// `providers()` for non-pinned models — it just never wins here.)
#[cfg(not(feature = "bench-stub"))]
pub(crate) fn accel_active(b: &BackendConfig) -> bool {
    cfg!(feature = "cuda")
        && matches!(
            b.execution_provider,
            ExecutionProvider::Cuda | ExecutionProvider::Auto
        )
}

/// CoreML EP for our fixed-shape encoder. NOTE: the `NeuralNetwork`
/// format is deliberate — `MLProgram` "supports more ops" in general
/// but **miscompiles NomicBert rotary** (`rotary_emb/Mul` broadcast
/// hard-fails, even at batch 1; verified on every CodeRankEmbed
/// export). `NeuralNetwork` runs every model correctly. Static input
/// shapes (we already pad to fixed `[batch, seq]` buckets), all
/// compute units, latency-first specialization, and a persistent
/// compiled-model cache so a process restart does not recompile — that
/// recompile is the slow startup + macOS "Context leak detected"
/// os_log spam the CPU pin used to avoid.
#[cfg(not(feature = "bench-stub"))]
fn coreml_ep(cache_dir: &Path) -> fastembed::ExecutionProviderDispatch {
    use ort::ep::coreml::{ComputeUnits, ModelFormat, SpecializationStrategy};
    let mlcache = cache_dir.join("coreml");
    // Best-effort: ORT writes the compiled model here; pre-create so the
    // first run already caches (a failure just means no cache reuse).
    let _ = std::fs::create_dir_all(&mlcache);
    ort::ep::CoreML::default()
        .with_model_format(ModelFormat::NeuralNetwork)
        .with_static_input_shapes(true)
        .with_compute_units(ComputeUnits::All)
        .with_specialization_strategy(SpecializationStrategy::FastPrediction)
        .with_model_cache_dir(mlcache.to_string_lossy())
        .build()
}

#[cfg(not(feature = "bench-stub"))]
pub(crate) fn providers(
    b: &BackendConfig,
    force_cpu: bool,
    cache_dir: &Path,
) -> Vec<fastembed::ExecutionProviderDispatch> {
    use ort::ep::CPU;
    let mac_arm = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    let mut v: Vec<fastembed::ExecutionProviderDispatch> = Vec::new();

    // A model whose ONNX export fragments the CoreML/CUDA partitioner
    // pins CPU regardless of the backend setting (resolved per-model
    // from `OnnxFiles` — see `ModelSpec::onnx`).
    let ep = if force_cpu {
        ExecutionProvider::Cpu
    } else {
        b.execution_provider
    };
    match ep {
        ExecutionProvider::Coreml => {
            if mac_arm {
                v.push(coreml_ep(cache_dir));
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
                v.push(coreml_ep(cache_dir));
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
        // hf-hub defaults to 0 retries: a single transient HTTP
        // timeout (slow link, big model/tokenizer file) then aborts
        // the whole load. Retry with backoff instead.
        .with_retries(4)
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
    // fastembed's own pooling (only used if we ever route through its
    // inference); our `OnnxEncoder` pools itself per `config::Pooling`.
    let mut m = UserDefinedEmbeddingModel::new(onnx, tok).with_pooling(fastembed::Pooling::Mean);
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

    // Architectures whose checkpoint is exported as an LM head but
    // whose ONNX still exposes a poolable encoder output usable as a
    // sentence embedder (jina v2 — fastembed mean-pools this exact
    // repo). Allowed despite the `*ForMaskedLM` name.
    const EMBED_OK_LM: &[&str] = &["JinaBertForMaskedLM"];
    let bad = |n: &str| {
        !EMBED_OK_LM.contains(&n)
            && (n.ends_with("ForMaskedLM")
                || n.ends_with("ForCausalLM")
                || n.ends_with("ForPreTraining")
                || n.contains("LMHead"))
    };
    if !names.is_empty() && names.iter().all(|n| bad(n)) {
        // This exact repo may already be a built-in served a working
        // way (e.g. jina-code via the fastembed bundle) — point there
        // instead of leaving the user to re-discover it.
        let hint = crate::config::model_spec(repo).map_or_else(
            || {
                "Use a sentence-embedding ONNX export, a built-in \
                 (`models list`), or a remote endpoint (`models add-remote`)."
                    .to_string()
            },
            |s| {
                format!(
                    "'{0}' is already a built-in served correctly — just \
                     `embedding-search models set-default {0}` (do NOT \
                     `models add` the raw repo).",
                    s.name
                )
            },
        );
        return Err(Error::Embed(format!(
            "HF repo '{repo}': its ONNX is a language-model head ({}), not a \
             sentence-embedding model — pooling its vocab logits is not an \
             embedding and these exports (ALiBi/long-context) consume tens \
             of GB of RAM. {hint}",
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

/// A with-KV-cache decoder ONNX graph names its inputs
/// `past_key_values.<n>.{key,value}`, present as plain protobuf
/// strings in the (small) graph file — so this is detectable before
/// the multi-GB `.onnx_data` download. Such exports need
/// `past_key_values` inputs the encoder embedding feed never supplies.
#[cfg(not(feature = "bench-stub"))]
fn is_kv_cache_decoder(onnx: &[u8]) -> bool {
    const M: &[u8] = b"past_key_values";
    onnx.windows(M.len()).any(|w| w == M)
}

/// Try to build the Apple-Silicon Metal (candle) backend for a
/// `candle_repo` model. Returns `None` (and logs why) on any failure —
/// no Metal device (headless/CI), download error, bad weights — so the
/// caller transparently falls back to the ONNX path.
#[cfg(candle_backend)]
fn try_candle(
    spec: &crate::config::ModelSpec,
    cfg: &Config,
    cache_dir: &Path,
) -> Option<crate::candle_encoder::CandleEncoder> {
    let repo = spec.candle_repo?;
    let build = || -> Result<crate::candle_encoder::CandleEncoder> {
        let (_repo, r) = hf_repo(repo, cache_dir)?;
        let get = |f: &str| -> Result<std::path::PathBuf> {
            r.get(f)
                .map_err(|e| Error::Embed(format!("hf-hub get {f}: {e}")))
        };
        let safetensors = get("model.safetensors")?;
        let config_json = read(&get("config.json")?)?;
        let tokenizer_json = read(&get("tokenizer.json")?)?;
        crate::candle_encoder::CandleEncoder::build(
            &safetensors,
            &config_json,
            &tokenizer_json,
            cfg.model.max_length.max(1),
        )
    };
    match build() {
        Ok(enc) => Some(enc),
        Err(e) => {
            tracing::warn!("candle Metal backend unavailable ({e}); using ONNX fallback");
            None
        }
    }
}

/// Download (cached) the chosen ONNX + tokenizer files from the HF
/// repo. `explicit` pins an exact file; else the precision mapping.
#[cfg(not(feature = "bench-stub"))]
pub(crate) fn load_user_defined(
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
    // Decoder/KV-cache export guard. onnx-community exports
    // decoder-LLM embedders (Qwen3-Embedding, etc.) as a with-past
    // graph that requires `past_key_values.*` inputs the encoder
    // embedding pipeline never supplies → a cryptic ORT "Missing
    // Input: past_key_values.0.value" at the probe. The graph `.onnx`
    // (here, before the multi-GB `.onnx_data`) carries those input
    // names as plain protobuf strings — reject up front instead.
    if is_kv_cache_decoder(&onnx) {
        return Err(Error::Embed(format!(
            "HF repo '{repo}': '{model_file}' is a decoder export with \
             KV-cache inputs (past_key_values) — it cannot run as a \
             sentence embedder via this pipeline. Use an encoder/embedding \
             ONNX export, a built-in (`models list`), or a remote \
             OpenAI-compatible endpoint (`models add-remote`)."
        )));
    }

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

/// Sequence-length buckets (ascending, last == `max_length`). Every
/// batch is padded up to the smallest bucket that fits its longest
/// sequence, so an ONNX session sees at most this many distinct input
/// shapes. fastembed's own tokenizer pads to the batch's longest
/// sequence (`BatchLongest`) — with small batches that is a *new*
/// `[batch, seq]` shape almost every call, and the CoreML execution
/// provider recompiles + retains the whole graph per shape (the
/// multi-GB, erratic-RSS blowup on real repos). Few fixed shapes ⇒
/// CoreML compiles a handful of graphs once; peak memory is bounded by
/// the largest bucket, not the corpus.
#[cfg(not(feature = "bench-stub"))]
pub(crate) fn seq_buckets(max_length: usize) -> Vec<usize> {
    let mut b: Vec<usize> = [64, 128, 256, 512]
        .into_iter()
        .filter(|&x| x < max_length)
        .collect();
    b.push(max_length.max(1));
    b
}

/// Build an ORT session + its tokenizer from already-fetched model
/// bytes, shared by `OnnxEncoder` (embedding) and `Reranker`
/// (cross-encoder). Returns `(session, tokenizer, pad_id,
/// need_type_ids)`. The tokenizer truncates to `max_length` and does
/// NOT pad — callers pad manually to a fixed bucket so the EP only ever
/// sees a tiny bounded set of input shapes (the memory-bounding fix).
/// Download / LM-head / KV-cache guards already ran in
/// `load_user_defined`.
/// The one owner of the `name@variant` identity convention. Folded
/// into the index fingerprint, so every backend that changes the
/// weights (ONNX f32↔int8 via `variant_tag`, the candle Metal path)
/// must route its tag through here — ad-hoc `format!` risks a stale
/// index silently surviving a backend switch.
#[cfg(not(feature = "bench-stub"))]
fn tagged_model_name(name: &str, variant: Option<&str>) -> String {
    match variant {
        Some(v) => format!("{name}@{v}"),
        None => name.to_string(),
    }
}

/// HF tokenizer truncated to `max_length`, padding off — we always pad
/// manually (fixed ONNX buckets / candle batch-longest) so the
/// tokenizer must not. Shared by the ONNX and candle backends.
#[cfg(not(feature = "bench-stub"))]
pub(crate) fn load_tokenizer(bytes: &[u8], max_length: usize) -> Result<Tokenizer> {
    let mut tok =
        Tokenizer::from_bytes(bytes).map_err(|e| Error::Embed(format!("tokenizer: {e}")))?;
    tok.with_truncation(Some(TruncationParams {
        max_length: max_length.max(1),
        ..Default::default()
    }))
    .map_err(|e| Error::Embed(format!("tokenizer truncation: {e}")))?;
    tok.with_padding(None);
    Ok(tok)
}

#[cfg(not(feature = "bench-stub"))]
pub(crate) fn build_onnx_session(
    udm: UserDefinedEmbeddingModel,
    backend: &BackendConfig,
    force_cpu: bool,
    max_length: usize,
    cache_dir: &Path,
) -> Result<(Session, Tokenizer, i64, bool)> {
    // pad_token_id lives in config.json (default 0, the BERT/e5/jina
    // convention) — pure padding, masked out anyway.
    let pad_id = serde_json::from_slice::<serde_json::Value>(&udm.tokenizer_files.config_file)
        .ok()
        .and_then(|v| v.get("pad_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as i64;

    let tok = load_tokenizer(&udm.tokenizer_files.tokenizer_file, max_length)?;

    let threads = available_parallelism().map_err(Error::Io)?.get();
    let mut b = Session::builder()
        .map_err(|e| Error::Embed(e.to_string()))?
        .with_execution_providers(providers(backend, force_cpu, cache_dir))
        .map_err(|e| Error::Embed(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| Error::Embed(e.to_string()))?
        .with_intra_threads(threads)
        .map_err(|e| Error::Embed(e.to_string()))?
        // Off: the memory-pattern planner caches a per-shape allocation
        // plan; with even a few buckets that is wasted retention. Our
        // shapes are already bounded by `buckets`.
        .with_memory_pattern(false)
        .map_err(|e| Error::Embed(e.to_string()))?;
    for ext in udm.external_initializers {
        b = b
            .with_external_initializer_file_in_memory(ext.file_name, ext.buffer.into())
            .map_err(|e| Error::Embed(e.to_string()))?;
    }
    let session = b
        .commit_from_memory(&udm.onnx_file)
        .map_err(|e| Error::Embed(e.to_string()))?;
    let need_type_ids = session
        .inputs()
        .iter()
        .any(|i| i.name() == "token_type_ids");
    Ok((session, tok, pad_id, need_type_ids))
}

/// Pack a tokenized batch into the FIXED shape `[batch, seq]`: pick the
/// smallest `buckets` entry that fits the longest sequence, then fill
/// `≤ batch` real rows over `batch * seq` pad-filled `(ids, mask)`
/// buffers. Pinning BOTH axes means the shape-caching CoreML/CUDA EP
/// compiles `buckets.len()` graphs total, not one per (partial-batch ×
/// bucket). Shared by the embedder and reranker run paths.
#[cfg(not(feature = "bench-stub"))]
pub(crate) fn fill_fixed_batch(
    encs: &[Encoding],
    pad_id: i64,
    batch: usize,
    buckets: &[usize],
) -> (Vec<i64>, Vec<i64>, usize) {
    let longest = encs.iter().map(|e| e.get_ids().len()).max().unwrap_or(1);
    let seq = buckets
        .iter()
        .copied()
        .find(|&b| b >= longest)
        .unwrap_or_else(|| *buckets.last().unwrap())
        .max(1);
    let mut ids = vec![pad_id; batch * seq];
    let mut mask = vec![0i64; batch * seq];
    for (row, e) in encs.iter().enumerate() {
        let off = row * seq;
        let n = e.get_ids().len().min(seq);
        for (j, (&id, &m)) in e
            .get_ids()
            .iter()
            .zip(e.get_attention_mask())
            .take(n)
            .enumerate()
        {
            ids[off + j] = id as i64;
            mask[off + j] = m as i64;
        }
    }
    (ids, mask, seq)
}

/// Direct ONNX-Runtime sentence encoder. Bypasses fastembed's inference
/// (which hard-codes `BatchLongest` padding) so we control the input
/// shape: tokenize → pad to a fixed `[batch, seq]` bucket → one ORT run
/// → attention-masked mean pool → L2 normalize. Bounded, predictable
/// memory regardless of corpus/chunk-length distribution.
#[cfg(not(feature = "bench-stub"))]
struct OnnxEncoder {
    session: Mutex<Session>,
    tok: Tokenizer,
    pad_id: i64,
    need_type_ids: bool,
    buckets: Vec<usize>,
    /// Fixed batch width every ORT run is padded up to. `seq_buckets`
    /// already pins the sequence axis; this pins the *batch* axis so
    /// the CoreML/CUDA EP — which compiles and *retains* one model per
    /// distinct input shape — sees at most `buckets.len()` shapes total
    /// instead of one per (partial-batch × bucket) combination (the
    /// multi-GB blowup on a fragmented raw `torch.onnx.export`).
    batch: usize,
    /// How to reduce the rank-3 token states to one vector per row. A
    /// ready rank-2 sentence/pooler output (sentence-transformers
    /// export) is always used as-is and bypasses this.
    pooling: Pooling,
    dim: usize,
}

#[cfg(not(feature = "bench-stub"))]
impl OnnxEncoder {
    /// Build the encoder from already-fetched model bytes. `cfg` drives
    /// max sequence length and execution providers (download / LM-head /
    /// KV-cache guards already ran in `load_user_defined`).
    fn build(
        udm: UserDefinedEmbeddingModel,
        cfg: &Config,
        force_cpu: bool,
        pooling: Pooling,
    ) -> Result<Self> {
        let max_length = cfg.model.max_length.max(1);
        let cache_dir = cfg.model_cache_dir()?;
        let (session, tok, pad_id, need_type_ids) =
            build_onnx_session(udm, &cfg.backend, force_cpu, max_length, &cache_dir)?;

        let mut enc = Self {
            session: Mutex::new(session),
            tok,
            pad_id,
            need_type_ids,
            buckets: seq_buckets(max_length),
            // The resolved per-model embed batch (auto = the model's
            // `rec_batch`). Every run is padded to exactly this width.
            batch: cfg.embed_batch().max(1),
            pooling,
            dim: 0,
        };
        // Probe the output width (user-defined ONNX has no static spec).
        enc.dim = enc
            .embed(&["probe"])?
            .first()
            .map(Vec::len)
            .ok_or_else(|| Error::Embed("ONNX encoder: empty probe".into()))?;
        Ok(enc)
    }

    /// Embed every text, one fixed-shape ORT call per `self.batch`
    /// slice. Splitting here (not in the caller) guarantees the only
    /// batch width ORT — and the shape-caching CoreML/CUDA EP — ever
    /// sees is `self.batch`.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch) {
            out.extend(self.run_batch(chunk)?);
        }
        Ok(out)
    }

    /// One ORT run on the FIXED shape `[self.batch, bucket]`: the `≤
    /// batch` real rows plus pure-pad filler rows (pad-id ids, zero
    /// mask) so the EP only ever compiles `buckets.len()` graphs.
    /// Tokenize → masked mean-pool → L2-normalize; the filler rows'
    /// outputs are discarded (transformer rows are independent, so they
    /// cannot perturb the real ones).
    fn run_batch(&self, chunk: &[&str]) -> Result<Vec<Vec<f32>>> {
        let real = chunk.len();
        let encs = self
            .tok
            .encode_batch(chunk.to_vec(), true)
            .map_err(|e| Error::Embed(format!("tokenize batch: {e}")))?;
        // Pinned to `[self.batch, bucket]` (filler rows pad the batch
        // axis) so the EP only ever compiles `buckets.len()` graphs.
        let bsz = self.batch;
        let (ids, mask, seq) = fill_fixed_batch(&encs, self.pad_id, bsz, &self.buckets);

        let shape = vec![bsz as i64, seq as i64];
        let mut inputs = ort::inputs![
            "input_ids" => Tensor::from_array((shape.clone(), ids))
                .map_err(|e| Error::Embed(e.to_string()))?,
            "attention_mask" => Tensor::from_array((shape.clone(), mask.clone()))
                .map_err(|e| Error::Embed(e.to_string()))?,
        ];
        if self.need_type_ids {
            inputs.push((
                "token_type_ids".into(),
                Tensor::from_array((shape, vec![0i64; bsz * seq]))
                    .map_err(|e| Error::Embed(e.to_string()))?
                    .into(),
            ));
        }

        // Prefer a ready rank-2 sentence/pooler output; else mean-pool a
        // rank-3 token-state output ([batch, seq, hidden]) with the
        // attention mask. Copy out inside the scope so the session
        // borrow (and the ORT run arena) is released before pooling.
        let (rank2, rank3) = {
            let mut sess = self
                .session
                .lock()
                .map_err(|_| Error::Embed("ONNX session lock poisoned".into()))?;
            let outputs = sess.run(inputs).map_err(|e| Error::Embed(e.to_string()))?;
            let mut rank2: Option<(usize, Vec<f32>)> = None;
            let mut rank3: Option<(usize, usize, Vec<f32>)> = None;
            for (name, val) in outputs.iter() {
                let Ok((sh, data)) = val.try_extract_tensor::<f32>() else {
                    continue;
                };
                match sh.len() {
                    2 if rank2.is_none()
                        || name.contains("sentence")
                        || name.contains("pooler") =>
                    {
                        rank2 = Some((sh[1] as usize, data.to_vec()));
                    }
                    3 if rank3.is_none() || name.contains("hidden") => {
                        rank3 = Some((sh[1] as usize, sh[2] as usize, data.to_vec()));
                    }
                    _ => {}
                }
            }
            (rank2, rank3)
        };

        // Only the first `real` rows are caller data; the rest is
        // filler that pinned the batch axis — drop it.
        let mut out = Vec::with_capacity(real);
        if let Some((hidden, data)) = rank2 {
            for r in 0..real {
                out.push(l2_normalized(&data[r * hidden..(r + 1) * hidden]));
            }
        } else if let Some((s, hidden, data)) = rank3 {
            let t = Rank3 {
                data: &data,
                s,
                hidden,
                seq,
                mask: &mask,
            };
            for r in 0..real {
                out.push(pool_rank3(self.pooling, &t, r));
            }
        } else {
            return Err(Error::Embed(
                "ONNX encoder: model produced no float tensor output".into(),
            ));
        }
        Ok(out)
    }
}

/// A rank-3 `[batch, s, hidden]` token-state output plus the padded
/// bucket width `seq` its `mask` is indexed by (`s` is the model's
/// actual output length). Borrowed view — pooled one row at a time.
#[cfg(not(feature = "bench-stub"))]
struct Rank3<'a> {
    data: &'a [f32],
    s: usize,
    hidden: usize,
    seq: usize,
    mask: &'a [i64],
}

/// Reduce row `r` of `t` to one L2-normalized vector per `pooling`.
#[cfg(not(feature = "bench-stub"))]
fn pool_rank3(pooling: Pooling, t: &Rank3, r: usize) -> Vec<f32> {
    let Rank3 {
        data,
        s,
        hidden,
        seq,
        mask,
    } = *t;
    let row = |i: usize| {
        let base = (r * s + i) * hidden;
        &data[base..base + hidden]
    };
    let valid = s.min(seq);
    match pooling {
        // [CLS] is always the first (unmasked) token.
        Pooling::Cls => l2_normalized(row(0)),
        // Last non-padding token (right-padded → highest t with
        // mask==1); an all-pad row falls back to token 0.
        Pooling::LastToken => {
            let last = (0..valid)
                .rev()
                .find(|&t| mask[r * seq + t] != 0)
                .unwrap_or(0);
            l2_normalized(row(last))
        }
        // Attention-masked mean over the real tokens.
        Pooling::Mean => {
            let mut acc = vec![0f32; hidden];
            let mut cnt = 0f32;
            for t in 0..valid {
                if mask[r * seq + t] == 0 {
                    continue;
                }
                for (a, &x) in acc.iter_mut().zip(row(t)) {
                    *a += x;
                }
                cnt += 1.0;
            }
            if cnt > 0.0 {
                for a in &mut acc {
                    *a /= cnt;
                }
            }
            l2_normalized(&acc)
        }
    }
}

/// L2-normalize (e5 / jina-code / nomic embeddings are unit-norm; this
/// matches sentence-transformers and keeps cosine == dot).
#[cfg(not(feature = "bench-stub"))]
fn l2_normalized(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
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
            contract: Contract::from_spec(spec),
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
            return Self::finish_user_defined(
                udm,
                cfg,
                name,
                Contract::from_prefixes(
                    cfg.model.onnx_query_prefix.clone(),
                    cfg.model.onnx_doc_prefix.clone(),
                ),
                false,
            );
        }

        // Registered custom model (HF repo id or direct .onnx URL),
        // downloaded/cached like a built-in.
        if let Some(cm) = cfg.custom_model() {
            let (udm, name) = match (&cm.repo, &cm.url) {
                (Some(repo), None) => {
                    // Model2Vec → static backend; else the ONNX path.
                    if let Some(sm) = try_model2vec(repo, &cache_dir)? {
                        let name = format!("model2vec:{}", crate::config::normalize_hf_repo(repo));
                        return Ok(Self::from_static(sm, name, Contract::from_custom(cm)));
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
            // A registered custom model has no per-model CPU pin: a raw
            // export that OOMs CoreML/CUDA is better promoted to a
            // built-in (`force_cpu`) or run with `[backend]
            // execution_provider = "cpu"`.
            return Self::finish_user_defined(udm, cfg, name, Contract::from_custom(cm), false);
        }

        let spec = cfg.model_spec()?;
        let repo = || {
            spec.hf_repo
                .ok_or_else(|| Error::Embed(format!("built-in model {} has no hf_repo", spec.name)))
        };

        // Dispatch on the declared architecture (the one switch).
        let model = match &spec.arch {
            ModelArch::Static => {
                // arch already says Model2Vec — `try_model2vec`'s None
                // (config.json not Model2Vec) can only mean a mis-
                // curated registry entry, surfaced as a clear error.
                let sm = try_model2vec(repo()?, &cache_dir)?.ok_or_else(|| {
                    Error::Embed(format!("built-in {}: repo is not Model2Vec", spec.name))
                })?;
                return Ok(Self::from_static(
                    sm,
                    spec.name.to_string(),
                    Contract::from_spec(spec),
                ));
            }
            ModelArch::OnnxEncoder => {
                // Apple Silicon: a `candle_repo` model runs f32 on the
                // Metal GPU via candle (~1.8x the int8 ONNX CPU path;
                // the ORT CoreML EP can't accelerate it). Falls through
                // to ONNX if Metal is unreachable (headless/CI) — the
                // identity carries `@candle` so switching backend (f32
                // vs int8 = different vectors) busts the index.
                #[cfg(candle_backend)]
                if let Some(enc) = try_candle(spec, cfg, &cache_dir) {
                    return Ok(Self {
                        dimensions: enc.dim,
                        backend: Backend::Candle(Box::new(enc)),
                        model_name: tagged_model_name(spec.name, Some("candle")),
                        contract: Contract::from_spec(spec),
                    });
                }
                // `accel_active` (CUDA-only — CoreML never wins for
                // these, see its doc) flips an `AccelCpu` model to its
                // f32 file; otherwise the int8 file is pinned to CPU.
                let accel = accel_active(&cfg.backend);
                let (file, pin_cpu) = spec.onnx.resolve(accel);
                let precision = spec.effective_precision(cfg.model.precision);
                let udm = load_user_defined(repo()?, precision, &cache_dir, file)?;
                let enc = OnnxEncoder::build(udm, cfg, pin_cpu, spec.pooling)?;
                return Ok(Self {
                    dimensions: enc.dim,
                    backend: Backend::Onnx(enc),
                    // f32↔int8 are different weights; the tag busts the
                    // index when the resolved file flips.
                    model_name: tagged_model_name(spec.name, spec.onnx.variant_tag(accel)),
                    contract: Contract::from_spec(spec),
                });
            }
            ModelArch::Fastembed(m) => {
                // The only public HF export for these is an LM head;
                // fastembed bundles a proper embedding ONNX + pooling.
                let eps = providers(&cfg.backend, false, &cache_dir);
                let init = InitOptions::new(m.clone())
                    .with_cache_dir(cache_dir)
                    .with_show_download_progress(true)
                    .with_max_length(cfg.model.max_length)
                    .with_execution_providers(eps);
                TextEmbedding::try_new(init).map_err(|e| Error::Embed(e.to_string()))?
            }
        };

        Ok(Self {
            backend: Backend::Local(Some(Mutex::new(model))),
            dimensions: spec.dimensions,
            model_name: spec.name.to_string(),
            contract: Contract::from_spec(spec),
        })
    }

    /// Wrap a loaded Model2Vec matrix as a static-backend embedder.
    /// Shared by the declared-Static built-in and the auto-detected
    /// custom Model2Vec repo (they differ only in name / e5 prefix).
    #[cfg(not(feature = "bench-stub"))]
    fn from_static(sm: StaticModel2Vec, model_name: String, contract: Contract) -> Self {
        Self {
            dimensions: sm.dim,
            model_name,
            contract,
            backend: Backend::Static(sm),
        }
    }

    /// Build a user-defined ONNX encoder (output dimensions probed at
    /// build time). Shared by the `onnx_path` and registered
    /// custom-model (HF repo / direct-URL) paths.
    #[cfg(not(feature = "bench-stub"))]
    fn finish_user_defined(
        udm: UserDefinedEmbeddingModel,
        cfg: &Config,
        name: String,
        contract: Contract,
        force_cpu: bool,
    ) -> Result<Self> {
        let enc = OnnxEncoder::build(udm, cfg, force_cpu, contract.pooling)?;
        Ok(Self {
            dimensions: enc.dim,
            backend: Backend::Onnx(enc),
            model_name: name,
            contract,
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
            contract: Contract::from_prefixes(cfg.query_prefix.clone(), cfg.doc_prefix.clone()),
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
            #[cfg(not(feature = "bench-stub"))]
            // Fixed-shape ORT. `OnnxEncoder` owns the split into
            // `self.batch` runs (so the EP sees one batch width), and
            // peak memory is bounded by `batch * bucket * hidden`
            // regardless of the group size handed in by sync.
            Backend::Onnx(e) => e.embed(&texts),
            #[cfg(candle_backend)]
            // Metal GPU; owns its own sub-batch split like `OnnxEncoder`.
            Backend::Candle(e) => e.embed(&texts),
        }
    }

    /// Stable identity of this model's resolved input/output contract,
    /// folded into the index fingerprint so changing a model's prefix
    /// or pooling forces the automatic one-time re-embed.
    pub fn fingerprint_tag(&self) -> String {
        self.contract.tag()
    }

    /// Embed indexed chunks, prepending the model's `doc_prefix` when
    /// it has one (verbatim, zero extra allocation, when it does not).
    pub fn embed_documents(&self, texts: &[&str], batch_size: usize) -> Result<Vec<Vec<f32>>> {
        match &self.contract.doc_prefix {
            Some(p) => {
                let owned: Vec<String> = texts.iter().map(|t| format!("{p}{t}")).collect();
                self.embed_raw(owned.iter().map(String::as_str).collect(), batch_size)
            }
            None => self.embed_raw(texts.to_vec(), batch_size),
        }
    }

    /// Embed a search query, prepending the model's `query_prefix`
    /// (e5 `query: `, nomic `search_query: `, a CLS-model instruction,
    /// or the full Qwen3 `Instruct: …\nQuery:` template) when set.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let owned;
        let q: &str = match &self.contract.query_prefix {
            Some(p) => {
                owned = format!("{p}{text}");
                &owned
            }
            None => text,
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
        // a true LM / causal head only → reject
        let mlm = br#"{"architectures":["LlamaForCausalLM"]}"#;
        assert!(assert_loadable_embedding("r", mlm).is_err());
        let bert_mlm = br#"{"architectures":["BertForMaskedLM"]}"#;
        assert!(assert_loadable_embedding("r", bert_mlm).is_err());
        // jina v2 exports as *ForMaskedLM but its ONNX is a poolable
        // encoder (fastembed mean-pools this repo) → explicitly allowed
        assert!(
            assert_loadable_embedding("r", br#"{"architectures":["JinaBertForMaskedLM"]}"#).is_ok()
        );
        // encoder export, mixed, unparseable/absent → allow
        assert!(assert_loadable_embedding("r", br#"{"architectures":["JinaBertModel"]}"#).is_ok());
        let mixed = br#"{"architectures":["BertModel","BertForMaskedLM"]}"#;
        assert!(assert_loadable_embedding("r", mixed).is_ok());
        assert!(assert_loadable_embedding("r", b"not json").is_ok());
        assert!(assert_loadable_embedding("r", b"{}").is_ok());
    }

    #[test]
    fn is_kv_cache_decoder_detects_past_key_values() {
        assert!(is_kv_cache_decoder(
            b"...graph...present_key_values...past_key_values.0.key...input_ids..."
        ));
        assert!(!is_kv_cache_decoder(
            b"input_ids attention_mask token_type_ids last_hidden_state"
        ));
        assert!(!is_kv_cache_decoder(b""));
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

    // One row, s=3, hidden=2, seq=3. Tokens: t0=[1,0] t1=[0,1]
    // t2=[9,9] (padding, mask=0). Each pooling must pick a different,
    // correct vector.
    const DATA: &[f32] = &[1.0, 0.0, 0.0, 1.0, 9.0, 9.0];
    fn rank3<'a>(mask: &'a [i64]) -> Rank3<'a> {
        Rank3 {
            data: DATA,
            s: 3,
            hidden: 2,
            seq: 3,
            mask,
        }
    }

    #[test]
    fn pool_rank3_cls_takes_first_token() {
        assert_eq!(
            pool_rank3(Pooling::Cls, &rank3(&[1, 1, 0]), 0),
            vec![1.0, 0.0]
        );
    }

    #[test]
    fn pool_rank3_last_token_takes_last_unmasked() {
        // t2 is masked → last real token is t1 = [0,1].
        assert_eq!(
            pool_rank3(Pooling::LastToken, &rank3(&[1, 1, 0]), 0),
            vec![0.0, 1.0]
        );
    }

    #[test]
    fn pool_rank3_mean_averages_unmasked_then_normalizes() {
        // mean([1,0],[0,1]) = [0.5,0.5] → L2 → [0.707,0.707].
        let v = pool_rank3(Pooling::Mean, &rank3(&[1, 1, 0]), 0);
        assert!((v[0] - 0.707).abs() < 1e-3 && (v[1] - 0.707).abs() < 1e-3);
    }

    #[test]
    fn pool_rank3_last_token_all_pad_falls_back_to_token0() {
        let v = pool_rank3(Pooling::LastToken, &rank3(&[0, 0, 0]), 0);
        assert_eq!(v, vec![1.0, 0.0]); // token 0
    }

    #[test]
    fn contract_from_prefixes_keeps_pair_and_mean() {
        let on = Contract::from_prefixes(Some("query: ".into()), Some("passage: ".into()));
        assert_eq!(on.query_prefix.as_deref(), Some("query: "));
        assert_eq!(on.doc_prefix.as_deref(), Some("passage: "));
        assert_eq!(on.pooling, Pooling::Mean);
        let off = Contract::from_prefixes(None, None);
        assert!(off.query_prefix.is_none() && off.doc_prefix.is_none());
    }

    #[test]
    fn contract_tag_changes_with_prefix_or_pooling() {
        let a = Contract::from_prefixes(Some("query: ".into()), Some("passage: ".into())).tag();
        let b = Contract::from_prefixes(None, None).tag();
        assert_ne!(a, b); // prefix difference shifts the fingerprint
        let cls = Contract {
            query_prefix: None,
            doc_prefix: None,
            pooling: Pooling::Cls,
        };
        assert_ne!(Contract::from_prefixes(None, None).tag(), cls.tag()); // pooling too
    }

    #[test]
    fn pooling_fromstr_accepts_aliases_and_rejects_junk() {
        use std::str::FromStr;
        assert_eq!(Pooling::from_str("mean").unwrap(), Pooling::Mean);
        assert_eq!(Pooling::from_str("cls").unwrap(), Pooling::Cls);
        assert_eq!(Pooling::from_str("last-token").unwrap(), Pooling::LastToken);
        assert_eq!(Pooling::from_str("last_token").unwrap(), Pooling::LastToken);
        assert_eq!(Pooling::LastToken.label(), "last-token");
        assert!(Pooling::from_str("bogus").is_err());
    }
}
