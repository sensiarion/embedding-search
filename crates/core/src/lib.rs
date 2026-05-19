pub mod chunker;
pub mod config;
pub mod db;
pub mod embedder;
pub mod error;
pub mod inspect;
pub mod rerank;
pub mod search;
pub mod sync;
pub mod vector;

pub use config::{model_spec, Config, ModelSpec, SUPPORTED_MODELS};
pub use error::{Error, Result};
pub use inspect::Inspector;
pub use search::SearchResult;
pub use sync::{IndexStatus, SyncEngine, SyncEvent, SyncStats};
