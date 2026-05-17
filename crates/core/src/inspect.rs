//! Read-only access to an existing index — never loads the embedding model.

use crate::config::{Config, PROJECT_INDEX_DIR};
use crate::db::{ChunkRow, Db, FileInfo};
use crate::error::Result;
use crate::sync::IndexStatus;
use crate::vector::VectorIndex;
use std::path::{Path, PathBuf};

pub struct Inspector {
    db: Db,
    vector: VectorIndex,
    config: Config,
}

impl Inspector {
    pub fn open(project_dir: &Path, config: Config) -> Result<Self> {
        let index_dir: PathBuf = project_dir.join(PROJECT_INDEX_DIR);
        let db = Db::open_or_create(&index_dir)?;
        let dims = db
            .get_meta("dimensions")?
            .and_then(|s| s.parse::<usize>().ok())
            .or_else(|| config.model_spec().ok().map(|s| s.dimensions))
            .unwrap_or(768);
        let vector = VectorIndex::open_or_create(&index_dir, dims)?;
        Ok(Self { db, vector, config })
    }

    pub fn status(&self) -> Result<IndexStatus> {
        let (files, chunks) = self.db.counts()?;
        let last = self.db.get_meta("last_sync_at")?;
        let model = self
            .db
            .get_meta("model_name")?
            .unwrap_or_else(|| self.config.model.default.clone());
        let merkle_root = self.db.get_meta("merkle_root")?;
        Ok(IndexStatus::assemble(
            files,
            chunks,
            self.vector.len(),
            model,
            last,
            merkle_root,
            &self.config,
        ))
    }

    pub fn list_files(&self) -> Result<Vec<FileInfo>> {
        self.db.list_files()
    }

    pub fn chunks_for_file(&self, rel: &str) -> Result<Vec<ChunkRow>> {
        self.db.chunks_for_file(rel)
    }
}
