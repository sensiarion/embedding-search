use crate::error::{Error, Result};
use std::path::{Path, PathBuf};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

pub struct VectorIndex {
    inner: Index,
    path: PathBuf,
}

impl VectorIndex {
    pub fn open_or_create(index_dir: &Path, dims: usize) -> Result<Self> {
        std::fs::create_dir_all(index_dir)?;
        let opts = IndexOptions {
            dimensions: dims,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let inner = Index::new(&opts).map_err(|e| Error::Index(e.to_string()))?;
        let path = index_dir.join("vectors.usearch");
        if path.exists() {
            inner
                .load(
                    path.to_str()
                        .ok_or_else(|| Error::Index("bad path".into()))?,
                )
                .map_err(|e| Error::Index(e.to_string()))?;
        } else {
            inner
                .reserve(1024)
                .map_err(|e| Error::Index(e.to_string()))?;
        }
        Ok(Self { inner, path })
    }

    fn ensure_capacity(&self, extra: usize) -> Result<()> {
        let needed = self.inner.size() + extra;
        if needed > self.inner.capacity() {
            let target = (needed.max(1024)).next_power_of_two();
            self.inner
                .reserve(target)
                .map_err(|e| Error::Index(e.to_string()))?;
        }
        Ok(())
    }

    pub fn add_batch(&self, keys: &[u64], vectors: &[Vec<f32>]) -> Result<()> {
        self.ensure_capacity(keys.len())?;
        for (k, v) in keys.iter().zip(vectors.iter()) {
            self.inner
                .add(*k, v)
                .map_err(|e| Error::Index(e.to_string()))?;
        }
        Ok(())
    }

    pub fn remove_many(&self, keys: &[u64]) -> Result<()> {
        for k in keys {
            // a key may be absent if a prior run crashed mid-write
            let _ = self.inner.remove(*k);
        }
        Ok(())
    }

    /// Returns (vector_id, similarity_score) pairs, best first. `exact`
    /// runs brute-force cosine over every vector (use on small indexes:
    /// HNSW is an approximate heuristic that buys nothing there and can
    /// miss the true nearest); otherwise the HNSW graph is queried.
    pub fn search(&self, query: &[f32], limit: usize, exact: bool) -> Result<Vec<(u64, f32)>> {
        if self.inner.size() == 0 {
            return Ok(Vec::new());
        }
        let m = if exact {
            self.inner.exact_search(query, limit)
        } else {
            self.inner.search(query, limit)
        }
        .map_err(|e| Error::Index(e.to_string()))?;
        Ok(m.keys
            .into_iter()
            .zip(m.distances)
            // cosine distance -> similarity
            .map(|(k, d)| (k, 1.0 - d))
            .collect())
    }

    pub fn save(&self) -> Result<()> {
        self.inner
            .save(
                self.path
                    .to_str()
                    .ok_or_else(|| Error::Index("bad path".into()))?,
            )
            .map_err(|e| Error::Index(e.to_string()))?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
