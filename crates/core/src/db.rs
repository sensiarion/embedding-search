use crate::error::Result;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension};
use std::collections::HashMap;
use std::path::Path;

const SCHEMA_VERSION: &str = "4";

#[derive(Debug, Clone)]
pub struct NewChunk {
    pub chunk_index: i32,
    pub content: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub node_type: String,
    pub language: String,
    /// blake3(header ++ body) — exact embed-text identity. Used for
    /// intra-file `Same` reuse where the path/symbol/sig haven't
    /// shifted.
    pub content_hash: String,
    /// blake3(body) only — used for cross-file `Copy` reuse so that
    /// renames / branch switches that place identical code at a new
    /// path can copy the source embedding instead of re-running the
    /// model. The source vector was computed against the OLD header,
    /// so the new chunk's embedding will be very slightly off (header
    /// is a few dozen tokens out of hundreds — dominant body signal
    /// is preserved). Acceptable trade-off for skipping the embed.
    pub body_hash: String,
}

#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub id: i64,
    pub file_id: i64,
    pub file_path: String,
    pub chunk_index: i32,
    pub content: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub node_type: String,
    pub language: String,
    pub vector_id: u64,
}

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub file_id: i64,
    pub path: String,
    pub content_hash: String,
    pub last_modified: i64,
    pub chunk_count: i64,
}

/// Indexed state of one file — a leaf of the file hash tree.
#[derive(Debug, Clone)]
pub struct FileState {
    pub id: i64,
    pub mtime: i64,
    pub size: i64,
    pub hash: String,
}

pub struct Db {
    conn: Connection,
}

impl Db {
    pub fn open_or_create(index_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(index_dir)?;
        let conn = Connection::open(index_dir.join("meta.db"))?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        let db = Self { conn };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                path          TEXT NOT NULL UNIQUE,
                last_modified INTEGER NOT NULL,
                size          INTEGER NOT NULL DEFAULT 0,
                content_hash  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                start_byte  INTEGER NOT NULL,
                end_byte    INTEGER NOT NULL,
                node_type   TEXT NOT NULL DEFAULT 'lines',
                language    TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL DEFAULT '',
                body_hash    TEXT NOT NULL DEFAULT '',
                vector_id   INTEGER NOT NULL UNIQUE
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id   ON chunks(file_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_body_hash ON chunks(body_hash);
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            "#,
        )?;
        // In-place column adds for pre-v3 databases (keeps existing
        // embeddings; CREATE above already has them for fresh DBs).
        // Legacy `chunks` get `content_hash = ''`, which matches no
        // blake3, so each such file re-embeds once on its next edit —
        // then chunk-level reuse kicks in. Best-effort ALTER is only
        // attempted on an older `schema_version` (a real failure on a
        // healthy DB then surfaces, not silently swallowed forever).
        let ver = self
            .get_meta("schema_version")?
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        let target = SCHEMA_VERSION.parse::<u32>().unwrap_or(0);
        if ver < target {
            let _ = self.conn.execute(
                "ALTER TABLE files ADD COLUMN size INTEGER NOT NULL DEFAULT 0",
                [],
            );
            let _ = self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''",
                [],
            );
            let _ = self.conn.execute(
                "ALTER TABLE chunks ADD COLUMN body_hash TEXT NOT NULL DEFAULT ''",
                [],
            );
            let _ = self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_body_hash ON chunks(body_hash)",
                [],
            );
        }
        self.set_meta("schema_version", SCHEMA_VERSION)?;
        if self.get_meta("next_vector_id")?.is_none() {
            self.set_meta("next_vector_id", "1")?;
        }
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| {
                r.get::<_, String>(0)
            })
            .optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// path -> indexed state for every file (the "hash tree" leaves:
    /// mtime+size are the cheap node check, content_hash the authority).
    pub fn file_states(&self) -> Result<HashMap<String, FileState>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, id, last_modified, size, content_hash FROM files")?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                FileState {
                    id: r.get::<_, i64>(1)?,
                    mtime: r.get::<_, i64>(2)?,
                    size: r.get::<_, i64>(3)?,
                    hash: r.get::<_, String>(4)?,
                },
            ))
        })?;
        Ok(rows.collect::<rusqlite::Result<HashMap<_, _>>>()?)
    }

    /// Root of the file hash tree: blake3 over the path-sorted
    /// `(path, content_hash)` leaves. A single value that changes iff
    /// any indexed file's content changed (Merkle-style state digest).
    pub fn merkle_root(&self) -> Result<String> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, content_hash FROM files ORDER BY path")?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        let mut h = blake3::Hasher::new();
        for row in rows {
            let (path, hash) = row?;
            h.update(path.as_bytes());
            h.update(b"\0");
            h.update(hash.as_bytes());
            h.update(b"\n");
        }
        Ok(h.finalize().to_hex().to_string())
    }

    pub fn upsert_file(&self, path: &str, mtime: i64, size: i64, hash: &str) -> Result<i64> {
        self.conn.execute(
            "INSERT INTO files(path, last_modified, size, content_hash) VALUES(?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET last_modified = excluded.last_modified,
                                             size          = excluded.size,
                                             content_hash  = excluded.content_hash",
            params![path, mtime, size, hash],
        )?;
        let id = self
            .conn
            .query_row("SELECT id FROM files WHERE path = ?1", [path], |r| {
                r.get::<_, i64>(0)
            })?;
        Ok(id)
    }

    /// Refresh just the (mtime, size) on an existing `files` row. Used
    /// by the sync's "content unchanged but stat moved" path (`git
    /// checkout` rewrites mtimes on every touched file): without this
    /// the fast `(mtime, size)` short-circuit in `classify` would never
    /// fire on a subsequent sync, and we'd re-read + re-hash the same
    /// bytes on every tick forever. Returns whether the row existed —
    /// `false` means a concurrent delete (Clear / external mutation)
    /// removed the row between the sync's snapshot and this call, so
    /// the caller can re-insert with `upsert_file` instead of silently
    /// leaving the file unindexed.
    pub fn touch_file_meta(&self, path: &str, mtime: i64, size: i64) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE files SET last_modified = ?2, size = ?3 WHERE path = ?1",
            params![path, mtime, size],
        )?;
        Ok(n > 0)
    }

    pub fn vector_ids_for_file(&self, file_id: i64) -> Result<Vec<u64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT vector_id FROM chunks WHERE file_id = ?1")?;
        let rows = stmt.query_map([file_id], |r| Ok(r.get::<_, i64>(0)? as u64))?;
        Ok(rows.collect::<rusqlite::Result<Vec<u64>>>()?)
    }

    /// Delete a file (CASCADE removes its chunks). Returns removed vector ids.
    pub fn delete_file_by_path(&self, path: &str) -> Result<Vec<u64>> {
        let file_id: Option<i64> = self
            .conn
            .query_row("SELECT id FROM files WHERE path = ?1", [path], |r| r.get(0))
            .optional()?;
        let Some(file_id) = file_id else {
            return Ok(Vec::new());
        };
        let vids = self.vector_ids_for_file(file_id)?;
        self.conn
            .execute("DELETE FROM files WHERE id = ?1", [file_id])?;
        Ok(vids)
    }

    /// Look up a vector_id for ANY chunk in the index whose BODY
    /// matches `body_hash`, regardless of which file owns it or what
    /// path/symbol prefix it had at index time. Used for cross-file
    /// `Copy` reuse: a chunk that moved across files (rename or `git
    /// checkout` to a branch with the same code at a new path) can
    /// copy the existing embedding instead of re-running the model.
    /// The source vector was computed against the OLD header — accept
    /// a small embedding drift in exchange for skipping the embed.
    /// Returns the lowest matching vector_id for determinism. Filters
    /// out legacy chunks with empty body_hash (pre-v4 schema).
    pub fn lookup_vector_id_by_body_hash(&self, body_hash: &str) -> Result<Option<u64>> {
        if body_hash.is_empty() {
            return Ok(None);
        }
        let row: Option<i64> = self
            .conn
            .query_row(
                "SELECT MIN(vector_id) FROM chunks WHERE body_hash = ?1",
                [body_hash],
                |r| r.get(0),
            )
            .optional()?
            .flatten();
        Ok(row.map(|v| v as u64))
    }

    /// `(content_hash, vector_id)` for every chunk of a file, in
    /// chunk order — the prior state used to reuse vectors for chunks
    /// whose content did not change across a file edit.
    pub fn chunk_fingerprints_for_file(&self, path: &str) -> Result<Vec<(String, u64)>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.content_hash, c.vector_id
             FROM chunks c JOIN files f ON f.id = c.file_id
             WHERE f.path = ?1 ORDER BY c.chunk_index",
        )?;
        let rows = stmt.query_map([path], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? as u64))
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// Reserve `n` contiguous vector ids; never reused after deletion.
    pub fn alloc_vector_ids(&self, n: usize) -> Result<Vec<u64>> {
        let cur: u64 = self
            .get_meta("next_vector_id")?
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let next = cur + n as u64;
        self.set_meta("next_vector_id", &next.to_string())?;
        Ok((cur..next).collect())
    }

    /// Atomically replace a file's chunk rows: delete the old set and
    /// insert `chunks` in one transaction (one durable commit). The
    /// `files` row is untouched so reused vector ids keep their owner.
    pub fn replace_chunks(
        &mut self,
        file_id: i64,
        chunks: &[NewChunk],
        vector_ids: &[u64],
    ) -> Result<()> {
        let tx = self.conn.transaction()?;
        {
            tx.execute("DELETE FROM chunks WHERE file_id = ?1", [file_id])?;
            let mut stmt = tx.prepare(
                "INSERT INTO chunks(file_id, chunk_index, content, start_byte,
                                    end_byte, node_type, language,
                                    content_hash, body_hash, vector_id)
                 VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            )?;
            for (c, vid) in chunks.iter().zip(vector_ids.iter()) {
                stmt.execute(params![
                    file_id,
                    c.chunk_index,
                    c.content,
                    c.start_byte,
                    c.end_byte,
                    c.node_type,
                    c.language,
                    c.content_hash,
                    c.body_hash,
                    *vid as i64,
                ])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    pub fn chunk_by_vector_id(&self, vector_id: u64) -> Result<Option<ChunkRow>> {
        Ok(self
            .conn
            .query_row(
                "SELECT c.id, c.file_id, f.path, c.chunk_index, c.content,
                        c.start_byte, c.end_byte, c.node_type, c.language, c.vector_id
                 FROM chunks c JOIN files f ON f.id = c.file_id
                 WHERE c.vector_id = ?1",
                [vector_id as i64],
                map_chunk_row,
            )
            .optional()?)
    }

    /// Batch form of `chunk_by_vector_id`: one prepared `IN (...)`
    /// query (one lock, one scan) keyed by `vector_id`. The search hot
    /// path materializes the whole over-fetched neighborhood at once —
    /// per-id round-trips would be O(candidates) Mutex+SQL hits.
    pub fn chunks_by_vector_ids(&self, ids: &[u64]) -> Result<HashMap<u64, ChunkRow>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let ph = vec!["?"; ids.len()].join(",");
        let sql = format!(
            "SELECT c.id, c.file_id, f.path, c.chunk_index, c.content,
                    c.start_byte, c.end_byte, c.node_type, c.language, c.vector_id
             FROM chunks c JOIN files f ON f.id = c.file_id
             WHERE c.vector_id IN ({ph})"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let bound = ids.iter().map(|&v| v as i64);
        let rows = stmt.query_map(params_from_iter(bound), map_chunk_row)?;
        let mut m = HashMap::with_capacity(ids.len());
        for r in rows {
            let c = r?;
            m.insert(c.vector_id, c);
        }
        Ok(m)
    }

    pub fn chunks_for_file(&self, path: &str) -> Result<Vec<ChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.file_id, f.path, c.chunk_index, c.content,
                    c.start_byte, c.end_byte, c.node_type, c.language, c.vector_id
             FROM chunks c JOIN files f ON f.id = c.file_id
             WHERE f.path = ?1 ORDER BY c.chunk_index",
        )?;
        let rows = stmt.query_map([path], map_chunk_row)?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    pub fn list_files(&self) -> Result<Vec<FileInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.id, f.path, f.content_hash, f.last_modified,
                    (SELECT COUNT(*) FROM chunks c WHERE c.file_id = f.id)
             FROM files f ORDER BY f.path",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(FileInfo {
                file_id: r.get(0)?,
                path: r.get(1)?,
                content_hash: r.get(2)?,
                last_modified: r.get(3)?,
                chunk_count: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
    }

    /// (file_count, chunk_count)
    pub fn counts(&self) -> Result<(i64, i64)> {
        let files: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
        let chunks: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok((files, chunks))
    }
}

fn map_chunk_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ChunkRow> {
    Ok(ChunkRow {
        id: r.get(0)?,
        file_id: r.get(1)?,
        file_path: r.get(2)?,
        chunk_index: r.get(3)?,
        content: r.get(4)?,
        start_byte: r.get(5)?,
        end_byte: r.get(6)?,
        node_type: r.get(7)?,
        language: r.get(8)?,
        vector_id: r.get::<_, i64>(9)? as u64,
    })
}
