use crate::db::ChunkRow;
use crate::error::Result;
use crate::sync::SyncEngine;
use std::collections::HashMap;

/// Compact reference to a chunk adjacent to (or enclosing) a hit. No
/// body — just enough for the agent to decide whether to fetch it
/// ("coarse global + fine local": restore the Module/Class awareness
/// fine-grained chunks lack, without paying full-context tokens).
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkRef {
    pub node_type: String,
    pub signature: String,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub file_path: String,
    pub language: String,
    pub node_type: String,
    /// First meaningful line of the chunk (the def/signature), trimmed.
    pub signature: String,
    /// Innermost enclosing scope (impl/class/module) — the higher-level
    /// context a fine-grained chunk otherwise loses.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<ChunkRef>,
    /// 1-based inclusive line range in the file.
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
    /// Previous / next sibling chunk in the same file (refs only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prev: Option<ChunkRef>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<ChunkRef>,
    pub score: f32,
}

/// First non-empty line, trimmed, capped — a cheap stand-in for the
/// symbol signature.
fn signature_of(content: &str) -> String {
    let line = content
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    line.chars().take(160).collect()
}

/// 1-based line number of `byte` within `text` (newlines before it + 1).
fn line_at(text: &str, byte: usize) -> usize {
    let end = byte.min(text.len());
    text.as_bytes()[..end]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

fn chunk_ref(text: Option<&str>, c: &ChunkRow) -> ChunkRef {
    ChunkRef {
        node_type: c.node_type.clone(),
        signature: signature_of(&c.content),
        start_line: text.map_or(0, |t| line_at(t, c.start_byte as usize)),
        end_line: text.map_or(0, |t| line_at(t, c.end_byte as usize)),
    }
}

/// Innermost OTHER chunk in `siblings` that byte-encloses `cur`
/// (smallest containing span) — the parent scope.
fn enclosing<'a>(siblings: &'a [ChunkRow], cur: &ChunkRow) -> Option<&'a ChunkRow> {
    siblings
        .iter()
        .filter(|s| {
            s.vector_id != cur.vector_id
                && s.start_byte <= cur.start_byte
                && s.end_byte >= cur.end_byte
        })
        .min_by_key(|s| s.end_byte - s.start_byte)
}

/// Normalize a scope to a `/`-separated, no-`./`, no-trailing-`/`
/// project-relative prefix.
fn norm_scope(s: &str) -> String {
    s.replace('\\', "/")
        .trim()
        .trim_start_matches("./")
        .trim_matches('/')
        .to_string()
}

/// A file is in scope if it equals the scope (a file) or sits under it
/// (a directory). Allocation-free.
fn in_scope(file_path: &str, scope: &str) -> bool {
    scope.is_empty()
        || file_path == scope
        || file_path
            .strip_prefix(scope)
            .is_some_and(|rest| rest.starts_with('/'))
}

/// Scoped queries have no path predicate in the vector index, so they
/// over-fetch then filter: `limit * FACTOR`, at least `MIN`.
const SCOPE_OVERFETCH_FACTOR: usize = 40;
const SCOPE_OVERFETCH_MIN: usize = 400;

pub fn run(
    engine: &SyncEngine,
    query: &str,
    limit: usize,
    scope: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let qvec = engine.embedder().embed_query(query)?;
    let scope = scope.map(norm_scope).filter(|s| !s.is_empty());

    // Scoped queries over-fetch then filter (the vector index has no
    // path predicate); unscoped fetch exactly `limit`.
    let k = if scope.is_some() {
        (limit * SCOPE_OVERFETCH_FACTOR).max(SCOPE_OVERFETCH_MIN)
    } else {
        limit
    };
    let exact = engine.use_exact();
    let hits = engine.with_vector(|v| v.search(&qvec, k, exact))?;

    // Per-file caches: each result file's text and its chunk list are
    // fetched at most once even when several hits share the file
    // (semantic hits cluster by file). Line numbers derive from byte
    // offsets against the on-disk text (kept consistent by the
    // freshness sync preceding search).
    let mut files: HashMap<String, Option<String>> = HashMap::new();
    let mut sibs: HashMap<String, Vec<crate::db::ChunkRow>> = HashMap::new();
    let mut out = Vec::with_capacity(limit);

    for (vid, score) in hits {
        if out.len() >= limit {
            break;
        }
        let Some(cur) = engine.with_db(|db| db.chunk_by_vector_id(vid))? else {
            continue;
        };
        if let Some(s) = &scope {
            if !in_scope(&cur.file_path, s) {
                continue;
            }
        }
        let text = files
            .entry(cur.file_path.clone())
            .or_insert_with(|| {
                std::fs::read_to_string(engine.project_dir().join(&cur.file_path)).ok()
            })
            .as_deref();

        let siblings = match sibs.entry(cur.file_path.clone()) {
            std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
            std::collections::hash_map::Entry::Vacant(e) => {
                e.insert(engine.with_db(|db| db.chunks_for_file(&cur.file_path))?)
            }
        };
        let pos = siblings.iter().position(|s| s.vector_id == cur.vector_id);
        let prev = pos
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| siblings.get(i))
            .map(|c| chunk_ref(text, c));
        let next = pos
            .map(|i| i + 1)
            .and_then(|i| siblings.get(i))
            .map(|c| chunk_ref(text, c));
        let parent = enclosing(siblings, &cur).map(|c| chunk_ref(text, c));

        out.push(SearchResult {
            signature: signature_of(&cur.content),
            start_line: text.map_or(0, |t| line_at(t, cur.start_byte as usize)),
            end_line: text.map_or(0, |t| line_at(t, cur.end_byte as usize)),
            parent,
            prev,
            next,
            file_path: cur.file_path,
            language: cur.language,
            node_type: cur.node_type,
            content: cur.content,
            score,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_at_is_one_based_and_clamped() {
        let t = "a\nbb\nccc\n";
        assert_eq!(line_at(t, 0), 1); // before first newline
        assert_eq!(line_at(t, 2), 2); // just past first '\n'
        assert_eq!(line_at(t, 5), 3); // past second '\n'
        assert_eq!(line_at(t, 9999), 4); // clamped to end
    }

    #[test]
    fn signature_skips_blank_lines_and_caps() {
        assert_eq!(signature_of("\n\n  fn foo() {\n  body\n}"), "fn foo() {");
        assert_eq!(signature_of(""), "");
        assert_eq!(signature_of(&"x".repeat(300)).len(), 160);
    }

    fn row(vid: u64, s: i64, e: i64) -> ChunkRow {
        ChunkRow {
            id: vid as i64,
            file_id: 1,
            file_path: "f".into(),
            chunk_index: vid as i32,
            content: String::new(),
            start_byte: s,
            end_byte: e,
            node_type: "x".into(),
            language: "rust".into(),
            vector_id: vid,
        }
    }

    #[test]
    fn enclosing_picks_innermost_other_container() {
        let outer = row(1, 0, 100);
        let mid = row(2, 10, 80);
        let cur = row(3, 20, 30);
        let sibs = vec![outer, mid.clone(), cur.clone()];
        let p = enclosing(&sibs, &cur).expect("has parent");
        assert_eq!(p.vector_id, mid.vector_id); // innermost, not outer
                                                // a chunk that nothing encloses → None
        assert!(enclosing(&sibs, &sibs[0]).is_none());
    }
}
