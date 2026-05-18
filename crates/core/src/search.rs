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

/// The vector index has no path/keyword predicate, so scoped OR hybrid
/// queries over-fetch the embedding neighborhood then filter/re-rank
/// it: `limit * FACTOR`, at least `MIN`.
const OVERFETCH_FACTOR: usize = 40;
const OVERFETCH_MIN: usize = 400;

/// RRF damping (Cormack et al.): rank+K constant. 60 is the standard
/// value — large enough that top ranks aren't winner-take-all.
const RRF_K: f32 = 60.0;

/// Split into lowercased `[a-z0-9_]` terms of length ≥ 2 — the unit
/// shared by the lexical document and query side.
fn tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .filter(|t| t.len() >= 2)
        .map(str::to_ascii_lowercase)
        .collect()
}

/// BM25 (k1=1.2, b=0.75) of each candidate against the query, with the
/// candidate set itself as the corpus — a lexical re-ranking of the
/// embedding neighborhood, no separate full-text index. `docs` are the
/// pre-tokenized candidate contents; returns one score per candidate.
fn bm25_scores(docs: &[Vec<String>], query_terms: &[String]) -> Vec<f32> {
    const K1: f32 = 1.2;
    const B: f32 = 0.75;
    let n = docs.len();
    if n == 0 {
        return Vec::new();
    }
    // One pass per doc: term-frequency map (borrowed keys). df/tf are
    // then map lookups, not repeated full-doc scans.
    let tf_maps: Vec<HashMap<&str, u32>> = docs
        .iter()
        .map(|d| {
            let mut m = HashMap::with_capacity(d.len());
            for t in d {
                *m.entry(t.as_str()).or_insert(0) += 1;
            }
            m
        })
        .collect();
    let avgdl = (docs.iter().map(Vec::len).sum::<usize>() as f32 / n as f32).max(1.0);
    let mut out = vec![0.0f32; n];
    for qt in query_terms {
        let qt = qt.as_str();
        let df = tf_maps.iter().filter(|m| m.contains_key(qt)).count();
        if df == 0 {
            continue;
        }
        let idf = (1.0 + (n as f32 - df as f32 + 0.5) / (df as f32 + 0.5)).ln();
        for (i, m) in tf_maps.iter().enumerate() {
            let Some(&c) = m.get(qt) else { continue };
            let tf = c as f32;
            let dl = docs[i].len() as f32;
            let denom = tf + K1 * (1.0 - B + B * dl / avgdl);
            out[i] += idf * (tf * (K1 + 1.0)) / denom;
        }
    }
    out
}

/// Rank positions (0 = best) for `scores`, descending. Ties keep input
/// (semantic) order — a stable sort over original indices.
fn ranks_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0usize; scores.len()];
    for (pos, &i) in idx.iter().enumerate() {
        rank[i] = pos;
    }
    rank
}

pub fn run(
    engine: &SyncEngine,
    query: &str,
    limit: usize,
    scope: Option<&str>,
) -> Result<Vec<SearchResult>> {
    let qvec = engine.embedder().embed_query(query)?;
    let scope = scope.map(norm_scope).filter(|s| !s.is_empty());
    // Distinct query terms: a repeated term would scale its own idf
    // contribution uniformly (no ranking effect) and waste a df pass.
    let mut q_terms = tokenize(query);
    q_terms.sort_unstable();
    q_terms.dedup();
    // Hybrid only helps when the query has lexical content; a
    // symbol-only query falls back to pure vector ranking.
    let hybrid = engine.search_config().hybrid && !q_terms.is_empty();

    // Over-fetch the neighborhood when we must filter (scope) or
    // re-rank (hybrid) it; otherwise fetch exactly `limit`.
    let k = if scope.is_some() || hybrid {
        (limit * OVERFETCH_FACTOR).max(OVERFETCH_MIN)
    } else {
        limit
    };
    let exact = engine.use_exact();
    let hits = engine.with_vector(|v| v.search(&qvec, k, exact))?;

    // Materialize the (scope-filtered) candidates in cosine order. One
    // batched DB hit for the whole neighborhood (not one lock+query per
    // id); the cosine order comes from `hits`, the rows from the map.
    let vids: Vec<u64> = hits.iter().map(|(v, _)| *v).collect();
    let mut rows = engine.with_db(|db| db.chunks_by_vector_ids(&vids))?;
    let mut cands: Vec<(crate::db::ChunkRow, f32)> = Vec::with_capacity(hits.len());
    for (vid, cos) in &hits {
        let Some(cur) = rows.remove(vid) else {
            continue;
        };
        if let Some(s) = &scope {
            if !in_scope(&cur.file_path, s) {
                continue;
            }
        }
        cands.push((cur, *cos));
    }

    // Fuse the cosine ranking with a BM25 lexical ranking by Reciprocal
    // Rank Fusion. `score` then reports the fused relevance (sum of
    // reciprocal ranks); without hybrid it stays the raw cosine.
    let order: Vec<usize> = if hybrid {
        let docs: Vec<Vec<String>> = cands.iter().map(|(c, _)| tokenize(&c.content)).collect();
        let cos: Vec<f32> = cands.iter().map(|(_, s)| *s).collect();
        let sem_rank = ranks_desc(&cos);
        let lex_rank = ranks_desc(&bm25_scores(&docs, &q_terms));
        let mut fused: Vec<(usize, f32)> = (0..cands.len())
            .map(|i| {
                let f = 1.0 / (RRF_K + sem_rank[i] as f32) + 1.0 / (RRF_K + lex_rank[i] as f32);
                (i, f)
            })
            .collect();
        fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (i, f) in &fused {
            cands[*i].1 = *f;
        }
        fused.into_iter().map(|(i, _)| i).collect()
    } else {
        (0..cands.len()).collect()
    };

    // Per-file caches: each result file's text and its chunk list are
    // fetched at most once even when several hits share the file
    // (semantic hits cluster by file). Line numbers derive from byte
    // offsets against the on-disk text (kept consistent by the
    // freshness sync preceding search).
    let mut files: HashMap<String, Option<String>> = HashMap::new();
    let mut sibs: HashMap<String, Vec<crate::db::ChunkRow>> = HashMap::new();
    let mut out = Vec::with_capacity(limit);

    // `order` is a permutation, so each slot is taken exactly once —
    // moving the row out (vs cloning `content`/paths per result).
    let mut slots: Vec<Option<(crate::db::ChunkRow, f32)>> = cands.into_iter().map(Some).collect();
    for ci in order.into_iter().take(limit) {
        let Some((cur, score)) = slots[ci].take() else {
            continue;
        };
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
    fn tokenize_lowercases_and_drops_short_runs() {
        assert_eq!(
            tokenize("fn parse_HF_Repo(x)!"),
            vec!["fn", "parse_hf_repo"]
        );
        assert!(tokenize("a + b - .").is_empty()); // all len<2
    }

    #[test]
    fn bm25_ranks_exact_term_match_first() {
        let docs = vec![
            tokenize("the quick brown fox"),
            tokenize("embedding search hybrid rerank"),
            tokenize("totally unrelated content here"),
        ];
        let q = tokenize("hybrid rerank");
        let s = bm25_scores(&docs, &q);
        let r = ranks_desc(&s);
        assert_eq!(r[1], 0); // doc 1 contains both query terms → best
        assert!(s[0] == 0.0 && s[2] == 0.0);
    }

    #[test]
    fn ranks_desc_breaks_ties_by_input_order() {
        // equal scores keep original order (stable)
        assert_eq!(ranks_desc(&[1.0, 1.0, 2.0]), vec![1, 2, 0]);
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
