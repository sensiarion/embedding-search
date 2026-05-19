use std::path::Path;
use tree_sitter::{Language, Node, Parser};

/// Semantic category of a chunk. Single source of truth for the
/// `node_type` string stored in the DB / returned in search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    Function,
    Method,
    Class,
    Struct,
    Impl,
    Trait,
    Enum,
    Module,
    Type,
    Block,
    Lines,
    Heading,
    YamlKey,
    TomlKey,
    JsonKey,
}

impl NodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NodeKind::Function => "function",
            NodeKind::Method => "method",
            NodeKind::Class => "class",
            NodeKind::Struct => "struct",
            NodeKind::Impl => "impl",
            NodeKind::Trait => "trait",
            NodeKind::Enum => "enum",
            NodeKind::Module => "module",
            NodeKind::Type => "type",
            NodeKind::Block => "block",
            NodeKind::Lines => "lines",
            NodeKind::Heading => "heading",
            NodeKind::YamlKey => "yaml_key",
            NodeKind::TomlKey => "toml_key",
            NodeKind::JsonKey => "json_key",
        }
    }

    /// Classify a tree-sitter node kind into a coarse category.
    /// Exact map of the closed set of tree-sitter kinds listed in the
    /// per-language `LangSpec` tables (no fragile substring matching).
    fn from_ts_kind(kind: &str) -> Self {
        match kind {
            "function_item"
            | "function_definition"
            | "function_declaration"
            | "generator_function_declaration" => NodeKind::Function,
            "method_definition" | "method_declaration" | "constructor_declaration" => {
                NodeKind::Method
            }
            "class_definition"
            | "class_declaration"
            | "abstract_class_declaration"
            | "class_specifier" => NodeKind::Class,
            "struct_item" | "struct_specifier" | "union_item" => NodeKind::Struct,
            "impl_item" => NodeKind::Impl,
            "trait_item" | "interface_declaration" => NodeKind::Trait,
            "enum_item" | "enum_declaration" | "enum_specifier" => NodeKind::Enum,
            "mod_item" | "module" | "internal_module" | "namespace_definition" => NodeKind::Module,
            "type_item" | "type_declaration" | "type_definition" => NodeKind::Type,
            _ => NodeKind::Block,
        }
    }

    /// Structured-data key category for a language label.
    fn structured(lang: &str) -> Self {
        match lang {
            "yaml" => NodeKind::YamlKey,
            "toml" => NodeKind::TomlKey,
            _ => NodeKind::JsonKey,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub content: String,
    pub start_byte: i64,
    pub end_byte: i64,
    pub node_type: String,
    /// Declared name of the AST node (fn/struct/impl-type/…), when one
    /// exists. `None` for line/markdown/structured chunks. Used only to
    /// build the code↔NL embed header — never stored or returned.
    pub symbol: Option<String>,
}

pub struct Chunker {
    max_chunk_bytes: usize,
}

struct LangSpec {
    label: &'static str,
    language: fn() -> Language,
    containers: &'static [&'static str],
    leaves: &'static [&'static str],
}

fn lang_for_ext(ext: &str) -> Option<LangSpec> {
    Some(match ext {
        "rs" => LangSpec {
            label: "rust",
            language: || tree_sitter_rust::LANGUAGE.into(),
            containers: &["impl_item", "mod_item", "trait_item"],
            leaves: &[
                "function_item",
                "struct_item",
                "enum_item",
                "type_item",
                "macro_definition",
                "const_item",
                "static_item",
                "union_item",
            ],
        },
        "py" | "pyi" => LangSpec {
            label: "python",
            language: || tree_sitter_python::LANGUAGE.into(),
            containers: &["class_definition"],
            leaves: &["function_definition", "decorated_definition"],
        },
        "ts" => LangSpec {
            label: "typescript",
            language: || tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            containers: &["class_declaration", "internal_module", "module"],
            leaves: &[
                "function_declaration",
                "method_definition",
                "abstract_class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
        },
        "tsx" => LangSpec {
            label: "tsx",
            language: || tree_sitter_typescript::LANGUAGE_TSX.into(),
            containers: &["class_declaration", "internal_module", "module"],
            leaves: &[
                "function_declaration",
                "method_definition",
                "abstract_class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
        },
        "js" | "jsx" | "mjs" | "cjs" => LangSpec {
            label: "javascript",
            language: || tree_sitter_javascript::LANGUAGE.into(),
            containers: &["class_declaration"],
            leaves: &[
                "function_declaration",
                "generator_function_declaration",
                "method_definition",
            ],
        },
        "go" => LangSpec {
            label: "go",
            language: || tree_sitter_go::LANGUAGE.into(),
            containers: &[],
            leaves: &[
                "function_declaration",
                "method_declaration",
                "type_declaration",
            ],
        },
        "java" => LangSpec {
            label: "java",
            language: || tree_sitter_java::LANGUAGE.into(),
            containers: &[
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ],
            leaves: &["method_declaration", "constructor_declaration"],
        },
        "c" | "h" => LangSpec {
            label: "c",
            language: || tree_sitter_c::LANGUAGE.into(),
            containers: &[],
            leaves: &[
                "function_definition",
                "struct_specifier",
                "enum_specifier",
                "type_definition",
            ],
        },
        "cpp" | "cc" | "cxx" | "hpp" | "hh" | "hxx" => LangSpec {
            label: "cpp",
            language: || tree_sitter_cpp::LANGUAGE.into(),
            containers: &[
                "class_specifier",
                "struct_specifier",
                "namespace_definition",
            ],
            leaves: &["function_definition", "enum_specifier", "type_definition"],
        },
        _ => return None,
    })
}

/// Declared name of an AST node: the tree-sitter `name` field if the
/// grammar exposes one (fn/struct/trait/mod/class…), else the first
/// `identifier`/`type_identifier` descendant (covers `impl Foo`, whose
/// name is its `type_identifier`, and grammars with no `name` field).
/// Only called on nodes that fit one chunk (≤ a few KB) so the bounded
/// pre-order descent is cheap.
fn node_symbol(node: &Node<'_>, src: &str) -> Option<String> {
    fn first_ident(node: Node<'_>, src: &str) -> Option<String> {
        if matches!(node.kind(), "identifier" | "type_identifier") {
            return src
                .get(node.start_byte()..node.end_byte())
                .map(str::to_owned);
        }
        let mut c = node.walk();
        let found = node
            .named_children(&mut c)
            .find_map(|ch| first_ident(ch, src));
        found
    }
    node.child_by_field_name("name")
        .and_then(|n| src.get(n.start_byte()..n.end_byte()))
        .map(str::to_owned)
        .or_else(|| first_ident(*node, src))
}

impl Chunker {
    pub fn new(max_chunk_bytes: usize) -> Self {
        Self { max_chunk_bytes }
    }

    /// Returns (language_label, chunks).
    pub fn chunk_file(&self, path: &Path, content: &str) -> (String, Vec<Chunk>) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        let (label, raw) = if let Some(spec) = lang_for_ext(&ext) {
            let chunks = self.ast_chunk(&spec, content);
            let chunks = if chunks.is_empty() {
                self.line_chunk(content, 0)
            } else {
                self.merge_small(chunks, content)
            };
            (spec.label.to_string(), chunks)
        } else {
            match ext.as_str() {
                "md" | "mdx" | "markdown" => ("markdown".into(), self.markdown_chunk(content)),
                "yaml" | "yml" => ("yaml".into(), self.structured_chunk(content, "yaml")),
                "toml" => ("toml".into(), self.structured_chunk(content, "toml")),
                "json" => ("json".into(), self.structured_chunk(content, "json")),
                _ => ("text".into(), self.line_chunk(content, 0)),
            }
        };
        (label, self.enforce_cap(raw))
    }

    /// Hard ceiling for EVERY chunk regardless of source path. Oversized
    /// chunks are split on UTF-8 boundaries — without this a single
    /// minified/long-line file becomes a multi-MB chunk that explodes the
    /// tokenizer's O(seq²) attention tensor.
    fn enforce_cap(&self, chunks: Vec<Chunk>) -> Vec<Chunk> {
        let cap = self.max_chunk_bytes.max(256);
        let mut out = Vec::with_capacity(chunks.len());
        for c in chunks {
            if c.content.len() <= cap {
                out.push(c);
                continue;
            }
            let base = c.start_byte;
            let mut off = 0usize;
            let bytes = c.content.as_bytes();
            while off < bytes.len() {
                let mut end = (off + cap).min(bytes.len());
                while end < bytes.len() && !c.content.is_char_boundary(end) {
                    end -= 1;
                }
                out.push(Chunk {
                    content: c.content[off..end].to_string(),
                    start_byte: base + off as i64,
                    end_byte: base + end as i64,
                    node_type: c.node_type.clone(),
                    symbol: c.symbol.clone(),
                });
                off = end;
            }
        }
        out
    }

    /// Coalesce adjacent AST chunks whose combined span stays within
    /// `max_chunk_bytes` (semble's strategy). One-chunk-per-leaf
    /// fragments a file into many tiny vectors — a 3-line fn becomes
    /// its own chunk — which retrieves poorly (especially for the
    /// static bag-of-token-means models, where a longer span is far
    /// more discriminative) and bloats the index. Merged content is
    /// re-sliced from `src`, so the span is contiguous: inter-node
    /// comments / blank lines between two merged nodes are kept rather
    /// than dropped. A homogeneous run keeps its node type (e.g.
    /// several functions → `function`, signature = the first); a mixed
    /// run becomes `block`; `symbol` is cleared once a run is merged
    /// (no single declaration owns it). Single chunks pass through
    /// unchanged — `src[start..end]` equals their original content,
    /// since every AST-path chunk is exactly its byte span.
    fn merge_small(&self, mut chunks: Vec<Chunk>, src: &str) -> Vec<Chunk> {
        if chunks.len() < 2 {
            return chunks;
        }
        // collect_node emits in pre-order (already ascending), but a
        // defensive sort makes the greedy pass independent of that.
        chunks.sort_by_key(|c| c.start_byte);
        let target = self.max_chunk_bytes.max(256);
        let mut out: Vec<Chunk> = Vec::with_capacity(chunks.len());
        let mut start = chunks[0].start_byte;
        let mut end = chunks[0].end_byte;
        let mut node_type = std::mem::take(&mut chunks[0].node_type);
        let mut symbol = chunks[0].symbol.take();
        let mut merged = false;
        let mut flush = |s: i64, e: i64, nt: String, sym: Option<String>, m: bool| {
            out.push(Chunk {
                // Single chunk: the slice == its original content (every
                // AST-path chunk is exactly its span), so re-slicing is
                // not a behavior change, just a uniform code path.
                content: src[s as usize..e as usize].to_string(),
                start_byte: s,
                end_byte: e,
                node_type: nt,
                symbol: if m { None } else { sym },
            });
        };
        for c in chunks.into_iter().skip(1) {
            // Only fold a strictly-following node in (collect_node never
            // overlaps, but a nested/duplicate span must start a new
            // run) and never past the cap, so enforce_cap can't later
            // split a merged run mid-node.
            if c.start_byte >= end && (c.end_byte - start) as usize <= target {
                if c.node_type != node_type {
                    node_type = NodeKind::Block.as_str().to_string();
                }
                end = c.end_byte;
                merged = true;
            } else {
                flush(start, end, std::mem::take(&mut node_type), symbol.take(), merged);
                start = c.start_byte;
                end = c.end_byte;
                node_type = c.node_type;
                symbol = c.symbol;
                merged = false;
            }
        }
        flush(start, end, node_type, symbol, merged);
        out
    }

    fn ast_chunk(&self, spec: &LangSpec, content: &str) -> Vec<Chunk> {
        let mut parser = Parser::new();
        if parser.set_language(&(spec.language)()).is_err() {
            return Vec::new();
        }
        let Some(tree) = parser.parse(content, None) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let root = tree.root_node();
        let mut cursor = root.walk();
        for child in root.named_children(&mut cursor) {
            self.collect_node(spec, &child, content, &mut out);
        }
        out
    }

    fn collect_node(&self, spec: &LangSpec, node: &Node<'_>, src: &str, out: &mut Vec<Chunk>) {
        let kind = node.kind();
        let is_container = spec.containers.contains(&kind);
        let is_leaf = spec.leaves.contains(&kind);
        if !is_container && !is_leaf {
            return;
        }
        let start = node.start_byte();
        let end = node.end_byte();
        let text = &src[start..end];
        let kind_str = NodeKind::from_ts_kind(kind).as_str();

        // Fits: one chunk.
        if text.len() <= self.max_chunk_bytes {
            out.push(Chunk {
                content: text.to_string(),
                start_byte: start as i64,
                end_byte: end as i64,
                node_type: kind_str.into(),
                symbol: node_symbol(node, src),
            });
            return;
        }

        let tag_lines = |this: &Self, out: &mut Vec<Chunk>| {
            let mut sub = this.line_chunk(text, start);
            for c in &mut sub {
                c.node_type = kind_str.into();
            }
            out.extend(sub);
        };

        // Oversized leaf: split by lines.
        if !is_container {
            tag_lines(self, out);
            return;
        }

        // Oversized container: descend; fall back to line split if it has
        // no extractable members.
        let before = out.len();
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            self.collect_node(spec, &child, src, out);
        }
        if out.len() == before {
            tag_lines(self, out);
        }
    }

    fn line_chunk(&self, content: &str, base_offset: usize) -> Vec<Chunk> {
        const LINES_PER: usize = 100;
        const OVERLAP: usize = 20;
        // Byte offset of each line start. `split_inclusive('\n')`
        // keeps the terminator so piece lengths are exact for `\n`
        // AND `\r\n` (the old `len()+1` drifted on CRLF, landing a
        // slice mid-char on multibyte text → panic). A line start is
        // right after `\n` (ASCII) so every offset is a char boundary.
        let mut line_starts = Vec::new();
        let mut pos = 0usize;
        for piece in content.split_inclusive('\n') {
            line_starts.push(pos);
            pos += piece.len();
        }
        let n = line_starts.len();
        if n == 0 {
            return Vec::new();
        }

        let mut out = Vec::new();
        let mut i = 0;
        while i < n {
            let end = (i + LINES_PER).min(n);
            let s = line_starts[i];
            let e = if end < n {
                line_starts[end]
            } else {
                content.len()
            };
            out.push(Chunk {
                content: content[s..e].to_string(),
                start_byte: (base_offset + s) as i64,
                end_byte: (base_offset + e) as i64,
                node_type: NodeKind::Lines.as_str().into(),
                symbol: None,
            });
            if end == n {
                break;
            }
            i += LINES_PER - OVERLAP;
        }
        out
    }

    fn markdown_chunk(&self, content: &str) -> Vec<Chunk> {
        let mut out = Vec::new();
        let mut sec_start = 0usize;
        let mut pos = 0usize;
        let mut started = false;
        for line in content.split_inclusive('\n') {
            let trimmed = line.trim_start();
            let is_header = {
                let hashes = trimmed.chars().take_while(|c| *c == '#').count();
                (1..=3).contains(&hashes) && trimmed.chars().nth(hashes) == Some(' ')
            };
            if is_header && started && pos > sec_start {
                out.push(Chunk {
                    content: content[sec_start..pos].to_string(),
                    start_byte: sec_start as i64,
                    end_byte: pos as i64,
                    node_type: NodeKind::Heading.as_str().into(),
                    symbol: None,
                });
                sec_start = pos;
            }
            started = true;
            pos += line.len();
        }
        if pos > sec_start {
            out.push(Chunk {
                content: content[sec_start..pos].to_string(),
                start_byte: sec_start as i64,
                end_byte: pos as i64,
                node_type: NodeKind::Heading.as_str().into(),
                symbol: None,
            });
        }
        if out.is_empty() {
            return self.line_chunk(content, 0);
        }
        out
    }

    fn structured_chunk(&self, content: &str, kind: &str) -> Vec<Chunk> {
        if content.len() < 2048 {
            return vec![Chunk {
                content: content.to_string(),
                start_byte: 0,
                end_byte: content.len() as i64,
                node_type: NodeKind::structured(kind).as_str().into(),
                symbol: None,
            }];
        }
        let split_here = |line: &str| -> bool {
            match kind {
                "toml" => line.starts_with('['),
                "yaml" => {
                    let c = line.chars().next();
                    c.is_some()
                        && !line.starts_with(char::is_whitespace)
                        && !line.starts_with('#')
                        && line.contains(':')
                }
                _ => false, // json: line fallback
            }
        };
        if kind == "json" {
            return self.line_chunk(content, 0);
        }
        let mut out = Vec::new();
        let mut sec_start = 0usize;
        let mut pos = 0usize;
        let mut started = false;
        for line in content.split_inclusive('\n') {
            if split_here(line) && started && pos > sec_start {
                out.push(Chunk {
                    content: content[sec_start..pos].to_string(),
                    start_byte: sec_start as i64,
                    end_byte: pos as i64,
                    node_type: NodeKind::structured(kind).as_str().into(),
                    symbol: None,
                });
                sec_start = pos;
            }
            started = true;
            pos += line.len();
        }
        if pos > sec_start {
            out.push(Chunk {
                content: content[sec_start..pos].to_string(),
                start_byte: sec_start as i64,
                end_byte: pos as i64,
                node_type: NodeKind::structured(kind).as_str().into(),
                symbol: None,
            });
        }
        if out.is_empty() {
            return self.line_chunk(content, 0);
        }
        out
    }
}
