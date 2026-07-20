///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Native Kimi (tiktoken) tokenizer for the kv-aware block hasher.
//!
//! Kimi-K2.6 ships a `TikTokenTokenizer` (tiktoken BPE): a `tiktoken.model`
//! rank table + a regex `pat_str`, but NO HF `tokenizer.json`, so the
//! `tokenizers` crate can't load it. To route Kimi *text* prompts cache-aware
//! (not only pre-tokenized token-id prompts) the router must reproduce the
//! engine's token ids byte-for-byte. This is a direct port of tiktoken's core:
//! split the text by the model's regex, then merge each piece by BPE rank.
//!
//! Encode matches `TikTokenTokenizer.encode(text, allow_special_tokens=True)`,
//! i.e. tiktoken `Encoding.encode(text, allowed_special="all")` — special-token
//! literals in the text map to their id, and no BOS/EOS is added.

use std::collections::HashMap;
use std::path::Path;

use base64::Engine as _;
use onig::Regex;
use serde_json::Value;

/// Token id / BPE rank. Matches the engine's `u32` token ids (and `hash_request`).
type Rank = u32;

/// Kimi's `pat_str` (tokenization_kimi.py), verbatim. It uses `&&` character
/// class intersection and a `(?!\S)` lookahead — Oniguruma supports both; the
/// `regex` crate rejects the lookahead, which is why we tokenize with `onig`.
const PAT: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// TikTokenTokenizer.num_reserved_special_tokens.
const NUM_RESERVED_SPECIAL_TOKENS: usize = 256;

pub struct KimiTokenizer {
    encoder: HashMap<Vec<u8>, Rank>,
    /// literal -> id, e.g. "<|im_end|>" -> 163586.
    special: HashMap<String, Rank>,
    /// Splits ordinary text into pre-token pieces.
    regex: Regex,
    /// Matches any special-token literal (longest-first); None if there are none.
    special_regex: Option<Regex>,
}

impl KimiTokenizer {
    /// Load from a model dir: needs `tiktoken.model`; reads special-token names
    /// from `tokenizer_config.json` (`added_tokens_decoder`) if present.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let model_path = dir.join("tiktoken.model");
        let encoder = load_bpe(&model_path)?;
        let num_base = encoder.len() as Rank;

        let names = load_special_names(&dir.join("tokenizer_config.json"));
        let mut special = HashMap::with_capacity(NUM_RESERVED_SPECIAL_TOKENS);
        for k in 0..NUM_RESERVED_SPECIAL_TOKENS as Rank {
            let id = num_base + k;
            let name = names
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
            special.insert(name, id);
        }

        let regex =
            Regex::new(PAT).map_err(|e| anyhow::anyhow!("tiktoken pat_str compile failed: {e}"))?;
        // Longest-first so a special that is a prefix of another still matches
        // fully (Oniguruma alternation is ordered / first-wins).
        let mut keys: Vec<&String> = special.keys().collect();
        keys.sort_by(|a, b| b.len().cmp(&a.len()).then(a.cmp(b)));
        let special_pat = keys
            .iter()
            .map(|k| escape_regex(k))
            .collect::<Vec<_>>()
            .join("|");
        let special_regex = if special_pat.is_empty() {
            None
        } else {
            Some(
                Regex::new(&special_pat)
                    .map_err(|e| anyhow::anyhow!("special-token regex compile failed: {e}"))?,
            )
        };

        Ok(KimiTokenizer {
            encoder,
            special,
            regex,
            special_regex,
        })
    }

    /// Encode text -> token ids, matching tiktoken `encode(allowed_special="all")`:
    /// special-token literals become their id; everything else is BPE'd.
    pub fn encode(&self, text: &str) -> Vec<Rank> {
        let mut out = Vec::new();
        let mut start = 0;
        loop {
            // Next special-token literal at or after `start`.
            let next = self
                .special_regex
                .as_ref()
                .and_then(|re| re.find(&text[start..]))
                .map(|(s, e)| (start + s, start + e));
            let end = next.map_or(text.len(), |(s, _)| s);
            self.encode_ordinary_into(&text[start..end], &mut out);
            match next {
                Some((s, e)) => {
                    if let Some(&id) = self.special.get(&text[s..e]) {
                        out.push(id);
                    }
                    start = e;
                }
                None => break,
            }
        }
        out
    }

    fn encode_ordinary_into(&self, text: &str, out: &mut Vec<Rank>) {
        for (s, e) in self.regex.find_iter(text) {
            let piece = &text.as_bytes()[s..e];
            match self.encoder.get(piece) {
                Some(&id) => out.push(id),
                None => byte_pair_encode(piece, &self.encoder, out),
            }
        }
    }
}

/// tiktoken's `byte_pair_merge`, then emit the id of each surviving segment.
/// Every segment is guaranteed to be an existing token; a single byte (0..=255)
/// always is, so the per-byte fallback can never miss.
fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>, out: &mut Vec<Rank>) {
    if piece.len() == 1 {
        if let Some(&id) = ranks.get(piece) {
            out.push(id);
        }
        return;
    }

    // parts[i] = (byte offset of segment i, rank of merging segment i with i+1).
    let mut parts: Vec<(usize, Rank)> = Vec::with_capacity(piece.len() + 1);
    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = ranks.get(&piece[i..i + 2]).copied().unwrap_or(Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = |parts: &Vec<(usize, Rank)>, i: usize| -> Rank {
        if i + 3 < parts.len() {
            let range = parts[i].0..parts[i + 3].0;
            ranks.get(&piece[range]).copied().unwrap_or(Rank::MAX)
        } else {
            Rank::MAX
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (j, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, j);
            }
        }
    }

    for w in parts.windows(2) {
        let seg = &piece[w[0].0..w[1].0];
        match ranks.get(seg) {
            Some(&id) => out.push(id),
            // Never expected (see fn doc), but degrade to raw bytes not a panic.
            None => {
                for b in seg {
                    if let Some(&id) = ranks.get(&[*b][..]) {
                        out.push(id);
                    }
                }
            }
        }
    }
}

/// Parse a `tiktoken.model` file: each non-empty line is `base64(token_bytes) rank`.
fn load_bpe(path: &Path) -> anyhow::Result<HashMap<Vec<u8>, Rank>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", path.display()))?;
    let mut ranks = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let (b64, rank) = match (it.next(), it.next()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue,
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| anyhow::anyhow!("bad base64 in {}: {e}", path.display()))?;
        let rank: Rank = rank
            .parse()
            .map_err(|e| anyhow::anyhow!("bad rank in {}: {e}", path.display()))?;
        ranks.insert(bytes, rank);
    }
    if ranks.is_empty() {
        anyhow::bail!("no ranks parsed from {}", path.display());
    }
    Ok(ranks)
}

/// Read `added_tokens_decoder` (id -> {"content": name}) from tokenizer_config.json.
fn load_special_names(path: &Path) -> HashMap<Rank, String> {
    let mut names = HashMap::new();
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return names,
    };
    let cfg: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return names,
    };
    if let Some(Value::Object(map)) = cfg.get("added_tokens_decoder") {
        for (id_str, entry) in map {
            if let (Ok(id), Some(content)) = (
                id_str.parse::<Rank>(),
                entry.get("content").and_then(|c| c.as_str()),
            ) {
                names.insert(id, content.to_string());
            }
        }
    }
    names
}

/// Escape a literal for use inside an Oniguruma alternation.
fn escape_regex(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for c in s.chars() {
        if "\\^$.|?*+()[]{}".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // The Kimi model dir (mounted on the perf hosts). Skipped gracefully when
    // absent so the suite still runs on a machine without the weights.
    const KIMI_DIR: &str = "/mnt/vast/john/huggingface/amd-Kimi-K2.6-MXFP4";

    fn load() -> Option<KimiTokenizer> {
        let p = Path::new(KIMI_DIR);
        if !p.join("tiktoken.model").exists() {
            eprintln!("skip: {KIMI_DIR} not present");
            return None;
        }
        Some(KimiTokenizer::load(p).expect("load kimi tokenizer"))
    }

    #[test]
    fn matches_python_ground_truth() {
        let Some(tk) = load() else { return };
        // Generated by tiktoken.Encoding(...).encode(s, allowed_special="all")
        // against this exact tiktoken.model (see conversation log).
        let cases: &[(&str, &[Rank])] = &[
            ("Hello world", &[19180, 2695]),
            ("The quick brown fox.", &[1008, 5072, 16331, 69275, 13]),
            ("你好,世界!机器学习", &[33845, 11, 2243, 0, 101773]),
            (
                "def foo(x): return x+1",
                &[1166, 21633, 3940, 3118, 745, 1288, 10, 16],
            ),
            (
                "  spaces   and\ttabs\n",
                &[220, 14803, 256, 316, 5604, 5609, 198],
            ),
        ];
        for (text, want) in cases {
            assert_eq!(&tk.encode(text), want, "encode({text:?})");
        }
    }

    #[test]
    fn special_token_literal_maps_to_its_id() {
        let Some(tk) = load() else { return };
        // "<|im_end|>" is id 163586 (num_base 163584 + 2).
        let ids = tk.encode("<|im_end|>");
        assert_eq!(ids, vec![163586]);
        // Surrounded by text: special id sits between the two halves.
        let ids = tk.encode("a<|im_end|>b");
        assert!(ids.contains(&163586), "got {ids:?}");
    }

    #[test]
    fn empty_and_ascii_roundtrip_shapes() {
        let Some(tk) = load() else { return };
        assert!(tk.encode("").is_empty());
        // Deterministic + non-empty for ordinary text.
        assert!(!tk.encode("hello").is_empty());
        assert_eq!(tk.encode("hello"), tk.encode("hello"));
    }
}
