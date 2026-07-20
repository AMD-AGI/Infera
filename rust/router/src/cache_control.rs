///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Parse client cache-control hints from a chat/completion body — the Rust twin
//! of `infera.router.cache_control`.
//!
//! Two shapes are accepted. Anthropic-style: `system[]/tools[]/messages[].content[]`
//! blocks carrying `cache_control: {type: "ephemeral", ttl?: "1h"|"5m"}`. OpenAI-style:
//! `prompt_cache_retention` ("24h"/"1h"/"5m") plus `prompt_cache_key`.
//! Overall retention is the MAX across all hinted blocks (none < short < long).
//! Never panics on malformed bodies — treats them as no-hint.

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Retention {
    // Ordinal order matters: None < Short < Long so `max` gives the strongest.
    None,
    Short,
    Long,
}

impl Retention {
    pub fn as_str(self) -> &'static str {
        match self {
            Retention::None => "none",
            Retention::Short => "short",
            Retention::Long => "long",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CacheHints {
    pub retention: Retention,
    pub session_id: Option<String>,
    /// Were ANY blocks explicitly tagged? Distinguishes "implicit short" (no
    /// hint at all) from "client said short" (deliberate ephemeral).
    pub explicit_hint_seen: bool,
    /// True if the body carries image/audio/video/file blocks. The router-side
    /// hasher is text-only, so MM requests must fall back to pure load balance.
    pub has_multimodal_content: bool,
}

impl CacheHints {
    fn none(session_id: Option<String>, has_mm: bool) -> Self {
        CacheHints {
            retention: Retention::None,
            session_id,
            explicit_hint_seen: false,
            has_multimodal_content: has_mm,
        }
    }
}

/// Content-block types that signal non-text input (permissive on purpose).
const MM_BLOCK_TYPES: &[&str] = &[
    "image",
    "image_url",
    "input_audio",
    "document",
    "file",
    "video",
    "audio",
];

pub fn parse_cache_hints(body: &Value) -> CacheHints {
    let obj = match body.as_object() {
        Some(o) => o,
        None => return CacheHints::none(None, false),
    };

    let openai_retention = parse_openai_retention(obj);
    let openai_session = obj
        .get("prompt_cache_key")
        .or_else(|| obj.get("session_id"))
        .and_then(coerce_session_id);

    let (anthropic_retention, anthropic_seen) = scan_anthropic(obj);
    let has_mm = detect_multimodal(obj);

    let mut candidates: Vec<Retention> = Vec::new();
    if let Some(r) = openai_retention {
        candidates.push(r);
    }
    if let Some(r) = anthropic_retention {
        candidates.push(r);
    }

    if candidates.is_empty() {
        return CacheHints::none(openai_session, has_mm);
    }
    let retention = candidates.into_iter().max().unwrap();
    CacheHints {
        retention,
        session_id: openai_session,
        explicit_hint_seen: anthropic_seen || openai_retention.is_some(),
        has_multimodal_content: has_mm,
    }
}

fn coerce_session_id(v: &Value) -> Option<String> {
    match v.as_str() {
        Some(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        _ => None,
    }
}

fn parse_openai_retention(obj: &serde_json::Map<String, Value>) -> Option<Retention> {
    if let Some(r) = obj.get("prompt_cache_retention").and_then(|v| v.as_str()) {
        let r = r.trim().to_lowercase();
        return match r.as_str() {
            "24h" | "1h" | "long" => Some(Retention::Long),
            "5m" | "short" => Some(Retention::Short),
            "none" | "off" | "disabled" => Some(Retention::None),
            _ => None,
        };
    }
    // `prompt_cache_key` alone (no retention) => implicit short.
    if obj
        .get("prompt_cache_key")
        .map(|v| !v.is_null())
        .unwrap_or(false)
    {
        return Some(Retention::Short);
    }
    None
}

/// Scan Anthropic `system`/`tools`/`messages[].content` for `cache_control`.
/// Returns (max retention, any_block_seen).
fn scan_anthropic(obj: &serde_json::Map<String, Value>) -> (Option<Retention>, bool) {
    let mut found: Vec<Retention> = Vec::new();
    for block in iter_blocks_with_cache_control(obj) {
        if let Some(r) = retention_for_block(block) {
            found.push(r);
        }
    }
    match found.into_iter().max() {
        Some(r) => (Some(r), true),
        None => (None, false),
    }
}

fn iter_blocks_with_cache_control(obj: &serde_json::Map<String, Value>) -> Vec<&Value> {
    let mut out: Vec<&Value> = Vec::new();
    let has_cc = |v: &&Value| {
        v.as_object()
            .map(|m| m.contains_key("cache_control"))
            .unwrap_or(false)
    };

    if let Some(list) = obj.get("system").and_then(|v| v.as_array()) {
        out.extend(list.iter().filter(has_cc));
    }
    if let Some(list) = obj.get("tools").and_then(|v| v.as_array()) {
        out.extend(list.iter().filter(has_cc));
    }
    if let Some(msgs) = obj.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            if let Some(content) = msg.get("content").and_then(|v| v.as_array()) {
                out.extend(content.iter().filter(has_cc));
            }
        }
    }
    out
}

fn retention_for_block(block: &Value) -> Option<Retention> {
    let bt = block.get("type").and_then(|v| v.as_str());
    // thinking blocks are never cached even if a buggy client tags them.
    if matches!(bt, Some("thinking") | Some("redacted_thinking")) {
        return Some(Retention::None);
    }
    let cc = block.get("cache_control")?.as_object()?;
    if cc.get("type").and_then(|v| v.as_str()) != Some("ephemeral") {
        return None;
    }
    if let Some(ttl) = cc.get("ttl").and_then(|v| v.as_str()) {
        let ttl = ttl.trim().to_lowercase();
        if matches!(ttl.as_str(), "1h" | "1hr" | "60m" | "3600s" | "long") {
            return Some(Retention::Long);
        }
        if matches!(ttl.as_str(), "5m" | "5min" | "300s" | "short") {
            return Some(Retention::Short);
        }
    }
    // ephemeral with no ttl: Anthropic default is 5 min => short.
    Some(Retention::Short)
}

fn detect_multimodal(obj: &serde_json::Map<String, Value>) -> bool {
    let is_mm_block = |b: &Value| {
        b.get("type")
            .and_then(|v| v.as_str())
            .map(|t| MM_BLOCK_TYPES.contains(&t))
            .unwrap_or(false)
    };

    if let Some(msgs) = obj.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            if let Some(content) = msg.get("content").and_then(|v| v.as_array()) {
                if content.iter().any(is_mm_block) {
                    return true;
                }
            }
        }
    }
    if obj
        .get("images")
        .and_then(|v| v.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false)
    {
        return true;
    }
    if let Some(audio) = obj.get("audio") {
        if audio.is_object() || audio.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            return true;
        }
    }
    if let Some(system) = obj.get("system").and_then(|v| v.as_array()) {
        if system.iter().any(is_mm_block) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn ordering_none_short_long() {
        assert!(Retention::None < Retention::Short);
        assert!(Retention::Short < Retention::Long);
        assert_eq!(
            [Retention::Short, Retention::Long, Retention::None]
                .into_iter()
                .max()
                .unwrap(),
            Retention::Long
        );
    }

    #[test]
    fn no_hint_is_none() {
        let h = parse_cache_hints(&json!({"model": "m", "prompt": "hi"}));
        assert_eq!(h.retention, Retention::None);
        assert!(!h.explicit_hint_seen);
        assert!(!h.has_multimodal_content);
    }

    #[test]
    fn openai_retention_and_key() {
        let h = parse_cache_hints(
            &json!({"prompt_cache_retention": "24h", "prompt_cache_key": "sess-1"}),
        );
        assert_eq!(h.retention, Retention::Long);
        assert_eq!(h.session_id.as_deref(), Some("sess-1"));
        assert!(h.explicit_hint_seen);

        // key alone => implicit short
        let h = parse_cache_hints(&json!({"prompt_cache_key": "s"}));
        assert_eq!(h.retention, Retention::Short);
    }

    #[test]
    fn anthropic_max_across_blocks() {
        let body = json!({
            "system": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "y", "cache_control": {"type": "ephemeral"}}
            ]}]
        });
        let h = parse_cache_hints(&body);
        assert_eq!(h.retention, Retention::Long); // 1h(system) beats ephemeral-default(short)
        assert!(h.explicit_hint_seen);
    }

    #[test]
    fn multimodal_detected() {
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {}}, {"type": "text", "text": "what is this"}
        ]}]});
        let h = parse_cache_hints(&body);
        assert!(h.has_multimodal_content);
    }
}
