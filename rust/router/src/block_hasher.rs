///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Tokenize a request and chain block hashes — the Rust twin of
//! `infera.router.kv_event.block_hasher`.
//!
//! The router's token ids must match the serving engine's, or every block hash
//! diverges and cache lookups always miss. We load the model's HF *fast*
//! tokenizer (`tokenizer.json`) via the `tokenizers` crate — the Rust
//! equivalent of Python's `AutoTokenizer` fast path. For a chat request we
//! render the model's Jinja `chat_template` (from `tokenizer_config.json`) with
//! minijinja; for a raw `prompt` we tokenize it directly.
//!
//! Any tokenisation failure degrades to "no cache info" (empty hashes) so the
//! cost function falls back to load-only routing — never a 500 — exactly like
//! the Python side.

use std::path::Path;

use minijinja::{context, Environment};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::hasher::hash_request;
use crate::tiktoken::KimiTokenizer;

pub struct BlockHasher {
    tokenizer: Option<Tokenizer>,
    /// Kimi-style tiktoken tokenizer (no `tokenizer.json`). Takes precedence
    /// over `tokenizer` for the text path when present.
    tiktoken: Option<KimiTokenizer>,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl BlockHasher {
    /// Load from a model dir (containing `tokenizer.json` [+ `tokenizer_config.json`])
    /// or a direct `tokenizer.json` path. A missing/unloadable tokenizer yields a
    /// hasher that always returns empty (load-only routing), never an error.
    pub fn load(path: &str) -> Self {
        let p = Path::new(path);
        let tok_json = if p.is_dir() {
            p.join("tokenizer.json")
        } else {
            p.to_path_buf()
        };
        let tokenizer = match Tokenizer::from_file(&tok_json) {
            Ok(t) => {
                tracing::info!(path = %tok_json.display(), "kv-aware: loaded tokenizer");
                Some(t)
            }
            Err(e) => {
                tracing::warn!(path = %tok_json.display(), err = %e, "kv-aware: tokenizer load failed; cache locality disabled");
                None
            }
        };

        // chat_template + bos/eos live in tokenizer_config.json alongside.
        let cfg_dir = if p.is_dir() {
            p.to_path_buf()
        } else {
            p.parent().map(|d| d.to_path_buf()).unwrap_or_default()
        };

        // Kimi (and other tiktoken-only models) have no `tokenizer.json` but do
        // ship a `tiktoken.model` — load it so text prompts still tokenize.
        let tiktoken = if cfg_dir.join("tiktoken.model").exists() {
            match KimiTokenizer::load(&cfg_dir) {
                Ok(t) => {
                    tracing::info!(dir = %cfg_dir.display(), "kv-aware: loaded tiktoken tokenizer");
                    Some(t)
                }
                Err(e) => {
                    tracing::warn!(dir = %cfg_dir.display(), err = %e, "kv-aware: tiktoken load failed");
                    None
                }
            }
        } else {
            None
        };

        let (chat_template, bos_token, eos_token) =
            load_config(&cfg_dir.join("tokenizer_config.json"));
        // Kimi ships its chat template as a standalone `chat_template.jinja`
        // (NOT embedded in tokenizer_config.json), so without this fallback a
        // chat request renders empty -> no tokens -> load-only routing (zero
        // cache locality). Prefer the embedded one when present.
        let chat_template = chat_template.or_else(|| {
            let p = cfg_dir.join("chat_template.jinja");
            match std::fs::read_to_string(&p) {
                Ok(s) => {
                    tracing::info!(path = %p.display(), "kv-aware: loaded standalone chat_template.jinja");
                    Some(s)
                }
                Err(_) => None,
            }
        });

        BlockHasher {
            tokenizer,
            tiktoken,
            chat_template,
            bos_token,
            eos_token,
        }
    }

    /// A no-op hasher (no tokenizer) — used when `--kv-tokenizer-path` is unset.
    pub fn disabled() -> Self {
        BlockHasher {
            tokenizer: None,
            tiktoken: None,
            chat_template: None,
            bos_token: None,
            eos_token: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.tokenizer.is_some() || self.tiktoken.is_some()
    }

    /// Chained block hashes for a request body, or empty on any failure.
    pub fn hash_for(&self, body: &Value, block_size: usize) -> Vec<u64> {
        if block_size == 0 {
            return Vec::new();
        }
        // Fast path: the request already carries token ids (`prompt` as an array
        // of ints — the OpenAI-legal pre-tokenized form the engines accept
        // verbatim). Hash them directly: no tokenizer, and the ids match the
        // engine's kv-event token ids byte-for-byte, so there's zero tokenizer
        // mismatch. This is how tiktoken-only models (Kimi) are routed, since
        // the HF `tokenizers` crate can't load their vocab.
        if let Some(ids) = token_ids_from_prompt(body) {
            return hash_request(&ids, block_size);
        }
        // Text path: needs a loaded tokenizer (HF fast or Kimi tiktoken).
        if self.tokenizer.is_none() && self.tiktoken.is_none() {
            return Vec::new();
        }
        let text = match self.render_text(body) {
            Some(t) => t,
            None => return Vec::new(),
        };
        // Kimi tiktoken takes precedence — it reproduces the engine's ids for a
        // model the HF `tokenizers` crate can't load.
        if let Some(tk) = &self.tiktoken {
            return hash_request(&tk.encode(&text), block_size);
        }
        let tok = self.tokenizer.as_ref().expect("checked above");
        // add_special_tokens=false: the template/prompt already carries any
        // leading special token as text (matches how the engines tokenize).
        match tok.encode(text, false) {
            Ok(enc) => hash_request(enc.get_ids(), block_size),
            Err(e) => {
                tracing::warn!(err = %e, "kv-aware: tokenisation failed");
                Vec::new()
            }
        }
    }

    fn render_text(&self, body: &Value) -> Option<String> {
        if let Some(messages) = body.get("messages") {
            if messages.is_array() {
                return self.apply_chat_template(messages);
            }
        }
        // Completion `prompt`: only a plain string is tokenizable here.
        if let Some(prompt) = body.get("prompt") {
            if let Some(s) = prompt.as_str() {
                return Some(s.to_string());
            }
        }
        None
    }

    fn apply_chat_template(&self, messages: &Value) -> Option<String> {
        let template = self.chat_template.as_ref()?;
        let mut env = Environment::new();
        // HF chat templates use the Python dict method `msg.get('key')` (7x in
        // Kimi's), which minijinja lacks natively -> the template errors -> empty
        // render -> no cache locality for chat. Supply just `.get(key[,default])`.
        env.set_unknown_method_callback(|_state, value, method, args| {
            use minijinja::{Error, ErrorKind, Value};
            if method == "get" {
                let key = args.first().cloned().unwrap_or(Value::UNDEFINED);
                let default = args.get(1).cloned().unwrap_or_else(|| Value::from(()));
                return Ok(match value.get_item(&key) {
                    Ok(v) if !v.is_undefined() => v,
                    _ => default,
                });
            }
            Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("object has no method {method}"),
            ))
        });
        // HF templates call raise_exception(msg) on malformed input.
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        if let Err(e) = env.add_template("chat", template) {
            tracing::warn!(err = %e, "kv-aware: chat_template parse failed");
            return None;
        }
        let tmpl = env.get_template("chat").ok()?;
        match tmpl.render(context! {
            messages => messages,
            add_generation_prompt => true,
            bos_token => self.bos_token,
            eos_token => self.eos_token,
        }) {
            Ok(rendered) => Some(rendered),
            Err(e) => {
                tracing::warn!(err = %e, "kv-aware: chat_template render failed");
                None
            }
        }
    }
}

/// If `prompt` is a flat array of non-negative integers (pre-tokenized ids),
/// return them as `u32`s. Any non-int / negative / non-array shape → None
/// (fall through to the text tokenizer path).
fn token_ids_from_prompt(body: &Value) -> Option<Vec<u32>> {
    let arr = body.get("prompt")?.as_array()?;
    if arr.is_empty() {
        return None;
    }
    let mut ids = Vec::with_capacity(arr.len());
    for v in arr {
        let n = v.as_u64()?;
        ids.push(u32::try_from(n).ok()?);
    }
    Some(ids)
}

fn load_config(path: &Path) -> (Option<String>, Option<String>, Option<String>) {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return (None, None, None),
    };
    let cfg: Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => return (None, None, None),
    };
    // chat_template may be a string or (rarely) a list of {name, template}.
    let chat_template = match cfg.get("chat_template") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(arr)) => arr.iter().find_map(|e| {
            e.get("template")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        }),
        _ => None,
    };
    let tok_str = |v: Option<&Value>| -> Option<String> {
        match v {
            Some(Value::String(s)) => Some(s.clone()),
            // {"content": "<s>"} form
            Some(Value::Object(m)) => m
                .get("content")
                .and_then(|c| c.as_str())
                .map(|s| s.to_string()),
            _ => None,
        }
    };
    (
        chat_template,
        tok_str(cfg.get("bos_token")),
        tok_str(cfg.get("eos_token")),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn disabled_hasher_returns_empty() {
        let h = BlockHasher::disabled();
        assert!(!h.is_enabled());
        assert!(h.hash_for(&json!({"prompt": "hello world"}), 4).is_empty());
    }

    #[test]
    fn missing_tokenizer_path_is_graceful() {
        let h = BlockHasher::load("/nonexistent/path/tokenizer.json");
        assert!(!h.is_enabled());
        assert!(h.hash_for(&json!({"prompt": "x"}), 4).is_empty());
    }

    #[test]
    fn token_id_prompt_hashes_without_a_tokenizer() {
        // The tiktoken/pre-tokenized path: even with no tokenizer, an integer
        // `prompt` hashes directly and matches hash_request on the same ids.
        let h = BlockHasher::disabled();
        let got = h.hash_for(&json!({"prompt": [1, 2, 3, 4, 5, 6, 7, 8]}), 4);
        assert_eq!(
            got,
            crate::hasher::hash_request(&[1, 2, 3, 4, 5, 6, 7, 8], 4)
        );
        assert_eq!(got.len(), 2);
        // a string prompt with no tokenizer still degrades to empty
        assert!(h.hash_for(&json!({"prompt": "text"}), 4).is_empty());
    }

    // Kimi ships chat_template.jinja (not embedded) + tiktoken. This guards the
    // whole chat path: standalone-template fallback, the `.get()` method, and the
    // `{% break %}` loop control all have to work or a chat request hashes to
    // empty (load-only routing, no cache locality). Skips if weights absent.
    #[test]
    fn kimi_chat_request_renders_and_hashes() {
        const KIMI_DIR: &str = "/mnt/vast/john/huggingface/amd-Kimi-K2.6-MXFP4";
        if !Path::new(KIMI_DIR).join("chat_template.jinja").exists() {
            eprintln!("skip: {KIMI_DIR} not present");
            return;
        }
        let h = BlockHasher::load(KIMI_DIR);
        assert!(h.is_enabled());
        // A multi-turn chat (incl. a tool message → exercises break/.get) must
        // render to a non-trivial token stream -> at least one 16-token block.
        let body = json!({"messages": [
            {"role": "system", "content": "You are an agent. ".repeat(40)},
            {"role": "user", "content": "list the files ".repeat(40)},
            {"role": "assistant", "content": null,
             "tool_calls": [{"id": "c1", "type": "function",
                             "function": {"name": "ls", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "a.txt b.txt ".repeat(40)},
            {"role": "user", "content": "summarize"},
        ]});
        assert!(
            !h.hash_for(&body, 16).is_empty(),
            "kimi chat template must render + tokenize (else 0 cache locality)"
        );
    }
}
