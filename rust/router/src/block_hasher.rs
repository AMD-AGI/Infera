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

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU8, Ordering};

use minijinja::Environment;
#[cfg(test)]
use minijinja::context;
use serde::Serialize;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::encoding_dsv4::{encode_messages, DSV4_THINKING_MODE};
use crate::hasher::hash_request;
use crate::tiktoken::KimiTokenizer;

/// `json.dumps` default separators: `", "` between items, `": "` after a key.
/// serde_json packs both, and the engine's prompt has the spaces.
pub(crate) struct PyJsonFormatter;

impl serde_json::ser::Formatter for PyJsonFormatter {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            w.write_all(b", ")
        }
    }
    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }
}

/// Python's `ensure_ascii=True`: every non-ASCII scalar becomes `\uXXXX`, and
/// anything above the BMP becomes a surrogate pair.
fn escape_non_ascii(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0u16; 2]) {
                // write! rather than push_str(&format!(..)): straight into the
                // buffer instead of a String per code unit. Infallible for String.
                let _ = write!(out, "\\u{unit:04x}");
            }
        }
    }
    out
}

/// A rendered prompt and the `add_special_tokens` flag the engine tokenizes it
/// with. Carrying the flag alongside the text keeps the two in step: picking it
/// at the encode site instead would mean re-deciding which renderer produced
/// the string.
struct Rendered {
    text: String,
    add_special_tokens: bool,
}

impl Rendered {
    /// Text that already spells out its own special tokens.
    fn verbatim(text: String) -> Self {
        Self {
            text,
            add_special_tokens: false,
        }
    }

    /// Text the engine feeds to a plain `tokenizer.encode`.
    fn with_specials(text: String) -> Self {
        Self {
            text,
            add_special_tokens: true,
        }
    }
}

/// `tracing::warn!`, but only the 1st and then every 1000th occurrence.
///
/// Every caller below sits in `hash_for`, which runs on *every* routing
/// decision, and each condition they report is steady state rather than
/// transient: a Codex client with `store: true` sends `previous_response_id`
/// on every turn after the first, so an unthrottled line here is one WARN per
/// request for the life of the session. That is the same pathology as the
/// kv_block_size ERROR this router used to print on every startup -- a line
/// that is always there is a line nobody reads. The count rides along so a
/// throttled line still says how big the problem is.
macro_rules! warn_throttled {
    ($($arg:tt)*) => {{
        static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if n == 1 || n % 1000 == 0 {
            tracing::warn!(occurrences = n, $($arg)*);
        }
    }};
}

/// Has this model's chat template been observed to need OpenAI tool-call
/// `arguments` strings parsed into objects? See `BlockHasher::tool_args`.
const TOOL_ARGS_UNDECIDED: u8 = 0;
const TOOL_ARGS_VERBATIM: u8 = 1;
const TOOL_ARGS_PARSED: u8 = 2;

pub struct BlockHasher {
    tokenizer: Option<Tokenizer>,
    /// Kimi-style tiktoken tokenizer (no `tokenizer.json`). Takes precedence
    /// over `tokenizer` for the text path when present.
    tiktoken: Option<KimiTokenizer>,
    chat_template: Option<String>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    /// This model speaks DeepSeek-V4's native chat encoding instead of a Jinja
    /// `chat_template` (see `render_dsv4`).
    dsv4: bool,
    /// This model speaks Kimi-K3's native XTML chat encoding instead of a Jinja
    /// `chat_template` (see `crate::encoding_k3`).
    kimi_k3: bool,
    /// Does this model's chat template want OpenAI's tool-call `arguments`
    /// JSON *string* parsed back into an object before it will render?
    ///
    /// Both answers are live templates on this fleet and they want opposite
    /// things:
    ///
    /// * Qwen3 -- `{%- if tool_call.arguments is string %}{{ tool_call.arguments }}`
    ///   emits the client's bytes verbatim. Hand it an object instead and it
    ///   takes the `| tojson` branch, which re-serialises with `json.dumps`
    ///   separators; every block after the first tool call then diverges from
    ///   the engine's.
    /// * GLM-5.3 -- `{%- for k, v in _args.items() %}` raises on a string, the
    ///   render returns None, and the router loses the whole prompt.
    ///
    /// So the answer is settled by observation rather than by model name:
    /// render verbatim, and only if *that* fails retry with the arguments
    /// parsed. The first tool-carrying request decides it; steady state is one
    /// render either way. `Ordering::Relaxed` throughout -- a racing pair of
    /// requests may both probe, and the worst case is one extra render.
    tool_args: AtomicU8,
}

/// Qwen3's chat template, verbatim from Qwen3-30B-A3B's tokenizer_config.json.
/// Kept whole rather than excerpted: the assertion below is the exact output
/// transformers produces, and trimming the template would only prove that a
/// trimmed template renders.
#[cfg(test)]
const QWEN3_TEMPLATE: &str = r###"{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
{%- for message in messages %}
    {%- if message.content is string %}
        {%- set content = message.content %}
    {%- else %}
        {%- set content = '' %}
    {%- endif %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
                {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
                {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
                {%- if (loop.first and content) or (not loop.first) %}
                    {{- '\n' }}
                {%- endif %}
                {%- if tool_call.function %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '<tool_call>\n{"name": "' }}
                {{- tool_call.name }}
                {{- '", "arguments": ' }}
                {%- if tool_call.arguments is string %}
                    {{- tool_call.arguments }}
                {%- else %}
                    {{- tool_call.arguments | tojson }}
                {%- endif %}
                {{- '}\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}"###;

/// Python's data model, as far as HF chat templates lean on it.
///
/// These templates are written for Jinja2 running on Python, so they call
/// methods minijinja has no notion of. A missing one is not a loud failure:
/// the template errors, the render comes back empty, the prompt hashes to
/// nothing, and kv-aware quietly becomes load-only routing while every health
/// signal stays green. Kimi's needs `msg.get('key')` (7x), GLM-5.2's needs
/// `content.strip()` and `content.split('</think>')[-1]`, Qwen3's needs
/// `startswith`/`endswith`.
fn unknown_method(
    _state: &minijinja::State,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::value::ValueKind;
    use minijinja::{Error, ErrorKind, Value};
    let as_text = |v: &Value| -> Result<String, Error> {
        v.as_str().map(str::to_owned).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("{method}() expects a string, got {}", v.kind()),
            )
        })
    };
    match method {
        "get" => {
            let key = args.first().cloned().unwrap_or(Value::UNDEFINED);
            let default = args.get(1).cloned().unwrap_or_else(|| Value::from(()));
            Ok(match value.get_item(&key) {
                Ok(v) if !v.is_undefined() => v,
                _ => default,
            })
        }
        // Python strips whitespace with no argument and the given
        // character SET (not substring) with one.
        "strip" | "lstrip" | "rstrip" => {
            let s = as_text(value)?;
            let chars: Option<Vec<char>> = match args.first() {
                Some(a) if !a.is_undefined() && !a.is_none() => Some(as_text(a)?.chars().collect()),
                _ => None,
            };
            let matches = |c: char| match &chars {
                Some(set) => set.contains(&c),
                None => c.is_whitespace(),
            };
            Ok(Value::from(match method {
                "lstrip" => s.trim_start_matches(matches).to_owned(),
                "rstrip" => s.trim_end_matches(matches).to_owned(),
                _ => s.trim_matches(matches).to_owned(),
            }))
        }
        // `s.split(sep)` keeps empty fields, so "a<t>b".split("<t>") is
        // ["a", "b"] and "".split("<t>") is [""] -- str::split agrees.
        // Bare `s.split()` is the different, whitespace-collapsing form.
        "split" => {
            let s = as_text(value)?;
            let parts: Vec<String> = match args.first() {
                Some(a) if !a.is_undefined() && !a.is_none() => {
                    let sep = as_text(a)?;
                    if sep.is_empty() {
                        return Err(Error::new(
                            ErrorKind::InvalidOperation,
                            "split() with an empty separator",
                        ));
                    }
                    s.split(sep.as_str()).map(str::to_owned).collect()
                }
                _ => s.split_whitespace().map(str::to_owned).collect(),
            };
            Ok(Value::from(parts))
        }
        // Python's dict views. minijinja iterates a map as its keys, so
        // the other two are built from that.
        "items" | "keys" | "values" => {
            if value.kind() != ValueKind::Map {
                return Err(Error::new(
                    ErrorKind::InvalidOperation,
                    format!("{method}() expects a mapping, got {}", value.kind()),
                ));
            }
            let at = |k: &Value| value.get_item(k).unwrap_or(Value::UNDEFINED);
            let keys = value.try_iter()?;
            Ok(Value::from(match method {
                "keys" => keys.collect::<Vec<_>>(),
                "values" => keys.map(|k| at(&k)).collect(),
                _ => keys
                    .map(|k| {
                        let v = at(&k);
                        Value::from(vec![k, v])
                    })
                    .collect(),
            }))
        }
        // Python takes either one candidate or a tuple of them, and an empty
        // needle always matches.
        "startswith" | "endswith" => {
            let s = as_text(value)?;
            let needle = args.first().cloned().unwrap_or(Value::UNDEFINED);
            let candidates: Vec<String> = match needle.kind() {
                ValueKind::Seq => needle
                    .try_iter()?
                    .map(|v| as_text(&v))
                    .collect::<Result<_, _>>()?,
                _ => vec![as_text(&needle)?],
            };
            let hit = candidates.iter().any(|c| {
                if method == "startswith" {
                    s.starts_with(c.as_str())
                } else {
                    s.ends_with(c.as_str())
                }
            });
            Ok(Value::from(hit))
        }
        _ => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("object has no method {method}"),
        )),
    }
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
            dsv4: detect_dsv4(&cfg_dir),
            kimi_k3: detect_kimi_k3(&cfg_dir),
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
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
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
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
        // `/v1/responses` carries the conversation in `input`; every renderer
        // below keys off `messages`/`prompt` and would hash nothing at all.
        // Normalise it into the chat body `OpenAIServingResponses._make_request`
        // builds, once, here — so the encoders stay single-schema and the two
        // endpoints render through identical code.
        // Captured before the shadowing below: once a Responses body is
        // normalised, `top_level_keys` on it reports the `messages`/`tools`
        // this router invented, and the `input`/`instructions`/`reasoning` the
        // client actually sent -- the schema a reader needs to see -- is gone
        // from the log line that exists to identify it.
        let client_keys = top_level_keys(body);
        let normalized;
        let body = if crate::responses_input::is_responses_body(body) {
            match crate::responses_input::to_chat_body(body) {
                Some(b) => {
                    normalized = b;
                    &normalized
                }
                None => {
                    warn_throttled!(
                        "kv-aware: /v1/responses body cannot be reproduced (a \
                         `previous_response_id` whose history lives in the engine, or an \
                         input item this port does not model); routing on load for this request"
                    );
                    return Vec::new();
                }
            }
        } else {
            body
        };
        // Kimi-K3 renders chat in imperative Python and ships no Jinja template
        // at all, so the chat-template path below has nothing to apply and every
        // chat request would hash to nothing. Encode it natively instead. The
        // encoder refuses (returns None) anything it can't reproduce exactly,
        // which falls through to that same empty render — never worse.
        if self.kimi_k3 {
            if let Some(tk) = &self.tiktoken {
                if let Some(segments) = crate::encoding_k3::encode_chat(body) {
                    return hash_request(&tk.encode_segments(&segments), block_size);
                }
            }
        }
        let Rendered {
            text,
            add_special_tokens,
        } = match self.render_text(body) {
            Some(t) => t,
            // Every other degradation in this file says so; this one used to be
            // the exception, and it is the one that fires on a whole endpoint at
            // a time. `policy="kv-aware"` keeps logging normally either way, so
            // silence here reads as a healthy router with an inexplicable 0% hit
            // rate — which is exactly how the `/v1/responses` outage hid.
            None => {
                warn_throttled!(
                    keys = ?client_keys,
                    "kv-aware: no tokenizable prompt in this request body (no `messages`, \
                     no string `prompt`); routing on load for this request"
                );
                return Vec::new();
            }
        };
        // Kimi tiktoken takes precedence — it reproduces the engine's ids for a
        // model the HF `tokenizers` crate can't load.
        if let Some(tk) = &self.tiktoken {
            // ...but only for text the engine also tokenizes plainly. This
            // encoder adds no BOS/EOS by construction (see `crate::tiktoken`),
            // so it cannot reproduce what a tokenizer asked for specials would
            // emit, and a prefix one token short of the engine's does not error
            // — it silently never hits the cache again. Refusing costs the same
            // load-only routing an empty render already gives, and says so.
            //
            // Unreachable today: the only `add_special_tokens` render is the
            // dsv4 encoder's, and dsv4 and tiktoken are different model
            // families. It is guarded rather than asserted because the next
            // native encoder added here would otherwise inherit the bug.
            if add_special_tokens {
                tracing::warn!(
                    "kv-aware: this prompt is tokenized with special tokens, which the \
                     tiktoken encoder cannot reproduce; routing on load for this request"
                );
                return Vec::new();
            }
            return hash_request(&tk.encode(&text), block_size);
        }
        let tok = self.tokenizer.as_ref().expect("checked above");
        match tok.encode(text, add_special_tokens) {
            Ok(enc) => hash_request(enc.get_ids(), block_size),
            Err(e) => {
                tracing::warn!(err = %e, "kv-aware: tokenisation failed");
                Vec::new()
            }
        }
    }

    /// A prompt string plus the `add_special_tokens` the engine tokenizes it
    /// with. The chat-template and completion paths carry any leading special
    /// token as text already, so a tokenizer that prepends BOS would double it;
    /// the dsv4 encoder's output is tokenized plainly instead.
    fn render_text(&self, body: &Value) -> Option<Rendered> {
        if let Some(messages) = body.get("messages") {
            if let Some(arr) = messages.as_array() {
                if self.dsv4 {
                    if let Some(rendered) = self.render_dsv4(body, arr) {
                        // serving_chat tokenizes the dsv4 encoder's output with a
                        // plain `tokenizer.encode(real_input)`, unlike the
                        // chat-template site which passes add_special_tokens=False.
                        return Some(Rendered::with_specials(rendered));
                    }
                }
                return self
                    .apply_chat_template(body, messages)
                    .map(Rendered::verbatim);
            }
        }
        // Completion `prompt`: only a plain string is tokenizable here.
        if let Some(prompt) = body.get("prompt") {
            if let Some(s) = prompt.as_str() {
                return Some(Rendered::verbatim(s.to_string()));
            }
        }
        None
    }

    #[cfg(test)]
    fn render_text_str(&self, body: &Value) -> Option<String> {
        self.render_text(body).map(|r| r.text)
    }

    /// Render a chat body with DeepSeek-V4's native encoder.
    ///
    /// DSv4 ships no `chat_template` and no `chat_template.jinja` — only an
    /// `encoding/` directory — so `apply_chat_template` returns None and every
    /// chat request against a DSv4 worker hashes to nothing. That failure is
    /// silent: no error, just a permanent 0% kv-aware hit rate. Returning None
    /// here leaves the caller on the previous path, so anything this cannot
    /// reproduce exactly is no worse than today.
    fn render_dsv4(&self, body: &Value, messages: &[Value]) -> Option<String> {
        // The engine renders tools from a full pydantic `Tool.model_dump()`,
        // which materialises defaults the client never sent (`strict`,
        // `defer_loading`). There is no pydantic model to dump here, and a
        // prefix a few tokens off misses silently, so skip rather than guess —
        // the same call the Python side makes when it cannot import that model.
        if body
            .get("tools")
            .is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()))
        {
            tracing::debug!("kv-aware: dsv4 encoder skipped, request carries tools");
            return None;
        }
        // serving_chat prepends an empty system message when the conversation
        // does not start with one; tools are hung off that message, never
        // passed to the encoder directly.
        let mut msgs: Vec<Value> = Vec::with_capacity(messages.len() + 1);
        if messages
            .first()
            .is_some_and(|m| m.get("role").and_then(Value::as_str) != Some("system"))
        {
            msgs.push(serde_json::json!({"role": "system", "content": ""}));
        }
        msgs.extend(messages.iter().cloned());

        let rendered = encode_messages(&msgs, DSV4_THINKING_MODE);
        if rendered.is_none() {
            tracing::warn!("kv-aware: dsv4 encoder could not render this conversation");
        }
        rendered
    }

    fn apply_chat_template(&self, body: &Value, messages: &Value) -> Option<String> {
        let template = self.chat_template.as_ref()?;
        let mut env = Environment::new();
        // transformers renders with both of these on. minijinja defaults them off,
        // which leaves a newline after every `{% %}` that is alone on its line --
        // invisible in the plain chat path, which trims explicitly with `{%- -%}`,
        // and two stray newlines in GLM-5.2's tool-call branch, which does not.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(unknown_method);
        // transformers does not use jinja2's tojson either -- it installs
        // `json.dumps(x, ensure_ascii=False, sort_keys=False)`, explicitly to stop
        // HTML characters being escaped. minijinja's builtin escapes `<`, `>` and
        // `&` and packs the separators, so both of them would move the token
        // stream away from the engine's.
        env.add_filter(
            "tojson",
            |v: minijinja::Value, kwargs: minijinja::value::Kwargs| {
                use minijinja::{Error, ErrorKind};
                let ascii = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
                // Ignoring an argument silently would only shift the divergence
                // somewhere harder to see; failing outright would blind the
                // router completely, which is worse.
                if let Err(e) = kwargs.assert_all_used() {
                    tracing::warn!(err = %e, "kv-aware: tojson() argument ignored");
                }
                let mut buf = Vec::new();
                v.serialize(&mut serde_json::Serializer::with_formatter(
                    &mut buf,
                    PyJsonFormatter,
                ))
                .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
                let out = String::from_utf8(buf)
                    .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
                Ok(minijinja::Value::from(if ascii {
                    escape_non_ascii(&out)
                } else {
                    out
                }))
            },
        );
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
        let base = self.template_context(body);
        let render = |msgs: &Value| -> Result<String, minijinja::Error> {
            let mut ctx = base.clone();
            ctx.insert("messages".to_string(), minijinja::Value::from_serialize(msgs));
            tmpl.render(ctx)
        };

        // Tool-call `arguments` policy -- see the `tool_args` field. Undecided
        // means "try the client's bytes first"; a template that cannot render
        // them says so by failing, and only then do we parse.
        let policy = self.tool_args.load(Ordering::Relaxed);
        if policy != TOOL_ARGS_PARSED {
            match render(messages) {
                Ok(out) => {
                    if policy == TOOL_ARGS_UNDECIDED {
                        self.tool_args.store(TOOL_ARGS_VERBATIM, Ordering::Relaxed);
                    }
                    return Some(out);
                }
                Err(e) if policy == TOOL_ARGS_VERBATIM => {
                    tracing::warn!(err = %e, "kv-aware: chat_template render failed");
                    return None;
                }
                Err(e) => {
                    tracing::debug!(
                        err = %e,
                        "kv-aware: chat_template render failed on verbatim tool-call \
                         arguments; retrying with them parsed"
                    );
                }
            }
        }
        let parsed = normalize_tool_call_arguments(messages);
        let parsed = parsed.as_ref().unwrap_or(messages);
        match render(parsed) {
            Ok(out) => {
                self.tool_args.store(TOOL_ARGS_PARSED, Ordering::Relaxed);
                Some(out)
            }
            Err(e) => {
                // Undecided on purpose: this render failed for some reason
                // other than the arguments shape, so it proves nothing about
                // which form the template wants.
                tracing::warn!(err = %e, "kv-aware: chat_template render failed");
                None
            }
        }
    }

    /// Everything `serving_chat` puts in the Jinja context besides `messages`.
    ///
    /// Reference, sglang 0.5.18 `OpenAIServingChat._process_messages`
    /// (serving_chat.py:1219 for `tools`, :1449 for the rest):
    ///
    /// ```python
    /// tools = [item.model_dump() for item in request.tools]   # see `chat_tools`
    /// extra_template_kwargs = {}
    /// if request.reasoning_effort is not None:
    ///     extra_template_kwargs["reasoning_effort"] = request.reasoning_effort
    /// if request.chat_template_kwargs:
    ///     extra_template_kwargs.update(request.chat_template_kwargs)
    /// apply_chat_template(msgs, add_generation_prompt=True, tools=tools,
    ///                     **extra_template_kwargs)
    /// ```
    ///
    /// Passing only `messages` was not a small omission. GLM-5.3's template
    /// reads five names and this supplied two of them, and the two it dropped
    /// both land in the *first* block of the prompt:
    ///
    /// ```jinja
    /// {%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined
    ///        and reasoning_effort in ['low', 'high'] else 'high' -%}
    /// {%- if ... -%}<|system|>Reasoning Effort: {{ effective_reasoning_effort | capitalize }}
    /// {%- if tools -%}<|system|>\n# Tools\n ...
    /// ```
    ///
    /// so a client sending `reasoning_effort="low"`, or any tools at all, put
    /// the router's block 0 somewhere the engine's never was -- and every
    /// block chains off block 0. Measured on the 1P1D GLM-5.3 fleet before
    /// this fix: `cache_hits=0` on all 5562 picks, 2765 of them with a
    /// non-empty `request_blocks`, and the router's own disagreement detector
    /// sitting at `streak=1024`.
    ///
    /// `bos_token`/`eos_token` are ours, not the engine's: transformers binds
    /// them from the tokenizer, and they are always defined (as none when the
    /// tokenizer has none) because a template may test `is defined`.
    ///
    /// Not modelled: `--default-chat-template-kwargs`, which the engine merges
    /// into `request.chat_template_kwargs` server-side (serving_chat.py:1196).
    /// The router is not told that flag, so a fleet using it still diverges;
    /// that needs a router-side channel for the same value, not a change here.
    fn template_context(&self, body: &Value) -> BTreeMap<String, minijinja::Value> {
        use minijinja::Value as JValue;
        let mut ctx: BTreeMap<String, JValue> = BTreeMap::new();
        ctx.insert("add_generation_prompt".into(), JValue::from(true));
        ctx.insert("bos_token".into(), JValue::from_serialize(&self.bos_token));
        ctx.insert("eos_token".into(), JValue::from_serialize(&self.eos_token));
        // `tools=tools` is passed unconditionally, so the name is always bound
        // -- to none when the request carries none, which templates test with
        // a plain `{%- if tools -%}`.
        ctx.insert("tools".into(), JValue::from_serialize(chat_tools(body)));
        // `if request.reasoning_effort is not None` -- absent, not none, so a
        // template's `reasoning_effort is defined` stays false.
        if let Some(effort) = body.get("reasoning_effort").filter(|v| !v.is_null()) {
            ctx.insert("reasoning_effort".into(), JValue::from_serialize(effort));
        }
        // `**request.chat_template_kwargs` -- spread, not nested.
        if let Some(kwargs) = body.get("chat_template_kwargs").and_then(Value::as_object) {
            for (k, v) in kwargs {
                ctx.insert(k.clone(), JValue::from_serialize(v));
            }
        }
        ctx
    }
}

/// `tools`, exactly as `_process_messages` hands it to the chat template.
///
/// sglang 0.5.18, serving_chat.py:1219 --
///
/// ```python
/// tools = None
/// effective_tools = self._effective_tools(request)      # request.tools + any
///                                                       # hung off a system/
///                                                       # developer message
/// if effective_tools and request.tool_choice != "none":
///     if not isinstance(request.tool_choice, str):      # {"function": {"name": ...}}
///         tools = [item.model_dump() for item in request.tools or []
///                  if item.function.name == request.tool_choice.function.name] or None
///     elif request.tools:
///         tools = [item.model_dump() for item in request.tools]
/// ```
///
/// Note `model_dump()` and not `model_dump(exclude_unset=True)`: fields the
/// client never sent are materialised with their defaults. That is not cosmetic
/// here, because GLM-5.3's template renders a tool by iterating it --
///
/// ```jinja
/// {%- for k, v in tool.items() -%}
///     {%- if k != 'defer_loading' and k != 'strict' -%}
///         "{{ k }}": {{ v | tojson(ensure_ascii=False) }}
/// ```
///
/// -- so a client tool of `{"name": "f", "parameters": {...}}` must reach the
/// template as `{"description": null, "name": "f", "parameters": {...},
/// "strict": false}` or the router renders a shorter tool block than the
/// engine and misses from block 0 onward. Both the added keys and their order
/// are load-bearing; the order is pydantic's field-declaration order, which is
/// why this builds the object rather than copying the client's.
fn chat_tools(body: &Value) -> Option<Vec<Value>> {
    let request_tools = body
        .get("tools")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty());
    // `_effective_tools` also collects tools hung off a system/developer
    // message. Those gate whether `tools` is populated but are never the
    // source of it -- the `elif request.tools` branch reads `request.tools`.
    let message_tools = body
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|msgs| {
            msgs.iter().any(|m| {
                matches!(
                    m.get("role").and_then(Value::as_str),
                    Some("system") | Some("developer")
                ) && m
                    .get("tools")
                    .is_some_and(|t| t.as_array().is_some_and(|a| !a.is_empty()))
            })
        });
    if request_tools.is_none() && !message_tools {
        return None;
    }
    let tool_choice = body.get("tool_choice");
    if tool_choice.and_then(Value::as_str) == Some("none") {
        return None;
    }
    let arr = request_tools?;
    // A non-string `tool_choice` names one function; the engine dumps only it.
    let only = tool_choice
        .filter(|c| c.as_str().is_none())
        .and_then(|c| c.get("function"))
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str);
    let out: Vec<Value> = arr
        .iter()
        .filter(|t| match only {
            None => true,
            Some(name) => {
                t.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(Value::as_str)
                    == Some(name)
            }
        })
        .map(tool_model_dump)
        .collect();
    (!out.is_empty()).then_some(out)
}

/// One `Tool.model_dump()`.
///
/// The field order and defaults are those of sglang 0.5.18's protocol models,
/// read off the running engine rather than guessed:
///
/// ```text
/// Tool      type='function'  function  defer_loading=None
/// Function  description=None  name  parameters=None  strict=False  defer_loading=None
/// ```
///
/// `Function` also carries a `@model_serializer(mode="wrap")` that pops
/// `defer_loading` when it is None -- so it appears on the tool but not on the
/// function unless the client set it. `Tool.defer_loading` has no such hook and
/// is always emitted.
fn tool_model_dump(tool: &Value) -> Value {
    let f = tool.get("function");
    let get = |k: &str| f.and_then(|f| f.get(k)).cloned();
    let mut func = serde_json::Map::new();
    func.insert("description".into(), get("description").unwrap_or(Value::Null));
    func.insert("name".into(), get("name").unwrap_or(Value::Null));
    func.insert("parameters".into(), get("parameters").unwrap_or(Value::Null));
    func.insert("strict".into(), get("strict").unwrap_or(Value::Bool(false)));
    if let Some(dl) = get("defer_loading").filter(|v| !v.is_null()) {
        func.insert("defer_loading".into(), dl);
    }
    let mut out = serde_json::Map::new();
    out.insert(
        "type".into(),
        tool.get("type")
            .cloned()
            .unwrap_or_else(|| Value::String("function".into())),
    );
    out.insert("function".into(), Value::Object(func));
    out.insert(
        "defer_loading".into(),
        tool.get("defer_loading").cloned().unwrap_or(Value::Null),
    );
    Value::Object(out)
}

/// Normalise OpenAI-style `tool_calls[].function.arguments` for chat templates.
///
/// The OpenAI wire format serialises tool-call arguments as a JSON *string*.
/// GLM-5.3's template iterates them as a mapping (`_args.items()`), so a
/// verbatim body renders as `items() expects a mapping, got string` and the
/// router loses the whole prompt. transformers' own `apply_chat_template` is
/// fed already-parsed dicts by the serving layer, which is why the engine
/// never sees this. Parse each string that is a JSON object back into an
/// object; leave every other shape untouched so a template that *wants* the
/// string still gets it.
///
/// Returns `None` when nothing needed rewriting, so the common path clones
/// nothing.
fn normalize_tool_call_arguments(messages: &Value) -> Option<Value> {
    let arr = messages.as_array()?;
    let mut out: Option<Vec<Value>> = None;
    for (i, msg) in arr.iter().enumerate() {
        let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        let mut new_calls: Option<Vec<Value>> = None;
        for (j, call) in calls.iter().enumerate() {
            // The template accepts both `tc.function.arguments` and `tc.arguments`.
            let path: &str = if call.get("function").is_some() {
                "function"
            } else {
                ""
            };
            let args = if path.is_empty() {
                call.get("arguments")
            } else {
                call.get("function").and_then(|f| f.get("arguments"))
            };
            let Some(Value::String(raw)) = args else {
                continue;
            };
            // Only an object is a mapping; a bare string/number/array left as-is
            // would still fail `.items()`, but rewriting it would be a lie.
            let Ok(parsed @ Value::Object(_)) = serde_json::from_str::<Value>(raw) else {
                continue;
            };
            let calls_vec = new_calls.get_or_insert_with(|| calls.clone());
            if path.is_empty() {
                calls_vec[j]["arguments"] = parsed;
            } else {
                calls_vec[j]["function"]["arguments"] = parsed;
            }
        }
        if let Some(calls_vec) = new_calls {
            let out_vec = out.get_or_insert_with(|| arr.clone());
            out_vec[i]["tool_calls"] = Value::Array(calls_vec);
        }
    }
    out.map(Value::Array)
}

/// Top-level keys of a request body, so an unrecognised schema is identifiable
/// from a log line without ever printing the prompt.
fn top_level_keys(body: &Value) -> Vec<String> {
    body.as_object()
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default()
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

/// `config.json`'s `architectures[0]`, the field sglang's
/// `resolve_chat_encoding_spec` dispatches on.
fn first_architecture(cfg_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(cfg_dir.join("config.json")).ok()?;
    let cfg = serde_json::from_str::<Value>(&text).ok()?;
    Some(
        cfg.get("architectures")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    )
}

/// Which native chat encoder this model needs, if any.
///
/// Mirrors the architecture arm of sglang's `resolve_chat_encoding_spec`, which
/// is what the engine itself dispatches on. The remaining specs (dsv32 /
/// inkling) still need their own encoders and stay on the `apply_chat_template`
/// path. `INFERA_KV_CHAT_ENCODING` forces one on for a tokenizer path whose
/// `config.json` the router cannot see.
fn detect_encoding(cfg_dir: &Path, spec: &str, arch_marker: &str) -> bool {
    if std::env::var("INFERA_KV_CHAT_ENCODING").as_deref() == Ok(spec) {
        tracing::info!(%spec, "kv-aware: chat encoding forced by INFERA_KV_CHAT_ENCODING");
        return true;
    }
    let arch = first_architecture(cfg_dir).unwrap_or_default();
    let matched = arch.contains(arch_marker);
    if matched {
        tracing::info!(%arch, %spec, "kv-aware: using a native chat encoder");
    }
    matched
}

fn detect_dsv4(cfg_dir: &Path) -> bool {
    detect_encoding(cfg_dir, "dsv4", "DeepseekV4")
}

fn detect_kimi_k3(cfg_dir: &Path) -> bool {
    detect_encoding(cfg_dir, "kimi_k3", "KimiK3")
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

    /// Model dir for a weight-backed test, from the environment.
    ///
    /// These used to carry a hardcoded absolute path from whichever machine the
    /// test was written on. That is worse than no test: everywhere else the
    /// path is missing, the test returns early, and the run reports a pass
    /// having checked nothing. Naming the variable at least makes the skip
    /// legible and lets CI opt in.
    ///
    /// `marker` is the file the test actually needs, so the skip is honest
    /// about what was missing rather than assuming every model ships the same
    /// layout -- Kimi-K3, notably, ships no Jinja template of any kind.
    fn model_dir_from_env(var: &str, marker: &str) -> Option<String> {
        match std::env::var(var) {
            Ok(p) if Path::new(&p).join(marker).exists() => Some(p),
            Ok(p) => {
                eprintln!("skip: {var}={p} has no {marker}");
                None
            }
            Err(_) => {
                eprintln!("skip: set {var} to a model dir to run this test");
                None
            }
        }
    }

    // Kimi-K2.6 ships chat_template.jinja (not embedded) + tiktoken. This guards
    // the whole chat path: standalone-template fallback, the `.get()` method, and
    // the `{% break %}` loop control all have to work or a chat request hashes to
    // empty (load-only routing, no cache locality). Skips if weights absent.
    #[test]
    fn kimi_chat_request_renders_and_hashes() {
        let Some(kimi_dir) = model_dir_from_env("INFERA_TEST_KIMI_DIR", "chat_template.jinja")
        else {
            return;
        };
        let h = BlockHasher::load(&kimi_dir);
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

    /// Kimi-K3 has no Jinja template at all, so `apply_chat_template` finds
    /// nothing and every chat request used to hash to empty -- kv-aware
    /// degrading to load-only routing with every health signal green. The
    /// native XTML encoder is the only thing standing between this model and a
    /// permanent 0% cache hit rate. Skips if weights absent.
    #[test]
    fn kimi_k3_chat_request_hashes_without_a_jinja_template() {
        let Some(dir) = model_dir_from_env("INFERA_TEST_KIMI_K3_DIR", "tiktoken.model") else {
            return;
        };
        let h = BlockHasher::load(&dir);
        assert!(h.is_enabled());
        assert!(
            h.chat_template.is_none(),
            "K3 is the no-template case; a template here means this test is \
             exercising something else"
        );
        assert!(
            h.kimi_k3,
            "config.json architectures[0] must select the encoder"
        );

        let body = json!({"messages": [
            {"role": "system", "content": "You are an agent. ".repeat(40)},
            {"role": "user", "content": "list the files ".repeat(40)},
        ]});
        let hashes = h.hash_for(&body, 16);
        assert!(!hashes.is_empty(), "K3 chat must tokenize natively");

        // A longer prompt sharing this one's prefix must share its leading block
        // hashes -- that shared prefix *is* the cache locality kv-aware routes on.
        let longer = json!({"messages": [
            {"role": "system", "content": "You are an agent. ".repeat(40)},
            {"role": "user", "content": "list the files ".repeat(80)},
        ]});
        let longer_hashes = h.hash_for(&longer, 16);
        assert!(longer_hashes.len() > hashes.len());
        assert_eq!(longer_hashes[0], hashes[0]);

        // A request the encoder refuses degrades to today's behaviour, not to
        // wrong hashes.
        let multimodal = json!({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]}]});
        assert!(h.hash_for(&multimodal, 16).is_empty());
    }

    /// End-to-end on real K3 weights: a `/v1/responses` body must hash to exactly
    /// what the equivalent `/v1/chat/completions` body hashes to.
    ///
    /// Measured before this fix, on a 1P1D K3 fleet: 462 of 528 routing decisions
    /// logged `request_blocks=0` -- every Codex turn, because Codex speaks the
    /// Responses API by default (`wire_api = "responses"`). kv-aware silently
    /// became round-robin for that whole endpoint while `policy="kv-aware"` kept
    /// logging normally. Equality, not non-emptiness, is the bar: a prefix a few
    /// tokens off the engine's does not error, it just never hits again.
    /// Opt in with `INFERA_TEST_KIMI_K3_DIR=/path/to/Kimi-K3`.
    #[test]
    fn kimi_k3_responses_request_hashes_identically_to_chat() {
        let Some(dir) = model_dir_from_env("INFERA_TEST_KIMI_K3_DIR", "tiktoken.model") else {
            return;
        };
        let h = BlockHasher::load(&dir);
        assert!(h.is_enabled() && h.kimi_k3);

        let tools = json!([{"type": "function", "name": "shell",
                            "parameters": {"type": "object"}}]);
        let responses = json!({
            "model": "kimi-k3",
            "instructions": "You are a coding agent. ".repeat(20),
            "input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "list the files ".repeat(40)}]},
                {"type": "reasoning", "summary": [{"text": "I should run ls"}]},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1",
                 "output": "a.txt b.txt ".repeat(40)},
            ],
            "tools": tools,
            "stream": true,
            "store": false,
        });
        let chat = json!({
            "messages": [
                {"role": "system", "content": "You are a coding agent. ".repeat(20)},
                {"role": "user", "content": [
                    {"type": "text", "text": "list the files ".repeat(40)}]},
                {"role": "assistant",
                 "reasoning_content": "I should run ls",
                 "tool_calls": [{"id": "call_1", "type": "function",
                                 "function": {"name": "shell",
                                              "arguments": "{\"cmd\":\"ls\"}"}}]},
                {"role": "tool", "tool_call_id": "call_1",
                 "content": "a.txt b.txt ".repeat(40)},
            ],
            "tools": [{"type": "function", "function": {
                "name": "shell", "description": null,
                "parameters": {"type": "object"}, "strict": null}}],
        });

        let from_responses = h.hash_for(&responses, 16);
        let from_chat = h.hash_for(&chat, 16);
        assert!(
            !from_responses.is_empty(),
            "/v1/responses must hash (this is the whole bug)"
        );
        assert_eq!(
            from_responses, from_chat,
            "router must hash a Responses turn exactly as the engine renders it"
        );

        // A prefix-sharing follow-up turn keeps its leading blocks -- that shared
        // prefix is the cache locality kv-aware exists to route on, and it is what
        // multi-prefill orchestration traffic was scattering.
        let mut next = responses.clone();
        next["input"]
            .as_array_mut()
            .unwrap()
            .push(json!({"type": "message", "role": "user",
                   "content": [{"type": "input_text", "text": "now summarize"}]}));
        let follow_up = h.hash_for(&next, 16);
        assert!(follow_up.len() > from_responses.len());
        assert_eq!(follow_up[0], from_responses[0]);

        // Unreproducible bodies degrade to load routing, not to wrong hashes.
        let with_history = json!({"input": "hi", "previous_response_id": "resp_1"});
        assert!(h.hash_for(&with_history, 16).is_empty());
    }

    /// GLM-5.2's template splits the thinking block out and strips the rest:
    ///
    ///   {%- set content = content.split('</think>')[-1] %}
    ///   {%- if content.strip() -%}{{ content.strip() }}{%- endif -%}
    ///
    /// minijinja has neither method natively. Without them the render errors,
    /// the prompt yields no tokens, and kv-aware degrades to load-only routing
    /// while every health signal stays green -- measured as 0.00% predicted
    /// hits against the Python router's 83.33% on the same trace. No weights
    /// needed: the template text is the whole subject.
    #[test]
    fn python_str_methods_render_glm_style_template() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{%- for m in messages -%}\
                 {%- set content = m['content'] -%}\
                 {%- set reasoning = content.split('</think>')[0].split('<think>')[-1] -%}\
                 {%- set content = content.split('</think>')[-1] -%}\
                 {%- if content.strip() -%}[{{ content.strip() }}|{{ reasoning.strip() }}]\
                 {%- endif -%}{%- endfor -%}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        let messages = json!([{"content": "<think>  weighing it  </think>  the answer  "}]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some("[the answer|weighing it]"),
            "split() must keep empty fields and index from the end; strip() must \
             trim both ends -- anything else changes the token stream and every \
             block hash with it"
        );
    }

    /// Qwen3's template guards its multi-step-tool branch with both methods at
    /// once:
    ///
    ///   {%- if ... and not(message.content.startswith('<tool_response>')
    ///                      and message.content.endswith('</tool_response>')) %}
    ///
    /// Measured on a live MI300X fleet before this was fixed: every
    /// /v1/chat/completions request logged `request_blocks=0`, so kv-aware had
    /// nothing to score and fell back to pure load balancing -- while the
    /// router stayed healthy and the same prompts through /v1/completions,
    /// which never touches the template, scored 242/242 hits. Qwen3 is not a
    /// corner case, and neither is the failure mode: it is silent.
    #[test]
    fn python_str_methods_render_qwen3_style_template() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{%- for m in messages -%}\
                 {%- if not(m['content'].startswith('<tool_response>') and \
                 m['content'].endswith('</tool_response>')) -%}\
                 [user:{{ m['content'] }}]\
                 {%- else -%}[tool]{%- endif -%}{%- endfor -%}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        let messages = json!([
            {"content": "<tool_response>result</tool_response>"},
            {"content": "plain question"},
        ]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some("[tool][user:plain question]"),
            "both methods must exist, or the whole template errors and every \
             chat request hashes to nothing"
        );
    }

    /// Python's forms these have to match: a tuple of candidates matches if any
    /// does, and an empty needle is always true.
    #[test]
    fn startswith_and_endswith_follow_python_semantics() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{{ s.startswith('ab') }}|{{ s.endswith('yz') }}|\
                 {{ s.startswith(('q', 'ab')) }}|{{ s.endswith(('q', 'yz')) }}|\
                 {{ s.startswith('') }}|{{ s.startswith('nope') }}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        let mut env = Environment::new();
        env.set_unknown_method_callback(unknown_method);
        let tmpl = env
            .template_from_str(h.chat_template.as_ref().unwrap())
            .unwrap();
        assert_eq!(
            tmpl.render(context! { s => "abcxyz" }).unwrap(),
            "true|true|true|true|true|false"
        );
    }

    /// The exact bytes transformers produces for Qwen3, so the token stream --
    /// and every block hash derived from it -- matches the engine's.
    ///
    /// Generated by running `tokenizer.apply_chat_template(..., tokenize=False,
    /// add_generation_prompt=True)` against Qwen3-30B-A3B. Note what the
    /// template does beyond the two methods: it strips the `<think>` block out
    /// of the assistant turn. Rendering *something* is not the bar; rendering
    /// what the engine saw is.
    #[test]
    fn qwen3_render_matches_transformers() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(QWEN3_TEMPLATE.to_string()),
            bos_token: None,
            eos_token: None,
        };
        let messages = json!([
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "<tool_response>prior</tool_response>"},
            {"role": "assistant", "content": "<think>reasoning</think>the answer"},
            {"role": "user", "content": "a plain question"},
        ]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some(
                "<|im_start|>system\nYou are helpful.<|im_end|>\n\
                 <|im_start|>user\n<tool_response>prior</tool_response><|im_end|>\n\
                 <|im_start|>assistant\nthe answer<|im_end|>\n\
                 <|im_start|>user\na plain question<|im_end|>\n\
                 <|im_start|>assistant\n"
            ),
        );
    }

    /// The pieces of Python's string semantics the templates actually lean on,
    /// where minijinja's nearest builtin differs: `split(sep)` KEEPS empty
    /// fields (bare `split()` does not), and `strip(chars)` takes a character
    /// SET, not a suffix.
    #[test]
    fn str_methods_follow_python_semantics() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{%- set s = messages[0]['content'] -%}\
                 {{ s.split(',') | length }}|{{ s.split() | length }}|\
                 {{ s.strip(' x') }}|{{ s.lstrip(' ') }}|{{ s.rstrip(' ') }}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        // "a,,b " -> split(',') = ["a","","b "] (3, empties kept)
        //         -> split()   = ["a,,b"]      (1, whitespace-collapsing)
        //         -> strip(" x") trims spaces and 'x' from both ends
        let messages = json!([{"content": "a,,b "}]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some("3|1|a,,b|a,,b |a,,b")
        );
    }

    /// `transformers` swaps jinja2's tojson for `json.dumps(..., ensure_ascii=
    /// False)`, so the engine's prompt has Python's `", "` / `": "` spacing and
    /// literal `<`, `>`, `&`. minijinja's builtin does the opposite on both
    /// counts, and object order has to survive serde_json and minijinja (each
    /// sorts keys unless told otherwise) or the arguments come out alphabetised.
    #[test]
    fn tojson_matches_transformers_json_dumps() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{{ messages[0] | tojson }}|{{ messages[0] | tojson(ensure_ascii=True) }}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        let messages = json!([{"z": "<a&b>", "a": [1, 2], "u": "中文"}]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some(
                r#"{"z": "<a&b>", "a": [1, 2], "u": "中文"}|{"z": "<a&b>", "a": [1, 2], "u": "\u4e2d\u6587"}"#
            ),
            "must equal json.dumps(x, ensure_ascii=...) byte for byte"
        );
    }

    /// minijinja has none of Python's dict views, and GLM-5.2 iterates
    /// `arguments.items()` on every tool call, so their absence takes the whole
    /// render down rather than just that branch.
    #[test]
    fn dict_views_follow_python_semantics() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: false,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(
                "{%- set d = messages[0] -%}\
                 {%- for k, v in d.items() -%}{{ k }}={{ v }};{%- endfor -%}\
                 |{{ d.keys() | join(',') }}|{{ d.values() | join(',') }}"
                    .to_string(),
            ),
            bos_token: None,
            eos_token: None,
        };
        let messages = json!([{"path": "/etc/hosts", "limit": 40}]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some("path=/etc/hosts;limit=40;|path,limit|/etc/hosts,40"),
            "items() must unpack as (key, value), agree with keys()/values(), and \
             keep the request's own key order rather than alphabetising it"
        );
    }

    /// The dsv4 gate is decided off `config.json`'s first architecture, exactly
    /// like sglang's `resolve_chat_encoding_spec`. Every other model — including
    /// the DeepseekV3 that dsv32 would claim — must stay on the chat-template
    /// path, since engaging the wrong native encoder silently produces a prefix
    /// that can never match.
    #[test]
    fn dsv4_is_detected_only_for_the_deepseek_v4_architecture() {
        let dir = std::env::temp_dir().join(format!("infera-dsv4-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("config.json");
        for (arch, want) in [
            (r#"["DeepseekV4ForCausalLM"]"#, true),
            (r#"["DeepseekV3ForCausalLM"]"#, false),
            (r#"["KimiK3ForCausalLM"]"#, false),
            (r#"[]"#, false),
        ] {
            std::fs::write(&cfg, format!(r#"{{"architectures": {arch}}}"#)).unwrap();
            assert_eq!(detect_dsv4(&dir), want, "architectures={arch}");
        }
        std::fs::write(&cfg, "not json").unwrap();
        assert!(!detect_dsv4(&dir));
        std::fs::remove_dir_all(&dir).ok();
        // No config.json at all.
        assert!(!detect_dsv4(&dir));
    }

    /// The whole point of the native encoder: a DSv4 chat body renders to the
    /// engine's prefix instead of to nothing. Hermetic — no weights needed,
    /// since the render is the subject and the tokenizer is the next stage.
    #[test]
    fn dsv4_chat_body_renders_without_a_chat_template() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: true,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: None,
            bos_token: None,
            eos_token: None,
        };
        // No system turn: serving_chat prepends an empty one, which renders to
        // nothing but must not shift anything else.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert_eq!(
            h.render_text_str(&body).as_deref(),
            Some("<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>")
        );

        // Same body without the gate: no chat_template, so nothing renders --
        // the behaviour this fix replaces.
        let off = BlockHasher { dsv4: false, ..h };
        assert!(off.render_text_str(&body).is_none());
    }

    /// serving_chat tokenizes the dsv4 encoder's output with a plain
    /// `tokenizer.encode(real_input)` but passes `add_special_tokens=False` at
    /// the chat-template site. Getting this backwards costs a doubled BOS on any
    /// tokenizer with `add_bos_token: true` -- and that misses silently, which is
    /// the whole failure mode the dsv4 encoder exists to remove.
    #[test]
    fn dsv4_output_is_tokenized_with_specials_and_templates_without() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: true,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: Some(QWEN3_TEMPLATE.to_string()),
            bos_token: None,
            eos_token: None,
        };
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(h.render_text(&body).unwrap().add_special_tokens);

        // Tools push the same body onto the chat-template path.
        let templated = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        });
        assert!(!h.render_text(&templated).unwrap().add_special_tokens);

        // Completion prompts spell out their own specials too.
        let prompt = json!({"prompt": "hi"});
        assert!(!h.render_text(&prompt).unwrap().add_special_tokens);
    }

    /// The tiktoken path must not silently drop the flag the HF path honours.
    ///
    /// `KimiTokenizer` adds no BOS/EOS by construction, so it cannot reproduce
    /// what a tokenizer asked for specials emits -- and being one token short of
    /// the engine is not an error, it is a permanent 0% hit rate. Hashing
    /// nothing costs exactly the load-only routing an unhashable body already
    /// gets; hashing the wrong thing costs the cache.
    #[test]
    fn the_tiktoken_path_refuses_text_that_wants_special_tokens() {
        let Ok(dir) = std::env::var("INFERA_TEST_KIMI_K3_DIR") else {
            eprintln!("skip: set INFERA_TEST_KIMI_K3_DIR to a Kimi model dir");
            return;
        };
        let tk = crate::tiktoken::KimiTokenizer::load(std::path::Path::new(&dir))
            .expect("load tiktoken");
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: Some(tk),
            dsv4: true,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: None,
            bos_token: None,
            eos_token: None,
        };
        // dsv4's encoder output is the one render tokenized with specials.
        let body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(h.render_text(&body).unwrap().add_special_tokens);
        assert!(
            h.hash_for(&body, 4).is_empty(),
            "refuse rather than hash a prefix the engine will not match"
        );
    }

    /// Tools are rendered by the engine from a full pydantic `Tool.model_dump()`,
    /// whose defaults are not in the request. Skipping is the honest outcome: a
    /// near-miss prefix hashes to a permanent miss with nothing to show for it.
    #[test]
    fn dsv4_skips_tool_carrying_bodies_rather_than_guess() {
        let h = BlockHasher {
            tokenizer: None,
            tiktoken: None,
            dsv4: true,
            kimi_k3: false,
            tool_args: AtomicU8::new(TOOL_ARGS_UNDECIDED),
            chat_template: None,
            bos_token: None,
            eos_token: None,
        };
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
        });
        assert!(h.render_text_str(&body).is_none());
        // An empty tools array is not a tool-carrying body.
        let body = json!({"messages": [{"role": "user", "content": "hi"}], "tools": []});
        assert!(h.render_text_str(&body).is_some());
    }

    /// End-to-end on real GLM-5.2 weights: template render -> tokenize -> block
    /// hashes. Guards the whole chat path the way the Kimi test does. The
    /// hermetic test above is the one that actually pins the bug; this one
    /// catches a template that changes under us. Opt in with
    /// `INFERA_TEST_GLM_DIR=/path/to/GLM-5.2-*`.
    #[test]
    fn glm52_chat_request_renders_and_hashes() {
        let Some(glm_dir) = model_dir_from_env("INFERA_TEST_GLM_DIR", "chat_template.jinja") else {
            return;
        };
        // The render failure this guards against is only ever a warn log, so
        // without a subscriber the assert below says "blind" and not why.
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::WARN)
            .try_init();
        let h = BlockHasher::load(&glm_dir);
        assert!(h.is_enabled());
        let body = json!({"messages": [
            {"role": "user", "content": "read the config file ".repeat(40)},
            {"role": "assistant", "content": "<think>looking</think>here it is ".repeat(40)},
            {"role": "user", "content": "now summarize it ".repeat(40)},
        ]});
        assert!(
            !h.hash_for(&body, 64).is_empty(),
            "GLM-5.2 chat template must render + tokenize, else kv-aware is blind"
        );
    }

    /// The tool-call branch reaches template code the plain chat turns never do
    /// (`arguments.items()`, `tojson`, the tool role, and `{% %}` tags with no
    /// explicit whitespace control), and the trace this router was tuned on has
    /// no tool calls -- so the path stayed broken with every benchmark green.
    ///
    /// A hit needs the router's tokens to equal the engine's, so "it rendered"
    /// is not the bar; this pins the exact string. Regenerate the expected value
    /// by rendering `chat_template.jinja` through the environment built in
    /// `transformers.utils.chat_template_utils._compile_jinja_template`.
    #[test]
    fn glm52_tool_call_render_matches_transformers() {
        let Some(glm_dir) = model_dir_from_env("INFERA_TEST_GLM_DIR", "chat_template.jinja") else {
            return;
        };
        let h = BlockHasher::load(&glm_dir);
        // Ordered so that alphabetising the keys shows up, and carrying values
        // that separate Python's json.dumps from serde_json's: a nested object,
        // a list, an HTML character and a non-ASCII one.
        let messages = json!([
            {"role": "user", "content": "read the config file"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "function": {"name": "read_file", "arguments": {
                    "path": "/etc/hosts",
                    "opts": {"depth": 2, "glob": "<*.py>", "note": "中文"},
                    "tags": ["a", "b"],
                    "limit": 40,
                }},
            }]},
            {"role": "tool", "content": "127.0.0.1 localhost"},
        ]);
        assert_eq!(
            h.apply_chat_template(&json!({}), &messages).as_deref(),
            Some(concat!(
                "[gMASK]<sop><|system|>Reasoning Effort: Max",
                "<|user|>read the config file",
                "<|assistant|><think></think>",
                "<tool_call>read_file",
                "<arg_key>path</arg_key><arg_value>/etc/hosts</arg_value>",
                r#"<arg_key>opts</arg_key><arg_value>{"depth": 2, "glob": "<*.py>", "note": "中文"}</arg_value>"#,
                r#"<arg_key>tags</arg_key><arg_value>["a", "b"]</arg_value>"#,
                "<arg_key>limit</arg_key><arg_value>40</arg_value>",
                "</tool_call>",
                "<|observation|><tool_response>127.0.0.1 localhost</tool_response>",
                "<|assistant|><think>",
            ))
        );
    }
}

#[cfg(test)]
mod tool_call_args_tests {
    use super::normalize_tool_call_arguments;
    use serde_json::json;

    #[test]
    fn parses_openai_string_arguments_into_object() {
        let messages = json!([
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "tool_calls": [
                {"type": "function", "function": {
                    "name": "shell",
                    "arguments": "{\"cmd\": [\"ls\", \"-l\"], \"timeout\": 5}"
                }}
            ]}
        ]);
        let out = normalize_tool_call_arguments(&messages).expect("should rewrite");
        let args = &out[1]["tool_calls"][0]["function"]["arguments"];
        assert!(args.is_object(), "got {args}");
        assert_eq!(args["cmd"][0], json!("ls"));
        assert_eq!(args["timeout"], json!(5));
    }

    #[test]
    fn handles_flattened_tool_call_shape() {
        let messages = json!([
            {"role": "assistant", "tool_calls": [
                {"name": "get_weather", "arguments": "{\"city\": \"Beijing\"}"}
            ]}
        ]);
        let out = normalize_tool_call_arguments(&messages).expect("should rewrite");
        assert_eq!(
            out[0]["tool_calls"][0]["arguments"]["city"],
            json!("Beijing")
        );
    }

    #[test]
    fn leaves_untouched_when_nothing_to_do() {
        // already an object
        let objs = json!([
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": {"a": 1}}}
            ]}
        ]);
        assert!(normalize_tool_call_arguments(&objs).is_none());

        // plain chat, no tool calls
        let plain = json!([{"role": "user", "content": "hi"}]);
        assert!(normalize_tool_call_arguments(&plain).is_none());

        // a string that is not a JSON object stays a string
        let scalar = json!([
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": "not json"}}
            ]}
        ]);
        assert!(normalize_tool_call_arguments(&scalar).is_none());
    }
}
