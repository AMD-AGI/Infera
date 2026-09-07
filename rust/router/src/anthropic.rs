// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! Anthropic Messages API <-> OpenAI Chat translation — the Rust twin of
//! `infera.api.anthropic`.
//!
//! Workers (vLLM, SGLang) only speak OpenAI Chat, while clients like openclaw
//! speak the Anthropic Messages API. Translating at the router edge, rather
//! than proxying, keeps the body visible to `cache_control` parsing, kv-aware
//! routing and stable prompt hashing.
//!
//! Scope, matching the Python reference:
//!
//! - Request: `messages` (text blocks), `system` (string or block array),
//!   `max_tokens`, `stop_sequences`, `temperature`, `top_p`, `top_k`,
//!   `stream`, `metadata.user_id` -> `user`, tool definitions, `tool_choice`,
//!   assistant `tool_use` and user `tool_result`.
//! - Response (unary): OpenAI completion JSON -> Anthropic Messages JSON,
//!   with `tool_calls` becoming `tool_use` blocks.
//! - Response (streaming): OpenAI SSE -> Anthropic SSE events, with tool-call
//!   deltas reassembled into `tool_use` blocks carrying `input_json_delta`.
//!
//! Out of scope: multimodal content (`image`/`audio`/`video`/`document`,
//! including nested in `tool_result.content`) is rejected with a message the
//! caller maps to HTTP 400; extended thinking blocks are not modelled and
//! are ignored.

use serde_json::{json, Map, Value};

/// Content-block types that carry non-text payloads. Accepting them would
/// silently drop the payload, so they are refused at the front door instead.
const MULTIMODAL_TYPES: [&str; 4] = ["image", "audio", "video", "document"];

// ---------------------------------------------------------------------------
// Request translation: Anthropic Messages -> OpenAI Chat
// ---------------------------------------------------------------------------

/// Translate an Anthropic Messages request body into an OpenAI Chat body.
///
/// `Err` carries a client-facing reason: an unsupported feature (multimodal
/// content) or a violated precondition (`tool_result` without `tool_use_id`).
/// The caller answers those with HTTP 400.
pub fn translate_request(body: &Value) -> Result<Value, String> {
    let obj = match body.as_object() {
        Some(o) => o,
        None => return Err("request body must be a JSON object".to_string()),
    };

    reject_if_multimodal(obj)?;

    let mut messages: Vec<Value> = Vec::new();

    // Anthropic carries the system prompt at the top level, as a string or an
    // array of text blocks; OpenAI wants one system message up front.
    let system_text = flatten_system(obj.get("system"));
    if !system_text.is_empty() {
        messages.push(json!({"role": "system", "content": system_text}));
    }

    // Per-role translation. Tool blocks make this non-trivial: one Anthropic
    // message can expand into several OpenAI messages.
    if let Some(list) = obj.get("messages").and_then(Value::as_array) {
        for msg in list {
            let role = match msg.get("role").and_then(Value::as_str) {
                // Anthropic has no in-band system role, so anything else is
                // not representable and is skipped rather than guessed at.
                Some(r @ ("user" | "assistant")) => r,
                _ => continue,
            };
            messages.extend(translate_message(role, msg.get("content"))?);
        }
    }

    let mut out = Map::new();
    out.insert(
        "model".to_string(),
        obj.get("model").cloned().unwrap_or_else(|| json!("")),
    );
    out.insert("messages".to_string(), Value::Array(messages));

    if let Some(tools) = obj.get("tools").and_then(Value::as_array) {
        if !tools.is_empty() {
            let defs: Vec<Value> = tools
                .iter()
                .filter(|t| t.is_object())
                .map(translate_tool_def)
                .collect();
            out.insert("tools".to_string(), Value::Array(defs));
        }
    }

    if let Some(tc) = obj.get("tool_choice") {
        if let Some(translated) = translate_tool_choice(tc) {
            out.insert("tool_choice".to_string(), translated);
        }
        // Anthropic expresses "one tool at a time" inside `tool_choice`;
        // OpenAI expresses it with a top-level `parallel_tool_calls`.
        if tc.get("disable_parallel_tool_use") == Some(&Value::Bool(true)) {
            out.insert("parallel_tool_calls".to_string(), Value::Bool(false));
        }
    }

    // Direct field mapping. `top_k` is not an OpenAI chat field, but both
    // engines accept it as an extra and receivers that don't simply ignore it.
    for key in ["max_tokens", "temperature", "top_p", "top_k"] {
        if let Some(v) = obj.get(key) {
            out.insert(key.to_string(), v.clone());
        }
    }
    if let Some(stop) = obj.get("stop_sequences") {
        if truthy(stop) {
            out.insert("stop".to_string(), stop.clone());
        }
    }
    if let Some(stream) = obj.get("stream") {
        out.insert("stream".to_string(), Value::Bool(truthy(stream)));
    }
    if let Some(user) = obj
        .get("metadata")
        .and_then(Value::as_object)
        .and_then(|m| m.get("user_id"))
        .and_then(Value::as_str)
    {
        out.insert("user".to_string(), json!(user));
    }

    Ok(Value::Object(out))
}

/// Python truthiness, so a body written against the reference behaves the same.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
    }
}

/// Flatten `system`, which is either a string or an array of text blocks.
fn flatten_system(system: Option<&Value>) -> String {
    match system {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Refuse any non-text, non-tool content block, one level into
/// `tool_result.content` as well: a tool that returns a screenshot must not be
/// degraded to an empty tool response.
fn reject_if_multimodal(body: &Map<String, Value>) -> Result<(), String> {
    let messages = match body.get("messages").and_then(Value::as_array) {
        Some(m) => m,
        None => return Ok(()),
    };
    for msg in messages {
        let content = match msg.get("content").and_then(Value::as_array) {
            Some(c) => c,
            None => continue,
        };
        for block in content {
            let block_type = match block.get("type").and_then(Value::as_str) {
                Some(t) => t,
                None => continue,
            };
            if MULTIMODAL_TYPES.contains(&block_type) {
                return Err(format!(
                    "content type '{block_type}' not supported in this Infera build. \
                     Multimodal blocks land in a followup PR. \
                     Submit text or tool blocks to proceed."
                ));
            }
            if block_type != "tool_result" {
                continue;
            }
            let inner = match block.get("content").and_then(Value::as_array) {
                Some(i) => i,
                None => continue,
            };
            for sub in inner {
                if let Some(sub_type) = sub.get("type").and_then(Value::as_str) {
                    if MULTIMODAL_TYPES.contains(&sub_type) {
                        return Err(format!(
                            "tool_result returned a '{sub_type}' block, which is not \
                             supported in this Infera build. Multimodal tool outputs \
                             land in a followup PR. Convert tool output to text first."
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Anthropic tool definition -> OpenAI `{type:"function", function:{...}}`.
///
/// `input_schema` becomes `parameters`. An absent or empty description is
/// omitted rather than declared empty, which strict receivers surface back to
/// the model as noise.
fn translate_tool_def(t: &Value) -> Value {
    let mut function = Map::new();
    function.insert(
        "name".to_string(),
        t.get("name").cloned().unwrap_or_else(|| json!("")),
    );
    let params = match t.get("input_schema") {
        Some(s) if truthy(s) => s.clone(),
        _ => json!({}),
    };
    function.insert("parameters".to_string(), params);
    if let Some(desc) = t.get("description").and_then(Value::as_str) {
        if !desc.is_empty() {
            function.insert("description".to_string(), json!(desc));
        }
    }
    json!({"type": "function", "function": Value::Object(function)})
}

/// Anthropic `tool_choice` -> OpenAI `tool_choice`.
///
/// `auto` -> `"auto"`, `any` -> `"required"`, `none` -> `"none"`, and
/// `{type:"tool", name}` -> `{type:"function", function:{name}}`. Unknown
/// shapes return `None` so the field is dropped and the engine picks, which
/// beats a 400 when Anthropic adds a type we have not modelled yet.
fn translate_tool_choice(tc: &Value) -> Option<Value> {
    let choice_type = tc.get("type").and_then(Value::as_str)?;
    match choice_type {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or("");
            if name.is_empty() {
                tracing::warn!(
                    "anthropic tool_choice type=tool with empty/missing name -- \
                     falling back to engine default"
                );
                return None;
            }
            Some(json!({"type": "function", "function": {"name": name}}))
        }
        other => {
            tracing::warn!(
                "anthropic tool_choice type={} not recognized -- falling back to engine default",
                other
            );
            None
        }
    }
}

/// Convert one Anthropic message into one or more OpenAI messages.
///
/// An assistant message folds its text and `tool_use` blocks into a single
/// message with `content` plus `tool_calls`. A user message splits: leftover
/// text becomes a `user` message, and each `tool_result` becomes its own
/// `tool` message, in the order the blocks appeared.
fn translate_message(role: &str, content: Option<&Value>) -> Result<Vec<Value>, String> {
    let blocks = match content {
        Some(Value::String(s)) => return Ok(vec![json!({"role": role, "content": s})]),
        Some(Value::Array(b)) => b,
        _ => return Ok(vec![json!({"role": role, "content": ""})]),
    };

    if role == "assistant" {
        let mut text_parts: Vec<&str> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        text_parts.push(text);
                    }
                }
                Some("tool_use") => {
                    // Anthropic's `input` is an object; OpenAI's `arguments` is
                    // a JSON string of it, so serialize even when empty.
                    let input = match block.get("input") {
                        Some(v) if truthy(v) => v.clone(),
                        _ => json!({}),
                    };
                    let id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .unwrap_or_else(|| format!("call_{}", random_hex(24)));
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(Value::as_str).unwrap_or(""),
                            "arguments": input.to_string(),
                        },
                    }));
                }
                _ => {}
            }
        }
        let mut msg = Map::new();
        msg.insert("role".to_string(), json!("assistant"));
        // OpenAI requires `content` to be present even when null alongside
        // `tool_calls`.
        if text_parts.is_empty() {
            msg.insert("content".to_string(), Value::Null);
        } else {
            msg.insert("content".to_string(), json!(text_parts.join("\n")));
        }
        if !tool_calls.is_empty() {
            msg.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
        return Ok(vec![Value::Object(msg)]);
    }

    let mut text_parts: Vec<&str> = Vec::new();
    let mut tool_messages: Vec<Value> = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    text_parts.push(text);
                }
            }
            Some("tool_result") => {
                // `tool_call_id` is what links the result back to the
                // assistant's earlier `tool_calls[].id`. Without it the engine
                // would either error or attribute the result to the wrong call.
                let tool_use_id = block
                    .get("tool_use_id")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .ok_or_else(|| {
                        "tool_result block missing required 'tool_use_id' -- each tool_result \
                         must reference the assistant's prior tool_use.id."
                            .to_string()
                    })?;
                let text = match block.get("content") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Array(parts)) => parts
                        .iter()
                        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                        .map(|b| b.get("text").and_then(Value::as_str).unwrap_or(""))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    _ => String::new(),
                };
                tool_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_use_id,
                    "content": text,
                }));
            }
            _ => {}
        }
    }

    let mut out: Vec<Value> = Vec::new();
    if !text_parts.is_empty() {
        out.push(json!({"role": "user", "content": text_parts.join("\n")}));
    }
    out.extend(tool_messages);
    if out.is_empty() {
        // Empty content: keep the turn structure so the chat template still
        // sees the alternation it expects.
        out.push(json!({"role": "user", "content": ""}));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Response translation (unary): OpenAI completion -> Anthropic Messages
// ---------------------------------------------------------------------------

/// Translate an OpenAI chat completion into an Anthropic Messages response.
///
/// `model` should be the model named by the client request, since engines
/// normalise the one they echo back; `None` falls back to the response's own
/// `model`. `request_id` seeds the Anthropic message id so client and server
/// logs correlate, and falls back to a random id.
pub fn translate_response(
    openai_response: &Value,
    model: Option<&str>,
    request_id: Option<&str>,
) -> Value {
    let first = openai_response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first());
    let message = first.and_then(|c| c.get("message"));
    let finish_reason = first
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    let usage = openai_response.get("usage");

    let mut content: Vec<Value> = Vec::new();
    if let Some(text) = message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_str)
    {
        if !text.is_empty() {
            content.push(json!({"type": "text", "text": text}));
        }
    }
    if let Some(calls) = message
        .and_then(|m| m.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            let function = call.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("");
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("toolu_{}", random_hex(20)));
            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": parse_tool_arguments(function.and_then(|f| f.get("arguments")), name),
            }));
        }
    }
    // The engine returned neither text nor tool calls; Anthropic clients still
    // expect a non-empty content array.
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let model_name = model
        .filter(|m| !m.is_empty())
        .or_else(|| openai_response.get("model").and_then(Value::as_str))
        .unwrap_or("");

    json!({
        "id": message_id(request_id),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_name,
        "stop_reason": map_finish_reason(finish_reason),
        "stop_sequence": Value::Null,
        "usage": {
            "input_tokens": usage_field(usage, "prompt_tokens"),
            "output_tokens": usage_field(usage, "completion_tokens"),
            // Engines do not expose per-request cache hit counts on the OpenAI
            // usage block, so the keys are present but zero.
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    })
}

/// Read one OpenAI usage counter, defaulting to 0 when absent or null.
fn usage_field(usage: Option<&Value>, key: &str) -> u64 {
    usage
        .and_then(|u| u.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

/// OpenAI `arguments` (a JSON string) -> Anthropic `input` (an object).
///
/// Engines occasionally emit malformed or non-object arguments; substituting
/// `{}` keeps the client seeing a well-formed `tool_use` block instead of a 500.
fn parse_tool_arguments(arguments: Option<&Value>, tool_name: &str) -> Value {
    match arguments {
        Some(Value::Object(o)) => Value::Object(o.clone()),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(o)) => Value::Object(o),
            Ok(_) => {
                tracing::warn!(
                    "tool_call {} returned non-object arguments -- substituting {{}}",
                    tool_name
                );
                json!({})
            }
            Err(err) => {
                tracing::warn!(
                    "tool_call {} returned malformed arguments JSON: {} -- substituting {{}}",
                    tool_name,
                    err
                );
                json!({})
            }
        },
        _ => json!({}),
    }
}

/// OpenAI `finish_reason` -> Anthropic `stop_reason`.
///
/// Unknown values fall back to `end_turn` so engine-specific reasons
/// (SGLang's `eos_token`, vLLM's `abort`) still terminate cleanly.
/// `content_filter` maps to `end_turn` because Anthropic exposes no
/// moderation stop.
fn map_finish_reason(reason: Option<&str>) -> Value {
    match reason {
        None | Some("") => Value::Null,
        Some("stop") | Some("content_filter") => json!("end_turn"),
        Some("length") => json!("max_tokens"),
        Some("tool_calls") => json!("tool_use"),
        Some(other) => {
            tracing::info!(
                "openai finish_reason={} not in our mapping; falling back to end_turn",
                other
            );
            json!("end_turn")
        }
    }
}

// ---------------------------------------------------------------------------
// Response translation (streaming): OpenAI SSE -> Anthropic SSE
// ---------------------------------------------------------------------------
//
// OpenAI:
//   data: {"choices":[{"delta":{"content":"Hello"}}]}
//   data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
//   data: [DONE]
//
// Anthropic (two lines per event):
//   event: message_start
//   data: {"type":"message_start","message":{...}}
//   event: content_block_start
//   data: {"type":"content_block_start","index":0,"content_block":{...}}
//   event: content_block_delta
//   data: {"type":"content_block_delta","index":0,"delta":{...}}
//   event: content_block_stop
//   data: {"type":"content_block_stop","index":0}
//   event: message_delta
//   data: {"type":"message_delta","delta":{...},"usage":{...}}
//   event: message_stop
//   data: {"type":"message_stop"}

/// Incremental OpenAI-SSE -> Anthropic-SSE converter.
///
/// Feed engine bytes to [`SseTranslator::push`] as they arrive and forward
/// whatever it returns; call [`SseTranslator::finish`] once the engine stream
/// ends so a stream cut short without `[DONE]` still terminates cleanly.
/// Partial lines are buffered across chunks, so callers may split the engine
/// body anywhere.
///
/// Block-index policy is first-come-first-served: whichever block type is seen
/// first in the OpenAI deltas takes Anthropic index 0, the next distinct block
/// takes 1, and so on. Index 0 is not reserved for text, because clients key
/// on the index to map deltas back to content blocks.
///
/// Malformed JSON, blank lines and `:` keepalives are skipped rather than
/// failing the stream, since the client already holds a 200 and part of a body.
pub struct SseTranslator {
    model: String,
    message_id: String,
    /// Bytes of an SSE line not yet terminated by `\n`.
    buf: Vec<u8>,
    /// True once `message_start` has been emitted. Closers are gated on it:
    /// a `message_stop` without a `message_start` is a protocol violation.
    started: bool,
    /// True once the terminal events have been emitted, or `[DONE]` arrived
    /// before anything else did. Further input is ignored.
    done: bool,
    text_index: Option<u64>,
    /// OpenAI `delta.tool_calls[].index` -> Anthropic content block index,
    /// in the order the tool calls first appeared.
    tool_indices: Vec<(i64, u64)>,
    next_block_index: u64,
    finish_reason: Option<String>,
    output_tokens: u64,
}

impl SseTranslator {
    /// Build a translator for one response. `model` is the model named by the
    /// client request; `request_id` seeds the Anthropic message id.
    pub fn new(model: &str, request_id: Option<&str>) -> Self {
        Self {
            model: model.to_string(),
            message_id: message_id(request_id),
            buf: Vec::new(),
            started: false,
            done: false,
            text_index: None,
            tool_indices: Vec::new(),
            next_block_index: 0,
            finish_reason: None,
            output_tokens: 0,
        }
    }

    /// Feed one chunk of the engine's SSE body; returns the Anthropic SSE bytes
    /// it produced, which may be empty when the chunk held no complete line.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        self.buf.extend_from_slice(chunk);
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = line.trim_ascii();
            if line.is_empty() || line.starts_with(b":") || !line.starts_with(b"data:") {
                continue;
            }
            let data = line["data:".len()..].trim_ascii();
            if data == b"[DONE]" {
                if self.started {
                    self.close(&mut out);
                }
                self.done = true;
                self.buf.clear();
                return out;
            }
            if let Ok(obj) = serde_json::from_slice::<Value>(data) {
                self.handle_chunk(&obj, &mut out);
            }
            if self.done {
                self.buf.clear();
                return out;
            }
        }
        out
    }

    /// Emit the terminal events for a stream that ended without `[DONE]`.
    /// Idempotent. A stream that never started is an error, not a silent 200.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.done {
            return Vec::new();
        }
        if !self.started {
            return self.error("worker stream ended before any tokens");
        }
        let mut out = Vec::new();
        self.close(&mut out);
        self.done = true;
        out
    }

    /// Terminate the stream with an Anthropic error event.
    pub fn error(&mut self, message: &str) -> Vec<u8> {
        if self.done {
            return Vec::new();
        }
        let mut out = Vec::new();
        push_event(
            &mut out,
            "error",
            json!({
                "error": {
                    "type": "api_error",
                    "message": message,
                }
            }),
        );
        self.done = true;
        out
    }

    /// Translate one decoded OpenAI chunk into Anthropic events.
    fn handle_chunk(&mut self, obj: &Value, out: &mut Vec<u8>) {
        if self.done {
            return;
        }
        if let Some(error) = obj.get("error") {
            let message = error
                .get("message")
                .and_then(Value::as_str)
                .or_else(|| error.as_str())
                .unwrap_or("worker stream failed");
            out.extend(self.error(message));
            return;
        }

        let choices = obj.get("choices").and_then(Value::as_array);
        let choice = match choices.and_then(|c| c.first()) {
            Some(c) => c,
            None => {
                // Some engines close with a usage-only chunk carrying no
                // choices; take the token count from it.
                if let Some(n) = obj
                    .get("usage")
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(Value::as_u64)
                {
                    self.output_tokens = self.output_tokens.max(n);
                }
                return;
            }
        };

        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            if !reason.is_empty() {
                self.finish_reason = Some(reason.to_string());
            }
        }

        // Anthropic requires `message_start` before any content block event.
        if !self.started {
            push_event(
                out,
                "message_start",
                json!({
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self.model,
                        "stop_reason": Value::Null,
                        "stop_sequence": Value::Null,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                }),
            );
            self.started = true;
        }

        let delta = match choice.get("delta") {
            Some(d) => d,
            None => return,
        };

        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                let index = match self.text_index {
                    Some(i) => i,
                    None => {
                        let i = self.next_block_index;
                        self.next_block_index += 1;
                        self.text_index = Some(i);
                        push_event(
                            out,
                            "content_block_start",
                            json!({"index": i, "content_block": {"type": "text", "text": ""}}),
                        );
                        i
                    }
                };
                push_event(
                    out,
                    "content_block_delta",
                    json!({"index": index, "delta": {"type": "text_delta", "text": text}}),
                );
            }
        }

        // Tool calls arrive in pieces: a first delta with id and name, then
        // any number of deltas appending to `function.arguments`. Each
        // argument fragment is a valid JSON prefix, not a complete value.
        if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for call in calls {
                if !call.is_object() {
                    continue;
                }
                let openai_index = call.get("index").and_then(Value::as_i64).unwrap_or(0);
                let function = call.get("function");
                let block_index = match self
                    .tool_indices
                    .iter()
                    .find(|(oi, _)| *oi == openai_index)
                    .map(|(_, bi)| *bi)
                {
                    Some(bi) => bi,
                    None => {
                        let bi = self.next_block_index;
                        self.next_block_index += 1;
                        self.tool_indices.push((openai_index, bi));
                        let id = call
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("toolu_{}", random_hex(20)));
                        push_event(
                            out,
                            "content_block_start",
                            json!({
                                "index": bi,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": id,
                                    "name": function
                                        .and_then(|f| f.get("name"))
                                        .and_then(Value::as_str)
                                        .unwrap_or(""),
                                    // Clients expect `input` to exist from the
                                    // start; the deltas accumulate into it.
                                    "input": {},
                                },
                            }),
                        );
                        bi
                    }
                };
                if let Some(fragment) = function
                    .and_then(|f| f.get("arguments"))
                    .and_then(Value::as_str)
                {
                    if !fragment.is_empty() {
                        push_event(
                            out,
                            "content_block_delta",
                            json!({
                                "index": block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": fragment,
                                },
                            }),
                        );
                    }
                }
            }
        }
    }

    /// Emit `content_block_stop` for every open block, then `message_delta`
    /// and `message_stop`.
    fn close(&self, out: &mut Vec<u8>) {
        // Deterministic order: text first, then tool blocks by index. Only the
        // stop-per-open-block invariant matters for protocol correctness.
        let mut indices: Vec<u64> = Vec::new();
        if let Some(i) = self.text_index {
            indices.push(i);
        }
        let mut tool: Vec<u64> = self.tool_indices.iter().map(|(_, bi)| *bi).collect();
        tool.sort_unstable();
        indices.extend(tool);
        for index in indices {
            push_event(out, "content_block_stop", json!({ "index": index }));
        }
        push_event(
            out,
            "message_delta",
            json!({
                "delta": {
                    "stop_reason": map_finish_reason(self.finish_reason.as_deref()),
                    "stop_sequence": Value::Null,
                },
                "usage": {"output_tokens": self.output_tokens},
            }),
        );
        push_event(out, "message_stop", json!({}));
    }
}

/// Append one Anthropic SSE event: an `event:` line and a `data:` line whose
/// payload carries the event name as its `type`.
fn push_event(out: &mut Vec<u8>, event_type: &str, payload: Value) {
    let mut body = Map::new();
    body.insert("type".to_string(), json!(event_type));
    if let Value::Object(fields) = payload {
        for (key, value) in fields {
            body.insert(key, value);
        }
    }
    out.extend_from_slice(
        format!("event: {}\ndata: {}\n\n", event_type, Value::Object(body)).as_bytes(),
    );
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Anthropic message id, derived from the router request id so client and
/// server logs correlate. Falls back to a random id.
fn message_id(request_id: Option<&str>) -> String {
    match request_id.filter(|s| !s.is_empty()) {
        Some(id) => format!("msg_{}", crate::util::truncate_chars(id, 24)),
        None => format!("msg_{}", random_hex(24)),
    }
}

/// `len` hex characters of randomness, standing in for a uuid4 hex prefix.
fn random_hex(len: usize) -> String {
    let mut s = String::with_capacity(32);
    while s.len() < len {
        s.push_str(&format!("{:016x}", rand::random::<u64>()));
    }
    s.truncate(len);
    s
}

#[cfg(test)]
mod sse_tests {
    use super::*;

    #[test]
    fn an_error_line_does_not_emit_later_content_from_the_same_chunk() {
        let mut t = SseTranslator::new("m", Some("abc"));
        let chunk = concat!(
            "data: {\"error\":{\"message\":\"boom\"}}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
        );
        let out = String::from_utf8(t.push(chunk.as_bytes())).unwrap();
        assert!(out.contains("event: error"));
        assert!(!out.contains("message_start"));
        assert!(!out.contains("content_block"));
    }

    #[test]
    fn finish_before_start_emits_an_error() {
        let mut t = SseTranslator::new("m", None);
        let out = String::from_utf8(t.finish()).unwrap();
        assert!(out.contains("event: error"));
        assert!(!out.contains("message_start"));
    }
}
