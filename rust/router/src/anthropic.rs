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
//!   assistant `tool_use` and user `tool_result`. Top-level `thinking`
//!   maps to `chat_template_kwargs.enable_thinking` and `thinking`;
//!   optional `thinking.budget_tokens` maps to SGLang
//!   `custom_logit_processor` + `custom_params.thinking_budget` when the
//!   model id is GLM-5.3
//!   (budgets are clamped to leave room to close thinking and answer); assistant `thinking` content
//!   blocks map to OpenAI `reasoning_content`.
//! - Response (unary): OpenAI completion JSON -> Anthropic Messages JSON,
//!   with `tool_calls` becoming `tool_use` blocks and `reasoning_content`
//!   becoming a leading `thinking` content block.
//! - Response (streaming): OpenAI SSE -> Anthropic SSE events, with tool-call
//!   deltas reassembled into `tool_use` blocks carrying `input_json_delta`,
//!   and `delta.reasoning_content` becoming `thinking` / `thinking_delta`.
//!
//! Out of scope: multimodal content (`image`/`audio`/`video`/`document`,
//! including nested in `tool_result.content`) is rejected with a message the
//! caller maps to HTTP 400; Anthropic thinking signatures and
//! `redacted_thinking` history are not modelled.

use serde_json::{json, Map, Value};

/// Content-block types that carry non-text payloads. Accepting them would
/// silently drop the payload, so they are refused at the front door instead.
const MULTIMODAL_TYPES: [&str; 4] = ["image", "audio", "video", "document"];

/// dill STACK_GLOBAL for
/// `infera.engine.sglang.thinking_budget.Glm53ThinkingBudgetLogitProcessor`.
/// Keep in sync with `GLM53_THINKING_BUDGET_PROCESSOR_STR` in Python.
const GLM53_THINKING_BUDGET_PROCESSOR_STR: &str = concat!(
    r#"{"callable":""#,
    "8004954e000000000000008c24696e666572612e656e67696e652e73676c616e672e7468",
    "696e6b696e675f627564676574948c21476c6d35335468696e6b696e674275646765744c",
    "6f67697450726f636573736f729493942e",
    r#""}"#,
);

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

    // Extended thinking: Anthropic `thinking:{type}` -> engine
    // `chat_template_kwargs` (`enable_thinking` + `thinking`). The thinking
    // budget (default, `budget_tokens`, `-1` for none) -> SGLang hard cap for
    // GLM-5.3 model ids only.
    let thinking = obj.get("thinking").and_then(Value::as_object);
    if let Some(thinking) = thinking {
        match thinking.get("type").and_then(Value::as_str) {
            Some("enabled") => {
                out.insert(
                    "chat_template_kwargs".to_string(),
                    json!({"enable_thinking": true, "thinking": true}),
                );
            }
            Some("disabled") => {
                out.insert(
                    "chat_template_kwargs".to_string(),
                    json!({"enable_thinking": false, "thinking": false}),
                );
            }
            _ => {}
        }
    }
    apply_thinking_budget(
        &mut out,
        thinking,
        obj.get("max_tokens"),
        obj.get("model").and_then(Value::as_str),
    )?;

    Ok(Value::Object(out))
}

fn is_glm53_model(model: Option<&str>) -> bool {
    let Some(model) = model else {
        return false;
    };
    let m = model.to_ascii_lowercase().replace('_', "-");
    m.contains("glm-5-3") || m.contains("glm-5.3") || m.contains("glm53")
}

/// Tokens forced after the budget is reached: newline, then `</think>`.
const THINKING_FORCED_TOKENS: u64 = 2;
/// Upper bound of completion tokens kept for the answer after thinking ends.
const MAX_ANSWER_RESERVE: u64 = 1024;
/// Smallest `max_tokens` that fits one thinking token, the forced tokens and one answer token.
const MIN_MAX_TOKENS_FOR_BUDGET: u64 = 4;

/// Clamp `budget` so the forced tokens and an answer reserve fit in `max_tokens`.
fn resolve_thinking_budget(budget: u64, max_tokens: u64) -> u64 {
    let answer_reserve = (max_tokens / 4).clamp(1, MAX_ANSWER_RESERVE);
    let ceiling = max_tokens.saturating_sub(THINKING_FORCED_TOKENS + answer_reserve);
    budget.min(ceiling).max(1)
}

/// Budget used when a GLM-5.3 request does not carry `budget_tokens`.
const DEFAULT_THINKING_BUDGET: u64 = 1024;
/// `budget_tokens` value that leaves thinking uncapped.
const UNLIMITED_THINKING_BUDGET: i64 = -1;

/// Integer value of a JSON number (not a bool, string or float).
fn json_int(v: &Value) -> Option<i64> {
    v.as_i64()
        .or_else(|| v.as_u64().and_then(|n| i64::try_from(n).ok()))
}

/// Return the thinking budget to enforce for a GLM-5.3 request, or `None` for no cap.
///
/// `type=disabled` and `budget_tokens=-1` leave thinking uncapped. An omitted
/// budget uses `DEFAULT_THINKING_BUDGET`. The budget is clamped so the forced
/// tokens and an answer reserve fit in `max_tokens`; a missing `max_tokens`
/// needs no clamp. An omitted budget is skipped when `max_tokens` cannot fit
/// one, an explicit one is rejected.
fn resolve_request_thinking_budget(
    thinking: Option<&Map<String, Value>>,
    max_tokens: Option<&Value>,
) -> Result<Option<u64>, String> {
    if thinking.and_then(|t| t.get("type")).and_then(Value::as_str) == Some("disabled") {
        return Ok(None);
    }
    let max_tokens = match max_tokens {
        None | Some(Value::Null) => None,
        Some(v) => match json_int(v) {
            Some(n) if n >= MIN_MAX_TOKENS_FOR_BUDGET as i64 => Some(Some(n as u64)),
            _ => Some(None),
        },
    };
    let budget = match thinking.and_then(|t| t.get("budget_tokens")) {
        Some(v) => match json_int(v) {
            Some(UNLIMITED_THINKING_BUDGET) => return Ok(None),
            Some(n) if n >= 1 && !v.is_boolean() => n as u64,
            _ => return Err("thinking.budget_tokens must be a positive integer or -1".to_string()),
        },
        None => {
            if max_tokens == Some(None) {
                return Ok(None);
            }
            DEFAULT_THINKING_BUDGET
        }
    };
    match max_tokens {
        None => Ok(Some(budget)),
        Some(Some(n)) => Ok(Some(resolve_thinking_budget(budget, n))),
        Some(None) => Err(format!(
            "max_tokens must be >= {MIN_MAX_TOKENS_FOR_BUDGET} when thinking.budget_tokens is set"
        )),
    }
}

fn insert_thinking_budget(out: &mut Map<String, Value>, budget: u64) {
    out.insert(
        "custom_logit_processor".to_string(),
        json!(GLM53_THINKING_BUDGET_PROCESSOR_STR),
    );
    out.insert(
        "custom_params".to_string(),
        json!({"thinking_budget": budget}),
    );
}

/// Map the Anthropic thinking budget onto SGLang hard-cap fields (GLM-5.3 only).
fn apply_thinking_budget(
    out: &mut Map<String, Value>,
    thinking: Option<&Map<String, Value>>,
    max_tokens: Option<&Value>,
    model: Option<&str>,
) -> Result<(), String> {
    if !is_glm53_model(model) {
        return Ok(());
    }
    if let Some(budget) = resolve_request_thinking_budget(thinking, max_tokens)? {
        insert_thinking_budget(out, budget);
    }
    Ok(())
}

/// Apply the thinking controls of an OpenAI chat body in place (GLM-5.3 only).
///
/// The top-level `thinking` object maps to `chat_template_kwargs` without
/// overriding keys the client set, and is removed from the forwarded body. The
/// budget follows `resolve_request_thinking_budget` with `max_completion_tokens`
/// taking precedence over `max_tokens`. Thinking turned off through
/// `chat_template_kwargs` is not capped.
pub(crate) fn apply_openai_thinking_budget(body: &mut Value) -> Result<(), String> {
    let Some(obj) = body.as_object_mut() else {
        return Ok(());
    };
    if !is_glm53_model(obj.get("model").and_then(Value::as_str)) {
        return Ok(());
    }
    let thinking = match obj.get("thinking") {
        Some(Value::Object(_)) => obj.remove("thinking").and_then(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        }),
        _ => None,
    };
    if let Some(flag) = thinking
        .as_ref()
        .and_then(|t| t.get("type"))
        .and_then(Value::as_str)
        .and_then(|t| match t {
            "enabled" => Some(true),
            "disabled" => Some(false),
            _ => None,
        })
    {
        let kwargs = obj
            .entry("chat_template_kwargs")
            .or_insert_with(|| json!({}));
        if let Some(kwargs) = kwargs.as_object_mut() {
            kwargs.entry("enable_thinking").or_insert(json!(flag));
            kwargs.entry("thinking").or_insert(json!(flag));
        }
    }
    if let Some(kwargs) = obj.get("chat_template_kwargs").and_then(Value::as_object) {
        if kwargs.get("enable_thinking") == Some(&json!(false))
            || kwargs.get("thinking") == Some(&json!(false))
        {
            return Ok(());
        }
    }
    let max_tokens = match obj.get("max_completion_tokens") {
        Some(v) if !v.is_null() => Some(v.clone()),
        _ => obj.get("max_tokens").cloned(),
    };
    if let Some(budget) = resolve_request_thinking_budget(thinking.as_ref(), max_tokens.as_ref())? {
        insert_thinking_budget(obj, budget);
    }
    Ok(())
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
        let mut thinking_parts: Vec<&str> = Vec::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        text_parts.push(text);
                    }
                }
                Some("thinking") => {
                    // Multi-turn replay: map Anthropic thinking to engine
                    // `reasoning_content` so prior CoT is visible.
                    if let Some(thought) = block.get("thinking").and_then(Value::as_str) {
                        if !thought.is_empty() {
                            thinking_parts.push(thought);
                        }
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
        if !thinking_parts.is_empty() {
            msg.insert(
                "reasoning_content".to_string(),
                json!(thinking_parts.join("\n")),
            );
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
    if let Some(reasoning) = message
        .and_then(|m| m.get("reasoning_content"))
        .and_then(Value::as_str)
    {
        if !reasoning.is_empty() {
            content.push(json!({
                "type": "thinking",
                "thinking": reasoning,
                "signature": "",
            }));
        }
    }
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

/// What the caller should do after one translated SSE line.
enum LineOutcome {
    /// Read on.
    Continue,
    /// The stream is over; `done` is already set.
    Stop,
    /// The line carried a `data:` payload that is not valid JSON.
    Unparsed,
}

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
    thinking_index: Option<u64>,
    text_index: Option<u64>,
    /// OpenAI `delta.tool_calls[].index` -> Anthropic content block index,
    /// in the order the tool calls first appeared.
    tool_indices: Vec<(i64, u64)>,
    next_block_index: u64,
    finish_reason: Option<String>,
    output_tokens: u64,
    /// Prompt tokens: the router's estimate until the engine reports usage.
    input_tokens: Option<u64>,
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
            thinking_index: None,
            text_index: None,
            tool_indices: Vec::new(),
            next_block_index: 0,
            finish_reason: None,
            output_tokens: 0,
            input_tokens: None,
        }
    }

    /// Seed `message_start.usage.input_tokens` with the router's prompt token
    /// count; the engine's final usage chunk replaces it in `message_delta`.
    pub fn with_input_tokens(mut self, input_tokens: Option<u64>) -> Self {
        self.input_tokens = input_tokens.filter(|n| *n > 0);
        self
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
            if let LineOutcome::Stop = self.handle_line(&line, &mut out) {
                self.buf.clear();
                return out;
            }
        }
        out
    }

    /// Translate one SSE line. Mid-stream, an unparseable `data:` payload is
    /// skipped rather than failed: the client already holds a 200 and part of
    /// a body. Only [`Self::finish`] treats it as evidence of a cut stream.
    fn handle_line(&mut self, line: &[u8], out: &mut Vec<u8>) -> LineOutcome {
        let line = line.trim_ascii();
        if line.is_empty() || line.starts_with(b":") || !line.starts_with(b"data:") {
            return LineOutcome::Continue;
        }
        let data = line["data:".len()..].trim_ascii();
        if data == b"[DONE]" {
            if self.started {
                self.close(out);
            }
            self.done = true;
            return LineOutcome::Stop;
        }
        match serde_json::from_slice::<Value>(data) {
            Ok(obj) => {
                self.handle_chunk(&obj, out);
                if self.done {
                    LineOutcome::Stop
                } else {
                    LineOutcome::Continue
                }
            }
            Err(_) => LineOutcome::Unparsed,
        }
    }

    /// Emit the terminal events for a stream that ended without `[DONE]`.
    /// Idempotent. A stream that never started is an error, not a silent 200.
    pub fn finish(&mut self) -> Vec<u8> {
        if self.done {
            return Vec::new();
        }
        let mut out = Vec::new();
        // The engine's last line may never have been newline-terminated.
        // Dropping it would close the stream over content the client still
        // needs, and a tool-call fragment lost that way arrives as a JSON
        // parse error the client can only attribute to the model.
        let tail = std::mem::take(&mut self.buf);
        let outcome = self.handle_line(&tail, &mut out);
        if let LineOutcome::Stop = outcome {
            return out;
        }
        if !self.started {
            out.append(&mut self.error("worker stream ended before any tokens"));
            return out;
        }
        if let LineOutcome::Unparsed = outcome {
            // Half a `data:` payload means bytes were lost in transit. Closing
            // normally would hand the client a truncated answer wrapped in a
            // clean `message_stop`, which reads as a model defect.
            out.append(&mut self.error("worker stream ended mid-event"));
            return out;
        }
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
                let usage = obj.get("usage");
                if let Some(n) = usage
                    .and_then(|u| u.get("completion_tokens"))
                    .and_then(Value::as_u64)
                {
                    self.output_tokens = self.output_tokens.max(n);
                }
                if let Some(n) = usage
                    .and_then(|u| u.get("prompt_tokens"))
                    .and_then(Value::as_u64)
                    .filter(|n| *n > 0)
                {
                    self.input_tokens = Some(n);
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
                        "usage": {"input_tokens": self.input_tokens.unwrap_or(0), "output_tokens": 0},
                    }
                }),
            );
            self.started = true;
        }

        let delta = match choice.get("delta") {
            Some(d) => d,
            None => return,
        };

        if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
            if !reasoning.is_empty() {
                self.close_text_if_open(out);
                let index = match self.thinking_index {
                    Some(i) => i,
                    None => {
                        let i = self.next_block_index;
                        self.next_block_index += 1;
                        self.thinking_index = Some(i);
                        push_event(
                            out,
                            "content_block_start",
                            json!({
                                "index": i,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "",
                                },
                            }),
                        );
                        i
                    }
                };
                push_event(
                    out,
                    "content_block_delta",
                    json!({
                        "index": index,
                        "delta": {"type": "thinking_delta", "thinking": reasoning},
                    }),
                );
            }
        }

        if let Some(text) = delta.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                self.close_thinking_if_open(out);
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
                        self.close_thinking_if_open(out);
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

    /// Close an open thinking block before text/tool content starts.
    fn close_thinking_if_open(&mut self, out: &mut Vec<u8>) {
        if let Some(index) = self.thinking_index.take() {
            push_event(out, "content_block_stop", json!({ "index": index }));
        }
    }

    /// Close an open text block before a later thinking block starts.
    fn close_text_if_open(&mut self, out: &mut Vec<u8>) {
        if let Some(index) = self.text_index.take() {
            push_event(out, "content_block_stop", json!({ "index": index }));
        }
    }

    /// Emit `content_block_stop` for every open block, then `message_delta`
    /// and `message_stop`.
    fn close(&self, out: &mut Vec<u8>) {
        // Deterministic order: thinking (if still open), text, then tools.
        // Only the stop-per-open-block invariant matters for protocol
        // correctness; order matches Anthropic's usual content shape.
        let mut indices: Vec<u64> = Vec::new();
        if let Some(i) = self.thinking_index {
            indices.push(i);
        }
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
                "usage": self.final_usage(),
            }),
        );
        push_event(out, "message_stop", json!({}));
    }
}

impl SseTranslator {
    /// `message_delta.usage`: output tokens, plus input and cache tokens when known.
    fn final_usage(&self) -> Value {
        match self.input_tokens {
            Some(input) => json!({
                "input_tokens": input,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "output_tokens": self.output_tokens,
            }),
            None => json!({"output_tokens": self.output_tokens}),
        }
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

    fn stream_events(t: &mut SseTranslator, chunks: &[&str]) -> Vec<(String, Value)> {
        let mut raw = Vec::new();
        for c in chunks {
            raw.extend(t.push(c.as_bytes()));
        }
        raw.extend(t.finish());
        let text = String::from_utf8(raw).unwrap();
        let mut out = Vec::new();
        let mut name = String::new();
        for line in text.lines() {
            if let Some(n) = line.strip_prefix("event: ") {
                name = n.to_string();
            } else if let Some(d) = line.strip_prefix("data: ") {
                out.push((name.clone(), serde_json::from_str(d).unwrap()));
            }
        }
        out
    }

    fn event<'a>(events: &'a [(String, Value)], name: &str) -> &'a Value {
        &events.iter().find(|(n, _)| n == name).unwrap().1
    }

    const TEXT_CHUNK: &str = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    const STOP_CHUNK: &str = "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    const USAGE_CHUNK: &str =
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3015,\"completion_tokens\":61}}\n\n";
    const DONE_CHUNK: &str = "data: [DONE]\n\n";

    #[test]
    fn message_delta_reports_input_tokens_from_usage_chunk() {
        let mut t = SseTranslator::new("m", Some("abc"));
        let events = stream_events(&mut t, &[TEXT_CHUNK, STOP_CHUNK, USAGE_CHUNK, DONE_CHUNK]);
        let usage = &event(&events, "message_delta")["usage"];
        assert_eq!(usage["input_tokens"], 3015);
        assert_eq!(usage["output_tokens"], 61);
        assert_eq!(usage["cache_read_input_tokens"], 0);
        assert_eq!(usage["cache_creation_input_tokens"], 0);
    }

    #[test]
    fn message_start_uses_the_input_token_estimate() {
        let mut t = SseTranslator::new("m", Some("abc")).with_input_tokens(Some(3015));
        let events = stream_events(&mut t, &[TEXT_CHUNK, STOP_CHUNK, DONE_CHUNK]);
        assert_eq!(
            event(&events, "message_start")["message"]["usage"]["input_tokens"],
            3015
        );
        assert_eq!(
            event(&events, "message_delta")["usage"]["input_tokens"],
            3015
        );
    }

    #[test]
    fn reported_usage_overrides_the_input_token_estimate() {
        let mut t = SseTranslator::new("m", Some("abc")).with_input_tokens(Some(3000));
        let events = stream_events(&mut t, &[TEXT_CHUNK, STOP_CHUNK, USAGE_CHUNK, DONE_CHUNK]);
        assert_eq!(
            event(&events, "message_start")["message"]["usage"]["input_tokens"],
            3000
        );
        assert_eq!(
            event(&events, "message_delta")["usage"]["input_tokens"],
            3015
        );
    }

    #[test]
    fn unknown_input_tokens_keep_zero_start_and_omit_the_delta_field() {
        let mut t = SseTranslator::new("m", Some("abc"));
        let events = stream_events(&mut t, &[TEXT_CHUNK, STOP_CHUNK, DONE_CHUNK]);
        assert_eq!(
            event(&events, "message_start")["message"]["usage"]["input_tokens"],
            0
        );
        assert!(event(&events, "message_delta")["usage"]
            .get("input_tokens")
            .is_none());
    }

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

    #[test]
    fn a_tail_line_without_a_newline_still_reaches_the_client() {
        // An upstream that ends its body on a line it never terminated leaves
        // that line in the buffer. Closing the stream over it hands the client
        // a well-formed `message_stop` above a tool input missing its last
        // fragment, which is a parse error the client blames on the model.
        let mut t = SseTranslator::new("m", Some("abc"));
        let opened = String::from_utf8(t.push(
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"f","arguments":"{\"a\":1"}}]}}]}
"#,
        ))
        .unwrap();
        assert!(opened.contains("content_block_start"));

        let unterminated =
            br#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}"#;
        let mid = String::from_utf8(t.push(unterminated)).unwrap();
        let tail = String::from_utf8(t.finish()).unwrap();
        let out = format!("{mid}{tail}");

        assert!(
            out.contains("input_json_delta"),
            "the closing argument fragment was dropped: {out}"
        );
    }

    #[test]
    fn a_body_cut_mid_event_is_reported_rather_than_closed() {
        // Half a `data:` payload is unrecoverable. Emitting `message_stop` over
        // it presents the loss as a finished answer, so the client consumes a
        // truncated result instead of retrying.
        let mut t = SseTranslator::new("m", Some("abc"));
        t.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        t.push(b"data: {\"choices\":[{\"delta\":{\"cont");
        let out = String::from_utf8(t.finish()).unwrap();

        assert!(out.contains("event: error"), "{out}");
        assert!(!out.contains("message_stop"), "{out}");
    }

    #[test]
    fn a_clean_body_without_done_still_closes_normally() {
        // Ending on a terminated line leaves nothing buffered, so this keeps
        // the existing contract: no `[DONE]` is not by itself a truncation.
        let mut t = SseTranslator::new("m", Some("abc"));
        t.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        let out = String::from_utf8(t.finish()).unwrap();

        assert!(out.contains("message_stop"), "{out}");
        assert!(!out.contains("event: error"), "{out}");
    }

    #[test]
    fn reasoning_content_becomes_thinking_deltas() {
        let mut t = SseTranslator::new("m", Some("abc"));
        let chunk = concat!(
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think \"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"more\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\n",
            "data: [DONE]\n\n",
        );
        let out = String::from_utf8(t.push(chunk.as_bytes())).unwrap();
        assert!(out.contains("\"type\":\"thinking\""), "{out}");
        assert!(out.contains("thinking_delta"), "{out}");
        assert!(out.contains("think "), "{out}");
        assert!(out.contains("more"), "{out}");
        assert!(out.contains("text_delta"), "{out}");
        assert!(out.contains("answer"), "{out}");
        let thinking_pos = out.find("\"type\":\"thinking\"").unwrap();
        let text_pos = out.find("\"type\":\"text\"").unwrap();
        assert!(thinking_pos < text_pos, "{out}");
        let thinking_stop = out[thinking_pos..]
            .find("event: content_block_stop")
            .map(|i| thinking_pos + i)
            .unwrap();
        assert!(thinking_stop < text_pos, "{out}");
    }

    #[test]
    fn unary_reasoning_content_becomes_thinking_block() {
        let openai = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "chain",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        });
        let out = translate_response(&openai, Some("m"), Some("req1"));
        let content = out.get("content").and_then(Value::as_array).unwrap();
        assert_eq!(
            content[0],
            json!({"type": "thinking", "thinking": "chain", "signature": ""})
        );
        assert_eq!(content[1], json!({"type": "text", "text": "answer"}));
    }

    #[test]
    fn request_thinking_enabled_sets_chat_template_kwargs() {
        let body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "thinking": {"type": "enabled"},
        });
        let out = translate_request(&body).unwrap();
        assert_eq!(
            out.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": true, "thinking": true}))
        );
        assert!(out.get("custom_logit_processor").is_none());
        assert!(out.get("custom_params").is_none());
    }

    #[test]
    fn request_thinking_budget_tokens_maps_to_sglang_hard_cap() {
        let body = json!({
            "model": "glm-5-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let out = translate_request(&body).unwrap();
        assert_eq!(
            out.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": true, "thinking": true}))
        );
        assert_eq!(
            out.get("custom_logit_processor"),
            Some(&json!(GLM53_THINKING_BUDGET_PROCESSOR_STR))
        );
        assert_eq!(
            out.get("custom_params"),
            Some(&json!({"thinking_budget": 1024}))
        );
    }

    #[test]
    fn request_thinking_budget_ignored_for_non_glm53() {
        let body = json!({
            "model": "qwen3-32b",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        let out = translate_request(&body).unwrap();
        assert!(out.get("custom_logit_processor").is_none());
        assert!(out.get("custom_params").is_none());
    }

    #[test]
    fn request_thinking_budget_tokens_clamped_to_leave_close_and_answer_room() {
        let body = json!({
            "model": "glm-5-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        });
        let out = translate_request(&body).unwrap();
        assert_eq!(
            out.get("custom_params"),
            Some(&json!({"thinking_budget": 8192 - 2 - 1024}))
        );
    }

    fn budget_of(out: &Value) -> Option<u64> {
        out.get("custom_params")
            .and_then(|p| p.get("thinking_budget"))
            .and_then(Value::as_u64)
    }

    #[test]
    fn request_thinking_budget_defaults_to_1024_without_budget_tokens() {
        for thinking in [Some(json!({"type": "enabled"})), None] {
            let mut body = json!({
                "model": "glm-5-3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
            });
            if let Some(t) = thinking {
                body["thinking"] = t;
            }
            let out = translate_request(&body).unwrap();
            assert_eq!(budget_of(&out), Some(1024));
        }
    }

    #[test]
    fn request_thinking_budget_minus_one_and_disabled_leave_thinking_uncapped() {
        for thinking in [
            json!({"type": "enabled", "budget_tokens": -1}),
            json!({"type": "disabled"}),
        ] {
            let body = json!({
                "model": "glm-5-3",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4096,
                "thinking": thinking,
            });
            let out = translate_request(&body).unwrap();
            assert!(out.get("custom_logit_processor").is_none());
            assert!(out.get("custom_params").is_none());
        }
    }

    fn resolve(thinking: Value, max_tokens: Value) -> Result<Option<u64>, String> {
        resolve_request_thinking_budget(thinking.as_object(), Some(&max_tokens))
    }

    #[test]
    fn resolve_request_budget_default_clamp_and_skip() {
        assert_eq!(resolve(json!({}), json!(4096)), Ok(Some(1024)));
        assert_eq!(resolve(json!({}), json!(1000)), Ok(Some(1000 - 2 - 250)));
        assert_eq!(resolve(json!({}), json!(3)), Ok(None));
        assert_eq!(resolve(json!({}), json!("x")), Ok(None));
        assert_eq!(resolve_request_thinking_budget(None, None), Ok(Some(1024)));
        assert_eq!(
            resolve(json!({"budget_tokens": 10000}), json!(8192)),
            Ok(Some(8192 - 2 - 1024))
        );
        assert_eq!(
            resolve_request_thinking_budget(json!({"budget_tokens": 2048}).as_object(), None),
            Ok(Some(2048))
        );
    }

    #[test]
    fn resolve_request_budget_rejects_invalid_explicit_values() {
        for bad in [json!(0), json!(-2), json!(true), json!("1024"), json!(1.5)] {
            let err = resolve(json!({"budget_tokens": bad}), json!(4096)).unwrap_err();
            assert!(err.contains("positive integer or -1"), "{err}");
        }
        let err = resolve(json!({"budget_tokens": 1024}), json!(3)).unwrap_err();
        assert!(err.contains("max_tokens must be >= 4"), "{err}");
        assert_eq!(resolve(json!({"budget_tokens": -1}), json!(4096)), Ok(None));
        assert_eq!(resolve(json!({"type": "disabled"}), json!(4096)), Ok(None));
    }

    fn openai(mut body: Value) -> Result<Value, String> {
        apply_openai_thinking_budget(&mut body)?;
        Ok(body)
    }

    #[test]
    fn openai_glm53_body_gets_the_default_budget() {
        let out = openai(json!({"model": "glm-5-3", "messages": [], "max_tokens": 4096})).unwrap();
        assert_eq!(
            out.get("custom_logit_processor"),
            Some(&json!(GLM53_THINKING_BUDGET_PROCESSOR_STR))
        );
        assert_eq!(budget_of(&out), Some(1024));
        assert!(out.get("chat_template_kwargs").is_none());
    }

    #[test]
    fn openai_thinking_object_sets_budget_and_template_kwargs() {
        let out = openai(json!({
            "model": "glm-5-3",
            "max_completion_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
        }))
        .unwrap();
        assert_eq!(budget_of(&out), Some(2000));
        assert_eq!(
            out.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": true, "thinking": true}))
        );
        assert!(out.get("thinking").is_none());
    }

    #[test]
    fn openai_max_completion_tokens_wins_over_max_tokens() {
        let out = openai(json!({
            "model": "glm-5-3",
            "max_tokens": 100000,
            "max_completion_tokens": 1000,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
        }))
        .unwrap();
        assert_eq!(budget_of(&out), Some(1000 - 2 - 250));
    }

    #[test]
    fn openai_minus_one_and_disabled_are_not_capped() {
        let out = openai(json!({
            "model": "glm-5-3",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": -1},
        }))
        .unwrap();
        assert!(out.get("custom_logit_processor").is_none());
        assert!(out.get("custom_params").is_none());
        let out = openai(json!({
            "model": "glm-5-3",
            "max_tokens": 4096,
            "thinking": {"type": "disabled"},
        }))
        .unwrap();
        assert!(out.get("custom_logit_processor").is_none());
        assert_eq!(
            out.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": false, "thinking": false}))
        );
        let out = openai(json!({
            "model": "glm-5-3",
            "max_tokens": 4096,
            "chat_template_kwargs": {"enable_thinking": false},
        }))
        .unwrap();
        assert!(out.get("custom_logit_processor").is_none());
    }

    #[test]
    fn openai_explicit_template_kwargs_are_not_overridden() {
        let out = openai(json!({
            "model": "glm-5-3",
            "max_tokens": 4096,
            "thinking": {"type": "enabled"},
            "chat_template_kwargs": {"enable_thinking": true, "extra": 1},
        }))
        .unwrap();
        assert_eq!(
            out.get("chat_template_kwargs"),
            Some(&json!({"enable_thinking": true, "extra": 1, "thinking": true}))
        );
    }

    #[test]
    fn openai_other_models_are_left_untouched_and_invalid_budget_errors() {
        let body =
            json!({"model": "qwen3-32b", "max_tokens": 4096, "thinking": {"type": "enabled"}});
        assert_eq!(openai(body.clone()).unwrap(), body);
        let err = openai(json!({
            "model": "glm-5-3",
            "thinking": {"type": "enabled", "budget_tokens": 0},
        }))
        .unwrap_err();
        assert!(err.contains("positive integer or -1"), "{err}");
    }

    #[test]
    fn resolve_thinking_budget_matches_python_ceiling() {
        assert_eq!(resolve_thinking_budget(1024, 4096), 1024);
        assert_eq!(resolve_thinking_budget(31999, 32000), 32000 - 2 - 1024);
        assert_eq!(resolve_thinking_budget(100000, 32000), 32000 - 2 - 1024);
        assert_eq!(resolve_thinking_budget(1, 4), 1);
        assert_eq!(resolve_thinking_budget(100, 10), 10 - 2 - 2);
    }

    #[test]
    fn request_thinking_budget_rejects_max_tokens_below_minimum() {
        let body = json!({
            "model": "glm-5-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 3,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        });
        assert!(translate_request(&body).is_err());
    }

    #[test]
    fn request_assistant_thinking_maps_to_reasoning_content() {
        let body = json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "step 1"},
                        {"type": "text", "text": "answer"},
                    ],
                },
            ],
            "max_tokens": 16,
        });
        let out = translate_request(&body).unwrap();
        let messages = out.get("messages").and_then(Value::as_array).unwrap();
        let assistant = &messages[1];
        assert_eq!(assistant.get("content"), Some(&json!("answer")));
        assert_eq!(assistant.get("reasoning_content"), Some(&json!("step 1")));
    }
}
