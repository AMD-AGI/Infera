// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

//! `/v1/responses` bodies, normalised into the chat body the engine renders.
//!
//! The Responses API carries its conversation in `input`, not `messages`. Every
//! renderer in `block_hasher` keys off `messages`/`prompt`, so a Responses body
//! hashes to nothing and kv-aware silently degrades to load balancing — measured
//! on a live Kimi-K3 PD fleet as `request_blocks=0` on 462 of 528 routing
//! decisions, every one of them Codex orchestration traffic, while
//! `/v1/chat/completions` through the same router scored up to 94/118 hits.
//!
//! Rather than teach each renderer a second schema, this reproduces
//! `OpenAIServingResponses._make_request`: it builds the very
//! `ChatCompletionRequest` the engine builds, and hands that to the encoders
//! unchanged. `OpenAIServingResponses` subclasses `OpenAIServingChat` and reuses
//! `_process_messages`, so once the body is in chat shape the two endpoints
//! render through identical code — which is the only way to stay byte-aligned.
//!
//! The harmony path (`_make_request_with_harmony`) is not modelled: `use_harmony`
//! is `hf_config.model_type == "gpt_oss"`, a model-level property, and gpt-oss
//! ships a Jinja chat template that the ordinary `render_text` path handles.
//!
//! Refuses (returns `None`) rather than guessing, matching `render_dsv4`: a
//! prefix a few tokens off the engine's does not error, it silently never hits
//! the cache again, and refusing costs only the load-based routing an empty
//! render already gives.

use serde_json::{json, Map, Value};

/// Python truthiness, for the shapes the reference tests with a bare `if`.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
    }
}

/// True when this body is a Responses request rather than a chat/completion one.
pub fn is_responses_body(body: &Value) -> bool {
    body.get("messages").is_none() && body.get("input").is_some()
}

/// `_construct_input_messages` + `_make_request`, as a chat body.
///
/// `None` means "cannot reproduce" — the caller must fall back to load routing.
pub fn to_chat_body(body: &Value) -> Option<Value> {
    // Conversation history lives in the engine's in-process `msg_store`, keyed by
    // a response id the router never sees. Nothing here can reconstruct it, and a
    // body missing its history hashes a prefix that is real but wrong — it would
    // route to a worker holding a *different* conversation. Refuse.
    if body
        .get("previous_response_id")
        .is_some_and(|v| !v.is_null())
    {
        return None;
    }
    // `store=true` is what makes a *later* turn carry `previous_response_id`; this
    // turn is still self-contained, so it stays reproducible.

    let mut messages: Vec<Value> = Vec::new();

    // `if request.instructions:` — Python truthiness, so `""` contributes nothing.
    if truthy(body.get("instructions")) {
        messages.push(json!({
            "role": "system",
            "content": body.get("instructions")?.clone(),
        }));
    }

    match body.get("input") {
        // "Responses API supports simple text inputs without chat format."
        Some(Value::String(s)) => messages.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for item in items {
                // `_normalize_response_message_for_chat` returns None for an empty
                // reasoning item, which the reference skips.
                if let Some(m) = normalize_item(item)? {
                    messages.push(m);
                }
            }
        }
        _ => return None,
    }

    let messages = merge_consecutive_assistant_messages(messages);
    let messages = coalesce_system_messages(messages);

    let mut out = Map::new();
    out.insert("messages".to_string(), Value::Array(messages));

    // `tools=chat_tools or None` — an all-builtin tool list yields no chat tools.
    if let Some(tools) = response_tools_to_chat_tools(body)? {
        out.insert("tools".to_string(), Value::Array(tools));
    }

    // `reasoning_effort=request.reasoning.effort if request.reasoning else None`.
    // The Responses-level `reasoning` object is consumed here and deliberately not
    // forwarded: `encode_chat` refuses any body carrying one, because on the *chat*
    // path it rewrites `reasoning_effort` in ways that port does not model. On this
    // path the engine reads nothing from it but `.effort`, which is now explicit.
    if let Some(reasoning) = body.get("reasoning").filter(|v| !v.is_null()) {
        // Anything other than an object is a 422 upstream, never a render.
        let obj = reasoning.as_object()?;
        if let Some(effort) = obj.get("effort").filter(|v| !v.is_null()) {
            out.insert("reasoning_effort".to_string(), effort.clone());
        }
    }

    Some(Value::Object(out))
}

/// `_response_tools_to_chat_tools`.
///
/// The engine constructs `Tool(type=..., function=Function(name=..., description=...,
/// parameters=..., strict=...))` with all four function fields passed explicitly, so
/// pydantic marks all four *set* and `model_dump(exclude_unset=True)` — which
/// `encoding_k3::dump_tools` emulates by copying the keys present in this JSON —
/// keeps them even when they are null. Emitting them unconditionally is therefore
/// required, not sloppy: dropping a null `strict` here renders a shorter tool
/// declaration than the engine's and misses the cache on every subsequent turn.
fn response_tools_to_chat_tools(body: &Value) -> Option<Option<Vec<Value>>> {
    let Some(tools) = body.get("tools").filter(|v| !v.is_null()) else {
        return Some(None);
    };
    let arr = tools.as_array()?;
    let mut out = Vec::new();
    for tool in arr {
        let obj = tool.as_object()?;
        // "Only `function` tools flow to chat; built-ins go through harmony."
        if obj.get("type").and_then(Value::as_str) != Some("function") {
            continue;
        }
        out.push(json!({
            "type": "function",
            "function": {
                "name": obj.get("name").cloned().unwrap_or(Value::Null),
                "description": obj.get("description").cloned().unwrap_or(Value::Null),
                "parameters": obj.get("parameters").cloned().unwrap_or(Value::Null),
                "strict": obj.get("strict").cloned().unwrap_or(Value::Null),
            },
        }));
    }
    Some(if out.is_empty() { None } else { Some(out) })
}

/// `_normalize_response_message_for_chat`.
///
/// `Ok(None)` skips the item (an empty reasoning block); the outer `None` refuses
/// the whole request, standing in for the reference's `ValueError`.
fn normalize_item(item: &Value) -> Option<Option<Value>> {
    // "if not isinstance(message, dict): return message" — a bare scalar reaches
    // the chat request as-is and blows up in rendering, so it is a 422 in practice.
    let obj = item.as_object()?;

    // "collapse `developer` to `system` at the boundary"
    let role = match obj.get("role").and_then(Value::as_str) {
        Some("developer") => Some("system"),
        other => other,
    };

    match obj.get("type").and_then(Value::as_str) {
        Some("function_call") => {
            let arguments = match obj.get("arguments") {
                // A string that parses to a JSON object is passed through
                // *verbatim* — not re-serialised — so byte-for-byte what the
                // client sent. Anything else becomes "{}".
                Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                    Ok(Value::Object(_)) if !s.is_empty() => s.clone(),
                    _ => "{}".to_string(),
                },
                Some(v @ Value::Object(_)) => serde_json::to_string(v).ok()?,
                _ => "{}".to_string(),
            };
            // `message.get("call_id") or message.get("id")` — Python `or`, so an
            // empty-string call_id falls through to `id`.
            let id = match obj.get("call_id") {
                Some(v) if truthy(Some(v)) => v.clone(),
                _ => obj.get("id").cloned().unwrap_or(Value::Null),
            };
            Some(Some(json!({
                "role": "assistant",
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": obj.get("name").cloned().unwrap_or(Value::Null),
                        "arguments": arguments,
                    },
                }],
            })))
        }
        Some("function_call_output") => Some(Some(json!({
            "role": "tool",
            "tool_call_id": obj.get("call_id").cloned().unwrap_or(Value::Null),
            "content": obj.get("output").cloned().unwrap_or_else(|| json!("")),
        }))),
        Some("reasoning") => {
            // "Prefer `summary`; fall back to `content` only when summary is empty,
            // since clients often populate both with the same text."
            let mut parts = collect_texts(obj.get("summary"));
            if parts.is_empty() {
                parts = collect_texts(obj.get("content"));
            }
            if parts.is_empty() {
                return Some(None);
            }
            Some(Some(json!({
                "role": "assistant",
                "reasoning_content": parts.join("\n"),
            })))
        }
        None | Some("message") => {
            let mut m = obj.clone();
            if let Some(r) = role {
                m.insert("role".to_string(), Value::String(r.to_string()));
            }
            if let Some(Value::Array(parts)) = obj.get("content") {
                let normalized: Vec<Value> = parts.iter().map(normalize_content_part).collect();
                m.insert("content".to_string(), Value::Array(normalized));
            }
            // "if v is not None and k not in ('id', 'status', 'type')"
            m.retain(|k, v| !v.is_null() && !matches!(k.as_str(), "id" | "status" | "type"));
            Some(Some(Value::Object(m)))
        }
        // `raise ValueError(f"Unsupported Responses API input item type: ...")`.
        Some(_) => None,
    }
}

/// The `_collect` closure inside the reasoning branch: non-empty `text` fields.
fn collect_texts(parts: Option<&Value>) -> Vec<String> {
    let Some(Value::Array(arr)) = parts else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|e| e.as_object()?.get("text")?.as_str())
        .filter(|t| !t.is_empty())
        .map(str::to_string)
        .collect()
}

/// `_normalize_response_content_part_for_chat`.
///
/// Image parts are normalised but not made hashable — `encoding_k3::render_content`
/// refuses any part without a `text`, so a multimodal turn declines downstream on
/// its own rather than hashing a prefix that omits the vision tokens.
fn normalize_content_part(part: &Value) -> Value {
    let Some(obj) = part.as_object() else {
        return part.clone();
    };
    match obj.get("type").and_then(Value::as_str) {
        Some("input_text") | Some("output_text") => json!({
            "type": "text",
            "text": obj.get("text").cloned().unwrap_or_else(|| json!("")),
        }),
        Some("input_image") => {
            let mut image_url = match obj.get("image_url") {
                Some(Value::Object(m)) => m.clone(),
                other => {
                    let mut m = Map::new();
                    m.insert("url".to_string(), other.cloned().unwrap_or(Value::Null));
                    m
                }
            };
            if !truthy(image_url.get("detail")) {
                let detail = obj.get("detail").filter(|v| truthy(Some(v))).cloned();
                image_url.insert(
                    "detail".to_string(),
                    detail.unwrap_or_else(|| json!("auto")),
                );
            }
            for key in ["min_dynamic_patch", "max_dynamic_patch"] {
                if let Some(v) = obj.get(key) {
                    image_url
                        .entry(key.to_string())
                        .or_insert_with(|| v.clone());
                }
            }
            json!({"type": "image_url", "image_url": Value::Object(image_url)})
        }
        Some("image_url") => {
            let mut out = obj.clone();
            let image_url = match obj.get("image_url") {
                Some(Value::String(url)) => json!({
                    "url": url,
                    "detail": obj.get("detail").cloned().unwrap_or_else(|| json!("auto")),
                }),
                Some(Value::Object(m)) => {
                    let mut m = m.clone();
                    if !truthy(m.get("detail")) {
                        let detail = obj.get("detail").filter(|v| truthy(Some(v))).cloned();
                        m.insert(
                            "detail".to_string(),
                            detail.unwrap_or_else(|| json!("auto")),
                        );
                    }
                    Value::Object(m)
                }
                other => other.cloned().unwrap_or(Value::Null),
            };
            out.insert("image_url".to_string(), image_url);
            Value::Object(out)
        }
        // "text" and everything unrecognised pass through untouched.
        _ => part.clone(),
    }
}

/// `_merge_consecutive_assistant_messages`.
fn merge_consecutive_assistant_messages(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        let is_assistant = |v: &Value| v.get("role").and_then(Value::as_str) == Some("assistant");
        if is_assistant(&msg) && merged.last().is_some_and(is_assistant) {
            let prev = merged.last_mut().expect("checked above");
            merge_into_assistant(prev, &msg);
            continue;
        }
        merged.push(msg);
    }
    merged
}

fn merge_into_assistant(prev: &mut Value, msg: &Value) {
    let Some(prev_obj) = prev.as_object_mut() else {
        return;
    };
    if let Some(new_content) = msg
        .get("content")
        .filter(|c| truthy(Some(c)) || c.is_array())
    {
        // `if new_content is not None and new_content != ""` — a non-empty list or
        // any non-empty string; an empty list is falsy for `truthy` but still != "".
        if !(new_content.is_string() && !truthy(Some(new_content))) {
            let prev_content = prev_obj.get("content").cloned().unwrap_or(Value::Null);
            let prev_empty = prev_content.is_null()
                || (prev_content.is_string() && !truthy(Some(&prev_content)));
            let joined = if prev_empty {
                new_content.clone()
            } else if let (Value::String(a), Value::String(b)) = (&prev_content, new_content) {
                // `sep = "\n\n" if prev_content and new_content else ""`
                let sep = if !a.is_empty() && !b.is_empty() {
                    "\n\n"
                } else {
                    ""
                };
                Value::String(format!("{a}{sep}{b}"))
            } else {
                // "Lift mixed str/list content to list parts so non-text parts
                // survive when the two sides differ in shape."
                let mut parts = as_parts(&prev_content);
                parts.extend(as_parts(new_content));
                Value::Array(parts)
            };
            prev_obj.insert("content".to_string(), joined);
        }
    }
    if let Some(Value::Array(new_calls)) = msg.get("tool_calls").filter(|v| truthy(Some(v))) {
        let mut calls = match prev_obj.get("tool_calls") {
            Some(Value::Array(a)) => a.clone(),
            _ => Vec::new(),
        };
        calls.extend(new_calls.iter().cloned());
        prev_obj.insert("tool_calls".to_string(), Value::Array(calls));
    }
    if let Some(new_reasoning) = msg
        .get("reasoning_content")
        .filter(|v| truthy(Some(v)))
        .and_then(Value::as_str)
    {
        let joined = match prev_obj.get("reasoning_content").and_then(Value::as_str) {
            Some(p) if !p.is_empty() => format!("{p}\n{new_reasoning}"),
            _ => new_reasoning.to_string(),
        };
        prev_obj.insert("reasoning_content".to_string(), Value::String(joined));
    }
}

/// The `_as_parts` closure: a list stays a list, a non-empty string becomes one
/// text part, anything else contributes nothing.
fn as_parts(c: &Value) -> Vec<Value> {
    match c {
        Value::Array(a) => a.clone(),
        Value::String(s) if !s.is_empty() => vec![json!({"type": "text", "text": s})],
        _ => Vec::new(),
    }
}

/// The `system_chunks` pass at the tail of `_construct_input_messages`: "Most chat
/// templates expect a single leading `system` message; coalesce any `instructions`
/// + interleaved `developer` entries."
fn coalesce_system_messages(messages: Vec<Value>) -> Vec<Value> {
    let mut chunks: Vec<String> = Vec::new();
    let mut others: Vec<Value> = Vec::with_capacity(messages.len());
    for m in messages {
        if m.get("role").and_then(Value::as_str) == Some("system") {
            match m.get("content") {
                Some(Value::String(s)) => chunks.push(s.clone()),
                Some(Value::Array(parts)) => {
                    for p in parts {
                        if let Some(t) = p.as_object().and_then(|o| o.get("text")) {
                            if let Some(t) = t.as_str() {
                                chunks.push(t.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
            // The reference drops the original message either way; a system entry
            // whose content is neither str nor list contributes nothing and
            // disappears. Matching that exactly matters more than it looks: an
            // extra empty system block shifts every token after it.
            continue;
        }
        others.push(m);
    }
    if chunks.is_empty() {
        return others;
    }
    let mut out = Vec::with_capacity(others.len() + 1);
    out.push(json!({"role": "system", "content": chunks.join("\n\n")}));
    out.extend(others);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding_k3::encode_chat;

    /// The assertion that actually matters: a Responses body and the chat body a
    /// client would have sent for the same turn must render to the *same* K3
    /// segments. Comparing normalised JSON would only prove this module matches
    /// itself; comparing renders proves the router matches the engine.
    fn assert_renders_same(responses: &Value, chat: &Value) {
        let flat = |b: &Value| -> String {
            encode_chat(b)
                .expect("renders")
                .iter()
                .map(|s| s.text.as_str())
                .collect()
        };
        let normalized = to_chat_body(responses).expect("reproducible");
        assert_eq!(
            flat(&normalized),
            flat(chat),
            "\nnormalized: {normalized:#}"
        );
    }

    #[test]
    fn plain_string_input_matches_the_equivalent_chat_request() {
        assert_renders_same(
            &json!({"model": "kimi-k3", "input": "hi", "store": false}),
            &json!({"messages": [{"role": "user", "content": "hi"}]}),
        );
    }

    #[test]
    fn message_items_match_the_equivalent_chat_request() {
        assert_renders_same(
            &json!({"input": [
                {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "message", "role": "assistant", "id": "m1", "status": "completed",
                 "content": [{"type": "output_text", "text": "yo"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
            ]}),
            &json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
                {"role": "user", "content": [{"type": "text", "text": "again"}]},
            ]}),
        );
    }

    #[test]
    fn instructions_and_developer_items_coalesce_into_one_system_message() {
        // `instructions` first, then any `developer` item, joined with a blank
        // line and hoisted to the front -- Codex sends both on every turn.
        assert_renders_same(
            &json!({
                "instructions": "A",
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    {"type": "message", "role": "developer", "content": "B"},
                ],
            }),
            &json!({"messages": [
                {"role": "system", "content": "A\n\nB"},
                {"role": "user", "content": "hi"},
            ]}),
        );
    }

    #[test]
    fn an_empty_instructions_string_contributes_no_system_block() {
        // Python `if request.instructions:` -- an empty string is falsy. Emitting
        // an empty system block here would shift every token after it.
        let out = to_chat_body(&json!({"instructions": "", "input": "hi"})).unwrap();
        assert_eq!(out["messages"], json!([{"role": "user", "content": "hi"}]));
    }

    #[test]
    fn function_call_becomes_an_assistant_tool_call() {
        assert_renders_same(
            &json!({"input": [
                {"type": "message", "role": "user", "content": "run it"},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1",
                 "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "a.txt"},
            ]}),
            &json!({"messages": [
                {"role": "user", "content": "run it"},
                {"role": "assistant", "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "a.txt"},
            ]}),
        );
    }

    #[test]
    fn function_call_arguments_survive_verbatim_and_degrade_to_an_empty_object() {
        let args = |v: Value| -> Value {
            let body = json!({"input": [
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": v},
            ]});
            to_chat_body(&body).unwrap()["messages"][0]["tool_calls"][0]["function"]["arguments"]
                .clone()
        };
        // A valid object string is passed through byte-for-byte -- NOT reparsed
        // and redumped, which would normalise whitespace the engine keeps.
        assert_eq!(args(json!("{\"a\": 1}")), json!("{\"a\": 1}"));
        // A dict is dumped compactly, orjson-style.
        assert_eq!(args(json!({"a": 1})), json!("{\"a\":1}"));
        // Truncated, non-object, and missing all collapse to "{}".
        assert_eq!(args(json!("{\"a\":")), json!("{}"));
        assert_eq!(args(json!("[1,2]")), json!("{}"));
        assert_eq!(args(json!("")), json!("{}"));
        assert_eq!(args(Value::Null), json!("{}"));
    }

    #[test]
    fn a_falsy_call_id_falls_through_to_the_item_id() {
        let id = |item: Value| -> Value {
            to_chat_body(&json!({"input": [item]})).unwrap()["messages"][0]["tool_calls"][0]["id"]
                .clone()
        };
        assert_eq!(
            id(json!({"type": "function_call", "id": "fc", "call_id": "", "name": "f"})),
            json!("fc")
        );
        assert_eq!(
            id(json!({"type": "function_call", "id": "fc", "name": "f"})),
            json!("fc")
        );
    }

    #[test]
    fn reasoning_prefers_summary_and_empty_items_drop() {
        let reasoning = |item: Value| {
            to_chat_body(&json!({"input": [item]})).unwrap()["messages"]
                .as_array()
                .unwrap()
                .len()
        };
        // "clients often populate both with the same text" -- summary wins.
        let out = to_chat_body(&json!({"input": [
            {"type": "reasoning", "summary": [{"text": "S"}], "content": [{"text": "C"}]},
        ]}))
        .unwrap();
        assert_eq!(out["messages"][0]["reasoning_content"], json!("S"));
        // Falls back to content only when summary yields nothing.
        let out = to_chat_body(&json!({"input": [
            {"type": "reasoning", "summary": [], "content": [{"text": "C1"}, {"text": "C2"}]},
        ]}))
        .unwrap();
        assert_eq!(out["messages"][0]["reasoning_content"], json!("C1\nC2"));
        // An empty reasoning item injects nothing -- Codex sends these constantly.
        assert_eq!(reasoning(json!({"type": "reasoning", "summary": []})), 0);
    }

    #[test]
    fn consecutive_assistant_items_merge_into_one_turn() {
        // "One Responses-API assistant turn maps to multiple input items" --
        // reasoning + message + two function_calls is a single Codex turn.
        let out = to_chat_body(&json!({"input": [
            {"type": "reasoning", "summary": [{"text": "think"}]},
            {"type": "message", "role": "assistant", "content": "text"},
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call", "call_id": "c2", "name": "g", "arguments": "{}"},
        ]}))
        .unwrap();
        let msgs = out["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 1, "{out:#}");
        assert_eq!(msgs[0]["reasoning_content"], json!("think"));
        assert_eq!(msgs[0]["content"], json!("text"));
        assert_eq!(msgs[0]["tool_calls"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn tools_carry_every_field_pydantic_marks_set() {
        // `Function(name=..., description=..., parameters=..., strict=...)` passes
        // all four explicitly, so `exclude_unset=True` keeps them even when null.
        // `dump_tools` copies whatever keys are present, so dropping the nulls
        // here would render a shorter tool declaration than the engine's.
        let out = to_chat_body(&json!({
            "input": "hi",
            "tools": [{"type": "function", "name": "shell",
                       "parameters": {"type": "object"}}],
        }))
        .unwrap();
        assert_eq!(
            out["tools"],
            json!([{
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": null,
                    "parameters": {"type": "object"},
                    "strict": null,
                },
            }])
        );
    }

    #[test]
    fn builtin_tools_are_dropped_and_an_all_builtin_list_yields_no_tools_key() {
        let out = to_chat_body(&json!({
            "input": "hi",
            "tools": [
                {"type": "web_search"},
                {"type": "function", "name": "f", "description": "d",
                 "parameters": {}, "strict": true},
            ],
        }))
        .unwrap();
        assert_eq!(out["tools"].as_array().unwrap().len(), 1);
        assert_eq!(out["tools"][0]["function"]["name"], json!("f"));

        // `tools=chat_tools or None`
        let out = to_chat_body(&json!({"input": "hi", "tools": [{"type": "web_search"}]})).unwrap();
        assert!(out.get("tools").is_none(), "{out:#}");
    }

    #[test]
    fn reasoning_effort_is_lifted_and_the_reasoning_object_is_not_forwarded() {
        // `encode_chat` refuses any body carrying a `reasoning` object, so leaving
        // it in place would trade one silent 0%-hit path for another.
        let out = to_chat_body(&json!({
            "input": "hi",
            "reasoning": {"effort": "high", "summary": "auto"},
        }))
        .unwrap();
        assert_eq!(out["reasoning_effort"], json!("high"));
        assert!(out.get("reasoning").is_none(), "{out:#}");
        assert!(encode_chat(&out).is_some(), "encoder must not refuse");
        assert_renders_same(
            &json!({"input": "hi", "reasoning": {"effort": "high"}}),
            &json!({"messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "high"}),
        );
    }

    #[test]
    fn unreproducible_requests_are_refused_rather_than_guessed() {
        // History lives in the engine's `msg_store`; hashing this turn alone would
        // produce a prefix that is real but belongs to a different conversation.
        assert!(to_chat_body(&json!({"input": "hi", "previous_response_id": "resp_1"})).is_none());
        // The reference raises ValueError on an unknown item type.
        assert!(to_chat_body(&json!({"input": [{"type": "custom_tool_call"}]})).is_none());
        // Not a Responses body at all.
        assert!(!is_responses_body(&json!({"messages": []})));
        assert!(is_responses_body(&json!({"input": "hi"})));
        // Images expand to vision tokens the router cannot count -- normalisation
        // succeeds, and `render_content` declines downstream, as it does for chat.
        let out = to_chat_body(&json!({"input": [
            {"type": "message", "role": "user", "content": [
                {"type": "input_image", "image_url": "http://x/y.png"},
            ]},
        ]}))
        .unwrap();
        assert!(encode_chat(&out).is_none(), "{out:#}");
    }
}
