///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! Kimi-K3 XTML chat encoding.
//!
//! A port of `encoding_k3.build_chat_segments` plus the request-level
//! preparation `serving_chat` does ahead of it, the two halves that turn an
//! OpenAI chat body into a Kimi-K3 prompt. Kimi-K3 renders chat in imperative
//! Python — `tokenization_kimi.apply_chat_template` calls straight into
//! `encoding_k3` — and ships no Jinja `chat_template` at all, neither embedded
//! in `tokenizer_config.json` nor as a standalone `chat_template.jinja`. So
//! `BlockHasher::apply_chat_template` returns None for every chat request and
//! the prompt hashes to nothing: kv-aware silently degrades to load-only
//! routing at a permanent 0% hit rate. This module is what lets the router
//! speak the format natively, the same way `encoding_dsv4` does for DSv4.
//!
//! Only the encode direction is ported; the router never parses model output.
//!
//! **Segments, not a string.** The reference encodes each `EncodeSegment`
//! separately, with special-token literals live only inside structural markers
//! (`<|open|>`, `<|sep|>`, ...) and inert inside user text — so a user who
//! types `<|end_of_msg|>` gets ordinary BPE tokens, not the control token.
//! Flattening to one string before tokenizing would both specialise that text
//! and merge across segment boundaries, moving the ids. `KimiTokenizer::
//! encode_segments` consumes these the way `_encode_chat_segments` does.
//!
//! A prefix that is even one token off from the engine's fails silently — no
//! error, just a permanent 0% cache hit rate — so anything this cannot
//! reproduce exactly returns `None` and leaves the caller no worse off than
//! the empty render it has today. Currently refused: multimodal content,
//! `continue_final_message`, the `reasoning` object (it rewrites
//! `reasoning_effort` inside the request model), and chat_template_kwargs
//! beyond `thinking` / `thinking_effort`.
//!
//! The tests below are the standing check on the port. They were written
//! against output captured from the reference itself — sglang's `serving_chat`
//! preparation followed by the model's own `build_chat_segments`, cross-checked
//! against `apply_chat_template` — rather than from reading it, because the
//! failure mode is a silent one-token drift. Reproducing that capture needs the
//! weights and a working sglang, so it is not something CI can re-run; when the
//! model's rendering changes, re-capture against the new reference rather than
//! adjusting the expectations here to match this code.

use serde::Serialize;
use serde_json::{Map, Value};

use crate::block_hasher::PyJsonFormatter;

const OPEN_TOKEN: &str = "<|open|>";
const CLOSE_TOKEN: &str = "<|close|>";
const SEP_TOKEN: &str = "<|sep|>";
const END_OF_MSG_TOKEN: &str = "<|end_of_msg|>";
const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";
const IMAGE_PLACEHOLDER_ESCAPED: &str = "<| kimi_image_placeholder |>";

/// `_VALID_THINKING_EFFORTS`. Note "medium" is not one of them, despite being
/// named in the message body the encoder emits.
const VALID_THINKING_EFFORTS: [&str; 3] = ["low", "high", "max"];

/// `apply_chat_template` does `kwargs.setdefault("thinking_effort", "max")`, so
/// an unset effort is not "no effort message" — it is this one.
const DEFAULT_THINKING_EFFORT: &str = "max";

/// One `EncodeSegment`: text plus whether special-token literals inside it are
/// control tokens (structural markers) or ordinary text (anything a client sent).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub allow_special: bool,
}

/// Accumulator mirroring the reference's `_segment` / `_open_tag` / `_attr`
/// helpers one for one.
///
/// The split points are load-bearing, not cosmetic: each segment is BPE'd on
/// its own, so ` role` + `="` + `user` + `"` is a different token stream from
/// the ` role="user"` you would get by concatenating them first.
#[derive(Default)]
struct Segments(Vec<Segment>);

impl Segments {
    /// `_segment`: an empty string produces no segment at all.
    fn push(&mut self, text: &str, allow_special: bool) {
        if text.is_empty() {
            return;
        }
        self.0.push(Segment {
            text: text.to_string(),
            allow_special,
        });
    }

    fn control(&mut self, text: &str) {
        self.push(text, true);
    }

    fn text(&mut self, text: &str) {
        self.push(text, false);
    }

    fn attr(&mut self, key: &str, value: &str) {
        self.text(&format!(" {key}"));
        self.text("=\"");
        self.text(&escape_attr_value(value));
        self.text("\"");
    }

    fn open_tag(&mut self, tag: &str, attrs: &[(&str, String)]) {
        self.control(OPEN_TOKEN);
        self.text(tag);
        for (key, value) in attrs {
            self.attr(key, value);
        }
        self.control(SEP_TOKEN);
    }

    fn close_tag(&mut self, tag: &str) {
        self.control(CLOSE_TOKEN);
        self.text(tag);
        self.control(SEP_TOKEN);
    }

    fn end_of_msg(&mut self) {
        self.control(END_OF_MSG_TOKEN);
    }

    /// `_internal_system_message`: a system turn the encoder synthesises.
    fn internal_system_message(&mut self, message_type: &str, body: &str) {
        self.open_tag(
            "message",
            &[
                ("role", "system".to_string()),
                ("type", message_type.to_string()),
            ],
        );
        self.text(body.trim());
        self.close_tag("message");
        self.end_of_msg();
    }
}

fn escape_attr_value(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

/// Python truthiness — the reference leans on it throughout (`if tools:`,
/// `if tool_calls:`): an empty list/string/object is *absent*.
fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
    }
}

/// A truthy field, or None — the shape every `if x:` in the reference wants.
fn field<'a>(msg: &'a Value, key: &str) -> Option<&'a Value> {
    msg.get(key).filter(|v| truthy(Some(v)))
}

fn role(msg: &Value) -> &str {
    msg.get("role").and_then(Value::as_str).unwrap_or_default()
}

/// `json.dumps(value, ensure_ascii=False)` — Python's default `", "` / `": "`
/// separators, no `\uXXXX` escaping.
fn to_json(v: &Value) -> Option<String> {
    let mut buf = Vec::new();
    v.serialize(&mut serde_json::Serializer::with_formatter(
        &mut buf,
        PyJsonFormatter,
    ))
    .ok()?;
    String::from_utf8(buf).ok()
}

/// `_json_compact`: `separators=(",", ":")`, which is serde_json's default.
fn to_json_compact(v: &Value) -> Option<String> {
    serde_json::to_string(v).ok()
}

/// `_xtml_type`. serde_json has no bool-is-an-int problem, so the ordering the
/// reference needs (bool before number) is automatic.
fn xtml_type(v: &Value) -> &'static str {
    match v {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

/// `_xtml_value`: a string is itself, anything else is JSON.
fn xtml_value(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        other => to_json(other),
    }
}

/// `deep_sort_dict`: recursively order object keys.
///
/// Rust compares `String` by UTF-8 bytes and Python's `sorted` compares by code
/// point; for any well-formed key the two orders agree.
fn deep_sort(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), deep_sort(&m[k]));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(deep_sort).collect()),
        other => other.clone(),
    }
}

/// `neutralize_kimi_k3_image_placeholder`: the engine defuses a placeholder a
/// client typed, so it cannot be mistaken for a real image slot.
fn neutralize(text: &str) -> String {
    text.replace(IMAGE_PLACEHOLDER, IMAGE_PLACEHOLDER_ESCAPED)
}

fn neutralize_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(neutralize(s)),
        Value::Array(a) => Value::Array(a.iter().map(neutralize_value).collect()),
        Value::Object(m) => Value::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), neutralize_value(v)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Emulate `[Tool(**t).model_dump(exclude_unset=True, by_alias=True) for t in tools]`.
///
/// Unlike the DSv4 path — which skips tools entirely because the engine dumps
/// them with defaults the client never sent — `exclude_unset=True` means the
/// dump *is* what the client sent, so it is reproducible from the raw body. The
/// three departures from a straight copy are all mechanical: pydantic ignores
/// keys the model does not declare, `Tool._propagate_defer_loading` copies a
/// tool-level flag down onto the function (and the assignment marks it set, so
/// it survives `exclude_unset`), and `Function._serialize` drops a null
/// `defer_loading`. Neither model declares an alias, so `by_alias` is a no-op.
///
/// Key order does not matter here: `deep_sort_dict` sorts every key downstream.
fn dump_tools(raw: &Value) -> Option<Vec<Value>> {
    let arr = raw.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for t in arr {
        let obj = t.as_object()?;
        let mut tool = Map::new();
        for key in ["type", "function", "defer_loading"] {
            if let Some(v) = obj.get(key) {
                tool.insert(key.to_string(), v.clone());
            }
        }
        let f = obj.get("function")?.as_object()?;
        let mut func = Map::new();
        for key in [
            "description",
            "name",
            "parameters",
            "strict",
            "defer_loading",
        ] {
            if let Some(v) = f.get(key) {
                func.insert(key.to_string(), v.clone());
            }
        }
        // `name: str` is required; a tool without one never reaches the encoder
        // (the request 422s), so refusing rather than rendering matches.
        func.get("name")?.as_str()?;
        if let Some(d) = tool.get("defer_loading").filter(|v| !v.is_null()) {
            if func.get("defer_loading").is_none_or(Value::is_null) {
                func.insert("defer_loading".to_string(), d.clone());
            }
        }
        if func.get("defer_loading").is_some_and(Value::is_null) {
            func.shift_remove("defer_loading");
        }
        tool.insert("function".to_string(), Value::Object(func));
        out.push(Value::Object(tool));
    }
    Some(out)
}

/// `_render_tool_declare`.
fn render_tool_declare(segs: &mut Segments, tools: &Value, dynamic: bool) -> Option<()> {
    let schemas = to_json_compact(tools)?;
    let body = if dynamic {
        format!(
            "## New Tools Available\n\
             The system dynamically extends the toolset via lazy-loading.\n\
             You have access to all existing and extended tools.\n\
             Here are the specs for the extended tools.\n\n\
             ```json\n{schemas}\n```"
        )
    } else {
        format!(
            "# Tools\n\
             Here are the available tools, described in JSONSchema.\n\n\
             ```json\n{schemas}\n```"
        )
    };
    segs.open_tag(
        "message",
        &[
            ("role", "system".to_string()),
            ("type", "tool-declare".to_string()),
        ],
    );
    segs.text(&body);
    segs.close_tag("message");
    segs.end_of_msg();
    Some(())
}

/// `_render_content_segments`, minus the image arm (refused in `prepare`).
///
/// A parts list stays a list: each part is its own segment, so collapsing them
/// into one string first would change the token stream.
fn render_content(segs: &mut Segments, content: Option<&Value>) -> Option<()> {
    match content {
        Some(Value::String(s)) => segs.text(s),
        Some(Value::Array(parts)) => {
            for part in parts {
                segs.text(part.get("text")?.as_str()?);
            }
        }
        None | Some(Value::Null) => {}
        // Anything else has no rendering in the reference either — it raises.
        Some(_) => return None,
    }
    Some(())
}

/// `_render_assistant_segments`.
fn render_assistant(segs: &mut Segments, msg: &Value, thinking: bool) -> Option<()> {
    if thinking {
        // The `<think>` channel is structural: present on every assistant turn
        // in thinking mode, even with nothing to put in it.
        let reasoning = field(msg, "reasoning_content").or_else(|| field(msg, "reasoning"));
        segs.open_tag("think", &[]);
        if let Some(r) = reasoning {
            let r = r.as_str()?;
            if !r.trim().is_empty() {
                segs.text(r);
            }
        }
        segs.close_tag("think");
    }

    segs.open_tag("response", &[]);
    render_content(segs, msg.get("content"))?;
    segs.close_tag("response");

    let Some(tool_calls) = field(msg, "tool_calls").and_then(Value::as_array) else {
        return Some(());
    };
    segs.open_tag("tools", &[]);
    for (i, tool_call) in tool_calls.iter().enumerate() {
        // `fn = tool_call.get("function", tool_call)`: a flattened call carries
        // the name and arguments directly.
        let f = tool_call.get("function").unwrap_or(tool_call);
        let name = f.get("name")?.as_str()?;
        segs.open_tag(
            "call",
            &[("tool", name.to_string()), ("index", (i + 1).to_string())],
        );
        match f.get("_xtml_json_block").filter(|v| !v.is_null()) {
            // Arguments that were not valid JSON are passed through verbatim.
            Some(block) => {
                segs.open_tag("json", &[("type", "object".to_string())]);
                segs.text(block.as_str()?);
                segs.close_tag("json");
            }
            None => {
                if let Some(args) = f.get("arguments").and_then(Value::as_object) {
                    for (key, value) in args {
                        segs.open_tag(
                            "argument",
                            &[("key", key.clone()), ("type", xtml_type(value).to_string())],
                        );
                        segs.text(&xtml_value(value)?);
                        segs.close_tag("argument");
                    }
                }
            }
        }
        segs.close_tag("call");
    }
    segs.close_tag("tools");
    Some(())
}

/// `normalize_tool_arguments`, composed with the `parse_tool_call_arguments`
/// pass `serving_chat` runs first (non-strict: a failure is left for the
/// encoder to deal with).
///
/// Returns `(arguments_object, xtml_json_block)`.
fn normalize_tool_arguments(arguments: Option<&Value>) -> Option<(Value, Option<String>)> {
    match arguments {
        None | Some(Value::Null) => Some((Value::Object(Map::new()), None)),
        Some(Value::Object(m)) => Some((Value::Object(m.clone()), None)),
        Some(Value::String(s)) => {
            if s.trim().is_empty() {
                return Some((Value::Object(Map::new()), None));
            }
            match serde_json::from_str::<Value>(s) {
                Ok(Value::Object(m)) => Some((Value::Object(m), None)),
                // Valid JSON that is not an object raises in the reference.
                Ok(_) => None,
                Err(_) => Some((Value::Object(Map::new()), Some(s.clone()))),
            }
        }
        // Any other type raises a TypeError there.
        Some(_) => None,
    }
}

/// `normalize_message`: sort a message's own tool schemas, and turn each tool
/// call's arguments into an object (or an opaque json block).
fn normalize_message(msg: &Value) -> Option<Value> {
    let Some(obj) = msg.as_object() else {
        return Some(msg.clone());
    };
    let mut out = obj.clone();
    if let Some(tools) = obj.get("tools").filter(|v| !v.is_null()) {
        out.insert("tools".to_string(), deep_sort(tools));
    }
    let Some(tool_calls) = field(msg, "tool_calls").and_then(Value::as_array) else {
        return Some(Value::Object(out));
    };
    let mut normalized = Vec::with_capacity(tool_calls.len());
    for tc in tool_calls {
        let Some(tc_obj) = tc.as_object() else {
            normalized.push(tc.clone());
            continue;
        };
        let mut tc_out = tc_obj.clone();
        // The arguments live on `function` when there is one, on the call itself
        // otherwise; either way the normalised pair is written back in place.
        let nested = tc_obj.get("function").and_then(Value::as_object).cloned();
        let mut target = nested.clone().unwrap_or_else(|| tc_obj.clone());
        let (args, block) = normalize_tool_arguments(target.get("arguments"))?;
        target.insert("arguments".to_string(), args);
        match block {
            Some(b) => {
                target.insert("_xtml_json_block".to_string(), Value::String(b));
            }
            None => {
                target.shift_remove("_xtml_json_block");
            }
        }
        if nested.is_some() {
            tc_out.insert("function".to_string(), Value::Object(target));
        } else {
            tc_out = target;
        }
        normalized.push(Value::Object(tc_out));
    }
    out.insert("tool_calls".to_string(), Value::Array(normalized));
    Some(Value::Object(out))
}

/// `_tool_call_id_index`: `tool_calls[].id` -> (1-based position, function name).
fn tool_call_id_index(tool_calls: Option<&Value>) -> Vec<(String, usize, Option<String>)> {
    let mut index: Vec<(String, usize, Option<String>)> = Vec::new();
    let Some(calls) = tool_calls.and_then(Value::as_array) else {
        return index;
    };
    for (position, tc) in calls.iter().enumerate() {
        if !tc.is_object() {
            continue;
        }
        let Some(id) = tc.get("id").filter(|v| !v.is_null()) else {
            continue;
        };
        // `str(call_id)` — an id is a string in every real request; anything
        // else would stringify differently in Python, so leave it unmatched.
        let Some(key) = id.as_str() else { continue };
        if index.iter().any(|(k, _, _)| k == key) {
            continue;
        }
        let name = tc
            .get("function")
            .and_then(Value::as_object)
            .map_or_else(|| tc.get("name"), |f| f.get("name"))
            .and_then(Value::as_str)
            .map(str::to_string);
        index.push((key.to_string(), position + 1, name));
    }
    index
}

/// `normalize_xtml_tool_result_messages`: re-sort each run of consecutive tool
/// messages into the order of the assistant `tool_calls` they answer, and let
/// the matched call name the tool. A run that cannot be fully matched is left
/// exactly as it came in.
fn normalize_tool_results(messages: &[Value]) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut current: Vec<(String, usize, Option<String>)> = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];
        if role(msg) == "assistant" {
            current = match field(msg, "tool_calls") {
                Some(tc) => tool_call_id_index(Some(tc)),
                None => Vec::new(),
            };
            out.push(msg.clone());
            i += 1;
            continue;
        }
        if role(msg) != "tool" {
            out.push(msg.clone());
            i += 1;
            continue;
        }

        // (position, original offset, message, resolved name)
        let mut run: Vec<(usize, usize, &Value, Option<String>)> = Vec::new();
        let mut unresolved = false;
        let mut offset = 0;
        while i < messages.len() && role(&messages[i]) == "tool" {
            let m = &messages[i];
            let call_id = m
                .get("tool_call_id")
                .or_else(|| m.get("id"))
                .filter(|v| !v.is_null())
                .and_then(Value::as_str);
            match call_id.and_then(|id| current.iter().find(|(k, _, _)| k == id)) {
                Some((_, position, name)) => run.push((*position, offset, m, name.clone())),
                None => {
                    unresolved = true;
                    run.push((0, offset, m, None));
                }
            }
            offset += 1;
            i += 1;
        }

        if unresolved {
            out.extend(run.into_iter().map(|(_, _, m, _)| m.clone()));
            continue;
        }
        run.sort_by_key(|(position, offset, _, _)| (*position, *offset));
        for (_, _, m, name) in run {
            match (name, m.as_object()) {
                (Some(name), Some(obj)) => {
                    let mut resolved = obj.clone();
                    resolved.insert("tool".to_string(), Value::String(name.clone()));
                    if resolved.contains_key("name") {
                        resolved.insert("name".to_string(), Value::String(name));
                    }
                    out.push(Value::Object(resolved));
                }
                _ => out.push(m.clone()),
            }
        }
    }
    out
}

/// Roles that parse as `ChatCompletionMessageGenericParam`; "user" is the one
/// role with its own model. A role in neither 422s before the engine sees it.
const GENERIC_ROLES: [&str; 6] = [
    "system",
    "assistant",
    "tool",
    "function",
    "developer",
    "latest_reminder",
];

/// Project a message through its pydantic model, the way `msg.model_dump()`
/// does before `_prepare_kimi_k3_messages` runs.
///
/// This is not tidying: the projection is what the encoder actually reads.
/// A `name` on a *user* message is dropped (the user model has no such field),
/// an assistant `reasoning` is dropped (only `reasoning_content` is declared),
/// and — because the dump has no `exclude_unset` — every declared field is
/// present even when unset, which is why the reference's
/// `message.get("tool", message.get("name"))` falls through to `name` instead
/// of finding a null `tool`.
fn project_message(msg: &Value) -> Option<Map<String, Value>> {
    let obj = msg.as_object()?;
    let role = obj.get("role")?.as_str()?;
    let mut out = Map::new();
    if role == "user" {
        out.insert("role".to_string(), Value::String(role.to_string()));
        out.insert(
            "content".to_string(),
            obj.get("content").cloned().unwrap_or(Value::Null),
        );
        return Some(out);
    }
    // The generic model's validator lowercases the role before matching.
    let role = role.to_lowercase();
    if !GENERIC_ROLES.contains(&role.as_str()) {
        return None;
    }
    out.insert("role".to_string(), Value::String(role));
    for key in [
        "content",
        "tool_call_id",
        "name",
        "reasoning_content",
        "tool_calls",
        "tools",
    ] {
        out.insert(
            key.to_string(),
            obj.get(key).cloned().unwrap_or(Value::Null),
        );
    }
    // ToolCall / FunctionResponse project too, and `function` is required.
    if let Some(calls) = out.get("tool_calls").and_then(Value::as_array) {
        let mut projected = Vec::with_capacity(calls.len());
        for tc in calls {
            let tc = tc.as_object()?;
            let f = tc.get("function")?.as_object()?;
            projected.push(serde_json::json!({
                "id": tc.get("id").cloned().unwrap_or(Value::Null),
                "index": tc.get("index").cloned().unwrap_or(Value::Null),
                "type": tc.get("type").cloned().unwrap_or_else(|| "function".into()),
                "function": {
                    "name": f.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": f.get("arguments").cloned().unwrap_or(Value::Null),
                },
            }));
        }
        out.insert("tool_calls".to_string(), Value::Array(projected));
    }
    Some(out)
}

/// `_prepare_kimi_k3_messages`, plus the `model_dump` normalisations
/// `serving_chat` applies to every message before it.
fn prepare_messages(body: &Value) -> Option<Vec<Value>> {
    let raw = body.get("messages")?.as_array()?;
    let mut out = Vec::with_capacity(raw.len());
    for msg in raw {
        let mut m = project_message(msg)?;
        let obj = &m.clone();

        match obj.get("content") {
            Some(Value::String(s)) => {
                m.insert("content".to_string(), Value::String(neutralize(s)));
            }
            Some(Value::Array(parts)) => {
                let mut kept = Vec::with_capacity(parts.len());
                for part in parts {
                    let p = part.as_object()?;
                    match p.get("type").and_then(Value::as_str) {
                        Some("text" | "input_text") => {
                            let text = p.get("text")?.as_str()?;
                            kept.push(serde_json::json!({
                                "type": "text",
                                "text": neutralize(text),
                            }));
                        }
                        // Images route through `image_prompts` and expand to
                        // vision tokens the router cannot count. The policy
                        // already drops text overlap for multimodal requests, so
                        // there is nothing to gain by guessing here.
                        _ => return None,
                    }
                }
                m.insert("content".to_string(), Value::Array(kept));
            }
            None | Some(Value::Null) => {
                m.insert("content".to_string(), Value::String(String::new()));
            }
            Some(_) => return None,
        }

        let role = obj.get("role").and_then(Value::as_str).unwrap_or_default();
        if role == "assistant" {
            if let Some(v) = obj.get("reasoning_content").filter(|v| !v.is_null()) {
                m.insert("reasoning_content".to_string(), neutralize_value(v));
            }
            if let Some(calls) = obj.get("tool_calls").and_then(Value::as_array) {
                let mut normalized = Vec::with_capacity(calls.len());
                for tc in calls {
                    let mut tc_out = tc.as_object()?.clone();
                    let mut f = tc_out.get("function")?.as_object()?.clone();
                    // `normalize_assistant_tool_call_arguments(strict=False)`
                    // runs first: only a JSON *object* replaces the string, and
                    // anything else is left for the encoder to deal with.
                    if let Some(s) = f.get("arguments").and_then(Value::as_str) {
                        if let Ok(Value::Object(parsed)) = serde_json::from_str::<Value>(s) {
                            f.insert("arguments".to_string(), Value::Object(parsed));
                        }
                    }
                    // Only the arguments are neutralised, not the name or id.
                    let args = f.get("arguments").map(neutralize_value);
                    if let Some(args) = args {
                        f.insert("arguments".to_string(), args);
                    }
                    tc_out.insert("function".to_string(), Value::Object(f));
                    normalized.push(Value::Object(tc_out));
                }
                m.insert("tool_calls".to_string(), Value::Array(normalized));
            }
        }

        // A system/developer message may carry its own tool schemas, which the
        // encoder renders as a dynamic tool-declare turn.
        if matches!(role, "system" | "developer") {
            if let Some(tools) = obj.get("tools").filter(|v| truthy(Some(v))) {
                m.insert("tools".to_string(), Value::Array(dump_tools(tools)?));
            }
        }
        if role == "developer" {
            m.insert("role".to_string(), Value::String("system".to_string()));
        }

        out.push(Value::Object(m));
    }
    Some(out)
}

/// `extract_response_schema`.
fn extract_response_schema(response_format: &Value) -> Option<&Value> {
    let json_schema = response_format
        .get("json_schema")
        .filter(|v| !v.is_null())?;
    Some(
        json_schema
            .get("schema")
            .or_else(|| json_schema.get("json_schema"))
            .unwrap_or(json_schema),
    )
}

/// Which thinking effort the encoder will render, or `None` to refuse.
fn resolve_thinking_effort(body: &Value, from_kwargs: Option<&str>) -> Option<String> {
    if let Some(e) = from_kwargs {
        // The reference asserts on an unsupported value.
        return VALID_THINKING_EFFORTS.contains(&e).then(|| e.to_string());
    }
    // `serving_chat` forwards `reasoning_effort` only when it is one of the
    // supported values; everything else (including "none") warns and falls back
    // to the tokenizer's own default.
    let effort = body
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .filter(|e| VALID_THINKING_EFFORTS.contains(e))
        .unwrap_or(DEFAULT_THINKING_EFFORT);
    Some(effort.to_string())
}

/// Encode an OpenAI chat body into Kimi-K3 prompt segments.
///
/// Mirrors `serving_chat._encode_messages` for `chat_encoding_spec == "kimi_k3"`
/// with `add_generation_prompt=True`, which is how the engine calls it. Returns
/// `None` if anything about the request cannot be reproduced exactly.
pub fn encode_chat(body: &Value) -> Option<Vec<Segment>> {
    if truthy(body.get("continue_final_message")) {
        return None;
    }

    // `ChatCompletionRequest` normalizes the reasoning knobs into
    // `chat_template_kwargs` before the encoder ever runs, so the body the
    // engine renders is not quite the body the client sent. The `reasoning`
    // object also rewrites `reasoning_effort`, which this port does not model.
    if body.get("reasoning").is_some_and(|v| !v.is_null()) {
        return None;
    }
    // `thinking = effort != "none"` for any effort at all, numeric ones included.
    let derived_thinking = body
        .get("reasoning_effort")
        .filter(|v| !v.is_null())
        .map(|e| e.as_str() != Some("none"));

    // `template_kwargs = dict(request.chat_template_kwargs or {})`, minus the
    // three keys serving_chat pops. Anything else changes the render in ways
    // this port does not model, so refuse rather than ignore it.
    let mut thinking_kwarg: Option<bool> = None;
    let mut effort_kwarg: Option<String> = None;
    if let Some(kwargs) = body.get("chat_template_kwargs").filter(|v| !v.is_null()) {
        for (key, value) in kwargs.as_object()? {
            match key.as_str() {
                "thinking" => thinking_kwarg = Some(value.as_bool()?),
                "thinking_effort" => effort_kwarg = Some(value.as_str()?.to_string()),
                // The alias other model families read. K3 takes `thinking`, and
                // `build_chat_segments` swallows the rest in `**kwargs`.
                "enable_thinking" => {}
                "tokenize" | "return_dict" | "image_prompts" => {}
                _ => return None,
            }
        }
    }
    // `ctk.setdefault("thinking", ...)`: an explicit kwarg outranks the
    // validator's derivation, and with neither the tokenizer defaults to on.
    let thinking = thinking_kwarg.or(derived_thinking).unwrap_or(true);
    let thinking_effort = resolve_thinking_effort(body, effort_kwarg.as_deref())?;

    let messages = prepare_messages(body)?;
    let messages = normalize_tool_results(&messages);
    let messages: Vec<Value> = messages
        .iter()
        .map(normalize_message)
        .collect::<Option<_>>()?;

    let tools = match body.get("tools").filter(|v| truthy(Some(v))) {
        Some(t) => Some(deep_sort(&Value::Array(dump_tools(t)?))),
        None => None,
    };

    let mut segs = Segments::default();

    if let Some(tools) = &tools {
        render_tool_declare(&mut segs, tools, false)?;
    }

    if thinking {
        segs.internal_system_message(
            "thinking-effort",
            &format!(
                "`thinking_effort` guides on how much to think in your thinking \
                 channel (not including the response channel), supported values \
                 include `low`, `medium`, `high`, and `max`.\n\
                 Now the system is invoked with `thinking_effort={thinking_effort}`."
            ),
        );
    }

    // The most recent assistant turn's tool_calls, for naming the tool messages
    // that answer it by position when they carry no name of their own.
    let mut last_tool_calls: Option<&Value> = None;
    let mut tool_index = 0usize;

    for msg in &messages {
        match role(msg) {
            "user" => {
                let mut attrs = vec![("role", "user".to_string())];
                if let Some(name) = field(msg, "name").and_then(Value::as_str) {
                    attrs.push(("name", name.to_string()));
                }
                segs.open_tag("message", &attrs);
                render_content(&mut segs, msg.get("content"))?;
                segs.close_tag("message");
                segs.end_of_msg();
            }
            "system" => match field(msg, "tools") {
                Some(tools) => render_tool_declare(&mut segs, tools, true)?,
                None => {
                    let mut attrs = vec![("role", "system".to_string())];
                    if let Some(name) = field(msg, "name").and_then(Value::as_str) {
                        attrs.push(("name", name.to_string()));
                    }
                    segs.open_tag("message", &attrs);
                    render_content(&mut segs, msg.get("content"))?;
                    segs.close_tag("message");
                    segs.end_of_msg();
                }
            },
            "tool" => {
                tool_index += 1;
                // `message.get("tool", message.get("name"))`: the default is
                // evaluated eagerly, so a present-but-null `tool` wins over `name`.
                let named = match msg.get("tool") {
                    Some(t) => t.as_str().map(str::to_string),
                    None => msg.get("name").and_then(Value::as_str).map(str::to_string),
                };
                let tool_name = match named {
                    Some(n) => n,
                    None => last_tool_calls
                        .and_then(Value::as_array)
                        .filter(|calls| tool_index <= calls.len())
                        .map(|calls| &calls[tool_index - 1])
                        .map(|tc| tc.get("function").unwrap_or(tc))
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)?
                        .to_string(),
                };
                segs.open_tag(
                    "message",
                    &[
                        ("role", "tool".to_string()),
                        ("tool", tool_name),
                        ("index", tool_index.to_string()),
                    ],
                );
                render_content(&mut segs, msg.get("content"))?;
                segs.close_tag("message");
                segs.end_of_msg();
            }
            "assistant" => {
                last_tool_calls = msg.get("tool_calls").filter(|v| !v.is_null());
                tool_index = 0;
                let mut attrs = vec![("role", "assistant".to_string())];
                if let Some(name) = field(msg, "name").and_then(Value::as_str) {
                    attrs.push(("name", name.to_string()));
                }
                segs.open_tag("message", &attrs);
                render_assistant(&mut segs, msg, thinking)?;
                segs.close_tag("message");
                segs.end_of_msg();
            }
            // Every other role is skipped by the reference, silently.
            _ => {}
        }
    }

    // `_effective_tools` is the request's own tools PLUS any declared on a
    // system or developer message -- not, as this said before, tools from an
    // MCP tool server. `normalize_message` has already renamed developer to
    // system and kept those schemas, so the union is readable from here.
    //
    // The distinction decides whether this turn is rendered at all: a client
    // that declares its tools on a system message and asks for
    // `tool_choice: "required"` gets the turn from the engine, and a router
    // keying off the top-level `tools` alone would omit it and hash a prefix
    // the engine never produced -- 0 cache hits for that whole conversation.
    //
    // Only the gate widens. The tool-declare turn at the top still renders
    // `tools`, because `serving_chat.py` passes `request_tools` -- the
    // request's own -- to `apply_chat_template`, not the union.
    let effective_tools = tools.is_some()
        || messages
            .iter()
            .any(|m| role(m) == "system" && field(m, "tools").is_some());
    if effective_tools {
        match body.get("tool_choice").and_then(Value::as_str) {
            Some("required") => segs.internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=required`.\n\
                 You MUST call tools in the next message.",
            ),
            Some("none") => segs.internal_system_message(
                "tool-choice",
                "The system is invoked with `tool_choice=none`.\n\
                 You MUST NOT call any tools in the next message.",
            ),
            _ => {}
        }
    }

    if let Some(rf) = body.get("response_format").filter(|v| !v.is_null()) {
        match rf.get("type").and_then(Value::as_str) {
            Some("json_object") => segs.internal_system_message(
                "response-format",
                "The system is invoked with `response_format=json_object`.\n\
                 Your response must be raw JSON data without markdown code \
                 blocks (```json) or any additional formatting.",
            ),
            Some("json_schema") => {
                let schema = extract_response_schema(rf).map(deep_sort);
                let schema = to_json_compact(schema.as_ref().unwrap_or(&Value::Null))?;
                segs.internal_system_message(
                    "response-format",
                    &format!(
                        "The system is invoked with `response_format=json_schema`.\n\
                         Your response must be raw JSON data without markdown code \
                         blocks (```json) or any additional formatting.\n\
                         The JSON data must match the following schema:\n\
                         ```json\n{schema}\n```"
                    ),
                );
            }
            _ => {}
        }
    }

    // add_generation_prompt: an assistant turn left open on its first channel.
    segs.open_tag("message", &[("role", "assistant".to_string())]);
    segs.open_tag(if thinking { "think" } else { "response" }, &[]);

    Some(segs.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn flat(body: &Value) -> String {
        encode_chat(body)
            .expect("renders")
            .iter()
            .map(|s| s.text.as_str())
            .collect()
    }

    #[test]
    fn plain_chat_renders_the_xtml_frame() {
        let out = flat(&json!({"messages": [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "hi"},
        ]}));
        // The thinking-effort turn is synthesised by the tokenizer default, not
        // by anything the client sent -- leaving it out shifts the whole prefix.
        assert!(
            out.starts_with("<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>"),
            "{out}"
        );
        assert!(out.contains("thinking_effort=max"), "{out}");
        assert!(
            out.ends_with(
                "<|open|>message role=\"user\"<|sep|>hi<|close|>message<|sep|><|end_of_msg|>\
                 <|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
            ),
            "{out}"
        );
    }

    #[test]
    fn structural_markers_are_special_and_client_text_is_not() {
        // The whole reason this returns segments: a marker a user typed must
        // tokenize as ordinary text, or their prompt could forge a message
        // boundary -- and would hash to ids the engine never produced.
        let segs = encode_chat(&json!({"messages": [
            {"role": "user", "content": "say <|end_of_msg|> please"},
        ]}))
        .expect("renders");
        let user_text = segs
            .iter()
            .find(|s| s.text.contains("say "))
            .expect("user text is a segment");
        assert!(!user_text.allow_special);
        assert!(segs
            .iter()
            .any(|s| s.text == END_OF_MSG_TOKEN && s.allow_special));
    }

    #[test]
    fn attribute_values_are_escaped_and_split_the_reference_way() {
        let segs = encode_chat(&json!({"messages": [
            {"role": "system", "name": "a\"&b", "content": "x"},
        ]}))
        .expect("renders");
        let texts: Vec<&str> = segs.iter().map(|s| s.text.as_str()).collect();
        // ` name`, `="`, value, `"` are four separate segments; concatenating
        // them before tokenizing would BPE across the boundaries.
        let at = texts.iter().position(|t| *t == " name").expect("name attr");
        assert_eq!(&texts[at..at + 4], &[" name", "=\"", "a&quot;&amp;b", "\""]);
    }

    #[test]
    fn fields_the_request_models_do_not_declare_never_reach_the_encoder() {
        // The user message model is role+content only, so a `name` there is
        // dropped by pydantic long before the encoder -- rendering it would put
        // an attribute in the prefix that the engine's never has.
        let out = flat(&json!({"messages": [
            {"role": "user", "name": "bob", "content": "hi"},
        ]}));
        assert!(!out.contains("name=\"bob\""), "{out}");
        // Same for an assistant `reasoning`: only `reasoning_content` is declared.
        let out = flat(&json!({"messages": [
            {"role": "assistant", "content": "a", "reasoning": "dropped"},
        ]}));
        assert!(!out.contains("dropped"), "{out}");
        // A role neither model accepts 422s upstream; hashing it would be
        // hashing a request the engine never runs.
        assert!(encode_chat(&json!({"messages": [{"role": "nobody", "content": "x"}]})).is_none());
    }

    #[test]
    fn no_thinking_opens_the_response_channel_instead() {
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"thinking": false},
        }));
        assert!(!out.contains("thinking-effort"), "{out}");
        assert!(out.ends_with("<|open|>response<|sep|>"), "{out}");
    }

    #[test]
    fn reasoning_effort_reaches_the_thinking_effort_turn() {
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "low",
        }));
        assert!(out.contains("thinking_effort=low"), "{out}");
        // "medium" is not a valid effort: the engine warns and uses the default.
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "medium",
        }));
        assert!(out.contains("thinking_effort=max"), "{out}");
    }

    #[test]
    fn tool_calls_render_one_argument_tag_per_key() {
        let out = flat(&json!({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\": \"SF\", \"days\": 3}"},
            }]},
        ]}));
        assert!(
            out.contains("<|open|>call tool=\"get_weather\" index=\"1\"<|sep|>"),
            "{out}"
        );
        assert!(
            out.contains("<|open|>argument key=\"city\" type=\"string\"<|sep|>SF"),
            "{out}"
        );
        assert!(
            out.contains("<|open|>argument key=\"days\" type=\"number\"<|sep|>3"),
            "{out}"
        );
    }

    #[test]
    fn unparseable_tool_arguments_become_an_opaque_json_block() {
        let out = flat(&json!({"messages": [
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "function": {"name": "f", "arguments": "{oops"},
            }]},
        ]}));
        assert!(
            out.contains("<|open|>json type=\"object\"<|sep|>{oops"),
            "{out}"
        );
    }

    #[test]
    fn tool_results_are_reordered_into_call_order() {
        // Delivered 2-then-1; the encoder must index them by the call order.
        let out = flat(&json!({"messages": [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "function": {"name": "first", "arguments": "{}"}},
                {"id": "b", "function": {"name": "second", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
        ]}));
        let first = out
            .find("role=\"tool\" tool=\"first\" index=\"1\"<|sep|>A")
            .expect("first result, renamed and renumbered by its call");
        let second = out
            .find("role=\"tool\" tool=\"second\" index=\"2\"<|sep|>B")
            .expect("second result");
        assert!(first < second, "{out}");
    }

    #[test]
    fn tools_are_declared_sorted_and_compact() {
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {
                "name": "f", "description": "d",
                "parameters": {"type": "object", "properties": {}},
            }}],
        }));
        // deep_sort_dict orders every key, and _json_compact drops the spaces.
        assert!(
            out.contains(
                r#"[{"function":{"description":"d","name":"f","parameters":{"properties":{},"type":"object"}},"type":"function"}]"#
            ),
            "{out}"
        );
    }

    #[test]
    fn tool_choice_and_response_format_add_their_own_turns() {
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "s", "schema": {"type": "object", "additionalProperties": false},
            }},
        }));
        assert!(out.contains("`tool_choice=required`"), "{out}");
        assert!(
            out.contains(r#"{"additionalProperties":false,"type":"object"}"#),
            "{out}"
        );
    }

    /// The tool-choice turn follows `_effective_tools`, not `request.tools`.
    ///
    /// SGLang gates it on the union of the request's tools and any declared on
    /// a system or developer message. A client that puts its schemas on the
    /// system message -- which the encoder renders as a dynamic tool-declare
    /// turn either way -- still gets the turn from the engine, so a router that
    /// omitted it hashed a prefix the engine never produced and drove the whole
    /// conversation to zero cache hits.
    #[test]
    fn tool_choice_follows_tools_declared_on_a_system_message() {
        let sys_tools = json!([{"type": "function", "function": {"name": "f"}}]);
        for role in ["system", "developer"] {
            let out = flat(&json!({
                "messages": [
                    {"role": role, "content": "S", "tools": sys_tools},
                    {"role": "user", "content": "hi"},
                ],
                "tool_choice": "required",
            }));
            assert!(
                out.contains("`tool_choice=required`"),
                "{role} message tools must open the gate: {out}"
            );
        }

        // And the gate stays shut where neither source declares any: the turn
        // is about tools that exist, not about the parameter being present.
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required",
        }));
        assert!(!out.contains("`tool_choice=required`"), "{out}");
    }

    #[test]
    fn unrepresentable_requests_are_refused_rather_than_guessed() {
        // Multimodal: images expand to vision tokens the router cannot count.
        assert!(
            encode_chat(&json!({"messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ]}]}))
            .is_none()
        );
        // A chat_template_kwarg this port does not model.
        assert!(encode_chat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "chat_template_kwargs": {"custom_flag": true},
        }))
        .is_none());
        assert!(encode_chat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "continue_final_message": true,
        }))
        .is_none());
        // `reasoning` rewrites reasoning_effort inside the request model.
        assert!(encode_chat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "high"},
        }))
        .is_none());
        // Not a chat body at all.
        assert!(encode_chat(&json!({"prompt": "hi"})).is_none());
    }

    /// The request model derives `thinking` from `reasoning_effort` and injects
    /// it -- plus the `enable_thinking` alias -- before the encoder runs, so the
    /// encoder has to derive the same value from what the client actually sent.
    #[test]
    fn thinking_follows_the_reasoning_effort_the_client_sent() {
        let with_effort = |v: Value| {
            flat(&json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": v}))
        };
        assert!(with_effort(json!("low")).contains("thinking_effort=low"));
        assert!(!with_effort(json!("none")).contains("thinking-effort"));
        // Unsupported efforts still think, just at the tokenizer's default.
        assert!(with_effort(json!("medium")).contains("thinking_effort=max"));

        // An explicit kwarg outranks the derivation; the alias is inert here.
        let out = flat(&json!({
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none",
            "chat_template_kwargs": {"thinking": true, "enable_thinking": false},
        }));
        assert!(out.contains("thinking_effort=max"), "{out}");
    }

    #[test]
    fn a_developer_turn_becomes_a_system_turn() {
        let out = flat(&json!({"messages": [{"role": "developer", "content": "D"}]}));
        assert!(
            out.contains("<|open|>message role=\"system\"<|sep|>D"),
            "{out}"
        );
    }

    #[test]
    fn a_typed_image_placeholder_is_defused() {
        let out = flat(&json!({"messages": [
            {"role": "user", "content": "look <|kimi_image_placeholder|> here"},
        ]}));
        assert!(
            out.contains("look <| kimi_image_placeholder |> here"),
            "{out}"
        );
    }

    #[test]
    fn json_helpers_match_python_separators() {
        let v: Value = serde_json::from_str(r#"{"b":1,"a":[1,2],"u":"中"}"#).unwrap();
        assert_eq!(to_json(&v).unwrap(), r#"{"b": 1, "a": [1, 2], "u": "中"}"#);
        assert_eq!(
            to_json_compact(&v).unwrap(),
            r#"{"b":1,"a":[1,2],"u":"中"}"#
        );
        assert_eq!(
            to_json_compact(&deep_sort(&v)).unwrap(),
            r#"{"a":[1,2],"b":1,"u":"中"}"#
        );
    }
}
