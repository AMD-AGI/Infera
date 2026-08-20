///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! DeepSeek-V4 chat encoding.
//!
//! A port of the *encode* half of `encoding/encoding_dsv4.py`, the reference
//! implementation shipped inside the DeepSeek-V4 model repository (MIT,
//! Copyright (c) 2023 DeepSeek) and imported by sglang as
//! `sglang.srt.entrypoints.openai.encoding_dsv4`. DeepSeek-V4 ships no Jinja
//! `chat_template` at all, so the router cannot render its chat prompts the way
//! it renders every other model's — it has to speak this format natively.
//!
//! Only the encoder is ported: the router never parses model output, and the
//! `reasoning_effort` prefix is left out because nothing plumbs that field to
//! the router (it is unreachable here, not approximated).
//!
//! A prefix that is even one token off from the engine's fails silently — no
//! error, just a permanent 0% cache hit rate — so every helper below returns
//! `None` rather than guess, and the unit tests are byte-exact against the
//! golden input/output pairs shipped alongside the reference implementation.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::{json, Map, Value};

use crate::block_hasher::PyJsonFormatter;

/// The router renders chat prompts in `"chat"` mode: `"thinking"` drifts one
/// token against the engine when the last assistant turn carries a tool_call.
/// Mirrors `DSV4_THINKING_MODE` on the Python side.
pub const DSV4_THINKING_MODE: &str = "chat";

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const DSML_TOKEN: &str = "｜DSML｜";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
const TOOL_CALLS_BLOCK_NAME: &str = "tool_calls";

const RESPONSE_FORMAT_TEMPLATE: &str =
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}";

const TOOLS_TEMPLATE: &str = r#"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"#;

/// Special tokens for DeepSeek's internal classification tasks.
fn task_sp_token(task: &str) -> Option<&'static str> {
    match task {
        "action" => Some("<｜action｜>"),
        "query" => Some("<｜query｜>"),
        "authority" => Some("<｜authority｜>"),
        "domain" => Some("<｜domain｜>"),
        "title" => Some("<｜title｜>"),
        "read_url" => Some("<｜read_url｜>"),
        _ => None,
    }
}

/// Python truthiness, which the reference implementation leans on throughout
/// (`if tools:`, `if tool_calls:`): an empty list/string/object is *absent*.
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

fn str_or_empty(v: Option<&Value>) -> &str {
    v.and_then(Value::as_str).unwrap_or("")
}

fn role(msg: &Value) -> &str {
    str_or_empty(msg.get("role"))
}

/// A truthy field, or None — the shape every `if x:` in the reference wants.
fn field<'a>(msg: &'a Value, key: &str) -> Option<&'a Value> {
    msg.get(key).filter(|v| truthy(Some(v)))
}

/// `json.dumps(value, ensure_ascii=False)`: Python's separators (`", "`, `": "`)
/// and no escaping of non-ASCII. Key order is the document's — serde_json is
/// built here with `preserve_order`, so it matches Python's insertion order.
fn to_json(v: &Value) -> Option<String> {
    let mut buf = Vec::new();
    v.serialize(&mut serde_json::Serializer::with_formatter(
        &mut buf,
        PyJsonFormatter,
    ))
    .ok()?;
    String::from_utf8(buf).ok()
}

fn find_last_user_index(messages: &[Value]) -> isize {
    for (idx, msg) in messages.iter().enumerate().rev() {
        if matches!(role(msg), "user" | "developer") {
            return idx as isize;
        }
    }
    -1
}

/// Render the tool schemas that go into a system/developer message.
fn render_tools(tools: &[Value]) -> Option<String> {
    // tools_from_openai_format: each entry is rendered as its `function` object.
    let mut schemas = Vec::with_capacity(tools.len());
    for t in tools {
        schemas.push(to_json(t.get("function")?)?);
    }
    // Substitute the token placeholders before `{tool_schemas}`: `str.format`
    // fills every field in one pass, so a schema that happened to contain a
    // placeholder must not itself be expanded.
    Some(
        TOOLS_TEMPLATE
            .replace("{dsml_token}", DSML_TOKEN)
            .replace("{thinking_start_token}", THINKING_START_TOKEN)
            .replace("{thinking_end_token}", THINKING_END_TOKEN)
            .replace("{tool_schemas}", &schemas.join("\n")),
    )
}

/// Encode one tool call's arguments (an OpenAI JSON *string*) as DSML parameters.
fn encode_arguments_to_dsml(tool_call: &Value) -> Option<String> {
    let raw = tool_call.get("arguments")?;
    let args: Map<String, Value> = match raw.as_str().map(serde_json::from_str::<Value>) {
        Some(Ok(Value::Object(m))) => m,
        // A parse that yields a non-object has no `.items()` in Python — it
        // raises there, so refuse to render rather than invent a shape.
        Some(Ok(_)) => return None,
        // Not a string, or not valid JSON: the reference wraps the raw value in
        // a single `arguments` parameter.
        _ => {
            let mut m = Map::new();
            m.insert("arguments".to_string(), raw.clone());
            m
        }
    };
    let mut parts = Vec::with_capacity(args.len());
    for (key, v) in &args {
        let (is_str, value) = match v {
            Value::String(s) => ("true", s.clone()),
            other => ("false", to_json(other)?),
        };
        parts.push(format!(
            "<{DSML_TOKEN}parameter name=\"{key}\" string=\"{is_str}\">{value}</{DSML_TOKEN}parameter>"
        ));
    }
    Some(parts.join("\n"))
}

/// Python's `repr` of a content block's `type`, for the `[Unsupported ...]` text.
fn block_type_repr(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        None | Some(Value::Null) => "None".to_string(),
        Some(other) => other.to_string(),
    }
}

/// Render the message at `index` into its encoded string form.
fn render_message(
    index: usize,
    messages: &[Value],
    thinking_mode: &str,
    drop_thinking: bool,
) -> Option<String> {
    let msg = messages.get(index)?;
    let last_user_idx = find_last_user_index(messages);
    let idx = index as isize;

    let mut prompt = String::new();
    let content = str_or_empty(msg.get("content"));
    let tools = field(msg, "tools").and_then(Value::as_array);
    let response_format = field(msg, "response_format");

    // Both system and developer hang tools / response_format off their content;
    // developer additionally opens with the user marker.
    let push_tail = |prompt: &mut String| -> Option<()> {
        if let Some(t) = tools {
            prompt.push_str("\n\n");
            prompt.push_str(&render_tools(t)?);
        }
        if let Some(rf) = response_format {
            prompt.push_str("\n\n");
            prompt.push_str(&RESPONSE_FORMAT_TEMPLATE.replace("{schema}", &to_json(rf)?));
        }
        Some(())
    };

    match role(msg) {
        "system" => {
            prompt.push_str(content);
            push_tail(&mut prompt)?;
        }
        "developer" => {
            // The reference asserts on an empty developer message.
            if content.is_empty() {
                return None;
            }
            prompt.push_str(USER_SP_TOKEN);
            prompt.push_str(content);
            push_tail(&mut prompt)?;
        }
        "user" => {
            prompt.push_str(USER_SP_TOKEN);
            match field(msg, "content_blocks").and_then(Value::as_array) {
                Some(blocks) => {
                    let mut parts = Vec::with_capacity(blocks.len());
                    for block in blocks {
                        match str_or_empty(block.get("type")) {
                            "text" => parts.push(str_or_empty(block.get("text")).to_string()),
                            "tool_result" => {
                                let inner = match block.get("content") {
                                    Some(Value::Array(items)) => items
                                        .iter()
                                        .map(|b| {
                                            if str_or_empty(b.get("type")) == "text" {
                                                str_or_empty(b.get("text")).to_string()
                                            } else {
                                                format!(
                                                    "[Unsupported {}]",
                                                    block_type_repr(b.get("type"))
                                                )
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .join("\n\n"),
                                    other => str_or_empty(other).to_string(),
                                };
                                parts.push(format!("<tool_result>{inner}</tool_result>"));
                            }
                            _ => parts.push(format!(
                                "[Unsupported {}]",
                                block_type_repr(block.get("type"))
                            )),
                        }
                    }
                    prompt.push_str(&parts.join("\n\n"));
                }
                None => prompt.push_str(content),
            }
        }
        "latest_reminder" => {
            prompt.push_str(LATEST_REMINDER_SP_TOKEN);
            prompt.push_str(content);
        }
        // merge_tool_messages folds these into user turns; reaching one here
        // means the preprocessing was skipped.
        "tool" => return None,
        "assistant" => {
            let mut tc_content = String::new();
            if let Some(tool_calls) = field(msg, "tool_calls").and_then(Value::as_array) {
                let mut rendered = Vec::with_capacity(tool_calls.len());
                for tc in tool_calls {
                    // tool_calls_from_openai_format
                    let f = tc.get("function")?;
                    let name = f.get("name")?.as_str()?;
                    let arguments = encode_arguments_to_dsml(f)?;
                    rendered.push(format!(
                        "<{DSML_TOKEN}invoke name=\"{name}\">\n{arguments}\n</{DSML_TOKEN}invoke>"
                    ));
                }
                tc_content = format!(
                    "\n\n<{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>\n{}\n</{DSML_TOKEN}{TOOL_CALLS_BLOCK_NAME}>",
                    rendered.join("\n")
                );
            }
            // A preceding message with a task makes this a task output, which
            // never carries thinking.
            let prev_has_task = index >= 1
                && messages[index - 1]
                    .get("task")
                    .is_some_and(|t| !t.is_null());
            if thinking_mode == "thinking"
                && !prev_has_task
                && (!drop_thinking || idx > last_user_idx)
            {
                prompt.push_str(str_or_empty(msg.get("reasoning_content")));
                prompt.push_str(THINKING_END_TOKEN);
            }
            prompt.push_str(content);
            prompt.push_str(&tc_content);
            if !truthy(msg.get("wo_eos")) {
                prompt.push_str(EOS_TOKEN);
            }
        }
        _ => return None,
    }

    // Transition tokens, appended only when what follows is a generation point.
    if let Some(next) = messages.get(index + 1) {
        if !matches!(role(next), "assistant" | "latest_reminder") {
            return Some(prompt);
        }
    }

    match msg.get("task").filter(|t| !t.is_null()) {
        Some(task) => {
            let task = task.as_str()?;
            let sp_token = task_sp_token(task)?;
            if task == "action" {
                prompt.push_str(ASSISTANT_SP_TOKEN);
                prompt.push_str(if thinking_mode == "thinking" {
                    THINKING_START_TOKEN
                } else {
                    THINKING_END_TOKEN
                });
            }
            prompt.push_str(sp_token);
        }
        None => {
            if matches!(role(msg), "user" | "developer") {
                prompt.push_str(ASSISTANT_SP_TOKEN);
                let open_thinking =
                    thinking_mode == "thinking" && (!drop_thinking || idx >= last_user_idx);
                prompt.push_str(if open_thinking {
                    THINKING_START_TOKEN
                } else {
                    THINKING_END_TOKEN
                });
            }
        }
    }

    Some(prompt)
}

/// DeepSeek-V4 has no standalone `tool` role: tool results become
/// `<tool_result>` blocks inside the user turn that follows them.
fn merge_tool_messages(messages: &[Value]) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());

    // The reference tests `"content_blocks" in merged[-1]`; only this function
    // creates that key, and always as an array.
    fn mergeable_user(m: Option<&Value>, require_no_task: bool) -> bool {
        m.is_some_and(|m| {
            role(m) == "user"
                && m.get("content_blocks").and_then(Value::as_array).is_some()
                && (!require_no_task || m.get("task").is_none_or(Value::is_null))
        })
    }
    fn push_block(merged: &mut [Value], block: Value) {
        if let Some(blocks) = merged
            .last_mut()
            .and_then(|m| m.get_mut("content_blocks"))
            .and_then(Value::as_array_mut)
        {
            blocks.push(block);
        }
    }

    for msg in messages {
        match role(msg) {
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id").cloned().unwrap_or_else(|| json!("")),
                    "content": msg.get("content").cloned().unwrap_or_else(|| json!("")),
                });
                if mergeable_user(merged.last(), false) {
                    push_block(&mut merged, block);
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            "user" => {
                let text = msg.get("content").cloned().unwrap_or_else(|| json!(""));
                let block = json!({"type": "text", "text": text});
                if mergeable_user(merged.last(), true) {
                    push_block(&mut merged, block);
                } else {
                    let mut new_msg = Map::new();
                    new_msg.insert("role".to_string(), json!("user"));
                    new_msg.insert(
                        "content".to_string(),
                        msg.get("content").cloned().unwrap_or_else(|| json!("")),
                    );
                    new_msg.insert("content_blocks".to_string(), json!([block]));
                    // Preserve the extra fields the renderer reads.
                    for key in ["task", "wo_eos", "mask"] {
                        if let Some(v) = msg.get(key) {
                            new_msg.insert(key.to_string(), v.clone());
                        }
                    }
                    merged.push(Value::Object(new_msg));
                }
            }
            _ => merged.push(msg.clone()),
        }
    }

    merged
}

/// Order the `tool_result` blocks of a user turn by the `tool_calls` order of
/// the assistant turn that requested them.
fn sort_tool_results_by_call_order(messages: &mut [Value]) {
    let mut last_order: HashMap<String, usize> = HashMap::new();

    for msg in messages.iter_mut() {
        if role(msg) == "assistant" && truthy(msg.get("tool_calls")) {
            last_order.clear();
            if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
                for (idx, tc) in tool_calls.iter().enumerate() {
                    let id = tc
                        .get("id")
                        .and_then(Value::as_str)
                        .or_else(|| {
                            tc.get("function")
                                .and_then(|f| f.get("id"))
                                .and_then(Value::as_str)
                        })
                        .unwrap_or("");
                    if !id.is_empty() {
                        last_order.insert(id.to_string(), idx);
                    }
                }
            }
        } else if role(msg) == "user" && truthy(msg.get("content_blocks")) {
            if last_order.is_empty() {
                continue;
            }
            let Some(blocks) = msg.get_mut("content_blocks").and_then(Value::as_array_mut) else {
                continue;
            };
            let mut tool_blocks: Vec<Value> = blocks
                .iter()
                .filter(|b| str_or_empty(b.get("type")) == "tool_result")
                .cloned()
                .collect();
            if tool_blocks.len() <= 1 {
                continue;
            }
            // Stable, like Python's `sorted`, and unknown ids sort to 0.
            tool_blocks.sort_by_key(|b| {
                *last_order
                    .get(str_or_empty(b.get("tool_use_id")))
                    .unwrap_or(&0)
            });
            let mut next = tool_blocks.into_iter();
            for block in blocks.iter_mut() {
                if str_or_empty(block.get("type")) == "tool_result" {
                    if let Some(sorted) = next.next() {
                        *block = sorted;
                    }
                }
            }
        }
    }
}

/// Drop reasoning content — and developer turns entirely — from before the last
/// user message.
fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last_user_idx = find_last_user_index(messages);
    const KEEP_ROLES: [&str; 5] = [
        "user",
        "system",
        "tool",
        "latest_reminder",
        "direct_search_results",
    ];

    let mut result = Vec::with_capacity(messages.len());
    for (idx, msg) in messages.iter().enumerate() {
        let r = role(msg);
        if KEEP_ROLES.contains(&r) || idx as isize >= last_user_idx {
            result.push(msg.clone());
        } else if r == "assistant" {
            let mut msg = msg.clone();
            if let Some(obj) = msg.as_object_mut() {
                obj.remove("reasoning_content");
            }
            result.push(msg);
        }
        // developer and other roles before the last user turn are dropped
    }
    result
}

/// Encode a conversation into the DeepSeek-V4 prompt format.
///
/// Mirrors `encode_messages(messages, thinking_mode)` with the reference
/// defaults (`context=[]`, `drop_thinking=True`, `add_default_bos_token=True`),
/// which is how the engine calls it. Returns `None` if anything about the
/// conversation cannot be rendered exactly.
pub fn encode_messages(messages: &[Value], thinking_mode: &str) -> Option<String> {
    if !matches!(thinking_mode, "chat" | "thinking") {
        return None;
    }
    let mut msgs = merge_tool_messages(messages);
    sort_tool_results_by_call_order(&mut msgs);

    // Tools anywhere in the conversation force thinking to be kept.
    let drop_thinking = !msgs.iter().any(|m| truthy(m.get("tools")));
    if thinking_mode == "thinking" && drop_thinking {
        msgs = drop_thinking_messages(&msgs);
    }

    let mut prompt = String::from(BOS_TOKEN);
    for idx in 0..msgs.len() {
        prompt.push_str(&render_message(idx, &msgs, thinking_mode, drop_thinking)?);
    }
    Some(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden input/output pairs, copied verbatim from `encoding/tests/` in the
    // DeepSeek-V4-Pro model repository (MIT). They are the byte-exact oracle
    // for this port: they exercise tools, DSML tool calls, tool-result merging,
    // drop_thinking, developer/latest_reminder roles and the action task.
    macro_rules! fixture {
        ($name:literal) => {
            include_str!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/dsv4/",
                $name
            ))
        };
    }

    fn messages(src: &str) -> Vec<Value> {
        serde_json::from_str(src).expect("fixture parses")
    }

    #[test]
    fn golden_1_thinking_with_tools() {
        // The reference harness hangs the tool list off the system message.
        let td: Value = serde_json::from_str(fixture!("test_input_1.json")).unwrap();
        let mut msgs = td["messages"].as_array().unwrap().clone();
        msgs[0]["tools"] = td["tools"].clone();
        assert_eq!(
            encode_messages(&msgs, "thinking").as_deref(),
            Some(fixture!("test_output_1.txt"))
        );
    }

    #[test]
    fn golden_2_thinking_drops_earlier_reasoning() {
        let msgs = messages(fixture!("test_input_2.json"));
        let out = encode_messages(&msgs, "thinking").expect("renders");
        assert_eq!(out, fixture!("test_output_2.txt"));
        assert!(!out.contains("The user said hello"));
    }

    #[test]
    fn golden_3_interleaved_thinking_and_search() {
        let msgs = messages(fixture!("test_input_3.json"));
        assert_eq!(
            encode_messages(&msgs, "thinking").as_deref(),
            Some(fixture!("test_output_3.txt"))
        );
    }

    #[test]
    fn golden_4_chat_mode_action_task() {
        let msgs = messages(fixture!("test_input_4.json"));
        assert_eq!(
            encode_messages(&msgs, "chat").as_deref(),
            Some(fixture!("test_output_4.txt"))
        );
    }

    #[test]
    fn chat_mode_renders_a_plain_conversation() {
        // The router's own mode: no thinking block, `</think>` opens the turn.
        let msgs = messages(r#"[{"role":"system","content":"S"},{"role":"user","content":"hi"}]"#);
        assert_eq!(
            encode_messages(&msgs, "chat").as_deref(),
            Some("<｜begin▁of▁sentence｜>S<｜User｜>hi<｜Assistant｜></think>")
        );
    }

    #[test]
    fn an_unknown_thinking_mode_is_refused() {
        let msgs = messages(r#"[{"role":"user","content":"hi"}]"#);
        assert!(encode_messages(&msgs, "reasoning").is_none());
    }

    #[test]
    fn json_uses_python_separators_and_keeps_key_order() {
        let v: Value = serde_json::from_str(r#"{"b":1,"a":[1,2],"u":"中"}"#).unwrap();
        assert_eq!(to_json(&v).unwrap(), r#"{"b": 1, "a": [1, 2], "u": "中"}"#);
    }
}
