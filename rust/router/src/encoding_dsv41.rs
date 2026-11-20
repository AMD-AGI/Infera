///////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
///////////////////////////////////////////////////////////////////////////////
//! DeepSeek-V4.1 chat encoding, for conversations without tools.
//!
//! V4.1 is not V4 with a new name: the prompt opens with `<｜System｜>` whenever
//! the first message is a system turn or a reasoning-effort prefix is due, the
//! engine renders in thinking mode unless asked not to, and that prefix
//! ("Reasoning Effort: 75 ...") sits before everything else. A router that
//! renders V4.1 with the V4 encoder diverges at token 1 on every request, and
//! kv-aware routing silently never hits.
//!
//! This ports what vLLM's `tokenizer-mode deepseek_v41` does to a chat request:
//! the template kwargs merge of `ChatCompletionRequest.build_chat_params`, the
//! defaults and message normalization of `deepseek_v41.py`, and the plain-chat
//! paths of `deepseek_v41_encoding.encode_messages`. The engine tokenizes the
//! result with `add_special_tokens=False` (the BOS is already in the text).
//!
//! Tools, tool calls and results, images, tasks, response formats and
//! continued final messages are refused (`None`) rather than approximated: a
//! prefix that is one token off misses from block 0 onward without any error,
//! and refusing costs only what load-only routing already costs.

use serde_json::{Map, Value};

const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
const DEFAULT_REASONING_EFFORT: u64 = 75; // "high"

#[derive(Clone, Copy, PartialEq, Eq)]
enum Role {
    System,
    Developer,
    User,
    Assistant,
}

struct Msg {
    role: Role,
    content: Option<String>,
    /// User turns only: consecutive user messages merge into one turn whose
    /// text blocks render joined by a blank line.
    blocks: Vec<String>,
    reasoning: Option<String>,
}

struct Options {
    thinking: bool,
    effort: u64,
    drop_thinking: bool,
    add_generation_prompt: bool,
}

/// Python truthiness of a JSON value.
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

/// `merge_kwargs` value filter: `None` and `"auto"` leave the default in place.
fn is_unset(v: &Value) -> bool {
    v.is_null() || v.as_str() == Some("auto")
}

impl Options {
    /// The template kwargs the engine hands the V4.1 renderer, and its defaults.
    fn from_body(body: &Map<String, Value>) -> Option<Options> {
        let mut kw = match body.get("chat_template_kwargs") {
            None | Some(Value::Null) => Map::new(),
            Some(Value::Object(o)) => o.clone(),
            Some(_) => return None,
        };
        // build_chat_params: the request's own fields (pydantic defaults
        // included) override the client's chat_template_kwargs unless unset; a
        // requested effort, even "auto", also decides enable_thinking when the
        // client did not.
        let user_set_enable = kw.contains_key("enable_thinking");
        for (key, default) in [("add_generation_prompt", true), ("continue_final_message", false)] {
            let v = body.get(key).cloned().unwrap_or(Value::Bool(default));
            if !is_unset(&v) {
                kw.insert(key.to_string(), v);
            }
        }
        if let Some(e) = body.get("reasoning_effort").filter(|v| !v.is_null()) {
            if !user_set_enable {
                kw.insert("enable_thinking".into(), Value::Bool(e.as_str() != Some("none")));
            }
            if !is_unset(e) {
                kw.insert("reasoning_effort".into(), e.clone());
            }
        }
        if truthy(kw.get("continue_final_message")) {
            return None;
        }
        let add_generation_prompt = kw.get("add_generation_prompt").map(|v| truthy(Some(v))).unwrap_or(true);

        // deepseek_v41.py apply_chat_template
        let mut thinking = truthy(kw.get("thinking")) || truthy(kw.get("enable_thinking"));
        if !kw.contains_key("thinking") && !kw.contains_key("enable_thinking") {
            thinking = true;
        }
        let mut effort = kw.get("reasoning_effort").filter(|v| !v.is_null()).cloned();
        if effort.as_ref().and_then(Value::as_str) == Some("none") {
            thinking = false;
            effort = None;
        }
        let effort = match effort {
            None => DEFAULT_REASONING_EFFORT,
            Some(Value::String(s)) => match s.as_str() {
                "low" => 50,
                "high" | "xhigh" => 75,
                "max" => 100,
                _ => return None,
            },
            // `type(effort) is int`: an integral JSON number, not a float or bool.
            Some(Value::Number(n)) => match n.as_u64() {
                Some(e @ 1..=100) => e,
                _ => return None,
            },
            Some(_) => return None,
        };
        let drop_thinking = kw.get("drop_thinking").map(|v| truthy(Some(v))).unwrap_or(true);
        Some(Options { thinking, effort, drop_thinking, add_generation_prompt })
    }
}

/// One message's text, as `_normalize_messages` leaves it: list content joins
/// its text parts with a blank line; any non-text part is refused.
fn normalized_content(v: Option<&Value>) -> Option<Option<String>> {
    let text = match v {
        None | Some(Value::Null) => return Some(None),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut out = Vec::with_capacity(parts.len());
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    // `block.get("text", "")`: absent is empty, null breaks the engine's join.
                    Some("text" | "input_text" | "output_text") => match part.get("text") {
                        None => out.push(String::new()),
                        Some(Value::String(s)) => out.push(s.clone()),
                        Some(_) => return None,
                    },
                    _ => return None,
                }
            }
            out.join("\n\n")
        }
        Some(_) => return None,
    };
    if text.contains(IMAGE_PLACEHOLDER) {
        return None;
    }
    Some(Some(text))
}

/// Normalize, refuse what this port does not model, and merge consecutive
/// user turns (`merge_tool_messages` without tool results).
fn conversation(messages: &[Value]) -> Option<Vec<Msg>> {
    let mut out: Vec<Msg> = Vec::with_capacity(messages.len());
    for m in messages {
        let obj = m.as_object()?;
        for key in ["tool_calls", "tools", "task", "response_format", "wo_eos", "content_blocks", "updated_tools"] {
            if truthy(obj.get(key)) {
                return None;
            }
        }
        let role = match obj.get("role").and_then(Value::as_str)? {
            "system" => Role::System,
            "developer" => Role::Developer,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            _ => return None,
        };
        let content = normalized_content(obj.get("content"))?;
        // `if "reasoning" in message: message["reasoning_content"] = message["reasoning"]`
        let reasoning_v = if obj.contains_key("reasoning") { obj.get("reasoning") } else { obj.get("reasoning_content") };
        let reasoning = match reasoning_v {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) if !s.contains(IMAGE_PLACEHOLDER) => Some(s.clone()),
            Some(_) => return None,
        };
        if role == Role::User {
            // A null user content reaches `"\n\n".join` as None in the engine.
            let text = content?;
            if let Some(prev) = out.last_mut().filter(|p| p.role == Role::User) {
                prev.blocks.push(text);
                continue;
            }
            out.push(Msg { role, content: None, blocks: vec![text], reasoning: None });
            continue;
        }
        out.push(Msg { role, content, blocks: Vec::new(), reasoning });
    }
    Some(out)
}

/// V4.1's last user turn: user/developer, or a system turn past index 0.
fn find_last_user_index(msgs: &[Msg]) -> isize {
    for (idx, m) in msgs.iter().enumerate().rev() {
        if matches!(m.role, Role::User | Role::Developer) || (m.role == Role::System && idx > 0) {
            return idx as isize;
        }
    }
    -1
}

/// `_drop_thinking_messages`: before the last user turn, assistants lose their
/// reasoning and developer turns are dropped.
fn drop_thinking_messages(msgs: Vec<Msg>) -> Vec<Msg> {
    let last_user = find_last_user_index(&msgs);
    msgs.into_iter()
        .enumerate()
        .filter_map(|(idx, mut m)| {
            let past = idx as isize >= last_user;
            match m.role {
                Role::User | Role::System => Some(m),
                _ if past => Some(m),
                Role::Assistant => {
                    m.reasoning = None;
                    Some(m)
                }
                Role::Developer => None,
            }
        })
        .collect()
}

fn render_message(index: usize, msgs: &[Msg], o: &Options) -> Option<String> {
    let msg = &msgs[index];
    let last_user = find_last_user_index(msgs);
    let idx = index as isize;
    let is_last = index + 1 == msgs.len();

    let effort_prompt = if index == 0 && o.thinking {
        format!(
            "Reasoning Effort: {} (range 1-100, the higher the value, the more thorough the reasoning)\n\n",
            o.effort
        )
    } else {
        String::new()
    };
    let mut prompt = String::new();
    if index == 0 && (!effort_prompt.is_empty() || msg.role == Role::System) {
        prompt.push_str(SYSTEM_SP_TOKEN);
    }
    prompt.push_str(&effort_prompt);

    match msg.role {
        Role::System => {
            if index > 0 {
                prompt.push_str(SYSTEM_SP_TOKEN);
            }
            prompt.push_str(msg.content.as_deref().unwrap_or(""));
        }
        Role::Developer => {
            // The reference asserts a non-empty developer message.
            let content = msg.content.as_deref().filter(|c| !c.is_empty())?;
            prompt.push_str(USER_SP_TOKEN);
            prompt.push_str(content);
        }
        Role::User => {
            prompt.push_str(USER_SP_TOKEN);
            prompt.push_str(&msg.blocks.join("\n\n"));
        }
        Role::Assistant => {
            if o.thinking && (!o.drop_thinking || idx > last_user) {
                prompt.push_str(msg.reasoning.as_deref().unwrap_or(""));
                prompt.push_str(THINKING_END_TOKEN);
            }
            prompt.push_str(msg.content.as_deref().unwrap_or(""));
            prompt.push_str(EOS_TOKEN);
        }
    }

    if is_last && !o.add_generation_prompt {
        return Some(prompt);
    }
    if let Some(next) = msgs.get(index + 1) {
        if next.role != Role::Assistant {
            return Some(prompt);
        }
    }
    let opens_turn = matches!(msg.role, Role::User | Role::Developer)
        || (msg.role == Role::System && index > 0)
        || (is_last && msg.role == Role::Assistant);
    if opens_turn {
        prompt.push_str(ASSISTANT_SP_TOKEN);
        let open_thinking = o.thinking && (!o.drop_thinking || idx >= last_user);
        prompt.push_str(if open_thinking { THINKING_START_TOKEN } else { THINKING_END_TOKEN });
    }
    Some(prompt)
}

/// The prompt text the engine builds for a V4.1 chat request, or `None` if
/// this request carries something the port does not reproduce exactly.
pub fn encode_chat(body: &Value) -> Option<String> {
    let obj = body.as_object()?;
    if truthy(obj.get("tools")) {
        return None;
    }
    let opts = Options::from_body(obj)?;
    let mut msgs = conversation(obj.get("messages")?.as_array()?)?;
    if msgs.is_empty() {
        return None;
    }
    if opts.thinking && opts.drop_thinking {
        msgs = drop_thinking_messages(msgs);
    }
    let mut prompt = String::from(BOS_TOKEN);
    for i in 0..msgs.len() {
        prompt.push_str(&render_message(i, &msgs, &opts)?);
    }
    Some(prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Byte-exact against the engine's renderer: each pair is a request body and
    /// the prompt vLLM's `deepseek_v41` tokenizer mode builds for it.
    #[test]
    fn matches_the_engine_on_golden_pairs() {
        let cases: Vec<Value> = serde_json::from_str(include_str!("encoding_dsv41_golden.json")).unwrap();
        assert!(cases.len() >= 20);
        for case in cases {
            let name = case["name"].as_str().unwrap();
            let got = encode_chat(&case["body"]);
            assert_eq!(got.as_deref(), case["prompt"].as_str(), "case {name}");
        }
    }

    #[test]
    fn refuses_what_it_cannot_reproduce() {
        let refused = [
            json!({"messages": [{"role": "user", "content": "q"}],
                   "tools": [{"type": "function", "function": {"name": "f"}}]}),
            json!({"messages": [{"role": "user", "content": "q"},
                                {"role": "assistant", "content": null,
                                 "tool_calls": [{"id": "1", "type": "function",
                                                 "function": {"name": "f", "arguments": "{}"}}]}]}),
            json!({"messages": [{"role": "tool", "content": "r", "tool_call_id": "1"}]}),
            json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}),
            json!({"messages": [{"role": "user", "content": null}]}),
            json!({"messages": [{"role": "user", "content": "q"}], "continue_final_message": true}),
            json!({"messages": [{"role": "user", "content": "q"}], "reasoning_effort": "extreme"}),
            json!({"messages": [{"role": "user", "content": "q"}], "chat_template_kwargs": {"reasoning_effort": 101}}),
            json!({"messages": [{"role": "developer", "content": ""}]}),
            json!({"messages": []}),
        ];
        for body in refused {
            assert_eq!(encode_chat(&body), None, "{body}");
        }
    }
}
