#!/usr/bin/env python3
"""Golden (request body -> prompt) pairs for the router's DeepSeek-V4.1 encoder.

Renders each body the way the vLLM engine does: ChatCompletionRequest.build_chat_params (top-level
reasoning_effort / add_generation_prompt merged into chat_template_kwargs), deepseek_v41.py's
apply_chat_template (message normalization, thinking default, effort default) and
deepseek_v41_encoding.encode_messages, loaded from the given encoder file.

    gen_dsv41_golden.py /path/to/deepseek_v41_encoding.py > golden.json
"""

import copy
import importlib.util
import json
import sys

spec = importlib.util.spec_from_file_location("enc", sys.argv[1])
enc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(enc)


def normalize(messages):
    out = []
    for m in messages:
        m = copy.deepcopy(m)
        m["_image_indices"] = []
        if "reasoning" in m:
            m["reasoning_content"] = m["reasoning"]
        c = m.get("content")
        if isinstance(c, list):
            m["content"] = "\n\n".join(b.get("text", "") for b in c)
        out.append(m)
    return out


def kwargs_of(body):
    ctk = dict(body.get("chat_template_kwargs") or {})
    effort = body.get("reasoning_effort")
    extra = {
        "add_generation_prompt": body.get("add_generation_prompt", True),
        "continue_final_message": body.get("continue_final_message", False),
        "documents": None,
        "reasoning_effort": effort,
    }
    if effort is not None and "enable_thinking" not in ctk:
        extra["enable_thinking"] = effort != "none"
    return ctk | {k: v for k, v in extra.items() if v not in (None, "auto")}


def render(body):
    kw = kwargs_of(body)
    conv = normalize(body["messages"])
    thinking = bool(kw.get("thinking") or kw.get("enable_thinking"))
    if "thinking" not in kw and "enable_thinking" not in kw:
        thinking = True
    effort = kw.get("reasoning_effort")
    if effort == "none":
        thinking, effort = False, None
    if effort is None:
        effort = "high"
    return enc.encode_messages(
        conv,
        thinking_mode="thinking" if thinking else "chat",
        drop_thinking=kw.get("drop_thinking", True),
        reasoning_effort=effort,
        add_generation_prompt=kw.get("add_generation_prompt", True),
        continue_final_message=kw.get("continue_final_message", False),
    )


U = lambda t: {"role": "user", "content": t}
A = lambda t, r=None: {"role": "assistant", "content": t, **({"reasoning_content": r} if r is not None else {})}
S = lambda t: {"role": "system", "content": t}
CASES = {
    "single_user_defaults_to_thinking_high": {"messages": [U("Hi there")]},
    "system_then_user": {"messages": [S("You are terse."), U("2+2?")]},
    "multi_turn_drops_earlier_reasoning": {
        "messages": [S("sys"), U("q1"), A("a1", "r1"), U("q2"), A("a2", "r2"), U("q3")]
    },
    "chat_mode_via_kwargs": {"messages": [S("sys"), U("q1"), A("a1"), U("q2")], "chat_template_kwargs": {"thinking": False}},
    "enable_thinking_false": {"messages": [U("q")], "chat_template_kwargs": {"enable_thinking": False}},
    "top_level_effort_max": {"messages": [U("q")], "reasoning_effort": "max"},
    "top_level_effort_none_is_chat": {"messages": [S("s"), U("q")], "reasoning_effort": "none"},
    "kwarg_effort_low": {"messages": [U("q")], "chat_template_kwargs": {"reasoning_effort": "low"}},
    "numeric_effort": {"messages": [U("q")], "chat_template_kwargs": {"reasoning_effort": 33}},
    "top_level_effort_overrides_kwarg": {
        "messages": [U("q")], "reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "max"}
    },
    "consecutive_users_merge": {"messages": [U("part one"), U("part two")]},
    "content_parts_join": {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
    },
    "developer_before_last_user_dropped": {
        "messages": [S("s"), {"role": "developer", "content": "dev note"}, U("q1"), A("a1", "r1"), U("q2")]
    },
    "developer_last": {"messages": [S("s"), U("q1"), A("a1"), {"role": "developer", "content": "dev"}]},
    "mid_conversation_system": {"messages": [U("q1"), A("a1"), S("new rules"), U("q2")]},
    "final_assistant_gets_generation_header": {"messages": [U("q1"), A("a1", "r1")]},
    "no_generation_prompt": {"messages": [U("q1"), A("a1")], "add_generation_prompt": False},
    "keep_thinking_when_drop_thinking_false": {
        "messages": [U("q1"), A("a1", "r1"), U("q2")], "chat_template_kwargs": {"drop_thinking": False}
    },
    "reasoning_field_alias": {"messages": [U("q1"), {"role": "assistant", "content": "a1", "reasoning": "r1"}, U("q2")],
                              "chat_template_kwargs": {"drop_thinking": False}},
    "null_assistant_content": {"messages": [U("q1"), {"role": "assistant", "content": None}, U("q2")]},
    "unicode_content": {"messages": [S("系统提示"), U("你好，世界 ✓")]},
    "top_level_auto_effort_turns_thinking_on": {
        "messages": [U("q")], "reasoning_effort": "auto", "chat_template_kwargs": {"thinking": False}
    },
    "kwarg_add_generation_prompt_is_overridden": {
        "messages": [U("q1"), A("a1")], "chat_template_kwargs": {"add_generation_prompt": False}
    },
    "missing_text_in_part_is_empty": {"messages": [{"role": "user", "content": [{"type": "text"}, {"type": "text", "text": "x"}]}]},
}

print(json.dumps([{"name": k, "body": v, "prompt": render(v)} for k, v in CASES.items()], ensure_ascii=False, indent=1))
