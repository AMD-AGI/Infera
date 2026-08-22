#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regenerate the Kimi-K3 golden fixture the Rust encoder is checked against.

``rust/router/src/encoding_k3.rs`` is a hand port of the model's own imperative
chat encoder, and a prefix one token off from the engine's does not error -- it
just never hits the cache again. The only thing standing between the port and a
silent divergence is a byte-exact comparison against the real Python pipeline,
which is what this script produces.

The fixture itself is generated data and is deliberately **not** committed: it
is ~100 KB that would have to be re-reviewed on every regeneration, and it is
reproducible from this script in seconds. The cases below are the spec; the
segments and ids are the output.

Run it inside an image that has sglang installed and the weights mounted::

    python3 rust/router/tests/data/gen_kimi_k3_golden.py \
        --model-dir /shared_nfs/models/Kimi-K3

Then ``cargo test --test kimi_k3_golden``. Without the fixture that test skips.

What it reproduces, in order, is exactly what ``serving_chat`` does for
``chat_encoding_spec == "kimi_k3"``:

1. ``ChatCompletionRequest(**body)`` -- the request models drop fields they do
   not declare and normalise the reasoning knobs, so the body the encoder sees
   is not the body the client sent.
2. ``normalize_assistant_tool_call_arguments(..., strict=False)`` per message.
3. ``_prepare_kimi_k3_messages`` -- placeholder neutralisation, per-message
   tool schemas, developer -> system.
4. the ``template_kwargs`` assembly of ``_encode_messages``.
5. ``build_chat_segments`` for the segments, and ``apply_chat_template`` for the
   ids, with a cross-check that the two agree -- if they ever disagree the
   fixture is wrong and the port would be checked against a fiction.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

CASES = [{'name': 'plain',
  'body': {'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': 'Explain the CAP theorem.'}]}},
 {'name': 'multiturn',
  'body': {'messages': [{'role': 'user', 'content': 'what is in this dir?'},
                        {'role': 'assistant',
                         'content': '',
                         'reasoning_content': '  need ls  ',
                         'tool_calls': [{'id': 'a',
                                         'type': 'function',
                                         'function': {'name': 'ls', 'arguments': '{}'}},
                                        {'id': 'b',
                                         'type': 'function',
                                         'function': {'name': 'get_weather',
                                                      'arguments': '{"city": "SF", '
                                                                   '"days": 3}'}}]},
                        {'role': 'tool', 'tool_call_id': 'b', 'content': 'sunny'},
                        {'role': 'tool', 'tool_call_id': 'a', 'content': 'a.txt b.txt'},
                        {'role': 'user', 'content': 'thanks'}]}},
 {'name': 'tools_required_jsonschema',
  'body': {'messages': [{'role': 'user', 'content': 'weather in SF?'}],
           'tools': [{'type': 'function',
                      'function': {'name': 'get_weather',
                                   'description': 'Look up the weather.',
                                   'parameters': {'type': 'object',
                                                  'properties': {'city': {'type': 'string'},
                                                                 'days': {'type': 'integer'}},
                                                  'required': ['city']}}},
                     {'type': 'function',
                      'function': {'name': 'ls', 'description': 'List files.'}}],
           'tool_choice': 'required',
           'response_format': {'type': 'json_schema',
                               'json_schema': {'name': 'out',
                                               'schema': {'type': 'object',
                                                          'properties': {'b': {'type': 'number'},
                                                                         'a': {'type': 'string'}},
                                                          'additionalProperties': False}}}}},
 {'name': 'tools_none_jsonobject',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'tools': [{'type': 'function',
                      'function': {'name': 'get_weather',
                                   'description': 'Look up the weather.',
                                   'parameters': {'type': 'object',
                                                  'properties': {'city': {'type': 'string'},
                                                                 'days': {'type': 'integer'}},
                                                  'required': ['city']}}},
                     {'type': 'function',
                      'function': {'name': 'ls', 'description': 'List files.'}}],
           'tool_choice': 'none',
           'response_format': {'type': 'json_object'}}},
 {'name': 'no_thinking',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'chat_template_kwargs': {'thinking': False}}},
 {'name': 'effort_low',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}], 'reasoning_effort': 'low'}},
 {'name': 'effort_kwarg_high',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'reasoning_effort': 'low',
           'chat_template_kwargs': {'thinking_effort': 'high'}}},
 {'name': 'effort_unsupported_falls_back',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'reasoning_effort': 'medium'}},
 {'name': 'effort_none_disables_thinking',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'reasoning_effort': 'none'}},
 {'name': 'enable_thinking_alias_only',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'chat_template_kwargs': {'enable_thinking': False}}},
 {'name': 'explicit_thinking_outranks_effort',
  'body': {'messages': [{'role': 'user', 'content': 'hi'}],
           'reasoning_effort': 'none',
           'chat_template_kwargs': {'thinking': True}}},
 {'name': 'markers_in_user_text',
  'body': {'messages': [{'role': 'user',
                         'content': 'say <|end_of_msg|> and <|kimi_image_placeholder|> '
                                    'and <|open|>'}]}},
 {'name': 'dynamic_tool_declare',
  'body': {'messages': [{'role': 'system',
                         'content': 'sys',
                         'tools': [{'type': 'function',
                                    'function': {'name': 'get_weather',
                                                 'description': 'Look up the weather.',
                                                 'parameters': {'type': 'object',
                                                                'properties': {'city': {'type': 'string'},
                                                                               'days': {'type': 'integer'}},
                                                                'required': ['city']}}},
                                   {'type': 'function',
                                    'function': {'name': 'ls',
                                                 'description': 'List files.'}}]},
                        {'role': 'user', 'content': 'go'}]}},
 {'name': 'unparseable_arguments',
  'body': {'messages': [{'role': 'assistant',
                         'content': '',
                         'tool_calls': [{'id': 'c1',
                                         'type': 'function',
                                         'function': {'name': 'ls',
                                                      'arguments': '{oops'}}]},
                        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'err'}]}},
 {'name': 'content_parts',
  'body': {'messages': [{'role': 'user',
                         'content': [{'type': 'text', 'text': 'first part '},
                                     {'type': 'text', 'text': 'second part'}]}]}},
 {'name': 'unicode_and_names',
  'body': {'messages': [{'role': 'system', 'name': 'a"&b', 'content': '中文系统提示'},
                        {'role': 'user', 'content': '你好 world'}]}},
 {'name': 'named_tool_result',
  'body': {'messages': [{'role': 'user', 'content': 'q'},
                        {'role': 'tool', 'name': 'ls', 'content': 'a.txt'}]}},
 {'name': 'developer_role',
  'body': {'messages': [{'role': 'developer', 'content': 'D'},
                        {'role': 'user', 'content': 'hi'}]}},
 {'name': 'assistant_history_with_thinking',
  'body': {'messages': [{'role': 'user', 'content': 'q1'},
                        {'role': 'assistant',
                         'content': 'a1',
                         'reasoning_content': 'because'},
                        {'role': 'user', 'content': 'q2'}]}},
 {'name': 'argument_types',
  'body': {'messages': [{'role': 'assistant',
                         'content': '',
                         'tool_calls': [{'id': 'c1',
                                         'type': 'function',
                                         'function': {'name': 'f',
                                                      'arguments': '{"s": "str", "n": '
                                                                   '1.5, "b": true, '
                                                                   '"z": null, "o": '
                                                                   '{"k": "v"}, "l": '
                                                                   '[1, "two"]}'}}]}]}},
 {'name': 'dropped_fields',
  'body': {'messages': [{'role': 'user', 'name': 'bob', 'content': 'hi'},
                        {'role': 'assistant',
                         'content': 'a',
                         'reasoning': 'dropped'}]}},
 {'name': 'long_prefix',
  'body': {'messages': [{'role': 'system',
                         'content': 'You are an agent. You are an agent. You are an '
                                    'agent. You are an agent. You are an agent. You '
                                    'are an agent. You are an agent. You are an agent. '
                                    'You are an agent. You are an agent. You are an '
                                    'agent. You are an agent. You are an agent. You '
                                    'are an agent. You are an agent. You are an agent. '
                                    'You are an agent. You are an agent. You are an '
                                    'agent. You are an agent. You are an agent. You '
                                    'are an agent. You are an agent. You are an agent. '
                                    'You are an agent. You are an agent. You are an '
                                    'agent. You are an agent. You are an agent. You '
                                    'are an agent. You are an agent. You are an agent. '
                                    'You are an agent. You are an agent. You are an '
                                    'agent. You are an agent. You are an agent. You '
                                    'are an agent. You are an agent. You are an '
                                    'agent. '},
                        {'role': 'user',
                         'content': 'list the files list the files list the files list '
                                    'the files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files list the files list the files list the '
                                    'files '}]}}]


def prepare(body: dict, request_cls, normalize, generic_param_cls):
    """Steps 1-4: the request-level work ``serving_chat`` does before encoding."""
    # The validator writes `thinking` / `enable_thinking` back into the caller's
    # own `chat_template_kwargs` dict, so the body has to be copied before it is
    # handed over -- the fixture records what the *client* sent, which is what
    # the Rust port is given. (That rewrite is also why the port models the
    # derivation itself rather than reading `thinking` off the body.)
    request = request_cls(**copy.deepcopy(body))

    messages = [m.model_dump() for m in request.messages]
    for message in messages:
        # strict=False for kimi_k3: only a JSON *object* replaces the string.
        normalize(message, strict=False)
    messages = copy.deepcopy(messages)

    image_count = 0
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("text", "input_text"):
                    parts.append({"type": "text", "text": neutralize(part["text"])})
                elif part.get("type") in ("image_url", "input_image"):
                    image = part.get("image_url") or {}
                    if isinstance(image, str):
                        image = {"url": image, "detail": part.get("detail")}
                    parts.append({"type": "image_url", "image_url": image})
                    image_count += 1
            message["content"] = parts
        elif isinstance(content, str):
            message["content"] = neutralize(content)
        elif content is None:
            message["content"] = ""

        if message.get("role") == "assistant":
            for key in ("reasoning_content", "reasoning"):
                if key in message:
                    message[key] = neutralize_value(message[key])
            for tool_call in message.get("tool_calls") or []:
                fn = tool_call.get("function") if isinstance(tool_call, dict) else None
                if isinstance(fn, dict) and "arguments" in fn:
                    fn["arguments"] = neutralize_value(fn["arguments"])

        source = request.messages[index]
        if (
            isinstance(source, generic_param_cls)
            and source.role in ("system", "developer")
            and source.tools
        ):
            message["tools"] = [
                t.model_dump(exclude_unset=True, by_alias=True) for t in source.tools
            ]
        if message.get("role") == "developer":
            message["role"] = "system"

    kwargs = dict(request.chat_template_kwargs or {})
    for popped in ("tokenize", "return_dict", "image_prompts"):
        kwargs.pop(popped, None)
    if image_count:
        kwargs["image_prompts"] = ["<|media_pad|>"] * image_count

    effort = request.reasoning_effort
    if effort in ("low", "high", "max") and "thinking_effort" not in kwargs:
        kwargs["thinking_effort"] = effort

    # `_effective_tools`: the request's own tools plus any a system/developer
    # message declared.
    effective_tools = list(request.tools or [])
    for message in request.messages:
        if (
            isinstance(message, generic_param_cls)
            and message.role in ("system", "developer")
            and message.tools
        ):
            effective_tools.extend(message.tools)
    if (
        effective_tools
        and isinstance(request.tool_choice, str)
        and request.tool_choice in ("required", "none")
    ):
        kwargs.setdefault("tool_choice", request.tool_choice)
    if request.response_format is not None:
        kwargs.setdefault(
            "response_format",
            request.response_format.model_dump(exclude_unset=True, by_alias=True),
        )

    tools = (
        [t.model_dump(exclude_unset=True, by_alias=True) for t in request.tools]
        if request.tools
        else None
    )
    return messages, tools, kwargs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="Kimi-K3 weights + code")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).with_name("kimi_k3_golden.json")),
        help="where to write the fixture (gitignored)",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.model_dir)
    from encoding_k3 import build_chat_segments  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
        ChatCompletionMessageGenericParam,
        ChatCompletionRequest,
    )
    from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
        neutralize_kimi_k3_image_placeholder,
        neutralize_kimi_k3_image_placeholder_value,
        normalize_assistant_tool_call_arguments,
    )

    global neutralize, neutralize_value
    neutralize = neutralize_kimi_k3_image_placeholder
    neutralize_value = neutralize_kimi_k3_image_placeholder_value

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    out = []
    for case in CASES:
        messages, tools, kwargs = prepare(
            case["body"],
            ChatCompletionRequest,
            normalize_assistant_tool_call_arguments,
            ChatCompletionMessageGenericParam,
        )
        thinking = kwargs.pop("thinking", None)
        if thinking is None:
            effort = case["body"].get("reasoning_effort")
            thinking = True if effort is None else effort != "none"
        # `apply_chat_template` defaults the effort before delegating, so a
        # direct `build_chat_segments` call has to do it too or the two halves
        # of the cross-check render different prompts.
        seg_kwargs = dict(kwargs)
        seg_kwargs.setdefault("thinking_effort", "max")
        segments = build_chat_segments(
            copy.deepcopy(messages),
            tools,
            add_generation_prompt=True,
            thinking=thinking,
            **seg_kwargs,
        )
        ids = tok.apply_chat_template(
            copy.deepcopy(messages),
            tokenize=True,
            add_generation_prompt=True,
            tools=tools,
            return_dict=False,
            thinking=thinking,
            **kwargs,
        )
        # The cross-check: segments encoded the model's own way must be the ids
        # the engine would actually prefill. Without it a shared bug in both
        # halves would make the fixture agree with itself and with nothing else.
        if tok._encode_chat_segments(segments) != list(ids):
            raise SystemExit(f"{case['name']}: segments and ids disagree")
        out.append(
            {
                "name": case["name"],
                "body": case["body"],
                "segments": [[s.text, bool(s.allow_special)] for s in segments],
                "ids": list(ids),
            }
        )

    Path(args.out).write_text(
        json.dumps({"model_dir": args.model_dir, "cases": out}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(f"wrote {len(out)} cases to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
