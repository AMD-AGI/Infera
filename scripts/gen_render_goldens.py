#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Record what the *engine* renders for each request body in the parity corpus.

The router re-implements the engine's prompt rendering -- in Rust, on
minijinja, against a context it assembles itself -- so that it can hash a
prefix the engine will recognise. When the two diverge there is no error and no
log line: the block hashes simply never match again and kv-aware routing
degrades to load balancing while every health signal stays green. That is how
GLM-5.3 sat at cache_hits=0 across 5562 routing decisions for 17 hours.

The only defence is a byte-for-byte diff against the real thing. This script is
the "real thing" half: it renders each body in
``rust/router/tests/render_parity/bodies`` with ``transformers`` the way
sglang's ``serving_chat`` does, and writes the result next to it under
``goldens/<model>/``. ``render_parity_matches_the_engine`` in block_hasher.rs
is the other half.

Adding a model is one command; there is no per-model code on either side::

    python3 scripts/gen_render_goldens.py --model-dir /shared_nfs/models/GLM-5.3 --name glm53
    INFERA_TEST_RENDER_PARITY=glm53=/shared_nfs/models/GLM-5.3 cargo test -p infera-router

Regenerate goldens only when the model's own ``chat_template`` changes -- a diff
here is the signal the test exists to produce, so never regenerate to make a
red test pass without reading it first.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "rust" / "router" / "tests" / "render_parity"


def effective_tools(body: dict) -> list[dict] | None:
    """`serving_chat`'s `tools` argument.

    Mirrors serving_chat.py: tools are dumped from the pydantic `Tool` model --
    which materialises defaults the client never sent (`strict`,
    `defer_loading`) and fixes their key order -- `tool_choice == "none"`
    suppresses them, and a named choice narrows the list. All three are
    load-bearing for a template like GLM-5.3's that renders a tool by iterating
    `tool.items()`.
    """
    tools = body.get("tools")
    if not tools or body.get("tool_choice") == "none":
        return None
    from sglang.srt.entrypoints.openai.protocol import Tool

    dumped = [Tool(**t).model_dump() for t in tools]
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        wanted = (choice.get("function") or {}).get("name")
        if wanted:
            dumped = [t for t in dumped if (t.get("function") or {}).get("name") == wanted]
    return dumped


def render(tokenizer, body: dict) -> str:
    """The prompt string `serving_chat` hands to `tokenizer.encode`.

    Every transformation here is *imported* from sglang rather than
    re-implemented. That is the point: a golden produced by a second hand-port
    would agree with the router's first hand-port for exactly the reasons both
    are wrong. Only the control flow is mirrored, and only because the router
    has to mirror the same flow.

    Mirrors `OpenAIServingChat._apply_jinja_template`'s branch (serving_chat.py,
    the `else` after the custom-encoder specs):

      1. `msg.model_dump()` -- pydantic materialises message defaults.
      2. `normalize_assistant_tool_call_arguments` -- OpenAI sends
         `tool_calls[].function.arguments` as a JSON *string*; templates that
         iterate it as a mapping (GLM-5.3's `_args.items()`) need it parsed,
         and templates with a verbatim string branch (Qwen3's) need it not to
         be. The engine parses; so must we.
      3. `process_content_for_template_format` + `normalize_tool_content`.
      4. `reasoning_effort`, then `chat_template_kwargs` over the top.
      5. `reasoning_config.effort_kwarg` -- some templates take a boolean
         toggle (`low_effort=True`) rather than the effort string, and sglang
         maps `reasoning_effort: "low"` onto it. Derived from the template text
         by sglang's own `detect_reasoning_pattern`.
      6. On any render failure, retry with tools flattened to bare functions
         (Mistral-style templates want no OpenAI wrapper).
    """
    import copy

    from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
    from sglang.srt.entrypoints.openai.serving_chat import (
        normalize_assistant_tool_call_arguments,
    )
    from sglang.srt.parser.conversation import generate_chat_conv  # noqa: F401  (import guard)
    from sglang.srt.parser.template_detection import detect_reasoning_pattern

    request = ChatCompletionRequest(model="parity", **body)
    messages = [m.model_dump() for m in request.messages]
    for message in messages:
        normalize_assistant_tool_call_arguments(message)

    from sglang.srt.parser.jinja_template_utils import (
        detect_jinja_template_content_format,
        process_content_for_template_format,
    )

    template = tokenizer.chat_template
    content_format = detect_jinja_template_content_format(template)
    try:
        from sglang.srt.entrypoints.openai.serving_chat import normalize_tool_content
    except ImportError:  # older sglang

        def normalize_tool_content(_role, content):
            return content

    prepared = []
    for msg in copy.deepcopy(messages):
        if msg.get("content") is None:
            msg["content"] = ""
        processed = process_content_for_template_format(msg, content_format, [], [], [], [])
        processed["content"] = normalize_tool_content(processed["role"], processed.get("content"))
        prepared.append(processed)

    extra: dict = {}
    if request.reasoning_effort is not None:
        extra["reasoning_effort"] = request.reasoning_effort
    if request.chat_template_kwargs:
        extra.update(request.chat_template_kwargs)
    _, reasoning_config = detect_reasoning_pattern(template)
    if reasoning_config is not None and reasoning_config.effort_kwarg is not None:
        if request.reasoning_effort == "low":
            extra.setdefault(reasoning_config.effort_kwarg, True)

    tools = effective_tools(body)
    try:
        return tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools,
            return_dict=False,
            **extra,
        )
    except Exception:
        flat = [t.get("function", t) for t in tools] if tools else None
        return tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            tools=flat,
            return_dict=False,
            **extra,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="local path holding tokenizer_config.json")
    ap.add_argument("--name", required=True, help="short key, e.g. glm53 (names the golden dir)")
    ap.add_argument(
        "--check", action="store_true", help="diff against existing goldens, write none"
    )
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    out_dir = CORPUS / "goldens" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    drift = 0
    bodies = sorted((CORPUS / "bodies").glob("*.json"))
    if not bodies:
        print(f"no bodies in {CORPUS / 'bodies'}", file=sys.stderr)
        return 2
    for body_path in bodies:
        body = json.loads(body_path.read_text())
        try:
            text = render(tokenizer, body)
        except Exception as exc:
            # A template that refuses a body is itself a fact worth recording:
            # the router must refuse it too, rather than render something.
            text = f"__RENDER_ERROR__ {type(exc).__name__}: {exc}"
        golden = out_dir / f"{body_path.stem}.txt"
        if args.check:
            previous = golden.read_text() if golden.exists() else None
            if previous != text:
                drift += 1
                print(f"DRIFT {golden.relative_to(ROOT)}", file=sys.stderr)
            continue
        golden.write_text(text)
        print(f"wrote {golden.relative_to(ROOT)} ({len(text)} chars)")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
