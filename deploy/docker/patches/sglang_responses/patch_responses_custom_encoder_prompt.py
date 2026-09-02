#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Make `/v1/responses` pick the same prompt representation `/v1/chat/completions`
already picks, so models with a custom chat encoder stop 400ing.

WHAT: on a Kimi-K3 server -- aggregated or PD, it makes no difference -- every
`POST /v1/responses` returns HTTP 400

    {"error":{"message":"texts cannot be empty and tokenizer must be initialized",
              "type":"invalid_request_error","param":null,"code":400}}

in ~5 ms, while the same prompt through `POST /v1/chat/completions` answers
normally. The error text names the tokenizer, which is a red herring: the
tokenizer is fine, the prompt handed to it is the empty string.

WHY IT HAPPENS:

  1. `OpenAIServingResponses` subclasses `OpenAIServingChat` and reuses its
     `_process_messages`. For a model with a custom chat encoder --
     `chat_encoding_spec` of "kimi_k3" or "inkling" -- that method takes the
     `prompt_ids is not None` branch (serving_chat.py, the
     `if self.chat_encoding_spec in ("inkling", "kimi_k3")` arm): the encoder
     hands back PRE-RENDERED token ids, and the local `prompt` string is never
     assigned, so `MessageProcessingResult.prompt` keeps its `prompt = ""`
     initial value. That is by design -- the ids are the authoritative
     rendering, the text is not.

  2. `OpenAIServingChat.create_chat_completion` knows this. Its prompt_kwargs
     ladder routes `kimi_k3` (and a non-empty-ids `inkling`) to
     `{"input_ids": ...}` BEFORE the generic `is_multimodal` arm that would
     take `{"text": processed_messages.prompt}`.

  3. `OpenAIServingResponses._make_request` does not. It branches on
     `is_multimodal` alone and takes `processed_messages.prompt`. Kimi-K3 is
     multimodal, so `engine_prompts` becomes `[""]`, `create_responses` turns a
     `str` engine prompt into `GenerateReqInput(text="")`, and
     `managers/tokenizer_manager.py` raises on `if not texts or ...` -- `""` is
     falsy. The `ValueError` is caught by the blanket `except ValueError` in
     `create_responses` and returned as a 400 with no traceback, which is why
     nothing in the engine log explains it.

FIX: replicate the chat endpoint's selection, verbatim in behaviour, in
`_make_request`. Multimodal models WITHOUT a custom encoder keep the text path
they have today; only the encoders that leave `prompt` empty are rerouted to
the ids they actually produced.

NOT A PD BUG, and separate from patch_responses_pd_bootstrap.py in
`sglang_disagg/`. That one plumbs `bootstrap_host/port/room` through so a
disaggregated pair accepts the request at all; this one fixes the prompt the
request carries. Both are needed on a Kimi-K3 PD deployment, and the order they
apply in does not matter -- disjoint anchors, different functions. On an
aggregated Kimi-K3 server only this one applies.

WHY IT MATTERS HERE: the Codex CLI/SDK speaks the Responses API by default
(`model_providers.<id>.wire_api = "responses"`), so Hyperloom's
inference-optimizer -- and any other Codex-driven agent -- never gets a single
successful turn against a Kimi-K3 endpoint without this.

UPSTREAM: worth filing. It reads as an oversight: the custom-encoder arms were
added to `create_chat_completion`'s ladder and `_make_request` was not revisited,
so the two paths silently disagree about which field of the shared
`MessageProcessingResult` is authoritative. DROP THIS SCRIPT once base sglang
routes both through one helper; it then reports "already present" and no-ops.

VERIFIED: anchor present exactly once in the sglang v0.5.17 tree shipped in the
mi35x engine image (`/sgl-workspace/sglang`). Runtime verification:
`curl $EP/v1/responses -d '{"model":"kimi-k3","input":"1+1=?","store":false}'`
returns 200 with a non-empty `output[]` instead of the 400 above, and prompt
token accounting matches the equivalent chat request.

Self-locating and idempotent. Single edit, all-or-nothing: a missing or
no-longer-unique anchor writes NOTHING and fails (exit 1), because an image
whose Responses endpoint 400s on every request should not ship quietly.
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[responses-custom-encoder-prompt]"

_OLD = """        if is_multimodal:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]
"""

_NEW = """        # Mirror OpenAIServingChat.create_chat_completion's prompt_kwargs
        # ladder. A custom chat encoder ("kimi_k3", "inkling") returns
        # pre-rendered token ids and leaves MessageProcessingResult.prompt at
        # its "" initial value, so taking .prompt here would build
        # GenerateReqInput(text="") and tokenizer_manager would raise
        # "texts cannot be empty and tokenizer must be initialized" -- a 400 on
        # every Responses request. Marker: _infera_responses_custom_encoder_ids.
        _infera_responses_custom_encoder_ids = False
        if is_multimodal:
            _spec = getattr(self, "chat_encoding_spec", None)
            if _spec == "kimi_k3":
                _infera_responses_custom_encoder_ids = True
            elif (
                _spec == "inkling"
                and isinstance(processed_messages.prompt_ids, list)
                and processed_messages.prompt_ids
            ):
                _infera_responses_custom_encoder_ids = True

        if is_multimodal and not _infera_responses_custom_encoder_ids:
            request_prompts = [processed_messages.prompt]
            engine_prompts = [processed_messages.prompt]
        else:
            request_prompts = [processed_messages.prompt_ids]
            engine_prompts = [processed_messages.prompt_ids]
"""

_REL = "entrypoints/openai/serving_responses.py"


def _srt_dir():
    spec = importlib.util.find_spec("sglang")
    if not spec or not spec.origin:
        return None
    d = Path(spec.origin).parent / "srt"
    return d if d.is_dir() else None


def main():
    srt = _srt_dir()
    if srt is None:
        print(f"{_TAG} sglang not importable — skipping")
        return 0

    f = srt / _REL
    if not f.is_file():
        print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
        return 1

    src = f.read_text()
    if "_infera_responses_custom_encoder_ids" in src:
        print(f"{_TAG} already present — skipping")
        return 0

    found = src.count(_OLD)
    if found != 1:
        where = "absent" if found == 0 else f"{found}x ambiguous"
        print(f"{_TAG} anchor {where} in {_REL}: {_OLD.splitlines()[0]!r}")
        print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
        return 1

    f.write_text(src.replace(_OLD, _NEW, 1))
    print(f"{_TAG} patched {f}")
    print(f"{_TAG} /v1/responses now uses prompt_ids for custom chat encoders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
