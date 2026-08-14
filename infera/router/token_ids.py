###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Reading the token ids an engine reports alongside the text it streams.

Migration continues a generation on another worker. Handing over the decoded
text works, but re-encoding it is not guaranteed to reproduce the ids the model
actually sampled -- tokenizers are not injective on the way back, so a boundary
can shift and the second worker resumes from a slightly different sequence.
Asking the engine for the ids removes that step entirely.

Both supported engines can report them, and neither agrees on where:

  vLLM      chunk["choices"][i]["token_ids"]   -- the deltas in this chunk
            chunk["prompt_token_ids"]          -- the prompt, first chunk only
  SGLang    chunk["sglext"]["completion_token_ids"] -- one list per choice
            chunk["sglext"]["prompt_token_ids"]

Both shapes are accepted wherever they appear, because the field has moved
between engine releases and a router pinned to one layout would silently stop
being exact. Silently is the problem: every reader here returns None rather
than a guess, and a caller that gets None falls back to carrying text, which is
approximate but never wrong about what was produced.
"""

from __future__ import annotations

from infera.common.worker_pool import EngineType


def deltas_from_chunk(obj: object) -> list[int] | None:
    """The ids generated in this chunk, or None if the engine did not say.

    None is not an error. It is the ordinary state for an engine that was not
    asked for ids, or was asked and does not support it on this endpoint.
    """
    if not isinstance(obj, dict):
        return None
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        ids = _int_list(choices[0].get("token_ids"))
        if ids is not None:
            return ids
    ext = obj.get("sglext")
    if isinstance(ext, dict):
        per_choice = ext.get("completion_token_ids")
        # One list per choice; only the first is ours (n > 1 is rejected
        # upstream for migratable requests).
        if isinstance(per_choice, list) and per_choice:
            return _int_list(per_choice[0])
        return _int_list(ext.get("token_ids"))
    return None


def prompt_from_chunk(obj: object) -> list[int] | None:
    """The prompt ids, which engines attach to the first chunk only.

    This is what makes an exact continuation possible at all: without it the
    router would have to re-encode the prompt, reintroducing on the input side
    exactly the ambiguity the output ids were fetched to avoid.
    """
    if not isinstance(obj, dict):
        return None
    ids = _int_list(obj.get("prompt_token_ids"))
    if ids is not None:
        return ids
    choices = obj.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        ids = _int_list(choices[0].get("prompt_token_ids"))
        if ids is not None:
            return ids
    ext = obj.get("sglext")
    if isinstance(ext, dict):
        return _int_list(ext.get("prompt_token_ids"))
    return None


def strip_token_ids(obj: dict) -> bool:
    """Remove the id fields from a chunk on its way to the client.

    The router asks for these; the caller did not. Leaving them in would put a
    field in the response that the same request does not produce when migration
    is off, which is a difference an operator's config should not make.

    Returns whether anything was removed, so a caller can skip re-serialising a
    chunk that does not need it.
    """
    touched = obj.pop("prompt_token_ids", None) is not None
    touched |= obj.pop("sglext", None) is not None
    choices = obj.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                touched |= choice.pop("token_ids", None) is not None
                touched |= choice.pop("prompt_token_ids", None) is not None
    return touched


def supports_streaming_ids(engine: EngineType | None, path: str) -> bool:
    """Whether asking this engine for ids on this endpoint is safe.

    SGLang rejects the request outright -- not ignores it -- when ids are asked
    for on streaming chat, so sending it anyway would turn every migratable
    chat request into a 400. Its completions endpoint is fine, and vLLM
    supports both.

    ATOM is left out: it is not known to accept the field, and an engine that
    errors on an unknown parameter would fail requests that work today.
    """
    if engine == EngineType.VLLM:
        return True
    if engine == EngineType.SGLANG:
        return not path.endswith("/chat/completions")
    return False


def _int_list(value: object) -> list[int] | None:
    """A list of ids, or None for anything else.

    Rejects rather than filters: a partially-parsed sequence would be a
    plausible-looking prefix of the truth, and continuing from it drops output
    the client already read.
    """
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        # bool is an int in Python and never a token id.
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        out.append(item)
    return out
