#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Let `/v1/responses` carry PD bootstrap params, so the Responses API works
under disaggregation.

WHAT: `POST /v1/responses` is unusable on any worker started with
`--disaggregation-mode prefill|decode`. It returns HTTP 400
"Invalid request: Disaggregated request received without bootstrap room id"
for every request, no matter what the client or the proxy in front of it sends.

WHY IT HAPPENS, in three steps:

  1. `entrypoints/openai/protocol.py` gives `CompletionRequest` and
     `ChatCompletionRequest` a `bootstrap_host` / `bootstrap_port` /
     `bootstrap_room` trio. `ResponsesRequest` has no such fields, and declares
     no `model_config`, so pydantic's default `extra="ignore"` DROPS them
     silently — a PD proxy's annotation never reaches the model object, and
     nothing anywhere logs that it was discarded.
  2. `entrypoints/openai/serving_responses.py` builds its `GenerateReqInput`
     without mentioning bootstrap at all, so `bootstrap_room` is None.
  3. `managers/scheduler.py`, in any disaggregation mode with a non-FAKE
     transfer backend, sees `recv_req.bootstrap_room is None` and calls
     `prepare_abort(..., status_code=HTTPStatus.BAD_REQUEST)`.

FIX: two edits, mirroring `ChatCompletionRequest` verbatim.
  protocol.py           add the three fields to `ResponsesRequest`.
  serving_responses.py  pass them into the `GenerateReqInput` that
                        `create_responses` builds.

Nothing else is needed, because `_make_request` already converts a
`ResponsesRequest` into a `ChatCompletionRequest` and runs it through the same
`_process_messages` → prompt path as the chat endpoint. The generation pipeline
was always shared; only the bootstrap plumbing was missing.

WHY IT MATTERS HERE: the Codex CLI/SDK speaks the Responses API by default
(`model_providers.<id>.wire_api = "responses"`), so every agentic workload
driven by Codex — Hyperloom's inference-optimizer among them — is locked out of
a PD deployment. Infera's Rust router registers `/v1/responses` and threads the
bootstrap trio through both legs exactly as it does for chat; this patch is the
engine half of that path.

THE HARMONY BUILTIN-TOOL LOOP IS DELIBERATELY NOT PATCHED. `serving_responses`
has a second `GenerateReqInput`, in `_generate_with_builtin_tools`, which issues
a FOLLOW-UP generation after a server-side tool call. Under PD that turn would
need its own bootstrap room and its own prefill leg, and no proxy has arranged
either — copying the first turn's room would hand the decode worker a room the
prefill side already consumed, and it would block on KVPoll until the ~300 s
timeout. Left unpatched, that path fails fast with the same 400 instead. Harmony
builtin tools under PD is a separate feature, not a field-plumbing fix.
Non-harmony models never reach it: `SimpleContext.need_builtin_tool_call()`
returns False, so the loop breaks after the first turn.

STATELESS CALLS ONLY, independent of this patch. `store` / `previous_response_id`
and the `GET /v1/responses/{id}` and `.../cancel` routes read a per-process
`response_store` dict, so they only resolve when the follow-up lands on the same
worker. Behind any multi-worker router, clients must send `store: false`.

UPSTREAM: not submitted. The gap looks like an oversight rather than a decision —
the two older request models grew the fields and `ResponsesRequest` was added
later without them. Worth filing; DROP THIS SCRIPT once base sglang carries the
fields, at which point it reports "already present" and no-ops.

VERIFIED: the missing fields and forwarding were re-checked against both
supported bases on 2026-09-02. v0.5.16 ends the GenerateReqInput call at
`background=request.background`; v0.5.18 adds `require_reasoning` and changes
background streaming semantics. The forwarding edit accepts exactly those two
source shapes. Runtime verification is the end-to-end one: a Responses request
through the router against a 1P1D pair returns 200 instead of 400, and prefill
and decode log the SAME bootstrap_room — which is what proves the KV actually
moved over Mooncake rather than the decode leg quietly recomputing the prompt.

Self-locating and idempotent. Both edits or none: `ResponsesRequest` accepting a
field that `create_responses` then drops is exactly the silent-discard bug this
patch exists to remove, so an anchor that is missing or no longer unique writes
NOTHING and fails (exit 1).
"""

import importlib.util
import sys
from pathlib import Path

_TAG = "[responses-pd-bootstrap]"

# rel path -> [(anchor, anchor + our edit), ...]. Each anchor must occur exactly
# once; the replacement doubles as the already-applied marker.
_EDITS: dict[str, list[tuple[str, str]]] = {
    "entrypoints/openai/protocol.py": [
        # Appended after the last of the core OpenAI fields, ahead of the
        # "Extra SGLang parameters" block, so the file keeps its own ordering
        # convention. Types are copied verbatim from ChatCompletionRequest.
        (
            """    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None
""",
            """    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    # PD disaggregation. Same trio, same types, as CompletionRequest and
    # ChatCompletionRequest. Without these, pydantic's default extra="ignore"
    # drops a proxy's annotation and the scheduler 400s the request.
    bootstrap_host: Optional[Union[List[str], str]] = None
    bootstrap_port: Optional[Union[List[Optional[int]], int]] = None
    bootstrap_room: Optional[Union[List[int], int]] = None
""",
        ),
    ],
}

_SERVING_REL = "entrypoints/openai/serving_responses.py"
_SERVING_MARKER = "bootstrap_host=request.bootstrap_host,"
_SERVING_VARIANTS: tuple[tuple[str, str], ...] = (
    # v0.5.16: the call ends directly after background.
    (
        """                        background=request.background,
                    )
""",
        """                        background=request.background,
                        # PD disaggregation: forward the proxy's bootstrap trio.
                        # None on an aggregated server, which is what
                        # GenerateReqInput already defaults to.
                        bootstrap_host=request.bootstrap_host,
                        bootstrap_port=request.bootstrap_port,
                        bootstrap_room=request.bootstrap_room,
                    )
""",
    ),
    # v0.5.18: background streaming semantics and require_reasoning were added.
    (
        """                        # background+stream streams on this connection, so don't detach.
                        background=request.background and not request.stream,
                        require_reasoning=require_reasoning,
                    )
""",
        """                        # background+stream streams on this connection, so don't detach.
                        background=request.background and not request.stream,
                        require_reasoning=require_reasoning,
                        # PD disaggregation: forward the proxy's bootstrap trio.
                        # None on an aggregated server, which is what
                        # GenerateReqInput already defaults to.
                        bootstrap_host=request.bootstrap_host,
                        bootstrap_port=request.bootstrap_port,
                        bootstrap_room=request.bootstrap_room,
                    )
""",
    ),
)


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

    # Resolve and check EVERY anchor before writing anything: a tree with the
    # field added but not forwarded is the silent-discard bug again.
    planned: list[tuple[Path, str]] = []
    for rel, edits in _EDITS.items():
        f = srt / rel
        if not f.is_file():
            print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
            return 1
        src = out = f.read_text()
        for old, new in edits:
            if new in out:
                continue  # this edit is already in the tree
            found = out.count(old)
            if found != 1:
                where = "absent" if found == 0 else f"{found}x ambiguous"
                print(f"{_TAG} anchor {where} in {rel}: {old.splitlines()[0]!r}")
                print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
                return 1
            out = out.replace(old, new, 1)
        if out != src:
            planned.append((f, out))

    # The GenerateReqInput tail differs between the two supported releases.
    # Require exactly one known pristine shape, or the explicit applied marker;
    # never guess across a new source layout.
    f = srt / _SERVING_REL
    if not f.is_file():
        print(f"{_TAG} {f} is missing — sglang layout changed, re-anchor the patch")
        return 1
    src = out = f.read_text()
    if _SERVING_MARKER not in out:
        matches = [(old, new) for old, new in _SERVING_VARIANTS if out.count(old) == 1]
        if len(matches) != 1:
            counts = ", ".join(
                f"variant-{i}={out.count(old)}" for i, (old, _) in enumerate(_SERVING_VARIANTS, 1)
            )
            print(
                f"{_TAG} expected exactly one supported GenerateReqInput tail "
                f"in {_SERVING_REL}; found {len(matches)} ({counts})"
            )
            print(f"{_TAG} sglang drifted — re-cut the patch, nothing written")
            return 1
        old, new = matches[0]
        out = out.replace(old, new, 1)
    if out != src:
        planned.append((f, out))

    if not planned:
        print(f"{_TAG} already present — skipping")
        return 0
    for f, out in planned:
        f.write_text(out)
        print(f"{_TAG} patched {f}")
    print(f"{_TAG} /v1/responses now carries bootstrap_host/port/room under PD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
