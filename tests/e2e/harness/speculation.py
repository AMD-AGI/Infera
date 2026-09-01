###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Did speculative decoding (MTP/EAGLE) actually run?

A row that asks for MTP and silently doesn't get it is indistinguishable from
one that does, from the outside: the replies are identical and every probe still
passes. The draft head can fail to load, the engine can decide the config is
unusable and warn once at startup, or the flag can simply be the wrong spelling
for that engine — all three end in ordinary autoregressive decode, and a green
case that claims MTP.

So after the probes have generated some tokens, read the worker's own counters
and say what happened. Requested-but-inactive fails the case; unreadable
counters are reported and do not, because "this engine exposes no metrics" is
not evidence of anything about the draft head.

The three engines spell the request three different ways
(``--speculative-algorithm EAGLE`` / ``--speculative-config {...}`` /
``--method mtp``) and name their counters differently too, so both sides of this
match on shape rather than on an exact list that would rot.
"""

from __future__ import annotations

import json
import re

import httpx

from .adapter import emit_reporter_line
from .params import EngineParams

__all__ = ["mtp_requested", "report_speculation"]

# A Prometheus sample line: `name{labels} value`, labels optional.
_SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^}]*\})?\s+(?P<value>\S+)$")


def mtp_requested(params: EngineParams) -> bool:
    """True if this case's argv asks for speculative decoding, in any engine's spelling."""
    args = list(params.extra_args)
    for i, tok in enumerate(args):
        flag, _, inline = tok.partition("=")
        value = inline or (args[i + 1] if i + 1 < len(args) else "")
        if flag == "--speculative-algorithm":  # sglang
            return True
        if flag == "--speculative-config":  # vllm
            # `{"method": "..."}`; an empty or null config is not a request.
            try:
                return bool(json.loads(value or "{}").get("method"))
            except ValueError:
                return bool(value)
        if flag == "--method" and value in ("mtp", "eagle3"):  # atom
            return True
    return False


def _spec_counters(metrics_text: str) -> dict[str, float]:
    """Speculative-decoding samples, by metric name.

    Matched on shape — a name mentioning speculation and either drafting or
    acceptance — because the engines disagree: sglang publishes
    ``sglang:spec_accept_length``, vLLM ``vllm:spec_decode_num_draft_tokens_total``
    and ``…_num_accepted_tokens_total``.
    """
    out: dict[str, float] = {}
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE.match(line)
        if not m:
            continue
        name = m.group("name")
        low = name.lower()
        if "spec" not in low or not any(k in low for k in ("draft", "accept")):
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        # TP>1 publishes one sample per rank/label set; sum rather than clobber.
        out[name] = out.get(name, 0.0) + value
    return out


async def report_speculation(
    port: int, params: EngineParams, *, engine: str, host: str = "127.0.0.1"
) -> None:
    """Read the worker's spec-decode counters, report them, fail if MTP is dead.

    ``port`` is the worker's own HTTP port (``WorkerHandle.port``), not the
    router's: these counters are the engine's, and the router does not proxy
    ``/metrics``. ``host`` is likewise the worker's, which is only the local one
    in the mixed tier — under PD the leg that speculates is on another node.
    """
    requested = mtp_requested(params)
    try:
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(f"http://{host}:{port}/metrics")
        counters = _spec_counters(r.text) if r.status_code == 200 else {}
        reachable = r.status_code == 200
    except httpx.HTTPError as e:
        counters, reachable = {}, False
        emit_reporter_line(f"[e2e mtp] metrics unreadable: {type(e).__name__}: {e}")

    if not reachable or not counters:
        detail = "metrics unreachable" if not reachable else "no spec counters exposed"
        emit_reporter_line(f"[e2e mtp] {engine}: requested={requested} — {detail}")
        if requested and engine in {"sglang", "vllm"}:
            raise AssertionError(
                f"{engine} requested speculative decoding, but {detail}; "
                "the test cannot verify that the draft head ran"
            )
        # ATOM currently exposes no compatible Prometheus counters.
        return

    summary = " ".join(f"{k}={v:g}" for k, v in sorted(counters.items()))
    # "Did any drafting happen" is the question; acceptance length can legitimately
    # be low, but a proposer that never proposed means the draft path is not live.
    active = any(v > 0 for k, v in counters.items() if "draft" in k.lower())
    if not active:
        active = any(v > 0 for v in counters.values())

    state = "active" if active else "INACTIVE"
    emit_reporter_line(f"[e2e mtp] {engine}: requested={requested} {state}  {summary}")

    assert not (requested and not active), (
        f"this case asks for speculative decoding but the worker drafted nothing: {summary}. "
        "The draft head did not load, or the flags are not the spelling this engine reads "
        "(sglang: --speculative-algorithm, vllm: --speculative-config, atom: --method mtp)."
    )
