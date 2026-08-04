###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What each engine calls the metrics we need, in one place.

Every engine exposes the same three facts — requests running, requests queued,
KV cache in use — under a different name, and the names drift between releases.
Anything that reads them (graceful drain, an autoscaler, the fake worker) needs
the same mapping, so it lives here rather than being spelled out at each call
site where one of them would quietly rot.

Provenance, because it is uneven and matters:

* **vLLM** — from its published metrics documentation.
* **SGLang** — second-hand, not verified against a running engine. Treat a
  lookup failure as "unknown", never as "zero"; the difference decides whether a
  drain waits or gives up.
* **ATOM** — unknown. Deliberately absent rather than guessed: a wrong name
  reads as an idle engine, and an idle engine is exactly the answer that makes a
  drain cut live requests.
"""

from __future__ import annotations

import re

from infera.common.worker_pool import EngineType

#: metric key -> per-engine exposition name. A missing engine means "we do not
#: know", which callers must distinguish from "the value is zero".
_NAMES: dict[str, dict[EngineType, str]] = {
    "requests_running": {
        EngineType.VLLM: "vllm:num_requests_running",
        EngineType.SGLANG: "sglang:num_running_reqs",
    },
    "requests_waiting": {
        EngineType.VLLM: "vllm:num_requests_waiting",
        EngineType.SGLANG: "sglang:num_queue_reqs",
    },
    "kv_cache_usage": {
        EngineType.VLLM: "vllm:gpu_cache_usage_perc",
        EngineType.SGLANG: "sglang:token_usage",
    },
}

#: Extra per-engine gauges that also represent unfinished work, counted only
#: when draining. These are the PD handoff queues: a prefill worker can show no
#: running and no queued requests while KV transfers are still outstanding, and
#: killing it there strands the decode workers waiting on that KV -- the failure
#: every PD system in the field documents and none of them prevents.
#: Verified present on SGLang 0.5.15 (`--enable-metrics`).
_DRAIN_EXTRA: dict[EngineType, tuple[str, ...]] = {
    EngineType.SGLANG: (
        "sglang:num_prefill_bootstrap_queue_reqs",
        "sglang:num_prefill_inflight_queue_reqs",
        "sglang:num_decode_prealloc_queue_reqs",
        "sglang:num_decode_transfer_queue_reqs",
    ),
}


def metric_name(key: str, engine: EngineType) -> str | None:
    """Exposition name for ``key`` on ``engine``, or None if not known."""
    return _NAMES[key].get(engine)


def parse_metric(text: str, name: str) -> float | None:
    """Sum every label set of a gauge in Prometheus text exposition.

    Engines label these per rank -- SGLang emits
    ``sglang:num_running_reqs{tp_rank="0",...}`` and one series per rank -- so
    reading only the first match would let a busy rank hide behind an idle one.
    Summing is safe for the question a drain asks, because the sum is zero
    exactly when every rank is zero.

    Returns None when the series is absent, which is not the same as 0.0: a
    caller draining in-flight work must not read "metric missing" as "idle".
    """
    total = 0.0
    found = False
    for m in re.finditer(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE
    ):
        try:
            total += float(m.group(1))
        except ValueError:
            continue
        found = True
    return total if found else None


def inflight_from_metrics(text: str, engine: EngineType) -> float | None:
    """Requests the engine is running plus those it has queued.

    Queued requests count: a request the engine has accepted but not started is
    still work the client is waiting on, and killing the process loses it just
    as surely as one mid-generation. So do the PD handoff queues, where the
    request may be finished locally while its KV is still in transit.

    None means the engine's in-flight count could not be determined.
    """
    total = 0.0
    seen = False
    for key in ("requests_running", "requests_waiting"):
        name = metric_name(key, engine)
        if name is None:
            continue
        value = parse_metric(text, name)
        if value is None:
            continue
        total += value
        seen = True
    if not seen:
        return None
    # Absent PD queues are genuinely zero here rather than unknown: the engine
    # published a metrics page and simply is not running disaggregated.
    for name in _DRAIN_EXTRA.get(engine, ()):
        total += parse_metric(text, name) or 0.0
    return total
