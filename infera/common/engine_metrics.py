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


def metric_name(key: str, engine: EngineType) -> str | None:
    """Exposition name for ``key`` on ``engine``, or None if not known."""
    return _NAMES[key].get(engine)


def parse_metric(text: str, name: str) -> float | None:
    """Read a single unlabelled gauge out of Prometheus text exposition.

    Returns None when the series is absent, which is not the same as 0.0 — a
    caller draining in-flight work must not read "metric missing" as "idle".
    """
    m = re.search(rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def inflight_from_metrics(text: str, engine: EngineType) -> float | None:
    """Requests the engine is running plus those it has queued.

    Queued requests count: a request the engine has accepted but not started is
    still work the client is waiting on, and killing the process loses it just
    as surely as one mid-generation.

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
    return total if seen else None
