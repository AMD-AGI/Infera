###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What each engine calls the metrics we need, in one place.

Every engine exposes the same three facts — requests running, requests queued,
KV cache in use — under a different name, and the names drift between releases.
Anything that reads them (graceful drain, an autoscaler) needs the same mapping,
so it lives here rather than being spelled out at each call site where one of
them would quietly rot.

Provenance, because it is uneven and matters:

* **vLLM** — verified against a running engine (vLLM 0.1.dev19253, Qwen3-8B on
  MI355X). Note ``kv_cache_usage_perc`` was ``gpu_cache_usage_perc`` in older
  builds; the alias list below covers both.
* **SGLang** — verified against a running engine (SGLang 0.5.15, Qwen3-8B on
  MI355X). Note sglang serves ``/metrics`` only with ``--enable-metrics``; the
  worker entrypoint injects it. Treat a lookup failure as "unknown", never as
  "zero"; the difference decides whether a drain waits or gives up.
* **ATOM** — unknown. Deliberately absent rather than guessed: a wrong name
  reads as an idle engine, and an idle engine is exactly the answer that makes a
  drain cut live requests.
"""

from __future__ import annotations

import logging
import re

from infera.common.worker_pool import EngineType

logger = logging.getLogger(__name__)

#: metric key -> per-engine exposition name(s). A missing engine means "we do
#: not know", which callers must distinguish from "the value is zero". Several
#: names per entry means the engine renamed the series between releases and both
#: spellings are in the wild.
_ALIASES: dict[str, dict[EngineType, tuple[str, ...]]] = {
    "kv_cache_usage": {
        EngineType.VLLM: ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    },
}

_NAMES: dict[str, dict[EngineType, str]] = {
    "requests_running": {
        EngineType.VLLM: "vllm:num_requests_running",
        EngineType.SGLANG: "sglang:num_running_reqs",
    },
    "requests_waiting": {
        EngineType.VLLM: "vllm:num_requests_waiting",
        EngineType.SGLANG: "sglang:num_queue_reqs",
    },
    # vLLM renamed this: older builds expose gpu_cache_usage_perc, current ones
    # kv_cache_usage_perc. Both are listed and callers sum whichever is present,
    # because pinning one silently returns "no KV in use" on the other.
    "kv_cache_usage": {
        EngineType.VLLM: "vllm:kv_cache_usage_perc",
        EngineType.SGLANG: "sglang:token_usage",
    },
    "prefix_cache_hit_rate": {
        EngineType.SGLANG: "sglang:cache_hit_rate",
        # vLLM exposes hits/queries counters; callers derive the ratio in PromQL.
    },
}

#: PD handoff / KV-transfer queue gauges re-exported on the frontend /metrics.
_TRANSFER_QUEUES: dict[EngineType, tuple[tuple[str, str], ...]] = {
    EngineType.SGLANG: (
        ("prefill_bootstrap", "sglang:num_prefill_bootstrap_queue_reqs"),
        ("prefill_inflight", "sglang:num_prefill_inflight_queue_reqs"),
        ("decode_prealloc", "sglang:num_decode_prealloc_queue_reqs"),
        ("decode_transfer", "sglang:num_decode_transfer_queue_reqs"),
    ),
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
    """Primary exposition name for ``key`` on ``engine``, or None if unknown."""
    return _NAMES[key].get(engine)


def metric_names(key: str, engine: EngineType) -> tuple[str, ...]:
    """Every spelling of ``key`` on ``engine``, newest first."""
    alias = _ALIASES.get(key, {}).get(engine)
    if alias:
        return alias
    name = _NAMES[key].get(engine)
    return (name,) if name else ()


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


def transfer_queue_names(engine: EngineType) -> tuple[tuple[str, str], ...]:
    """``(queue_label, exposition_name)`` pairs for PD KV-transfer queues."""
    return _TRANSFER_QUEUES.get(engine, ())


def mean_metric(text: str, name: str) -> float | None:
    """Average label sets of a gauge (for per-rank fractions like cache hit rate)."""
    total = 0.0
    n = 0
    for m in re.finditer(
        rf"^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)\s*$", text, re.MULTILINE
    ):
        try:
            total += float(m.group(1))
        except ValueError:
            continue
        n += 1
    return (total / n) if n else None


# ----------------------------------------------------------------------
# Frontend federation (plan A): re-export selected engine series on /metrics
# ----------------------------------------------------------------------
# Covers the stock "Monitoring vLLM Inference Server" panels. Histograms keep
# their buckets so PromQL quantiles stay valid. Both old and new vLLM spellings
# are listed because the gauge/histogram renames landed mid-flight.

_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count", "_created")

_FEDERATE_FAMILIES: dict[EngineType, frozenset[str]] = {
    EngineType.VLLM: frozenset(
        {
            # E2E / TTFT / TPOT
            "vllm:e2e_request_latency_seconds",
            "vllm:time_to_first_token_seconds",
            "vllm:time_per_output_token_seconds",
            "vllm:inter_token_latency_seconds",
            # Throughput
            "vllm:prompt_tokens_total",
            "vllm:prompt_tokens",
            "vllm:generation_tokens_total",
            "vllm:generation_tokens",
            # Scheduler
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:num_requests_swapped",
            # Cache util (gpu_cache renamed to kv_cache)
            "vllm:gpu_cache_usage_perc",
            "vllm:kv_cache_usage_perc",
            "vllm:cpu_cache_usage_perc",
            # Length heatmaps
            "vllm:request_prompt_tokens",
            "vllm:request_generation_tokens",
            # Finish reason
            "vllm:request_success_total",
            "vllm:request_success",
            # Queue / prefill / decode / max gen
            "vllm:request_queue_time_seconds",
            "vllm:request_prefill_time_seconds",
            "vllm:request_decode_time_seconds",
            "vllm:request_max_num_generation_tokens",
        }
    ),
    EngineType.SGLANG: frozenset(
        {
            "sglang:num_running_reqs",
            "sglang:num_queue_reqs",
            "sglang:token_usage",
            "sglang:cache_hit_rate",
            "sglang:num_prefill_bootstrap_queue_reqs",
            "sglang:num_prefill_inflight_queue_reqs",
            "sglang:num_decode_prealloc_queue_reqs",
            "sglang:num_decode_transfer_queue_reqs",
            # Present on recent SGLang builds with --enable-metrics.
            "sglang:time_to_first_token_seconds",
            "sglang:e2e_request_latency_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total",
            "sglang:num_used_tokens",
        }
    ),
}

_SAMPLE_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(.+)$")


def _metric_family(name: str) -> str:
    """Strip Prometheus histogram/summary suffixes to the family base name."""
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _inject_labels(sample_line: str, extra: dict[str, str]) -> str:
    """Add labels to one Prometheus sample line; existing keys are left alone."""
    m = _SAMPLE_LINE.match(sample_line)
    if not m:
        return sample_line
    name, label_block, rest = m.group(1), m.group(2), m.group(3)
    existing: set[str] = set()
    if label_block:
        for key in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)="', label_block):
            existing.add(key)
    additions = [f'{k}="{_escape_label_value(v)}"' for k, v in extra.items() if k not in existing]
    if not additions:
        return sample_line
    if label_block:
        inner = label_block[1:-1]
        new_block = "{" + (inner + "," if inner else "") + ",".join(additions) + "}"
    else:
        new_block = "{" + ",".join(additions) + "}"
    return f"{name}{new_block} {rest}"


def federate_engine_metrics(
    text: str,
    *,
    worker_id: str,
    engine: str | EngineType,
) -> str:
    """Filter allowlisted engine series and stamp ``worker_id`` / ``engine``.

    Used by the frontend ``/metrics`` handler so a single Prometheus scrape of
    the router sees the engine panels (scheduler, KV cache, queue/prefill
    times, finish reason, …) without scraping every worker.

    Callers that scrape multiple workers must pass the results through
    :func:`merge_federated_exposition` so ``# HELP`` / ``# TYPE`` appear once
    per family (Prometheus rejects duplicate metadata directives).
    """
    if isinstance(engine, EngineType):
        eng = engine
        eng_label = engine.value
    else:
        try:
            eng = EngineType(engine)
        except ValueError:
            return ""
        eng_label = eng.value
    families = _FEDERATE_FAMILIES.get(eng)
    if not families or not text:
        return ""

    extra = {"worker_id": worker_id, "engine": eng_label}
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if line.startswith("#"):
            # `# HELP name ...` / `# TYPE name ...`
            parts = line.split(None, 3)
            if len(parts) < 3:
                continue
            name = parts[2]
            if _metric_family(name) in families:
                out.append(line)
            continue
        m = _SAMPLE_LINE.match(line)
        if not m:
            continue
        if _metric_family(m.group(1)) not in families:
            continue
        out.append(_inject_labels(line, extra))
    if not out:
        return ""
    return "\n".join(out) + "\n"


def merge_federated_exposition(parts: list[str]) -> str:
    """Join per-worker federated text, keeping one HELP/TYPE per family."""
    seen_meta: set[tuple[str, str]] = set()
    out: list[str] = []
    for part in parts:
        if not part:
            continue
        for raw in part.splitlines():
            line = raw.rstrip()
            if not line:
                continue
            if line.startswith("#"):
                pieces = line.split(None, 3)
                if len(pieces) < 3:
                    continue
                kind = pieces[1].upper()
                if kind not in ("HELP", "TYPE"):
                    continue
                key = (kind, _metric_family(pieces[2]))
                if key in seen_meta:
                    continue
                seen_meta.add(key)
                out.append(line)
                continue
            out.append(line)
    return ("\n".join(out) + "\n") if out else ""


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
    missing: list[str] = []
    for key in ("requests_running", "requests_waiting"):
        name = metric_name(key, engine)
        if name is None:
            continue
        value = parse_metric(text, name)
        if value is None:
            missing.append(name)
            continue
        total += value
        seen = True
    if not seen:
        return None
    if missing:
        # Partial readings are still worth acting on -- refusing them outright
        # would mean one renamed series stops the drain waiting for anything at
        # all, which is the worse failure. But it must not pass silently: the
        # absent series contributes zero, so a drain can finish while the work
        # it describes is still outstanding. These names do drift; the vLLM KV
        # gauge was renamed under exactly this module.
        logger.warning(
            "drain: %s not found on the %s metrics page; its work counts as zero, "
            "so in-flight requests may be cut. Check the exposition names.",
            ", ".join(missing),
            engine.value,
        )
    # Absent PD queues are genuinely zero here rather than unknown: the engine
    # published a metrics page and simply is not running disaggregated.
    for name in _DRAIN_EXTRA.get(engine, ()):
        total += parse_metric(text, name) or 0.0
    return total
