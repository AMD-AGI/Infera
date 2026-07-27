"""[[DSV4-TRACE]] runtime attention-backend path tracer (study instrumentation).

Injected into SGLang v0.5.15.post1 to confirm which attention forward paths
DeepSeek-V4 actually hits under eager / cuda_graph / +dpa configs.

Design (see infera.sglang.study/CLAUDE.md):
- one warning line per FIRST-SEEN (entrypoint, layer_id, mode, compress_ratio,
  capture) combo, only for layer_id < LAYER_WATCH (DSv4's early layers differ);
- every call still increments a counter for its combo;
- atexit dumps the full counter table so we see "reached" AND "how often".
All calls are wrapped so instrumentation can never crash the engine.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading

logger = logging.getLogger("sglang.dsv4_trace")

# DSv4 first layers are architecturally different; compress_ratios[:6] =
# [128,128,4,128,4,128] covers the heterogeneous early layers + both compress
# paths. Watch 0..5 by default; override with DSV4_TRACE_LAYERS.
LAYER_WATCH = int(os.environ.get("DSV4_TRACE_LAYERS", "6"))

_counts: dict = {}
_seen: set = set()
_lock = threading.Lock()
_dumped = False


def _capture_mode() -> str:
    """get_is_capture_mode() may be unavailable early; degrade gracefully."""
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        return "cap" if get_is_capture_mode() else "run"
    except Exception:
        return "?"


def _mode_str(forward_batch) -> str:
    try:
        return forward_batch.forward_mode.name
    except Exception:
        try:
            return str(forward_batch.forward_mode)
        except Exception:
            return "?"


def dsv4_trace(
    entrypoint: str,
    *,
    layer_id=None,
    forward_batch=None,
    compress_ratio=None,
    extra: str = "",
) -> None:
    """Record one hit of a forward entrypoint. Never raises."""
    try:
        lid = layer_id if layer_id is not None else -1
        mode = _mode_str(forward_batch) if forward_batch is not None else "-"
        cap = _capture_mode()
        cr = compress_ratio if compress_ratio is not None else "-"
        key = (entrypoint, lid, mode, cr, cap)
        with _lock:
            _counts[key] = _counts.get(key, 0) + 1
            first = key not in _seen
            if first:
                _seen.add(key)
        if first and (lid < LAYER_WATCH):
            logger.warning(
                "[[DSV4-TRACE]] %s | layer=%s mode=%s cr=%s capture=%s%s",
                entrypoint,
                lid,
                mode,
                cr,
                cap,
                (" | " + extra) if extra else "",
            )
    except Exception:
        pass


@atexit.register
def _dump():
    global _dumped
    if _dumped:
        return
    _dumped = True
    try:
        pid = os.getpid()
        lines = ["[[DSV4-TRACE]] ==== count summary (pid=%d) ====" % pid]
        for key in sorted(_counts):
            ep, lid, mode, cr, cap = key
            lines.append(
                "[[DSV4-TRACE]] %6d  %s L=%s mode=%s cr=%s cap=%s"
                % (_counts[key], ep, lid, mode, cr, cap)
            )
        logger.warning("\n".join(lines))
    except Exception:
        pass


# NOTE: intentionally no signal handler — overriding SIGTERM/SIGINT risks
# interfering with SGLang's multi-process shutdown. The live first-seen warning
# lines are the primary evidence; the atexit summary is best-effort.
