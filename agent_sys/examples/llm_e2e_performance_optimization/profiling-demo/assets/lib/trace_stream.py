#!/usr/bin/env python3
"""Read the `traceEvents` array of a chrome trace one event at a time.

`json.load` on these files is what this replaces. A `with_stack` capture of this
deployment is ~8x the size of the same window without it, because SGLang then
records the whole `python_function` chain; the reference kit measured 122 MB
against 14 MB per rank. Decompressed and materialised as Python objects that is
several gigabytes for one rank, and `manifest.py` reads eight of them in a row.
Streaming keeps peak memory at one event plus the read buffer.

Imports nothing from `agent_sys`, nothing from Magpie and nothing from
Hyperloom: this is package data run as a subprocess on the compute node, and the
only contract it depends on is the chrome-trace layout.

**Partial recovery is deliberate.** torch writes the trace from the profiler
callback after `stop` has already returned, so a capture whose flush was cut
short leaves a file with a valid prefix and no closing bracket. Events before
the cut are real measurements. This yields them and records the truncation in
`errors`, so a caller can tell "the window held nothing" from "the file was
clipped" -- `json.load` reports both as the same exception and loses the data.
"""

from __future__ import annotations

import codecs
import gzip
import json
from pathlib import Path
from typing import IO, Iterator

#: Read/decode chunk, and the threshold at which consumed input is discarded.
#: 8 MiB is Hyperloom's `_bypass_trace_reader.stream_events` default; matching it
#: keeps the two readers' behaviour on a truncated file comparable.
BUFSIZE = 8 << 20

_KEY = '"traceEvents"'
_DECODER = json.JSONDecoder()


def open_trace(path: Path | str) -> IO[bytes]:
    """Open a trace for reading, decompressing `.gz` transparently.

    The caller closes it. SGLang writes `*.trace.json.gz`; the uncompressed form
    is accepted so a hand-extracted file can be analysed without repacking it.
    """
    path = Path(path)
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def stream_events(
    fileobj: IO[bytes],
    bufsize: int = BUFSIZE,
    *,
    errors: list[str] | None = None,
) -> Iterator[dict]:
    """Yield each object inside `traceEvents`, in file order.

    Args:
        fileobj: A binary file, possibly the decompressing one `open_trace`
            returns.
        bufsize: Read chunk in bytes, and the point at which consumed input is
            dropped from the buffer.
        errors: Optional list, appended to with one line per structural fault.
            An empty list after a full pass means the array closed cleanly.

    Yields:
        Trace-event dicts. Non-dict array elements are skipped and recorded --
        the format permits them and nothing here can use one.
    """
    def record(message: str) -> None:
        if errors is not None:
            errors.append(message)

    chunk = max(1, int(bufsize))
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    buf = ""
    pos = 0
    eof = False

    def refill() -> bool:
        """Append one chunk of decoded text. False once nothing more can arrive."""
        nonlocal buf, eof
        if eof:
            return False
        try:
            raw = fileobj.read(chunk)
        except (EOFError, OSError, gzip.BadGzipFile) as exc:
            # A gzip member truncated mid-stream. Everything decoded so far is
            # still valid; report and stop rather than discard it.
            record(f"read failed: {type(exc).__name__}: {exc}")
            eof = True
            raw = b""
        if not raw:
            eof = True
            tail = decoder.decode(b"", final=True)
            buf += tail
            return bool(tail)
        buf += decoder.decode(raw, final=False)
        return True

    def compact() -> None:
        """Drop consumed input once it exceeds one chunk."""
        nonlocal buf, pos
        if pos > chunk:
            buf = buf[pos:]
            pos = 0

    def skip_space() -> str:
        """The next non-whitespace character, or '' at end of input."""
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                return buf[pos]
            if not refill():
                return ""

    # ---- locate the array ---------------------------------------------------
    # `traceEvents` is not always first: SGLang emits `schemaVersion` and
    # `distributedInfo` around it depending on version, so the key is searched
    # for rather than assumed at a position.
    while True:
        found = buf.find(_KEY, pos)
        if found >= 0:
            pos = found + len(_KEY)
            break
        # Keep the last len(_KEY)-1 characters: the key may straddle two chunks.
        keep = len(_KEY) - 1
        if len(buf) > keep:
            buf = buf[-keep:]
            pos = 0
        if not refill():
            record(f"{_KEY} not found; this is not a chrome trace")
            return

    if skip_space() != ":":
        record(f"{_KEY} is not followed by ':'")
        return
    pos += 1
    if skip_space() != "[":
        record(f"{_KEY} is not an array")
        return
    pos += 1

    # ---- walk it ------------------------------------------------------------
    while True:
        char = skip_space()
        if char == "":
            record("input ended inside traceEvents; events before the cut are complete")
            return
        if char == "]":
            return
        if char == ",":
            pos += 1
            continue

        while True:
            try:
                value, end = _DECODER.raw_decode(buf, pos)
            except ValueError:
                # Either the object straddles the buffer end, or it is
                # malformed. More input distinguishes the two; at EOF it cannot.
                if refill():
                    continue
                record("traceEvents holds an element that does not parse; stopping here")
                return
            break

        pos = end
        compact()
        if isinstance(value, dict):
            yield value
        else:
            record(f"skipped a non-object element of type {type(value).__name__}")


def count_categories(path: Path | str, *, errors: list[str] | None = None) -> dict:
    """One streaming pass: per-category event counts, kernel time and time span.

    Returned keys are what `manifest.py` publishes and `check_trace_coverage`
    reads. `gpu_memcpy` and `gpu_memset` are counted separately from `kernel`
    rather than folded into it: they are GPU work and not compute, and pooling
    them would let a capture of pure data movement clear a kernel-count floor.
    """
    counts: dict[str, int] = {}
    events = 0
    kernel_us = 0.0
    low: float | None = None
    high: float | None = None

    with open_trace(path) as handle:
        for event in stream_events(handle, errors=errors):
            events += 1
            category = event.get("cat")
            if isinstance(category, str):
                counts[category] = counts.get(category, 0) + 1
                if category == "kernel":
                    duration = event.get("dur") or 0
                    if isinstance(duration, (int, float)):
                        kernel_us += duration
            stamp = event.get("ts")
            if isinstance(stamp, (int, float)):
                low = stamp if low is None or stamp < low else low
                high = stamp if high is None or stamp > high else high

    return {
        "events": events,
        "categories": counts,
        "gpu_kernels": counts.get("kernel", 0),
        "gpu_kernel_us": round(kernel_us, 1),
        # The signal that a capture asked for `with_stack`. Zero here with a
        # non-zero kernel count is a trace no launcher can be resolved from.
        "python_functions": counts.get("python_function", 0),
        "span_s": round((high - low) / 1e6, 3) if low is not None and high is not None else 0.0,
    }
