###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Measured kernel speedups, applied to the kernel report and to the step cost.

A tuner or kernel agent that sped up some of the kernels a replay reported
(:mod:`kernel_shapes`) hands the result back as a speedups file::

    {"schema": "inferasim.kernel_speedups", "version": 1,
     "entries": [
       {"op": "moe_gate_up", "speedup": 1.3},
       {"op": "attn_qkv", "dims": {"n": 1280, "k": 2880}, "ranges": {"m": [1, 512]},
        "baseline_us": 21.4, "tuned_us": 15.0, "source": "gemm-tune"},
       {"op": "attn_decode", "pool": "decode", "dims": {"batch": 64}, "us": 41.0}
     ]}

An entry matches a report row when every field it names matches: ``op``,
optionally ``pool``, ``kind``, ``dtype``, exact ``dims`` and inclusive
``ranges``; ``"op": "*"`` matches every op. The most specific match wins (most
fields named, a named op counting as one), then the last one listed. Its speedup is ``speedup``, or ``baseline_us / tuned_us`` (a measured
ratio, applied to the simulated time), or the simulated time over ``us`` (a
measured time for that exact shape).

On the report each matched row keeps ``est_us_base`` and gains ``speedup``;
``est_us`` becomes the sped-up time. On the step cost, a step is charged its
simulated time scaled by how much its own kernels got faster: the sum of their
sped-up times over the sum of their simulated ones. Kernels overlap inside a
step, so this is Amdahl's estimate, not a re-simulation of the kernels.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field

SCHEMA = "inferasim.kernel_speedups"

_ACTIVE: Speedups | None = None


@dataclass
class Entry:
    op: str
    pool: str | None = None
    kind: str | None = None
    dtype: str | None = None
    dims: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)
    speedup: float | None = None
    us: float | None = None
    source: str | None = None

    @property
    def specificity(self) -> int:
        return (
            (self.op != "*")
            + sum(x is not None for x in (self.pool, self.kind, self.dtype))
            + len(self.dims)
            + len(self.ranges)
        )

    def matches(self, row: dict) -> bool:
        if self.op != "*" and row.get("op") != self.op:
            return False
        for name in ("pool", "kind", "dtype"):
            want = getattr(self, name)
            if want is not None and str(row.get(name)) != str(want):
                return False
        for dim, want in self.dims.items():
            if row.get(dim) != want:
                return False
        for dim, (lo, hi) in self.ranges.items():
            v = row.get(dim)
            if v is None or not (lo <= v <= hi):
                return False
        return True

    def factor(self, est_us: float | None) -> float | None:
        if self.speedup is not None:
            return self.speedup
        if self.us is not None and est_us:
            return est_us / self.us
        return None


def _entry(raw: dict) -> Entry:
    if "op" not in raw:
        raise ValueError(f"kernel speedup entry without an op: {raw}")
    speedup = raw.get("speedup")
    if speedup is None and raw.get("baseline_us") and raw.get("tuned_us"):
        speedup = float(raw["baseline_us"]) / float(raw["tuned_us"])
    us = raw.get("us")
    if speedup is None and us is None:
        raise ValueError(f"kernel speedup entry for {raw['op']} names no speedup, times or us")
    if speedup is not None and (not math.isfinite(float(speedup)) or float(speedup) <= 0):
        raise ValueError(f"kernel speedup for {raw['op']} must be positive: {speedup}")
    return Entry(
        op=str(raw["op"]),
        pool=raw.get("pool"),
        kind=raw.get("kind"),
        dtype=raw.get("dtype"),
        dims=dict(raw.get("dims") or {}),
        ranges={k: (v[0], v[1]) for k, v in (raw.get("ranges") or {}).items()},
        speedup=None if speedup is None else float(speedup),
        us=None if us is None else float(us),
        source=raw.get("source"),
    )


class Speedups:
    def __init__(self, entries: list[Entry]) -> None:
        self.entries = entries

    @classmethod
    def load(cls, path: str) -> Speedups:
        with open(path) as fh:
            raw = json.load(fh)
        return cls.parse(raw)

    @classmethod
    def parse(cls, raw) -> Speedups:
        if isinstance(raw, dict):
            schema = raw.get("schema")
            if schema not in (None, SCHEMA):
                raise ValueError(f"not a kernel speedups file: schema {schema!r}")
            raw = raw.get("entries", [])
        return cls([_entry(r) for r in raw])

    def match(self, row: dict) -> Entry | None:
        best, rank = None, -1
        for e in self.entries:
            if e.matches(row) and e.specificity >= rank:
                best, rank = e, e.specificity
        return best

    def factor(self, row: dict, est_us: float | None) -> float:
        e = self.match(row)
        f = e.factor(est_us) if e is not None else None
        return f if f and f > 0 else 1.0

    def annotate(self, row: dict) -> None:
        """Apply the matching speedup to one kernel-report row, in place."""
        e = self.match(row)
        if e is None:
            return
        base = row.get("est_us")
        f = e.factor(base)
        if f is None:
            if e.us is not None:
                row["est_us_base"] = None
                row["est_us"] = round(e.us, 3)
                row["est_total_ms"] = round(e.us * row["calls"] / 1000.0, 3)
                row["est_source"] = "measured"
            return
        row["est_us_base"] = base
        row["speedup"] = round(f, 4)
        if base is not None:
            row["est_us"] = round(base / f, 3)
            row["est_total_ms"] = round(base / f * row["calls"] / 1000.0, 3)
            row["est_source"] = f"{row.get('est_source')}+{e.source or 'speedup'}"

    def step_factor(self, timed: list[tuple[dict, float, int]]) -> float:
        """Sped-up over simulated time of one step's ``(row, est_us, calls)`` kernels."""
        before = after = 0.0
        for row, us, calls in timed:
            if not us:
                continue
            before += us * calls
            after += us * calls / self.factor(row, us)
        return after / before if before > 0 else 1.0


def active() -> Speedups | None:
    return _ACTIVE


@contextlib.contextmanager
def applying(speedups: Speedups | None) -> Iterator[Speedups | None]:
    """Charge every DES step inside this block its kernels' measured speedups."""
    global _ACTIVE
    prev, _ACTIVE = _ACTIVE, speedups
    try:
        yield speedups
    finally:
        _ACTIVE = prev
