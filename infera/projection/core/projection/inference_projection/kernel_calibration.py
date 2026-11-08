###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A serving stack's measured kernel efficiency, applied to the kernel report.

The kernel report (:mod:`kernel_shapes`) times every kernel as an efficient
implementation of it would run: Origami, the SDPA simulator, a roofline. A real
engine runs some kernels far from that (a top-k over a few experts at 1000x its
roofline) and runs kernels no model lists (a pad's fill, layout copies). One
profile of the stack, matched against a replay of the same deployment, gives
both, and they carry to the stack's other deployments::

    {"schema": "inferasim.kernel_calibration", "version": 1,
     "source": {"model": "...", "engine": "sglang", "image_digest": "..."},
     "factors": [
       {"op": "moe_topk", "factor": 1320.0},
       {"op": "attn_decode", "kind": "attention", "factor": 1.4}
     ],
     "extras": [
       {"op": "engine_fill", "share": 0.36, "kernels": ["FillFunctor ..."]}
     ]}

A factor is measured over estimated time; it matches report rows as a kernel
speedup does (:class:`kernel_speedups.Entry`: ``op``, optionally ``pool``,
``kind``, ``dtype``, ``dims`` and ``ranges``, the most specific match winning)
and multiplies their time. An extra is a kernel the engine adds: it is charged
``share`` of the time of the kernels its pool's steps launch, and is listed as
its own row.

This changes the report and how much a step speeds up under kernel speedups
(its share of the step), not the DES step time, which the projector's own
benchmark calibration sets.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from .kernel_speedups import Entry

SCHEMA = "inferasim.kernel_calibration"


@dataclass
class Extra:
    op: str
    share: float
    pool: str | None = None
    kernels: list = field(default_factory=list)


@dataclass
class Calibration:
    factors: list[Entry]
    extras: list[Extra]
    source: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> Calibration:
        with open(path) as fh:
            return cls.parse(json.load(fh))

    @classmethod
    def parse(cls, raw: dict) -> Calibration:
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"not a kernel calibration file: schema {raw.get('schema')!r}")
        factors = []
        for f in raw.get("factors") or []:
            v = float(f["factor"])
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"kernel calibration factor for {f.get('op')} must be positive: {v}")
            factors.append(
                Entry(
                    op=str(f["op"]),
                    pool=f.get("pool"),
                    kind=f.get("kind"),
                    dtype=f.get("dtype"),
                    dims=dict(f.get("dims") or {}),
                    ranges={k: (v[0], v[1]) for k, v in (f.get("ranges") or {}).items()},
                    speedup=v,
                )
            )
        extras = []
        for e in raw.get("extras") or []:
            share = float(e["share"])
            if not math.isfinite(share) or share < 0:
                raise ValueError(f"kernel calibration share for {e.get('op')} must be >= 0: {share}")
            extras.append(Extra(str(e["op"]), share, e.get("pool"), list(e.get("kernels") or [])))
        return cls(factors, extras, dict(raw.get("source") or {}))

    def factor(self, row: dict) -> float | None:
        best, rank = None, -1
        for e in self.factors:
            if e.matches(row) and e.specificity >= rank:
                best, rank = e, e.specificity
        return None if best is None else best.speedup

    def extras_for(self, pool: str) -> list[Extra]:
        base = pool.split("/", 1)[0]
        return [e for e in self.extras if e.pool in (None, pool, base) and e.share > 0]
