###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Throughput-vs-interactivity Pareto frontiers, InferenceX-style.

For one workload -- a fixed ISL/OSL scenario or a recorded agentic trace --
this projects every combination of model, engine, precision, parallelism,
concurrency, speculative decoding (MTP / EAGLE / draft model) and deployment
knobs (scheduler policy, step budget, chunked prefill, KV dtype, memory
fraction, HBM size, ... or any projection flag) the caller lists, through the same closed-loop DES the corpus is
scored with, and keeps the configurations no other configuration beats on both
axes. The default axes are InferenceX's: interactivity (tokens/s each user
sees, ``1000 / TPOT``) against total token throughput per GPU.

Each point is projected in one of two modes, and every row says which:

* ``benchmark`` -- calibrated from a measured anchor of the same model and
  engine. A point no anchor covers is reported as uncovered rather than
  silently priced analytically, because the two answers are not equally
  trustworthy and a chart that mixes them unlabelled reads as one.
* ``simulate`` -- the analytical path, no GPU. Ranking-quality.

Outputs are a CSV and JSON of every point, a self-contained interactive HTML
page with model / scenario / precision / mode selectors and selectable axes,
a static SVG per chart, and a PNG per chart when matplotlib is installed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import functools
import glob
import html
import io
import itertools
import json
import math
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

# Dtype flags per precision name. The KV cache follows what the serving
# recipes actually run: fp8 KV for every low-precision weight format.
PRECISIONS: dict[str, dict[str, str]] = {
    "bf16": dict(weight="bf16", kv="bf16", moe="bf16", linear="bf16"),
    "fp8": dict(weight="fp8", kv="fp8", moe="fp8", linear="fp8"),
    "mxfp4": dict(weight="mxfp4", kv="fp8", moe="mxfp4", linear="mxfp4"),
    "fp4": dict(weight="mxfp4", kv="fp8", moe="mxfp4", linear="mxfp4"),
    # MXFP4 experts behind bf16 attention and dense layers, as gpt-oss ships;
    # its anchors record the checkpoint's format, mxfp4.
    "mxfp4moe": dict(weight="bf16", kv="bf16", moe="mxfp4", linear="bf16", match="mxfp4"),
}

# Per-step token budget each engine runs with when the recipe sets none
# (vLLM --max-num-batched-tokens, SGLang --chunked-prefill-size).
ENGINE_STEP_BUDGET: dict[str, int] = {"vllm": 8192, "sglang": 16384, "atom": 16384}

# metric -> (label, higher is better)
METRICS: dict[str, tuple[str, bool]] = {
    "interactivity": ("Interactivity (tok/s/user)", True),
    "total_tput_per_gpu": ("Token throughput per GPU (tok/s/gpu)", True),
    "output_tput_per_gpu": ("Output token throughput per GPU (tok/s/gpu)", True),
    "input_tput_per_gpu": ("Input token throughput per GPU (tok/s/gpu)", True),
    "ttft_ms": ("Mean TTFT (ms)", False),
    "ttft_p90_ms": ("P90 TTFT (ms)", False),
    "tpot_ms": ("Mean TPOT (ms)", False),
    "e2e_ms": ("Mean end-to-end latency (ms)", False),
    "cost_per_mtok": ("Cost per million total tokens ($)", False),
    "prefix_hit_rate": ("Prefix-cache block reuse", True),
}

_PAR_RE = re.compile(r"^tp(\d+)(?:ep(\d+))?(?:dp(\d+))?(?:dcp(\d+))?(?:pp(\d+))?$")


@dataclass(frozen=True)
class Parallelism:
    tp: int
    ep: int = 1
    dp_attn: int = 1
    pp: int = 1
    dcp: int = 1

    @classmethod
    def parse(cls, spec: str) -> Parallelism:
        """``tp8``, ``tp4ep4``, ``tp8ep8dp8`` (attention DP), ``tp4dcp4`` (decode context parallel), ``tp4pp2``."""
        m = _PAR_RE.match(spec.strip().lower())
        if not m:
            raise ValueError(
                f"bad parallelism '{spec}' (expected e.g. tp8, tp8ep8, tp8ep8dp8, tp4dcp4)"
            )
        tp, ep, dp, dcp, pp = (int(g) if g else 1 for g in m.groups())
        if dp > tp:
            raise ValueError(f"'{spec}': attention DP cannot exceed TP")
        if tp % dcp:
            raise ValueError(f"'{spec}': decode context parallelism must divide TP")
        return cls(tp=tp, ep=ep, dp_attn=dp, pp=pp, dcp=dcp)

    @property
    def gpus(self) -> int:
        return self.tp * self.pp

    @property
    def label(self) -> str:
        s = f"TP{self.tp}"
        if self.ep > 1:
            s += f" EP{self.ep}"
        if self.dp_attn > 1:
            s += f" DPA{self.dp_attn}"
        if self.dcp > 1:
            s += f" DCP{self.dcp}"
        if self.pp > 1:
            s += f" PP{self.pp}"
        return s


@dataclass(frozen=True)
class ModelSpec:
    """A preset plus the checkpoint it is served as.

    ``glm5@GLM-5.2-MXFP4`` projects the ``glm5`` preset and matches anchors
    harvested from a checkpoint named ``GLM-5.2-MXFP4``. Without ``@`` the
    preset name is both. A trailing ``=PRECISION`` states the checkpoint's
    format when neither its name nor its anchors record one
    (``deepseek_v4_pro@DeepSeek-V4-Pro=fp8``), so benchmark mode does not
    calibrate another precision from it.
    """

    preset: str
    checkpoint: str
    precision: str | None = None

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        spec, _, prec = spec.partition("=")
        preset, _, ckpt = spec.partition("@")
        prec = prec.strip().lower() or None
        if prec and prec not in PRECISIONS:
            raise ValueError(f"unknown precision {prec!r} in {spec!r}")
        return cls(preset=preset.strip(), checkpoint=(ckpt or preset).strip(), precision=prec)


@dataclass(frozen=True)
class Scenario:
    """A fixed ISL/OSL point, or a trace replayed closed-loop."""

    name: str
    isl: int
    osl: int
    trace: str | None = None

    @classmethod
    def parse(cls, spec: str, trace_max_requests: int = 0) -> Scenario:
        """``8k/1k``, ``1024/1024``, or ``trace:/path/to/agentx.jsonl``.

        A trace path holding ``{c}`` is one file per concurrency, e.g.
        ``trace:lanes/agentx_256k_c{c}.jsonl``: a trace interleaving many more
        conversations than there are clients evicts every prefix before its
        next turn arrives, so a closed loop at C clients should replay a trace
        laid out as C lanes. The listed ISL/OSL is that of the first file found.
        """
        if spec.startswith("trace:"):
            path = spec[len("trace:") :]
            probe = path
            if "{c}" in path:
                found = sorted(glob.glob(path.replace("{c}", "*")))
                if not found:
                    raise ValueError(f"no trace matches '{path}'")
                probe = found[0]
            isl, osl = _trace_mean_lengths(probe, trace_max_requests)
            name = (
                os.path.splitext(os.path.basename(path))[0].replace("_c{c}", "").replace("{c}", "")
            )
            return cls(name=f"trace:{name}", isl=isl, osl=osl, trace=path)
        a, _, b = spec.lower().partition("/")
        if not b:
            raise ValueError(f"bad scenario '{spec}' (expected ISL/OSL like 8k/1k, or trace:PATH)")
        return cls(name=spec, isl=_tokens(a), osl=_tokens(b))


def _tokens(s: str) -> int:
    s = s.strip().lower()
    if s.endswith("k"):
        return int(float(s[:-1]) * 1024)
    return int(s)


def _trace_rows(path: str, limit: int) -> list[str]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(line)
                if limit and len(out) >= limit:
                    break
    return out


@functools.cache
def _trace_stats(path: str, limit: int) -> tuple[int, int, int]:
    """Mean ISL, mean OSL and the longest context of a trace."""
    rows = [json.loads(x) for x in _trace_rows(path, limit)]
    if not rows:
        raise ValueError(f"trace '{path}' is empty")
    lens = [(int(r.get("input_length", 1)), int(r.get("output_length", 1))) for r in rows]
    isl = sum(i for i, _ in lens) / len(lens)
    osl = sum(o for _, o in lens) / len(lens)
    return max(1, int(isl)), max(1, int(osl)), max(i + o for i, o in lens)


def _trace_mean_lengths(path: str, limit: int) -> tuple[int, int]:
    return _trace_stats(path, limit)[:2]


AGENTX_IDLE_CAP_MS = 10000.0


def _sliced_trace(path: str, limit: int, workdir: str) -> str:
    """The first ``limit`` rows of a trace, so a sweep is not paced by its length.

    Taken from the head because rows reference earlier rows (``hp``) and
    sessions reference earlier turns; a head slice keeps every reference whole.
    """
    if not limit:
        return path
    os.makedirs(workdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(workdir, f"{base}.head{limit}.jsonl")
    if not os.path.exists(out):
        with open(out, "w") as f:
            f.writelines(_trace_rows(path, limit))
    return out


@dataclass(frozen=True)
class KvConfig:
    """A KV memory hierarchy to deploy with: ``hbm`` alone, or HBM plus tiers.

    ``dram=dram:512`` and ``dram+ssd=dram:512,ssd:4000`` name a hierarchy and
    list its tiers in ``--kv-tier`` syntax, fastest first.
    """

    label: str
    tiers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, spec: str) -> KvConfig:
        label, _, body = spec.partition("=")
        label = label.strip()
        if not body:
            if label.lower() in ("hbm", "none", ""):
                return cls(label="hbm")
            body, label = label, ""
        tiers = tuple(t.strip() for t in body.split(",") if t.strip())
        from .kv_tiers import KvTier

        for t in tiers:
            KvTier.parse(t)
        return cls(label=label or "+".join(t.split(":")[0] for t in tiers), tiers=tiers)


# method -> the draft depth it runs at when the setting names none:
# InferenceX's recipes' ``speculative-num-steps 3``, 4 for a separate draft
# model and n-gram lookup.
SPEC_DEFAULTS: dict[str, int] = {
    "mtp": 3,
    "nextn": 3,
    "eagle": 3,
    "eagle3": 3,
    "dspark": 3,
    "draft_model": 4,
    "draft": 4,
    "ngram": 4,
}


@dataclass(frozen=True)
class SpecConfig:
    """One speculative-decoding setting: ``none`` or ``METHOD[:K[:ACCEPT[:DRAFT_COST]]]``.

    Left out (or ``golden``), ``ACCEPT`` is what the measured acceptance-length
    files give the model at that depth (:mod:`.golden_al`: its own curve, its
    sibling head's, else the method's default) and ``DRAFT_COST`` is the
    drafter's own forward, priced from its layers; the projector resolves both
    and reports where they came from. A number states a per-draft-token
    acceptance or a flat draft cost per proposed token instead.

    ``K`` may be a range (``mtp:1-8``, or ``mtp:golden`` for every depth the
    curves cover); each depth is projected, but the method stays one line, so
    the frontier picks the depth at each concurrency.
    """

    method: str = "none"
    k: int = 0
    accept: float | None = None
    draft_cost: float | None = None
    sweep: bool = False
    thinking: str = "on"

    @classmethod
    def parse_many(cls, spec: str) -> list[SpecConfig]:
        parts = [p.strip() for p in spec.strip().lower().split(":")]
        if len(parts) < 2 or (parts[1] != "golden" and "-" not in parts[1]):
            return [cls.parse(spec)]
        if parts[0] not in SPEC_DEFAULTS:
            raise ValueError(f"unknown speculative method {parts[0]!r} in {spec!r}")
        from .golden_al import DEFAULT_DEPTHS

        rng, rest = ("", parts[2:]) if parts[1] == "golden" else (parts[1], parts[2:])
        lo, _, hi = rng.partition("-")
        try:
            ks = range(int(lo or 1), int(hi or lo or max(DEFAULT_DEPTHS)) + 1)
        except ValueError:
            raise ValueError(f"bad speculative setting {spec!r}") from None
        if not ks or ks[0] < 1:
            raise ValueError(f"bad speculative setting {spec!r}")
        one = cls.parse(":".join([parts[0], str(ks[0]), *rest]))
        return [replace(one, k=k, sweep=True) for k in ks]

    @classmethod
    def parse(cls, spec: str) -> SpecConfig:
        parts = [p.strip() for p in spec.strip().lower().split(":")]
        if parts[0] in ("", "none", "off", "0"):
            return cls()
        if parts[0] not in SPEC_DEFAULTS:
            raise ValueError(f"unknown speculative method {parts[0]!r} in {spec!r}")
        if len(parts) > 1 and (parts[1] == "golden" or "-" in parts[1]):
            raise ValueError(f"a depth range names several settings; use parse_many: {spec!r}")
        k, acc, dc = SPEC_DEFAULTS[parts[0]], None, None
        try:
            if len(parts) > 1 and parts[1]:
                k = int(parts[1])
            if len(parts) > 2 and parts[2] not in ("", "golden"):
                acc = float(parts[2])
            if len(parts) > 3 and parts[3]:
                dc = float(parts[3])
        except ValueError:
            raise ValueError(f"bad speculative setting {spec!r}") from None
        if k < 1 or (acc is not None and not 0.0 <= acc <= 1.0) or (dc is not None and dc < 0.0):
            raise ValueError(f"bad speculative setting {spec!r}")
        return cls(method=parts[0], k=k, accept=acc, draft_cost=dc)

    @property
    def label(self) -> str:
        if self.method == "none":
            return "none"
        m = self.method.upper()
        # A depth sweep is one line per method: its depths are a choice the
        # frontier makes.
        s = m if self.sweep else f"{m}{'-' if m[-1].isdigit() else ''}{self.k}"
        if self.accept is not None:
            s += f"@{self.accept:g}"
        if self.draft_cost is not None:
            s += f"/d{self.draft_cost:g}"
        return s

    def argv(self) -> list[str]:
        if self.method == "none":
            return []
        method = "draft_model" if self.method == "draft" else self.method
        out = [
            "--speculative-method",
            method,
            "--speculative-num-tokens",
            str(self.k),
            "--speculative-thinking",
            self.thinking,
        ]
        if self.accept is not None:
            out += ["--speculative-acceptance-rate", str(self.accept)]
        if self.draft_cost is not None:
            out += [
                "--speculative-draft-cost-factor",
                str(self.draft_cost),
                "--speculative-draft-cost-model",
                "flat",
            ]
        return out


def _sched_argv(v: str) -> list[str]:
    policy, _, cap = v.partition(":")
    if policy not in ("unified", "exclusive"):
        raise ValueError(f"scheduler policy must be unified[:N] or exclusive[:N], got {v!r}")
    out = ["--des-exclusive-prefill"] if policy == "exclusive" else []
    return out + (["--des-max-prefill-seqs", str(int(cap))] if cap else [])


# Named deployment knobs -> the projection flags one value turns into. ``budget``
# and ``hbm`` are applied by :func:`point_argv` itself: the step budget is split
# across attention-DP ranks and the HBM size feeds the memory model.
KNOBS: dict[str, tuple[str, Any]] = {
    "sched": (
        "scheduler: unified | exclusive (prefill-only steps), ':N' caps new seqs/step",
        _sched_argv,
    ),
    "budget": ("step token budget (max-num-batched-tokens), 0 = engine default", None),
    "hbm": ("HBM capacity per GPU in GB", None),
    "chunk": (
        "chunked-prefill chunk size in tokens (0 disables)",
        lambda v: ["--chunked-prefill-size", str(int(v))],
    ),
    "kvdtype": ("KV-cache dtype, overriding the precision's", lambda v: ["--kv-cache-dtype", v]),
    "memfrac": (
        "fraction of HBM the engine may use (mem_fraction_static)",
        lambda v: ["--kv-cache-memory-fraction", str(float(v))],
    ),
    "maxseqs": ("max running sequences per instance", lambda v: ["--max-num-seqs", str(int(v))]),
    "think": (
        "client think time between turns, ms",
        lambda v: ["--des-client-think-ms", str(float(v))],
    ),
    "block": ("KV block size in tokens", lambda v: ["--kv-block-size", str(int(v))]),
}


@dataclass(frozen=True)
class KnobAxis:
    """One swept knob: a :data:`KNOBS` name or any projection ``--flag``.

    A raw flag takes ``on`` / ``off`` for a switch and anything else as its
    value, so a knob the sweep has no name for is still one ``--knob`` away.
    """

    name: str
    values: tuple[str, ...]

    @classmethod
    def parse(cls, spec: str) -> KnobAxis:
        name, sep, body = spec.partition("=")
        name = name.strip()
        vals = tuple(v.strip() for v in body.split(",") if v.strip())
        if not sep or not name or not vals:
            raise ValueError(f"knob must be NAME=V1[,V2...], got {spec!r}")
        if name not in KNOBS and not name.startswith("--"):
            raise ValueError(f"unknown knob {name!r}: one of {sorted(KNOBS)} or a --flag")
        ax = cls(name, vals)
        for v in vals:
            ax.argv(v)
        return ax

    def argv(self, value: str) -> list[str]:
        if self.name in ("budget", "hbm"):
            float(value)
            return []
        if self.name in KNOBS:
            return KNOBS[self.name][1](value)
        if value.lower() in ("on", "true"):
            return [self.name]
        if value.lower() in ("off", "false"):
            return []
        return [self.name, value]

    @property
    def short(self) -> str:
        return self.name.lstrip("-")


def knob_combos(axes: Sequence[KnobAxis]) -> list[tuple[tuple[str, str], ...]]:
    """Every combination of the swept knob values, as ``((name, value), ...)``."""
    if not axes:
        return [()]
    return [
        tuple(zip((a.name for a in axes), vals, strict=True))
        for vals in itertools.product(*(a.values for a in axes))
    ]


def _knob_label(combo: Sequence[tuple[str, str]]) -> str:
    return " ".join(f"{n.lstrip('-')}={v}" for n, v in combo)


@dataclass
class ParetoPoint:
    model: str
    checkpoint: str
    engine: str
    precision: str
    gpu_arch: str
    scenario: str
    isl: int
    osl: int
    mode: str  # benchmark | simulate
    kv_config: str
    parallelism: str
    tp: int
    ep: int
    dp_attn: int
    pp: int
    gpus: int
    concurrency: int
    anchor: str = ""
    dcp: int = 1
    spec: str = "none"
    spec_k: int = 0
    spec_accept: float = 0.0
    spec_al_source: str = ""
    knobs: str = ""
    series_label: str = ""

    feasible: bool = True
    reason: str = ""

    ttft_ms: float = 0.0
    ttft_p90_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_ms: float = 0.0
    interactivity: float = 0.0
    total_tput_per_gpu: float = 0.0
    output_tput_per_gpu: float = 0.0
    input_tput_per_gpu: float = 0.0
    cost_per_mtok: float = 0.0
    prefix_hit_rate: float = 0.0
    memory_per_gpu_gb: float = 0.0
    kv_tiers: str = ""
    spec_tokens_per_step: float = 1.0
    spec_draft_cost: str = ""
    spec_draft_ms: float = 0.0
    on_frontier: bool = False

    def series(self) -> str:
        if self.series_label:
            return self.series_label
        kv = "" if self.kv_config == "hbm" else f" KV:{self.kv_config}"
        sp = "" if self.spec == "none" else f" {self.spec}"
        return f"{self.gpu_arch.upper()} {self.engine}{sp}{kv} ({self.mode})"

    def chart(self) -> tuple[str, str, str]:
        return (self.model, self.scenario, self.precision)

    def short(self) -> str:
        """Point label on a chart: layout, clients and, if speculating, the depth."""
        k = f" k{self.spec_k}" if self.spec != "none" and self.spec_k else ""
        return f"{self.parallelism} c{self.concurrency}{k}"


@dataclass
class ParetoResult:
    points: list[ParetoPoint] = field(default_factory=list)
    elapsed_s: float = 0.0
    x: str = "interactivity"
    y: str = "total_tput_per_gpu"


# -- anchors -----------------------------------------------------------------


def _anchor_files(specs: Sequence[str]) -> list[str]:
    out: list[str] = []
    for s in specs or ():
        if os.path.isdir(s):
            out += sorted(glob.glob(os.path.join(s, "*.json")))
        else:
            out += sorted(glob.glob(s)) or [s]
    return out


def _anchor_meta(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            art = json.load(f)
    except (OSError, ValueError):
        return None
    if not (isinstance(art, dict) and isinstance(art.get("meta"), dict)):
        return None
    sweep = art.get("sweep") or []
    if not any(isinstance(e, dict) and e.get("decode_ms") for e in sweep):
        return None
    m = art["meta"]
    model = (art.get("measured") or {}).get("model") or {}
    return dict(
        prefill=m.get("prefill_anchor") or {},
        has_prefill=model.get("prefill_ms") is not None,
        # The unblocked step rather than mean TPOT, which already carries the
        # prefill stalls the scheduler models.
        decode_ok=str(m.get("decode_observable") or "").startswith("median_itl"),
        path=path,
        backend=str(art.get("backend") or "").lower(),
        checkpoint=os.path.basename(str(m.get("model") or "").rstrip("/")),
        tp=int(m.get("benchmark_tp") or m.get("tp") or 1),
        ep=int(m.get("benchmark_ep") or m.get("ep") or 1),
        isl=int(m.get("input_len") or 0),
        gpus=int(m.get("benchmark_gpus") or 0),
        mtime=os.path.getmtime(path),
        precision=_named_precision(str(m.get("weight_dtype") or "").replace("bfloat16", "bf16")),
    )


def with_decode_context(
    anchor: dict[str, Any], anchors: Sequence[dict[str, Any]], out_dir: str
) -> str:
    """The anchor's path, or a derived copy carrying a decode-vs-context grid.

    An anchor's batch sweep reads exactly its own prompt length of KV, and the
    projector bills decode flat in context unless the artifact holds a
    ``decode_ctx`` grid, which the serving harness never writes. Anchors are
    harvested long and read short, so the error is one-sided: TPOT at ISL 1024
    off an ISL 8192 anchor reads high. The grid is drawn from siblings of the
    same checkpoint, engine and width harvested at other lengths, as the
    corpus scorer does, so a sweep point and the config it reproduces agree.
    """
    grid, donors = [], []
    for o in anchors:
        if (
            o["path"] == anchor["path"]
            or not o["isl"]
            or o["isl"] == anchor["isl"]
            or o["checkpoint"].lower() != anchor["checkpoint"].lower()
            or o["backend"] != anchor["backend"]
            or o["tp"] != anchor["tp"]
        ):
            continue
        try:
            with open(o["path"]) as f:
                sweep = json.load(f).get("sweep") or []
        except (OSError, ValueError):
            continue
        pts = [
            e
            for e in sweep
            if isinstance(e, dict) and e.get("batch") and float(e.get("decode_ms") or 0) > 0
        ]
        donors += [o["path"]] if pts else []
        grid += [
            dict(batch=int(e["batch"]), context=float(o["isl"]), decode_ms=float(e["decode_ms"]))
            for e in pts
        ]
    if not grid or not anchor["isl"]:
        return anchor["path"]
    name = os.path.splitext(os.path.basename(anchor["path"]))[0]
    dest = os.path.join(out_dir, f"{name}_ctx.json")
    if not os.path.exists(dest):
        with open(anchor["path"]) as f:
            doc = json.load(f)
        doc["decode_ctx"] = grid
        doc.setdefault("meta", {})["decode_ctx_donors"] = [os.path.basename(p) for p in donors]
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{dest}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(doc, f)
        os.replace(tmp, dest)
    return dest


def _prefill_points(pa: dict[str, Any]) -> list[tuple[float, float]]:
    return sorted(
        (float(p.get("input_len") or 0.0), float(p.get("mean_ttft_ms") or 0.0))
        for p in (pa.get("points") or [])
    )


def _prefill_covers(pa: dict[str, Any], isl: int) -> bool:
    """Whether the prefill probe came within reach of ``isl`` (or recorded no lengths)."""
    lengths = [n for n, _ in _prefill_points(pa) if n > 0]
    return not lengths or isl >= 0.9 * min(lengths)


def _curve_is_physical(pa: dict[str, Any], isl: int) -> bool:
    """Whether the prefill length fit describes prefill at ``isl``.

    Refused when its intercept or linear rate is negative, its rate at ``isl``
    is not positive, ``isl`` lies below the probed range, or the measured
    points fall by more than 5% of their span between neighbours.
    """
    cf = pa.get("curve_fit") or {}
    if not cf:
        return False
    a = float(cf.get("ms_per_token") or 0.0)
    b = float(cf.get("ms_per_token_sq") or 0.0)
    if a <= 0.0 or float(cf.get("fixed_ms") or 0.0) < 0.0 or a + b * isl <= 0.0:
        return False
    if not _prefill_covers(pa, isl):
        return False
    ys = [y for _, y in _prefill_points(pa)]
    span = max(ys, default=0.0) - min(ys, default=0.0)
    return not (len(ys) >= 3 and span > 0 and min(q - p for p, q in zip(ys, ys[1:])) < -0.05 * span)


def _packed_is_usable(pa: dict[str, Any]) -> bool:
    """Whether a packed-prefill probe agrees with the single-sequence curve it rides on."""
    cf = pa.get("curve_fit") or {}
    pk = pa.get("packed") or {}
    pts = pk.get("points") or []
    if not (cf and pk.get("ms_per_token") and len(pts) >= 2):
        return False
    a = float(cf.get("ms_per_token") or 0.0)
    b = float(cf.get("ms_per_token_sq") or 0.0)
    fx = float(cf.get("fixed_ms") or 0.0)
    n, t = max((int(p.get("step_tokens") or 0), float(p.get("last_ttft_ms") or 0.0)) for p in pts)
    single = fx + a * n + b * n * n
    if n <= 0 or t <= 0 or single <= 0 or not 0.5 <= t / single <= 1.5:
        return False
    return 0.6 <= (fx + float(pk["ms_per_token"]) * n) / t <= 1.4


def _named_precision(name: str) -> str | None:
    """The weight format a checkpoint name declares, if it declares one."""
    n = name.lower()
    if "fp4" in n:
        return "mxfp4"
    if "fp8" in n:
        return "fp8"
    if "bf16" in n:
        return "bf16"
    return None


def pick_anchor(
    anchors: Sequence[dict[str, Any]],
    *,
    checkpoint: str,
    engine: str,
    par: Parallelism,
    isl: int,
    max_anchor_tp: int = 4,
    precision: str | None = None,
    native: str | None = None,
) -> dict[str, Any] | None:
    """The anchor a benchmark-mode point is calibrated from, or None.

    ``native`` is the checkpoint's precision when the caller knows it and the
    name does not say.

    Same checkpoint and same engine are requirements: no transport converts one
    engine's kernels into another's. Among those, the ranking is the corpus
    scorer's, so a sweep point and the scored config it reproduces share an
    anchor: a prefill curve that is physical at this prompt length, then a
    recorded prefill that reaches it, then the unblocked decode observable,
    then width -- a target at TP1/TP2 uses an anchor at its own width,
    anything wider restores from TP4 -- then a usable packed-prefill probe,
    which is what prices a step holding several prompts. After those come
    agreement on whether experts are sharded, prompt length, and the most
    recent harvest.
    """
    want_tp = par.tp if par.tp <= 2 else min(4, max_anchor_tp)
    # An anchor measures one checkpoint's kernels, and a checkpoint is one
    # precision. When its name says which, a point at another is not covered.
    # Otherwise the anchor's recorded weight dtype decides; one that recorded
    # none (or only "quark") is taken at the checkpoint's word.
    named = _named_precision(checkpoint)
    if native and not named:
        named = PRECISIONS[native].get("match") or PRECISIONS[native]["weight"]
    want = None
    if precision:
        want = PRECISIONS[precision].get("match") or PRECISIONS[precision]["weight"]
    if want and named and want != named:
        return None
    cands = [
        a
        for a in anchors
        if a["checkpoint"].lower() == checkpoint.lower()
        and not (want and a.get("precision") and a["precision"] != want)
        and a["backend"] == engine.lower()
        and a["tp"] <= max(max_anchor_tp, 1)
        and a["tp"] <= par.tp
    ]
    if not cands:
        return None

    def rank(a):
        pa = a.get("prefill") or {}
        return (
            -int(_curve_is_physical(pa, isl)),
            -int(bool(a.get("has_prefill")) and _prefill_covers(pa, isl)),
            -int(bool(a.get("decode_ok"))),
            abs(a["tp"] - want_tp),
            -int(_packed_is_usable(pa)),
            int((a["ep"] > 1) != (par.ep > 1)),
            abs(a["isl"] - isl) / max(1, isl),
            -a["mtime"],
        )

    return min(cands, key=rank)


# -- one point ---------------------------------------------------------------


def _workload_yaml() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "..", "configs", "inferasim_workload.yaml")
    )


def point_argv(
    *,
    scenario: Scenario,
    trace_path: str | None,
    precision: str,
    engine: str,
    par: Parallelism,
    concurrency: int,
    gpu_arch: str,
    hbm_gb: float,
    num_requests: int,
    anchor: str | None,
    checkpoint: str,
    kv: KvConfig | None = None,
    pool_from_memory: bool = False,
    spec: SpecConfig | None = None,
    knobs: Sequence[tuple[str, str]] = (),
    extra: Sequence[str] = (),
    warmup_requests: int = 0,
) -> list[str]:
    knob = dict(knobs)
    hbm_gb = float(knob.get("hbm") or hbm_gb)
    dt = PRECISIONS[precision]
    # Attention DP runs one scheduler per rank over its share of the clients,
    # and the ranks meet at every MoE layer. The DES simulates one of them with
    # the step priced at the combined batch (INFERASIM_DES_LOCKSTEP), and the
    # throughput is scaled back up by the rank count in ``_fill_metrics``.
    ranks = max(1, par.dp_attn)
    concurrency = -(-int(concurrency) // ranks)
    num_requests = max(concurrency, -(-int(num_requests) // ranks))
    argv = [
        "inference",
        "--config",
        _workload_yaml(),
        "--inference-mode",
        "both",
        "--profiling-mode",
        "simulate",
        "--serving-model",
        "continuous",
        "--serving-engine",
        engine,
        "--input-len",
        str(scenario.isl),
        "--output-len",
        str(scenario.osl),
        "--max-concurrency",
        str(concurrency),
        "--inference-batch-size",
        str(concurrency),
        "--gpu-arch",
        gpu_arch,
        "--hbm-capacity-gb",
        str(hbm_gb),
        "--weight-dtype",
        dt["weight"],
        "--kv-cache-dtype",
        dt["kv"],
        "--moe-expert-dtype",
        dt["moe"],
        "--linear-weight-dtype",
        dt["linear"],
        "--des-closed-loop",
        "--des-num-requests",
        str(num_requests),
        "--des-warmup-requests",
        str(min(int(warmup_requests), num_requests - 1)),
        "--des-warmup-frac",
        "0",
    ]
    budget = int(float(knob.get("budget") or 0)) or ENGINE_STEP_BUDGET.get(engine.lower(), 8192)
    if par.dp_attn > 1:
        argv += ["--attention-dp-size", str(par.dp_attn)]
        # Each attention-DP rank schedules its own share of the step budget.
        budget = max(1, budget // par.dp_attn)
    if par.dcp > 1:
        argv += ["--decode-context-parallel-size", str(par.dcp)]
    argv += ["--max-num-batched-tokens", str(budget)]
    if trace_path:
        # The engine is launched with a context that fits the trace's longest
        # request. Left at the default (mean ISL + mean OSL) every request above
        # the mean can never finish its prefill, and the replay stalls on them.
        argv += [
            "--des-mooncake-trace",
            trace_path,
            "--max-context-len",
            str(_trace_stats(trace_path, 0)[2] + 1),
            # The AgentX harness waits each turn's recorded think time (the
            # trace's ``think_ms``) but pulls pending turns forward once nothing
            # has been in flight for 10 s (AIPerf ``system_idle_gap_cap_seconds``).
            "--des-client-idle-cap-ms",
            str(AGENTX_IDLE_CAP_MS),
        ]
    if anchor:
        argv += ["--load-benchmark", anchor, "--bench-model", checkpoint]
    for t in kv.tiers if kv else ():
        argv += ["--kv-tier", t]
    if pool_from_memory:
        argv += ["--des-kv-pool-from-memory"]
    if spec is not None:
        argv += spec.argv()
    # After the defaults above: argparse keeps the last value of a repeated flag.
    for name, value in knobs:
        argv += KnobAxis(name, (value,)).argv(value)
    argv += list(extra)
    argv += [
        f"tensor_model_parallel_size={par.tp}",
        f"expert_model_parallel_size={par.ep}",
        f"pipeline_model_parallel_size={par.pp}",
    ]
    return argv


def _fill_metrics(pt: ParetoPoint, res: dict[str, Any], gpu_cost_per_hour: float) -> None:
    perf = res.get("performance")
    mem = res.get("memory")
    des = (res.get("des") or {}).get("point")
    gpus = int(getattr(perf, "replica_gpus", 0) or 0) or pt.gpus
    pt.gpus = gpus
    extras = getattr(perf, "extras", None) or {}
    pt.spec_tokens_per_step = float(extras.get("speculative_tokens_per_step") or 1.0)
    pt.spec_al_source = str(extras.get("speculative_acceptance_source") or "")
    pt.spec_draft_cost = str(extras.get("speculative_draft_cost") or "")
    pt.spec_draft_ms = float(extras.get("speculative_draft_ms") or 0.0)
    if mem is not None:
        pt.memory_per_gpu_gb = float(getattr(mem, "total_bytes", 0) or 0) / (1024.0**3)
        # Unrunnable only when weights and the step's activations leave no HBM
        # for KV at all. A pool smaller than every client's full context is not
        # infeasible -- the engine admits what fits and queues the rest, and
        # the DES, whose pool is sized from the same memory model, does too.
        cap = float(getattr(mem, "hbm_capacity_bytes", 0) or 0)
        fixed = float(getattr(mem, "weight_bytes", 0) or 0) + float(
            getattr(mem, "activation_bytes", 0) or 0
        )
        if cap and fixed >= cap:
            pt.feasible = False
            pt.reason = (
                f"weights+activations need {fixed / 1024.0**3:.0f} GB/GPU, "
                f"device has {cap / 1024.0**3:.0f} GB"
            )
    if des is None:
        pt.feasible = False
        pt.reason = pt.reason or "no DES result"
        return
    pt.ttft_ms = float(des.ttft_arrival.get("mean") or des.ttft.get("mean") or 0.0)
    pt.ttft_p90_ms = float(des.ttft_arrival.get("p90") or des.ttft.get("p90") or 0.0)
    pt.tpot_ms = float(des.tpot.get("mean") or 0.0)
    pt.e2e_ms = float(des.e2e.get("mean") or 0.0)
    pt.interactivity = 1000.0 / pt.tpot_ms if pt.tpot_ms > 0 else 0.0
    ranks = max(1, pt.dp_attn)
    out_tps = float(des.system_throughput_tps or 0.0) * ranks
    tot_tps = float(des.system_total_throughput_tps or 0.0) * ranks or out_tps
    pt.output_tput_per_gpu = out_tps / gpus
    pt.total_tput_per_gpu = tot_tps / gpus
    pt.input_tput_per_gpu = max(0.0, tot_tps - out_tps) / gpus
    if gpu_cost_per_hour and pt.total_tput_per_gpu > 0:
        pt.cost_per_mtok = gpu_cost_per_hour / (pt.total_tput_per_gpu * 3600.0) * 1e6
    pfx = getattr(des, "prefix", None) or {}
    pt.prefix_hit_rate = float(pfx.get("block_hit_rate") or 0.0)
    tiers = pfx.get("tiers")
    if isinstance(tiers, dict):
        pt.kv_tiers = json.dumps(tiers, sort_keys=True)
    if pt.tpot_ms <= 0 or tot_tps <= 0:
        pt.feasible = False
        pt.reason = pt.reason or "DES produced no tokens"
    stalled = float((getattr(des, "packing", None) or {}).get("stalled_reqs") or 0.0)
    if stalled:
        pt.feasible = False
        pt.reason = pt.reason or f"DES stalled with {stalled:.0f} request(s) unschedulable"


def project_point(task: dict[str, Any]) -> ParetoPoint:
    """Project one configuration. Runs in a worker process."""
    pt = ParetoPoint(**task["point"])
    os.environ["INFERASIM_MODEL"] = task["preset"]
    os.environ["INFERASIM_TP"] = str(pt.tp)
    os.environ["INFERASIM_EP"] = str(pt.ep)
    if pt.dp_attn > 1:
        os.environ["INFERASIM_DES_LOCKSTEP"] = str(pt.dp_attn)
    else:
        os.environ.pop("INFERASIM_DES_LOCKSTEP", None)
    # SGLang's attention-DP decode measures at the dense charge; the sparse
    # DPA decode cost describes ATOM's kernels.
    if pt.engine == "sglang" and pt.dp_attn > 1:
        os.environ.setdefault("INFERASIM_DPA_DECODE_DENSE", "1")
    try:
        from infera.projection.cli import build_parser

        from . import launch_projection_from_cli

        parsed, overrides = build_parser().parse_known_args(task["argv"])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = launch_projection_from_cli(parsed, overrides)
        if pt.mode == "benchmark" and not res.get("calibrated"):
            pt.feasible = False
            pt.reason = "anchor did not calibrate the projection"
        _fill_metrics(pt, res, float(task.get("gpu_cost_per_hour") or 0.0))
    except Exception as exc:  # noqa: BLE001 - one bad point must not stop a sweep
        pt.feasible = False
        pt.reason = f"{type(exc).__name__}: {str(exc)[:300]}"
    return pt


# -- frontier ----------------------------------------------------------------


def pareto_front(points: Sequence[ParetoPoint], x: str, y: str) -> list[ParetoPoint]:
    """Points no other point matches-or-beats on both axes, ordered along ``x``."""
    xb, yb = METRICS[x][1], METRICS[y][1]
    sx = 1.0 if xb else -1.0
    sy = 1.0 if yb else -1.0
    pts = [p for p in points if p.feasible]
    # Best x first; walking down x, a point is on the frontier iff its y beats
    # every point already seen.
    pts.sort(key=lambda p: (-sx * getattr(p, x), -sy * getattr(p, y)))
    front: list[ParetoPoint] = []
    best = None
    for p in pts:
        v = sy * getattr(p, y)
        if best is None or v > best:
            front.append(p)
            best = v
    return sorted(front, key=lambda p: getattr(p, x))


def mark_frontiers(points: Sequence[ParetoPoint], x: str, y: str) -> None:
    groups: dict[tuple, list[ParetoPoint]] = {}
    for p in points:
        p.on_frontier = False
        groups.setdefault((p.chart(), p.series()), []).append(p)
    for grp in groups.values():
        for p in pareto_front(grp, x, y):
            p.on_frontier = True


# -- driver ------------------------------------------------------------------


def build_tasks(
    *,
    models: Sequence[ModelSpec],
    engines: Sequence[str],
    precisions: Sequence[str],
    parallelisms: Sequence[Parallelism],
    concurrencies: Sequence[int],
    scenarios: Sequence[Scenario],
    modes: Sequence[str],
    anchors: Sequence[dict[str, Any]],
    kv_configs: Sequence[KvConfig] = (KvConfig("hbm"),),
    gpu_arch: str = "mi355x",
    hbm_gb: float = 288.0,
    max_gpus: int = 8,
    max_anchor_tp: int = 4,
    requests_per_client: int = 10,
    min_requests: int = 64,
    warmup_requests: int = 0,
    trace_max_requests: int = 0,
    workdir: str = "pareto_out/traces",
    gpu_cost_per_hour: float = 0.0,
    specs: Sequence[SpecConfig] = (SpecConfig(),),
    knob_axes: Sequence[KnobAxis] = (),
    series_by: Sequence[str] = ("spec", "kv"),
    extra: Sequence[str] = (),
    golden_thinking: str = "on",
) -> tuple[list[dict[str, Any]], list[ParetoPoint]]:
    """Every point to project, plus the benchmark points no anchor covers.

    Speculation and the KV hierarchy split series by default, as InferenceX
    draws MTP as its own line; every other knob is a deployment choice the
    frontier picks among, the way it picks a parallelism. ``series_by`` names
    the axes -- ``spec``, ``kv`` or a knob -- that get their own line instead.
    """
    tasks: list[dict[str, Any]] = []
    uncovered: list[ParetoPoint] = []
    by = set(series_by)
    for m, eng, prec, par, conc, sc, mode, kv, spec, combo in itertools.product(
        models,
        engines,
        precisions,
        parallelisms,
        concurrencies,
        scenarios,
        modes,
        kv_configs,
        specs,
        knob_combos(knob_axes),
    ):
        if par.gpus > max_gpus:
            continue
        if kv.tiers and not sc.trace:
            # A fixed-length scenario never reuses a block, so a tier below HBM
            # has nothing to hold and every hierarchy would plot the same line.
            continue
        # An attention-DP rank is simulated with its share of the clients, and
        # a conversation-affine router gives it the same share of the lanes;
        # replaying all C lanes through one rank's cache would thrash it.
        lanes = -(-conc // max(1, par.dp_attn))
        trace = sc.trace.replace("{c}", str(lanes)) if sc.trace else None
        if spec.method != "none":
            spec = replace(spec, thinking=golden_thinking)
        base = dict(
            model=m.preset,
            checkpoint=m.checkpoint,
            engine=eng,
            precision=prec,
            gpu_arch=gpu_arch,
            scenario=sc.name,
            isl=sc.isl,
            osl=sc.osl,
            mode=mode,
            kv_config=kv.label,
            parallelism=par.label,
            tp=par.tp,
            ep=par.ep,
            dp_attn=par.dp_attn,
            pp=par.pp,
            dcp=par.dcp,
            gpus=par.gpus,
            concurrency=conc,
            spec=spec.label,
            spec_k=spec.k,
            spec_accept=spec.accept or 0.0,
            knobs=_knob_label(combo),
        )
        tag = ""
        if "spec" in by and spec.method != "none":
            tag += f" {spec.label}"
        if "kv" in by and kv.label != "hbm":
            tag += f" KV:{kv.label}"
        split = [(n, v) for n, v in combo if n in by or n.lstrip("-") in by]
        if split:
            tag += f" {_knob_label(split)}"
        base["series_label"] = f"{gpu_arch.upper()} {eng}{tag} ({mode})"
        if trace and not os.path.exists(trace):
            uncovered.append(ParetoPoint(**base, feasible=False, reason=f"no trace {trace}"))
            continue
        if trace:
            if trace != sc.trace:
                base["isl"], base["osl"] = _trace_mean_lengths(trace, trace_max_requests)
            trace = _sliced_trace(trace, trace_max_requests, workdir)
        anchor = None
        if mode == "benchmark":
            a = pick_anchor(
                anchors,
                checkpoint=m.checkpoint,
                engine=eng,
                par=par,
                isl=sc.isl,
                max_anchor_tp=max_anchor_tp,
                precision=prec,
                native=m.precision,
            )
            if a is None:
                uncovered.append(
                    ParetoPoint(
                        **base,
                        feasible=False,
                        reason=f"no {eng} {prec} anchor for {m.checkpoint} at or below TP{par.tp}",
                    )
                )
                continue
            base["anchor"] = a["path"]
            anchor = with_decode_context(
                a, anchors, os.path.join(os.path.dirname(workdir), "anchors")
            )
        nreq = max(min_requests, requests_per_client * conc)
        argv = point_argv(
            scenario=sc,
            trace_path=trace,
            precision=prec,
            engine=eng,
            par=par,
            concurrency=conc,
            gpu_arch=gpu_arch,
            hbm_gb=hbm_gb,
            num_requests=nreq,
            anchor=anchor,
            checkpoint=m.checkpoint,
            kv=kv,
            # Every point runs against the pool its HBM actually leaves, so a
            # concurrency past it queues instead of being priced as resident,
            # and a tier below HBM has something to catch.
            pool_from_memory=True,
            spec=spec,
            knobs=combo,
            extra=extra,
            warmup_requests=warmup_requests,
        )
        tasks.append(
            dict(point=base, preset=m.preset, argv=argv, gpu_cost_per_hour=gpu_cost_per_hour)
        )
    return tasks, uncovered


def run(
    tasks: Sequence[dict[str, Any]],
    *,
    jobs: int = 1,
    progress: bool = True,
    sink: str | None = None,
) -> list[ParetoPoint]:
    """Project every task. ``sink`` gets each point as one JSON line the moment
    it finishes, so a sweep killed part-way keeps what it had done."""
    out: list[ParetoPoint] = []
    sink_f = open(sink, "a") if sink else None  # noqa: SIM115 - closed below

    def _report(i: int, p: ParetoPoint) -> None:
        if sink_f is not None:
            sink_f.write(json.dumps(asdict(p)) + "\n")
            sink_f.flush()
        if not progress:
            return
        tag = "ok" if p.feasible else "X "
        print(
            f"  [{i:>4}/{len(tasks)}] {tag} {p.model:<14} {p.engine:<7} {p.precision:<6} "
            f"{p.mode:<9} {p.kv_config:<10} {p.spec:<6} {p.parallelism:<16} c{p.concurrency:<4} "
            f"{p.knobs + ' ' if p.knobs else ''}"
            f"intvty={p.interactivity:7.1f} tput/gpu={p.total_tput_per_gpu:8.1f}"
            f"{'  ' + p.reason if p.reason else ''}",
            flush=True,
        )

    try:
        if jobs <= 1:
            for i, t in enumerate(tasks, 1):
                p = project_point(t)
                out.append(p)
                _report(i, p)
            return out

        import multiprocessing as mp
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

        # Fork, so each worker inherits the parent's already-imported cost
        # model (and any patches applied to it) instead of paying the import
        # per process. One fresh worker per point: a trace replay leaves
        # gigabytes behind that a reused worker never returns, and a worker
        # the kernel kills for memory then costs its own point, not the sweep.
        # (A pool's max_tasks_per_child cannot be combined with fork.)
        ctx = mp.get_context("fork")
        queue = list(tasks)
        live: dict[Any, tuple[dict[str, Any], ProcessPoolExecutor]] = {}
        i = 0
        while queue or live:
            while queue and len(live) < jobs:
                t = queue.pop(0)
                ex = ProcessPoolExecutor(max_workers=1, mp_context=ctx)
                live[ex.submit(project_point, t)] = (t, ex)
            finished, _ = wait(list(live), return_when=FIRST_COMPLETED)
            for f in finished:
                t, ex = live.pop(f)
                ex.shutdown(wait=False)
                try:
                    p = f.result()
                except Exception as exc:  # noqa: BLE001 - a dead worker loses one point
                    p = ParetoPoint(**t["point"], feasible=False)
                    p.reason = f"worker failed: {type(exc).__name__}: {str(exc)[:200]}"
                i += 1
                out.append(p)
                _report(i, p)
        return out
    finally:
        if sink_f is not None:
            sink_f.close()


_IDENTITY = (
    "model",
    "checkpoint",
    "engine",
    "precision",
    "gpu_arch",
    "scenario",
    "mode",
    "kv_config",
    "spec",
    "spec_k",
    "knobs",
    "tp",
    "ep",
    "dp_attn",
    "pp",
    "dcp",
    "concurrency",
)


def load_points(path: str) -> list[ParetoPoint]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        doc = json.load(f)
    names = set(ParetoPoint.__dataclass_fields__)
    return [
        ParetoPoint(**{k: v for k, v in p.items() if k in names}) for p in doc.get("points", [])
    ]


def load_partial(path: str) -> list[ParetoPoint]:
    """Points a sweep streamed out as it went (see :func:`run`)."""
    if not os.path.exists(path):
        return []
    names = set(ParetoPoint.__dataclass_fields__)
    out = []
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue  # a line cut short by the kill
            out.append(ParetoPoint(**{k: v for k, v in d.items() if k in names}))
    return out


def _key(p: ParetoPoint) -> tuple:
    return tuple(getattr(p, k) for k in _IDENTITY)


def merge_points(old: Sequence[ParetoPoint], new: Sequence[ParetoPoint]) -> list[ParetoPoint]:
    """``old`` with every point ``new`` re-projects replaced by its new row."""
    fresh = {_key(p) for p in new}
    return [p for p in old if _key(p) not in fresh] + list(new)


# -- writers -----------------------------------------------------------------


def write_csv(points: Sequence[ParetoPoint], path: str) -> None:
    rows = [asdict(p) for p in points]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_json(result: ParetoResult, path: str, meta: dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(
            dict(
                meta=meta,
                x=result.x,
                y=result.y,
                elapsed_s=result.elapsed_s,
                metrics={k: dict(label=v[0], higher_is_better=v[1]) for k, v in METRICS.items()},
                points=[asdict(p) for p in result.points],
            ),
            f,
            indent=1,
        )


_PALETTE = [
    "#3b6fb6",
    "#e07b39",
    "#2f9e6e",
    "#c8414b",
    "#7d5bb5",
    "#1fa3b8",
    "#d4a72c",
    "#d1609e",
    "#7a6a58",
    "#5b8c2a",
    "#9a9a9a",
    "#264653",
]


def _slug(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts)).strip("_")


def _split_series(s: str) -> tuple[str, str]:
    """('MI355X sglang MTP3', 'simulate') from 'MI355X sglang MTP3 (simulate)'."""
    m = re.match(r"^(.*?)\s*\((benchmark|simulate)\)$", s)
    return (m.group(1), m.group(2)) if m else (s, "")


def _nice_ticks(hi: float, n: int = 7) -> list[float]:
    """Round tick values from 0 up to hi; the axis itself ends at hi, not at the next tick."""
    if hi <= 0:
        return [0.0, 1.0]
    raw = hi / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= raw)
    return [i * step for i in range(int(hi / step + 1e-9) + 1)]


def _fmt_tick(v: float) -> str:
    if abs(v) >= 1000:
        s = f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return f"{v:.1f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.0f}"


def _chart_layout(points: Sequence[ParetoPoint], x: str, y: str):
    """Feasible points, per-config colour, per-series frontier, and the overall envelope."""
    feas = [p for p in points if p.feasible]
    groups = sorted({_split_series(p.series())[0] for p in feas})
    col = {g: _PALETTE[i % len(_PALETTE)] for i, g in enumerate(groups)}
    series = sorted(
        {p.series() for p in feas}, key=lambda s: (_split_series(s)[0], _split_series(s)[1])
    )
    fronts = {s: pareto_front([p for p in feas if p.series() == s], x, y) for s in series}
    envelope = pareto_front(feas, x, y) if feas else []
    return feas, col, series, fronts, envelope


def _label_points(envelope: Sequence[ParetoPoint], px, py, min_gap: float) -> list[ParetoPoint]:
    """Envelope points spaced at least min_gap apart in plot units, so labels do not pile up."""
    out: list[ParetoPoint] = []
    for p in envelope:
        if all(math.hypot(px(p) - px(q), py(p) - py(q)) >= min_gap for q in out):
            out.append(p)
    return out


def write_svg(points: Sequence[ParetoPoint], path: str, x: str, y: str, title: str) -> None:
    """A static frontier chart with no plotting dependency."""
    feas, col, series, fronts, envelope = _chart_layout(points, x, y)
    if not feas:
        return
    W, T, B, L = 1000, 64, 58, 72
    R = 300
    H = max(580, T + 40 + 20 * (len(col) + 3))
    x1 = max(getattr(p, x) for p in feas) * 1.06 or 1.0
    y1 = max(getattr(p, y) for p in feas) * 1.08 or 1.0
    xt, yt = _nice_ticks(x1), _nice_ticks(y1)

    def px(v):
        return L + v / x1 * (W - L - R)

    def py(v):
        return H - B - v / y1 * (H - T - B)

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif" font-size="12" fill="#333">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{L}" y="28" font-size="17" font-weight="600" fill="#111">{html.escape(title)}</text>',
        f'<text x="{L}" y="47" font-size="12" fill="#777">Pareto frontier per configuration. '
        "Solid = benchmark, dashed = simulate; faint dots are dominated configurations.</text>",
    ]
    if len(envelope) > 1:
        d = " ".join(f"{px(getattr(p, x)):.1f},{py(getattr(p, y)):.1f}" for p in envelope)
        e0, e1 = envelope[0], envelope[-1]
        o.append(
            f'<polygon points="{px(0):.1f},{py(0):.1f} {px(0):.1f},{py(getattr(e0, y)):.1f} {d} '
            f'{px(getattr(e1, x)):.1f},{py(0):.1f}" '
            'fill="#3b6fb6" fill-opacity="0.05"/>'
        )
    for v in xt:
        o.append(
            f'<line x1="{px(v):.1f}" y1="{T}" x2="{px(v):.1f}" y2="{H - B}" stroke="#eef0f3"/>'
        )
        o.append(
            f'<text x="{px(v):.1f}" y="{H - B + 18}" text-anchor="middle" fill="#666">{_fmt_tick(v)}</text>'
        )
    for v in yt:
        o.append(
            f'<line x1="{L}" y1="{py(v):.1f}" x2="{W - R}" y2="{py(v):.1f}" stroke="#eef0f3"/>'
        )
        o.append(
            f'<text x="{L - 8}" y="{py(v) + 4:.1f}" text-anchor="end" fill="#666">{_fmt_tick(v)}</text>'
        )
    o.append(f'<line x1="{L}" y1="{H - B}" x2="{W - R}" y2="{H - B}" stroke="#999"/>')
    o.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{H - B}" stroke="#999"/>')
    o.append(
        f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" font-size="13" fill="#444">'
        f"{html.escape(METRICS[x][0])}</text>"
    )
    o.append(
        f'<text transform="translate(18,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle" '
        f'font-size="13" fill="#444">{html.escape(METRICS[y][0])}</text>'
    )
    for s in series:
        g, mode = _split_series(s)
        c = col[g]
        for p in feas:
            if p.series() == s and p not in fronts[s]:
                o.append(
                    f'<circle cx="{px(getattr(p, x)):.1f}" cy="{py(getattr(p, y)):.1f}" r="2.2" '
                    f'fill="{c}" fill-opacity="0.18"/>'
                )
    for s in series:
        g, mode = _split_series(s)
        c, front = col[g], fronts[s]
        sim = mode == "simulate"
        dash = ' stroke-dasharray="7,4"' if sim else ""
        if len(front) > 1:
            d = " ".join(f"{px(getattr(p, x)):.1f},{py(getattr(p, y)):.1f}" for p in front)
            o.append(
                f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2.2" '
                f'stroke-linejoin="round"{dash}/>'
            )
        for p in front:
            fill = "white" if sim else c
            o.append(
                f'<circle cx="{px(getattr(p, x)):.1f}" cy="{py(getattr(p, y)):.1f}" r="3.8" '
                f'fill="{fill}" stroke="{c}" stroke-width="1.6">'
                f"<title>{html.escape(f'{s}: {p.short()}')}</title></circle>"
            )
    for p in _label_points(envelope, lambda q: px(getattr(q, x)), lambda q: py(getattr(q, y)), 46):
        lx, ly = px(getattr(p, x)) + 6, py(getattr(p, y)) - 7
        txt = html.escape(p.short())
        o.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10" fill="#333" stroke="white" '
            f'stroke-width="3" paint-order="stroke">{txt}</text>'
        )
    lx0, ly = W - R + 24, T + 6
    o.append(
        f'<text x="{lx0}" y="{ly}" font-size="12" font-weight="600" fill="#111">Configuration</text>'
    )
    for g, c in col.items():
        ly += 20
        o.append(f'<rect x="{lx0}" y="{ly - 9}" width="14" height="10" rx="2" fill="{c}"/>')
        o.append(f'<text x="{lx0 + 22}" y="{ly}" font-size="11.5">{html.escape(g)}</text>')
    ly += 30
    o.append(f'<text x="{lx0}" y="{ly}" font-size="12" font-weight="600" fill="#111">Mode</text>')
    for lab, dash, fill in (
        ("benchmark", "", "#555"),
        ("simulate", ' stroke-dasharray="7,4"', "white"),
    ):
        ly += 20
        o.append(
            f'<line x1="{lx0}" y1="{ly - 4}" x2="{lx0 + 30}" y2="{ly - 4}" stroke="#555" stroke-width="2.2"{dash}/>'
            f'<circle cx="{lx0 + 15}" cy="{ly - 4}" r="3.8" fill="{fill}" stroke="#555" stroke-width="1.6"/>'
        )
        o.append(f'<text x="{lx0 + 38}" y="{ly}" font-size="11.5">{lab}</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))


def write_png(
    points: Sequence[ParetoPoint],
    path: str,
    x: str,
    y: str,
    title: str,
    measured: Sequence[dict[str, Any]] = (),
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        return False
    feas, col, series, fronts, envelope = _chart_layout(points, x, y)
    if not feas:
        return False
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax, lax) = plt.subplots(
        1, 2, figsize=(13, 6.6), dpi=150, gridspec_kw={"width_ratios": [3.0, 1.15], "wspace": 0.04}
    )
    lax.axis("off")
    meas = [q for q in measured if q.get(x) is not None and q.get(y) is not None]
    x1 = max([getattr(p, x) for p in feas] + [q[x] for q in meas]) * 1.06 or 1.0
    y1 = max([getattr(p, y) for p in feas] + [q[y] for q in meas]) * 1.08 or 1.0
    xt, yt = _nice_ticks(x1), _nice_ticks(y1)
    if len(envelope) > 1:
        ex = [0.0] + [getattr(p, x) for p in envelope]
        ey = [getattr(envelope[0], y)] + [getattr(p, y) for p in envelope]
        ax.fill_between(ex, ey, 0, color="#3b6fb6", alpha=0.05, lw=0, zorder=0)
    for s in series:
        g, _ = _split_series(s)
        dom = [p for p in feas if p.series() == s and p not in fronts[s]]
        ax.scatter(
            [getattr(p, x) for p in dom],
            [getattr(p, y) for p in dom],
            s=9,
            color=col[g],
            alpha=0.18,
            lw=0,
            zorder=1,
        )
    for s in series:
        g, mode = _split_series(s)
        c, front = col[g], fronts[s]
        sim = mode == "simulate"
        ax.plot(
            [getattr(p, x) for p in front],
            [getattr(p, y) for p in front],
            ls=(0, (5, 3)) if sim else "-",
            lw=2,
            color=c,
            marker="o",
            ms=4.5,
            mfc="white" if sim else c,
            mec=c,
            mew=1.4,
            zorder=3,
            solid_joinstyle="round",
        )
    pkey = {
        "interactivity": "interactivity",
        "total_tput_per_gpu": "tput_per_gpu",
        "ttft_ms": "ttft_ms",
    }
    for q in meas:
        g = f"MI355X {q['engine']}" + ("" if q["spec"] == "none" else f" {q['spec']}")
        c = col.get(g, "#111")
        pj = (q.get("proj") or {}).get("benchmark") or {}
        if not pj.get("tput_per_gpu"):
            pj = (q.get("proj") or {}).get("simulate") or {}
        if x in pkey and y in pkey and pj.get(pkey[x]) and pj.get(pkey[y]):
            ax.plot([q[x], pj[pkey[x]]], [q[y], pj[pkey[y]]], color=c, lw=0.9, alpha=0.6, zorder=4)
        ax.scatter([q[x]], [q[y]], marker="x", s=42, lw=2.0, color=c, zorder=5)
    ax.set_xlim(0, x1)
    ax.set_ylim(0, y1)
    fig.canvas.draw()

    def to_px(p):
        return ax.transData.transform((getattr(p, x), getattr(p, y)))

    for p in _label_points(envelope, lambda q: to_px(q)[0], lambda q: to_px(q)[1], 60):
        ax.annotate(
            p.short(),
            (getattr(p, x), getattr(p, y)),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            color="#333",
            zorder=4,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_tick(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _fmt_tick(v)))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999")
    ax.tick_params(colors="#555", length=0, pad=6)
    ax.grid(color="#e9ecef", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(METRICS[x][0], color="#444", labelpad=8)
    ax.set_ylabel(METRICS[y][0], color="#444", labelpad=8)
    fig.suptitle(title, x=0.06, ha="left", fontsize=14, fontweight="bold", color="#111", y=0.975)
    ax.set_title(
        "Pareto frontier per configuration; faint dots are dominated configurations",
        loc="left",
        fontsize=9.5,
        color="#777",
        pad=10,
    )
    cfg = [Patch(color=c, label=g) for g, c in col.items()]
    leg1 = lax.legend(
        handles=cfg,
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        frameon=False,
        fontsize=8.5,
        title_fontproperties={"weight": "bold", "size": 9},
        alignment="left",
    )
    lax.add_artist(leg1)
    modes = [
        Line2D([], [], color="#555", lw=2, marker="o", ms=4.5, label="benchmark"),
        Line2D(
            [],
            [],
            color="#555",
            lw=2,
            ls=(0, (5, 3)),
            marker="o",
            ms=4.5,
            mfc="white",
            label="simulate",
        ),
    ]
    if meas:
        modes.append(
            Line2D([], [], color="#555", lw=0, marker="x", ms=7, mew=2, label="measured run")
        )
    lax.legend(
        handles=modes,
        title="Mode",
        loc="lower left",
        bbox_to_anchor=(0.0, 0.0),
        frameon=False,
        fontsize=8.5,
        title_fontproperties={"weight": "bold", "size": 9},
        alignment="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    return True


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>InferaSim Pareto</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--ink:#1d2330;--mute:#6b7280;--line:#e5e7eb;--bg:#f6f7f9;--card:#fff;--acc:#3b6fb6}
*{box-sizing:border-box}
body{font-family:Inter,"Segoe UI",system-ui,-apple-system,sans-serif;margin:0;color:var(--ink);background:var(--bg)}
header{padding:22px 32px 6px}
header h1{font-size:22px;margin:0 0 4px;font-weight:650;letter-spacing:-.01em}
header p{margin:0;color:var(--mute);font-size:13px}
main{padding:12px 32px 40px;max-width:1500px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;
box-shadow:0 1px 2px rgba(16,24,40,.04);margin-bottom:16px}
.ctl{display:flex;flex-wrap:wrap;gap:12px 18px;align-items:flex-end;position:relative;z-index:5}
.grp{display:flex;flex-direction:column;gap:5px}
.grp>span{font-size:11px;font-weight:600;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
.dd{position:relative}
.ddb{font:inherit;font-size:13px;color:var(--ink);min-width:140px;max-width:320px;display:flex;justify-content:space-between;
align-items:center;gap:12px;background:#fff;border:1px solid #d1d5db;border-radius:8px;padding:6px 10px;cursor:pointer;
line-height:1.3;text-align:left}
.ddb span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.ddb:hover,.dd.open .ddb{border-color:var(--acc)}
.ddb::after{content:"";flex:none;border:4.5px solid transparent;border-top-color:#6b7280;margin-top:4px}
.dd.open .ddb::after{border-top-color:transparent;border-bottom-color:var(--acc);margin:-4px 0 0}
.ddm{display:none;position:absolute;top:calc(100% + 4px);left:0;min-width:100%;max-height:340px;overflow:auto;background:#fff;
border:1px solid var(--line);border-radius:8px;box-shadow:0 10px 28px rgba(16,24,40,.14);padding:4px;z-index:20}
.dd.open .ddm{display:block}
.ddo{display:block;width:100%;text-align:left;font:inherit;font-size:13px;color:var(--ink);background:none;border:0;
border-radius:6px;padding:6px 10px;cursor:pointer;white-space:nowrap}
.ddo:hover{background:#f3f7fc}
.ddo.on{color:var(--acc);font-weight:600;background:#eef4fb}
.plot{display:flex;gap:18px;align-items:flex-start}
#chart{flex:1;min-width:0;width:100%;height:auto;display:block}
#legend{width:290px;flex:none;font-size:12.5px}
#legend h4{margin:2px 0 8px;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute)}
.lg{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:6px;cursor:pointer;user-select:none}
.lg:hover{background:#f3f4f6}.lg.off{opacity:.35}
.sw{width:12px;height:12px;border-radius:3px;flex:none}
.md{display:flex;align-items:center;gap:8px;padding:3px 6px;color:#444}
.hint{font-size:11.5px;color:var(--mute);margin-top:10px}
#tip{position:fixed;pointer-events:none;background:rgba(17,24,39,.94);color:#f9fafb;border-radius:8px;
padding:8px 10px;font-size:12px;line-height:1.45;display:none;box-shadow:0 6px 18px rgba(0,0,0,.25);white-space:pre;z-index:9}
#tip b{color:#fff}
.tw{overflow:auto;max-height:460px;border:1px solid var(--line);border-radius:10px}
table{border-collapse:separate;border-spacing:0;font-size:12px;width:100%}
td,th{padding:6px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid #f0f1f3}
th{background:#f9fafb;position:sticky;top:0;font-weight:600;color:#374151;border-bottom:1px solid var(--line)}
tr:hover td{background:#f8fafc}
td:first-child,th:first-child{text-align:left}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px;vertical-align:0}
.note{font-size:12.5px;color:var(--mute);line-height:1.55;max-width:1050px;margin:0}
#uncov{margin-top:6px}
h3{font-size:14px;margin:0 0 10px;font-weight:600}
circle.pt{transition:r .08s}
.mx{cursor:pointer}
circle.pt:hover{r:7}
</style></head><body>
<header><h1>InferaSim Pareto frontier</h1>
<p>Throughput versus interactivity across every parallelism, concurrency and deployment knob swept.</p></header>
<main>
<div class="card ctl">
<div class="grp"><span>Model</span><div class="dd" id="model"></div></div>
<div class="grp"><span>Scenario</span><div class="dd" id="scenario"></div></div>
<div class="grp"><span>Precision</span><div class="dd" id="precision"></div></div>
<div class="grp"><span>Mode</span><div class="dd" id="mode"></div></div>
<div class="grp"><span>Speculative</span><div class="dd" id="spec"></div></div>
<div class="grp"><span>Draft depth</span><div class="dd" id="depth"></div></div>
<div class="grp"><span>Points</span><div class="dd" id="show"></div></div>
<div class="grp" id="measgrp" style="display:none"><span>Measured runs</span><div class="dd" id="meas"></div></div>
<div class="grp"><span>X axis</span><div class="dd" id="x"></div></div>
<div class="grp"><span>Y axis</span><div class="dd" id="y"></div></div>
</div>
<div class="card">
<div class="plot"><svg id="chart" viewBox="0 0 1000 600" preserveAspectRatio="xMinYMin meet"></svg><div id="legend"></div></div>
<p class="note" style="margin-top:8px">Solid lines with filled markers are benchmark mode (calibrated from a
measured anchor of the same model and engine); dashed lines with hollow markers are simulate mode (analytical,
no GPU). Each line is the Pareto frontier of one GPU / engine / speculative setting / KV hierarchy; faint dots
are dominated configurations, and the shaded area sits under the best frontier overall. Hover a point for its
configuration; click a legend entry to hide or show it. Crosses are measured runs; the thin line from each cross
ends at the projection of that same configuration (benchmark mode, or simulate when benchmark has no anchor), so its
length is the error. A cross below a frontier is usually a dominated configuration, not a frontier miss.</p>
<div id="uncov" class="note"></div>
</div>
<div class="card"><h3>Frontier points</h3><div class="tw"><table id="tbl"></table></div></div>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const M = DATA.metrics, P = (() => {
  const C = DATA.points, ks = Object.keys(C.cols), out = new Array(C.n);
  for (let i = 0; i < C.n; i++) {
    const o = {};
    for (const k of ks) { const v = C.cols[k][i]; o[k] = k in C.dicts ? C.dicts[k][v] : v; }
    out[i] = o;
  }
  return out;
})();
const PAL = ["#3b6fb6","#e07b39","#2f9e6e","#c8414b","#7d5bb5","#1fa3b8","#d4a72c","#d1609e","#7a6a58","#5b8c2a","#9a9a9a","#264653"];
const $ = id => document.getElementById(id);
const hidden = new Set();
const V={};
const uniq = a => [...new Set(a)].sort();
const esc = s => String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function chips(id, vals, sel, lab){
  if(!vals.includes(sel)) sel=vals[0]; V[id]=sel;
  const L=v=>esc(lab?lab(v):v);
  $(id).innerHTML=`<button class="ddb" type="button"><span>${L(sel)}</span></button><div class="ddm">`+
    vals.map(v=>`<button class="ddo ${v===sel?'on':''}" type="button" data-v="${esc(v)}">${L(v)}</button>`).join('')+'</div>';
}
const closeMenus = () => document.querySelectorAll('.dd.open').forEach(d=>d.classList.remove('open'));
document.addEventListener('click',e=>{ if(!e.target.closest('.dd')) closeMenus(); });
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeMenus(); });
const short = k => M[k].label.replace(/\\s*\\(.*\\)$/,'');
function refreshDeps(){
  const m=V.model;
  chips('scenario', uniq(P.filter(p=>p.model===m).map(p=>p.scenario)), V.scenario);
  chips('precision', uniq(P.filter(p=>p.model===m&&p.scenario===V.scenario).map(p=>p.precision)), V.precision);
  const sp=uniq(P.filter(p=>p.model===m&&p.scenario===V.scenario&&p.precision===V.precision).map(p=>p.spec||'none'));
  chips('spec', ['all',...sp], V.spec||'all', v=>v==='all'?'all settings':v);
  const ks=[...new Set(P.filter(p=>p.model===m&&p.scenario===V.scenario&&p.precision===V.precision&&(p.spec||'none')!=='none'
    &&(V.spec==='all'||p.spec===V.spec)&&p.spec_k).map(p=>p.spec_k))].sort((a,b)=>a-b).map(String);
  chips('depth', ['best',...ks], V.depth||'best', v=>v==='best'?'best per point':'k = '+v);
}
chips('model', uniq(P.map(p=>p.model)), null);
chips('mode', ['all','benchmark','simulate'], 'all', v=>v==='all'?'benchmark + simulate':v);
chips('show', ['all','front'], 'all', v=>v==='all'?'all':'frontier only');
chips('x', Object.keys(M), DATA.x, short);
const MS = DATA.measured || [];
if(MS.length){ $('measgrp').style.display=''; chips('meas', ['show','hide'], 'show', v=>v==='show'?'show':'hide'); }
chips('y', Object.keys(M), DATA.y, short);
const series = p => p.series_label || `${p.gpu_arch.toUpperCase()} ${p.engine}${(p.spec||'none')==='none'?'':' '+p.spec}${p.kv_config==='hbm'?'':' KV:'+p.kv_config} (${p.mode})`;
const group = s => s.replace(/\\s*\\((benchmark|simulate)\\)$/,'');
function front(pts,x,y){
  const sx=M[x].higher_is_better?1:-1, sy=M[y].higher_is_better?1:-1;
  const s=[...pts].sort((a,b)=>(sx*b[x]-sx*a[x])||(sy*b[y]-sy*a[y]));
  const f=[]; let best=null;
  for(const p of s){ const v=sy*p[y]; if(best===null||v>best){f.push(p);best=v;} }
  return f.sort((a,b)=>a[x]-b[x]);
}
function ticks(hi,n=7){ if(hi<=0) return [0,1]; const raw=hi/(n-1), mag=Math.pow(10,Math.floor(Math.log10(raw)));
  const step=[1,2,2.5,5,10].map(k=>k*mag).find(v=>v>=raw); const t=[]; for(let i=0;i*step<=hi*(1+1e-9);i++) t.push(i*step); return t; }
function fmt(v){ if(Math.abs(v)>=1000) return (v/1000).toFixed(1).replace(/\\.0$/,'')+'k'; return v<10?(+v.toFixed(1)).toString():v.toFixed(0); }
function num(v,d=1){ return v.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:0}); }
function draw(){
  const m=V.model, sc=V.scenario, pr=V.precision, md=V.mode, sp=V.spec, dk=V.depth;
  const x=V.x, y=V.y, onlyF=V.show==='front';
  // A chosen depth narrows the speculative lines only; the no-speculation line stays as the reference.
  const atK = p => dk==='best'||(p.spec||'none')==='none'||String(p.spec_k)===dk;
  const sel=P.filter(p=>p.model===m&&p.scenario===sc&&p.precision===pr&&(md==='all'||p.mode===md)&&(sp==='all'||(p.spec||'none')===sp)&&atK(p));
  const all=sel.filter(p=>p.feasible);
  const groups=uniq(all.map(p=>group(series(p)))); const col={}; groups.forEach((g,i)=>col[g]=PAL[i%PAL.length]);
  const pts=all.filter(p=>!hidden.has(group(series(p))));
  const mgroup = q => `${'MI355X'} ${q.engine}${q.spec==='none'?'':' '+q.spec}`;
  const msel = V.meas==='show' ? MS.filter(q=>q.model===m&&q.scenario===sc&&q.precision===pr&&(sp==='all'||q.spec===sp)&&atK(q)
    &&!hidden.has(mgroup(q))&&q[x]!=null&&q[y]!=null) : [];
  const W=1000,H=600,L=70,R=24,T=20,B=56; let o=[];
  const x1=Math.max(...pts.map(p=>p[x]),...msel.map(q=>q[x]),1)*1.06, y1=Math.max(...pts.map(p=>p[y]),...msel.map(q=>q[y]),1)*1.08, xt=ticks(x1), yt=ticks(y1);
  const X=v=>L+v/x1*(W-L-R), Y=v=>H-B-v/y1*(H-T-B);
  const env=front(pts,x,y);
  if(env.length>1) o.push(`<polygon fill="#3b6fb6" fill-opacity=".06" points="${X(0)},${Y(0)} ${X(0)},${Y(env[0][y])} ${env.map(p=>X(p[x])+','+Y(p[y])).join(' ')} ${X(env[env.length-1][x])},${Y(0)}"/>`);
  xt.forEach(v=>o.push(`<line x1=${X(v)} y1=${T} x2=${X(v)} y2=${H-B} stroke="#eef0f3"/><text x=${X(v)} y=${H-B+20} font-size=12 fill="#6b7280" text-anchor=middle>${fmt(v)}</text>`));
  yt.forEach(v=>o.push(`<line x1=${L} y1=${Y(v)} x2=${W-R} y2=${Y(v)} stroke="#eef0f3"/><text x=${L-10} y=${Y(v)+4} font-size=12 fill="#6b7280" text-anchor=end>${fmt(v)}</text>`));
  o.push(`<line x1=${L} y1=${H-B} x2=${W-R} y2=${H-B} stroke="#9ca3af"/><line x1=${L} y1=${T} x2=${L} y2=${H-B} stroke="#9ca3af"/>`);
  o.push(`<text x=${(L+W-R)/2} y=${H-12} text-anchor=middle font-size=13 fill="#374151">${M[x].label}</text>`);
  o.push(`<text transform="translate(16,${(T+H-B)/2}) rotate(-90)" text-anchor=middle font-size=13 fill="#374151">${M[y].label}</text>`);
  const ser=uniq(pts.map(series)).sort((a,b)=>group(a).localeCompare(group(b))||a.localeCompare(b));
  const F={}; ser.forEach(s=>F[s]=front(pts.filter(p=>series(p)===s),x,y));
  const rows=[], dots=[], lines=[];
  ser.forEach(s=>{
    const c=col[group(s)], g=pts.filter(p=>series(p)===s), f=F[s], sim=s.endsWith('(simulate)');
    if(!onlyF) g.filter(p=>!f.includes(p)).forEach(p=>dots.push(`<circle class=pt cx=${X(p[x])} cy=${Y(p[y])} r=2.4 fill="${c}" fill-opacity=.2 data-i="${P.indexOf(p)}"/>`));
    if(f.length>1) lines.push(`<polyline fill=none stroke="${c}" stroke-width=2.3 stroke-linejoin=round ${sim?'stroke-dasharray="7,4"':''} points="${f.map(p=>X(p[x])+','+Y(p[y])).join(' ')}"/>`);
    f.forEach(p=>{lines.push(`<circle class=pt cx=${X(p[x])} cy=${Y(p[y])} r=4.2 fill="${sim?'#fff':c}" stroke="${c}" stroke-width=1.7 data-i="${P.indexOf(p)}"/>`); rows.push([s,p]);});
  });
  o.push(...dots, ...lines);
  const PK={interactivity:'interactivity',total_tput_per_gpu:'tput_per_gpu',ttft_ms:'ttft_ms'};
  msel.forEach(q=>{ const cx=X(q[x]), cy=Y(q[y]), c=col[mgroup(q)]||'#111', d=5;
    const pr=q.proj||{}; const pj=(md!=='simulate'&&pr.benchmark&&pr.benchmark.tput_per_gpu)?pr.benchmark:pr.simulate;
    if(pj&&PK[x]&&PK[y]&&pj[PK[x]]&&pj[PK[y]]) o.push(`<line x1=${cx} y1=${cy} x2=${X(pj[PK[x]])} y2=${Y(pj[PK[y]])} stroke="${c}" stroke-width=1 stroke-opacity=.55 pointer-events=none />`);
    o.push(`<g class=mx data-m="${MS.indexOf(q)}"><circle cx=${cx} cy=${cy} r=8 fill="#fff" fill-opacity=.01 data-m="${MS.indexOf(q)}"/>`+
      `<path d="M${cx-d},${cy-d}L${cx+d},${cy+d}M${cx-d},${cy+d}L${cx+d},${cy-d}" stroke="#fff" stroke-width=4.5 data-m="${MS.indexOf(q)}"/>`+
      `<path d="M${cx-d},${cy-d}L${cx+d},${cy+d}M${cx-d},${cy+d}L${cx+d},${cy-d}" stroke="${c}" stroke-width=2.2 data-m="${MS.indexOf(q)}"/></g>`); });
  const lab=[]; env.forEach(p=>{ if(lab.every(q=>Math.hypot(X(p[x])-X(q[x]),Y(p[y])-Y(q[y]))>=50)) lab.push(p); });
  lab.forEach(p=>o.push(`<text x=${X(p[x])+7} y=${Y(p[y])-8} font-size=11 fill="#1f2937" stroke="#fff" stroke-width=3 paint-order=stroke pointer-events=none>${p.parallelism} c${p.concurrency}${(p.spec||'none')!=='none'&&p.spec_k?' k'+p.spec_k:''}</text>`));
  $('chart').innerHTML=o.join('');
  let lg='<h4>Configuration</h4>'+groups.map(g=>`<div class="lg ${hidden.has(g)?'off':''}" data-g="${g}"><span class=sw style="background:${col[g]}"></span>${g}</div>`).join('');
  lg+='<h4 style="margin-top:16px">Mode</h4>'+
    '<div class=md><svg width=34 height=10><line x1=0 y1=5 x2=34 y2=5 stroke="#555" stroke-width=2.2 /><circle cx=17 cy=5 r=3.8 fill="#555"/></svg>benchmark</div>'+
    '<div class=md><svg width=34 height=10><line x1=0 y1=5 x2=34 y2=5 stroke="#555" stroke-width=2.2 stroke-dasharray="7,4"/><circle cx=17 cy=5 r=3.8 fill="#fff" stroke="#555" stroke-width=1.6 /></svg>simulate</div>'+
    (MS.length?'<div class=md><svg width=34 height=12><path d="M12,1L22,11M12,11L22,1" stroke="#555" stroke-width=2.2 /></svg>measured run</div>':'')+
    `<div class=hint>${pts.length} configurations plotted, ${rows.length} on a frontier${msel.length?`, ${msel.length} measured`:''}.</div>`;
  $('legend').innerHTML=lg;
  document.querySelectorAll('.lg').forEach(el=>el.onclick=()=>{const g=el.dataset.g; hidden.has(g)?hidden.delete(g):hidden.add(g); draw();});
  const un=sel.filter(p=>!p.feasible);
  $('uncov').textContent = un.length ? `${un.length} configuration(s) not plotted (did not fit, or no anchor covered them in benchmark mode).` : '';
  const cols=[['parallelism','parallelism'],['concurrency','conc'],['gpus','GPUs'],['mode','mode'],['kv_config','KV'],['spec','spec'],['spec_k','k'],['spec_tokens_per_step','tok/step'],['knobs','knobs'],['interactivity','tok/s/user'],['total_tput_per_gpu','tput/GPU'],['output_tput_per_gpu','out tput/GPU'],['ttft_ms','TTFT ms'],['tpot_ms','TPOT ms'],['prefix_hit_rate','prefix hit']];
  const cell=v=>typeof v==='number'?(Number.isInteger(v)?v:num(v,2)):(v??'');
  $('tbl').innerHTML='<tr><th>series</th>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr>'+
    rows.map(([s,p])=>`<tr><td><span class=dot style="background:${col[group(s)]}"></span>${s}</td>`+cols.map(c=>`<td>${cell(p[c[0]])}</td>`).join('')+'</tr>').join('');
}
$('chart').addEventListener('mousemove',e=>{
  const t=$('tip'); const mi=e.target.getAttribute&&e.target.getAttribute('data-m');
  if(mi!==null&&mi!==undefined){ const q=MS[+mi]; let h=`<b>measured · ${q.parallelism} · conc ${q.concurrency}</b>\n${q.model} ${q.engine} ${q.precision}${q.spec==='none'?'':' '+q.spec+(q.spec_k?' k'+q.spec_k:'')} · median of ${q.runs} run(s)\n`+
      `interactivity ${num(q.interactivity)} tok/s/user\nthroughput/GPU ${num(q.total_tput_per_gpu,0)}\nTTFT ${num(q.ttft_ms,0)} ms`;
    t.innerHTML=h; t.style.display='block'; const r=t.getBoundingClientRect(); let lx=e.clientX+14, ly=e.clientY+12;
    if(lx+r.width>innerWidth-8) lx=e.clientX-r.width-14; if(ly+r.height>innerHeight-8) ly=e.clientY-r.height-12;
    t.style.left=lx+'px'; t.style.top=ly+'px'; return; }
  const i=e.target.getAttribute&&e.target.getAttribute('data-i');
  if(i===null||i===undefined){t.style.display='none';return;}
  const p=Object.fromEntries(Object.entries(P[+i]).map(([k,v])=>[k,typeof v==='string'?v.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])):v]));
  t.innerHTML=`<b>${p.parallelism} · conc ${p.concurrency} · ${p.gpus} GPU</b>\\n${p.model} ${p.engine} ${p.precision} (${p.mode})\\n`+
   `interactivity ${num(p.interactivity)} tok/s/user\\nthroughput/GPU ${num(p.total_tput_per_gpu,0)} (output ${num(p.output_tput_per_gpu,0)})\\n`+
   `TTFT ${num(p.ttft_ms,0)} ms · TPOT ${num(p.tpot_ms,2)} ms`+(p.anchor?`\\nanchor ${p.anchor.split('/').pop()}`:'')+(p.kv_tiers?`\\nKV tiers ${p.kv_tiers}`:'')+
   ((p.spec||'none')!=='none'?`\\nspeculative ${p.spec} k=${p.spec_k}, ${(p.spec_tokens_per_step||1).toFixed(2)} tok/step`+
     (p.spec_al_source?`\\n  acceptance ${p.spec_al_source}`:'')+(p.spec_draft_cost?`\\n  draft ${p.spec_draft_cost}${p.spec_draft_ms?' = '+num(p.spec_draft_ms,2)+' ms/step':''}`:''):'')+
   (p.knobs?`\\n${p.knobs}`:'');
  t.style.display='block';
  const r=t.getBoundingClientRect(); let lx=e.clientX+14, ly=e.clientY+12;
  if(lx+r.width>innerWidth-8) lx=e.clientX-r.width-14; if(ly+r.height>innerHeight-8) ly=e.clientY-r.height-12;
  t.style.left=lx+'px'; t.style.top=ly+'px';
});
$('chart').addEventListener('mouseleave',()=>$('tip').style.display='none');
['model','scenario','precision','mode','spec','depth','x','y','show','meas'].forEach(id=>$(id).addEventListener('click',e=>{
  if(e.target.closest('.ddb')){ const open=$(id).classList.contains('open'); closeMenus(); if(!open) $(id).classList.add('open'); return; }
  const b=e.target.closest('.ddo'); if(!b) return;
  closeMenus();
  V[id]=b.dataset.v; $(id).querySelector('.ddb span').textContent=b.textContent;
  $(id).querySelectorAll('.ddo').forEach(c=>c.classList.toggle('on',c===b));
  if(['model','scenario','precision','spec'].includes(id)){ if(id==='model') hidden.clear(); refreshDeps(); }
  draw();
}));
refreshDeps(); draw();
</script></body></html>
"""


_HTML_INFEASIBLE = ("model", "scenario", "precision", "mode", "spec", "feasible", "reason")


def _compact(v: Any) -> Any:
    """Five significant digits keep the embedded page small; the CSV and JSON keep full precision."""
    return float(f"{v:.5g}") if isinstance(v, float) else v


def _columns(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Rows as one array per field, non-numeric fields as indices into a table of distinct values.

    A sweep of ~200k points embedded row by row is >100 MB of repeated keys and
    strings, more than a browser will parse inline; the page rebuilds the rows.
    """
    keys = sorted({k for r in rows for k in r})
    cols: dict[str, list[Any]] = {}
    dicts: dict[str, list[Any]] = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if any(isinstance(v, (str, list, dict)) for v in vals):
            seen: dict[str, int] = {}
            table: list[Any] = []
            idx = []
            for v in vals:
                s = json.dumps(v, sort_keys=True)
                if s not in seen:
                    seen[s] = len(table)
                    table.append(v)
                idx.append(seen[s])
            cols[k], dicts[k] = idx, table
        else:
            cols[k] = vals
    return dict(n=len(rows), cols=cols, dicts=dicts)


def write_html(
    result: ParetoResult,
    path: str,
    measured: Sequence[dict[str, Any]] | None = None,
) -> None:
    """The dashboard; ``measured`` runs are drawn over the frontiers."""
    data = dict(
        measured=list(measured or []),
        x=result.x,
        y=result.y,
        metrics={k: dict(label=v[0], higher_is_better=v[1]) for k, v in METRICS.items()},
        # The page only counts the points it cannot plot, so those carry just
        # the fields its filters read; the CSV and JSON keep every column.
        points=_columns(
            [
                {k: _compact(v) for k, v in asdict(p).items()}
                if p.feasible
                else {k: getattr(p, k) for k in _HTML_INFEASIBLE}
                for p in result.points
            ]
        ),
    )
    blob = json.dumps(data).replace("</", "<\\/")
    with open(path, "w") as f:
        f.write(_HTML.replace("__DATA__", blob))


def load_measured(path: str | None) -> dict[str, list[dict[str, Any]]]:
    """Measured runs to draw over the frontiers.

    A JSON object with ``measured``: one record per measured configuration
    (model, scenario, precision, engine, spec, parallelism, concurrency, the
    metric values, and optionally ``proj`` -- the projection of that same
    configuration per mode).
    """
    if not path:
        return {}
    with open(path) as f:
        doc = json.load(f)
    return {"measured": list(doc.get("measured") or [])}


def write_outputs(
    result: ParetoResult,
    out_dir: str,
    meta: dict[str, Any],
    measured: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, list[str]]:
    measured = measured or {}
    runs = measured.get("measured") or []
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, list[str]] = {"csv": [], "json": [], "html": [], "svg": [], "png": []}
    p = os.path.join(out_dir, "pareto_points.csv")
    write_csv(result.points, p)
    written["csv"].append(p)
    p = os.path.join(out_dir, "pareto_points.json")
    write_json(result, p, meta)
    written["json"].append(p)
    p = os.path.join(out_dir, "pareto.html")
    write_html(result, p, runs)
    written["html"].append(p)
    charts: dict[tuple, list[ParetoPoint]] = {}
    for pt in result.points:
        charts.setdefault(pt.chart(), []).append(pt)
    for (model, scen, prec), pts in sorted(charts.items()):
        title = f"{model} | {scen} | {prec}"
        stem = os.path.join(out_dir, "pareto_" + _slug(model, scen, prec))
        write_svg(pts, stem + ".svg", result.x, result.y, title)
        written["svg"].append(stem + ".svg")
        mine = [
            q
            for q in runs
            if (q.get("model"), q.get("scenario"), q.get("precision")) == (model, scen, prec)
        ]
        if write_png(pts, stem + ".png", result.x, result.y, title, mine):
            written["png"].append(stem + ".png")
    return written


# -- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="inferasim pareto",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Model preset(s), optionally PRESET@CHECKPOINT so benchmark mode can "
        "match anchors harvested from that checkpoint (e.g. glm5@GLM-5.2-MXFP4).",
    )
    ap.add_argument("--engine", nargs="+", default=["sglang"], help="Serving engine(s).")
    ap.add_argument(
        "--precision", nargs="+", default=["fp8"], choices=sorted(PRECISIONS), help="Precision(s)."
    )
    ap.add_argument(
        "--parallelism",
        nargs="+",
        default=["tp1", "tp2", "tp4", "tp8", "tp8ep8", "tp8ep8dp8"],
        help="Parallelism configs: tpN[epN][dpN][ppN]; dp is attention data parallel.",
    )
    ap.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    ap.add_argument(
        "--scenario",
        nargs="+",
        default=["8k/1k"],
        help="ISL/OSL (e.g. 1k/1k, 8k/1k) or trace:PATH for a Mooncake/AgentX trace.",
    )
    ap.add_argument(
        "--mode",
        nargs="+",
        default=["benchmark", "simulate"],
        choices=["benchmark", "simulate"],
    )
    ap.add_argument(
        "--anchors",
        nargs="*",
        default=[],
        help="Anchor artifacts for benchmark mode: files, globs or directories.",
    )
    ap.add_argument(
        "--kv-config",
        nargs="+",
        default=["hbm"],
        help="KV memory hierarchies to sweep, each its own series: 'hbm', or "
        "LABEL=TIER[,TIER...] in --kv-tier syntax, fastest first "
        "(e.g. dram=dram:512 dram+ssd=dram:512,ssd:4000).",
    )
    ap.add_argument(
        "--spec",
        nargs="+",
        default=["none"],
        help="Speculative decoding settings, each its own series: 'none' or "
        "METHOD[:K[:ACCEPT[:DRAFT_COST]]] with METHOD in "
        f"{'/'.join(SPEC_DEFAULTS)} (e.g. none mtp eagle3:4 mtp:1-8 mtp:3:0.8). "
        "K is the draft depth (default "
        + ", ".join(f"{m}={k}" for m, k in SPEC_DEFAULTS.items())
        + "), or a range swept as one line ('mtp:golden' = every depth the "
        "curves cover). ACCEPT left out reads the measured acceptance-length "
        "files for the model (own curve, sibling head's, else the method "
        "default); a number states a per-draft-token rate. DRAFT_COST left out "
        "prices the drafter from its layers; a number is a flat cost per draft "
        "token as a fraction of a target decode step.",
    )
    ap.add_argument(
        "--golden-thinking",
        choices=("on", "off"),
        default="on",
        help="Which thinking-mode sweep of the acceptance-length files to read "
        "(InferenceX's default is on)",
    )
    knobs = ap.add_argument_group(
        "deployment knobs",
        "Each takes a list of values; the sweep crosses them with everything else "
        "and, unless --series-by names it, the frontier picks the best value the "
        "way it picks a parallelism.",
    )
    knobs.add_argument("--sched", nargs="+", default=None, help=KNOBS["sched"][0] + ".")
    knobs.add_argument("--step-budget", nargs="+", default=None, help=KNOBS["budget"][0] + ".")
    knobs.add_argument("--chunked-prefill", nargs="+", default=None, help=KNOBS["chunk"][0] + ".")
    knobs.add_argument("--kv-dtype", nargs="+", default=None, help=KNOBS["kvdtype"][0] + ".")
    knobs.add_argument("--mem-fraction", nargs="+", default=None, help=KNOBS["memfrac"][0] + ".")
    knobs.add_argument("--max-num-seqs", nargs="+", default=None, help=KNOBS["maxseqs"][0] + ".")
    knobs.add_argument("--think-ms", nargs="+", default=None, help=KNOBS["think"][0] + ".")
    knobs.add_argument("--kv-block-size", nargs="+", default=None, help=KNOBS["block"][0] + ".")
    knobs.add_argument(
        "--hbm-capacity-gb",
        nargs="+",
        default=None,
        help="HBM capacity per GPU in GB (default 288); several values sweep it.",
    )
    knobs.add_argument(
        "--knob",
        action="append",
        default=[],
        metavar="NAME=V1[,V2...]",
        help="Any other knob: a name above (" + ", ".join(KNOBS) + ") or a raw "
        "projection --flag (on/off for a switch), e.g. "
        "--knob=--prefix-cache-hit-rate=0,0.5. Repeatable.",
    )
    knobs.add_argument(
        "--series-by",
        nargs="+",
        default=["spec", "kv"],
        help="Axes that get their own line instead of being pooled into one "
        "frontier: spec, kv, or a knob name (e.g. sched budget).",
    )
    ap.add_argument("--max-anchor-tp", type=int, default=4)
    ap.add_argument("--gpu-arch", default="mi355x")
    ap.add_argument("--max-gpus", type=int, default=8, help="Skip configs wider than this.")
    ap.add_argument("--x", default="interactivity", choices=sorted(METRICS))
    ap.add_argument("--y", default="total_tput_per_gpu", choices=sorted(METRICS))
    ap.add_argument(
        "--requests-per-client",
        type=int,
        default=10,
        help="requests per client; InferenceX's harness sends 10 x concurrency",
    )
    ap.add_argument("--min-requests", type=int, default=64)
    ap.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="leading completions left out of the metrics. 0 is the benchmark "
        "harness's convention, which averages over every request including the "
        "opening burst where all clients prefill at once; pass the concurrency "
        "for steady-state TTFT",
    )
    ap.add_argument(
        "--trace-max-requests",
        type=int,
        default=0,
        help="Replay only the first N rows of a trace (0 = all). Lane-interleaved "
        "traces put every conversation's first turn first, so a short head has "
        "no reuse in it.",
    )
    ap.add_argument("--gpu-cost-per-hour", type=float, default=0.0)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out-dir", default="pareto_out")
    ap.add_argument(
        "--measured",
        default=None,
        help="JSON of measured runs ({'measured': [...]}) to draw over the "
        "frontiers as crosses, each joined to the projection of the same configuration",
    )
    ap.add_argument(
        "--append",
        action="store_true",
        help="Merge into the points already in --out-dir (a re-projected point "
        "replaces its old row), so several sweeps -- one per model, say -- "
        "build one dashboard.",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip points already in --out-dir (merged or streamed out by a run that "
        "was killed), so a restarted sweep only projects what is missing.",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="List the points that would be projected and exit."
    )
    ap.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="After '--': extra flags passed to every projection (e.g. KV tier flags).",
    )
    return ap


_KNOB_OPTIONS = (
    ("sched", "sched"),
    ("step_budget", "budget"),
    ("chunked_prefill", "chunk"),
    ("kv_dtype", "kvdtype"),
    ("mem_fraction", "memfrac"),
    ("max_num_seqs", "maxseqs"),
    ("think_ms", "think"),
    ("kv_block_size", "block"),
)


def knob_axes_from_args(a: argparse.Namespace) -> tuple[list[KnobAxis], float]:
    """The swept knobs, and the single HBM size when it is not one of them."""
    axes = [
        KnobAxis.parse(f"{name}={','.join(getattr(a, attr))}")
        for attr, name in _KNOB_OPTIONS
        if getattr(a, attr)
    ]
    hbm = [str(float(v)) for v in (a.hbm_capacity_gb or [])]
    if len(hbm) > 1:
        axes.append(KnobAxis("hbm", tuple(hbm)))
    axes += [KnobAxis.parse(k) for k in a.knob]
    names = [x.name for x in axes]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"[inferasim:pareto] knob(s) given twice: {sorted(dup)}")
    return axes, float(hbm[0]) if len(hbm) == 1 else 288.0


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    extra = [t for t in (a.extra or []) if t != "--"]
    models = [ModelSpec.parse(m) for m in a.model]
    pars = [Parallelism.parse(p) for p in a.parallelism]
    scen = [Scenario.parse(s, a.trace_max_requests) for s in a.scenario]
    specs = list(dict.fromkeys(c for s in a.spec for c in SpecConfig.parse_many(s)))
    axes, hbm_gb = knob_axes_from_args(a)
    anchors = [m for m in (_anchor_meta(p) for p in _anchor_files(a.anchors)) if m]
    if "benchmark" in a.mode and not anchors:
        print(
            "[inferasim:pareto] benchmark mode requested but no anchors given (--anchors); "
            "benchmark points will be reported uncovered."
        )
    tasks, uncovered = build_tasks(
        models=models,
        engines=a.engine,
        precisions=a.precision,
        parallelisms=pars,
        concurrencies=a.concurrency,
        scenarios=scen,
        modes=a.mode,
        anchors=anchors,
        kv_configs=[KvConfig.parse(k) for k in a.kv_config],
        gpu_arch=a.gpu_arch,
        hbm_gb=hbm_gb,
        max_gpus=a.max_gpus,
        max_anchor_tp=a.max_anchor_tp,
        requests_per_client=a.requests_per_client,
        min_requests=a.min_requests,
        warmup_requests=a.warmup_requests,
        trace_max_requests=a.trace_max_requests,
        workdir=os.path.join(a.out_dir, "traces"),
        gpu_cost_per_hour=a.gpu_cost_per_hour,
        specs=specs,
        knob_axes=axes,
        series_by=a.series_by,
        extra=extra,
        golden_thinking=a.golden_thinking,
    )
    print(
        f"[inferasim:pareto] {len(tasks)} point(s) to project, "
        f"{len(uncovered)} benchmark point(s) without an anchor"
    )
    if a.dry_run:
        for t in tasks:
            b = t["point"]
            print(
                f"  {b['model']} {b['engine']} {b['precision']} {b['mode']} "
                f"{b['parallelism']} c{b['concurrency']} {b['scenario']} {b['spec']} "
                f"{b['knobs']} {os.path.basename(b.get('anchor', '') or '')}"
            )
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    final = os.path.join(a.out_dir, "pareto_points.json")
    partial = os.path.join(a.out_dir, "pareto_points.partial.jsonl")
    # Points a killed run streamed out but never merged.
    keep_old = a.append or a.resume
    salvaged = merge_points(load_points(final) if keep_old else [], load_partial(partial))
    if a.resume:
        done = {_key(p) for p in salvaged if not p.reason.startswith("worker failed")}
        before = len(tasks)
        tasks = [t for t in tasks if _key(ParetoPoint(**t["point"])) not in done]
        print(f"[inferasim:pareto] --resume: {before - len(tasks)} point(s) already done")
        if not tasks and os.path.exists(final) and not os.path.exists(partial):
            return 0
    t0 = time.perf_counter()
    pts = run(tasks, jobs=a.jobs, sink=partial) + uncovered
    pts = merge_points(salvaged, pts)
    res = ParetoResult(points=pts, elapsed_s=time.perf_counter() - t0, x=a.x, y=a.y)
    mark_frontiers(res.points, a.x, a.y)
    meta = dict(argv=list(argv) if argv is not None else sys.argv[1:], n_tasks=len(tasks))
    written = write_outputs(res, a.out_dir, meta, load_measured(a.measured))
    with contextlib.suppress(FileNotFoundError):
        os.remove(partial)
    n_ok = sum(p.feasible for p in res.points)
    print(f"[inferasim:pareto] {n_ok} feasible of {len(res.points)} in {res.elapsed_s:.1f} s")
    for kind, paths in written.items():
        for p in paths:
            print(f"  {kind:<4} {p}")
    if not written["png"]:
        print("  (no PNG: matplotlib is not installed; the SVG and HTML carry the same charts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
