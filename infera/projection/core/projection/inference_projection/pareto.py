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
import os
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
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

_PAR_RE = re.compile(r"^tp(\d+)(?:ep(\d+))?(?:dp(\d+))?(?:pp(\d+))?$")


@dataclass(frozen=True)
class Parallelism:
    tp: int
    ep: int = 1
    dp_attn: int = 1
    pp: int = 1

    @classmethod
    def parse(cls, spec: str) -> Parallelism:
        """``tp8``, ``tp4ep4``, ``tp8ep8dp8`` (attention DP), ``tp4pp2``."""
        m = _PAR_RE.match(spec.strip().lower())
        if not m:
            raise ValueError(f"bad parallelism '{spec}' (expected e.g. tp8, tp8ep8, tp8ep8dp8)")
        tp, ep, dp, pp = (int(g) if g else 1 for g in m.groups())
        if dp > tp:
            raise ValueError(f"'{spec}': attention DP cannot exceed TP")
        return cls(tp=tp, ep=ep, dp_attn=dp, pp=pp)

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
        if self.pp > 1:
            s += f" PP{self.pp}"
        return s


@dataclass(frozen=True)
class ModelSpec:
    """A preset plus the checkpoint it is served as.

    ``glm5@GLM-5.2-MXFP4`` projects the ``glm5`` preset and matches anchors
    harvested from a checkpoint named ``GLM-5.2-MXFP4``. Without ``@`` the
    preset name is both.
    """

    preset: str
    checkpoint: str

    @classmethod
    def parse(cls, spec: str) -> ModelSpec:
        preset, _, ckpt = spec.partition("@")
        return cls(preset=preset.strip(), checkpoint=(ckpt or preset).strip())


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


# method -> (draft depth, per-draft-token acceptance, draft cost per proposed
# token as a fraction of one target decode step). The depth is InferenceX's
# recipes' (``speculative-num-steps 3``); an MTP head is one extra layer plus
# the LM head, so its draft costs a few percent of a 60-90 layer target.
SPEC_DEFAULTS: dict[str, tuple[int, float, float]] = {
    "mtp": (3, 0.75, 0.05),
    "nextn": (3, 0.75, 0.05),
    "eagle": (3, 0.70, 0.08),
    "eagle3": (3, 0.75, 0.08),
    "draft": (4, 0.65, 0.15),
    "ngram": (4, 0.40, 0.0),
}


@dataclass(frozen=True)
class SpecConfig:
    """One speculative-decoding setting: ``none`` or ``METHOD[:K[:ACCEPT[:DRAFT_COST]]]``."""

    method: str = "none"
    k: int = 0
    accept: float = 0.0
    draft_cost: float = 0.0

    @classmethod
    def parse(cls, spec: str) -> SpecConfig:
        parts = [p.strip() for p in spec.strip().lower().split(":")]
        if parts[0] in ("", "none", "off", "0"):
            return cls()
        if parts[0] not in SPEC_DEFAULTS:
            raise ValueError(f"unknown speculative method {parts[0]!r} in {spec!r}")
        k, acc, dc = SPEC_DEFAULTS[parts[0]]
        if len(parts) > 1 and parts[1]:
            k = int(parts[1])
        if len(parts) > 2 and parts[2]:
            acc = float(parts[2])
        if len(parts) > 3 and parts[3]:
            dc = float(parts[3])
        if k < 1 or not 0.0 <= acc <= 1.0 or dc < 0.0:
            raise ValueError(f"bad speculative setting {spec!r}")
        return cls(method=parts[0], k=k, accept=acc, draft_cost=dc)

    @property
    def label(self) -> str:
        if self.method == "none":
            return "none"
        _, acc, dc = SPEC_DEFAULTS[self.method]
        s = f"{self.method.upper()}{self.k}"
        if abs(self.accept - acc) > 1e-9:
            s += f"@{self.accept:g}"
        if abs(self.draft_cost - dc) > 1e-9:
            s += f"/d{self.draft_cost:g}"
        return s

    @property
    def tokens_per_step(self) -> float:
        """Expected tokens a verify step keeps: the bonus token plus the accepted run."""
        return sum(self.accept**i for i in range(self.k + 1)) if self.k else 1.0

    def argv(self) -> list[str]:
        if self.method == "none":
            return []
        return [
            "--speculative-num-tokens",
            str(self.k),
            "--speculative-acceptance-rate",
            str(self.accept),
            "--speculative-draft-cost-factor",
            str(self.draft_cost),
        ]


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
    spec: str = "none"
    spec_k: int = 0
    spec_accept: float = 0.0
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
    on_frontier: bool = False

    def series(self) -> str:
        if self.series_label:
            return self.series_label
        kv = "" if self.kv_config == "hbm" else f" KV:{self.kv_config}"
        sp = "" if self.spec == "none" else f" {self.spec}"
        return f"{self.gpu_arch.upper()} {self.engine}{sp}{kv} ({self.mode})"

    def chart(self) -> tuple[str, str, str]:
        return (self.model, self.scenario, self.precision)


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
    return dict(
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
) -> dict[str, Any] | None:
    """The anchor a benchmark-mode point is calibrated from, or None.

    Same checkpoint and same engine are requirements: no transport converts one
    engine's kernels into another's. Among those, width follows the benchmark
    policy -- a target at TP1/TP2 uses an anchor at its own width, anything
    wider restores from TP4 -- then agreement on whether experts are sharded,
    then prompt length, then the most recent harvest.
    """
    want_tp = par.tp if par.tp <= 2 else min(4, max_anchor_tp)
    # An anchor measures one checkpoint's kernels, and a checkpoint is one
    # precision. When its name says which, a point at another is not covered.
    # Otherwise the anchor's recorded weight dtype decides; one that recorded
    # none (or only "quark") is taken at the checkpoint's word.
    named = _named_precision(checkpoint)
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
        return (
            abs(a["tp"] - want_tp),
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
        str(concurrency),
        "--des-warmup-frac",
        "0",
    ]
    budget = int(float(knob.get("budget") or 0)) or ENGINE_STEP_BUDGET.get(engine.lower(), 8192)
    if par.dp_attn > 1:
        argv += ["--attention-dp-size", str(par.dp_attn)]
        # Each attention-DP rank schedules its own share of the step budget.
        budget = max(1, budget // par.dp_attn)
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
    requests_per_client: int = 4,
    min_requests: int = 64,
    trace_max_requests: int = 0,
    workdir: str = "pareto_out/traces",
    gpu_cost_per_hour: float = 0.0,
    specs: Sequence[SpecConfig] = (SpecConfig(),),
    knob_axes: Sequence[KnobAxis] = (),
    series_by: Sequence[str] = ("spec", "kv"),
    extra: Sequence[str] = (),
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
            gpus=par.gpus,
            concurrency=conc,
            spec=spec.label,
            spec_k=spec.k,
            spec_accept=spec.accept,
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
            anchor = a["path"]
            base["anchor"] = anchor
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
    "knobs",
    "tp",
    "ep",
    "dp_attn",
    "pp",
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
    "#e4572e",
    "#17bebb",
    "#ffc914",
    "#76b041",
    "#2e86ab",
    "#a23b72",
    "#f18f01",
    "#6c757d",
    "#3b1f2b",
    "#8338ec",
]


def _slug(*parts: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", "_".join(parts)).strip("_")


def write_svg(points: Sequence[ParetoPoint], path: str, x: str, y: str, title: str) -> None:
    """A static frontier chart with no plotting dependency."""
    W, H, L, R, T, B = 900, 560, 80, 220, 50, 60
    feas = [p for p in points if p.feasible]
    if not feas:
        return
    xs = [getattr(p, x) for p in feas]
    ys = [getattr(p, y) for p in feas]
    x0, x1 = 0.0, max(xs) * 1.05 or 1.0
    y0, y1 = 0.0, max(ys) * 1.08 or 1.0

    def px(v):
        return L + (v - x0) / (x1 - x0) * (W - L - R)

    def py(v):
        return H - B - (v - y0) / (y1 - y0) * (H - T - B)

    series = sorted({p.series() for p in feas})
    col = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(series)}
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="white"/>',
        f'<text x="{L}" y="28" font-size="15" font-weight="bold">{html.escape(title)}</text>',
    ]
    for i in range(6):
        gx = x0 + (x1 - x0) * i / 5
        gy = y0 + (y1 - y0) * i / 5
        o.append(f'<line x1="{px(gx):.1f}" y1="{T}" x2="{px(gx):.1f}" y2="{H - B}" stroke="#eee"/>')
        o.append(f'<line x1="{L}" y1="{py(gy):.1f}" x2="{W - R}" y2="{py(gy):.1f}" stroke="#eee"/>')
        o.append(f'<text x="{px(gx):.1f}" y="{H - B + 16}" text-anchor="middle">{gx:.0f}</text>')
        o.append(f'<text x="{L - 6}" y="{py(gy) + 4:.1f}" text-anchor="end">{gy:.0f}</text>')
    o.append(
        f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="none" stroke="#888"/>'
    )
    o.append(
        f'<text x="{(L + W - R) / 2}" y="{H - 18}" text-anchor="middle">{html.escape(METRICS[x][0])}</text>'
    )
    o.append(
        f'<text transform="translate(18,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle">'
        f"{html.escape(METRICS[y][0])}</text>"
    )
    for si, s in enumerate(series):
        grp = [p for p in feas if p.series() == s]
        front = pareto_front(grp, x, y)
        c = col[s]
        dash = ' stroke-dasharray="6,4"' if "(simulate)" in s else ""
        if len(front) > 1:
            d = " ".join(f"{px(getattr(p, x)):.1f},{py(getattr(p, y)):.1f}" for p in front)
            o.append(f'<polyline points="{d}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        for p in grp:
            on = p in front
            o.append(
                f'<circle cx="{px(getattr(p, x)):.1f}" cy="{py(getattr(p, y)):.1f}" '
                f'r="{4 if on else 2.5}" fill="{c}" fill-opacity="{1 if on else 0.3}">'
                f"<title>{html.escape(f'{p.parallelism} c{p.concurrency}')}</title></circle>"
            )
        for p in front:
            o.append(
                f'<text x="{px(getattr(p, x)) + 5:.1f}" y="{py(getattr(p, y)) - 5:.1f}" '
                f'font-size="9" fill="{c}">{html.escape(f"{p.parallelism} c{p.concurrency}")}</text>'
            )
        ly = T + 10 + si * 18
        o.append(
            f'<line x1="{W - R + 15}" y1="{ly}" x2="{W - R + 40}" y2="{ly}" stroke="{c}" stroke-width="2"{dash}/>'
        )
        o.append(f'<text x="{W - R + 46}" y="{ly + 4}">{html.escape(s)}</text>')
    o.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(o))


def write_png(points: Sequence[ParetoPoint], path: str, x: str, y: str, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    feas = [p for p in points if p.feasible]
    if not feas:
        return False
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=130)
    for i, s in enumerate(sorted({p.series() for p in feas})):
        grp = [p for p in feas if p.series() == s]
        c = _PALETTE[i % len(_PALETTE)]
        ls = "--" if "(simulate)" in s else "-"
        ax.scatter(
            [getattr(p, x) for p in grp], [getattr(p, y) for p in grp], s=10, color=c, alpha=0.3
        )
        front = pareto_front(grp, x, y)
        ax.plot(
            [getattr(p, x) for p in front],
            [getattr(p, y) for p in front],
            ls,
            marker="o",
            ms=4,
            color=c,
            label=s,
        )
        for p in front:
            ax.annotate(
                f"{p.parallelism} c{p.concurrency}",
                (getattr(p, x), getattr(p, y)),
                fontsize=6,
                color=c,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel(METRICS[x][0])
    ax.set_ylabel(METRICS[y][0])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>InferaSim Pareto</title>
<style>
body{font-family:system-ui,sans-serif;margin:18px;color:#222}
.ctl{display:flex;flex-wrap:wrap;gap:14px;align-items:end;margin-bottom:10px}
.ctl label{display:flex;flex-direction:column;font-size:12px;color:#555}
select{font-size:13px;padding:3px}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #999;padding:6px 8px;
font-size:12px;display:none;box-shadow:0 2px 6px rgba(0,0,0,.2);white-space:pre}
table{border-collapse:collapse;font-size:12px;margin-top:14px}
td,th{border:1px solid #ddd;padding:3px 7px;text-align:right}
th{background:#f4f4f4} td:first-child,th:first-child{text-align:left}
.note{font-size:12px;color:#666;max-width:1000px}
</style></head><body>
<h2>InferaSim Pareto frontier</h2>
<div class="ctl">
<label>Model<select id="model"></select></label>
<label>Scenario<select id="scenario"></select></label>
<label>Precision<select id="precision"></select></label>
<label>Mode<select id="mode"><option value="all">benchmark + simulate</option>
<option value="benchmark">benchmark</option><option value="simulate">simulate</option></select></label>
<label>Speculative<select id="spec"></select></label>
<label>X axis<select id="x"></select></label>
<label>Y axis<select id="y"></select></label>
<label>Points<select id="show"><option value="all">all</option><option value="front">frontier only</option></select></label>
</div>
<svg id="chart" width="1000" height="600"></svg>
<div id="tip"></div>
<p class="note">Solid lines are benchmark mode (calibrated from a measured anchor of the same
model and engine); dashed lines are simulate mode (analytical, no GPU, ranking-quality).
Each line is the Pareto frontier of one GPU / engine / speculative setting / KV hierarchy / mode
across every parallelism, concurrency and deployment knob swept; faint dots are dominated
configurations. Hover a point for its config.</p>
<div id="uncov" class="note"></div>
<table id="tbl"></table>
<script>
const DATA = __DATA__;
const M = DATA.metrics, P = DATA.points;
const PAL = ["#e4572e","#17bebb","#ffc914","#76b041","#2e86ab","#a23b72","#f18f01","#6c757d","#3b1f2b","#8338ec"];
const $ = id => document.getElementById(id);
function opts(el, vals, sel){ el.innerHTML = vals.map(v=>`<option ${v===sel?'selected':''}>${v}</option>`).join(''); }
const uniq = a => [...new Set(a)].sort();
opts($('model'), uniq(P.map(p=>p.model)));
function refreshDeps(){
  const m=$('model').value;
  const sc=uniq(P.filter(p=>p.model===m).map(p=>p.scenario)); const cur=$('scenario').value;
  opts($('scenario'), sc, sc.includes(cur)?cur:sc[0]);
  const pr=uniq(P.filter(p=>p.model===m && p.scenario===$('scenario').value).map(p=>p.precision));
  const cp=$('precision').value; opts($('precision'), pr, pr.includes(cp)?cp:pr[0]);
  const sp=uniq(P.filter(p=>p.model===m && p.scenario===$('scenario').value).map(p=>p.spec||'none'));
  const cs=$('spec').value; opts($('spec'), ['all',...sp], ['all',...sp].includes(cs)?cs:'all');
}
$('x').innerHTML = Object.keys(M).map(k=>`<option value="${k}" ${k===DATA.x?'selected':''}>${M[k].label}</option>`).join('');
$('y').innerHTML = Object.keys(M).map(k=>`<option value="${k}" ${k===DATA.y?'selected':''}>${M[k].label}</option>`).join('');
const series = p => p.series_label || `${p.gpu_arch.toUpperCase()} ${p.engine}${(p.spec||'none')==='none'?'':' '+p.spec}${p.kv_config==='hbm'?'':' KV:'+p.kv_config} (${p.mode})`;
function front(pts,x,y){
  const sx=M[x].higher_is_better?1:-1, sy=M[y].higher_is_better?1:-1;
  const s=[...pts].sort((a,b)=>(sx*b[x]-sx*a[x])||(sy*b[y]-sy*a[y]));
  const f=[]; let best=null;
  for(const p of s){ const v=sy*p[y]; if(best===null||v>best){f.push(p);best=v;} }
  return f.sort((a,b)=>a[x]-b[x]);
}
function draw(){
  const m=$('model').value, sc=$('scenario').value, pr=$('precision').value, md=$('mode').value, sp=$('spec').value;
  const x=$('x').value, y=$('y').value, onlyF=$('show').value==='front';
  const sel=P.filter(p=>p.model===m&&p.scenario===sc&&p.precision===pr&&(md==='all'||p.mode===md)&&(sp==='all'||(p.spec||'none')===sp));
  const pts=sel.filter(p=>p.feasible);
  const svg=$('chart'); const W=1000,H=600,L=80,R=250,T=30,B=60;
  let o=[];
  const xs=pts.map(p=>p[x]), ys=pts.map(p=>p[y]);
  const x1=(Math.max(...xs,1))*1.05, y1=(Math.max(...ys,1))*1.08;
  const X=v=>L+v/x1*(W-L-R), Y=v=>H-B-v/y1*(H-T-B);
  for(let i=0;i<=5;i++){const gx=x1*i/5, gy=y1*i/5;
    o.push(`<line x1=${X(gx)} y1=${T} x2=${X(gx)} y2=${H-B} stroke="#eee"/><line x1=${L} y1=${Y(gy)} x2=${W-R} y2=${Y(gy)} stroke="#eee"/>`);
    o.push(`<text x=${X(gx)} y=${H-B+16} font-size=11 text-anchor=middle>${gx.toFixed(gx<10?1:0)}</text><text x=${L-6} y=${Y(gy)+4} font-size=11 text-anchor=end>${gy.toFixed(gy<10?1:0)}</text>`);}
  o.push(`<rect x=${L} y=${T} width=${W-L-R} height=${H-T-B} fill=none stroke="#888"/>`);
  o.push(`<text x=${(L+W-R)/2} y=${H-18} text-anchor=middle font-size=13>${M[x].label}</text>`);
  o.push(`<text transform="translate(20,${(T+H-B)/2}) rotate(-90)" text-anchor=middle font-size=13>${M[y].label}</text>`);
  const ser=uniq(pts.map(series)); const rows=[];
  ser.forEach((s,i)=>{
    const c=PAL[i%PAL.length], g=pts.filter(p=>series(p)===s), f=front(g,x,y), dash=s.includes('(simulate)')?'stroke-dasharray="6,4"':'';
    if(f.length>1) o.push(`<polyline fill=none stroke="${c}" stroke-width=2 ${dash} points="${f.map(p=>X(p[x])+','+Y(p[y])).join(' ')}"/>`);
    (onlyF?f:g).forEach(p=>{const on=f.includes(p);
      o.push(`<circle cx=${X(p[x])} cy=${Y(p[y])} r=${on?5:3} fill="${c}" fill-opacity=${on?1:.3} data-i="${P.indexOf(p)}"/>`);});
    f.forEach(p=>rows.push([s,p]));
    const ly=T+12+i*20;
    o.push(`<line x1=${W-R+15} y1=${ly} x2=${W-R+42} y2=${ly} stroke="${c}" stroke-width=2 ${dash}/><text x=${W-R+48} y=${ly+4} font-size=12>${s}</text>`);
  });
  svg.innerHTML=o.join('');
  const un=sel.filter(p=>!p.feasible);
  $('uncov').textContent = un.length ? `${un.length} configuration(s) not plotted (did not fit, or no anchor covered them in benchmark mode).` : '';
  const cols=['parallelism','concurrency','gpus','mode','kv_config','spec','spec_tokens_per_step','knobs','interactivity','total_tput_per_gpu','output_tput_per_gpu','ttft_ms','tpot_ms','prefix_hit_rate'];
  $('tbl').innerHTML='<tr><th>series</th>'+cols.map(c=>`<th>${c}</th>`).join('')+'</tr>'+
    rows.map(([s,p])=>'<tr><td>'+s+'</td>'+cols.map(c=>`<td>${typeof p[c]==='number'&&!Number.isInteger(p[c])?p[c].toFixed(2):p[c]}</td>`).join('')+'</tr>').join('');
}
$('chart').addEventListener('mousemove',e=>{
  const i=e.target.getAttribute&&e.target.getAttribute('data-i'); const t=$('tip');
  if(i===null||i===undefined){t.style.display='none';return;}
  const p=P[+i];
  t.textContent=`${p.model} ${p.engine} ${p.precision} (${p.mode})\\n${p.parallelism}  conc ${p.concurrency}  ${p.gpus} GPU\\n`+
   `interactivity ${p.interactivity.toFixed(1)} tok/s/user\\ntput/gpu ${p.total_tput_per_gpu.toFixed(1)} (out ${p.output_tput_per_gpu.toFixed(1)})\\n`+
   `TTFT ${p.ttft_ms.toFixed(0)} ms  TPOT ${p.tpot_ms.toFixed(2)} ms`+(p.anchor?`\\nanchor ${p.anchor.split('/').pop()}`:'')+(p.kv_tiers?`\\nKV tiers ${p.kv_tiers}`:'')+
   ((p.spec||'none')!=='none'?`\\nspeculative ${p.spec} (k=${p.spec_k}, accept ${p.spec_accept}, ${(p.spec_tokens_per_step||1).toFixed(2)} tok/step)`:'')+
   (p.knobs?`\\n${p.knobs}`:'');
  t.style.display='block'; t.style.left=(e.clientX+14)+'px'; t.style.top=(e.clientY+10)+'px';
});
['model','scenario','precision'].forEach(id=>$(id).addEventListener('change',()=>{refreshDeps();draw();}));
['mode','spec','x','y','show'].forEach(id=>$(id).addEventListener('change',draw));
refreshDeps(); draw();
</script></body></html>
"""


_HTML_INFEASIBLE = ("model", "scenario", "precision", "mode", "spec", "feasible", "reason")


def write_html(result: ParetoResult, path: str) -> None:
    data = dict(
        x=result.x,
        y=result.y,
        metrics={k: dict(label=v[0], higher_is_better=v[1]) for k, v in METRICS.items()},
        # The page only counts the points it cannot plot, so those carry just
        # the fields its filters read; the CSV and JSON keep every column.
        points=[
            asdict(p) if p.feasible else {k: getattr(p, k) for k in _HTML_INFEASIBLE}
            for p in result.points
        ],
    )
    blob = json.dumps(data).replace("</", "<\\/")
    with open(path, "w") as f:
        f.write(_HTML.replace("__DATA__", blob))


def write_outputs(result: ParetoResult, out_dir: str, meta: dict[str, Any]) -> dict[str, list[str]]:
    os.makedirs(out_dir, exist_ok=True)
    written: dict[str, list[str]] = {"csv": [], "json": [], "html": [], "svg": [], "png": []}
    p = os.path.join(out_dir, "pareto_points.csv")
    write_csv(result.points, p)
    written["csv"].append(p)
    p = os.path.join(out_dir, "pareto_points.json")
    write_json(result, p, meta)
    written["json"].append(p)
    p = os.path.join(out_dir, "pareto.html")
    write_html(result, p)
    written["html"].append(p)
    charts: dict[tuple, list[ParetoPoint]] = {}
    for pt in result.points:
        charts.setdefault(pt.chart(), []).append(pt)
    for (model, scen, prec), pts in sorted(charts.items()):
        title = f"{model} | {scen} | {prec}"
        stem = os.path.join(out_dir, "pareto_" + _slug(model, scen, prec))
        write_svg(pts, stem + ".svg", result.x, result.y, title)
        written["svg"].append(stem + ".svg")
        if write_png(pts, stem + ".png", result.x, result.y, title):
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
        f"{'/'.join(SPEC_DEFAULTS)} (e.g. none mtp mtp:3:0.8 eagle3:4). K is the "
        "draft depth, ACCEPT the per-draft-token acceptance, DRAFT_COST one draft "
        "pass as a fraction of a target decode step; omitted parts take "
        "the method's defaults "
        + ", ".join(f"{m}={k}:{a:g}:{d:g}" for m, (k, a, d) in SPEC_DEFAULTS.items())
        + ".",
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
    ap.add_argument("--requests-per-client", type=int, default=4)
    ap.add_argument("--min-requests", type=int, default=64)
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
    specs = list(dict.fromkeys(SpecConfig.parse(s) for s in a.spec))
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
        trace_max_requests=a.trace_max_requests,
        workdir=os.path.join(a.out_dir, "traces"),
        gpu_cost_per_hour=a.gpu_cost_per_hour,
        specs=specs,
        knob_axes=axes,
        series_by=a.series_by,
        extra=extra,
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
    t0 = time.perf_counter()
    pts = run(tasks, jobs=a.jobs, sink=partial) + uncovered
    pts = merge_points(salvaged, pts)
    res = ParetoResult(points=pts, elapsed_s=time.perf_counter() - t0, x=a.x, y=a.y)
    mark_frontiers(res.points, a.x, a.y)
    meta = dict(argv=list(argv) if argv is not None else sys.argv[1:], n_tasks=len(tasks))
    written = write_outputs(res, a.out_dir, meta)
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
