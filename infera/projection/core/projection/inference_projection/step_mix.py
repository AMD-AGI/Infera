###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A replay's step mix: record it once, re-derive any deployment's kernels from it.

A DES replay decides two things: which steps run (how many sequences each
carries and how far back they attend), and how long each takes. The kernels a
step launches, and their shapes, follow from the first and the deployment
alone. So a step mix, the replay's distinct step compositions with their
counts, is enough to report every kernel for another tensor, expert or
pipeline split, other weight, KV or activation dtypes, another drafter or
another GPU, without replaying the trace again.

Each step also keeps the key the DES's cost kernel priced it by, so the mix can
be re-priced under the new deployment: the same steps, charged what the new
configuration's cost model charges them. That gives an end-to-end estimate
without a replay, and, with measured kernel speedups
(:mod:`kernel_speedups`), an estimate of what they buy end to end.

Reuse is exact when the schedule could not have changed: the same workload,
scheduler limits and speculation, a closed loop with no think time (a client's
next request waits on completions, not on the clock), and a KV pool that still
holds what the replay kept resident. :func:`check` says which of these hold.
An open-loop replay's steps interleave with arrivals, so a deployment that is
much faster or slower would have batched differently; :func:`check` reports
how much the step time moved.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field

from . import gemm_shapes
from . import kernel_shapes as ks
from . import layer_plan as lp

SCHEMA = "inferasim.step_mix"
VERSION = 1

# Fields that set the schedule itself. A change to any of them means the
# replay's steps are not the steps the new deployment would run.
SCHEDULE_KEYS = (
    "input_len",
    "output_len",
    "max_concurrency",
    "max_num_batched_tokens",
    "chunked_prefill_size",
    "arrival",
    "request_rate",
    "num_requests",
    "seed",
    "warmup_frac",
    "burstiness",
    "range_ratio",
    "workload",
    "think_ms",
    "prefix",
    "spec",
    "disaggregated",
    "kv_pool",
    "block_size",
)


# ── the recorded schedule ───────────────────────────────────────────────────


def _file_digest(path: str | None) -> str | None:
    if not path:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except OSError:
        return None


def _spec(projector) -> list:
    req = projector.cfg.request_config
    k = int(getattr(req, "speculative_num_tokens", 0) or 0)
    rates = None
    sched = getattr(projector, "speculative_schedule", None)
    if callable(sched):
        try:
            got = sched()
        except Exception:
            got = None
        if got is not None:
            k, rates = got
    if k <= 0:
        return [0, None]
    if rates is None:
        rates = [float(getattr(req, "speculative_acceptance_rate", 0.0) or 0.0)]
    return [int(k), [round(float(r), 6) for r in rates]]


# Set by the replay's own flags. Re-deriving a mix inherits them: the mix is that replay.
REPLAY_KEYS = (
    "arrival",
    "num_requests",
    "seed",
    "warmup_frac",
    "burstiness",
    "range_ratio",
    "workload",
    "think_ms",
    "prefix",
    "kv_pool",
    "block_size",
)


def schedule_of(cfg, args, projector, replay: dict | None = None) -> dict:
    """What decided the replay's steps, as the CLI and config gave it.

    With ``replay`` (a mix's recorded schedule) the replay's own settings are
    taken from it, and only what the deployment's config decides is read here.
    """
    req = cfg.request_config
    closed = bool(getattr(args, "des_closed_loop", False))
    arrival = "closed" if closed else (getattr(req, "arrival_model", "closed") or "closed")
    trace = getattr(args, "des_mooncake_trace", None)
    workload = getattr(args, "des_workload_file", None)
    if trace or workload:
        arrival = "trace"
    split = getattr(cfg, "disaggregation_config", None)
    disagg = None
    if getattr(split, "enabled", False):
        disagg = {
            k: getattr(split, k, None)
            for k in ("prefill_replicas", "decode_replicas")
            if getattr(split, k, None) is not None
        } or {"enabled": True}
    sched = {
        "input_len": int(req.input_seq_len),
        "output_len": int(req.output_seq_len),
        "max_concurrency": int(req.resolved_max_concurrency()),
        "max_num_batched_tokens": int(req.max_num_batched_tokens or 0),
        "chunked_prefill_size": int(req.chunked_prefill_size or 0),
        "arrival": arrival,
        "request_rate": float(req.request_rate or 0.0) if arrival != "closed" else 0.0,
        "num_requests": int(getattr(args, "des_num_requests", 400) or 400),
        "seed": int(getattr(args, "des_seed", 0) or 0),
        "warmup_frac": float(getattr(args, "des_warmup_frac", 0.1) or 0.0),
        "burstiness": float(getattr(args, "des_burstiness", 1.0) or 1.0),
        "range_ratio": float(getattr(args, "des_range_ratio", 1.0) or 1.0),
        "workload": _file_digest(trace or workload),
        "think_ms": float(getattr(args, "des_client_think_ms", 0.0) or 0.0),
        "prefix": [
            int(getattr(args, "des_num_prefixes", 0) or 0),
            int(getattr(args, "des_prefix_len", 0) or 0),
            float(getattr(args, "des_prefix_zipf", 0.0) or 0.0),
        ],
        "spec": _spec(projector),
        "disaggregated": disagg,
        "kv_pool": "memory"
        if getattr(args, "des_kv_pool_from_memory", False)
        else int(getattr(args, "des_kv_cache_tokens", 0) or 0),
        "block_size": int(getattr(args, "des_block_size", 0) or 0),
    }
    if replay is not None:
        sched.update({k: replay.get(k) for k in REPLAY_KEYS if k in replay})
        if sched.get("arrival") == "closed":
            sched["request_rate"] = 0.0
    return sched


def deployment_of(cfg, args=None) -> dict:
    """What the kernels' shapes depend on, for the record (not for reuse checks)."""
    mc, mp, req = cfg.model_config, cfg.model_parallel_config, cfg.request_config
    tp, atp, dp, ep, dcp = ks._ranks(mp)
    return {
        "model": getattr(mc, "model_name", None) or os.environ.get("INFERASIM_MODEL"),
        "num_layers": int(mc.num_layers),
        "hidden_size": int(mc.hidden_size),
        "tp": tp,
        "attention_dp": dp,
        "ep": ep,
        "pp": int(getattr(mp, "pipeline_model_parallel_size", 1) or 1),
        "dcp": dcp,
        "kv_cache_dtype": ks._kv_dtype(cfg),
        "linear_weight_dtype": gemm_shapes._weight_dtype(cfg, "linear_weight_dtype"),
        "moe_expert_dtype": gemm_shapes._weight_dtype(cfg, "moe_expert_dtype"),
        "act_quant_dtype": ks._act_quant_dtype(cfg),
        "serving_engine": getattr(req, "serving_engine", None),
        "cudagraph_mode": getattr(req, "cudagraph_mode", None),
        "gpu_arch": getattr(args, "gpu_arch", None) if args is not None else None,
        "window": list(lp.window_of(cfg)),
        "decode_window_capped": bool(getattr(mc, "sink_window_decode_capped", False)),
    }


# ── the mix ─────────────────────────────────────────────────────────────────


def _seqs(s: ks.Seqs | None):
    return None if s is None else [s.n, s.q, s.max_q, s.max_kv, s.mean_kv]


def _unseqs(v) -> ks.Seqs | None:
    return None if v is None else ks.Seqs(*(int(x) for x in v))


@dataclass
class Pool:
    q_len: int = 1
    dpa_spread: bool = False
    window: tuple[int, float] = (0, 0.0)
    capped: bool = False
    # (step, cost key) -> count, and the recorded price of each cost key.
    priced: Counter = field(default_factory=Counter)
    recorded_ms: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return sum(self.priced.values())


@dataclass
class Mix:
    pools: dict[str, Pool]
    schedule: dict
    deployment: dict
    des: dict
    path: str | None = None

    @property
    def num_steps(self) -> int:
        return sum(p.num_steps for p in self.pools.values())


def _cost_kernel(projector, pool: str, meta: Pool):
    from .des import _CostKernel

    return _CostKernel(projector, meta.q_len, dpa_spread=meta.dpa_spread, pool=pool)


def price(kernel, cost) -> float | None:
    """What ``kernel`` (a DES cost kernel) charges a step recorded as ``cost``."""
    if not cost:
        return None
    with gemm_shapes.paused():
        if cost[0] == "decode":
            return float(kernel.decode_step_ms(int(cost[1]), int(cost[2])))
        if cost[0] == "mixed":
            nd, tok, ctx, pkv, reqs = (int(x) for x in cost[1:6])
            return float(kernel.mixed_step_ms(nd, tok, ctx, pkv, reqs))
    return None


def from_trace(
    trace: ks.KernelTrace, projectors: dict, schedule: dict, deployment: dict, des: dict
):
    """The step mix a :class:`kernel_shapes.KernelTrace` recorded, priced as it ran."""
    pools = {}
    for name, priced in trace._priced.items():
        meta = trace._pool_meta.get(name, {})
        cfg = trace._cfgs[name]
        p = Pool(
            q_len=int(meta.get("q_len", 1)),
            dpa_spread=bool(meta.get("dpa_spread", False)),
            window=lp.window_of(cfg),
            capped=bool(getattr(cfg.model_config, "sink_window_decode_capped", False)),
            priced=Counter(priced),
        )
        proj = projectors.get(name)
        if proj is not None:
            kernel = _cost_kernel(proj, name, p)
            for _, cost in priced:
                if cost and cost not in p.recorded_ms:
                    p.recorded_ms[cost] = price(kernel, cost)
        pools[name] = p
    return Mix(pools, schedule, deployment, des)


def dump(mix: Mix, path: str) -> None:
    pools = {}
    for name, p in mix.pools.items():
        steps = []
        for (step, cost), n in sorted(
            p.priced.items(), key=lambda kv: (-kv[1], kv[0][0].tokens, str(kv[0][1]))
        ):
            steps.append(
                {
                    "count": n,
                    "tokens": step.tokens,
                    "sampled_rows": step.sampled_rows,
                    "decode": _seqs(step.decode),
                    "prefill": _seqs(step.prefill),
                    "decode_window": _seqs(step.decode_window),
                    "cost": list(cost) if cost else None,
                    "ms": None if not cost else p.recorded_ms.get(cost),
                }
            )
        pools[name] = {
            "q_len": p.q_len,
            "dpa_spread": p.dpa_spread,
            "window": list(p.window),
            "decode_window_capped": p.capped,
            "num_steps": p.num_steps,
            "steps": steps,
        }
    doc = {
        "schema": SCHEMA,
        "version": VERSION,
        "schedule": mix.schedule,
        "deployment": mix.deployment,
        "des": mix.des,
        "pools": pools,
    }
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)


def load(path: str) -> Mix:
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{path} is not a step mix (schema {doc.get('schema')!r})")
    if int(doc.get("version", 0)) > VERSION:
        raise ValueError(f"{path} is step-mix version {doc['version']}; this reads {VERSION}")
    pools = {}
    for name, raw in doc["pools"].items():
        p = Pool(
            q_len=int(raw.get("q_len", 1)),
            dpa_spread=bool(raw.get("dpa_spread", False)),
            window=tuple(raw.get("window") or (0, 0.0)),
            capped=bool(raw.get("decode_window_capped", False)),
        )
        for s in raw["steps"]:
            step = ks.Step(
                int(s["tokens"]),
                int(s["sampled_rows"]),
                _unseqs(s.get("decode")),
                _unseqs(s.get("prefill")),
                _unseqs(s.get("decode_window")),
            )
            cost = tuple(s["cost"]) if s.get("cost") else None
            p.priced[(step, cost)] += int(s["count"])
            if cost and s.get("ms") is not None:
                p.recorded_ms[cost] = float(s["ms"])
        pools[name] = p
    return Mix(pools, doc.get("schedule", {}), doc.get("deployment", {}), doc.get("des", {}), path)


# ── reuse ───────────────────────────────────────────────────────────────────


def _rewindow(step: ks.Step, cfg) -> ks.Step:
    """The step's windowed decode spans under ``cfg``'s window.

    Exact when the window is the recorded one; otherwise each span is capped
    at the new window from the step's longest and mean spans, which is exact
    when every span is at least the window (the usual case: windows are short
    next to the contexts they slide over).
    """
    w, frac = lp.window_of(cfg)
    capped = w and frac > 0 and getattr(cfg.model_config, "sink_window_decode_capped", False)
    if not capped or step.decode is None:
        return ks.Step(step.tokens, step.sampled_rows, step.decode, step.prefill, None)
    d, cap = step.decode, ks._up(w, ks.KV_BUCKET)
    return ks.Step(
        step.tokens,
        step.sampled_rows,
        d,
        step.prefill,
        ks.Seqs(d.n, d.q, d.max_q, min(d.max_kv, cap), min(d.mean_kv, cap)),
    )


def to_trace(mix: Mix, projectors: dict) -> ks.KernelTrace:
    """A kernel trace of the mix's steps, under each pool's new ``projectors[pool]``."""
    trace = ks.KernelTrace()
    for name, p in mix.pools.items():
        proj = projectors.get(name)
        if proj is None:
            raise ValueError(f"the step mix has pool {name!r}; no projector was given for it")
        trace.bind(name, proj, p.q_len, p.dpa_spread)
        same_window = tuple(lp.window_of(proj.cfg)) == tuple(p.window)
        for (step, cost), n in p.priced.items():
            if not same_window:
                step = _rewindow(step, proj.cfg)
            trace.add_step(name, step, cost, n)
            trace._steps[name][(ks.graph_tokens(proj.cfg, step), step.sampled_rows)] += n
    return trace


@dataclass
class Check:
    exact: bool
    allowed: bool
    reasons: list[str]

    def as_dict(self) -> dict:
        return {"exact": self.exact, "allowed": self.allowed, "reasons": list(self.reasons)}


def _resident_peak(p: Pool) -> int:
    """An upper bound on the KV the replay kept resident in one step (bucketed spans)."""
    peak = 0
    for step, _ in p.priced:
        t = 0
        for s in (step.decode, step.prefill):
            if s is not None:
                t += s.n * s.mean_kv
        peak = max(peak, t)
    return peak


def check(mix: Mix, schedule: dict, kv_capacity: dict | None = None) -> Check:
    """Whether the mix's steps are the steps ``schedule`` would run.

    ``kv_capacity``: per pool, the tokens of KV the new deployment's pool holds
    (0 or missing when the replay's pool was unbounded).
    """
    reasons, allowed, exact = [], True, True
    rec = mix.schedule
    for key in SCHEDULE_KEYS:
        if key in rec and key in schedule and rec[key] != schedule[key]:
            allowed = exact = False
            reasons.append(f"{key} differs: recorded {rec[key]!r}, now {schedule[key]!r}")
    if rec.get("arrival") != "closed":
        exact = False
        reasons.append(
            f"the replay was open-loop ({rec.get('arrival')}): its steps interleave with "
            "arrivals, so a faster or slower deployment would batch differently"
        )
    elif float(rec.get("think_ms") or 0.0) > 0:
        exact = False
        reasons.append(
            "closed-loop clients think between requests, so issue times follow the clock"
        )
    for name, cap in (kv_capacity or {}).items():
        p = mix.pools.get(name)
        if p is None or not cap:
            continue
        peak = _resident_peak(p)
        if peak > cap:
            exact = False
            reasons.append(
                f"{name}: the replay kept up to ~{peak} KV tokens resident; this deployment "
                f"holds {cap}, so it would preempt or admit fewer"
            )
    return Check(exact, allowed, reasons)


@dataclass
class Estimate:
    """The mix re-priced under a deployment, per pool and end to end."""

    pools: dict
    des: dict

    def as_dict(self) -> dict:
        return {"pools": self.pools, "des": self.des}


def reprice(mix: Mix, trace: ks.KernelTrace, projectors: dict, speedups=None) -> Estimate:
    """Charge every recorded step what the new deployment's cost model charges it.

    With ``speedups`` each step is further scaled by how much its own kernels
    got faster (:meth:`kernel_speedups.Speedups.step_factor`).
    """
    pools = {}
    for name, p in mix.pools.items():
        kernel = _cost_kernel(projectors[name], name, p)
        cfg = trace._cfgs[name]
        stages = lp.stage_counts(cfg)
        draft = trace._drafts.get(name)
        prices: dict = {}
        rec = new = base = dec_rec = dec_new = 0.0
        for (step, cost), n in p.priced.items():
            if cost not in prices:
                prices[cost] = price(kernel, cost)
            ms = prices[cost]
            if ms is None:
                continue
            f = _step_factor(trace, name, cfg, stages, draft, step, speedups) if speedups else 1.0
            base += n * ms
            new += n * ms * f
            r = p.recorded_ms.get(cost)
            rec += n * (r if r is not None else ms)
            if step.prefill is None:
                dec_rec += n * (r if r is not None else ms)
                dec_new += n * ms * f
        pools[name] = {
            "steps": p.num_steps,
            "recorded_busy_ms": round(rec, 3),
            "busy_ms": round(base, 3),
            "busy_ms_with_speedups": round(new, 3) if speedups else None,
            "ratio": round(new / rec, 4) if rec else None,
            "decode_ratio": round(dec_new / dec_rec, 4) if dec_rec else None,
        }
    return Estimate(pools, _rescale(mix.des, pools))


def _step_factor(trace, pool, cfg, stages, draft, step, speedups) -> float:
    memo = trace.__dict__.setdefault("_factors", {})
    key = (pool, step)
    if key in memo:
        return memo[key]
    timed = []
    for stage, c in enumerate(stages):
        for k, m in ks.step_kernels(cfg, step, c):
            us, _ = trace.time(pool, k)
            timed.append(
                (
                    {
                        "pool": pool,
                        "stage": stage,
                        "kind": k.kind,
                        "op": k.op,
                        "dtype": k.dtype,
                        **dict(k.dims),
                    },
                    us,
                    m,
                )
            )
    if draft is not None and step.decode is not None:
        dpool = f"{pool}/draft"
        for s, m, extra in ks.draft_passes(draft, step.decode):
            for k, c in ks.step_kernels(draft.cfg, s) + extra:
                us, _ = trace.time(dpool, k)
                timed.append(
                    (
                        {
                            "pool": dpool,
                            "kind": k.kind,
                            "op": k.op,
                            "dtype": k.dtype,
                            **dict(k.dims),
                        },
                        us,
                        m * c,
                    )
                )
    memo[key] = speedups.step_factor(timed)
    return memo[key]


def step_of_cost(cfg, cost, q_len: int = 1) -> ks.Step:
    """A representative step for a cost-kernel key: every sequence at the key's mean span."""
    up = ks._up
    if cost[0] == "decode":
        b, ctx = int(cost[1]), int(cost[2])
        dec = ks.Seqs(
            b, b * q_len, q_len, up(ctx + q_len, ks.KV_BUCKET), up(ctx + q_len, ks.KV_BUCKET)
        )
        return _rewindow(ks.Step(b * q_len, b * q_len, dec, None, None), cfg)
    nd, tok, ctx, pkv, reqs = (int(x) for x in cost[1:6])
    reqs = max(1, reqs)
    dec = None
    if nd:
        span = up(ctx + q_len, ks.KV_BUCKET)
        dec = ks.Seqs(nd, nd * q_len, q_len, span, span)
    span = up(pkv, ks.KV_BUCKET)
    pre = ks._bucket_q(ks.Seqs(reqs, tok, max(1, -(-tok // reqs)), span, span))
    return _rewindow(ks.Step(nd * q_len + tok, nd * q_len + reqs, dec, pre, None), cfg)


def cost_step_factor(speedups, projector, pool: str, q_len: int, cost) -> float:
    """How much a DES step priced as ``cost`` speeds up under ``speedups``."""
    memo = speedups.__dict__.setdefault("_cost_factors", {})
    key = (id(projector), pool, tuple(cost))
    if key in memo:
        return memo[key]
    traces = speedups.__dict__.setdefault("_traces", {})
    tkey = (id(projector), pool)
    if tkey not in traces:
        t = ks.KernelTrace()
        t.bind(pool, projector, q_len)
        traces[tkey] = t
    trace = traces[tkey]
    cfg = projector.cfg
    step = step_of_cost(cfg, cost, q_len)
    memo[key] = _step_factor(
        trace, pool, cfg, lp.stage_counts(cfg), trace._drafts.get(pool), step, speedups
    )
    return memo[key]


def _rescale(des: dict, pools: dict) -> dict:
    """The recorded replay's headline numbers, moved by the re-priced busy time.

    The makespan keeps the replay's idle time and swaps the busiest pool's busy
    time for its re-priced one; throughputs scale inversely with it, TPOT and
    ITL with the decode steps' ratio. TTFT and end-to-end latency include
    queueing that moves non-linearly with the step time, so they are scaled by
    the overall ratio and should be read as indicative.
    """
    if not des or not pools:
        return {}
    busiest = max(pools.values(), key=lambda v: v["recorded_busy_ms"])
    rec_busy = busiest["recorded_busy_ms"]
    new_busy = busiest["busy_ms_with_speedups"] or busiest["busy_ms"]
    makespan = float(des.get("makespan_ms") or 0.0)
    if not makespan or not rec_busy:
        return {}
    idle = max(0.0, makespan - rec_busy)
    est = idle + new_busy
    for v in pools.values():
        est = max(est, v["busy_ms_with_speedups"] or v["busy_ms"])
    ratio = est / makespan
    dec = busiest.get("decode_ratio") or busiest.get("ratio") or 1.0
    out = {"makespan_ms": round(est, 3), "makespan_ratio": round(ratio, 4)}
    for key in ("system_throughput_tps", "system_total_throughput_tps", "achieved_rate"):
        if des.get(key):
            out[key] = round(float(des[key]) / ratio, 3)
    for key, r in (("tpot", dec), ("itl", dec), ("ttft", ratio), ("e2e", ratio)):
        if isinstance(des.get(key), dict):
            out[key] = {p: round(float(v) * r, 3) for p, v in des[key].items()}
    return out


def des_summary(point) -> dict:
    """The replay's headline numbers, kept with the mix to rescale later."""
    out = {}
    for key in (
        "makespan_ms",
        "utilization",
        "num_requests",
        "system_throughput_tps",
        "system_total_throughput_tps",
        "achieved_rate",
    ):
        v = getattr(point, key, None)
        if v is not None:
            out[key] = v
    for key in ("ttft", "tpot", "itl", "e2e"):
        v = getattr(point, key, None)
        if isinstance(v, dict) and v:
            out[key] = {k: float(x) for k, x in v.items() if isinstance(x, (int, float))}
    return out
