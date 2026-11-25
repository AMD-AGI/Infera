###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: role of glm52's MTP pieces (`_build_spec_worker`, `run_step`, `draft_ops`, `spec_phases`,
# `--full-step`) in AKLR `aklr/benchmarks/models/glm52/{model_bench,profile_decode}.py` @ 33689332 and of
# sglang_inproc/tool2_perop/dsv41/spec_dspark.py, rebuilt on vLLM's runner: the drafter is the runner's own
# speculator (DSpark, vllm/models/deepseek_v41/amd/dspark.py), driven per-op by vLLM bench forward's
# DraftBench (fixed, fully accepted target results; vllm/benchmarks/forward/speculator.py), and whole
# steps (`--full-step`) are execute_model + sample_tokens under the engine's own cuda graphs.

"""DSpark draft model rows and whole-step (`--full-step`) driving for tool 2 (vLLM)."""

import contextlib
import time

import torch

import model_bench


def speculator(runner):
    return getattr(runner, "speculator", None) if model_bench.spec_config(runner) is not None else None


def draft_model(runner):
    spec = speculator(runner)
    if spec is None:
        return None
    for name in ("model", "draft_model"):
        m = getattr(spec, name, None)
        if isinstance(m, torch.nn.Module):
            return m
    return None


def draft_layers(runner) -> list:
    m = draft_model(runner)
    if m is None:
        return []
    return [x for x in m.modules() if hasattr(x, "attn") and hasattr(x, "ffn") and hasattr(x, "mhc_pre_delayed")]


def draft_ops(runner) -> list:
    """The first draft layer's rows, keyed `draft|<row>` (one row set; the draft model is one roll-up row)."""
    layers = draft_layers(runner)
    return model_bench.layer_ops(layers[0], "draft") if layers else []


class Drafter:
    """vLLM bench forward's DraftBench on the current batch: propose() with every draft accepted."""

    def __init__(self, runner):
        from vllm.benchmarks.forward.speculator import DraftBench

        self.bench = DraftBench(runner)
        self.runner = runner

    def prepare(self, fwd) -> None:
        self.bench.prepare(fwd.input_batch, fwd.ctx[0])

    def propose(self, output):
        self.bench.propose(output)
        return self.bench.tokens


def forward_note(runner) -> str:
    return (f"target verify forward + logits ({1 + model_bench.num_spec(runner)} query tokens/request) and the"
            " DSpark drafter (all drafts accepted, vLLM bench forward's DraftBench), each under its own graph,"
            " summed")


# ----------------------------------------------------------------------------------------------
# Whole steps under the engine's own graphs.


class Stepper:
    """execute_model + sample_tokens on a fixed decode/verify batch. Step 0 adds the requests (untimed); each
    later step is a cached-request step at the same context (num_computed_tokens held at the case's prefix,
    every step re-verifying the same positions), so every sample has the reference shape. The runner's CPU
    input prep, graph dispatch, sampling / rejection sampling and the drafter are inside the clock."""

    def __init__(self, runner, batch, drafts: int):
        self.runner, self.batch, self.drafts = runner, batch, drafts
        self.k = 0
        self.replays = {"n": 0}
        mgr = getattr(runner, "cudagraph_manager", None)
        self._orig_full = getattr(mgr, "run_fullgraph", None) if mgr is not None else None
        if self._orig_full is not None:
            def counted(*a, **kw):
                self.replays["n"] += 1
                return self._orig_full(*a, **kw)

            mgr.run_fullgraph = counted

    def _run(self, so):
        so.num_spec_tokens_to_schedule = self.drafts
        out = self.runner.execute_model(so)
        if out is None:
            out = self.runner.sample_tokens(None)
        if hasattr(out, "get_output"):
            out = out.get_output()
        return out

    def first(self):
        self._run(self.batch.scheduler_output)

    def cached(self):
        from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

        b = self.batch
        ids = b.req_ids
        self.k += 1
        so = SchedulerOutput.make_empty()
        so.scheduled_cached_reqs = CachedRequestData(
            req_ids=list(ids), new_token_ids=[[] for _ in ids], new_block_ids=[None] * len(ids),
            num_computed_tokens=[b.expected[r][1] for r in ids], num_output_tokens=[self.k] * len(ids))
        so.num_scheduled_tokens = {r: b.expected[r][0] for r in ids}
        if self.drafts:
            so.scheduled_spec_decode_tokens = {r: [0] * self.drafts for r in ids}
        so.total_num_scheduled_tokens = sum(so.num_scheduled_tokens.values())
        return lambda: self._run(so)

    def close(self):
        mgr = getattr(self.runner, "cudagraph_manager", None)
        if self._orig_full is not None:
            mgr.__dict__.pop("run_fullgraph", None)
        self.runner.execute_model(self.batch.finish())
        self.runner.execute_model_state = None


def phases(runner) -> list:
    """CUDA-event phases of one step (glm52 spec_phases): graph forward, sampler, drafter."""
    out = [("forward:target (graph)", getattr(runner, "cudagraph_manager", None), "run_fullgraph"),
           ("sample", runner, "sample")]
    spec = getattr(runner, "speculator", None)
    if spec is not None:
        out.append(("propose:draft", spec, "propose"))
    return out


@contextlib.contextmanager
def time_calls(targets, into: dict):
    """CUDA-event timing around chosen methods, one event pair per call."""
    saved = []
    for label, obj, name in targets:
        if obj is None or not hasattr(obj, name):
            continue
        orig = getattr(obj, name)
        had = name in getattr(obj, "__dict__", {})

        def wrapped(*a, _l=label, _o=orig, **kw):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = _o(*a, **kw)
            e.record()
            into.setdefault(_l, []).append((s, e))
            return out

        setattr(obj, name, wrapped)
        saved.append((obj, name, orig, had))
    try:
        yield
    finally:
        for obj, name, orig, had in reversed(saved):
            if had:
                setattr(obj, name, orig)
            else:
                obj.__dict__.pop(name, None)


def elapsed(into: dict) -> dict:
    return {k: (sum(s.elapsed_time(e) for s, e in pairs), len(pairs)) for k, pairs in into.items()}


def full_step(runner, batch, n: int, rank_print, conc: int) -> dict:
    """N+2 synced cached steps, fastest wins (glm52 --full-step); returns the numbers for the log/JSON."""
    st = Stepper(runner, batch, model_bench.num_spec(runner))
    try:
        st.first()
        torch.cuda.synchronize()
        ms, evs = [], []
        for _ in range(n + 2):
            run = st.cached()  # built outside the clock
            ev = {}
            torch.cuda.synchronize()
            t = time.perf_counter()
            with time_calls(phases(runner), ev):
                run()
            torch.cuda.synchronize()
            ms.append((time.perf_counter() - t) * 1000.0)
            evs.append(ev)
    finally:
        st.close()
    best = min(range(len(ms)), key=ms.__getitem__)
    rank_print("synced per step ms: " + " ".join(f"{m:.3f}" for m in ms))
    ph = elapsed(evs[best])
    for label, (total, calls) in ph.items():
        rank_print(f"  {label}: {total:.3f} ms  {calls} calls/step")
    rank_print(f"full-graph replays during {len(ms) + 1} steps: {st.replays['n']}")
    acc = model_bench.accept_len(runner)
    spec = model_bench.spec_config(runner) is not None
    tok = conc * (acc or 1) / ms[best] * 1000
    rank_print(f"CONC {conc} step: synced {ms[best]:.3f} ms"
               + (f"  {tok:.1f} tok/s" if (not spec or acc) else "  tok/s not reported (no synthetic acceptance)")
               + (f" at accept len {acc:.2f}" if acc else ""))
    return {"conc": conc, "step_ms": ms[best], "samples_ms": ms, "phases_ms": {k: v[0] for k, v in ph.items()},
            "graph_replays": st.replays["n"], "steps": len(ms) + 1, "accept_len": acc,
            "tok_s": tok if (not spec or acc) else None}
