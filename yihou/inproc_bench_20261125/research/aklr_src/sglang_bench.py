###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################

"""Shared bench plumbing: sglang forward setup, batch prep, graph/eager timing, MoE layouts, versions and tables."""

import contextlib
import inspect
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import torch

# Repeating an op inside one graph amortises the ~10us host launch that would otherwise inflate small kernels.
GRAPH_REPS = 20

# Cap on tokens one graph pushes through; each repeat allocates its own activations in the graph pool.
GRAPH_TOKEN_BUDGET = 16384

# Any constant; it only has to be the same every run.
MOE_ROUTING_SEED = 1234

# Run from git checkouts, so the commit identifies them; version strings move too slowly.
GIT_PACKAGES = ("sglang", "aiter")


def require_aiter() -> None:
    """Fail loudly if aiter resolved to an empty namespace package."""
    import aiter

    if getattr(aiter, "__file__", None) is None:
        raise RuntimeError(
            "aiter is an empty namespace package -- PYTHONPATH is missing its "
            "directory (/aiter or /sgl-workspace/aiter, depending on the image)."
        )


def load_tag(server_args) -> str:
    """`dummy` for a synthetic-weight run, empty for a real one."""
    return "dummy" if server_args.load_format == "dummy" else ""


class MoeParallel(NamedTuple):
    """How the routed experts are split across ranks, read off the runner."""

    ep_size: int
    tp_size: int
    a2a: str | None


def parallel_state(model_runner):
    """The runner's parallel state, or SGLang's process-local replacement."""
    if hasattr(model_runner, "ps"):
        return model_runner.ps
    from sglang.srt.runtime_context import get_parallel

    return get_parallel()


def publish_once(server_args, gpu_id: int, tp_rank: int) -> None:
    """Publish the config bags if nothing has yet."""
    from sglang.srt.runtime_context import get_context, publish

    if not get_context().is_config_namespace_published("model"):
        if "ranks" in inspect.signature(publish).parameters:
            from sglang.srt.runtime_context import SpawnRanks, spawn_world_rank

            publish(
                server_args,
                role="scheduler",
                ranks=SpawnRanks(
                    world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
                    gpu_id=gpu_id,
                ),
            )
        else:
            publish(server_args, role="scheduler")


def moe_parallel(model_runner) -> MoeParallel:
    from sglang.srt.distributed import get_moe_tensor_parallel_world_size

    backend = model_runner.server_args.moe_a2a_backend
    return MoeParallel(
        ep_size=parallel_state(model_runner).moe_ep_size,
        tp_size=get_moe_tensor_parallel_world_size(),
        a2a=None if backend == "none" else backend,
    )


class A2aImpl(NamedTuple):
    """Sizing and dtypes of the all-to-all, read off its dispatcher."""

    cap: int
    dispatch_dtype: str
    combine_dtype: str


def assert_lockstep(saved: dict) -> None:
    """Raise if the ranks recorded different op lists."""
    from sglang.srt.distributed import get_world_group

    names = list(saved)
    seen = get_world_group().all_gather_object(names)
    for rank, other in enumerate(seen):
        if other == names:
            continue
        at = next(
            (i for i, (a, b) in enumerate(zip(names, other)) if a != b),
            min(len(names), len(other)),
        )
        raise RuntimeError(
            f"world rank {rank} and this rank recorded different op lists ({len(other)}"
            f" vs {len(names)} ops), first difference at {at}: {other[at : at + 1]}"
            f" vs {names[at : at + 1]}; timing them would hang"
        )


def agreed(ok: bool) -> bool:
    """``ok`` on every rank, or False on all of them."""
    from sglang.srt.distributed import get_world_group

    try:
        group = get_world_group()
    except Exception:  # noqa: BLE001 -- single-rank runs have no world
        return ok
    return all(group.all_gather_object(bool(ok)))


@contextlib.contextmanager
def forward_ctx(model_runner):
    """Publish the ForwardContext that layers read via get_forward_context()."""
    from sglang.srt.model_executor.forward_context import (
        ForwardContext,
        forward_context,
    )

    with forward_context(ForwardContext(attn_backend=model_runner.attn_backend)):
        yield


def local_conc(args, model_runner) -> int:
    """How many requests this rank runs."""
    return args.CONC // parallel_state(model_runner).attn_dp_size


def forward_batch_of(schedule_batch, model_runner):
    """Rebuild the ForwardBatch that extend()/decode() throw away."""
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    return ForwardBatch.init_new(schedule_batch, model_runner, return_hidden_states_before_norm=False)


class PrefillStep(NamedTuple):
    """One extend forward: ``extend`` new tokens on top of ``prefix`` cached ones."""

    prefix: int
    extend: int


def prefill_conc(args) -> int:
    """One request per attention rank, which is what chunked prefill produces."""
    return args.dp_size if args.enable_dp_attention else 1


def prepare_decode(batch, model_runner, input_ids=None) -> None:
    """Advance the batch to its next decode step, on ``input_ids`` or random tokens."""
    from sglang.benchmark.one_batch import _maybe_prepare_mlp_sync_batch

    if input_ids is None:
        vocab = model_runner.model_config.vocab_size
        input_ids = torch.randint(0, vocab, (len(batch.reqs),), device=batch.device)
    batch.input_ids = input_ids
    batch.prepare_for_decode()
    _maybe_prepare_mlp_sync_batch(batch, model_runner)


def graph_reps_for(extend_tokens: int) -> int:
    """How many times one op repeats inside a captured graph, by chunk size."""
    return max(1, min(GRAPH_REPS, GRAPH_TOKEN_BUDGET // max(extend_tokens, 1)))


def bench(fns, reps: int, inner: int = GRAPH_REPS, rounds: int = 1) -> float:
    """GPU-busy ms per call of `fns`, the fastest of `rounds` timing passes."""
    if not isinstance(fns, (list, tuple)):
        fns = [fns]

    def body():
        for i in range(inner):
            fns[i % len(fns)]()

    g = _capture(body)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            g.replay()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps * 1000.0 / inner)

    # Hand the graph's private pool back before the next op captures its own.
    del g
    torch.cuda.empty_cache()
    return best


def bench_eager(fn: Callable, reps: int, rounds: int = 1) -> float:
    """ms per call with the launches left in, the way an eager forward pays them."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps * 1000.0)
    return best


def _capture(fn: Callable) -> torch.cuda.CUDAGraph:
    """Record fn into a CUDA graph."""
    from sglang.srt.distributed.parallel_state import graph_capture

    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    with graph_capture() as ctx:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=getattr(ctx, "stream", None)):
            fn()
    torch.cuda.synchronize()
    return g


def _balanced_forward(experts) -> Callable:
    real, cache = experts.forward, {}

    def forward(*args, **kwargs):
        key = "topk_output"
        out = kwargs[key] if key in kwargs else args[1]
        ids = out.topk_ids
        if cache.get("shape") != ids.shape:
            n_shared = getattr(experts, "num_fused_shared_experts", 0)
            n_pick = ids.shape[1] - n_shared
            # Seeded: the draw sets how many distinct experts are read, which moves the byte count.
            gen = torch.Generator(device=ids.device)
            gen.manual_seed(MOE_ROUTING_SEED)
            pick = torch.rand(
                ids.shape[0],
                experts.num_experts - n_shared,
                device=ids.device,
                generator=gen,
            ).argsort(1)
            new = ids.clone()
            new[:, :n_pick] = pick[:, :n_pick].to(ids.dtype)
            cache["shape"], cache["ids"] = ids.shape, new
        out = out._replace(topk_ids=cache["ids"])
        if key in kwargs:
            return real(*args, **{**kwargs, key: out})
        return real(args[0], out, *args[2:], **kwargs)

    return forward


def _partitions(times: dict, kind: str, name: str) -> bool:
    """Whether ``kind|name`` can be summed without double-counting."""
    parent = name.rpartition(".")[0]
    return not parent or f"{kind}|{parent}" in times


def _git_rev(path: str) -> str:
    """`abc1234` for the checkout owning `path`, `+dirty` when it has edits."""
    import subprocess

    # Set safe.directory on the command: a bind-mounted repo owned by the host user is otherwise unreadable.
    def git(*cmd) -> str:
        try:
            done = subprocess.run(
                ("git", "-c", "safe.directory=*", "-C", path) + cmd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    rev = git("rev-parse", "--short", "HEAD")
    return rev + ("+dirty" if rev and git("status", "--porcelain") else "")


def versions() -> list[tuple]:
    """Every version that can move a number in the report."""
    import importlib

    def one(name: str) -> str:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            # Optional packages may fail during initialization.
            return "absent"
        parts = [getattr(mod, "__version__", None) or ""]
        if name in GIT_PACKAGES and mod.__file__:
            parts.append(_git_rev(os.path.dirname(mod.__file__)))
        return " ".join(p for p in parts if p) or "unknown"

    return [
        # The bench itself, so a change in this directory is distinguishable from a model regression.
        ("bench", _git_rev(os.path.dirname(os.path.abspath(__file__)))),
        ("sglang", one("sglang")),
        ("aiter", one("aiter")),
        ("torch", torch.__version__),
        ("rocm", torch.version.hip or "unknown"),
        ("triton", one("triton")),
        ("flydsl", one("flydsl")),
        ("tilelang", one("tilelang")),
    ]


def table(rows: list[tuple], headers: tuple, fmt: str = ".3f") -> str:
    """A markdown table, padded so the raw text lines up as well."""
    cells = [[format(c, fmt) if isinstance(c, float) else str(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        widths = [max(w, len(c)) for w, c in zip(widths, r)]

    def line(vals):
        return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths)) + " |"

    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(headers), rule] + [line(c) for c in cells])


def merged_skips(results) -> dict:
    """Every op any chunk could not replay, labelled with where it first fell."""
    out = {}
    for i, r in enumerate(results):
        for name, err in r.skipped.items():
            out.setdefault(name, f"chunk {i + 1}: {err}")
    return out


def trace_step(args, fb, mr, rank_print, tp_rank) -> None:
    """Kernel-level breakdown of one step: what the per-op rows are made of."""
    from torch.profiler import ProfilerActivity, profile

    mr.forward(fb)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        mr.forward(fb)
        torch.cuda.synchronize()

    # Self time, not total: device_time_total on an op includes the kernels it launched.
    rows = {}
    for e in prof.key_averages():
        if e.self_device_time_total <= 0:
            continue
        rows[e.key] = (e.self_device_time_total / 1000.0, e.count)
    total = sum(ms for ms, _ in rows.values())
    rank_print(f"kernels: {len(rows)} distinct, {sum(n for _, n in rows.values())} launches, {total:.3f} ms")
    for name, (ms, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:35]:
        rank_print(f"  {ms:8.3f} ms  {n:5d} x {ms * 1000 / n:7.2f} us  {name[:78]}")
    # Rank 0 only: export_chrome_trace renames <path>.tmp, so ranks sharing one path race.
    if tp_rank != 0:
        return
    path = Path(args.out or "trace.json")
    prof.export_chrome_trace(str(path))
    rank_print(f"trace -> {path.resolve()}")
