###############################################################################
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
###############################################################################
# Provenance: adapted from AKLR `aklr/benchmarks/models/sglang_bench.py` @ 33689332
# (research/aklr_src/sglang_bench.py) and its SGLang DSv4.1 port (sglang_inproc/tool2_perop/sglang_bench.py).
# vLLM changes (vllm 0.34.0rc1.dev897+g9c1d45c21, image infera-vllm-dsv41:0007cd8087f1-nightly-9c1d45c2):
# - world / collectives from vllm.distributed (torchrun + external_launcher, one process per GPU);
# - graphs captured inside vllm.distributed.parallel_state.graph_capture(device), as vLLM's own
#   `vllm bench forward` (vllm/benchmarks/forward/capture.py) does;
# - the ambient state an op reads is vLLM's ForwardContext (attention metadata per layer prefix, slot
#   mappings, DP metadata, batch descriptor): snapshotted field by field at record time and re-entered with
#   vllm.forward_context.override_forward_context on every replay (the role glm52's attn_inputs + MLP flags
#   played in SGLang);
# - seeded balanced MoE routing: the AKLR draw (seed 1234, uniform distinct experts per token) is injected at
#   the router-logits boundary (GateLinear output -> fixed logits whose top-k is the draw), the mechanism of
#   vLLM's own routing fixture (vllm/benchmarks/forward/routing.py), because vLLM's MoE runner routes inside
#   its fused/monolithic kernels and exposes no `topk_output` argument to replace.

"""Shared bench plumbing: world/lockstep, graph/eager timing, forward-context replay, balanced routing,
versions and tables (vLLM)."""

import contextlib
import dataclasses
import hashlib
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch

# Repeating an op inside one graph amortises the ~10us host launch that would otherwise inflate small kernels.
GRAPH_REPS = 20

# Cap on tokens one graph pushes through; each repeat allocates its own activations in the graph pool.
GRAPH_TOKEN_BUDGET = 16384

# Any constant; it only has to be the same every run.
MOE_ROUTING_SEED = 1234

# Fixture logits: chosen experts score far above the rest under any monotonic scoring function (vLLM routing.py).
_CHOSEN_LOGIT = 4.0
_OTHER_LOGIT = -4.0


def require_aiter() -> None:
    """Fail loudly if aiter resolved to an empty namespace package."""
    import aiter

    if getattr(aiter, "__file__", None) is None:
        raise RuntimeError("aiter is an empty namespace package; check PYTHONPATH")


def load_tag(load_format) -> str:
    """`dummy` for a synthetic-weight run, empty for a real one."""
    return "dummy" if str(load_format) == "dummy" else ""


# ----------------------------------------------------------------------------------------------
# World: one process per GPU under torchrun; vLLM's world group spans DP x TP.


def world():
    from vllm.distributed.parallel_state import get_world_group

    return get_world_group()


def world_rank() -> int:
    try:
        return world().rank
    except Exception:  # noqa: BLE001 -- before init
        return int(os.environ.get("RANK", "0"))


def gather_obj(obj) -> list:
    """all_gather_object over the world's CPU group ([obj] on a single rank)."""
    try:
        w = world()
    except Exception:  # noqa: BLE001
        return [obj]
    if w.world_size == 1:
        return [obj]
    out = [None] * w.world_size
    torch.distributed.all_gather_object(out, obj, group=w.cpu_group)
    return out


def assert_lockstep(saved: dict) -> None:
    """Raise if the ranks recorded different op lists (timing them would unpair collectives and hang)."""
    names = list(saved)
    for rank, other in enumerate(gather_obj(names)):
        if other == names:
            continue
        at = next((i for i, (a, b) in enumerate(zip(names, other)) if a != b), min(len(names), len(other)))
        raise RuntimeError(
            f"world rank {rank} and this rank recorded different op lists ({len(other)} vs {len(names)} ops),"
            f" first difference at {at}: {other[at : at + 1]} vs {names[at : at + 1]}; timing them would hang"
        )


def agreed(ok: bool) -> bool:
    """``ok`` on every rank, or False on all of them."""
    return all(gather_obj(bool(ok)))


def world_times(times: dict, reduce: str) -> dict:
    """Per-key max (``reduce='max'``: collectives make the slowest rank the step) or min over ranks."""
    allv = gather_obj(times)
    pick = max if reduce == "max" else min
    return {k: pick(v[k] for v in allv if k in v) for k in times}


def barrier() -> None:
    try:
        w = world()
    except Exception:  # noqa: BLE001
        return
    if w.world_size > 1:
        torch.distributed.barrier(group=w.cpu_group)


# ----------------------------------------------------------------------------------------------
# Timing.


def graph_reps_for(extend_tokens: int) -> int:
    """How many times one op repeats inside a captured graph, by chunk size."""
    return max(1, min(GRAPH_REPS, GRAPH_TOKEN_BUDGET // max(extend_tokens, 1)))


def bench(fns, reps: int, inner: int = GRAPH_REPS, rounds: int = 1) -> float:
    """GPU-busy ms per call of `fns`, the fastest of `rounds` timing passes (AKLR `bench`)."""
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
    """Record fn into a CUDA graph inside vLLM's graph_capture (communicators set up for capture)."""
    from vllm.distributed.parallel_state import graph_capture

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    device = torch.device("cuda", torch.cuda.current_device())
    with graph_capture(device) as ctx:
        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=ctx.stream):
            fn()
    torch.cuda.synchronize()
    return g


# ----------------------------------------------------------------------------------------------
# Recorded state: argument snapshots and the forward context.


def snapshot(value, memo: dict):
    """Clone tensors so in-place mutation after the hook cannot change the recorded input. One clone per
    (tensor id, data_ptr, version) within one recording forward: an unchanged tensor handed to several ops is
    cloned once; an in-place write bumps the version and gets a new clone. Descends into tuples/lists/dicts,
    NamedTuples and dataclasses (vLLM's QuantizedActivation); other objects are shared."""
    if isinstance(value, torch.Tensor):
        key = (id(value), value.data_ptr() if value.numel() else 0, _mutation_tag(value, memo))
        if key not in memo:
            memo[key] = value.detach().clone()
        return memo[key]
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(snapshot(v, memo) for v in value))
    if isinstance(value, (list, tuple)):
        return type(value)(snapshot(v, memo) for v in value)
    if isinstance(value, dict):
        return {k: snapshot(v, memo) for k, v in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        try:
            return dataclasses.replace(
                value, **{f.name: snapshot(getattr(value, f.name), memo) for f in dataclasses.fields(value) if f.init}
            )
        except (TypeError, ValueError):
            return value
    return value


def _mutation_tag(t: torch.Tensor, memo: dict):
    """What changes when `t` is written in place: its version counter, or -- for inference tensors, which
    track none (vLLM runs under torch.inference_mode) -- a byte-sum fingerprint of its contents (one device
    reduction + sync, recording forward only); a non-contiguous inference tensor is cloned per call."""
    try:
        return ("v", t._version)
    except RuntimeError:
        pass
    if t.numel() and t.is_contiguous():
        try:
            return ("s", int(t.view(torch.uint8).sum(dtype=torch.int64)))
        except (RuntimeError, TypeError):
            pass
    return ("c", memo.get("__call__", 0))


def ctx_now():
    """(ForwardContext, its fields) at the call, or None outside a forward."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
    except ImportError:  # CPU tests without vLLM
        return None
    if not is_forward_context_available():
        return None
    c = get_forward_context()
    return c, dict(vars(c))


@contextlib.contextmanager
def ctx_scope(saved):
    """Re-enter a recorded ForwardContext with every field restored (models update it in place)."""
    if saved is None:
        yield
        return
    from vllm.forward_context import override_forward_context

    c, fields = saved
    for name, value in fields.items():
        setattr(c, name, value)
    with override_forward_context(c):
        yield


# ----------------------------------------------------------------------------------------------
# Seeded balanced routing (AKLR draw, injected at the router-logits boundary).


def aklr_draw(n_tokens: int, n_experts: int, top_k: int, device) -> torch.Tensor:
    """AKLR `_balanced_forward`'s draw: seeded uniform distinct experts per token ([n, top_k])."""
    gen = torch.Generator(device=device)
    gen.manual_seed(MOE_ROUTING_SEED)
    return torch.rand(n_tokens, n_experts, device=device, generator=gen).argsort(1)[:, :top_k]


def fixture_logits(ids: torch.Tensor, n_experts: int, dtype) -> torch.Tensor:
    """Logits whose top-k (any monotonic score, bias zeroed) is exactly ``ids``; distinct values, no ties."""
    n = ids.shape[0]
    gen = torch.Generator(device=ids.device)
    gen.manual_seed(MOE_ROUTING_SEED + 1)
    logits = torch.full((n, n_experts), _OTHER_LOGIT, dtype=torch.float32, device=ids.device)
    logits.scatter_(1, ids.long(), _CHOSEN_LOGIT)
    logits += torch.rand(logits.shape, device=ids.device, generator=gen) * 0.5
    return logits.to(dtype)


class BalancedRouting:
    """Install the AKLR draw on every given (moe_module, gate) pair; `remove()` restores the model.

    The gate GEMM still runs (its cost is real); its output is replaced by cached fixture logits, cached per
    row count as AKLR caches per ids shape. `e_score_correction_bias` is zeroed while installed (restored on
    remove) so the selection is exactly the draw. `fired[i]` counts gate calls, so a layer whose gate is
    bypassed (fused away) is reported instead of silently routing natively."""

    def __init__(self, pairs):
        self.pairs = pairs
        self.fired = [0] * len(pairs)
        self.hooks, self.biases = [], []
        for i, (module, gate) in enumerate(pairs):
            self.hooks.append(gate.register_forward_hook(self._hook(i, module, gate)))
            for owner in (gate, module):
                bias = getattr(owner, "e_score_correction_bias", None)
                if isinstance(bias, torch.Tensor) and all(bias is not b for b, _ in self.biases):
                    self.biases.append((bias, bias.detach().clone()))
                    bias.data.zero_()

    def _hook(self, i, module, gate):
        cache = {}
        n_experts = gate.weight.shape[0]
        top_k = top_k_of(module)

        def hook(_m, _inputs, output):
            self.fired[i] += 1
            logits = output[0] if isinstance(output, tuple) else output
            key = (logits.shape[0], logits.dtype)
            if key not in cache:
                ids = aklr_draw(logits.shape[0], n_experts, top_k, logits.device)
                cache[key] = fixture_logits(ids, n_experts, logits.dtype)
            return (cache[key], *output[1:]) if isinstance(output, tuple) else cache[key]

        return hook

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        for bias, saved in self.biases:
            bias.data.copy_(saved)
        self.hooks, self.biases = [], []


def top_k_of(module) -> int:
    for sub in module.modules():
        for name in ("n_activated_experts", "top_k"):
            v = getattr(sub, name, None)
            if isinstance(v, int):
                return v
    raise ValueError(f"no top_k on {type(module).__name__}")


# ----------------------------------------------------------------------------------------------
# Report helpers.


def _partitions(times: dict, kind: str, name: str) -> bool:
    """Whether ``kind|name`` can be summed without double-counting (its parent row was measured)."""
    parent = name.rpartition(".")[0]
    return not parent or f"{kind}|{parent}" in times


def _git_rev(path: str) -> str:
    import subprocess

    def git(*cmd) -> str:
        try:
            done = subprocess.run(
                ("git", "-c", "safe.directory=*", "-C", path) + cmd, capture_output=True, text=True, timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    rev = git("rev-parse", "--short", "HEAD")
    return rev + ("+dirty" if rev and git("status", "--porcelain", "--", ".") else "")


def _kit_hash() -> str:
    """sha256 over this tool's .py files (a node copy has no git)."""
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if "tests" in p.parts or "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def versions() -> list[tuple]:
    """Every version that can move a number in the report."""
    import importlib

    def one(name: str) -> str:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            return "absent"
        return getattr(mod, "__version__", None) or "unknown"

    return [
        ("bench", (_git_rev(os.path.dirname(os.path.abspath(__file__))) or "no git") + f" files-sha256 {_kit_hash()}"),
        ("vllm", one("vllm")),
        ("aiter", one("aiter")),
        ("torch", torch.__version__),
        ("rocm", torch.version.hip or "unknown"),
        ("triton", one("triton")),
        ("flydsl", one("flydsl")),
        ("tilelang", one("tilelang")),
        ("aklr", one("aklr")),
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


def kernel_rows(prof, limit: int = 35):
    """Self device time per op key of a torch profile, hottest first."""
    rows = {}
    for e in prof.key_averages():
        if e.self_device_time_total <= 0:
            continue
        rows[e.key] = (e.self_device_time_total / 1000.0, e.count)
    return sorted(rows.items(), key=lambda kv: -kv[1][0])[:limit], rows
