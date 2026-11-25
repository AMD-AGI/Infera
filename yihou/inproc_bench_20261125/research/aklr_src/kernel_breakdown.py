"""Kineto kernel breakdown for the standalone module replays."""

import contextlib
import importlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Collection
from pathlib import Path
from typing import NamedTuple

ROOT = "bench.module"
CHILD = "bench.child"
# Plumbing that only forwards a call; hiding it exposes the real caller.
_PLUMBING = (
    "/torch/_ops.py",
    "/torch/autograd/",
    "/torch/nn/modules/module.py",
    "/torch/profiler/",
    "/aiter/jit/core.py",
)
_PACKAGES = frozenset({"aiter", "sglang", "torch"})
# Substrings of a kernel name or of any enclosing frame that mark a cross-rank collective.
COLLECTIVE_MARKS = (
    "all_reduce",
    "allreduce",
    "cross_device_reduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "all_to_all",
    "alltoall",
    "nccl",
    "rccl",
    "quickreduce",
    "qr_all_reduce",
    "msccl",
    "epdispatch",
    "epcombine",
    "dp_gather",
)
TRUSTED_COLLECTIVE_SHARE = 0.30
TIE_GAP = 0.10
TOP_TARGETS = 5
TOP_SHOWN = 10


def contains_launch(scope, launch):
    return (
        (scope.get("pid"), scope.get("tid")) == (launch.get("pid"), launch.get("tid"))
        and scope["ts"] <= launch["ts"]
        and launch["ts"] + launch["dur"] <= scope["ts"] + scope["dur"]
    )


def python_api(event):
    """Turn a Kineto Python frame into `module.function:defline`."""
    name = event.get("name", "")
    if name.startswith("<") or "): " not in name:
        return None
    location, function = name.rsplit("): ", 1)
    raw_path, _, line = location.rpartition("(")
    if not line.isdigit():
        return None
    path = "/" + raw_path.replace("\\", "/").strip("/")
    if any(part in path for part in _PLUMBING) or not _PACKAGES.intersection(path.split("/")):
        return None
    for marker in ("/site-packages/", "/python/", "/sgl-workspace/"):
        if marker in path:
            path = path.split(marker, 1)[1]
            break
    module = path.strip("/").removesuffix(".py").replace("/", ".")
    return f"{module}.{function}:{line}"


def package_roots(profiles):
    """Absolute directory of every package named in the profiled APIs."""
    modules = importlib.import_module("sys").modules
    roots = {}
    for kernels in profiles:
        for _, api in kernels:
            for part in api.split(" → "):
                if "::" in part:
                    continue
                top = part.split(".", 1)[0]
                path = getattr(modules.get(top), "__file__", None)
                if path:
                    roots[top] = str(Path(path).parent)
    return dict(sorted(roots.items()))


def launch_api(launch, frames, cpu_ops):
    """Outer-to-inner call path for one launch and its innermost-first frame chain."""
    inside = [(frame, api) for frame, api in frames if contains_launch(frame, launch)]
    parts = [(frame["ts"], -frame["dur"], api) for frame, api in inside[:1]]
    operators = cpu_ops.get(launch.get("args", {}).get("External id"), ())
    if operators:
        operator = min(operators, key=lambda op: op["dur"])
        parts.append((operator["ts"], -operator["dur"], operator["name"]))
    parts.sort()
    chain = tuple(dict.fromkeys(api for _, api in inside))
    return " → ".join(part[-1] for part in parts) or "unavailable", chain


def add_sample(dest, key, ms, calls, frames) -> None:
    """Accumulate one `(ms, calls, frames)` sample into `dest[key]`."""
    old_ms, old_calls, old_frames = dest.get(key, (0.0, 0, ()))
    dest[key] = (
        old_ms + ms,
        old_calls + calls,
        tuple(dict.fromkeys((*old_frames, *frames))),
    )


def summarize(events):
    """Return {(kernel, Python/dispatcher API): (GPU ms, calls, frames)} from one Kineto trace."""
    roots, children, gpu, python_events, launches, cpu_ops = [], [], [], [], {}, {}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat", "")
        args = event.get("args", {})
        if category == "user_annotation":
            if event["name"] == ROOT:
                roots.append(event)
            elif event["name"] == CHILD:
                children.append(event)
        elif category in {"kernel", "gpu_memcpy", "gpu_memset"}:
            gpu.append(event)
        elif category == "python_function":
            python_events.append(event)
        elif category == "cpu_op" and "External id" in args:
            cpu_ops.setdefault(args["External id"], []).append(event)
        elif category.endswith("_runtime") or category in {"cuda_driver", "hip_driver"}:
            correlation = args.get("correlation")
            if correlation is not None:
                launches.setdefault(correlation, []).append(event)
    if len(roots) != 1 or not gpu:
        raise RuntimeError("Kineto needs one module scope and GPU kernel/copy events")
    # Resolve API names once; innermost first, so the first container wins.
    frames = sorted(
        ((frame, api) for frame in python_events if (api := python_api(frame))),
        key=lambda pair: pair[0]["dur"],
    )
    result = {}
    for event in gpu:
        matches = launches.get(event.get("args", {}).get("correlation"), ())
        if len(matches) != 1 or not contains_launch(roots[0], matches[0]):
            raise RuntimeError("Missing/ambiguous Kineto module launch correlation")
        launch = matches[0]
        if any(contains_launch(child, launch) for child in children):
            continue
        duration = event["dur"]
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError("Invalid Kineto GPU duration")
        api, chain = launch_api(launch, frames, cpu_ops)
        add_sample(result, (event["name"], api), duration / 1000.0, 1, chain)
    return result


@contextlib.contextmanager
def child_ranges(children):
    """Hook only children with their own timing row; fused work stays in the parent."""
    record_function = importlib.import_module("torch.profiler").record_function

    handles, active = [], []

    def enter(module, _args):
        scope = record_function(CHILD)
        scope.__enter__()
        active.append((module, scope))

    def leave(module, _args, _output):
        if active and active[-1][0] is module:
            active.pop()[1].__exit__(None, None, None)

    try:
        for module in dict.fromkeys(children):
            handles.append(module.register_forward_pre_hook(enter))
            handles.append(module.register_forward_hook(leave, always_call=True))
        yield
    finally:
        for handle in handles:
            handle.remove()
        while active:
            active.pop()[1].__exit__(None, None, None)


def profile_call(call, children=()):
    """One warmup + one annotated eager replay, separate from graph timing."""
    torch = importlib.import_module("torch")
    profiler = importlib.import_module("torch.profiler")
    activity = profiler.ProfilerActivity
    if activity.CUDA not in profiler.supported_activities():
        raise RuntimeError("Kineto GPU profiling is not available")
    with child_ranges(children):
        call()
        torch.cuda.synchronize()
        with profiler.profile(activities=[activity.CPU, activity.CUDA], with_stack=True) as prof:
            with profiler.record_function(ROOT):
                call()
            torch.cuda.synchronize()
    # Temporary and rank-local: no separate trace artifact.
    with tempfile.TemporaryDirectory(prefix="aklr-kineto-") as directory:
        path = Path(directory) / "trace.json"
        try:
            prof.export_chrome_trace(str(path))
            return summarize(json.loads(path.read_text())["traceEvents"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f"Cannot read Kineto kernel trace: {exc}") from exc


class HotKernel(NamedTuple):
    api: str
    ms: float  # calibrated to graph time
    kineto_ms: float  # raw Kineto ms
    calls: int
    share: float  # of the leaf roll-up
    hottest: str  # kernel symbol with the most calibrated ms
    collective: bool


class TimedRow(NamedTuple):
    """One timed leaf row's Kineto kernels, split into collective and compute ms."""

    graph_ms: float
    kernels: dict  # {(kernel, api): (ms, calls, frames)}
    collective: dict  # {(kernel, api): bool}
    collective_ms: float
    compute_ms: float

    @classmethod
    def of(cls, graph_ms: float, kernels: dict) -> "TimedRow":
        collective = {
            key: any(mark in " ".join((key[0], *frames)).lower() for mark in COLLECTIVE_MARKS)
            for key, (_, _, frames) in kernels.items()
        }
        collective_ms = sum(v[0] for key, v in kernels.items() if collective[key])
        compute_ms = sum(v[0] for v in kernels.values()) - collective_ms
        return cls(graph_ms, kernels, collective, collective_ms, compute_ms)

    @property
    def heavy(self) -> bool:
        total = self.collective_ms + self.compute_ms
        return self.collective_ms > TRUSTED_COLLECTIVE_SHARE * total

    @property
    def factor(self) -> float:
        if not self.compute_ms:
            return 1.0
        return max(self.graph_ms - self.collective_ms, 0.0) / self.compute_ms


class Ranking(NamedTuple):
    """Kernel groups of one step, hottest first, calibrated to the timed leaf rows."""

    kernels: list[HotKernel]
    factors: tuple[float, float]  # (lo, hi) over the trusted rows
    clamped: int  # heavy rows clamped into [lo, hi]

    @classmethod
    def of(cls, kernels_by_row: dict, leaves: dict[str, float], routed: Collection[str]) -> "Ranking":
        rows = [
            TimedRow.of(leaves[name], kernels) for name, kernels in kernels_by_row.items() if name in leaves
        ]
        rows = [row for row in rows if row.collective_ms + row.compute_ms > 0]
        trusted = [row.factor for row in rows if not row.heavy]
        lo, hi = (min(trusted), max(trusted)) if trusted else (1.0, 1.0)
        by_last = defaultdict(list)
        for api in routed:
            by_last[api.rpartition(".")[2]].append(api)
        ms_of, calls_of, kineto = Counter(), Counter(), Counter()
        symbols = defaultdict(Counter)
        for row in rows:
            factor = min(max(row.factor, lo), hi) if row.heavy else row.factor
            residual = max(row.graph_ms - factor * row.compute_ms, 0.0)
            for (kernel, api), (ms, calls, frames) in row.kernels.items():
                collective = row.collective[kernel, api]
                if not collective:
                    calibrated = factor * ms
                elif row.collective_ms:
                    calibrated = residual * ms / row.collective_ms
                else:
                    calibrated = 0.0
                hit = None
                # innermost first: exact routed name, else unique last component
                for frame in frames:
                    name = frame.rsplit(":", 1)[0]
                    same = [name] if name in routed else by_last.get(name.rpartition(".")[2], [])
                    if len(same) == 1:
                        hit = same[0]
                        break
                group = (
                    hit or (api if frames else f"unavailable:{kernel}"),
                    collective,
                )
                ms_of[group] += calibrated
                calls_of[group] += calls
                kineto[group] += ms
                symbols[group][kernel] += calibrated
        rollup = sum(leaves.values())
        hot = []
        for group, ms in ms_of.items():
            api, collective = group
            hottest = symbols[group].most_common(1)[0][0]
            share = ms / rollup if rollup else 0.0
            hot.append(HotKernel(api, ms, kineto[group], calls_of[group], share, hottest, collective))
        hot.sort(key=lambda k: (-k.ms, k.api))
        return cls(hot, (lo, hi), sum(row.heavy for row in rows))

    def targets(self, n: int = TOP_TARGETS) -> list[HotKernel]:
        """The first `n` non-collective groups, extended by the tie chain crossing rank `n`."""
        picked = [k for k in self.kernels if not k.collective]
        end = min(n, len(picked))
        while (
            0 < end < len(picked)
            and picked[end - 1].share - picked[end].share < TIE_GAP * picked[end - 1].share
        ):
            end += 1
        return picked[:end]

    def section(self, table: Callable) -> str:
        """The hottest groups as a markdown `table`; `target` marks the AKLR top-5 targets."""
        picked = self.targets()
        rows = [
            (
                i,
                f"`{k.api}`",
                k.ms,
                k.kineto_ms,
                f"{k.share * 100:.1f}%",
                k.calls,
                "yes" if k.collective else "",
                "yes" if k in picked else "",
                f"`{k.hottest}`",
            )
            for i, k in enumerate(self.kernels, 1)
            if i <= TOP_SHOWN or k in picked
        ]
        headers = (
            "rank",
            "API",
            "ms",
            "Kineto ms",
            "share",
            "calls",
            "collective",
            "target",
            "hottest kernel",
        )
        lo, hi = self.factors
        note = (
            "\n\nKineto ms of each timed leaf row scaled to its graph time (factors"
            f" {lo:.2f}..{hi:.2f}, {self.clamped} collective-heavy row(s) clamped);"
            " collectives take the row's residual. Targets are the top"
            f" {TOP_TARGETS} non-collective groups plus ties."
        )
        return table(rows, headers) + note
