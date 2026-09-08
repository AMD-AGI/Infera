#!/usr/bin/env python3
"""Resolve each ranked device kernel to the Python frame that launched it.

Magpie's gap analysis answers "which kernel owns the GPU time". It cannot answer
"which source file do I edit", because a device symbol is often a compilation
artefact: `main_kernel` is what TileLang names every kernel it generates, and a
Triton kernel's symbol is assembled from its own tuning constants. The next stage
of this pipeline (`analyze-demo`'s `identify`) has to name a framework-level
entry point, and without a call stack its only evidence is a grep over the
symbol name -- which cannot distinguish a definition from a test that mentions
it, and has nothing at all to offer for `main_kernel`.

A `with_stack` capture carries the answer. torch records the enclosing
`python_function` chain for every launch, so the frame that called the kernel is
in the trace next to the kernel itself. This recovers it.

**The output field names are Hyperloom's `LauncherFrame`, not this package's.**
`Hyperloom/src/hyperloom/agents/kernel/tools/_trace_launcher_resolver.py` defines
`source_file` / `line` / `function` / `sample_count` / `launch_api`, and
`analyze-demo`'s `kernel_table` kind reserves a `launcher` block under exactly
those names (its DESIGN.md section 4.4). Writing them here means neither side
needs a translation layer.

**The algorithm is Hyperloom's, reimplemented rather than imported.** Hyperloom
is a separate repository that is not on this cluster's compute nodes, and every
body in this package is stand-alone package data run as a subprocess. The
duplication is bounded to the correlation rule and the two frame filters below,
and is the same bounded duplication `assets/lib/redact.py` makes of the seal's
allow-list. Two things are deliberately *not* copied:

- **TraceLens elision handling.** Hyperloom matches a symbol truncated with a
  trailing `...`, because TraceLens shortens long mangled names. Magpie does
  not: measured on the sample gap analysis, a 700-character Tensile symbol and a
  full `_ZN5aiter...` mangling both arrive complete. Matching a prefix is how a
  wrong kernel gets bound to a right-looking frame, so it is left out.
- **The `codes` / patchability classification.** That is the consumer's job.

Usage:

    launchers.py --csv <gap_analysis.csv> --stacks <dir of *.trace.json.gz> \
                 --out <launchers.json> [--top-n 50] [--max-files 2] [--samples 3]
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import trace_stream  # noqa: E402 — the path insert above is what makes it importable

# --------------------------------------------------------------------------- #
# Kineto categories. Only these three are read.

_CAT_KERNEL = "kernel"
_CAT_RUNTIME = "cuda_runtime"
_CAT_PYTHON = "python_function"

#: Substring shared by every kernel-dispatch runtime API on HIP and CUDA
#: (`hipModuleLaunchKernel`, `hipExtModuleLaunchKernel`, `cudaLaunchKernel`).
#: Correlated runtime calls that are not launches -- memcpy, synchronize, malloc
#: -- can never be a kernel's dispatch point, and keeping them would size the
#: probe map by total runtime events rather than by launches.
_LAUNCH_MARKER = "launch"

#: Shared by `hipGraphLaunch` / `cudaGraphLaunch` / `cuGraphLaunch`. A graph
#: replay has one Python frame for the whole graph and none per kernel, so a
#: replay probe would resolve every kernel in the graph to `replay`. This round
#: runs with decode graphs off, so eager probes exist for everything that
#: matters; the filter is here because prefill capture-and-replay does not
#: depend on that flag.
_GRAPH_LAUNCH_MARKER = "graphlaunch"

#: Reaching this frame means the launch came from a graph replay whatever the
#: API name said.
_GRAPH_REPLAY_MARKER = "torch/cuda/graphs.py"

#: A `python_function` event's name, which torch writes as `<path>(<line>): <func>`.
_FRAME_RE = re.compile(r"^(?P<path>.+?)\((?P<line>\d+)\):\s*(?P<func>.+)$")

#: Frames between the user's call site and the launch. Every JIT toolchain
#: contributes one, and it is never the kernel's source:
#:   triton   -> `triton/runtime/jit.py: run`
#:   aiter    -> `aiter/jit/core.py: wrapper`
#:   FlyDSL   -> `flydsl/compiler/jit_executor.py: __call__`
#:   TileLang -> `tilelang/jit/kernel.py: __call__`
#: Admitting one collapses every kernel that toolchain compiled onto a single
#: file, so two different MoE GEMMs would both resolve to `jit_executor.py`.
#:
#: **`tilelang/jit/` and `tilelang/engine/` are this package's addition to
#: Hyperloom's list, and they are the entries that matter most for GLM-5.3-Flash.**
#: The model serves DSA prefill and decode through TileLang, whose device symbol
#: is `main_kernel` for every kernel it generates -- 3.29% of GPU time in the
#: sample profile under one meaningless name. Without these two, that kernel
#: resolves to TileLang's own dispatcher, which is neither editable nor specific
#: to it. With them, the walk continues outward to the sglang backend that called
#: it, which is the framework-level entry `analyze-demo`'s DESIGN.md section 4.4
#: says a workset's unit actually is.
#:
#: Scope matters, and it is the same distinction FlyDSL needs: only the
#: toolchain's own dispatch is plumbing. A kernel *written* in TileLang or FlyDSL
#: is a legitimate target, so `aiter/ops/flydsl/` and an authored `*_tilelang.py`
#: under sglang must stay visible -- and `aiter/ops/flydsl/` is where the sample
#: profile's two hottest routable kernels live.
_SKIP_FRAME_RE = re.compile(
    r"(?:"
    r"triton/runtime/|triton/backends/|triton/compiler/"
    r"|aiter/jit/"
    r"|flydsl/compiler/"
    r"|tilelang/jit/|tilelang/engine/"
    r"|torch/nn/modules/module\.py"
    r"|torch/utils/_contextlib\.py"
    r"|torch/_dynamo/|torch/_inductor/"
    r"|/_ops\.py"
    r"|^<"
    r")"
)

#: Generic decorator frames. A path filter cannot catch these: a guard or a
#: no-grad decorator lives in an ordinary repository file, and its body is still
#: plumbing. The function name is the stable signal.
_WRAPPER_FUNC_NAMES = frozenset(
    {
        "call",
        "custom_wrapper",
        "decorate",
        "decorate_context",
        "handle_torch_function",
        "inner",
        "outer_wrapper",
        "wrapped",
        "wrapper",
        "wrapper_custom",
        "_fn",
        "_inner",
        "_wrapped",
        "_wrapper",
    }
)

#: Probes collected per kernel before voting. Oversampled against `--samples`
#: because a probe can land on a stack that resolves to nothing.
_PROBE_OVERSAMPLE = 4

# --------------------------------------------------------------------------- #
# Container roots, and the two shapes a recorded frame path comes in.
#
# **torch writes a frame path with the longest matching `sys.path` entry stripped
# off the front, so the same capture yields absolute and relative paths side by
# side.** Measured in the engine image on 2026-09-01 (`temp/manual/FINDINGS.md`):
#
#     aiter/ops/triton/softmax.py(10): softmax
#     torch/profiler/profiler.py(812): __enter__
#     /sgl-workspace/sglang/python/sglang/srt/utils/common.py(3341): next_power_of_2
#
# `/sgl-workspace/aiter` and the site-packages directory are both on `sys.path`,
# so those two came out relative. sglang is installed **editable** -- `sys.path`
# carries `__editable__.sglang-0.5.18...finder.__path_hook__` and not the real
# `/sgl-workspace/sglang/python` -- so nothing matched and its frame came out
# absolute. That is not an edge case to tolerate: the GLM-5.3-Flash tree *is* the
# editable PR #36507 overlay, so the frames this stage most wants are exactly the
# ones that arrive absolute.
#
# The form is also stable against the process' working directory, which was
# checked rather than assumed: the same frames were recorded from `cwd=/` and
# `cwd=/tmp` unchanged.
#
# **This table must agree with `analyze-demo/assets/lib/container_roots.yaml`,
# and the two are separate files on purpose** -- see that file's own header. A
# frame is published as a placeholder plus a relative path, never as an absolute
# one, because a handoff should not name an absolute host path outside a small
# allow-list and `/sgl-workspace/` is not on it. Note the seal does NOT enforce
# this (`store.py`: `locality.check` is not called); `assets/lib/redact.py` does,
# which is why the shape below has to be right at the producer.
#
# Splitting rather than substituting is what makes it pass redact.py: a
# substituted `@SGLANG_ROOT@/srt/layers/moe/fused_moe.py` is one string with two
# path segments in it and `locality.py`'s regex -- which redact.py reuses -- would
# offer `/srt/layers/...` as a fresh candidate, while a relative path has no
# leading slash and cannot match.
CONTAINER_ROOTS: tuple[tuple[str, str, str], ...] = (
    # (owner, container path, placeholder)
    ("sglang", "/sgl-workspace/sglang/python/sglang", "SGLANG_ROOT"),
    ("sgl_kernel", "/sgl-workspace/sglang/sgl-kernel", "SGL_KERNEL_ROOT"),
    ("aiter", "/sgl-workspace/aiter", "AITER_ROOT"),
    ("tilelang", "/opt/tilelang", "TILELANG_ROOT"),
)

#: Leading path segment of a `sys.path`-relative frame -> the owner it belongs
#: to. Only needed for the relative form; the absolute form is matched exactly
#: against `CONTAINER_ROOTS` and needs no guessing.
#:
#: Which `sys.path` entry a relative path was stripped against is not recoverable
#: from the string -- `aiter/ops/...` is consistent with an entry of
#: `/sgl-workspace/aiter` and with one of `/sgl-workspace`. So this names the
#: owner and `path_form` below says the path still needs binding to a file. The
#: consumer is where that is settled, because the consumer has the repository
#: checked out and can test which candidate exists; guessing here would be a
#: producer asserting something it cannot see.
RELATIVE_OWNERS: dict[str, str] = {
    "aiter": "aiter",
    "sglang": "sglang",
    "tilelang": "tilelang",
    "sgl_kernel": "sgl_kernel",
    "sgl-kernel": "sgl_kernel",
}

#: Leading segments that are a real frame and never an optimisation target:
#: third-party runtime and the standard library. Reported as such rather than as
#: an unknown root, so a genuinely missing root stays visible in `unmapped`.
FOREIGN_PREFIXES = frozenset({"torch", "triton", "numpy", "vllm", "transformers", "ray"})

#: How `source_file` should be read.
FORM_ABSOLUTE = "container_absolute"
FORM_RELATIVE = "sys_path_relative"


def _classify_frame(path: str) -> tuple[str, str, str, str, str]:
    """`(owner, placeholder, relative, path_form, reject_reason)` for a frame path.

    A non-empty `reject_reason` means the frame is real and this package will not
    publish a location for it. That is a supported outcome: the kernel simply
    falls to the consumer's name-search tier, and recording the reason is what
    keeps a missing container root visible instead of showing up as a quietly
    lower resolve count.
    """
    path = (path or "").strip()
    if not path:
        return "", "", "", "", "the frame carries no path"

    if path.startswith("/"):
        for owner, root, placeholder in sorted(CONTAINER_ROOTS, key=lambda row: -len(row[1])):
            stem = root.rstrip("/")
            if path == stem or path.startswith(stem + "/"):
                return owner, "${%s}" % placeholder, path[len(stem) + 1:], FORM_ABSOLUTE, ""
        return "", "", "", "", "absolute path outside every known container root"

    head = path.split("/", 1)[0]
    if head in FOREIGN_PREFIXES:
        return "", "", "", "", f"launched from {head}, which is not an editable target here"
    owner = RELATIVE_OWNERS.get(head)
    if owner is None:
        # A single-segment path is a script on `sys.path[0]`, and anything else
        # unrecognised is a package this table does not know about.
        return "", "", "", "", f"relative path under an unrecognised root {head!r}"

    placeholder = next(
        ("${%s}" % row[2] for row in CONTAINER_ROOTS if row[0] == owner),
        "",
    )
    return owner, placeholder, path, FORM_RELATIVE, ""


# --------------------------------------------------------------------------- #
# Pass 1: which (pid, tid, ts) to look at.


def _has_symbol_boundaries(event_name: str, name: str) -> bool:
    """Whether `name` occurs in `event_name` as one complete identifier token.

    What this admits is a decoration the runtime added -- `<...>` template
    arguments, a trailing `.kd` -- and what it rejects is a longer identifier
    that merely starts with the same characters, so `moe_gemm1` cannot bind
    `moe_gemm1_0`.
    """
    start = 0
    while True:
        start = event_name.find(name, start)
        if start < 0:
            return False
        end = start + len(name)
        before_ok = start == 0 or not (
            event_name[start - 1].isalnum() or event_name[start - 1] == "_"
        )
        after_ok = end == len(event_name) or not (
            event_name[end].isalnum() or event_name[end] == "_"
        )
        if before_ok and after_ok:
            return True
        start += 1


def _match_kernel(event_name: str, exact: frozenset[str], wanted: tuple[str, ...]) -> str | None:
    """The wanted kernel this event is, or None.

    Exact identity is a set lookup and is the normal case, because Magpie writes
    the trace's own `name` field through unchanged. The boundary scan is the
    fallback for a runtime that decorated the symbol, and it costs a pass over
    `wanted` only for events that were not an exact hit.

    A tie between two equally specific boundary matches resolves to None. The
    alternative -- returning whichever the iteration reached first -- made the
    binding depend on dict order, and a wrong binding here is worse than no
    binding, because it reaches the consumer ahead of its own name search.
    """
    if event_name in exact:
        return event_name
    hits = [name for name in wanted if _has_symbol_boundaries(event_name, name)]
    if not hits:
        return None
    longest = max(len(name) for name in hits)
    finalists = [name for name in hits if len(name) == longest]
    return finalists[0] if len(finalists) == 1 else None


def _event_pid(event: dict) -> int:
    """The event's process id, 0 when the trace omits one."""
    try:
        return int(event.get("pid") or 0)
    except (TypeError, ValueError):
        return 0


def _collect_probes(
    trace: Path,
    wanted: frozenset[str],
    scan_order: tuple[str, ...],
    budget: int,
    errors: list[str],
) -> dict[str, list[tuple[int, int, float, str]]]:
    """Pass 1: pick `(pid, tid, ts, launch_api)` probes per kernel.

    Reads only `kernel` and `cuda_runtime`, which are order 10^4 events against
    the 10^6 `python_function` events pass 2 walks.

    **Paired on `correlation` alone, never on `(pid, correlation)`.** Kineto
    records the two sides of a launch against different processes -- the kernel
    against the device, the runtime call against the host -- so a pid-qualified
    key never matches on a real trace. `correlation` is the field designed to
    span that boundary. Its cost is that a merged multi-rank trace restarts the
    ids, which is handled by detecting the collision and dropping the id rather
    than by keying on a pid that means different things on either side.
    """
    corr_kernel: dict[object, str] = {}
    runtimes: dict[object, tuple[int, int, float, str]] = {}
    ambiguous: set[object] = set()

    with trace_stream.open_trace(trace) as handle:
        for event in trace_stream.stream_events(handle, errors=errors):
            category = event.get("cat")
            if category == _CAT_KERNEL:
                matched = _match_kernel(str(event.get("name") or ""), wanted, scan_order)
                if matched is None:
                    continue
                args = event.get("args")
                corr = args.get("correlation") if isinstance(args, dict) else None
                if corr is None:
                    continue
                held = corr_kernel.get(corr)
                if held is not None and held != matched:
                    # Two kernels answer to one id: a merged trace reused it.
                    # Neither binding can be trusted.
                    ambiguous.add(corr)
                else:
                    corr_kernel[corr] = matched
            elif category == _CAT_RUNTIME:
                api = str(event.get("name") or "")
                # Kernel and runtime events arrive in no guaranteed order, so
                # this cannot filter on corr_kernel yet. Gating on the launch
                # marker bounds the map by launches instead.
                if _LAUNCH_MARKER not in api.lower():
                    continue
                args = event.get("args")
                corr = args.get("correlation") if isinstance(args, dict) else None
                if corr is None:
                    continue
                if corr in runtimes:
                    ambiguous.add(corr)
                    continue
                runtimes[corr] = (
                    _event_pid(event),
                    int(event.get("tid") or 0),
                    float(event.get("ts") or 0.0),
                    api,
                )

    probes: dict[str, list[tuple[int, int, float, str]]] = defaultdict(list)
    for corr, kernel in corr_kernel.items():
        if corr in ambiguous:
            continue
        runtime = runtimes.get(corr)
        if runtime is None:
            continue
        if _GRAPH_LAUNCH_MARKER in runtime[3].lower():
            continue
        if len(probes[kernel]) < budget:
            probes[kernel].append(runtime)
    return {kernel: samples for kernel, samples in probes.items() if samples}


# --------------------------------------------------------------------------- #
# Pass 2: the frames enclosing those timestamps.


def _collect_frames(
    trace: Path,
    probes: dict[str, list[tuple[int, int, float, str]]],
    errors: list[str],
) -> dict[tuple[int, int, float], list[tuple[float, float, str]]]:
    """Pass 2: the `python_function` frames whose span covers each probe.

    Only enclosing frames are retained, so memory is proportional to the probe
    count rather than to the trace. Probes are keyed by `(pid, tid, ts)`: a
    merged multi-rank trace reuses thread ids, and without the pid one rank's
    frames would answer for another rank's launch.
    """
    by_thread: dict[tuple[int, int], list[float]] = defaultdict(list)
    for samples in probes.values():
        for pid, tid, stamp, _api in samples:
            by_thread[(pid, tid)].append(stamp)
    stamps_of = {key: sorted(set(values)) for key, values in by_thread.items()}

    enclosing: dict[tuple[int, int, float], list[tuple[float, float, str]]] = defaultdict(list)
    with trace_stream.open_trace(trace) as handle:
        for event in trace_stream.stream_events(handle, errors=errors):
            if event.get("cat") != _CAT_PYTHON:
                continue
            key = (_event_pid(event), int(event.get("tid") or 0))
            stamps = stamps_of.get(key)
            if not stamps:
                continue
            start = float(event.get("ts") or 0.0)
            duration = float(event.get("dur") or 0.0)
            low = bisect.bisect_left(stamps, start)
            high = bisect.bisect_right(stamps, start + duration)
            if low >= high:
                continue
            name = str(event.get("name") or "")
            for stamp in stamps[low:high]:
                # `dur` is kept alongside `start`: profiler timestamps are
                # microsecond-granular, so adjacent frames on a fast call chain
                # routinely share a `ts` and start time alone cannot order them.
                enclosing[(key[0], key[1], stamp)].append((start, duration, name))
    return enclosing


def _innermost_user_frame(frames: list[tuple[float, float, str]]) -> tuple[str, int, str] | None:
    """The innermost frame that is a user `.py` call site, or None.

    Ordered innermost-first by `(start, -dur)`: a later start is deeper, and
    among frames that started in the same microsecond the narrower span is the
    nested one. Sorting on start alone leaves same-`ts` frames in trace write
    order, which picks an arbitrary nesting level.
    """
    for _start, _duration, name in sorted(frames, key=lambda item: (item[0], -item[1]), reverse=True):
        if _GRAPH_REPLAY_MARKER in name:
            return None
        if _SKIP_FRAME_RE.search(name):
            continue
        match = _FRAME_RE.match(name)
        if not match:
            continue
        path = match.group("path").strip()
        if not path.endswith(".py"):
            continue
        function = match.group("func").strip()
        if function in _WRAPPER_FUNC_NAMES:
            continue
        return path, int(match.group("line")), function
    return None


# --------------------------------------------------------------------------- #


def resolve(
    traces: list[Path],
    wanted: frozenset[str],
    *,
    samples: int = 3,
    max_files: int = 2,
    log=print,
) -> tuple[dict[str, dict], list[str], list[dict]]:
    """`(resolved, notes, unmapped)` for a set of kernel names.

    `max_files` defaults to 2 for the reason Hyperloom's own ceiling is 2: every
    rank runs the same Python, so one rank's trace resolves everything that is
    resolvable and the second is corroboration. Reading eight costs eight pairs
    of streaming passes for no new frames.

    A kernel unresolved in the first file is retried in the second rather than
    written off: the ranks run the same Python but not the same requests, so a
    probe that landed on a graph replay on one rank can land on an eager launch
    on another. What that retry must not do is report the same finding twice,
    which is what `_note` and the `unmapped` key below are for.
    """
    resolved: dict[str, dict] = {}
    unmapped: dict[tuple, dict] = {}
    notes: list[str] = []
    budget = max(1, int(samples)) * _PROBE_OVERSAMPLE

    def _note(message: str) -> None:
        """Record a diagnostic once, however many ranks reach the same one."""
        if message not in notes:
            notes.append(message)

    for trace in traces[: max(1, int(max_files))]:
        remaining = frozenset(wanted - set(resolved))
        if not remaining:
            break
        errors: list[str] = []
        try:
            probes = _collect_probes(trace, remaining, tuple(sorted(remaining)), budget, errors)
            frames = _collect_frames(trace, probes, errors) if probes else {}
        except (EOFError, OSError, ValueError, TypeError, AttributeError) as exc:
            # Fail soft per file: one clipped trace must not take the tier down.
            _note(f"{trace.name}: {type(exc).__name__}: {exc}")
            continue
        for error in errors:
            _note(f"{trace.name}: {error}")

        log(f"[launchers] {trace.name}: probes for {len(probes)}/{len(remaining)} kernel(s)")

        for kernel, picked in probes.items():
            votes: Counter[tuple[str, int, str]] = Counter()
            apis: Counter[str] = Counter()
            for pid, tid, stamp, api in picked:
                if sum(votes.values()) >= samples:
                    break
                found = _innermost_user_frame(frames.get((pid, tid, stamp)) or [])
                if found is None:
                    continue
                votes[found] += 1
                apis[api] += 1
            if not votes:
                continue
            ranked = votes.most_common(2)
            (path, line, function), count = ranked[0]
            # A plurality is not agreement. With probes split across distinct
            # frames, `most_common` returns whichever was counted first, which
            # is trace order rather than evidence, so a tie at the top is left
            # unresolved for the consumer's name search to answer.
            if len(ranked) > 1 and ranked[1][1] == count:
                _note(f"{kernel[:60]}: probes disagree ({count} vs {ranked[1][1]}); left unresolved")
                continue

            owner, placeholder, relative, form, rejected = _classify_frame(path)
            if rejected:
                # Named, not published. Only the basename is recorded: it carries
                # the useful part and has no leading slash for the seal to reject.
                unmapped[(kernel, Path(path).name, line, function)] = {
                    "kernel": kernel,
                    "file_name": Path(path).name,
                    "function": function,
                    "line": line,
                    "reason": rejected,
                }
                continue

            resolved[kernel] = {
                # Hyperloom `LauncherFrame`, with the absolute root split off.
                "source_file": relative,
                "line": line,
                "function": function,
                "sample_count": count,
                "launch_api": apis.most_common(1)[0][0] if apis else "",
                # What the relative path is relative to. `identify` reads this as
                # `image_repo_path`; the expansion is in the consumer's
                # `container_roots.yaml`.
                "container_root": placeholder,
                "owner": owner,
                # `container_absolute` means `source_file` is exactly relative to
                # `container_root`. `sys_path_relative` means torch stripped a
                # `sys.path` entry this package cannot identify, so the consumer
                # has to bind it to a file under the checkout it indexed -- it can
                # test which candidate exists and this producer cannot.
                "path_form": form,
            }

    return resolved, notes, list(unmapped.values())


def _wanted_from_csv(path: Path, top_n: int) -> list[str]:
    """The `top_n` kernel names with the most self CUDA time.

    Bounded rather than "every row" because `_match_kernel`'s boundary scan runs
    per kernel event per unmatched name. The bound is generous against what the
    consumer selects: `analyze-demo`'s `rank` picks its top N from the *routable*
    bucket only, and collectives hold 79% of the time in the sample profile, so a
    routable winner can sit well down the raw ranking.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Name" not in reader.fieldnames:
            raise SystemExit(f"launchers: {path} has no 'Name' column (header: {reader.fieldnames})")
        rows = [row for row in reader if (row.get("Name") or "").strip()]

    def self_us(row: dict) -> float:
        try:
            return float((row.get("Self CUDA total (us)") or "0").strip().replace(",", ""))
        except ValueError:
            return 0.0

    rows.sort(key=self_us, reverse=True)
    return [(row["Name"] or "").strip() for row in rows[:top_n]]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", required=True, type=Path, help="gap_analysis.csv")
    parser.add_argument("--stacks", required=True, type=Path, help="directory of with_stack traces")
    parser.add_argument("--out", required=True, type=Path, help="launchers.json to write")
    parser.add_argument("--top-n", type=int, default=50, help="kernels to resolve, by self CUDA time")
    parser.add_argument("--max-files", type=int, default=2, help="trace files to read")
    parser.add_argument("--samples", type=int, default=3, help="agreeing probes per kernel")
    args = parser.parse_args(argv)

    wanted = _wanted_from_csv(args.csv, args.top_n)
    traces = sorted(args.stacks.glob("*.trace.json.gz")) if args.stacks.is_dir() else []

    # **Absence is reported, not fatal.** A run whose stack window failed still
    # produces a usable ranking; what it must not do is leave the consumer
    # unable to tell "no launcher was resolvable" from "nobody looked".
    if not traces:
        args.out.write_text(
            json.dumps(
                {
                    "available": False,
                    "reason": f"no *.trace.json.gz under {args.stacks.name}/",
                    "wanted": len(wanted),
                    "resolved": 0,
                    "launchers": {},
                    "unmapped": [],
                    "notes": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"launchers: no stack traces under {args.stacks}; nothing to resolve")
        return 0

    resolved, notes, unmapped = resolve(
        traces,
        frozenset(wanted),
        samples=args.samples,
        max_files=args.max_files,
    )

    by_owner: dict[str, int] = {}
    by_form: dict[str, int] = {}
    for frame in resolved.values():
        by_owner[frame["owner"]] = by_owner.get(frame["owner"], 0) + 1
        by_form[frame["path_form"]] = by_form.get(frame["path_form"], 0) + 1

    args.out.write_text(
        json.dumps(
            {
                "available": True,
                "reason": "",
                "wanted": len(wanted),
                "resolved": len(resolved),
                "trace_files_read": min(len(traces), max(1, args.max_files)),
                "trace_files_present": len(traces),
                "samples_per_kernel": args.samples,
                "by_owner": by_owner,
                # How many frames arrived in each of the two shapes torch writes.
                # A consumer reads this to know how much of the table it has to
                # bind to a file itself; see `path_form` on each record.
                "by_path_form": by_form,
                "launchers": resolved,
                # Resolved to a frame that is real and outside every known
                # container root. Recorded so a new root shows up as a named
                # gap rather than as a silently lower resolve count.
                "unmapped": unmapped,
                "notes": notes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"launchers: resolved {len(resolved)}/{len(wanted)} kernel(s) from "
        f"{min(len(traces), max(1, args.max_files))} of {len(traces)} trace file(s); "
        f"by owner {by_owner or '{}'}, by form {by_form or '{}'}, "
        f"{len(unmapped)} not published"
    )
    for kernel, frame in resolved.items():
        print(
            f"  {frame['container_root']}/{frame['source_file']}:{frame['line']} "
            f"{frame['function']} <- {kernel[:56]}"
        )
    for row in unmapped:
        print(f"  (unpublished) {row['file_name']}:{row['line']} {row['function']}: {row['reason']}")
    for note in notes:
        print(f"  (note) {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
