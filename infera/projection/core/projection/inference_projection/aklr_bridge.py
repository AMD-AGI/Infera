###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""AKLR kernels for a replay's kernels, and what they buy end to end.

AKLR (github.com/AMD-AGI/AKLR) keeps kernels as schemas: an operator contract
pinned to one shape (``gemm_n5120_k2880``), the workloads it is scored on, and
the kernel's sources. Its ``forge_task`` harness hands a schema without a
kernel to KernelForge's agents and scores each candidate against AKLR's
baseline (aiter), workload by workload.

This module joins that to a replay's kernel tasks (:mod:`kernel_tasks`):

* ``targets``: the tasks an AKLR contract can express, one target per schema,
  its workloads the shapes the replay runs, heaviest first;
* ``schemas``: each target written as a kernel-less AKLR schema in an AKLR
  checkout, by AKLR's own code (it needs torch, so run it in AKLR's image);
* ``campaign``: the commands that prepare and run one target's KernelForge
  campaign;
* ``speedups``: the campaigns' kept kernels, or AKLR's committed ones, as an
  ``inferasim.kernel_speedups`` file, which ``--kernel-speedups`` charges to
  the replay's steps.

A speedup is measured on the GPU, the kernel's time against the kernel the
serving stack runs, and applied to InferaSim's time for that shape: it is the
ratio that carries over, not the absolute time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import sys
from dataclasses import dataclass, field

from . import kernel_speedups as ksp

TARGETS_SCHEMA = "inferasim.aklr_targets"

# AKLR's contract per (task kind, dtype), and how the task's dims name its axes.
_CONTRACTS = {
    ("gemm", "bf16"): {
        "contract": "gemm@1.0",
        "op_dir": "gemm",
        "var": ("m", "m"),
        "const": (("n", "n"), ("k", "k")),
        "name": "gemm_n{n}_k{k}",
        "dtypes": {"a": "bfloat16", "b": "bfloat16", "bias": None, "out": "bfloat16"},
        "attrs": {"trans_b": True, "has_bias": False},
    },
    ("gemm", "fp16"): {
        "contract": "gemm@1.0",
        "op_dir": "gemm",
        "var": ("m", "m"),
        "const": (("n", "n"), ("k", "k")),
        "name": "gemm_n{n}_k{k}_fp16",
        "dtypes": {"a": "float16", "b": "float16", "bias": None, "out": "float16"},
        "attrs": {"trans_b": True, "has_bias": False},
    },
    ("norm", "bf16"): {
        "contract": "add_rms_norm@1.0",
        "op_dir": "rms_norm",
        "var": ("tokens", "n_tokens"),
        "const": (("width", "hidden_size"),),
        "name": "add_rms_norm_hidden_size{hidden_size}",
        "dtypes": {
            "x": "bfloat16",
            "residual": "bfloat16",
            "weight": "bfloat16",
            "normalized": "bfloat16",
            "residual_out": "bfloat16",
        },
        "attrs": {},
        # 1e-5 as float32 holds it
        "scalars": {"epsilon": 9.999999747378752e-06},
        "starting_kernel": True,
    },
}
_LAYOUT = "row_major"
_GPU_TARGETS = {"mi355x": {"arch": "gfx950", "hardware": "MI355X"}}


@dataclass
class Workload:
    value: int
    calls: int = 0
    est_us: float = 0.0
    ops: list[str] = field(default_factory=list)

    @property
    def total_ms(self) -> float:
        return self.calls * self.est_us / 1000.0


@dataclass
class Target:
    name: str
    contract: str
    op_dir: str
    kind: str
    dtype: str
    var_axis: str
    var_dim: str
    const: dict[str, int]
    const_dims: dict[str, int]
    workloads: list[Workload]
    share_pct: float = 0.0

    @property
    def ops(self) -> list[str]:
        return sorted({op for w in self.workloads for op in w.ops})

    @property
    def total_ms(self) -> float:
        return sum(w.total_ms for w in self.workloads)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "contract": self.contract,
            "op_dir": self.op_dir,
            "kind": self.kind,
            "dtype": self.dtype,
            "ops": self.ops,
            "axes": {**self.const, self.var_axis: [w.value for w in self.workloads]},
            "task_dims": dict(self.const_dims),
            "var_dim": self.var_dim,
            "share_pct": round(self.share_pct, 3),
            "est_total_ms": round(self.total_ms, 3),
            "workloads": [
                {
                    "axes": {self.var_axis: w.value},
                    "calls": w.calls,
                    "est_us": round(w.est_us, 3),
                    "ops": sorted(set(w.ops)),
                }
                for w in self.workloads
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Target:
        spec = _spec(d["kind"], d["dtype"])
        var_axis = spec["var"][1]
        const = {k: v for k, v in d["axes"].items() if k != var_axis}
        return cls(
            name=d["name"],
            contract=d["contract"],
            op_dir=d["op_dir"],
            kind=d["kind"],
            dtype=d["dtype"],
            var_axis=var_axis,
            var_dim=d["var_dim"],
            const=const,
            const_dims=dict(d["task_dims"]),
            workloads=[
                Workload(
                    w["axes"][var_axis],
                    w.get("calls", 0),
                    w.get("est_us", 0.0),
                    list(w.get("ops", [])),
                )
                for w in d["workloads"]
            ],
            share_pct=d.get("share_pct", 0.0),
        )


def _spec(kind: str, dtype: str) -> dict:
    spec = _CONTRACTS.get((kind, str(dtype)))
    if spec is None:
        raise ValueError(f"no AKLR contract for a {dtype} {kind} task")
    return spec


def targets(
    tasks_doc: dict,
    *,
    ops: list[str] | None = None,
    max_var: int | None = None,
    min_share_pct: float = 0.0,
    max_workloads: int = 8,
) -> list[Target]:
    """The replay's tasks AKLR can express, one per schema, heaviest first.

    A task's cases with the same constant axes become one schema; cases of
    several ops sharing those axes (two linear layers of one shape) share it.
    ``max_var`` keeps the cases whose variable axis (a GEMM's m, a norm's
    tokens) is at most that: decode's small, memory-bound calls, where a
    hand-written kernel beats the library's.
    """
    total = sum((tasks_doc.get("totals") or {}).get("est_gpu_ms", {}).values()) or 0.0
    by_name: dict[str, Target] = {}
    for task in tasks_doc.get("tasks", []):
        if ops and task.get("op") not in ops:
            continue
        spec = _CONTRACTS.get((task.get("kind"), str(task.get("dtype"))))
        if spec is None:
            continue
        var_dim, var_axis = spec["var"]
        for case in task.get("cases", []):
            dims = case.get("dims") or {}
            if dims.get("groups", 1) not in (None, 1) or dims.get("topk", 0) not in (None, 0):
                continue
            if any(dims.get(d) is None for d, _ in spec["const"]) or dims.get(var_dim) is None:
                continue
            value = int(dims[var_dim])
            if max_var is not None and value > max_var:
                continue
            const = {axis: int(dims[d]) for d, axis in spec["const"]}
            name = spec["name"].format(**const)
            target = by_name.get(name)
            if target is None:
                target = by_name[name] = Target(
                    name=name,
                    contract=spec["contract"],
                    op_dir=spec["op_dir"],
                    kind=task["kind"],
                    dtype=str(task["dtype"]),
                    var_axis=var_axis,
                    var_dim=var_dim,
                    const=const,
                    const_dims={d: int(dims[d]) for d, _ in spec["const"]},
                    workloads=[],
                )
            work = next((w for w in target.workloads if w.value == value), None)
            if work is None:
                work = Workload(value)
                target.workloads.append(work)
            # one shape run by several ops: their calls add, at the calls-weighted time
            calls = int(case.get("calls") or 0)
            est = float(case.get("est_us") or 0.0)
            both = work.calls + calls
            work.est_us = (work.est_us * work.calls + est * calls) / both if both else est
            work.calls = both
            work.ops.append(task["op"])
    out = []
    for target in by_name.values():
        target.workloads.sort(key=lambda w: -w.total_ms)
        target.workloads = sorted(target.workloads[:max_workloads], key=lambda w: w.value)
        target.share_pct = 100.0 * target.total_ms / total if total else 0.0
        if target.share_pct >= min_share_pct:
            out.append(target)
    return sorted(out, key=lambda t: -t.total_ms)


def targets_document(found: list[Target], tasks_path: str = "") -> dict:
    return {
        "schema": TARGETS_SCHEMA,
        "version": 1,
        "kernel_tasks": tasks_path,
        "targets": [t.to_dict() for t in found],
    }


def load_targets(path: str) -> list[Target]:
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("schema") != TARGETS_SCHEMA:
        raise ValueError(f"{path}: not an {TARGETS_SCHEMA} file")
    return [Target.from_dict(t) for t in doc["targets"]]


# ── AKLR schemas (AKLR's code: run where AKLR and torch import) ──────────────


def _import_aklr(aklr_root: str):
    if aklr_root not in sys.path:
        sys.path.insert(0, aklr_root)
    from aklr.core.contract.registry import ContractRegistry
    from aklr.core.contract.spec import PointSpec

    return ContractRegistry.default(), PointSpec


def committed(aklr_root: str, target: Target) -> list[str]:
    """The schema files AKLR already holds for ``target``'s name, any language."""
    root = os.path.join(aklr_root, "aklr", "ops", target.op_dir, "impls")
    found = []
    if os.path.isdir(root):
        for language in sorted(os.listdir(root)):
            folder = os.path.join(root, language, target.name)
            if os.path.isdir(folder):
                found += [
                    os.path.join(folder, f)
                    for f in sorted(os.listdir(folder))
                    if f.endswith(".json")
                ]
    return found


def schema_text(
    aklr_root: str, target: Target, *, language: str = "triton", gpu: str = "mi355x"
) -> tuple[str, str]:
    """A kernel-less AKLR schema for ``target``: its text, and its path under the op directory."""
    registry, point_spec = _import_aklr(aklr_root)
    spec = _spec(target.kind, target.dtype)
    contract = registry.get(target.contract)
    first = {target.var_axis: target.workloads[0].value}
    point = point_spec(
        contract_id=target.contract,
        axes={**target.const, **first},
        dtypes=dict(spec["dtypes"]),
        attrs=dict(spec["attrs"]),
        layout_variant_name=_LAYOUT,
    )
    kernel = {
        "language": language,
        "targets": [_GPU_TARGETS[gpu]],
        "author": "InferaSim kernel tasks",
        "entry_point": "kernel.py::run",
        "sources": [],
    }
    scalars = dict(spec.get("scalars") or {})
    workloads = [
        {"axes": {target.var_axis: w.value}, **({"scalars": scalars} if scalars else {})}
        for w in target.workloads
    ]
    schema = contract.narrow(point, workloads=workloads, kernel=kernel).model_copy(
        update={"name": target.name}
    )
    return schema.schema_text(), str(schema.relative_path())


def write_schema(aklr_root: str, target: Target, *, language: str = "triton") -> str:
    """Write ``target`` as a kernel-less schema in an AKLR checkout; its path."""
    text, rel = schema_text(aklr_root, target, language=language)
    path = os.path.join(aklr_root, "aklr", "ops", target.op_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    return path


# ── KernelForge campaigns ───────────────────────────────────────────────────


def campaign_script(
    target: Target,
    *,
    aklr_root: str,
    schema_path: str,
    run_dir: str,
    hours: float,
    model: str,
    language: str = "triton",
    gpu: int = 0,
    starting_kernel: str = "",
    anchor_start: bool = False,
) -> str:
    """The shell commands of one target's campaign, from AKLR's checkout: prepare the task, then KernelForge's loop.

    AKLR starts a GEMM schema from torch; any other needs ``starting_kernel``. ``anchor_start`` scores against
    the starting kernel, for a schema AKLR's baseline cannot run.
    """
    q = shlex.quote
    if _spec(target.kind, target.dtype).get("starting_kernel") and not starting_kernel:
        raise ValueError(
            f"{target.name}: AKLR needs a starting kernel for a {target.contract} schema"
        )
    prepare = f"--schema-file {q(schema_path)} --run-dir {q(run_dir)} --kernel-backend {language}"
    if starting_kernel:
        prepare += f" --starting-kernel {q(starting_kernel)}"
    if anchor_start:
        prepare += " --anchor start"
    loop = [
        f"{q(run_dir)}/forge_loop.sh",
        "--max-hours",
        str(hours),
        "--model",
        model,
        "--gpu",
        str(gpu),
        "--lanes",
        "1",
        "--no-experience-kb",
        "--no-kb-warmstart",
        "--no-pr-kb",
    ]
    return "\n".join(
        [
            "#!/bin/bash",
            f"# {target.name}: {', '.join(target.ops)}, {target.share_pct:.2f}% of the replay's GPU time",
            "set -e",
            f"cd {q(aklr_root)}",
            f"[ -f {q(run_dir)}/baseline.json ] || python harness/scripts/forge_task prepare {prepare}",
            " ".join(loop),
            "",
        ]
    )


# ── results as speedups ─────────────────────────────────────────────────────


def _case_value(case_id: str, axis: str) -> int | None:
    for part in str(case_id).split(","):
        name, _, value = part.partition("=")
        if name.strip() == axis and value.strip().lstrip("-").isdigit():
            return int(value)
    return None


def _entries(target: Target, times: dict[int, tuple[float, float]], source: str) -> list[dict]:
    """One entry per op and workload: the kernel's measured time over the baseline's."""
    out = []
    for value, (base_ms, ms) in sorted(times.items()):
        if (
            not (base_ms and ms and math.isfinite(base_ms) and math.isfinite(ms))
            or base_ms <= 0
            or ms <= 0
        ):
            continue
        work = next((w for w in target.workloads if w.value == value), None)
        for op in sorted(set(work.ops if work else target.ops)):
            out.append(
                {
                    "op": op,
                    "kind": target.kind,
                    "dims": dict(target.const_dims),
                    "ranges": {target.var_dim: [value, value]},
                    "baseline_us": round(base_ms * 1000.0, 4),
                    "tuned_us": round(ms * 1000.0, 4),
                    "source": source,
                }
            )
    return out


def campaign_times(run_dir: str, axis: str) -> dict[int, tuple[float, float]]:
    """Per workload, AKLR's baseline and the kept kernel's time, from a forge_task run directory.

    KernelForge records both in its run state; a kernel no better than the
    baseline on a workload is reported as it measured (a speedup below 1
    there), since a dispatcher that serves the schema serves it everywhere.
    """
    state_path = os.path.join(run_dir, "workspace", "forge_experiments", "run_state.json")
    base: dict = {}
    best: dict = {}
    if os.path.exists(state_path):
        with open(state_path) as fh:
            state = json.load(fh)
        base = state.get("baseline_case_times") or {}
        best = state.get("best_case_times") or {}
    baseline_json = os.path.join(run_dir, "baseline.json")
    if os.path.exists(baseline_json):
        with open(baseline_json) as fh:
            base = (json.load(fh).get("case_times") or {}) or base
    out = {}
    for case_id, ms in best.items():
        value = _case_value(case_id, axis)
        if value is not None and case_id in base:
            out[value] = (float(base[case_id]), float(ms))
    return out


def bench_times(path: str, axis: str) -> dict[int, tuple[float, float]]:
    """Per workload, the baseline's and a kernel's time from a bench file.

    ``{"cases": {"m=64": {"baseline_ms": .., "kernel_ms": .., "library_ms": ..}}}``:
    a kernel timed again after its campaign. With ``library_ms`` (the library
    the stack would call instead, e.g. hipBLASLt through torch) the baseline is
    the faster of the two, so a kernel is credited only for beating both.
    """
    with open(path) as fh:
        doc = json.load(fh)
    out = {}
    for case_id, rec in (doc.get("cases") or {}).items():
        value = _case_value(case_id, axis)
        if value is None:
            continue
        base = [float(x) for x in (rec.get("baseline_ms"), rec.get("library_ms")) if x]
        if base and rec.get("kernel_ms"):
            out[value] = (min(base), float(rec["kernel_ms"]))
    return out


def committed_times(schema_path: str, axis: str) -> dict[int, tuple[float, float]]:
    """Per workload, AKLR's recorded speedup over its baseline, as (speedup, 1.0)."""
    with open(schema_path) as fh:
        doc = json.load(fh)
    out = {}
    for w in doc.get("workloads", []):
        value = (w.get("axes") or {}).get(axis)
        if value is not None and w.get("speedup") and w.get("accuracy") != "failed":
            out[int(value)] = (float(w["speedup"]), 1.0)
    return out


def speedups(results: list[tuple[Target, dict[int, tuple[float, float]], str]]) -> dict:
    entries = []
    for target, times, source in results:
        entries += _entries(target, times, source)
    doc = {"schema": ksp.SCHEMA, "version": 1, "entries": entries}
    ksp.Speedups.parse(doc)
    return doc


# ── CLI ─────────────────────────────────────────────────────────────────────


def _write(path: str, doc: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def _cmd_targets(args) -> dict:
    with open(args.kernel_tasks) as fh:
        doc = json.load(fh)
    found = targets(
        doc,
        ops=args.op or None,
        max_var=args.max_m,
        min_share_pct=args.min_share,
        max_workloads=args.max_workloads,
    )
    out = targets_document(found, args.kernel_tasks)
    _write(args.out, out)
    return {
        "targets": [
            {
                "name": t.name,
                "ops": t.ops,
                "share_pct": round(t.share_pct, 3),
                "workloads": len(t.workloads),
            }
            for t in found
        ]
    }


def _cmd_schemas(args) -> dict:
    written = {}
    for target in load_targets(args.targets):
        if args.name and target.name not in args.name:
            continue
        existing = [
            p for p in committed(args.aklr, target) if f"{os.sep}{args.language}{os.sep}" in p
        ]
        written[target.name] = (
            existing[0]
            if existing and not args.force
            else write_schema(args.aklr, target, language=args.language)
        )
    return {"schemas": written}


def _cmd_campaign(args) -> dict:
    target = next((t for t in load_targets(args.targets) if t.name == args.name), None)
    if target is None:
        raise ValueError(f"no target {args.name} in {args.targets}")
    text = campaign_script(
        target,
        aklr_root=os.path.abspath(args.aklr),
        schema_path=os.path.abspath(args.schema),
        run_dir=os.path.abspath(args.run_dir),
        hours=args.hours,
        model=args.model,
        language=args.language,
        gpu=args.gpu,
        starting_kernel=os.path.abspath(args.starting_kernel) if args.starting_kernel else "",
        anchor_start=args.anchor_start,
    )
    with open(args.out, "w") as fh:
        fh.write(text)
    os.chmod(args.out, 0o755)
    return {"campaign": args.out}


def _cmd_speedups(args) -> dict:
    by_name = {t.name: t for t in load_targets(args.targets)}
    results = []
    for kind, raw in (
        [("run", r) for r in args.run]
        + [("bench", r) for r in args.bench]
        + [("schema", r) for r in args.committed]
    ):
        name, _, path = raw.partition("=")
        target = by_name.get(name)
        if target is None:
            raise ValueError(f"no target {name} in {args.targets}")
        if kind == "run":
            times = campaign_times(path, target.var_axis)
        elif kind == "bench":
            times = bench_times(path, target.var_axis)
        else:
            times = committed_times(path, target.var_axis)
        results.append((target, times, f"aklr:{target.name}:{kind}"))
    doc = speedups(results)
    _write(args.out, doc)
    return {"kernel_speedups": args.out, "entries": len(doc["entries"])}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inferasim aklr", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("targets", help="the kernel tasks AKLR can express, as AKLR targets")
    t.add_argument("--kernel-tasks", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--op", action="append", default=[], help="only these ops (repeatable)")
    t.add_argument(
        "--max-m",
        type=int,
        default=None,
        help="only cases whose m (a norm's tokens) is at most this",
    )
    t.add_argument(
        "--min-share",
        type=float,
        default=0.0,
        help="only targets with this share of GPU time, in %%",
    )
    t.add_argument("--max-workloads", type=int, default=8)
    s = sub.add_parser(
        "schemas", help="write the targets as kernel-less AKLR schemas (needs AKLR and torch)"
    )
    s.add_argument("--targets", required=True)
    s.add_argument("--aklr", required=True, help="an AKLR checkout")
    s.add_argument("--name", action="append", default=[])
    s.add_argument("--language", default="triton", choices=("triton", "flydsl", "hip"))
    s.add_argument(
        "--force", action="store_true", help="write even when AKLR already holds the schema"
    )
    c = sub.add_parser("campaign", help="a KernelForge campaign script for one target")
    c.add_argument("--targets", required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--aklr", required=True)
    c.add_argument("--schema", required=True)
    c.add_argument("--run-dir", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--hours", type=float, default=4.0)
    c.add_argument("--model", default="claude-opus-5-5")
    c.add_argument("--language", default="triton", choices=("triton", "flydsl", "hip"))
    c.add_argument("--gpu", type=int, default=0)
    c.add_argument(
        "--starting-kernel", default="", help="kernel.py, or a directory of them (non-GEMM schemas)"
    )
    c.add_argument("--anchor-start", action="store_true", help="score against the starting kernel")
    r = sub.add_parser(
        "speedups", help="kernels' measured speedups as an inferasim.kernel_speedups file"
    )
    r.add_argument("--targets", required=True)
    r.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="NAME=RUN_DIR",
        help="a forge_task run directory",
    )
    r.add_argument(
        "--bench", action="append", default=[], metavar="NAME=JSON", help="a kernel timed again"
    )
    r.add_argument(
        "--committed",
        action="append",
        default=[],
        metavar="NAME=SCHEMA",
        help="an AKLR schema file",
    )
    r.add_argument("--out", required=True)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "targets": _cmd_targets,
        "schemas": _cmd_schemas,
        "campaign": _cmd_campaign,
        "speedups": _cmd_speedups,
    }
    print(json.dumps(commands[args.command](args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
