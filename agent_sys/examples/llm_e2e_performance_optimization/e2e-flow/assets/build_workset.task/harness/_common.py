#!/usr/bin/env python3
"""The measuring instrument every workset ships, and **no agent writes it**.

`build_workset` is an AI task. The one thing it must not produce is the thing
that decides whether its own output is correct and how fast it is: an agent that
writes its own oracle controls its own result, and every number after that is
unfalsifiable. So the two entrypoints and this module are **package data**,
copied into the workset verbatim by `scaffold.py`, and `check_workset_shape`
compares the copy against this directory byte for byte.

What the agent does supply is the *Definition* — `reference` and `baseline` as
Python source, in flashinfer-bench's own shape. That is judgement work and a
model is the right instrument for it. This file then runs it, and
`check_workset_runs` re-runs this file. The split is the whole reason M4.3.5
could be reversed: m4 divides by a number produced by an instrument neither it
nor the agent that built the workset could edit.

Three properties the two entrypoints share, and each exists because its absence
was a measured failure somewhere in this effort:

* **One document shape for baseline and candidate**, distinguished only by
  `impl`. A consumer comparing its kernel against the incumbent is comparing two
  runs of one script, not two scripts.
* **`--shape CASE_ID` narrows to one case.** Without it `check_workset_runs`
  cannot re-measure affordably, and the trust chain becomes a claim about a
  claim.
* **The abort is a behaviour, not a paragraph.** `ground_truth.abort_on_mismatch`
  is checked here, so a run on the wrong architecture stops rather than
  producing a number somebody later divides by.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent

__all__ = ["Ctx", "load_definition", "load_impl", "setup", "finish", "now"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ctx:
    """Everything both entrypoints need, resolved once."""

    def __init__(self, doc, args, report, operators):
        self.doc, self.args, self.report, self.operators = doc, args, report, operators

    def shapes(self, operator):
        """The shapes this invocation covers, honouring `--shape`."""
        return [s for s in operator["shapes"] if self.args.shape in (None, s["case_id"])]


def _observed(field: str):
    """What the host says about one `environment.fixed` field, or `None`.

    Read from `E2E_<FIELD>` rather than probed. Probing a GPU architecture from
    inside a validator would mean this module needs a ROCm import to decide
    whether it may run, and a missing import would then read as a mismatch. The
    runner exports what it knows; an unset variable means *unknown*, and unknown
    is not a mismatch.
    """
    return os.environ.get("E2E_" + field.upper())


def setup(what: str) -> Ctx:
    import yaml

    doc = yaml.safe_load((HERE / "workset.yaml").read_text(encoding="utf-8"))
    # **The flag spelling comes from the workset, not from here.**
    #
    # It was hardcoded, and `entrypoints.<what>.flags` declared the same four
    # strings in the manifest — one fact, two readers, and the reader that could
    # not be told was this one. m4 drives these entrypoints by reading `flags`
    # from the manifest, so a workset declaring a different spelling would have
    # produced a consumer passing `--implementation` to a parser that only knew
    # `--impl`. That is the shape m4 named — *one authority, two readers, one of
    # them narrower* — and it was the fourth instance between us in two days.
    #
    # Found by auditing my own code for the shape after claiming it was clean.
    # It was not.
    flags = {"operator": "--operator", "shape": "--shape", "impl": "--impl", "report": "--json"}
    flags.update((doc.get("entrypoints") or {}).get(what, {}).get("flags") or {})
    ap = argparse.ArgumentParser(description=f"one-click {what} over this workset")
    ap.add_argument(flags["operator"], dest="operator", help="restrict to one operator_id")
    ap.add_argument(flags["shape"], dest="shape", help="restrict to one case_id")
    ap.add_argument(flags["impl"], dest="impl",
                    help="a replacement implementation; omit to exercise the baseline")
    ap.add_argument(flags["report"], dest="json", help="where to write the report")
    args = ap.parse_args()

    ground = doc["ground_truth"]
    fixed = ground["environment"]["fixed"]
    for field in ground["abort_on_mismatch"]:
        seen = _observed(field)
        if seen is not None and str(seen) != str(fixed.get(field)):
            sys.exit(
                f"abort: {field} is {seen!r} on this host and the workset's evidence was taken "
                f"at {fixed.get(field)!r}. Every number in evidence/ is about a different "
                f"machine; a ratio across the two would not mean anything (M4.3.5)."
            )
    for field in ground.get("warn_on_mismatch") or []:
        seen = _observed(field)
        if seen is not None and str(seen) != str(fixed.get(field)):
            print(f"warning: {field} is {seen!r} here, the evidence was taken at "
                  f"{fixed.get(field)!r}", file=sys.stderr)

    report = {
        "schema_version": 1,
        "generated_by": pathlib.Path(sys.argv[0]).name,
        "environment": {
            # **`E2E_NODE` first, and `os.uname()` only as a fallback.** Measured
            # on the first real GPU run: inside a container `os.uname().nodename`
            # is the container id — the report said it was measured on
            # `9aae0135bc3d`, which names the run's own sandbox and not the
            # machine. That is precisely the "record of a configuration rather
            # than of a run" CONTRACT §2 exists to prevent, and it would have
            # made `evidence.measured_on.node` useless for the one question it
            # answers: was this measured on the box m4 is standing on.
            "node": os.environ.get("E2E_NODE") or os.uname().nodename,
            "gpu_arch": fixed["gpu_arch"],
            "gpu_count": fixed["gpu_count"],
            "tp_size": fixed["tp_size"],
            "container": ground["environment"]["runtime"]["container"],
            "image_id": fixed["image_id"],
        },
        "started_at": now(),
        "impl": "candidate" if args.impl else "baseline",
        "impl_path": args.impl,
        "operators": [],
    }
    operators = [o for o in doc["operators"] if args.operator in (None, o["operator_id"])]
    if not operators:
        sys.exit(f"no operator matches --operator {args.operator!r}")
    return Ctx(doc, args, report, operators)


def finish(ctx: Ctx, ok: bool) -> None:
    ctx.report["finished_at"] = now()
    if ctx.args.json:
        out = pathlib.Path(ctx.args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(ctx.report, indent=2), encoding="utf-8")
    print(json.dumps({"impl": ctx.report["impl"], "operators": len(ctx.report["operators"]), "passed": ok}))
    # Exit code and the report's own verdict say the same thing, on purpose: a
    # caller that reads only one of the two is not then reading a different
    # answer from a caller that reads the other.
    sys.exit(0 if ok else 1)


def _exec_source(source: str, label: str):
    """One flashinfer-bench source string as a module namespace.

    `reference` and `baseline` are Python source ending in `def run(...)`, which
    is the format's own convention (`rank0/definitions/`). Executed rather than
    imported because a Definition is one JSON document and splitting it across
    files to satisfy the import system would break the correspondence
    `check_workset_shape` checks.
    """
    namespace: dict = {}
    exec(compile(source, f"<{label}>", "exec"), namespace)  # noqa: S102 — the artefact under test
    if "run" not in namespace:
        raise SystemExit(f"the Definition's {label!r} defines no `run`")
    return namespace["run"]


def load_definition(operator: dict) -> dict:
    return json.loads((HERE / operator["definition"]).read_text(encoding="utf-8"))


def load_impl(operator: dict, definition: dict, impl_path: str | None, which: str):
    """The callable under test.

    `which` is `reference` or `baseline`. When `--impl` was given it replaces the
    **baseline** and never the reference: the reference is what correctness is
    judged against, so a candidate that could replace it would be grading itself.
    """
    if which == "baseline" and impl_path:
        source = pathlib.Path(impl_path).read_text(encoding="utf-8")
        return _exec_source(source, f"candidate:{operator['operator_id']}")
    return _exec_source(definition[which], f"{which}:{operator['operator_id']}")


def build_inputs(definition: dict, shape: dict):
    """Tensors for one workload line, from the Definition's `inputs` and the
    shape's `axes`.

    The Definition names every axis as `var` or `const`, so a shape supplies only
    the `var` ones and this resolves the rest — which is what makes the workload
    JSONL short enough to read and impossible to disagree with the Definition
    about a constant.
    """
    import torch

    axes = {name: spec["value"] for name, spec in definition["axes"].items() if spec["type"] == "const"}
    axes.update(shape["axes"])

    dtypes = {
        "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
        "float64": torch.float64, "int32": torch.int32, "int64": torch.int64,
        "uint8": torch.uint8, "bool": torch.bool,
    }
    built = {}
    for name, spec in definition["inputs"].items():
        if spec.get("shape") is None:
            built[name] = spec.get("value")
            continue
        dims = [axes[d] if isinstance(d, str) else d for d in spec["shape"]]
        dtype = dtypes.get(spec["dtype"])
        if dtype is None:
            raise SystemExit(f"input {name!r} has dtype {spec['dtype']!r}, which this harness cannot build. "
                             f"A packed or quantised dtype needs a per-operator builder; see the readme's STEP 5.")
        if dtype in (torch.int32, torch.int64):
            built[name] = torch.randint(0, max(1, axes.get("num_experts", 8)), dims, dtype=dtype, device=_device())
        elif dtype == torch.bool:
            built[name] = torch.zeros(dims, dtype=dtype, device=_device())
        else:
            built[name] = torch.randn(dims, dtype=dtype, device=_device())
    return built


def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
