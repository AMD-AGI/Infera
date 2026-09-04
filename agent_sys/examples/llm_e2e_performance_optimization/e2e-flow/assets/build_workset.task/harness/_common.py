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
  producing a number somebody later divides by. **That sentence was false for
  the life of this file** — the check read `E2E_<FIELD>` variables that nothing
  in this package declares, so it compared against `None` and passed every
  time. It now compares this run's environment record against the workset's,
  and says so in the report; see `_run_record`. The lesson is not about the
  seven names: a paragraph asserting a behaviour is not one, and this one was
  read many times by its own author without being run.
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


def _run_record(explicit: str | None) -> tuple[dict | None, str]:
    """This run's environment record, and where it was found.

    **This replaced an `E2E_<FIELD>` lookup that could never succeed.** The old
    reader derived `E2E_GPU_ARCH`, `E2E_GPU_COUNT`, `E2E_TP_SIZE`, `E2E_DTYPE`,
    `E2E_IMAGE_ID`, `E2E_ROCM` and `E2E_TORCH` from the two mismatch lists —
    **not one of the seven is declared anywhere in this package**, so it
    returned `None` every time and both loops below have never been able to
    fire. Its docstring said *"an unset variable means unknown, and unknown is
    not a mismatch"*; unknown was always. The abort that stops a measurement
    being taken on a machine the workset's evidence did not come from has been
    inert for the life of this file, and the docstring at the top of it called
    that abort a behaviour rather than a paragraph.

    The record is the right source and the env channel was the wrong one, for
    the reason the sibling path already demonstrated: `measure_in_container.sh`
    and m4's `run_in_container.sh` both take the node, job and transport from
    `environment.yaml` and refuse an ambient value only when it *disagrees*,
    and that path came through the same outage intact. An env channel here is
    also a second authority for a fact the record already states, which is the
    one thing this package has learnt to avoid, and this is the worst place to
    have it.

    Two documents, and the comparison is between them rather than within one:

    * **this run's record**, returned here — where the entrypoint is executing;
    * **the evidence's record**, `ground_truth.environment` in `workset.yaml` —
      where the numbers in `evidence/` were taken.

    They are the same document when the workset is measured where it was built,
    which is m3's own path and why it agrees trivially there. They are two
    different documents in m4's, which is the case M4.3.5 is about: m4 renders
    `items/codes/environment.yaml` from the `deploy_kit` — *this* run — while
    the workset it re-measures carries m3's.

    Found in that order: `--environment`, then a record sitting beside this
    module. **Absent is reported, not passed over.** Silence is what made the
    old reader invisible for so long, and a gate that cannot run should say so
    in the same transcript as the numbers it did not guard.
    """
    import yaml

    for path, origin in ((explicit, "--environment"), (HERE / "environment.yaml", "beside the harness")):
        if not path:
            continue
        candidate = pathlib.Path(path)
        if candidate.is_file():
            return yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}, f"{origin}: {candidate}"
    return None, ""


def _observed(record: dict | None, field: str):
    """What **this run's** record says about one `environment.fixed` field.

    `None` when there is no record, or when the record does not carry the
    field — a record that omits a field genuinely does not know it, which is
    the one case the old reader's "unknown is not a mismatch" rule was written
    for and the only case it still covers.
    """
    if record is None:
        return None
    value = (record.get("fixed") or {}).get(field)
    return None if value is None else value


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
    flags = {"operator": "--operator", "shape": "--shape", "impl": "--impl", "report": "--json",
             "environment": "--environment"}
    flags.update((doc.get("entrypoints") or {}).get(what, {}).get("flags") or {})
    ap = argparse.ArgumentParser(description=f"one-click {what} over this workset")
    ap.add_argument(flags["operator"], dest="operator", help="restrict to one operator_id")
    ap.add_argument(flags["shape"], dest="shape", help="restrict to one case_id")
    ap.add_argument(flags["impl"], dest="impl",
                    help="a replacement implementation; omit to exercise the baseline")
    ap.add_argument(flags["report"], dest="json", help="where to write the report")
    ap.add_argument(flags["environment"], dest="environment",
                    help="this run's environment.yaml, against which the workset's is checked; "
                         "defaults to one sitting beside this module")
    args = ap.parse_args()

    ground = doc["ground_truth"]
    fixed = ground["environment"]["fixed"]
    record, origin = _run_record(args.environment)
    if record is None:
        # **Loud, and in the same transcript as the numbers.** The gate this
        # guards is the one that stops a ratio being taken across two machines,
        # and it spent the life of this file returning `None` in silence. A run
        # that cannot check its premise may still be the right run — m3's own
        # measurement is — but it may not look like a checked one.
        print(f"warning: no environment record for this run, so "
              f"{', '.join(ground['abort_on_mismatch'])} were NOT checked against it. "
              f"Pass {flags['environment']} <path to this run's environment.yaml>, or place one "
              f"beside the entrypoints. The workset claims {fixed.get('gpu_arch')!r} / "
              f"tp {fixed.get('tp_size')!r}; nothing here confirmed it.", file=sys.stderr)
    elif record == ground["environment"]:
        # **The resolved record IS the workset's own, so the loops below compare
        # a document with itself and agree by construction.**
        #
        # Found by m4 in review, in the fallback rather than in the flag: the
        # record beside this module is the one `ground_truth.environment` was
        # copied from. Their reading, and it is the right one — this file's own
        # "a paragraph asserting a behaviour is not one", one layer out. A field
        # asserting `checked` for a run that checked nothing is that sentence
        # with a JSON key instead of a docstring.
        #
        # **This is a legitimate state, not a fault**, and the wording says so
        # in both places deliberately. m3's own measurement resolves here and is
        # *correct* in doing so: it measures where the workset was built, on the
        # node the record names, so the record beside it genuinely is the one
        # the evidence comes from. What is wrong is only the reporting — naming
        # a path reads as two documents. So `premise_checked_against` says
        # `self` and carries no path, and a consumer wanting the two-document
        # comparison passes the flag (m4's two callers do).
        origin = "self (not independently confirmed)"
        print(f"note: this run's environment record is the workset's own, so "
              f"{', '.join(ground['abort_on_mismatch'])} agree by construction and were not "
              f"independently confirmed. That is expected when a workset is measured where it "
              f"was built. Pass {flags['environment']} <this run's environment.yaml> for a "
              f"comparison between two documents.", file=sys.stderr)
    for field in ground["abort_on_mismatch"]:
        seen = _observed(record, field)
        if seen is not None and str(seen) != str(fixed.get(field)):
            sys.exit(
                f"abort: {field} is {seen!r} in this run's environment record ({origin}) and the "
                f"workset's evidence was taken at {fixed.get(field)!r}. Every number in evidence/ "
                f"is about a different machine; a ratio across the two would not mean anything "
                f"(M4.3.5)."
            )
    for field in ground.get("warn_on_mismatch") or []:
        seen = _observed(record, field)
        if seen is not None and str(seen) != str(fixed.get(field)):
            # Warn and carry on, deliberately, and `image_id` is on this list
            # rather than the abort list on m4's reading — a rebuilt image with
            # the same architecture and topology is a software difference the
            # record's own `warnings[]` channel carries to m5, not a reason to
            # refuse a measurement.
            print(f"warning: {field} is {seen!r} in this run's record, the evidence was taken at "
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
            # **These five said `fixed[...]` — the workset's own claim, copied
            # into a block whose whole job is to say where the run happened.**
            # So even a working abort gate would have had no artefact behind it:
            # the report agreed with the premise by construction, for the same
            # reason `node` once agreed with the container id. This run's record
            # is preferred and the claim is the fallback, which differs only
            # where the two disagree — and for `abort_on_mismatch` fields the
            # run has already stopped by here, so in practice this is about the
            # warn-level ones.
            "gpu_arch": _observed(record, "gpu_arch") or fixed["gpu_arch"],
            "gpu_count": _observed(record, "gpu_count") or fixed["gpu_count"],
            "tp_size": _observed(record, "tp_size") or fixed["tp_size"],
            "container": ((record or {}).get("runtime") or {}).get("container")
            or ground["environment"]["runtime"]["container"],
            "image_id": _observed(record, "image_id") or fixed["image_id"],
            # Which of the above were read and which were assumed. Without it a
            # reader cannot tell a checked run from an unchecked one, and that
            # is exactly the distinction that went missing for this file's life.
            "premise_checked_against": origin or None,
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
