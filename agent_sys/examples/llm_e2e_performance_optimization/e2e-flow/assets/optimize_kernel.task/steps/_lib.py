#!/usr/bin/env python3
"""What the seven STEP scripts share: locating things, and reading the workset.

One module, because the alternative is seven readers of one layout and seven
chances for "the workset's three shapes" to mean three different things. The
same argument `assets/lib/workset_io.py` makes for the producer/validator split,
applied one level down.

Nothing here decides anything. Every function returns what a file says.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: Where the structured artefacts live inside the packup. Fixed, not derived:
#: two validators, this task and m5 all open them, and a path each derives is a
#: path each can derive differently. `kernel_optimization.schema.json` pins the
#: first two with `const`.
DOC = "results/kernel_optimization.json"
SNAPSHOT = "results/workset.snapshot.yaml"
BASELINE_REPORT = "results/workset.baseline_report.json"
APPARATUS = "scripts/workset"


def die(message: str, code: int = 1) -> "None":
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def package() -> Path:
    """The staged copy of this package, from whichever row exported it.

    **Both variables, always.** A body that reads one of the two works in
    testing and fails in a phase; it has already cost one run.
    """
    for var in ("AGENT_SYS_TASK_PACKAGE", "AGENT_SYS_DEMO_PACKAGE"):
        value = os.environ.get(var)
        if value:
            return Path(value)
    return Path(__file__).resolve().parents[3]


def _lib():
    sys.path.insert(0, str(package() / "assets" / "lib"))


def load_yaml(path: Path):
    import yaml  # a declared agent_sys dependency

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def input_content(kind: str) -> Path:
    """An input handoff's `content/` directory.

    **The asymmetry that bites**: `$AGENT_SYS_INPUT_<KIND>` points at the
    handoff's *version* directory, so `content/` is a hop below it, while
    `$AGENT_SYS_OUTPUT_<KIND>` points at `content/` itself. They look like a
    pair and they are one level apart.
    """
    var = "AGENT_SYS_INPUT_" + "".join(c if c.isalnum() else "_" for c in kind).upper()
    root = os.environ.get(var)
    if not root:
        die(f"{var} is unset; this task does not have {kind} as an input")
    version = Path(str(root))
    content = version / "content"
    return content if content.is_dir() else version


def workset_root() -> Path:
    """The workset root **is** `items/codes/`, with no wrapper directory.

    Mirrors `assets/lib/workset_io.workset_root`, and the reason is worth
    keeping: a workset carries `definitions/`, `workloads/`, `operators/` and
    `evidence/` side by side, so the merged kind puts the workset at the root
    and the *operators* in a list inside it. A `find the one directory` rule is
    right for a packup and wrong here.
    """
    return input_content("operator_workset") / "items" / "codes"


def load_workset() -> dict:
    path = workset_root() / "workset.yaml"
    if not path.is_file():
        die(f"no workset.yaml at {path}")
    return load_yaml(path)


def load_environment() -> dict:
    """m1's environment record, out of the `deploy_kit`.

    CONTRACT §2: a `code` handoff carries it at `items/codes/environment.yaml`.
    This is the record this run is *in*; the workset's own
    `ground_truth.environment` is the record the baseline was measured in, and
    STEP 2 is the comparison between them.
    """
    path = input_content("deploy_kit") / "items" / "codes" / "environment.yaml"
    if not path.is_file():
        die(f"no environment.yaml at {path}; m1's kit does not carry the environment record")
    return load_yaml(path)


def pick_operator(workset: dict, wanted: str | None) -> dict:
    operators = [o for o in workset.get("operators") or () if isinstance(o, dict)]
    if not operators:
        die("the workset declares no operators")
    if wanted:
        for operator in operators:
            if operator.get("operator_id") == wanted:
                return operator
        die(f"no operator {wanted!r} in the workset (has: {[o.get('operator_id') for o in operators]})")
    if len(operators) > 1:
        die(
            "the workset carries "
            f"{len(operators)} operators {[o.get('operator_id') for o in operators]} and this task "
            "optimises one; pass --operator <id>. One handoff per operator is deferred (todo.md T3)"
        )
    return operators[0]


def entrypoints(workset: dict, operator: dict) -> dict:
    """An operator's entrypoints, falling back to the workset's own.

    `workset.schema.json` declares `entrypoints` in both places and requires it
    at the top level. The per-operator block wins where present, because a
    workset with several operators may drive them differently.
    """
    merged = dict(workset.get("entrypoints") or {})
    merged.update(operator.get("entrypoints") or {})
    return merged


def shapes(operator: dict, role: str) -> list[str]:
    """Case ids for one role. `correctness-and-performance` counts for both."""
    wanted = {role, "correctness-and-performance"}
    return [
        str(s["case_id"])
        for s in operator.get("shapes") or ()
        if isinstance(s, dict) and s.get("case_id") and s.get("role") in wanted
    ]


def report_medians(report: dict, operator_id: str) -> dict[str, float]:
    """case_id -> ms out of a `performance_report`.

    `weighted_mean_ms` is the reduction the workset's own `check_workset_runs`
    recomputes, so it is what everything downstream divides by; `median_ms` is
    the fallback for a workset whose protocol reduces that way.
    """
    out: dict[str, float] = {}
    for entry in report.get("operators") or ():
        if not isinstance(entry, dict) or entry.get("operator_id") != operator_id:
            continue
        for shape in entry.get("shapes") or ():
            if not isinstance(shape, dict):
                continue
            value = shape.get("weighted_mean_ms", shape.get("median_ms"))
            if isinstance(value, (int, float)) and shape.get("case_id"):
                out[str(shape["case_id"])] = float(value)
    return out


def scratch() -> Path:
    """`$KFO_SCRATCH_ROOT`, required and never defaulted.

    A default that happens to be writable is how the NFS `root_squash` trap gets
    re-set: `/tmp` inside a container is not the `/tmp` outside it, and a
    network home maps a container's root to nobody so that writes fail
    *silently*.
    """
    value = os.environ.get("KFO_SCRATCH_ROOT")
    if not value:
        die("KFO_SCRATCH_ROOT is unset; it must be local disk inside a `yihou/` directory")
    path = Path(str(value))
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate(name: str, doc) -> list[str]:
    """Validate against one of the package's schemas. Returns problems, never raises."""
    _lib()
    import schema  # noqa: PLC0415 — resolved from the staged package, not importable at module load

    try:
        schema.validate(name, doc)
    except schema.SchemaError as exc:
        return str(exc).splitlines()[1:]
    return []
