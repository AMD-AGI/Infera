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
#: patchkit's manifest, where `apply_patch` globs for it (`*/apply/manifest.json`).
APPLY_MANIFEST = "apply/manifest.json"


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


def report_per_case_ms(report: dict, operator_id: str) -> dict[str, float]:
    """case_id -> ms out of a `performance_report`.

    **`weighted_mean_ms` first, `median_ms` only as the fallback**, which is why
    this is not called `report_medians` any more. The weighted mean is weighted
    by `shapes[].calls` and is the reduction `check_workset_runs` recomputes
    from `per_group_ms` — so it is both the figure m3 would rather be divided
    by and the one a record cannot disagree with itself about.

    Indexed per shape and never aggregated across them: aggregating first hides
    a candidate that got faster on the primary shape and slower on the other
    two, which is a real outcome and not a rounding detail.
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


def container_path_for(target_file: str, repo_root_var: str = "") -> str | None:
    """One of `integration.target_files` as patchkit's `@ROOT@/...`.

    Two frames for one file. The workset writes a path **relative to
    `edit_target.repo_root_var`**; patchkit's manifest writes it relative to a
    root in `assets/lib/container_roots.yaml`
    (`@SGLANG_ROOT@/srt/layers/sampler.py`, where `SGLANG_ROOT` is
    `/sgl-workspace/sglang/python/sglang`).

    **`@NAME@` and never `${NAME}`**, and the reason is mechanical rather than
    stylistic: `handoff/locality.py` refuses to seal content naming an absolute
    path outside a small allow-list, and its lookbehind excludes `@` but not
    `}` — so `${SGLANG_ROOT}/srt/x.py` leaves `/srt/x.py` as a fresh
    two-segment candidate and is refused anyway.

    When `repo_root_var` names a root, that is the answer and no inference is
    needed — the workset said which root it meant. m3 records the variable in
    three legal forms (`@NAME@`, `${NAME}`, and empty for *owner unknown*), so
    all three are normalised here rather than at each call site.

    The suffix search is the fallback for the empty form. It is a real case:
    three of the top routable candidates on the real 124-kernel table are
    Triton kernels no owner rule matches.

    Returns `None` when nothing matches, which is a refusal rather than a
    guess — a container path invented here is one m5 would apply to a real
    image.
    """
    roots = load_yaml(package() / "assets" / "lib" / "container_roots.yaml").get("roots") or {}
    relative = str(target_file).lstrip("/")

    declared = str(repo_root_var or "").strip()
    if declared.startswith("@") and declared.endswith("@"):
        declared = declared[1:-1]
    elif declared.startswith("${") and declared.endswith("}"):
        declared = declared[2:-1]
    if declared:
        if declared not in roots:
            return None
        depth_and_rest = _strip_root_tail(relative, roots[declared])
        rest = relative if depth_and_rest is None else depth_and_rest[1]
        return f"@{declared}@/{rest}"

    best: tuple[int, str] | None = None
    for name, entry in roots.items():
        found = _strip_root_tail(relative, entry)
        if found is None:
            continue
        depth, rest = found
        if best is None or depth > best[0]:
            best = (depth, f"@{name}@/{rest}")
    return best[1] if best else None


def _strip_root_tail(relative: str, entry) -> tuple[int, str] | None:
    """Drop the root's own trailing path segments from the front of `relative`.

    **Both callers, and it used to be only one.** `/sgl-workspace/sglang/python/sglang`
    against `sglang/python/sglang/kernels/ops/…` shares `sglang/python/sglang`,
    and joining the two without dropping it yields
    `/sgl-workspace/sglang/python/sglang/sglang/python/sglang/kernels/…` — a path
    that exists nowhere, produced from two documents that are each correct.

    The suffix search below the declared branch always did this; the declared
    branch returned `@ROOT@/{relative}` verbatim, on the reasoning that a
    workset naming its `repo_root_var` has said which root it meant and no
    inference is needed. True about the *root*, and it says nothing about
    whether the path was written relative to that root or to the checkout above
    it — and m3 writes `edit_target.source_file` and
    `invocation_spec.json`'s `sources` in the second form while naming
    `@SGLANG_ROOT@`. So the branch that was told the answer was the one that got
    it wrong, and the branch that had to guess was right.

    Measured 2026-09-04: with the prefix present, STEP 6 refused resolving
    `base_sha256` against the doubled path; with it stripped by hand, it
    completed. Stripping is safe for a path that is genuinely root-relative —
    it only fires when the root's own tail is actually there.

    Returns `(segments matched, the remainder)`, or `None` when the root's tail
    is not there at all — which the suffix search reads as "this root does not
    describe this path" and the declared branch reads as "already relative to
    the root it named".
    """
    tail = str((entry or {}).get("path") or "").strip("/").split("/")
    for length in range(len(tail), 0, -1):
        prefix = "/".join(tail[-length:]) + "/"
        if relative.startswith(prefix):
            return length, relative[len(prefix):]
    return None


def render_environment(out: Path, warnings: list[dict] | None = None) -> None:
    """MOCK-MAP (A) / CONTRACT §2: write `items/codes/environment.yaml`.

    **Every handoff carries the environment record** (mission G5), and this one
    did not — in the real path as well as the mock. The leader reported it as a
    mock adaptation, which is where it was found; it is not one. `mock.sh`
    copies sealed bytes and the sealed stage-4 handoff predates the record
    entirely, so the mock has nothing to copy — but `60_write_handoff.py` was
    never writing it either, so a *real* m4 run would have failed
    `check_environment` for the same reason with no mock involved.

    **Inherited from the `deploy_kit`, verbatim, and never rebuilt.** m1 is the
    sole producer because the flow's premise is that modules 1–4 are talking
    about one container: a stage that re-derived the record could differ from
    m1's and nothing would notice, while a stage that inherits it cannot. The
    `deploy_kit` rather than the workset because that is where the record
    originates and it is the source `load_environment` already uses — one
    source in the real path and the mock path, so the mock exercises the real
    wiring rather than a parallel one.

    The `warnings` are M4.3.5's software-level differences. They travel in the
    record's own `warnings[]` channel, which `environment.schema.json` defines
    for exactly this, so m5 sees them without knowing anything about m4's
    premise block.
    """
    source = input_content("deploy_kit") / "items" / "codes" / "environment.yaml"
    if not source.is_file():
        die(f"no environment.yaml at {source}; m1's kit does not carry the environment record")
    argv = [
        sys.executable, str(package() / "assets" / "lib" / "env_render.py"),
        "--inherit", str(source), "--content-type", "code", "--out", str(out),
    ]
    for warning in warnings or []:
        argv += ["--warn", f"{warning['field']}={warning['expected']}!={warning['actual']}"]
    import subprocess

    # `E2E_STAGE` is what env_render stamps each warning with. Unset it reads as
    # `stage: ''`, and a warning that does not say which stage noticed the
    # difference is markedly less useful to m5 than one that does.
    environment = dict(os.environ)
    environment.setdefault("E2E_STAGE", "m4")
    done = subprocess.run(argv, capture_output=True, text=True, env=environment)
    if done.returncode != 0:
        # env_render validates before it writes, so a non-zero exit means the
        # record is malformed and nothing was written. A handoff carrying a
        # malformed environment record is worse than one carrying none: both
        # fail `check_environment`, and the malformed one looks like a record.
        die(f"env_render failed: {(done.stderr or done.stdout).strip()[-400:]}")


def expand_container_path(container_path: str) -> Path | None:
    """`@SGLANG_ROOT@/srt/x.py` -> the real path, using `container_roots.yaml`.

    Resolvable **because m1 through m4 share one container** (CONTRACT §5): the
    engine tree m5 will patch is on this filesystem right now. Outside that
    container it returns a path that does not exist, and every caller treats a
    missing file as a refusal rather than as a zero hash.
    """
    if not container_path.startswith("@") or "@/" not in container_path:
        return None
    name, _, relative = container_path[1:].partition("@/")
    roots = load_yaml(package() / "assets" / "lib" / "container_roots.yaml").get("roots") or {}
    root = (roots.get(name) or {}).get("path")
    return Path(str(root)) / relative if root else None


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    except Exception as exc:  # noqa: BLE001
        # `schema.validate` promises `SchemaError` and can raise other things.
        # One is live: with `referencing` unimportable the loader falls back to
        # a registry-less validator, and both `kernel_optimization` and
        # `workset` `$ref` `environment.schema.json`, so validation raises
        # `Unresolvable` rather than returning problems. Measured 2026-09-03.
        # Reported as a problem, because "could not be validated" and "is valid"
        # are not the same answer.
        return [f"the schema loader raised {type(exc).__name__}: {exc}; {name} was NOT validated"]
    return []
