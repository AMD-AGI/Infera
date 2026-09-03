#!/usr/bin/env python3
"""Run one of the workset's entrypoints and enforce its acceptance criterion.

Shared by STEP 4 and STEP 5, because the two differ only in which entrypoint
they name and what counts as acceptance. Writing it twice would be two places
for the candidate-selection convention to drift.

**The three flag names are configurable, and that is deliberate rather than
lazy.** `workset.schema.json`'s `$defs/entrypoint` fixes `cmd` and `report` and
says nothing about how a *candidate* implementation is selected, while its
`$defs/performance_report` carries `impl` and `impl_path` — so a selector exists
and its spelling is m3's to declare. Until it is in the schema these defaults are
the convention, and a workset that spells it differently sets
`KFO_IMPL_FLAG` / `KFO_IMPL_PATH_FLAG` / `KFO_REPORT_FLAG` rather than being
silently mis-driven. `check_speedup_substantiated` carries the same three as
`args`, and **the two must agree** — a workset driven one way here and another
way there produces two reports of the same kernel that disagree for no reason a
reader can see.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _lib as lib  # noqa: E402

#: A PATH a compiler can actually work in. A Triton kernel compiles
#: `hip_utils.c` through `/bin/gcc`, which then needs `as`, `ld` and
#: `collect2`. The baseline side survives a missing PATH because plain torch
#: compiles nothing, so the symptom is *only the candidate fails* — which reads
#: exactly like a broken kernel and is not one. Measured 2026-09-01.
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _env(scratch: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = env.get("PATH") or _PATH
    env.setdefault("TRITON_CACHE_DIR", str(scratch / "triton_cache"))
    env.setdefault("KNOWLEDGE_LOCAL_ROOT", str(scratch / "knowledge"))
    env["TMPDIR"] = env.get("TMPDIR") or str(scratch / "tmp")
    for key in ("TRITON_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    # HIP_VISIBLE_DEVICES is deliberately NOT defaulted: on a shared host card 0
    # is somebody else's, and a default here moves the measurement onto it
    # silently.
    return env


def _accept_correctness(report: dict, declared: list[str], operator_id: str) -> list[str]:
    problems: list[str] = []
    if report.get("passed") is not True:
        problems.append("the report says passed != true")
    seen: set[str] = set()
    for entry in report.get("operators") or ():
        if not isinstance(entry, dict) or entry.get("operator_id") != operator_id:
            continue
        if entry.get("ran") is not True:
            problems.append(f"{operator_id}: ran != true ({entry.get('failure')!r})")
        if entry.get("passed") is not True:
            problems.append(f"{operator_id}: passed != true ({entry.get('failure')!r})")
        for shape in entry.get("shapes") or ():
            if not isinstance(shape, dict):
                continue
            seen.add(str(shape.get("case_id")))
            if shape.get("passed") is not True:
                problems.append(f"{shape.get('case_id')}: failed ({shape.get('failure')!r})")
    missing = sorted(c for c in declared if c not in seen)
    if missing:
        problems.append(f"the workset declares correctness cases that were never run: {missing}")
    return problems


def _accept_performance(report: dict, declared: list[str], operator_id: str) -> list[str]:
    problems: list[str] = []
    measured = lib.report_medians(report, operator_id)
    missing = sorted(c for c in declared if c not in measured)
    if missing:
        problems.append(f"no figure for {missing}")
    for entry in report.get("operators") or ():
        if not isinstance(entry, dict) or entry.get("operator_id") != operator_id:
            continue
        if entry.get("ran") is not True:
            problems.append(f"{operator_id}: ran != true ({entry.get('failure')!r})")
        for shape in entry.get("shapes") or ():
            if isinstance(shape, dict) and not isinstance(shape.get("rsd"), (int, float)):
                problems.append(f"{shape.get('case_id')}: no rsd recorded, so the spread is unknown")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--role", required=True, choices=("correctness", "performance"))
    ap.add_argument("--candidate", default=None, help="omit to run the workset's own baseline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pinned = lib.load_json(Path(a.inputs))
    operator_id = str(pinned["operator_id"])
    root = Path(pinned["workset_root"])
    entry = (pinned["entrypoints"] or {})[a.role]
    cmd = str(entry.get("cmd") or "").strip()
    if not cmd:
        lib.die(f"the workset declares no {a.role} entrypoint")

    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = [*cmd.split(), os.environ.get("KFO_REPORT_FLAG", "--report"), str(out)]
    impl_flag = os.environ.get("KFO_IMPL_FLAG", "--impl")
    if a.candidate:
        argv += [impl_flag, "candidate",
                 os.environ.get("KFO_IMPL_PATH_FLAG", "--impl-path"), str(Path(a.candidate).resolve())]
    else:
        argv += [impl_flag, "baseline"]

    timeout = float(entry.get("timeout_s") or 3600)
    print(f"running: {' '.join(argv)}  (cwd {root})", file=sys.stderr)
    try:
        proc = subprocess.run(
            argv, cwd=root, env=_env(lib.scratch()), timeout=timeout,
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        lib.die(f"{argv[0]} is not executable from {root}")
    except subprocess.TimeoutExpired:
        lib.die(f"the {a.role} entrypoint exceeded {timeout:.0f}s")
    sys.stderr.write(proc.stderr[-4000:])
    if proc.returncode != 0:
        lib.die(f"the {a.role} entrypoint exited {proc.returncode}")
    if not out.is_file():
        lib.die(f"the {a.role} entrypoint wrote no report at {out}")

    report = json.loads(out.read_text(encoding="utf-8"))
    # No `schema.validate` call here: the report's schema is
    # `workset.schema.json#/$defs/{correctness,performance}_report`, a `$defs`
    # entry rather than a top-level schema, and `assets/lib/schema.py` resolves
    # by *file name*. The acceptance criteria below read the same fields those
    # `$defs` require, so the check is made rather than skipped — it is just not
    # made by the loader.
    declared = (pinned.get("shapes") or {}).get(a.role) or []
    if a.role == "correctness":
        problems = _accept_correctness(report, declared, operator_id)
    else:
        problems = _accept_performance(report, declared, operator_id)

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        if a.role == "correctness":
            print(
                "Correctness is not a percentage. Do not proceed to STEP 5; go to STEP 6 and "
                "write the handoff with this failure in it.",
                file=sys.stderr,
            )
        return 1

    if a.role == "performance":
        medians = lib.report_medians(report, operator_id)
        print("ok: " + ", ".join(f"{c} {medians[c] * 1000:.2f}us" for c in sorted(medians)))
    else:
        print(f"ok: {len(declared)} correctness case(s) passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
