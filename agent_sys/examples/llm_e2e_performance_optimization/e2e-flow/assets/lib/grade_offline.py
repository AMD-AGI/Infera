#!/usr/bin/env python3
"""Grade a sealed handoff with a validator's real body, offline, no node.

**Method is m2's** (2026-09-05, on `p9`'s `profiling_evidence`): build the zone
`validator/phase.py` builds, then run the validator's own `check.py` in it. The
verdict comes from the validator, not from a person reading numbers.

**Zone shape is not reconstructed** -- it is taken from two in-package sources
that already build one:
  assets/lib/zone.py            docstring: "validator/phase.py:236 names them:
                                args.json, inputs.json, materials.json, verdict.json"
  optimize_kernel.task/steps/70_selfcheck.sh   a working example of all three

  inputs.json     ["<handoff id>"]         zone.inputs() -> "the handoff ids
                                            this body is validating"
  materials.json  {"<handoff id>": "<relpath from zone to content dir>"}
  args.json       the validator module's own `args` block, WITH VARS RENDERED
  verdict.json    written by the body; all-true == PASS

**Why the var rendering is not optional.** Every validator here declares `args`,
and those blocks carry `${var}` / `${var:-default}` templates. An unrendered
template does not fail cleanly: it reaches the body as a literal string and the
body crashes, and a crash READS AS A REFUSAL. That exact mistake was made on
this package on 2026-09-05 -- ten lines of crash reported as ten rejections.
So `render()` below refuses to write an args.json still containing `${`, and
`main()` reports that as SKIPPED rather than as a verdict.

**What a PASS here is worth: level 3, not level 4.** It says a validator accepts
an artefact that a stage which really executed produced. It says NOTHING about
whether that validator would refuse a fault. Level 4 stays zero.

**A `--var` you supply is an argument you are grading, not the validator.**
m2's warning, and it bites here: the run's OWN staged package under
`zones/.../package/` is **not rendered** -- it still reads
`expect_ranks: '${expect_ranks:-8}'` -- so **a run does not record the `--var`
values it was launched with**, and no amount of reading the run tree recovers
them. Supplying `expect_ranks=4` turns `check_trace_coverage` on a TP4 run from
REFUSED into PASS. Both are honest; the verdict is a function of an argument
the run does not record, and a row like that should be reported as conditional
rather than as a pass. Prefer values read from the run's own artefacts
(`environment.yaml`'s `tp_size`) over ones you remember, and say which you did.

**Run a case with a known answer first.** Defect 1 below was caught only because
m2 had independently passed `p9`'s `profiling_evidence` and my harness said the
opposite. Nothing in here detected it; the contradiction did.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

#: Never run these offline.
#: `check_deploy_serves` does a real bring-up plus a 180 s stress run; CLAUDE.md
#: rule 10c keeps it out of any loop that runs on the login node.
REFUSE_TO_RUN = {"check_deploy_serves"}


def render(obj, variables: dict):
    """Substitute `${var}` / `${var:-default}` through a nested structure."""
    if isinstance(obj, dict):
        return {k: render(v, variables) for k, v in obj.items()}
    if isinstance(obj, list):
        return [render(v, variables) for v in obj]
    if isinstance(obj, str):
        def one(m):
            name, default = m.group(1), m.group(2)
            if name in variables:
                return str(variables[name])
            return default if default is not None else m.group(0)
        return VAR.sub(one, obj)
    return obj


def load_specs(package: Path):
    """(validators, kinds) from the package's yaml."""
    validators, kinds = {}, {}
    for f in sorted(package.glob("steps/*.yaml")) + [package / "shared.yaml"]:
        if not f.is_file():
            continue
        for doc in yaml.safe_load_all(f.read_text(encoding="utf-8")):
            for it in (doc if isinstance(doc, list) else [doc]):
                if not isinstance(it, dict):
                    continue
                if it.get("module") == "validator":
                    validators[it["name"]] = it
                elif it.get("module") == "handoff":
                    kinds[it["name"]] = it
    return validators, kinds


def sealed(run: Path):
    """Every `valid` handoff version in a run, with the closure that made it."""
    tasks = {}
    for p in glob.glob(str(run / "store" / "task" / "*.json")):
        d = json.load(open(p))
        tasks[d["id"]] = d
    rows = []
    for p in glob.glob(str(run / "store" / "handoff" / "*.json")):
        d = json.load(open(p))
        for v in d.get("versions", []):
            if v.get("status") != "valid":
                continue
            # **Do NOT trust `v['version']` for the directory name.** Measured
            # 2026-09-05 on run 20260905T163424-bdb4d8, handoff ae9d7162: the
            # store records `version: 0, status: valid` while the content is in
            # `v1/` and `v0/content/` is an EMPTY DIRECTORY.
            #
            # This is the false-refusal class and it is silent: an empty content
            # dir is a well-formed input, so every validator ran, wrote a
            # verdict, and refused for missing files. Four confident FAILs on an
            # artefact m2 had already passed. **Nothing here detected it -- the
            # contradiction with their result did.** So: take the newest
            # non-empty `content/`, and report which one was used.
            candidates = sorted(
                (p for p in (run / "handoffs" / d["id"]).glob("v*/content")
                 if p.is_dir() and any(p.iterdir())),
                key=lambda p: int(p.parent.name[1:]))
            if not candidates:
                continue
            content = candidates[-1]
            rows.append({
                "kind": d["type"],
                "id": d["id"],
                "closure": tasks.get(v.get("producer_task_id") or "", {}).get("closure", "?"),
                "content": content,
                "vdir": content.parent.name,
                "vrec": v["version"],
            })
    return sorted(rows, key=lambda r: (r["closure"], r["kind"]))


def grade(package: Path, spec: dict, handoff: dict, variables: dict, python: str):
    """Run one validator body over one handoff. Returns (state, detail)."""
    name = spec["name"]
    if name in REFUSE_TO_RUN:
        return "REFUSED", "not run offline: real bring-up + stress (CLAUDE.md 10c)"
    # **`entry.sh`, never `check.py` directly** (m2's recipe, RUN-PLAN "Grading a
    # sealed artefact offline"). `entry.sh` resolves `AGENT_SYS_DEMO_PYTHON`;
    # a validation zone gets a policy-derived `PATH` where `python3` is
    # `/usr/bin/python3`, which on this host has no `referencing`, so the body
    # dies importing `assets/lib/schema.py` BEFORE writing `verdict.json` and
    # the phase reports "nothing was decided" -- a validator that could not
    # start looks like one that was never asked. m5 paid for that one.
    body = package / "assets" / f"{name}.validator" / "entry.sh"
    if not body.is_file():
        return "SKIPPED", f"no entry.sh at {body}"

    args = render(spec.get("args") or {}, variables)
    blob = json.dumps(args, indent=2)
    if "${" in blob:
        left = sorted(set(m.group(0) for m in VAR.finditer(blob)))
        # A crash from an unrendered template reads as a refusal. Refuse to be
        # that: report the missing names instead of producing a verdict.
        return "SKIPPED", f"unrendered args: {', '.join(left)}"

    with tempfile.TemporaryDirectory(prefix="grade.", dir=os.path.expanduser("~")) as zone:
        z = Path(zone)
        (z / "args.json").write_text(blob, encoding="utf-8")
        (z / "inputs.json").write_text(json.dumps([handoff["id"]]), encoding="utf-8")
        (z / "materials.json").write_text(json.dumps(
            {handoff["id"]: os.path.relpath(handoff["content"], z)}, indent=2), encoding="utf-8")
        env = dict(os.environ,
                   AGENT_SYS_TASK_PACKAGE=str(package),
                   AGENT_SYS_DEMO_PYTHON=python)
        proc = subprocess.run(["sh", str(body)], cwd=z, env=env,
                              capture_output=True, text=True, timeout=900)
        vpath = z / "verdict.json"
        if not vpath.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            # rc alone cannot tell a refusal from a crash; say which this was.
            return "CRASH", f"rc={proc.returncode}, no verdict.json | " + (
                " / ".join(tail[-3:]) if tail else "no output")
        verdict = json.loads(vpath.read_text(encoding="utf-8"))
        ok = all(verdict.values())
        failed = [k for k, v in verdict.items() if not v]
        report = (z / "validator_report.txt")
        detail = "" if ok else "false: " + ", ".join(failed)
        if not ok and report.is_file():
            detail += " | " + " / ".join(report.read_text(encoding="utf-8").strip().splitlines()[:2])
        return ("PASS" if ok else "FAIL"), detail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--run", required=True, action="append")
    ap.add_argument("--var", action="append", default=[])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only-kind", default="")
    a = ap.parse_args()

    variables = dict(v.split("=", 1) for v in a.var)
    package = Path(a.package).resolve()
    validators, kinds = load_specs(package)

    print(f"{'run':<24} {'closure':<24} {'kind':<32} {'dir':<4} {'validator':<30} {'verdict':<8} detail")
    for r in a.run:
        run = Path(r).resolve()
        for h in sealed(run):
            if a.only_kind and h["kind"] != a.only_kind:
                continue
            for vname in (kinds.get(h["kind"], {}).get("validators") or []):
                spec = validators.get(vname)
                if spec is None:
                    state, detail = "SKIPPED", "validator not declared in this package"
                else:
                    state, detail = grade(package, spec, h, variables, a.python)
                print(f"{run.name:<24} {h['closure']:<24} {h['kind']:<32} {h['vdir']:<4} "
                      f"{vname:<30} {state:<8} {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
