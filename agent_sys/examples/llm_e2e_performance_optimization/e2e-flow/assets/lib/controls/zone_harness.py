#!/usr/bin/env python3
"""Run one of this package's validators in a synthetic zone, as the runner does.

Committed because `artefact_neg.py` beside it cannot run without it, and a
negative-test battery nobody can execute is not much better than no battery.

`validator/phase.py:236` names the four files a zone carries — `args.json`,
`inputs.json`, `materials.json`, `verdict.json`. The body is started with `cwd`
set to a fresh zone holding the first three and owes the fourth.

**The environment matters and is the point.** A validator declares no agent, so
the package's `env:` block never reaches it: it gets the GLOBAL row and nothing
else, and `python3` resolves through the zone's PATH. Reproducing that with
`env -i` rather than inheriting your shell is what makes a result here mean
anything — a by-hand run that inherits your environment is a different
experiment and will tell you the subject is fine. That hole has bitten this
package five times; do not reintroduce it in the harness that tests for it.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile

PKG = pathlib.Path(__file__).resolve().parents[3]

#: The GLOBAL row and nothing else. `cli/main.py:666,668` exports these two.
BASE_ENV = {
    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "AGENT_SYS_DEMO_PACKAGE": str(PKG),
}


def run(validator: str, materials: dict, args: dict, *, extra_env: dict | None = None,
        home: str | None = None) -> tuple[subprocess.CompletedProcess, dict | None, str]:
    """Returns `(proc, verdict_or_None, zone_path)`. The zone is left on disk."""
    zone = pathlib.Path(tempfile.mkdtemp(prefix="zone."))
    (zone / "args.json").write_text(json.dumps(args))
    (zone / "inputs.json").write_text(json.dumps(list(materials)))
    (zone / "materials.json").write_text(
        json.dumps({k: str(v) for k, v in materials.items()}))
    env = dict(BASE_ENV, HOME=home or str(zone))
    env.update(extra_env or {})
    proc = subprocess.run(
        ["/bin/sh", str(PKG / "assets" / f"{validator}.validator" / "entry.sh")],
        cwd=zone, env=env, capture_output=True, text=True)
    verdict_file = zone / "verdict.json"
    verdict = json.loads(verdict_file.read_text()) if verdict_file.is_file() else None
    return proc, verdict, str(zone)


def show(title: str, validator: str, materials: dict, args: dict,
         expect: bool | None = None, **kw) -> bool:
    proc, verdict, zone = run(validator, materials, args, **kw)
    got = None if verdict is None else all(bool(v) for v in verdict.values())
    mark = "   " if expect is None else ("OK " if got == expect else "!! ")
    print(f"{mark}{title}  [{validator}]  verdict={got} rc={proc.returncode}")
    for line in (proc.stdout or "").strip().splitlines()[-6:]:
        print("      |", line[:160])
    if proc.returncode and (proc.stderr or "").strip():
        print("     E|", (proc.stderr or "").strip().splitlines()[-1][:160])
    report = pathlib.Path(zone) / "validator_report.txt"
    if report.is_file():
        print(f"      report: {len(report.read_text().splitlines())} line(s) in {zone}")
    return got == expect if expect is not None else True
