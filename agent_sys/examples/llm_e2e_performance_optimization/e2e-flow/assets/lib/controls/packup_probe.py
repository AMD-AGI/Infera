#!/usr/bin/env python3
"""`check_packup_shape`, run as a VALIDATOR on a staged handoff.

    python3 packup_probe.py                     # the out-of-band kit
    E2E_PACKUP_KIT=<dir> python3 packup_probe.py   # any other content dir

Sibling of `artefact_neg.py` and deliberately a **second instrument, not a
replacement**. m5's drives their own harness over their own seven; this one was
built independently and reached fourteen cases with zero probe corrections where
theirs needed three. Collapsing them to one would delete the only independent
check either of us has, which is m5's call and their words.

## WHY IT RUNS THE ENTRY AND NOT `check()`

`check_packup_shape` was, until 2026-09-04, **the only validator in the package
that had never produced a verdict** — `e2e_packup` is `created` in all 69 run
trees, so nothing ever asked it a question. m5's battery already reached its
refusal text, but that calls the function. This starts the real `entry.sh` in a
real zone — `args.json` read out of `steps/m5_integration.yaml`, `inputs.json`,
`materials.json`, the producer env row, `/usr/bin/python3` — which is the shape
`validator/phase.py:146` starts it in.

*A bar that is read* and *a check that detects* are different claims (m5); *the
function refuses* and *the validator refuses* are two more.

## WHAT FOURTEEN PASSING CASES DO AND DO NOT LICENSE

They show this validator detects **the specific breakage injected here**. They
do **not** show it detects the breakage the world produces. m5's standing caveat
applies unchanged, and this file is its own proof: `logs/ = one empty
subdirectory` and `scripts/ = one zero-byte file` were injected breakages that
the validator did **not** detect, in a battery that was otherwise passing.
A green run of this file is evidence about fourteen cases, not about the class.

## THE LIVENESS CONTROLS ARE NOT DECORATION — THEY ARE WHY THE TWO GAPS WERE FILABLE

A PASS you suspect is a hole has two explanations, and **the PASS itself cannot
distinguish them** (m3): the input genuinely satisfies the check, or the check
never ran on that path. m5 lost three probes to the second in one morning, all
three returning a confident PASS against a working validator.

The refusing control decides it in one run. `CONTROL scripts/ emptied` refuses,
so the body demonstrably opens `scripts/` — which makes `scripts/ = one
zero-byte file` passing a real hole rather than a misaimed probe. Every case
below marked `GAP` is paired with a `CONTROL` on the same path, and a `GAP`
without its control is not a finding.

CONTRACT.md §4.4 already carries two thirds of this: *prove the probe can fail
before believing it passed*, and *prove a reader can pass before believing it
failed*. This is the third corner — **prove the path is live before believing a
PASS is a hole.**

## THE BASELINE IS A CASE

It must PASS. A battery that only ever sees refusals cannot tell a working
validator from one that refuses everything (§4.4).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

PKG = Path(os.environ.get("AGENT_SYS_TASK_PACKAGE") or Path(__file__).resolve().parents[3])
ENTRY = PKG / "assets" / "check_packup_shape.validator" / "entry.sh"

#: The 47-file kit produced out of band by `integration`'s own unmodified
#: `packup.py`, which is what MOCK-MAP (F) feeds the mock arm. Override to run
#: against a real one — every case below has been confirmed identical against
#: m5's 73-file `e2e_packup_real`, which is how we know the gaps were in the
#: body and not in the fixture. A single-kit battery cannot make that claim.
KIT = Path(os.environ.get(
    "E2E_PACKUP_KIT",
    "/shared_nfs/yihou/agent_sys/debugging/integration/packup-out-of-band/content"))

SCRATCH = Path(os.environ.get("E2E_PROBE_SCRATCH", tempfile.gettempdir())) / "packup-probe"


def args_from_yaml() -> dict:
    """`steps/m5_integration.yaml`'s `args`, read rather than retyped.

    So a bar its owner changes changes here too — a battery carrying its own
    copy of the thresholds is testing the copy.
    """
    import yaml
    for doc in yaml.safe_load_all((PKG / "steps" / "m5_integration.yaml").read_text()):
        for entry in (doc if isinstance(doc, list) else [doc]):
            if isinstance(entry, dict) and entry.get("module") == "validator" \
                    and entry.get("name") == "check_packup_shape":
                return dict(entry.get("args") or {})
    raise SystemExit("check_packup_shape is not defined in steps/m5_integration.yaml")


def run(label: str, mutate, *, stage: bool = True) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        t = Path(tmp)
        zone, content = t / "zone", t / "staged"
        zone.mkdir()
        if stage:
            shutil.copytree(KIT, content, symlinks=True)
            # The sealed kits are read-only on NFS; a mutation has to be able to
            # unlink. Scoped to this temporary copy and never to the source.
            for p in [content, *content.rglob("*")]:
                p.chmod(p.stat().st_mode | 0o200)
            mutate(content)
        # Deterministic, so a schema complaint about the id reads as a schema
        # complaint rather than as this harness's spelling. Learned the hard way
        # on the interpreter sweep, where naming staged inputs by kind made m4's
        # error indistinguishable from mine.
        hid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"packup-probe/{label}"))
        (zone / "args.json").write_text(json.dumps(args_from_yaml()))
        (zone / "inputs.json").write_text(json.dumps([hid]))
        (zone / "materials.json").write_text(json.dumps({hid: str(content)}))
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
            "AGENT_SYS_TASK_PACKAGE": str(PKG),
            "AGENT_SYS_DEMO_PACKAGE": str(PKG),
            "AGENT_SYS_DEMO_PYTHON": "/usr/bin/python3",
        }
        r = subprocess.run(["/bin/sh", str(ENTRY)], cwd=zone, env=env,
                           capture_output=True, text=True, timeout=300)
        vf = zone / "verdict.json"
        verdict = json.loads(vf.read_text()) if vf.is_file() else None
        said = [l.strip()[2:] for l in (r.stdout or "").splitlines()
                if l.strip().startswith("- ")]
        return {"rc": r.returncode, "verdict_file": vf.is_file(),
                "result": None if verdict is None else verdict.get(hid),
                "said": said, "stderr": (r.stderr or "").strip()[-400:]}


# --------------------------------------------------------------------------- #
# The mutations. One plausible producer mistake each.

def m_none(c):
    pass


def m_no_items(c):
    shutil.rmtree(c); c.mkdir()


def m_codes_empty(c):
    d = c / "items" / "codes"; shutil.rmtree(d); d.mkdir(parents=True)


def m_readme_empty(c):
    (c / "items" / "codes" / "README.md").write_text("# Kit\n\n")


def m_reproduce_prose(c):
    """The commands survive as prose. A producer who reformatted the document
    and lost its fences leaves every word and nothing executable."""
    p = c / "items" / "codes" / "REPRODUCE.md"
    p.write_text("\n".join(l for l in p.read_text().splitlines()
                           if not l.lstrip().startswith(("```", "~~~"))
                           and not l.startswith("    ")))


def m_results_thin(c):
    d = c / "items" / "codes" / "results"
    for p in sorted(d.iterdir())[3:]:
        p.unlink() if p.is_file() else shutil.rmtree(p)


def m_placeholder(c):
    p = c / "items" / "codes" / "notes.md"
    p.write_text(p.read_text() + "\n## Follow-up\nTBD\n")


def _wipe(c, name):
    d = c / "items" / "codes" / name; shutil.rmtree(d); d.mkdir()


def m_logs_emptied(c):
    _wipe(c, "logs")


def m_scripts_emptied(c):
    _wipe(c, "scripts")


def m_results_emptied(c):
    _wipe(c, "results")


def m_logs_one_empty_subdir(c):
    """REGRESSION for m5's `ad6d431`. Passed before it; must refuse after.

    The real `logs/` is 17 files under seven subdirectories, and `require_dirs`
    used to accept `any(path.iterdir())` — so all of it could go, leaving one
    empty directory, and the kit still graded complete."""
    _wipe(c, "logs"); (c / "items" / "codes" / "logs" / "bench_stock").mkdir()


def m_scripts_one_zero_byte(c):
    """REGRESSION for `ad6d431`. The same shape on the other unmeasured dir."""
    _wipe(c, "scripts"); (c / "items" / "codes" / "scripts" / "run.sh").write_text("")


def m_readme_fenced_output(c):
    """NOT a defect, and kept to record that it was considered.

    25 lines of pasted terminal output clear the 20-line README floor, because
    `content_lines` counts inside fences. m5's judgement, which I accept: the
    floor is a *substance* floor and pasted output is the most valuable thing a
    README carries. Only the name misleads. Expected PASS — if this ever starts
    refusing, someone narrowed the floor to prose without saying so."""
    (c / "items" / "codes" / "README.md").write_text(
        "# Kit\n\n```\n" + "\n".join(f"[rank0] step {i} ok" for i in range(25)) + "\n```\n")


#: `(label, mutation, expected verdict, stage the content at all)`
CASES = [
    ("baseline — MUST PASS",                        m_none,                  True,  True),
    ("absent packup — nothing staged",              m_none,                  False, False),
    ("empty content dir — no items/",               m_no_items,              False, True),
    ("items/codes/ exists and is empty",            m_codes_empty,           False, True),
    ("README.md emptied",                           m_readme_empty,          False, True),
    ("REPRODUCE.md reformatted to prose",           m_reproduce_prose,       False, True),
    ("results/ pruned to 3 files",                  m_results_thin,          False, True),
    ("notes.md carries TBD",                        m_placeholder,           False, True),
    ("CONTROL logs/ emptied",                       m_logs_emptied,          False, True),
    ("CONTROL scripts/ emptied",                    m_scripts_emptied,       False, True),
    ("CONTROL results/ emptied",                    m_results_emptied,       False, True),
    ("GAP-CLOSED logs/ one empty subdir",           m_logs_one_empty_subdir, False, True),
    ("GAP-CLOSED scripts/ one zero-byte file",      m_scripts_one_zero_byte, False, True),
    ("BY DESIGN README 25 fenced output lines",     m_readme_fenced_output,  True,  True),
]


def main() -> int:
    if not KIT.is_dir():
        print(f"no kit at {KIT} — set E2E_PACKUP_KIT to a handoff content directory",
              file=sys.stderr)
        return 2

    rows = [(label, expect, run(label, mut, stage=stage))
            for label, mut, expect, stage in CASES]

    print(f"check_packup_shape as a validator, kit = {KIT}")
    print("=" * 100)
    print(f"{'case':<42} {'rc':>3} {'verdict.json':>12} {'result':>7} {'':>6}  said")
    for label, expect, r in rows:
        ok = r["result"] is expect
        print(f"{label:<42} {r['rc']:>3} "
              f"{('yes' if r['verdict_file'] else '**NO**'):>12} "
              f"{str(r['result']):>7} {('ok' if ok else '<<<<<<'):>6}  "
              f"{(r['said'][0][:60] if r['said'] else '')}")

    # **The two shapes that are worse than a wrong verdict**, in the order that
    # matters: a body that decided nothing, then a battery that cannot observe a
    # pass. Either makes every other row unreadable.
    dead = [l for l, _, r in rows if not r["verdict_file"]]
    wrong = [l for l, e, r in rows if r["result"] is not e]
    print("\n" + "=" * 100)
    print("died writing no verdict (the dangerous category):", dead or "none")
    print("verdict differs from the bar:                    ", wrong or "none")
    if not any(r["result"] is True for _, _, r in rows):
        print("!! this battery never observed a PASS — it cannot tell a working "
              "validator from one that refuses everything")
        return 2
    if not any(r["result"] is False for _, _, r in rows):
        print("!! this battery never observed a refusal — the paths may be dead")
        return 2
    return 1 if (dead or wrong) else 0


if __name__ == "__main__":
    sys.exit(main())
