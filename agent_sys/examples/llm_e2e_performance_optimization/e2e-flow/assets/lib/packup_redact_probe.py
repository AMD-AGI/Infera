#!/usr/bin/env python3
"""Predict `redact.py`'s verdict on a packup BEFORE `packup` runs.

**Why this exists.** `packup.py:529` runs `redact.py` with `check=True`. One
unnameable absolute path in a `.py`/`.sh`/`.json`/`.jsonl` exits 1, the exception
propagates, `packup.py` dies, and `e2e_packup` — `is_end: true` — is never
written. **A successful five-stage chain dies at its last rung.** This answers
the question minutes earlier, for free, without touching the run.

**It asks redact, it does not imitate redact.** It imports the real module and
calls the real `substitute()` then `offenders()`. A grep for a literal path
would answer a different question — what a path *looks like* — and the thing
that decides is what `redact` *refuses* after substitution.

**What it scans, and why not everything.** `packup.py` is SELECTIVE, despite its
own comment at :501 saying it "carries eight upstream handoffs verbatim". The
trees it copies wholesale or as refusable suffixes:

    packup.py:180       kernel_optimization/items/codes/**   copytree, WHOLESALE
    packup.py:172-177   {patch_overlay, stock.measurement,
                         patched.measurement}/items/command  -> scripts/*.sh
    packup.py:156-166   <kind>/items/logs/*                  (several kinds)
    packup.py:133       deploy_kit  ONE items/codes/**/README.md   (.md, not refusable)

`--all` scans every handoff instead: an over-approximation, useful as an early
warning, and it WILL report things that never travel. The default models the
selection above and cites the lines so a reader can check the model rather than
trust it.

Usage
-----
    python3 packup_redact_probe.py <run-dir> [--pid <orchestrator pid>] [--all]
    python3 packup_redact_probe.py --self-test

Prefixes are reconstructed the way `packup.py:511-527` builds them. With
`--pid` they are read from the live launch line, which is the only place a run
records what it was launched with.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path("/home/yihou/dev/git.16-19/infera/agent_sys/examples/"
            "llm_e2e_performance_optimization/e2e-flow")
REFUSABLE = {".py", ".sh", ".json", ".jsonl"}


def load_redact():
    """The real module. Anything else answers a different question.

    **Every failure path here names the module, because this probe's whole
    value is that it imports the real one.** `spec_from_file_location` returns
    `ModuleSpec | None` and a spec's `loader` may be `None`; passing either
    straight through fails with *"'NoneType' object has no attribute 'loader'"*
    at the one moment the reader needs to be told *which file was missing*.
    Same idea as the `NOTHING TO SCAN` exit, one level down: a tool that cannot
    do its job must say what it could not do.
    """
    path = REPO / "assets" / "lib" / "redact.py"
    if not path.is_file():
        sys.exit(f"probe: redact.py not found at {path} — cannot answer, and a "
                 f"hand-rolled imitation would answer a different question")
    spec = importlib.util.spec_from_file_location("redact", path)
    if spec is None or spec.loader is None:
        sys.exit(f"probe: {path} exists but python produced no import spec/loader "
                 f"for it — cannot load the real redact module")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # noqa: BLE001 — the reason matters more than the type
        sys.exit(f"probe: importing {path} raised {type(exc).__name__}: {exc}")
    for needed in ("substitute", "offenders"):
        if not hasattr(mod, needed):
            sys.exit(f"probe: {path} has no {needed}() — redact's interface moved "
                     f"and this probe is now measuring nothing")
    return mod


def vars_from_pid(pid: str) -> dict:
    """The launch line is the only record of what a run was launched with."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError as exc:
        sys.exit(f"probe: cannot read /proc/{pid}/cmdline: {exc}")
    out = {}
    for tok in raw.split("\0"):
        if "=" in tok and not tok.startswith("-"):
            k, _, v = tok.partition("=")
            out[k] = v
    return out


def prefixes(v: dict) -> list[tuple[str, str]]:
    """`packup.py:511-527`, rebuilt. Absolute and non-root only, as it requires."""
    pairs = [
        ("TASK_PACKAGE", str(REPO)),
        ("TMPDIR", "/tmp"),
        ("HOME", str(Path.home())),
    ]
    model = v.get("model_path", os.environ.get("E2E_MODEL_PATH", ""))
    if model:
        pairs.append(("MODEL_MOUNT", str(Path(model).parent)))
    for name, key in (("WORK_ROOT", "work_root"), ("MOCK_ROOT", "mock_root")):
        val = v.get(key, "")
        if val:
            pairs.append((name, val))
    kept = [(p.rstrip("/"), n) for n, p in pairs if p.startswith("/") and p.rstrip("/")]
    # Longest first, exactly as packup/redact require: a nested pair must not
    # depend on argument order.
    kept.sort(key=lambda pair: len(pair[0]), reverse=True)
    return kept


def targets(run: Path, scan_all: bool) -> list[Path]:
    handoffs = run / "handoffs"
    if not handoffs.is_dir():
        sys.exit(f"probe: no handoffs/ under {run}")
    if scan_all:
        return [p for p in handoffs.rglob("*") if p.is_file() and p.suffix in REFUSABLE]
    # **Modelled per SOURCE KIND, not per handoff**, because `packup.py` takes a
    # different slice from each one. An earlier version of this function scanned
    # `items/codes` for every kind and reported sixteen offenders, of which zero
    # travel: twelve were `deploy_kit`'s kit scripts and four were
    # `operator_workset`'s, and `packup.py` copies neither. **A probe with the
    # wrong population is not a conservative probe, it is a wrong one.**
    kind_of = {}
    for rec in sorted((run / "store" / "handoff").glob("*.json")):
        import json
        d = json.loads(rec.read_text())
        kind_of[d["id"]] = d["type"]

    LOGS = ("deploy_kit", "profiling_evidence", "operator_workset",
            "kernel_optimization", "patch_overlay",
            "stock.measurement", "patched.measurement")          # packup.py:156-158
    COMMAND = ("patch_overlay", "stock.measurement", "patched.measurement")  # :172
    CODES_WHOLESALE = ("kernel_optimization",)                    # :180 copytree

    picked: list[Path] = []
    for h in handoffs.iterdir():
        kind = kind_of.get(h.name)
        if kind is None:
            continue
        for ver in sorted(h.glob("v*")):
            c = ver / "content"
            if kind in CODES_WHOLESALE and (c / "items" / "codes").is_dir():
                picked += [p for p in (c / "items" / "codes").rglob("*")
                           if p.is_file() and p.suffix in REFUSABLE]
            if kind in LOGS and (c / "items" / "logs").is_dir():
                picked += [p for p in (c / "items" / "logs").iterdir()
                           if p.is_file() and p.suffix in REFUSABLE]
            if kind in COMMAND and (c / "items" / "command").is_file():
                picked.append(c / "items" / "command")
    return picked


def scan(mod, files: list[Path], mapping) -> list[tuple[Path, int, str]]:
    hits = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # redact scans UTF-8; a binary is not its business
        for lineno, path in mod.offenders(mod.substitute(text, mapping)):
            hits.append((f, lineno, path))
    return hits


def self_test(mod) -> int:
    """A case whose answer is already known, before the instrument is trusted."""
    mapping = [("/data/yihou/e2e_flow11", "WORK_ROOT"),
               ("/apps/data/models", "MODEL_MOUNT"),
               ("/home/yihou", "HOME"), ("/tmp", "TMPDIR")]
    mapping.sort(key=lambda p: len(p[0]), reverse=True)
    cases = [
        ("/data/yihou/e2e_flow11/kfo/x.json", False),
        ("/apps/data/models/Qwen3-32B", False),
        ("/dev/md0", False),
        ("/shared_nfs", False),          # one segment: CANDIDATE needs two
        ("/data/yihou/Magpie/tools", True),
        ("/mnt/m2m_nobackup/yihou", True),
    ]
    bad = 0
    for text, expect in cases:
        got = bool(mod.offenders(mod.substitute(f"x = {text}", mapping)))
        ok = got == expect
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {text:<34} refused={got} expected={expect}")
    print("self-test PASSED" if not bad else f"self-test FAILED ({bad})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--pid", default=None, help="orchestrator pid; reads its launch line")
    ap.add_argument("--all", action="store_true", help="scan every handoff, not packup's selection")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    mod = load_redact()
    if a.self_test:
        return self_test(mod)
    if not a.run:
        ap.error("a run directory is required unless --self-test")

    v = vars_from_pid(a.pid) if a.pid else {}
    mapping = prefixes(v)
    print(f"probe: prefixes redact will hold ({'from pid ' + a.pid if a.pid else 'no pid given'}):")
    for p, n in mapping:
        print(f"    {n}={p}")
    if not a.pid:
        print("    NOTE: no --pid, so WORK_ROOT/MODEL_MOUNT/MOCK_ROOT may be missing")
        print("          and this run will OVER-report. Pass --pid for the real answer.")

    files = targets(Path(a.run), a.all)
    print(f"probe: {len(files)} refusable file(s) "
          f"({'ALL handoffs' if a.all else 'packup.py selection: items/codes, items/logs, items/command'})")
    if not files:
        # **"Zero offenders" and "nothing to look at" are two different answers
        # and only one of them is reassuring.** The exposure lives in artefacts
        # that may not exist yet; reporting a clean result over an empty
        # population is how a probe fails toward reassurance.
        print("\nprobe: NOTHING TO SCAN — no refusable file in packup's selection "
              "exists yet.\n       This is NOT a clean result. The exposure is "
              "kernel_optimization/items/codes\n       (copytree, packup.py:180) and "
              "m5's items/command scripts; re-run when\n       those seal.")
        return 3

    hits = scan(mod, files, mapping)
    if not hits:
        print("\nprobe: NO OFFENDERS. redact would not refuse this content.")
        return 0
    print(f"\nprobe: {len(hits)} OFFENDER(S) — packup would exit 1 and e2e_packup "
          f"(is_end: true) would never be written:\n")
    for f, lineno, path in hits:
        print(f"  {path}\n      {f}:{lineno}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
