#!/usr/bin/env python3
"""Run every validator against the sealed corpus, in parallel, with no run.

**Why this exists.** Every mock failure in this effort has been a validator
refusing replayed data, and they were discovered ONE PER FULL-CHAIN RUN — the
graph stops at the first invalid handoff, so a run costing minutes to hours
returns exactly one mismatch. Five stages of mismatches therefore take days.

A validator reads three files from its cwd (`assets/lib/zone.py`):

    inputs.json      ["<hid>"]
    args.json        the parameters the step yaml declares
    materials.json   {"<hid>": "<path to the staged content dir>"}

and writes `verdict.json`. Nothing else. So a synthetic zone is three small
JSON files, and the whole 21-validator surface can be enumerated at once.

Usage:  python3 probe_validators.py [--corpus DIR] [--jobs N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile

#: The package this file lives in — two levels up from `assets/lib/`. Derived,
#: not hardcoded, so the probe travels with the package instead of pinning one
#: checkout.
PKG = pathlib.Path(__file__).resolve().parents[2]
CORPUS = pathlib.Path("/shared_nfs/yihou/agent_sys/cheat_for_mock")


def declared() -> list[tuple[str, str, dict]]:
    """(kind, validator, args) for every validator declared in the step yamls.

    **Read from the `module: validator` blocks, which is where the args are.**

    The first version walked the *handoffs* and took args from the validator
    entry there. Every handoff in this package writes the string form —
    `validators: [check_environment, check_profiling_evidence]` — so that
    branch returned `{}` for all 44 pairs and **118 declared args across 21
    validators were silently discarded**. Found by m2, 2026-09-05, by running
    their own row both ways on the same body and corpus:

        args = {}           verdict FALSE  "parts.json claims part(s) … that
                                            this kind does not declare"
        args = the step's   verdict TRUE   passed, four informative notes

    With `require_parts` empty, every part the artefact declares is
    "undeclared" — the refusal was the empty arg set talking.

    **The failure was worse in the passing direction.** A validator stripped of
    its thresholds passes trivially, so the 26 `PASS` rows of the first run were
    the weakest possible evidence, not the strongest. §4.4's doctrine, in the
    tool written to apply it: a check that cannot fail is not a check.

    The docstring above this function already said *"the args are half the
    contract — a validator run without the parameters its step passes is not
    the validator the graph runs"*, and the code under it did the opposite.
    """
    import yaml

    out: list[tuple[str, str, dict]] = []
    for step in sorted((PKG / "steps").glob("*.yaml")) + [PKG / "shared.yaml"]:
        if not step.is_file():
            continue
        doc = yaml.safe_load(step.read_text(encoding="utf-8")) or {}
        for mod in doc if isinstance(doc, list) else [doc]:
            for v in _walk_validators(mod):
                name = str(v.get("name") or "")
                if not name.startswith("check_"):
                    continue
                args = _substitute(v.get("args") or v.get("parameters") or {}, MOCK_VARS)
                if str((v.get("tags") or {}).get("cost", "")) == "gpu_hours":
                    _GPU_HOURS.add(name)
                for kind in v.get("inputs") or []:
                    out.append((str(kind), name, args))
    return out


#: Validators the probe **cannot** grade, by their own declaration.
#:
#: `tags.cost: gpu_hours` says the check re-measures on a node. The probe runs on
#: a login node with no card and no live allocation, so these refuse every
#: artefact identically — a fact about where the probe runs, not about the
#: artefact. m3 argued the category from `check_workset_runs`; it covers three:
#:
#:     check_deploy_serves          brings a deployment up and loads it
#:     check_workset_runs           measure_in_container.sh:142, "no measurement card"
#:     check_speedup_substantiated  re-measures the kernel; that IS its purpose
#:
#: **`NO_CORPUS` and `NO_VERDICT` exist for the same reason** — *could not be
#: graded* is a different fact from *graded and refused*, and collapsing them is
#: what put three permanent non-defects on a fourteen-row worklist. Every future
#: run would have reported them again.
#:
#: Reported, never silently skipped: the row still appears, saying why.
_GPU_HOURS: set[str] = set()


#: The launch-line vars a **mock** run passes, which the declared defaults do
#: not cover. Without these the probe grades a mocked artefact under a real
#: run's expectations and refuses correctly for a reason about the launch line
#: rather than about the artefact — the fourth defect in this file, and the same
#: sentence as the first three: *the validator the graph runs* is the one with
#: its launch line, not just its defaults.
#:
#: Measured, each one resolving a row that was otherwise a correct refusal of
#: the wrong question (m2, 2026-09-05):
#:
#:     expect_ranks=2   check_trace_coverage    "expected 8 rank(s), the manifest lists 2"
#:                      — the sealed torch_trace is a TP-2 capture (`m2_profiling.yaml:116`)
#:     adhoc_cases=0    check_acceptance x2     "adhoc.json is missing and 3 ad-hoc case(s)
#:                      are required (M5.4)" — no sealed handoff carries them (MOCK-MAP D")
#:
#: **`measure_gpu` is deliberately NOT here**, m3's warning: supplying it would
#: have the probe attempt a real measurement on a login node with no card, which
#: moves the refusal rather than removing it. A `cost: gpu_hours` validator is
#: not gradeable here at all — see `NOT_GRADEABLE`.
#:
#: Only vars whose effect has been measured belong in this dict. A var added
#: because it appears in a launch line, without a row to show for it, is a
#: threshold quietly relaxed.
MOCK_VARS: dict[str, str] = {
    "expect_ranks": "2",
    "adhoc_cases": "0",
}

_VAR = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*))?\}$")


def _substitute(node, overrides: dict[str, str] | None = None):
    """Resolve `${name:-default}` to its default, recursively.

    **The third defect in this file, and the same sentence catches all three.**
    v1 passed `{}`; v2 passed the raw template text; neither is *"the validator
    the graph runs"*. A run substitutes these at load time (`spec_loader`); the
    probe read the **definition** rather than the loaded spec — the same
    distinction as reading the tree rather than the store.

    The symptom was a crash, not a refusal:

        check_bench_report   ValueError: invalid literal for int(): '${bench_rounds:-1}'
        check_identity_resolved  could not convert string to float: '${min_resolve_ratio:-0.0}'

    **Ten of the fourteen refusals crashed on it; six of those resolve once
    substituted.** The other four crashed *and* refuse afterwards — the crash
    was HIDING a real refusal, not manufacturing a false one, which is the
    opposite of what a bare "12 of 14 were this" implies. Three counts, three
    different questions: 12 rows contain a template, 10 crashed on one, 6 are
    resolved by substituting. Across four owners, and every owner
    correctly refused to convert an artefact to satisfy it. Found by m5 and m3
    independently within a minute of each other; m2 classified the whole table.

    A bare `${name}` with no default resolves to `""`, which is what an
    unset var gives a run. CONTRACT §4.2: these arrive as strings, which is why
    a numeric coercion is where it surfaces.
    """
    if isinstance(node, dict):
        return {k: _substitute(v, overrides) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, overrides) for v in node]
    if isinstance(node, str):
        m = _VAR.match(node.strip())
        if m:
            name, default = m.group(1), m.group(2)
            if overrides and name in overrides:
                return overrides[name]
            return default if default is not None else ""
    return node


def _walk_validators(node):
    """Every `module: validator` block, at any depth."""
    if isinstance(node, dict):
        if node.get("module") == "validator":
            yield node
        for v in node.values():
            yield from _walk_validators(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_validators(v)


def _walk_handoffs(node):
    """Every mapping that looks like a handoff spec, at any depth."""
    if isinstance(node, dict):
        if node.get("module") == "handoff" or ("validators" in node and "name" in node):
            yield node
        for v in node.values():
            yield from _walk_handoffs(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_handoffs(v)


#: kind -> the mocked content dir, built once from a real run's handoffs.
#:
#: **Not the corpus.** Only three kinds are plain renames
#: (`profiling_mode_off.bench_result:aiperf_baseline`,
#: `profiling_mode_on.bench_result:aiperf_profiled`,
#: `profiling_mode_on.profile_result:torch_trace`); the rest are either
#: identity or, for m5, ASSEMBLED by `mock_m5.sh` from several corpus dirs
#: (`bench_stock` + `acceptance_stock` + `deployment_stock` -> one
#: `stock.measurement`). Probing the corpus by kind name therefore reports
#: "no corpus" for artefacts that the mock builds perfectly well — a
#: limitation of the probe, not a defect in the package.
#:
#: A run's sealed handoffs are the mock's own output, after every adaptation,
#: which is the thing a validator actually meets.
_INDEX: dict[str, pathlib.Path] = {}


def build_index(run: pathlib.Path) -> dict[str, pathlib.Path]:
    """kind -> newest content dir, keyed from **the store**, not from prose.

    **The first version parsed the kind out of the handoff `README.md`'s first
    line** and it silently mis-indexed anything whose title is prose:

        "# Kernel optimization — sglang sampler vocabulary softmax"
          -> head.split()[0] == "Kernel"        -> kernel_optimization never indexed

    A missing key is not a loud failure here — `corpus_content` falls back to
    `cheat_for_mock`, sealed **2026-09-02**, which predates `environment.yaml`
    and `results/kernel_optimization.json`. So three validators were shown a
    pre-new-format artefact and correctly reported the new files absent, and the
    table read that as three defects in m4's stage.

    **Had that table been worked, three validators would have been rewritten to
    accept a format their own producer stopped emitting** — deleting the checks
    that catch a producer regressing to the old shape. Found by m4, 2026-09-05,
    who ran their rows against the right handoff and got PASS with zero changes.

    `store/handoff/*.json` carries `type` exactly, which is why `kit_env.sh`
    reads it that way. Prose is not an identifier.
    """
    out: dict[str, pathlib.Path] = {}
    for rec in sorted((run / "store" / "handoff").glob("*.json")):
        try:
            d = json.loads(rec.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        kind, hid = d.get("type"), d.get("id")
        if not kind or not hid:
            continue
        # Newest version with content on disk wins; a version dir can exist and
        # hold nothing (92 of 283 measured), so presence of files is the test.
        best: pathlib.Path | None = None
        for c in sorted((run / "handoffs" / str(hid)).glob("v*/content")):
            if any(c.rglob("*")):
                best = c
        if best is not None and (
            kind not in out or best.stat().st_mtime > out[kind].stat().st_mtime
        ):
            out[str(kind)] = best
    return out


def corpus_content(kind: str) -> pathlib.Path | None:
    if kind in _INDEX:
        return _INDEX[kind]
    for stage in sorted(CORPUS.glob("*")):
        c = stage / kind / "content"
        if c.is_dir():
            return c
    hits = sorted(CORPUS.glob(f"*/{kind}/content"))
    return hits[0] if hits else None


def run_one(kind: str, validator: str, args: dict) -> dict:
    body = PKG / "assets" / f"{validator}.validator" / "check.py"
    if not body.is_file():
        return {"kind": kind, "validator": validator, "state": "NO_BODY"}
    if validator in _GPU_HOURS:
        # Declared `cost: gpu_hours` — it re-measures on a node. Running it here
        # produces a refusal about the login node, not about the artefact.
        return {"kind": kind, "validator": validator, "state": "NOT_GRADEABLE",
                "why": "declares cost: gpu_hours — re-measures on a node; "
                       "the probe has no card and no live allocation"}
    content = corpus_content(kind)
    if content is None:
        return {"kind": kind, "validator": validator, "state": "NO_CORPUS"}

    hid = f"probe-{kind}"
    with tempfile.TemporaryDirectory(prefix="vprobe-", dir="/tmp") as zone:
        z = pathlib.Path(zone)
        (z / "inputs.json").write_text(json.dumps([hid]), encoding="utf-8")
        (z / "args.json").write_text(json.dumps(args), encoding="utf-8")
        (z / "materials.json").write_text(json.dumps({hid: str(content)}), encoding="utf-8")

        env = dict(os.environ)
        env["AGENT_SYS_TASK_PACKAGE"] = str(PKG)
        env["AGENT_SYS_DEMO_PACKAGE"] = str(PKG)
        try:
            p = subprocess.run(
                [sys.executable, str(body)],
                cwd=z, env=env, capture_output=True, text=True, timeout=180,
            )
        except subprocess.TimeoutExpired:
            return {"kind": kind, "validator": validator, "state": "TIMEOUT"}

        vpath = z / "verdict.json"
        verdict = None
        if vpath.is_file():
            try:
                verdict = json.loads(vpath.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                verdict = "unparseable"
        report = ""
        rp = z / "validator_report.txt"
        if rp.is_file():
            report = rp.read_text(encoding="utf-8", errors="replace")

        if verdict is None:
            # **The state that matters most.** A validator that cannot start
            # looks, from the graph, like one that was never asked.
            tail = (p.stderr or p.stdout or "").strip().splitlines()
            return {"kind": kind, "validator": validator, "state": "NO_VERDICT",
                    "rc": p.returncode, "why": tail[-1][:200] if tail else ""}
        ok = all(bool(v) for v in verdict.values()) if isinstance(verdict, dict) else False
        why = ""
        for line in report.splitlines():
            if "PROBLEM" in line or "REFUSED" in line:
                why = line.strip()[:220]
                break
        return {"kind": kind, "validator": validator,
                "state": "PASS" if ok else "REFUSE", "rc": p.returncode, "why": why}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--run", action="append", default=[],
                    help="a run root whose sealed handoffs supply the mocked content; "
                         "repeatable, later ones win")
    a = ap.parse_args()

    for r in a.run:
        _INDEX.update(build_index(pathlib.Path(r)))
    if a.run:
        print(f"  indexed {len(_INDEX)} kind(s) from {len(a.run)} run(s): "
              f"{', '.join(sorted(_INDEX))}\n")

    pairs = declared()
    if not pairs:
        print("probe: no validators found in steps/*.yaml — the walker is wrong, "
              "not the package", file=sys.stderr)
        return 2
    print(f"  {len(pairs)} (kind, validator) pairs declared across steps/*.yaml\n")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for r in ex.map(lambda t: run_one(*t), pairs):
            rows.append(r)

    order = {"NO_VERDICT": 0, "TIMEOUT": 1, "REFUSE": 2, "NO_CORPUS": 3,
             "NOT_GRADEABLE": 4, "NO_BODY": 5, "PASS": 6}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r["kind"], r["validator"]))
    for r in rows:
        line = f"  {r['state']:<10} {r['kind']:<34} {r['validator']}"
        if r.get("why"):
            line += f"\n             {r['why']}"
        print(line)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    print("\n  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
