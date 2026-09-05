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

    Read from the specs rather than from the directory listing, because the
    args are half the contract — a validator run without the parameters its
    step passes is not the validator the graph runs.
    """
    import yaml

    out: list[tuple[str, str, dict]] = []
    for step in sorted((PKG / "steps").glob("*.yaml")):
        doc = yaml.safe_load(step.read_text(encoding="utf-8")) or {}
        for mod in doc if isinstance(doc, list) else [doc]:
            for hs in _walk_handoffs(mod):
                kind = hs.get("name") or hs.get("kind")
                for v in hs.get("validators") or []:
                    if isinstance(v, str):
                        out.append((str(kind), v, {}))
                    elif isinstance(v, dict):
                        name = v.get("name") or v.get("validator")
                        if name:
                            out.append((str(kind), str(name), dict(v.get("parameters") or {})))
    return out


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
    out: dict[str, pathlib.Path] = {}
    for h in sorted((run / "handoffs").glob("*/v*/content")):
        readme = h / "README.md"
        if not readme.is_file():
            continue
        first = readme.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
        if not first:
            continue
        # "# deploy_kit — Qwen/... " -> deploy_kit
        head = first[0].lstrip("# ").strip()
        kind = head.split()[0].split("—")[0].strip() if head else ""
        if kind and (kind not in out or h.stat().st_mtime > out[kind].stat().st_mtime):
            out[kind] = h
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

    order = {"NO_VERDICT": 0, "TIMEOUT": 1, "REFUSE": 2, "NO_CORPUS": 3, "NO_BODY": 4, "PASS": 5}
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
