#!/usr/bin/env python3
"""Artefact-driven refusals for m5's validators — and the three probes that missed.

    ARMS=<dir with stock.measurement/ and e2e_packup/> python3 artefact_neg.py

**Why this exists.** m2's sweep found two validators with zero refusals across
269 recorded verdicts, and produced the refusal by breaking an ARTEFACT. Applying
that method to m5's seven split my own coverage in two:

    artefact-driven   check_measurement_order, check_no_regression,
                      check_overlay_applies, check_patch_live          (4)
    argument-driven   check_acceptance, check_bench_report,
                      check_packup_shape                               (3)

**Three of seven had only ever been shown to READ their bars** — `min_requests=500`,
`min_result_files=9999`, `needle_min_depths_retrieved=99`. That proves the
argument is consulted. It does not prove the check detects a broken artefact.
*A bar that is read* and *a check that detects* are different claims, and I had
been reporting the first as the second.

## WHAT SIX PASSING PROBES DO AND DO NOT LICENSE

They show these validators detect **the specific breakage injected here**. They
do **not** show they detect the breakage the world produces — which is m4's
standing caveat about their own re-measurement wearing different clothes. A
green run of this file is evidence about six cases, not about the class.

## THE THREE PROBES THAT WERE WRONG, KEPT BECAUSE THEY ARE THE INSTRUCTIVE PART

All three returned a confident PASS and none of them changed anything. Had I
stopped there I would have filed three defects against three working validators
— the mirror of m2's near-miss, reproduced three times in a row an hour after
they warned me.

    A1  the packup has TWO README.md, and `next(a.rglob("README.md"))` handed me
        the handoff's own rather than `items/codes/README.md`, which is the one
        check_packup_shape reads.
    A5  smoke.json's `checks` is a LIST of {"name": ...}, not a dict. The probe
        was guarded by `if isinstance(doc.get("checks"), dict)` — so it applied
        NO tamper at all and reported the validator as passing a removed check.
    A6  needle depths carry `ok`; the RUN carries `retrieved`/`of`. The probe set
        a `retrieved` field on each depth, which nothing reads.

**A PASS from a probe you have just written is not evidence.** Check that the
field you edited is the field the validator reads, then re-run.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from zone_harness import show  # noqa: E402

ARMS = pathlib.Path(os.environ.get("ARMS", "/home/yihou/m5ws/out"))
WORK = pathlib.Path(tempfile.mkdtemp(prefix="artefact_neg."))

AC = dict(require_frozen_checks=["arithmetic", "long_generation", "workers", "engine_log"],
          min_adhoc_cases=0, min_scored_per_eval=20,
          needle_min_token_ratio=0.95, needle_min_depths_retrieved=1)
BR = dict(schema="bench_result", min_requests="50", expect_rounds="1",
          require_metrics=["request_count", "output_token_throughput_tps", "ttft_ms",
                           "inter_token_latency_ms"], max_error_rate="0.05")
PS = dict(require_files=["README.md", "REPRODUCE.md", "environment.md", "notes.md"],
          min_content_lines={"README.md": 20, "REPRODUCE.md": 15,
                             "environment.md": 12, "notes.md": 8},
          min_command_lines=8, require_dirs=["results", "logs", "scripts"],
          min_result_files=4)


def clone(src: pathlib.Path, name: str) -> pathlib.Path:
    dst = WORK / name
    shutil.copytree(src, dst)
    return dst


def main() -> int:
    packup, arm = ARMS / "e2e_packup", ARMS / "stock.measurement"
    for p in (packup, arm):
        if not p.is_dir():
            print(f"missing {p} — set ARMS to a directory holding e2e_packup/ and "
                  "stock.measurement/ (a sealed arm and a packup, real ones)")
            return 2
    ok = []

    print("### A1  check_packup_shape: README.md present but EMPTY")
    print("       its own yaml calls this the failure worth naming — satisfies a file")
    print("       list and tells a reproducer nothing.")
    a = clone(packup, "empty_readme")
    (a / "items" / "codes" / "README.md").write_text("")   # the KIT readme, not the handoff's
    ok.append(show("present-but-empty README", "check_packup_shape", {"pk": a}, PS, expect=False))

    print("### A2  check_packup_shape: REPRODUCE.md deleted")
    b = clone(packup, "no_reproduce")
    for f in list(b.rglob("REPRODUCE.md")):
        f.unlink()
    ok.append(show("the one file a reproducer executes is gone", "check_packup_shape",
                   {"pk": b}, PS, expect=False))

    print("### A3  check_bench_report: every request errored (m2's shape)")
    c = clone(arm, "all_errored")
    jl = next(c.rglob("profile_export.jsonl"), None)
    if jl:
        rows = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        for r in rows:
            r.setdefault("metadata", {})["was_cancelled"] = True
            r["error"] = {"type": "ServerError", "code": 500}
        jl.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"       marked {len(rows)} request record(s) errored")
    ok.append(show("100% of requests errored", "check_bench_report", {"sm": c}, BR, expect=False))

    print("### A4  check_bench_report: the AIPerf export corrupted")
    d = clone(arm, "bad_export")
    ex = next(d.rglob("profile_export_aiperf.json"), None)
    if ex:
        ex.write_text('{"not": "an aiperf export"}')
    ok.append(show("summary export fails its schema", "check_bench_report",
                   {"sm": d}, BR, expect=False))

    print("### A5  check_acceptance: a frozen smoke check removed")
    e = clone(arm, "no_frozen")
    sm = next(e.rglob("smoke.json"), None)
    if sm:
        doc = json.loads(sm.read_text())
        before = len(doc.get("checks") or [])
        doc["checks"] = [x for x in doc.get("checks", []) if x.get("name") != "workers"]
        print(f"       removed the 'workers' check ({before} -> {len(doc['checks'])})")
        sm.write_text(json.dumps(doc, indent=2))
    ok.append(show("a required frozen check is absent", "check_acceptance",
                   {"sm": e}, AC, expect=False))

    print("### A6  check_acceptance: needle retrieved NOTHING")
    f = clone(arm, "needle_zero")
    nd = next(f.rglob("needle.json"), None)
    if nd:
        doc = json.loads(nd.read_text())
        for run_ in doc.get("runs", []):
            run_["retrieved"] = 0
            run_["passed"] = False
            for dep in run_.get("depths", []):
                dep["ok"] = False        # depths carry `ok`, not `retrieved`
        doc["ok"] = False
        nd.write_text(json.dumps(doc, indent=2))
        print("       set every depth ok=False and each run retrieved=0")
    ok.append(show("no depth retrieved", "check_acceptance", {"sm": f}, AC, expect=False))

    print(f"\n{sum(ok)}/{len(ok)} refused as expected   (scratch: {WORK})")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
