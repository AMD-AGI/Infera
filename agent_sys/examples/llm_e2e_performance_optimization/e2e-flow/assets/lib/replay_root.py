#!/usr/bin/env python3
"""Materialise a completed run's sealed handoffs into a `mock_root`.

    replay_root.py --out <dir> --run <run> [--run <run> ...] [--kind K ...]
    replay_root.py --out <dir> --run <run> --list          # decide nothing, print

Then point a debug run at it:

    --var mock_root=<dir> --var mock_stages=m1,m2 --var m1_agent=runner ...

**THIS IS A DEBUGGING ACCELERATOR AND NEVER AN ACCEPTANCE PATH.** The user was
explicit: *final acceptance still requires one full real e2e*. A green run that
skipped stages proves the stages it ran, and nothing about the ones it replayed
— those were proven on the day they were produced, by the run this tool names in
the promotion record. If you are reading this because you are tempted to accept
on a skip-ahead run: that is the thing this sentence is here to stop.

## Why this is small

**The mock source root is already a `--var`** (`shared.yaml`,
`E2E_MOCK_ROOT: '${mock_root:-…}'`), and a mock leaf already copies
`<stage>/<kind>/content/` into `$AGENT_SYS_OUTPUT_<KIND>`. That *is* handoff
injection; the whole package has been doing it all day from the sealed
2026-09-02 corpus. This tool only points the same machinery at **the last good
real run** instead, so there is no new injection mechanism to trust — only a new
source directory, in a layout `mock.sh` already reads.

## What it refuses to do

**Read the store, do not pattern-match paths, and refuse rather than pick.**
m2's `kit_env.sh` established the idiom this afternoon after the one-liner it
replaced turned out to be a coin flip between three handoffs; its header has the
measurements and they are not repeated here. Two consequences copied verbatim:

* the `<run>/handoffs/<id>/v*` glob is structurally the first filter — 14 of 17
  path matches on a full tree are staged copies under `zones/` and validation
  `materials/`, and 8 of those carry `deploy_kit`'s own id, so scoping by kind
  alone does not reach one path;
* **a version directory can exist and hold nothing.** 92 of 283 handoff
  directories keep their content somewhere other than `v0`. Skipping empty
  directories is what does the work; ordering is a tie-break that has never
  been needed.

## "Stable" is about verdicts, not exits

**A run that finished is not a run that passed**, and the two came apart on
2026-09-04: rung 1 *sealed* `deploy_kit` — README with all three headings, every
probe green, load clean — and a validator then refused it on one number. So
stability here is computed from `handoffs/<id>/v<N>/validation.yaml`, which
records, per handoff version, **which validator, what result, what strength and
when**. Nothing else in a run tree carries the validator's *name*: the zone holds
`args.json`/`inputs.json`/`materials.json`/`verdict.json` and the verdict is
keyed by handoff id, so two validators' verdicts are distinguishable only by
their args. `validation.yaml` is the only place the name survives.

A kind is **stable** at threshold N when N distinct runs each produced it with

* the **same set** of validators, and
* every one of them `result: true`.

A different validator set between runs is *not* stability at a lower count — it
means the package changed underneath, and the two artefacts were not graded by
the same thing. That is reported as `unstable: validator set changed`, with both
sets, rather than being averaged away.

## The safety net you get for free, which is m2's

`check_environment` carries `compare_fixed_across_inputs: [node, gpu_arch,
image_id, model_path]` and runs across **every** handoff staged in a phase. So an
injected handoff from a different node, a different image or a different model
is **already a refusal** — loudly, at the phase that stages it, not silently
three stages later. That property is what makes skip-ahead safe to use casually,
and it was designed for something else entirely.

**It does not cover a stale live resource.** A record can name the right node
and the right image and still name a container that no longer exists. Which
kinds are exposed to that is a per-seam question — *does stage N+1 consume stage
N's artefact, or stage N's running process?* — and it is recorded in
`SKIPPABLE` below rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
import datetime as dt

#: `mock.sh`'s stage directories, by the kind each stage produces (CONTRACT §1).
STAGE_OF = {
    "deploy_kit": "stage1-deploy",
    "profiling_mode_off.bench_result": "stage2-profiling",
    "profiling_mode_on.bench_result": "stage2-profiling",
    "profiling_mode_on.profile_result": "stage2-profiling",
    "profiling_mode_on.kernel_table": "stage2-profiling",
    "profiling_evidence": "stage2-profiling",
    "kernel_worklist": "stage3-analyze",
    "operator_identity": "stage3-analyze",
    "operator_workset": "stage3-analyze",
    "kernel_optimization": "stage4-kernel-opt",
    "patch_overlay": "stage5-integration",
    "stock.measurement": "stage5-integration",
    "patched.measurement": "stage5-integration",
    "integration_report": "stage5-integration",
    "e2e_packup": "stage5-integration",
}

#: **Whether a kind survives being replayed, per seam.** `None` means the
#: question has been asked and not yet answered — the tool will materialise it
#: and say so, rather than guess, because a wrong entry here costs a debugging
#: session that looks like a real defect in somebody else's stage.
#:
#: The question for each seam is *does the consumer read this artefact, or the
#: process it describes?* A `deploy_kit` from a run that ended three hours ago
#: names a torn-down container in `runtime.container`; if anything downstream
#: connects to `runtime.endpoint`, replaying it produces a confident wrong
#: failure inside the consumer.
SKIPPABLE: dict[str, bool | None] = {k: None for k in STAGE_OF}


def load_handoffs(run: pathlib.Path) -> list[dict]:
    """Every handoff record in a run's store, as `{id, type, versions}`."""
    store = run / "store" / "handoff"
    out = []
    if not store.is_dir():
        return out
    for f in sorted(store.glob("*.json")):
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 — a corrupt record is not this tool's business
            continue
    return out


def populated_version(run: pathlib.Path, hid: str) -> pathlib.Path | None:
    """The one version directory of `hid` that holds content, newest first."""
    base = run / "handoffs" / hid
    versions = sorted(
        (p for p in base.glob("v*") if p.name[1:].isdigit()),
        key=lambda p: int(p.name[1:]),
        reverse=True,
    )
    for v in versions:
        content = v / "content"
        if content.is_dir() and any(content.rglob("*")):
            return v
    return None


def verdicts_of(version_dir: pathlib.Path) -> list[dict]:
    """`validation.yaml`'s rows: validator, result, strength, when.

    Returns `[]` when the file is absent, which is **not** the same as "passed
    nothing" — an unvalidated handoff is reported as such rather than counted.
    """
    f = version_dir / "validation.yaml"
    if not f.is_file():
        return []
    try:
        import yaml

        doc = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001
        return [{"validator": f"<unreadable: {type(exc).__name__}>", "result": False}]
    return list(doc.get("verdicts") or [])


def survey(runs: list[pathlib.Path], kinds: list[str] | None) -> dict[str, list[dict]]:
    """For each kind, one row per run that produced it. Newest run last."""
    found: dict[str, list[dict]] = {}
    for run in runs:
        for rec in load_handoffs(run):
            kind = rec.get("type")
            if kind not in STAGE_OF or (kinds and kind not in kinds):
                continue
            version = populated_version(run, rec["id"])
            if version is None:
                continue
            rows = verdicts_of(version)
            found.setdefault(kind, []).append({
                "run": run.name,
                "run_path": str(run),
                "handoff_id": rec["id"],
                "version": version.name,
                "content": version / "content",
                "validators": sorted(r.get("validator", "?") for r in rows),
                "all_passed": bool(rows) and all(r.get("result") is True for r in rows),
                "verdicts": [
                    {"validator": r.get("validator"), "result": r.get("result"),
                     "strength": r.get("strength"), "at": r.get("at")}
                    for r in rows
                ],
            })
    return found


def stability(rows: list[dict], threshold: int) -> tuple[bool, str]:
    """Whether these rows clear the bar, and the sentence explaining it."""
    passing = [r for r in rows if r["all_passed"]]
    if len(passing) < threshold:
        return False, (f"{len(passing)} of {len(rows)} run(s) passed every validator; "
                       f"threshold is {threshold}")
    sets = {tuple(r["validators"]) for r in passing}
    if len(sets) > 1:
        listed = " | ".join(",".join(s) or "<none>" for s in sorted(sets))
        return False, ("the validator set changed between runs, so these artefacts were "
                       f"not graded by the same thing: {listed}")
    if not any(passing[0]["validators"]):
        return False, "no validator graded this kind in any run — a green with nothing behind it"
    return True, f"{len(passing)} run(s), each passing {', '.join(passing[0]['validators'])}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True,
                    help="a completed run directory; repeat, oldest first")
    ap.add_argument("--out", help="the mock_root to write (omit with --list)")
    ap.add_argument("--kind", action="append", help="restrict to these kinds")
    ap.add_argument("--threshold", type=int, default=3,
                    help="runs that must have passed every validator (default 3)")
    ap.add_argument("--list", action="store_true",
                    help="report and write nothing")
    ap.add_argument("--allow-unstable", action="store_true",
                    help="materialise kinds below the threshold, marked in the record")
    args = ap.parse_args()

    runs = [pathlib.Path(r).resolve() for r in args.run]
    for r in runs:
        if not (r / "store" / "handoff").is_dir():
            print(f"replay_root: {r} has no store/handoff — not a run directory", file=sys.stderr)
            return 2

    found = survey(runs, args.kind)
    if not found:
        print("replay_root: no handoff of any known kind in these runs", file=sys.stderr)
        return 1

    promotions, skipped = [], []
    for kind in sorted(found):
        rows = found[kind]
        ok, why = stability(rows, args.threshold)
        newest = [r for r in rows if r["all_passed"]][-1] if any(r["all_passed"] for r in rows) else None
        mark = "STABLE  " if ok else "unstable"
        print(f"{mark} {kind:34s} {why}")
        if newest is None:
            skipped.append({"kind": kind, "reason": why})
            continue
        if not ok and not args.allow_unstable:
            skipped.append({"kind": kind, "reason": why})
            continue
        promotions.append({"kind": kind, "row": newest, "stable": ok, "why": why})

    if args.list or not args.out:
        if not args.list:
            print("replay_root: --out is required unless --list", file=sys.stderr)
            return 2
        return 0

    out = pathlib.Path(args.out).resolve()
    # **Never delete a tree this tool did not write.** A `mock_root` may be the
    # sealed corpus, which is not ours; overwriting per kind is the widest thing
    # allowed here.
    if out.exists() and not (out / "PROMOTION.json").is_file() and any(out.iterdir()):
        print(f"replay_root: {out} is not empty and holds no PROMOTION.json — refusing to "
              "write into a directory this tool did not create. Pick a new --out.",
              file=sys.stderr)
        return 2

    record = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        # **The command, beside its own output** — CONTRACT §4.4, the eighth
        # face. A derived table that does not cite its derivation is a claim,
        # and the reader of a replayed handoff three weeks from now has this
        # file and nothing else.
        "command": " ".join(sys.argv),
        "threshold": args.threshold,
        "runs_surveyed": [str(r) for r in runs],
        "ACCEPTANCE": "This root is for debugging only. Final acceptance requires one "
                      "full real e2e with --var mock_stages=none.",
        "promoted": [],
        "not_promoted": skipped,
    }

    for p in promotions:
        row, kind = p["row"], p["kind"]
        dest = out / STAGE_OF[kind] / kind / "content"
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(row["content"], dest, symlinks=True)
        n = sum(1 for f in dest.rglob("*") if f.is_file())
        record["promoted"].append({
            "kind": kind,
            "stage_dir": STAGE_OF[kind],
            "stable": p["stable"],
            "why": p["why"],
            "from_run": row["run"],
            "from_run_path": row["run_path"],
            "handoff_id": row["handoff_id"],
            "version": row["version"],
            "files": n,
            "verdicts": row["verdicts"],
            "skippable_seam": SKIPPABLE.get(kind),
        })
        print(f"  -> {STAGE_OF[kind]}/{kind}/content  {n} file(s)  "
              f"from {row['run']} {row['version']}")

    (out / "PROMOTION.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    unknown = [p["kind"] for p in record["promoted"] if p["skippable_seam"] is None]
    if unknown:
        # stdout is block-buffered when piped and stderr is not, so without this
        # the note lands *above* the promotion lines it refers to. Same trap
        # `apply.py`'s NodeError diagnosis documents, and it reads as a bug in
        # the tool rather than in the terminal.
        sys.stdout.flush()
        print(f"\nreplay_root: NOTE {len(unknown)} promoted kind(s) have no recorded answer to "
              "'does the consumer read this artefact or the process it describes?': "
              + ", ".join(unknown)
              + "\n  A replayed record can name a container that no longer exists. Until the "
              "seam is answered, treat a failure in the consuming stage as possibly this "
              "and not that stage's defect.", file=sys.stderr)

    print(f"\nreplay_root: wrote {out}/PROMOTION.json — "
          f"{len(record['promoted'])} promoted, {len(skipped)} not")
    print("replay_root: debugging only. Final acceptance is one full real e2e.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
