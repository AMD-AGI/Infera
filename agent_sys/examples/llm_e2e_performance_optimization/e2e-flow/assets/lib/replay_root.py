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
three stages later. It was designed for something else entirely.

### Two things that guard does NOT cover, and the second is not obvious

**A stale live resource.** A record can name the right node and the right image
and still name a container that no longer exists. Per-seam question — *does
stage N+1 consume stage N's artefact, or stage N's running process?* — recorded
in `SKIPPABLE` below rather than assumed.

**An engine configuration difference, which is invisible in every compared
field.** m2's measurement, 2026-09-04, one node, one image, one tp, same cards,
one flag:

    --cuda-graph-bs-decode max  8   ITL 42.15 ms    312 tok/s
    --cuda-graph-bs-decode max 16   ITL  9.31 ms   1649 tok/s

**4.5x on decode**, and `node`, `gpu_arch`, `image_id` and `model_path` are all
identical across those two runs. m1 swept every kit: there is **no default** —
the producing agent writes the ceiling fresh at each bring-up, and four real runs
chose 16, 16, 8, 32.

So **an injected kit can pass `compare_fixed_across_inputs` and still deploy an
engine 4.5x slower than the one the numbers beside it came from**, which
surfaces downstream as a regression that is really a configuration difference.
The guard is real and it is not sufficient; `assets/lib/graph_ceiling.py` is the
bar that covers this gap, and it is an **absolute** bar for the reason in
CONTRACT §4.6 — both sides can share the fault, so no comparison can find it.

Named here at m2's request rather than left to be discovered during a debugging
session.
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

#: **`deploy_kit` is a recipe, and by construction rather than by intent.**
#: m1's answer, 2026-09-04, and it is stronger than the question deserved:
#: `deploy_and_prove` writes the kit at STEP 5 and tears the deployment down at
#: STEP 7, both inside m1's own task — so `runtime.container` names a dead
#: container **immediately**, on every run, not only after a replay. The kit
#: says so itself in `runtime.notes`. Replay therefore introduces no staleness
#: the graph does not already exercise every time.
#:
#: `runtime.endpoint` has **no reader anywhere** — every occurrence outside m1
#: is a write.
#:
#: **Confirmed by the consumer, not only inferred from the producer.** m2,
#: 2026-09-04: `load/line.sh:131-149` reads `fixed.{node,image,image_id,
#: model_name,served_model_name,tp_size,gpu_devices}` and `runtime.replayed_from`
#: — a static provenance string that cannot go stale — **and nothing else**.
#:
#: The precision that makes it checkable rather than asserted: `runtime.endpoint`,
#: `container` and `ports` *do* appear in `line.sh`, and they come from
#: `deployment.json` — **the handshake m2's own `deploy.sh` just wrote**
#: (`:288`, `:300`, `:306`) — not from the injected kit. `:263` runs
#: `deploy.sh` out of the kit and brings up m2's own engine; `:288` refuses if
#: it wrote no handshake. **The kit supplies *how to deploy*; the deployment
#: supplies *where to send traffic*.**
#:
#: And it is the design rather than a shortcut that happens to work while m1
#: runs first: M2.5, quoted at `line.sh:19` — *"m1 的 output 已经包含了如何部署
#: 的全量信息"* — the same rule that removed `serve_*` and `check_service_live`
#: from stage 2.
SKIPPABLE["deploy_kit"] = True


def rewrite_environment(record: dict, source_run: str) -> list[str]:
    """Make a replayed environment record honest about being replayed.

    Returns the list of changes, for the promotion record. **Mutates the copy
    under `--out`, never the source run.**

    ## The blocker this exists for

    `_agree_or_die` (`run_in_container.sh:96-104`, and the same three lines in
    m3's `measure_in_container.sh:127-129`) **exits 1** when an ambient value
    and the record's value are both non-empty and differ. It guards
    `fixed.node`, `runtime.slurm_jobid` and `runtime.transport`.

    A replayed kit carries the *old* job id; a debug run has a new one; both
    non-empty, both different — so m3 and m4 refuse with
    *"slurm_jobid is 'X' in the environment and 'Y' in the record"*, which reads
    as a misconfigured launch and not as an artefact of injection. m1 found this
    and it is the one failure that would have looked like somebody else's
    defect.

    **Blanking is the fix, not rewriting.** `_agree_or_die` returns the ambient
    value when the record's is empty, so an absent `slurm_jobid` lets the debug
    run's own job id win — which is the true one. Verified against the schema:
    `runtime.required` is `[container, endpoint, started_at]`, so neither
    `slurm_jobid` nor `transport` is required and removing them still validates.

    ## `fixed.node` is deliberately NOT rewritten

    m1 suggested rewriting it when the node moves. **Declining, and saying so
    rather than doing it silently:** the replayed artefact really was produced
    on the old node, and rewriting the field would make the record claim a
    measurement happened somewhere it did not. Letting `_agree_or_die` and
    `check_environment`'s `compare_fixed_across_inputs` refuse is the correct
    outcome — it means *skip-ahead requires the same node*, which is a real
    constraint of the mechanism and better documented than papered over.

    ## `runtime.container` is made unresolvable on purpose

    m1's sharpest point, and the only silent failure in the set. Container names
    carry a run tag, but not uniformly — one 2026-09-04 kit used a date-only tag
    — so a replayed name **can resolve to a live container that is a different
    process**. Then `docker inspect` succeeds, node/jobid/transport all agree,
    and m4 execs into the wrong container with every field validating. m1 hit
    exactly that on 2026-09-04 for an unrelated reason: the record said
    `started_at: 09:03:51Z` while docker reported `StartedAt: 09:37:18Z` with
    `RestartCount: 0`.

    A name that **cannot** resolve takes m4's ephemeral path
    (`run_in_container.sh:210-232`), which builds a fresh container from
    `fixed.image`, records `mode=ephemeral`, and states that a speedup measured
    there is a different claim from one measured in the live deployment. Wrong
    loudly beats wrong silently, and here the loud path is also correct.
    """
    changes: list[str] = []
    runtime = record.setdefault("runtime", {})

    for field in ("slurm_jobid", "transport"):
        if runtime.get(field) not in (None, ""):
            changes.append(f"runtime.{field}={runtime[field]!r} removed "
                           f"(_agree_or_die takes the ambient value when the record is empty)")
            runtime.pop(field, None)

    dead = f"replayed-from-{source_run}-NOT-RUNNING"
    if runtime.get("container") != dead:
        changes.append(f"runtime.container={runtime.get('container')!r} -> {dead!r} "
                       f"(unresolvable on purpose: a name that resolves to a different "
                       f"process validates every field and is wrong silently)")
        runtime["container"] = dead
    if runtime.get("endpoint"):
        changes.append(f"runtime.endpoint={runtime['endpoint']!r} -> a dead marker "
                       f"(required by the schema, read by nothing)")
        runtime["endpoint"] = f"http://replayed-from-{source_run}.invalid:0"

    # **The marker already exists and is already consumed** — five readers:
    # `check_deploy_kit/check.py:404-425`, `kit_status.py:66,158`,
    # `check_measurement_order/check.py:280,330`, and m2's `line.sh:149`, which
    # carries it into the numbers rather than into a log line because a reader
    # meets the number long after the message that qualified it. `mock_adapt.sh`
    # sets it for the mock path, so this is the second producer of an understood
    # field rather than a new one.
    #
    # `runtime.additionalProperties` is `true`, so setting it validates —
    # checked, because an undeclared field under a closed object is the trap m4
    # hit with `base_sha256_from` this afternoon.
    # **A replay of a replay must not erase the first one.** Found by running
    # m1's typo test against a real kit: rung 1's `deploy_kit` *already* carried
    # `replayed_from: /shared_nfs/…/cheat_for_mock/stage1-deploy/deploy_kit`,
    # because that run mocked stage 1 from the sealed corpus. Overwriting it
    # collapses `sealed corpus -> run X -> here` into `run X -> here`, which
    # tells a reader the numbers came from a real bring-up one hop back when
    # they never came from one at all.
    #
    # Chained into the one field rather than into a second one: consumers treat
    # this value as opaque and truthy — `required_unless` in m1's layout,
    # a message in `check_deploy_kit`, a column in `kit_status`, and m2's
    # `line.sh:149` carrying it into the numbers — so a longer string is safe,
    # where a new undeclared key would be the hazard m1 just measured.
    prior = runtime.get("replayed_from")
    value = source_run if not prior or prior == source_run else f"{source_run} <- {prior}"
    if prior != value:
        changes.append(f"runtime.replayed_from={prior!r} -> {value!r}"
                       + (" (chained: the source run was itself a replay)" if prior else ""))
        runtime["replayed_from"] = value
    return changes


def rewrite_in_tree(content: pathlib.Path, source_run: str) -> list[str]:
    """Apply `rewrite_environment` to every environment record under `content`.

    All fifteen kinds carry the same document (CONTRACT §2) at one of three
    paths, so this globs rather than being told which.
    """
    import yaml

    out: list[str] = []
    for rel in ("items/env/environment.yaml", "items/codes/environment.yaml",
                "items/codes/*/environment.yaml"):
        for path in sorted(content.glob(rel)):
            try:
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                out.append(f"{path.name}: UNREADABLE, left alone: {type(exc).__name__}")
                continue
            if not isinstance(doc, dict):
                continue
            changed = rewrite_environment(doc, source_run)
            if changed:
                path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
                out.extend(f"{path.relative_to(content)}: {c}" for c in changed)
                # **Read the marker back through the accessor a CONSUMER uses,
                # and refuse if it is not there.** m1's suggestion, and it
                # guards a failure this tool would otherwise cause itself.
                #
                # `environment.schema.json` leaves `runtime` open
                # (`additionalProperties: true`), so `replayed_from` is
                # undeclared and a **misspelling validates silently**. m1
                # measured it against the real rung-1 record with a stock
                # `Draft202012Validator`: `replayed_from`, `replayed_form` and
                # `zzz_not_a_field` all give **zero schema errors**, and the two
                # wrong ones read back as `None`.
                #
                # `None` means *"not a replay"* to all five consumers. So one
                # transposed character here turns a replayed kit into one the
                # whole flow treats as a real bring-up — the unsafe direction,
                # from a typo, in the field whose entire job is to say this
                # artefact is not what it looks like.
                #
                # Declaring the field would not fix that while the object stays
                # open, and closing it is a shared-schema decision with the
                # leader (three undeclared keys across 241 records, m1's sweep).
                # This costs nothing and closes it here, where the tool is the
                # producer.
                back = yaml.safe_load(path.read_text(encoding="utf-8"))
                got = ((back or {}).get("runtime") or {}).get("replayed_from")
                # `startswith`, because the value chains when the source run was
                # itself a replay. The gate is "does it name this hop", not
                # "is it exactly this hop".
                if not isinstance(got, str) or not got.startswith(source_run):
                    raise SystemExit(
                        f"replay_root: wrote {path} but reading `runtime.replayed_from` "
                        f"back gives {got!r}, not {source_run!r}.\n"
                        "  Every consumer reads it exactly that way and treats absent as "
                        "'this is a real bring-up', so shipping this root would hand the "
                        "flow a replayed kit wearing a real one's face. Refusing."
                    )
    return out


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
    # `__doc__` is None under `python3 -OO`, which strips docstrings — so
    # `__doc__.splitlines()[0]` is an AttributeError before argparse ever runs,
    # and `--help` dies with a traceback. Found by the leader; the package never
    # invokes with -OO, so this is a courtesy to whoever does.
    ap = argparse.ArgumentParser(
        description=(__doc__ or "Materialise a run's handoffs into a mock_root.").splitlines()[0])
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
    ap.add_argument("--no-rewrite", action="store_true",
                    help="copy the environment records verbatim. Leaves the stale "
                         "slurm_jobid that makes m3 and m4 refuse, and leaves a container "
                         "name that may resolve to a different process. For inspecting "
                         "what a run really produced, not for a debug run.")
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
        # On the copy, never the source run. A run tree is evidence.
        rewrites = [] if args.no_rewrite else rewrite_in_tree(dest, row["run"])
        n = sum(1 for f in dest.rglob("*") if f.is_file())
        record["promoted"].append({
            "environment_rewrites": rewrites,
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
