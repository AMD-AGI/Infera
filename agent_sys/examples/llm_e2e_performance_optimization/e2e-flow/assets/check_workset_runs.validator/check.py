#!/usr/bin/env python3
"""`check_workset_runs` — trustworthiness, strong.

The workset's own one-click correctness and performance entrypoints run, on this
hardware, and agree with the numbers it prints.

**This validator is the whole trust chain of the package** (CONTRACT.md §4.0),
so it is worth saying exactly what it holds up. m4 is told to take its ground
truth *strictly* from the workset and to **abort** rather than re-measure when
the premise does not hold — M4.3.5, reversed from the previous round's "do not
trust the workset's printed number". That instruction is only safe because
something has already run the workset's own tests on this hardware.

`build_workset` builds **and** measures — there is no separate `verify_workset`
task, because splitting build from measure across two agents is what M2.5
forbids. So the evidence in the workset is the producer's own claim about
itself. Grading the shape of that claim would make the chain a claim about a
claim.

**Therefore this validator re-measures.** It runs at least `reverify_shapes`
shapes through the workset's own `--shape` selector and checks its own number
against the recorded one. `build_workset` asserts a baseline; this confirms the
assertion; m4 may then divide by it. The same move `check_no_regression` makes
one stage later by recomputing instead of reading a verdict field.

Cost is `gpu_hours` and honest: one shape, five groups. `--shape` exists on the
entrypoints precisely so this check does not double the workset's GPU bill at
every seal.

**Two failures that are not failures of the workset, and are still failures.**
A node too busy to give a stable measurement fails `max_rsd`, and a host on
which the entrypoints refuse to run fails outright. Neither is a defect in the
artefact and both are correct verdicts: the artefact's claim is *"these numbers
hold on this hardware"*, and neither case establishes it. `todo.md` T10 records
that this and `min_pass_ratio` express opposite philosophies about forgiveness.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io as W  # noqa: E402
import zone  # noqa: E402

#: How far this validator's own re-measurement may sit from the recorded one
#: before the record is called wrong rather than the node called busy. Wider
#: than `max_rsd` on purpose: the two runs are minutes and one co-tenant apart,
#: which is the very effect `todo.md` T7 measured at 12% between two arms that
#: were 1.1% apart under matched load.
_REVERIFY_TOLERANCE = 0.25


def _fail(problems: list[str], message: str) -> bool:
    problems.append(message)
    return False


def _check_reports(content: Path, document: dict, args: dict, problems: list[str], notes: list[str]) -> list[dict]:
    """Grade the recorded evidence and return the per-operator performance rows.

    Returns `[]` when the evidence cannot be read at all, which is a failure the
    caller has already recorded.
    """
    evidence = document.get("evidence")
    if not evidence:
        _fail(problems, "the workset carries no evidence block; there is nothing to confirm")
        return []
    try:
        perf = W.load_report(content, evidence["performance_report"])
        correct = W.load_report(content, evidence["correctness_report"])
    except (json.JSONDecodeError, OSError) as error:
        _fail(problems, f"the evidence does not load: {error}")
        return []

    if perf.get("impl") != "baseline":
        _fail(problems, f"the recorded performance report has impl {perf.get('impl')!r}, expected 'baseline'")
    if not correct.get("passed"):
        _fail(problems, "the recorded correctness report did not pass; a timing of a wrong kernel is not evidence")

    # **The report was produced under the protocol the manifest declares.**
    #
    # `protocol` lives in `workset.yaml` and is echoed into the performance
    # report by the harness that used it — one fact written twice, and nothing
    # compared them. m4 copies the *manifest's* protocol and re-measures its
    # candidate under it, then divides by a baseline the *report* recorded. If
    # the two ever differed, that ratio would be across two protocols and would
    # look entirely normal.
    #
    # Same shape as the withheld shape and the interpreter: one authority, two
    # readers. Found by auditing for the shape rather than by it failing.
    declared, used = document.get("protocol") or {}, perf.get("protocol") or {}
    for key in ("groups", "iters_per_group", "warmup", "timing"):
        if key in declared and key in used and declared[key] != used[key]:
            _fail(problems, f"protocol.{key} is {declared[key]!r} in workset.yaml and {used[key]!r} in the "
                            f"performance report. A consumer re-measures under the manifest's protocol and "
                            f"divides by this report's number; across two protocols that ratio means nothing")

    correct_by_id = {o["operator_id"]: o for o in correct.get("operators") or []}

    # **Every declared shape must appear in the report, and this is checked
    # before anything is graded.**
    #
    # Without it the body grades whatever came back: a harness that silently
    # reported nothing for one shape leaves that shape out of
    # `operators[].shapes`, every remaining row passes, and the verdict is a
    # clean PASS over a partial measurement. m4 found exactly this in their own
    # validator and warned me — and they found it with a stub that could
    # *withhold* a shape on demand, after eight readings of the code had not.
    # I had the same hole.
    #
    # The workset is the authority for what should have been measured, not the
    # report. Anything else lets the thing being audited decide the scope of
    # its own audit.
    perf_by_id = {o["operator_id"]: o for o in perf.get("operators") or []}
    for declared in document["operators"]:
        label = declared["operator_id"]
        # **Every declared shape, not only the ones whose `role` names this
        # report.** The first version filtered by role and let a withheld shape
        # through: `evidence/` is produced by a *full* run — no `--shape` — and
        # the harness iterates every shape regardless of role, so a full report
        # that is missing one is missing it for a reason nobody recorded. The
        # role governs what a shape is *for*, not whether the harness touched
        # it, and reading it as the latter is what reopened the hole one commit
        # after closing it.
        for report, by_id, role, what in ((perf, perf_by_id, "performance", "timed"),
                                          (correct, correct_by_id, "correctness", "checked")):
            wanted = {s["case_id"] for s in declared["shapes"]}
            row = by_id.get(label)
            if row is None:
                _fail(problems, f"{label}: declared in workset.yaml and absent from the {role} report; "
                                f"a missing operator is not a passing one")
                continue
            if not row.get("ran"):
                continue  # an honest non-run is reported below, not here
            got = {s["case_id"] for s in row.get("shapes") or []}
            missing = sorted(wanted - got)
            if missing and row.get("failure"):
                # A run that broke off partway says so, and the shortfall is
                # then explained rather than silent. It is still not evidence —
                # `min_pass_ratio` decides whether the workset survives it — but
                # it is not the failure this rule is about.
                notes.append(f"{label}: {len(missing)} shape(s) not {what} after a recorded failure "
                             f"({row['failure'][:80]}) — {missing}")
            elif missing:
                _fail(problems, f"{label}: {len(missing)} declared shape(s) never {what} and no failure "
                                f"recorded — {missing}. The report covers {sorted(got)}; grading only "
                                f"what came back would pass a partial measurement as a clean one")
            extra = sorted(got - {s["case_id"] for s in declared["shapes"]})
            if extra:
                _fail(problems, f"{label}: the {role} report carries shape(s) {extra} the workset does "
                                f"not declare; the two documents describe different work")
    min_groups = W.arg_num(args, "min_groups", 5, int)
    min_iters = W.arg_num(args, "min_iters_per_group", 10, int)
    max_rsd = W.arg_num(args, "max_rsd", 0.10)

    for operator in perf.get("operators") or []:
        label = operator["operator_id"]
        if not operator.get("ran"):
            notes.append(f"{label}: did not run ({operator.get('failure') or 'no reason recorded'})")
            continue

        companion = correct_by_id.get(label)
        if companion is None:
            _fail(problems, f"{label}: timed, but the correctness report does not mention it")
        elif not companion.get("passed"):
            _fail(problems, f"{label}: timed, but its correctness failed ({companion.get('failure')})")

        for shape in operator["shapes"]:
            where = f"{label}/{shape['case_id']}"
            groups = shape["groups"]
            per_group = shape["per_group_ms"]
            if groups < min_groups:
                _fail(problems, f"{where}: {groups} measurement group(s), the protocol floor is {min_groups}")
            if len(per_group) != groups:
                _fail(problems, f"{where}: per_group_ms has {len(per_group)} entries for {groups} groups")
            if shape["iters_total"] < min_groups * min_iters:
                _fail(problems, f"{where}: {shape['iters_total']} iterations, below {min_groups}x{min_iters}")

            # Recompute rather than trust. The producer and this import the same
            # `weighted_mean`, so a stored figure that disagrees with the raw
            # numbers cannot come from a different formula — only from the record
            # having been edited after it was measured.
            recomputed = W.weighted_mean(per_group, shape["iters_total"])
            stored = shape["weighted_mean_ms"]
            if stored > 0 and abs(recomputed - stored) / stored > 0.01:
                _fail(problems, f"{where}: weighted_mean_ms is {stored:.6f}, the per-group figures give "
                                f"{recomputed:.6f}; the record disagrees with itself")
            recomputed_rsd = W.rsd(per_group)
            if abs(recomputed_rsd - shape["rsd"]) > 0.01:
                _fail(problems, f"{where}: rsd is {shape['rsd']:.4f}, the per-group figures give {recomputed_rsd:.4f}")
            if recomputed_rsd > max_rsd:
                _fail(problems, f"{where}: run-to-run spread {recomputed_rsd:.4f} exceeds {max_rsd}. The machine was "
                                f"not quiet, and an optimiser working against this baseline would chase noise")

    return list(perf.get("operators") or [])


def _transport_env(args: dict) -> dict[str, str]:
    """`os.environ` plus what `spur` needs and a validation zone strips.

    **The bug that cost four rung-0 runs and three of my own non-reproductions.**
    A validator declares no agent, so it runs in a closed environment: no
    `SPUR_CONTROLLER_ADDR`, and a `PATH` that `sh` fills in as `/usr/bin:/bin`
    while `spur` lives in `/usr/local/bin`. The re-measurement then dies with

        1: tcp connect error | 2: tcp connect error | 3: Connection refused

    which `require_visible_on_node` reported as *"the workset is not visible on
    the node"* — a filesystem claim for a missing environment variable.

    **Why I could not reproduce it three times:** my shell has
    `SPUR_CONTROLLER_ADDR`. Every hand-invocation inherited it. I even stripped
    `E2E_JOBID`, `E2E_NODE`, `E2E_TRANSPORT` and `E2E_MEASURE_GPU` to imitate
    the zone and **kept the one that mattered**, because it is not an `E2E_*`
    name and nothing pointed at it. A fixture more convenient than production
    (§4.4), where the convenience was my own login shell.

    m1 solved this for `check_deploy_serves` (`check.py:95-111`) and
    `RUN-PLAN.md`'s var table already recorded it as costing three rung-0 runs
    and two wrong attributions — **in m1's stage**. This is the same hole in
    mine, and the parameters are theirs by name so one `--var` drives both.
    """
    env = dict(os.environ)
    extra = str(args.get("transport_path") or "")
    if extra:
        parts = [p for p in env.get("PATH", "").split(":") if p]
        env["PATH"] = ":".join(parts + [p for p in extra.split(":") if p and p not in parts])
    for pair in str(args.get("transport_env") or "").split():
        name, _, value = pair.partition("=")
        if name and value:
            env[name] = value
    # **The measurement card, which the producer now refuses to default.**
    # `measure_in_container.sh` used to fall back to card 4; it does not, because
    # a shared card returns *slower numbers rather than an error* and this
    # validator would re-measure on the same contaminated card and agree. So the
    # card must be chosen, and a validator gets no `E2E_*` block — same hole as
    # `SPUR_CONTROLLER_ADDR`, same remedy.
    #
    # Re-measuring on a different card than the producer used is fine and is why
    # this is its own arg rather than read from the artefact: the comparison is
    # between two runs on one architecture, and `abort_on_mismatch` guards the
    # architecture. **The workset does not record which card produced it** —
    # `evidence.measured_on` carries node, arch and container but not the index —
    # which is T19's unrecorded half and is noted in `todo.md`.
    card = str(args.get("measure_gpu") or "")
    if card:
        env["E2E_MEASURE_GPU"] = card
    return env


def _reverify(content: Path, document: dict, recorded: list[dict], args: dict,  # noqa: PLR0913
              problems: list[str], notes: list[str]) -> None:
    """Re-measure `reverify_shapes` shapes here, and compare.

    This is the step that makes the chain evidence rather than a claim about a
    claim. Picking the **primary** shape of each operator in turn: it is the one
    a headline number refers to, and the one m4 will divide by first.
    """
    wanted = W.arg_num(args, "reverify_shapes", 1, int)
    if wanted <= 0:
        _fail(problems, "reverify_shapes is 0; that turns this validator into a reader of the producer's own claim")
        return

    root = W.workset_root(content)
    entry = document["entrypoints"]["performance"]
    by_id = {o["operator_id"]: o for o in recorded}

    picked: list[tuple[str, str]] = []
    for operator in document["operators"]:
        primary = next((s for s in operator["shapes"] if s.get("is_primary")), None)
        if primary is not None:
            picked.append((operator["operator_id"], primary["case_id"]))
    if not picked:
        _fail(problems, "no operator declares a primary shape; there is nothing to re-measure")
        return

    # **Which operators were re-measured, and which were taken on trust.**
    #
    # `picked[:wanted]` samples, and with `reverify_shapes: 1` and one operator
    # that is the whole workset — which is why it has been free so far. With
    # *two* operators it silently becomes: operator 1 is verified on every run
    # and operator 2 is verified on none, because the order is stable. **A
    # sample that never moves is not a sample of the population; it is a
    # census of one member.**
    #
    # Not fixed by raising the default, and deliberately so. Each re-verify is
    # a container start and a torch import — measured today at roughly 90 s
    # against **3 s** of actual timing, so the cost is ~30x the measurement and
    # scales with operator count, not with shapes. Choosing that spend is the
    # operator's call and depends on how much of a rung they are willing to
    # give this gate. What is *not* the operator's call is being told: the
    # unverified set is now named in the report, per operator, so a reader sees
    # `recorded, NOT re-measured` beside every number this validator did not
    # actually check.
    #
    # The first `wanted` are the highest-ranked, because `operators` is in the
    # ranker's order — stated because it was previously true by accident.
    verified, unverified = picked[:wanted], picked[wanted:]
    picked = verified
    for operator_id, case_id in unverified:
        notes.append(
            f"{operator_id}/{case_id}: recorded, NOT re-measured — reverify_shapes={wanted} of "
            f"{wanted + len(unverified)} operator(s) with a primary shape. This number is the "
            f"producer's claim and this run did not check it"
        )
    if unverified:
        notes.append(
            f"re-measuring all {wanted + len(unverified)} would cost about "
            f"{90 * len(unverified)}s more (a container start and a torch import each, against "
            f"~3s of timing). Raise `reverify_shapes` to trade rung time for coverage; the "
            f"default samples the top-ranked operator only"
        )

    for operator_id, case_id in picked:
        # **The report lands beside the staged content, not in a host tempdir.**
        # When the re-run goes through a container, only `/shared_nfs` is
        # mounted — a `/tmp` path on the host is invisible inside it, so the
        # entrypoint exited 0 having written a report nothing could read, and
        # the body reported "wrote no --json report" for a run that had in fact
        # succeeded. The zone is on the shared filesystem and is disposable by
        # construction, so a sibling of the staged content is visible from both
        # sides and cleaned up with the zone.
        with tempfile.TemporaryDirectory(dir=str(content.parent)) as tmp:
            out = Path(tmp) / "reverify.json"
            flags = {"operator": "--operator", "shape": "--shape", "impl": "--impl", "report": "--json"}
            flags.update(entry.get("flags") or {})
            inner = [*entry["cmd"].split(), flags["operator"], operator_id,
                     flags["shape"], case_id, flags["report"], str(out)]

            # **Re-measure where the producer measured.** Measured by the leader:
            # `spur exec <job> python3 -c "import torch"` fails — the node's
            # *host* has no torch, only the containers do. So a validator that
            # ran the entrypoint directly would fail on every host in this
            # cluster and read as a broken workset.
            #
            # When torch is importable here, this body is already inside a
            # container and runs the entrypoint directly. When it is not, it
            # goes through the **same** `measure_in_container.sh` the producer
            # used — one instrument, two callers. A validator re-measuring
            # through a different arrangement than the producer used would not
            # be re-measuring the same thing.
            if importlib.util.find_spec("torch") is not None:
                command, where = inner, root
            else:
                package = os.environ.get("AGENT_SYS_TASK_PACKAGE") or os.environ.get("AGENT_SYS_DEMO_PACKAGE")
                if not package:
                    _fail(problems, f"{operator_id}/{case_id}: no torch here and no package path to reach "
                                    f"the container helper; cannot re-measure, and grading the record "
                                    f"without re-measuring is what this validator exists not to do")
                    continue
                command = ["bash", str(pathlib.Path(package) / "assets/build_workset.task/measure_in_container.sh"),
                           str(content), " ".join(inner)]
                where = None
            try:
                finished = subprocess.run(  # noqa: S603 — the command comes from the artefact under test
                    command, cwd=where, capture_output=True, text=True,
                    env=_transport_env(args),
                    timeout=int(entry.get("timeout_s") or 1800),
                )
            except subprocess.TimeoutExpired:
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint timed out")
                continue
            except OSError as error:
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint would not start: {error}")
                continue

            if finished.returncode != 0:
                # 12 and not 3: a body's refusal is written for a person and runs to
                # several lines. Three kept the last three, which for a
                # well-written message is the *end* of the explanation and not
                # the instruction. Measured against the no-card refusal below.
                tail = (finished.stderr or finished.stdout or "").strip().splitlines()[-12:]
                _fail(problems, f"{operator_id}/{case_id}: the performance entrypoint exited "
                                f"{finished.returncode}: {' | '.join(tail)}")
                continue
            if not out.is_file():
                _fail(problems, f"{operator_id}/{case_id}: the entrypoint exited 0 and wrote no --json report")
                continue

            try:
                fresh = json.loads(out.read_text(encoding="utf-8"))
            except json.JSONDecodeError as error:
                _fail(problems, f"{operator_id}/{case_id}: the re-run's report is not valid JSON: {error}")
                continue

            measured = _one_shape(fresh, operator_id, case_id)
            claimed = _one_shape({"operators": [by_id.get(operator_id) or {}]}, operator_id, case_id)
            if measured is None:
                _fail(problems, f"{operator_id}/{case_id}: the re-run's report does not contain this shape")
                continue
            if claimed is None:
                _fail(problems, f"{operator_id}/{case_id}: the recorded evidence does not contain this shape")
                continue

            drift = abs(measured - claimed) / claimed if claimed > 0 else 1.0
            line = (f"{operator_id}/{case_id}: recorded {claimed:.4f} ms, re-measured {measured:.4f} ms "
                    f"({drift * 100:.1f}% apart)")
            if drift > _REVERIFY_TOLERANCE:
                _fail(problems, line + f" — beyond {_REVERIFY_TOLERANCE * 100:.0f}%. The recorded baseline does not "
                                       f"hold on this hardware, and m4 is about to divide by it")
            else:
                notes.append(line)


def _one_shape(report: dict, operator_id: str, case_id: str) -> float | None:
    for operator in report.get("operators") or []:
        if not operator or operator.get("operator_id") != operator_id:
            continue
        for shape in operator.get("shapes") or []:
            if shape.get("case_id") == case_id:
                return float(shape.get("weighted_mean_ms") or 0.0)
    return None


def _check(content: Path, args: dict, problems: list[str], notes: list[str]) -> bool:
    try:
        document = W.load_workset(content)
    except Exception as error:  # noqa: BLE001
        return _fail(problems, f"items/codes/workset.yaml does not load: {error}")

    recorded = _check_reports(content, document, args, problems, notes)
    if not recorded:
        return False
    _reverify(content, document, recorded, args, problems, notes)

    # `min_pass_ratio` forgives an operator that legitimately cannot run yet —
    # `identify`'s `agent_recovered` case — without letting the whole step block
    # the operators that are ready.
    ran = [o for o in recorded if o.get("ran")]
    ratio = len(ran) / len(recorded)
    floor = W.arg_num(args, "min_pass_ratio", 0.5)
    if ratio < floor:
        _fail(problems, f"{len(ran)} of {len(recorded)} operator(s) measured (ratio {ratio:.2f}, floor {floor})")

    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    findings: dict[str, tuple[list[str], list[str]]] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        notes: list[str] = []
        content = zone.content_of(hid)
        if content is None:
            problems.append("the phase staged no content for this handoff")
            verdicts[hid] = False
        else:
            try:
                verdicts[hid] = _check(content, args, problems, notes)
            except Exception as error:  # noqa: BLE001
                # **A crash and a refusal are not the same event**, and this
                # validator is the one where confusing them costs most: it is
                # the gate that re-measures, so "it failed" reads as "the
                # numbers do not hold" rather than "the instrument broke".
                # Same treatment as `check_workset_shape`, and the same
                # limitation — `verdict.json` is `dict[str, bool]` and has no
                # third state, so False is written because a check that did not
                # execute has established nothing.
                problems.append(
                    f"THIS VALIDATOR DID NOT RUN — {type(error).__name__}: {error}. An instrument "
                    f"failure, not a finding: no re-measurement happened, so nothing about these "
                    f"numbers was established either way. Traceback below"
                )
                problems.append(traceback.format_exc())
                verdicts[hid] = False
        findings[hid] = (problems, notes)
        for note in notes:
            print(f"{hid}: {note}")
        for problem in problems:
            print(f"{hid}: {problem}")
    # **Before the verdict, so a crash in the writer cannot take the reasons
    # with it.** Ordering learnt from the reclaim: teardown that runs after the
    # thing it protects is teardown that does not run.
    W.write_report("check_workset_runs", findings, verdicts)
    zone.write_verdict(verdicts)
    print(f"check_workset_runs: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
