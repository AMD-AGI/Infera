#!/usr/bin/env python3
"""`check_speedup_substantiated` — was the claim taken from the workset, and did the premise hold?

**This body is a reversal of the one it replaces, and the reversal is the whole
of the difference.** `kernel-opt-demo`'s version re-measured the baseline here
and disbelieved the workset's; its loudest rule was *"最要命的那条：不要拿
workset 里印的数字当分母"*. Mission M4.3.5 overrules it:

    这一点不成立，优化任务的 ground truth 本身就应该严格的从 workset 中来，
    如果最基础的硬件、优化前提不一样，直接报错 abort。软件环境不太一样可以报
    warning。

So the denominator is the workset's own recorded baseline, always, and the
question this body asks changes from *"does your number reproduce against a
baseline I measured"* to two questions:

1. **Did you take the workset's baseline?** Exact comparison against the copy of
   the workset's own performance report that the handoff carries. Not "close
   to"; equal.
2. **Did the premise hold?** Every field the workset named in
   `ground_truth.abort_on_mismatch` must match between the workset's environment
   and this run's. A mismatch **aborts**. A field in `warn_on_mismatch` may
   differ, and then it must be *recorded* — differing is tolerated, hiding it is
   not.

**Why an abort rather than a smaller pass.** A speedup measured against a
different architecture is not a weaker result, it is the answer to a different
question, and the damage is that it looks entirely legitimate. Measured: the
2026-09-02 run timed `B8_V151936` at 50.18 µs on gfx950 against the workset's
55.40 µs on gfx942 — 9.6% of speedup available for free from a comparison
nobody downstream could detect. The old rule's response was to silently
re-baseline, which makes the report internally consistent and still answers the
wrong question.

**Why taking the workset's number on trust is safe here, and what would make it
unsafe.** `check_workset_runs` (m3, `cost: gpu_hours`) executes the workset's own
correctness and performance entrypoints **on this hardware** before m4 starts,
and it runs again in m4's own input phase. The workset's printed baseline is
therefore a number that has been reproduced here, not a number carried in from
another machine. **If that validator is ever weakened to a shape check, this
body has to go back to re-measuring the baseline itself.** Nothing in the code
below can detect that happening; it is a standing dependency between two
validators and it is written here because there is nowhere else to write it.

**What is still measured here, and why the cost tag stays `gpu_hours`.** The
*optimised* side. The reversal is about the denominator, not about believing the
numerator: a producer's own claim about the kernel it just wrote is exactly the
claim worth re-running. The seed is re-run too, against the workset's baseline —
not to replace it, but because a disagreement there is the premise failing
empirically, and that is a finding rather than a licence to substitute a number.

**Why `strong`.** It refuses on the two things that cannot be recovered
downstream: a claim whose denominator is not the ground truth, and a premise
that did not hold. Both make every number after them meaningless, and m5 has no
way to notice either.
"""

from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402 — the path insert above is what makes it importable

_DOC = "results/kernel_optimization.json"
_SNAPSHOT = "results/workset.snapshot.yaml"
_BASELINE_REPORT = "results/workset.baseline_report.json"
#: The runnable copy of the workset the handoff carries, so that this body can
#: re-run the workset's entrypoints without reaching for a handoff it was not
#: handed. See the module docstring of `check_optimization_shape` for why it
#: cannot be handed one.
_APPARATUS = "scripts/workset"


# --------------------------------------------------------------------------- #
# environment


def _interpreter(problems: list[str], notes: list[str]) -> str | None:
    """An interpreter that can actually `import torch`, or `None`.

    **This function exists because of a bug that made the previous validator
    fail every real run, and the failure looked like a measurement
    disagreement.** A validator body is started by `/bin/sh` with a *closed*
    environment (`validator/environment.py`): `os.environ` is not inherited and
    `PATH` is deliberately absent, so POSIX `sh` substitutes its built-in
    default. The template idiom `"${AGENT_SYS_DEMO_PYTHON:-python3}"` then
    resolves to `/usr/bin/python3` on the **output** phase, because the PRODUCER
    row shadows the GLOBAL row that carries `AGENT_SYS_DEMO_PYTHON`
    (`kernel-opt-demo/bugs/002-validator-env-row-shadows-demo-python.md`).

    `/usr/bin/python3` has no `torch`, so the measurement died on the import in
    about 0.1 s — faithfully reported as "measurement failed" and folded into a
    FAIL. Measured 2026-09-01 across three campaigns; the same handoff passed
    when re-run by hand with the venv interpreter.

    The bug record said the package was immune because its bodies import stdlib
    only. That was **wrong**: the body imports stdlib and then shells out to a
    script that needs the whole ML stack. Immunity to a missing import is not
    immunity to picking the wrong interpreter.
    """
    seen: list[str] = []
    for candidate in (
        os.environ.get("KFO_PYTHON"),
        os.environ.get("AGENT_SYS_DEMO_PYTHON"),
        sys.executable,
        "/opt/venv/bin/python3",
        "python3",
    ):
        if not candidate or candidate in seen:
            continue
        seen.append(candidate)
        try:
            probe = subprocess.run(
                [candidate, "-c", "import torch; print(torch.__version__)"],
                capture_output=True,
                text=True,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            notes.append(f"interpreter {candidate} (torch {probe.stdout.strip()})")
            return candidate
    problems.append(
        "no interpreter with torch found; tried "
        + ", ".join(seen)
        + ". Set KFO_PYTHON on the agent spec's env block to one that has it"
    )
    return None


#: A PATH a compiler can actually work in. **Not decoration.**
#:
#: An optimised kernel here is usually a Triton kernel, and the first thing
#: Triton does on this backend is compile `hip_utils.c` by shelling out to
#: `/bin/gcc` — which then needs `as`, `ld` and `collect2` off `PATH`. A
#: validator body's environment is closed and carries no `PATH`, so a subprocess
#: started from it inherits none and the compile dies. Measured 2026-09-01. The
#: baseline side survives it because plain `torch.softmax` compiles nothing, so
#: the symptom is *only the optimised side fails*, which reads exactly like a
#: broken optimised kernel. It is not.
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _measure_env(scratch: Path) -> dict[str, str]:
    """The environment the measurement subprocess needs, built rather than inherited.

    `TRITON_CACHE_DIR` because Triton otherwise writes to `$HOME/.triton`, and
    `$HOME` on this class of host is an NFS mount whose writes fail silently for
    a container user. `HOME` because several libraries probe it and an unset one
    is not the same as a writable one. `TMPDIR` because a `$TMPDIR` pointing at
    a directory that does not exist makes every HIP kernel launch segfault with
    no output while `torch.cuda.is_available()` still reports `True` — the trap
    that cost the 2026-09-02 run 25 minutes.

    `HIP_VISIBLE_DEVICES` is deliberately **not** defaulted. It arrives from the
    agent spec's `env:` block through the PRODUCER row, and inventing a default
    would silently move the measurement onto card 0 — which on a shared host is
    somebody else's.
    """
    env = dict(os.environ)
    env["PATH"] = env.get("PATH") or _PATH
    env.setdefault("TRITON_CACHE_DIR", str(scratch / "triton_cache"))
    env.setdefault("HOME", str(scratch))
    env["TMPDIR"] = str(scratch / "tmp")
    for key in ("TRITON_CACHE_DIR", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def _num(value: object, fallback: float) -> float:
    """Coerce an `args.json` value. Substitution yields **strings**, always."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------- #
# the premise


def _dig(doc: dict, dotted: str):
    """`fixed.gpu_arch` out of an environment record, or `KeyError`-free `None`."""
    node = doc
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _resolve(field: str, environment: dict) -> tuple[str, object]:
    """A premise field name, as a dotted path into an environment record.

    The workset writes dotted paths (`fixed.gpu_arch`); the step yaml's own
    `abort_on_premise_mismatch` writes bare leaves (`gpu_arch`), because that is
    how the mission names them. Both are accepted, and a bare name is looked up
    under `fixed` and then `runtime` — so the two spellings cannot silently
    check different fields.
    """
    if "." in field:
        return field, _dig(environment, field)
    for section in ("fixed", "runtime"):
        value = _dig(environment, f"{section}.{field}")
        if value is not None:
            return f"{section}.{field}", value
    return f"fixed.{field}", None


def _check_premise(doc: dict, args: dict, problems: list[str], notes: list[str]) -> bool:
    """The M4.3.5 gate. Returns False on an abort."""
    premise = doc.get("premise") or {}
    workset_env = premise.get("workset_environment") or {}
    run_env = premise.get("run_environment") or {}

    # The lists are the workset's. The step yaml's own lists are a floor: a
    # field the mission names must be checked even if a workset forgot it, and a
    # field the workset adds is checked because the workset knows what its
    # numbers depend on.
    abort_fields = list(dict.fromkeys(
        list(premise.get("abort_on_mismatch") or []) + list(args.get("abort_on_premise_mismatch") or [])
    ))
    warn_fields = list(dict.fromkeys(
        list(premise.get("warn_on_mismatch") or []) + list(args.get("warn_on_mismatch") or [])
    ))
    if not abort_fields:
        problems.append("the premise declares no abort_on_mismatch fields; there is nothing to hold it to")
        return False

    # **Deduplicated by *resolved path*, not by the name written down.** The
    # workset spells `fixed.gpu_arch` and the step yaml spells `gpu_arch`; both
    # resolve to the same field, and a set of the raw strings keeps both — which
    # reported one mismatch twice in the first smoke test. Same field, one line.
    aborted: list[str] = []
    seen: set[str] = set()
    for field in abort_fields:
        # `operator`, `shapes` and `dtype` are not environment fields; they are
        # checked against the workset in `_check_ground_truth` and skipped here.
        if field in ("operator", "shapes", "dtype"):
            continue
        path, expected = _resolve(field, workset_env)
        if path in seen:
            continue
        seen.add(path)
        _, actual = _resolve(field, run_env)
        if expected != actual:
            aborted.append(f"{path}: the workset was measured on {expected!r}, this run is on {actual!r}")

    if aborted:
        problems.append(
            "ABORT — the optimisation premise does not hold, so no ratio computed here answers the "
            "question the workset asked (M4.3.5):"
        )
        problems.extend(f"  {line}" for line in aborted)
        return False

    # A field that differed and was not recorded is the fault. Differing is not.
    recorded = {
        str(w.get("field"))
        for w in (run_env.get("warnings") or []) + (premise.get("verdict", {}).get("warnings") or [])
        if isinstance(w, dict)
    }
    seen = set()
    for field in warn_fields:
        path, expected = _resolve(field, workset_env)
        if path in seen:
            continue
        seen.add(path)
        _, actual = _resolve(field, run_env)
        if expected == actual:
            continue
        if path in recorded or field in recorded:
            notes.append(f"WARNING carried forward — {path}: workset {expected!r}, this run {actual!r}")
        else:
            problems.append(
                f"{path} differs from the workset ({expected!r} vs {actual!r}) and is recorded in "
                "neither run_environment.warnings[] nor premise.verdict.warnings[]. A tolerated "
                "difference that is not written down is indistinguishable from one nobody noticed"
            )

    # The producer's own verdict must agree with the one just computed. A
    # document that says the premise held while it did not is worse than one
    # that says nothing, because m5 reads the field and not this transcript.
    if (premise.get("verdict") or {}).get("held") is not True:
        problems.append("premise.verdict.held is not true, but no abort-level field differs — the record disagrees with itself")
    return not problems


def _check_ground_truth(doc: dict, snapshot: dict, args: dict, problems: list[str]) -> None:
    """`operator`, `shapes` and `dtype`: the abort-level fields that are not environment."""
    operator_id = doc.get("operator")
    operator = next(
        (o for o in snapshot.get("operators") or () if isinstance(o, dict) and o.get("operator_id") == operator_id),
        None,
    )
    if operator is None:
        problems.append(f"ABORT — operator {operator_id!r} is not one the workset defines")
        return

    declared = [
        s.get("case_id")
        for s in operator.get("shapes") or ()
        if isinstance(s, dict) and s.get("role") in ("performance", "correctness-and-performance")
    ]
    performance = (doc.get("evidence") or {}).get("performance") or {}
    measured = sorted((performance.get("measured") or {}).get("per_case_ms") or {})
    if sorted(c for c in declared if c) != measured:
        problems.append(
            f"ABORT — the shapes measured are not the shapes the workset declares "
            f"(workset {sorted(c for c in declared if c)}, measured {measured}). A speedup over a "
            "different shape set is a different question, not a partial answer"
        )
    floor = int(_num(args.get("min_shapes_measured"), 3))
    if len(measured) < floor:
        problems.append(f"{len(measured)} shapes measured, the workset contract requires >= {floor}")

    # The claim must carry the workset's own floor, not one of m4's choosing.
    claim = performance.get("claim") or {}
    if claim and claim.get("noise_floor") != operator.get("noise_floor"):
        problems.append(
            f"claim.noise_floor is {claim.get('noise_floor')!r}, the workset declares "
            f"{operator.get('noise_floor')!r}. The workset derives it from its own measured "
            "spread; a consumer that restates it differently has chosen when to call its own "
            "result significant"
        )

    # `dtype` is on the mission's abort list and is **not** an environment field.
    # It lives in the flashinfer-bench Definition (`inputs[].dtype`), which this
    # body cannot open — the snapshot is `workset.yaml`, not the whole workset
    # tree. Rather than skip it silently, which is the failure where two owners
    # each assume the other checks a thing, say what was not checked. The
    # protection that remains is real and is m3's: the entrypoints refuse to run
    # on a mismatched host at all, and `check_workset_runs` re-runs them here.
    if "dtype" in (args.get("abort_on_premise_mismatch") or []):
        declared_dtype = (doc.get("premise") or {}).get("dtype")
        workset_dtype = (snapshot.get("ground_truth") or {}).get("dtype")
        if workset_dtype is None:
            print(
                "note: dtype is on the abort list and the workset does not lift it into "
                "ground_truth, so it was NOT compared here. It is in the Definition's "
                "inputs[].dtype, which this validator cannot reach",
                flush=True,
            )
        elif declared_dtype != workset_dtype:
            problems.append(
                f"ABORT — dtype {declared_dtype!r} is not the workset's {workset_dtype!r}"
            )


def _check_denominator(doc: dict, baseline_report: dict, problems: list[str], notes: list[str]) -> None:
    """*Prove you took the workset's own baseline.* Exact, per case."""
    performance = (doc.get("evidence") or {}).get("performance") or {}
    claimed = (performance.get("baseline") or {}).get("per_case_ms") or {}

    # `evidence/performance.json` and the report a candidate run writes are the
    # **same document shape**, distinguished only by `impl`. That is m3's design
    # and it is what makes a speedup a ratio between two runs of one instrument
    # — and it also means a producer that carried the wrong one forward would
    # be dividing a candidate by a candidate. One field, checked once.
    impl = baseline_report.get("impl")
    if impl != "baseline":
        problems.append(
            f"{_BASELINE_REPORT} has impl={impl!r}, not 'baseline'. The denominator must be the "
            "workset's baseline run, and a candidate report has the same shape as one (M4.3.5)"
        )
        return

    truth: dict[str, float] = {}
    for entry in baseline_report.get("operators") or ():
        if not isinstance(entry, dict) or entry.get("operator_id") != doc.get("operator"):
            continue
        for shape in entry.get("shapes") or ():
            if isinstance(shape, dict) and shape.get("case_id"):
                value = shape.get("weighted_mean_ms", shape.get("median_ms"))
                if isinstance(value, (int, float)):
                    truth[str(shape["case_id"])] = float(value)
    if not truth:
        problems.append(
            f"{_BASELINE_REPORT} carries no figures for operator {doc.get('operator')!r}; "
            "there is nothing to prove the denominator against"
        )
        return

    for case, value in sorted(claimed.items()):
        if case not in truth:
            problems.append(f"baseline.per_case_ms names {case!r}, which the workset's own report does not")
        elif abs(float(value) - truth[case]) > 1e-9 * max(truth[case], 1.0):
            problems.append(
                f"baseline.per_case_ms[{case}] is {value}, the workset's own report says {truth[case]} — "
                "the denominator is not the workset's (M4.3.5)"
            )
    missing = sorted(c for c in truth if c not in claimed)
    if missing:
        problems.append(f"the workset baselines {missing}, which this handoff did not carry forward")
    if not problems:
        notes.append(f"denominator is the workset's own, {len(claimed)} case(s), exactly")


def _check_correctness(doc: dict, snapshot: dict, args: dict, problems: list[str]) -> None:
    """Boolean, and every declared case. Correctness is not a percentage."""
    if not args.get("require_correctness_pass", True):
        return
    correctness = (doc.get("evidence") or {}).get("correctness") or {}
    if correctness.get("passed") is not True:
        problems.append(
            "evidence.correctness.passed is not true. A kernel that is faster and wrong is not a "
            "partial success, and it must never have reached a timing loop"
        )
    failed = [
        s.get("case_id") for s in correctness.get("shapes") or ()
        if isinstance(s, dict) and s.get("passed") is not True
    ]
    if failed:
        problems.append(f"correctness failed on {failed}")

    operator = next(
        (o for o in snapshot.get("operators") or ()
         if isinstance(o, dict) and o.get("operator_id") == doc.get("operator")),
        None,
    ) or {}
    declared = {
        s.get("case_id") for s in operator.get("shapes") or ()
        if isinstance(s, dict) and s.get("role") in ("correctness", "correctness-and-performance")
    }
    covered = {s.get("case_id") for s in correctness.get("shapes") or () if isinstance(s, dict)}
    uncovered = sorted(c for c in declared if c and c not in covered)
    if uncovered:
        problems.append(f"the workset declares correctness cases that were never run: {uncovered}")


# --------------------------------------------------------------------------- #
# the measurement


def _run_entrypoint(
    root: Path, cmd: str, impl_path: Path | None, report: Path, args: dict, env: dict, timeout: float
) -> str | None:
    """The workset's own performance entrypoint. Returns an error string or `None`.

    The contract m3 shipped is::

        ./run_performance.sh [--operator ID] [--impl PATH] [--shape CASE_ID] [--json OUT]

    **No `--impl` means the workset's own baseline; `--impl PATH` means the
    candidate**, judged against the same reference, the same shapes and the same
    protocol. So a speedup is a ratio between two runs of one instrument, which
    is the property that makes any of this comparable.

    The two flag names stay `args` rather than literals. Not because the
    contract is unsettled — it is settled — but because this body and
    `optimize_kernel`'s `steps/run_entrypoint.py` must drive the workset
    *identically*, and a pair of literals in two files is a pair that can be
    edited apart. **The failure mode if they ever disagree is silent**: a wrong
    flag that makes the entrypoint run the baseline while this body records it
    as the candidate produces two measurements of the same code, a ratio of
    1.000, and no error anywhere.
    """
    argv = [
        *cmd.split(),
        str(args.get("report_flag") or "--json"), str(report),
    ]
    if impl_path is not None:
        argv += [str(args.get("impl_flag") or "--impl"), str(impl_path)]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, cwd=root, env=env, timeout=timeout
        )
    except FileNotFoundError:
        return f"{argv[0]} is not executable from {root}"
    except subprocess.TimeoutExpired:
        return f"the entrypoint exceeded {timeout:.0f}s"
    if proc.returncode != 0:
        return f"the entrypoint exited {proc.returncode}: {proc.stderr.strip()[-400:]}"
    if not report.is_file():
        return f"the entrypoint wrote no report at {report}"
    return None


def _medians(report: Path, operator_id: str) -> dict[str, float]:
    loaded = json.loads(report.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for entry in loaded.get("operators") or ():
        if not isinstance(entry, dict) or entry.get("operator_id") != operator_id:
            continue
        for shape in entry.get("shapes") or ():
            value = shape.get("weighted_mean_ms", shape.get("median_ms"))
            if isinstance(value, (int, float)):
                out[str(shape.get("case_id"))] = float(value)
    return out


def _remeasure(
    packup: Path, doc: dict, baseline_truth: dict[str, float], args: dict,
    problems: list[str], notes: list[str],
) -> None:
    """Re-run the workset's performance entrypoint on both sides, here, now."""
    apparatus = packup / _APPARATUS
    if not apparatus.is_dir():
        problems.append(f"{_APPARATUS}/ is missing; the kit does not carry what measures it")
        return

    performance = (doc.get("evidence") or {}).get("performance") or {}
    entrypoint = str(performance.get("entrypoint") or "").strip()
    if not entrypoint:
        problems.append("evidence.performance.entrypoint is empty; there is no protocol to re-run")
        return

    optimized_src = packup / "results" / "optimized_kernel.py"
    if not optimized_src.is_file():
        problems.append("results/optimized_kernel.py is missing")
        return

    timeout = _num(args.get("timeout_seconds"), 1800)
    tolerance = _num(args.get("tolerance"), 0.15)
    premise_tolerance = _num(args.get("baseline_agreement_tolerance"), 0.05)

    # `scratch_dir` is an argument and `TMPDIR` only the fallback. A validation
    # zone forces `TMPDIR` to `<zone>/tmp` and treats that as an invariant of
    # the zone (`validator/environment.py:233,86`), so the producer's `env`
    # cannot reach it. The zone sits under `--demo-root`; with a run root on
    # this cluster's NFS every ROCm kernel launch segfaults, and it does so
    # *after* the copies and the first round, so the run is lost at its most
    # expensive point.
    scratch_dir = str(args.get("scratch_dir") or "").strip() or os.environ.get("TMPDIR") or ""
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)
        notes.append(f"scratch under {scratch_dir}")
    root = Path(tempfile.mkdtemp(prefix="substantiate-", dir=scratch_dir or None))
    seed_root, candidate_root = root / "seed", root / "candidate"
    shutil.copytree(apparatus, seed_root)
    shutil.copytree(apparatus, candidate_root)

    if _interpreter(problems, notes) is None:
        return
    env = _measure_env(root)
    operator_id = str(doc.get("operator"))

    seed_report = seed_root / "substantiate_seed.json"
    failure = _run_entrypoint(seed_root, entrypoint, None, seed_report, args, env, timeout)
    if failure:
        problems.append(f"the seed re-measurement failed: {failure}")
        return
    seed = _medians(seed_report, operator_id)

    candidate_report = candidate_root / "substantiate_candidate.json"
    failure = _run_entrypoint(
        candidate_root, entrypoint, optimized_src.resolve(), candidate_report, args, env, timeout
    )
    if failure:
        problems.append(f"the optimised re-measurement failed: {failure}")
        return
    candidate = _medians(candidate_report, operator_id)

    # **Every case the workset baselines must come back from both sides.**
    #
    # Found by `stubkit` case 4, and it is the exact failure the shape of this
    # body invites: `_medians` skips a shape whose figure is absent or
    # non-numeric, `shared` is then an intersection, and the mean is taken over
    # whatever came back. A candidate that fails to measure on one shape is
    # therefore scored on the two it managed — which flatters precisely the
    # kernel that is fast on the easy shapes and broken on the hard one.
    #
    # An entrypoint that exits 0 and reports nothing for a case is not a smaller
    # sample, it is a failed measurement, and the two must not fold together.
    for label, side in (("seed", seed), ("optimised", candidate)):
        missing = sorted(c for c in baseline_truth if c not in side)
        if missing:
            problems.append(
                f"the {label} re-measurement returned no figure for {missing} — the entrypoint "
                "exited 0 and reported nothing for them. A case that did not measure is a failed "
                "measurement, not a smaller sample, and averaging over the rest scores a kernel "
                "on the shapes it happened to manage"
            )
    if problems:
        return

    # --- the premise, made empirical ---------------------------------------
    #
    # This is **not** the deleted rule coming back. It does not substitute a
    # denominator; it refuses. If the workset's baseline does not reproduce
    # here, the premise did not hold, and M4.3.5's answer to that is abort.
    drifted = [
        f"{case}: workset {baseline_truth[case]:.6f} ms, re-measured {seed[case]:.6f} ms "
        f"({(seed[case] / baseline_truth[case] - 1) * 100:+.1f}%)"
        for case in sorted(set(seed) & set(baseline_truth))
        if abs(seed[case] / baseline_truth[case] - 1.0) > premise_tolerance
    ]
    if drifted:
        problems.append(
            f"ABORT — the workset's own baseline does not reproduce on this machine within "
            f"{premise_tolerance:.0%}, so the premise did not hold empirically even though the "
            "environment records agree. The workset's numbers are the ground truth and cannot be "
            "replaced by these; the run is answering a different question (M4.3.5):"
        )
        problems.extend(f"  {line}" for line in drifted)
        return

    # --- the numerator ------------------------------------------------------
    stated = (performance.get("measured") or {}).get("per_case_ms") or {}
    for case in sorted(set(candidate) & set(stated)):
        if abs(candidate[case] / float(stated[case]) - 1.0) > tolerance:
            problems.append(
                f"measured.per_case_ms[{case}] is {stated[case]}, re-measured {candidate[case]:.6f} ms "
                f"— more than {tolerance:.0%} apart"
            )

    claim = performance.get("claim")
    if not claim:
        notes.append("no claim is made, so there is no ratio to substantiate")
        return

    shared = sorted(set(baseline_truth) & set(candidate))
    if not shared:
        problems.append(f"no case measured on both sides (workset {sorted(baseline_truth)}, here {sorted(candidate)})")
        return
    per_case = {case: baseline_truth[case] / candidate[case] for case in shared if candidate[case] > 0}
    measured_mean = statistics.fmean(per_case.values())
    claimed_mean = float(claim.get("mean_case_speedup", 0.0))

    # **No default.** An earlier draft fell back to 1.05, and m3 was right to
    # object: a consumer with a fallback floor is a consumer that silently picks
    # its own significance threshold on the one occasion the workset failed to
    # state one. The workset derives it from the measured spread as
    # `1 + 2.83 x rsd_max` — the two-sample 2-sigma separation, so a noisy host
    # correctly demands a bigger win — and the field is required on m3's side,
    # so its absence means something is wrong upstream and should say so.
    noise_floor = claim.get("noise_floor")
    if not isinstance(noise_floor, (int, float)):
        problems.append(
            f"claim.noise_floor is {noise_floor!r}, not a number. It is the workset's to declare "
            "and m4's to carry; nothing here substitutes a value for it"
        )
        return
    noise_floor = float(noise_floor)

    notes.append(
        "re-measured " + ", ".join(f"{c} {per_case[c]:.3f}x" for c in shared)
        + f"; mean {measured_mean:.3f}x against a claim of {claimed_mean:.3f}x"
    )

    if measured_mean < noise_floor:
        problems.append(
            f"re-measured {measured_mean:.3f}x is below the workset's noise floor {noise_floor:.3f}x — "
            "not distinguishable from measurement spread on this machine"
        )
    # One-sided on purpose. A handoff that under-claims is honest; a handoff
    # that over-claims is the thing this validator exists to catch.
    if measured_mean < claimed_mean * (1.0 - tolerance):
        problems.append(
            f"re-measured {measured_mean:.3f}x is more than {tolerance:.0%} below the claimed {claimed_mean:.3f}x"
        )


# --------------------------------------------------------------------------- #


def _load(packup: Path, rel: str, problems: list[str]):
    path = packup / rel
    if not path.is_file():
        problems.append(f"missing {rel}")
        return None
    try:
        if path.suffix in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(path.read_text(encoding="utf-8"))
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        problems.append(f"{rel} does not parse: {exc}")
        return None


def _check(hid: str, args: dict, problems: list[str], notes: list[str]) -> bool:
    content = zone.content_of(hid)
    if content is None:
        problems.append("the phase staged no content for this handoff")
        return False
    packup, why = zone.find_packup(content)
    if packup is None:
        problems.append(why)
        return False

    doc = _load(packup, _DOC, problems)
    snapshot = _load(packup, _SNAPSHOT, problems)
    baseline_report = _load(packup, _BASELINE_REPORT, problems)
    if doc is None or snapshot is None or baseline_report is None:
        return False

    # The free gates first, and the abort ones before anything is spent: a run
    # whose premise did not hold must not reach a timing loop, for the same
    # reason a kernel that is wrong must not.
    if not _check_premise(doc, args, problems, notes):
        return False
    _check_ground_truth(doc, snapshot, args, problems)
    _check_correctness(doc, snapshot, args, problems)
    if problems:
        return False

    _check_denominator(doc, baseline_report, problems, notes)
    if problems:
        return False

    baseline_truth = (doc["evidence"]["performance"]["baseline"] or {}).get("per_case_ms") or {}
    _remeasure(packup, doc, {k: float(v) for k, v in baseline_truth.items()}, args, problems, notes)
    return not problems


def main() -> int:
    args = zone.args()
    verdicts: dict[str, bool] = {}
    for hid in zone.inputs():
        problems: list[str] = []
        notes: list[str] = []
        verdicts[hid] = _check(hid, args, problems, notes)
        for note in notes:
            print(f"{hid} note: {note}")
        for problem in problems:
            print(f"{hid} problem: {problem}")
    # One entry per declared handoff. A missing entry raises at `PhaseRunner`'s
    # seam rather than folding as falsy.
    zone.write_verdict(verdicts)
    print(f"check_speedup_substantiated: {sum(verdicts.values())}/{len(verdicts)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
