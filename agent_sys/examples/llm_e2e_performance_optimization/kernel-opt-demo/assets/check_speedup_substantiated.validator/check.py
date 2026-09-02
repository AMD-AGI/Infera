#!/usr/bin/env python3
"""`check_speedup_substantiated` — does the claimed speedup re-measure, here, now?

The expensive half of the output gate. It takes the optimized kernel and the
measurement apparatus **both out of the handoff itself**, runs the 5-round
protocol against the seed and against the optimized kernel, and compares the
median-of-medians it got to the number the handoff claims.

**Why it reads one handoff and not two.** The first version declared
`inputs: [kernel_optimization, workset]`, because the driver lives in the
workset and an output phase stages only outputs. That is the documented route to
a handoff a phase would not otherwise stage — and it was wrong here, measured
2026-09-01. A phase's validator set is `closures.validators_for(<closure>)`, the
union of the closure's own list **and every validator joined to one of its
handoff kinds** — and a validator is joined to a kind by naming it in `inputs`.
So naming `workset` also **bound this body to the workset kind's phases**: it
ran in the publisher's output phase, against a workset with no optimization
beside it, and recorded a `trustworthiness / weak / FAIL` on a task that had
done nothing wrong.

Widening the body to tolerate missing materials would have been the wrong fix —
a validator that passes when its inputs are absent is worse than one scoped
correctly. Instead the **handoff carries its own apparatus**: the producer
copies `driver.py`, `graph_harness.py`, `measure_baseline.py` and the seed
kernel into `scripts/kernel/`, `check_optimization_shape` requires them, and
this body reads them from there. A reproduction kit that does not ship the thing
that measures it was under-specified anyway.

**Why `weak`.** It establishes one thing: the claimed number reproduces on this
machine today. It does not establish that the kernel is correct in general, that
it is fast on another shape, that it is safe to integrate, or that the speedup
survives at the service level. Those are four different claims and this body
makes none of them. The strength field is where that is said out loud.

**Why the tolerance is wide.** The optimized side of a comparison like this has
measured ~8% round-to-round spread on this hardware against the baseline's
~2%. A tight tolerance would fail honest handoffs more often than dishonest
ones, which is the wrong direction for a trustworthiness check to fail in.
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


def _interpreter(problems: list[str], notes: list[str]) -> str | None:
    """An interpreter that can actually `import torch`, or `None`.

    **This function exists because of a bug that made this validator fail every
    real run, and the failure looked like a measurement disagreement.**

    A validator body is started by `/bin/sh` with a *closed* environment
    (`validator/environment.py`): `os.environ` is not inherited and `PATH` is
    deliberately absent, so POSIX `sh` substitutes its built-in default. The
    template idiom `"${AGENT_SYS_DEMO_PYTHON:-python3}"` then resolves to
    `/usr/bin/python3` on the **output** phase, because the PRODUCER row shadows
    the GLOBAL row that carries `AGENT_SYS_DEMO_PYTHON` (see
    `bugs/002-validator-env-row-shadows-demo-python.md`).

    `/usr/bin/python3` is not the interpreter the supervisor runs under and has
    no `torch`. So `sys.executable` here is the *wrong* interpreter, and
    `measure_baseline.py` died on `import torch` in about 0.1 s — which this
    body faithfully reported as "measurement failed" and folded into a FAIL.
    Measured 2026-09-01 across three separate campaigns; the same handoff passed
    when re-run by hand with the venv interpreter.

    The bug record says this package was immune because its bodies import stdlib
    only. That was **wrong**: the body imports stdlib, and then shells out to a
    script that needs the whole ML stack. Immunity to a missing import is not
    immunity to picking the wrong interpreter.

    So: try candidates in order, verify each one really has `torch`, and say
    which was chosen. A body that guesses silently is how this cost three runs.
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


def _num(value: object, fallback: float) -> float:
    """Coerce an `args.json` value. Substitution yields **strings**, always."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


#: A PATH a compiler can actually work in. **Not decoration.**
#:
#: An optimized kernel here is a Triton kernel, and the first thing Triton does
#: on this backend is compile `hip_utils.c` by shelling out to `/bin/gcc` — which
#: then needs `as`, `ld` and `collect2` off `PATH`. A validator body's
#: environment is closed and deliberately carries no `PATH`
#: (`validator/environment.py`), so a subprocess started from it inherits none
#: and the compile dies:
#:
#:     CalledProcessError: ['/bin/gcc', '.../hip_utils.c', '-O3', '-shared', ...]
#:     returned non-zero exit status 1
#:
#: Measured 2026-09-01. The baseline side survives this because it is plain
#: `torch.softmax` and compiles nothing — so the symptom is *only the optimized
#: side fails*, which reads exactly like a broken optimized kernel. It is not.
_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _measure_env(scratch: Path) -> dict[str, str]:
    """The environment the measurement subprocess needs, built rather than inherited.

    A validator body gets five closed channels and nothing else, so anything the
    measurement needs has to be supplied here explicitly. Three things do:

    - **`PATH`**, for the compiler's own sub-tools — see `_PATH` above.
    - **`TRITON_CACHE_DIR`**, because Triton otherwise writes to `$HOME/.triton`,
      and `$HOME` on this class of host is an NFS mount whose writes fail
      silently for a container user. Pointed inside the scratch tree.
    - **`HOME`**, because several libraries probe it and an unset one is not the
      same as a writable one.

    `HIP_VISIBLE_DEVICES` is *not* defaulted. It arrives from the agent spec's
    `env:` block through the PRODUCER row, and inventing a default here would
    silently move the measurement onto card 0 — which on a shared host is
    somebody else's.
    """
    env = dict(os.environ)
    env["PATH"] = env.get("PATH") or _PATH
    env.setdefault("TRITON_CACHE_DIR", str(scratch / "triton_cache"))
    env.setdefault("HOME", str(scratch))
    Path(env["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    return env


def _run_protocol(
    python: str, kernel_dir: Path, scratch: Path, rounds: int, iters: int, timeout: float
) -> dict[str, float] | str:
    """The workset's own protocol, in a fresh process. Returns medians or an error string."""
    script = kernel_dir / "measure_baseline.py"
    if not script.is_file():
        return "the workset carries no kernel/measure_baseline.py"
    out = kernel_dir / "measured.json"
    try:
        proc = subprocess.run(
            [python, str(script), "--rounds", str(rounds), "--iters", str(iters), "--json", str(out)],
            capture_output=True,
            text=True,
            cwd=kernel_dir,
            env=_measure_env(scratch),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"measurement exceeded {timeout:.0f}s"
    if proc.returncode != 0:
        return f"measure_baseline.py exited {proc.returncode}: {proc.stderr.strip()[-400:]}"
    if not out.is_file():
        return "measure_baseline.py wrote no json"
    loaded = json.loads(out.read_text(encoding="utf-8"))
    return {case: float(v["median_ms"]) for case, v in loaded["cases"].items()}


def _check(optimization: str, args: dict, problems: list[str], notes: list[str]) -> bool:
    opt_content = zone.content_of(optimization)
    if opt_content is None:
        problems.append("the phase staged no content for this handoff")
        return False

    packup, _ = zone.find_packup(opt_content)
    if packup is None:
        problems.append("no packup in the optimization handoff")
        return False

    verification = packup / "results" / "verification.json"
    if not verification.is_file():
        problems.append("results/verification.json is missing; there is no claim to substantiate")
        return False
    claim = json.loads(verification.read_text(encoding="utf-8"))

    # A mock claims nothing, so there is nothing to substantiate and this body
    # must not invent a failure. `check_optimization_shape` has already refused
    # a mock that claims a speedup or hides that it is one.
    if bool(claim.get("mock")):
        notes.append("mock run: no speedup claimed, nothing to re-measure")
        return True

    claimed = claim.get("mean_case_speedup")
    if not isinstance(claimed, (int, float)):
        problems.append(f"mean_case_speedup is {claimed!r}, not a number")
        return False

    # The apparatus travels inside the handoff, so there is no second handoff to
    # find and no store to reach for.
    apparatus = packup / "scripts" / "kernel"
    if not apparatus.is_dir():
        problems.append("scripts/kernel/ is missing; the kit does not carry what measures it")
        return False

    rounds = int(_num(args.get("rounds"), 5))
    iters = int(_num(args.get("iters"), 30))
    tolerance = _num(args.get("tolerance"), 0.15)
    noise_floor = _num(args.get("noise_floor"), 1.05)
    timeout = _num(args.get("timeout_seconds"), 1800)

    optimized_src = packup / "results" / "optimized_kernel.py"
    if not optimized_src.is_file():
        problems.append("results/optimized_kernel.py is missing")
        return False

    # Two sibling scratch copies of the workset's kernel directory: one left as
    # the seed, one with the optimized kernel dropped in. Created under a fresh
    # mkdtemp so nothing existing is touched and nothing is deleted.
    root = Path(tempfile.mkdtemp(prefix="substantiate-", dir=os.environ.get("TMPDIR") or None))
    base_dir, opt_dir = root / "baseline", root / "optimized"
    shutil.copytree(apparatus, base_dir)
    shutil.copytree(apparatus, opt_dir)

    # The seed kernel is the one module in kernel/ that the driver imports and
    # that is neither the driver, the harness nor the protocol script.
    reserved = {"driver.py", "graph_harness.py", "measure_baseline.py"}
    seeds = [p for p in opt_dir.glob("*.py") if p.name not in reserved]
    if len(seeds) != 1:
        problems.append(f"expected exactly one kernel module in the workset, found {[p.name for p in seeds]}")
        return False
    shutil.copyfile(optimized_src, seeds[0])

    python = _interpreter(problems, notes)
    if python is None:
        return False

    base = _run_protocol(python, base_dir, root, rounds, iters, timeout)
    if isinstance(base, str):
        problems.append(f"baseline re-measurement failed: {base}")
        return False
    opt = _run_protocol(python, opt_dir, root, rounds, iters, timeout)
    if isinstance(opt, str):
        problems.append(f"optimized re-measurement failed: {opt}")
        return False

    shared = sorted(set(base) & set(opt))
    if not shared:
        problems.append(f"no case measured on both sides (baseline {sorted(base)}, optimized {sorted(opt)})")
        return False

    per_case = {case: base[case] / opt[case] for case in shared if opt[case] > 0}
    measured = statistics.fmean(per_case.values())
    notes.append(
        "re-measured "
        + ", ".join(f"{c} {base[c]*1000:.2f}->{opt[c]*1000:.2f}us {per_case[c]:.3f}x" for c in shared)
    )
    notes.append(f"re-measured mean {measured:.3f}x against a claim of {float(claimed):.3f}x")

    if measured < noise_floor:
        problems.append(
            f"re-measured {measured:.3f}x is below the noise floor {noise_floor:.2f}x — "
            "not distinguishable from measurement spread on this machine"
        )
    # One-sided on purpose. A handoff that under-claims is honest; a handoff
    # that over-claims is the thing this validator exists to catch.
    if measured < float(claimed) * (1.0 - tolerance):
        problems.append(
            f"re-measured {measured:.3f}x is more than {tolerance:.0%} below the claimed "
            f"{float(claimed):.3f}x"
        )
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
