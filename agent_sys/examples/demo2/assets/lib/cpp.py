#!/usr/bin/env python3
"""Compiling, running and timing one C++ translation unit.

Four validator bodies in this package — `check_compiles`, `check_one_binary`,
`check_extra_tests`, `check_faster` — all need the same three motions: turn a
`.cpp` into a binary, feed it stdin and collect stdout, and say how long that
took. This is that, and nothing else.

**It imports nothing from `agent_sys`, and it may not.** A task package is data;
`../store.py`'s docstring states the rule and this file is under it. Only the
standard library is used, so the same reason that admits `subprocess` here
admits it in any package outside this repository.

**Why `subprocess` and not a build tool.** The repository rule is to prefer a
mature library over writing it yourself, and the mature thing here is `g++`
itself — invoked directly. A single translation unit with no dependencies, no
link order and no incremental rebuild is the case CMake, Meson and `make` all
exist *above*; wiring one in would add a generator step, a build directory and a
second failure mode in exchange for nothing this package needs. The Python
ecosystem has no de facto "compile one C++ file" library — `cppyy` and
`pybind11` solve embedding, not batch compilation — so `subprocess.run` around
the compiler is the thin wrapper, and it is the whole of it.

Timeouts are the reason this is a module rather than three call sites. A
submitted program that loops for ever must fail as *timed out* and not hang the
run, and getting that right once is worth more than getting it nearly right four
times.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

#: `g++`'s own default is `-O0`, and a validator that ranks two programs by wall
#: clock must not be ranking the optimiser's absence. `-std=c++17` is pinned for
#: the same reason a lockfile is: g++ 11's default is `gnu++17`, g++ 14's is
#: `gnu++17` too but g++ 15 moved to `gnu++20`, so leaving it unset makes the
#: language a property of the machine.
COMPILE_FLAGS = ("-O2", "-std=c++17")

#: Compilation is bounded because a pathological template can genuinely not
#: terminate, and a submitted source is not trusted input.
COMPILE_TIMEOUT = 60.0

#: `timeout(1)`'s exit status for *the command timed out*. A killed process
#: reports `-9` through `Popen.returncode`, which is indistinguishable from a
#: program that was killed for some other reason, so the timed-out case is given
#: a code of its own and the convention chosen is the one already in every
#: shell script that needed it.
TIMEOUT_RETURNCODE = 124


def compile(src: Path, out: Path) -> tuple[bool, str]:  # noqa: A001
    """Build `src` into the executable `out`. Returns `(ok, diagnostics)`.

    `diagnostics` is the compiler's stderr, verbatim and unparsed — a validator
    reporting *why* a submission failed to build should quote the compiler
    rather than summarise it, and every attempt to classify a g++ diagnostic by
    regex is a second, worse copy of the compiler's own message.

    The name shadows the builtin. That is deliberate: the call site reads
    `cpp.compile(...)`, the module is never `import *`-ed, and naming it
    `compile_source` to dodge a builtin nothing here uses would be worse.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["g++", *COMPILE_FLAGS, "-o", str(out), str(src)]
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"g++ did not finish within {COMPILE_TIMEOUT:.0f}s"
    except FileNotFoundError:
        # Not the submission's fault, and saying so is the difference between a
        # verdict a reviewer can act on and one that blames the wrong party.
        return False, "g++ is not on PATH"
    return done.returncode == 0, done.stderr


def run(
    binary: Path,
    stdin_text: str,
    *,
    timeout: float = 30.0,
    argv: Sequence[str] = (),
) -> tuple[int, str, str, float]:
    """Run `binary` with `stdin_text` on stdin. Returns `(rc, out, err, seconds)`.

    `seconds` is wall clock measured with `time.perf_counter`, which is the
    monotonic clock and so cannot go backwards if the machine's time is
    adjusted mid-run. It is wall and not CPU time on purpose: `check_faster`
    compares two programs a human would wait on.

    On timeout the return code is `TIMEOUT_RETURNCODE` and `seconds` is the
    elapsed budget — the caller gets a number rather than an exception, because
    *timed out* is a verdict this package records and not an error it recovers
    from.

    **`argv` was missing and its absence was a bug in the graph, not a gap in
    this file.** The unified harness selects a case by `argv[1]`; with no way to
    pass one, `score.py` prepended the case id to stdin and the harness read it
    with `fgets`. Every solution begins `std::ios::sync_with_stdio(false)`, which
    discards the position any earlier stdio read left — so the harness consumed
    the input into the C buffer and the solution's first `std::cin >> n` failed.
    Every case scored zero, the binary exited 0, and every validator passed:
    `check_one_binary` asks whether one executable exists and dispatches, and
    `check_scores` asks whether the arithmetic is reproducible. Both were true.

    Measured over seven programs in
    `scratch/demo2-2026-08/probe_fgets_eats_stdin.py`: `fgets`, `getchar` and
    `getline` **all** starve a desynced solution, and all three are fine with a
    synced one. The choice of function is not the variable; reading stdin at all
    before the solution runs is. So there is no correct stdin fallback, and the
    id belongs on argv.
    """
    started = time.perf_counter()
    try:
        done = subprocess.run(
            [str(binary), *argv],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as expired:
        elapsed = time.perf_counter() - started
        # `TimeoutExpired` carries whatever was read before the kill, as `bytes`
        # or `str` depending on how the child was opened. Text mode was asked
        # for, but the attribute is `None` when nothing arrived, so neither the
        # type nor the presence can be assumed.
        return TIMEOUT_RETURNCODE, _text(expired.stdout), _text(expired.stderr), elapsed
    return done.returncode, done.stdout, done.stderr, time.perf_counter() - started


def check_cases(binary: Path, cases: list[dict], *, argv: Sequence[str] = ()) -> list[dict]:
    """Run every case and say, per case, whether the output matched.

    `argv` is passed to every case — it selects *which* solution the unified
    harness dispatches to, and that is constant across one pair's cases while
    the input is not. See `run`'s docstring for why it cannot travel on stdin.

    A case is `{"input": str, "expected": str}`. The result adds `ok`,
    `returncode`, `stdout`, `stderr`, `seconds` and `timed_out`, and keeps
    `expected` beside them so that one entry is a complete record of one
    comparison — a caller writing a failure report needs no second lookup.

    **Comparison is stripped and per-line stripped**, which is the loosest rule
    that still catches a wrong answer: leading and trailing blank lines go, and
    each surviving line is stripped of its own surrounding whitespace. A
    competitive-programming judge does exactly this, and the alternative —
    byte equality — fails a correct program over a trailing newline.
    """
    results: list[dict] = []
    for case in cases:
        rc, out, err, seconds = run(binary, case["input"], argv=argv)
        timed_out = rc == TIMEOUT_RETURNCODE
        results.append(
            {
                "ok": rc == 0 and _normalise(out) == _normalise(case["expected"]),
                "returncode": rc,
                "stdout": out,
                "stderr": err,
                "seconds": seconds,
                "timed_out": timed_out,
                "expected": case["expected"],
            }
        )
    return results


def _normalise(text: str) -> str:
    """Strip the whole, then strip each line. See `check_cases`."""
    return "\n".join(line.strip() for line in text.strip().splitlines())


def _text(raw: object) -> str:
    """Whatever `TimeoutExpired` captured, as `str`. See `run`."""
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)
