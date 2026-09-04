#!/usr/bin/env python3
"""Launch `agent-sys run` with a stall threshold a real stage can survive.

**Why this exists.** `agent_sys/cli/main.py:911` defaults `stall_after` to
**20 seconds**, and `main.py:1015` ends the run when

    (not holding or blocked) and now - last_change > stall_after

`holding` counts only an attempt **mid-model-call** (`main.py:980`). So a
`kind: ai` stage is safe for as long as its conversation runs, and a long
**program** stage is not: it produces no status change, holds no executor, and
after twenty seconds the run is declared stalled while it is working.

Measured on rung 2b, 2026-09-04:

    deploy_and_prove            succeeded, 3 strong verdicts, deploy_kit valid
    run_profiling_mode_off      RUNNING
    profiling_mode_off.bench_result   GENERATING
    -> "Nothing has changed for 20 s" -> run ended

**This is not a hypothetical about slow tasks.** m2's three closures are
`agent: runner` *literally* — that stage has no `kind: ai` body by design,
because bring-up folds into the task that uses it (mission M2.5). It is
therefore the one stage in the package structurally unable to hold an executor,
and every rung from 2 onward runs it for real. Rungs 3, 4 and 5 all pass
through it.

**Why a launcher rather than a fix.** `agent_sys/cli/` is outside this effort's
activity scope, and both halves of the failure are already recorded:

    temp/bugs/2026-09-03-the-stall-detector-ends-a-run-while-a-task-is-still-working.md
    temp/bugs/2026-09-04-an-escalation-with-no-recipient.md

The first states that `stall_after` is **not exposed on the CLI, so there is no
operator-side knob** — true, and the reason this file exists. `--timeout` is an
absolute ceiling on the whole run and is a different thing; it is left alone.

**What this gives up, stated rather than buried.** A genuinely hung run now
takes `--stall-after` seconds to be declared stalled instead of twenty. That is
the entire safety property being traded, and `--timeout` still bounds the run
(4 h by default). A hang costs one long wait; the 20 s default costs every real
rung, which is why the trade is worth making *here* and would not be in general.

Usage — identical to `python3 -m agent_sys.cli.main`, plus one flag:

    python3 assets/lib/run_with_long_stall.py --stall-after 3600 run --package … --var …

**Verified before use** (2026-09-04): `_settle` is called at `main.py:405` with
only `timeout=`, so `stall_after` genuinely comes from the default; and
`__kwdefaults__` is externally mutable with the change visible to that caller.
Both checked by measurement, not by reading.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    seconds = 3600.0
    if argv and argv[0] == "--stall-after":
        if len(argv) < 2:
            print("run_with_long_stall: --stall-after needs a value", file=sys.stderr)
            return 2
        try:
            seconds = float(argv[1])
        except ValueError:
            print(f"run_with_long_stall: {argv[1]!r} is not a number", file=sys.stderr)
            return 2
        argv = argv[2:]

    # `python3 <path>/run_with_long_stall.py` puts *this file's* directory on
    # `sys.path`, not the repo root, so `agent_sys` is not importable — measured,
    # not guessed. Every documented invocation runs from the repo root, which is
    # what `python3 -m agent_sys.cli.main` relies on too, so the cwd is the
    # right thing to add and the failure without it is loud.
    import os

    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())

    from agent_sys.cli import main as cli

    # **Refuse rather than silently do nothing.** If upstream renames the
    # parameter, makes it positional, or starts passing it at the call site,
    # the patch stops applying — and a launcher that quietly reverts to 20 s
    # would present the framework's own defect as this package's. That is the
    # `${VAR:?}`-is-inert shape and the reason this is a hard exit.
    kw = getattr(cli._settle, "__kwdefaults__", None)
    if not kw or "stall_after" not in kw:
        print(
            "run_with_long_stall: agent_sys.cli.main._settle no longer takes a "
            "keyword-only `stall_after`. The patch cannot apply and the 20 s "
            "default would silently return. Re-read main.py:905 before running.",
            file=sys.stderr,
        )
        return 2

    before = kw["stall_after"]
    kw["stall_after"] = seconds
    print(
        f"run_with_long_stall: stall_after {before:g}s -> {seconds:g}s "
        f"(see this file's docstring for what that trades away)",
        file=sys.stderr,
    )

    return cli.main(argv)


if __name__ == "__main__":
    sys.exit(main())
