#!/usr/bin/env python3
"""Negative control for `check_deploy_serves`' NCCL retry.

**Why this exists.** The retry has never executed a line. It fires only on an
intermittent fault, and three real deploys on 2026-09-05 all took the clean
path. Unexecuted code that runs *only* when something has already gone wrong is
the worst kind to leave untested — and `_note`'s missing `file=` keyword,
found the same day two lines below this block, is what that looks like when it
lands.

So the fault is injected rather than waited for. `check.on` is replaced with a
recorder that fails the first bring-up with the measured signature and succeeds
after. Nothing here touches a node.

Four assertions, each naming a condition that fails:

  A. the signature is recognised -> both retry notes appear
  B. teardown runs BETWEEN the two attempts, not after both
  C. a non-matching failure does NOT retry (the narrowness is the point)
  D. two consecutive failures are reported as new, not swallowed

Run it from the package root; it takes seconds and needs no node:

    python3 assets/check_deploy_serves.validator/retry_gate.py

**A gate that passes is the weakest evidence in this package**, so this one was
shown to fail before it was kept. Three mutations on a scratch copy, 2026-09-05,
each caught by the assertion aimed at it and the baseline restored to 4/4 after
each:

    _sig = False                    -> A, A2, B, D fail (6 assertions)
    _sig = True (retry widened)     -> C fails, all three of its assertions
    teardown call deleted           -> B fails on call 2 not being the teardown

**And one assertion here was wrong before the code was.** The first version
asserted case C makes *exactly one call*; it saw two and the second was the
teardown that `check_one`'s exit path runs on every route, including a failed
bring-up. The code was right and the gate was wrong — corrected to count
bring-ups, and to assert the cleanup positively rather than forbid it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = Path(sys.argv[1] if len(sys.argv) > 1 else
           "/home/yihou/dev/git/infera.aiopt.real.task_package/agent_sys/examples/"
           "llm_e2e_performance_optimization/e2e-flow")
sys.path.insert(0, str(PKG / "assets/lib"))
sys.path.insert(0, str(PKG / "assets/check_deploy_serves.validator"))

import check  # noqa: E402

# The two measured signatures, verbatim from the comment above the retry.
NCCL_TEXT = ("[TP1] NCCL error: unhandled cuda error\n"
             "HIP failure: 'invalid argument'\n")
MINUS_9_TEXT = "worker exited with code -9 before reporting ready\n"
OTHER_TEXT = "deploy.sh: line 12: model path does not exist\n"


def fixture(tmp: Path) -> Path:
    codes = tmp / "items" / "codes"
    (codes / "retry-gate.packup_20260905").mkdir(parents=True)
    (codes / "environment.yaml").write_text(
        "fixed:\n  served_model_name: gate/model\n  node: gate-node\n"
    )
    return tmp


def run_case(name: str, script: list, *, expect_retry: bool) -> tuple[list[str], list[str], list[str]]:
    """`script` is a list of ('fail', text) / ('ok', stdout) per `on` call."""
    import tempfile

    calls: list[str] = []
    seq = list(script)

    def fake_on(command, transport, *, check_=True, **kw):
        calls.append(command)
        if not seq:
            # Past the point this case is about. Return empty rather than abort:
            # `check_one` then fails naturally on the unreadable handshake and
            # still runs its teardown, so the case ends the way a real one does
            # instead of unwinding through a half-finished function.
            return ""
        kind, payload = seq.pop(0)
        if kind == "fail":
            raise check.NodeError(command, 1, payload)
        return payload

    # `on` is called with `check=` by some sites; accept it under any spelling.
    def shim(command, transport, **kw):
        return fake_on(command, transport, **{k: v for k, v in kw.items() if k != "check"})

    real_on = check.on
    check.on = shim
    try:
        with tempfile.TemporaryDirectory() as td:
            content = fixture(Path(td))
            notes: list[str] = []
            faults = check.check_one(
                content,
                {"work_root": "/tmp/gate", "port_base": 8999},
                {"kind": "local"},
                {},
                notes,
            )
    finally:
        check.on = real_on
    return notes, faults, calls


def main() -> int:
    bad: list[str] = []

    # ---- A + B: the signature retries, and teardown sits between attempts ----
    notes, faults, calls = run_case(
        "A", [("fail", NCCL_TEXT), ("ok", ""), ("ok", "DEPLOY_OK\n")], expect_retry=True
    )
    joined = "\n".join(notes)
    if "failed on attempt 1 with the intermittent NCCL signature" not in joined:
        bad.append("A: attempt-1 note missing — the signature was not recognised")
    if "retrying once" not in joined:
        bad.append("A: 'retrying once' note missing")
    if len(calls) < 3:
        bad.append(f"B: expected 3 calls (deploy, teardown, deploy), saw {len(calls)}")
    else:
        if "teardown" not in calls[1]:
            bad.append(f"B: call 2 is not the teardown: {calls[1][:90]}")
        if "deploy" not in calls[2]:
            bad.append(f"B: call 3 is not the second bring-up: {calls[2][:90]}")

    # the -9 spelling is the other measured signature and must behave the same
    notes2, _, calls2 = run_case(
        "A2", [("fail", MINUS_9_TEXT), ("ok", ""), ("ok", "DEPLOY_OK\n")], expect_retry=True
    )
    if "retrying once" not in "\n".join(notes2):
        bad.append("A2: the 'exited with code -9' signature did not retry")

    # ---- C: anything else must NOT retry ------------------------------------
    notes3, faults3, calls3 = run_case("C", [("fail", OTHER_TEXT)], expect_retry=False)
    if "retrying once" in "\n".join(notes3):
        bad.append("C: an unrelated bring-up failure was retried — the retry is too wide")
    if not faults3 or "failed (rc=1)" not in faults3[0]:
        bad.append(f"C: expected a plain bring-up fault, got {faults3!r}")
    # Exactly ONE bring-up. There is a second call and it is the teardown that
    # `check_one`'s finally runs on every exit path — measured, not assumed: an
    # earlier version of this gate asserted "exactly 1 call" and failed, and the
    # code was right. A failed bring-up still has to be cleaned up.
    bringups3 = [c for c in calls3 if "deploy.sh" in c]
    if len(bringups3) != 1:
        bad.append(f"C: expected exactly 1 bring-up, saw {len(bringups3)} in {len(calls3)} calls")
    if not any("teardown.sh" in c for c in calls3):
        bad.append("C: a failed bring-up did not run the teardown")

    # ---- D: two consecutive failures are NEW, not swallowed ------------------
    notes4, faults4, calls4 = run_case(
        "D", [("fail", NCCL_TEXT), ("ok", ""), ("fail", NCCL_TEXT)], expect_retry=True
    )
    if not faults4 or "failed twice" not in faults4[0]:
        bad.append(f"D: two failures did not produce the 'failed twice' fault: {faults4!r}")
    if faults4 and "new" not in faults4[0].lower():
        bad.append("D: the 'failed twice' fault does not say this is new")

    for line in bad:
        print(f"FAIL  {line}")
    if bad:
        print(f"\nretry_gate: {len(bad)} assertion(s) failed")
        return 1
    print("retry_gate: A signature recognised, B teardown between attempts, "
          "C narrowness held, D two failures reported as new — 4/4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
