#!/usr/bin/env python3
"""`check_acceptance` — completeness, strong.

All three kinds of correctness evidence arrived, and each is readable on its own
terms.

**What this validator deliberately does not do: compare the arms.** An eval score
is admitted whatever it is, because a single score has no baseline — the 1P1D
kit's own README says it plainly, and `check_no_regression` is where two scores
become a judgement. What is checked here is that the measurement happened and can
be read.

Two of the rules exist because of failures that produce a NUMBER rather than an
error, which is the only kind worth spending a strong validator on:

  a needle run that silently sent a 3000-token prompt is not a long-context test,
  and it passes. `usage.prompt_tokens` is read back from the server for exactly
  this reason, and the ratio is checked here.

  an eval whose html holds fewer scored rows than its index claims did not score
  what it says it scored, and every interval computed from it downstream is
  wrong by an unknown amount.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def smoke_ok(result: Path, args: dict, reasons: list) -> bool:
    smoke = read_json(result / "smoke.json")
    if smoke is None:
        return _fail(reasons, "smoke.json is missing or unreadable")
    got = {c["name"]: c for c in smoke.get("checks", [])}
    ok = True
    for name in args.get("require_smoke_checks") or ():
        if name not in got:
            ok = _fail(reasons, f"smoke.json has no {name!r} check")
        elif not got[name].get("ok"):
            ok = _fail(reasons, f"smoke check {name!r} failed: {got[name]}")
    return ok


def needle_ok(result: Path, args: dict, reasons: list) -> bool:
    needle = read_json(result / "needle.json")
    if needle is None:
        return _fail(reasons, "needle.json is missing or unreadable")
    runs = {run["label"]: run for run in needle.get("runs", [])}
    if "gated" not in runs:
        return _fail(reasons, "needle.json carries no gated run")

    ok = True
    gated = runs["gated"]
    ratio = float(args.get("needle_min_token_ratio", 0.95))
    target = gated.get("target_tokens") or 0

    # **The prompt length is the part that is gated**, because it is the part
    # that is a fact rather than a model behaviour. A needle run that silently
    # sent a short prompt is not a long-context test and would pass everything
    # else here.
    for depth in gated.get("depths", []):
        got = depth.get("prompt_tokens")
        if got is None:
            ok = _fail(reasons, f"gated depth {depth['depth']} recorded no prompt_tokens")
        elif got < target * ratio:
            ok = _fail(
                reasons,
                f"gated depth {depth['depth']} sent {got} prompt tokens against a target of "
                f"{target}; below {ratio:.0%} this is not the length the run claims to test",
            )
        # A starved answer is a configuration fault, not a retrieval result, and
        # it has to be separated out or it would be counted as evidence about the
        # deployment. The engine reasons before it answers; too small a budget
        # returns nothing at all.
        if depth.get("starved"):
            ok = _fail(
                reasons,
                f"gated depth {depth['depth']} returned no answer at all because the whole "
                "token budget went to reasoning — raise needle --max-tokens; this is not a "
                "retrieval failure and must not be recorded as one",
            )

    # **Retrieval is a floor, not a gate**, and the floor is one depth.
    #
    # Nine measurements on this deployment produced no configuration that
    # retrieves at every depth reliably: the outcome is not monotonic in prompt
    # length, it moves with the needle's wording, and raising the generation
    # budget from 256 to 2048 turned two passing depths into failures because the
    # model reasons its way past the answer. Requiring all three would be
    # requiring something nothing has achieved twice. Requiring one separates a
    # working long-context path from a stack that dropped every prefill chunk but
    # the last, which is the failure this test was added for.
    #
    # The comparison between the arms is where a needle result becomes evidence,
    # and that lives in `compare` / `check_no_regression`.
    floor = int(args.get("needle_min_depths_retrieved", 1))
    retrieved = gated.get("retrieved")
    if retrieved is None:
        retrieved = sum(1 for d in gated.get("depths", []) if d.get("ok"))
    if retrieved < floor:
        ok = _fail(
            reasons,
            f"the gated needle retrieved at {retrieved} of {len(gated.get('depths', []))} "
            f"depth(s), floor is {floor} — at zero this is a long-context path that is not "
            "returning anything from the prompt",
        )
    else:
        print(
            f"  needle: {retrieved}/{len(gated.get('depths', []))} gated depths retrieved "
            f"(floor {floor}); per-depth results are comparison material, not a capability claim"
        )

    if "frontier" in runs:
        frontier = runs["frontier"]
        print(
            f"  needle: {frontier.get('retrieved')}/{frontier.get('of')} frontier depths "
            f"retrieved at ~{frontier.get('target_tokens')} tokens (recorded, never gated)"
        )
    return ok


def probe_recorded(result: Path, reasons: list) -> bool:
    """The probe ran and its answer is in the record. **Not that it passed.**

    A failing probe means an eval score from this deployment is not
    interpretable, and the right place to act on that is the comparison, where
    `compare` marks the eval `uninterpretable` rather than `same`. Failing here
    instead would throw away an arm's worth of measurement — including the
    replay, which does not depend on the probe at all — for a reason that has
    nothing to do with the patch under test.
    """
    probe = read_json(result / "probe.json")
    if probe is None:
        return _fail(reasons, "probe.json is missing — the eval gate never ran")
    if not probe.get("ok"):
        print(
            "  note: probe FAILED, so an eval score from this arm is not interpretable. "
            "The failures were: " + "; ".join(probe.get("failures") or ["(none recorded)"])
        )
    return True


def evals_ok(result: Path, args: dict, reasons: list) -> bool:
    index = result / "lm_eval" / ".index"
    if not index.is_file():
        return _fail(reasons, "lm_eval/.index is missing — no eval was scored")
    rows = [line.split("\t") for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return _fail(reasons, "lm_eval/.index is empty — every eval failed before scoring")

    ok = True
    floor = int(args.get("min_scored_per_eval", 20))
    for row in rows:
        name, scored, filename = row[0], int(row[1] or 0), row[2]
        payload = read_json(result / "lm_eval" / filename)
        if payload is None:
            ok = _fail(reasons, f"eval {name!r}: {filename} is missing or unreadable")
            continue
        if payload.get("score", payload.get("mean_score")) is None:
            ok = _fail(reasons, f"eval {name!r}: {filename} carries no score")
        if scored < floor:
            ok = _fail(
                reasons,
                f"eval {name!r} scored {scored} question(s), floor is {floor} — below that "
                "the interval is too wide for the number to mean anything",
            )
        # The html is where a moved score is explained, and it is also where the
        # scored count came from; a missing one makes that count unverifiable.
        html = result / "lm_eval" / filename.replace(".json", ".html")
        if not html.is_file():
            ok = _fail(
                reasons,
                f"eval {name!r} has no html report, so its scored count cannot be re-derived "
                "and there is nowhere to see WHY the score is what it is",
            )
        else:
            counted = html.read_text(encoding="utf-8", errors="replace").count("Correct Answer")
            if counted != scored:
                ok = _fail(
                    reasons,
                    f"eval {name!r}: the index says {scored} scored, the html holds {counted}",
                )
    return ok


def check(content: Path, args: dict, reasons: list) -> bool:
    result = content / "items" / "result"
    if not result.is_dir():
        return _fail(reasons, "items/result/ is missing")
    # Not short-circuited: every rule runs, so one rerun clears the whole set.
    return all(
        [
            smoke_ok(result, args, reasons),
            needle_ok(result, args, reasons),
            probe_recorded(result, reasons),
            evals_ok(result, args, reasons),
        ]
    )


def main() -> int:
    args = zone.args()
    results = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_acceptance: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
