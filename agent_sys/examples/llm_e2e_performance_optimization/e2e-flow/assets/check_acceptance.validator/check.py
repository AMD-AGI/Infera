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

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import workset_io  # noqa: E402 — the shared report writer; see _report()
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
    """The **frozen** set (M5.4). These live in `assets/accept/` and ship with the
    package, so what they test cannot be chosen after the numbers are in."""
    smoke = read_json(result / "smoke.json")
    if smoke is None:
        return _fail(reasons, "smoke.json is missing or unreadable")
    got = {c["name"]: c for c in smoke.get("checks", [])}
    ok = True
    for name in args.get("require_frozen_checks") or ():
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


def adhoc_ok(result: Path, args: dict, reasons: list) -> list[str]:
    """M5.4 — 同时还要临时 ai 生成几个。免得作弊.

    **Why a frozen set is not enough.** Every check in `assets/accept/` is in the
    repository, so an optimisation — or an agent driving one — can be made to
    satisfy exactly those and nothing else, and the suite then measures
    compliance with itself. A handful of cases invented per run cannot be
    prepared for.

    **Why the prompts are recorded rather than only the results.** A generated
    case whose text is thrown away is unauditable: a later reader cannot tell a
    hard question that was answered from a trivial one that was asked instead,
    and "three ad-hoc cases passed" then means nothing. The mission asks for the
    cases; the handoff carries what was asked as well as what came back.

    Three rules beyond the count, each closing a way the requirement could be
    met without being met:

    - **no ad-hoc case may repeat a frozen one.** Regenerating the shipped suite
      satisfies the count and adds no coverage.
    - **the ad-hoc prompts must differ from each other.** Three copies of one
      case is one case.
    - **both arms must run the same ad-hoc set**, which is checked across the
      two handoffs by the caller. Different cases per arm is not a comparison,
      and it is the shape a regression could hide behind.

    Returns the case ids, so the caller can compare the arms.
    """
    # `args.get(k, d)` and **not** `args.get(k) or d`: `0 or 3` is `3`, so the
    # `or` form reads a deliberate `min_adhoc_cases: 0` as the default and
    # silently re-arms a rule the operator turned off. Harmless here because
    # both values are 0 — kept in the safe shape so that changing the default
    # later cannot introduce the bug. Raised by m3, who found the live case.
    floor = int(args.get("min_adhoc_cases", 0))
    payload = read_json(result / "adhoc.json")
    if payload is None:
        if floor:
            _fail(
                reasons,
                f"adhoc.json is missing and {floor} ad-hoc case(s) are required (M5.4). The "
                "frozen suite is in the repository and can be satisfied by construction; the "
                "per-run cases are the part that cannot.",
            )
        return []

    generator = payload.get("generator") or {}
    if not str(generator.get("prompt") or "").strip():
        _fail(
            reasons,
            "adhoc.json records no generator prompt. What was ASKED is half the evidence — "
            "without it, a reader cannot tell a hard case that was answered from an easy one "
            "that was substituted.",
        )

    cases = payload.get("cases") or []
    if len(cases) < floor:
        _fail(reasons, f"adhoc.json carries {len(cases)} case(s), floor is {floor}")

    frozen_prompts = {
        str(c.get("prompt") or "").strip().lower()
        for c in (read_json(result / "smoke.json") or {}).get("checks", [])
        if c.get("prompt")
    }
    seen: set[str] = set()
    ids: list[str] = []
    for i, case in enumerate(cases):
        where = f"adhoc case {case.get('id') or i}"
        prompt = str(case.get("prompt") or "").strip()
        if not prompt:
            _fail(reasons, f"{where} carries no prompt")
            continue
        if not str(case.get("expectation") or "").strip():
            _fail(reasons, f"{where} states no expectation, so its `ok` is an opinion")
        if case.get("answer") is None:
            _fail(reasons, f"{where} records no answer — the case was generated and not run")
        if not isinstance(case.get("ok"), bool):
            _fail(reasons, f"{where} has no boolean `ok`")
        key = prompt.lower()
        if key in frozen_prompts:
            _fail(reasons, f"{where} repeats a frozen case; the ad-hoc set adds no coverage")
        if key in seen:
            _fail(reasons, f"{where} repeats another ad-hoc case")
        seen.add(key)
        ids.append(str(case.get("id") or f"#{i}"))

    failed = [c.get("id") for c in cases if c.get("ok") is False]
    if failed:
        # Not this validator's refusal. A single arm's ad-hoc failure is a fact
        # about that deployment; a failure on patched that passed on stock is a
        # regression, and that comparison is `check_no_regression`'s.
        print(f"  note: ad-hoc case(s) {failed} did not pass on this arm")
    else:
        print(f"  ad-hoc: {len(cases)} case(s), all passed")
    return ids


def check(content: Path, args: dict, reasons: list) -> tuple[bool, list[str]]:
    result = content / "items" / "result"
    if not result.is_dir():
        return _fail(reasons, "items/result/ is missing"), []
    adhoc_reasons: list = []
    ids = adhoc_ok(result, args, adhoc_reasons)
    reasons.extend(adhoc_reasons)
    # Not short-circuited: every rule runs, so one rerun clears the whole set.
    ok = all(
        [
            not adhoc_reasons,
            smoke_ok(result, args, reasons),
            needle_ok(result, args, reasons),
            probe_recorded(result, reasons),
            evals_ok(result, args, reasons),
        ]
    )
    return ok, ids


def main() -> int:
    args = zone.args()
    results = {}
    findings: dict = {}
    adhoc: dict[str, list[str]] = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        notes: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            # Captured and re-echoed. The needle line -- "3/3 gated depths
            # retrieved (floor 1); per-depth results are comparison material,
            # not a capability claim" -- explains a PASS and is exactly the kind
            # of sentence a zone loses, because a zone keeps no stdout.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    results[hid], adhoc[hid] = check(content, args, reasons)
            except Exception as exc:  # noqa: BLE001 — a crash is not a refusal
                results[hid] = False
                adhoc[hid] = []
                reasons.append(f"THIS VALIDATOR DID NOT RUN: {type(exc).__name__}: {exc}")
            sys.stdout.write(buffer.getvalue())
            notes = [ln.strip() for ln in buffer.getvalue().splitlines() if ln.strip()]
        findings[hid] = ([] if results[hid] else list(reasons),
                         notes + (list(reasons) if results[hid] else []))
        print(f"check_acceptance: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")

    # Both arms are produced by one task (M5.2) and so arrive in one phase.
    # Different ad-hoc cases per arm is not a comparison — and it is exactly the
    # shape a regression could hide behind, since the arm that failed a case
    # could simply have been asked a different one.
    if len(adhoc) > 1 and len({tuple(v) for v in adhoc.values()}) > 1:
        for hid in adhoc:
            results[hid] = False
            findings[hid] = (list(findings.get(hid, ([], []))[0]) +
                             ["the arms ran different ad-hoc case sets, so the per-run cases "
                              "no longer make the suite ungameable"],
                             list(findings.get(hid, ([], []))[1]))
            print(
                f"check_acceptance: {hid} FAIL\n  - the arms ran different ad-hoc case sets "
                f"({ {k: v for k, v in adhoc.items()} }). The per-run cases exist so the suite "
                "cannot be gamed; running a different set per arm gives that back."
            )

    _report(findings, results)
    zone.write_verdict(results)
    return 0


def _report(findings: dict, results: dict) -> None:
    """`workset_io.write_report`, and never a second implementation of it.

    m3 measured 16 of 21 validators persisting nothing; seven were this stage's.
    It matters most here because **stage 5 has never been reached** — every
    other stage has had refusals to learn from, and m5's first would otherwise
    arrive with the diagnostics off.

    `verdicts` is passed rather than inferred from `problems` being non-empty:
    this body keeps informational lines in the same list, which is the case the
    argument was added for.

    Wrapped so a failure to write the report cannot fail the validation — the
    report is evidence *about* a verdict, never the reason there is not one.
    """
    try:
        workset_io.write_report("check_acceptance", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print(f"check_acceptance: could not write the report: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
