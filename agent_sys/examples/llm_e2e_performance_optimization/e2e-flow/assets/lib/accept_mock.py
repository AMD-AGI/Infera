#!/usr/bin/env python3
"""Does a mock run of this package meet the deliverable? — reads artefacts, not the exit code.

**Why this exists at all.** `agent-sys run` on the mock exits **5**, forever, and
that is correct: the corpus's `integration_report` carries a *refused* verdict,
`check_no_regression` recomputes `REJECTED` from the numbers independently, and
so one handoff seals `invalid`. There is no passing fixture to swap in — making
one means changing numbers nobody chose, and the corpus's whole value is that
nobody chose them. Widening the bar was tried once and was the wrong answer
(`DELIVERY-NOTE-FROM-LEADER.md`).

So the run's exit code cannot carry the claim, and this script does instead —
CLAUDE.md principle 1, *read the artefact, not the exit code; every acceptance
claim names a file to open and a condition that fails.*

### The claim, in one sentence

> One run produces **15 handoffs** and **43 verdicts**, of which **42 are true**;
> the single false verdict is **`check_no_regression` on `integration_report`**;
> and that validator's **`validator_report.txt` carries exactly the four
> `PROBLEM:` lines below** — no more, no fewer, none different.

### Why it is not a mechanism for turning red into green

Nothing here inverts a verdict, relaxes a bar or touches a validator. The run
still exits 5 and the verdict is still false. This is a *stricter* statement
laid over the run, not a softer one, and it fails in three directions the run's
own exit code cannot distinguish:

- a **different** refusal from the same validator on the same kind — a real
  regression next week wearing the approved refusal's clothes — produces
  different `PROBLEM:` lines and fails here;
- a **second** refusal anywhere else fails the 42/1 split;
- the expected refusal **disappearing** fails it too, because 43 true is not the
  claim either. An artefact that stopped refusing is a finding, not a success.

That third property is the one the framework's own `xfail(strict=True)` gets
right and is the reason this is modelled on it rather than on a skip.

### What was considered and rejected: `cli/expectations.py`

The framework has this concept already — `EXPECTED_FAILURE` / `UNEXPECTED_SUCCESS`
/ `EXPECTATION_UNREACHED`, pytest's vocabulary, and it is *better* than what a
package would invent. Three measured reasons it is not used here, recorded in
`temp/bugs/2026-09-04-declaring-one-expected-failure-disables-completion-\
checking-for-the-whole-run.md`:

1. `_BY_PACKAGE` is a hardcoded `{"demo": DEMO}` keyed on directory name, so
   declaring anything means editing framework code;
2. `main.py:1409` — `gaps = ... if nothing_promised else []`. Declaring **one**
   promise switches off `_completion_gaps` for **all fifteen handoffs**;
3. the sealed verdict record has **no reason field** (`validator`, `result`,
   `strength`, `dimension`, `task_id`, `agent_id`, `environment`, `at`), so a
   promise can only be keyed on `(validator, kind, false)` — which accepts any
   refusal from that validator on that kind. That is the failure mode this file
   exists to make impossible.

The reason *is* recorded — in `validator_report.txt`, in structured form,
because the shared `workset_io.write_report` helper puts it there. This script
reads it. That is the whole trick, and it is only available because the report
file exists.

### Maintenance

`EXPECTED_*` below are the claim. When the graph legitimately changes — a
validator added, a kind added — these numbers change **deliberately, in a commit
that says why**. A failure here is not noise to be silenced by editing the
constant; it is the claim asking to be re-stated by whoever changed the graph.

Usage:

    python3 assets/lib/accept_mock.py                     # newest run under the default root
    python3 assets/lib/accept_mock.py --run <run-dir>
    python3 assets/lib/accept_mock.py --root ~/agent_sys_runroot

Exit 0 = the claim holds. Exit 1 = it does not, and every line of the difference
is printed. Exit 2 = the script could not tell (no run, unreadable artefacts),
which is deliberately **not** 1: *cannot judge* and *judged and refused* are
different facts, the same distinction the validators themselves draw.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

DEFAULT_ROOT = pathlib.Path("~/agent_sys_runroot").expanduser()

#: The graph's shape. See "Maintenance" above before editing.
EXPECTED_HANDOFFS = 15
EXPECTED_VERDICTS = 43
EXPECTED_TRUE = 42

#: The one refusal the corpus makes unavoidable.
EXPECTED_REFUSAL_VALIDATOR = "check_no_regression"
EXPECTED_REFUSAL_KIND = "integration_report"

#: **The reason, pinned.** Each entry is a name and the substrings that must all
#: appear in one `PROBLEM:` line. The match must be a bijection: every expected
#: problem matches exactly one reported one and vice versa.
#:
#: Substrings rather than whole lines, and the trade is stated rather than
#: hidden. Whole-line equality would also catch a wording change that alters the
#: *meaning* of a problem — but it breaks on any rewording at all, including m5
#: improving a sentence, and a check that cries wolf gets edited until it stops.
#: What is pinned is every number the problem is *about* (the producer's 35%/30%
#: against the validator's 5%/10%, the count of unusable metrics) plus the metric
#: names, so a different refusal cannot wear this one's clothes.
EXPECTED_PROBLEMS: dict[str, tuple[str, ...]] = {
    "producer chose its own throughput bar": (
        "max_throughput_regression=35%",
        "this validator's bar is 5%",
    ),
    "producer chose its own latency bar": (
        "max_latency_regression=30%",
        "this validator's bar is 10%",
    ),
    "four metrics regressed": (
        "the patch regressed:",
        "output_token_throughput_tps (avg)",
        "request_throughput_rps (avg)",
        "ttft_ms (p90)",
        "request_latency_ms (p90)",
    ),
    "six metrics are above their noise floor, so the run cannot judge": (
        "this run cannot judge the patch",
        "6 metric(s) have a noise floor above their bar",
    ),
}


class Undecidable(Exception):
    """The script could not reach a judgement. Exit 2, never 1."""


def newest_run(root: pathlib.Path) -> pathlib.Path:
    runs = root / "runs"
    if not runs.is_dir():
        raise Undecidable(f"no runs directory at {runs}")
    candidates = sorted((p for p in runs.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not candidates:
        raise Undecidable(f"{runs} contains no run")
    return candidates[-1]


def load_yaml(path: pathlib.Path):
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment, not logic
        raise Undecidable(f"pyyaml is not importable: {exc}") from exc
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - the message is the finding
        raise Undecidable(f"{path} is unreadable: {type(exc).__name__}: {exc}") from exc


def collect(run: pathlib.Path) -> tuple[int, list[dict], list[dict]]:
    """(handoff count, all verdicts, false verdicts). Each verdict gains a `kind`."""
    handoffs = run / "handoffs"
    if not handoffs.is_dir():
        raise Undecidable(f"no handoffs directory under {run}")
    ids = sorted(p for p in handoffs.iterdir() if p.is_dir())
    every: list[dict] = []
    for hid in ids:
        for version in sorted(hid.glob("v*")):
            validation, manifest = version / "validation.yaml", version / "manifest.yaml"
            if not validation.is_file():
                continue
            # `kind`, and `type` only as a fallback. Measured on
            # `20260904T114914-0a0cdd`: a version manifest carries
            # `digest, algorithm, kind, producer, created_at`. The first draft
            # read `type` alone — the spelling the *spec* uses — and every
            # handoff came back `?`, which the claim then reported as the
            # refusal being on the wrong kind. A near-miss worth the comment:
            # the check failed loudly and named a real field, so it read like a
            # finding about the run rather than a defect in the reader.
            kind = "?"
            if manifest.is_file():
                loaded = load_yaml(manifest) or {}
                kind = loaded.get("kind") or loaded.get("type") or "?"
            for verdict in (load_yaml(validation) or {}).get("verdicts") or []:
                every.append({**verdict, "kind": kind, "version": version})
    return len(ids), every, [v for v in every if not v.get("result")]


def report_for(verdict: dict, run: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    """The refusing validator's own report, and its `PROBLEM:` lines.

    Located via the verdict's recorded `environment.zone` rather than by
    searching, because the zone is the one link between a sealed verdict and the
    directory the body wrote in — and a search would happily find *another*
    validator's report and grade against it.
    """
    zone = ((verdict.get("environment") or {}).get("zone")) or ""
    if not zone:
        raise Undecidable(
            f"the {verdict.get('validator')} verdict records no environment.zone, "
            f"so its report cannot be located from the verdict alone"
        )
    path = pathlib.Path(zone) / "validator_report.txt"
    if not path.is_file():
        raise Undecidable(
            f"no validator_report.txt at {path} — the validator recorded a verdict "
            f"without writing a report, so its reason is not on disk to be pinned"
        )
    problems = [
        line.strip()[len("PROBLEM:") :].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("PROBLEM:")
    ]
    return path, problems


def match_problems(reported: list[str]) -> list[str]:
    """Bijection between what was expected and what was reported. Findings, or []."""
    findings: list[str] = []
    unmatched = list(range(len(reported)))
    for name, needles in EXPECTED_PROBLEMS.items():
        hits = [i for i in unmatched if all(n in reported[i] for n in needles)]
        if not hits:
            missing = [n for n in needles if not any(n in reported[i] for i in unmatched)]
            findings.append(
                f"expected problem NOT reported: {name}\n"
                f"      no PROBLEM line contains all of {list(needles)}\n"
                f"      (absent from every remaining line: {missing})"
            )
            continue
        if len(hits) > 1:
            findings.append(
                f"expected problem matched {len(hits)} reported lines: {name} — "
                f"the fingerprint no longer discriminates and must be narrowed"
            )
        unmatched.remove(hits[0])
    for i in unmatched:
        findings.append(
            f"UNEXPECTED problem reported, matching no entry in EXPECTED_PROBLEMS:\n"
            f"      {reported[i][:300]}"
        )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", help="a run directory; default is the newest under --root")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help=f"run root (default {DEFAULT_ROOT})")
    args = parser.parse_args()

    try:
        run = pathlib.Path(args.run) if args.run else newest_run(pathlib.Path(args.root).expanduser())
        if not run.is_dir():
            raise Undecidable(f"{run} is not a directory")
        handoffs, every, refusals = collect(run)
    except Undecidable as exc:
        print(f"CANNOT JUDGE: {exc}", file=sys.stderr)
        print("  This is exit 2, not 1: no claim was tested.", file=sys.stderr)
        return 2

    findings: list[str] = []
    if handoffs != EXPECTED_HANDOFFS:
        findings.append(f"handoffs: {handoffs}, expected {EXPECTED_HANDOFFS}")
    if len(every) != EXPECTED_VERDICTS:
        findings.append(f"verdicts: {len(every)}, expected {EXPECTED_VERDICTS}")
    true_count = len(every) - len(refusals)
    if true_count != EXPECTED_TRUE:
        findings.append(f"true verdicts: {true_count}, expected {EXPECTED_TRUE}")

    if len(refusals) != 1:
        findings.append(
            f"refusals: {len(refusals)}, expected exactly 1 — "
            + (
                "; ".join(f"{r.get('validator')} on {r.get('kind')}" for r in refusals)
                if refusals
                else "none, so the fixture stopped refusing and that is a finding, not a pass"
            )
        )
    else:
        refusal = refusals[0]
        named = (refusal.get("validator"), refusal.get("kind"))
        if named != (EXPECTED_REFUSAL_VALIDATOR, EXPECTED_REFUSAL_KIND):
            findings.append(
                f"the refusal is {named[0]} on {named[1]}, expected "
                f"{EXPECTED_REFUSAL_VALIDATOR} on {EXPECTED_REFUSAL_KIND}"
            )
        else:
            try:
                path, problems = report_for(refusal, run)
            except Undecidable as exc:
                print(f"CANNOT JUDGE: {exc}", file=sys.stderr)
                return 2
            print(f"report: {path}", file=sys.stderr)
            findings.extend(match_problems(problems))

    print(f"run:    {run}", file=sys.stderr)
    if findings:
        print(
            f"\nMOCK E2E NOT ACCEPTED — {len(findings)} difference(s) from the claim:",
            file=sys.stderr,
        )
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        print(
            "\nEach line above is the claim disagreeing with the artefact. Fix the\n"
            "artefact, or re-state the claim in a commit that says why it changed.",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nMOCK E2E ACCEPTED — {handoffs} handoffs, {len(every)} verdicts, "
        f"{true_count} true; the one refusal is {EXPECTED_REFUSAL_VALIDATOR} on "
        f"{EXPECTED_REFUSAL_KIND} and its report gives exactly the "
        f"{len(EXPECTED_PROBLEMS)} expected reasons.\n"
        "The run's own exit code is 5 and that is correct; this is the deliverable's green.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
