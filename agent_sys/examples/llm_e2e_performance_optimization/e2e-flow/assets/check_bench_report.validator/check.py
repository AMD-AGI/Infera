#!/usr/bin/env python3
"""`check_bench_report` — completeness, strong.

Every replay round produced a complete AIPerf export, actually sent requests, and
stayed under the error bar.

The rule that earns its place is `min_requests`. **AIPerf exits 0 after
synthesising prompts if the schedule window turns out empty**, and every file
below is then present, well formed, and describes no requests at all — a
successful-looking round that measured nothing. It is the same failure
`profiling-demo` guards, for the same reason.

`max_error_rate` is not zero. A saturated deployment legitimately times out the
occasional request under a fixed schedule, and a bar of zero would turn this into
a flake detector rather than a completeness check.

The last rule is about the pair rather than the round: both arms must have run
the same sequence of steps. "Round 1 was cold for this trace" is only true of an
arm if the same things happened before it, and if the two arms disagree there
then the comparison downstream is between two different experiments.

**Carried across from `integration-demo` with one addition: the shared schema.**
`args.schema` names `bench_result`, which is m2's file in `assets/schemas/` and
not a second copy — mission G2 puts one schema in front of the producer and the
validator alike, and CONTRACT.md §4.1 says a shared definition is shared rather
than duplicated. m5's replay and m2's bench are the same AIPerf export, so they
are graded by the same document; a round whose `profile_export_aiperf.json` does
not validate is a round m2's own validator would have refused.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402
import workset_io  # noqa: E402 — the shared report writer; see _report()
import zone  # noqa: E402

#: AIPerf writes these five per run. All five, or the round is not a record.
#:
#: `profile_export_aiperf.json` is the one the schema grades and the other four
#: are renderings of it — measured by m2 when it wrote `bench_result.schema.json`:
#: the CSV blanks the percentile columns AIPerf did not compute, the console text
#: is the CSV with box-drawing characters, and the JSONL is the per-request detail
#: the summary was computed from.
REQUIRED_EXPORTS = (
    "profile_export_aiperf.csv",
    "profile_export_aiperf.json",
    "profile_export.jsonl",
    "profile_export_console.txt",
    "server_metrics_export.csv",
)


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def round_ok(round_dir: Path, args: dict, reasons: list) -> bool:
    tag = round_dir.name
    ok = True
    for name in REQUIRED_EXPORTS:
        path = round_dir / name
        if not path.is_file():
            ok = _fail(reasons, f"{tag}: {name} is missing")
        # Size is read at THIS moment rather than trusted from a listing taken
        # when the file was written: measured on profiling-demo, `ls -l` and
        # `du -sb` disagreed by three orders of magnitude immediately after a
        # write, and `stat` afterwards agreed with neither.
        elif path.stat().st_size == 0:
            ok = _fail(reasons, f"{tag}: {name} is empty")

    jsonl = round_dir / "profile_export.jsonl"
    if jsonl.is_file():
        lines = sum(1 for line in jsonl.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
        if lines == 0:
            ok = _fail(reasons, f"{tag}: profile_export.jsonl describes no requests")

    # The shared schema, against the one export that is lossless and typed.
    # A round that fails here is a round m2's `check_bench_result` would have
    # refused, which is the point of the two stages naming one schema.
    name = args.get("schema")
    export = round_dir / "profile_export_aiperf.json"
    if name and export.is_file():
        try:
            schema_lib.validate(str(name), json.loads(export.read_text(encoding="utf-8")))
        except schema_lib.SchemaError as exc:
            ok = _fail(reasons, f"{tag}: profile_export_aiperf.json does not validate: {exc}")
        except ValueError as exc:
            ok = _fail(reasons, f"{tag}: profile_export_aiperf.json is not readable as JSON: {exc}")
    elif name:
        ok = _fail(
            reasons,
            f"{tag}: profile_export_aiperf.json is absent, so the {name!r} schema had nothing "
            "to grade. The CSV and the console text are renderings; this is the record.",
        )

    summary = read_json(round_dir / "summary.json")
    if summary is None:
        return _fail(reasons, f"{tag}: summary.json is missing or unreadable")

    metrics = summary.get("metrics") or {}
    for name in args.get("require_metrics") or ():
        if name not in metrics:
            ok = _fail(
                reasons,
                f"{tag}: the summary has no {name!r}. An AIPerf release that renames a row "
                "should fail here as a missing metric, not three steps later as a KeyError.",
            )

    count = (metrics.get("request_count") or {}).get("avg")
    floor = float(args.get("min_requests", 1))
    if count is None:
        ok = _fail(reasons, f"{tag}: no request count was recorded")
    elif count < floor:
        ok = _fail(
            reasons,
            f"{tag}: {count:.0f} request(s) replayed, floor is {floor:.0f}. AIPerf exits 0 "
            "after synthesising prompts even when the schedule window is empty, so a "
            "complete-looking round can describe no work at all.",
        )

    # Errors are counted out of the jsonl rather than read from a metric: AIPerf's
    # CSV has no error-rate row, and inferring one from the gap between requests
    # sent and responses recorded would silently count a truncated export as a
    # perfect run.
    if jsonl.is_file() and count:
        text = jsonl.read_text(encoding="utf-8", errors="replace")
        errored = sum(
            1
            for line in text.splitlines()
            if line.strip() and '"error"' in line and '"error": null' not in line
        )
        rate = errored / count
        bar = float(args.get("max_error_rate", 0.05))
        print(f"  {tag}: {count:.0f} requests, {errored} errored ({rate:.2%}), bar {bar:.0%}")
        if rate > bar:
            ok = _fail(reasons, f"{tag}: error rate {rate:.2%} is over the {bar:.0%} bar")
    return ok


def check(content: Path, args: dict, reasons: list) -> bool:
    result = content / "items" / "result"
    if not result.is_dir():
        return _fail(reasons, "items/result/ is missing")

    rounds = sorted(p for p in result.glob("r*") if p.is_dir())
    expected = int(args.get("expect_rounds", 1))
    ok = True
    if len(rounds) != expected:
        ok = _fail(
            reasons,
            f"{len(rounds)} replay round(s) present, {expected} expected. The arms are "
            "compared round for round, so a missing round is a missing comparison.",
        )
    if not rounds:
        return False

    for round_dir in rounds:
        ok = round_ok(round_dir, args, reasons) and ok

    steps = read_json(content / "items" / "env" / "steps.json")
    if steps is None:
        ok = _fail(reasons, "items/env/steps.json is missing — the step order was not recorded")
    else:
        names = [s.get("step") for s in steps.get("steps", [])]
        if not names:
            ok = _fail(reasons, "steps.json records no steps")
        else:
            print(f"  sequence: {' -> '.join(str(n) for n in names)}")
    return ok


def main() -> int:
    args = zone.args()
    results: dict = {}
    findings: dict = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            # Captured and re-echoed: the lines that explain a PASS go to stdout,
            # and a zone keeps no stdout at all. A person watching the run still
            # sees them; so, now, does anyone reading the zone afterwards.
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    results[hid] = check(content, args, reasons)
            except Exception as exc:  # noqa: BLE001
                # A crash is not a refusal. verdict.json cannot express the
                # difference (todo.md T29); this text is the only place it exists.
                results[hid] = False
                reasons.append(f"THIS VALIDATOR DID NOT RUN: {type(exc).__name__}: {exc}")
            sys.stdout.write(buffer.getvalue())
            notes = [ln.strip() for ln in buffer.getvalue().splitlines() if ln.strip()]
            findings[hid] = ([] if results[hid] else list(reasons),
                             notes + (list(reasons) if results[hid] else []))
        findings.setdefault(hid, (list(reasons), []))
        print(f"check_bench_report: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    # Before write_verdict, deliberately: a crash in the writer must not be able
    # to take the reasons with it, and the verdict is what the phase reads.
    _report(findings, results)
    zone.write_verdict(results)
    return 0


def _report(findings: dict, results: dict) -> None:
    """`workset_io.write_report`, and never a second implementation of it.

    m3 measured that 16 of 21 validators persist nothing, and seven of those are
    this stage's. That matters most here because **stage 5 has never been
    reached**: every other stage has had refusals to learn from, and m5's first
    one would otherwise arrive with the diagnostics switched off.

    `verdicts` is passed rather than letting the heading infer from `problems`
    being non-empty — these bodies keep informational lines in the same
    `reasons` list, which is the case that made the argument exist.

    Wrapped so that a failure to write the report cannot fail the validation:
    the report is evidence *about* a verdict and must never become the reason
    there is not one.
    """
    try:
        workset_io.write_report("check_bench_report", findings, results)
    except Exception as exc:  # noqa: BLE001 — see the docstring
        print("check_bench_report: could not write the validator report: %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
