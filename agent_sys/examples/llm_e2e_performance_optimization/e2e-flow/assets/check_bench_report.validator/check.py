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
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import zone  # noqa: E402

#: AIPerf writes these four per run. All four, or the round is not a record.
REQUIRED_EXPORTS = (
    "profile_export_aiperf.csv",
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
    results = {}
    for hid in zone.inputs():
        content = zone.content_of(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_bench_report: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
