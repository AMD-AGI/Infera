#!/usr/bin/env python3
"""`check_bench_result` — completeness, strong. Six rules over a bench.

Carried across from `../../profiling-demo/assets/check_aiperf_report.validator/`,
which had four of them and had been driven to a real cluster run. What is new
here is mission M2.2.1: the bench's record is a JSON document with a schema in
`assets/schemas/`, and the schema is checked from **both** sides — the producer
validates before it seals, this validates after.

**The failure this exists to catch is not a crash.** AIPerf exits 0 when its
schedule window turns out to contain nothing, and then every export is present,
every file is well-formed, and the whole thing describes zero requests. A shape
check passes that; a request count does not.

The rules:

1. **Shape.** The five exports and the three environment records are present and
   non-empty.
2. **Schema.** `profile_export_aiperf.json` validates against `bench_result`.
   This is the one that says the document is the format the rest of the flow was
   written against, rather than a format that happens to have the fields today's
   reader looks at.
3. **The load really ran.** The request count, read from the **export** and not
   from the summary, clears a floor.
4. **The summary is a rendering and not a second opinion.** `summary.json` is
   derived from the CSV; the CSV is derived from the same run as the JSON. When
   they disagree about how many requests there were, one of the four exports is
   from a different run and there is no way to tell which from inside.
5. **Errors, counted per request.** From `profile_export.jsonl.gz`, not from a
   summary field, because AIPerf's own error accounting has moved between
   releases and a missing key would read as zero errors — the one wrong answer
   that looks like success.
6. **The replay's own configuration is recorded**, and it names a window and a
   trace.

**It does not judge the numbers, and that is deliberate.** The same rules run
over both modes, and `profiling_mode_on` is expected to be several times slower
because its deployment has CUDA graphs off — measured on the sealed pair: 15.65
ms mean ITL against 124.98 ms, 8x apart, both correct. A validator that failed a
report for being slow would be enforcing a policy nobody wrote, in the one place
where slowness is the intent.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import schema as schema_lib  # noqa: E402 — the path insert above is what makes these importable
import zone  # noqa: E402

MIN_BYTES = 2

#: AIPerf writes four renderings of one run plus the per-request detail. All
#: five are required: the JSON is the record, and the other four are what a
#: person opens when the record says something surprising.
REQUIRED = {
    "result": [
        "summary.json",
        "profile_export_aiperf.csv",
        "profile_export_aiperf.json",
        "profile_export_console.txt",
        "profile_export.jsonl.gz",
    ],
    "env": ["load.json", "engine_argv.txt", "router_cmd.txt"],
}

#: The document the schema grades. Named once, because the producer's
#: `entry.sh` names the same path and a disagreement between the two would mean
#: the producer validated a file nobody else reads.
RECORD = ("result", "profile_export_aiperf.json")


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def shape_ok(content: Path, reasons: list) -> bool:
    """Rule 1."""
    ok = True
    for item, files in REQUIRED.items():
        base = content / "items" / item
        if not base.is_dir():
            ok = _fail(reasons, f"items/{item}/ is missing")
            continue
        for name in files:
            path = base / name
            if not path.is_file():
                ok = _fail(reasons, f"items/{item}/{name} is missing")
            elif path.stat().st_size < MIN_BYTES:
                ok = _fail(reasons, f"items/{item}/{name} is empty")
    for name in ("command", "watchout"):
        if not (content / "items" / name).is_file():
            ok = _fail(reasons, f"items/{name} is missing")
    if not (content / "items" / "logs").is_dir():
        ok = _fail(reasons, "items/logs/ is missing")
    return ok


def _read_record(content: Path, reasons: list):
    path = content / "items" / RECORD[0] / RECORD[1]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(reasons, f"items/{RECORD[0]}/{RECORD[1]} is not readable JSON: {exc}")
        return None


def schema_ok(export: dict, args: dict, reasons: list) -> bool:
    """Rule 2. The same file, resolved the same way, as the producer used.

    `args.schema` is a **name** and not a path (`assets/lib/schema.py`), so a
    producer and the validator that grades it cannot end up one directory apart
    and never notice.
    """
    name = args.get("schema") or "bench_result"
    try:
        schema_lib.validate(name, export)
    except schema_lib.SchemaError as exc:
        for line in str(exc).splitlines()[1:]:
            _fail(reasons, line.strip())
        return _fail(reasons, f"the export does not validate against the {name!r} schema")
    return True


def load_really_ran(export: dict, args: dict, reasons: list) -> bool:
    """Rule 3, and the reason this validator exists."""
    ok = True
    count = ((export.get("request_count") or {}).get("avg"))
    floor = float(args.get("min_requests", 1))
    if count is None:
        return _fail(reasons, "the export carries no request count")
    if count < floor:
        ok = _fail(
            reasons,
            f"the replay sent {count:.0f} request(s), want at least {floor:.0f}",
        )

    if export.get("was_cancelled"):
        ok = _fail(
            reasons,
            "the run was cancelled — the export is a partial window and its "
            "throughput describes however much of the schedule got sent",
        )

    # Streaming off is a legitimate AIPerf run and a useless stage-2 one: with no
    # stream there is no TTFT and no inter-token latency, and those are two of
    # the three numbers m5's stock arm has to reproduce.
    endpoint = (export.get("input_config") or {}).get("endpoint") or {}
    if args.get("require_streaming", True) and not endpoint.get("streaming"):
        ok = _fail(
            reasons,
            "the load ran with streaming off, so this export carries no TTFT and "
            "no inter-token latency — the two numbers stage 5 has to reproduce",
        )
    return ok


def summary_agrees(content: Path, export: dict, args: dict, reasons: list) -> bool:
    """Rules 4. `summary.json` is a rendering, and this is what makes that checkable."""
    path = content / "items" / "result" / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"summary.json is not readable JSON: {exc}")

    ok = True
    found = summary.get("metrics") or {}
    for name in args.get("require_metrics") or []:
        if name not in found:
            ok = _fail(reasons, f"summary.json has no metric {name!r}")

    missing = summary.get("missing")
    if missing:
        ok = _fail(
            reasons,
            f"summary.json reports {len(missing)} metric(s) it could not find in the "
            f"CSV: {missing}",
        )

    rendered = (found.get("request_count") or {}).get("avg")
    recorded = (export.get("request_count") or {}).get("avg")
    if rendered is None:
        ok = _fail(reasons, "summary.json carries no request count")
    elif recorded is not None and abs(float(rendered) - float(recorded)) > 0.5:
        ok = _fail(
            reasons,
            f"summary.json says {rendered:.0f} request(s) and the JSON export says "
            f"{recorded:.0f} — these are supposed to be two renderings of one run, so "
            f"one of the four exports came from a different one",
        )
    return ok


def errors_ok(content: Path, args: dict, reasons: list) -> bool:
    """Rule 5, read per request."""
    path = content / "items" / "result" / "profile_export.jsonl.gz"
    total = errored = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    errored += 1
                    continue
                if record.get("error") or (record.get("metadata") or {}).get("error"):
                    errored += 1
    except (OSError, EOFError, gzip.BadGzipFile) as exc:
        return _fail(reasons, f"profile_export.jsonl.gz is not readable: {exc}")

    if total == 0:
        return _fail(reasons, "profile_export.jsonl.gz holds no request records")
    rate = errored / total
    bar = float(args.get("max_error_rate", 0.05))
    if rate > bar:
        return _fail(reasons, f"{errored}/{total} requests errored ({rate:.1%}), bar is {bar:.1%}")
    reasons.append(f"(note) {total} request record(s), {errored} errored ({rate:.1%})")
    return True


def load_config_ok(content: Path, reasons: list) -> bool:
    """Rule 6."""
    path = content / "items" / "env" / "load.json"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"load.json is not readable JSON: {exc}")

    ok = True
    window = cfg.get("trace_window_ms")
    if not (isinstance(window, list) and len(window) == 2 and window[1] > window[0]):
        ok = _fail(reasons, f"load.json has no usable trace window: {window!r}")
    if not cfg.get("trace"):
        ok = _fail(reasons, "load.json does not name the trace that was replayed")
    return ok


def check(content: Path, args: dict, reasons: list) -> bool:
    if not shape_ok(content, reasons):
        # Every rule below reads one of the files rule 1 just found missing.
        # Reporting them all would be six restatements of one fact.
        return False
    export = _read_record(content, reasons)
    if export is None:
        return False
    return all(
        [
            schema_ok(export, args, reasons),
            load_really_ran(export, args, reasons),
            summary_agrees(content, export, args, reasons),
            errors_ok(content, args, reasons),
            load_config_ok(content, reasons),
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
            reasons.append("no staged content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_bench_result: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    zone.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
