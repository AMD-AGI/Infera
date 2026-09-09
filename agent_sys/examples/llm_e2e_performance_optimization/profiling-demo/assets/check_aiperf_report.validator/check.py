#!/usr/bin/env python3
"""`check_aiperf_report` — completeness, strong.

Four rules over a replay report. Reads only files.

The failure this exists to catch is not a crash. AIPerf exits 0 when its schedule
window turns out to contain nothing, and then every export is present, every file
is well-formed, and the whole thing describes zero requests. A shape check passes
that; a request count does not.

**It does not judge the numbers.** The same rules run over both rounds, and the
profiled round is expected to be several times slower because its deployment has
CUDA graphs off. A validator that failed a report for being slow would be
enforcing a policy nobody wrote, in the one place where slowness is the intent.
"""

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import store  # noqa: E402 — the path insert above is what makes it importable

MIN_BYTES = 2

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


def _fail(reasons: list, message: str) -> bool:
    reasons.append(message)
    return False


def shape_ok(content: Path, reasons: list) -> bool:
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
        path = content / "items" / name
        if not path.is_file():
            ok = _fail(reasons, f"items/{name} is missing")
    if not (content / "items" / "logs").is_dir():
        ok = _fail(reasons, "items/logs/ is missing")
    return ok


def metrics_ok(content: Path, args: dict, reasons: list) -> bool:
    """The metrics summarise.py was asked to find are there, and requests happened."""
    path = content / "items" / "result" / "summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(reasons, f"summary.json is not readable JSON: {exc}")

    found = summary.get("metrics") or {}
    ok = True
    for name in args.get("require_metrics") or []:
        if name not in found:
            ok = _fail(reasons, f"summary.json has no metric {name!r}")

    count = (found.get("request_count") or {}).get("avg")
    floor = float(args.get("min_requests", 1))
    if count is None:
        ok = _fail(reasons, "summary.json carries no request count")
    elif count < floor:
        # The whole point of this validator. AIPerf exits 0 on an empty window.
        ok = _fail(reasons, f"the replay sent {count:.0f} request(s), want at least {floor:.0f}")
    return ok


def errors_ok(content: Path, args: dict, reasons: list) -> bool:
    """Errored requests as a share of the total, read from the per-request records.

    Read from `profile_export.jsonl.gz` and not from a summary field, because
    AIPerf's own error accounting has moved between releases and a missing key
    would read as zero errors — which is the one wrong answer that looks like
    success.
    """
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
    return True


def load_ok(content: Path, reasons: list) -> bool:
    """The replay's own configuration is recorded, and it replayed a window."""
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
    return all(
        [
            shape_ok(content, reasons),
            metrics_ok(content, args, reasons),
            errors_ok(content, args, reasons),
            load_ok(content, reasons),
        ]
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        reasons: list = []
        if content is None:
            results[hid] = False
            reasons.append("no published content for this handoff")
        else:
            results[hid] = check(content, args, reasons)
        print(f"check_aiperf_report: {hid} {'PASS' if results[hid] else 'FAIL'}")
        for reason in reasons:
            print(f"  - {reason}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
