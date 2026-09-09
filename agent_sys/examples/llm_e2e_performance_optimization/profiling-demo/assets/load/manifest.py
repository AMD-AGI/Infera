#!/usr/bin/env python3
"""Describe a directory of torch traces well enough to check the capture.

A file listing says eight files arrived. It does not say whether any of them
holds GPU work — and a capture whose window landed on an idle scheduler produces
eight perfectly well-formed traces with nothing in them, which is the failure
this manifest exists to make visible.

So each rank is opened, decompressed and parsed, and what is recorded is the
count of GPU kernel events and the span they cover. That costs about a second per
rank against files of this size and it is the only way the claim "the capture
covered the load" can be checked from the handoff alone.

**Parsed by streaming rather than with `json.load`.** A `with_stack` capture is
13x the bytes of the same window without one, measured, and materialising one of
those as Python objects is gigabytes for a file this then throws away. The
streaming reader also recovers the events before a truncation instead of
reporting the whole file as unreadable, which matters because torch writes the
trace from the profiler callback after `stop` has already returned.

    manifest.py <dir of *.trace.json.gz> <manifest.json>
"""

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import trace_stream  # noqa: E402 — the path insert above is what makes it importable

#: SGLang names each file `<epoch>-TP-<n>[-DP-<n>][-EP-<n>].trace.json.gz`. The
#: rank is read from the name rather than from the file's contents because the
#: contents do not carry it; a name that does not match is reported as rank None
#: rather than dropped, so a naming change surfaces instead of shrinking the set.
RANK = re.compile(r"-TP-(\d+)")


def describe(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    row = {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "rank": int(match.group(1)) if (match := RANK.search(path.name)) else None,
    }

    errors: list[str] = []
    try:
        counts = trace_stream.count_categories(path, errors=errors)
    except (OSError, EOFError, ValueError) as exc:
        row["readable"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    # A file whose array closed cleanly is readable. A file with a structural
    # fault is not, even when events were recovered from it -- every rule in
    # `check_trace_coverage` is a count, and a count off a clipped file is a
    # number nobody can act on. The recovered figures are still reported, so the
    # failure says how far the capture got.
    row["readable"] = not errors
    if errors:
        row["error"] = "; ".join(errors)
    row["events"] = counts["events"]
    row["gpu_kernels"] = counts["gpu_kernels"]
    row["gpu_kernel_us"] = counts["gpu_kernel_us"]
    # Non-zero exactly when the capture asked for `with_stack`. This is what
    # makes "the stack window really carries stacks" checkable from the handoff:
    # a `with_stack` request that silently did not take effect produces a trace
    # that is correct in every other respect.
    row["python_functions"] = counts["python_functions"]
    row["span_s"] = counts["span_s"]
    return row


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: manifest.py <trace dir> <manifest.json>", file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    files = sorted(src.glob("*.trace.json.gz"))
    if not files:
        print(f"manifest: no *.trace.json.gz under {src}", file=sys.stderr)
        return 1

    ranks = [describe(path) for path in files]
    manifest = {
        "ranks": ranks,
        "totals": {
            "files": len(ranks),
            "bytes": sum(r["bytes"] for r in ranks),
            "gpu_kernels": sum(r.get("gpu_kernels", 0) for r in ranks),
            "python_functions": sum(r.get("python_functions", 0) for r in ranks),
            "readable": sum(1 for r in ranks if r.get("readable")),
        },
    }
    dst.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    totals = manifest["totals"]
    print(
        f"manifest: {totals['files']} rank(s), {totals['readable']} readable, "
        f"{totals['gpu_kernels']} GPU kernel events, "
        f"{totals['python_functions']} python_function events, {totals['bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
