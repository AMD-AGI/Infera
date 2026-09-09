#!/usr/bin/env python3
"""`check_kernel_table` — usability, strong.

Four rules, each total over the document:

1. The CSV header is one of the two shapes Magpie writes — six base columns, or
   those six plus the thirteen `KernelSourceInfo` columns.
2. There are at least `min_kernel_rows` data rows.
3. The `% Total` column sums into `[pct_total_min, pct_total_max]`.
4. Every `launcher` block that is present is one `identify` can act on.

Rule 3 is a range and not an equality because Magpie rounds each percentage to
two decimals; over ~143 rows the sum drifts. The sample profile sums to 99.94.
The floor is loose so that a table truncated by `--top-k` still passes.

Rule 4 checks presence-conditional shape rather than coverage. Whether *enough*
kernels carry a call site is a question about the capture, and it belongs to the
producing package's own validator, which knows whether a `with_stack` window was
asked for. What belongs here is that a block which exists is usable: a resolved
path with no root to resolve it against would send the next stage nowhere, and it
would do so looking complete.

**The document is located rather than assumed.** This kind has two producers —
this package's `seed_table` mock and `profiling-demo`'s `kernel_scan` — whose
`content_type`s differ, so the document sits at `items/text.json` or at
`items/result/text.json`. `assets/lib/kernel_table.py` is the only place that
knows there are two.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))

import csv_io  # noqa: E402
import kernel_table  # noqa: E402
import store  # noqa: E402


def check(content: Path, args: dict) -> tuple[bool, str]:
    # Located through `kernel_table` rather than by path: this kind has two
    # producers with different `content_type`s, so the document sits at
    # `items/text.json` or at `items/result/text.json` depending on which one
    # wrote it. That file is the only place which knows there are two.
    try:
        data = kernel_table.read(content)
    except kernel_table.LayoutError as error:
        return False, str(error)

    columns = data.get("columns") or []
    missing = [c for c in csv_io.BASE_COLUMNS if c not in columns]
    if missing:
        return False, f"header is missing {missing}"

    kernels = data["kernels"]
    minimum = int(args.get("min_kernel_rows") or 20)
    if len(kernels) < minimum:
        return False, f"{len(kernels)} rows, need at least {minimum}"

    # The raw CSV has to be there too: the kind declares it, and a downstream
    # consumer that wants a column this package did not parse reads it from
    # there rather than re-running the profile.
    if kernel_table.csv_dir(content) is None:
        return False, (
            "no gap-analysis CSV: looked for "
            + " and ".join(str(p) for p in kernel_table.CSV_DIRS)
        )

    total = sum(float(k.get("pct_total") or 0.0) for k in kernels)
    low = float(args.get("pct_total_min") or 50.0)
    high = float(args.get("pct_total_max") or 100.5)
    if not (low <= total <= high):
        return False, f"% Total sums to {total:.3f}, outside [{low}, {high}]"

    unnamed = [k for k in kernels if not (k.get("name") or "").strip()]
    if unnamed:
        return False, f"{len(unnamed)} row(s) carry no kernel name"

    # A `launcher` block is optional -- the mock producer never has one -- but a
    # malformed one is not. It is read as evidence by `identify`, and a block
    # missing its root would send the next stage at a path with no repository to
    # resolve it against.
    for kernel in kernels:
        launcher = kernel.get("launcher")
        if not isinstance(launcher, dict) or not launcher:
            continue
        name = (kernel.get("name") or "?")[:40]
        if not (launcher.get("source_file") or "").strip():
            return False, f"{name}: carries a launcher block with no source_file"
        if launcher["source_file"].startswith("/"):
            return False, (
                f"{name}: launcher source_file {launcher['source_file']!r} is absolute. "
                f"The seal refuses absolute paths and a container path is not resolvable "
                f"against a checkout; the producer splits the root off"
            )
        if not (launcher.get("container_root") or "").strip():
            return False, f"{name}: launcher names no container_root to resolve against"

    coverage = kernel_table.launcher_coverage(data)
    return True, (
        f"{len(kernels)} rows, % Total sums to {total:.2f}, "
        f"{coverage['rows_with_frames']} row(s) carry a launcher frame"
    )


def main() -> int:
    args = store.args()
    results = {}
    for hid in store.inputs():
        content = store.staged_content(hid) or store.content_dir(hid)
        if content is None:
            results[hid] = False
            print(f"check_kernel_table: {hid}: no published content")
            continue
        ok, why = check(content, args)
        results[hid] = ok
        print(f"check_kernel_table: {hid}: {'PASS' if ok else 'FAIL'} — {why}")
    store.write_verdict(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
