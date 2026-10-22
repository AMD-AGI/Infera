#!/usr/bin/env python3
"""Collect the TP8/EP8/DPA sweep into a CSV, gating every point.

Reuses verify_point_yihou.check(), so a row only appears as `pass` if it satisfies the same
assertions: complete, exact useful_output_tokens, topology (ep=8, tp=8, dp=8, dpa=True),
local_batch == C/8, realized_accept_length invariant, driver exit_code == 0.

Exit code is non-zero if any collected point fails, so this is a gate, not a formatter.
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_point_yihou import check  # noqa: E402

COLUMNS = [
    "concurrency",
    "ep_size",
    "local_batch_size",
    "verdict",
    "tpot_ms",
    "output_tokens_per_second",
    "output_tokens_per_second_per_gpu",
    "decode_seconds",
    "load_pool_capture_seconds",
    "verify_iterations",
    "realized_accept_length",
    "reserved_tokens",
    "max_memory_allocated_bytes",
    "point",
    "problems",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("iterations_dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--prefix",
        default="tp8_ep8_dpa_on_c",
        help="only collect iteration dirs starting with this prefix "
        "(tp8_ep8_dpa_on_c for the EP8 sweep, tp8_noep_dpa_on_c for the EP-off sweep)",
    )
    parser.add_argument("--expect-ep", type=int, default=8, help="expected ep_size (8 or 1)")
    args = parser.parse_args()

    rows, failed = [], False
    for point in sorted(args.iterations_dir.iterdir()):
        if not point.is_dir() or not point.name.startswith(args.prefix):
            continue
        verdict, summary = check(point, expect_ep=args.expect_ep)
        failed |= verdict != "pass"
        # A point still running (or dead before writing its result) yields a sparse summary;
        # fill every column so the CSV shape and the sort key are stable.
        rows.append(
            {
                **{k: summary.get(k) for k in COLUMNS},
                "verdict": verdict,
                "point": point.name,
                "problems": "; ".join(summary.get("problems", [])),
            }
        )

    rows.sort(key=lambda r: (r["concurrency"] is None, r["concurrency"] or 0))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"C={str(row['concurrency']):<5} ep={str(row['ep_size']):<3} "
            f"local={str(row['local_batch_size']):<4} {row['verdict']:<7} "
            f"TPOT={row['tpot_ms']} tok/s={row['output_tokens_per_second']} {row['problems']}"
        )
    print(f"\n{len(rows)} points -> {args.output}")
    return 1 if failed or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
