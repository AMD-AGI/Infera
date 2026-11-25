#!/usr/bin/env python3
"""Collect a sweep into a CSV, gating every point.

Port of the GLM-5.2 collect_sweep.py. Reuses verify_point.check() (the GLM-5.2 copy imported the
pre-rename module name verify_point_yihou, handoff trap #17; fixed here), so a row is `pass` only
if it satisfies the same assertions. Exit code is non-zero if any collected point fails or none
was collected: this is a gate, not a formatter.
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_point import add_expectation_args, check, expectations  # noqa: E402

COLUMNS = [
    "concurrency", "ep_size", "local_batch_size", "verdict", "tpot_ms", "output_tokens_per_second",
    "output_tokens_per_second_per_gpu", "decode_seconds", "load_pool_capture_seconds", "verify_iterations",
    "realized_accept_length", "reserved_tokens", "max_memory_allocated_bytes", "point", "problems",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("iterations_dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--prefix", default="tp4_ep4_dpa_on_c",
                        help="only collect iteration dirs starting with this prefix "
                             "(tp4_ep4_dpa_on_c for the EP4 sweep, tp4_noep_dpa_on_c for the EP-off sweep)")
    add_expectation_args(parser)
    args = parser.parse_args()

    rows, failed = [], False
    for point in sorted(args.iterations_dir.iterdir()):
        if not point.is_dir() or not point.name.startswith(args.prefix):
            continue
        verdict, summary = check(point, **expectations(args))
        failed |= verdict != "pass"
        rows.append({**{k: summary.get(k) for k in COLUMNS}, "verdict": verdict, "point": point.name,
                     "problems": "; ".join(summary.get("problems", []))})

    rows.sort(key=lambda r: (r["concurrency"] is None, r["concurrency"] or 0))
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"C={str(row['concurrency']):<5} ep={str(row['ep_size']):<3} local={str(row['local_batch_size']):<4} "
              f"{row['verdict']:<7} TPOT={row['tpot_ms']} tok/s={row['output_tokens_per_second']} {row['problems']}")
    print(f"\n{len(rows)} points -> {args.output}")
    return 1 if failed or not rows else 0


if __name__ == "__main__":
    sys.exit(main())
