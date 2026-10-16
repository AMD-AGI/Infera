"""Write a synthetic Mooncake trace for the profiling demo.

The profiling lines replay a Mooncake trace with AIPerf at its recorded pacing:
one JSON object per request with `timestamp` (ms), `input_length`,
`output_length` and `hash_ids` (one id per `block_size` tokens of prompt). This
writes a steady load with no prefix reuse, so every run of the demo replays the
same requests:

    python3 make_trace.py --out /tmp/$USER/trace.jsonl
"""

import argparse
import json
import math


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--requests", type=int, default=240)
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second")
    ap.add_argument("--input-length", type=int, default=1024)
    ap.add_argument("--output-length", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=512, help="must match --var block_size")
    a = ap.parse_args()
    blocks = math.ceil(a.input_length / a.block_size)
    with open(a.out, "w", encoding="utf-8") as f:
        for i in range(a.requests):
            row = {
                "timestamp": int(i * 1000 / a.rate),
                "input_length": a.input_length,
                "output_length": a.output_length,
                "hash_ids": list(range(i * blocks, (i + 1) * blocks)),
            }
            f.write(json.dumps(row) + "\n")
    print(f"{a.requests} requests over {a.requests / a.rate:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
