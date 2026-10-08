#!/usr/bin/env python3
"""Classify each sender ACK timeout in a run by what its retransmits found.

Within +-WINDOW samples of a timeout on a rail, compare the source's
retransmitted packets with the destination's resp_rx_dup_request:
  no-retx   the timer fired with nothing retransmitted;
  dup-only  every retransmit reached the destination as a duplicate, i.e. the
            originals had all arrived (lost <= SLACK);
  loss      some retransmits filled real gaps.
The two nodes are sampled one after the other, hence the window. A per-rail
tx - rx packet gap is printed for reference; it swings with traffic during the
sampling skew, so it is not used to classify.

Usage: ack_timeout_events.py RUN_DIR [--window N] [--slack PKTS]
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

COUNTERS = ("tx_rdma_ack_timeout", "tx_rdma_retx_pkts", "resp_rx_dup_request")


def open_timeseries(run: Path):
    path = run / "counters/timeseries.jsonl"
    return open(path) if path.exists() else gzip.open(f"{path}.gz", "rt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--slack", type=int, default=50)
    args = parser.parse_args()
    workers = json.loads((args.run / "live-config.json").read_text())["workers"]
    source = next(v["node"] for k, v in workers.items() if "prefill" in k)
    destination = next(v["node"] for k, v in workers.items() if "decode" in k)

    rows = []  # (time, {hca: (gap, ack_timeout, retx, dup)})
    with open_timeseries(args.run) as stream:
        for line in stream:
            sample = json.loads(line)
            src, dst = sample["nodes"][source], sample["nodes"][destination]
            rows.append((sample["captured_at"][11:19], {
                hca: (
                    src[hca]["sysfs"]["tx_rdma_ucast_pkts"]
                    - dst[hca]["sysfs"]["rx_rdma_ucast_pkts"],
                    src[hca]["sysfs"][COUNTERS[0]],
                    src[hca]["sysfs"][COUNTERS[1]],
                    dst[hca]["sysfs"][COUNTERS[2]],
                )
                for hca in src
            }))

    totals = {"no-retx": 0, "dup-only": 0, "loss": 0}
    print(f"{source} -> {destination}, {len(rows)} samples")
    print(f"{'time':8s} {'hca':8s} {'ackto':>5s} {'retx':>7s} {'dup':>7s} {'gap':>9s}  class")
    for i in range(1, len(rows)):
        time, now = rows[i]
        for hca, values in sorted(now.items()):
            ackto = values[1] - rows[i - 1][1][hca][1]
            if ackto <= 0:
                continue
            lo = rows[max(0, i - args.window)][1][hca]
            hi = rows[min(len(rows) - 1, i + args.window)][1][hca]
            gap, retx, dup = hi[0] - lo[0], hi[2] - lo[2], hi[3] - lo[3]
            kind = ("no-retx" if retx == 0
                    else "dup-only" if retx - dup <= args.slack else "loss")
            totals[kind] += ackto
            print(f"{time:8s} {hca:8s} {ackto:5d} {retx:7d} {dup:7d} {gap:9d}  {kind}")
    print(f"ACK timeouts by class: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
