#!/usr/bin/env python3
"""Per-rail packet conservation between the Prefill (source) and Decode
(destination) HCAs of a run, from counters/before.json and after.json.

  mac     source frames_tx_all - destination frames_rx_all: lost on the wire.
          Includes a few thousand frames per run sent to other hosts.
  data    source tx_rdma_ucast_pkts - destination rx_rdma_ucast_pkts.
  inside  destination frames_rx_all - rx_rdma_ucast_pkts: frames that reached
          the destination MAC but not its RDMA engine (non-RDMA frames or
          dropped inside the NIC).
  ack     destination tx_rdma_ucast_pkts - source rx_rdma_ucast_pkts.
  ackto/retx/nak  source tx_rdma_ack_timeout, tx_rdma_retx_pkts, req_rx_pkt_seq_err.
  dup     destination resp_rx_dup_request.
  pause   PFC pause frames sent by the destination.

Usage: rail_conservation.py RUN_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    workers = json.loads((run / "live-config.json").read_text())["workers"]
    source = next(v["node"] for k, v in workers.items() if "prefill" in k)
    destination = next(v["node"] for k, v in workers.items() if "decode" in k)
    before, after = (
        json.loads((run / f"counters/{name}.json").read_text())["nodes"]
        for name in ("before", "after")
    )

    def delta(node: str, hca: str, counter: str) -> int:
        def read(snapshot: dict) -> int:
            entry = snapshot[node][hca]
            if counter in entry["sysfs"]:
                return entry["sysfs"][counter]
            return sum(e.get(counter, 0) for e in entry["ethtool"].values())
        return read(after) - read(before)

    columns = ("mac", "data", "inside", "ack", "ackto", "retx", "nak", "dup", "pause")
    totals = dict.fromkeys(columns, 0)
    print(f"{source} -> {destination}")
    print(f"{'hca':8s}" + "".join(f"{c:>12s}" for c in columns))
    for hca in sorted(after[destination]):
        s = lambda counter: delta(source, hca, counter)
        d = lambda counter: delta(destination, hca, counter)
        row = {
            "mac": s("frames_tx_all") - d("frames_rx_all"),
            "data": s("tx_rdma_ucast_pkts") - d("rx_rdma_ucast_pkts"),
            "inside": d("frames_rx_all") - d("rx_rdma_ucast_pkts"),
            "ack": d("tx_rdma_ucast_pkts") - s("rx_rdma_ucast_pkts"),
            "ackto": s("tx_rdma_ack_timeout"),
            "retx": s("tx_rdma_retx_pkts"),
            "nak": s("req_rx_pkt_seq_err"),
            "dup": d("resp_rx_dup_request"),
            "pause": d("frames_tx_pause") + d("frames_tx_pripause"),
        }
        for column in columns:
            totals[column] += row[column]
        print(f"{hca:8s}" + "".join(f"{row[c]:12d}" for c in columns))
    print(f"{'total':8s}" + "".join(f"{totals[c]:12d}" for c in columns))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
