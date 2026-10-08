#!/usr/bin/env python3
"""Summarize the receiver-drop mechanism from two full HCA counter snapshots.

nic_rx_loss is MAC-received unicast frames minus RDMA packets accepted by the
RDMA engine on the same port; on this fabric the MAC counts match the peer's
transmitted frames, so the difference is loss inside the receiving NIC.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSFS = {
    "rdma_rx": "rx_rdma_ucast_pkts",
    "rdma_tx": "tx_rdma_ucast_pkts",
    "rdma_rx_bytes": "rx_rdma_ucast_bytes",
    "oos": "resp_rx_outouf_seq",
    "seq_nak_tx": "resp_tx_pkt_seq_err",
    "dup_request": "resp_rx_dup_request",
    "seq_nak_rx": "req_rx_pkt_seq_err",
    "retx_pkts": "tx_rdma_retx_pkts",
    "ack_timeout": "tx_rdma_ack_timeout",
    "retry_excd": "req_tx_retry_excd_err",
    "ecn_rx": "rx_rdma_ecn_pkts",
    "cnp_tx": "tx_rdma_cnp_pkts",
    "cnp_rx": "rx_rdma_cnp_pkts",
    "req_cqe_err": "req_rx_cqe_err",
}
ETHTOOL = {
    "mac_rx_uc": "frames_rx_unicast",
    "mac_tx_uc": "frames_tx_unicast",
    "pause_tx": "frames_tx_pripause",
    "pause_rx": "frames_rx_pripause",
    "pause_tx_us": "tx_pripause_3_1us_count",
    "mac_rx_dropped": "frames_rx_dropped",
    "hw_rx_dropped": "hw_rx_dropped",
}
FIELDS = tuple(SYSFS) + tuple(ETHTOOL) + ("nic_rx_loss",)


def _delta(before: dict, after: dict, source: str, name: str):
    if source == "sysfs":
        old = before.get("sysfs", {}).get(name)
        new = after.get("sysfs", {}).get(name)
    else:
        old = new = None
        for netdev, stats in before.get("ethtool", {}).items():
            old = stats.get(name)
            new = after.get("ethtool", {}).get(netdev, {}).get(name)
            break
    if isinstance(old, int) and isinstance(new, int):
        return new - old
    return None


def per_hca(before: dict, after: dict) -> dict:
    result = {}
    for node, hcas in before.get("nodes", {}).items():
        for hca, record in hcas.items():
            later = after.get("nodes", {}).get(node, {}).get(hca)
            if later is None:
                continue
            row = {key: _delta(record, later, "sysfs", name) for key, name in SYSFS.items()}
            row.update(
                {key: _delta(record, later, "ethtool", name) for key, name in ETHTOOL.items()}
            )
            if row["mac_rx_uc"] is not None and row["rdma_rx"] is not None:
                row["nic_rx_loss"] = row["mac_rx_uc"] - row["rdma_rx"]
            else:
                row["nic_rx_loss"] = None
            result.setdefault(node, {})[hca] = row
    return result


def totals(rows: dict) -> dict:
    result = {}
    for node, hcas in rows.items():
        result[node] = {
            field: sum(row[field] for row in hcas.values() if row.get(field) is not None)
            for field in FIELDS
        }
    return result


def summarize(before: dict, after: dict) -> dict:
    rows = per_hca(before, after)
    return {"per_hca": rows, "totals": totals(rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(
        json.loads(args.before.read_text(encoding="utf-8")),
        json.loads(args.after.read_text(encoding="utf-8")),
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
