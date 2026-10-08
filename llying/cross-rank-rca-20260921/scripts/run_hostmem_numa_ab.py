#!/usr/bin/env python3
"""Host-memory RDMA WRITE A/B: receiver buffer NUMA-local vs cross-NUMA to its HCA.

Every rail ionic_k carries one ib_write_bw stream from the source node to the
same-named HCA on the destination node, all eight concurrently. Sender memory
and all CPUs stay NUMA-local to their HCA; the only variable is the NUMA node
of the receiver's buffer. No GPU is used on either node.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

from capture_hca_counters import snapshot
from hca_mechanism import summarize
from rca_nodes import DESTINATION_IP, DESTINATION_NODE, SOURCE_NODE


def hca_numa(index: int) -> int:
    return 0 if index < 4 else 1


def ssh(node: str, script: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, "bash", "-s"],
        input=script,
        text=True,
        capture_output=True,
        timeout=timeout,
        errors="replace",
    )


def perftest(hca: int, port: int, seconds: int, qps: int, size: int, extra: list[str]) -> list[str]:
    return [
        "ib_write_bw", "-d", f"ionic_{hca}", "-x", "1", "-F", "--report_gbits",
        "-D", str(seconds), "-q", str(qps), "-s", str(size), "-p", str(port), *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver-numa", choices=("local", "cross"), required=True)
    parser.add_argument("--sender-numa", choices=("local", "cross"), default="local")
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--qps", type=int, default=8)
    parser.add_argument("--size", type=int, default=1 << 20)
    parser.add_argument("--hcas", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--base-port", type=int, default=18700)
    parser.add_argument(
        "--perftest-args", default="", help="extra ib_write_bw options for both ends, e.g. --tclass=104"
    )
    args = parser.parse_args()
    hcas = [int(value) for value in args.hcas.split(",")]
    extra = shlex.split(args.perftest_args)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = out.name

    servers, clients = [], []
    for hca in hcas:
        numa = hca_numa(hca)
        rx_numa = numa if args.receiver_numa == "local" else 1 - numa
        tx_numa = numa if args.sender_numa == "local" else 1 - numa
        port = args.base_port + hca
        server = ["numactl", f"--cpunodebind={numa}", f"--membind={rx_numa}"]
        server += perftest(hca, port, args.seconds, args.qps, args.size, extra)
        client = ["numactl", f"--cpunodebind={numa}", f"--membind={tx_numa}"]
        client += perftest(hca, port, args.seconds, args.qps, args.size, extra) + [DESTINATION_IP]
        log = f"/tmp/mc-rca-hostmem-{tag}-{hca}"
        servers.append(
            f"timeout {args.seconds + 60} {shlex.join(server)} > {log}-server.log 2>&1 &"
        )
        clients.append(
            f"timeout {args.seconds + 60} {shlex.join(client)} > {log}-client.log 2>&1 &"
        )
    contract = {
        "source_node": SOURCE_NODE,
        "destination_node": DESTINATION_NODE,
        "receiver_numa": args.receiver_numa,
        "sender_numa": args.sender_numa,
        "seconds": args.seconds,
        "qps_per_hca": args.qps,
        "message_bytes": args.size,
        "perftest_args": extra,
        "hcas": hcas,
        "servers": servers,
        "clients": clients,
    }
    (out / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")

    before = snapshot()
    (out / "counters-before.json").write_text(json.dumps(before) + "\n")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    ssh(DESTINATION_NODE, "\n".join(servers) + "\nsleep 1\n", 60)
    time.sleep(3)
    ssh(SOURCE_NODE, "\n".join(clients) + "\nwait\n", args.seconds + 120)
    time.sleep(5)
    completed = dt.datetime.now(dt.timezone.utc).isoformat()
    after = snapshot()
    (out / "counters-after.json").write_text(json.dumps(after) + "\n")

    collect = "\n".join(
        f"echo '### {hca}'; cat /tmp/mc-rca-hostmem-{tag}-{hca}-client.log; "
        f"rm -f /tmp/mc-rca-hostmem-{tag}-{hca}-client.log"
        for hca in hcas
    )
    client_logs = ssh(SOURCE_NODE, collect, 60).stdout
    server_logs = ssh(
        DESTINATION_NODE,
        "\n".join(
            f"echo '### {hca}'; cat /tmp/mc-rca-hostmem-{tag}-{hca}-server.log; "
            f"rm -f /tmp/mc-rca-hostmem-{tag}-{hca}-server.log"
            for hca in hcas
        ),
        60,
    ).stdout
    (out / "client.log").write_text(client_logs)
    (out / "server.log").write_text(server_logs)

    bandwidth = {}
    errors = {}
    for block in client_logs.split("### ")[1:]:
        hca, _, body = block.partition("\n")
        match = re.search(rf"^\s*{args.size}\s+\d+\s+[\d.]+\s+([\d.]+)", body, re.M)
        bandwidth[f"ionic_{hca.strip()}"] = float(match.group(1)) if match else None
        errors[f"ionic_{hca.strip()}"] = len(re.findall(r"cqe with error|Completion with error", body))
    mechanism = summarize(before, after)
    result = {
        **contract,
        "started_at": started,
        "completed_at": completed,
        "bw_gbps": bandwidth,
        "bw_total_gbps": round(sum(v for v in bandwidth.values() if v), 2),
        "client_cqe_errors": errors,
        "mechanism_totals": mechanism["totals"],
    }
    (out / "mechanism.json").write_text(json.dumps(mechanism, indent=2, sort_keys=True) + "\n")
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: result[k] for k in ("receiver_numa", "bw_gbps", "bw_total_gbps", "client_cqe_errors")}, indent=1))
    for node, values in mechanism["totals"].items():
        print(node, {k: values[k] for k in ("pause_tx", "pause_rx", "nic_rx_loss", "oos", "seq_nak_tx", "retx_pkts", "ack_timeout", "retry_excd", "ecn_rx", "cnp_tx")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
