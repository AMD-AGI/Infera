#!/usr/bin/env python3
"""Apply the cross-rank RCA hard gates to a matrix and final AgentX run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SHARED_HCAS = ",".join(f"ionic_{index}" for index in range(8))
ZERO_FIELDS = (
    "cqe_error_12",
    "retry_exhausted",
    "req_tx_retry_excd_err",
    "tx_rdma_ack_timeout",
)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def scoped(record: dict, field: str):
    # Concurrent waves snapshot HCA counters once per 8-pair wave; a zero wave
    # delta covers every pair in it.
    if record.get("counter_scope") == "wave" and f"wave_{field}" in record:
        return record[f"wave_{field}"]
    return record.get(field)


def matrix_gates(matrix: Path, required_policy: str) -> list[dict]:
    contract = load(matrix / "contract.json")
    records = load(matrix / "summary.json")
    selected = [
        record for record in records if record.get("rail_policy") == required_policy
    ]
    pairs = [
        (record.get("source_gpu"), record.get("destination_gpu"))
        for record in selected
    ]
    expected_pairs = {(source, destination) for source in range(8) for destination in range(8)}
    gates = [
        {
            "name": "matrix-contract-policy",
            "passed": required_policy in contract.get("policies", []),
            "detail": required_policy,
        },
        {
            "name": "matrix-64-unique-pairs",
            "passed": len(selected) == 64 and set(pairs) == expected_pairs,
            "detail": {
                "records": len(selected),
                "unique_pairs": len(set(pairs)),
                "missing": sorted(expected_pairs - set(pairs)),
            },
        },
        {
            "name": "matrix-byte-verify",
            "passed": bool(selected) and all(record.get("verified") is True for record in selected),
            "detail": sum(record.get("verified") is not True for record in selected),
        },
        {
            "name": "matrix-process-success",
            "passed": bool(selected) and all(record.get("passed") is True for record in selected),
            "detail": sum(record.get("passed") is not True for record in selected),
        },
        {
            "name": "matrix-counter-snapshots-complete",
            "passed": bool(selected)
            and all(
                scoped(record, "counter_snapshot_complete") is True
                for record in selected
            ),
            "detail": sum(
                scoped(record, "counter_snapshot_complete") is not True
                for record in selected
            ),
        },
    ]
    for field in ZERO_FIELDS:
        gates.append(
            {
                "name": f"matrix-zero-{field}",
                "passed": bool(selected)
                and all(scoped(record, field) == 0 for record in selected),
                "detail": {
                    "nonzero_records": sum(
                        scoped(record, field) != 0 for record in selected
                    )
                },
            }
        )
    return gates


def live_contract_gates(run: Path) -> list[dict]:
    live = load(run / "live-config.json")
    workers = list(live.get("workers", {}).values())
    return [
        {
            "name": "agentx-router-affinity-off",
            "passed": live.get("router_affinity") is False,
            "detail": live.get("router_affinity"),
        },
        {
            "name": "agentx-shared-8-hca",
            "passed": len(workers) == 2
            and all(worker.get("hcas") == SHARED_HCAS for worker in workers),
            "detail": [worker.get("hcas") for worker in workers],
        },
        {
            "name": "agentx-hicache-off",
            "passed": len(workers) == 2
            and all(worker.get("hicache") is False for worker in workers),
            "detail": [worker.get("hicache") for worker in workers],
        },
    ]


def agentx_gates(run: Path) -> list[dict]:
    summary = load(run / "analysis" / "summary.json")
    signatures = summary.get("signatures", {})
    benchmark = summary.get("benchmark", {})
    hard_counters = summary.get("hca_hard_counter_totals", {})
    router_pairs = load(run / "analysis" / "router-pairs.json")
    decode_counts = Counter(pair.get("decode_rank") for pair in router_pairs)
    gates = live_contract_gates(run)
    gates.extend(
        [
            {
                "name": "agentx-duration-1200s",
                "passed": (benchmark.get("duration_seconds") or 0) >= 1200,
                "detail": benchmark.get("duration_seconds"),
            },
            {
                "name": "agentx-client-errors-zero",
                "passed": signatures.get("client_errors") == 0,
                "detail": signatures.get("client_errors"),
            },
            {
                "name": "agentx-transfer-failures-zero",
                "passed": signatures.get("prefill_transfer_failed") == 0
                and signatures.get("decode_transfer_failed") == 0,
                "detail": {
                    "prefill": signatures.get("prefill_transfer_failed"),
                    "decode": signatures.get("decode_transfer_failed"),
                },
            },
            {
                # Requests cancelled by the client at the end of profiling.
                "name": "agentx-transfer-aborts-client-cancelled",
                "passed": signatures.get("transfer_client_aborted", 0)
                <= (benchmark.get("cancelled_requests") or 0),
                "detail": {
                    "transfer_client_aborted": signatures.get("transfer_client_aborted"),
                    "client_cancelled": benchmark.get("cancelled_requests"),
                },
            },
            {
                "name": "agentx-cqe12-zero",
                "passed": signatures.get("cqe_error_12") == 0,
                "detail": signatures.get("cqe_error_12"),
            },
            {
                "name": "agentx-retry-exhausted-zero",
                "passed": signatures.get("retry_exhausted") == 0,
                "detail": signatures.get("retry_exhausted"),
            },
            {
                "name": "agentx-session-failed-zero",
                "passed": signatures.get("session_failed") == 0,
                "detail": signatures.get("session_failed"),
            },
            {
                "name": "agentx-hca-hard-counters-zero",
                "passed": all(hard_counters.get(field, 0) == 0 for field in ZERO_FIELDS[2:]),
                "detail": hard_counters,
            },
            {
                "name": "agentx-decode-ranks-global",
                "passed": set(decode_counts) == set(range(8))
                and all(decode_counts[rank] > 0 for rank in range(8)),
                "detail": {str(rank): decode_counts[rank] for rank in range(8)},
            },
        ]
    )
    return gates


def same_image_gate(matrix: Path, run: Path) -> dict:
    matrix_image = load(matrix / "contract.json").get("image_id")
    workers = load(run / "live-config.json").get("workers", {}).values()
    agentx_images = sorted({worker.get("image") for worker in workers})
    return {
        "name": "matrix-agentx-same-image",
        "passed": agentx_images == [matrix_image],
        "detail": {"matrix": matrix_image, "agentx": agentx_images},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--agentx-run", type=Path, required=True)
    parser.add_argument(
        "--required-policy",
        choices=("source-local", "destination-local", "auto"),
        default="auto",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.agentx_run / "analysis" / "certification.json"
    gates = matrix_gates(args.matrix.resolve(), args.required_policy)
    gates.extend(agentx_gates(args.agentx_run.resolve()))
    gates.append(same_image_gate(args.matrix.resolve(), args.agentx_run.resolve()))
    result = {
        "matrix": str(args.matrix.resolve()),
        "agentx_run": str(args.agentx_run.resolve()),
        "required_policy": args.required_policy,
        "passed": all(gate["passed"] for gate in gates),
        "failed_gates": [gate["name"] for gate in gates if not gate["passed"]],
        "gates": gates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
