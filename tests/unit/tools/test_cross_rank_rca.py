from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SCRIPTS = (
    REPO
    / "bench/glm5p2_pd/results/cross-rank-rca-20260921/scripts"
)
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


analyze = load_script("analyze_reproduction")
matrix = load_script("run_pair_matrix")
certify = load_script("certify_results")


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_router_pair_parser_strips_ansi():
    router = "\n".join(
        [
            "\x1b[3mrole\x1b[0m=\x1b[0mPrefill \x1b[3mpicked\x1b[0m=\x1b[0mhost#dp2",
            "\x1b[3mrole\x1b[0m=\x1b[0mDecode \x1b[3mpicked\x1b[0m=\x1b[0mhost#dp6",
        ]
    )
    assert analyze.parse_router_pairs(router) == [
        {
            "prefill_rank": 2,
            "decode_rank": 6,
            "rank_class": "cross-rank",
            "numa_class": "cross-numa",
        }
    ]


def test_transfer_pair_and_client_error_correlation():
    prefill = (
        "2026-09-21T13:30:53.790819852Z [DP1 TP1] Prefill transfer failed "
        "for request rank=1 req.rid='p' req.bootstrap_room=123 with exception: "
        "Failed to send kv chunk\n"
    )
    decode = (
        "2026-09-21T13:30:53.700000000Z [DP6 TP6] Decode transfer failed "
        "for request rank=6 decode_req.req.rid='d' "
        "decode_req.req.bootstrap_room=123 with exception: Failed to get kvcache\n"
    )
    transfers = analyze.parse_transfer_lines(prefill, decode)
    assert transfers[0]["rank_class"] == "cross-rank"
    assert transfers[0]["numa_class"] == "cross-numa"
    ended = int(
        analyze.epoch("2026-09-21T13:30:53.600000+00:00") * 1_000_000_000
    )
    errors = [{"metadata": {"request_end_ns": ended, "x_request_id": "client"}}]
    matches = analyze.correlate_client_errors(errors, transfers)
    assert matches[0]["bootstrap_room"] == "123"
    assert matches[0]["delta_ms"] == 100.0


def test_decode_port_mapping_and_cqe_parser():
    transfers = [
        {
            "decode_rank": rank,
            "lines": [
                {
                    "role": "prefill",
                    "line": f"remote mooncake session host:{15000 + rank} is not alive",
                }
            ],
        }
        for rank in range(1, 8)
    ]
    decode = "\n".join(
        f"Transfer Engine RPC using P2P handshake, listening on host:{15000 + rank}"
        for rank in range(8)
    )
    mapping, evidence = analyze.decode_port_map(decode, transfers)
    assert mapping[15000] == 0
    assert evidence["15000"]["method"] == "elimination"
    line = (
        "2026-09-21T13:30:18.445453852Z Worker: Process failed for slice "
        "(opcode: 1, source_addr: 0x1, length: 65536, dest_addr: 0x2, "
        "local_nic: ionic_3, peer_nic: host:15006@ionic_3, dest_rkey: 1): "
        "transport retry counter exceeded"
    )
    events = analyze.parse_cqe_events(line, mapping)
    assert events[0]["destination_rank"] == 6
    assert events[0]["same_named_hca"] is True


def test_pair_matrix_contract_helpers():
    assert matrix.rail("source-local", 2, 6) == "ionic_2"
    assert matrix.rail("destination-local", 2, 6) == "ionic_6"
    assert matrix.rail("auto", 2, 6) == matrix.SHARED_HCAS
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY=1" in matrix.docker_command(
        "on", {"dest_affinity": True, "physical_gpu": 2}
    )
    assert "MC_ENABLE_DEST_DEVICE_AFFINITY=1" not in matrix.docker_command(
        "off", {"dest_affinity": False, "physical_gpu": 2}
    )
    assert "HIP_VISIBLE_DEVICES=2" in matrix.docker_command(
        "gpu", {"dest_affinity": True, "physical_gpu": 2}
    )
    assert matrix.geometry("batch", 3) == {
        "size": 1 << 30,
        "chunk": 32 << 20,
        "nchunk": 32,
        "seconds": 3,
    }
    before = {
        "nodes": {
            "source": {
                "ionic_0": {
                    "req_tx_retry_excd_err": 1,
                    "tx_rdma_ack_timeout": 2,
                }
            }
        }
    }
    after = {
        "nodes": {
            "source": {
                "ionic_0": {
                    "req_tx_retry_excd_err": 4,
                    "tx_rdma_ack_timeout": 7,
                }
            }
        }
    }
    _, totals = matrix.fault_counter_deltas(before, after)
    assert totals == {"req_tx_retry_excd_err": 3, "tx_rdma_ack_timeout": 5}


def test_certification_happy_path(tmp_path: Path):
    matrix_dir = tmp_path / "matrix"
    run = tmp_path / "run"
    dump(matrix_dir / "contract.json", {"policies": ["auto"]})
    dump(
        matrix_dir / "summary.json",
        [
            {
                "rail_policy": "auto",
                "source_gpu": source,
                "destination_gpu": destination,
                "verified": True,
                "passed": True,
                "counter_snapshot_complete": True,
                "cqe_error_12": 0,
                "retry_exhausted": 0,
                "req_tx_retry_excd_err": 0,
                "tx_rdma_ack_timeout": 0,
            }
            for source in range(8)
            for destination in range(8)
        ],
    )
    dump(
        run / "live-config.json",
        {
            "router_affinity": False,
            "workers": {
                role: {"hcas": certify.SHARED_HCAS, "hicache": False}
                for role in ("prefill", "decode")
            },
        },
    )
    dump(
        run / "analysis/summary.json",
        {
            "benchmark": {"duration_seconds": 1200},
            "signatures": {
                "client_errors": 0,
                "prefill_transfer_failed": 0,
                "decode_transfer_failed": 0,
                "cqe_error_12": 0,
                "retry_exhausted": 0,
                "session_failed": 0,
            },
            "hca_hard_counter_totals": {},
        },
    )
    dump(
        run / "analysis/router-pairs.json",
        [{"decode_rank": rank} for rank in range(8)],
    )
    gates = certify.matrix_gates(matrix_dir, "auto") + certify.agentx_gates(run)
    assert all(gate["passed"] for gate in gates)
