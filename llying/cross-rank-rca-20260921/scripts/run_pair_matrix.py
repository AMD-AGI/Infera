#!/usr/bin/env python3
"""Run independent source-GPU -> destination-GPU Mooncake WRITE pairs."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

from capture_hca_counters import snapshot
from rca_nodes import (
    DESTINATION_IP,
    DESTINATION_NODE,
    IMAGE,
    IMAGE_ID,
    SOURCE_IP,
    SOURCE_NODE,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SHARED_HCAS = ",".join(f"ionic_{index}" for index in range(8))
MODULE = "infera.tools.preflight.network.mooncakeperf"
# 3: workers see all eight GPUs, so a buffer's Mooncake location is its
# physical GPU, as in the SGLang ranks.
SCHEMA_VERSION = 3
FAULT_COUNTERS = ("req_tx_retry_excd_err", "tx_rdma_ack_timeout")
FAULT_COUNTER_SCRIPT = r"""
import glob
import json
import os
import pathlib

result = {}
for ib_path in sorted(glob.glob("/sys/class/infiniband/ionic_*")):
    hca = os.path.basename(ib_path)
    counters = {}
    for name in ("req_tx_retry_excd_err", "tx_rdma_ack_timeout"):
        path = pathlib.Path(ib_path) / "ports" / "1" / "hw_counters" / name
        try:
            counters[name] = int(path.read_text().strip(), 0)
        except (OSError, ValueError):
            counters[name] = None
    result[hca] = counters
print(json.dumps(result, sort_keys=True))
"""


def remote(node: str, command: list[str], **kwargs):
    ssh_host = {
        SOURCE_NODE: SOURCE_IP,
        DESTINATION_NODE: DESTINATION_IP,
    }.get(node, node)
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            ssh_host,
            shlex.join(command),
        ],
        text=True,
        errors="replace",
        **kwargs,
    )


def image_id(node: str) -> str:
    completed = remote(
        node,
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return completed.stdout.strip()


def write_json(path: Path, payload) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def fault_counters() -> dict:
    nodes = {}
    command = ["python3", "-c", FAULT_COUNTER_SCRIPT]
    for node in (SOURCE_NODE, DESTINATION_NODE):
        completed = remote(
            node,
            command,
            check=True,
            capture_output=True,
            timeout=30,
        )
        nodes[node] = json.loads(completed.stdout)
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "nodes": nodes,
    }


def fault_counter_deltas(before: dict, after: dict) -> tuple[list[dict], dict]:
    rows = []
    totals = Counter({counter: 0 for counter in FAULT_COUNTERS})
    for node, hcas in before.get("nodes", {}).items():
        for hca, counters in hcas.items():
            after_counters = after.get("nodes", {}).get(node, {}).get(hca, {})
            for counter in FAULT_COUNTERS:
                old = counters.get(counter)
                new = after_counters.get(counter)
                if isinstance(old, int) and isinstance(new, int):
                    delta = new - old
                    rows.append(
                        {
                            "node": node,
                            "hca": hca,
                            "counter": counter,
                            "before": old,
                            "after": new,
                            "delta": delta,
                        }
                    )
                    totals[counter] += delta
    return rows, dict(totals)


def rail(policy: str, source_gpu: int, destination_gpu: int) -> str:
    if policy == "source-local":
        return f"ionic_{source_gpu}"
    if policy == "destination-local":
        return f"ionic_{destination_gpu}"
    if policy == "auto":
        return SHARED_HCAS
    raise ValueError(policy)


def docker_command(name: str, spec: dict) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "host",
        "--ipc",
        "host",
        "--ipc",
        "host",
        "--shm-size",
        "8g",
        "--device",
        "/dev/kfd",
        "--device",
        "/dev/dri",
        "--device",
        "/dev/infiniband",
        "--group-add",
        "video",
        "--group-add",
        "render",
        "--cap-add",
        "IPC_LOCK",
        "--ulimit",
        "memlock=-1:-1",
        "--ulimit",
        "nofile=65536:65536",
        "-v",
        f"{REPO}:{REPO}",
        "-v",
        "/lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro",
        "--workdir",
        str(REPO),
        "-e",
        f"PYTHONPATH={REPO}",
        "-e",
        "MC_GID_INDEX=1",
        "-e",
        "MC_DISABLE_HIP_TRANSPORT=1",
        "-e",
        "MOONCAKE_DISABLE_HIP_DMABUF=0",
        "-e",
        "RDMAV_FORK_SAFE=1",
    ]
    if spec.get("dest_affinity", True):
        command.extend(["-e", "MC_ENABLE_DEST_DEVICE_AFFINITY=1"])
    command.extend(
        [
        IMAGE,
        "python3",
        "-m",
        MODULE,
        json.dumps(spec, separators=(",", ":")),
        ]
    )
    return command


def wait_file(path: Path, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        if process.poll() is not None:
            return path.exists()
        time.sleep(0.2)
    return path.exists()


def cleanup(name: str, node: str) -> None:
    remote(
        node,
        ["docker", "rm", "-f", name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
    )


def run_pair(
    output: Path,
    profile: str,
    policy: str,
    source_gpu: int,
    destination_gpu: int,
    geometry: dict,
    capture_pair_counters: bool = True,
    dest_affinity: bool = True,
) -> dict:
    label = f"p{source_gpu}-d{destination_gpu}"
    pair_dir = output / policy / label
    pair_dir.mkdir(parents=True, exist_ok=True)
    result_path = pair_dir / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("orchestrator_complete")
            and existing.get("passed") is True
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("profile") == profile
            and existing.get("rail_policy") == policy
            and existing.get("source_gpu") == source_gpu
            and existing.get("destination_gpu") == destination_gpu
            and existing.get("geometry") == geometry
            and existing.get("counter_scope", "pair")
            == ("pair" if capture_pair_counters else "wave")
            and existing.get("dest_device_affinity", True) is dest_affinity
        ):
            return existing
    for stale in ("target.json", "target.json.tmp", "verify.json", "verify.json.tmp", "done"):
        (pair_dir / stale).unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)

    pattern_id = source_gpu * 8 + destination_gpu
    device = rail(policy, source_gpu, destination_gpu)
    metadata = {
        "source_gpu": source_gpu,
        "destination_gpu": destination_gpu,
        "rail_policy": policy,
        "profile": profile,
        "source_node": SOURCE_NODE,
        "destination_node": DESTINATION_NODE,
        "dest_device_affinity": dest_affinity,
    }
    target_spec = {
        "role": "target",
        "sig": str(pair_dir),
        "hostname": f"{DESTINATION_IP}:0",
        "host": DESTINATION_NODE,
        "protocol": "rdma",
        "operation": "write",
        "loc": "gpu",
        "gpu": destination_gpu,
        "physical_gpu": destination_gpu,
        "dev": device,
        "pattern_id": pattern_id,
        "geometry": geometry,
        "metadata": metadata,
        "dest_affinity": dest_affinity,
    }
    initiator_spec = {
        "role": "initiator",
        "sig": str(pair_dir),
        "hostname": f"{SOURCE_IP}:0",
        "host": SOURCE_NODE,
        "protocol": "rdma",
        "operation": "write",
        "loc": "gpu",
        "gpu": source_gpu,
        "physical_gpu": source_gpu,
        "dev": device,
        "pattern_id": pattern_id,
        "geometry": geometry,
        "metadata": metadata,
        "dest_affinity": dest_affinity,
    }
    token = re.sub(r"[^a-z0-9-]", "-", f"{profile}-{policy}-{label}".lower())
    target_name = f"mc-rca-t-{token}"[:63]
    initiator_name = f"mc-rca-i-{token}"[:63]
    started = dt.datetime.now(dt.timezone.utc)
    timeout = float(geometry["seconds"]) + 420
    target = None
    target_log = None
    counters_before = None
    try:
        if capture_pair_counters:
            counters_before = fault_counters()
            write_json(pair_dir / "counters-before.json", counters_before)
        target_log = (pair_dir / "target.log").open("w", encoding="utf-8")
        target_command = shlex.join(docker_command(target_name, target_spec))
        target = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", DESTINATION_IP, target_command],
            stdout=target_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if not wait_file(pair_dir / "target.json", target, 180):
            raise RuntimeError("target did not publish its registered segment")
        initiator = remote(
            SOURCE_NODE,
            docker_command(initiator_name, initiator_spec),
            check=False,
            capture_output=True,
            timeout=timeout,
        )
        (pair_dir / "initiator.log").write_text(
            (initiator.stdout or "") + (initiator.stderr or ""),
            encoding="utf-8",
        )
        try:
            target_rc = target.wait(timeout=180)
        except subprocess.TimeoutExpired:
            target.kill()
            target_rc = -9
        target_log.flush()
        payload = {}
        worker_result = pair_dir / "result.json"
        if worker_result.exists():
            payload = json.loads(worker_result.read_text(encoding="utf-8"))
        verify_path = pair_dir / "verify.json"
        verify = (
            json.loads(verify_path.read_text(encoding="utf-8"))
            if verify_path.exists()
            else {}
        )
        logs = (
            (pair_dir / "initiator.log").read_text(encoding="utf-8", errors="replace")
            + (pair_dir / "target.log").read_text(encoding="utf-8", errors="replace")
        )
        if capture_pair_counters:
            counters_after = fault_counters()
            write_json(pair_dir / "counters-after.json", counters_after)
            counter_rows, counter_totals = fault_counter_deltas(
                counters_before, counters_after
            )
            write_json(pair_dir / "counter-deltas.json", counter_rows)
            counter_snapshot_complete = (
                len(counter_rows) == 2 * 8 * len(FAULT_COUNTERS)
            )
        else:
            counter_totals = {}
            counter_snapshot_complete = None
        payload.update(metadata)
        payload.update(
            {
                "schema_version": SCHEMA_VERSION,
                "device_filter": device,
                "pattern_id": pattern_id,
                "geometry": geometry,
                "initiator_rc": initiator.returncode,
                "target_rc": target_rc,
                "verified": verify.get("verified", payload.get("verified")),
                "cqe_error_12": logs.count("cqe with error 12"),
                "retry_exhausted": logs.count("transport retry counter exceeded"),
                "counter_scope": "pair" if capture_pair_counters else "wave",
                "req_tx_retry_excd_err": counter_totals.get(
                    "req_tx_retry_excd_err"
                ),
                "tx_rdma_ack_timeout": counter_totals.get("tx_rdma_ack_timeout"),
                "counter_snapshot_complete": counter_snapshot_complete,
                "started_at": started.isoformat(),
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "numa_class": (
                    "same-numa"
                    if (source_gpu < 4) == (destination_gpu < 4)
                    else "cross-numa"
                ),
                "orchestrator_complete": True,
            }
        )
        payload["passed"] = bool(
            initiator.returncode == 0
            and target_rc == 0
            and payload.get("gb_s") is not None
            and payload.get("verified") is True
            and payload["cqe_error_12"] == 0
            and payload["retry_exhausted"] == 0
            and (
                not capture_pair_counters
                or (
                    payload["req_tx_retry_excd_err"] == 0
                    and payload["tx_rdma_ack_timeout"] == 0
                    and payload["counter_snapshot_complete"] is True
                )
            )
        )
        if not payload["passed"] and not payload.get("reason"):
            failed_gates = []
            if initiator.returncode != 0:
                failed_gates.append("initiator_rc")
            if target_rc != 0:
                failed_gates.append("target_rc")
            if payload.get("gb_s") is None:
                failed_gates.append("gb_s")
            if payload.get("verified") is not True:
                failed_gates.append("verified")
            for key in ("cqe_error_12", "retry_exhausted"):
                if payload.get(key) != 0:
                    failed_gates.append(key)
            if capture_pair_counters:
                for key in ("req_tx_retry_excd_err", "tx_rdma_ack_timeout"):
                    if payload.get(key) != 0:
                        failed_gates.append(key)
                if payload.get("counter_snapshot_complete") is not True:
                    failed_gates.append("counter_snapshot_complete")
            payload["reason"] = "failed gates: " + ",".join(failed_gates)
        write_json(result_path, payload)
        return payload
    except Exception as exc:
        if capture_pair_counters and counters_before is not None:
            try:
                counters_after = fault_counters()
                write_json(pair_dir / "counters-after.json", counters_after)
                counter_rows, counter_totals = fault_counter_deltas(
                    counters_before, counters_after
                )
                write_json(pair_dir / "counter-deltas.json", counter_rows)
            except Exception as counter_exc:
                counter_totals = {}
                counter_error = f"{type(counter_exc).__name__}: {counter_exc}"
            else:
                counter_error = None
        else:
            counter_totals = {}
            counter_error = (
                "before snapshot unavailable" if capture_pair_counters else None
            )
        payload = {
            **metadata,
            "schema_version": SCHEMA_VERSION,
            "device_filter": device,
            "pattern_id": pattern_id,
            "geometry": geometry,
            "passed": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "counter_error": counter_error,
            "counter_scope": "pair" if capture_pair_counters else "wave",
            "req_tx_retry_excd_err": counter_totals.get("req_tx_retry_excd_err"),
            "tx_rdma_ack_timeout": counter_totals.get("tx_rdma_ack_timeout"),
            "counter_snapshot_complete": False,
            "started_at": started.isoformat(),
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "orchestrator_complete": True,
        }
        write_json(result_path, payload)
        return payload
    finally:
        if target is not None and target.poll() is None:
            target.kill()
        if target_log is not None:
            target_log.close()
        cleanup(target_name, DESTINATION_NODE)
        cleanup(initiator_name, SOURCE_NODE)


def parse_pairs(value: str) -> list[tuple[int, int]]:
    if value == "all":
        return [(source, destination) for source in range(8) for destination in range(8)]
    if value == "cross-numa":
        return [
            (source, destination)
            for source in range(8)
            for destination in range(8)
            if (source < 4) != (destination < 4)
        ]
    pairs = []
    for item in value.split(","):
        match = re.fullmatch(r"(\d+)[-:](\d+)", item.strip())
        if not match:
            raise ValueError(f"invalid pair: {item}")
        pair = (int(match.group(1)), int(match.group(2)))
        if any(index not in range(8) for index in pair):
            raise ValueError(f"pair outside GPU 0..7: {item}")
        pairs.append(pair)
    return pairs


def geometry(profile: str, seconds: float) -> dict:
    if profile == "single":
        return {"size": 64 << 10, "chunk": 64 << 10, "nchunk": 1, "seconds": 0}
    if profile == "batch":
        return {"size": 1 << 30, "chunk": 32 << 20, "nchunk": 32, "seconds": seconds}
    if profile == "soak":
        return {"size": 1 << 30, "chunk": 32 << 20, "nchunk": 32, "seconds": seconds}
    raise ValueError(profile)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("single", "batch", "soak"), required=True)
    parser.add_argument("--policies", default="source-local,destination-local,auto")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--confirm-exclusive",
        action="store_true",
        help="required for execution: confirms both nodes are reserved and drained",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the immutable work plan without SSH, Docker, or counter access",
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    seconds = (
        args.seconds
        if args.seconds is not None
        else {"single": 0.0, "batch": 3.0, "soak": 120.0}[args.profile]
    )
    if seconds < 0 or args.jobs <= 0:
        parser.error("seconds must be non-negative and jobs positive")
    if args.jobs != 1:
        parser.error("jobs must be 1 so per-pair HCA counters remain attributable")
    policies = [value.strip() for value in args.policies.split(",") if value.strip()]
    if not policies or any(value not in {"source-local", "destination-local", "auto"} for value in policies):
        parser.error("invalid rail policy")
    pairs = parse_pairs(args.pairs)
    if len(pairs) != len(set(pairs)):
        parser.error("duplicate GPU pair")
    transfer_geometry = geometry(args.profile, seconds)
    work = [
        (policy, source_gpu, destination_gpu)
        for policy in policies
        for source_gpu, destination_gpu in pairs
    ]
    contract = {
        "schema_version": SCHEMA_VERSION,
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "source_node": SOURCE_NODE,
        "destination_node": DESTINATION_NODE,
        "profile": args.profile,
        "policies": policies,
        "pairs": pairs,
        "geometry": transfer_geometry,
        "jobs": args.jobs,
        "work_items": len(work),
        "per_pair_fault_counters": list(FAULT_COUNTERS),
    }
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    if not args.confirm_exclusive:
        parser.error("--confirm-exclusive is required for hardware execution")
    args.output.mkdir(parents=True, exist_ok=True)

    for node in (SOURCE_NODE, DESTINATION_NODE):
        actual = image_id(node)
        if actual != IMAGE_ID:
            raise SystemExit(f"{node}: image {actual}, expected {IMAGE_ID}")
    write_json(args.output / "contract.json", contract)
    write_json(args.output / "counters-before.json", snapshot())
    records: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                run_pair,
                args.output,
                args.profile,
                policy,
                source_gpu,
                destination_gpu,
                transfer_geometry,
            ): (policy, source_gpu, destination_gpu)
            for policy, source_gpu, destination_gpu in work
        }
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            print(
                record["rail_policy"],
                f"P{record['source_gpu']}->D{record['destination_gpu']}",
                "PASS" if record["passed"] else f"FAIL {record.get('reason', '')}",
                flush=True,
            )

    write_json(args.output / "counters-after.json", snapshot())
    records.sort(
        key=lambda record: (
            record["rail_policy"],
            record["source_gpu"],
            record["destination_gpu"],
        )
    )
    write_json(args.output / "summary.json", records)
    keys = (
        "rail_policy",
        "source_gpu",
        "destination_gpu",
        "numa_class",
        "device_filter",
        "passed",
        "verified",
        "gb_s",
        "gib",
        "cqe_error_12",
        "retry_exhausted",
        "req_tx_retry_excd_err",
        "tx_rdma_ack_timeout",
        "counter_snapshot_complete",
        "reason",
    )
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    failures = [record for record in records if not record["passed"]]
    print(f"completed={len(records)} failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
