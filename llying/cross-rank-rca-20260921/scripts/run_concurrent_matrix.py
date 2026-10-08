#!/usr/bin/env python3
"""Run eight GPU pairs concurrently per wave: permutations, or fan-in groups."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path

from hca_mechanism import summarize
from run_pair_matrix import (
    DESTINATION_IP,
    DESTINATION_NODE,
    FAULT_COUNTERS,
    IMAGE,
    IMAGE_ID,
    MODULE,
    REPO,
    SCHEMA_VERSION,
    SOURCE_IP,
    SOURCE_NODE,
    cleanup,
    docker_command,
    fault_counter_deltas,
    fault_counters,
    geometry,
    image_id,
    rail,
    remote,
    snapshot,
    write_json,
)


CONCURRENT_SCHEMA_VERSION = 2


def wave_pairs(offset: int, fan_in: int = 1) -> list[tuple[int, int]]:
    # Each group of fan_in consecutive sources writes to one destination GPU.
    return [(source, (source - source % fan_in + offset) % 8) for source in range(8)]


def policies(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    valid = {"source-local", "destination-local", "auto"}
    if not result or any(item not in valid for item in result):
        raise ValueError("invalid rail policy")
    return result


def offsets(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(set(result)) != len(result) or any(
        item < 0 or item > 7 for item in result
    ):
        raise ValueError("offsets must be unique integers from 0 through 7")
    return result


def pair_specs(
    output: Path,
    profile: str,
    policy: str,
    source: int,
    destination: int,
    transfer_geometry: dict,
    dest_affinity: bool,
) -> tuple[dict, dict, dict]:
    pair_dir = output / policy / f"p{source}-d{destination}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    for stale in (
        "target.json",
        "target.json.tmp",
        "verify.json",
        "verify.json.tmp",
        "done",
        "result.json",
    ):
        (pair_dir / stale).unlink(missing_ok=True)
    metadata = {
        "source_gpu": source,
        "destination_gpu": destination,
        "rail_policy": policy,
        "profile": profile,
        "source_node": SOURCE_NODE,
        "destination_node": DESTINATION_NODE,
        "dest_device_affinity": dest_affinity,
    }
    common = {
        "sig": str(pair_dir),
        "protocol": "rdma",
        "operation": "write",
        "loc": "gpu",
        "dev": rail(policy, source, destination),
        "pattern_id": source * 8 + destination,
        "geometry": transfer_geometry,
        "metadata": metadata,
        "dest_affinity": dest_affinity,
    }
    target = {
        **common,
        "role": "target",
        "hostname": f"{DESTINATION_IP}:0",
        "host": DESTINATION_NODE,
        "gpu": destination,
        "physical_gpu": destination,
    }
    initiator = {
        **common,
        "role": "initiator",
        "hostname": f"{SOURCE_IP}:0",
        "host": SOURCE_NODE,
        "gpu": source,
        "physical_gpu": source,
    }
    return target, initiator, metadata


def shared_container_command(
    name: str, specs: list[dict], log_name: str, worker_env: list[str]
) -> list[str]:
    template = docker_command(name, specs[0])
    image_index = template.index(IMAGE)
    prefix = template[:image_index]
    filtered = []
    index = 0
    child_only = (
        "MC_TE_FILTERS=",
        "MC_ENABLE_DEST_DEVICE_AFFINITY=",
    )
    while index < len(prefix):
        if (
            prefix[index] == "-e"
            and index + 1 < len(prefix)
            and prefix[index + 1].startswith(child_only)
        ):
            index += 2
            continue
        filtered.append(prefix[index])
        index += 1

    launches = ["rc=0", "pids=''"]
    for spec in specs:
        pair_dir = Path(spec["sig"])
        env = ["env", f"MC_TE_FILTERS={spec['dev']}"]
        if spec.get("dest_affinity", True):
            env.append("MC_ENABLE_DEST_DEVICE_AFFINITY=1")
        command = env + worker_env + [
            "python3",
            "-m",
            MODULE,
            json.dumps(spec, separators=(",", ":")),
        ]
        log_path = pair_dir / log_name
        launches.append(
            f"{shlex.join(command)} > {shlex.quote(str(log_path))} 2>&1 & "
            "pids=\"$pids $!\""
        )
    launches.append('for pid in $pids; do wait "$pid" || rc=1; done')
    launches.append("exit $rc")
    return filtered + [IMAGE, "bash", "-lc", "; ".join(launches)]


def wait_target_files(
    paths: list[Path], process: subprocess.Popen, timeout: float
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(path.exists() for path in paths):
            return True
        if process.poll() is not None:
            return all(path.exists() for path in paths)
        time.sleep(0.2)
    return all(path.exists() for path in paths)


def run_shared_pairs(
    output: Path,
    profile: str,
    policy: str,
    offset: int,
    fan_in: int,
    transfer_geometry: dict,
    dest_affinity: bool,
    worker_env: list[str],
) -> list[dict]:
    target_specs = []
    initiator_specs = []
    metadatas = []
    pairs = wave_pairs(offset, fan_in)
    for source, destination in pairs:
        target, initiator, metadata = pair_specs(
            output,
            profile,
            policy,
            source,
            destination,
            transfer_geometry,
            dest_affinity,
        )
        target_specs.append(target)
        initiator_specs.append(initiator)
        metadatas.append(metadata)

    token = f"{profile}-{policy}-o{offset}".replace("_", "-").lower()
    target_name = f"mc-rca-wave-t-{token}"[:63]
    initiator_name = f"mc-rca-wave-i-{token}"[:63]
    wave_dir = output / "waves" / policy / f"offset-{offset}"
    target_container_log = wave_dir / "target-container.log"
    target_stream = target_container_log.open("w", encoding="utf-8")
    target_process = None
    started = dt.datetime.now(dt.timezone.utc)
    try:
        target_command = shlex.join(
            shared_container_command(target_name, target_specs, "target.log", worker_env)
        )
        target_process = subprocess.Popen(
            ["ssh", "-o", "BatchMode=yes", DESTINATION_IP, target_command],
            stdout=target_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        target_paths = [Path(spec["sig"]) / "target.json" for spec in target_specs]
        if not wait_target_files(target_paths, target_process, 180):
            raise RuntimeError("not all target workers published registered segments")
        initiator = remote(
            SOURCE_NODE,
            shared_container_command(
                initiator_name, initiator_specs, "initiator.log", worker_env
            ),
            check=False,
            capture_output=True,
            timeout=float(transfer_geometry["seconds"]) + 420,
        )
        (wave_dir / "initiator-container.log").write_text(
            (initiator.stdout or "") + (initiator.stderr or ""),
            encoding="utf-8",
        )
        try:
            target_rc = target_process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            target_process.kill()
            target_rc = -9
        target_stream.flush()

        records = []
        for (source, destination), metadata in zip(pairs, metadatas):
            pair_dir = output / policy / f"p{source}-d{destination}"
            result_path = pair_dir / "result.json"
            payload = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {}
            )
            verify_path = pair_dir / "verify.json"
            verify = (
                json.loads(verify_path.read_text(encoding="utf-8"))
                if verify_path.exists()
                else {}
            )
            logs = "".join(
                (pair_dir / name).read_text(
                    encoding="utf-8", errors="replace"
                )
                if (pair_dir / name).exists()
                else ""
                for name in ("initiator.log", "target.log")
            )
            payload.update(metadata)
            payload.update(
                {
                    "schema_version": SCHEMA_VERSION,
                    "device_filter": rail(policy, source, destination),
                    "pattern_id": source * 8 + destination,
                    "geometry": transfer_geometry,
                    "initiator_rc": (
                        0 if payload.get("gb_s") is not None else initiator.returncode
                    ),
                    "target_rc": target_rc,
                    "verified": verify.get("verified", payload.get("verified")),
                    "cqe_error_12": logs.count("cqe with error 12"),
                    "retry_exhausted": logs.count(
                        "transport retry counter exceeded"
                    ),
                    "counter_scope": "wave",
                    "req_tx_retry_excd_err": None,
                    "tx_rdma_ack_timeout": None,
                    "counter_snapshot_complete": None,
                    "started_at": started.isoformat(),
                    "completed_at": dt.datetime.now(
                        dt.timezone.utc
                    ).isoformat(),
                    "numa_class": (
                        "same-numa"
                        if (source < 4) == (destination < 4)
                        else "cross-numa"
                    ),
                    "orchestrator_complete": True,
                }
            )
            payload["passed"] = bool(
                payload["initiator_rc"] == 0
                and target_rc == 0
                and payload.get("gb_s") is not None
                and payload.get("verified") is True
                and payload["cqe_error_12"] == 0
                and payload["retry_exhausted"] == 0
            )
            if not payload["passed"] and not payload.get("reason"):
                payload["reason"] = "shared-container wave gate failed"
            write_json(result_path, payload)
            records.append(payload)
        return records
    finally:
        if target_process is not None and target_process.poll() is None:
            target_process.kill()
        target_stream.close()
        cleanup(target_name, DESTINATION_NODE)
        cleanup(initiator_name, SOURCE_NODE)


def run_wave(
    output: Path,
    profile: str,
    policy: str,
    offset: int,
    fan_in: int,
    transfer_geometry: dict,
    dest_affinity: bool,
    worker_env: list[str],
) -> tuple[dict, list[dict]]:
    wave_dir = output / "waves" / policy / f"offset-{offset}"
    wave_dir.mkdir(parents=True, exist_ok=True)
    wave_result = wave_dir / "result.json"
    if wave_result.exists():
        existing = json.loads(wave_result.read_text(encoding="utf-8"))
        if (
            existing.get("orchestrator_complete")
            and existing.get("schema_version") == CONCURRENT_SCHEMA_VERSION
            and existing.get("geometry") == transfer_geometry
            and existing.get("dest_device_affinity") is dest_affinity
            and existing.get("worker_env", []) == worker_env
            and existing.get("fan_in", 1) == fan_in
        ):
            return existing, existing["records"]

    before = fault_counters()
    write_json(wave_dir / "counters-before.json", before)
    full_before = snapshot()
    started = dt.datetime.now(dt.timezone.utc)
    pairs = wave_pairs(offset, fan_in)
    records = run_shared_pairs(
        output,
        profile,
        policy,
        offset,
        fan_in,
        transfer_geometry,
        dest_affinity,
        worker_env,
    )

    after = fault_counters()
    write_json(wave_dir / "counters-after.json", after)
    mechanism = summarize(full_before, snapshot())
    write_json(wave_dir / "mechanism.json", mechanism)
    counter_rows, counter_totals = fault_counter_deltas(before, after)
    write_json(wave_dir / "counter-deltas.json", counter_rows)
    counters_complete = len(counter_rows) == 2 * 8 * len(FAULT_COUNTERS)
    counter_clean = counters_complete and all(
        counter_totals.get(name, 0) == 0 for name in FAULT_COUNTERS
    )
    for record in records:
        record["wave_offset"] = offset
        record["wave_counter_snapshot_complete"] = counters_complete
        record["wave_req_tx_retry_excd_err"] = counter_totals.get(
            "req_tx_retry_excd_err"
        )
        record["wave_tx_rdma_ack_timeout"] = counter_totals.get(
            "tx_rdma_ack_timeout"
        )
        if not counter_clean:
            record["passed"] = False
            suffix = "wave HCA hard-counter gate failed"
            record["reason"] = (
                f"{record['reason']}; {suffix}" if record.get("reason") else suffix
            )
        pair_result = (
            output
            / policy
            / f"p{record['source_gpu']}-d{record['destination_gpu']}"
            / "result.json"
        )
        write_json(pair_result, record)
    records.sort(key=lambda record: record["source_gpu"])
    result = {
        "schema_version": CONCURRENT_SCHEMA_VERSION,
        "policy": policy,
        "offset": offset,
        "fan_in": fan_in,
        "geometry": transfer_geometry,
        "dest_device_affinity": dest_affinity,
        "worker_env": worker_env,
        "pairs": pairs,
        "started_at": started.isoformat(),
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "counter_snapshot_complete": counters_complete,
        "counter_totals": counter_totals,
        "mechanism_totals": mechanism["totals"],
        "passed": counter_clean and all(record["passed"] for record in records),
        "records": records,
        "orchestrator_complete": True,
    }
    write_json(wave_result, result)
    return result, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("batch", "soak"), default="soak")
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument(
        "--policies", default="source-local,destination-local,auto"
    )
    parser.add_argument(
        "--dest-affinity",
        choices=("on", "off"),
        default="on",
        help="whether MC_ENABLE_DEST_DEVICE_AFFINITY is present",
    )
    parser.add_argument("--offsets", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--fan-in",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
        help="sources per destination GPU in each wave (1 = permutation)",
    )
    parser.add_argument(
        "--worker-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra environment for every worker on both nodes",
    )
    parser.add_argument("--confirm-exclusive", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any("=" not in item for item in args.worker_env):
        parser.error("--worker-env takes KEY=VALUE")
    if args.seconds < 0:
        parser.error("seconds must be non-negative")
    try:
        selected_policies = policies(args.policies)
        selected_offsets = offsets(args.offsets)
    except ValueError as exc:
        parser.error(str(exc))
    args.output = args.output.resolve()
    transfer_geometry = geometry(args.profile, args.seconds)
    dest_affinity = args.dest_affinity == "on"
    contract = {
        "schema_version": CONCURRENT_SCHEMA_VERSION,
        "pair_worker_schema_version": SCHEMA_VERSION,
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "source_node": SOURCE_NODE,
        "destination_node": DESTINATION_NODE,
        "profile": args.profile,
        "geometry": transfer_geometry,
        "policies": selected_policies,
        "dest_device_affinity": dest_affinity,
        "worker_env": args.worker_env,
        "waves_per_policy": len(selected_offsets),
        "fan_in": args.fan_in,
        "pairs_per_wave": 8,
        "pair_concurrency": 8,
        "pairs": {
            str(offset): wave_pairs(offset, args.fan_in) for offset in selected_offsets
        },
        "counter_scope": "wave",
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

    records = []
    failed_waves = []
    for policy in selected_policies:
        for offset in selected_offsets:
            wave, wave_records = run_wave(
                args.output,
                args.profile,
                policy,
                offset,
                args.fan_in,
                transfer_geometry,
                dest_affinity,
                args.worker_env,
            )
            records.extend(wave_records)
            status = "PASS" if wave["passed"] else "FAIL"
            rx = wave["mechanism_totals"][DESTINATION_NODE]
            tx = wave["mechanism_totals"][SOURCE_NODE]
            print(
                f"{policy} offset={offset} {status} pairs={len(wave_records)} "
                f"gb_s={[record.get('gb_s') for record in wave_records]} "
                f"rx_loss={rx['nic_rx_loss']} oos={rx['oos']} pause_tx={rx['pause_tx']} "
                f"retx={tx['retx_pkts']} ack_timeout={tx['ack_timeout']} "
                f"retry_excd={tx['retry_excd']}",
                flush=True,
            )
            if not wave["passed"]:
                failed_waves.append((policy, offset))

    write_json(args.output / "counters-after.json", snapshot())
    records.sort(
        key=lambda record: (
            record["rail_policy"],
            record["wave_offset"],
            record["source_gpu"],
        )
    )
    write_json(args.output / "summary.json", records)
    keys = (
        "rail_policy",
        "wave_offset",
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
        "wave_req_tx_retry_excd_err",
        "wave_tx_rdma_ack_timeout",
        "wave_counter_snapshot_complete",
        "reason",
    )
    with (args.output / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(
        f"completed={len(records)} failed_waves={len(failed_waves)} "
        f"failed_pairs={sum(not record['passed'] for record in records)}"
    )
    return 1 if failed_waves else 0


if __name__ == "__main__":
    raise SystemExit(main())
