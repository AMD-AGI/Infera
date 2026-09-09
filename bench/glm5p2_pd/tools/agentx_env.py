#!/usr/bin/env python3
"""Inspect a running multi-P/D service and write AgentX's actual runtime env."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


class InspectError(ValueError):
    pass


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", ""}:
        return False
    raise InspectError(f"invalid boolean value: {value!r}")


def normalize_url(value: str) -> str:
    return value.strip().rstrip("/")


def load_topology(path: Path, env: Mapping[str, str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    counts = {"prefill": 0, "decode": 0}
    result = []
    prefix = env["CONTAINER_PREFIX"]
    bases = [
        int(env["ENGINE_PORT_BASE"]),
        int(env["BOOTSTRAP_PORT_BASE"]),
        int(env["KV_EVENT_PORT_BASE"]),
        int(env["SNAPSHOT_PORT_BASE"]),
    ]
    for index, row in enumerate(rows):
        role = row["role"].strip().lower()
        role_index = counts[role]
        counts[role] += 1
        result.append(
            {
                "instance": f"{role}-{role_index}",
                "role": role,
                "node": row["node"].strip(),
                "ip": row["data_ip"].strip(),
                "url": f"http://{row['data_ip'].strip()}:{bases[0] + index}",
                "ports": [base + index for base in bases],
                "container": f"{prefix}-{role}-{role_index}",
            }
        )
    return result


def fetch_json(url: str, timeout: float = 30) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise InspectError(f"request failed for {url}: {exc}") from exc


def worker_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        payload = (
            payload.get("workers")
            or payload.get("data")
            or payload.get("instances")
            or []
        )
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise InspectError("router /v1/workers did not return a worker list")
    return payload


def worker_url(worker: Mapping[str, Any]) -> str:
    value = (
        worker.get("base_url")
        or worker.get("url")
        or worker.get("worker_url")
        or worker.get("endpoint")
    )
    if not isinstance(value, str) or not value.strip():
        raise InspectError(f"worker has no URL: {worker!r}")
    return normalize_url(value)


def worker_role(worker: Mapping[str, Any]) -> str:
    value = worker.get("disagg_mode") or worker.get("role") or worker.get("mode")
    role = str(value or "").strip().lower()
    if role not in {"prefill", "decode"}:
        raise InspectError(f"worker has no valid P/D role: {worker!r}")
    return role


def server_info(url: str) -> dict[str, Any]:
    errors = []
    for endpoint in ("/get_server_info", "/v1/server_info", "/server_info"):
        try:
            value = fetch_json(url + endpoint)
            if not isinstance(value, dict):
                raise InspectError(f"{url + endpoint} did not return an object")
            return value
        except InspectError as exc:
            errors.append(str(exc))
    raise InspectError("; ".join(errors))


def ssh_output(
    ssh_options: list[str], node: str, command: list[str], timeout: float = 60
) -> str:
    try:
        result = subprocess.run(
            ["ssh", *ssh_options, node, shlex.join(command)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise InspectError(f"{node}: SSH command timed out") from exc
    if result.returncode:
        raise InspectError(
            f"{node}: {(result.stderr or result.stdout).strip() or 'SSH command failed'}"
        )
    return result.stdout


def inspect_container(
    ssh_options: list[str], row: Mapping[str, Any]
) -> dict[str, Any]:
    payload = json.loads(
        ssh_output(
            ssh_options,
            str(row["node"]),
            ["docker", "inspect", str(row["container"])],
        )
    )
    if not isinstance(payload, list) or len(payload) != 1:
        raise InspectError(f"{row['instance']}: invalid docker inspect response")
    return payload[0]


def command_value(command: list[str], option: str, default: str = "") -> str:
    try:
        index = command.index(option)
    except ValueError:
        return default
    if index + 1 >= len(command):
        raise InspectError(f"{option} has no value in container command")
    return command[index + 1]


def served_model(info: Mapping[str, Any]) -> str:
    value = info.get("served_model_name") or info.get("model_name")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def expected_role(env: Mapping[str, str], role: str) -> dict[str, Any]:
    prefix = role.upper()
    return {
        "tp": int(env[f"{prefix}_TP"]),
        "ep": int(env[f"{prefix}_EP"]),
        "dp": int(env[f"{prefix}_DP"]),
        "dpa": parse_bool(env[f"{prefix}_DPA"]),
        "hicache": parse_bool(env[f"{prefix}_HICACHE"]),
        "mtp": role == "decode" and parse_bool(env["DECODE_MTP"]),
        "max_running": int(env[f"{prefix}_MAX_RUNNING"]),
    }


def actual_role(
    row: Mapping[str, Any],
    info: Mapping[str, Any],
    container: Mapping[str, Any],
    env: Mapping[str, str],
) -> dict[str, Any]:
    command = [str(item) for item in (container.get("Config") or {}).get("Cmd", [])]
    container_env = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in (container.get("Config") or {}).get("Env", [])
        if isinstance(item, str) and "=" in item
    }
    actual = {
        "tp": int(command_value(command, "--tp-size")),
        "ep": int(command_value(command, "--ep-size", "1")),
        "dp": int(command_value(command, "--dp-size", "1")),
        "dpa": "--enable-dp-attention" in command,
        "hicache": "--enable-hierarchical-cache" in command,
        "mtp": "--speculative-algorithm" in command,
        "max_running": int(command_value(command, "--max-running-requests")),
        "kv_transfer": command_value(command, "--disaggregation-transfer-backend"),
        "model_path": command_value(command, "--model-path"),
        "simulate_acc_len": container_env.get("SGLANG_SIMULATE_ACC_LEN", ""),
    }
    expected = expected_role(env, str(row["role"]))
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise InspectError(
                f"{row['instance']}: live {key}={actual[key]!r}, "
                f"config expects {expected_value!r}"
            )
    if actual["kv_transfer"] != env["KV_P2P_TRANSFER"]:
        raise InspectError(
            f"{row['instance']}: live KV transfer is {actual['kv_transfer']!r}, "
            f"config expects {env['KV_P2P_TRANSFER']!r}"
        )
    if actual["model_path"] != env["MODEL"]:
        raise InspectError(
            f"{row['instance']}: live model path is {actual['model_path']!r}, "
            f"config expects {env['MODEL']!r}"
        )
    expected_simulation = (
        env.get("DECODE_SIMULATE_ACC_LEN", "") if row["role"] == "decode" else ""
    )
    if actual["simulate_acc_len"] != expected_simulation:
        raise InspectError(
            f"{row['instance']}: live simulated acceptance is "
            f"{actual['simulate_acc_len'] or 'off'}, config expects "
            f"{expected_simulation or 'off'}"
        )

    info_values = {
        "tp": int(info.get("tp_size") or 1),
        "ep": int(info.get("ep_size") or 1),
        "dp": int(info.get("dp_size") or 1),
    }
    for key, value in info_values.items():
        if value != actual[key]:
            raise InspectError(
                f"{row['instance']}: server_info {key}={value}, "
                f"container command has {actual[key]}"
            )
    return actual


def detect_hardware(ssh_options: list[str], nodes: list[str]) -> dict[str, str]:
    result = {}
    for node in nodes:
        output = ssh_output(ssh_options, node, ["rocm-smi", "--showproductname"])
        match = re.search(r"\bMI[0-9]{3}[A-Z]*\b", output, re.IGNORECASE)
        if not match:
            raise InspectError(f"{node}: cannot detect GPU model")
        result[node] = match.group(0).lower()
    if len(set(result.values())) != 1:
        raise InspectError(f"topology has mixed GPU models: {result}")
    return result


def detect_dram(ssh_options: list[str], nodes: list[str]) -> dict[str, int]:
    result = {}
    command = ["awk", "/MemTotal/ {printf \"%.0f\\n\", $2/1024/1024}", "/proc/meminfo"]
    for node in nodes:
        value = ssh_output(ssh_options, node, command).strip()
        if not value.isdigit() or int(value) <= 0:
            raise InspectError(f"{node}: cannot detect CPU DRAM capacity")
        result[node] = int(value)
    if max(result.values()) - min(result.values()) > 2:
        raise InspectError(f"topology has different CPU DRAM capacities: {result}")
    return result


def make_runtime_env(
    *,
    topology: list[dict[str, Any]],
    workers_payload: Any,
    infos: Mapping[str, Mapping[str, Any]],
    containers: Mapping[str, Mapping[str, Any]],
    hardware: Mapping[str, str],
    dram: Mapping[str, int],
    env: Mapping[str, str],
    router_url: str,
    concurrency: int,
    duration: int,
    output_dir: Path,
    runtime_dir: Path,
    hf_home: Path,
) -> dict[str, str]:
    discovered = {worker_url(item): item for item in worker_list(workers_payload)}
    expected_urls = {normalize_url(str(row["url"])) for row in topology}
    if set(discovered) != expected_urls:
        raise InspectError(
            f"router workers differ from topology: "
            f"expected={sorted(expected_urls)}, actual={sorted(discovered)}"
        )

    role_settings: dict[str, list[dict[str, Any]]] = {"prefill": [], "decode": []}
    models = set()
    for row in topology:
        url = normalize_url(str(row["url"]))
        if worker_role(discovered[url]) != row["role"]:
            raise InspectError(f"{row['instance']}: router reports the wrong role")
        info = infos[str(row["instance"])]
        model = served_model(info)
        if not model:
            raise InspectError(f"{row['instance']}: server_info has no served model")
        models.add(model)
        role_settings[str(row["role"])].append(
            actual_role(row, info, containers[str(row["instance"])], env)
        )
    if len(models) != 1:
        raise InspectError(f"workers serve different model names: {sorted(models)}")
    live_served_model = next(iter(models))
    if live_served_model != env["SERVED_MODEL"]:
        raise InspectError(
            f"live served model {live_served_model!r} does not match "
            f"config {env['SERVED_MODEL']!r}"
        )
    for role, settings in role_settings.items():
        if len({json.dumps(item, sort_keys=True) for item in settings}) != 1:
            raise InspectError(f"{role} workers do not share one runtime configuration")

    p = role_settings["prefill"][0]
    d = role_settings["decode"][0]
    model_paths = {
        str(settings["model_path"])
        for values in role_settings.values()
        for settings in values
    }
    if len(model_paths) != 1:
        raise InspectError(f"workers use different model paths: {sorted(model_paths)}")
    live_model_path = next(iter(model_paths))
    hw = next(iter(hardware.values()))
    detected_dram = min(dram.values())
    configured_dram = env.get("TOTAL_CPU_DRAM_GB", "").strip()
    if configured_dram:
        if not configured_dram.isdigit() or int(configured_dram) <= 0:
            raise InspectError("TOTAL_CPU_DRAM_GB must be a positive integer")
        if abs(int(configured_dram) - detected_dram) > max(2, detected_dram // 50):
            raise InspectError(
                f"configured TOTAL_CPU_DRAM_GB={configured_dram} differs from "
                f"detected capacity {detected_dram}"
            )
        total_dram = int(configured_dram)
    else:
        total_dram = detected_dram

    hicache = p["hicache"] or d["hicache"]
    parsed_router = urlsplit(router_url)
    image_ids = {
        str(row["instance"]): str(containers[str(row["instance"])].get("Image") or "")
        for row in topology
    }
    metrics_urls = [normalize_url(router_url) + "/metrics"] + [
        normalize_url(str(row["url"])) + "/metrics" for row in topology
    ]
    values: dict[str, Any] = {
        "MODEL": live_model_path,
        "MODEL_NAME": live_served_model,
        "SERVED_MODEL_NAME": live_served_model,
        "MODEL_PREFIX": "glm5.2",
        "FRAMEWORK": "sglang",
        "PRECISION": "fp4",
        "RUNNER_TYPE": hw,
        "PREFILL_HARDWARE": hw,
        "DECODE_HARDWARE": hw,
        "IMAGE": env["IMAGE"],
        "IMAGE_IDS": json.dumps(image_ids, sort_keys=True, separators=(",", ":")),
        "PORT": parsed_router.port or 80,
        "AIPERF_SERVER_URL": normalize_url(router_url),
        "AIPERF_SERVER_METRICS_URLS": ",".join(metrics_urls),
        "AIPERF_REQUIRED_SERVER_METRIC_PREFIX": "sglang:",
        "AIPERF_FAILED_REQUEST_THRESHOLD": env["AGENTX_FAILED_REQUEST_THRESHOLD"],
        "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": env[
            "AGENTX_FAILED_REQUEST_THRESHOLD"
        ],
        "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
        "CONC": concurrency,
        "DURATION": duration,
        "IS_AGENTIC": "1",
        "SCENARIO_TYPE": "agentic-coding",
        "IS_MULTINODE": "true",
        "DISAGG": "true",
        "TP": p["tp"],
        "EP_SIZE": p["ep"],
        "DP_ATTENTION": str(p["dpa"] or d["dpa"]).lower(),
        "PREFILL_NUM_WORKERS": len(role_settings["prefill"]),
        "PREFILL_TP": p["tp"],
        "PREFILL_EP": p["ep"],
        "PREFILL_DP_SIZE": p["dp"],
        "PREFILL_DP_ATTN": str(p["dpa"]).lower(),
        "DECODE_NUM_WORKERS": len(role_settings["decode"]),
        "DECODE_TP": d["tp"],
        "DECODE_EP": d["ep"],
        "DECODE_DP_SIZE": d["dp"],
        "DECODE_DP_ATTN": str(d["dpa"]).lower(),
        "SPEC_DECODING": "mtp" if d["mtp"] else "none",
        "SIMULATE_ACC_LEN": d["simulate_acc_len"],
        "KV_OFFLOADING": "dram" if hicache else "none",
        "KV_OFFLOAD_BACKEND": "hicache" if hicache else "",
        "KV_OFFLOAD_BACKEND_METADATA": '{"name":"hicache"}' if hicache else "",
        "KV_P2P_TRANSFER": p["kv_transfer"],
        "TOTAL_CPU_DRAM_GB": total_dram,
        "ENABLE_AGENTX_POWER": "0",
        "PYTHONNOUSERSITE": "1",
        "INFMAX_CONTAINER_WORKSPACE": env["INFERENCEX_DIR"],
        "AGENTIC_OUTPUT_DIR": str(output_dir),
        "RESULT_FILENAME": f"agentx_conc{concurrency}",
        "AIPERF_RUNTIME_DIR": str(runtime_dir),
        "HF_HOME": str(hf_home),
        "HOST_UID": os.getuid(),
        "HOST_GID": os.getgid(),
    }
    return {key: str(value) for key, value in values.items()}


def write_env(path: Path, values: Mapping[str, str]) -> None:
    for key, value in values.items():
        if any(character in value for character in "\r\n\0"):
            raise InspectError(f"{key} contains a character invalid in an env file")
    path.write_text(
        "# Generated from the running service; do not edit.\n"
        + "".join(f"{key}={values[key]}\n" for key in sorted(values)),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def source_version(path: str) -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", path, "status", "--porcelain"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"INFERENCEX_COMMIT": "unknown", "INFERENCEX_DIRTY": "unknown"}
    return {
        "INFERENCEX_COMMIT": commit,
        "INFERENCEX_DIRTY": str(dirty).lower(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--duration", required=True, type=int)
    parser.add_argument("--ssh-options", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.concurrency <= 0 or args.duration <= 0:
        raise SystemExit("concurrency and duration must be positive")
    env = os.environ
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots = output_dir / "service"
    snapshots.mkdir(exist_ok=True)
    try:
        topology = load_topology(args.topology, env)
        workers_payload = fetch_json(normalize_url(args.router_url) + "/v1/workers")
        write_json(snapshots / "workers.json", workers_payload)
        infos = {}
        containers = {}
        ssh_options = shlex.split(args.ssh_options)
        for row in topology:
            instance = str(row["instance"])
            infos[instance] = server_info(str(row["url"]))
            containers[instance] = inspect_container(ssh_options, row)
            write_json(snapshots / f"{instance}-server-info.json", infos[instance])
            write_json(snapshots / f"{instance}-container.json", containers[instance])
        nodes = list(dict.fromkeys(str(row["node"]) for row in topology))
        hardware = detect_hardware(ssh_options, nodes)
        dram = detect_dram(ssh_options, nodes)
        write_json(
            snapshots / "hardware.json",
            {"gpu": hardware, "cpu_dram_gb": dram},
        )
        values = make_runtime_env(
            topology=topology,
            workers_payload=workers_payload,
            infos=infos,
            containers=containers,
            hardware=hardware,
            dram=dram,
            env=env,
            router_url=args.router_url,
            concurrency=args.concurrency,
            duration=args.duration,
            output_dir=output_dir,
            runtime_dir=args.runtime_dir.resolve(),
            hf_home=args.hf_home.resolve(),
        )
        values.update(source_version(env["INFERENCEX_DIR"]))
        write_env(output_dir / "runtime.env", values)
    except (InspectError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agentx_env: ERROR: {exc}", file=os.sys.stderr)
        return 1
    print(
        f"agentx_env: PASS "
        f"({len(topology)} workers, runtime={output_dir / 'runtime.env'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
