#!/usr/bin/env python3
"""Purpose: Inspect a live multi-P/D service and build AgentX's runtime contract.
Usage:
    python3 tools/agentx_env.py --topology FILE --router-url URL --output-dir DIR
    --runtime-dir DIR --hf-home DIR --concurrency N --duration N [--ssh-options OPTS]
Artifacts:
    runtime.env plus worker, container, server-info, and hardware snapshots.
Artifact paths:
    OUTPUT_DIR/runtime.env and OUTPUT_DIR/service/*.json.

Inspect a running multi-P/D service and write AgentX runtime.env.

Usage: agentx_env.py --topology FILE --router-url URL --output-dir DIR \
--runtime-dir DIR --hf-home DIR --concurrency N --duration N [--ssh-options OPTS]
"""

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
from urllib.parse import urlsplit


def fetch(url: str):
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except (OSError, ValueError, urllib.error.HTTPError) as exc:
        raise ValueError(f"request failed for {url}: {exc}") from exc


def ssh(options: list[str], node: str, command: list[str]) -> str:
    try:
        result = subprocess.run(
            ["ssh", *options, node, shlex.join(command)],
            check=False, capture_output=True, text=True, errors="replace", timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"{node}: SSH command timed out") from exc
    if result.returncode:
        raise ValueError(
            f"{node}: {(result.stderr or result.stdout).strip() or 'SSH command failed'}"
        )
    return result.stdout


def bool_value(value) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", ""}:
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def command_value(command: list[str], option: str, default: str = "") -> str:
    if option not in command:
        return default
    index = command.index(option)
    if index + 1 >= len(command):
        raise ValueError(f"{option} has no value")
    return command[index + 1]


def load_topology(path: Path, env) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        source = list(csv.DictReader(stream, delimiter="\t"))
    counts = {"prefill": 0, "decode": 0}
    bases = [
        int(env["ENGINE_PORT_BASE"]), int(env["BOOTSTRAP_PORT_BASE"]),
        int(env["KV_EVENT_PORT_BASE"]), int(env["SNAPSHOT_PORT_BASE"]),
    ]
    rows = []
    for index, row in enumerate(source):
        role = row["role"].strip().lower()
        instance = f"{role}-{counts[role]}"
        counts[role] += 1
        ip = row["data_ip"].strip()
        rows.append({
            "instance": instance, "role": role, "node": row["node"].strip(),
            "ip": ip, "url": f"http://{ip}:{bases[0] + index}",
            "container": f"{env['CONTAINER_PREFIX']}-{instance}",
        })
    return rows


def workers(payload) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("workers") or payload.get("data") or payload.get("instances") or []
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("router did not return a worker list")
    return payload


def worker_url(worker: dict) -> str:
    value = (
        worker.get("base_url") or worker.get("url")
        or worker.get("worker_url") or worker.get("endpoint")
    )
    if not isinstance(value, str) or not value:
        raise ValueError(f"worker has no URL: {worker!r}")
    return value.rstrip("/")


def worker_role(worker: dict) -> str:
    role = str(worker.get("disagg_mode") or worker.get("role") or worker.get("mode") or "").lower()
    if role not in {"prefill", "decode"}:
        raise ValueError(f"worker has invalid role: {worker!r}")
    return role


def server_info(url: str) -> dict:
    errors = []
    for endpoint in ("/get_server_info", "/v1/server_info", "/server_info"):
        try:
            value = fetch(url + endpoint)
            if isinstance(value, dict):
                return value
            errors.append(f"{endpoint} did not return an object")
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors))


def expected(env, role: str) -> dict:
    prefix = role.upper()
    return {
        "tp": int(env[f"{prefix}_TP"]),
        "ep": int(env[f"{prefix}_EP"]),
        "dp": int(env[f"{prefix}_DP"]),
        "dpa": bool_value(env[f"{prefix}_DPA"]),
        "hicache": bool_value(env[f"{prefix}_HICACHE"]),
        "mtp": role == "decode" and bool_value(env["DECODE_MTP"]),
        "max_running": int(env[f"{prefix}_MAX_RUNNING"]),
        "graph_max_bs": int(env[f"{prefix}_GRAPH_MAX_BS"]),
        "dsa_prefill_backend": env.get("DSA_PREFILL_BACKEND", "triton"),
        "dsa_decode_backend": env.get("DSA_DECODE_BACKEND", "triton"),
        "dsa_topk_backend": env.get("DSA_TOPK_BACKEND", ""),
        "jit_grouped_topk": bool_value(
            env.get("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "0")
        ),
        "model_override": env.get("JSON_MODEL_OVERRIDE_ARGS", ""),
    }


def verify_pd_contract(row: dict, command: list[str], container_env: dict, env) -> dict:
    """A configuration knob is useful only if it reached the actual worker."""
    actual_rdma = command_value(command, "--disaggregation-ib-device")
    wanted_rdma = env["RDMA_DEVICE"]
    # JSON key order is irrelevant; a list and a per-GPU map are not equivalent.
    def rdma(value):
        return json.loads(value) if value.startswith("{") else value
    if rdma(actual_rdma) != rdma(wanted_rdma):
        raise ValueError(f"{row['instance']}: per-GPU RDMA mapping differs from config")
    keys = ("MC_GID_INDEX", "MC_TE_FILTERS", "MC_ENABLE_DEST_DEVICE_AFFINITY",
            "MC_DISABLE_HIP_TRANSPORT", "MOONCAKE_DISABLE_HIP_DMABUF", "NCCL_IB_DISABLE",
            "SGLANG_ENABLE_FAILED_SESSION_PROBE", "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S")
    values = {key: container_env.get(key, "") for key in keys}
    for key, value in values.items():
        if value != env[key]:
            raise ValueError(f"{row['instance']}: live {key}={value!r}, config expects {env[key]!r}")
    scratch = env[f"{row['role'].upper()}_HSA_NO_SCRATCH_RECLAIM"]
    if container_env.get("HSA_NO_SCRATCH_RECLAIM") != scratch:
        raise ValueError(f"{row['instance']}: role-specific HSA scratch setting differs")
    if not bool_value(container_env.get("SGLANG_DSA_FUSE_TOPK", "1")):
        raise ValueError(f"{row['instance']}: withdrawn SGLANG_DSA_FUSE_TOPK=0 workaround is active")
    if row["role"] == "decode" and bool_value(env["DECODE_MTP"]):
        if "--speculative-use-rejection-sampling" in command:
            raise ValueError("PD decode cannot explicitly enable rejection sampling without draft_probs handoff")
        for flag, wanted in (
            ("--speculative-algorithm", "EAGLE"),
            ("--speculative-num-steps", env["DECODE_SPEC_STEPS"]),
            ("--speculative-eagle-topk", env["DECODE_SPEC_TOPK"]),
            ("--speculative-num-draft-tokens", env["DECODE_SPEC_DRAFT_TOKENS"]),
        ):
            if command_value(command, flag) != wanted:
                raise ValueError(f"{row['instance']}: live {flag} differs from config")
    return {"rdma_device": actual_rdma, **values, "HSA_NO_SCRATCH_RECLAIM": scratch}


def actual(row: dict, info: dict, container: dict, env) -> dict:
    command = [str(item) for item in (container.get("Config") or {}).get("Cmd", [])]
    graph_option = (
        "--cuda-graph-max-bs-prefill"
        if row["role"] == "prefill"
        else "--cuda-graph-max-bs-decode"
    )
    container_env = {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in (container.get("Config") or {}).get("Env", [])
        if isinstance(item, str) and "=" in item
    }
    value = {
        "tp": int(command_value(command, "--tp-size")),
        "ep": int(command_value(command, "--ep-size", "1")),
        "dp": int(command_value(command, "--dp-size", "1")),
        "dpa": "--enable-dp-attention" in command,
        "hicache": "--enable-hierarchical-cache" in command,
        "mtp": "--speculative-algorithm" in command,
        "max_running": int(command_value(command, "--max-running-requests")),
        "graph_max_bs": int(command_value(command, graph_option)),
        "dsa_prefill_backend": command_value(command, "--dsa-prefill-backend"),
        "dsa_decode_backend": command_value(command, "--dsa-decode-backend"),
        "dsa_topk_backend": command_value(command, "--dsa-topk-backend"),
        "jit_grouped_topk": bool_value(
            container_env.get("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "0")
        ),
        "kv_transfer": command_value(command, "--disaggregation-transfer-backend"),
        "model_path": command_value(command, "--model-path"),
        "simulate_acc_len": container_env.get("SGLANG_SIMULATE_ACC_LEN", ""),
        "model_override": command_value(command, "--json-model-override-args"),
        "pd_contract": verify_pd_contract(row, command, container_env, env),
    }
    for key, wanted in expected(env, row["role"]).items():
        if value[key] != wanted:
            raise ValueError(
                f"{row['instance']}: live {key}={value[key]!r}, config expects {wanted!r}"
            )
    if value["kv_transfer"] != env["KV_P2P_TRANSFER"]:
        raise ValueError(f"{row['instance']}: KV transfer differs from config")
    if value["model_path"] != env["MODEL"]:
        raise ValueError(f"{row['instance']}: model path differs from config")
    wanted_simulation = env.get("DECODE_SIMULATE_ACC_LEN", "") if row["role"] == "decode" else ""
    if value["simulate_acc_len"] != wanted_simulation:
        raise ValueError(f"{row['instance']}: simulated acceptance differs from config")
    for key in ("tp", "ep", "dp"):
        reported = int(info.get(f"{key}_size") or 1)
        if reported != value[key]:
            raise ValueError(f"{row['instance']}: server_info {key}={reported}, command={value[key]}")
    return value


def detect_hardware(options: list[str], nodes: list[str]) -> str:
    models = set()
    for node in nodes:
        output = ssh(options, node, ["rocm-smi", "--showproductname"])
        match = re.search(r"\bMI[0-9]{3}[A-Z]*\b", output, re.IGNORECASE)
        if not match:
            raise ValueError(f"{node}: cannot detect GPU model")
        models.add(match.group(0).lower())
    if len(models) != 1:
        raise ValueError(f"mixed GPU models: {sorted(models)}")
    return next(iter(models))


def detect_dram(options: list[str], nodes: list[str]) -> int:
    values = []
    command = ["awk", "/MemTotal/ {printf \"%.0f\\n\", $2/1024/1024}", "/proc/meminfo"]
    for node in nodes:
        value = ssh(options, node, command).strip()
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"{node}: cannot detect CPU DRAM")
        values.append(int(value))
    if max(values) - min(values) > 2:
        raise ValueError(f"nodes have different CPU DRAM: {values}")
    return min(values)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    parser.add_argument("--hf-home", required=True, type=Path)
    parser.add_argument("--concurrency", required=True, type=int)
    parser.add_argument("--duration", required=True, type=int)
    parser.add_argument("--ssh-options", default="")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.duration <= 0:
        raise SystemExit("concurrency and duration must be positive")
    env = os.environ
    options = shlex.split(args.ssh_options)
    output = args.output_dir.resolve()
    service = output / "service"
    service.mkdir(parents=True, exist_ok=True)
    try:
        warmup = env["AGENTX_WARMUP_REQUESTS_PER_LANE"]
        if not warmup.isdigit() or int(warmup) <= 0:
            raise ValueError("AGENTX_WARMUP_REQUESTS_PER_LANE must be positive")
        topology = load_topology(args.topology, env)
        worker_payload = fetch(args.router_url.rstrip("/") + "/v1/workers")
        write_json(service / "workers.json", worker_payload)
        discovered = {worker_url(item): item for item in workers(worker_payload)}
        expected_urls = {row["url"] for row in topology}
        if set(discovered) != expected_urls:
            raise ValueError(
                f"router workers differ: expected={sorted(expected_urls)}, actual={sorted(discovered)}"
            )
        role_settings = {"prefill": [], "decode": []}
        image_ids = {}
        served_models = set()
        for row in topology:
            if worker_role(discovered[row["url"]]) != row["role"]:
                raise ValueError(f"{row['instance']}: router reports wrong role")
            info = server_info(row["url"])
            inspect = json.loads(ssh(options, row["node"], ["docker", "inspect", row["container"]]))[0]
            write_json(service / f"{row['instance']}-server-info.json", info)
            write_json(service / f"{row['instance']}-container.json", inspect)
            role_settings[row["role"]].append(actual(row, info, inspect, env))
            image_ids[row["instance"]] = str(inspect.get("Image") or "")
            model = info.get("served_model_name") or info.get("model_name")
            if isinstance(model, list):
                model = model[0] if model else ""
            served_models.add(str(model or ""))
        if served_models != {env["SERVED_MODEL"]}:
            raise ValueError(f"served model mismatch: {served_models}")
        if len(set(image_ids.values())) != 1:
            raise ValueError(f"workers use different images: {image_ids}")
        for role, settings in role_settings.items():
            if len({json.dumps(item, sort_keys=True) for item in settings}) != 1:
                raise ValueError(f"{role} workers have different settings")
        p = role_settings["prefill"][0]
        d = role_settings["decode"][0]
        nodes = list(dict.fromkeys(row["node"] for row in topology))
        hardware = detect_hardware(options, nodes)
        dram = detect_dram(options, nodes)
        write_json(service / "hardware.json", {"gpu": hardware, "cpu_dram_gb": dram})
        configured_dram = env.get("TOTAL_CPU_DRAM_GB", "")
        if configured_dram:
            if not configured_dram.isdigit() or int(configured_dram) <= 0:
                raise ValueError("TOTAL_CPU_DRAM_GB must be a positive integer")
            if abs(int(configured_dram) - dram) > max(2, dram // 50):
                raise ValueError(
                    f"TOTAL_CPU_DRAM_GB={configured_dram} differs from detected {dram}"
                )
            dram = int(configured_dram)
        parsed = urlsplit(args.router_url)
        metrics = [args.router_url.rstrip("/") + "/metrics"]
        metrics.extend(row["url"] + "/metrics" for row in topology)
        values = {
            "MODEL": p["model_path"], "MODEL_NAME": env["SERVED_MODEL"],
            "SERVED_MODEL_NAME": env["SERVED_MODEL"], "MODEL_PREFIX": "glm5.2",
            "FRAMEWORK": "sglang", "PRECISION": "fp4", "RUNNER_TYPE": hardware,
            "PREFILL_HARDWARE": hardware, "DECODE_HARDWARE": hardware,
            "IMAGE": env["IMAGE"],
            "IMAGE_IDS": json.dumps(image_ids, sort_keys=True, separators=(",", ":")),
            "DSA_PREFILL_BACKEND": p["dsa_prefill_backend"],
            "DSA_DECODE_BACKEND": p["dsa_decode_backend"],
            "DSA_TOPK_BACKEND": p["dsa_topk_backend"],
            "SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK": str(
                p["jit_grouped_topk"]
            ).lower(),
            "PORT": str(parsed.port or 80), "AIPERF_SERVER_URL": args.router_url.rstrip("/"),
            "AIPERF_SERVER_METRICS_URLS": ",".join(metrics),
            "AIPERF_REQUIRED_SERVER_METRIC_PREFIX": "sglang:",
            "AIPERF_FAILED_REQUEST_THRESHOLD": env["AGENTX_FAILED_REQUEST_THRESHOLD"],
            "AIPERF_LIVE_FAILED_REQUEST_THRESHOLD": env["AGENTX_FAILED_REQUEST_THRESHOLD"],
            "AIPERF_WARMUP_REQUESTS_PER_LANE": warmup,
            "AIPERF_HTTP_TCP_USER_TIMEOUT": "900000",
            "CONC": str(args.concurrency), "DURATION": str(args.duration),
            "IS_AGENTIC": "1", "SCENARIO_TYPE": "agentic-coding",
            "IS_MULTINODE": "true", "DISAGG": "true",
            "TP": str(p["tp"]), "EP_SIZE": str(p["ep"]),
            "DP_ATTENTION": str(p["dpa"] or d["dpa"]).lower(),
            "PREFILL_NUM_WORKERS": str(len(role_settings["prefill"])),
            "PREFILL_TP": str(p["tp"]), "PREFILL_EP": str(p["ep"]),
            "PREFILL_DP_SIZE": str(p["dp"]), "PREFILL_DP_ATTN": str(p["dpa"]).lower(),
            "DECODE_NUM_WORKERS": str(len(role_settings["decode"])),
            "DECODE_TP": str(d["tp"]), "DECODE_EP": str(d["ep"]),
            "DECODE_DP_SIZE": str(d["dp"]), "DECODE_DP_ATTN": str(d["dpa"]).lower(),
            "SPEC_DECODING": "mtp" if d["mtp"] else "none",
            "SIMULATE_ACC_LEN": d["simulate_acc_len"],
            "KV_OFFLOADING": "dram" if p["hicache"] or d["hicache"] else "none",
            "KV_OFFLOAD_BACKEND": "hicache" if p["hicache"] or d["hicache"] else "",
            "KV_OFFLOAD_BACKEND_METADATA": '{"name":"hicache"}' if p["hicache"] or d["hicache"] else "",
            "KV_P2P_TRANSFER": p["kv_transfer"], "TOTAL_CPU_DRAM_GB": str(dram),
            "ENABLE_AGENTX_POWER": "0", "PYTHONNOUSERSITE": "1",
            "INFMAX_CONTAINER_WORKSPACE": env["INFERENCEX_DIR"],
            "INFERENCEX_COMMIT": env["INFERENCEX_REF"], "INFERENCEX_DIRTY": "false",
            "AGENTIC_OUTPUT_DIR": str(output),
            "RESULT_FILENAME": f"agentx_conc{args.concurrency}",
            "AIPERF_RUNTIME_DIR": str(args.runtime_dir.resolve()),
            "HF_HOME": str(args.hf_home.resolve()),
            "UV_CACHE_DIR": str(args.runtime_dir.resolve() / "uv-cache"),
            "UV_PYTHON_INSTALL_DIR": str(args.runtime_dir.resolve() / "python"),
            "PIP_CACHE_DIR": str(args.runtime_dir.resolve() / "pip-cache"),
            "XDG_CACHE_HOME": str(args.runtime_dir.resolve() / "xdg-cache"),
            # AIPerf creates Unix sockets below tempfile.gettempdir(); the
            # absolute shared-NFS path exceeds sockaddr_un's 107-byte limit.
            # agentx_bench.sh bind-mounts output/tmp here, keeping data in .tmp.
            "TMPDIR": "/ax-tmp",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOST_UID": str(os.getuid()), "HOST_GID": str(os.getgid()),
        }
        for key, value in values.items():
            if any(char in value for char in "\r\n\0"):
                raise ValueError(f"{key} is invalid for an env file")
        (output / "runtime.env").write_text(
            "# Generated from the running service; do not edit.\n"
            + "".join(f"{key}={values[key]}\n" for key in sorted(values)),
            encoding="utf-8",
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"agentx_env: ERROR: {exc}", file=__import__("sys").stderr)
        return 1
    print(f"agentx_env: PASS ({len(topology)} workers, {output / 'runtime.env'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
