#!/usr/bin/env python3
"""Fail unless the live P8D8 C144 service matches the experiment contract."""

from __future__ import annotations

import json
import subprocess
import urllib.request
from typing import Any


EXPECTED_IMAGE = (
    "sha256:fd7220a57b7d3b58efd875c41f7a9ef46b93469581102d96cbeb6f5451e91d35"
)
EXPECTED_HCAS = {
    str(rank): f"ionic_{rank}"
    for rank in range(8)
}
HCA_LIST = ",".join(f"ionic_{rank}" for rank in range(8))
PREFIX = "glm52-pd-c144-runtime-rca"
WORKERS = (
    (
        "crsuse2-m2m-138",
        f"{PREFIX}-prefill-0",
        "prefill",
        "http://10.245.157.237:29001",
    ),
    (
        "crsuse2-m2m-136",
        f"{PREFIX}-decode-0",
        "decode",
        "http://10.245.154.168:29002",
    ),
)


def fail(message: str) -> None:
    raise SystemExit(f"live C144 assertion failed: {message}")


def ssh_json(node: str, *command: str) -> Any:
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", node, *command],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )
    return json.loads(completed.stdout)


def http_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def env_map(container: dict[str, Any]) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in (container.get("Config") or {}).get("Env", [])
        if isinstance(item, str) and "=" in item
    }


def command(container: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for value in (container.get("Config") or {}).get("Cmd", [])
    ]


def option(argv: list[str], name: str) -> str:
    try:
        return argv[argv.index(name) + 1]
    except (ValueError, IndexError):
        fail(f"missing {name}")
        raise AssertionError


def require_option(argv: list[str], name: str, expected: str) -> None:
    actual = option(argv, name)
    if actual != expected:
        fail(f"{name}={actual!r}, expected {expected!r}")


def server_info(base_url: str) -> dict[str, Any]:
    for path in ("/get_server_info", "/v1/server_info"):
        try:
            payload = http_json(base_url + path)
        except Exception:
            continue
        if isinstance(payload, dict):
            return payload
    fail(f"no server_info at {base_url}")
    raise AssertionError


def main() -> int:
    summary: dict[str, Any] = {"workers": {}}
    for node, name, role, url in WORKERS:
        payload = ssh_json(node, "docker", "inspect", name)
        if not isinstance(payload, list) or len(payload) != 1:
            fail(f"{name} inspect returned an invalid payload")
        container = payload[0]
        if not (container.get("State") or {}).get("Running"):
            fail(f"{name} is not running")
        if container.get("Image") != EXPECTED_IMAGE:
            fail(f"{name} image={container.get('Image')}")

        argv = command(container)
        env = env_map(container)
        if env.get("HIP_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
            fail(f"{name} HIP_VISIBLE_DEVICES={env.get('HIP_VISIBLE_DEVICES')}")
        if env.get("MC_TE_FILTERS") != HCA_LIST:
            fail(f"{name} MC_TE_FILTERS={env.get('MC_TE_FILTERS')}")
        if env.get("MC_ENABLE_DEST_DEVICE_AFFINITY") != "1":
            fail(f"{name} destination affinity is disabled")
        if env.get("MC_GID_INDEX") != "1":
            fail(f"{name} MC_GID_INDEX={env.get('MC_GID_INDEX')}")

        require_option(argv, "--tp-size", "8")
        require_option(argv, "--dp-size", "8")
        require_option(argv, "--ep-size", "1")
        require_option(argv, "--max-running-requests", "256")
        require_option(argv, "--disaggregation-mode", role)
        require_option(argv, "--dsa-prefill-backend", "tilelang")
        require_option(argv, "--dsa-decode-backend", "tilelang")
        graph_option = (
            "--cuda-graph-max-bs-prefill"
            if role == "prefill"
            else "--cuda-graph-max-bs-decode"
        )
        require_option(argv, graph_option, "256")
        if "--enable-dp-attention" not in argv:
            fail(f"{name} DP attention is disabled")
        try:
            hcas = json.loads(option(argv, "--disaggregation-ib-device"))
        except json.JSONDecodeError as exc:
            fail(f"{name} RDMA map is not JSON: {exc}")
        if hcas != EXPECTED_HCAS:
            fail(f"{name} RDMA map={hcas}")
        override = json.loads(option(argv, "--json-model-override-args"))
        if override != {"index_share_for_mtp_iteration": False}:
            fail(f"{name} IndexShare override={override}")

        hicache = "--enable-hierarchical-cache" in argv
        if role == "prefill":
            if not hicache:
                fail("prefill HiCache is disabled")
            require_option(argv, "--hicache-ratio", "1.5")
            require_option(argv, "--hicache-write-policy", "write_through")
            if "SGLANG_SIMULATE_ACC_LEN" in env:
                fail("prefill unexpectedly has simulated acceptance")
        else:
            if hicache:
                fail("decode HiCache is enabled")
            if env.get("SGLANG_SIMULATE_ACC_LEN") != "3.61":
                fail(f"decode simulated acceptance={env.get('SGLANG_SIMULATE_ACC_LEN')}")
            require_option(argv, "--speculative-algorithm", "EAGLE")
            require_option(argv, "--speculative-num-steps", "5")
            require_option(argv, "--speculative-num-draft-tokens", "6")

        info = server_info(url)
        for key, expected in (
            ("tp_size", 8),
            ("dp_size", 8),
            ("max_running_requests", 256),
            ("enable_dp_attention", True),
            ("enable_hierarchical_cache", role == "prefill"),
        ):
            if info.get(key) != expected:
                fail(f"{name} server_info {key}={info.get(key)!r}, expected {expected!r}")
        summary["workers"][name] = {
            "node": node,
            "role": role,
            "image": container.get("Image"),
            "rdma_map": hcas,
            "hicache": hicache,
            "max_running": 256,
            "graph_max_bs": 256,
        }

    router = ssh_json(
        "crsuse2-m2m-138",
        "docker",
        "inspect",
        f"{PREFIX}-router",
    )[0]
    if router.get("Image") != EXPECTED_IMAGE:
        fail("router image differs")
    if env_map(router).get("INFERA_PD_DP_RANK_AFFINITY") != "true":
        fail("router DP rank affinity is not true")
    workers = http_json("http://10.245.157.237:28000/v1/workers")
    worker_list = workers.get("workers", workers) if isinstance(workers, dict) else workers
    if not isinstance(worker_list, list) or len(worker_list) != 2:
        fail(f"router reports {len(worker_list) if isinstance(worker_list, list) else 'invalid'} workers")

    summary["router_affinity"] = True
    summary["router_workers"] = len(worker_list)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
