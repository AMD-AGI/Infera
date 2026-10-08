#!/usr/bin/env python3
"""Fail unless the live P8D8 service matches the fixed RCA contract."""

from __future__ import annotations

import json
import subprocess

from rca_nodes import DESTINATION_NODE, IMAGE_ID as EXPECTED_IMAGE, SOURCE_NODE


SHARED_HCAS = "ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
WORKERS = (
    (SOURCE_NODE, "glm52-pd-crossrank-rca-prefill-0", "prefill"),
    (DESTINATION_NODE, "glm52-pd-crossrank-rca-decode-0", "decode"),
)


def ssh_json(node: str, *command: str):
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", node, *command],
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return json.loads(completed.stdout)


def fail(message: str) -> None:
    raise SystemExit(f"live RCA assertion failed: {message}")


def env_map(container: dict) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in (container.get("Config") or {}).get("Env", [])
        if isinstance(item, str) and "=" in item
    }


def option(command: list[str], name: str) -> str:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        fail(f"missing {name}")
        raise AssertionError


def main() -> int:
    summary = {"workers": {}}
    for node, name, role in WORKERS:
        container = ssh_json(node, "docker", "inspect", name)[0]
        if container.get("Image") != EXPECTED_IMAGE:
            fail(f"{name} image={container.get('Image')}")
        command = [str(value) for value in (container.get("Config") or {}).get("Cmd", [])]
        env = env_map(container)
        if env.get("HIP_VISIBLE_DEVICES") != "0,1,2,3,4,5,6,7":
            fail(f"{name} HIP_VISIBLE_DEVICES={env.get('HIP_VISIBLE_DEVICES')}")
        if env.get("MC_TE_FILTERS") != SHARED_HCAS:
            fail(f"{name} MC_TE_FILTERS={env.get('MC_TE_FILTERS')}")
        if env.get("MC_ENABLE_DEST_DEVICE_AFFINITY") != "1":
            fail(f"{name} destination affinity is not enabled")
        if env.get("MC_GID_INDEX") != "1":
            fail(f"{name} MC_GID_INDEX={env.get('MC_GID_INDEX')}")
        if option(command, "--disaggregation-ib-device") != SHARED_HCAS:
            fail(f"{name} does not use the shared HCA list")
        if "{" in option(command, "--disaggregation-ib-device"):
            fail(f"{name} uses a per-GPU RDMA map")
        if "--enable-hierarchical-cache" in command:
            fail(f"{name} unexpectedly enables HiCache")
        if option(command, "--dp-size") != "8" or option(command, "--tp-size") != "8":
            fail(f"{name} is not DP8/TP8")
        if role == "decode":
            if env.get("SGLANG_SIMULATE_ACC_LEN") != "3.61":
                fail(f"{name} simulated acceptance differs")
            override = option(command, "--json-model-override-args")
            if json.loads(override) != {"index_share_for_mtp_iteration": False}:
                fail(f"{name} IndexShare override differs")
        summary["workers"][name] = {
            "node": node,
            "role": role,
            "image": container.get("Image"),
            "hcas": option(command, "--disaggregation-ib-device"),
            "hicache": False,
            "dest_local_rail": env.get("MC_ENABLE_DEST_LOCAL_RAIL"),
            "ib_timeout": env.get("MC_IB_TIMEOUT"),
        }

    router = ssh_json(
        SOURCE_NODE, "docker", "inspect", "glm52-pd-crossrank-rca-router"
    )[0]
    router_env = env_map(router)
    if router.get("Image") != EXPECTED_IMAGE:
        fail("router image differs")
    if router_env.get("INFERA_PD_DP_RANK_AFFINITY") != "false":
        fail(
            "router affinity="
            + str(router_env.get("INFERA_PD_DP_RANK_AFFINITY"))
        )
    summary["router_affinity"] = False
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
