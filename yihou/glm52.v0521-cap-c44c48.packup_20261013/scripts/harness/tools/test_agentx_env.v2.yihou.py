#!/usr/bin/env python3
"""Offline check: agentx_env.expected() == actual() for the dry-rendered E1/E2 argv.
Usage: test_agentx_env.v2.yihou.py DRYRUN.txt CONFIG  (YIHOU_* must be in env)
Builds a fake `docker inspect` (Cmd, Env) and server_info from the dryrun argv."""
import os, shlex, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench-harness", "tools"))
import agentx_env  # noqa: E402

dry, config = sys.argv[1], sys.argv[2]
env = dict(l.split("=", 1) for l in subprocess.run(
    ["bash", "-c", f"set -a; source {shlex.quote(config)} >/dev/null; env -0"],
    check=True, capture_output=True).stdout.decode().split("\0") if "=" in l)
cur, rc = None, 0
for line in open(dry):
    if line.startswith("## "):
        cur = line.split()[1].rsplit("-", 1)[0]
    if not line.startswith(" + docker"):
        continue
    argv = [shlex.split(t)[0] if t.strip() else "" for t in line.strip()[2:].split(" + ")]
    img = argv.index(env["IMAGE"])
    cenv = [argv[i + 1] for i in range(img) if argv[i] == "-e"]
    cmd = argv[img + 1:]
    size = int(agentx_env.command_value(cmd, "--hicache-size", "0"))
    row = {"role": cur, "instance": f"{cur}-0"}
    try:
        v = agentx_env.actual(row, {"hicache_size": size, "tp_size": 4, "attn_dp_size": 4, "ep_size": 1},
                              {"Config": {"Cmd": cmd, "Env": cenv},
                               "HostConfig": {"PidMode": argv[argv.index("--pid") + 1] if "--pid" in argv[:img] else ""}}, env)
        print(f"PASS {cur}: hicache_size={v['hicache_size']} ratio={v['hicache_ratio']} "
              f"fused_off={v['dsa_fused_indexer_off']} writer_off={v['aiter_fused_fp8_writer_off']} "
              f"chunk={v['host_register_chunk_gb']} pid_host={v['pid_host']} ipc_legacy={v['ipc_legacy']} "
              f"override={v['model_override']}")
    except ValueError as e:
        rc = 1; print(f"FAIL {cur}: {e}")
sys.exit(rc)
