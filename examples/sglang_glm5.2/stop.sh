#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Tear down whatever this node is running. Run on both nodes; `--all` also stops etcd.
# A leftover engine keeps its VRAM, so the next launch OOMs — always stop before relaunching.
set -euo pipefail

pkill -f "infera.engine.sglang" 2>/dev/null || true
pkill -f "infera.server" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true
[[ "${1:-}" == "--all" ]] && { pkill -f "etcd --name infera-etcd" 2>/dev/null || true; }

for _ in $(seq 60); do
    pgrep -f "infera.engine.sglang|sglang.launch_server" >/dev/null || break
    sleep 2
done
pkill -9 -f "infera.engine.sglang|sglang.launch_server" 2>/dev/null || true
echo "[stop] remaining sglang procs: $(pgrep -cf 'infera.engine.sglang|sglang.launch_server' || true)"

# SGLang spawns a scheduler process per rank; VRAM comes back tens of seconds after
# the last one exits, and rocm-smi is the only honest signal that it has. Relaunching
# before this reads 0 is how you get an OOM that looks like a config problem.
for _ in $(seq 60); do
    used="$(rocm-smi --showmemuse 2>/dev/null | grep -c 'VRAM%): [1-9]' || true)"
    [[ "$used" == "0" ]] && break
    sleep 3
done
rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" || true
