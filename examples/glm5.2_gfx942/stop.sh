#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Stop the router and engine processes. Run INSIDE the engine container on BOTH
# nodes. The container itself survives; remove it from the host with
# `bash host_container.sh --rm`, and etcd with `docker rm -f infera-glm52-etcd`.
set -euo pipefail

pkill -f "infera.engine.sglang" 2>/dev/null || true
pkill -f "infera.server" 2>/dev/null || true
# ROUTER_BACKEND=rust execs this binary over the python process, so after the exec
# the "infera.server" pattern above no longer matches the router.
pkill -f "infera-router" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true

# Give the engines time to release VRAM; a relaunch that races them OOMs.
for _ in $(seq 60); do
  pgrep -f "infera.engine.sglang|sglang.launch_server" >/dev/null || break
  sleep 2
done
pkill -9 -f "infera.engine.sglang|sglang.launch_server" 2>/dev/null || true

# kvd after the engines, so its cache backend does not vanish from under one that
# is still running. Only the RAM tier dies here: L3 is on disk with a journal and
# is recovered on the next start. Delete KVD_L3_DIR to discard it.
pkill -f "infera.kvd" 2>/dev/null || true

echo "[stop] remaining engine procs: $(pgrep -cf 'infera.engine.sglang|sglang.launch_server' || true)"
rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" || true
