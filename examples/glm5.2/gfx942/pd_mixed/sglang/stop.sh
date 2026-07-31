#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Stop the server run_sglang.sh started, then wait for the VRAM to actually come back:
# SGLang holds one scheduler process per rank and they release tens of seconds after
# the launcher exits. Relaunching too early OOMs in a way that reads like a bad
# --mem-fraction-static.
set -euo pipefail

PID_FILE="${PID_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_sglang.pid}"
[[ -f "$PID_FILE" ]] && kill "$(cat "$PID_FILE")" 2>/dev/null || true
pkill -f sglang.launch_server 2>/dev/null || true
rm -f "$PID_FILE"

for _ in $(seq 60); do
    pgrep -f sglang.launch_server >/dev/null || break
    sleep 2
done
pkill -9 -f sglang.launch_server 2>/dev/null || true

for _ in $(seq 60); do
    rocm-smi --showmemuse 2>/dev/null | grep -q 'VRAM%): [1-9]' || break
    sleep 3
done
echo "[stop] done"
