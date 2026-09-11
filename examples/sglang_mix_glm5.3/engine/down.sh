#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Tear down and WAIT for VRAM to actually drain. The wait is the point, not the
# kill: `docker rm -f` returns before the driver has reclaimed the allocations,
# and relaunching too early aborts the next distributed bootstrap with a
# misleading "memory capacity is unbalanced" error that reads like a model or
# config problem.
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"

# Named explicitly, never a pattern -- on a shared node a pattern is how you
# remove somebody else's container.
docker rm -f "$CTR" "${CTR}_etcd" >/dev/null 2>&1
echo "removed $CTR and ${CTR}_etcd; waiting for VRAM to drain"
for i in $(seq 1 60); do
  busy=$(rocm-smi --showmemuse 2>/dev/null | awk -v want=",$GPUS," '
    match($0, /GPU\[([0-9]+)\]/, m) && /VRAM%/ {
      split($0, f, ": "); if (index(want, "," m[1] ",") && f[2]+0 > 5) printf "%s ", m[1] }')
  [ -z "$busy" ] && { echo "GPUs $GPUS at baseline after $((i * 5))s"; exit 0; }
  sleep 5
done
echo "TIMEOUT: GPU(s) $busy still above baseline." >&2
echo "If they are not ours, that is expected on a shared node -- check first." >&2
rocm-smi --showpids 2>/dev/null | sed -n '3,12p' >&2
exit 1
