#!/usr/bin/env bash
# Stop router/engine processes inside the long-lived container. Run on both nodes
# from inside the container. Use host_container.sh --rm on the host to remove it.
set -euo pipefail

pkill -f "infera.engine.sglang" 2>/dev/null || true
pkill -f "infera.server" 2>/dev/null || true
pkill -f "sglang.launch_server" 2>/dev/null || true

for _ in $(seq 60); do
  pgrep -f "infera.engine.sglang|sglang.launch_server" >/dev/null || break
  sleep 2
done
pkill -9 -f "infera.engine.sglang|sglang.launch_server" 2>/dev/null || true

# kvd after the engine, so its cache backend does not disappear from under a
# still-running engine. The L3 tier is on disk with a journal and is recovered on
# the next start; only the RAM tier dies here. Delete KVD_L3_DIR to discard L3.
pkill -f "infera.kvd" 2>/dev/null || true

echo "[stop] remaining engine procs: $(pgrep -cf 'infera.engine.sglang|sglang.launch_server' || true)"
rocm-smi --showmemuse 2>/dev/null | grep "VRAM%" || true
