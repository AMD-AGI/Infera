#!/usr/bin/env bash
# what: capture the node + image + engine state that served this run.
# why : a packup that records the image TAG only is not reproducible — tags move. Capture
#       the image ID, the resolved engine cmdline, and the driver versions.
# how : run ON the node WHILE the deployment is live.  bash envsnap.sh > env_<node>.txt
set -uo pipefail
CTR="${CTR:-glm52_mix}"

echo "===== captured $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname) ====="
echo
echo "--- host ---"
uname -r; echo "cpu: $(nproc) threads"; free -g | head -2
echo
echo "--- gpus ---"
rocm-smi --showproductname 2>/dev/null | grep -iE "card series|card model" | head -2
rocm-smi --csv --showmeminfo vram 2>/dev/null | tail -8 \
  | awk -F, '{s+=$3; t+=$2} END {printf "VRAM %.0f / %.0f GB used\n", s/1073741824, t/1073741824}'
echo "driver: $(cat /sys/module/amdgpu/version 2>/dev/null)"
echo
echo "--- image ---"
docker inspect "$CTR" --format 'container created: {{.Created}}'
docker inspect "$CTR" --format 'image tag       : {{.Config.Image}}'
docker inspect "$CTR" --format 'image id        : {{.Image}}'
docker inspect "$CTR" --format 'binds           : {{json .HostConfig.Binds}}'
echo
echo "--- versions inside the container ---"
docker exec "$CTR" python3 -c "
import torch, sglang
print('sglang:', sglang.__version__)
print('torch :', torch.__version__)
" 2>/dev/null
docker exec "$CTR" pip show amd-infera 2>/dev/null | grep -E '^(Name|Version|Location)' || true
echo
echo "--- the engine cmdline that actually ran (resolved, not requested) ---"
docker exec "$CTR" bash -c "ps -eo args= | grep '[i]nfera.engine.sglang' | head -1 | tr ' ' '\n' | paste -d' ' - - | head -40"
echo
echo "--- router cmdline ---"
docker exec "$CTR" bash -c "ps -eo args= | grep '[i]nfera.server' | head -1"
echo
echo "--- kvd cmdline + counters ---"
docker exec "$CTR" bash -c "ps -eo args= | grep '[i]nfera.kvd --socket' | head -1"
docker exec "$CTR" python3 -m infera.kvd.statctl --socket "${KVD_SOCK:-/tmp/kvd/kvd.sock}" 2>&1 | head -12
echo
echo "--- resolved server_args (from the engine's own log) ---"
docker exec "$CTR" bash -c "strings /tmp/glm52_mix_${TAG:-base}.log 2>/dev/null \
  | grep -o 'server_args=ServerArgs(.*' | head -1 | cut -c1-4000"
