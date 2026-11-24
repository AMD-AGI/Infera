#!/bin/bash
# cru_pd_chain.sh <p-gpus> <d-gpus> <port-offset> <cfg:conc>... — run PD points back to back on a crusoe node (fresh stack per
# point), so the GPUs never sit idle between points. Logs: $DSV41_NFS_ROOT/launch/<tag>.{serve,point}.log
set -u
export DSV41_NFS_ROOT=/shared_nfs/yihou/dsv41_vllm_pd_20261119
export DSV41_LOCAL_ROOT=/mnt/m2m_nobackup/yihou/dsv41_vllm_pd_20261119
export DSV41_MODEL=/shared_nfs/models/DeepSeek-V4.1-Flash
export KIT_VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
export KIT_IX=$DSV41_NFS_ROOT/stage/inferencex-e2e
K=$DSV41_NFS_ROOT/ws/scripts/kit; L=$DSV41_NFS_ROOT/launch; mkdir -p "$L"
PG=$1; DG=$2; OFF=$3; shift 3; PORT=$((8000+OFF))
for item in "$@"; do
  cfg=${item%%:*}; c=${item##*:}; T=${cfg%%-*}-c$c-$(date -u +%m%d%H%M)
  bash "$K/serve_pd.sh" "$cfg" "$PG" "$DG" "$T" --port-offset "$OFF" --wait > "$L/$T.serve.log" 2>&1 \
    && bash "$K/run_point.sh" "$T" "http://127.0.0.1:$PORT" "$c" 4 official > "$L/$T.point.log" 2>&1
  echo "CHAIN_DONE rc=$?" >> "$L/$T.point.log"
  bash "$K/stop.sh" "$T" --prefix >> "$L/$T.point.log" 2>&1
done
