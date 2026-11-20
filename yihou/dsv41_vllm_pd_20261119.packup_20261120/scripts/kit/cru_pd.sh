#!/bin/bash
# usage: cru_pd.sh <pd-cfg> <p-gpus> <d-gpus> <tagprefix> [--port-offset N]
set -u
export DSV41_NFS_ROOT=/shared_nfs/yihou/dsv41_vllm_pd_20261119
export DSV41_LOCAL_ROOT=/mnt/m2m_nobackup/yihou/dsv41_vllm_pd_20261119
export DSV41_MODEL=/shared_nfs/models/DeepSeek-V4.1-Flash
export KIT_VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
export KIT_IX=$DSV41_NFS_ROOT/stage/inferencex-e2e
K=$DSV41_NFS_ROOT/ws/scripts/kit; L=$DSV41_NFS_ROOT/launch; mkdir -p "$L"
CFG=$1; PG=$2; DG=$3; T=$4-$(date -u +%m%d%H%M); shift 4
setsid nohup bash -c "bash $K/serve_pd.sh $CFG $PG $DG $T $* --wait > $L/$T.serve.log 2>&1; echo SERVE_DONE rc=\$? >> $L/$T.serve.log" >/dev/null 2>&1 < /dev/null &
echo "launched $T"
