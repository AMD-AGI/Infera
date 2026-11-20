#!/bin/bash
# usage: cru_launch.sh <cfg> <gpus> <offset> <conc> <tagprefix> [bench_gpus]
set -u
export DSV41_NFS_ROOT=/shared_nfs/yihou/dsv41_vllm_pd_20261119
export DSV41_LOCAL_ROOT=/mnt/m2m_nobackup/yihou/dsv41_vllm_pd_20261119
export DSV41_MODEL=/shared_nfs/models/DeepSeek-V4.1-Flash
export KIT_VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
export KIT_IX=$DSV41_NFS_ROOT/stage/inferencex-e2e
K=$DSV41_NFS_ROOT/ws/scripts/kit; L=$DSV41_NFS_ROOT/launch; mkdir -p "$L"
CFG=$1; G=$2; OFF=$3; C=$4; T=$5-$(date -u +%m%d%H%M); BG=${6:-2}; P=$((8000+OFF))
setsid nohup bash -c "bash $K/serve.sh $CFG $G $T --port-offset $OFF --wait > $L/$T.serve.log 2>&1 && bash $K/run_point.sh $T http://127.0.0.1:$P $C $BG official > $L/$T.point.log 2>&1; echo CHAIN_DONE rc=\$? >> $L/$T.point.log" >/dev/null 2>&1 < /dev/null &
echo "launched $T"
