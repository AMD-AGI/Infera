# Source on a crusoe node: points the kit at crusoe paths (NFS + node-local + model), keeps the openai-built venv path in containers.
export DSV41_NFS_ROOT=/shared_nfs/yihou/dsv41_vllm_pd_20261119
export DSV41_LOCAL_ROOT=/mnt/m2m_nobackup/yihou/dsv41_vllm_pd_20261119
export DSV41_MODEL=/shared_nfs/models/DeepSeek-V4.1-Flash
export KIT_VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
export KIT_IX=$DSV41_NFS_ROOT/stage/inferencex-e2e
K=$DSV41_NFS_ROOT/ws/scripts/kit
