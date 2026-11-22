# Shared settings for the kf teammate's DSv4.1-Flash Infera + vLLM P/D profiling stack (openai or crusoe MI355X).
# Port of llying's /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/env.sh (read 2026-11-20):
# same image / Infera; our names and paths only. Sourced by the host-side run script and the in-container launchers
# (ctr_*.sh). W is mounted at the same path in every container, so paths below are valid on host and in containers.
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)     # staged copy of <workspace>/pd
# llying's image: vllm-project/vllm@9c1d45c2, built 2026-11-19 06:12 UTC (vllm 0.34.0rc1.dev897, ROCm 10).
IMAGE_REPO=vllm/vllm-openai-rocm
IMAGE_TAG=nightly-9c1d45c21836370104d4eb51d3b46eda20719c47
IMAGE_DIGEST=sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
IMAGE="$IMAGE_REPO@$IMAGE_DIGEST"
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
CTR_PREFIX=yihou-dsv41kf-
ETCD_PORT=${ETCD_PORT:-22479}
ROUTER_PORT=${ROUTER_PORT:-28100}
if [ -d /shared_nfs/models ]; then
    # crusoe: no RDMA container setup (Code 209 at kernel warmup on these nodes, leader E7 11-20) -> Mooncake over TCP.
    KF_CLUSTER=crusoe
    MODEL=/shared_nfs/models/DeepSeek-V4.1-Flash
    NODE_CACHE=/mnt/m2m_nobackup/yihou/dsv41_kf_20261120/cache
    CALIB_LOCAL=/mnt/m2m_nobackup/yihou/dsv41_kf_20261120/calib
    LEADER=/shared_nfs/yihou/dsv41_vllm_pd_20261119
    KF_MC_PROTOCOL=${KF_MC_PROTOCOL:-tcp}
else
    KF_CLUSTER=openai
    MODEL=/perf_apps/data/models/DeepSeek-V4.1-Flash
    NODE_CACHE=/data/yihou/dsv41_kf_20261120/cache
    CALIB_LOCAL=/data/yihou/dsv41_kf_20261120/calib
    LEADER=/perf_apps/yihou/dsv41_vllm_pd_20261119
    KF_MC_PROTOCOL=${KF_MC_PROTOCOL:-rdma}
fi
# AgentX client: the PD leader's staged InferenceX e2e, prebuilt agentx-harness venv (built at the openai path; its
# absolute symlinks need it mounted at VENV_CTR) and HF cache holding cc-traces-weka-062126 — read-only reuse.
IX_E2E=$LEADER/stage/inferencex-e2e
VENV_HOST=$LEADER/venv-agentx
VENV_CTR=/perf_apps/yihou/dsv41_vllm_pd_20261119/venv-agentx
HF_HOME_HOST=$LEADER/hf_home
CALIB_HOOK=$W/calib/hook
node_ip() { [ -n "${NODE_IP:-}" ] && echo "$NODE_IP" || { ip -4 -o addr show dev fenic 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | grep . || hostname -I | awk '{print $1}'; }; }
ts() { date -u +%FT%TZ; }
