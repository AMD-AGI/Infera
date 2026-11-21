# Shared settings for the kf teammate's DSv4.1-Flash Infera + vLLM P/D profiling stack on openai MI355X.
# Port of llying's /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/env.sh (read 2026-11-20):
# same image / model / Infera; our names and paths only. Sourced by the host-side run script and the in-container
# launchers (ctr_*.sh); /perf_apps is mounted at the same path in the containers.
W=/perf_apps/yihou/dsv41_kf_20261120/pd                 # staged copy of <workspace>/pd (stage_pd.sh)
# llying's image: vllm-project/vllm@9c1d45c2, built 2026-11-19 06:12 UTC (vllm 0.34.0rc1.dev897, ROCm 10).
IMAGE_REPO=vllm/vllm-openai-rocm
IMAGE_TAG=nightly-9c1d45c21836370104d4eb51d3b46eda20719c47
IMAGE_DIGEST=sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
IMAGE="$IMAGE_REPO@$IMAGE_DIGEST"
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
MODEL=/perf_apps/data/models/DeepSeek-V4.1-Flash
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
CTR_PREFIX=yihou-dsv41kf-
ETCD_PORT=${ETCD_PORT:-22479}
ROUTER_PORT=${ROUTER_PORT:-28100}
# dockerd cannot read /home (root-squashed NFS); caches live on node-local scratch.
NODE_CACHE=/data/yihou/dsv41_kf_20261120/cache
# InferenceX e2e (AgentX client): the PD leader's staged copy (read-only reuse).
IX_E2E=/perf_apps/yihou/dsv41_vllm_pd_20261119/stage/inferencex-e2e
# Calibrated rocprofiler fork build (node-local) and hook (staged): see <workspace>/scripts/calib.
CALIB_LOCAL=/data/yihou/dsv41_kf_20261120/calib
CALIB_HOOK=$W/calib/hook
node_ip() { [ -n "${NODE_IP:-}" ] && echo "$NODE_IP" || ip -4 -o addr show dev fenic 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
ts() { date -u +%FT%TZ; }
