# Shared settings for the DSv4.1-Flash infera+vLLM bring-up on AUS MI355X. Sourced by the host-side items
# (holder/queue/*.sh) and by the in-container launchers (ctr_*.sh); /perf_apps is mounted at the same path.
W=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119
# Latest vLLM ROCm nightly at the start of the task (vllm-project/vllm@9c1d45c2, built 2026-11-19 06:12 UTC).
IMAGE_REPO=vllm/vllm-openai-rocm
IMAGE_TAG=nightly-9c1d45c21836370104d4eb51d3b46eda20719c47
IMAGE_DIGEST=sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
IMAGE="$IMAGE_REPO@$IMAGE_DIGEST"
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
MODEL=/perf_apps/data/models/DeepSeek-V4.1-Flash
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
CTR_PREFIX=llying-dsv41-
ETCD_PORT=${ETCD_PORT:-22379}
ROUTER_PORT=${ROUTER_PORT:-28000}
# dockerd (root) cannot read /home (root-squashed NFS); caches live on node-local scratch.
NODE_CACHE=/scratch/liyingli/dsv41-cache
# Inside containers (no iproute2 guaranteed) NODE_IP is passed in by exec_bg.
node_ip() { [ -n "${NODE_IP:-}" ] && echo "$NODE_IP" || ip -4 -o addr show dev fenic 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
ts() { date -u +%FT%TZ; }
