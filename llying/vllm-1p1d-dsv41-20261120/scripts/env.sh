# Shared settings for the DSv4.1-Flash Infera + vLLM 1P1D example work (AUS MI355X). Sourced by host-side scripts
# that run on the compute nodes (srun --overlap into jobs 36739 / 36738); never run heavy work on the login node.
W=/perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120
# Latest vLLM ROCm nightly when the work started (vllm-project/vllm@9c1d45c2, 2026-11-19 06:12 UTC); the Infera
# image is built on top of it, and the agg baseline runs on it unmodified.
BASE_IMAGE=vllm/vllm-openai-rocm@sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
BASE_TAG=nightly-9c1d45c21836370104d4eb51d3b46eda20719c47
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
MODEL=/perf_apps/data/models/DeepSeek-V4.1-Flash
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
CTR_PREFIX=llying-v1p1d-
ROUTER_PORT=${ROUTER_PORT:-28000}
# dockerd (root) cannot read /home (root-squashed NFS); caches live on node-local scratch.
NODE_CACHE=/scratch/liyingli/dsv41-cache
IX=/perf_apps/liyingli/bench_agentx/InferenceX
HFC=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/hf-cache
ts() { date -u +%FT%TZ; }
host_gids() { local g gid; for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && echo "--group-add $gid"; done; }
node_ip() { ip -4 -o addr show dev fenic 2>/dev/null | awk '{print $4}' | cut -d/ -f1; }
# Docker Hub login in ~/.docker/config.json carries an expired token that fails even public pulls: retry anonymously.
pull_image() {
    docker image inspect "$1" >/dev/null 2>&1 && return 0
    docker pull "$1" && return 0
    local d; d="$(mktemp -d)"; DOCKER_CONFIG="$d" docker pull "$1"; local rc=$?; rm -rf "$d"; return $rc
}
