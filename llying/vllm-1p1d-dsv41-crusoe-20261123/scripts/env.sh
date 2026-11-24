# Shared settings for the DeepSeek-V4.1-Flash Infera + vLLM 1P1D reproduction on crusoe spur (one MI355X VM node).
# Sourced by the job's scripts on the compute node; the login node only submits and reads logs.
W=/home/liyingli/bench_agentx/vllm-dsv41-crusoe-20261123
# The kit at llying/example/vllm_1p1d_dsv41_flash (054985cc1; engine image content identical to AUS c33de6336) plus
# the crusoe wrapper (cluster/cluster.crusoe.sh) and the MC_ENV_EXTRA pass-through.
REPO=/home/liyingli/bench_agentx/baseline/Infera-dsv41-kit
KIT=$REPO/examples/vllm_1p1d_dsv41_flash
BASE_IMAGE=vllm/vllm-openai-rocm:nightly-9c1d45c21836370104d4eb51d3b46eda20719c47@sha256:694108707f333787b1041caa0e948dd18b86f2450365a94e38819f026275ee32
RUST_IMAGE=docker.io/library/rust:1-bookworm
ETCD_IMAGE=quay.io/coreos/etcd:v3.5.14
# Built from the kit at 054985cc1e03. Later kit commits touch only examples/, which the image does not contain
# (build_image.sh marks a tree dirty only under infera/, rust/, deploy/docker/, pyproject.toml), so the tag stays.
IMAGE=infera-vllm-dsv41:054985cc1e03-nightly-9c1d45c2
IMAGE_TAR=/shared_nfs/liyingli/images/${IMAGE//[:\/]/_}.tar
MODEL_SRC=/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4.1-Flash
NODE_ROOT=/mnt/m2m_nobackup/liyingli_dsv41
MODEL_MOUNT=$NODE_ROOT/models
MODEL=$MODEL_MOUNT/DeepSeek-V4.1-Flash
SERVED=deepseek-ai/DeepSeek-V4.1-Flash
CTR=lyl-dsv41-pd
INFERENCEX_DIR=$W/InferenceX
GSM8K_PARQUET=/shared_nfs/huggingface_models/hub/datasets--openai--gsm8k/snapshots/740312add88f781978c0658806c59bc2815b9866/main/test-00000-of-00001.parquet
ts() { date -u +%FT%TZ; }
# Docker Hub pulls with a stale login token fail even for public images: retry anonymously.
pull_image() {
    docker image inspect "$1" > /dev/null 2>&1 && return 0
    timeout 1800 docker pull -q "$1" && return 0
    local d; d="$(mktemp -d)"; DOCKER_CONFIG="$d" timeout 1800 docker pull -q "$1"; local rc=$?; rm -rf "$d"; return $rc
}
