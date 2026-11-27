# Site and recipe settings for the GLM-5.2-MXFP4 P4D4 kit (one MI355X node, Infera + vLLM). Edit the site block for
# another cluster; everything below it is the recipe. Sourced by engine/up.sh, agentx.sh, eval.sh, down.sh.

# ---- site (AUS MI355X) ----
MODEL=${MODEL:-/perf_apps/data/models/GLM-5.2-MXFP4}
SERVED=${SERVED:-glm5.2-mxfp4}
IXM=${IXM:-/perf_apps/liyingli/bench_agentx/InferenceX-main-d490919c}   # InferenceX main checkout (client + eval)
PROBE_DIR=${PROBE_DIR:-/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127/results/probe-$(hostname -s)}
# KV memory registration, decided per node by probe/probe_node.sh (results/probe-<node>/):
#   GEM=1  KV pools in GEM BOs (amdgpu 6.19, or 6.16 without peer memory: dma-buf-pinned VRAM is charged twice)
#   GEM=0  plain hipMalloc registered through peer memory (amdgpu 6.16 with peer memory)
GEM=${GEM:-$(cat "$PROBE_DIR/gem" 2>/dev/null || echo 1)}
HOST_LIBIONIC=${HOST_LIBIONIC:-$(cat "$PROBE_DIR/libionic" 2>/dev/null || echo 0)}
# RDMA NIC of host GPU 0..7 (nearest on PCIe), from the probe; crusoe and AUS both map GPU i -> ionic_i.
GPU_NICS=${GPU_NICS:-$(awk '{print $2}' "$PROBE_DIR/gpu-nic.txt" 2>/dev/null | paste -sd, -)}
GPU_NICS=${GPU_NICS:-ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7}

# ---- recipe ----
IMAGE=${IMAGE:?IMAGE (an engine image built by docker/build.sh)}
MTP_N=${MTP_N:-4}
# InferenceX golden AL (glm-5.2-fp8 MTP, thinking on): the synthetic acceptance a score must use for MTP_N.
golden_al() { case $1 in 1) echo 1.84;; 2) echo 2.50;; 3) echo 2.99;; 4) echo 3.33;; 5) echo 3.61;; 6) echo 3.78;; 7) echo 3.91;; 8) echo 4.06;; esac; }
CPU_BYTES=${CPU_BYTES:-1900000000000}             # prefill CPU KV tier (vLLM OffloadingConnector, LRU)
# Numerics (InferenceX: allowed when evals pass; the MTP draft layer 78 keeps its shipped precision):
#   online FP8 per-channel on the target dense projections (layers 0-77), MXFP4 indexer K cache (target layers)
K_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ "${NUMERICS:-1}" == 1 ]]; then
    NUMERICS_ARGS="--attention-config '{\"indexer_kv_dtype\":\"mxfp4\"}' --quantization-config '$(python3 -c 'import json, sys; print(json.dumps(json.load(open(sys.argv[1])), separators=(",", ":")))' "$K_DIR/numerics/dense_fp8_qcfg.json")'"
fi
ROUTER_ARGS=${ROUTER_ARGS:---router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on}
ROUTER_ENV=${ROUTER_ENV:-INFERA_VLLM_MOONCAKE_SERIAL=1}

ts() { date -u +%FT%TZ; }
pull_image() {   # anonymous retry: an expired Docker Hub token in ~/.docker/config.json fails even public pulls
    docker image inspect "$1" > /dev/null 2>&1 && return 0
    docker pull -q "$1" && return 0
    local d; d="$(mktemp -d)"; DOCKER_CONFIG="$d" docker pull -q "$1"; local rc=$?; rm -rf "$d"; return $rc
}
