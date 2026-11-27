# AUS MI355X site settings for the GLM-5.2 P4D4 (Infera + vLLM) work. Sourced by scripts that run on the compute nodes
# (holder items); nothing heavy runs on the login node (16 cores, 62 GB).
W=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127
REF=/perf_apps/liyingli/bench_agentx/infera-vllm-glm52-ref          # Infera dev/pd_opt/vllm_glm52_agentx (crusoe recipe)
REF_COMMIT=0fcce3334ed4767d772bc518b542cdbf9dcff70c
MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
IX=/perf_apps/liyingli/bench_agentx/InferenceX                       # local clone; scripts check out what they need into worktrees
# v0.34.0 image of the crusoe recipe (llying/vllm-image-v0340-20261117/Dockerfile at REF_COMMIT)
IMAGE_V0340=infera-vllm:v0340-${REF_COMMIT:0:8}
# Latest vLLM ROCm nightly when this work started (vllm-project/vllm@de7cf7f8, 2026-11-27 06:06 UTC, ROCm 10.0)
NIGHTLY_BASE=vllm/vllm-openai-rocm:nightly-rocm100-de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3
NIGHTLY_COMMIT=de7cf7f8cb5dc47d2e65207e0fabbe2c8f0d73b3
# dockerd (root) cannot read /home (root-squashed NFS): every path given to a container lives under /perf_apps.
NODE_CACHE=/tmp/liyingli-p4d4                                         # node-local scratch (no /scratch on these nodes)
ts() { date -u +%FT%TZ; }
# Docker Hub login in ~/.docker/config.json carries an expired token that fails even public pulls: retry anonymously.
pull_image() {
    docker image inspect "$1" > /dev/null 2>&1 && return 0
    docker pull -q "$1" && return 0
    local d; d="$(mktemp -d)"; DOCKER_CONFIG="$d" docker pull -q "$1"; local rc=$?; rm -rf "$d"; return $rc
}
# GEM decision recorded by probe/probe_node.sh: results/probe-<node>/gem (1 = KV pools in GEM BOs, 0 = plain hipMalloc).
node_gem() { cat "$W/results/probe-$(hostname -s)/gem" 2>/dev/null || echo 1; }
