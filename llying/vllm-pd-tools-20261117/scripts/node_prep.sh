#!/usr/bin/env bash
# Prepare a fresh node for the P4D4 rounds: stage the model to the local NVMe, build infera-vllm:v0340-0c5e5e89 from
# commit 0c5e5e89 (git archive), and check that the vLLM files the decode overlays replace are the stock v0.34.0 ones
# (the image applies no vLLM source patch). Prints one "prep ok" / "prep FAILED" line at the end.
#   node_prep.sh
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
h=$(hostname -s)
REPO=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52
IMAGE=infera-vllm:v0340-0c5e5e89
BASE=vllm/vllm-openai-rocm:v0.34.0@sha256:3cb773d23f063afa86291ef75773cdd486154ddea94c213300f4ed7cc42b7af1
bash "$R/scripts/stage_model.sh" /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4 > "$R/logs/stage-$h.log" 2>&1 &
st=$!
if ! docker image inspect "$IMAGE" > /dev/null 2>&1; then
    d=/mnt/m2m_nobackup/liyingli_vpd/build-0c5e5e89
    rm -rf "$d"; mkdir -p "$d"
    git -C "$REPO" archive 0c5e5e89 | tar -x -C "$d"
    ( cd "$d" && docker build --progress=plain -f llying/vllm-image-v0340-20261117/Dockerfile -t "$IMAGE" . > "$R/logs/build-$h-0c5e5e89.log" 2>&1 )
    echo "build rc=$?"
fi
wait $st; st_rc=$?
tail -1 "$R/logs/stage-$h.log"
V=/usr/local/lib/python3.12/dist-packages/vllm
files="$V/v1/attention/ops/rocm_aiter_mla_sparse.py $V/distributed/device_communicators/cuda_communicator.py"
docker image inspect "$BASE" > /dev/null 2>&1 || docker pull -q "$BASE" > /dev/null 2>&1
a=$(docker run --rm --network none --entrypoint md5sum "$IMAGE" $files 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
b=$(docker run --rm --network none --entrypoint md5sum "$BASE" $files 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
echo "overlay targets: image $a / base $b"
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | grep -E "infera-vllm|vllm-openai-rocm|etcd"
if docker image inspect "$IMAGE" > /dev/null 2>&1 && (( st_rc == 0 )) && [[ "$a" == "$b" && -n "$a" ]]; then
    echo "prep ok on $h"
else
    echo "prep FAILED on $h (stage rc=$st_rc)"
fi
exit 0
