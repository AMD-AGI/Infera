#!/usr/bin/env bash
# On the node: run one kf/ python script in the image on one GPU, optionally with an overlay's vLLM files mounted.
#   run_kf.sh GPU OUTFILE SCRIPT [args...]     env OVL (overlay dir with vllm/..., e.g. $R/overlay-v3), IMAGE,
#                                               EXTRA_ENV (space-separated K=V), TIMEOUT (900 s)
R=/home/liyingli/bench_agentx/vllm-trackA-20261118
IMAGE=${IMAGE:-vllm/vllm-openai-rocm:v0.34.0}
P=/usr/local/lib/python3.12/dist-packages/vllm
gpu=$1 outf=$2 script=$3; shift 3
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
mounts=(-v "$R/kf:/kf" -v "vta-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit")
if [[ -n "${OVL:-}" ]]; then
  while IFS= read -r f; do mounts+=(-v "$OVL/$f:$P/${f#vllm/}:ro"); done < <(cd "$OVL" && find vllm -name '*.py')
fi
envs=(-e HIP_VISIBLE_DEVICES="$gpu" -e VLLM_ROCM_USE_AITER=1)
for kv in ${EXTRA_ENV:-}; do envs+=(-e "$kv"); done
name=vta-kf-$gpu
docker rm -f "$name" > /dev/null 2>&1
timeout "${TIMEOUT:-900}" docker run --rm --name "$name" --network host --ipc host \
    --device /dev/kfd --device /dev/dri "${gids[@]}" --security-opt seccomp=unconfined \
    "${mounts[@]}" "${envs[@]}" -w /kf --entrypoint python3 "$IMAGE" "/kf/$script" "$@" > "$outf" 2>&1
rc=$?
docker rm -f "$name" > /dev/null 2>&1
echo "[$script rc=$rc]" >> "$outf"
exit 0
