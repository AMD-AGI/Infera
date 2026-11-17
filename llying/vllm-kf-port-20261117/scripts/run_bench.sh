#!/usr/bin/env bash
# On the node: run kf/bench_sparse_mla.py, one candidate per GPU in parallel, into kf/results/<tag>/<candidate>.txt.
#   run_bench.sh TAG [rows ...]     env GPUS (0 1 2 3), CANDS (asm aiter_tr sgl47278 kf), TIMEOUT (1200 s)
# Also runs kf/bench_paged_logits.py on GPU LOGITS_GPU (4; empty to skip) into paged_logits.txt.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
tag=${1:?tag}; shift
out=$R/kf/results/$tag
mkdir -p "$out"
read -r -a gpus <<< "${GPUS:-0 1 2 3}"
read -r -a cands <<< "${CANDS:-asm aiter_tr sgl47278 kf}"
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
for i in "${!cands[@]}"; do
    c=${cands[$i]} g=${gpus[$((i % ${#gpus[@]}))]}
    docker rm -f "vpd-kf-$c" > /dev/null 2>&1
    timeout "${TIMEOUT:-1200}" docker run --rm --name "vpd-kf-$c" --network host --ipc host \
        --device /dev/kfd --device /dev/dri "${gids[@]}" --security-opt seccomp=unconfined \
        -v "$R/kf:/kf" -v "vpd-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" \
        -e HIP_VISIBLE_DEVICES="$g" -e ONLY="$c" -e VLLM_ROCM_USE_AITER=1 \
        --entrypoint python3 "$IMAGE" /kf/bench_sparse_mla.py "$@" > "$out/$c.txt" 2>&1 &
done
if [[ -n "${LOGITS_GPU-4}" ]]; then
    docker rm -f vpd-kf-logits > /dev/null 2>&1
    timeout "${TIMEOUT:-1200}" docker run --rm --name vpd-kf-logits --network host --ipc host \
        --device /dev/kfd --device /dev/dri "${gids[@]}" --security-opt seccomp=unconfined \
        -v "$R/kf:/kf" -v "vpd-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" \
        -e HIP_VISIBLE_DEVICES="${LOGITS_GPU-4}" --entrypoint python3 "$IMAGE" /kf/bench_paged_logits.py \
        > "$out/paged_logits.txt" 2>&1 &
    cands+=(logits)
fi
wait
for c in "${cands[@]}"; do docker rm -f "vpd-kf-$c" > /dev/null 2>&1; done
for c in "${cands[@]}"; do f=$out/$c.txt; [[ $c == logits ]] && f=$out/paged_logits.txt; echo "== $c"; grep -v -E "^\s*$" "$f" | grep -E "^(device|rows|batch|[0-9]+ [|(])|Error|error" | head -20; done
exit 0
