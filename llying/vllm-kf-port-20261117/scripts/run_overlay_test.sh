#!/usr/bin/env bash
# On the node: kf/test_overlay_logits.py in the image with the vLLM overlay bind-mounted, on GPU ${GPU:-3}.
R=/home/liyingli/bench_agentx/vllm-pd-20261117
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
V=/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
gids=()
for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
docker rm -f vpd-kf-ovl > /dev/null 2>&1
timeout "${TIMEOUT:-900}" docker run --rm --name vpd-kf-ovl --network host --ipc host \
    --device /dev/kfd --device /dev/dri "${gids[@]}" --security-opt seccomp=unconfined \
    -v "$R/kf:/kf" -v "vpd-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" \
    -v "$R/overlay/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/rocm_aiter_mla_sparse.py:ro" \
    -v "$R/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/rocm_gluon_paged_mqa_logits.py:ro" \
    -e HIP_VISIBLE_DEVICES="${GPU:-3}" -e VLLM_ROCM_USE_AITER=1 \
    --entrypoint python3 "$IMAGE" /kf/test_overlay_logits.py
rc=$?
docker rm -f vpd-kf-ovl > /dev/null 2>&1
echo "[test rc=$rc]"
exit 0
