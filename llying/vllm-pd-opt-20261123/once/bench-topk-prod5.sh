# Indexer decode top-k on production-like rows (scripts/bench_topk_prod.py): aiter one-block (v0.34 default) vs
# vLLM's in-tree kernel vs the sglang #47872 Triton port from the A-line. 1 GPU, ~10 min. ONCE=bench-topk-prod.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
KF=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-decode-trackA-20261118/scripts/kf
O=$B/results/topk-prod5
IMAGE=infera-vllm:v0340-0c5e5e89
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
for dist in ar zsparse randn; do
    timeout 1500 docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 -e DIST=$dist -e FULLERR=1 -e CANDS="aiter triton_v" \
        -v "$jit:/usr/local/lib/python3.12/dist-packages/aiter/jit" -v "$B:$B:ro" -v "$KF:$KF:ro" --entrypoint python3 "$IMAGE" \
        "$B/scripts/bench_topk_prod.py" "$KF" "$B/scripts" 2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/bench-$dist.txt"
done
timeout 600 docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 -e VPD_TRITON_TOPK=1 \
    -v "$jit:/usr/local/lib/python3.12/dist-packages/aiter/jit" -v "$B:$B:ro" --entrypoint bash "$IMAGE" -c \
    "PYTHONPATH=$B/scripts python3 -c 'import vpd_triton_topk_decode as t; print(\"sort\", t._SORT)' && python3 $B/scripts/make_tritontopk_overlay.py $R/overlay-v3comm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py /tmp/ops.py && PYTHONPATH=$B/scripts python3 $B/scripts/test_tritontopk.py /tmp/ops.py" \
    2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/unit-test.txt"
echo "$(date -u +%m%d-%T) topk-prod bench in $O" >> logs/summary.out
exit 0
