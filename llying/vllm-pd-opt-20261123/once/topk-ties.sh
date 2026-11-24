# Which tied elements each decode top-k picks (scripts/topk_tie_positions.py). 1 GPU, a few minutes. ONCE=topk-ties.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/topk-ties
IMAGE=infera-vllm:v0340-0c5e5e89
mkdir -p "$O"
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
timeout 900 docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 \
    -v "vpd-aiter-jit-${image_id:7:12}:/usr/local/lib/python3.12/dist-packages/aiter/jit" -v "$B:$B:ro" --entrypoint python3 "$IMAGE" \
    "$B/scripts/topk_tie_positions.py" "$B/scripts" 2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/tie-positions.txt"
echo "$(date -u +%m%d-%T) topk-ties in $O" >> logs/summary.out
exit 0
