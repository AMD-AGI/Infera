# GPU tests + microbenchmark of the packed FP4 dispatch (item 8: scripts/make_fp4dispatch_pack_overlay.py,
# VPD_FP4_DISPATCH_PACK=1 = one all-gather per MoE layer instead of four) before any S_FP4PACK=1 pair. Build the overlay
# from the image's two files (also the base FP4 overlay, to check rocm_aiter_moe.py is the same), then (the test loads
# the patched naive_dp_ep.py by path, so nothing is mounted over the image):
#   1. base   scripts/test_fp4dispatch.py on the pack overlay (switch-off regression, GPU 0);
#   2. layout switch off = base module state; row layout and receive views vs the 4 unpacked gathers (4 ranks by cat);
#   3. gpu    Triton pack kernel (quant=triton / hip) bit-exact vs aiter HIP quant + ids / weights, M 1..256, topk 8 / 9,
#             recv view == split, kernel times (GPU 0);
#   4. moe    aiter fused_moe on the strided recv views vs contiguous copies: is VPD_FP4_DISPATCH_PACK_RECV=view safe (GPU 0);
#   5. dist   GPUs 0-3 (a real aiter custom all-gather needs the 4 ranks; the name says 1gpu as the other steps use one):
#             packed 1-AG vs today's 4-AG FP4 path, bit-exact per rank (eager and CUDA-graph replay), per-layer time.
# ~10 min. Results in $B/results/fp4pack-1gpu (one file per step, summary.txt). ONCE=test-fp4pack-1gpu.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/fp4pack-1gpu
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
NDP=$SP/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py
RAM=$SP/vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
ov=$R/runs/fp4pack-1gpu-$JOB-ov
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint bash "$IMAGE" -c \
    "md5sum $NDP $RAM && python3 -B $B/scripts/make_fp4dispatch_pack_overlay.py $NDP $ov/naive_dp_ep.py $RAM $ov/rocm_aiter_moe.py \
     && python3 -B $B/scripts/make_fp4dispatch_overlay.py $NDP $ov/base_naive_dp_ep.py $RAM $ov/base_rocm_aiter_moe.py \
     && cmp $ov/rocm_aiter_moe.py $ov/base_rocm_aiter_moe.py && echo 'rocm_aiter_moe.py identical to the base FP4 overlay' \
     && { diff $ov/base_naive_dp_ep.py $ov/naive_dp_ep.py > $ov/pack-vs-base.diff; [ \$? -eq 1 ]; }" > "$O/build.txt" 2>&1 \
    || { echo "$(date -u +%m%d-%T) fp4pack-1gpu: overlay build failed, see $O/build.txt" >> logs/summary.out; exit 0; }
cp "$ov/pack-vs-base.diff" "$O/"
gpu() {   # NAME GPUS TIMEOUT CMD...: run CMD (bash -c) on HIP devices GPUS, output in $O/NAME.txt
    local name=$1 dev=$2 t=$3; shift 3
    timeout "$t" docker run --rm --init --network none --ipc host --shm-size 16g --device /dev/kfd --device /dev/dri "${gids[@]}" \
        -e HIP_VISIBLE_DEVICES="$dev" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
        -v "$jit:$SP/aiter/jit" -v "$B:$B:ro" -v "$ov:$ov:ro" --entrypoint bash "$IMAGE" -c "cd /tmp && $*" \
        2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/$name.txt"
    local rc=${PIPESTATUS[0]}
    echo "$name rc=$rc $(grep -a -E "passed|failed|error|Error" "$O/$name.txt" | tail -1)" | tee -a "$O/summary.txt"
}
: > "$O/summary.txt"
T="python3 $B/scripts/test_fp4dispatch_pack.py"
gpu base 0 600 "python3 $B/scripts/test_fp4dispatch.py $ov/naive_dp_ep.py $ov/rocm_aiter_moe.py"
gpu layout 0 600 "$T layout $ov/naive_dp_ep.py"
gpu gpu 0 1200 "$T gpu $ov/naive_dp_ep.py"
gpu moe 0 1200 "$T moe $ov/naive_dp_ep.py"
gpu dist 0,1,2,3 1200 "$T dist $ov/naive_dp_ep.py"
grep -h -E "^(FAIL|M |topk )|fused_moe" "$O"/{gpu,moe,dist}.txt >> "$O/summary.txt" 2>/dev/null
echo "$(date -u +%m%d-%T) fp4pack-1gpu tests in $O: $(grep -c ' rc=0 ' "$O/summary.txt")/5 rc=0" >> logs/summary.out
exit 0
