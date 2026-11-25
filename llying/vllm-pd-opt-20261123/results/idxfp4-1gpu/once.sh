# GPU tests (1 GPU, gfx950) of the MXFP4 indexer K cache overlay (scripts/make_idxfp4_overlay.py, vLLM #61985 backported
# to v0.34.0, target layers only), before any S_IDXFP4=1 pair: build the overlay from the image's files, then with
# its five files mounted over the image's own:
#   1. scripts/test_idxfp4.py cpu (layer selection, imports, hook guards) on the GPU platform this time;
#   2. the PR's kernel tests (overlay-idxfp4/tests, from v0.34.0 + #61985): MXFP4 K writes byte-exact in aiter's page
#      order and scored by aiter's paged_mxfp4_mqa_logits, the MXFP4 index query, and the FP8 fused_norm_rope / fused_q
#      tests of the same file as regression; the plan test (incl. the #61985 model check); v0.34.0's DSv4 paged
#      MXFP4 indexer test (aiter 0.1.28.post1's kernel on this layout);
#   3. scripts/test_idxfp4.py gpu with VPD_TOPK_IMPL=triton and =native: exact top-2048 through the decode hook.
# ~15 min. Results in $B/results/idxfp4-1gpu (one file per step, summary.txt). ONCE=test-idxfp4-1gpu.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/idxfp4-1gpu
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
ov=$R/runs/idxfp4-1gpu-$JOB-ov
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
docker run --rm --network none -v "$B:$B:ro" -v "$ov:$ov" --entrypoint python3 "$IMAGE" \
    "$B/scripts/make_idxfp4_overlay.py" "$SP" "$ov" > "$O/build.txt" 2>&1 \
    || { echo "$(date -u +%m%d-%T) idxfp4-1gpu: overlay build failed, see $O/build.txt" >> logs/summary.out; exit 0; }
idx_v=(); while IFS=: read -r f rel; do idx_v+=(-v "$ov/$f:$SP/$rel:ro"); done < "$ov/mounts.txt"
gpu() {   # NAME TIMEOUT CMD...: run CMD (bash -c) on GPU 0 with the overlay mounted, output in $O/NAME.txt
    local name=$1 t=$2; shift 2
    timeout "$t" docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 \
        -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v "$jit:$SP/aiter/jit" -v "$B:$B:ro" "${idx_v[@]}" --entrypoint bash "$IMAGE" -c "cd /tmp && $*" \
        2>&1 | grep -v -E "^\[aiter\]|queue-intercept" > "$O/$name.txt"
    local rc=${PIPESTATUS[0]}
    echo "$name rc=$rc $(grep -a -E "passed|failed|error" "$O/$name.txt" | tail -1)" | tee -a "$O/summary.txt"
}
: > "$O/summary.txt"
T=$B/overlay-idxfp4/tests
PT="python3 -m pytest -q -p no:cacheprovider -o addopts="
gpu select 600 "python3 $B/scripts/test_idxfp4.py cpu"
gpu norm_rope 1800 "$PT $T/test_fused_deepseek_v32_norm_rope.py"
gpu plan 600 "$PT $T/test_rocm_paged_mxfp4_indexer_plan.py"
gpu dsv4_mxfp4 1800 "$PT $T/test_rocm_paged_mxfp4_indexer.py"
gpu hook_triton 600 "PYTHONPATH=$B/scripts VPD_TOPK_IMPL=triton python3 $B/scripts/test_idxfp4.py gpu"
gpu hook_native 600 "VPD_TOPK_IMPL=native python3 $B/scripts/test_idxfp4.py gpu"
# item 9b (dense online FP8): which layers v0.34.0's --quantization-config targets / ignore (overlay-dfp8/) would quantize;
# needs only the model's config.json, no overlay
M=/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4
idx_v=(-v "$M:$M:ro")   # the overlay is not mounted for this step
gpu dfp8_map 600 "python3 $B/overlay-dfp8/test_dense_fp8_mapping.py $M $B/overlay-dfp8/dense_fp8_qcfg.json"
echo "$(date -u +%m%d-%T) idxfp4-1gpu tests in $O: $(grep -c ' rc=0 ' "$O/summary.txt")/7 rc=0" >> logs/summary.out
exit 0
