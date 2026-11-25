# Item 9b (dense online FP8) microbenchmark, 1 GPU (gfx950): GLM-5.2 TP1 dense target GEMMs (fused_qkv_a, q_b, o_proj,
# indexer wq_b, layer 0-2 MLP gate_up / down; shapes from the model's config.json), BF16 as decode runs it today
# (vLLM's UnquantizedLinearMethod -> rocm_unquantized_gemm, with our decode BF16 table mounted exactly as S_GEMM=1 in
# p4d4-pair6.sh) vs v0.34.0's online fp8_per_channel (Fp8PtpcOnlineLinearMethod, kernel chosen by vLLM's ROCm list,
# per-token activation quant included), plus aiter CK bpreshuffle untuned default (--ck) as headroom.
# M = 1..256 (decode rows per DP rank incl. MTP) and 2048 / 4096 (prefill). Median us, rel. error vs fp32, per-step
# sum at M = 56. Read-only image, no overlay. ~15-25 min (FlyDSL / CK JIT on first use goes to the shared JIT volume).
# Results in $B/results/dfp8-bench (table.md, bench.json, run.txt). ONCE=bench-dense-fp8-1gpu.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/dfp8-bench
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
M=/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4
mkdir -p "$O"
cp "$0" "$O/once.sh" 2>/dev/null
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
image_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
jit=vpd-aiter-jit-${image_id:7:12}
# the decode BF16 GEMM table, same source and mount point as p4d4-pair6.sh (S_GEMM=1)
gemm=$R/runs/a4-gemmtune/glm52_decode_bf16_tuned_gemm.csv
[ -s "$gemm" ] || gemm=$B/gemm-tune/glm52_decode_bf16_tuned_gemm.csv
ov=$R/runs/dfp8-bench-$JOB-ov
rm -rf "$ov"; mkdir -p "$ov" && chmod 777 "$ov"
cp "$gemm" "$ov/glm52_decode_bf16_tuned_gemm.csv"
echo "table $gemm ($(wc -l < "$gemm") lines) image $image_id jit $jit" > "$O/config.txt"
# decode container env (deploy/vllm_up.sh) minus the networking ones
timeout 3600 docker run --rm --init --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONHASHSEED=0 -e HSA_NO_SCRATCH_RECLAIM=1 \
    -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
    -v "$jit:$SP/aiter/jit" -v "$B:$B:ro" -v "$M/config.json:$M/config.json:ro" -v "$O:$O" \
    -v "$ov/glm52_decode_bf16_tuned_gemm.csv:$SP/aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv:ro" \
    --entrypoint bash "$IMAGE" -c "cd /tmp && python3 $B/scripts/bench_dense_fp8.py $M $O --ck" > "$O/run.txt" 2>&1
rc=$?
grep -a -v -E "^\[aiter\]|queue-intercept" "$O/run.txt" > "$O/run-filtered.txt"
s=$(grep -a "^SUMMARY" "$O/run.txt" | tail -1)
echo "$(date -u +%m%d-%T) dfp8-bench rc=$rc in $O: ${s:-no summary, see run.txt}" | tee -a "$O/summary.txt" >> logs/summary.out
rm -rf "$ov"
exit 0
