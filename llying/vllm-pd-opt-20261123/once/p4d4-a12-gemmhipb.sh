# a12: decode dense GEMMs tuned with hipBLASLt solutions included (a4's tuning dropped them for time). 60 shapes:
# o_proj 6144x16384, q_b 16384x2048, fused_qkv_a 2624x6144, indexer wq_b 4096x2048 at the target-model M of FULL
# graphs (32..128 step 8, 136, 144). Tuner builds go to a throwaway JIT volume (v3-186's local disk filled up after
# a4's tuning). scripts/merge_tuned_gemm2.py keeps, per shape, the faster of the new row and what the engine reads now
# (o_proj into our glm52 table; shapes the image's tables already have via modified copies, same keys). Then side B of
# once/p4d4-pair2.sh with score-a12-B.env (S_GEMM_DIR set here, gate on). ONCE=p4d4-a12-gemmhipb, pinned next to a10/a11.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
O=$B/results/a12
IMAGE=infera-vllm:v0340-0c5e5e89
mkdir -p "$O"
cp "$0" "$O/once-a12.sh" 2>/dev/null
tw=$R/runs/a12-gemmtune
gd=$R/runs/a12-gemm
rm -rf "$tw" "$gd"; mkdir -p "$gd" && chmod 777 "$gd"
vol=vpd-tune-jit-${JOB:-x}
echo "$(date -u +%T) a12 tuning starts on $(hostname -s); disk: $(df -h / | tail -1 | awk '{print $4" free"}')" | tee "$O/tune-progress.txt"
TUNE_JIT_VOL=$vol TUNE_BATCH=8 TUNE_HIPB=1 bash "$B/scripts/tune_decode_gemm.sh" "$B/gemm-tune/glm52_decode_shapes_hipb60.txt" "$tw" 0,1,2,3,4,5,6,7 4200
trc=$?
docker volume rm -f "$vol" > /dev/null 2>&1
cp "$tw"/{glm52_decode_bf16_tuned_gemm.csv,tune-summary.txt,untuned.csv} "$O/" 2>/dev/null
tail -40 "$tw/tune.log" > "$O/tune-log-tail.txt" 2>/dev/null
echo "$(date -u +%T) tuning done rc=$trc; disk: $(df -h / | tail -1 | awk '{print $4" free"}')" | tee -a "$O/tune-progress.txt"
(( trc == 0 )) || { echo "$(date -u +%T) a12: tuning produced no table" | tee -a logs/summary.out; exit 0; }
ours=$R/runs/a4-gemmtune/glm52_decode_bf16_tuned_gemm.csv
[ -s "$ours" ] || ours=$B/gemm-tune/glm52_decode_bf16_tuned_gemm.csv
docker run --rm --network none -v "$B:$B:ro" -v "$tw:$tw:ro" -v "$gd:$gd" -v "$(dirname "$ours"):/ours:ro" --entrypoint python3 "$IMAGE" \
    "$B/scripts/merge_tuned_gemm2.py" "/ours/$(basename "$ours")" "$tw/glm52_decode_bf16_tuned_gemm.csv" "$gd" > "$O/merge.txt" 2>&1 \
    || { echo "$(date -u +%T) a12: merge failed"; exit 0; }
cp -r "$gd" "$O/gemm-merged" 2>/dev/null
echo "$(date -u +%T) merged: $(tail -1 "$O/merge.txt")" | tee -a "$O/tune-progress.txt"
SCORE_ENV=a12 PAIR_SIDES=B bash once/p4d4-pair2.sh
