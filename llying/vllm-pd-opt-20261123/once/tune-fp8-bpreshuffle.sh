# Tune aiter's a8w8 bpreshuffle table (the one vLLM's AiterPreshuffledPerTokenFp8ScaledMMLinearKernel needs) for the two
# GLM-5.2 online-FP8 (PTPC) dense shapes v0.34.0 leaves on hipBLASLt (results/dfp8-bench/table.md):
#   o_proj 6144x16384 (78 layers / step) and dense MLP gate_up 24576x6144 (3 layers), M = TUNE_MS (decode 1..256 per rank,
#   prefill 512..4096), with the image's own aiter (0.1.28.post1) tuner, all GPUs of the node. Then:
#   - merge: our rows, minus any key an image table already has (a duplicate key makes aiter raise), per (N, K) all M
#     or nothing (scripts/merge_tuned_fp8_bpreshuffle.py);
#   - rebuild aiter's runtime bpreshuffle modules (module_gemm_a8w8_bpreshuffle[_cktile].so): their CK / CK-Tile
#     kernel lookup is generated FROM the tuned table at JIT build time (optCompilerConfig.json blob_gen_cmd
#     --tune_file), and the serving JIT volume holds copies built from the image table, which would run our "ck" /
#     "cktile" rows with the heuristic default kernel;
#   - verify (scripts/verify_fp8_bpreshuffle.py, GPU 0, table mounted): kernel vLLM selects now, us vs the hipBLASLt
#     path at decode / prefill M, rel. error vs fp32.
# Everything runs on a throwaway JIT volume (removed at the end; TUNE_KEEP_VOL=1 keeps it), never the serving one.
# Output $B/gemm-tune/fp8-bpreshuffle/: a8w8_bpreshuffle_tuned_gemm_glm52_ptpc.csv, jit/*.so, merged-preview.csv,
# verify.md / verify.json, tune.log, prof_all.csv, summary.txt, and mounts.txt (src:dst, src relative to the dir) ONLY
# if the merge is complete and the verification passes (else mounts.txt.candidate). Use: S_FP8GEMM_DIR=<that dir> in
# once/p4d4-pair11.sh.
# Env: TUNE_MS, TUNE_GPUS (0,...,7), TUNE_LIBTYPE (ck,cktile,flydsl: what aiter's runtime dispatches; "asm" rows are
# ignored by gemm_a8w8_bpreshuffle), TUNE_BATCH (8), TUNE_TIMEOUT (8400 s), TUNE_GROUPED (1 = --shape_grouped),
# TUNE_KEEP_VOL (0). Expected: 1.5-2.5 h (tune-module JIT builds 20-40 min, 40 shapes on 8 GPUs 40-90 min, runtime
# module rebuild 10-25 min, verify ~10 min); bounded by the timeouts to ~3.3 h, inside run_once.sbatch's 4 h.
# Run by run_once.sbatch: --export=ALL,ONCE=tune-fp8-bpreshuffle.
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
OUT=$B/gemm-tune/fp8-bpreshuffle
IMAGE=infera-vllm:v0340-0c5e5e89
SP=/usr/local/lib/python3.12/dist-packages
NAME=a8w8_bpreshuffle_tuned_gemm_glm52_ptpc.csv
MS=${TUNE_MS:-8,16,32,48,64,96,128,192,256,1024,2048,4096}
gpus=${TUNE_GPUS:-0,1,2,3,4,5,6,7} lib=${TUNE_LIBTYPE:-ck} batch=${TUNE_BATCH:-8} tmo=${TUNE_TIMEOUT:-8400}
JOB=${JOB:-$$}
n=$(awk -F, '{print NF}' <<<"$gpus")
jit=vpd-aiter-jit-fp8tune-$JOB
mkdir -p "$OUT" && chmod 777 "$OUT"
ST=$OUT/stage-$JOB                                         # per job: files the container writes may not be deletable from the host
mkdir -p "$ST/jit" "$ST/verify" && chmod -R 777 "$ST"
rm -f "$OUT/mounts.txt" "$OUT/mounts.txt.candidate" "$OUT"/{verify.md,verify.json,merged-preview.csv,new_tuned.csv,$NAME} "$OUT"/jit/*.so          # a half-finished run never leaves a usable mounts.txt
cp "$0" "$OUT/once.sh" 2>/dev/null
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
log "job $JOB on $(hostname -s); Ms $MS; GPUs $gpus; libtype $lib; volume $jit; disk: $(df -h /var/lib/docker 2>/dev/null | tail -1 | awk '{print $4" free"}')"
free_gb=$(df -BG --output=avail /var/lib/docker 2>/dev/null | tail -1 | tr -dc 0-9)
if (( ${free_gb:-0} < 60 )); then log "NOT STARTED: only ${free_gb:-?} GB free under /var/lib/docker"; exit 0; fi
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
run() {   # run TIMEOUT HIP_DEVICES EXTRA_MOUNT CMD
    timeout -k 60 "$1" docker run --rm --init --name vpd-fp8tune --network none --device /dev/kfd --device /dev/dri "${gids[@]}" \
        -e HIP_VISIBLE_DEVICES="$2" -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e PYTHONHASHSEED=0 -e HSA_NO_SCRATCH_RECLAIM=1 \
        -e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
        -e GPU_ARCHS=gfx950 -e AITER_ROCM_ARCH=gfx950 \
        -v "$jit:$SP/aiter/jit" -v "$OUT:/w" -v "$B:$B:ro" $3 --entrypoint bash "$IMAGE" -c "$4"
    local rc=$?
    docker rm -f vpd-fp8tune >/dev/null 2>&1
    return $rc
}
cleanup() { [[ "${TUNE_KEEP_VOL:-0}" == 1 ]] || docker volume rm -f "$jit" >/dev/null 2>&1; }

# 1. tuner location (image: aiter_meta/csrc/...) and its options
FIND_T="T=\$(ls $SP/aiter_meta/csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py 2>/dev/null || find / -xdev -name gemm_a8w8_bpreshuffle_tune.py 2>/dev/null | head -1); [ -n \"\$T\" ] || exit 3; cd \$(dirname \$T)/../.. && echo tuner \$T"
run 600 0 "" "$FIND_T && python3 \$T --help && head -2 $SP/aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv && python3 -c 'import importlib.metadata as m; print(\"aiter\", m.version(\"amd-aiter\"))' || true" \
    > "$OUT/help.txt" 2>&1 || { log "tuner not found / --help failed (help.txt)"; exit 0; }
log "$(grep -m1 '^tuner' "$OUT/help.txt")"

# 2. input: M,N,K,q_dtype_w (gfx950 = OCP float8_e4m3fn, what vLLM's check passes as q_dtype_w)
{ echo "M,N,K,q_dtype_w"; for nk in 6144,16384 24576,6144; do for m in ${MS//,/ }; do echo "$m,$nk,torch.float8_e4m3fn"; done; done; } > "$OUT/untuned.csv"

# GPU-utilisation watchdog on this cluster (user, 11-26): a job whose GPUs stay under 25 % for 45 min is killed (313306 was,
# during the tuner's 8-way JIT build). So: (a) log the mean GPU busy of the node every 5 min; (b) pre-build the tuner module
# once on GPU 0 (one process, no 8-way lock contention), bounded to 35 min, and stop the job ourselves if it is not done;
# (c) then tune on all 8 GPUs with the module already built. GPU_ARCHS=gfx950 (the image builds gfx942;gfx950 = twice the
# compile; 314276's single-process build of the 159-instance tune module ran past the tuner's 1800 s task timeout).
( while sleep 300; do b=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | awk '{s+=$1; n++} END {if (n) printf "%.0f", s / n}'); log "node GPU busy mean ${b:-?} %"; done ) &
BUSY=$!; trap 'kill $BUSY 2>/dev/null; cleanup' EXIT
{ echo "M,N,K,q_dtype_w"; echo "64,6144,16384,torch.float8_e4m3fn"; } > "$OUT/prebuild.csv"
t0=$(date +%s)
run "${TUNE_PREBUILD_TIMEOUT:-2100}" 0 "" "$FIND_T && python3 \$T -i /w/prebuild.csv -o /w/stage-$JOB/prebuild_tuned.csv --mp 1 --batch 1 --libtype $lib --timeout ${TUNE_PREBUILD_TIMEOUT:-2100}" > "$OUT/prebuild.log" 2>&1
prc=$?
log "prebuild rc=$prc in $(( $(date +%s) - t0 )) s"
(( prc == 0 )) || { [[ $prc == 124 ]] && why="did not finish within ${TUNE_PREBUILD_TIMEOUT:-2100} s" || why="failed (rc $prc; the tuner's own per-task timeout counts the module build)"; log "tuner module pre-build $why; stopping before the 45-min GPU-idle watchdog (prebuild.log)"; exit 0; }

# 3. tune (the tuner writes its table every --batch shapes, so a timeout keeps what is done)
grp=""; [[ "${TUNE_GROUPED:-1}" == 1 ]] && grp="--shape_grouped"
t0=$(date +%s)
run "$tmo" "$gpus" "" "$FIND_T && python3 \$T -i /w/untuned.csv -o /w/stage-$JOB/new_tuned.csv -o2 /w/prof_all.csv --mp $n --batch $batch --libtype $lib $grp" \
    > "$OUT/tune.log" 2>&1
rc=$?
rows=$(( $(wc -l < "$ST/new_tuned.csv" 2>/dev/null || echo 1) - 1 ))
log "tune rc=$rc: $rows rows of $(( $(wc -l < "$OUT/untuned.csv") - 1 )) shapes in $(( $(date +%s) - t0 )) s"
(( rows > 0 )) || { log "no tuned rows (tune.log)"; exit 0; }

# 4. merge against the image's tables (rc 0 complete, 2 some (N, K) left out, 1 nothing usable)
run 600 0 "" "python3 $B/scripts/merge_tuned_fp8_bpreshuffle.py /w/stage-$JOB/new_tuned.csv /w/stage-$JOB --ms $MS" > "$OUT/merge.log" 2>&1
mrc=$?
log "merge rc=$mrc: $(tail -1 "$OUT/merge.log")"
[[ -s "$ST/$NAME" && -s "$ST/mounts.txt" ]] || { log "nothing to mount (merge.log)"; exit 0; }

# 5. rebuild the runtime bpreshuffle modules from the merged table (ours mounted), copy them out, then verify on GPU 0
cat > "$ST/build_bpreshuffle.py" <<'EOF'
import os, torch
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle_ck, gemm_a8w8_bpreshuffle_cktile
from aiter.jit.core import AITER_CONFIGS
print("merged table", AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE, flush=True)
M, N, K = 64, 6144, 16384
x = torch.randn(M, K, device="cuda").to(dtypes.fp8)
w = shuffle_weight((torch.randn(N, K, device="cuda") * 0.1).to(dtypes.fp8), layout=(16, 16))
xs, ws = torch.ones(M, 1, device="cuda"), torch.ones(N, 1, device="cuda")
y = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
for f in (gemm_a8w8_bpreshuffle_ck, gemm_a8w8_bpreshuffle_cktile):   # first call JIT-builds the module
    try:
        f(x, w, xs, ws, y, 0); torch.cuda.synchronize(); print(f.__name__, "ran", flush=True)
    except Exception as e:
        print(f.__name__, "call failed after build:", type(e).__name__, str(e)[:200], flush=True)
EOF
J=$SP/aiter/jit
MODS="module_gemm_a8w8_bpreshuffle module_gemm_a8w8_bpreshuffle_cktile"
CSVM="-v $ST/$NAME:$SP/aiter/configs/model_configs/$NAME:ro"
t1=$(date +%s)
run 3600 0 "$CSVM" "set -e; for m in $MODS; do rm -f $J/\$m.so; rm -rf $J/build/\$m; done
    python3 /w/stage-$JOB/build_bpreshuffle.py
    for m in $MODS; do test -s $J/\$m.so; cp $J/\$m.so /w/stage-$JOB/jit/; done; md5sum /w/stage-$JOB/jit/*.so
    cd /tmp && python3 $B/scripts/verify_fp8_bpreshuffle.py /w/stage-$JOB/verify" > "$OUT/verify.log" 2>&1
vrc=$?
log "rebuild + verify rc=$vrc in $(( $(date +%s) - t1 )) s: $(grep -a '^SUMMARY' "$OUT/verify.log" | tail -1)"
grep -a -A3 "^## kernel per" "$OUT/verify.log" | tee -a "$OUT/summary.txt"

# 6. publish
cp "$ST/$NAME" "$ST/merged-preview.csv" "$ST/new_tuned.csv" "$OUT/" 2>/dev/null
cp "$ST/verify/verify.md" "$ST/verify/verify.json" "$OUT/" 2>/dev/null
mkdir -p "$OUT/jit"; cp "$ST"/jit/*.so "$OUT/jit/" 2>/dev/null
mt=$OUT/mounts.txt.candidate
{ cat "$ST/mounts.txt"; for m in $MODS; do [ -s "$OUT/jit/$m.so" ] && echo "jit/$m.so:$J/$m.so"; done; } > "$mt"
nso=$(ls "$OUT"/jit/*.so 2>/dev/null | wc -l)
if [[ "$mrc" == 0 && "$vrc" == 0 && "$nso" == 2 ]]; then
    mv "$mt" "$OUT/mounts.txt"; log "OK: $OUT/mounts.txt ($(wc -l < "$OUT/mounts.txt") mounts)"
else
    log "NOT PUBLISHED as mounts.txt (merge rc $mrc, verify rc $vrc, modules $nso/2): see $mt, merge.log, verify.log"
fi
echo "$(date -u +%m%d-%T) tune-fp8-bpreshuffle job $JOB: $(tail -1 "$OUT/summary.txt")" >> logs/summary.out
exit 0
