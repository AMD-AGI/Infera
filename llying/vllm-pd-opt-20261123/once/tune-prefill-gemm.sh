# aiter BF16 GEMM table for the GLM-5.2 PREFILL (C152, prefill is the throughput ceiling). The a54 prefill log lists BF16
# GEMMs that fall back to hipBLASLt's default solution ("not found tuned config ... using torch solution:0"):
#   target layers (all PP stages): kv_b 28672 x 512 (M = chunk lengths up to 4096);
#   MTP draft layer (PP3, the slightly busiest stage): fused q_a+kv_a 2624 x 6144, o_proj 6144 x 16384, 4096 x 6144,
#   indexer wq_b 4096 x 2048, at any M up to 4096 (about 2700 distinct M each in one hour).
# aiter looks a shape up at M, then at get_padded_m(M, N, K, 0) and (.., 1); so 1) compute in the image, for every
# M in 1..4096, the coarsest padded M and tune only those; 2) tune with the decode wrapper (scripts/tune_decode_gemm.sh,
# the image's own tuner, hipBLASLt included); 3) write the prefill table and a decode + prefill combined table.
# The cluster kills a job whose GPUs stay under 25 % busy for 45 min (user, 11-26): the mean node GPU busy is logged
# every minute and the tuner container is stopped after 35 min under 25 %.
# Run by run_once.sbatch (ONCE=tune-prefill-gemm); results in $B/gemm-tune/prefill/.
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}   # run_once.sbatch does not export R
cd "$R" || exit 0
B=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-pd-opt-20261123
OUT=$B/gemm-tune/prefill
IMAGE=infera-vllm:v0340-0c5e5e89
JOB=${SLURM_JOB_ID:-$$}
mkdir -p "$OUT" && chmod 777 "$OUT"
cp "$0" "$OUT/once.sh" 2>/dev/null
log() { echo "$(date -u +%T) $*" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
log "job $JOB on $(hostname -s)"
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done

free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
log "GPU free GiB: ${free:-unknown}"
for x in $free; do (( x < 250 )) && { log "NOT STARTED, a GPU has only $x GiB free"; echo "$(date -u +%m%d-%T) tune-prefill-gemm: NOT STARTED (GPU busy)" >> logs/summary.out; exit 0; }; done

# 1. shapes: the coarsest padded M of every M in 1..4096, per (N, K)
timeout 600 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES=0 \
    -v "$OUT:/w" --entrypoint python3 "$IMAGE" -c '
from aiter.ops.gemm_op_common import get_padded_m
NK = [(28672, 512), (2624, 6144), (6144, 16384), (4096, 6144), (4096, 2048)]
rows = []
for n, k in NK:
    ms = sorted({get_padded_m(m, n, k, 1) for m in range(1, 4097)})
    ms0 = len({get_padded_m(m, n, k, 0) for m in range(1, 4097)})
    print(f"N {n} K {k}: {len(ms)} distinct padded M at gl=1 ({ms0} at gl=0): {ms[:6]} .. {ms[-4:]}", flush=True)
    rows += [f"{m},{n},{k}" for m in ms]
open("/w/shapes.txt", "w").write("\n".join(rows) + "\n")
print("shapes", len(rows))
' > "$OUT/shapes.log" 2>&1
cat "$OUT/shapes.log" >> "$OUT/summary.txt"
nshapes=$(wc -l < "$OUT/shapes.txt" 2>/dev/null || echo 0)
(( nshapes > 0 )) || { log "no shapes (shapes.log)"; echo "$(date -u +%m%d-%T) tune-prefill-gemm: no shapes" >> logs/summary.out; exit 0; }
(( nshapes <= 400 )) || { log "$nshapes shapes is more than about 1.5 h of tuning; stopping (shapes.log)"; echo "$(date -u +%m%d-%T) tune-prefill-gemm: too many shapes ($nshapes)" >> logs/summary.out; exit 0; }

# 2. tune, with the GPU-idle guard
( low=0; while sleep 60; do
      b=$(cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null | awk '{s+=$1; n++} END {if (n) printf "%.0f", s / n}')
      echo "$(date -u +%T) node GPU busy mean ${b:-?} %" >> "$OUT/gpu-busy.txt"
      if (( ${b:-0} < 25 )); then low=$((low + 1)); else low=0; fi
      if (( low >= 35 )); then log "GPU busy under 25 % for 35 min: stopping the tuner before the cluster watchdog"; docker rm -f vpd-gemmtune >/dev/null 2>&1; low=0; fi
  done ) &
GUARD=$!
trap 'kill $GUARD 2>/dev/null' EXIT
tw=$R/runs/tune-prefill-gemm-$JOB
log "tuning $nshapes shapes on 8 GPUs"
bash "$B/scripts/tune_decode_gemm.sh" "$OUT/shapes.txt" "$tw" 0,1,2,3,4,5,6,7 6000
trc=$?
cat "$tw/tune-summary.txt" >> "$OUT/summary.txt" 2>/dev/null
tail -40 "$tw/tune.log" > "$OUT/tune-log-tail.txt" 2>/dev/null
cp "$tw"/{untuned.csv,make-untuned.log} "$OUT/" 2>/dev/null
(( trc == 0 )) || { log "tuning produced no table (rc $trc)"; echo "$(date -u +%m%d-%T) tune-prefill-gemm: no table" >> logs/summary.out; exit 0; }
cp "$tw/glm52_decode_bf16_tuned_gemm.csv" "$OUT/glm52_prefill_bf16_tuned_gemm.csv"

# 3. decode + prefill combined table (one file mounts as model_configs/glm52_bf16_tuned_gemm.csv on either engine)
python3 - "$B/gemm-tune/glm52_decode_bf16_tuned_gemm.csv" "$OUT/glm52_prefill_bf16_tuned_gemm.csv" "$OUT/glm52_bf16_tuned_gemm_combined.csv" <<'EOF' >> "$OUT/summary.txt" 2>&1
import csv, sys
dec, pre, out = sys.argv[1:4]
rows, seen, header = [], set(), None
for path in (pre, dec):                      # prefill rows win on a key clash (none expected: different M / N, K)
    with open(path) as f:
        r = csv.DictReader(f)
        header = header or r.fieldnames
        assert r.fieldnames == header, f"{path}: header differs"
        for row in r:
            key = tuple(row[c] for c in ("gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"))
            if key not in seen:
                seen.add(key); rows.append(row)
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
print(f"combined table: {len(rows)} rows -> {out}")
EOF
log "done: $(( $(wc -l < "$OUT/glm52_prefill_bf16_tuned_gemm.csv") - 1 )) prefill rows"
echo "$(date -u +%m%d-%T) tune-prefill-gemm: $(( $(wc -l < "$OUT/glm52_prefill_bf16_tuned_gemm.csv") - 1 )) rows in $OUT" >> logs/summary.out
