#!/usr/bin/env bash
# sglang#48302 end-to-end A/B on one MI355X node: two aggregated GLM-5.2-MXFP4 servers side by side
# (TP4 / DP4 attention, EAGLE 3/1/4, custom all-reduce on), the same nightly image; arm "main" runs main's
# Python (src-main), arm "pr" the PR's (src-pr), via PYTHONPATH.
#   slot 0: GPUs 0-3, CPUs of NUMA 0     slot 1: GPUs 4-7, CPUs of NUMA 1
#   cfg P: image defaults (SGLANG_USE_ROCM700A=1: DP gather / combine by all-reduce, custom AR in the graphs)
#   cfg E: E11 (SGLANG_USE_ROCM700A=0: aiter all-gather / reduce-scatter in the graphs)
# The phases run in step on both slots: start-up -> correctness (GSM8K 400 @64 and 16k long context 64 @32
# at once, so DP ranks hold sequences on both sides of index_topk) -> throughput (bench_serving from the
# image's own sglang, identical client for both arms) -> stop.
# usage: ab_node.sh <node> <cfg P|E> <round> <arm slot0> <arm slot1>
set -uo pipefail
AB=/perf_apps/liyingli/sgl48302-ab-20261118
IMG=/apps/liyingli/squash/lmsysorg_sglang-rocm_v0.5.23-rocm720-mi35x-20261117_sha256_e016f33423250be30bde9e0888d159b67b01e00944de1edc11f36dd1f3dd0269.sqsh
MODEL=/perf_apps/data/models/GLM-5.2-MXFP4
SHAREGPT=/perf_apps/data/models/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json
JOB=${JOB:-36659}
node=$1 cfg=$2 round=$3
arms=("$4" "$5")
gpus=(0,1,2,3 4,5,6,7)
cpus=(0-63,128-191 64-127,192-255)
ports=(30010 30020)
OUT=$AB/runs/r$round-$cfg
mkdir -p "$OUT"
say() { echo "$(date -u +%FT%TZ) [r$round $cfg $node] $*" | tee -a "$OUT/summary.txt" >> "$AB/summary.txt"; }
cstep() {
    srun --jobid "$JOB" --overlap --nodes 1 --ntasks 1 --nodelist "$node" \
        --container-image "$IMG" --no-container-entrypoint --no-container-mount-home \
        --container-mounts /perf_apps:/perf_apps "$@"
}
nstep() { srun --jobid "$JOB" --overlap --nodes 1 --ntasks 1 --nodelist "$node" "$@"; }
tag() { echo "s$1-${arms[$1]}"; }
vram_used_mb() { nstep amd-smi monitor -v 2>/dev/null | awk 'NR>1 {gsub(/ MB/,"",$0); s+=$3} END {print s+0}'; }

# Leftovers from an earlier round would skew both arms: wait for the GPUs to drain.
for t in $(seq 1 40); do
    u=$(vram_used_mb); [ "${u:-0}" -lt 8000 ] && break
    [ "$t" = 1 ] && say "waiting for GPUs to drain (VRAM used ${u} MB)"
    nstep pkill -f sglang.launch_server 2>/dev/null
    sleep 15
done
say "start: slot0=${arms[0]} slot1=${arms[1]} (main $(cat $AB/src-main/SOURCE_REV), pr $(cat $AB/src-pr/SOURCE_REV)), VRAM used $(vram_used_mb) MB"

common=$(cat "$AB/server_args.txt")
pids=()
for s in 0 1; do
    envs="PYTHONPATH=$AB/src-${arms[$s]}/python HIP_VISIBLE_DEVICES=${gpus[$s]} SGLANG_SET_CPU_AFFINITY=0 HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1 SAFETENSORS_FAST_GPU=1 NCCL_IB_DISABLE=1 PYTHONHASHSEED=0"
    [ "$cfg" = E ] && envs="$envs SGLANG_USE_ROCM700A=0"
    cstep bash -c "cd /tmp && export $envs && python3 -c 'import sglang; print(\"SGLANG_FILE\", sglang.__file__, \"ROCM700A\", __import__(\"os\").environ[\"SGLANG_USE_ROCM700A\"])' && exec taskset -c ${cpus[$s]} python3 -m sglang.launch_server $common --port ${ports[$s]}" \
        > "$OUT/$(tag $s).server.log" 2>&1 &
    pids+=($!)
done
trap 'kill ${pids[*]} 2>/dev/null; sleep 10; nstep pkill -f sglang.launch_server 2>/dev/null' EXIT
ready() { curl -s -m 5 -o /dev/null -w '%{http_code}' "http://$node:${ports[$1]}/v1/models" 2>/dev/null; }
t0=$(date +%s)
for t in $(seq 1 120); do  # up to 60 min
    [ "$(ready 0)" = 200 ] && [ "$(ready 1)" = 200 ] && break
    if ! kill -0 "${pids[0]}" 2>/dev/null || ! kill -0 "${pids[1]}" 2>/dev/null; then break; fi
    sleep 30
done
say "ready after $(( $(date +%s) - t0 )) s: slot0=$(ready 0) slot1=$(ready 1)"
for s in 0 1; do
    log="$OUT/$(tag $s).server.log"
    say "$(tag $s) start-up: $(grep -a -m1 SGLANG_FILE "$log") dual_graph_lines=$(grep -ac 'DSA dual-graph enabled' "$log") tracebacks=$(grep -ac Traceback "$log")"
done
if [ "$(ready 0)" != 200 ] || [ "$(ready 1)" != 200 ]; then say "a server did not come up; stopping"; exit 1; fi

correctness() {  # correctness <slot>
    local s=$1 url="http://$node:${ports[$1]}" t; t=$(tag "$1")
    cstep bash -c "cd $AB/probe && python3 gsm8k_probe.py gsm8k_probe_prompts.jsonl $url glm $OUT/$t.gsm8k.jsonl --limit 400 --concurrency 64" > "$OUT/$t.gsm8k.log" 2>&1 &
    local p1=$!
    cstep bash -c "cd $AB/probe && python3 longctx_probe.py gsm8k_probe_prompts.jsonl $url glm $OUT/$t.longctx.jsonl --ctx-tokens 16000 --n 64 --concurrency 32" > "$OUT/$t.longctx.log" 2>&1 &
    local p2=$!
    wait $p1 $p2
}
say "correctness phase"
correctness 0 & c0=$!
correctness 1 & c1=$!
wait $c0 $c1
for s in 0 1; do
    t=$(tag $s)
    say "$t gsm8k: $(cstep python3 $AB/probe/probe_stats.py "$OUT/$t.gsm8k.jsonl" --review "$OUT/$t.review.txt" 2>&1 | tail -1)"
    say "$t longctx: $(grep -aE '^(needle|gsm8k): n=' "$OUT/$t.longctx.log" | tr '\n' ' ')"
done

perf() {  # perf <slot> <name> <isl> <osl> <num> <conc> <seed>
    local s=$1 name=$2 t; t=$(tag "$1")
    curl -s -m 60 -X POST "http://$node:${ports[$s]}/flush_cache" > /dev/null
    cstep bash -c "cd /tmp && python3 -m sglang.benchmark.serving --backend sglang --host $node --port ${ports[$s]} --tokenizer $MODEL \
        --dataset-name random --dataset-path $SHAREGPT --random-input-len $3 --random-output-len $4 --random-range-ratio 0.8 \
        --num-prompts $5 --max-concurrency $6 --seed $7 --warmup-requests 8 --output-file $OUT/$t.$name.jsonl" > "$OUT/$t.$name.log" 2>&1
}
for spec in "p1k 1024 1024 768 64 1" "p8k 8192 1024 192 32 2"; do
    set -- $spec
    say "throughput phase $1 (ISL $2 / OSL $3, ratio 0.8, $4 prompts @ $5)"
    perf 0 "$@" & q0=$!
    perf 1 "$@" & q1=$!
    wait $q0 $q1
    for s in 0 1; do
        t=$(tag $s)
        say "$t $1: $(python3 - "$OUT/$t.$1.jsonl" <<'EOF'
import json, sys
try:
    r = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
except Exception as e:
    print(f"no result ({e})"); sys.exit()
print(f"done={r['completed']} out_tok/s={r['output_throughput']:.1f} total_tok/s={r['total_throughput']:.1f} "
      f"req/s={r['request_throughput']:.3f} tpot_mean={r['mean_tpot_ms']:.2f} itl_p50={r['median_itl_ms']:.2f} "
      f"itl_p99={r['p99_itl_ms']:.2f} ttft_mean={r['mean_ttft_ms']:.0f} accept={r.get('accept_length')} dur={r['duration']:.0f}s")
EOF
)"
    done
done
for s in 0 1; do
    log="$OUT/$(tag $s).server.log"
    say "$(tag $s) server log: tracebacks=$(grep -ac Traceback "$log") nan_or_error=$(grep -aciE 'nan detected|RuntimeError|CUDA error|HIP error' "$log")"
done
say "done"
