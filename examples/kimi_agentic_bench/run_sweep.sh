#!/bin/bash
# Concurrency sweep for the Kimi agentic (multi-turn) benchmark.
#
# Drives vLLM's multi-turn tool (benchmark_serving_multi_turn.py) against a running
# endpoint (the PD router, or the single-node baseline), sweeping the number of
# concurrent sessions and recording per-point throughput, latency and cache-hit rate.
#
# Real cache-hit rate comes from the prefill engine's /metrics prefix_cache_* delta
# (the tool's own approx_cached_percent needs a server-returned cached_tokens field
# that the router does not populate).
#
# Usage:
#   SESSIONS="1 2 4 8 16 32 64 96 128" NGPU=12 \
#   ROUTER=http://<router-host>:8100 \
#   PREFILL_METRICS=http://<prefill-host>:30000/metrics \
#   TOOL_DIR=/path/to/vllm/benchmarks/multi_turn \
#   TXT=/path/to/corpus.txt \
#   OUT=./results/pd bash run_sweep.sh
set -uo pipefail

# vLLM's multi-turn benchmark tool (from the vllm repo: benchmarks/multi_turn/).
TOOL_DIR=${TOOL_DIR:?set TOOL_DIR to vllm/benchmarks/multi_turn}
PY=${PY:-python3}
# Tokenizer/model: a local path or HF id; must match the served model's tokenizer.
MODEL_TOK=${MODEL_TOK:-/models/Kimi-K2.6-MXFP4}
SERVED=${SERVED:-kimi2.6-mxfp4}
# A large plain-text corpus used as the token source (e.g. a Project Gutenberg dump).
# It must contain more distinct tokens than a single session consumes; large session
# counts (N>=64) need a multi-million-token file.
TXT=${TXT:?set TXT to a large plain-text corpus}

ROUTER=${ROUTER:-http://127.0.0.1:8100}
PREFILL_METRICS=${PREFILL_METRICS:-http://127.0.0.1:30000/metrics}
NGPU=${NGPU:-8}
OUT=${OUT:-./results}
SESSIONS="${SESSIONS:-1 2 4 8 16 32 64 96 128}"

# Workload knobs (defaults reproduce the published multi-turn agentic settings).
COMMON=${COMMON:-20000}; PREFIX=${PREFIX:-10000}; PERTURN=${PERTURN:-2048}
OUTTOK=${OUTTOK:-900}; UTURNS=${UTURNS:-30}
NTURNS=$((UTURNS*2))   # the tool counts user+assistant turns; 30 user-turns => 60

# Stop the sweep once a point's mean ITL exceeds this (ms): beyond it the per-user
# rate is below the interactivity target, so higher-concurrency points are not goodput.
ITL_STOP=${ITL_STOP:-40}

mkdir -p "$OUT"
SUMMARY=$OUT/summary.csv
echo "sessions,req_s,ttft_ms_mean,ttft_p50,ttft_p99,itl_ms_mean,itl_p50,isl_mean,isl_max,osl_mean,dur_s,in_out_tps_total,in_out_tps_gpu,l1_hit_pct,l3_hit_pct,combined_hit_pct" > "$SUMMARY"

snap() {
  local m lq lh eq eh
  m=$(curl -s --max-time 15 "$PREFILL_METRICS" 2>/dev/null)
  lq=$(echo "$m" | grep -E '^vllm:prefix_cache_queries_total'          | awk '{print $2}' | head -1)
  lh=$(echo "$m" | grep -E '^vllm:prefix_cache_hits_total'             | awk '{print $2}' | head -1)
  eq=$(echo "$m" | grep -E '^vllm:external_prefix_cache_queries_total' | awk '{print $2}' | head -1)
  eh=$(echo "$m" | grep -E '^vllm:external_prefix_cache_hits_total'    | awk '{print $2}' | head -1)
  echo "${lq:-0} ${lh:-0} ${eq:-0} ${eh:-0}"
}

for N in $SESSIONS; do
  rd=$OUT/s${N}; mkdir -p "$rd"
  conf=$rd/gen.json
  cat > "$conf" <<JSON
{
    "filetype": "generate_conversations",
    "num_conversations": $N,
    "text_files": ["$TXT"],
    "print_stats": false,
    "prompt_input": {
        "num_turns": {"distribution": "constant", "value": $NTURNS},
        "common_prefix_num_tokens": {"distribution": "constant", "value": $COMMON},
        "prefix_num_tokens": {"distribution": "constant", "value": $PREFIX},
        "num_tokens": {"distribution": "constant", "value": $PERTURN}
    },
    "prompt_output": {
        "num_tokens": {"distribution": "constant", "value": $OUTTOK}
    }
}
JSON
  echo "==================== sessions=$N  ($(date +%H:%M:%S)) ===================="
  B=$(snap); echo "  metrics before: $B"
  # --warmup-step primes the engine (runs the first turn of every conversation before
  # the measured pass) so the shared prefix + first-turn caches are hot at measurement
  # start; without it a heavy prior point can evict the shared prefix and inflate the
  # next point's cold-start ITL/TTFT. Measured stats exclude the warmup pass.
  ( cd "$TOOL_DIR" && AIPERF_PD_IGNORE_EOS="${AIPERF_PD_IGNORE_EOS:-}" $PY benchmark_serving_multi_turn.py \
      --model $MODEL_TOK --served-model-name $SERVED --url $ROUTER \
      --input-file "$conf" --num-clients $N --max-active-conversations $N \
      --limit-min-tokens $((OUTTOK-1)) --limit-max-tokens $OUTTOK --warmup-step \
      --trust-remote-code --stats-json-output "$rd/stats.json" ) > "$rd.log" 2>&1
  A=$(snap); echo "  metrics after:  $A"

  if [ -f "$rd/stats.json" ]; then
    HIT=$(echo "$B $A" | awk '{lq=$5-$1; lh=$6-$2; eq=$7-$3; eh=$8-$4; c=lh+eh;
      printf "%.2f,%.2f,%.2f",(lq>0?100*lh/lq:0),(eq>0?100*eh/eq:0),(lq>0?100*c/lq:0)}')
    ROW=$($PY - "$rd/stats.json" "$N" "$NGPU" "$HIT" <<'PYEOF'
import json,sys,statistics as st
recs=json.load(open(sys.argv[1])); N=sys.argv[2]; NGPU=float(sys.argv[3]); HIT=sys.argv[4]
def pct(v,p):
    v=sorted(v); k=(len(v)-1)*p/100; f=int(k); return v[f] if f+1>=len(v) else v[f]+(v[f+1]-v[f])*(k-f)
ttft=[r['ttft_ms'] for r in recs]; tpot=[r['tpot_ms'] for r in recs if r.get('tpot_ms')]
isl=[r['input_num_tokens'] for r in recs]; osl=[r['output_num_tokens'] for r in recs]
t0=min(r['start_time_ms'] for r in recs); t1=max(r['start_time_ms']+r['latency_ms'] for r in recs)
dur=(t1-t0)/1000.0
tot=sum(isl)+sum(osl); req_s=len(recs)/dur if dur>0 else 0
tg=tot/dur/NGPU if dur>0 else 0
print(f"{N},{req_s:.3f},{st.mean(ttft):.1f},{pct(ttft,50):.1f},{pct(ttft,99):.1f},"
      f"{st.mean(tpot):.2f},{pct(tpot,50):.2f},{st.mean(isl):.0f},{max(isl):.0f},"
      f"{st.mean(osl):.0f},{dur:.1f},{tot/dur:.0f},{tg:.0f},{HIT}")
PYEOF
)
    echo "$ROW" >> "$SUMMARY"
    echo "  -> $ROW"
    itlnow=$(echo "$ROW" | awk -F, '{print $6}')
    if awk "BEGIN{exit !(${itlnow:-0} > ${ITL_STOP:-40})}"; then
      echo "  -> mean ITL ${itlnow}ms > ${ITL_STOP}ms: stopping the sweep here"
      break
    fi
  else
    echo "$N,FAILED,,,,,,,,,,,,,," >> "$SUMMARY"
    echo "  -> FAILED (see $rd.log)"
  fi
done
echo; echo "===== $SUMMARY ====="; cat "$SUMMARY"
