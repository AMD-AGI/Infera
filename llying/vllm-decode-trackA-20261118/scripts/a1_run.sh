#!/usr/bin/env bash
# A1 step 2, on the running pair from a1_pair_up.sh: correctness gate on both (aggK = #47897 on, aggR = stock, as
# an A/A reference), then the decode-dominated bench on both at once, then per-rank kernel breakdowns of the traces.
#   a1_run.sh TAG [N IN OUT]
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
tag=${1:-pair1} n=${2:-104} in=${3:-65536} outlen=${4:-4096}
out=$R/results/a1/$tag
cd "$R"
for side in aggK:8100 aggR:8200; do
  c=${side%%:*} p=${side##*:}
  PREFILL_NUM_WORKERS=0 bash scripts/gate.sh "a1-$tag-$c" "http://127.0.0.1:$p" "http://127.0.0.1:$p/metrics" \
    > "$out/gate-$c.out" 2>&1 &
done
wait
tail -1 "$out"/gate-*.out
correct=$(grep -o 'correct=[0-9]*/200' "$out/gate-aggK.out" | head -1 | cut -d= -f2 | cut -d/ -f1)
if [[ -z "$correct" || "$correct" -lt 190 ]]; then
  echo "gate on aggK failed (GSM8K ${correct:-?}/200); no bench" | tee -a logs/summary.out
  exit 0
fi
echo "gluon lines after gate: $(grep -a -c 'gfx950 gluon paged-MQA logits' logs/engines/vta-aggK.log)"
TRACE_S=3 bash scripts/decode_bench.sh vta-aggK 8100 "$out/bench-aggK" "$n" "$in" "$outlen" &
TRACE_S=3 bash scripts/decode_bench.sh vta-aggR 8200 "$out/bench-aggR" "$n" "$in" "$outlen" &
wait
for c in aggK aggR; do
  for t in "$out/trace-$c"/*.pt.trace.json.gz; do
    [[ -e "$t" ]] || continue
    python3 scripts/trace_cat.py "$t" --top 30 > "${t%.pt.trace.json.gz}.cat.txt" 2>&1
  done
done
echo "$(date -u +%m%d-%T) a1 $tag: K $(cat $out/bench-aggK/window.json 2>/dev/null); R $(cat $out/bench-aggR/window.json 2>/dev/null)" \
  | tee -a logs/summary.out
exit 0
