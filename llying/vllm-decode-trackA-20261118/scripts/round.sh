#!/usr/bin/env bash
# One A/B round: start engines side by side, gate each, run the decode-dominated bench on all at once, break the
# torch traces down per DP rank. Results in $R/results/<tag>/<name>/.
#   round.sh TAG "NAME GPUS PORT RPC MODE [extra vllm args...]" ...
# MODE as in agg_up.sh (stock, gluon, gluon3, comm, joined by +).
# Env: STOP (containers to remove first), KEEP (1 = leave the engines running), BENCH ("N IN OUT", default
#      "104 65536 4096"), GATE_MIN (GSM8K correct out of 200 needed before benching, default 190).
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
tag=$1; shift
specs=("$@")
out=$R/results/$tag; mkdir -p "$out"
cd "$R"
read -r bn bin bout <<< "${BENCH:-104 65536 4096}"
for c in ${STOP:-}; do docker rm -f "$c" > /dev/null 2>&1; done
[[ -n "${STOP:-}" ]] && sleep 15
for s in "${specs[@]}"; do
  read -r name gpus port rpc mode extra <<< "$s"
  mkdir -p "$out/$name"
  # shellcheck disable=SC2086
  TRACE_DIR=$out/$name/trace bash scripts/agg_up.sh "vta-$name" "$gpus" "$port" "$rpc" "$mode" $extra \
    > "$out/$name/up.log" 2>&1 &
  sleep 20
done
wait
for s in "${specs[@]}"; do
  read -r name gpus port rpc mode extra <<< "$s"
  f=logs/engines/vta-$name.log
  {
    tail -n 1 "$out/$name/up.log"
    echo "gluon: $(grep -a -c 'gfx950 gluon paged-MQA logits' $f); aiter AG/RS: $(grep -a -c 'Using AITER custom AG/RS' $f)"
    grep -a -m 3 -E 'DSA indexer decode path|GPU KV cache size|Using AITER custom AG/RS|all2all|MoRI|mori' $f | cut -c1-300
  } > "$out/$name/check.txt" 2>&1
  if curl -sf -m 5 "http://127.0.0.1:$port/v1/models" > /dev/null; then
    PREFILL_NUM_WORKERS=0 bash scripts/gate.sh "$tag-$name" "http://127.0.0.1:$port" "http://127.0.0.1:$port/metrics" \
      > "$out/$name/gate.out" 2>&1 &
  else
    echo "not up" > "$out/$name/gate.out"
  fi
done
wait
for s in "${specs[@]}"; do
  read -r name gpus port rpc mode extra <<< "$s"
  correct=$(grep -o 'correct=[0-9]*/200' "$out/$name/gate.out" | head -1 | cut -d= -f2 | cut -d/ -f1)
  if [[ -n "$correct" && "$correct" -ge "${GATE_MIN:-190}" ]]; then
    TRACE_S=3 bash scripts/decode_bench.sh "vta-$name" "$port" "$out/$name/bench" "$bn" "$bin" "$bout" &
  else
    echo "$name: gate failed (GSM8K ${correct:-?}/200), no bench" | tee -a "$out/summary.txt"
  fi
done
wait
for s in "${specs[@]}"; do
  read -r name gpus port rpc mode extra <<< "$s"
  for t in "$out/$name/trace"/*.pt.trace.json.gz; do
    [[ -e "$t" ]] || continue
    python3 scripts/trace_cat.py "$t" --top 30 > "${t%.pt.trace.json.gz}.cat.txt" 2>&1
  done
  echo "$name ($mode $extra): $(tail -n 1 "$out/$name/gate.out" | grep -o 'probe n=200[^;]*' | cut -c1-60); \
$(cat "$out/$name/bench/window.json" 2>/dev/null)" >> "$out/summary.txt"
  if [[ "${KEEP:-0}" != 1 ]]; then
    docker logs "vta-$name" 2>&1 | gzip > "$out/$name/engine.log.gz"; docker rm -f "vta-$name" > /dev/null 2>&1
  fi
done
echo "$(date -u +%m%d-%T) $tag: $(tr '\n' ' ' < "$out/summary.txt")" >> logs/summary.out
cat "$out/summary.txt"
exit 0
