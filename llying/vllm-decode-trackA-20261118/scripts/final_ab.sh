#!/usr/bin/env bash
# Final A/B: the recommended decode config (overlay v3 + AITER AG/RS in FULL graphs) against stock, side by side,
# gate + c104 bench (round.sh, engines kept), then the same pair at lower concurrency (N requests, 64K in, 4K out).
#   final_ab.sh TAG [N ...]
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-trackA-20261118}
tag=${1:-final}; shift
cd "$R"
KEEP=1 bash scripts/round.sh "$tag" "final 0,1,2,3 8100 29610 gluon3+comm" "stock 4,5,6,7 8200 29710 stock"
out=$R/results/$tag
ns=("$@"); [[ ${#ns[@]} -eq 0 ]] && ns=(32 64)
park() {  # move this engine's trace files into trace/<label>/
  local d=$out/$1/trace; mkdir -p "$d/$2"; find "$d" -maxdepth 1 -type f -exec mv {} "$d/$2/" \;
}
for c in final stock; do park $c c104; done
for n in "${ns[@]}"; do
  if curl -sf -m 5 http://127.0.0.1:8100/v1/models > /dev/null && curl -sf -m 5 http://127.0.0.1:8200/v1/models > /dev/null; then
    TRACE_S=3 bash scripts/decode_bench.sh vta-final 8100 "$out/final/bench-c$n" "$n" 65536 4096 &
    TRACE_S=3 bash scripts/decode_bench.sh vta-stock 8200 "$out/stock/bench-c$n" "$n" 65536 4096 &
    wait
    for c in final stock; do
      park $c c$n
      for t in "$out/$c/trace/c$n"/*.pt.trace.json.gz; do
        [[ -e "$t" ]] && python3 scripts/trace_cat.py "$t" --top 30 > "${t%.pt.trace.json.gz}.cat.txt" 2>&1
      done
      echo "$c c$n: $(cat "$out/$c/bench-c$n/window.json" 2>/dev/null)" >> "$out/summary.txt"
    done
  fi
done
for c in final stock; do docker logs "vta-$c" 2>&1 | gzip > "$out/$c/engine.log.gz"; docker rm -f "vta-$c" > /dev/null 2>&1; done
echo "$(date -u +%m%d-%T) $tag low-concurrency done" >> logs/summary.out
exit 0
