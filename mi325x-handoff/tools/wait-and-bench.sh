#!/usr/bin/env bash
# Wait for both legs of an already-launched pair, then score them.
#
#   TAG=mtp-topk2 CONC=1 NUM_PROMPTS=20 bash temp/wait-and-bench.sh
#
# The second half of tune-cycle.sh, split out so a cold start that is already
# under way is not thrown away when the wait needs restarting.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
EX="$(cd "$HERE/.." && pwd)/examples/glm5.2_gfx942"
source "$EX/cluster.env"

TAG="${TAG:?set TAG}"
CONC="${CONC:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"
LOG_DIR="${LOG_DIR:-$EX/logs_$TAG}"
CONTAINER=infera-glm52-gfx942

t0=$SECONDS
for _ in $(seq 100); do
  p=$(curl -s -m 3 "$PREFILL_URL/health_generate" -o /dev/null -w '%{http_code}' || true)
  d=$(curl -s -m 3 "$DECODE_URL/health_generate" -o /dev/null -w '%{http_code}' || true)
  [[ "$p" == "200" && "$d" == "200" ]] && break
  for n in "$PREFILL_NODE:prefill" "$DECODE_NODE:decode"; do
    alive=$(ssh -n "${n%:*}" "docker exec $CONTAINER pgrep -cf 'infera.engine.sglang' 2>/dev/null || echo 0")
    if [[ "${alive:-0}" == "0" ]]; then
      echo "=== [$TAG] ${n#*:} launcher is gone after $((SECONDS-t0))s; last lines:"
      ssh -n "${n%:*}" "tail -20 $LOG_DIR/${n#*:}.log" | sed 's/^/    /'
      exit 1
    fi
  done
  sleep 20
done
[[ "$p" == "200" && "$d" == "200" ]] || { echo "=== [$TAG] legs did not come up"; exit 1; }
echo "=== [$TAG] ready after $((SECONDS-t0))s"

curl -s -m 5 "$PREFILL_URL/get_server_info" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print("  effective: chunk=%s dp=%s tp=%s spec=%s/%s/%s" % (
   d["chunked_prefill_size"], d["dp_size"], d["tp_size"],
   d.get("speculative_num_steps"), d.get("speculative_eagle_topk"), d.get("speculative_num_draft_tokens")))'

rm -f "$HERE/exec-logs/agentic/${TAG}_c${CONC}_n${NUM_PROMPTS}."*
CONC="$CONC" NUM_PROMPTS="$NUM_PROMPTS" IMAGE="$IMAGE" MODEL="$MODEL" \
  bash "$HERE/run-agentic.sh" "$TAG" "$PREFILL_IP:8000" "$PREFILL_URL" "$DECODE_URL" >/dev/null 2>&1

echo "=== [$TAG] result"
grep -E "Successful requests|Benchmark duration|Output token throughput|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Mean TTFT" \
  "$HERE/exec-logs/agentic/${TAG}_c${CONC}_n${NUM_PROMPTS}.log" | sed 's/  */ /g;s/^/  /'
grep -E "efficiency|actual hit" "$HERE/exec-logs/agentic/${TAG}_c${CONC}_n${NUM_PROMPTS}.score.txt" | sed 's/^/  /'
ssh -n "$DECODE_NODE" "grep -oE 'accept len: [0-9.]+' $LOG_DIR/decode.log | tail -200" \
  | awk -F': ' '{s+=$2; n++} END {if(n) printf "  mean accept len over last %d batches: %.2f\n", n, s/n}'
