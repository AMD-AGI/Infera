#!/usr/bin/env bash
# One tuning iteration: relaunch both legs with a config, wait, score it.
#
#   TAG=mtp0 MTP=0 bash temp/tune-cycle.sh
#   TAG=mtp8 MTP_STEPS=7 MTP_DRAFT_TOKENS=8 bash temp/tune-cycle.sh
#
# Any variable env.sh reads can be passed through; they are forwarded verbatim to
# both launch scripts. DP and CHUNK default to the dp-attention-off shape, which
# is what §10 of the handoff is tuning on top of.
#
# Both legs have to be relaunched together for anything speculative: SGLang
# refuses a disaggregated pair whose MTP shape differs, and it refuses it *after*
# the weights load, so a one-sided change costs a full cold start to discover.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
EX="$REPO/examples/glm5.2_gfx942"
source "$EX/cluster.env"

TAG="${TAG:?set TAG}"
CONC="${CONC:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-20}"

# Forwarded to the launch scripts. Listed explicitly rather than exporting the
# caller's whole environment, so a leftover export from an earlier iteration
# cannot silently ride along into the next one.
KNOBS=(DP TP CHUNK MEM_FRAC MAX_RUNNING MTP MTP_STEPS MTP_TOPK MTP_DRAFT_TOKENS)
DP="${DP:-1}"
CHUNK="${CHUNK:-1024}"

CFG=""
for k in "${KNOBS[@]}"; do
  [[ -n "${!k:-}" ]] && CFG+="$k=${!k} "
done
LOG_DIR="$EX/logs_$TAG"
CFG+="LOG_DIR=$LOG_DIR"

echo "=== [$TAG] $CFG"
CONTAINER=infera-glm52-gfx942

for node in "$PREFILL_NODE" "$DECODE_NODE"; do
  ssh -n "$node" "docker exec $CONTAINER bash -lc 'cd $EX && source cluster.env && bash stop.sh'" >/dev/null 2>&1 || true
done
sleep 10

ssh -n -f "$DECODE_NODE" \
  "docker exec $CONTAINER bash -lc 'cd $EX && source cluster.env && $CFG bash launch/launch_decode.sh' >/tmp/$TAG-dec.out 2>&1"
docker exec "$CONTAINER" bash -lc "cd $EX && source cluster.env && $CFG bash launch/launch_prefill.sh" | sed 's/^/  /'
docker exec "$CONTAINER" bash -lc "cd $EX && source cluster.env && bash launch/launch_router.sh" >/dev/null 2>&1 &

echo "=== [$TAG] waiting for both legs (cold start 12-25 min)"
t0=$SECONDS
for _ in $(seq 100); do
  # `|| true` is load-bearing under `set -e`: curl exits 7 while the port is still
  # closed, which is the normal state for the first 15 minutes of this loop.
  p=$(curl -s -m 3 "$PREFILL_URL/health_generate" -o /dev/null -w '%{http_code}' || true)
  d=$(curl -s -m 3 "$DECODE_URL/health_generate" -o /dev/null -w '%{http_code}' || true)
  [[ "$p" == "200" && "$d" == "200" ]] && break
  # An argument SGLang rejects dies in seconds, but the ready loop would still
  # sit here for half an hour. Fail on a dead launcher instead of on the timeout.
  for n in "$PREFILL_NODE:prefill" "$DECODE_NODE:decode"; do
    alive=$(ssh -n "${n%:*}" "docker exec $CONTAINER pgrep -cf 'infera.engine.sglang' || true")
    [[ "${alive:-0}" == "0" ]] && {
      echo "=== [$TAG] ${n#*:} launcher is gone; last lines:"
      ssh -n "${n%:*}" "tail -15 $LOG_DIR/${n#*:}.log" | sed 's/^/    /'
      exit 1
    }
  done
  sleep 20
done
[[ "$p" == "200" && "$d" == "200" ]] || { echo "=== [$TAG] legs did not come up"; exit 1; }
echo "=== [$TAG] ready after $((SECONDS-t0))s"

# What actually took effect, not what was asked for -- CHUNK in particular is
# divided by dp_size on the way in (gotcha 2).
curl -s -m 5 "$PREFILL_URL/get_server_info" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print("  effective: chunk=%s/rank x dp=%s tp=%s" % (d["chunked_prefill_size"], d["dp_size"], d["tp_size"]))'

CONC="$CONC" NUM_PROMPTS="$NUM_PROMPTS" \
IMAGE="$IMAGE" MODEL="$MODEL" \
bash "$HERE/run-agentic.sh" "$TAG" "$PREFILL_IP:8000" "$PREFILL_URL" "$DECODE_URL"

echo
echo "=== [$TAG] $CFG"
grep -E "Mean TPOT|Median TPOT|Mean ITL|Benchmark duration|Successful requests" \
  "$HERE/exec-logs/agentic/${TAG}_c${CONC}_n${NUM_PROMPTS}.log" | sed 's/  */ /g;s/^/  /'
grep -oE "accept len: [0-9.]+" "$LOG_DIR/decode.log" 2>/dev/null | tail -1 | sed 's/^/  MTP /'
