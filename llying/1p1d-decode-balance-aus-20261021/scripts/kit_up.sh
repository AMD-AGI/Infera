#!/usr/bin/env bash
# Queue step: bring the PR 210 kit up (engine/up.sh on the prefill node; it starts the decode over the ssh shim and
# returns once the router lists 1 prefill + 1 decode), then follow the three engine logs onto the login host as
# pr210/runs/<RUN>/logs/{prefill-0,decode-0,router}.log for patrol and analysis.
# Usage: kit_up.sh RUN=<id> [MTP=0|1] [SIM_ACCEPT_LEN=3.61|] [PREFILL0_GPUS=4,5,6,7] [PREFILL_EXTRA_ARGS=...]
set -uo pipefail
for a in "$@"; do export "$a"; done
: "${RUN:?}"
source "$(dirname "$0")/kit_env.sh"
OUT="$PR_RUNS/$RUN"; LOGS="$OUT/logs"
mkdir -p "$LOGS"
# MTP=1: the prefill also gets the decode's EAGLE flags (the kit has them on the decode only; RESULT §2 #1).
[[ "${MTP:-0}" == 1 ]] && PREFILL_EXTRA_ARGS="--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6${PREFILL_EXTRA_ARGS:+ $PREFILL_EXTRA_ARGS}"
extra=()
[[ -n "${PREFILL_EXTRA_ARGS:-}" ]] && extra+=("PREFILL_EXTRA_ARGS=$PREFILL_EXTRA_ARGS")
say "up RUN=$RUN on P=$P_NODE ($P_IP) D=$D_NODE ($D_IP) SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN-3.61} ${extra[*]}"
echo "$RUN" > "$R/pr210/current_run"
for pair in "$CTR-prefill0 $P_NODE prefill-0" "$CTR-decode $D_NODE decode-0" "$CTR-router $P_NODE router"; do
    read -r name node log <<< "$pair"
    follow_when_created "$name" "$node" "$LOGS/$log.log"
done
kit_on "$P_NODE" "RUN_ID=$(q "$RUN") $(q "${extra[@]}") bash engine/up.sh" > "$OUT/up.log" 2>&1
rc=$?
tail -n 3 "$OUT/up.log"
(( rc == 0 )) || die "up.sh rc=$rc (see $OUT/up.log)"
ssh "$P_NODE" "cat /scratch/liyingli_2p1d/pr210/work/runs/$RUN/engine-config.env" > "$OUT/engine-config.env"
curl -sf -m 10 "$ROUTER_URL/v1/workers" > "$OUT/workers.json" || die "router workers"
say "up: $(python3 -c 'import json,sys; print(sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' < "$OUT/workers.json")"
