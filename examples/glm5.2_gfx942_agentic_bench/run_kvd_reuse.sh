#!/usr/bin/env bash
# Measure what kvd actually buys, by replaying the same agentic trace twice with
# the GPU and host tiers wiped in between. Run on the prefill node inside the
# container, after launch + smoke.
#
#   NUM_PROMPTS=20 CONC=4 bash run_kvd_reuse.sh
#
# WHY A SECOND PASS. With a 54 GB device KV pool per rank this deployment already
# serves the agentic trace at ~100% of the growing-prefix ideal, so "enable kvd
# and watch the hit rate" measures nothing: there is almost nothing left to
# recover. What kvd uniquely provides is a prefix that survives leaving the GPU.
# So: fill L3 (pass 1), throw away everything above it, replay (pass 2).
#
# The isolation is narrower than "any pass 2 hit came off kvd", which is what this
# script used to claim. Pass 2 replays whole conversations, so each conversation
# refills the GPU tier as it goes and every turn but the first is served from
# there. Only the conversation-opening turns are attributable to kvd, and the
# scorer's tier breakdown is what proves it -- read that, not the hit rate.
#
#   /clear_hicache_storage_backend  drops our namespace in kvd (L3 + RAM tier)
#   /flush_cache                    resets the radix tree and the host pool,
#                                   and does NOT touch the storage backend
#
# Pass 2 is scored with --warm: pass 1 stored every turn's whole prompt, so the
# growing-prefix ideal (turn 0 reuses nothing) is no longer the ceiling.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

[[ "$KVD" == "1" ]] || { echo "[reuse] needs KVD=1" >&2; exit 1; }

NUM_PROMPTS="${NUM_PROMPTS:-20}"
CONC="${CONC:-4}"
TAG="${TAG:-c${CONC}_n${NUM_PROMPTS}}"

stats() { python3 -m infera.kvd.statctl --socket "$KVD_SOCKET"; }
counter() { stats | python3 -c "import json,sys; print(json.load(sys.stdin)['$1'])"; }

echo "== clearing kvd so pass 1 starts cold =="
curl -sf -X POST "$PREFILL_URL/clear_hicache_storage_backend" || echo "(clear endpoint refused)"
sleep 3
stats

echo; echo "== pass 1: cold everywhere -- fills kvd =="
S0_SETS="$(counter sets_total)"
S0_GETS="$(counter gets_total)"
S0_HITS="$(counter hits_total)"
NUM_PROMPTS="$NUM_PROMPTS" CONC="$CONC" bash "$HERE/run_agentic_trace.sh" "kvd_p1_${TAG}"
S1_SETS="$(counter sets_total)"
S1_GETS="$(counter gets_total)"
S1_HITS="$(counter hits_total)"

echo; echo "== pass 2: same trace, GPU + host tiers wiped, kvd kept =="
# run_agentic_trace.sh flushes both legs before it starts, which is exactly the
# wipe this pass needs -- no extra step here, and no accidental L3 clear.
NUM_PROMPTS="$NUM_PROMPTS" CONC="$CONC" SCORE_ARGS="--warm" \
  bash "$HERE/run_agentic_trace.sh" "kvd_p2_${TAG}"
S2_GETS="$(counter gets_total)"
S2_HITS="$(counter hits_total)"

echo; echo "== kvd traffic (deltas within each pass) =="
printf '  pass 1 writes      %12d\n' "$((S1_SETS - S0_SETS))"
printf '  pass 1 lookups     %12d  hits %d  (cleared first, so hits should be ~0)\n' \
  "$((S1_GETS - S0_GETS))" "$((S1_HITS - S0_HITS))"
printf '  pass 2 lookups     %12d  hits %d\n' \
  "$((S2_GETS - S1_GETS))" "$((S2_HITS - S1_HITS))"
stats | tee "$RESULT_DIR/kvd_stats_${TAG}.json"

echo; echo "== read this as =="
echo "  pass 1 hit rate            what the GPU tier alone recovers"
echo "  pass 2 'cached by tier'    the line that answers whether kvd did anything."
echo "    Do NOT read pass 2's total hit rate as kvd's: pass 2 replays whole"
echo "    conversations, so from turn 2 on each conversation has refilled the GPU"
echo "    tier from its own traffic and serves itself. Measured here: 99.6% of"
echo "    pass 2's hits came from device, 0.4% from storage."
echo "  pass 2 'conversation-opening turns'   the isolated number. Only those"
echo "    turns have no tier above kvd holding their prefix, and the scorer"
echo "    prints what fraction of them kvd served."
echo "  a kvd hit says the block came from the daemon, NOT off NVMe: it answers"
echo "    from its pinned RAM arena first and the counters do not split by tier."
grep -aE "prefetch|storage_hit|hicache" "$LOG_DIR/prefill.log" | tail -8 || true
