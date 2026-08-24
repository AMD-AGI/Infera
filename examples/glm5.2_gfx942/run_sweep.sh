#!/usr/bin/env bash
# Concurrency sweep through the router. Run INSIDE the engine container on the
# prefill node:  bash run_sweep.sh
#
# 128 is the aggregate max_running_requests, i.e. the point where the engine
# admits everything instead of queueing.
#
# The two lines that matter are the seed and the flush. bench.sh defaults to
# SEED=42 and sizes NUM_PROMPTS from CONC, so a sweep run at the default makes
# each point's prompt set a SUPERSET of the point below it -- and the GPU radix
# tree still holds the smaller one. Measured that way the cache-hit line reads
# ~50% at every point from CONC=16 up (and 12.1% at CONC=8, which is 4/32), so
# the higher-concurrency points are scored with half their prefill already
# cached and the points are not comparable. A distinct seed per point makes the
# prompt sets disjoint; the flush drops what the previous point left behind.
set -uo pipefail
source cluster.env
source env.sh
require_ips

for C in 1 8 16 32 64 128; do
  echo "################ CONC=$C ################"
  for URL in "$PREFILL_URL" "$DECODE_URL"; do
    curl -sf -m 30 -X POST "$URL/flush_cache" >/dev/null \
      || echo "[sweep] WARNING: $URL/flush_cache did not answer; this point starts warm"
  done
  sleep 5
  # -u NUM_PROMPTS because sourcing env.sh above already exported it, computed
  # from the DEFAULT CONC rather than this loop's. Left set, every point runs the
  # same 64 prompts: half an hour of serial requests at CONC=1, and at CONC=128
  # fewer prompts than the concurrency limit, so that point never reaches it.
  # Unsetting hands the derivation back to env.sh, which sizes it from CONC.
  env -u NUM_PROMPTS SEED=$((1000 + C)) CONC=$C bash bench.sh 2>&1 | tail -32
done
