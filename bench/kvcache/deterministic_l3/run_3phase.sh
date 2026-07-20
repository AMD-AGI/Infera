#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Three-phase L3 (kvd) cache validation.
#
# Phase 1 — warmup. Run the corpus once against a fresh vLLM + fresh
#           kvd. Every turn-N prompt is a kvd MISS, every turn writes
#           to kvd. End state: kvd contains every (session, turn-prefix)
#           the corpus describes.
#
# Phase 2 — cross-restart, L2 OFF. Restart vLLM (drops L1 + L2),
#           replay the SAME corpus. Same prompts → same content
#           hashes → kvd hits. We measure how much kvd alone can do
#           (no in-process L2).
#
# Phase 3 — cross-restart, L2 ON. Restart vLLM with the 8 GiB pinned
#           L2 pool, replay the same corpus once more. First touch
#           of each prefix hits kvd (UDS, ~1100 µs / block); the
#           pool then captures the bytes. Subsequent re-reads in
#           the same uptime hit L2 instead (~80 µs / block).
#
# The clean signal is the gap between Phase 2 and Phase 3 — both
# kvd-warm, only L2 enable bit differs. If Phase 3 is faster the
# L2 pool earned its keep; if it isn't, either the pool implementation
# is broken or this workload's access pattern doesn't repeat within
# one vLLM uptime (in which case use --runs >1 to amplify).
#
# Prerequisites:
#   - `infera-kvd` container running with --max-bytes large enough
#     to hold the entire corpus (corpus size depends on --turn-tokens
#     and total turns; budget ~150 KB per KV block per layer).
#   - vLLM launcher at /tmp/run_vllm_l2.sh from the parent session.
#   - light-trace-benchmark tokenizer paths exported for HF cache.

set -euo pipefail

OUT="${OUT:-/tmp/l3-bench-results}"
CORPUS="${CORPUS:-$OUT/corpus.json}"
SERVER="${SERVER:-http://localhost:30000}"
MODEL="${MODEL:-MiniMax-M2.5}"
CONCURRENCY="${CONCURRENCY:-8}"
SESSIONS="${SESSIONS:-32}"
TURNS="${TURNS:-8}"
TURN_TOKENS="${TURN_TOKENS:-6000}"
KVD_SOCKET_HOST="${KVD_SOCKET_HOST:-/tmp/kvd-sock-C/kvd.sock}"
KVD_CONTAINER="${KVD_CONTAINER:-infera-kvd}"
LAUNCH_VLLM="${LAUNCH_VLLM:-/tmp/run_vllm_l2.sh}"

mkdir -p "$OUT"
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"

# Helper: capture kvd stats into a json file (idempotent).
kvd_stats() {
  docker exec "$KVD_CONTAINER" python3 -m infera.kvd.statctl \
    --socket /run/infera-kvd/kvd.sock 2>/dev/null
}

# Helper: capture vLLM prefix-cache metric set as CSV row.
# `|| true` so a transient /metrics 404 / timeout (common in the first
# few seconds after relaunch) doesn't trip set -e + pipefail and kill
# the whole 3-phase run.
vllm_cache_stats() {
  (curl -s -m 5 "$SERVER/metrics" 2>/dev/null || true) | awk -v ts="$(date +%s)" '
    /^vllm:prefix_cache_queries_total{/ {lq=$NF}
    /^vllm:prefix_cache_hits_total{/    {lh=$NF}
    /^vllm:external_prefix_cache_queries_total{/ {eq=$NF}
    /^vllm:external_prefix_cache_hits_total{/    {eh=$NF}
    /^vllm:kv_cache_usage_perc{/        {ku=$NF}
    END {printf("%s,%s,%s,%s,%s,%s\n", ts, ku, lq, lh, eq, eh)}
  ' || true
}

# Helper: wait until vLLM ready.
wait_vllm() {
  echo -n "waiting for vLLM at $SERVER ..."
  until curl -s -m 3 "$SERVER/v1/models" 2>/dev/null | grep -q "$MODEL"; do
    sleep 5
    echo -n "."
  done
  echo " ready."
}

# Step 0: corpus.
if [ ! -s "$CORPUS" ]; then
  echo "==> generating corpus ($SESSIONS sessions × $TURNS turns × $TURN_TOKENS tokens)"
  python3 -m bench.kvcache.deterministic_l3.corpus \
    --sessions "$SESSIONS" --turns "$TURNS" --turn-tokens "$TURN_TOKENS" \
    --out "$CORPUS"
else
  echo "==> using existing corpus at $CORPUS"
fi

run_phase() {
  local phase="$1"   # 1 | 2 | 3
  local label="$2"   # human-readable
  local l2gb="$3"    # 0 = off, 8 = on
  local restart="$4" # yes | no

  echo
  echo "================================================================="
  echo "PHASE $phase: $label  (L2=$l2gb GiB)"
  echo "================================================================="

  if [ "$restart" = "yes" ]; then
    echo "==> relaunching vLLM with INFERA_KVD_L2_GB=$l2gb"
    "$LAUNCH_VLLM" "$l2gb" > "$OUT/phase${phase}-launch.log" 2>&1 &
    disown
    wait_vllm
  fi

  echo "==> kvd state pre-phase:"
  kvd_stats | tee "$OUT/phase${phase}-kvd-pre.json"
  echo "==> vLLM cache state pre-phase:"
  vllm_cache_stats | tee "$OUT/phase${phase}-vllm-pre.csv"

  echo "==> replaying corpus"
  python3 -m bench.kvcache.deterministic_l3.replay \
    --corpus "$CORPUS" \
    --server "$SERVER" \
    --model "$MODEL" \
    --concurrency "$CONCURRENCY" \
    --out "$OUT/phase${phase}-replay.json"

  echo "==> kvd state post-phase:"
  kvd_stats | tee "$OUT/phase${phase}-kvd-post.json"
  echo "==> vLLM cache state post-phase:"
  vllm_cache_stats | tee "$OUT/phase${phase}-vllm-post.csv"
}

# Phase 1 — warmup. Assume vLLM is already up; if not, launch with L2 off.
if ! curl -s -m 3 "$SERVER/v1/models" 2>/dev/null | grep -q "$MODEL"; then
  run_phase 1 "warmup (fresh vLLM)" 0 yes
else
  echo "==> vLLM already running; using it for Phase 1 (NB: L2 status unknown)"
  run_phase 1 "warmup (existing vLLM)" 0 no
fi

# Phase 2 — restart, L2 off, replay same corpus.
run_phase 2 "cross-restart, L2 OFF" 0 yes

# Phase 3 — restart, L2 on, replay same corpus.
run_phase 3 "cross-restart, L2 ON" 8 yes

echo
echo "================================================================="
echo " summary"
echo "================================================================="
for phase in 1 2 3; do
  if [ -f "$OUT/phase${phase}-replay.json" ]; then
    python3 - "$OUT/phase${phase}-replay.json" "$phase" <<'PY'
import json, sys, statistics
path, ph = sys.argv[1], sys.argv[2]
data = json.load(open(path))
results = data["results"]
ok = [r for r in results if r["ok"] and r["ttft_ms"] is not None]
ttfts = sorted(r["ttft_ms"] for r in ok)
if not ttfts:
    print(f"Phase {ph}: NO successful samples")
    sys.exit(0)
p50 = ttfts[len(ttfts)//2]
p90 = ttfts[int(len(ttfts)*0.9)]
p99 = ttfts[min(int(len(ttfts)*0.99), len(ttfts)-1)]
mean = statistics.mean(ttfts)
print(f"Phase {ph}: {len(ok)}/{len(results)} ok  "
      f"TTFT mean={mean:.0f}  p50={p50:.0f}  p90={p90:.0f}  p99={p99:.0f}  "
      f"wall {data['wall_total_s']:.1f}s")
PY
  fi
done

# kvd delta across the full run
python3 - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
prev = None
for ph in [0, 1, 2, 3]:
    f = os.path.join(out, f"phase{ph}-kvd-{'pre' if ph==1 else 'post'}.json")
    if not os.path.exists(f):
        continue
    d = json.load(open(f))
    if prev is None:
        prev = d
        continue
    delta_gets = d["gets_total"] - prev["gets_total"]
    delta_sets = d["sets_total"] - prev["sets_total"]
    delta_evict = d["evictions_total"] - prev["evictions_total"]
    print(f"kvd delta during phase {ph}: gets+{delta_gets:,}  sets+{delta_sets:,}  evict+{delta_evict:,}")
    prev = d
PY
