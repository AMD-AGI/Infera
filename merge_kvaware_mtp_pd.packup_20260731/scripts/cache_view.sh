#!/usr/bin/env bash
# Sum the router's mirrored KV cache view across all DP ranks of each worker.
#
# This is the discriminator for the bigram fix. With MTP on, SGLang's radix keys
# are bigrams (``is_eagle`` is a global server arg, so this affects the PREFILL
# leg too) and kv-events carry ``(t[i], t[i+1])`` pairs. Without _flat_tokens the
# router hashes the pairs and every count below reads 0 -- kv-aware routing
# silently degrades to round-robin with nothing in any log.
#
# TIMING MATTERS: the view lives in the router PROCESS. A freshly restarted
# router reads 0 for a trivial reason. Drive traffic (prefix_reuse.py) first.
#
# Runs ON the prefill node.
set -u
CTR="${CTR:-merge_g0}"
ROUTER="${ROUTER:-http://10.2.122.10:8100}"
WORKERS="${WORKERS:-10.2.122.10:30000 10.2.122.44:30000}"
RANKS="${RANKS:-0 1 2 3 4 5 6 7}"

for w in $WORKERS; do
  tot=0
  for r in $RANKS; do
    n=$(docker exec "$CTR" bash -c \
        "curl -s '$ROUTER/v1/admin/cache-view/$w?dp_rank=$r'" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["block_count"])' 2>/dev/null || echo 0)
    tot=$((tot + n))
  done
  echo "$w total_blocks=$tot"
done
