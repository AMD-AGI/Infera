#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: prove the deployment is serving AND that each of the five features is actually on.
# why : a green /health proves the process is alive, not that PD paired, that KV moved over
#       RDMA, or that speculative decoding is doing anything. Every check below is one that
#       would go RED if the corresponding feature were silently absent.
# how : run through cluster/<your-cluster>.sh smoke.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"

require_env PREFILL_NODE; require_env DECODE_NODE; require_env PREFILL_IP
SSH_CMD="${SSH_CMD:-ssh -o StrictHostKeyChecking=no}"
on(){ local h="$1"; shift; $SSH_CMD "$h" "$*"; }
URL="http://$PREFILL_IP:$ROUTER_PORT"
PLOG="${PLOG:-/tmp/glm52_prefill.log}"
DLOG="${DLOG:-/tmp/glm52_decode.log}"

echo "===== 1. router + workers ====="
# active_workers must be 2, with one prefill and one decode. A single worker means the second
# leg never registered in etcd — which a per-leg /health check cannot tell you.
on "$PREFILL_NODE" "docker exec $CTR curl -s -m10 $URL/health" ; echo
on "$PREFILL_NODE" "docker exec $CTR curl -s -m10 $URL/v1/workers" | python3 -m json.tool 2>/dev/null \
  || on "$PREFILL_NODE" "docker exec $CTR curl -s -m10 $URL/v1/workers"

echo
echo "===== 2. a real completion through the router ====="
# Expect a coherent answer; GARBAGE OR REPEATED TOKENS means the DSA-on-ROCm env block did
# not take effect, and is not a sampling problem (README note 5). DO NOT lower max_tokens —
# reasoning is billed against the SAME budget and `content` then comes back EMPTY (README §5).
BODY=$(python3 -c "import json,os;print(json.dumps({
  'model': os.environ.get('SERVED','glm5.2-mxfp4'),
  'messages':[{'role':'user','content':'What is the capital of France? Answer in one short sentence.'}],
  'max_tokens': 512}))")
on "$PREFILL_NODE" "docker exec $CTR curl -s -m120 $URL/v1/chat/completions \
  -H 'Content-Type: application/json' -d '$BODY'" \
  | python3 -c "import sys,json
d=json.load(sys.stdin)
m=d['choices'][0]['message']
print('  content  :', (m.get('content') or '').strip())
r=(m.get('reasoning_content') or '').strip()
print('  reasoning:', (r[:80]+'…') if len(r)>80 else r or '(none)')
u=d.get('usage',{})
print('  usage    :', u.get('prompt_tokens'), 'prompt /', u.get('completion_tokens'), 'completion',
      '| cached', (u.get('prompt_tokens_details') or {}).get('cached_tokens'))" \
  || die "no completion — check: docker exec $CTR tail -50 /tmp/router.log"

echo
echo "===== 3. feature evidence ====="

echo "-- PD + mooncake: KV must move over RDMA, not TCP --"
# MC_FORCE_TCP appearing in a leg log means mooncake fell back to TCP. That path WORKS and
# looks fine; it is merely 5-20x slower, which is exactly why it has to be checked rather
# than assumed. 'GID is NULL' means the GID index is wrong for this node.
#
# `strings` because these logs carry binary bytes and a bare grep then reports only
# "binary file matches". Values go into variables first: an `echo -n "label="` followed by
# output arriving over ssh interleaves, leaving a bare column of digits.
#
# The two counters being 0 does NOT prove the container can see an RDMA device — a
# mismatched provider finds ZERO devices and prints neither string. Hence the device count.
for pair in "$PREFILL_NODE:$PLOG" "$DECODE_NODE:$DLOG"; do
  h="${pair%%:*}"; f="${pair#*:}"
  n_tcp=$(on "$h" "docker exec $CTR bash -c \"strings $f 2>/dev/null | grep -c MC_FORCE_TCP; true\"" | tr -d '\r\n')
  n_gid=$(on "$h" "docker exec $CTR bash -c \"strings $f 2>/dev/null | grep -c 'GID is NULL'; true\"" | tr -d '\r\n')
  n_dev=$(on "$h" "docker exec $CTR bash -c \"ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE; true\"" | tr -d '\r\n')
  printf '  %-10s MC_FORCE_TCP=%-4s GID-is-NULL=%-4s RDMA-devices-in-container=%s\n' \
    "$h" "${n_tcp:-?}" "${n_gid:-?}" "${n_dev:-?}"
done
echo "  (the two counters must be 0; the device count must be > 0)"

echo "-- DP-attention: resolved dp_size per leg --"
# Read it off the engine's own resolved server_args, not off what was requested. A leg
# launched with --dp-size that failed to enable DP-attention reports dp_size=1 here.
for pair in "$PREFILL_NODE:$PLOG" "$DECODE_NODE:$DLOG"; do
  h="${pair%%:*}"; f="${pair#*:}"
  echo -n "  $h  "
  on "$h" "docker exec $CTR bash -c \"strings $f 2>/dev/null | grep -o 'dp_size=[0-9]*' | head -1; strings $f 2>/dev/null | grep -o 'enable_dp_attention=[A-Za-z]*' | head -1\"" | tr '\n' ' '
  echo
done

echo "-- MTP: acceptance length on the decode leg --"
# Healthy is a MEDIAN of 2-3. Report the distribution, not the last few lines: a few
# percent of batches sit at 4.00 even on a healthy leg (measured 4.7% at median 2.88), so
# a `tail -5` routinely lands on them and reads as degenerate.
on "$DECODE_NODE" "docker exec $CTR bash -c \"strings $DLOG 2>/dev/null \
  | grep -o 'accept len: [0-9.]*' | awk '{print \\\$3}' | sort -n \
  | awk '{a[NR]=\\\$1; if (\\\$1>=4) f++} END {if (!NR) exit 1;
      printf \\\"  n=%d  p10=%s  MEDIAN=%s  p90=%s  at-4.00=%d (%.1f%%)\\n\\\",
             NR, a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], f, 100*f/NR}'\"" \
  || echo "  (no accept-len lines yet — send more traffic)"
echo "  (read the MEDIAN: 2-3 is healthy. A median at 4.00 is a repetition loop, not a win.)"

echo "-- kv-aware / kvd --"
# The Rust router prints `router_policy: "kv-aware"` inside a colourised Config{...} dump,
# never `router-policy=<x>` — a hyphenated, unquoted, ANSI-naive pattern prints EMPTY
# whatever the policy is. Strip ANSI, read the struct field, and `strings` for the binary bytes.
r_pol=$(on "$PREFILL_NODE" "docker exec $CTR bash -c \"strings /tmp/router.log 2>/dev/null \
  | sed -r 's/\\x1B\\[[0-9;]*[mK]//g' | grep -oE 'router_policy: \\\"[a-z-]+\\\"' | head -1\"" | tr -d '\r\n')
# kv-aware silently degrades to load-only routing if the tokenizer did not load, so the
# policy line alone is not enough.
r_tok=$(on "$PREFILL_NODE" "docker exec $CTR bash -c \"strings /tmp/router.log 2>/dev/null \
  | sed -r 's/\\x1B\\[[0-9;]*[mK]//g' | grep -c 'kv-aware: loaded tokenizer'\"" | tr -d '\r\n')
printf '  router %s   tokenizer-loaded=%s (must be >0 for kv-aware to steer)\n' \
  "${r_pol:-<not found>}" "${r_tok:-?}"
# kvd 'adapter connected' should appear once per DP rank on a leg with kvd enabled, and zero
# times on the decode leg — infera skips kvd there by design.
n_kvd=$(on "$PREFILL_NODE" "docker exec $CTR bash -c \"strings $PLOG 2>/dev/null | grep -c 'kvd adapter connected'; true\"" | tr -d '\r\n')
printf '  prefill kvd adapters connected: %s (expect one per DP rank on the kvd leg)\n' "${n_kvd:-?}"
if [ "${PREFILL_KVD:-1}" = "1" ]; then
  echo "  kvd counters:"
  on "$PREFILL_NODE" "docker exec $CTR python3 -m infera.kvd.statctl --socket $KVD_SOCK 2>&1 | head -12" || true
fi

echo
log "smoke complete — read the four blocks above, not just the exit code"
