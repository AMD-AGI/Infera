#!/usr/bin/env bash
# what: prove the MIX deployment serves AND that each feature is really on.
# why : a green /health proves the process is alive — not that DSA took effect, that MTP is
#       speculating, or that kvd is storing anything. Every check below goes RED if the
#       corresponding feature is silently absent.
# how : run ON the node.  bash mix_smoke.sh
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/mix_common.sh"

require_env MY_IP
URL="http://$MY_IP:$ROUTER_PORT"
TAG="${TAG:-base}"
ELOG="${ELOG:-/tmp/glm52_mix_${TAG}.log}"

echo "===== 1. router + worker registry ====="
# Exactly ONE worker, and its disaggregation mode must read mixed/null — a worker that
# registered as prefill or decode means the mode flags leaked back in.
docker exec "$CTR" curl -s -m10 "$URL/health"; echo
docker exec "$CTR" curl -s -m10 "$URL/v1/workers" | python3 -m json.tool 2>/dev/null \
  || docker exec "$CTR" curl -s -m10 "$URL/v1/workers"

echo
echo "===== 2. a real completion through the router ====="
# Expect a coherent answer. GARBAGE OR REPEATED TOKENS is not a sampling problem — it is the
# signature of the DSA-on-ROCm env block not taking effect. DO NOT lower max_tokens: this is
# a thinking model with --reasoning-parser glm45, reasoning is billed against the SAME
# budget, and at a small value `content` comes back EMPTY on a healthy deployment.
BODY=$(python3 -c "import json,os;print(json.dumps({
  'model': os.environ.get('SERVED','glm5.2-mxfp4'),
  'messages':[{'role':'user','content':'What is the capital of France? Answer in one short sentence.'}],
  'max_tokens': 512, 'temperature': 1.0, 'top_p': 0.95}))")
docker exec "$CTR" curl -s -m180 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -c "import sys,json
d=json.load(sys.stdin)
m=d['choices'][0]['message']
print('  content  :', (m.get('content') or '').strip())
r=(m.get('reasoning_content') or '').strip()
print('  reasoning:', (r[:80]+'…') if len(r)>80 else r or '(none)')
u=d.get('usage',{})
print('  usage    :', u.get('prompt_tokens'), 'prompt /', u.get('completion_tokens'), 'completion',
      '| cached', (u.get('prompt_tokens_details') or {}).get('cached_tokens'))" \
  || die "no completion — docker exec $CTR tail -50 /tmp/router.log"

echo
echo "===== 3. feature evidence ====="

echo "-- DP-attention: the engine's OWN resolved server_args --"
# Read the resolved value, not what was requested: a worker launched with --dp-size that
# failed to enable DP-attention still reports dp_size=1 here.
# `strings` because these logs carry binary bytes and a bare grep then reports only
# "binary file matches".
echo -n "  "
docker exec "$CTR" bash -c "strings $ELOG 2>/dev/null | grep -o 'dp_size=[0-9]*' | head -1; \
  strings $ELOG 2>/dev/null | grep -o 'enable_dp_attention=[A-Za-z]*' | head -1" | tr '\n' ' '
echo
echo -n "  live scheduler ranks: "
docker exec "$CTR" bash -c "ps aux | grep -c '[s]cheduler_DP'"

echo "-- MTP: acceptance-length DISTRIBUTION --"
# Healthy is a MEDIAN of 2-3. Report the distribution, not the last few lines: a few percent
# of batches sit at 4.00 even on a healthy engine, so a `tail -5` routinely lands on them
# and reads as degenerate. A median AT 4.00 is a repetition loop, not a win.
docker exec "$CTR" bash -c "strings $ELOG 2>/dev/null \
  | grep -o 'accept len: [0-9.]*' | awk '{print \$3}' | sort -n \
  | awk '{a[NR]=\$1; if (\$1>=4) f++} END {if (!NR) exit 1;
      printf \"  n=%d  p10=%s  MEDIAN=%s  p90=%s  at-4.00=%d (%.1f%%)\n\",
             NR, a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], f, 100*f/NR}'" \
  || echo "  (no accept-len lines yet — send more traffic)"

echo "-- kv-aware: router policy AND tokenizer --"
# The Rust router prints `router_policy: "kv-aware"` inside a colourised Config{...} dump,
# never `router-policy=<x>` — a hyphenated, unquoted, ANSI-naive pattern prints EMPTY
# whatever the policy is. kv-aware also degrades silently to load-only routing if the
# tokenizer did not load, so the policy line alone is not sufficient evidence.
r_pol=$(docker exec "$CTR" bash -c "strings /tmp/router.log 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;]*[mK]//g' | grep -oE 'router_policy: \"[a-z-]+\"' | head -1" | tr -d '\r\n')
r_tok=$(docker exec "$CTR" bash -c "strings /tmp/router.log 2>/dev/null \
  | sed -r 's/\x1B\[[0-9;]*[mK]//g' | grep -c 'kv-aware: loaded tokenizer'" | tr -d '\r\n')
printf '  router %s   tokenizer-loaded=%s (must be >0 for kv-aware to steer)\n' \
  "${r_pol:-<not found>}" "${r_tok:-?}"

echo "-- kvd: adapters + counters --"
# One adapter per DP rank. kvd IS legal on mix: infera skips it only when
# disaggregation_mode == "decode".
n_kvd=$(docker exec "$CTR" bash -c "strings $ELOG 2>/dev/null | grep -c 'kvd adapter connected'; true" | tr -d '\r\n')
printf '  kvd adapters connected: %s (expect one per DP rank)\n' "${n_kvd:-?}"
docker exec "$CTR" python3 -m infera.kvd.statctl --socket "$KVD_SOCK" 2>&1 | head -12 || true

echo "-- prefix cache: cached_tokens must be NON-ZERO on a repeat --"
# --enable-cache-report populates usage.prompt_tokens_details.cached_tokens. Without the
# flag every client-side cache metric reads 0 and a prefix-reuse target cannot be checked
# at all. Send the SAME long-ish prompt twice; the second must report a hit.
PROMPT=$(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 200)")
RBODY=$(python3 -c "import json,os,sys;print(json.dumps({
  'model': os.environ.get('SERVED','glm5.2-mxfp4'),
  'messages':[{'role':'user','content': sys.argv[1] + ' Reply with one word.'}],
  'max_tokens': 32, 'temperature': 1.0, 'top_p': 0.95}))" "$PROMPT")
for i in 1 2; do
  echo -n "  attempt $i cached_tokens = "
  docker exec "$CTR" curl -s -m120 "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$RBODY" \
    | python3 -c "import sys,json;u=json.load(sys.stdin).get('usage',{});
print((u.get('prompt_tokens_details') or {}).get('cached_tokens'), '/', u.get('prompt_tokens'), 'prompt')" \
    || echo "?"
done

echo
log "smoke complete — read the blocks above, not just the exit code"
