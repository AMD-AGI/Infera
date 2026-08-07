#!/bin/bash
# Smoke a GLM-5.2 MIX (aggregated) deployment: prove it serves AND each feature is REALLY on.
# One worker (not two). No MC/RDMA checks (mix has no KV transfer). Runs ON the node.
set -u
export DOCKER_CONFIG="${DOCKER_CONFIG:-/var/tmp/dockercfg_yihou}"; mkdir -p "$DOCKER_CONFIG"
MY_IP="${MY_IP:?}"; CTR="${CTR:-glm52_mix}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; SERVED="${SERVED:-glm5.2-mxfp4}"
MLOG="${MLOG:-/tmp/glm52_mix.log}"; KVD_SOCK=/tmp/kvd/kvd.sock
URL="http://$MY_IP:$ROUTER_PORT"

echo "===== 1. router + worker (expect exactly 1 worker) ====="
docker exec "$CTR" curl -s -m10 "$URL/health"; echo
docker exec "$CTR" curl -s -m10 "$URL/v1/workers" | python3 -m json.tool 2>/dev/null \
  || docker exec "$CTR" curl -s -m10 "$URL/v1/workers"

echo; echo "===== 2. a real completion through the router (garbage => DSA env not live) ====="
BODY=$(python3 -c "import json,os;print(json.dumps({
  'model': os.environ.get('SERVED','glm5.2-mxfp4'),
  'messages':[{'role':'user','content':'What is the capital of France? Answer in one short sentence.'}],
  'max_tokens': 512}))")
docker exec "$CTR" curl -s -m120 "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -c "import sys,json
d=json.load(sys.stdin); m=d['choices'][0]['message']
print('  content  :', (m.get('content') or '').strip())
r=(m.get('reasoning_content') or '').strip()
print('  reasoning:', (r[:80]+'…') if len(r)>80 else r or '(none)')
u=d.get('usage',{})
print('  usage    :', u.get('prompt_tokens'),'prompt /',u.get('completion_tokens'),'completion | cached',(u.get('prompt_tokens_details') or {}).get('cached_tokens'))" \
  || { echo '  no completion — router.log tail:'; docker exec "$CTR" tail -30 /tmp/router.log; }

echo; echo "===== 3. feature evidence ====="

echo "-- DSA env resolved on the worker --"
docker exec "$CTR" bash -c "strings $MLOG 2>/dev/null | grep -oE 'enable_dp_attention=[A-Za-z]*|ep_size=[0-9]*|speculative_algorithm=[A-Za-z]*' | sort -u | sed 's/^/  /'"

echo "-- MTP: acceptance length distribution (median 2-3 healthy; 4.00 = degeneration) --"
docker exec "$CTR" bash -c "strings $MLOG 2>/dev/null \
  | grep -o 'accept len: [0-9.]*' | awk '{print \$3}' | sort -n \
  | awk '{a[NR]=\$1; if (\$1>=4) f++} END {if (!NR) exit 1;
      printf \"  n=%d  p10=%s  MEDIAN=%s  p90=%s  at-4.00=%d (%.1f%%)\n\", NR,a[int(NR*0.1)+1],a[int(NR*0.5)+1],a[int(NR*0.9)+1],f,100*f/NR}'" \
  || echo "  (no accept-len lines yet — send more traffic)"

echo "-- kv-aware router policy + tokenizer --"
r_pol=$(docker exec "$CTR" bash -c "strings /tmp/router.log 2>/dev/null | sed -r 's/\x1B\[[0-9;]*[mK]//g' | grep -oE 'router_policy: \"[a-z-]+\"' | head -1" | tr -d '\r\n')
r_tok=$(docker exec "$CTR" bash -c "strings /tmp/router.log 2>/dev/null | sed -r 's/\x1B\[[0-9;]*[mK]//g' | grep -c 'kv-aware: loaded tokenizer'" | tr -d '\r\n')
printf '  router %s   tokenizer-loaded=%s\n' "${r_pol:-<not found>}" "${r_tok:-?}"

echo "-- kvd: adapters + counters (entries must be >0 after traffic) --"
n_kvd=$(docker exec "$CTR" bash -c "strings $MLOG 2>/dev/null | grep -c 'kvd adapter connected'; true" | tr -d '\r\n')
printf '  worker kvd adapters connected: %s (expect ~one per rank)\n' "${n_kvd:-?}"
docker exec "$CTR" python3 -m infera.kvd.statctl --socket $KVD_SOCK 2>&1 | head -12 || true

echo; echo "smoke complete — read the four blocks, not just the exit code"
