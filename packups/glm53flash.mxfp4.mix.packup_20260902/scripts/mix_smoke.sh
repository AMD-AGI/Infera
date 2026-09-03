#!/bin/bash
# Prove the GLM-5.3-Flash MIX deployment is really live — read the blocks, not the
# exit code. Hits the ROUTER (:8100), which is the only endpoint a client should use.
set -u
MY_IP="${MY_IP:?MY_IP=IP of this node}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
SERVED="${SERVED:-glm5.3-flash}"
CTR="${CTR:-glm53_mix}"
R="http://$MY_IP:$ROUTER_PORT"

echo "===== 1. workers registered (expect exactly 1, aggregated/mixed) ====="
curl -s -m10 "$R/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s -m10 "$R/v1/workers"; echo

echo "===== 2. models ====="
curl -s -m10 "$R/v1/models" | python3 -m json.tool 2>/dev/null; echo

echo "===== 3. coherent answer (garbage here means the ROCm/AITER path is off) ====="
curl -s -m300 "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Answer in one short sentence.\"}],
  \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 512
}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
c=d['choices'][0]; m=c['message']
print('reasoning_content present:', bool(m.get('reasoning_content')))
print('content :', m.get('content'))
print('finish  :', c['finish_reason'])
print('usage   :', d['usage'])
"; echo

echo "===== 4. arithmetic + instruction following ====="
curl -s -m300 "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Compute 17 * 23. Reply with only the number.\"}],
  \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 512
}" | python3 -c "
import json,sys
d=json.load(sys.stdin)
m=d['choices'][0]['message']
print('content :', repr(m.get('content')))
print('expected: 391')
"; echo

echo "===== 5. engine-side: both memory pools, and no faults ====="
docker exec "$CTR" bash -c "grep -a 'Decode batch' /tmp/glm53_mix.log | tail -2"
echo "--- fault / error scan (expect empty) ---"
docker exec "$CTR" bash -c "grep -aiE 'memory access fault|Traceback|CUDA error|HIP error' /tmp/glm53_mix.log | grep -v 'Ignore import error' | tail -5"
echo "--- AITER GLM table + mHC dispatch ---"
docker exec "$CTR" bash -c "grep -a 'glm5_bf16_tuned_gemm\|mHC' /tmp/glm53_mix.log | tail -3 | cut -c1-200"
