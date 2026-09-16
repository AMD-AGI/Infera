#!/usr/bin/env bash
# Prove the deployment is really live. RUNS ON THE COMPUTE NODE.
# From examples/glm53flash-demo/scripts/mix_smoke.sh. Read the blocks, not the
# exit code -- a 200 with garbage in it means the ROCm/AITER path is off, and
# that is exactly the failure mode the FP8 checkpoint could introduce.
set -u
MY_IP="${NODE_IP:?}"
ROUTER_PORT="${ROUTER_PORT:-8100}"
SERVED="${SERVED:-glm5.3-flash}"
CTR="${CTR:-glm53_mix}"
R="http://$MY_IP:$ROUTER_PORT"

echo "===== 1. workers registered (expect exactly 1, disagg_mode=mixed) ====="
curl -s -m10 "$R/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s -m10 "$R/v1/workers"; echo

echo "===== 2. models ====="
curl -s -m10 "$R/v1/models" | python3 -m json.tool 2>/dev/null; echo

echo "===== 3. coherent answer ====="
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
c=(m.get('content') or '').strip()
print('content :', repr(c))
print('expected: 391')
print('SMOKE_ARITHMETIC_OK' if '391' in c else 'SMOKE_ARITHMETIC_FAIL')
"; echo

echo "===== 5. engine-side: batches, faults, AITER dispatch ====="
docker exec "$CTR" bash -c "grep -a 'Decode batch' /tmp/glm53_mix.log | tail -2"
echo "--- fault / error scan (expect empty) ---"
docker exec "$CTR" bash -c "grep -aiE 'memory access fault|Traceback|CUDA error|HIP error' /tmp/glm53_mix.log | grep -v 'Ignore import error' | tail -5"
echo "--- AITER GLM table + mHC dispatch ---"
docker exec "$CTR" bash -c "grep -a 'glm5_bf16_tuned_gemm\|mHC' /tmp/glm53_mix.log | tail -3 | cut -c1-200"
echo "--- quantization the engine actually chose ---"
docker exec "$CTR" bash -c "grep -aiE 'quantization|quant_method|fp8' /tmp/glm53_mix.log | head -5 | cut -c1-200"
