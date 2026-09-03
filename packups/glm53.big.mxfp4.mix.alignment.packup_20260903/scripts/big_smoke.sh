#!/bin/bash
# Prove the GLM-5.3 big-model MIX deployment is REAL, not merely /health green.
#
# A green /health proves a process is alive. It does not prove the DSA-on-ROCm
# path took effect, and that is the failure this model has: with the env block
# missing the server starts, answers 200, and returns GARBAGE. So every check
# below is written to go RED when a feature is silently absent, not when the
# process is dead.
#
# Hits the ROUTER (:8110), which is the only endpoint a client should use.
# Read the blocks; the exit code is not the deliverable.
set -u
MY_IP="${MY_IP:?MY_IP=IP of this node}"
ROUTER_PORT="${ROUTER_PORT:-8110}"
VARIANT="${VARIANT:-mxfp4}"
case "$VARIANT" in
  fp8)   DEF_SERVED=glm-5.3-fp8   ;;
  mxfp4) DEF_SERVED=glm-5.3-mxfp4 ;;
  *) echo "VARIANT must be fp8 or mxfp4, got: $VARIANT" >&2; exit 2 ;;
esac
SERVED="${SERVED:-$DEF_SERVED}"
CTR="${CTR:-glm53_big_mix}"
LOG="${LOG:-/tmp/glm53_big_mix.log}"
R="http://$MY_IP:$ROUTER_PORT"

echo "===== 1. workers registered (expect exactly 1, aggregated/mixed) ====="
curl -s -m10 "$R/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s -m10 "$R/v1/workers"; echo

echo "===== 2. models (expect id == $SERVED) ====="
curl -s -m10 "$R/v1/models" | python3 -m json.tool 2>/dev/null; echo

# max_tokens is deliberately LARGE. GLM-5.3 is a thinking model and the worker
# passes --reasoning-parser glm45, so the chain of thought lands in
# reasoning_content -- but it is billed against the SAME budget. At a small value
# the model spends every token thinking, content comes back empty with
# finish_reason "length", and a correctly serving deployment reads as a failure.
echo "===== 3. coherent answer + reasoning separated ====="
echo "     (garbage or repeated tokens here is NOT sampling -- it is the DSA-on-ROCm"
echo "      env block not taking effect. Check block 6 before touching temperature.)"
curl -s -m600 "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Answer in one short sentence.\"}],
  \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 1024
}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d['choices'][0]; m = c['message']
r = m.get('reasoning_content') or ''
t = m.get('content') or ''
print('reasoning_content present:', bool(r), '(len %d)' % len(r))
print('content  :', repr(t))
print('finish   :', c['finish_reason'])
print('usage    :', d['usage'])
print('VERDICT  :', 'PASS' if ('Paris' in t and c['finish_reason'] == 'stop') else 'INSPECT -- read the content above')
"; echo

echo "===== 4. arithmetic + instruction following (a coherence check, not a benchmark) ====="
curl -s -m600 "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Compute 17 * 23. Reply with only the number.\"}],
  \"temperature\": 1.0, \"top_p\": 0.95, \"max_tokens\": 1024
}" | python3 -c "
import json, sys
d = json.load(sys.stdin)
t = (d['choices'][0]['message'].get('content') or '')
print('content  :', repr(t))
print('expected : 391')
print('VERDICT  :', 'PASS' if '391' in t else 'INSPECT')
"; echo

echo "===== 5. resolved server args -- did our flags actually land? ====="
# The engine echoes the fully resolved ServerArgs. This is where a flag that was
# silently dropped, renamed or overridden becomes visible.
docker exec "$CTR" bash -c "grep -aoE '(dsa_prefill_backend|dsa_decode_backend|ep_size|dp_size|enable_dp_attention|kv_cache_dtype|moe_runner_backend|quantization|mem_fraction_static|max_running_requests|chunked_prefill_size|context_length|speculative_algorithm|disable_shared_experts_fusion)=[^,)]*' '$LOG' | sort -u"
echo
echo "--- shared-experts fusion decision (the Flash arm faults here when it fuses) ---"
# glm4_moe.py does NOT consult the quant config in its fusion gate, so nothing in
# the engine would catch a precision mismatch. Read the decision, do not assume it.
docker exec "$CTR" bash -c "grep -aiE 'shared.experts.fusion|num_fused_shared_experts|fused_shared' '$LOG' | tail -4 | cut -c1-200"
echo

echo "===== 6. DSA / AITER evidence in the engine log ====="
echo "--- tilelang DSA indexer + AITER (absence here explains garbage in block 3) ---"
docker exec "$CTR" bash -c "grep -aiE 'tilelang|aiter|indexer|mHC|fused_moe|float4_e2m1|quark|mxfp4' '$LOG' | tail -12 | cut -c1-200"
echo "--- memory pool / batch progress (proves decode really ran) ---"
docker exec "$CTR" bash -c "grep -aE 'Decode batch|Prefill batch|KV Cache is allocated|max_total_num_tokens' '$LOG' | tail -4 | cut -c1-200"
echo

# 6b is a VERDICT, not an echo, and it is the one that catches a silent
# regression: with --moe-runner-backend triton (or aiter failing to bind) the
# MXFP4 checkpoint is dequantised to BF16 GEMMs. The server still starts, still
# answers correctly, and is several times slower, with nothing in any log
# saying so. Only the fused_moe dispatch line names the packed dtype.
if [ "$VARIANT" = "mxfp4" ]; then
  echo "===== 6b. native FP4 MoE path live? (mxfp4 only -- silent-dequant guard) ====="
  docker exec "$CTR" bash -c "grep -am1 'float4_e2m1fn_x2' '$LOG' | cut -c1-300"
  N4=$(docker exec "$CTR" bash -c "grep -ac 'float4_e2m1fn_x2' '$LOG'" 2>/dev/null | tr -d '\r')
  N32=$(docker exec "$CTR" bash -c "grep -ac 'QuantType.per_1x32' '$LOG'" 2>/dev/null | tr -d '\r')
  echo "float4_e2m1fn_x2 lines: ${N4:-0}   QuantType.per_1x32 lines: ${N32:-0}"
  if [ "${N4:-0}" -gt 0 ] && [ "${N32:-0}" -gt 0 ]; then
    echo "VERDICT  : PASS -- AITER native FP4 MoE, not dequantised to BF16"
  else
    echo "VERDICT  : FAIL -- no packed-FP4 dispatch seen; the MoE is running BF16"
  fi
  echo "--- untuned-kernel note (perf, not correctness) ---"
  docker exec "$CTR" bash -c "grep -ac 'no tuned FlyDSL config' '$LOG'" | sed 's/^/no tuned FlyDSL config lines: /'
  echo
fi

echo "===== 7. fault scan (expect EMPTY) ====="
docker exec "$CTR" bash -c "grep -aiE 'memory access fault|Traceback|CUDA error|HIP error|HSA_STATUS_ERROR|out of memory|Aborted|GID is NULL|unrecognized arguments|unknown option' '$LOG' | grep -v 'Ignore import error' | tail -12 | cut -c1-220"
echo "--- (if this is empty, no fault was logged) ---"
echo

echo "===== 8. router policy ====="
docker exec "$CTR" bash -c "grep -aiE 'router-policy|kv-aware|policy' /tmp/router.log | tail -3 | cut -c1-200"
