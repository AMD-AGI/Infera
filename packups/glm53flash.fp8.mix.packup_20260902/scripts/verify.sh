#!/bin/bash
# Correctness + health battery for the GLM-5.3-Flash FP8 MIX run. Runs on the
# HOST; talks to the container by exec so it does not depend on host curl
# reaching the container's ports.
#
# "No errors in the log" is NOT the health check here. The two that matter:
#   * 8 AITER mHC lines (2 per rank x 4 ranks). Without them the server still
#     starts, still answers correctly, and is 4.3-5.4x slower with nothing in
#     any log saying so.
#   * decode lines carrying BOTH `full token usage` (paged KV pool) and
#     `mamba usage` (the KDA state pool). A missing second pool means the KDA
#     allocator clamped and max_running_requests is silently capped.
set -u
CTR="${CTR:-yihou_f8_mix}"
MY_IP="${MY_IP:-$(hostname -I | awk '{print $1}')}"
PORT="${PORT:-31400}"
ROUTER_PORT="${ROUTER_PORT:-18105}"
LOG="${LOG:-/tmp/glm53_f8_mix.log}"
OUT="${OUT:-/apps/yihou/glm53.series.workspace_20260901/flash-fp8-0529/verify_$(date +%H%M%S).txt}"
X() { docker exec "$CTR" bash -lc "$1"; }

exec > >(tee "$OUT") 2>&1
echo "########## verify $(date -Is)  ctr=$CTR ##########"

echo; echo "===== A. infera worker registry (expect 1 worker, disagg_mode mixed) ====="
X "curl -s -m10 http://$MY_IP:$ROUTER_PORT/v1/workers" | python3 -m json.tool 2>/dev/null \
  || X "curl -s -m10 http://$MY_IP:$ROUTER_PORT/v1/workers"

echo; echo "===== B. /v1/models ====="
X "curl -s -m10 http://$MY_IP:$ROUTER_PORT/v1/models" | python3 -m json.tool 2>/dev/null \
  || X "curl -s -m10 http://$MY_IP:$ROUTER_PORT/v1/models"

echo; echo "===== C. arithmetic: 17 * 23 (expect 391) ====="
X "curl -s -m300 http://$MY_IP:$ROUTER_PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"glm5.3-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 23? Reply with just the number.\"}],\"max_tokens\":512,\"temperature\":0}'" \
  > /tmp/f8_c.json
python3 - <<'EOF'
import json
d = json.load(open("/tmp/f8_c.json"))
m = d["choices"][0]["message"]
rc = (m.get("reasoning_content") or "")
c  = (m.get("content") or "")
print("reasoning_content len:", len(rc), "| head:", rc[:200].replace("\n", " "))
print("content:", repr(c[:400]))
print("VERDICT 391 in content:", "391" in c)
print("VERDICT reasoning separated from content:", bool(rc) and "391" in c)
EOF

echo; echo "===== D. coherence: a short open question ====="
X "curl -s -m300 http://$MY_IP:$ROUTER_PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"glm5.3-flash\",\"messages\":[{\"role\":\"user\",\"content\":\"In two sentences, why is the sky blue?\"}],\"max_tokens\":512,\"temperature\":0}'" \
  > /tmp/f8_d.json
python3 - <<'EOF'
import json
m = json.load(open("/tmp/f8_d.json"))["choices"][0]["message"]
print("reasoning_content len:", len(m.get("reasoning_content") or ""))
print("content:", (m.get("content") or "")[:600])
EOF

echo; echo "===== E. AITER mHC lines (EXPECT 4 + 4 = 8 total, one of each per rank) ====="
echo "-- 'Using AITER gfx950 mHC pre/post kernels':"
X "grep -c 'Using AITER gfx950 mHC pre/post kernels' $LOG || true"
echo "-- 'Using fused AITER mHC attention-to-FFN boundary':"
X "grep -c 'Using fused AITER mHC attention-to-FFN boundary' $LOG || true"
echo "-- any line mentioning mHC (sample):"
X "grep -i 'mhc' $LOG | sort | uniq -c | head -20 || true"

echo; echo "===== F. both memory pools on decode lines ====="
X "grep -o 'Decode batch.*' $LOG | tail -5 || true"
echo "-- count of decode lines with BOTH 'full token usage' and 'mamba usage':"
X "grep 'Decode batch' $LOG | grep 'full token usage' | grep -c 'mamba usage' || true"
echo "-- count of decode lines total:"
X "grep -c 'Decode batch' $LOG || true"
echo "-- max_running_requests as the engine settled on it:"
X "grep -iE 'max_running_requests|Memory pool end|KV Cache is allocated|mamba' $LOG | head -20 || true"

echo; echo "===== G. shared-expert fusion: which arm actually ran ====="
X "grep -iE 'shared.experts|_load_w13|_load_w2' $LOG | head -20 || true"

echo; echo "===== H. fault scan ====="
echo "-- raw matches for 'memory access fault|HIP error|Traceback':"
RAW=$(X "grep -cE 'memory access fault|HIP error|Traceback' $LOG || true")
echo "   raw = $RAW"
echo "-- benign torch telemetry excluded (_dynamo/metrics_context):"
X "grep -nE 'memory access fault|HIP error|Traceback' $LOG | grep -c '_dynamo\|metrics_context' || true"
echo "-- NON-benign context (each Traceback with 6 following lines, telemetry filtered):"
X "grep -A6 -E 'memory access fault|HIP error|Traceback' $LOG | grep -vE '_dynamo|metrics_context' | head -60 || true"

echo; echo "===== I. VRAM after load (ours is GPUs 4-7) ====="
rocm-smi --showmeminfo vram 2>/dev/null | grep -E 'GPU\[[4-7]\].*Used' || true

echo; echo "########## end ##########"
echo "saved to $OUT"
