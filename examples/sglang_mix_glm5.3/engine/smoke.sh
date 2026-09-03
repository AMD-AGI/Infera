#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Prove the deployment is real, not merely alive.
#
# A green /health says a process is listening. It does not say the AITER fast
# path is dispatching, that both memory pools exist, or that the model is
# producing sense rather than noise. Each block below is chosen because it goes
# RED when a specific feature is silently absent.
set -uo pipefail
KIT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../env.sh
source "$KIT/env.sh"
SERVED="${SERVED:-glm5.3-$VARIANT}"
R="http://$MY_IP:$ROUTER_PORT"

echo "===== 1. router sees exactly one MIXED worker ====="
curl -s "$R/v1/workers" | python3 -c '
import json,sys
d=json.load(sys.stdin); w=d if isinstance(d,list) else d.get("workers",d)
w=w if isinstance(w,list) else []
print(f"  workers: {len(w)}  (want 1)")
for x in w: print(f"    {x.get('"'"'url'"'"',x.get('"'"'worker_id'"'"'))}  disagg_mode={x.get('"'"'disagg_mode'"'"')}  (want mixed)")
'

echo "===== 2. model is served under the expected name ====="
curl -s "$R/v1/models" | python3 -c 'import json,sys; print("  ", [m["id"] for m in json.load(sys.stdin)["data"]])'

echo "===== 3. coherent answer, and reasoning separated ====="
# max_tokens is deliberately generous. GLM-5.3 is a thinking model and the leg
# passes --reasoning-parser glm45, so the chain of thought lands in
# reasoning_content but is billed against the SAME budget. At a small value the
# model spends every token thinking, content comes back empty with
# finish_reason "length", and a healthy deployment reads as a failure.
#
# Garbage or repeated tokens here is NOT a sampling problem. On the big
# variants it is the signature of the DSA-on-ROCm env block not taking effect.
curl -s "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"In two sentences, explain why prefix caching helps agentic workloads.\"}],
  \"max_tokens\": 600, \"temperature\": 0}" | python3 -c '
import json,sys
d=json.load(sys.stdin); m=d["choices"][0]["message"]
print("  content :", (m.get("content") or "")[:220])
print("  reasoning_content chars:", len(m.get("reasoning_content") or ""), "(want > 0)")
print("  finish  :", d["choices"][0].get("finish_reason"), "| usage:", d["usage"])'

echo "===== 4. arithmetic, and instruction-following ====="
curl -s "$R/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$SERVED\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is 17 * 23? Reply with only the number.\"}],
  \"max_tokens\": 512, \"temperature\": 0}" \
  | python3 -c 'import json,sys; print("  answer:", repr(json.load(sys.stdin)["choices"][0]["message"]["content"]), "(want 391)")'

echo "===== 5. engine-side evidence ====="
docker exec "$CTR" bash -c '
L=/tmp/glm53_mix.log
# THE health check that matters. Two AITER mHC lines per rank. Their absence
# does not raise: the server answers correctly and is several times slower,
# with nothing anywhere saying so. Flash variants only.
echo "  AITER mHC lines        : $(grep -c "mHC" $L)   (flash: want 2 per rank)"
# Must be ABSENT. Present means shared-experts fusion is on, which on a
# mixed-precision Quark checkpoint mis-loads the BF16 shared expert into a
# packed routed slot. See the note in worker.sh.
echo "  fusion-enabled line    : $(grep -c "Shared experts fusion optimization enabled" $L)   (want 0 on mxfp4)"
# The flash family keeps TWO pools: the paged KV pool and a KDA state pool.
# Both must appear, or the linear-attention path is not what you think it is.
echo "  decode lines w/ 2 pools: $(grep -c "full token usage.*mamba usage" $L)"
# The KDA state pool caps concurrency independently of --max-running-requests.
echo "  resolved max_running   : $(grep -o "max_running_requests is capped to [0-9]*" $L | tail -1)"
echo "  memory access fault    : $(grep -c "memory access fault" $L)   (want 0)"
echo "  HIP error              : $(grep -c "HIP error" $L)   (want 0)"
# torch._dynamo/metrics_context tracebacks are compile-telemetry noise and
# appear in healthy runs; excluded here so a real one is visible.
echo "  Traceback (non-dynamo) : $(grep "Traceback" $L | grep -vc "_dynamo")   (want 0)"
grep -m1 "max_total_num_tokens" $L | cut -c1-150
' 2>/dev/null

echo "===== 6. router policy ====="
docker exec "$CTR" grep -om1 "kv-aware" /tmp/router.log 2>/dev/null | sed 's/^/  policy: /'
