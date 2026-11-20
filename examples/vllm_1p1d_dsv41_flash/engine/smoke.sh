#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: service check PLUS positive evidence that each feature of this deployment took effect.
# why : a green /health proves the processes are alive — not that P and D paired, that KV moved
#       over RDMA, that both legs run DP2 with fp8 KV, that UMBP is wired, or that DSpark drafts.
#       Every check below reads a signal that goes RED when its feature is silently absent.
# how : bash cluster/<your-cluster>.sh smoke    (exit 1 if any hard check fails)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"
require_env PREFILL_NODE; require_env PREFILL_IP; require_env DECODE_NODE
URL="http://$PREFILL_IP:$ROUTER_PORT"
PLOG="$LOG_DIR/prefill.log" DLOG="$LOG_DIR/decode.log"
bad=0
check(){ if [ "$2" = ok ]; then echo -e "  ${GRN}ok${NC}   $1"; else echo -e "  ${RED}FAIL${NC} $1"; bad=1; fi; }
count(){ grep -ac -- "$2" "$1" 2>/dev/null || true; }
chat(){  # chat PROMPT MAX_TOKENS [STREAM] -> response body on stdout
  python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}],
    "max_tokens": int(sys.argv[3]), "stream": sys.argv[4] == "1"}))' "$SERVED" "$1" "$2" "${3:-0}" \
  | curl -s -m 900 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d @-
}
nic_tx(){ on "$PREFILL_NODE" "cat /sys/class/infiniband/*/ports/1/counters/port_xmit_data 2>/dev/null | awk '{s+=\$1} END {print s+0}'"; }

echo "===== 1. router + PD pairing ====="
# One prefill and one decode worker, each with its DP ranks. A single worker means the second leg
# never registered in etcd — which a per-leg /health check cannot tell you.
w=$(curl -s -m10 "$URL/v1/workers")
echo "$w" | python3 -c 'import json,sys
for x in json.load(sys.stdin).get("workers", []):
    print("  ", x.get("disagg_mode"), x.get("url"), "dp_size=%s" % x.get("dp_size"), "kv_events=%s" % bool(x.get("kv_events_endpoint")))' 2>/dev/null
np=$(echo "$w" | grep -o '"disagg_mode": *"prefill"' | wc -l); nd=$(echo "$w" | grep -o '"disagg_mode": *"decode"' | wc -l)
check "router sees $np prefill + $nd decode worker(s) (want 1 + 1)" "$([ "$np" -ge 1 ] && [ "$nd" -ge 1 ] && echo ok)"

echo; echo "===== 2. completions through the router ====="
# max_tokens 512 on purpose: DeepSeek-V4.1 thinks first and the reasoning is billed against the
# same budget; at a small value content comes back empty with finish_reason=length.
r=$(chat "What is the capital of France? Answer in one short sentence." 512)
echo "$r" | python3 -c 'import json,sys
d=json.load(sys.stdin); c=d["choices"][0]; m=c["message"]; u=d.get("usage", {})
print("   content  :", (m.get("content") or "").strip()[:120])
print("   reasoning:", (m.get("reasoning_content") or m.get("reasoning") or "")[:80].replace("\n", " "), "...")
print("   finish   :", c.get("finish_reason"), "| usage", u.get("prompt_tokens"), "prompt /", u.get("completion_tokens"), "completion")' 2>/dev/null \
  || echo "   raw: ${r:0:300}"
check "non-streaming answer names Paris" "$(echo "$r" | grep -q Paris && echo ok)"
s=$(chat "Count from 1 to 20, separated by commas." 512 1)
ev=$(echo "$s" | grep -c '^data: {'); done_=$(echo "$s" | grep -c '^data: \[DONE\]')
check "streaming: $ev SSE chunks, [DONE]=$done_" "$([ "$ev" -gt 5 ] && [ "$done_" = 1 ] && echo ok)"

echo; echo "===== 3. KV moves over RDMA ====="
# A ~16k-token prompt forces a real KV transfer; the NIC byte counters must move with it. With
# MC_DISABLE_HIP=1 Mooncake has no HIP-IPC shortcut even when both legs share a node, so a flat
# counter means the KV took another path (TCP) — that path WORKS, it is just 5-20x slower.
tx0=$(nic_tx)
long=$(python3 -c 'import random; r=random.Random(7); w="river mountain window garden lantern bridge tower valley ocean island".split(); print(" ".join(r.choice(w) for _ in range(12000)) + " ... How many words were there, roughly? One number.")')
chat "$long" 64 >/dev/null
tx1=$(nic_tx)
mb=$(( (tx1 - tx0) * 4 / 1000000 ))   # port_xmit_data counts 4-byte words
check "RDMA NICs on the prefill node sent ${mb} MB during one 16k-token request" "$([ "$mb" -gt 10 ] && echo ok)"
for f in "$PLOG" "$DLOG"; do
  printf '   %-8s HCAs=%s  tcp-fallback=%s  handshake-errors=%s\n' "$(basename "$f" .log)" \
    "$(grep -aoE 'Found [0-9]+ HCAs?' "$f" | head -1 | grep -oE '[0-9]+')" \
    "$(count "$f" 'MC_FORCE_TCP\|falling back to TCP\|Fallback to TCP')" "$(count "$f" 'handshake.*fail\|GID is NULL')"
done
n_to=$(count "$DLOG" "Timeout waiting for P side ready")
check "decode never timed out waiting for a prefill send ($n_to)" "$([ "$n_to" = 0 ] && echo ok)"

echo; echo "===== 4. resolved engine settings (read back from the engines' own logs) ====="
for f in "$PLOG" "$DLOG"; do
  leg=$(basename "$f" .log)
  dp=$(grep -aoE "data_parallel_size[=:] ?[0-9]+" "$f" | head -1 | grep -oE '[0-9]+$')
  kvd=$(grep -aoE "kv_cache_dtype[=:] ?'?[a-z0-9_]+" "$f" | head -1 | grep -oE "[a-z0-9_]+$")
  kvc=$(grep -aoE 'GPU KV cache size: [0-9,]+ tokens' "$f" | head -1 | grep -oE '[0-9,]+')
  echo "   $leg: dp=${dp:-?} kv_cache_dtype=${kvd:-?} KV cache/rank=${kvc:-?} tokens"
  check "$leg runs DP2 with fp8_e4m3 KV" "$([ "${dp:-}" = 2 ] && [ "${kvd:-}" = fp8_e4m3 ] && echo ok)"
done
if [ "${MR_MODE:-}" = gem ]; then
  g=$(count "$PLOG" "KV cache allocated from amdgpu GEM BOs"); g2=$(count "$DLOG" "KV cache allocated from amdgpu GEM BOs")
  check "GEM KV allocator active (prefill $g, decode $g2 ranks)" "$([ "$g" -ge 1 ] && [ "$g2" -ge 1 ] && echo ok)"
fi
if [ "${UMBP:-1}" = "1" ]; then
  u=$(grep -aciE 'UMBPStoreConnector|umbp' "$PLOG")
  check "prefill wired to UMBP ($u log lines; server log: $(grep -ac 'StandaloneServer' "$LOG_DIR/umbp.log" 2>/dev/null))" "$([ "$u" -ge 1 ] && echo ok)"
fi

echo; echo "===== 5. DSpark on decode ====="
# Block (real) verification: a healthy mean acceptance on chat traffic is ~2.5-3.8 of 6. Synthetic
# (golden) mode reports the configured value instead — that is expected, not a measurement.
grep -aoE 'Mean acceptance length: [0-9.]+' "$DLOG" | tail -3 | sed 's/^/   /'
grep -aoE "'rejection_sample_method': '[a-z]+'|synthetic_acceptance_length[^,}]*" "$DLOG" | head -2 | sed 's/^/   /'

echo; echo "===== 6. kv-aware routing inputs ====="
r_tok=$(grep -ac 'loaded tokenizer' "$LOG_DIR/router.log"); r_sub=$(grep -aciE 'subscrib' "$LOG_DIR/router.log")
echo "   router: tokenizer-loaded=$r_tok kv-event-subscription-lines=$r_sub"

echo
[ "$bad" = 0 ] && log "smoke PASSED" || { warn "smoke FAILED — read the blocks above"; exit 1; }
