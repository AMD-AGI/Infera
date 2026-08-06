#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Verify the deployment before spending a benchmark on it. Run INSIDE the engine
# container on the PREFILL node:  bash verify.sh
#
# Five assertions and one readout, each aimed at a failure this stack produces
# WITHOUT returning an error: a leg that never registered, a KV hand-off that
# yields fluent nonsense, kv-aware silently degraded to load balancing, MTP
# silently dropped, and -- only when KVD=1 -- an offload tier that stores nothing.
# The last prints the decode leg's RDMA lines to read by eye. Exits non-zero if
# any assertion fails.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

SERVER="${SERVER:-$ROUTER_URL}"
FAILED=0
pass() { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*"; FAILED=1; }

# ---------------------------------------------------------------- 1. workers
echo "== workers (expect 1 prefill + 1 decode) =="
WORKERS="$(curl -sf -m 10 "$SERVER/v1/workers" || true)"
if [[ -z "$WORKERS" ]]; then
  fail "$SERVER/v1/workers did not answer -- is the router up? ($LOG_DIR/router.log)"
else
  echo "$WORKERS" | python3 -m json.tool 2>/dev/null || echo "$WORKERS"
  # {"workers": [{"disagg_mode": "prefill", "status": "active", ...}, ...]}
  COUNTS="$(echo "$WORKERS" | python3 -c '
import json, sys
try:
    items = json.load(sys.stdin)["workers"]
    modes = [w.get("disagg_mode") for w in items if w.get("status") == "active"]
except Exception:
    modes = []
print(modes.count("prefill"), modes.count("decode"))
' 2>/dev/null || echo "0 0")"
  read -r N_PREFILL N_DECODE <<<"$COUNTS"
  if (( N_PREFILL >= 1 && N_DECODE >= 1 )); then
    pass "$N_PREFILL prefill, $N_DECODE decode registered"
  else
    fail "$N_PREFILL prefill, $N_DECODE decode -- a leg has not registered in etcd yet"
  fi
fi

# ------------------------------------------------------------ 2. correctness
# A broken KV hand-off does not return an HTTP error: the decode leg reads a
# corrupt or empty prefix and produces fluent text that has nothing to do with the
# prompt. So assert an answer that is only reachable if the prefix survived the
# transfer, rather than just checking for HTTP 200.
#
# The filler pads the prompt past one 64-token router block: a short question
# hashes to zero blocks, which would make check 3 report "not steering" on a
# perfectly healthy router. The prompt is fixed rather than overridable because
# the check below asserts its answer.
echo; echo "== correctness (127 * 31 = 3937, temperature 0) =="
FILLER="$(python3 -c 'print(("The quick brown fox jumps over the lazy dog while the engine warms its caches. " * 40).strip())')"
PROMPT="$FILLER
What is 127 * 31? Answer with the number only."
BODY="$(MODEL="$MODEL" PROMPT="$PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 64, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}))')"

REPLY="$(curl -sf -m 300 "$SERVER/v1/chat/completions" \
  -H 'Content-Type: application/json' -d "$BODY" || true)"
if [[ -z "$REPLY" ]]; then
  fail "no completion -- check $LOG_DIR/{router,prefill}.log and the decode node's log"
else
  # GLM routes text through content or reasoning_content depending on the parser;
  # read both so the check does not depend on which one this build populates.
  TEXT="$(echo "$REPLY" | python3 -c '
import json, sys
msg = json.load(sys.stdin)["choices"][0]["message"]
print(" ".join(filter(None, (msg.get("content"), msg.get("reasoning_content")))))
' 2>/dev/null || true)"
  echo "  reply: ${TEXT:0:200}"
  if [[ -z "$TEXT" ]]; then
    fail "empty completion body"
  elif [[ "$TEXT" == *3937* ]]; then
    pass "correct answer through the PD pair"
  else
    fail "wrong answer -- fluent output with a bad prefix means the KV hand-off is broken"
  fi
fi

# ------------------------------------------------------- 3. kv-aware steering
# kv-aware fails soft: with no block hashes the router still answers, still looks
# healthy, and routes on load alone. Prove it is hashing before trusting a number.
echo; echo "== router: $ROUTER_POLICY / $ROUTER_BACKEND =="
if [[ "$ROUTER_POLICY" != "kv-aware" ]]; then
  echo "  (skipped: policy is $ROUTER_POLICY)"
elif [[ ! -f "$LOG_DIR/router.log" ]]; then
  fail "$LOG_DIR/router.log is missing -- run this on the prefill node"
elif grep -qa "degrades to pure load balancing" "$LOG_DIR/router.log"; then
  fail "kv-aware degraded to load balancing -- tokenizer not loaded from $MODEL"
else
  # Both backends log request_blocks=<n> per pick, but the rust one wraps every
  # field in ANSI colour codes and capitalises the role, so match on the number
  # rather than on role=prefill. Decode picks always report 0; a non-zero anywhere
  # means the tokenizer and block hasher are live.
  BLOCKS="$(sed 's/\x1b\[[0-9;]*m//g' "$LOG_DIR/router.log" \
            | grep -aoE 'request_blocks=[0-9]+' | cut -d= -f2 | sort -n | tail -1 || true)"
  if [[ -z "$BLOCKS" ]]; then
    fail "no pick logged with request_blocks -- the request above should have made one"
  elif (( BLOCKS > 0 )); then
    pass "$BLOCKS blocks hashed on the largest pick, so the block hasher is live"
  else
    fail "every pick reports request_blocks=0 on a prompt padded past one block -- routing on load, not cache overlap"
  fi
fi

# ------------------------------------------------------------------- 4. MTP
# Speculative decoding is silently dropped if the two legs disagree on its shape.
# The acceptance series only exists on the decode leg, and only with metrics on.
echo; echo "== MTP acceptance (decode /metrics) =="
if [[ "$MTP" == "0" ]]; then
  echo "  (skipped: MTP=0)"
else
  METRICS="$(curl -sf -m 10 "$DECODE_URL/metrics" 2>/dev/null || true)"
  if [[ -z "$METRICS" ]]; then
    fail "$DECODE_URL/metrics does not answer -- relaunch the decode leg with ENGINE_METRICS=1"
  elif grep -q "^sglang:spec_accept_length" <<<"$METRICS"; then
    # -m3 rather than `| head -3`: head closing the pipe early SIGPIPEs grep, and
    # under pipefail that failure would take the script down here.
    grep -m3 "^sglang:spec_accept_length" <<<"$METRICS" | sed 's/^/    /'
    # The series existing is the assertion; the values are 0 until those ranks
    # have decoded, which a freshly launched fleet legitimately has not.
    pass "MTP $MTP_STEPS/$MTP_TOPK/$MTP_DRAFT_TOKENS accepted by the decode leg"
  else
    fail "/metrics carries no sglang:spec_accept_length -- both legs must agree on the MTP shape"
  fi
fi

# --------------------------------------------------------------------- 5. kvd
# Skipped unless the offload tier is on; KVD=0 is the default (see env.sh). When
# on, it fails silently in both directions: a hicache that never wired up writes
# nothing at all, and a page too large for its tablespace slot is REJECTED rather
# than split, which leaves L3 empty while every other check here stays green.
echo; echo "== kvd offload tier (KVD=$KVD) =="
if [[ "$KVD" != "1" ]]; then
  echo "  (skipped: the GPU radix tree is the only cache)"
else
  sets_total() { python3 -m infera.kvd.statctl --socket "$KVD_SOCKET" \
                 | python3 -c 'import json, sys; print(json.load(sys.stdin)["sets_total"])'; }
  BEFORE="$(sets_total 2>/dev/null || true)"
  if [[ -z "$BEFORE" ]]; then
    fail "no daemon answering on $KVD_SOCKET -- run launch/launch_kvd.sh, then relaunch prefill"
  else
    # hicache writes back a page at a time, so a prompt shorter than one page
    # produces no kvd traffic and would prove nothing; this one fills several.
    # The nonce matters as much: the same text on a second run is already stored,
    # writes nothing, and would fail this check on a healthy tier.
    KVD_PROMPT="run $(date +%s%N). $(python3 -c "print('The quick brown fox jumps over the lazy dog. ' * 120)") Reply with OK."
    BODY="$(MODEL="$MODEL" PROMPT="$KVD_PROMPT" python3 -c '
import json, os
print(json.dumps({
    "model": os.environ["MODEL"],
    "messages": [{"role": "user", "content": os.environ["PROMPT"]}],
    "max_tokens": 8, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}))')"
    curl -sf -m 300 "$SERVER/v1/chat/completions" \
      -H 'Content-Type: application/json' -d "$BODY" >/dev/null || true
    sleep 5   # write-back to storage is asynchronous w.r.t. the response
    AFTER="$(sets_total 2>/dev/null || echo "$BEFORE")"

    if grep -aq "value_exceeds" "$LOG_DIR/kvd.log" 2>/dev/null; then
      fail "kvd rejected pages that outgrew their slot -- raise KVD_TABLESPACE_POOLS"
      grep -a "value_exceeds" "$LOG_DIR/kvd.log" | tail -2 | sed 's/^/    /'
    elif (( AFTER > BEFORE )); then
      pass "kvd took $((AFTER - BEFORE)) writes from the prefill engine"
    else
      fail "no kvd writes ($BEFORE -> $AFTER) -- prefill should log a hicache storage backend"
      grep -aiE "hierarchical|hicache|kvd" "$LOG_DIR/prefill.log" 2>/dev/null | tail -5 | sed 's/^/    /' || true
    fi
  fi
fi

# ------------------------------------------------------------ 6. RDMA hand-off
# Informational: decode.log lives on the other node unless LOG_DIR is shared.
echo; echo "== RDMA hand-off (decode log, rail=$IB_DEVICE) =="
if [[ -f "$LOG_DIR/decode.log" ]]; then
  grep -aE "GID index|installTransport|mooncake" "$LOG_DIR/decode.log" | tail -5 | sed 's/^/    /' \
    || echo "    (no hand-off lines yet)"
else
  echo "    (decode.log is on $DECODE_NODE; check it there for 'mooncake' transport lines)"
fi

echo
if (( FAILED )); then
  echo "VERIFY: FAILED -- do not benchmark this deployment yet."
  exit 1
fi
echo "VERIFY: all checks passed."
