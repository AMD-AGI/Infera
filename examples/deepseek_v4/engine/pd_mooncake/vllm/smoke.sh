#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: wait for router, smoke a completion through it. why: prove P/D pairing + KV hand-off.
# how: uses /v1/completions (vllm PD serves via completions). caller = user.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../../../common.sh"
require_env PREFILL_IP; require_env INFERA_MODEL
URL="http://$PREFILL_IP:$ROUTER_PORT"
wait_health "$URL/health" 200 || die "router health never came up"
log "workers:"; curl -s "$URL/v1/workers" | python3 -m json.tool 2>/dev/null || true
R=$(curl -s "$URL/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$INFERA_MODEL\",\"prompt\":\"The capital of France is\",\"max_tokens\":16,\"temperature\":0}")
echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['text'])" \
  || die "smoke failed: $R"
log "vllm PD smoke OK"
