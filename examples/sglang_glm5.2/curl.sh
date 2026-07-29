#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Smoke-test the PD pair through the router (prefill node :8000). /v1/workers is
# router-only; a coherent answer means the KV hand-off over RDMA worked.
# Override: PROMPT="..." bash curl.sh
set -euo pipefail

SERVER="${SERVER:-http://127.0.0.1:8000}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
PROMPT="${PROMPT:-What is 127 * 31? Answer with the number only.}"
MAX_TOKENS="${MAX_TOKENS:-256}"

echo "== workers =="
curl -s "$SERVER/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s "$SERVER/v1/workers"

echo; echo "== chat =="
# Build the body with python so a PROMPT/MODEL containing quotes cannot break it.
BODY="$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0}))')"
curl -s "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -m json.tool 2>/dev/null || echo "(request failed — check infera_1_server.log)"
