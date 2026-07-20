#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# Smoke-test the PD router (node-0:8000); hits /v1/workers, which is router-only (not on a single-node engine). Override: PROMPT="..." bash curl.sh
set -euo pipefail

SERVER="${SERVER:-http://127.0.0.1:8000}"
: "${MODEL:?set MODEL=/path/to/Kimi-K2.6-MXFP4 (must match the model path the router/engines were launched with)}"
PROMPT="${PROMPT:-What is 1+1? Answer directly.}"
MAX_TOKENS="${MAX_TOKENS:-64}"

echo "== workers =="
curl -s "$SERVER/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s "$SERVER/v1/workers"

echo; echo "== chat =="
# Build the JSON body with python so a PROMPT/MODEL containing quotes can't break it.
BODY="$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "max_tokens": int(os.environ["MAX_TOKENS"])}))')"
curl -s "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
    | python3 -m json.tool 2>/dev/null || echo "(request failed — check: docker logs infera-sgl-server)"
