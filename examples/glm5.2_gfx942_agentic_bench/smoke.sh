#!/usr/bin/env bash
# Smoke-test the router: worker list, one chat request, and the decode leg's RDMA
# hand-off lines. Run on the prefill node inside the container.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

SERVER="${SERVER:-$ROUTER_URL}"
PROMPT="${PROMPT:-What is 127 * 31? Answer with the number only.}"
MAX_TOKENS="${MAX_TOKENS:-128}"

echo "== workers =="   # expect one prefill + one decode
curl -s "$SERVER/v1/workers" | python3 -m json.tool 2>/dev/null || curl -s "$SERVER/v1/workers"

echo; echo "== chat =="
# Build the JSON body with python so a PROMPT/MODEL containing quotes can't break it.
BODY="$(MODEL="$MODEL" PROMPT="$PROMPT" MAX_TOKENS="$MAX_TOKENS" python3 -c 'import json, os; print(json.dumps({"model": os.environ["MODEL"], "messages": [{"role": "user", "content": os.environ["PROMPT"]}], "max_tokens": int(os.environ["MAX_TOKENS"]), "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}))')"
curl -s "$SERVER/v1/chat/completions" -H 'Content-Type: application/json' -d "$BODY" \
  | python3 -m json.tool 2>/dev/null \
  || { echo "(request failed; check $LOG_DIR/{router,prefill,decode}.log)"; exit 1; }

echo; echo "== rdma hand-off (decode log) =="
if [[ -f "$LOG_DIR/decode.log" ]]; then
  grep -aE "GID index|installTransport|mooncake" "$LOG_DIR/decode.log" | tail -5 \
    || echo "(no hand-off lines yet)"
else
  echo "(decode.log is on $DECODE_NODE)"
fi
