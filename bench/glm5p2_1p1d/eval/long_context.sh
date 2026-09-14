#!/usr/bin/env bash
# Exercise the pinned Aiter large-buffer fallback near the context limit.
# Usage: LONG_CONTEXT_TOKENS=250000 bash eval/long_context.sh  # requires a running stack
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

URL="http://$PREFILL_IP:$ROUTER_PORT"
TOKENS="$LONG_CONTEXT_TOKENS"
OUT="${LONG_CONTEXT_OUT_DIR:-$RUN_ROOT/correctness/long-context}"
rm -rf "$OUT"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1

flush_cache() {
    local role="$1" url="$2" attempt status
    local body="$OUT/flush_${role}.txt"
    for attempt in $(seq 1 30); do
        if ! status="$(curl -sS -m 20 -o "$body" -w '%{http_code}' \
            -X POST "$url/flush_cache")"; then
            echo "[long-context] $role flush request failed" >&2
            return 1
        fi
        [[ "$status" == "200" ]] && return 0
        if [[ "$status" == "400" && "$attempt" -lt 30 ]]; then
            sleep 2
            continue
        fi
        echo "[long-context] $role flush failed with HTTP $status: $(<"$body")" >&2
        return 1
    done
}

# SGLang returns HTTP 400 until its scheduler is idle.
flush_cache prefill "http://$PREFILL_IP:$PREFILL_PORT"
flush_cache decode "http://$DECODE_IP:$DECODE_PORT"

URL="$URL" SERVED_MODEL="$SERVED_MODEL" TOKENS="$TOKENS" OUT="$OUT" python3 - <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path

requested_tokens = int(os.environ["TOKENS"])
payload = json.dumps(
    {
        "model": os.environ["SERVED_MODEL"],
        "messages": [
            {
                "role": "user",
                "content": (" token" * requested_tokens) + "\nReply OK.",
            }
        ],
        "max_tokens": 4,
        "temperature": 0.0,
    }
).encode()
request = urllib.request.Request(
    f"{os.environ['URL']}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
)
start = time.monotonic()
with urllib.request.urlopen(request, timeout=900) as response:
    result = json.load(response)
elapsed = time.monotonic() - start

assert result.get("choices"), result
assert result["usage"]["prompt_tokens"] >= requested_tokens, result["usage"]
Path(os.environ["OUT"], "response.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
summary = {
    "elapsed_seconds": round(elapsed, 2),
    "prompt_tokens": result["usage"]["prompt_tokens"],
    "completion_tokens": result["usage"]["completion_tokens"],
    "finish_reason": result["choices"][0]["finish_reason"],
}
Path(os.environ["OUT"], "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
print(json.dumps(summary))
PY

echo "[long-context] PASS; artifacts: $OUT"
