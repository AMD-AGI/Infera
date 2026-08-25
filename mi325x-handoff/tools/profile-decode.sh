#!/usr/bin/env bash
# Capture a torch profiler trace of the decode leg over a handful of decode steps
# at concurrency 1, which is the operating point the tok/s/user target lives at.
#
# The request is fired first and profiling starts once it is past prefill, so the
# captured steps are decode steps rather than an idle scheduler loop.
#
#   MODEL=... bash profile-decode.sh <router host:port> <decode url> <out dir>
set -euo pipefail

ROUTER="${1:?usage: profile-decode.sh <router host:port> <decode url> <out dir>}"
DECODE="${2:?}"
OUT="${3:?}"
MODEL="${MODEL:?MODEL must point at the weights dir}"

STEPS="${STEPS:-15}"
OSL="${OSL:-400}"        # long enough that decode is still running when profiling ends
ISL_WORDS="${ISL_WORDS:-2000}"
TTFT_WAIT="${TTFT_WAIT:-3}"

mkdir -p "$OUT"

code=$(curl -s -m 60 -o /dev/null -w '%{http_code}' -X POST "$DECODE/flush_cache")
[ "$code" = "200" ] || { echo "[prof] decode flush returned $code, fleet not idle" >&2; exit 1; }

prompt=$(python3 -c "print(' '.join(['token']*$ISL_WORDS))")
python3 - "$ROUTER" "$MODEL" "$OSL" "$prompt" <<'PY' &
import json, sys, urllib.request
router, model, osl, prompt = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": osl, "temperature": 0.0, "stream": False,
}).encode()
req = urllib.request.Request(f"http://{router}/v1/chat/completions", body,
                             {"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=300).read()
PY
REQ_PID=$!

sleep "$TTFT_WAIT"
echo "[prof] capturing $STEPS decode steps -> $OUT"
curl -s -m 60 -X POST "$DECODE/start_profile" -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$OUT\",\"num_steps\":$STEPS,\"activities\":[\"CPU\",\"GPU\"],\"record_shapes\":true}"
echo

wait "$REQ_PID" || true
# The profiler flushes asynchronously after the last captured step.
for _ in $(seq 30); do
  sleep 2
  ls "$OUT"/*.trace.json.gz >/dev/null 2>&1 && break
done
ls -la "$OUT"
