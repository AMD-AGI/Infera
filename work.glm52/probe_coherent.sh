#!/usr/bin/env bash
# 4-question coherence probe, temp=0. Same questions the e2e harness uses.
set -uo pipefail
PORT="${PORT:-30000}"
MODEL="${MODEL_NAME:-glm5.2-fp8}"
U="http://127.0.0.1:${PORT}/v1/chat/completions"

ask() {
  local q="$1"
  local body
  body=$(printf '{"model":"%s","temperature":0,"max_tokens":48,"messages":[{"role":"user","content":"%s"}]}' "$MODEL" "$q")
  local r
  r=$(curl -sf -X POST "$U" -H 'Content-Type: application/json' -d "$body" 2>&1)
  if [ -z "$r" ]; then echo "  Q: $q -> <NO RESPONSE>"; return 1; fi
  python3 -c "
import json,sys
try:
    d=json.loads(sys.stdin.read())
    m=d['choices'][0]['message']
    c=(m.get('content') or '').strip().replace('\n',' ')
    rc=(m.get('reasoning_content') or '').strip().replace('\n',' ')
    print('  Q: $q')
    print('     A:', c[:160] if c else '(empty content)')
    if rc: print('     think:', rc[:100])
except Exception as e:
    print('  Q: $q -> PARSE FAIL', e)
" <<< "$r"
}

echo "=== coherence probe (temp=0) ==="
ask "What is the capital of France? Answer with just the city name."
ask "What is the capital of China? Answer with just the city name."
ask "What is 2+2? Answer with just the number."
ask "Name the largest planet in our solar system. One word."
