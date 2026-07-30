#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Is prefix-aware routing actually doing anything? Two turns that share a long
# prefix, sent through the router. Turn 2 should land on the rank that already
# holds turn 1's blocks, so it reports a large cached_tokens and a much lower
# TTFT. Round-robin gets that by luck 1/dp_size of the time.
#
# Reads three independent signals, because any one of them can be misleading:
#   - the engine's own accounting (usage.prompt_tokens_details.cached_tokens,
#     needs --enable-cache-report on the leg that prefills)
#   - the router's decision (pick ... cache_hits=N request_blocks=M in its log)
#   - wall-clock latency, which is the only one the user feels
#
# Usage: bash probe_kv_aware.sh [turns]
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
MODEL="${MODEL:-/wekafs/models/GLM-5.2-FP8}"
ROUTER_LOG="${ROUTER_LOG:-$(dirname "$0")/infera_1_server.log}"
TURNS="${1:-3}"
# ~1.1 tokens per word here, so 6000 words lands near 6.5k tokens: enough to
# span ~100 blocks at page_size 64 and to make a recompute obvious in the clock.
WORDS="${WORDS:-6000}"

prompt_file=$(mktemp)
trap 'rm -f "$prompt_file"' EXIT

python3 - "$prompt_file" "$WORDS" <<'PY'
import sys
path, words = sys.argv[1], int(sys.argv[2])
# Deterministic, non-repeating-enough text. A single repeated sentence would let
# the engine's radix tree collapse it, which would flatter the cache.
lines = []
for i in range(words // 10):
    lines.append(
        f"Section {i}: the module at index {i} validates tensor shape "
        f"{i * 7 % 97} against the manifest checksum {i * 13 % 89}."
    )
with open(path, "w") as f:
    f.write("\n".join(lines))
PY

echo "prefix: $(wc -w < "$prompt_file") words"
echo

for turn in $(seq 1 "$TURNS"); do
    payload=$(python3 - "$prompt_file" "$MODEL" "$turn" <<'PY'
import json, sys
path, model, turn = sys.argv[1], sys.argv[2], int(sys.argv[3])
base = open(path).read()
# Each turn appends a short delta, so the shared prefix grows monotonically —
# the shape a code agent produces between tool calls.
content = base + "\n" + "\n".join(
    f"Follow-up {i}: confirm section {i * 3} is unchanged." for i in range(turn)
)
content += "\n\nReply with the single word OK."
print(json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": content}],
    "max_tokens": 400,
    "temperature": 0,
}))
PY
)
    start=$(date +%s.%N)
    resp=$(curl -s -m 600 "$BASE/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$payload")
    end=$(date +%s.%N)
    python3 - "$resp" "$turn" "$start" "$end" <<'PY'
import json, sys
resp, turn = sys.argv[1], sys.argv[2]
elapsed = float(sys.argv[4]) - float(sys.argv[3])
try:
    d = json.loads(resp)
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    cached = det.get("cached_tokens")
    pt = u.get("prompt_tokens", 0)
    pct = f"{100.0 * cached / pt:5.1f}%" if cached is not None and pt else "  n/a"
    print(f"turn {turn}: {elapsed:7.2f}s  prompt={pt:>7}  "
          f"cached={str(cached):>7} ({pct})")
except Exception as exc:
    print(f"turn {turn}: unparseable response ({exc}): {resp[:200]}")
PY
done

echo
echo "=== router pick decisions (last $TURNS) ==="
grep -a "pick policy=" "$ROUTER_LOG" 2>/dev/null \
    | grep -oE "role=[a-z]+ .*picked=[^ ]+ cache_hits=[0-9]+ request_blocks=[0-9]+" \
    | tail -$((TURNS * 2)) || echo "(no pick lines — is --router-policy kv-aware set?)"
