#!/usr/bin/env bash
# Purpose: the R03 acceptance probe. Reproduces R01's exact ladder so the two
#   rounds can be diffed directly, then adds the two target prompts and a real
#   /metrics read of MTP acceptance.
# Usage: ./probe.yihou.sh <out_dir>
# Artifacts: <out_dir>/ladder-<n>.json, target-{arith,nonce}.json,
#   metrics-{idle,loaded}-dp<N>.txt, summary.md
set -euo pipefail

OUT="${1:?usage: probe.yihou.sh <out_dir>}"
mkdir -p "$OUT"

ROUTER="${ROUTER:-http://10.245.148.209:28000}"   # control node 135
DECODE_HOST="${DECODE_HOST:-10.245.157.237}"      # 138
DECODE_PORT_BASE="${DECODE_PORT_BASE:-29002}"     # decode engine, dp ranks share the port
MODEL="${MODEL_NAME:-glm5.2-mxfp4}"

say() { printf '\n=== %s ===\n' "$*"; }

ask() {  # ask <outfile> <max_tokens> <prompt>
    local out="$1" maxtok="$2" prompt="$3"
    curl -sS --max-time 180 "$ROUTER/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[3]}],
                  "temperature":0,"max_tokens":int(sys.argv[2])}))' \
            "$MODEL" "$maxtok" "$prompt")" >"$out"
    python3 - "$out" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
m=d["choices"][0]["message"]
txt=m.get("content") or ""
rsn=m.get("reasoning_content") or ""
u=d.get("usage",{})
print(f"  content={txt!r}")
print(f"  reasoning={rsn!r}")
print(f"  completion_tokens={u.get('completion_tokens')} reasoning_tokens={u.get('reasoning_tokens')}")
PY
}

# --- 1. R01's ladder, verbatim -- this is the 对拍 reference -------------------
# R01 recorded: max_tokens 1 -> "1"; 3 -> "1!!"; 8 -> "1!!!!!!!". The first token
# was right and every later one degenerated. Same prompt, same temperature, same
# max_tokens, so the two rounds are directly comparable.
say "ladder (R01 reference: 1 / 1!! / 1!!!!!!!)"
for n in 1 3 8; do
    echo "max_tokens=$n"
    ask "$OUT/ladder-$n.json" "$n" "What is 2+2? Answer with a single digit."
done

# --- 2. the two target prompts ------------------------------------------------
say "target 1 -- arithmetic, must not degenerate"
ask "$OUT/target-arith.json" 64 "What is 2+2? Answer with a single digit."

NONCE="YIHOU$(date -u +%H%M%S)"
say "target 2 -- must reproduce nonce $NONCE"
ask "$OUT/target-nonce.json" 64 "Reply with exactly: BASELINE_OK_$NONCE"
echo "$NONCE" >"$OUT/nonce.txt"

# --- 3. acceptance, read the RIGHT way ----------------------------------------
# TRAP: a rank that has served no decode tokens reports spec_accept_length 0.0,
# which is indistinguishable from "MTP accepts nothing". So snapshot the idle
# gauge first, then generate real decode traffic, then read again -- and only
# trust ranks whose counters actually moved.
say "metrics BEFORE load (expected to be uninformative -- recorded to prove the trap)"
curl -sS --max-time 30 "http://$DECODE_HOST:$DECODE_PORT_BASE/metrics" \
    >"$OUT/metrics-idle.txt" || echo "  (scrape failed)"
grep -E 'spec_accept_(length|rate)' "$OUT/metrics-idle.txt" | head -20 || true

say "generating real decode traffic (8 concurrent, 400 tokens each)"
for i in $(seq 1 8); do
    ask "$OUT/load-$i.json" 400 \
        "Write a detailed paragraph about distributed systems, part $i." >/dev/null 2>&1 &
done
wait
echo "  done"

say "metrics AFTER load -- this is the measurement"
curl -sS --max-time 30 "http://$DECODE_HOST:$DECODE_PORT_BASE/metrics" \
    >"$OUT/metrics-loaded.txt" || echo "  (scrape failed)"
grep -E 'spec_accept_(length|rate)' "$OUT/metrics-loaded.txt" || true

# --- 4. the same reading R01 took, from the same place ------------------------
# R01 read `accept len: 1.25, accept rate: 0.05` out of the decode server log
# (metrics_reporter.py:928 emits it every decode_log_interval steps under load).
# Reading the SAME source keeps the two rounds directly comparable; /metrics
# above is the independent second source.
if [[ -n "${DECODE_LOG:-}" && -f "$DECODE_LOG" ]]; then
    say "decode-log acceptance (same source as R01)"
    grep -oE 'accept len: [0-9.]+, accept rate: [0-9.]+' "$DECODE_LOG" \
        | tail -20 | tee "$OUT/accept-from-log.txt" || echo "  (no decode-batch lines yet)"
else
    echo "(set DECODE_LOG=<server-logs/decode-0.log> to also capture R01's source)"
fi

say "verdict"
python3 - "$OUT" <<'PY'
import json, os, re, sys
out = sys.argv[1]

def msg(p):
    d = json.load(open(os.path.join(out, p)))
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + (m.get("reasoning_content") or "")

lad = {n: msg(f"ladder-{n}.json") for n in (1, 3, 8)}
print("ladder:", lad)
# R01's signature: the first token is right, every later one is the same filler.
tail = lad[8][1:]
degenerate = len(set(tail)) == 1 and len(tail) > 1
print("degenerate-tail (R01 signature):", degenerate)

nonce = open(os.path.join(out, "nonce.txt")).read().strip()
got = msg("target-nonce.json")
print("nonce reproduced:", nonce in got)

vals = []
for line in open(os.path.join(out, "metrics-loaded.txt")):
    m = re.match(r'sglang:spec_accept_length\{([^}]*)\}\s+([\d.eE+-]+)', line)
    if m:
        vals.append((m.group(1), float(m.group(2))))
active = [v for _, v in vals if v > 0]
print("spec_accept_length, non-idle ranks:", active)
if active:
    print("min over active ranks:", min(active), "  PASS" if min(active) >= 2.0 else "  FAIL (bar is 2.0)")
else:
    print("no active rank reported -- INCONCLUSIVE, not a zero measurement")
PY
