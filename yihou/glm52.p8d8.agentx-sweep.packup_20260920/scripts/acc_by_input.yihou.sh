#!/usr/bin/env bash
# Purpose: test whether MTP acceptance length is driven by the INPUT rather than
#   by configuration. Same service, same engine process, same everything --
#   only the prompt class changes. This is the discriminating experiment for
#   "why is acceptance 4.7 here but 2.3-2.57 on AgentX".
# Usage: ./acc_by_input.yihou.sh <out_dir>
# Artifacts: <out_dir>/<class>/req-NN.json, <out_dir>/<class>/metrics.txt,
#            <out_dir>/summary.csv
#
# Method and why it is sound:
#   * `spec_accept_length` is a gauge holding a RECENT mean, so it tracks the
#     traffic that just ran. Each class is therefore run in isolation, followed
#     by a quiet period, and scraped immediately after that class drains.
#   * Classes are ordered from most to least predictable. If acceptance falls
#     monotonically with predictability, the input explains the gap and no
#     configuration difference need be invoked.
#   * Every class uses the SAME max_tokens and the SAME concurrency, so output
#     length and batch shape are not free variables.
set -uo pipefail

OUT="${1:?usage: acc_by_input.yihou.sh <out_dir>}"
mkdir -p "$OUT"
ROUTER="${ROUTER:-http://10.245.153.247:28000}"
DECODE_HOST="${DECODE_HOST:-10.245.154.168}"
DECODE_PORT="${DECODE_PORT:-29002}"
MODEL="${MODEL_NAME:-glm5.2-mxfp4}"
N="${N:-8}"
MAXTOK="${MAXTOK:-512}"
QUIET="${QUIET:-25}"

# class name | prompt.  Ordered most-predictable first.
read -r -d '' CLASSES <<'EOF' || true
memorized|List the first 10 prime numbers, separated by commas, then explain in one sentence why 1 is not prime.
boilerplate|Write a standard Python function named fibonacci that returns the nth Fibonacci number using memoization, with a docstring and type hints.
codeedit|Here is a Python file. Add error handling for a missing key and return None instead of raising.\n\ndef lookup(cfg, key):\n    return cfg[key]\n\nReturn only the rewritten function.
freeform|Invent an unusual metaphor for the feeling of debugging a distributed system at 3am, then write a short original paragraph developing it. Do not use any common cliche.
adversarial|Produce 40 random-looking lowercase alphanumeric tokens of length 6, separated by spaces. Do not repeat any token and do not use dictionary words.
EOF

echo "class,n,maxtok,mean_accept_length,min,max,mean_accept_rate" > "$OUT/summary.csv"

scrape_ranks() {  # scrape_ranks <file> -> prints "mean min max meanrate"
    curl -s -m 15 "http://$DECODE_HOST:$DECODE_PORT/metrics" > "$1" 2>/dev/null
    python3 - "$1" <<'PY'
import re, sys
acc, rate = [], []
for line in open(sys.argv[1]):
    m = re.match(r'sglang:spec_accept_length\{[^}]*\}\s+([\d.eE+-]+)', line)
    if m:
        acc.append(float(m.group(1))); continue
    m = re.match(r'sglang:spec_accept_rate\{[^}]*\}\s+([\d.eE+-]+)', line)
    if m:
        rate.append(float(m.group(1)))
acc = [a for a in acc if a > 0]
rate = [r for r in rate if r > 0]
if not acc:
    print("NA NA NA NA")
else:
    print(f"{sum(acc)/len(acc):.4f} {min(acc):.4f} {max(acc):.4f} "
          f"{(sum(rate)/len(rate) if rate else float('nan')):.4f}")
PY
}

while IFS='|' read -r cls prompt; do
    [[ -n "$cls" ]] || continue
    d="$OUT/$cls"; mkdir -p "$d"
    printf '\n=== class %s ===\n' "$cls"
    # Let the previous class drain so the gauge reflects THIS class only.
    sleep "$QUIET"
    pids=()
    for i in $(seq 1 "$N"); do
        payload="$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],
                  "messages":[{"role":"user","content":sys.argv[2].replace("\\n","\n")}],
                  "temperature":0,"max_tokens":int(sys.argv[3])}))' \
            "$MODEL" "$prompt" "$MAXTOK")"
        curl -sS --max-time 300 "$ROUTER/v1/chat/completions" \
            -H 'Content-Type: application/json' -d "$payload" \
            > "$d/req-$(printf '%02d' "$i").json" &
        pids+=("$!")
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    read -r mean mn mx mrate <<< "$(scrape_ranks "$d/metrics.txt")"
    echo "  accept_length mean=$mean min=$mn max=$mx  accept_rate=$mrate"
    # Show one completion so a garbled class is not silently scored.
    python3 - "$d" <<'PY'
import glob, json, os, sys
p = sorted(glob.glob(os.path.join(sys.argv[1], "req-*.json")))
if p:
    try:
        m = json.load(open(p[0]))["choices"][0]["message"]
        t = (m.get("content") or "") or (m.get("reasoning_content") or "")
        print(f"  sample: {t[:150]!r}")
    except Exception as e:
        print(f"  sample: <unreadable: {e}>")
PY
    echo "$cls,$N,$MAXTOK,$mean,$mn,$mx,$mrate" >> "$OUT/summary.csv"
done <<< "$CLASSES"

printf '\n=== summary ===\n'
cat "$OUT/summary.csv"
