#!/usr/bin/env bash
# Purpose: snapshot real MTP acceptance around an AgentX run. T1's headline.
# Usage: ./collect_accept.yihou.sh <out_dir> [before|after]
#   Run once with "before" immediately pre-AgentX and once with "after"
#   immediately post-AgentX, into the same out_dir.
# Artifacts: <out_dir>/metrics-<phase>.txt, accept-<phase>.txt, and on "after"
#   a verdict.txt.
#
# THE TRAP THIS SCRIPT EXISTS TO AVOID. A decode rank that has served no tokens
# reports spec_accept_length 0.0, which is indistinguishable from "accepts
# nothing". The "before" snapshot is expected to be uninformative and is kept
# only to prove the trap was considered; the measurement is "after", and only
# ranks whose gauge is non-zero are counted.
#
# accept_length and accept_rate are DIFFERENT metrics. The bar is on length.
set -euo pipefail

OUT="${1:?usage: collect_accept.yihou.sh <out_dir> [before|after]}"
PHASE="${2:-after}"
mkdir -p "$OUT"

DECODE_HOST="${DECODE_HOST:-10.245.157.237}"     # crsuse2-m2m-138
DECODE_PORT="${DECODE_PORT:-29002}"              # all decode DP ranks share it

curl -sS --max-time 30 "http://$DECODE_HOST:$DECODE_PORT/metrics" \
    >"$OUT/metrics-$PHASE.txt" || { echo "scrape failed" >&2; exit 1; }
grep -E 'spec_accept_(length|rate)' "$OUT/metrics-$PHASE.txt" \
    | tee "$OUT/accept-$PHASE.txt" || true

# The same reading R01 took, from the decode server log: metrics_reporter.py
# emits "accept len: X, accept rate: Y" every decode_log_interval steps under
# load. An independent second source for the same quantity.
if [[ -n "${DECODE_LOG:-}" && -f "$DECODE_LOG" ]]; then
    grep -oE 'accept len: [0-9.]+, accept rate: [0-9.]+' "$DECODE_LOG" \
        | tail -40 >"$OUT/accept-from-log-$PHASE.txt" || true
    echo "--- decode log, last lines ---"
    tail -5 "$OUT/accept-from-log-$PHASE.txt" 2>/dev/null || echo "  (none yet)"
fi

[[ "$PHASE" == after ]] || exit 0

python3 - "$OUT" <<'PY' | tee "$OUT/verdict.txt"
import re, sys, os
out = sys.argv[1]
rows = []
for line in open(os.path.join(out, "metrics-after.txt")):
    m = re.match(r'sglang:spec_accept_length\{([^}]*)\}\s+([\d.eE+-]+)', line)
    if m:
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        rows.append((labels.get("dp_rank", labels.get("rank", "?")), float(m.group(2))))
print("spec_accept_length by rank:", rows)
active = [v for _, v in rows if v > 0]
if not active:
    print("no rank reported a non-zero gauge -- INCONCLUSIVE, not a measurement of zero")
    sys.exit(0)
print(f"active ranks: {len(active)}/{len(rows)}   min={min(active):.4f}  mean={sum(active)/len(active):.4f}")
print("PASS (bar 2.0)" if min(active) >= 2.0 else "FAIL (bar 2.0)")
PY
