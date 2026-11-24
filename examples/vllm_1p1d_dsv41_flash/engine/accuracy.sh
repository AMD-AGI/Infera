#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# what: correctness gate through the router — GSM8K (zero-shot, 200 questions) and needle retrieval at
#       16k / 64k / 128k tokens (32 each), with a pass/fail verdict.
# why : a P/D deployment that drops, misplaces or mis-formats KV blocks still returns 200s; it just
#       answers worse, and mostly on LONG prompts (agentic turns are 100k+ tokens). Gate every bring-up
#       before spending an hour on AgentX: a single failed root request aborts a formal AgentX run.
# how : bash cluster/<your-cluster>.sh accuracy     (decode must run SPEC_MODE=block, the default)
#       Thresholds: GSM8K_MIN (0.95), NEEDLE_MIN (32 of 32), RUNAWAY_MAX (2, GSM8K only: a thinking
#       model at temperature 0 occasionally loops on an ambiguous question, aggregated or not).
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$DIR/../common.sh"
require_env PREFILL_NODE; require_env PREFILL_IP; require_env KIT_DIR
URL="http://$PREFILL_IP:$ROUTER_PORT"
OUT="$LOG_DIR/accuracy"; mkdir -p "$OUT" && chmod 777 "$OUT"
DLOG="$LOG_DIR/decode.log"
grep -q "'rejection_sample_method': 'synthetic'" "$DLOG" 2>/dev/null \
  && die "decode runs synthetic (golden) acceptance; run 'golden.sh block' first — synthetic scores ~0"
probe(){ on "$PREFILL_NODE" "docker exec ${GSM8K_FILE:+-e GSM8K_FILE=$GSM8K_FILE} $CTR timeout ${PROBE_TIMEOUT:-3600} python3 \
  $KIT_DIR/engine/probes/accuracy_probe.py $* ${GSM8K_FILE:+--gsm8k-file $GSM8K_FILE}"; }
t0=$(grep -ac "Timeout waiting for P side ready" "$DLOG" 2>/dev/null || true)

log "GSM8K (200, concurrency 32)"
probe gsm8k "$URL" "$SERVED" "$OUT/gsm8k.jsonl" --n "${GSM8K_N:-200}" --concurrency 32 | tee "$OUT/gsm8k.log" | tail -1
for k in ${NEEDLE_K:-16 64 128}; do
  log "needle at ${k}k tokens (32, concurrency 16)"
  probe needle "$URL" "$SERVED" "$OUT/needle${k}k.jsonl" --ctx-tokens $((k * 1024)) --n 32 --concurrency 16 \
    --max-tokens 8192 | tee "$OUT/needle${k}k.log" | tail -1
done
t1=$(grep -ac "Timeout waiting for P side ready" "$DLOG" 2>/dev/null || true)

python3 - "$OUT" "${GSM8K_MIN:-0.95}" "${NEEDLE_MIN:-32}" "${RUNAWAY_MAX:-2}" "$((t1 - t0))" ${NEEDLE_K:-16 64 128} <<'PY'
import re, sys
out, gmin, nmin, rmax, timeouts, ks = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6:]
def stats(name):
    line = open(f"{out}/{name}.log").read().strip().splitlines()[-1]
    return {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", line)}
fails = []
g = stats("gsm8k")
if g["accuracy"] < gmin: fails.append(f"GSM8K {g['accuracy']:.3f} < {gmin}")
if g["runaway"] > rmax: fails.append(f"GSM8K runaway {g['runaway']:.0f} > {rmax}")
for name, s in [("gsm8k", g)] + [(f"needle{k}k", stats(f"needle{k}k")) for k in ks]:
    if s["errors"]: fails.append(f"{name}: {s['errors']:.0f} errors")
    if s["garbled"]: fails.append(f"{name}: {s['garbled']:.0f} garbled")
    if name.startswith("needle") and s["correct"] < nmin: fails.append(f"{name}: {s['correct']:.0f}/{s['n']:.0f} < {nmin}")
if timeouts: fails.append(f"{timeouts} decode pulls timed out waiting for the prefill")
print("ACCURACY", "PASS" if not fails else "FAIL: " + "; ".join(fails))
sys.exit(1 if fails else 0)
PY
