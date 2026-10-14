#!/usr/bin/env bash
# Reproduce the best 3P(TP4)1D(TP8) configuration from scratch (REPRODUCE.zh-CN.md).
# Usage (detached on the control node, crsuse2-m2m-137):
#   ROUND=best1 POINTS="176" setsid nohup bash scripts/reproduce_best.sh > runtime/reproduce.best1.log 2>&1 < /dev/null &
# Steps: install config/best.sh as config/round.<ROUND>.sh -> run_3p1d.sh all with no AgentX points
# (preflight, launch at REAL acceptance, census, known-answer gate, switch decode to simulated
# acceptance 3.61, smoke) -> KV-transfer smoke (192 x 8k; on failure restart the prefills once and
# retry: a decode-only restart can leave stale prefill->decode transfer state) -> background sampler
# -> 30-min unscored warm-up -> one AgentX round per point with the best router switches.
set -euo pipefail
: "${ROUND:?set ROUND (new name, e.g. best1)}"
POINTS="${POINTS:-176}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
BEST="INFERA_PD_PREFILL_GUARD_RELEASE=completion INFERA_SESSION_AFFINITY=prefill"
BEST+=" INFERA_SESSION_AFFINITY_TTL_SECS=3600 INFERA_R2_DECODE_DEMAND=on"
say() { echo "$(date -u +%FT%TZ) reproduce_best: $*"; }
export ROUND

round_file="$R/config/round.$ROUND.sh"
[[ -e "$round_file" ]] || cp "$R/config/best.sh" "$round_file"
say "round overlay: $round_file"

POINTS=" " bash "$HERE/run_3p1d.sh" all
set -a; source "$R/config/config.3p1d.sh"; set +a
control_ip="$(python3 "$BENCH_DIR/tools/topology.py" node-ip "$TOPOLOGY" "$CONTROL_NODE")"
router_url="http://$control_ip:$ROUTER_PORT"
decode_log="$RUNS/launch/server-logs/decode-0.after-realacc.log"

if ! python3 "$HERE/transfer_smoke.py" "$router_url" "$SERVED_MODEL" "$decode_log" 192 8000; then
    say "transfer smoke failed; restarting the prefills once"
    bash "$HERE/restart_prefills.sh" "ROUND=$ROUND" "OVERLAY=$round_file" TAG=smokefix
    python3 "$HERE/transfer_smoke.py" "$router_url" "$SERVED_MODEL" "$decode_log" 192 8000 ||
        { say "transfer smoke failed again; stopping"; exit 1; }
fi
say "transfer smoke ok"

mkdir -p "$RUNS/tune"
setsid nohup python3 "$HERE/sampler.py" "$RUNS/tune/samples.jsonl" 60 > "$RUNS/tune/sampler.log" 2>&1 < /dev/null &
say "sampler started (pid $!)"

bash "$HERE/warmup_round.sh" "VARIANT=warm-$ROUND" CONC=144 MINUTES=30 "ROUND=$ROUND" || say "warm-up round returned non-zero (unscored)"
for c in $POINTS; do
    bash "$HERE/tune_round.sh" "VARIANT=best-$ROUND" "CONC=$c" "ROUND=$ROUND" "ROUTER_ENV=$BEST" ||
        { say "round C$c failed; stopping"; exit 1; }
done
say "done: results/best-$ROUND/"
