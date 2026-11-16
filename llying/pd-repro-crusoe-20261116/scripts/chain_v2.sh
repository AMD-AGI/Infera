#!/usr/bin/env bash
# Gate, then (only when the gate is clean: smoke 6/6, garbled 0, bad_start <= 1, real acceptance >= 3) one scoring run,
# as the AUS items 36 / 37 did. Usage: [MODE=1node] chain_gate_score.sh NAME ARM CONC [SCORE_EXTRA_ENV] [GATE_EXTRA_ENV]
set -uo pipefail
name="$1" arm="$2" conc="$3" sx="${4:-}" gx="${5:-}"
R=/home/liyingli/bench_agentx/pd-repro-20261116
o=$R/logs/opt.out
PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 EXTRA_ENV="$gx" bash $R/scripts/run_item_v2.sh "$name-gate" gate "$arm"
lab="$(grep -aoE "$(basename "${arm%.env}")[^ ]*-[0-9]{4}T[0-9]{4} probe" "$o" | tail -1 | cut -d' ' -f1)"
p="$(grep -a "$lab probe:" "$o" | tail -1)" s="$(grep -a "$lab smoke:" "$o" | tail -1)"
acc="$(grep -a "accept $lab" "$o" | tail -1 | grep -oE 'mean [0-9.]+' | cut -d' ' -f2)"
if [[ -z "$lab" || "$s" != *6/6* || "$p" != *" garbled=0 "* || ! "$p" =~ bad_start=[01]\  ]] || ! awk -v a="${acc:-0}" 'BEGIN {exit !(a >= 3)}'; then
    echo "$(date -u +%T) $name score skipped: gate not clean (${lab:-no gate}; acc ${acc:-?}; $s; $p)" | tee -a "$o"; exit 1
fi
EXTRA_ENV="$sx" bash $R/scripts/run_item_v2.sh "$name" score "$arm" "$conc"
