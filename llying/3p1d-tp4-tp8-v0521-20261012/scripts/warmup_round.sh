#!/usr/bin/env bash
# Unscored warm-up after a decode restart (the first AgentX run on a fresh decode has slower
# decode steps, REVIEW §8): one tune_round.sh with the best router switches and a shorter
# window. Published as results/<VARIANT>/c<CONC>/ but not used for comparisons; its
# engine_pace.txt shows whether decode steps have settled.
# Usage: warmup_round.sh VARIANT=<warm-...> [CONC=144] [MINUTES=30] [ROUND=r2]
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${VARIANT:?}"
[[ "$VARIANT" == warm-* ]] || { echo "VARIANT must start with warm-" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
best="INFERA_PD_PREFILL_GUARD_RELEASE=completion INFERA_SESSION_AFFINITY=prefill"
best+=" INFERA_SESSION_AFFINITY_TTL_SECS=3600 INFERA_R2_DECODE_DEMAND=on"
seconds=$(( ${MINUTES:-30} * 60 ))
exec bash "$HERE/tune_round.sh" "VARIANT=$VARIANT" "CONC=${CONC:-144}" "ROUND=${ROUND:-r2}" \
    "ROUTER_ENV=$best" "DURATION=$seconds" "AGENTX_DURATION=$seconds"
