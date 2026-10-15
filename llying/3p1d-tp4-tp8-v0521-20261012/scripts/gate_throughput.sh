#!/usr/bin/env bash
# Queue gate: fail (which stops the tuning queue) unless a published round reached MIN tok/s/GPU.
# Usage: gate_throughput.sh VARIANT=<v> CONC=<n> MIN=<tok/s/GPU>
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${VARIANT:?}" "${CONC:?}" "${MIN:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
result="$R/results/$VARIANT/c$CONC/agentx_conc$CONC.json"
[[ -s "$result" ]] || { echo "gate: no result $result" >&2; exit 1; }
got="$(python3 -c 'import json,sys; print(round(json.load(open(sys.argv[1]))["request_metrics"]["throughput"]["per_gpu"]["total_tput_tps"]))' "$result")"
echo "$(date -u +%FT%TZ) gate: $VARIANT C$CONC = $got tok/s/GPU (min $MIN)"
(( got >= MIN ))
