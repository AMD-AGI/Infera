#!/usr/bin/env bash
set -euo pipefail
export CONFIG=/perf_apps/liyingli/bench_agentx/router-capacity-20260928/config/b2-real.sh
set -a; source "$CONFIG"; set +a
python3 "$TRACE_RUNTIME/scripts/transition_two_node.py"
python3 "$TRACE_RUNTIME/scripts/radix_gate.py" --router "http://$PREFILL_IP:28000" --prefill "http://$PREFILL_IP:29001" --decode "http://$DECODE_IP:29002" --model "$SERVED_MODEL" --out "$RUN/radix-gate.json" --records 400 --trials 2
python3 "$TRACE_RUNTIME/scripts/check_radix_gate.py"
printf 'RADIX_GATE_PASSED\n' > "$RUN/STATUS"
