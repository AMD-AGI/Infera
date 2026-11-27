#!/usr/bin/env bash
# One P4D4 point, end to end: [eval] fresh deployment with real MTP acceptance -> InferenceX eval at the benchmark
# concurrency; [score] fresh deployment with the golden synthetic acceptance -> NIC / engine counters -> InferenceX AgentX.
#   run.sh VARIANT OUT_DIR
# VARIANT: kit/variants/<name>.env (image + per-variant engine args). Env: CONC (152), STEPS ("eval score"), FAST (0).
set -uo pipefail
K=$(cd "$(dirname "$0")" && pwd)
variant=${1:?variant} out=${2:?out dir}
set -a   # the variant's and env.sh's settings reach engine/up.sh, agentx.sh and eval.sh
. "$K/variants/$variant.env"
. "$K/env.sh"
set +a
CONC=${CONC:-152}
mkdir -p "$out"
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
murls="http://$IP:27201 http://$IP:27301"
rc=0
for step in ${STEPS:-eval score}; do
    d=$out/$step
    mkdir -p "$d"
    bash "$K/down.sh"
    if [[ "$step" == eval ]]; then
        SIM_AL="" bash "$K/engine/up.sh" "$d" > "$d/up.log" 2>&1 || { echo "$(ts) $variant eval: deployment failed ($(tail -1 "$d/up.log"))"; rc=1; continue; }
        IMAGE=$IMAGE MTP_N=$MTP_N bash "$K/eval.sh" "$d/eval" "$(cat "$d/router_url")" "$CONC" "$murls" > "$d/eval.log" 2>&1 || rc=1
        echo "$(ts) $variant eval C$CONC: $(cat "$d/eval/eval-rc.txt" 2>/dev/null) $(grep -a -E 'GSM8K rc|AL gate rc' "$d/eval.log" | tr '\n' ' ')"
    else
        SIM_AL=$(golden_al "$MTP_N") bash "$K/engine/up.sh" "$d" > "$d/up.log" 2>&1 || { echo "$(ts) $variant score: deployment failed ($(tail -1 "$d/up.log"))"; rc=1; continue; }
        for n in /sys/class/infiniband/*; do echo "$(basename "$n") $(cat "$n/ports/1/counters/port_xmit_data" "$n/ports/1/counters/port_rcv_data" 2>/dev/null | paste -sd' ')"; done > "$d/nic0.txt"
        for m in $murls; do curl -s -m 30 "$m/metrics"; done > "$d/metrics0.txt"
        FAST=${FAST:-0} IMAGE=$IMAGE MTP_N=$MTP_N bash "$K/agentx.sh" "$d/agentx" "$(cat "$d/router_url")" "$CONC" "$murls" > "$d/agentx.log" 2>&1 || rc=1
        for n in /sys/class/infiniband/*; do echo "$(basename "$n") $(cat "$n/ports/1/counters/port_xmit_data" "$n/ports/1/counters/port_rcv_data" 2>/dev/null | paste -sd' ')"; done > "$d/nic1.txt"
        for m in $murls; do curl -s -m 30 "$m/metrics"; done > "$d/metrics1.txt"
        echo "$(ts) $variant score C$CONC $( [[ "${FAST:-0}" == 1 ]] && echo FAST): $(tail -2 "$d/agentx/score.txt" 2>/dev/null | tr '\n' ' ')"
    fi
    for c in vpd-prefill vpd-decode vpd-router; do docker logs "$c" 2>&1 | gzip > "$d/logs/$c.full.log.gz"; done
done
bash "$K/down.sh"
exit $rc
