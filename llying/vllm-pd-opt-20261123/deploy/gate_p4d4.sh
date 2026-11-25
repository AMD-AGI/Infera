#!/usr/bin/env bash
# Correctness gate of a P4D4 deployment with real MTP acceptance (the chain_v2 checks in gate.sh: smoke, GSM8K 200,
# 16k long context x 64, acceptance from the engine counters), run by p4d4_run.sh with P4D4_STEP=gate_p4d4.sh so the
# deployment env is the scoring one minus the simulated acceptance.   gate_p4d4.sh RUN
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
run=${1:?run id}
unset SIM_ACCEPT_LEN
bash "$R/scripts/pd1_up.sh" "$run" || { echo "$(date -u +%m%d-%T) gate $run: deployment failed" | tee -a "$R/logs/summary.out"; exit 1; }
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
PREFILL_NUM_WORKERS=1 bash "$R/scripts/gate.sh" "$run" "$(cat "$R/runs/$run/router_url")" "http://$IP:27201 http://$IP:27301"
rc=$?
for c in vpd-prefill vpd-decode vpd-router; do docker logs "$c" > "$R/runs/$run/$c.log" 2>&1; done
exit $rc
