#!/usr/bin/env bash
# One scoring point for the one-node P4D4: a fresh deployment with simulated MTP acceptance (InferenceX agentic rule,
# K3 golden 2.99), NIC and engine counters before / after, AgentX C$CONC for $DURATION s, summary line in
# $R/logs/summary.out.   score_p4d4.sh RUN   (env as pd1_up.sh / agentx.sh; SIM_ACCEPT_LEN defaults to 2.99)
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
run=${1:?run id}
export SIM_ACCEPT_LEN=${SIM_ACCEPT_LEN:-2.99} CONC=${CONC:-104} DURATION=${DURATION:-3600}
RUN_DIR=$R/runs/$run
bash "$R/scripts/pd1_up.sh" "$run" || { echo "$(date -u +%m%d-%T) score $run: deployment failed" | tee -a "$R/logs/summary.out"; exit 1; }
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
murls="http://$IP:27201 http://$IP:27301"
# SMOKE=1: the router smoke first (simulated acceptance makes the text wrong, so only failures to answer count).
if [[ "${SMOKE:-0}" == 1 ]]; then
    PREFILL_NUM_WORKERS=1 timeout 900 python3 "$R/scripts/probes/smoke.py" "$(cat "$RUN_DIR/router_url")" glm5.2-mxfp4 \
        "$RUN_DIR/smoke" > "$RUN_DIR/smoke.out" 2>&1
    if ! grep -q '"name": "long_prefix_repeat"' "$RUN_DIR/smoke.out"; then
        echo "$(date -u +%m%d-%T) score $run: smoke did not complete ($(tail -1 "$RUN_DIR/smoke.out"))" | tee -a "$R/logs/summary.out"
        exit 1
    fi
fi
# shellcheck disable=SC2086
python3 "$R/scripts/metrics_snap.py" snap "$RUN_DIR/m0.json" $murls
bash "$R/scripts/nic_snap.sh" > "$RUN_DIR/nic0.txt"; t0=$(date +%s)
bash "$R/scripts/agentx.sh" "$run"; rc=$?
bash "$R/scripts/nic_snap.sh" > "$RUN_DIR/nic1.txt"; t1=$(date +%s)
# shellcheck disable=SC2086
python3 "$R/scripts/metrics_snap.py" snap "$RUN_DIR/m1.json" $murls
python3 "$R/scripts/nic_diff.py" "$RUN_DIR/nic0.txt" "$RUN_DIR/nic1.txt" $((t1 - t0)) > "$RUN_DIR/nic-delta.txt"
python3 "$R/scripts/metrics_snap.py" diff "$RUN_DIR/m0.json" "$RUN_DIR/m1.json" > "$RUN_DIR/metrics-delta.json"
for c in vpd-prefill vpd-decode vpd-router; do docker logs "$c" > "$RUN_DIR/$c.log" 2>&1; done
out=$RUN_DIR/agentx-c$CONC
tmp=$(mktemp -d)
ln -s "$out"/agentx_conc*.json "$tmp/"; ln -s "$out/aiperf_artifacts/profile_export_aiperf.json" "$tmp/"
row=$(python3 "$R/scripts/summarize_points.py" "$tmp" 2>&1 | tail -1)
rm -rf "$tmp"
echo "$(date -u +%m%d-%T) score $run C$CONC ${DURATION}s rc=$rc: $row; metrics $(cat "$RUN_DIR/metrics-delta.json")" | tee -a "$R/logs/summary.out"
exit $rc
