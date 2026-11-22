#!/usr/bin/env bash
# On n02-25: wait for the c384 run to end, build the GEM-IPC image (fix B), then the 1P1D c768 point with prefill on
# GEM at 0.9 + standalone UMBP (smoke + accuracy gate -> golden 3.51 -> formal AgentX). Runs on the node.
set -uo pipefail
source "$(dirname "$0")/env.sh"
until grep -aq "pd1p1d-c384-1120T0606: end" "$W/logs/summary.log"; do sleep 30; done
cd /perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
bash examples/vllm_1p1d_dsv41_flash/build_image.sh > "$W/logs/build-image-B-n0225.log" 2>&1 || {
    echo "$(ts) chain-c768: image build FAILED" | tee -a "$W/logs/summary.log"; exit 1; }
IMG=$(grep -ao "infera-vllm-dsv41:[0-9a-f]*-nightly-[0-9a-f]*" "$W/logs/build-image-B-n0225.log" | tail -1)
echo "$(ts) chain-c768: built $IMG on $(hostname -s)" | tee -a "$W/logs/summary.log"
exec env INFERA_IMAGE="$IMG" CONC=768 GATE=1 UMBP_MODE=standalone UMBP_GEM_IPC=1 RUN=pd1p1d-c768-B-$(date -u +%m%dT%H%M) \
    NOTE="B: P GEM 0.9 + standalone UMBP over GEM IPC, D GEM 0.9" bash "$W/scripts/run_1p1d_v2.sh"
