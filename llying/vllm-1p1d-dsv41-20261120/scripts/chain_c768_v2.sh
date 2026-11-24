#!/usr/bin/env bash
# On n02-25: build the image at the branch head (adds serial vLLM Mooncake dispatch, Infera#246), then the 1P1D c768
# point: prefill GEM 0.9 + standalone UMBP over GEM IPC, decode GEM 0.9, router dispatching decode legs after the
# prefill reply (smoke + accuracy gate -> golden 3.51 -> formal AgentX). Runs on the node.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd /perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
bash examples/vllm_1p1d_dsv41_flash/build_image.sh > "$W/logs/build-image-serial-n0225.log" 2>&1 || {
    echo "$(ts) chain-c768-v2: image build FAILED" | tee -a "$W/logs/summary.log"; exit 1; }
IMG=$(grep -ao "infera-vllm-dsv41:[0-9a-f]*-nightly-[0-9a-f]*" "$W/logs/build-image-serial-n0225.log" | tail -1)
echo "$(ts) chain-c768-v2: built $IMG on $(hostname -s)" | tee -a "$W/logs/summary.log"
exec env INFERA_IMAGE="$IMG" CONC=768 GATE=1 UMBP_MODE=standalone UMBP_GEM_IPC=1 MOONCAKE_SERIAL=1 \
    RUN=pd1p1d-c768-B-serial-$(date -u +%m%dT%H%M) \
    NOTE="B + serial Mooncake dispatch: P GEM 0.9 + standalone UMBP over GEM IPC, D GEM 0.9" bash "$W/scripts/run_1p1d_v2.sh"
