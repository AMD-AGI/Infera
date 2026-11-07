#!/usr/bin/env bash
# K5 C88 crash repro (3/3 HSA_STATUS_ERROR_OUT_OF_RESOURCES 26-29 min after start, each 1-4 min after a decode rank's KV
# first reached >= 0.98; QA-20261106 section 5): K5 leaves 2.10 GB free per decode GPU after CUDA graph capture, K3 6.03 GB.
# Same point with decode --mem-fraction-static 0.93 (~7.9 GB free), an 1800 s window, VRAM sampled every 20 s; one-node 4+4
# on the batch host. Only in 35231's idle tail (31r and 40 do not fit there); under any other job it goes back to
# holder1/parked for the user to decide.
if [[ "${SLURM_JOB_ID:-}" != 35231 ]]; then
    cp "$0" /apps/liyingli/holder1/parked/27-v2-k5-c88-mf093.sh
    echo "$(date -u +%T) v2 v2-k5-c88-mf093 not under 35231 (job ${SLURM_JOB_ID:-?}): back to holder1/parked" >> /perf_apps/liyingli/vfix-logs/v2/v2.out
    exit 0
fi
MTP_STEPS=5 DECODE_EXTRA_ARGS="--mem-fraction-static 0.93" DURATION=1800 EST_MIN=60 VRAM_SAMPLE=20 \
    exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-k5-c88-mf093 sweep 1node 88
