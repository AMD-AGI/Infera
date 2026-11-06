#!/usr/bin/env bash
# K5 C88 crash repro (3/3 HSA_STATUS_ERROR_OUT_OF_RESOURCES 26-29 min after start, each 1-4 min after a decode rank's KV
# first reached >= 0.98; QA-20261106 section 5): K5 leaves 2.10 GB free per decode GPU after CUDA graph capture, K3 6.03 GB.
# Same point with decode --mem-fraction-static 0.93 (~7.9 GB free), an 1800 s window, VRAM sampled every 20 s.
MTP_STEPS=5 DECODE_EXTRA_ARGS="--mem-fraction-static 0.93" DURATION=1800 EST_MIN=60 VRAM_SAMPLE=20 ITEM="$0" \
    exec bash /perf_apps/liyingli/bench_agentx/v2-aus/v2_item.sh v2-k5-c88-mf093 sweep 1node 88
