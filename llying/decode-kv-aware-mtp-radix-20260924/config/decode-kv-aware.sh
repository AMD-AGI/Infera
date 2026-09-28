#!/usr/bin/env bash
# Launch overrides for decode radix cache + decode KV events with MTP.
# Source after the harness config, with patch 03 applied to the harness and an
# image built from docker/Dockerfile. Starts nothing.
: "${IMAGE:?set IMAGE to an image built from docker/Dockerfile}"
export DECODE_KV_AWARE=1
# Decode HiCache stays off: engine.sh rejects decode HiCache together with MTP.
export DECODE_HICACHE=0
# Decode follows the Prefill-selected DP rank, so the rank a prefix was routed to
# on Prefill is also where its Decode radix entries accumulate.
export PD_DP_RANK_AFFINITY=1
true
