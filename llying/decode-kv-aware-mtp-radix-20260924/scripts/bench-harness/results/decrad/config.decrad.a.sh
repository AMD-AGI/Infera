#!/usr/bin/env bash
# Purpose: Arm A (decode radix cache OFF) of the decode-radix correctness gate.
#   yihou t2f (config.yihou.full.dcar.sh) on 135 prefill + 138 decode, GPUs 2-5,
#   TP4/DP4 + DPA, decode MTP EAGLE 5/1/6, --disable-custom-all-reduce,
#   index_share_for_mtp_iteration=false, prefill HiCache on.
# Deltas vs t2f: the image below, a separate container prefix, and REAL MTP
#   acceptance (simulated acceptance hides garbled decode output).
# Usage: CONFIG=<this file> TOPOLOGY=<this dir>/topology.yihou.tsv ./launch.sh
# Artifacts: none; this file only defines shell variables.
_DECRAD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DECODE_SIMULATE_ACC_LEN=""
: "${IMAGE:=infera-sglang:v0519-yihou-0917-nextnfix-hicache-decrad}"
: "${CONTAINER_PREFIX:=glm52-pd-llying-decrad}"

source "$_DECRAD_DIR/config.yihou.full.dcar.sh"
