#!/usr/bin/env bash
set -uo pipefail
source /mnt/vast/c_huggingface/pd_env.sh
export BACKEND=mooncake
export CTR=dsv4_pd_sgl
echo "[mooncake] KIT_DIR=$KIT_DIR TOPO=$TOPO"
echo "[mooncake] prefill=$PREFILL_NODE($PREFILL_IP) decode=$DECODE_NODE($DECODE_IP)"
echo "[mooncake] model=$INFERA_MODEL image=$INFERA_IMAGE"
bash "$KIT_DIR/engine/pd_mooncake/sglang/up.sh"
echo "[mooncake] up.sh returned rc=$?"
