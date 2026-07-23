#!/usr/bin/env bash
set -uo pipefail
source /mnt/vast/c_huggingface/pd_env.sh
export INFERA_IMAGE=infera/engine-sglang:pd-mcfix
export BACKEND=mori CTR=dsv4_pd_sgl_mori
echo "[mori] image=$INFERA_IMAGE prefill=$PREFILL_NODE decode=$DECODE_NODE"
bash "$KIT_DIR/engine/pd_mori/sglang/up.sh"
echo "[mori] up.sh rc=$?"
