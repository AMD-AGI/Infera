#!/usr/bin/env bash
# The TP4 / EP1 / DP-attention decode sweep (port of the GLM-5.2 sweeps/run_tp8_ep1_dpa.sh, re-pinned to the
# #5034 shape TP4 DPA4 and this port's reference ISL 8192 / OSL 1024 / accept 3.51). The EP4 and EP1 scripts differ
# only in --ep-size and the iteration name, so EP4-vs-EP1 is a single-variable comparison. Run both in the SAME
# container. Default: C=64. Usage: bash run_tp4_noep_dpa.sh C [C ...]   (C divisible by 4)
set -euo pipefail
TOOL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for concurrency in "${@:-64}"; do
    ARGS_FILES=presets/dsv41_ix5034.args bash "$TOOL_ROOT/scripts/run_decode.sh" "tp4_noep_dpa_on_c${concurrency}_yihou" \
        --tp-size 4 --ep-size 1 --enable-dp-attention --batch-size "$concurrency" \
        --max-running-requests "$concurrency" \
        --input-len 8192 --output-len 1024 --accept-length 3.51 \
        --warmup-steps 10 --mem-fraction-static 0.85
done
