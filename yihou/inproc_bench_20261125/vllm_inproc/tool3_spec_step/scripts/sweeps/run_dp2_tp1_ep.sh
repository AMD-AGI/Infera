#!/usr/bin/env bash
# EP twin of run_dp2_tp1.sh (--enable-expert-parallel over DP2 x TP1); see that script.
# sweeps/run_tp8_ep8_dpa.sh). run_dp2_tp1_ep.sh differs only in --ep-size 2 and the iteration name, so EP-vs-no-EP
# is a single-variable comparison. Run both in the SAME container.
# Usage: ISL=173000 OSL=1024 bash run_dp2_tp1.sh C [C ...]   (C = GLOBAL concurrency, divisible by 2; default 448)
set -euo pipefail
TOOL_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for concurrency in "${@:-448}"; do
    ARGS_FILES=presets/dsv41_llying_decode.args bash "$TOOL_ROOT/scripts/run_decode.sh" "dp2_tp1_ep2_c${concurrency}_yihou" \
        --tp-size 1 --dp-size 2 --ep-size 2 --batch-size "$concurrency" \
        --input-len "${ISL:-173000}" --output-len "${OSL:-1024}" --accept-length 3.51 --warmup-steps 10
done
