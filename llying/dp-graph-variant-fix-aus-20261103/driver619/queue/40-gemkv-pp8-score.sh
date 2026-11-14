#!/bin/bash
# DRIVER619-20261113 section 6 step 3: score of the 1P1D PP8 production arm with GEM-backed KV pools, prefill on the batch
# host (n04-29, amdgpu 6.19.14), decode on n04-33 (6.16.13), C208 / 3600 s.
# Reference (arm-C-1p1d-pp8, KFD KV, 11-07, n04-25 -> n04-29 when both ran 6.16.13): C208 56,593 tok/s, interactivity
# p90 70.9 (score-pp8.txt); different pair, PP8 throughput varies ~4 % between pairs.
ALLOW_PILOT_DRIVER=1 ITEM="$0" \
    bash /perf_apps/liyingli/bench_agentx/opt-aus-20261103/scripts/opt_item5.sh gemkv-pp8-score 125 score arm-C-1p1d-pp8-gemkv.env 208
