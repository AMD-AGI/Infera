#!/bin/bash
# arm 3: decode MTP OFF + decode radix cache ON. C=8 only, to compare against
# arm 1's C=8 point (prefill DPA=0, router 20.0/2.0 -- both restored).
set -uo pipefail
export CONCS=8
export OUT=/shared_nfs/yihou_agentx_caseA/bench/results_nomtp
mkdir -p "$OUT"
exec bash /shared_nfs/yihou_agentx_caseA/run_caseA.sh
