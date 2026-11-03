#!/usr/bin/env bash
# 11-02 (1) control for night30 variant B: plain E10 (packet-capture flag unset), uncapped pool, same quick check as variant B
#   (variant B: probe 0.930 / garbled 6 / hit-cap 13 / bad start 0; capped-pool E10 controls were 0.985 / 0 / 3);
# (2) two more capped-pool gates with decode custom all-reduce off (night31: bad start 0 of 200; with CAR on, 6 of 8 earlier gates
#     had 3-8 and 2 had 0, so one zero is weak evidence: three zeros in a row would be p ~ 0.25^3 = 1.6%).
# Usage: night33_1102.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
mkdir -p "$L/e11-check2" "$L/gate"
bash $S/disk_check.sh crsuse2-m2m-140 95 "before E10 control" || exit 0
PROBE_N=200 LONGCTX_SPECS="16000:32" bash $S/quick_correct.sh "$KIT" "$L/e11-check2" $C/site-llr35-v3-evict512k-bq16.env
echo "$(date -u +%T) night33: E10 control done"
for k in 2 3; do
    bash $S/disk_check.sh crsuse2-m2m-140 95 "before nocar gate $k" || exit 0
    g="$L/gate/full-site-140-gate-nocar-r$k-$(date -u +%m%dT%H%M).out"
    PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-140-gate-nocar.env >"$g" 2>&1
    grep -aE "probe:|longctx" "$g" | cut -c1-250
    echo "$(date -u +%T) night33: nocar gate $k verdict $(python3 $S/gate_pass2.py "$g" site-140-gate-nocar --passes 2 2>&1 | tail -1)"
done
echo "$(date -u +%T) night33 done"
