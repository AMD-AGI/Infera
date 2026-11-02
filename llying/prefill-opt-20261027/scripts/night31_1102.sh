#!/usr/bin/env bash
# 11-02 after night30: is the intermittent "bad start" tied to the decode custom all-reduce? Capped-pool gate of E10 (control: bad start
# 3-8 of 200, probe 0.965-0.985) with DECODE_CUSTOM_AR=0, same probe set and two 16k x 128 passes as night20.
# Usage: night31_1102.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
while pgrep -f "night30_110[2]" >/dev/null; do sleep 30; done
bash $S/disk_check.sh crsuse2-m2m-140 95 "before nocar gate" || exit 0
mkdir -p "$L/gate"
g="$L/gate/full-site-140-gate-nocar-$(date -u +%m%dT%H%M).out"
PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" $C/site-140-gate-nocar.env >"$g" 2>&1
grep -aE "smoke|probe:|longctx" "$g" | cut -c1-260
echo "$(date -u +%T) night31: verdict $(python3 $S/gate_pass2.py "$g" site-140-gate-nocar --passes 2 2>&1 | tail -2)"
echo "$(date -u +%T) night31 done"
