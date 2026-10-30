#!/usr/bin/env bash
# 10-30 queue on 138 -> 135 (140 refuses ssh): disk check around every step (stop at >= 95% on / or the data disk);
#   1. E10 (current best) C104 on the new pair -- new reference (node-to-node spread ~3% > E9's predicted gain);
#   2. E9 20-min short run: no divergence assert / traceback, each PP rank's mean eviction time >= 60 ms below E10's
#      on this pair, else skip to 5;
#   3. E9 capped-pool gate (gate_pass2.py PASS), else skip to 5;
#   4. E9 C104 scoring;
#   5. decode profile of E10 under AgentX load (data collection).
# A failed start-up (Mooncake ibv_reg_mr EINVAL flake, 10-29) is retried once.
# Usage: night23_1030.sh LOG_ROOT
set -uo pipefail
L="$1"
P=/home/liyingli/bench_agentx/baseline/Infera/llying
C=$P/prefill-opt-20261027/config
S=$P/prefill-opt-20261027/scripts
R=$P/prefill-opt-20261027/results
KIT=/home/liyingli/bench_agentx/baseline/2p1d-kit/llr35
G=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/2p1d/work/runs/correct-v2/gsm8k/samples_gsm8k_2026-10-20T16-13-32.558577.jsonl
DEC=crsuse2-m2m-135
disk() { bash $S/disk_check.sh $DEC 95 "$1" || { echo "$(date -u +%T) night23: disk limit reached ($1), stop"; exit 0; }; }
perf() {  # LABEL PLAN: one plan line, retried once on a failed start-up
    local label="$1" plan="$2" k
    for k in 1 2; do
        # pd_perf_chain.sh reports a failed start-up on its stdout ("c104 up failed"), not in c104.log.
        out="$(cd "$KIT" && bash $P/prefill-car-20261023/scripts/perf_plan.sh "$KIT" "$L/perf" "$plan" 2>&1)"
        echo "$out"
        grep -q "up failed" <<<"$out" || return 0
        echo "$(date -u +%T) night23: $label start-up failed (attempt $k)"
        mv "$L/perf/$label/c104.log" "$L/perf/$label/c104.log.failed$k"
    done
    return 1
}
analyze() {  # LABEL
    (cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh "$1" 104 > $R/run-$1-c104.txt 2>&1)
    grep -E "^\| C104|^pp[0-3]:" $R/run-$1-c104.txt
    local run; run=$(grep -o 'run=[^ ]*' $L/perf/$1/c104.log | head -1 | cut -d= -f2)
    [[ -n "$run" ]] && python3 $S/beat_vs_evict.py $run/logs/glm52_2p1d-prefill0.log \
        $run/agentx/aiperf_artifacts/server_metrics_export.json > $R/beat-vs-evict-1029/beat-vs-evict-$(basename $run).txt 2>&1
}
profile() {
    disk "before decode profile"
    (cd "$KIT" && SIM=2.99 bash $S/decode_profile_chain.sh "$KIT" $C/site-135-e10.env $L/dprof-135-e10-1030 104 1500 600 40 300)
    disk "after decode profile"
    echo "$(date -u +%T) night23 queue done"
    exit 0
}
disk "start"
E10=llr37-135-e10
echo "$(date -u +%T) night23: E10 C104 on 138 -> 135 ($E10)"
perf $E10 $C/plan-135-e10.txt || { echo "$(date -u +%T) night23: E10 never started, stop"; exit 0; }
analyze $E10
disk "after E10"
LS=llr37-135-e9-short
echo "$(date -u +%T) night23: E9 short run $LS"
perf $LS $C/plan-135-e9-short.txt || { echo "$(date -u +%T) night23: E9 short never started"; profile; }
(cd $P/prefill-opt-20261027 && bash scripts/analyze_run.sh $LS 104 > $R/run-$LS-c104.txt 2>&1)
grep -E "^\| C104|^pp[0-3]:" $R/run-$LS-c104.txt
RUN=$(grep -o 'run=[^ ]*' $L/perf/$LS/c104.log | head -1 | cut -d= -f2)
if grep -aqE "selections diverged|victims diverged|Traceback" $RUN/logs/glm52_2p1d-prefill0.log; then
    echo "$(date -u +%T) night23: assert / traceback in the prefill log, E9 stopped"
    grep -anE "diverged|Traceback" $RUN/logs/glm52_2p1d-prefill0.log | head -5
    profile
fi
ok=$(python3 - $R/run-$E10-c104.txt $R/run-$LS-c104.txt <<'PY'
import re, sys
def avg(f):
    return {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"^pp(\d): .*? avg (\d+)ms", open(f).read(), re.M)}
ref, got = avg(sys.argv[1]), avg(sys.argv[2])
good = len(ref) == 4 and len(got) == 4 and all(got[r] <= ref[r] - 60 for r in ref)
print(("PASS " if good else "FAIL ") + f"E10 {ref} E9 {got}")
PY
)
echo "$(date -u +%T) night23: mechanism check: $ok"
[[ "$ok" == PASS* ]] || { echo "$(date -u +%T) night23: E9 mechanism not as predicted, E9 stopped"; profile; }
disk "after E9 short"
g="$L/gate/full-site-135-e9-gate-$(date -u +%m%dT%H%M).out"
echo "$(date -u +%T) night23: E9 capped gate ($g)"
(cd "$KIT" && PROBE_N=200 PROBE_PROMPTS=$G LONGCTX_SPECS="16000:128 16000:128" bash $S/quick_correct.sh "$KIT" "$L/gate" \
    $C/site-135-e9-gate.env >"$g" 2>&1)
grep -E " smoke: | probe: | longctx " "$g" | cut -c1-300
verdict=$(python3 $S/gate_pass2.py "$g" site-135-e9-gate --passes 2)
echo "$(date -u +%T) night23: gate verdict: $verdict"
[[ "$verdict" == PASS ]] || { echo "$(date -u +%T) night23: E9 gate FAILED, no scoring"; profile; }
disk "after E9 gate"
E9=llr37-135-e9
echo "$(date -u +%T) night23: E9 C104 ($E9)"
perf $E9 $C/plan-135-e9.txt || { echo "$(date -u +%T) night23: E9 scoring never started"; profile; }
analyze $E9
disk "after E9"
profile
