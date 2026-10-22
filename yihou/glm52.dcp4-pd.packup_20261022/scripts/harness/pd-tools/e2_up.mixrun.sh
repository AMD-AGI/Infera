#!/usr/bin/env bash
# e2_up.mixrun.sh ROUND SIM(empty|3.61) [PD_EXTRA...]: E2 (DP2xDCP2 decode) bring-up on v3-070 via pdrun's run.v2 harness.
# Login host, detached: setsid nohup bash e2_up.mixrun.sh pdE2-r1 "" "DECODE_EXTRA_ARGS_ADD=--disable-cuda-graph" > $PD/runs/e2up.<round>.log 2>&1 < /dev/null &
set -uo pipefail
PD=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd; H=$PD/harness; JOB=230266
round=$1; sim=$2; shift 2; export PD_EXTRA="$*" PD_JOB=$JOB PD_CONC=${PD_CONC:-40}
ts(){ date -u +%FT%TZ; }
bash "$PD/tools/mk_env_e2.mixrun.sh" "$round" "$sim" || exit 1
cd "$H"; source "env.$round-v3-070.yihou.sh"
bash run.v2.yihou.sh dryrun > "dryrun.$round-v3-070.yihou.txt" 2>&1 || { echo "$(ts) dryrun failed"; exit 1; }
echo "$(ts) up round=$round sim='$sim' PD_EXTRA='$PD_EXTRA'"
bash run.v2.yihou.sh up || { echo "$(ts) UP FAILED (see $PD/runs/$round/status.tsv)"; exit 1; }
for m in census audit smoke; do echo "$(ts) $m"; bash run.v2.yihou.sh $m > "$PD/runs/$round/$m.txt" 2>&1; echo "$m rc=$?"; done
bash "$H/pdcheck.yihou.sh" "$round" > "$PD/runs/$round/pdcheck.txt" 2>&1
echo "$(ts) E2 UP DONE round=$round (stack left up)"
