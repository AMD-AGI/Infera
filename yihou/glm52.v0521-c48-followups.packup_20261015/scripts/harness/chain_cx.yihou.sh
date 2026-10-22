#!/usr/bin/env bash
# Task v0521cap2: correctness round driver (ON the node, local ssh shim): up -> census -> smoke -> audit -> correctness tests -> stop.
set -uo pipefail
ENVF=$1
N=/shared_nfs/yihou_glm52/v0521cap2_20261013
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export PATH=$N/shimbin:$PATH
ts(){ date -u +%FT%TZ; }
cd $H && source "$ENVF"
[[ "$(hostname -s)" == "$YIHOU_NODE" ]] || { echo "$(ts) STOP: run on $YIHOU_NODE"; exit 1; }
D="bash $H/run.v3.yihou.sh"; RD=$R/$YIHOU_ROUND; mkdir -p $RD $N/runs/$YIHOU_ROUND
echo "$(ts) cx start round=$YIHOU_ROUND overrides: $YIHOU_OVERRIDES"
kfd=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '); [[ "$kfd" == 0 ]] || { echo "$(ts) STOP: $kfd KFD procs"; exit 1; }
$D up || { echo "$(ts) STOP at up"; exit 1; }
$D census > $RD/census.txt 2>&1
$D smoke || echo "$(ts) WARN smoke FAIL (continuing to correctness tests)"
$D audit > $RD/audit.txt 2>&1; tail -1 $RD/audit.txt
echo "$(ts) correctness tests"
python3 -u $N/correctness_tests.yihou.py http://$YIHOU_IP:28100 glm5.2-mxfp4 $N/runs/$YIHOU_ROUND > $N/runs/$YIHOU_ROUND/correctness.log 2>&1
echo "$(ts) tests rc=$?"; tail -30 $N/runs/$YIHOU_ROUND/correctness.log
echo "$(ts) stop stack"; $D stop > $RD/stop.log 2>&1; echo "$(ts) cx DONE"
