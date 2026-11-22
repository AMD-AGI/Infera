#!/bin/bash
# a16-19 -> crusoe: start pd_profile.sh detached on the node of spur job JOB.
# usage: cru_launch_profile.sh JOB TAG STOP_BY_ISO [KEY=VAL ...]   (KEY=VAL: NP ND CONC GOLDEN D_MODE CALIB WINDOW_AT_S ...)
set -euo pipefail
JOB=$1 TAG=$2 STOP=$3; shift 3
D=/shared_nfs/yihou/dsv41_kf_20261120
SR=~/.claude/skills/machine-pool/scripts/spur_remote.sh
$SR "spur exec $JOB bash -c 'cd /tmp && mkdir -p $D/launch && setsid nohup env TAG=$TAG STOP_BY=$STOP $* bash $D/pd/scripts/pd_profile.sh > $D/launch/$TAG.log 2>&1 < /dev/null & sleep 5; head -3 $D/launch/$TAG.log'"
