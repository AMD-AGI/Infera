#!/bin/bash
# a16-19: one watchdog sample of a crusoe profiling run: summary tail, engine log errors, load, GPU use, containers.
# usage: cru_watch.sh JOB TAG
JOB=$1 TAG=$2
D=/shared_nfs/yihou/dsv41_kf_20261120/pd/logs
~/.claude/skills/machine-pool/scripts/spur_remote.sh "grep -a ' $TAG:' $D/summary.log | tail -4; tail -2 $D/$TAG/load.log 2>/dev/null; for f in $D/$TAG/prefill0.log $D/$TAG/decode0.log $D/$TAG/agg0.log $D/$TAG/agg3.log; do [ -f \$f ] && echo \"\$(basename \$f): \$(grep -acE 'Traceback|Error' \$f) err-lines; \$(tail -1 \$f | cut -c1-160)\"; done; grep -c 'stall seen' $D/$TAG/pyspy/watch.log 2>/dev/null; timeout 60 spur exec $JOB bash -c 'cd /tmp && docker ps --format {{.Names}} | grep yihou-dsv41kf; rocm-smi --showuse | grep -c \"GPU use (%): [1-9]\"'" 2>&1 | tail -15
