#!/usr/bin/env bash
# Task v0521cap TB: after pc40c A2, re-run the long-prompt burst with prompts big enough to cross the runshare cap
# (64 x 200k words ≈ 230k tokens; ~9-11 in flight per DP -> ~0.75 of the pool without the cap). Same stack.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; CAP=/shared_nfs/yihou_glm52/v0521cap_20261012
ts(){ date -u +%FT%TZ; }
until grep -qE "pc40c chain DONE|STOP" $R/pc40c.chain.log; do sleep 60; done
grep -q "pc40c chain DONE" $R/pc40c.chain.log || { echo "$(ts) chain stopped, no smoke2"; exit 1; }
DL=$R/pc40c/server-logs/decode-0.a1.log
b0=$(strings $DL | grep -ac 'YIHOU-RUNSHARE block')
setsid nohup python3 $CAP/tools/metrics_sampler.yihou.py $CAP/runs/pc40c/metrics.burst2.tsv 1500 2 \
  decode=http://10.245.147.130:29357 prefill=http://10.245.147.130:29101 > /dev/null 2>&1 < /dev/null &
echo "$(ts) smoke2 start (blocks before: $b0)"
python3 $CAP/tools/longprompt_burst.yihou.py http://10.245.147.130:28100 64 200000 16 > $R/pc40c/burst2.txt 2>&1; brc=$?
tail -1 $R/pc40c/burst2.txt
echo "$(ts) smoke2 rc=$brc blocks after: $(strings $DL | grep -ac 'YIHOU-RUNSHARE block')"
strings $DL | grep -a 'YIHOU-RUNSHARE block' | tail -5
echo "$(ts) max pre-allocated usage: $(strings $DL | grep -ao 'pre-allocated usage: [0-9.]*' | awk '{print $3}' | sort -g | tail -1)"
echo "$(ts) faults: $(strings $DL | grep -ac -E 'Memory access fault|Traceback')"
