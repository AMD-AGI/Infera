#!/usr/bin/env bash
# Task v0521cap2 (LOGIN HOST, node 136): replaces the tail of ofs_seq136l. Its KFD gate counts the crusoe exporter's
# gpuagent PID (always present on 136), so it would never hand over. Gate here = every GPU's VRAM < 1 GiB.
# wait ofg4b chain DONE/STOP -> run.v3 stop ofg4b -> VRAM gate -> chain_cap2l ofg2b 48 48 56 -> stop.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
vram_max(){ ssh -o ConnectTimeout=15 -o BatchMode=yes crsuse2-m2m-136 "rocm-smi --showmeminfo vram | grep 'Total Used' | awk '{print \$NF}' | sort -n | tail -1"; }
echo "$(ts) waiting for ofg4b chain"
until grep -qE "chain DONE|STOP" $R/ofg4b.chain.log 2>/dev/null; do sleep 60; done
tail -1 $R/ofg4b.chain.log
( cd $H && source $H/env.ofg4b-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) > $R/ofg4b/stop.seq.log 2>&1
echo "$(ts) ofg4b stopped"
for i in $(seq 1 60); do v=$(vram_max); [[ -n "$v" && "$v" -lt 1073741824 ]] && break; sleep 30; done
[[ -n "$v" && "$v" -lt 1073741824 ]] || { echo "$(ts) STOP: max VRAM $v B on 136 before ofg2b"; exit 1; }
echo "$(ts) -> ofg2b 48 48 56 (max VRAM $v B)"
bash $H/chain_cap2l.yihou.sh $H/env.ofg2b-136.yihou.sh 48 48 56 > $R/ofg2b.chain.log 2>&1 < /dev/null
tail -1 $R/ofg2b.chain.log
( cd $H && source $H/env.ofg2b-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) > $R/ofg2b/stop.seq.log 2>&1
echo "$(ts) ofg2b stopped; seq DONE"
