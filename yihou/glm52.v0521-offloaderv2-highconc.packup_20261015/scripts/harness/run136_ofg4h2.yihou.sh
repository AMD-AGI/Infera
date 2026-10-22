#!/usr/bin/env bash
# Task v0521cap2 (LOGIN HOST spur005, node 136): wait until every 136 GPU has < 1 GiB VRAM, then chain_cap2l ofg4h2 56 56 64,
# then run.v3 stop with a 600 s cap (its sync_back copy hung for 50 min once).
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
vram_max(){ ssh -o ConnectTimeout=15 -o BatchMode=yes crsuse2-m2m-136 "rocm-smi --showmeminfo vram | grep 'Total Used' | awk '{print \$NF}' | sort -n | tail -1"; }
for i in $(seq 1 90); do v=$(vram_max); [[ -n "$v" && "$v" -lt 1073741824 ]] && break; echo "$(ts) max VRAM $v B, waiting"; sleep 60; done
[[ -n "$v" && "$v" -lt 1073741824 ]] || { echo "$(ts) STOP: max VRAM $v B on 136 after 90 min"; exit 1; }
echo "$(ts) -> ofg4h2 56 56 64 (max VRAM $v B)"
bash $H/chain_cap2l.yihou.sh $H/env.ofg4h2-136.yihou.sh 56 56 64 > $R/ofg4h2.chain.log 2>&1 < /dev/null
tail -1 $R/ofg4h2.chain.log
( cd $H && source $H/env.ofg4h2-136.yihou.sh && timeout 600 bash $H/run.v3.yihou.sh stop ) > $R/ofg4h2/stop.seq.log 2>&1
echo "$(ts) ofg4h2 stopped; DONE"
