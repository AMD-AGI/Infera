#!/usr/bin/env bash
# Task v0521cap2 (LOGIN HOST, node 136; user chose "1" 10-15 13:2x): wait until 136 is reachable, stop any leftover
# ofg4h2 stack (run.v3s = streaming sync_back, 600 s cap), wait for VRAM < 1 GiB on every GPU, then chain_cap2ls ofg4h3 56 56 64.
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
s136(){ ssh -o ConnectTimeout=20 -o BatchMode=yes crsuse2-m2m-136 "$@"; }
for i in $(seq 1 120); do s136 true 2>/dev/null && break; echo "$(ts) 136 unreachable, waiting"; sleep 120; done
s136 true || { echo "$(ts) STOP: 136 unreachable for 4 h"; exit 1; }
left=$(s136 "docker ps --format '{{.Names}}' | grep -c '^glm52-v0521-yihou-136-ofg4h2-'")
echo "$(ts) 136 reachable; ofg4h2 containers running: $left"
if [[ "$left" != 0 ]]; then ( cd $H && source $H/env.ofg4h2-136.yihou.sh && timeout 600 bash $H/run.v3s.yihou.sh stop ) > $R/ofg4h2/stop.v3s.log 2>&1; echo "$(ts) ofg4h2 stop rc=$?"; fi
vram_max(){ s136 "rocm-smi --showmeminfo vram | grep 'Total Used' | awk '{print \$NF}' | sort -n | tail -1"; }
for i in $(seq 1 90); do v=$(vram_max); [[ -n "$v" && "$v" -lt 1073741824 ]] && break; echo "$(ts) max VRAM $v B, waiting"; sleep 60; done
[[ -n "$v" && "$v" -lt 1073741824 ]] || { echo "$(ts) STOP: max VRAM $v B on 136 after 90 min"; exit 1; }
echo "$(ts) -> ofg4h3 56 56 64 (max VRAM $v B)"
bash $H/chain_cap2ls.yihou.sh $H/env.ofg4h3-136.yihou.sh 56 56 64 > $R/ofg4h3.chain.log 2>&1 < /dev/null
tail -1 $R/ofg4h3.chain.log
( cd $H && source $H/env.ofg4h3-136.yihou.sh && timeout 600 bash $H/run.v3s.yihou.sh stop ) > $R/ofg4h3/stop.seq.log 2>&1
echo "$(ts) ofg4h3 stopped; DONE"
