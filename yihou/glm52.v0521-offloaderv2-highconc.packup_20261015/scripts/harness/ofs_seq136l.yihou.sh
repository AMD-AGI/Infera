#!/usr/bin/env bash
# Task v0521cap2 (LOGIN HOST, node 136 = plain ssh): run perf rounds back to back on 136 with chain_cap2l,
# stopping each stack and waiting for 0 KFD processes before the next. Usage: ofs_seq136l.yihou.sh "<round> C1 C2 C3" ...
set -uo pipefail
R=/shared_nfs/yihou_glm52/v0521dbg_20261010/runs; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
ts(){ date -u +%FT%TZ; }
kfd(){ ssh -o ConnectTimeout=15 -o BatchMode=yes crsuse2-m2m-136 "rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+ '"; }
for spec in "$@"; do
  set -- $spec; r=$1; shift
  for i in $(seq 1 60); do k=$(kfd); [[ "$k" == 0 ]] && break; sleep 30; done
  [[ "$k" == 0 ]] || { echo "$(ts) STOP: $k KFD processes on 136 before $r"; exit 1; }
  echo "$(ts) -> $r $*"
  bash $H/chain_cap2l.yihou.sh $H/env.$r-136.yihou.sh "$@" > $R/$r.chain.log 2>&1 < /dev/null
  tail -1 $R/$r.chain.log
  ( cd $H && source $H/env.$r-136.yihou.sh && bash $H/run.v3.yihou.sh stop ) > $R/$r/stop.seq.log 2>&1
  echo "$(ts) $r stopped"
done
echo "$(ts) seq DONE"
