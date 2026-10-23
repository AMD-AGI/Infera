#!/usr/bin/env bash
# mkenv_pp (task ppdc 2026-10-23, ON 135): env.ppdc1 = env.dcpx11 + PP4 prefill + SPF + write_back + no OffloaderV2 +
# prefill mem 0.88 / max-running 256 + decode empty_cache patch with mem 0.88 / max-running 96; dryrun + tokdiff.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
E=$W/env.ppdc1-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppdc1-135x136.yihou.sh (2026-10-23, user: PP prefill + llying opts + GMU decode, first round with trade-offs) = env.dcpx11 + prefill PP4xTP1 (partition 20,20,20,18, chunk 8192, overlap off, deepseek_v2 PP fix) + shortest-prefill-first + HiCache write_back + no OffloaderV2 + prefill mem 0.88 / max-running 256 + decode empty_cache patch 03 (+ memlog) with decode mem 0.88 / max-running 96. Kept: prefill MTP, hashed shard, DCP pack fix, xfer stats, graph 1..96."; grep -v "^# env.dcpx11" $W/env.dcpx11-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=dcpx11/YIHOU_ROUND=ppdc1/" \
  -e "s/DCPX_PACKNOWAIT=2 /DCPX_PACKNOWAIT=2 DCPX_PP=4 DCPX_SPF=1 DCPX_NOOFFLOAD=1 DCPX_DECODE_EC=1 PREFILL_HICACHE_WRITE_POLICY=write_back PREFILL_MAX_RUNNING=256 /" \
  -e "s/PREFILL_MEM_FRACTION=0.90 /PREFILL_MEM_FRACTION=0.88 /" -e "s/DECODE_MEM_FRACTION=0.85 /DECODE_MEM_FRACTION=0.88 /" -e "s/DECODE_MAX_RUNNING=128 /DECODE_MAX_RUNNING=96 /"; } > $E
grep -o "DCPX_PP=4\|DCPX_SPF=1\|DCPX_NOOFFLOAD=1\|DCPX_DECODE_EC=1\|write_back\|PREFILL_MAX_RUNNING=256\|PREFILL_MEM_FRACTION=0.88\|DECODE_MEM_FRACTION=0.88\|DECODE_MAX_RUNNING=96" $E | tr '\n' ' '; echo
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppdc1-135x136.yihou.txt 2>&1 || { echo DRYRUN_FAIL; tail -5 $W/dryrun.ppdc1-135x136.yihou.txt; }
echo "== tokdiff dcpx11 -> ppdc1"
diff <(tr ' ' '\n' < $W/dryrun.dcpx11-135x136.yihou.txt | sed 's/dcpx11/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppdc1-135x136.yihou.txt | sed 's/ppdc1/ROUND/g') | grep '^[<>]' | grep -v '^[<>] +$' | grep -v '^[<>] \\$' | tr '\n' ' '; echo
