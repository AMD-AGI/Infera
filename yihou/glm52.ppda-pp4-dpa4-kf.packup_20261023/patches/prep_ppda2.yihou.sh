#!/usr/bin/env bash
# prep_ppda2 (task ppdc 2026-10-23, ON 135): DCPX_KF=1 knob (kf-glm52 decode-only overlay: mount dir -> /yihou_overlay,
# PYTHONPATH + YIHOU_KF_SPARSE/INDEXER/MOE=1; engages only on a non-DCP EP1 decode), env.ppda2 = env.ppda1 + DCPX_KF=1,
# dryrun + token diff. Does NOT launch.
set -uo pipefail
W=/shared_nfs/yihou_glm52/g4tune_20261019; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness; D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc
K=/shared_nfs/yihou_glm52/kf_e2e_20261023/kf/e2e/overlay
ls $K | head; [[ -f $K/sitecustomize.py && -d $K/yihou_kf ]] || { echo "KF overlay missing"; exit 1; }
T=$H/mounts.yihou.dcpx-kf-decode.tsv
[[ -e $T ]] || printf '%s\t%s\n' $K /yihou_overlay > $T
cat -A $T
C=$H/config.yihou.dcpx.sh
grep -q DCPX_KF $C || { cp -p $C $D/config.yihou.dcpx.sh.before-kf; python3 -I - $C <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '[[ "$DCPX_DECODE_EC" == 1 ]] && export DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-ec-decode.tsv"\n'
assert s.count(a) == 1
s = s.replace(a, a + '''# DCPX_KF=1 (ppda2): kf-glm52 decode-kernel overlay (sparse MLA + indexer + MoE; engages only on a non-DCP EP1 decode,
# TARGET_VERIFY <= 128 rows, needs SGLANG_OPT_USE_TOPK_V2=false). Decode log must show "[yihou-kf] installed".
: "${DCPX_KF:=0}"
if [[ "$DCPX_KF" == 1 ]]; then
    [[ "$DCPX_DECODE_EC" == 1 ]] && { echo "config.yihou.dcpx.sh: DCPX_KF=1 with DCPX_DECODE_EC=1 needs a merged decode tsv" >&2; return 2 2>/dev/null || exit 2; }
    export DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-kf-decode.tsv"
    DECODE_EXTRA_ENV+=" PYTHONPATH=/yihou_overlay YIHOU_KF_SPARSE=1 YIHOU_KF_INDEXER=1 YIHOU_KF_MOE=1"
fi
''')
s = s.replace('-ec${DCPX_DECODE_EC}', '-ec${DCPX_DECODE_EC}-kf${DCPX_KF:-0}')
open(p, "w").write(s)
PY
}
bash -n $C && echo CONFIG_OK
E=$W/env.ppda2-135x136.yihou.sh
[[ -e $E ]] || { echo "# env.ppda2-135x136.yihou.sh (2026-10-23) = env.ppda1 + DCPX_KF=1 ONLY (kf-glm52 decode kernels on the DPA4 decode; same-node they cut TPOT 15 -> 11.2 ms)."; grep -v "^# env.ppda1" $W/env.ppda1-135x136.yihou.sh | sed -e "s/YIHOU_ROUND=ppda1/YIHOU_ROUND=ppda2/" -e "s/ DCPX_DECODE_EC=0 / DCPX_DECODE_EC=0 DCPX_KF=1 /"; } > $E
grep -c "DCPX_KF=1" $E
( source $E; cd $H; bash $H/run.v3x.yihou.sh dryrun ) > $W/dryrun.ppda2-135x136.yihou.txt 2>&1 || { echo DRYRUN_FAIL; tail -4 $W/dryrun.ppda2-135x136.yihou.txt; }
echo "== tokdiff ppda1 -> ppda2"; diff <(tr ' ' '\n' < $W/dryrun.ppda1-135x136.yihou.txt | sed 's/ppda1/ROUND/g') <(tr ' ' '\n' < $W/dryrun.ppda2-135x136.yihou.txt | sed 's/ppda2/ROUND/g') | grep '^[<>]' | grep -v '^[<>] +$' | tr '\n' ' '; echo
grep -o "SGLANG_OPT_USE_TOPK_V2=[a-z]*" $W/dryrun.ppda2-135x136.yihou.txt | sort | uniq -c
