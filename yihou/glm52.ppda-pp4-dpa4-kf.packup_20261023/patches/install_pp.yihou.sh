#!/usr/bin/env bash
# install_pp (task ppdc 2026-10-23, ON 135): PP4 prefill + llying prefill opts + GMU decode patch for the DCP4 decode.
#  - patched overlay copies: deepseek_v2.py (PP fix ab7a898b77), decode.py (+empty_cache patch 03), metrics_reporter.py
#    (+memlog patch 01), built under a yihou scratch tree from overlay 93034eba91
#  - mounts tsv: prefill (conn xferstats2 + dcp_pack nowait2 + deepseek_v2 ppfix), decode (decode.py + metrics_reporter.py)
#  - config.yihou.dcpx.sh knobs: DCPX_PP (4), DCPX_SPF (1), DCPX_NOOFFLOAD (1), DCPX_DECODE_EC (1)
#  - agentx_env.py: live pp; report PREFILL_TP / TP = tp*pp so the per-GPU score divides by 8 GPUs
set -euo pipefail
D=/shared_nfs/yihou_glm52/dcpx_20261022/ppdc; H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
O=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91
T=$D/tree.yihou; mkdir -p $T/python/sglang/srt/models $T/python/sglang/srt/disaggregation $T/python/sglang/srt/managers/scheduler_components
for f in python/sglang/srt/models/deepseek_v2.py python/sglang/srt/disaggregation/decode.py; do cp $O/$f $T/$f; done
MR=python/sglang/srt/managers/scheduler_components/metrics_reporter.py; [[ -f $O/$MR ]] || { echo "no $MR"; exit 1; }; echo "metrics_reporter: $MR"
mkdir -p $T/$(dirname $MR); cp $O/$MR $T/$MR
cd $T
sed -n '/^diff --git/,$p' $D/pp-deepseek.patch | patch -p1 --no-backup-if-mismatch
patch -p1 --no-backup-if-mismatch < $D/ec03.diff
patch -p1 --no-backup-if-mismatch < $D/memlog01.diff
grep -n "config.hidden_size == 7168\|yihou gmu\|mem_get_info" python/sglang/srt/models/deepseek_v2.py python/sglang/srt/disaggregation/decode.py $MR | head
grep -n "^import torch\|^    import torch" python/sglang/srt/disaggregation/decode.py | head -2
for f in python/sglang/srt/models/deepseek_v2.py python/sglang/srt/disaggregation/decode.py $MR; do python3 -m py_compile $f && cp $f $H/mounts-v0522/$(basename $f .py).yihou-ppdc.py; done
md5sum $H/mounts-v0522/*.yihou-ppdc.py
TP=$H/mounts.yihou.dcpx-pp-prefill.tsv; TD=$H/mounts.yihou.dcpx-ec-decode.tsv
[[ -e $TP ]] || { cat $H/mounts.yihou.dcpx-xstats2-packnowait2.tsv; printf '%s\t%s\n' $H/mounts-v0522/deepseek_v2.yihou-ppdc.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py; } > $TP
[[ -e $TD ]] || printf '%s\t%s\n%s\t%s\n' $H/mounts-v0522/decode.yihou-ppdc.py /sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py \
  $H/mounts-v0522/metrics_reporter.yihou-ppdc.py /sgl-workspace/sglang/$MR > $TD
cat $TP $TD
C=$H/config.yihou.dcpx.sh
grep -q DCPX_PP $C || { cp -p $C $D/config.yihou.dcpx.sh.before-pp; python3 -I - $C <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '# --- KV rails:'
assert s.count(a) == 1
s = s.replace(a, '''# --- ppdc (10-23): PP4 prefill + llying prefill options + GMU decode patch -------------------------------------------
# DCPX_PP=4: prefill TP1 x PP4 (one 4-GPU prefill instance, stage k = GPU k), layer partition 20,20,20,18, chunk and
# max-prefill-tokens 8192, overlap off (llying 2p1d-v0522 kit-v6-pp4.diff); mounts the deepseek_v2 PP fix (ab7a898b77)
# next to the conn/dcp_pack patches (needs DCPX_SHARD=xstats DCPX_PACKNOWAIT=2).
: "${DCPX_PP:=0}" "${DCPX_SPF:=0}" "${DCPX_NOOFFLOAD:=0}" "${DCPX_DECODE_EC:=0}"
if [[ "$DCPX_PP" != 0 ]]; then
    [[ "$DCPX_PP" =~ ^[2-8]$ ]] || { echo "config.yihou.dcpx.sh: DCPX_PP must be 0 or 2..8" >&2; return 2 2>/dev/null || exit 2; }
    [[ "$DCPX_SHARD" == xstats && "$DCPX_PACKNOWAIT" == 2 ]] || { echo "config.yihou.dcpx.sh: DCPX_PP needs DCPX_SHARD=xstats DCPX_PACKNOWAIT=2" >&2; return 2 2>/dev/null || exit 2; }
    PREFILL_TP=1; PREFILL_EP=1; PREFILL_DP=1; PREFILL_DPA=0; PREFILL_CHUNK_SIZE=8192
    PREFILL_EXTRA_ARGS+=" --pp-size $DCPX_PP --disable-overlap-schedule --max-prefill-tokens 8192"
    PREFILL_EXTRA_ENV+=" SGLANG_PP_LAYER_PARTITION=${DCPX_PP_PARTITION:-20,20,20,18}"
    export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-pp-prefill.tsv"
fi
# DCPX_SPF=1: prefill --schedule-policy shortest-prefill-first (llying 2p1d-aus stage-b4-spf, +15.4 % at 2P1D C176).
[[ "$DCPX_SPF" == 1 ]] && PREFILL_EXTRA_ARGS+=" --schedule-policy shortest-prefill-first"
# DCPX_NOOFFLOAD=1: drop OffloaderV2 from the prefill (PP stages hold ~94-120 GB of weights; S3 with it was -19 %).
if [[ "$DCPX_NOOFFLOAD" == 1 ]]; then
    PREFILL_EXTRA_ARGS="$(echo " $PREFILL_EXTRA_ARGS " | sed -E 's/ --offload-(group-size|num-in-group|prefetch-step|mode) [^ ]+//g; s/^ +//; s/ +$//')"
fi
# DCPX_DECODE_EC=1: decode.py all-rank empty_cache when free < 6 GiB every 8 loop iterations (other session's
# glm52.dcp4-decode-gmu.packup_20261023 patch 03) + memlog on the TP0 Decode-batch line (patch 01).
[[ "$DCPX_DECODE_EC" == 1 ]] && export DECODE_MOUNTS_ADD="$WS/harness/mounts.yihou.dcpx-ec-decode.tsv"

''' + a)
s = s.replace('${DCPX_GRAPH_BS:+-gbs${DCPX_GRAPH_BS##*,}}', '${DCPX_GRAPH_BS:+-gbs${DCPX_GRAPH_BS##*,}}-pp${DCPX_PP}-spf${DCPX_SPF}-nooff${DCPX_NOOFFLOAD}-ec${DCPX_DECODE_EC}')
open(p, "w").write(s)
PY
}
bash -n $C && echo CONFIG_OK
F=$H/bench-harness/tools/agentx_env.py
grep -q '"pp": int(command_value' $F || { cp -p $F $D/agentx_env.py.before-pp; python3 -I - $F <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
a = '        "tp": int(command_value(command, "--tp-size")),\n'
assert s.count(a) == 1
s = s.replace(a, a + '        # ppdc 2026-10-23: PP prefill; the reported TP below is tp*pp so per-GPU scores count every GPU.\n        "pp": int(command_value(command, "--pp-size", "1")),\n')
for k in ['"TP": str(p["tp"])', '"PREFILL_TP": str(p["tp"])']:
    assert s.count(k) == 1, k
    s = s.replace(k, k.replace('str(p["tp"])', 'str(p["tp"] * p.get("pp", 1))'))
open(p, "w").write(s)
PY
}
python3 -I -m py_compile $F && grep -n '"pp"\|p.get("pp"' $F
echo INSTALL_PP_DONE
