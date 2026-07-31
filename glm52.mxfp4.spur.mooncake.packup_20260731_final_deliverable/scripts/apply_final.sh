#!/bin/bash
# FINAL canonical patch set for GLM-5.2 PD + DP-attention + EAGLE MTP on gfx950.
#
# Each of the five patches is in the most upstream-canonical shape that our own
# measurements support. Where upstream's shape was tested and FAILED, we keep
# ours and say so -- this file is the record of that choice.
#
#   patch 1  dsa_indexer.py       #32762 shape (NPU precedent)  -- exp1 PASS
#   patch 2a dsa_backend.py       ours (no upstream counterpart found)
#   patch 2b dsa_backend.py       OURS -- #32209's trim shape FAILED 0/32 x7
#   patch 3  deepseek_nextn.py    ours (#32175 carries the same fix upstream)
#   patch 4  6 files              #32209 shape (rides existing all-gather) -- exp3b PASS
#
# NOTE: patch 1 v2 and patch 4 v2 have each been validated, but never TOGETHER.
# exp1 ran 1v2 with OUR patch 4; exp3b ran #32209's patch 4 with OUR patch 1.
# This arm is their first joint run and that is the point of the final validation.
set -u
SG=/sgl-workspace/sglang
SRT=$SG/python/sglang/srt
W=/shared_nfs/yihou_exp3way
K=$W/kit_patches

RESET="python/sglang/srt/layers/attention/dsa/dsa_indexer.py
python/sglang/srt/layers/attention/dsa_backend.py
python/sglang/srt/layers/dp_attention.py
python/sglang/srt/speculative/eagle_worker_v2.py
python/sglang/srt/speculative/eagle_info.py
python/sglang/srt/speculative/base_spec_worker.py
python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py
python/sglang/srt/models/deepseek_nextn.py
python/sglang/srt/managers/scheduler_components/dp_attn.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/disaggregation/decode.py"

echo "=== FINAL: reset to pristine ==="
cd $SG || exit 1
git checkout -- $RESET 2>/dev/null
find python/sglang/srt -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "  dirty after reset: $(git status --short --untracked-files=no python/sglang/srt | wc -l)"

ap() { echo "  applying $1"; git apply "$K/$1.diff" || { echo "APPLY FAILED: $1"; exit 1; }; }

echo "=== FINAL: applying ==="
ap deepseek_nextn_glm52_mtp_bf16                 # patch 3
ap dsa_indexer_hip_dp_padded_rows                # patch 1 v1 (base for v2)
python3 $W/e1/patch1_v2_32762_style.py || exit 1 # patch 1 -> #32762 shape
ap dsa_backend_dp_sync_and_page_table_rows       # patch 2a + our 2b
python3 $W/e3/patch4_32209_style.py   || exit 1  # patch 4 -> #32209 shape

echo "=== FINAL: verification (BYTECODE, identifiers not comments) ==="
fail=0
MARKERS="dsa_indexer.py:_p1v2_trim
dsa_backend.py:_glm52_match_page_table_rows
dp_attn.py:can_draft_cuda_graph
eagle_worker_v2.py:requires_dp_attention_eager_forward
eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph
forward_batch_info.py:can_run_dp_draft_cuda_graph
schedule_batch.py:force_disable_draft_cuda_graph
decode.py:force_disable_draft_cuda_graph"
# Exactly one draft-graph mechanism, and NOT the failed 2b port.
ANTI="eagle_worker_v2.py:_needs_eager_local
dsa_backend.py:_p2bv2_trim_decode_dp_padding
dsa_indexer.py:GLM52_DSTEP
dsa_backend.py:GLM52_MULTI"

n3=$(grep -c 'num_hidden_layers}.eh_proj' $SRT/models/deepseek_nextn.py || true)
echo "  patch3 nextn eh_proj -> src=$n3 (want 1)"; [ "$n3" -eq 1 ] || fail=1
n2a=$(grep -c 'GLM52_BUG2_FIX_A: needs_cpu_seq_lens=False nulls the host mirror' $SRT/layers/attention/dsa_backend.py || true)
echo "  patch2a max_seqlen_k -> src=$n2a (want 1)"; [ "$n2a" -eq 1 ] || fail=1

check() {
  local f="$1" m="$2" want="$3" p d b pyc n
  p=$(find $SRT -name "$f" | head -1)
  [ -z "$p" ] && { echo "  MISSING MODULE $f"; return 1; }
  d=$(dirname "$p"); b=$(basename "$p" .py)
  rm -f "$d/__pycache__/$b."*.pyc
  python3 -c "import py_compile;py_compile.compile('$p',doraise=True)" 2>&1 || { echo "  COMPILE FAIL $f"; return 1; }
  pyc=$(ls "$d/__pycache__/$b."*.pyc 2>/dev/null | head -1)
  n=$(strings "$pyc" | grep -c "$m" || true)
  if [ "$want" = gt0 ]; then echo "  WANT>0  $f :: $m -> pyc=$n"; [ "$n" -gt 0 ]
  else echo "  WANT=0  $f :: $m -> pyc=$n"; [ "$n" -eq 0 ]; fi
}
for s in $MARKERS; do check "${s%%:*}" "${s#*:}" gt0 || fail=1; done
for s in $ANTI;    do check "${s%%:*}" "${s#*:}" eq0 || fail=1; done

[ "$fail" -ne 0 ] && { echo "FINAL VERIFICATION FAILED"; exit 1; }
echo "FINAL ARM OK"
