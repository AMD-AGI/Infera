#!/bin/bash
# Apply one experiment arm's patch set inside the dbg2 container, then prove
# every marker reached the BYTECODE. A stale .pyc silently reverts a patch and
# has already invalidated one full experiment (CLAUDE.md), so source-level
# verification is not accepted here.
#
# Verification uses IDENTIFIERS, never `#` comment markers: the compiler
# discards comments, so a comment marker reads as a false negative
# (PITFALLS P6).
#
# The four baseline patches come from the packup kit's diffs, NOT from the
# ad-hoc fix_bug*.py scripts in ~/glm52_fix. Those scripts are from an earlier
# round and encode DIFFERENT fixes for patch 2a (an empty-batch .max() guard
# plus a base_spec_worker.py edit) than the kit shipped and verified
# (max_seqlen_k = req_to_token.shape[1], dsa_backend.py only). The kit diffs
# are the 2540/2540-verified artifacts; the scripts are not.
#
# Usage (from the host):
#   spur exec <job> bash -c 'docker exec dbg2 bash /shared_nfs/yihou_exp3way/common/apply_arm.sh e1'
set -u
ARM="${1:?e1|e2|e3}"
SG=/sgl-workspace/sglang
SRT=$SG/python/sglang/srt
W=/shared_nfs/yihou_exp3way
K=$W/kit_patches
ANTI=""

# Files any arm may touch. Reset them so the arm is applied to a pristine tree
# and cannot inherit a previous arm's edits.
RESET="python/sglang/srt/layers/attention/dsa/dsa_indexer.py
python/sglang/srt/layers/attention/dsa_backend.py
python/sglang/srt/speculative/eagle_worker_v2.py
python/sglang/srt/speculative/base_spec_worker.py
python/sglang/srt/speculative/eagle_draft_cuda_graph_runner.py
python/sglang/srt/models/deepseek_nextn.py
python/sglang/srt/managers/scheduler_components/dp_attn.py
python/sglang/srt/managers/schedule_batch.py
python/sglang/srt/model_executor/forward_batch_info.py
python/sglang/srt/disaggregation/decode.py"

echo "=== arm $ARM: resetting to pristine ==="
cd $SG || exit 1
git checkout -- $RESET 2>/dev/null
find python/sglang/srt -name __pycache__ -exec rm -rf {} + 2>/dev/null
echo "  dirty tracked files after reset: $(git status --short --untracked-files=no python/sglang/srt | wc -l)"

ap() {  # ap <kit diff basename>
  echo "  applying $1"
  git apply "$K/$1.diff" || { echo "APPLY FAILED: $1"; exit 1; }
}

echo "=== arm $ARM: applying ==="
# Patch 3 (nextn eh_proj bf16) is required by EVERY arm: without it the server
# dies at weight load with a 3072-vs-6144 shape mismatch on the draft head.
ap deepseek_nextn_glm52_mtp_bf16

case "$ARM" in
e1)
  # patch 1 v2 (#32762 shape) + 2a + 2b (kit) + 3 + 4 (kit).
  # Kit patch 1 goes on first; the v2 script rewrites its aiter block in place.
  ap dsa_indexer_hip_dp_padded_rows
  python3 $W/e1/patch1_v2_32762_style.py || exit 1
  ap dsa_backend_dp_sync_and_page_table_rows
  ap eagle_worker_v2_uniform_draft_graph
  MARKERS="dsa_indexer.py:_p1v2_trim
dsa_backend.py:_glm52_match_page_table_rows
eagle_worker_v2.py:_needs_eager_local"
  # Patch 2a is an expression change with no new identifier; its presence is
  # asserted separately below by source inspection at a known line.
  P2A=want
  ;;
e2)
  # patch 1 (kit) + patch 3 ONLY. IndexShare is disabled by a launcher flag,
  # not a patch; patches 2 and 4 are deliberately absent -- that is the
  # hypothesis under test.
  ap dsa_indexer_hip_dp_padded_rows
  MARKERS="dsa_indexer.py:_q_mqa"
  ANTI="dsa_backend.py:_glm52_match_page_table_rows
eagle_worker_v2.py:_needs_eager_local"
  P2A=absent
  ;;
e3)
  # patch 1 (kit) + 2a (kit) + 2b (#32209 shape) + 3 + 4 (#32209 shape).
  # The kit's dsa_backend diff carries BOTH 2a and 2b, so it is applied and
  # then 2b is reverted out of it, leaving 2a. The 2b-v2 script refuses to run
  # while the v1 helper is present, which is the guard that this worked.
  ap dsa_indexer_hip_dp_padded_rows
  ap dsa_backend_dp_sync_and_page_table_rows
  python3 $W/e3/strip_patch2b_v1.py    || exit 1
  python3 $W/e3/patch2b_32209_style.py || exit 1
  python3 $W/e3/patch4_32209_style.py  || exit 1
  MARKERS="dsa_indexer.py:_q_mqa
dsa_backend.py:_p2bv2_trim_decode_dp_padding
dp_attn.py:can_draft_cuda_graph
eagle_worker_v2.py:requires_dp_attention_eager_forward
eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph
forward_batch_info.py:can_run_dp_draft_cuda_graph
schedule_batch.py:force_disable_draft_cuda_graph
decode.py:force_disable_draft_cuda_graph"
  ANTI="dsa_backend.py:_glm52_match_page_table_rows
eagle_worker_v2.py:_needs_eager_local"
  P2A=want
  ;;
*) echo "unknown arm $ARM" >&2; exit 1 ;;
esac

echo "=== arm $ARM: verification ==="
fail=0

# Patch 3: a file-content check, the marker is a literal in an f-string.
n3=$(grep -c 'num_hidden_layers}.eh_proj' $SRT/models/deepseek_nextn.py || true)
echo "  patch3 nextn eh_proj  -> src=$n3 (want 1)"
[ "$n3" -eq 1 ] || fail=1

# Patch 2a: `max_seqlen_k = self.req_to_token.shape[1]` on the
# needs_cpu_seq_lens=False arm of init_forward_metadata. It introduces no new
# identifier, and the same expression already appears twice on the pre-existing
# `dsa_drop_wide_page_table` graph-capture paths -- so counting the expression
# gives 3, not 1. Key off the comment line the patch adds instead, which is
# unique. This is a SOURCE check by necessity; the bytecode cannot show it.
n2a=$(grep -c 'GLM52_BUG2_FIX_A: needs_cpu_seq_lens=False nulls the host mirror' $SRT/layers/attention/dsa_backend.py || true)
if [ "$P2A" = want ]; then
  echo "  patch2a max_seqlen_k  -> src=$n2a (want 1)"
  [ "$n2a" -eq 1 ] || fail=1
else
  echo "  patch2a max_seqlen_k  -> src=$n2a (want 0)"
  [ "$n2a" -eq 0 ] || fail=1
fi

check() {  # check <basename.py> <identifier> <gt0|eq0>
  local f="$1" m="$2" want="$3" p d b pyc n
  p=$(find $SRT -name "$f" | head -1)
  if [ -z "$p" ]; then echo "  MISSING MODULE $f"; return 1; fi
  d=$(dirname "$p"); b=$(basename "$p" .py)
  rm -f "$d/__pycache__/$b."*.pyc
  python3 -c "import py_compile;py_compile.compile('$p',doraise=True)" 2>&1 || {
    echo "  COMPILE FAIL $f"; return 1; }
  pyc=$(ls "$d/__pycache__/$b."*.pyc 2>/dev/null | head -1)
  n=$(strings "$pyc" | grep -c "$m" || true)
  if [ "$want" = gt0 ]; then
    echo "  WANT>0  $f :: $m  -> pyc=$n"; [ "$n" -gt 0 ]
  else
    echo "  WANT=0  $f :: $m  -> pyc=$n"; [ "$n" -eq 0 ]
  fi
}

for spec in $MARKERS; do check "${spec%%:*}" "${spec#*:}" gt0 || fail=1; done
for spec in $ANTI;    do check "${spec%%:*}" "${spec#*:}" eq0 || fail=1; done

if [ "$fail" -ne 0 ]; then
  echo "ARM $ARM VERIFICATION FAILED"
  exit 1
fi
echo "ARM $ARM OK"
