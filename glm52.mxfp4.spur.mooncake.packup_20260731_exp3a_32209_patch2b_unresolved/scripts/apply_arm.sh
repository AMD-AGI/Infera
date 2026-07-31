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
python/sglang/srt/disaggregation/decode.py
python/sglang/srt/layers/dp_attention.py"

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
e3a)
  # SPLIT of e3, half 1: #32209-shape patch 2b, but OUR patch 4.
  # Isolates whether the 2b port stands on its own. Everything else is the
  # e1 baseline (kit patch 1, kit 2a, kit 3).
  ap dsa_indexer_hip_dp_padded_rows
  ap dsa_backend_dp_sync_and_page_table_rows
  python3 $W/e3/strip_patch2b_v1.py    || exit 1
  python3 $W/e3/patch2b_32209_style.py || exit 1
  ap eagle_worker_v2_uniform_draft_graph
  # Round 2 instrumentation. Round 1 crashed at conc=32 with the
  # dp_gather_replicate size mismatch and left two facts:
  #   - pad_mode=1 (MAX_LEN) on ALL eight ranks -> the DpPaddingMode
  #     hypothesis (charter H3 / instr_e3.py C2) is refuted for this crash;
  #   - ranks 6,7 handed the collective 6 rows into a slot planned for 4.
  # instr_e3 (now reading global_num_tokens_cpu, which the first revision
  # got wrong and logged as None) shows the PLAN vs the actual local rows;
  # instr_p2bv2_rows shows whether the 2b trim fired at all on that
  # iteration, which decides whether 2b is causal or a bystander.
  python3 $W/e3/instr_e3.py            || exit 1
  python3 $W/e3/instr_p2bv2_rows.py    || exit 1
  # Round 3. Round 2 narrowed the crash to ONE fact: on the crashing
  # iteration rank 4 entered the DSA decode site once while its peers
  # entered twice, and it alone delivered 4 rows into a 3-row slot. Five
  # ranks trimmed 2 rows and still delivered 3, so trim/restore is
  # row-neutral and patch 2b does not create the extra row. This probe
  # decides the last binary question: did rank 4 replay the draft graph for
  # one step (A -- a replayed graph runs no Python, so a step goes
  # unrecorded), or did it skip the site (B)?
  python3 $W/e3/instr_draft_steps.py   || exit 1
  MARKERS="dsa_indexer.py:_q_mqa
dsa_backend.py:_p2bv2_trim_decode_dp_padding
eagle_worker_v2.py:_needs_eager_local
dp_attention.py:_e3i_log
dsa_backend.py:_p2brows_log
eagle_worker_v2.py:_dstep_log"
  # Our patch 4 must be the ONLY draft-graph mechanism present.
  ANTI="dsa_backend.py:_glm52_match_page_table_rows
dp_attn.py:can_draft_cuda_graph"
  P2A=want
  ;;
e3b)
  # SPLIT of e3, half 2: OUR patch 2b, but #32209-shape patch 4, plus the
  # graph-usage instrumentation. Isolates whether the 4 port stands on its
  # own AND whether it actually uses the draft graph -- charter criterion 5,
  # which a green conc=32 cannot answer.
  ap dsa_indexer_hip_dp_padded_rows
  ap dsa_backend_dp_sync_and_page_table_rows
  python3 $W/e3/patch4_32209_style.py       || exit 1
  python3 $W/common/instr_graph_usage.py    || exit 1
  MARKERS="dsa_indexer.py:_q_mqa
dsa_backend.py:_glm52_match_page_table_rows
dp_attn.py:can_draft_cuda_graph
eagle_worker_v2.py:requires_dp_attention_eager_forward
eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph
eagle_draft_cuda_graph_runner.py:_guse_record
eagle_worker_v2.py:_GUSE_WHY"
  # #32209's patch 4 must be the ONLY draft-graph mechanism present.
  ANTI="eagle_worker_v2.py:_needs_eager_local
dsa_backend.py:_p2bv2_trim_decode_dp_padding"
  P2A=want
  ;;
e3c)
  # e3a + the half of #32209's patch 2b that the first port omitted.
  #
  # e3a (#32209 trim only, our patch 4) crashed at conc=32 in all THREE runs,
  # on two different node pairs and a rebuilt image, always with
  # `output tensor size must be equal to world_size times input tensor size`.
  # Round-3 instrumentation ruled out every alternative: the padding mode
  # agrees on all 8 ranks, patch 4's vote is uniform (51 graph / 7 eager,
  # identical per rank), no rank replayed the graph on the crashing iteration,
  # no rank skipped a draft step, and trim+restore is row-neutral. What
  # remained is that the ranks whose real token count was below the group plan
  # delivered a row count matching NEITHER value -- a count carried over
  # through spec_info.hidden_states, which nothing on our path trims back.
  #
  # This arm adds upstream's own fix for exactly that carry
  # (_slice_draft_output_to_local_tokens) and changes nothing else, so a green
  # conc=32 here attributes to that hunk alone. Keeping e3a intact means the
  # failing configuration stays reproducible for comparison.
  ap dsa_indexer_hip_dp_padded_rows
  ap dsa_backend_dp_sync_and_page_table_rows
  python3 $W/e3/strip_patch2b_v1.py     || exit 1
  python3 $W/e3/patch2b_32209_style.py  || exit 1
  python3 $W/e3/patch2b_32209_slice.py  || exit 1
  ap eagle_worker_v2_uniform_draft_graph
  # Same probes as e3a round 3, so the two arms are read the same way.
  #
  # NOT instr_graph_usage.py: it anchors on the `can_run_dp_draft_cuda_graph`
  # term that #32209's patch 4 introduces, and this arm runs OUR patch 4, so
  # the anchor does not exist here. Draft-graph usage is instead read from
  # `GLM52_DSTEP vote graph=` -- the decision each rank ACTS on after our
  # patch 4's all-reduce, which answers the same charter-criterion-5 question
  # (is the graph provably taken, or has the arm silently become Variant B).
  python3 $W/e3/instr_e3.py             || exit 1
  python3 $W/e3/instr_p2bv2_rows.py     || exit 1
  python3 $W/e3/instr_draft_steps.py    || exit 1
  MARKERS="dsa_indexer.py:_q_mqa
dsa_backend.py:_p2bv2_trim_decode_dp_padding
eagle_worker_v2.py:_slice_draft_output_to_local_tokens
eagle_worker_v2.py:_needs_eager_local
dp_attention.py:_e3i_log
dsa_backend.py:_p2brows_log
eagle_worker_v2.py:_dstep_log"
  # Our patch 4 is the draft-graph mechanism here (as in e3a); #32209's must
  # be absent, and v1's page-table expansion must be gone.
  ANTI="dsa_backend.py:_glm52_match_page_table_rows
dp_attn.py:can_draft_cuda_graph"
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
