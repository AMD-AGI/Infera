#!/usr/bin/env bash
# Apply the GLM-5.2 DSA patch set to the sglang checkout in an engine image, and
# prove every patch reached the BYTECODE.
#
# Used by Dockerfile.sglang.dmabuf at build time, and runnable by hand inside a
# container for iteration.
#
# WHY BYTECODE VERIFICATION.  Python caches compiled modules in __pycache__ keyed
# on the source mtime.  A patch script that restores a backup with shutil.copy2
# preserves the original mtime, so the edited .py can match a stale .pyc and
# CPython silently runs the UNPATCHED bytecode.  This has already invalidated a
# full experiment here -- the source showed the fix, the runtime did not have it.
# So we drop __pycache__ and re-verify against freshly compiled bytecode.
#
# Verification greps for IDENTIFIERS, never for `#` comment markers: the compiler
# discards comments, so a comment marker reads as a false negative.
set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCH_DIR="${PATCH_DIR:-/tmp/sglang_dsa}"
SRT="$SGLANG_DIR/python/sglang/srt"

# Applied in this order.  01 must precede nothing in particular -- the four touch
# disjoint files except that 02 carries both patch 2a and 2b in one diff.
PATCHES=(
  dsa_indexer_hip_dp_padded_rows.diff
  dsa_backend_dp_sync_and_page_table_rows.diff
  deepseek_nextn_glm52_mtp_bf16.diff
  draft_cuda_graph_dp_vote.diff
)

# module basename : identifier that must be present in the compiled bytecode
MARKERS=(
  "dsa_indexer.py:_p1v2_trim"
  "dsa_backend.py:_glm52_match_page_table_rows"
  "dp_attn.py:can_draft_cuda_graph"
  "eagle_worker_v2.py:requires_dp_attention_eager_forward"
  "eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph"
  "forward_batch_info.py:can_run_dp_draft_cuda_graph"
  "schedule_batch.py:force_disable_draft_cuda_graph"
  "decode.py:force_disable_draft_cuda_graph"
)

echo "=== sglang DSA patches: applying to $SGLANG_DIR ==="
cd "$SGLANG_DIR"
for p in "${PATCHES[@]}"; do
  echo "  --- $p"
  # --fuzz=0: these target a pinned sglang commit.  A fuzzy apply that "succeeds"
  # against a different base is worse than a clean failure.
  patch -p1 --fuzz=0 < "$PATCH_DIR/$p"
done

echo "=== dropping __pycache__ so verification cannot read a stale .pyc ==="
find "$SGLANG_DIR/python/sglang/srt" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "=== verifying every patch reached the bytecode ==="
fail=0

# Patch 3's marker is a literal inside an f-string -- check the source, since the
# f-string is split across bytecode constants.
n3=$(grep -c 'num_hidden_layers}.eh_proj' "$SRT/models/deepseek_nextn.py" || true)
echo "  patch3 nextn eh_proj      -> src=$n3 (want 1)"
[ "$n3" -eq 1 ] || fail=1

# Patch 2a changes an expression and introduces no new identifier, and the same
# expression already appears on two pre-existing graph-capture paths -- so
# counting it gives 3, not 1.  Key off the unique comment the patch adds.  This
# is a SOURCE check by necessity; bytecode cannot show it.
n2a=$(grep -c 'GLM52_BUG2_FIX_A: needs_cpu_seq_lens=False nulls the host mirror' \
      "$SRT/layers/attention/dsa_backend.py" || true)
echo "  patch2a max_seqlen_k      -> src=$n2a (want 1)"
[ "$n2a" -eq 1 ] || fail=1

for spec in "${MARKERS[@]}"; do
  f="${spec%%:*}"; m="${spec#*:}"
  p=$(find "$SRT" -name "$f" | head -1)
  if [ -z "$p" ]; then echo "  MISSING MODULE $f"; fail=1; continue; fi
  d=$(dirname "$p"); b=$(basename "$p" .py)
  rm -f "$d/__pycache__/$b."*.pyc
  python3 -c "import py_compile; py_compile.compile('$p', doraise=True)"
  pyc=$(ls "$d/__pycache__/$b."*.pyc 2>/dev/null | head -1)
  n=$(strings "$pyc" | grep -c "$m" || true)
  printf "  %-42s -> pyc=%s (want >0)\n" "$f :: $m" "$n"
  [ "$n" -gt 0 ] || fail=1
done

if [ "$fail" -ne 0 ]; then
  echo "SGLANG DSA PATCH VERIFICATION FAILED" >&2
  exit 1
fi
echo "=== all sglang DSA patches verified in bytecode ==="
