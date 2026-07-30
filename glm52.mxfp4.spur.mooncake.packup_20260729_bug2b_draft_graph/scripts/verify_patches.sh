#!/bin/bash
# Verify all four patches apply to a pristine sglang tree with ZERO fuzz, and that
# the result compiles.
#
# Why this exists: `patch --dry-run` passing does NOT mean the hunks landed correctly
# -- patch fuzzes by default, and a hand-written diff once silently dropped its second
# hunk while still "passing". So this uses --fuzz=0, applies for real into a scratch
# copy, and byte-compiles the result.
#
# Usage:
#   verify_patches.sh                       # against /sgl-workspace/sglang (in-container)
#   verify_patches.sh /path/to/sglang       # against any pristine checkout
#
# Expected sglang commit: 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd (v0.5.15.post1)

set -u
SGL="${1:-/sgl-workspace/sglang}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATCHES="$HERE/patches"
SCRATCH="${SCRATCH:-/tmp/glm52_patch_verify}"

# Order matters: dsa_indexer carries Bug 1 + Bug 6, and dsa_backend's Bug 5 hunk sits
# next to it. deepseek_nextn is independent but MUST be applied -- without it the
# server dies at weight load with "size of tensor a (3072) must match ... b (6144)".
FILES=(
  "python/sglang/srt/layers/attention/dsa/dsa_indexer.py|dsa_indexer_hip_dp_padded_rows.diff"
  "python/sglang/srt/layers/attention/dsa_backend.py|dsa_backend_dp_sync_and_page_table_rows.diff"
  "python/sglang/srt/models/deepseek_nextn.py|deepseek_nextn_glm52_mtp_bf16.diff"
  "python/sglang/srt/speculative/eagle_worker_v2.py|eagle_worker_v2_uniform_draft_graph.diff"
)

echo "sglang tree : $SGL"
if [ -d "$SGL/.git" ]; then
  echo "commit      : $(cd "$SGL" && git rev-parse HEAD 2>/dev/null)"
fi
echo

rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"
rc=0
for entry in "${FILES[@]}"; do
  rel="${entry%%|*}"; diff="${entry##*|}"
  src="$SGL/$rel"
  if [ ! -f "$src" ]; then
    echo "MISSING  $rel"; rc=1; continue
  fi
  mkdir -p "$SCRATCH/$(dirname "$rel")"
  # take the pristine version out of git when available, so a dirty tree does not
  # produce a false failure
  if [ -d "$SGL/.git" ] && (cd "$SGL" && git cat-file -e "HEAD:$rel" 2>/dev/null); then
    (cd "$SGL" && git show "HEAD:$rel") > "$SCRATCH/$rel"
  else
    cp "$src" "$SCRATCH/$rel"
  fi

  if (cd "$SCRATCH" && patch -p1 --fuzz=0 --silent < "$PATCHES/$diff") 2>/dev/null; then
    if python3 -m py_compile "$SCRATCH/$rel" 2>/dev/null; then
      echo "OK       $diff"
    else
      echo "COMPILE  $diff  -- applied but does not byte-compile"; rc=1
    fi
  else
    echo "FAIL     $diff  -- does not apply at fuzz=0"; rc=1
  fi
done

echo
# The fix's own identifiers must survive into the patched source -- a comment-only
# marker would not, and cannot be checked in bytecode either.
EW="$SCRATCH/python/sglang/srt/speculative/eagle_worker_v2.py"
if [ -f "$EW" ]; then
  echo "eagle_worker_v2 _needs_eager_local refs : $(grep -c _needs_eager_local "$EW")   (want 4)"
  echo "eagle_worker_v2 get_tp_group import     : $(grep -c 'get_tp_group as _get_tp_group' "$EW")   (want 1)"
fi

[ $rc -eq 0 ] && echo "ALL PATCHES VERIFIED" || echo "VERIFICATION FAILED"
exit $rc
