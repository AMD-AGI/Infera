#!/usr/bin/env bash
# Apply the GLM-5.2 DSA patch set to the sglang checkout in an engine image, and
# prove every patch reached the BYTECODE.
#
# Used by Dockerfile.sglang and Dockerfile.sglang.gfx942 at build time, and
# runnable by hand inside a container for iteration.
#
# TWO ARMS, because the set is not uniformly portable across our engine bases:
#
#   DSA_PATCH_SET=full     (default; Dockerfile.sglang, mi35x / v0.5.17)
#         patch 01 + dsa_dp_sync + dsa_page_table_rows + draft_cuda_graph_dp_vote.
#         The three diffs are `--fuzz=0` against that one release, so this arm
#         only works there.
#   DSA_PATCH_SET=indexer  (Dockerfile.sglang.gfx942, mi30x / v0.5.16)
#         patch 01 only -- an anchor script, not a pinned diff, which is what
#         lets both bases share it.  Anchor coverage per base: patch 01's header.
#
#         dsa_page_table_rows and draft_cuda_graph_dp_vote are instead substituted
#         at RUNTIME on that base by
#         `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`
#         -- see patches/sglang_dsa/README.md; the gfx942 recipe MUST pass it.
#         dsa_dp_sync has no substitute and is not carried there: it has not been
#         re-cut or measured on v0.5.16.
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
DSA_PATCH_SET="${DSA_PATCH_SET:-full}"

case "$DSA_PATCH_SET" in
  full|indexer) ;;
  *) echo "DSA_PATCH_SET must be 'full' or 'indexer', got '$DSA_PATCH_SET'" >&2
     exit 1 ;;
esac

# Patch 01 is a self-locating anchor script rather than a pinned diff, precisely
# so both bases can share it.  It is idempotent and all-or-nothing, and exits
# non-zero only when its anchors drifted.
PATCH01=patch_dsa_indexer_hip_dp_padded_rows.py

# Applied in this order.  Order is not significant -- these touch files disjoint
# from each other and from patch 01.  dsa_dp_sync.diff is upstream PR sglang#33973
# verbatim, so it drops by deleting the file the day that merges.
PATCHES=(
  dsa_dp_sync.diff
  dsa_page_table_rows.diff
  draft_cuda_graph_dp_vote.diff
)

# module basename : identifier that must be present in the compiled bytecode
#
# Patch 01 needs TWO markers: `_p1v2_trim` alone is also satisfied by the earlier
# one-directional revision; only `_p1v2_rows` proves real > padded is handled.
MARKERS=("dsa_indexer.py:_p1v2_trim" "dsa_indexer.py:_p1v2_rows")
if [ "$DSA_PATCH_SET" = "full" ]; then
  # dsa_dp_sync introduces no identifier, but it does rewrite an assert message,
  # and a string constant survives into the .pyc. Its own substitution
  # (`self.req_to_token.shape[1]`) is NOT usable: two pre-existing graph-capture
  # paths already spell it, so a count cannot tell them apart.
  # draft_cuda_graph_dp_vote spans seven files; these two pin its ends -- the
  # scheduler side that feeds the vote and the gate that consumes it.  Either one
  # missing makes the patch inert rather than absent.
  MARKERS+=(
    "dsa_backend.py:must not be None for DRAFT_EXTEND_V2"
    "dsa_backend.py:_glm52_match_page_table_rows"
    "decode.py:force_disable_draft_cuda_graph"
    "eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph"
  )
fi

echo "=== sglang DSA patches ($DSA_PATCH_SET): applying to $SGLANG_DIR ==="
cd "$SGLANG_DIR"
echo "  --- $PATCH01"
python3 "$PATCH_DIR/$PATCH01"
if [ "$DSA_PATCH_SET" = "full" ]; then
  for p in "${PATCHES[@]}"; do
    echo "  --- $p"
    # --fuzz=0: these target a pinned sglang commit.  A fuzzy apply that "succeeds"
    # against a different base is worse than a clean failure.
    patch -p1 --fuzz=0 < "$PATCH_DIR/$p"
  done
fi

echo "=== dropping __pycache__ so verification cannot read a stale .pyc ==="
find "$SGLANG_DIR/python/sglang/srt" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "=== verifying every patch reached the bytecode ==="
fail=0

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
