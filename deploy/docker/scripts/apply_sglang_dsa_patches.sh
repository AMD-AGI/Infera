#!/usr/bin/env bash
# Apply the GLM-5.2 DSA patch set to the sglang checkout in an engine image, and
# prove every patch reached the BYTECODE.
#
# Used by Dockerfile.sglang and Dockerfile.sglang.gfx942 at build time, and
# runnable by hand inside a container for iteration.
#
# TWO ARMS, because the set is not uniformly portable across our engine bases:
#
#   DSA_PATCH_SET=full     (default; Dockerfile.sglang, mi35x / v0.5.15.post1)
#         patch 01 + 02 + 04.  02 and 04 are `--fuzz=0` context diffs pinned to
#         that one release, so this arm only works on that base.
#   DSA_PATCH_SET=indexer  (Dockerfile.sglang.gfx942, mi30x / v0.5.16)
#         patch 01 only.  It is an anchor script, not a pinned diff, and its two
#         edit sites are byte-identical on both releases.  02b and 04 are instead
#         substituted at RUNTIME on this base by
#         `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`
#         -- see patches/sglang_dsa/README.md; the gfx942 recipe MUST pass it.
#         02a has no substitute and is not carried here: its diff does not apply
#         to v0.5.16 and the fix has not been re-cut or measured on that base.
#
# DEPENDENCY (full arm only): the GLM-5.2 nextn eh_proj quark-exclude fix is NOT
# in this set -- deploy/docker/patches/sglang/patch_glm52_nextn_quark_exclude.py
# already carries it, and Dockerfile.sglang runs that loop BEFORE this script.
# It is a hard prerequisite (without it GLM-5.2 MTP dies at draft weight-load
# with a 3072-vs-6144 shape error), so we ASSERT it below rather than assume it.
# The indexer arm does not: v0.5.16 carries sglang#30265 upstream, which is why
# Dockerfile.sglang.gfx942 does not run patches/sglang/ at all.
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
# from each other and from patch 01, except that 02 carries 2a and 2b in one diff.
PATCHES=(
  dsa_backend_dp_sync_and_page_table_rows.diff
  draft_cuda_graph_dp_vote.diff
)

# module basename : identifier that must be present in the compiled bytecode
MARKERS=("dsa_indexer.py:_p1v2_trim")
if [ "$DSA_PATCH_SET" = "full" ]; then
  MARKERS+=(
    "dsa_backend.py:_glm52_match_page_table_rows"
    "dp_attn.py:can_draft_cuda_graph"
    "eagle_worker_v2.py:requires_dp_attention_eager_forward"
    "eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph"
    "forward_batch_info.py:can_run_dp_draft_cuda_graph"
    "schedule_batch.py:force_disable_draft_cuda_graph"
    "decode.py:force_disable_draft_cuda_graph"
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

if [ "$DSA_PATCH_SET" = "full" ]; then
  # PREREQUISITE, not one of our patches: patch_glm52_nextn_quark_exclude.py (run
  # earlier by Dockerfile.sglang) must have made the nextn eh_proj edit.  Asserted
  # here because GLM-5.2 MTP cannot load its draft weights without it, and because
  # a silent "skipped" from that idempotent script would otherwise go unnoticed
  # until the engine crashed at runtime.  The marker is a literal inside an
  # f-string, so this is a SOURCE check -- the f-string is split across bytecode
  # constants and does not appear whole in the .pyc.
  n3=$(grep -c 'num_hidden_layers}.eh_proj' "$SRT/models/deepseek_nextn.py" || true)
  echo "  PREREQ nextn eh_proj      -> src=$n3 (want 1)"
  [ "$n3" -eq 1 ] || { echo "  ^ run deploy/docker/patches/sglang/ first" >&2; fail=1; }

  # Patch 2a changes an expression and introduces no new identifier, and the same
  # expression already appears on two pre-existing graph-capture paths -- so
  # counting it gives 3, not 1.  Key off the unique comment the patch adds.  This
  # is a SOURCE check by necessity; bytecode cannot show it.
  n2a=$(grep -cF 'GLM52_BUG2A: BEFORE `seq_lens.max().item()`' \
        "$SRT/layers/attention/dsa_backend.py" || true)
  echo "  patch2a max_seqlen_k      -> src=$n2a (want 1)"
  [ "$n2a" -eq 1 ] || fail=1
fi

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
