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
#         patch 01 only.  02b is substituted at RUNTIME by
#         `--json-model-override-args '{"index_share_for_mtp_iteration":false}'`
#         -- see patches/sglang_dsa/README.md; the gfx942 recipe MUST pass it.
#
# A THIRD ARM `gfx942` (= 01 + 02a + 04) was added and then REMOVED.  Worth
# knowing why, because the reasoning that produced it was self-consistent and
# still wrong:
#
#   Three GPU faults on a 2 x 8 MI300X 1P1D deployment were each traced to a
#   plausible sglang cause -- a missing 02a on the draft-extend path, a missing
#   04 on the draft-cuda-graph vote, then a prefill kernel on multi-sequence
#   batches -- and 02a and 04 were wired in on that basis.  The real cause was
#   the ENVIRONMENT: that cluster runs amdgpu 6.3.x, whose supported ROCm
#   userspace ceiling is 7.0.x, against a 7.2.0 base image.  Rebuilt on
#   `v0.5.16-rocm700-mi30x` with patch 01 ALONE and nothing else changed, the
#   same deployment ran EAGLE MTP(5,1,6) on both legs clean -- warmup, conc 1,
#   and conc 8/16/32.  Neither 02a nor 04 was needed at any point.
#
#   The lesson generalises past this repo: a single environment mismatch can
#   present as several unrelated code bugs, each with a convincing stack, and
#   patches that "fix" it may only be shifting timing.  Before attributing a
#   `Memory access fault by GPU node-N` to sglang, check the host amdgpu version
#   against the container's /opt/rocm/.info/version.  That check is cheaper than
#   one image rebuild; three of them were spent here instead.
#
#   Kept from that work because it is independently correct:
#     - the scoped-rejection logic in the apply loop below (a non-zero `patch`
#       exit is tolerated only when the FAILED files match an arm's declared
#       expectation exactly).  No arm declares one today, so any rejection now
#       fails the build.
#     - patches/sglang_dsa/patch_draft_cuda_graph_dp_vote_v0516.py, which ports
#       04's dp_attn.py hunks to v0.5.16 by anchor (the diff's 7/7 hunks fail
#       there: v0.5.16 renamed the two neighbouring gates).  Verified in-image,
#       wired into no arm.  If 04 is ever genuinely needed on v0.5.16, it does
#       not have to be rewritten.
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
  gfx942) echo "DSA_PATCH_SET=gfx942 was removed: it added 02a + 04 on a misattributed" >&2
          echo "environment failure (see the header). Use 'indexer' -- patch 01 only." >&2
          exit 1 ;;
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
case "$DSA_PATCH_SET" in
  full)    PATCHES=(dsa_dp_sync.diff dsa_page_table_rows.diff draft_cuda_graph_dp_vote.diff) ;;
  indexer) PATCHES=() ;;
esac

# Per-arm escape hatch for a diff that is expected to reject ONE named file
# because an anchor script carries it instead.  Both are deliberately EMPTY: no
# arm needs this today.  The mechanism stays because it is the only safe way to
# express that tolerance -- scoped to a named file, so any OTHER rejection still
# fails the build.  An unset expectation waves nothing through; see the loop.
declare -A EXPECT_REJECT=()
PORT_SCRIPTS=()

# module basename : identifier that must be present in the compiled bytecode
#
# Patch 01 needs TWO markers: `_p1v2_trim` alone is also satisfied by the earlier
# one-directional revision; only `_p1v2_rows` proves real > padded is handled.
MARKERS=("dsa_indexer.py:_p1v2_trim" "dsa_indexer.py:_p1v2_rows")
if [ "$DSA_PATCH_SET" = "full" ]; then
  # 02a introduces no identifier, but it does rewrite an assert message, and a
  # string constant survives into the .pyc. Its own substitution
  # (`self.req_to_token.shape[1]`) is NOT usable: two pre-existing graph-capture
  # paths already spell it, so a count cannot tell them apart.
  MARKERS+=("dsa_backend.py:must not be None for DRAFT_EXTEND_V2")
  MARKERS+=("dsa_backend.py:_glm52_match_page_table_rows")
  # 04 spans seven files; these pin its ends -- the scheduler side that feeds the
  # vote, the collective that carries it, and the gate that consumes it.  Any one
  # missing makes the patch inert rather than absent, which is the failure mode
  # that matters: an inert 04 looks exactly like a working one until load.
  MARKERS+=(
    "decode.py:force_disable_draft_cuda_graph"
    "dp_attn.py:can_run_draft_cuda_graph"
    "eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph"
  )
fi

echo "=== sglang DSA patches ($DSA_PATCH_SET): applying to $SGLANG_DIR ==="
cd "$SGLANG_DIR"
echo "  --- $PATCH01"
python3 "$PATCH_DIR/$PATCH01"
for p in ${PATCHES[@]+"${PATCHES[@]}"}; do
  echo "  --- $p"
  # --fuzz=0: these target a pinned sglang commit.  A fuzzy apply that "succeeds"
  # against a different base is worse than a clean failure.  An OFFSET is not
  # fuzz: it means the hunk matched byte-for-byte somewhere else in the file,
  # which is why 02a carries to v0.5.16 while 04's dp_attn.py does not.
  # --batch so a "Reversed (or previously applied) patch detected!" prompt fails
  # instead of hanging a build waiting on stdin.
  out=$(patch -p1 --fuzz=0 --batch < "$PATCH_DIR/$p" 2>&1) && rc=0 || rc=$?
  echo "$out" | sed 's/^/      /'
  [ "$rc" -eq 0 ] && continue

  # Non-zero is allowed only when the FAILED files are exactly the ones this arm
  # declares it carries by script instead.  An empty list means patch bailed for
  # some other reason -- "previously applied", a missing file, a malformed diff --
  # and must not be waved through by an unset expectation.
  want="${EXPECT_REJECT[$p]:-}"
  rejected=$(echo "$out" | awk '/^(patching|checking) file /{f=$3} /FAILED/{print f}' | sort -u)
  if [ -z "$rejected" ] || [ "$rejected" != "$want" ]; then
    echo "  $p: exit $rc, FAILED files were: ${rejected:-<none, see output above>}" >&2
    echo "  expected exactly: ${want:-<none — this patch must apply cleanly>}" >&2
    exit 1
  fi
  echo "      ($want rejected as expected — carried by port script below)"
  find "$SGLANG_DIR/python" \( -name '*.rej' -o -name '*.orig' \) -delete
done

for s in ${PORT_SCRIPTS[@]+"${PORT_SCRIPTS[@]}"}; do
  echo "  --- $s"
  python3 "$PATCH_DIR/$s"
done

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
