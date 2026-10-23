#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Apply the GLM-5.2 DSA patch set to the v0.5.22 base (Dockerfile.sglang) and
# prove every patch reached the BYTECODE.
#
# The v0.5.16-v0.5.18 driver (deploy/docker/scripts/apply_sglang_dsa_patches.sh)
# is left untouched for Dockerfile.sglang.gfx942. On this base the set is:
#
#   01   ../patch_dsa_indexer_hip_dp_padded_rows.py   unchanged, self-locating
#   02a  ../dsa_dp_sync.diff                          unchanged, applies at an
#                                                     OFFSET (not fuzz) here
#   02b  ./dsa_page_table_rows.diff                   RE-CUT for v0.5.22 anchors
#   04   draft_cuda_graph_dp_vote                     upstream (#32196): asserted
#        below, so a base that loses it fails here instead of deadlocking the
#        decode DP group at load
#   --   GLM NextN fused-shared-experts architecture  upstream since v0.5.22:
#        asserted below (the NextN draft would otherwise silently run unfused)
#
# --fuzz=0, probe both directions before mutating, drop __pycache__, verify
# identifiers (never comments) in fresh .pyc.
set -euo pipefail

SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"
PATCH_DIR="${PATCH_DIR:-/tmp/sglang_dsa}"
SRT="$SGLANG_DIR/python/sglang/srt"

PATCH01="$PATCH_DIR/patch_dsa_indexer_hip_dp_padded_rows.py"
DIFFS=("$PATCH_DIR/dsa_dp_sync.diff" "$PATCH_DIR/v0522/dsa_page_table_rows.diff")

# module basename : identifier that must be in the compiled bytecode
MARKERS=(
  "dsa_indexer.py:_p1v2_trim"
  "dsa_indexer.py:_p1v2_rows"
  "dsa_backend.py:must not be None for DRAFT_EXTEND_V2"
  "dsa_backend.py:_glm52_match_page_table_rows"
  # upstream replacement for 04 -- required, not applied
  "dp_attn.py:_spec_input_cuda_graph_compatible"
  "eagle_draft_cuda_graph_runner.py:can_run_dp_draft_cuda_graph"
)

echo "=== sglang DSA patches (v0522): applying to $SGLANG_DIR ==="
cd "$SGLANG_DIR"
echo "  --- $(basename "$PATCH01")"
python3 "$PATCH01"
for p in "${DIFFS[@]}"; do
  echo "  --- ${p#"$PATCH_DIR"/}"
  if patch -p1 --fuzz=0 --batch --forward --dry-run < "$p" >/dev/null 2>&1; then
    patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch < "$p" | sed 's/^/      /'
  elif patch -p1 -R --fuzz=0 --batch --dry-run < "$p" >/dev/null 2>&1; then
    echo "      already applied — skipping"
  else
    patch -p1 --fuzz=0 --batch --forward --dry-run < "$p" | sed 's/^/      /' || true
    echo "  ${p#"$PATCH_DIR"/}: does not apply at --fuzz=0 to this base" >&2
    exit 1
  fi
done
find "$SGLANG_DIR/python" \( -name '*.rej' -o -name '*.orig' \) -delete

echo "=== dropping __pycache__ so verification cannot read a stale .pyc ==="
find "$SRT" -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

echo "=== verifying bytecode ==="
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
  printf "  %-58s -> pyc=%s (want >0)\n" "$f :: $m" "$n"
  [ "$n" -gt 0 ] || fail=1
done
# The NextN attribute adds no new identifier (the class name and the attribute
# name both pre-exist in the module), so check the class BODY's code object.
glm="$SRT/models/glm4_moe.py"
rm -f "$(dirname "$glm")/__pycache__/glm4_moe."*.pyc
python3 - "$glm" <<'EOF' || fail=1
import marshal, py_compile, sys, types
pyc = py_compile.compile(sys.argv[1], doraise=True)
code = marshal.loads(open(pyc, "rb").read()[16:])
body = next(
    c for c in code.co_consts
    if isinstance(c, types.CodeType) and c.co_name == "GlmMoeDsaForCausalLMNextN"
)
ok = ("fused_shared_experts_architecture" in body.co_names
      and "GlmMoeDsaForCausalLMNextN" in body.co_consts)
print(f"  {'glm4_moe.py :: NextN class body sets fused_shared_experts_architecture':58s} -> {ok}")
sys.exit(0 if ok else 1)
EOF

if [ "$fail" -ne 0 ]; then
  echo "SGLANG DSA PATCH VERIFICATION FAILED" >&2
  exit 1
fi
echo "=== all v0522 sglang DSA patches verified in bytecode ==="
