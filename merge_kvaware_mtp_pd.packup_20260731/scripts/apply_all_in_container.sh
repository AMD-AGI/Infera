#!/usr/bin/env bash
# Apply the FULL merged patch set inside a running engine container, and prove
# every piece reached the bytecode.
#
# This is the in-container prototype of what will become the Dockerfile layer:
# the same three scripts, in the same order, with the same verification.
#
#   1. sglang DSA patch set (3 diffs)   -- PD + DP-attention + EAGLE MTP on gfx950
#   2. mooncake early-send wait event   -- long-prompt chunked-prefill corruption
#   3. infera kv-event bigram flatten   -- kv-aware routing under MTP
#   4. infera decode-radix vs MTP gate  -- kvaware auto-append collides with EAGLE
#   5. infera decode-leg kvd skip       -- kvd is write-only on a PD decode leg
#
# ORDER MATTERS for (1): the GLM-5.2 nextn eh_proj quark-exclude fix is its hard
# prerequisite, and it is already baked into this image by the Dockerfile's
# patch loop -- apply_sglang_dsa_patches.sh ASSERTS it rather than assuming.
#
# Idempotent: re-running is a no-op that still re-verifies.
set -euo pipefail

PATCH_DIR="${PATCH_DIR:-/tmp/merge_patches}"
SGLANG_DIR="${SGLANG_DIR:-/sgl-workspace/sglang}"

echo "################ 1/5  sglang DSA patch set ################"
PATCH_DIR="$PATCH_DIR" SGLANG_DIR="$SGLANG_DIR" bash "$PATCH_DIR/apply_sglang_dsa_patches.sh"

echo
echo "################ 2/5  mooncake early-send wait event ################"
python3 "$PATCH_DIR/patch_mooncake_early_send_wait_event.py"

echo
echo "################ 3/5  infera kv-event bigram flatten ################"
python3 "$PATCH_DIR/patch_infera_kvevent_bigram.py"

echo
echo "################ 4/5  infera decode-radix vs speculative gate ################"
python3 "$PATCH_DIR/patch_infera_decode_radix_vs_mtp.py"

echo
echo "################ 5/5  infera decode-leg kvd skip ################"
python3 "$PATCH_DIR/patch_infera_decode_kvd_skip.py"

echo
echo "################ verification ################"
fail=0

# --- (2) mooncake wait event: three files, all or nothing ---------------------
SRT="$SGLANG_DIR/python/sglang/srt"
declare -A MC_MARKERS=(
  ["disaggregation/common/utils.py"]="wait_event"
  ["disaggregation/mooncake/conn.py"]="_early_send_wait_event"
  ["disaggregation/prefill.py"]="_early_send_wait_event"
)
for rel in "${!MC_MARKERS[@]}"; do
  m="${MC_MARKERS[$rel]}"
  p="$SRT/$rel"
  d=$(dirname "$p"); b=$(basename "$p" .py)
  rm -f "$d/__pycache__/$b."*.pyc
  python3 -c "import py_compile; py_compile.compile('$p', doraise=True)"
  pyc=$(ls "$d/__pycache__/$b."*.pyc 2>/dev/null | head -1)
  n=$(strings "$pyc" | grep -c "$m" || true)
  printf "  %-42s -> pyc=%s (want >0)\n" "$rel :: $m" "$n"
  [ "$n" -gt 0 ] || fail=1
done
# The synchronize() call is the actual barrier -- assert the source shows it in
# the transfer worker, since `synchronize` alone is too generic a bytecode hit.
n=$(grep -c 'kv_chunk.wait_event.synchronize()' "$SRT/disaggregation/mooncake/conn.py" || true)
echo "  mooncake transfer_worker synchronize()     -> src=$n (want 1)"
[ "$n" -eq 1 ] || fail=1

# --- (3) infera bigram --------------------------------------------------------
# BOTH copies: site-packages AND the image's /opt/infera source tree, which
# shadows it for any process whose cwd is the WORKDIR (i.e. every docker exec).
INFERA_ROOTS=$(python3 -c "
import importlib.util, os
roots, seen = [], set()
spec = importlib.util.find_spec('infera')
cand = [os.path.dirname(spec.origin)] if spec and spec.origin else []
cand.append('/opt/infera/infera')
for d in cand:
    d = os.path.realpath(d)
    if d not in seen and os.path.isfile(os.path.join(d, 'router/kv_event/client.py')):
        seen.add(d); roots.append(d)
print('\n'.join(roots))")
echo "  infera copies: $(echo "$INFERA_ROOTS" | tr '\n' ' ')"
while read -r INFERA_ROOT; do
  [ -n "$INFERA_ROOT" ] || continue
  p="$INFERA_ROOT/router/kv_event/client.py"
  d=$(dirname "$p"); b=$(basename "$p" .py)
  rm -f "$d/__pycache__/$b."*.pyc
  python3 -c "import py_compile; py_compile.compile('$p', doraise=True)"
  pyc=$(ls "$d/__pycache__/$b."*.pyc 2>/dev/null | head -1)
  n=$(strings "$pyc" | grep -c "_flat_tokens" || true)
  printf "  %-42s -> pyc=%s (want >0)\n" "${INFERA_ROOT}/...client.py" "$n"
  [ "$n" -gt 0 ] || fail=1
  # The msgspec field type is an annotation, evaluated lazily -- source check.
  n=$(grep -c 'token_ids: list\[int | tuple\[int, int\]\]' "$INFERA_ROOT/router/kv_event/events.py" || true)
  printf "  %-42s -> src=%s (want 1)\n" "${INFERA_ROOT}/...events.py" "$n"
  [ "$n" -eq 1 ] || fail=1
  # (4) the kvaware-gated decode-radix append must skip speculative decoding.
  n=$(grep -c 'incompatible with --speculative-algorithm' "$INFERA_ROOT/engine/sglang/args.py" || true)
  printf "  %-42s -> src=%s (want 1)\n" "${INFERA_ROOT}/...args.py spec gate" "$n"
  [ "$n" -eq 1 ] || fail=1
  # (5) kvd must not be wired on a PD decode leg.
  n=$(grep -c '_skip_kvd_on_decode_leg' "$INFERA_ROOT/engine/sglang/kvd_wiring.py" || true)
  printf "  %-42s -> src=%s (want 2)\n" "${INFERA_ROOT}/...kvd_wiring.py decode skip" "$n"
  [ "$n" -eq 2 ] || fail=1
done <<< "$INFERA_ROOTS"

# --- functional smoke: the flatten actually flattens ---------------------------
python3 - <<'PY'
from infera.router.kv_event.client import _flat_tokens
assert _flat_tokens([(1, 2), (2, 3), (3, 4)]) == [1, 2, 3], "bigram flatten wrong"
assert _flat_tokens([1, 2, 3]) == [1, 2, 3], "plain path changed"
assert _flat_tokens([]) == [], "empty path changed"
print("  _flat_tokens smoke                         -> OK")
PY

if [ "$fail" -ne 0 ]; then
  echo "MERGED PATCH VERIFICATION FAILED" >&2
  exit 1
fi
echo "=== all merged patches verified ==="
