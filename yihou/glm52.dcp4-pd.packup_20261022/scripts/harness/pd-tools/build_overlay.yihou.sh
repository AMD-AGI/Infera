#!/usr/bin/env bash
# Build the PD sglang overlay: a COPY of a synced branch tree with the Infera v0521/v0522 image
# recipe patches + the two kit runtime fixes re-applied, because the overlay mount shadows the
# patched /sgl-workspace/sglang/python/sglang of the image.
# Run INSIDE the PD image (python3 + patch present), CPU only:
#   docker run --rm --network none -v $PD:$PD -v $SRCROOT:$SRCROOT:ro --entrypoint bash IMAGE \
#     $PD/tools/build_overlay.yihou.sh SRC_PY_SGLANG OUT_DIR
# SRC_PY_SGLANG: .../src/sglang-<sha>/python/sglang  (read only)
# OUT_DIR      : new dir under a yihou path; must not exist. Result: OUT_DIR/python/sglang.
set -euo pipefail
SRC="${1:?src python/sglang dir}"; OUT="${2:?out dir}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
RP="${RECIPE_PATCHES:-/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/build/infera-src.yihou/deploy/docker/patches}"
case "$OUT" in *yihou*) ;; *) echo "OUT must be under a yihou path" >&2; exit 2;; esac
[[ -e "$OUT" ]] && { echo "OUT exists: $OUT (pick a new name)" >&2; exit 2; }
[[ -f "$SRC/__init__.py" ]] || { echo "not a sglang package dir: $SRC" >&2; exit 2; }
mkdir -p "$OUT/python"
cp -a "$SRC" "$OUT/python/sglang"
find "$OUT/python/sglang" -name __pycache__ -prune -exec rm -rf {} +
export PYTHONPATH="$OUT/python"
python3 - <<'PY'
import importlib.util, os
o = importlib.util.find_spec("sglang").origin
assert o.startswith(os.environ["PYTHONPATH"]), o
print("find_spec(sglang) ->", o)
PY
cd "$OUT"   # patch -p1 paths are python/sglang/...
echo "== recipe patches ($RP)"
python3 "$RP/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py"
for d in sglang_dsa/dsa_dp_sync.diff sglang_dsa/v0521/dsa_page_table_rows.diff; do
  patch -p1 --fuzz=0 --batch --forward --dry-run < "$RP/$d" >/dev/null ||
    { echo "$d does not apply at fuzz0" >&2; exit 1; }
  patch -p1 --fuzz=0 --batch --forward --no-backup-if-mismatch < "$RP/$d"
done
python3 "$RP/sglang_dsa/v0521/patch_glm_nextn_fused_shared_experts_arch.py"
python3 "$RP/sglang_disagg/v0521/patch_prefill_nonfinal_chunk_wait_event.py"
python3 "$RP/sglang_disagg/v0519/patch_pd_disable_implicit_rocm_rejection_sampling.py"
python3 "$RP/sglang_responses/v0519/patch_responses_unstreamed_tool_args.py"
python3 "$RP/sglang_rocm/patch_hicache_rocm_staged_write_back.py"
echo "== kit runtime fixes (R001 noblock, R003 noabsorbcfg)"
S=python/sglang/srt
patch --fuzz=0 --batch --forward --no-backup-if-mismatch "$S/managers/scheduler_components/kv_events_publisher.py" < "$TOOLS/overlay-patches/kv_events_publisher.noblock.diff"
patch --fuzz=0 --batch --forward --no-backup-if-mismatch "$S/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py" < "$TOOLS/overlay-patches/forward_mla_rocm.noabsorbcfg.diff"
find "$OUT/python" \( -name '*.rej' -o -name '*.orig' \) -print -delete
echo "== byte-compile check (no pyc kept)"
python3 - "$OUT/python/sglang" <<'PY'
import os, sys, py_compile, tempfile
root = sys.argv[1]; tmp = tempfile.mkdtemp(); n = 0
for d, _, fs in os.walk(root):
    for f in fs:
        if f.endswith(".py"):
            py_compile.compile(os.path.join(d, f), cfile=os.path.join(tmp, "x.pyc"), doraise=True); n += 1
print("compiled", n, "files OK")
PY
find "$OUT/python" -name __pycache__ -prune -exec rm -rf {} +
{ echo "src=$SRC"; echo "recipe=$RP"; echo "built=$(date -u +%FT%TZ)"; } > "$OUT/OVERLAY.yihou.txt"
(cd "$OUT/python" && diff -ru "$SRC" sglang > "$OUT/overlay-vs-src.yihou.diff" || true)
echo "diff vs src: $(grep -c '^+++ ' "$OUT/overlay-vs-src.yihou.diff") files -> $OUT/overlay-vs-src.yihou.diff"
echo "OVERLAY OK: $OUT/python/sglang"
