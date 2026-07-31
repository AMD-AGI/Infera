#!/bin/sh
# Harvest the compiled pieces of the overlay out of an image that has them, into
# /native/rocm<major>-py<minor>/, and record which capabilities that tree carries.
#
# Run once per ABI family. The families are (ROCm major, CPython minor) and the
# vendor bases disagree: vLLM ships 3.12, SGLang 3.10. Neither Mooncake nor
# hipFile can be shared across them, so each needs its own harvest from an image
# built for that family.
#
# Not every family carries every capability, and that is not a defect:
#
#   mooncake   PD KV transport            both families
#   hipfile    kvd GPU-direct L3          vLLM only, by design — the SGLang kvd
#              (loads L3 into VRAM)       route goes through host memory, so its
#                                         images never build hipFile
#
# So the tree writes a CAPABILITIES file naming what it actually has, and
# infera-exec checks the capability a deployment asked for rather than guessing
# from the directory's existence. A tree with mooncake and no hipfile is a
# perfectly good SGLang tree; treating it as broken blocks working deployments.
set -eu

ROCM_MAJOR="$(readlink -f /opt/rocm/lib/libamdhip64.so.* 2>/dev/null \
              | grep -oE 'so\.[0-9]+' | head -1 | cut -d. -f2)"
PYTAG="py$(python3 -c 'import sys;print("%d%d"%sys.version_info[:2])')"
: "${ROCM_MAJOR:?cannot determine ROCm major — is this a ROCm image?}"

OUT="/native/rocm${ROCM_MAJOR}-${PYTAG}"
mkdir -p "$OUT/lib" "$OUT/bin"
echo "harvest: $OUT"
CAPS=""

# ---- Mooncake ---------------------------------------------------------------
# The extension pulls ~40 system libraries (glog, jsoncpp, asio, gflags, ...)
# that a stock vendor image does not ship. Bundling them next to the .so is what
# lets the overlay work off-base; infera-exec points LD_LIBRARY_PATH at them.
MC="$(python3 -c 'import mooncake,os;print(os.path.dirname(mooncake.__file__))' 2>/dev/null || true)"
if [ -n "$MC" ]; then
    cp -a "$MC" "$OUT/mooncake"
    ldd "$MC"/engine*.so 2>/dev/null | awk '/=> \//{print $3}' \
      | grep -vE 'libamdhip64|libc\.so|libm\.so|libstdc\+\+|libgcc_s|ld-linux|libpthread|libdl|librt|libnuma|libibverbs' \
      | while read -r so; do cp -L "$so" "$OUT/lib/" 2>/dev/null || true; done
    CAPS="${CAPS}mooncake "
    echo "harvest:   mooncake <- $MC"
else
    echo "harvest:   mooncake ABSENT in this image"
fi

# ---- hipFile ----------------------------------------------------------------
# libhipfile plus the ais-check probe. A missing ais-check silently downgrades
# kvd's load path to a CPU bounce, so the capability is only claimed when the
# library, the probe AND the Python module are all present.
HF="$(python3 -c 'import hipfile,os;print(os.path.dirname(hipfile.__file__))' 2>/dev/null || true)"
HAVE_LIB=0
for f in /opt/rocm/lib/libhipfile.so*; do
    [ -e "$f" ] || continue
    cp -L "$f" "$OUT/lib/" 2>/dev/null && HAVE_LIB=1
done
cp /opt/rocm/bin/ais-check "$OUT/bin/" 2>/dev/null || true
[ -n "$HF" ] && cp -a "$HF" "$OUT/hipfile" || true
ln -sf libhipfile.so.0.3.0 "$OUT/lib/libhipfile.so.0" 2>/dev/null || true
ln -sf libhipfile.so.0 "$OUT/lib/libhipfile.so" 2>/dev/null || true

if [ "$HAVE_LIB" = 1 ] && [ -n "$HF" ] && [ -x "$OUT/bin/ais-check" ]; then
    CAPS="${CAPS}hipfile "
    echo "harvest:   hipfile <- $HF"
else
    echo "harvest:   hipfile ABSENT (expected on SGLang images — GPU-direct L3 is vLLM-only)"
fi

# One space-separated line — infera-exec word-matches against it.
echo $CAPS > "$OUT/CAPABILITIES"
echo "harvest: capabilities = [$(echo $CAPS)]"
du -sh "$OUT"

# A tree with neither capability is not worth shipping and means the harvest
# image was wrong — fail the build rather than emit an empty directory that
# infera-exec would later report as a missing payload.
[ -n "$CAPS" ] || { echo "harvest: FATAL — no capabilities found" >&2; exit 1; }
