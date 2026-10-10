#!/usr/bin/env bash
# Write runtime/docker/decode_prefix_diag.py: <image>'s disaggregation/decode.py
# plus the aus decode-prefix diagnostic, which engine.sh mounts over decode.py
# (DECODE_DIAG_SOURCE). Run on a node that has <image>.
# Usage: make_decode_diag.sh <image>
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$HERE/../router-capacity-campaign-20260928/build/decode-prefix-diagnostic.patch"
OUT="$HERE/runtime/docker/decode_prefix_diag.py"
mkdir -p "$(dirname "$OUT")"
docker run --rm --entrypoint cat "$1" \
    /sgl-workspace/sglang/python/sglang/srt/disaggregation/decode.py > "$OUT.tmp"
patch --fuzz=0 --batch "$OUT.tmp" < "$PATCH"
python3 -m py_compile "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
echo "$OUT $(sha256sum < "$OUT" | cut -d' ' -f1)"
