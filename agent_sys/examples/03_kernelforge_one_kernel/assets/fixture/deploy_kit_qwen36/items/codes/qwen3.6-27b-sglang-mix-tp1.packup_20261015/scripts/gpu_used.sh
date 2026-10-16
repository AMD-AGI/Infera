#!/usr/bin/env bash
# gpu_used.sh <comma-separated device indices>
# Prints "idx:MiB idx:MiB ..." of VRAM in use per named card, read from
# `rocm-smi --showmeminfo vram --json`. Exits NON-ZERO if any named card has no
# reading: an absent reading must never be mistaken for "free" or "drained".
set -u
DEVS="${1:?usage: gpu_used.sh <devices, e.g. 6 or 4,5,6,7>}"
rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import json, sys
devs = [d.strip() for d in sys.argv[1].split(",") if d.strip()]
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(2)
out = []
for d in devs:
    used = (data.get("card" + d) or {}).get("VRAM Total Used Memory (B)")
    if used is None:
        sys.exit(3)
    out.append("%s:%d" % (d, int(used) >> 20))
print(" ".join(out))
' "$DEVS"
