#!/usr/bin/env bash
# Source the B4 crsuse2 config into the environment and run one run_b4.py stage.
# Usage: [B4_PREFILL_NODE=<p>] [B4_DECODE_NODE=<d>] [B4_RUN_ID=<id>] run_b4.sh STAGE
#   STAGE: all | prepare | launch | gate | switch | preflight | measure | stop
#   Stages after prepare need the B4_RUN_ID that prepare printed.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export CONFIG="${CONFIG:-$HERE/../config/b4.crsuse2.sh}"
set -a
source "$CONFIG"
set +a
export B4_RUN_ID="$RUN_ID"
echo "B4_RUN_ID=$B4_RUN_ID"
exec python3 "$HERE/run_b4.py" "$@"
