#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "${CONFIG:-$ROOT/config/config.sh}"
set +a
python3 "$ROOT/scripts/capture_window.py"
