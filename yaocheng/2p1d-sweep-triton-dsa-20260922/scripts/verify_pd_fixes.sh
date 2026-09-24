#!/usr/bin/env bash
# Inspect image contents on CPU; never attach GPU devices or modify existing containers.
source "$(dirname "$0")/common.sh"
VERIFY_OUT_DIR="${VERIFY_OUT_DIR:-$SWEEP_TMP_DIR/validation/pd-fixes-$(date -u +%Y%m%dT%H%M%SZ)}"
exec python3 -B "$SCRIPT_DIR/verify_pd_fixes.py" --output "$VERIFY_OUT_DIR"
