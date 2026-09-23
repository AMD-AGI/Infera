#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/p8d8-chunk8k-31625-20260923
set -a
source "$ROOT/config/config.sh"
set +a
cp "$RUN/chunk8k-config-validation.json" "$RUN/chunk8k-config-validation.before-dynamic-port-fix.json"
cp "$ROOT/scripts/validate_chunk8k.py" "$RUN/snapshot/validate_chunk8k.after-port-fix.py"
python3 "$ROOT/scripts/validate_allocation.py"
python3 "$ROOT/scripts/validate_chunk8k.py"
python3 "$ROOT/scripts/smoke.py" --output "$RUN/smoke.json"
bash "$ROOT/scripts/capture_live.sh"
python3 "$ROOT/scripts/validate_smoke.py" "$RUN"
exec bash "$ROOT/scripts/run_chunk8k.sh" --resume-after-smoke
