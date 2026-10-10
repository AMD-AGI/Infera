#!/usr/bin/env bash
# Write runtime/reference/b4: B4's own c80/runtime.env as the BASELINE_RUN that
# validate_and_pin_client.py / validate_chunk8k_runtime.py compare against.
# Only the four keys that hold aus filesystem paths outside those scripts'
# allow-list are rewritten to this cluster's paths; every other key must then
# match B4 exactly at run time.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
set -a
source "$HERE/config/b4.crsuse2.sh"
set +a
SRC="$HERE/../router-capacity-campaign-20260928/b4/runtime.env"
OUT="$TRACE_RUNTIME/reference/b4/c80"
mkdir -p "$OUT"
declare -A rewrite=(
    [MODEL]="$MODEL"
    [HF_HOME]="$AGENTX_CACHE_DIR/hf"
    [INFMAX_CONTAINER_WORKSPACE]="$INFERENCEX_DIR"
    [UV_CONSTRAINT]="$AGENTX_CLIENT_CONSTRAINTS"
)
while IFS= read -r line; do
    key="${line%%=*}"
    if [[ -n "${rewrite[$key]+x}" ]]; then
        echo "$key=${rewrite[$key]}"
    else
        echo "$line"
    fi
done < "$SRC" > "$OUT/runtime.env"
python3 - "$AGENTX_CLIENT_CONSTRAINTS" "$OUT/client-dependency-constraints.json" <<'EOF'
import hashlib, json, sys
path = sys.argv[1]
json.dump({"path": path, "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
           "scope": "B4 constraint file, committed copy"}, open(sys.argv[2], "w"), indent=2)
EOF
diff "$SRC" "$OUT/runtime.env" || true
