#!/usr/bin/env bash
set -Eeuo pipefail
set -a; source "${CONFIG:?}"; set +a
out="$RUN/client-smoke"
mkdir -p "$out"
python3 "$BENCH_DIR/tools/agentx_env.py" --topology "$TOPOLOGY" --router-url "http://$PREFILL_IP:$ROUTER_PORT" --output-dir "$out" --runtime-dir "$AGENTX_CACHE_DIR/aiperf-r1r4-31719-client-smoke" --hf-home "$AGENTX_CACHE_DIR/hf" --concurrency 4 --duration 60 --ssh-options "$SSH_OPTS"
printf 'UV_CONSTRAINT=%s\n' "$AGENTX_CLIENT_CONSTRAINTS" >> "$out/runtime.env"
docker run --rm --name "$CONTAINER_PREFIX-agentx-client-smoke" --network host --ipc host -v /perf_apps:/perf_apps --env-file "$out/runtime.env" "$IMAGE" bash -c '
set -euo pipefail
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/benchmark_lib.sh"
if [[ -x "$AIPERF_PYTHON" ]] && "$AIPERF_PYTHON" -c "import aiperf, datasets, huggingface_hub"; then AIPERF_DEPS_READY=1; fi
        install_agentic_deps
"$AIPERF_PYTHON" -m aiperf profile --model "$SERVED_MODEL_NAME" --tokenizer "$MODEL" --url "$AIPERF_SERVER_URL" --endpoint-type chat --streaming --concurrency 4 --request-count 24 --synthetic-input-tokens-mean 512 --synthetic-input-tokens-stddev 0 --output-tokens-mean 32 --output-tokens-stddev 0 --use-server-token-count --no-gpu-telemetry --output-artifact-dir "$AGENTIC_OUTPUT_DIR/aiperf_artifacts"
chown -R "$HOST_UID:$HOST_GID" "$AGENTIC_OUTPUT_DIR"
' > "$out/runner.log" 2>&1
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=[json.loads(x) for x in (p/'aiperf_artifacts/profile_export.jsonl').read_text().splitlines()]
report={'requests':len(rows),'errors':[r.get('error') for r in rows if r.get('error')],'scope':'24-request synthetic AIPerf functional smoke, not formal workload performance'}
(p/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
assert len(rows)==24 and not report['errors']
PY
