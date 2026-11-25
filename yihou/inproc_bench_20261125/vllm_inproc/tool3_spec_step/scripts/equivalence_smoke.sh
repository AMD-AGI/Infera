#!/usr/bin/env bash
# Criterion-1 evidence: the current bench WITHOUT --profile vs a pristine copy, identical args, same container.
# Port of the GLM-5.2 equivalence_smoke.sh. PRISTINE is a bench/ directory the container can see, e.g. the
# bench_snapshot/ of an earlier iteration (it lives under WORK_DIR, mounted at /out; PYTHONPATH supplies the kit's
# common/ for a snapshot that cannot reach it relative to itself):
#   PRISTINE=/out/iterations/<old point>/bench_snapshot bash equivalence_smoke.sh
# Then compares the deterministic fields (verify_iterations, realized_accept_length, accept_histogram,
# useful_output_tokens, graph counts) and the config_yihou.json key sets; prints EQUIVALENCE: ok|FAILED.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib_v3.sh"
PRISTINE=${PRISTINE:?set PRISTINE=/out/iterations/<point>/bench_snapshot (container path)}
TAG=${TAG:-eqsmoke}
ARGS=(--tp-size 1 --dp-size 2 --ep-size 1 --batch-size 32 --input-len 8192 --output-len 128
      --accept-length 3.51 --warmup-steps 2 --max-steps 20)
mapfile -t PRESET < <(v3_expand_args "$HERE/../presets/dsv41_llying_decode.args")
v3_check_container
run() {  # run <tag> <entrypoint dir>
    local out="$WORK_DIR/iterations/$1" ctr="/out/iterations/$1"
    [[ ! -e "$out" ]] || { echo "refusing existing $out" >&2; return 1; }
    mkdir -p "$out"
    local cmd
    printf -v cmd '%q ' python3 "$2/profile_decode.py" --model-path "$MODEL" --result-dir "$ctr" "${PRESET[@]}" "${ARGS[@]}"
    docker exec -w "$ctr" "$CONTAINER" bash -lc "PYTHONPATH=/kit/common $cmd > $ctr/runtime.log 2>&1"
    echo "$1 exit=$?"
}
run "${TAG}_noprofile_patched_yihou" "$TOOL_IN/bench"
run "${TAG}_noprofile_pristine_yihou" "$PRISTINE"
python3 - "$WORK_DIR/iterations/${TAG}_noprofile_patched_yihou" "$WORK_DIR/iterations/${TAG}_noprofile_pristine_yihou" <<'PY'
import json, sys
a, b = (json.load(open(f"{p}/result_yihou.json")) for p in sys.argv[1:3])
ca, cb = (json.load(open(f"{p}/config_yihou.json")) for p in sys.argv[1:3])
keys = ("verify_iterations", "realized_accept_length", "accept_histogram", "useful_output_tokens",
        "raw_accept_tokens", "graph_execution_counts", "target_graph_iterations")
diff = {k: (a.get(k), b.get(k)) for k in keys if a.get(k) != b.get(k)}
if set(ca["benchmark"]) != set(cb["benchmark"]):
    diff["config_keys"] = sorted(set(ca["benchmark"]) ^ set(cb["benchmark"]))
tpot = (a["effective_token_latency_ms_per_user"], b["effective_token_latency_ms_per_user"])
print(json.dumps({"tpot_ms_patched_vs_pristine": tpot, "differences": diff}, indent=2))
print("EQUIVALENCE: " + ("ok" if not diff else "FAILED"))
sys.exit(1 if diff else 0)
PY
