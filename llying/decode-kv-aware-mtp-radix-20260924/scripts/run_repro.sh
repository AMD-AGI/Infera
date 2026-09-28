#!/usr/bin/env bash
# Purpose: CPU-only check of the decode radix-cache decision, before and after
#   the SGLang/Infera patches, inside the GLM-5.2 image.
# Usage: ./run_repro.sh [NODE]   (default crsuse2-m2m-138)
# Artifacts: analysis/evidence/repro-{unpatched,patched}.txt
# Starts no engine, passes no GPU device, and uses --network none. The patched
#   run applies the patches to a throwaway container only; the image is unchanged.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$DIR/../.." && pwd)"
NODE="${1:-crsuse2-m2m-138}"
IMAGE="${IMAGE:-infera-sglang:v0519-yihou-0917-nextnfix-hicache}"
MODEL="${MODEL:-/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4}"
EVIDENCE="$DIR/analysis/evidence"
mkdir -p "$EVIDENCE"

run() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
        docker run --rm --network none --entrypoint bash \
        -v "$MODEL:$MODEL:ro" -v "$DIR:/task:ro" -v "$REPO:/repo:ro" \
        -e PYTHONNOUSERSITE=1 -e "MODEL=$MODEL" "$IMAGE" -c "$(printf '%q' "$1")"
}

run 'for c in current forced; do python3 /task/scripts/repro_decode_args.py $c 2>&1 | grep -E "RESULT|infera.engine.sglang.args|pd_disaggregation_hook"; done' \
    | tee "$EVIDENCE/repro-unpatched.txt"

run 'set -e
cd /sgl-workspace/sglang && git apply /task/patches/01-sglang-decode-radix-allow-eagle.patch
cp -r /repo/infera /repo/tests /tmp/ && cd /tmp && git apply /task/patches/02-infera-decode-radix-spec-opt-in.patch
cd /tmp && export PYTHONPATH=/tmp
python3 /task/scripts/repro_decode_args.py patched 2>&1 | grep -E "RESULT|infera.engine.sglang.args|pd_disaggregation_hook"
unset SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC
python3 /task/scripts/repro_decode_args.py current 2>&1 | grep -E "RESULT|infera.engine.sglang.args|pd_disaggregation_hook"
python3 -m pytest -q -p no:cacheprovider tests/engine/sglang/test_decode_radix_cache_guard.py \
    -k "speculative or opted or opt_in or appends_for or skips_for or explicit or prefill_leg_is_untouched or kv_events_off" 2>&1 | tail -5' \
    | tee "$EVIDENCE/repro-patched.txt"
