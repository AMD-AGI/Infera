#!/usr/bin/env bash
# Unit tests for the vLLM fix (upstream/vllm-main, branch fix/mooncake-send-kv-on-stopped-prefill): MooncakeConnector's
# producer hands KV over on FINISHED_STOPPED as on FINISHED_LENGTH_CAPPED. Runs the branch's
# tests/v1/kv_connector/unit/test_mooncake_connector.py against the nightly image's vllm, first unpatched (the new
# STOPPED case should fail), then with the patched mooncake_connector.py mounted over the installed one. Test deps at
# the requirements/test/rocm.txt pins (21c failed on the missing tblib; 22b on a quoting error in this command).
set -uo pipefail
RUN=24-vllm-unit-$(date -u +%m%dT%H%M)
source /perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/scripts/lib.sh
REPO=/perf_apps/liyingli/bench_agentx/upstream/vllm-main
SITE=/usr/local/lib/python3.12/dist-packages
say "start on $(hostname -s), job $SLURM_JOB_ID, repo $(git -C "$REPO" rev-parse --short HEAD) + working tree"
ensure_images > "$R/pull.log" 2>&1 || { say "image pull failed"; exit 1; }
remove_ours
trap cleanup EXIT
run_tests() {  # run_tests LABEL [docker args...]
    local label=$1; shift
    docker run --rm --name "${CTR_PREFIX}unit" --network host --device /dev/kfd --device /dev/dri $(host_gids) --ipc host \
        --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 -v /perf_apps:/perf_apps -v "$NODE_CACHE:/root/.cache" \
        "$@" --entrypoint "" "$IMAGE" bash -c "
            python3 -m pip install -q anthropic==0.93.0 pytest==9.2.0 pytest-asyncio==1.4.0 pytest-forked==1.6.0 pytest-mock==3.15.1 pytest-rerunfailures==14.0 pytest-timeout==2.3.1 tblib==3.1.0 2>&1 | grep -v 'Running pip as' | tail -2
            mkdir -p /tmp/vt && cp -r $REPO/tests /tmp/vt/ && cd /tmp/vt
            python3 -c 'import vllm, os; print(\"vllm\", vllm.__version__, os.path.dirname(vllm.__file__))'
            grep -c FINISHED_STOPPED $SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
            python3 -m pytest -q -p no:cacheprovider tests/v1/kv_connector/unit/test_mooncake_connector.py -k request_finished
            python3 -m pytest -q -p no:cacheprovider tests/v1/kv_connector/unit/test_mooncake_connector.py
        " > "$R/$label.log" 2>&1
    say "$label: $(grep -aE '^[0-9]+ (passed|failed)|passed|failed|error' "$R/$label.log" | grep -aE ' in [0-9.]+s' | tr '\n' ' ')"
}
run_tests unpatched
run_tests patched -v "$W/vllm-overlay/mooncake_connector.py:$SITE/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py:ro"
say "end"
