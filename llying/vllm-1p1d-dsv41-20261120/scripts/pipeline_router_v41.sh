#!/usr/bin/env bash
# Holder item (job 36739): router DSv4.1 encoder fix -> checks -> image -> 1P1D with the kv-aware gate -> c768.
#   1. cargo fmt / clippy -D warnings / test in rust:1-bookworm via router_check_inner.sh, as this user (fmt fixes
#      are committed); stop on failure.
#   2. build the engine image at the branch head.
#   3. run_1p1d_v3.sh: B config + serial dispatch, smoke, accuracy gate, kv-aware check (render parity + cache_hits on
#      two-turn probes), golden 3.51, formal AgentX c768.
# Ends by touching the holder's STOP so the job releases the node. Not from an EXIT trap: a preempted item is
# requeued by the holder and must run again on the next node.
set -uo pipefail
source /perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/scripts/env.sh
H=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder
REPO=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
L=$W/logs/pipeline-v41-$(date -u +%m%dT%H%M)
mkdir -p "$L" && chmod 777 "$L"
say() { echo "$(ts) pipeline-v41: $*" | tee -a "$W/logs/summary.log"; }
finish() { say "$1; touching STOP so job ${SLURM_JOB_ID:-?} releases $(hostname -s)"; touch "$H/STOP"; exit "${2:-0}"; }

say "start on $(hostname -s) job ${SLURM_JOB_ID:-?}, amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null), branch $(git -C "$REPO" rev-parse --short HEAD)"
for img in "$BASE_IMAGE" docker.io/library/rust:1-bookworm "$ETCD_IMAGE"; do
    pull_image "$img" >> "$L/pull.log" 2>&1 || finish "pull of $img FAILED" 1
done

cd "$REPO"
docker run --rm -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -v "$REPO/rust:/work/rust" \
    -v "$W/scripts/router_check_inner.sh:/inner.sh:ro" docker.io/library/rust:1-bookworm bash /inner.sh \
    > "$L/router-test.log" 2>&1
rc=$?
say "router checks rc=$rc: $(grep -aE '^test result|error(\[|:)|warning: unused' "$L/router-test.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';' | head -c 500)"
((rc)) && finish "router checks FAILED ($L/router-test.log)" 1
if ! git diff --quiet -- rust; then
    git add rust && git -c gc.auto=0 commit -q -s --trailer "Co-authored-by: Cursor <cursoragent@cursor.com>" -m "style(router): cargo fmt"
    say "committed cargo fmt changes: $(git log --oneline -1)"
fi

bash examples/vllm_1p1d_dsv41_flash/build_image.sh > "$L/build.log" 2>&1 || finish "image build FAILED: $(tail -3 "$L/build.log" | tr '\n' ' ')" 1
IMG=$(grep -ao "infera-vllm-dsv41:[0-9a-f]*-nightly-[0-9a-f]*" "$L/build.log" | tail -1)
say "image $IMG"

env INFERA_IMAGE="$IMG" CONC=768 GATE=1 UMBP_MODE=standalone UMBP_GEM_IPC=1 MOONCAKE_SERIAL=1 \
    RUN=pd1p1d-c768-v41-$(date -u +%m%dT%H%M) \
    NOTE="B + serial + router DSv4.1 encoder: P GEM 0.9 + standalone UMBP over GEM IPC, D GEM 0.9" \
    bash "$W/scripts/run_1p1d_v3.sh"
finish "done (run_1p1d_v3 rc=$?)" 0
