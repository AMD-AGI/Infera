#!/usr/bin/env bash
# The PD example on its own image, once the chain running now (WAIT_PID) has ended: build
# deploy/docker/Dockerfile.sglang from a clean clone at REV on THIS node, copy the image to the decode
# node, run tests/unit and tests/engine/sglang inside it (logged, not gating), then the example's
# correctness run (pd_correct_chain.sh on CORRECT_SITE) and, if smoke is 6/6 and GSM8K strict-match is at
# least 0.96, the AgentX points listed in LOG_DIR/plan.txt ("SITE_ENV CONC" per line, read one line at a
# time, so the rest of the plan can be edited while it runs; each point is a pd_perf_chain.sh run).
# Usage: ex_campaign.sh WAIT_PID REV TAG KIT CORRECT_SITE LOG_DIR
set -uo pipefail
wait_pid="$1" rev="$2" tag="$3" kit="$4" correct_site="$5" log_dir="$6"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/home/liyingli/bench_agentx/baseline/Infera
PEER=crsuse2-m2m-140
GPU_FLAGS=(--init --privileged --ipc host --shm-size 16gb --ulimit memlock=-1
    --device /dev/kfd --device /dev/dri --group-add video --group-add render -v /boot:/boot:ro)
export DOCKER_CONFIG=/tmp/liyingli-docker-anon
mkdir -p "$log_dir" "$DOCKER_CONFIG" && [[ -s "$DOCKER_CONFIG/config.json" ]] || echo '{}' >"$DOCKER_CONFIG/config.json"
say() { echo "$(date -u +%T) $*"; }

while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
say "chain $wait_pid ended"

src="$log_dir/src"
rm -rf "$src" && git clone -q "$REPO" "$src" && git -C "$src" checkout -q "$rev" || { say "checkout failed"; exit 1; }
say "source $(git -C "$src" log --oneline -1)"
if ! docker image inspect "$tag" >/dev/null 2>&1; then
    docker build --network=host -f "$src/deploy/docker/Dockerfile.sglang" -t "$tag" "$src" >"$log_dir/build.log" 2>&1 ||
        { say "build failed (build.log)"; exit 1; }
fi
id="$(docker image inspect -f '{{.Id}}' "$tag")"
say "built $tag $id"
if [[ "$(ssh -o BatchMode=yes "$PEER" docker image inspect -f '{{.Id}}' "$tag" 2>/dev/null)" != "$id" ]]; then
    docker save "$tag" | ssh -o BatchMode=yes "$PEER" docker load >"$log_dir/load.log" 2>&1 ||
        { say "copy to $PEER failed (load.log)"; exit 1; }
fi
say "on $PEER: $(ssh -o BatchMode=yes "$PEER" docker image inspect -f '{{.Id}}' "$tag")"

mkdir -p "$log_dir/scratch"
in_image() {
    local name="$1"
    shift
    # GPUs 0-3: the previous chain's prefill (GPUs 4-7) may still be releasing its VRAM.
    docker run --rm "${GPU_FLAGS[@]}" -e HIP_VISIBLE_DEVICES=0,1,2,3 -v "$src":/workspace:ro \
        -v "$log_dir/scratch":/scratch -e HF_HOME=/scratch/hf -e PYTHONDONTWRITEBYTECODE=1 -w /workspace \
        --entrypoint bash "$tag" -lc 'exec bash tests/lib/container_pytest.sh "$@"' _ "$@" >"$log_dir/$name-image.log" 2>&1
    say "$name in image exited $? ($name-image.log)"
}
in_image unit "[unit]" -q tests/unit
in_image engine --per-file "[engine tests/engine/sglang]" tests/engine/sglang

run_id="correct-ex-$(date -u +%m%dT%H%M)"
bash "$S/pd_correct_chain.sh" "$kit" "$correct_site" "$log_dir" "$run_id"
smoke="$(grep -a 'checks passed' "$log_dir/$run_id.log" | tail -1)"
strict="$(grep -a 'strict-match' "$log_dir/$run_id.log" | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')"
say "correctness: ${smoke:-no smoke result}, GSM8K strict ${strict:-none}"
[[ "$smoke" == "6/6 checks passed" ]] && awk -v s="${strict:-0}" 'BEGIN {exit !(s >= 0.96)}' ||
    { say "correctness gate failed, no AgentX points"; exit 1; }

n=1
while line="$(sed -n "${n}p" "$log_dir/plan.txt")" && [[ -n "$line" ]]; do
    read -r site conc <<<"$line"
    say "point $n: $(basename "$site") C$conc"
    bash "$S/pd_perf_chain.sh" "$kit" "$site" "$log_dir/perf-$(basename "$site" .env)" "$conc"
    n=$((n + 1))
done
say "plan done"
