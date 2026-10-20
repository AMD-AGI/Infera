#!/usr/bin/env bash
# Run on the login host (detached) after submitting the hold job: waits for JOB to start, then
#   1. set_nodes.sh <first node> <second node>;
#   2. image on both nodes: reuse when present with the expected ID, copy node-to-node when only one has it,
#      otherwise rebuild on the prefill node (build/build_images.sh) and update EXPECTED_IMAGE_ID;
#   3. node-local dirs, AgentX dataset cache on the control node;
#   4. gpu_stream_check.sh on both nodes (every GPU must pass);
#   5. start the tuning queue for ROUND.
# Usage: JOB=<id> ROUND=<r> prepare_alloc.sh
set -uo pipefail
: "${JOB:?}" "${ROUND:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10 -n)
log() { echo "$(date -u +%FT%TZ) prepare: $*"; }
die() { log "FAIL $*"; exit 1; }

log "waiting for job $JOB"
while true; do
    st="$(squeue -h -j "$JOB" -o %T 2>/dev/null)"
    [[ "$st" == RUNNING ]] && break
    [[ -z "$st" ]] && die "job $JOB is gone ($(sacct -n -X -j "$JOB" -o State | xargs))"
    sleep 30
done
mapfile -t nodes < <(scontrol show hostnames "$(squeue -h -j "$JOB" -o %N)")
(( ${#nodes[@]} == 2 )) || die "expected 2 nodes, got ${nodes[*]}"
P="${nodes[0]}"; D="${nodes[1]}"
log "job $JOB running on P=$P D=$D"
for i in $(seq 1 20); do "${SSH[@]}" "$P" true && "${SSH[@]}" "$D" true && break; sleep 15; done
# Node-to-node copies (build_images.sh, image copy) use the real ssh; record new host keys first.
for n in "$P" "$D"; do /usr/bin/ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -n "$n" true 2>/dev/null; done
bash "$HERE/set_nodes.sh" "$P" "$D" || die "set_nodes"

set +u; set -a; source "$R/config/config.3p1d.sh"; set +a; set -u
id_on() { "${SSH[@]}" "$1" "docker image inspect --format '{{.Id}}' $IMAGE 2>/dev/null"; }
idp="$(id_on "$P")"; idd="$(id_on "$D")"
log "image ids: P=${idp:-none} D=${idd:-none} expected=$EXPECTED_IMAGE_ID"
if [[ "$idp" == "$EXPECTED_IMAGE_ID" && "$idd" == "$EXPECTED_IMAGE_ID" ]]; then
    log "image present on both nodes"
elif [[ "$idp" == "$EXPECTED_IMAGE_ID" || "$idd" == "$EXPECTED_IMAGE_ID" ]]; then
    src="$P"; dst="$D"; [[ "$idd" == "$EXPECTED_IMAGE_ID" ]] && { src="$D"; dst="$P"; }
    log "copying image $src -> $dst"
    "${SSH[@]}" "$src" "docker save $IMAGE | ssh -o BatchMode=yes $dst docker load" || die "image copy"
else
    log "rebuilding image on $P"
    "${SSH[@]}" "$P" "mkdir -p /tmp/liyingli-docker-anon && { [ -s /tmp/liyingli-docker-anon/config.json ] || echo '{}' > /tmp/liyingli-docker-anon/config.json; } && DOCKER_CONFIG=/tmp/liyingli-docker-anon docker pull -q $SGLANG_BASE_IMAGE" ||
        die "base pull"
    "${SSH[@]}" "$P" "bash $R/build/build_images.sh $D" > "$R/logs/build-images.$JOB.log" 2>&1 || die "build (logs/build-images.$JOB.log)"
    new="$(id_on "$P")"
    sed -i "s#^EXPECTED_IMAGE_ID=.*#EXPECTED_IMAGE_ID=$new#" "$R/config/config.3p1d.sh"
    log "rebuilt image $new"
fi
[[ "$("${SSH[@]}" "$D" "docker image inspect --format '{{.Id}}' $(printf %q "$IMAGE")")" == "$(grep -m1 '^EXPECTED_IMAGE_ID=' "$R/config/config.3p1d.sh" | cut -d= -f2)" ]] ||
    die "decode node image ID mismatch"
"${SSH[@]}" "$P" "DOCKER_CONFIG=/tmp/liyingli-docker-anon docker image inspect quay.io/coreos/etcd:v3.5.14 >/dev/null 2>&1 || { mkdir -p /tmp/liyingli-docker-anon; [ -s /tmp/liyingli-docker-anon/config.json ] || echo '{}' > /tmp/liyingli-docker-anon/config.json; DOCKER_CONFIG=/tmp/liyingli-docker-anon docker pull -q quay.io/coreos/etcd:v3.5.14; }" ||
    die "etcd image"

key="${P##*-}"
for n in "$P" "$D"; do
    # Containers of an earlier round (e.g. left running when 34389 was preempted) would hold the GPUs.
    old="$("${SSH[@]}" "$n" "docker ps -a --format '{{.Names}}' | grep '^glm52-llying-2p1d-' | grep -v '^$CONTAINER_PREFIX-'" || true)"
    if [[ -n "$old" ]]; then
        log "removing earlier-round containers on $n: $(echo $old)"
        "${SSH[@]}" "$n" "docker stop --time 120 $(echo $old) >/dev/null; docker rm $(echo $old) >/dev/null" || die "cleanup on $n"
    fi
    "${SSH[@]}" "$n" "mkdir -p $NODE_RW_ROOT" || die "mkdir on $n"
    # Containers run as root and leave root-owned cache dirs behind (e.g. from an earlier job on this node).
    "${SSH[@]}" "$n" "docker run --rm --network none --entrypoint chown -v $NODE_RW_ROOT:$NODE_RW_ROOT $IMAGE -R \$(id -u):\$(id -g) $NODE_RW_ROOT" ||
        die "chown on $n"
done
"${SSH[@]}" "$P" "mkdir -p $NODE_RW_ROOT/cache/$key/agentx/hf && cp -rn /perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/cache/agentx/hf/hub $NODE_RW_ROOT/cache/$key/agentx/hf/" ||
    die "dataset cache"
log "node dirs and dataset cache ready"

for n in "$P" "$D"; do
    out="$("${SSH[@]}" "$n" "bash $HERE/gpu_stream_check.sh $IMAGE")"
    echo "$out"
    [[ "$(grep -c ': ok ' <<< "$out")" == 8 ]] || die "GPU stream check on $n"
done
log "16 GPUs pass the stream check"

if [[ "${START_QUEUE:-1}" == 1 ]]; then
    mkdir -p "$R/runtime/runs/$ROUND/tune"
    bash "$HERE/start_queue.sh" "ROUND=$ROUND" || die "start_queue"
    log "queue started for ROUND=$ROUND"
fi
log "done"
