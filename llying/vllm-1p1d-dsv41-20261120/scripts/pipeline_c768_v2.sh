#!/usr/bin/env bash
# Holder item: c768 rerun with the router's hyper bump (chunked-terminator stall, hyperium/hyper#4143).
#   1. cargo fmt / clippy -D warnings / test in rust:1-bookworm via router_check_inner.sh; stop on failure.
#   2. build the engine image at the branch head; refuse it unless the router binary carries hyper 1.12.0.
#   3. run_1p1d_v3.sh: B config + serial dispatch, smoke, accuracy gate, kv-aware check, golden 3.51, formal
#      AgentX c768 -- with a passive stall watch alongside: if AIPerf holds in-flight requests while both engines
#      are idle, it records the router's client sockets and metrics. It never touches a connection.
# Ends by touching the holder's STOP so the job releases the node. Not from an EXIT trap: a preempted item is
# requeued by the holder and must run again on the next node.
set -uo pipefail
source /perf_apps/liyingli/bench_agentx/vllm-1p1d-dsv41-20261120/scripts/env.sh
H=/perf_apps/liyingli/bench_agentx/dsv41-infera-vllm-20261119/holder
REPO=/perf_apps/liyingli/bench_agentx/upstream/Infera-vllm-1p1d-dsv41
L=$W/logs/pipeline-c768v2-$(date -u +%m%dT%H%M)
mkdir -p "$L" && chmod 777 "$L"
say() { echo "$(ts) pipeline-c768v2: $*" | tee -a "$W/logs/summary.log"; }
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
say "router checks rc=$rc: $(grep -aE '^test result' "$L/router-test.log" | awk '{p+=$4; f+=$6} END{print p" passed, "f" failed"}')"
((rc)) && finish "router checks FAILED ($L/router-test.log)" 1
git diff --quiet -- rust || finish "router checks left changes in rust/ (cargo fmt?): commit them first" 1

bash examples/vllm_1p1d_dsv41_flash/build_image.sh > "$L/build.log" 2>&1 || finish "image build FAILED: $(tail -3 "$L/build.log" | tr '\n' ' ')" 1
IMG=$(grep -ao "infera-vllm-dsv41:[0-9a-f]*-nightly-[0-9a-f]*" "$L/build.log" | tail -1)
nh=$(docker run --rm --entrypoint bash "$IMG" -c 'grep -aoc "hyper-1\.12\.0" /usr/local/bin/infera-router')
old=$(docker run --rm --entrypoint bash "$IMG" -c 'grep -aoc "hyper-1\.10\." /usr/local/bin/infera-router')
((nh > 0 && old == 0)) || finish "image $IMG router does not carry hyper 1.12.0 (1.12.0 refs $nh, 1.10.x refs $old)" 1
say "image $IMG (router hyper 1.12.0: $nh refs, 1.10.x: $old)"

RUN=pd1p1d-c768-hyper-$(date -u +%m%dT%H%M)
IP=$(node_ip)
(
    # Passive: two consecutive 5-min samples with AIPerf in-flight > 0 and both engines at 0 running / 0 waiting.
    log=$W/logs/$RUN/agentx-c768/result/conc_768/aiperf_artifacts/logs/aiperf.log
    hits=0; n=0
    while sleep 300; do
        [ -f "$log" ] || continue
        line=$(tr '\r' '\n' < "$log" | grep -a "progress |" | tail -1)
        inflight=$(grep -aoE 'in_flight=[0-9,]+' <<<"$line" | tr -dc 0-9)
        busy=0
        for p in 30000 30001; do
            busy=$((busy + $(curl -s -m 5 "http://$IP:$p/metrics" | grep -aE '^vllm:num_requests_(running|waiting)\{' |
                awk '{s+=$2} END{printf "%d", s}')))
        done
        if [ "${inflight:-0}" -gt 0 ] && [ "$busy" -eq 0 ]; then hits=$((hits + 1)); else hits=0; fi
        if ((hits >= 2)); then
            n=$((n + 1)); f=$L/stall-$n.txt
            { date -u; echo "$line"; echo "== router client sockets"; ss -tnio state established "( dport = :28000 )"
              echo "== router metrics"; curl -s -m 5 "http://$IP:28000/metrics" | grep -aE '^infera_request_inflight'; } > "$f" 2>&1
            say "stall watch: AIPerf in_flight=$inflight with both engines idle for >=10 min -> $f (not intervening)"
            hits=0
        fi
    done
) &
watch_pid=$!

env INFERA_IMAGE="$IMG" CONC=768 GATE=1 UMBP_MODE=standalone UMBP_GEM_IPC=1 MOONCAKE_SERIAL=1 RUN="$RUN" \
    NOTE="B + serial + router DSv4.1 encoder + hyper 1.12.0: P GEM 0.9 + standalone UMBP over GEM IPC, D GEM 0.9" \
    bash "$W/scripts/run_1p1d_v3.sh"
rc=$?
kill "$watch_pid" 2>/dev/null
finish "done (run_1p1d_v3 rc=$rc)" 0
