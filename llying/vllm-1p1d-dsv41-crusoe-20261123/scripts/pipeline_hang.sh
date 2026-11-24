#!/usr/bin/env bash
# DP2 lockstep hang reproduction on this node: deploy an arm (decode on golden from start-up), then run
# scripts/hang_repro.py (idle IDLE_S -> one long request -> a burst of BURST requests, repeated) until BUDGET_S runs
# out. When an engine dies, its logs and hang dumps move to hangN/, the stack is redeployed and the loop continues.
# The last summary line counts cycles and hangs, so arms can be compared by hangs per cycle.
#   ARM (arms/<name>.sh, required), BUDGET_S (14400), IDLE_S (150), BURST (48), STEPLOG (1), PROFILE_AT (0 = off;
#   > 0 only adds --profiler-config, no profile is taken), RUN, plus any cluster.crusoe.sh variable.
# The arm file is sourced here (exports such as PREFILL_GMU / DECODE_GMU / D_PSI) and by engine/leg.sh (per-leg
# arrays P_ARGS_X / D_ARGS_X / P_ENV_X / D_ENV_X, UMBP_EXTRA_JSON).
set -uo pipefail
source "$(dirname "$0")/env.sh"
: "${ARM:?ARM=arms/<name>.sh}"
[[ "$ARM" == /* ]] || ARM=$W/$ARM
CONC=0 GATE=0 PROFILE_AT=${PROFILE_AT:-0} BUDGET_S=${BUDGET_S:-14400} IDLE_S=${IDLE_S:-150} BURST=${BURST:-48}
arm_name=$(basename "$ARM" .sh)
RUN=${RUN:-hang-$arm_name-$(date -u +%m%dT%H%M)}
export LOG_DIR=$W/logs/$RUN
mkdir -p "$LOG_DIR"/{steplog-p,steplog-d,trace-p,trace-d,analysis} && chmod -R 777 "$LOG_DIR"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }
# shellcheck disable=SC1090
source "$ARM"
# The legs source one file: the arm, then the diagnostics this pipeline adds.
{
    echo "# arm $ARM"
    cat "$ARM"
    echo
    echo "# pipeline_hang diagnostics"
    ((${STEPLOG:-1})) && echo "P_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-d)"
    if ((PROFILE_AT > 0)); then
        echo "P_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-p\",\"torch_profiler_with_stack\":false}')"
        echo "D_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-d\",\"torch_profiler_with_stack\":false}')"
    fi
} > "$LOG_DIR/arm.sh"
export ARM_FILE=$LOG_DIR/arm.sh
SP=/usr/local/lib/python3.12/dist-packages
# STEPLOG=0: no per-step recorder in the containers (neither the env var nor the .pth mount).
((${STEPLOG:-1})) && export CTR_MOUNT_EXTRA="$W/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $W/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro"
export INFERA_IMAGE=$IMAGE MODEL_MOUNT MODEL TOKENIZER=$MODEL SERVED CTR INFERENCEX_DIR UMBP_DIR=/dev/shm/lyl-dsv41-umbp
export ROUTER_RUST_LOG=${ROUTER_RUST_LOG:-info,infera_router::render_probe=debug}
cp "$W/data/gsm8k_test.jsonl" "$LOG_DIR/gsm8k_test.jsonl" && export GSM8K_FILE=$LOG_DIR/gsm8k_test.jsonl
wrap() { bash "$KIT/cluster/cluster.crusoe.sh" "$@"; }
nics() { bash "$W/scripts/ionic_counters.sh" > "$LOG_DIR/ionic-$1.txt" 2>&1; }
cleanup() { docker logs "$CTR" > "$LOG_DIR/docker-ctr.log" 2>&1; wrap down > "$LOG_DIR/down.log" 2>&1; }
trap cleanup EXIT
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
export PREFILL_GPUS=${PREFILL_GPUS:-0,1} DECODE_GPUS=${DECODE_GPUS:-4,5}
export ETCD_PORT=${ETCD_PORT:-22379} ROUTER_PORT=${ROUTER_PORT:-28700} PREFILL_PORT=${PREFILL_PORT:-31300} DECODE_PORT=${DECODE_PORT:-31301}
RP=$ROUTER_PORT
cp "$LOG_DIR/arm.sh" "$LOG_DIR/analysis/arm.sh"

say "start on $(hostname -s) job ${SLURM_JOB_ID:-?}, arm $arm_name (P_GMU=${PREFILL_GMU:-0.9} D_GMU=${DECODE_GMU:-0.9} D_PSI=${D_PSI:-11}), hang repro BUDGET_S=$BUDGET_S IDLE_S=$IDLE_S BURST=$BURST STEPLOG=${STEPLOG:-1} PROFILE_AT=$PROFILE_AT, P GPUs $PREFILL_GPUS D GPUs $DECODE_GPUS"
say "node co-tenants: $(squeue -w "$(hostname -s)" -h -o '%i:%u:%b:%C' 2>/dev/null | tr '\n' ' ')"
wrap down > "$LOG_DIR/down0.log" 2>&1
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES="$PREFILL_GPUS,$DECODE_GPUS" \
    --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
say "GPU free GiB ($PREFILL_GPUS,$DECODE_GPUS): ${free:-unknown}"
[[ $(wc -w <<<"${free:-}") == 4 ]] || { say "NOT STARTED: could not read the free memory of our 4 GPUs"; exit 1; }
for x in $free; do (( x >= 250 )) || { say "NOT STARTED: a GPU has only $x GiB free"; exit 1; }; done
busy=""
for p in $ETCD_PORT $((ETCD_PORT + 1)) $ROUTER_PORT $PREFILL_PORT $DECODE_PORT 29511 29513 29610 29612 28390 28391 40000 42000; do
    ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && busy+=" $p"
done
[[ -z "$busy" ]] || { say "NOT STARTED: ports already listening:$busy"; exit 1; }
(
    while :; do
        echo "$(ts) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" \
            "shm_used_gb=$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9) containers_other=$(docker ps --format '{{.Names}}' | grep -vc '^lyl-dsv41')" \
            "jobs=$(squeue -w "$(hostname -s)" -h -o '%i:%u:%b' 2>/dev/null | tr '\n' ',')"
        sleep 300
    done
) > "$LOG_DIR/host-monitor.txt" 2>&1 &
mon_pid=$!
trap 'kill $mon_pid 2>/dev/null; cleanup' EXIT

up_once() { if ((GATE)); then wrap up > "$LOG_DIR/up.log" 2>&1; else SPEC_MODE=golden wrap up > "$LOG_DIR/up.log" 2>&1; fi; }
kv() { for l in prefill decode; do echo -n "$l $(grep -aoE 'GPU KV cache size: [0-9,]+' "$LOG_DIR/$l.log" | grep -oE '[0-9,]+$' | sort -u | tr '\n' '/') "; done; }
nics up0
t0=$SECONDS
up_once; rc=$?
if ((rc)) && [[ -n "${PREFILL_GMU:-}${DECODE_GMU:-}" ]]; then
    say "up FAILED after $((SECONDS - t0)) s: $(grep -aE 'ERROR|died|never|out of memory|OutOfMemory' "$LOG_DIR/up.log" "$LOG_DIR"/prefill.log "$LOG_DIR"/decode.log 2>/dev/null | tail -3 | cut -c1-200 | tr '\n' ' ')"
    for l in prefill decode up; do mv "$LOG_DIR/$l.log" "$LOG_DIR/$l-try1.log" 2>/dev/null; done
    lower() { python3 -c "print(f'{max(0.85, float(\"$1\") - 0.03):.2f}')"; }
    [[ -n "${PREFILL_GMU:-}" ]] && export PREFILL_GMU=$(lower "$PREFILL_GMU")
    [[ -n "${DECODE_GMU:-}" ]] && export DECODE_GMU=$(lower "$DECODE_GMU")
    say "retrying with P_GMU=${PREFILL_GMU:-0.9} D_GMU=${DECODE_GMU:-0.9}"
    wrap down > "$LOG_DIR/down1.log" 2>&1
    t0=$SECONDS; up_once; rc=$?
fi
((rc)) && { say "up FAILED after $((SECONDS - t0)) s: $(grep -a 'ERROR\|died\|never' "$LOG_DIR/up.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1; }
say "up in $((SECONDS - t0)) s; KV tokens/rank: $(kv)"
for l in prefill decode; do
    say "$l: $(grep -aoE 'Found [0-9]+ HCAs?|installTransport, type=[a-z]+|couldn.t load driver[^ ]*|TcpTransport' "$LOG_DIR/$l.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';') graphs: $(tr '\r' '\n' < "$LOG_DIR/$l.log" | grep -aoE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
done

hang_watch() {
    for leg in "decode:$DECODE_PORT" "prefill:$PREFILL_PORT"; do
        nohup bash "$W/scripts/hang_watch2.sh" "$LOG_DIR" "$CTR" "${leg#*:}" none "${leg%%:*}" > "$LOG_DIR/hang-watch-${leg%%:*}-$1.log" 2>&1 &
    done
}
died_legs() { for l in prefill decode; do grep -aqE 'engine core reported fatal failure|LLM became permanently unhealthy' "$LOG_DIR/$l.log" 2>/dev/null && echo -n "$l "; done; }
wrap smoke > "$LOG_DIR/smoke.log" 2>&1
s=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | grep -avE 'answer names Paris|streaming:')
(( $(grep -c '^  FAIL' <<<"$s") == 0 )) || { say "smoke deployment checks FAILED: $(grep '^  FAIL' <<<"$s" | tr '\n' ';' | head -c 400)"; exit 1; }

cycles=0 hangs=0 n=0 tstart=$SECONDS
while :; do
    remaining=$((BUDGET_S - (SECONDS - tstart)))
    ((remaining > 900)) || break
    n=$((n + 1))
    hang_watch rep$n
    python3 "$W/scripts/hang_repro.py" "http://$IP:$RP" "$SERVED" "$LOG_DIR" "$remaining" "$IDLE_S" "$BURST" > "$LOG_DIR/repro-$n.log" 2>&1
    res=$(grep -a '^RESULT' "$LOG_DIR/repro-$n.log" | tail -1)
    c=$(grep -oE 'cycles=[0-9]+' <<<"$res" | tr -dc 0-9); h=$(grep -oE 'hangs=[0-9]+' <<<"$res" | tr -dc 0-9)
    cycles=$((cycles + ${c:-0})); hangs=$((hangs + ${h:-0}))
    say "repro segment $n: ${res:-no RESULT line}; last cycle: $(grep -a ' cycle ' "$LOG_DIR/repro-$n.log" | tail -1 | cut -c1-200)"
    ((${h:-0})) || break
    sleep 150   # let the watchers finish their second dump
    d=$LOG_DIR/hang$n; mkdir -p "$d"
    for f in prefill.log decode.log router.log umbp.log up.log steplog-p steplog-d hang-dumps hang-dumps-decode hang-dumps-prefill hang-watch*.log; do
        mv "$LOG_DIR"/$f "$d/" 2>/dev/null
    done
    docker logs "$CTR" > "$d/docker-ctr.log" 2>&1
    say "hang $hangs in $(grep -alE 'engine core reported fatal failure' "$d"/prefill.log "$d"/decode.log 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')after $cycles cycles; dumps: $(ls -d "$d"/hang-dumps* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' '); redeploying"
    wrap down > "$LOG_DIR/down-hang$n.log" 2>&1
    mkdir -p "$LOG_DIR"/{steplog-p,steplog-d} && chmod 777 "$LOG_DIR"/{steplog-p,steplog-d}
    t0=$SECONDS
    SPEC_MODE=golden wrap up > "$LOG_DIR/up.log" 2>&1 || { say "redeploy up FAILED after $((SECONDS - t0)) s"; break; }
    say "redeployed in $((SECONDS - t0)) s"
done
say "RESULT arm=$arm_name cycles=$cycles hangs=$hangs in $(( (SECONDS - tstart) / 60 )) min (IDLE_S=$IDLE_S BURST=$BURST STEPLOG=${STEPLOG:-1} PROFILE_AT=$PROFILE_AT)"
