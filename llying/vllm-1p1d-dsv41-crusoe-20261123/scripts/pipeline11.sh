#!/usr/bin/env bash
# One optimization round on this node (pipeline10.sh with the UMBP pool in HIP VMM memory and THP off; EXPERIMENT 4.11): fresh deployment with an arm's overrides -> smoke -> (GATE=1: accuracy,
# kv-aware, golden restart | GATE=0: deployment checks only, decode on golden from start-up) -> formal AgentX at
# CONC -> in-job analysis (per-step logs, profiler traces, comparison with the baseline) and a git-sized pack.
#   ARM (arms/<name>.sh, required), CONC (704), GATE (1), PROFILE_AT (s into the scored window, 3000; 0 = off),
#   BASELINE (packed or raw artifact dir:P_PORT,D_PORT of the reference run), RUN, plus any cluster.crusoe.sh variable.
#   STEPLOG (1): 0 drops the per-step recorder; PROFILE_AT=0 also drops --profiler-config.
# Host telemetry (scripts/host_telemetry.sh, every 2 s from before start-up: GPU busy, page migration / compaction, KFD
# queue evictions) runs throughout; the analysis lists every stall with what the host did meanwhile (slow_steps.py)
# and counts the times prefill found the UMBP server unresponsive (EXPERIMENT 4.4-4.6: page migration under
# GPU-registered host memory makes KFD evict the queues of every process that registered it).
# ROUTER_BACKLOG_SHIM (1): preload instr/listen_backlog.so into the router so it listens with a backlog of 4096 instead
# of 128; at AgentX's warmup phase switch ~1,000 connections arrive within a second and, with 128, the kernel resets
# those whose SYN cookies fail (EXPERIMENT 4.2: SyncookiesFailed equalled the resets).
# UMBP_POOL (vmm): the UMBP server's DRAM pool in HIP VMM host memory (instr/umbp_vmm_pool.so, UMBP_VMM_SO), its chunks
# alternating over the NUMA nodes (UMBP_VMM_NUMA=interleave); pinned = hipHostMalloc (instr/umbp_pinned_pool.so, still
# KFD userptr); register = UMBP's own hipHostRegister'd pages. THP_OFF (1): the engine processes run with THP disabled
# (instr/thp_off.pth), so they no longer drive direct compaction (hangF: 0 compaction stalls, 0 evictions in 84 min).
# HOSTALLOC_LOG (1): instr/hostalloc_log.so (HOSTALLOC_SO) after the GEM allocator in both legs logs each engine
# process's pinned / registered host memory (KFD userptr) to hostalloc/. The analysis adds every process's KFD evictions
# and the compaction counters (kfd_evict_summary.py). UVA_VMM (1): instr/uva_vmm.py puts vLLM's UVA buffers of at least
# UVA_VMM_MIN_BYTES (1 MiB; all_token_ids is 4 GiB) in HIP VMM host memory instead of pinned (userptr) memory, so another
# job's compaction cannot evict the engine workers through them (R13: 17-40 s stalls at 16:11). The UMBP server also gets
# the hostalloc logger (its own small pinned buffers kept it evicted ~0.5 s of every 2 s under that compaction).
# SHEXP_PROBE (0): instr/shexp_probe.py logs, in prefill, why the MoE runner does (not) put the shared experts on its aux
# stream. STOP_AFTER_SMOKE (0): end the run after the smoke test (whose 16k-token request is enough for such probes).
# AUX_PRIO (0): instr/aux_prio.py makes vLLM's aux stream (the shared experts' stream) high priority in prefill.
# PROFILE_SMOKE (0, with STOP_AFTER_SMOKE=1 and PROFILE_AT > 0 for --profiler-config): profile prefill and decode over a
# second smoke pass before stopping. Decode's step recorder writes every step and step entries (VPD_STEPLOG_FLUSH_S=0, VPD_STEPLOG_BEGIN=1), so
# a hung decode shows each rank's last step. The analysis adds the UMBP server's availability (umbp_avail.py).
# A DP2 lockstep hang (DECODE_HANG.zh-CN.md; decode, and prefill with MegaMoE) kills that engine after 300 s; when it
# happens during smoke or AgentX, the run's logs move to try1/ or ax-try1/ and the deployment is redone once (AgentX only while at least
# AX_RETRY_S, 12600 by default, remains of the 6 h job).
# The arm file is sourced here (exports such as PREFILL_GMU / DECODE_GMU / D_PSI) and by engine/leg.sh (per-leg
# arrays P_ARGS_X / D_ARGS_X / P_ENV_X / D_ENV_X, UMBP_EXTRA_JSON).
set -uo pipefail
source "$(dirname "$0")/env.sh"
: "${ARM:?ARM=arms/<name>.sh}"
[[ "$ARM" == /* ]] || ARM=$W/$ARM
CONC=${CONC:-704} GATE=${GATE:-1} PROFILE_AT=${PROFILE_AT:-3000}
UMBP_POOL=${UMBP_POOL:-vmm} UMBP_VMM_NUMA=${UMBP_VMM_NUMA:-interleave} THP_OFF=${THP_OFF:-1} HOSTALLOC_LOG=${HOSTALLOC_LOG:-1}
UVA_VMM=${UVA_VMM:-1} UVA_VMM_MIN_BYTES=${UVA_VMM_MIN_BYTES:-1048576}
SHEXP_PROBE=${SHEXP_PROBE:-0} STOP_AFTER_SMOKE=${STOP_AFTER_SMOKE:-0} AUX_PRIO=${AUX_PRIO:-0} PROFILE_SMOKE=${PROFILE_SMOKE:-0}
arm_name=$(basename "$ARM" .sh)
RUN=${RUN:-pd1p1d-c$CONC-$arm_name-$(date -u +%m%dT%H%M)}
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
    echo "# pipeline11 diagnostics"
    # Both legs run in GEM mode (LD_PRELOAD=the GEM allocator, set by the kit); the role's LD_PRELOAD replaces it.
    ((SHEXP_PROBE)) && echo "P_ENV_X+=(SHEXP_PROBE=1)"
    ((AUX_PRIO)) && echo "P_ENV_X+=(AUX_STREAM_PRIORITY=1)"
    ((UVA_VMM)) && echo "P_ENV_X+=(VLLM_UVA_VMM=1 VLLM_UVA_VMM_MIN_BYTES=$UVA_VMM_MIN_BYTES) D_ENV_X+=(VLLM_UVA_VMM=1 VLLM_UVA_VMM_MIN_BYTES=$UVA_VMM_MIN_BYTES)"
    ((HOSTALLOC_LOG)) && echo "P_ENV_X+=(LD_PRELOAD=/usr/local/lib/libhip_gem_alloc.so:$LOG_DIR/hostalloc_log.so HOSTALLOC_LOG_DIR=$LOG_DIR/hostalloc) D_ENV_X+=(LD_PRELOAD=/usr/local/lib/libhip_gem_alloc.so:$LOG_DIR/hostalloc_log.so HOSTALLOC_LOG_DIR=$LOG_DIR/hostalloc)"
    ((${STEPLOG:-1})) && echo "P_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-d VPD_STEPLOG_FLUSH_S=0 VPD_STEPLOG_BEGIN=1)"
    if ((PROFILE_AT > 0)); then
        echo "P_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-p\",\"torch_profiler_with_stack\":false}')"
        echo "D_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-d\",\"torch_profiler_with_stack\":false}')"
    fi
} > "$LOG_DIR/arm.sh"
export ARM_FILE=$LOG_DIR/arm.sh
SP=/usr/local/lib/python3.12/dist-packages
# STEPLOG=0: no per-step recorder in the containers (neither the env var nor the .pth mount).
mounts=""
((${STEPLOG:-1})) && mounts+=" $W/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $W/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro"
((THP_OFF)) && mounts+=" $W/instr/thp_off.py:$SP/thp_off.py:ro $W/instr/thp_off.pth:$SP/thp_off.pth:ro"
((UVA_VMM)) && mounts+=" $W/instr/uva_vmm.py:$SP/uva_vmm.py:ro $W/instr/uva_vmm.pth:$SP/uva_vmm.pth:ro"
((SHEXP_PROBE)) && mounts+=" $W/instr/shexp_probe.py:$SP/shexp_probe.py:ro $W/instr/shexp_probe.pth:$SP/shexp_probe.pth:ro"
((AUX_PRIO)) && mounts+=" $W/instr/aux_prio.py:$SP/aux_prio.py:ro $W/instr/aux_prio.pth:$SP/aux_prio.pth:ro"
[[ -n "$mounts" ]] && export CTR_MOUNT_EXTRA="${mounts# }"
export INFERA_IMAGE=$IMAGE MODEL_MOUNT MODEL TOKENIZER=$MODEL SERVED CTR INFERENCEX_DIR UMBP_DIR=/dev/shm/lyl-dsv41-umbp
export ROUTER_RUST_LOG=${ROUTER_RUST_LOG:-info,infera_router::render_probe=debug}
cp "$W/data/gsm8k_test.jsonl" "$LOG_DIR/gsm8k_test.jsonl" && export GSM8K_FILE=$LOG_DIR/gsm8k_test.jsonl
# LOG_DIR is mounted into the container at the same path.
((${ROUTER_BACKLOG_SHIM:-1})) && cp "$W/instr/listen_backlog.so" "$LOG_DIR/listen_backlog.so" && export ROUTER_PRELOAD=$LOG_DIR/listen_backlog.so
case $UMBP_POOL in
    vmm) cp "${UMBP_VMM_SO:-$W/instr/umbp_vmm_pool.so}" "$LOG_DIR/umbp_vmm_pool.so" && export UMBP_PRELOAD_EXTRA=$LOG_DIR/umbp_vmm_pool.so
         export UMBP_ENV_EXTRA="UMBP_VMM_NUMA=$UMBP_VMM_NUMA" ;;
    pinned) cp "$W/instr/umbp_pinned_pool.so" "$LOG_DIR/" && export UMBP_PRELOAD_EXTRA=$LOG_DIR/umbp_pinned_pool.so ;;
    register) ;;
    *) echo "UMBP_POOL must be vmm, pinned or register, got $UMBP_POOL" >&2; exit 1 ;;
esac
((HOSTALLOC_LOG)) && { mkdir -p "$LOG_DIR/hostalloc" && chmod 777 "$LOG_DIR/hostalloc" &&
    cp "${HOSTALLOC_SO:-$W/instr/hostalloc_log.so}" "$LOG_DIR/hostalloc_log.so" || exit 1
    export UMBP_PRELOAD_EXTRA="${UMBP_PRELOAD_EXTRA:+$UMBP_PRELOAD_EXTRA:}$LOG_DIR/hostalloc_log.so"
    export UMBP_ENV_EXTRA="${UMBP_ENV_EXTRA:+$UMBP_ENV_EXTRA }HOSTALLOC_LOG_DIR=$LOG_DIR/hostalloc"; }
wrap() { bash "$KIT/cluster/cluster.crusoe.sh" "$@"; }
nics() { bash "$W/scripts/ionic_counters.sh" > "$LOG_DIR/ionic-$1.txt" 2>&1; }
cleanup() { docker logs "$CTR" > "$LOG_DIR/docker-ctr.log" 2>&1; wrap down > "$LOG_DIR/down.log" 2>&1; }
trap cleanup EXIT
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
export PREFILL_GPUS=${PREFILL_GPUS:-0,1} DECODE_GPUS=${DECODE_GPUS:-4,5}
export ETCD_PORT=${ETCD_PORT:-22379} ROUTER_PORT=${ROUTER_PORT:-28700} PREFILL_PORT=${PREFILL_PORT:-31300} DECODE_PORT=${DECODE_PORT:-31301}
RP=$ROUTER_PORT
cp "$LOG_DIR/arm.sh" "$LOG_DIR/analysis/arm.sh"

say "start on $(hostname -s) job ${SLURM_JOB_ID:-?}, arm $arm_name (P_GMU=${PREFILL_GMU:-0.9} D_GMU=${DECODE_GMU:-0.9} D_PSI=${D_PSI:-11}), c$CONC GATE=$GATE PROFILE_AT=$PROFILE_AT, P GPUs $PREFILL_GPUS D GPUs $DECODE_GPUS"
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
bash "$W/scripts/host_telemetry.sh" "$LOG_DIR/host-telemetry.txt" 2 &
telemetry_pid=$!
trap 'kill $mon_pid $telemetry_pid 2>/dev/null; cleanup' EXIT

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
say "up in $((SECONDS - t0)) s; KV tokens/rank: $(kv); router listen backlog $(ss -Hltn "sport = :$RP" | awk '{print $3}')"
((UVA_VMM)) && say "uva_vmm: $(grep -ahE '\[uva_vmm\] (pid|HIP VMM allocation failed|patch failed)' "$LOG_DIR"/prefill.log "$LOG_DIR"/decode.log | cut -c1-160 | sort | uniq -c | sort -rn | head -6 | tr -s ' ' | tr '\n' ';')"
say "UMBP pool ($UMBP_POOL): $(grep -aE 'umbp_vmm_pool|umbp_pinned_pool|registered for GPU access|hipHostRegister of' "$LOG_DIR/umbp.log" | cut -c1-200 | tr '\n' ';')"
for l in prefill decode; do
    say "$l: $(grep -aoE 'Found [0-9]+ HCAs?|installTransport, type=[a-z]+|couldn.t load driver[^ ]*|TcpTransport' "$LOG_DIR/$l.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';') graphs: $(tr '\r' '\n' < "$LOG_DIR/$l.log" | grep -aoE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
done

hang_watch() {
    for leg in "decode:$DECODE_PORT" "prefill:$PREFILL_PORT"; do
        nohup bash "$W/scripts/hang_watch2.sh" "$LOG_DIR" "$CTR" "${leg#*:}" none "${leg%%:*}" > "$LOG_DIR/hang-watch-${leg%%:*}-$1.log" 2>&1 &
    done
}
died_legs() { for l in prefill decode; do grep -aqE 'engine core reported fatal failure|LLM became permanently unhealthy' "$LOG_DIR/$l.log" 2>/dev/null && echo -n "$l "; done; }
engine_died() { [[ -n "$(died_legs)" ]]; }
hang_watch up
wrap smoke > "$LOG_DIR/smoke.log" 2>&1; rc=$?
say "smoke rc=$rc: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | sed 's/^  //' | tr '\n' ';' | head -c 900)"
if ((STOP_AFTER_SMOKE)); then
    if ((PROFILE_SMOKE)); then
        for port in $PREFILL_PORT $DECODE_PORT; do curl -s -m 120 -X POST "http://$IP:$port/start_profile" > /dev/null; done
        wrap smoke > "$LOG_DIR/smoke-profiled.log" 2>&1
        for port in $PREFILL_PORT $DECODE_PORT; do curl -s -m 900 -X POST "http://$IP:$port/stop_profile" > /dev/null; done
        say "PROFILE_SMOKE: traces $(ls "$LOG_DIR"/trace-p "$LOG_DIR"/trace-d 2>/dev/null | grep -c json)"
    fi
    say "STOP_AFTER_SMOKE: aux_prio: $(grep -ah '\[aux_prio\] pid' "$LOG_DIR"/prefill.log | cut -c1-200 | sort -u | head -2 | tr '\n' ';')"
    sleep 20
    say "STOP_AFTER_SMOKE: shexp_probe: $(grep -ah '\[shexp_probe\]' "$LOG_DIR"/prefill.log | cut -c1-400 | sort | uniq -c | sort -rn | head -6 | tr -s ' ' | tr '\n' ';')"
    say "STOP_AFTER_SMOKE: prefill KV / graphs: $(tr '\r' '\n' < "$LOG_DIR/prefill.log" | grep -aoE 'Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB|Available KV cache memory: [0-9.]+ GiB' | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
    exit 0
fi
# A decode DP2 lockstep hang (DECODE_HANG.zh-CN.md) during smoke kills the engine after 300 s; keep its logs and
# redeploy once, since it has struck at random points of otherwise healthy runs.
if ((rc)) && engine_died; then
    say "engine died during smoke: $(died_legs)($(grep -ah -m1 -oE 'RPC call to [a-z_]+ timed out' "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log" | tr '\n' ' ')); hang dumps: $(ls -d "$LOG_DIR"/hang-dumps* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' '); redeploying once"
    sleep 5
    mkdir -p "$LOG_DIR/try1"
    for f in prefill.log decode.log up.log smoke.log router.log umbp.log docker-ctr.log steplog-p steplog-d hang-dumps hang-dumps-decode hang-dumps-prefill hang-watch*.log; do
        mv "$LOG_DIR/$f" "$LOG_DIR/try1/" 2>/dev/null
    done
    docker logs "$CTR" > "$LOG_DIR/try1/docker-ctr.log" 2>&1
    wrap down > "$LOG_DIR/down-try1.log" 2>&1
    mkdir -p "$LOG_DIR"/{steplog-p,steplog-d} && chmod 777 "$LOG_DIR"/{steplog-p,steplog-d}
    t0=$SECONDS; up_once; rc=$?
    ((rc)) && { say "redeploy up FAILED after $((SECONDS - t0)) s: $(grep -a 'ERROR\|died\|never' "$LOG_DIR/up.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1; }
    say "redeployed in $((SECONDS - t0)) s; KV tokens/rank: $(kv)"
    hang_watch up2
    wrap smoke > "$LOG_DIR/smoke.log" 2>&1; rc=$?
    say "smoke (redeploy) rc=$rc: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | sed 's/^  //' | tr '\n' ';' | head -c 900)"
fi
if ((GATE)); then
    ((rc)) && { say "smoke FAILED: no AgentX point"; exit 1; }
    wrap accuracy > "$LOG_DIR/accuracy.log" 2>&1; rc=$?
    say "accuracy: $(grep -ahE '^(gsm8k|needle@)' "$LOG_DIR"/accuracy/*.log | sed -E 's/ prompt_tokens_p50.*completion_tokens_max=/ ctmax=/' | tr '\n' ';' | head -c 700) $(grep -a '^ACCURACY' "$LOG_DIR/accuracy.log")"
    ((rc)) && { say "accuracy gate FAILED: no AgentX point"; exit 1; }
    bash "$W/scripts/kvaware_check.sh" "$LOG_DIR" "http://$IP:$RP" "http://$IP:$PREFILL_PORT" "$SERVED" > "$LOG_DIR/kvaware.log" 2>&1; rc=$?
    say "kv-aware check rc=$rc: $(grep -aE '^two-turn' "$LOG_DIR/kvaware.log" | head -c 300)"
    ((rc)) && { say "kv-aware check FAILED: no AgentX point"; exit 1; }
    t1=$SECONDS
    wrap golden > "$LOG_DIR/golden.log" 2>&1 || { say "golden restart failed: $(tail -2 "$LOG_DIR/golden.log" | tr '\n' ' ')"; exit 1; }
    say "decode on golden 3.51 after $((SECONDS - t1)) s; decode KV tokens/rank: $(grep -aoE 'GPU KV cache size: [0-9,]+' "$LOG_DIR/decode.log" | grep -oE '[0-9,]+$' | sort -u | tr '\n' '/')"
else
    s=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | grep -avE 'answer names Paris|streaming:')
    bad=$(grep -c '^  FAIL' <<<"$s"); oks=$(grep -c '^  ok' <<<"$s")
    (( bad == 0 && oks >= 8 )) || { say "smoke deployment checks FAILED ($bad failed, $oks ok): no AgentX point"; exit 1; }
    grep -q "'rejection_sample_method': 'synthetic'" "$LOG_DIR/decode.log" || { say "decode is not on golden acceptance: no AgentX point"; exit 1; }
    say "GATE=0: accuracy gate skipped; decode on golden from start-up"
fi
golden_smoke() {
    local code
    code=$(curl -s -o "$LOG_DIR/smoke-golden.json" -w '%{http_code}' -m 600 "http://$IP:$RP/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}")
    say "golden smoke HTTP $code; AgentX formal c$CONC starting"
}

# On-demand kernel compiles (FlyDSL MoE shapes) can stall prefill for minutes; at c768 AgentX's cold first wave already
# takes ~1,800 s, the most a root warmup request survives. Compile them before the run (JIT_WARM=0 skips).
jit_warm() {
    ((${JIT_WARM:-1})) || return 0
    local t=$SECONDS
    python3 "$W/scripts/jit_warm.py" "http://$IP:$RP" "$SERVED" 3 > "$LOG_DIR/jit-warm.log" 2>&1
    say "JIT warm-up in $((SECONDS - t)) s: $(tr '\n' ';' < "$LOG_DIR/jit-warm.log"); prefill JIT compiles so far: $(grep -ac 'JIT compilation during inference' "$LOG_DIR/prefill.log")"
}

AXLOG=$LOG_DIR/agentx-c$CONC/result/conc_$CONC/aiperf_artifacts/logs/aiperf.log
run_agentx() {
(
    # Passive stall watch, as pipeline.sh.
    hits=0; n=0
    while sleep 300; do
        [ -f "$AXLOG" ] || continue
        line=$(tr '\r' '\n' < "$AXLOG" | grep -a "progress |" | tail -1)
        inflight=$(grep -aoE 'in_flight=[0-9,]+' <<<"$line" | tr -dc 0-9)
        busy=0
        for p in $PREFILL_PORT $DECODE_PORT; do
            busy=$((busy + $(curl -s -m 5 "http://$IP:$p/metrics" | grep -aE '^vllm:num_requests_(running|waiting)\{' | awk '{s+=$2} END{printf "%d", s}')))
        done
        if [ "${inflight:-0}" -gt 0 ] && [ "$busy" -eq 0 ]; then hits=$((hits + 1)); else hits=0; fi
        if ((hits >= 2)); then
            n=$((n + 1)); f=$LOG_DIR/stall-$n.txt
            { date -u; echo "$line"; ss -tnio state established "( dport = :$RP )"; curl -s -m 5 "http://$IP:$RP/metrics" | grep -aE '^infera_request_inflight'; } > "$f" 2>&1
            say "stall watch: AIPerf in_flight=$inflight with both engines idle for >=10 min -> $f (not intervening)"
            hits=0
        fi
    done
) &
watch_pid=$!
prof_pid=""
if ((PROFILE_AT > 0)); then
    (
        # Torch profiles late in the scored window: prefill 5 s, then decode 3 s (never both at once).
        while ! grep -qa "Phase profiling (profiling) started" "$AXLOG" 2>/dev/null; do sleep 30; done
        sleep "$PROFILE_AT"
        for leg in "P:$PREFILL_PORT:5" "D:$DECODE_PORT:3"; do
            IFS=: read -r r port secs <<<"$leg"
            curl -s -m 120 -X POST "http://$IP:$port/start_profile" > /dev/null && echo "$(ts) $r start_profile"
            sleep "$secs"
            curl -s -m 900 -X POST "http://$IP:$port/stop_profile" > /dev/null && echo "$(ts) $r stop_profile"
            sleep 30
        done
    ) > "$LOG_DIR/profile.log" 2>&1 &
    prof_pid=$!
fi
nics agentx0
# Host TCP counters every 10 s during AgentX: listen-queue overflows and aborted connections identify who resets
# AIPerf's connections when the warmup switches phase (~all lanes send at once).
( while :; do echo "== $(date -u +%T.%N | cut -c1-12) estab_router=$(ss -Htn state established "( sport = :$RP )" 2>/dev/null | wc -l)"; grep -A1 '^TcpExt:' /proc/net/netstat; sleep 10; done ) > "$LOG_DIR/tcp-counters.txt" 2>&1 &
tcp_pid=$!
t2=$SECONDS
wrap agentx "$CONC" > "$LOG_DIR/agentx.log" 2>&1; rc=$?
nics agentx1
kill "$watch_pid" ${prof_pid:-} $tcp_pid 2>/dev/null
say "AgentX c$CONC rc=$rc in $(( (SECONDS - t2) / 60 )) min: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/agentx.log" | grep -aA4 '^AgentX c' | tr -s ' ' | tr '\n' ' ' | head -c 700)"
}

golden_smoke
jit_warm
run_agentx
AX_RETRY_S=${AX_RETRY_S:-12600}
if ((rc)) && engine_died && (( 21600 - 900 - SECONDS >= AX_RETRY_S )); then
    say "engine died during AgentX: $(died_legs)($(grep -ah -m1 -oE 'RPC call to [a-z_]+ timed out' "$LOG_DIR/prefill.log" "$LOG_DIR/decode.log" | tr '\n' ' ')); hang dumps: $(ls -d "$LOG_DIR"/hang-dumps* 2>/dev/null | xargs -r -n1 basename | tr '\n' ' '); redeploying on golden and repeating AgentX"
    mkdir -p "$LOG_DIR/ax-try1"
    for f in "agentx-c$CONC" agentx.log profile.log prefill.log decode.log decode-*.log router.log umbp.log up.log smoke.log \
        smoke-golden.json steplog-p steplog-d trace-p trace-d hang-dumps hang-dumps-decode hang-dumps-prefill hang-watch*.log ionic-agentx*.txt stall-*.txt; do
        mv "$LOG_DIR"/$f "$LOG_DIR/ax-try1/" 2>/dev/null
    done
    docker logs "$CTR" > "$LOG_DIR/ax-try1/docker-ctr.log" 2>&1
    wrap down > "$LOG_DIR/down-ax1.log" 2>&1
    mkdir -p "$LOG_DIR"/{steplog-p,steplog-d,trace-p,trace-d} && chmod 777 "$LOG_DIR"/{steplog-p,steplog-d,trace-p,trace-d}
    t0=$SECONDS
    SPEC_MODE=golden wrap up > "$LOG_DIR/up.log" 2>&1 || { say "AgentX redeploy up FAILED after $((SECONDS - t0)) s: $(grep -a 'ERROR\|died\|never' "$LOG_DIR/up.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1; }
    say "redeployed on golden in $((SECONDS - t0)) s; KV tokens/rank: $(kv)"
    hang_watch ax2
    wrap smoke > "$LOG_DIR/smoke.log" 2>&1
    s=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | grep -avE 'answer names Paris|streaming:')
    bad=$(grep -c '^  FAIL' <<<"$s"); oks=$(grep -c '^  ok' <<<"$s")
    (( bad == 0 && oks >= 8 )) || { say "smoke deployment checks after the redeploy FAILED ($bad failed, $oks ok): no AgentX point"; exit 1; }
    grep -q "'rejection_sample_method': 'synthetic'" "$LOG_DIR/decode.log" || { say "redeployed decode is not on golden acceptance: no AgentX point"; exit 1; }
    golden_smoke
    jit_warm
    run_agentx
fi

# ---- in-job analysis (on the node) ----
A=$LOG_DIR/analysis
ART=$(dirname "$(find "$LOG_DIR/agentx-c$CONC" -name server_metrics_export.csv | head -1)")
T0=$(python3 - "$AXLOG" <<'PY'
import re, sys, datetime
for line in open(sys.argv[1], errors="replace"):
    if "Phase profiling (profiling) started" in line:
        ts = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)", line).group(1)
        print(datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()); break
PY
)
if [[ -n "$T0" ]]; then
    T1=$(python3 -c "print($T0 + 3600)")
    { python3 "$W/scripts/steplog_summary.py" "$LOG_DIR/steplog-p" prefill "$T0" "$T1"
      python3 "$W/scripts/steplog_summary.py" "$LOG_DIR/steplog-d" decode "$T0" "$T1"; } > "$A/steplog.txt" 2>&1
fi
for f in $(find "$LOG_DIR"/trace-p "$LOG_DIR"/trace-d -name "*.json*" -type f 2>/dev/null | sort); do
    timeout 1200 python3 "$W/scripts/trace_cat.py" "$f" --top 40 > "$A/trace-$(basename "$(dirname "$f")")-$(basename "$f" | cut -c1-60).txt" 2>&1
done
[[ -n "$ART" ]] && python3 "$W/scripts/compare_runs.py" "this=$ART:$PREFILL_PORT,$DECODE_PORT" ${BASELINE:+"base=$BASELINE"} > "$A/compare.txt" 2>&1
python3 "$W/scripts/pack_raw.py" "$LOG_DIR" "$W/packed/$RUN" > "$A/pack.txt" 2>&1
cp -r "$A" "$W/packed/$RUN/analysis" 2>/dev/null
for d in steplog-p steplog-d; do
    mkdir -p "$W/packed/$RUN/$d" && for f in "$LOG_DIR/$d"/*.csv; do [ -s "$f" ] && gzip -c "$f" > "$W/packed/$RUN/$d/$(basename "$f").gz"; done
done
python3 "$W/scripts/slow_steps.py" "$LOG_DIR" 5 > "$A/slow-steps.txt" 2>&1
python3 "$W/scripts/umbp_avail.py" "$LOG_DIR" 3 > "$A/umbp-avail.txt" 2>&1
cp "$A/umbp-avail.txt" "$W/packed/$RUN/analysis/" 2>/dev/null
python3 "$W/scripts/kfd_evict_summary.py" "$LOG_DIR" > "$A/kfd-evictions.txt" 2>&1
cp "$A/kfd-evictions.txt" "$W/packed/$RUN/analysis/" 2>/dev/null
say "KFD evictions: $(head -1 "$A/kfd-evictions.txt" | cut -c1-300)"
if ((HOSTALLOC_LOG)); then
    for f in "$LOG_DIR"/hostalloc/*.log; do
        l=$(grep -a SUMMARY "$f" | tail -1); [[ $l =~ peak\ ([0-9]+) && ${BASH_REMATCH[1]} -gt 0 ]] && echo "$(basename "$f" .log) ${l:13}"
    done > "$A/hostalloc.txt" 2>/dev/null
    say "hostalloc (processes with pinned / registered host memory): $(cut -c1-160 "$A/hostalloc.txt" | tr '\n' ';')"
fi
cp "$A/slow-steps.txt" "$W/packed/$RUN/analysis/" 2>/dev/null
gzip -c "$LOG_DIR/host-telemetry.txt" > "$W/packed/$RUN/host-telemetry.txt.gz" 2>/dev/null
say "analysis: $(grep -aE 'cadence|tok/s|dt =|FULL graph' "$A/steplog.txt" 2>/dev/null | tr -s ' ' | tr '\n' ';' | head -c 900)"
say "stalls > 5 s: $(grep -ac '^[pd]' "$A/slow-steps.txt") ($(grep -a '^total stalled' "$A/slow-steps.txt" | tr '\n' ';')); $(grep -avE '^  ' "$A/umbp-avail.txt" | tr '\n' ';')"
say "end"
