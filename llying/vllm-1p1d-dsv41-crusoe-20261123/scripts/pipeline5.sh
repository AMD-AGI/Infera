#!/usr/bin/env bash
# One optimization round on this node: fresh deployment with an arm's overrides -> smoke -> (GATE=1: accuracy,
# kv-aware, golden restart | GATE=0: deployment checks only, decode on golden from start-up) -> formal AgentX at
# CONC -> in-job analysis (per-step logs, profiler traces, comparison with the baseline) and a git-sized pack.
#   ARM (arms/<name>.sh, required), CONC (704), GATE (1), PROFILE_AT (s into the scored window, 3000; 0 = off),
#   BASELINE (packed or raw artifact dir:P_PORT,D_PORT of the reference run), RUN, plus any cluster.crusoe.sh variable.
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
    echo "# pipeline5 diagnostics"
    echo "P_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-p) D_ENV_X+=(VPD_STEPLOG_DIR=$LOG_DIR/steplog-d)"
    if ((PROFILE_AT > 0)); then
        echo "P_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-p\",\"torch_profiler_with_stack\":false}')"
        echo "D_ARGS_X+=(--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$LOG_DIR/trace-d\",\"torch_profiler_with_stack\":false}')"
    fi
} > "$LOG_DIR/arm.sh"
export ARM_FILE=$LOG_DIR/arm.sh
SP=/usr/local/lib/python3.12/dist-packages
export CTR_MOUNT_EXTRA="$W/instr/vpd_steplog.py:$SP/vpd_steplog.py:ro $W/instr/vpd_steplog.pth:$SP/vpd_steplog.pth:ro"
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
engine_died() { [[ -n "$(died_legs)" ]]; }
hang_watch up
wrap smoke > "$LOG_DIR/smoke.log" 2>&1; rc=$?
say "smoke rc=$rc: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | sed 's/^  //' | tr '\n' ';' | head -c 900)"
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
t2=$SECONDS
wrap agentx "$CONC" > "$LOG_DIR/agentx.log" 2>&1; rc=$?
nics agentx1
kill "$watch_pid" ${prof_pid:-} 2>/dev/null
say "AgentX c$CONC rc=$rc in $(( (SECONDS - t2) / 60 )) min: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/agentx.log" | grep -aA4 '^AgentX c' | tr -s ' ' | tr '\n' ' ' | head -c 700)"
}

golden_smoke
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
say "analysis: $(grep -aE 'cadence|tok/s|dt =|FULL graph' "$A/steplog.txt" 2>/dev/null | tr -s ' ' | tr '\n' ';' | head -c 900)"
say "end"
