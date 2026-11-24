#!/usr/bin/env bash
# One round on this node through the kit's crusoe wrapper, as AUS run_1p1d_v3.sh: fresh deployment -> smoke ->
# accuracy gate (real DSpark verification) -> kv-aware check -> golden 3.51 -> formal AgentX at CONC.
# One line per step in logs/summary.log; the deployment is torn down on exit.
#   CONC (640), RUN, GATE, plus any cluster.crusoe.sh variable.
#   GATE=1: the full gate above. GATE=0: for a configuration whose accuracy already passed on this image -- decode
#   starts on golden, and only smoke's deployment checks gate the run (pairing, RDMA bytes, pull timeouts, DP2/fp8,
#   GEM, UMBP); its answer-content and streaming checks are not meaningful on synthetic acceptance and are ignored.
set -uo pipefail
source "$(dirname "$0")/env.sh"
CONC=${CONC:-640} GATE=${GATE:-1}
RUN=${RUN:-pd1p1d-c$CONC-crusoe-$(date -u +%m%dT%H%M)}
export LOG_DIR=$W/logs/$RUN
mkdir -p "$LOG_DIR" && chmod 777 "$LOG_DIR"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }
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

say "start on $(hostname -s) job ${SLURM_JOB_ID:-?}, amdgpu $(cat /sys/module/amdgpu/version), $(docker run --rm --entrypoint cat "$IMAGE" /opt/infera/BUILD_INFO | tr '\n' ' '), c$CONC, P GPUs $PREFILL_GPUS D GPUs $DECODE_GPUS NIC ionic_${DECODE_GPUS%%,*}"
say "node co-tenants: $(squeue -w "$(hostname -s)" -h -o '%i:%u:%b:%C' 2>/dev/null | tr '\n' ' ')"
wrap down > "$LOG_DIR/down0.log" 2>&1
# Another job's leftovers (or a co-tenant) can hold our GPUs: stop if one of them has less than 250 GiB free.
gids=(); for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [[ -n "$gid" ]] && gids+=(--group-add "$gid"); done
free=$(timeout 300 docker run --rm --network none --device /dev/kfd --device /dev/dri "${gids[@]}" -e HIP_VISIBLE_DEVICES="$PREFILL_GPUS,$DECODE_GPUS" \
    --entrypoint python3 "$IMAGE" \
    -c 'import torch; print(" ".join(str(int(torch.cuda.mem_get_info(i)[0] / 2**30)) for i in range(torch.cuda.device_count())))' 2>/dev/null | tail -1)
say "GPU free GiB ($PREFILL_GPUS,$DECODE_GPUS): ${free:-unknown}"
[[ $(wc -w <<<"${free:-}") == 4 ]] || { say "NOT STARTED: could not read the free memory of our 4 GPUs"; exit 1; }
for x in $free; do (( x >= 250 )) || { say "NOT STARTED: a GPU has only $x GiB free"; exit 1; }; done
# Fixed ports of the kit (leg.sh: DP RPC / master, VLLM_PORT, Mooncake bootstrap) and ours: none may be taken.
busy=""
for p in $ETCD_PORT $((ETCD_PORT + 1)) $ROUTER_PORT $PREFILL_PORT $DECODE_PORT 29511 29513 29610 29612 28390 28391 40000 42000; do
    ss -ltnH "sport = :$p" 2>/dev/null | grep -q . && busy+=" $p"
done
[[ -z "$busy" ]] || { say "NOT STARTED: ports already listening:$busy"; exit 1; }
(
    # Host-side interference record (shared node): every 5 min, load, free memory, shm, co-tenant jobs and containers.
    while :; do
        echo "$(ts) load=$(cut -d' ' -f1-3 /proc/loadavg) memavail_gb=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)" \
            "shm_used_gb=$(df -BG --output=used /dev/shm | tail -1 | tr -dc 0-9) containers_other=$(docker ps --format '{{.Names}}' | grep -vc '^lyl-dsv41')" \
            "jobs=$(squeue -w "$(hostname -s)" -h -o '%i:%u:%b' 2>/dev/null | tr '\n' ',')"
        sleep 300
    done
) > "$LOG_DIR/host-monitor.txt" 2>&1 &
mon_pid=$!
trap 'kill $mon_pid 2>/dev/null; cleanup' EXIT

nics up0
t0=$SECONDS
if ((GATE)); then wrap up > "$LOG_DIR/up.log" 2>&1; else SPEC_MODE=golden wrap up > "$LOG_DIR/up.log" 2>&1; fi
rc=$?
kv() { for l in prefill decode; do echo -n "$l $(grep -aoE 'GPU KV cache size: [0-9,]+' "$LOG_DIR/$l.log" | head -1 | grep -oE '[0-9,]+$') "; done; }
((rc)) && { say "up FAILED after $((SECONDS - t0)) s: $(grep -a 'ERROR\|died\|never' "$LOG_DIR/up.log" | tail -3 | tr '\n' ' ' | head -c 600)"; exit 1; }
say "up in $((SECONDS - t0)) s; KV tokens/rank: $(kv)"
for l in prefill decode; do
    say "$l transport: $(grep -aoE 'Found [0-9]+ HCAs?|installTransport, type=[a-z]+|couldn.t load driver[^ ]*|TcpTransport' "$LOG_DIR/$l.log" | sort | uniq -c | tr -s ' ' | tr '\n' ';')"
done

wrap smoke > "$LOG_DIR/smoke.log" 2>&1; rc=$?
say "smoke rc=$rc: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | sed 's/^  //' | tr '\n' ';' | head -c 900)"
if ((GATE)); then
    ((rc)) && { say "smoke FAILED: no AgentX point"; exit 1; }
    wrap accuracy > "$LOG_DIR/accuracy.log" 2>&1; rc=$?
    say "accuracy: $(grep -ahE '^(gsm8k|needle@)' "$LOG_DIR"/accuracy/*.log | sed -E 's/ prompt_tokens_p50.*completion_tokens_max=/ ctmax=/' | tr '\n' ';' | head -c 700) $(grep -a '^ACCURACY' "$LOG_DIR/accuracy.log")"
    ((rc)) && { say "accuracy gate FAILED: no AgentX point"; exit 1; }
    bash "$W/scripts/kvaware_check.sh" "$LOG_DIR" "http://$IP:$RP" "http://$IP:$PREFILL_PORT" "$SERVED" > "$LOG_DIR/kvaware.log" 2>&1; rc=$?
    say "kv-aware check rc=$rc: $(grep -aE '^(engine|two-turn)' "$LOG_DIR/kvaware.log" | tr '\n' ' ' | head -c 600)"
    ((rc)) && { say "kv-aware check FAILED: no AgentX point"; exit 1; }
    t1=$SECONDS
    wrap golden > "$LOG_DIR/golden.log" 2>&1 || { say "golden restart failed: $(tail -2 "$LOG_DIR/golden.log" | tr '\n' ' ')"; exit 1; }
    say "decode on golden 3.51 after $((SECONDS - t1)) s: $(grep -aoE 'synthetic_acceptance_length[^,}]*' "$LOG_DIR/decode.log" | head -1)"
else
    s=$(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/smoke.log" | grep -aE '^  (ok|FAIL)' | grep -avE 'answer names Paris|streaming:')
    bad=$(grep -c '^  FAIL' <<<"$s"); oks=$(grep -c '^  ok' <<<"$s")
    # 8 deployment checks: pairing, RDMA bytes, pull timeouts, DP2/fp8 x2, GEM x2, UMBP.
    (( bad == 0 && oks >= 8 )) || { say "smoke deployment checks FAILED ($bad failed, $oks ok): no AgentX point"; exit 1; }
    grep -q "'rejection_sample_method': 'synthetic'" "$LOG_DIR/decode.log" || { say "decode is not on golden acceptance: no AgentX point"; exit 1; }
    say "GATE=0: accuracy gate skipped (passed on this image and configuration); decode on golden from start-up"
fi
code=$(curl -s -o "$LOG_DIR/smoke-golden.json" -w '%{http_code}' -m 600 "http://$IP:$RP/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello.\"}],\"max_tokens\":64}")
say "golden smoke HTTP $code; AgentX formal c$CONC starting"

(
    # Passive stall watch: two consecutive 5-min samples with AIPerf in-flight > 0 and both engines at
    # 0 running / 0 waiting. It records the router's client sockets and metrics, never touches a connection.
    log=$LOG_DIR/agentx-c$CONC/result/conc_$CONC/aiperf_artifacts/logs/aiperf.log
    hits=0; n=0
    while sleep 300; do
        [ -f "$log" ] || continue
        line=$(tr '\r' '\n' < "$log" | grep -a "progress |" | tail -1)
        inflight=$(grep -aoE 'in_flight=[0-9,]+' <<<"$line" | tr -dc 0-9)
        busy=0
        for p in $PREFILL_PORT $DECODE_PORT; do
            busy=$((busy + $(curl -s -m 5 "http://$IP:$p/metrics" | grep -aE '^vllm:num_requests_(running|waiting)\{' | awk '{s+=$2} END{printf "%d", s}')))
        done
        if [ "${inflight:-0}" -gt 0 ] && [ "$busy" -eq 0 ]; then hits=$((hits + 1)); else hits=0; fi
        if ((hits >= 2)); then
            n=$((n + 1)); f=$LOG_DIR/stall-$n.txt
            { date -u; echo "$line"; echo "== router client sockets"; ss -tnio state established "( dport = :$RP )"
              echo "== router metrics"; curl -s -m 5 "http://$IP:$RP/metrics" | grep -aE '^infera_request_inflight'; } > "$f" 2>&1
            say "stall watch: AIPerf in_flight=$inflight with both engines idle for >=10 min -> $f (not intervening)"
            hits=0
        fi
    done
) &
watch_pid=$!
nics agentx0
t2=$SECONDS
wrap agentx "$CONC" > "$LOG_DIR/agentx.log" 2>&1; rc=$?
nics agentx1
kill "$watch_pid" 2>/dev/null
say "AgentX c$CONC rc=$rc in $(( (SECONDS - t2) / 60 )) min: $(sed -r 's/\x1B\[[0-9;]*m//g' "$LOG_DIR/agentx.log" | grep -aA4 '^AgentX c' | tr -s ' ' | tr '\n' ' ' | head -c 700)"
say "end"
