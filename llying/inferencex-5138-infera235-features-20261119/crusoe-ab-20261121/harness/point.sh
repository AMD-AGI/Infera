#!/usr/bin/env bash
# One AgentX point of a rendered #5138 configuration, from a fresh deployment as InferenceX runs each point:
# plan -> etcd, workers, router -> transport and GEM checks -> smoke -> AgentX client (InferenceX's srt_agentic.sh)
# with a guard, VRAM samples and ionic counters -> logs and memory facts -> teardown until the nodes are idle.
# Usage: NODE_HOSTS="h0 [h1]" point.sh LABEL CONFIG CONC fast|full|up PORT_BASE PLACE...
#   CONFIG names render/out/<CONFIG>.json; PLACE is role:node:gpus, one per worker (plan.py --place).
#   up: bring-up only (deployment, checks, smoke, Mooncake error counts), no AgentX. ROUTER_WAIT (s, default 1200)
#   bounds the wait for the router's registrations.
# Writes runs/<LABEL>/ and appends one JSON line to runs/points.jsonl. Exits 0 unless the arguments are wrong.
set -uo pipefail
source /home/liyingli/bench_agentx/ix5138-crusoe-20261120/harness/lib.sh
read -ra NODE_HOSTS <<<"${NODE_HOSTS:?}"
label=$1 config=$2 conc=$3 mode=$4 base=$5
shift 5
places=("$@")
[[ "$mode" == fast || "$mode" == full || "$mode" == up ]] || { echo "mode must be fast, full or up"; exit 2; }
run=$R/runs/$label
prefix=lyl-ix5138-$label
mkdir -p "$run/logs"
exec > >(tee -a "$run/point.log") 2>&1
status=ok
note() { echo "$*" >>"$run/notes.txt"; say "$*"; }
fail() { status=$1; note "FAIL $*"; }

nodes_used=$(printf '%s\n' "${places[@]}" | cut -d: -f2 | sort -u)
say "point $label: $config C$conc $mode, places ${places[*]}, nodes ${NODE_HOSTS[*]}"
facts=()
for i in "${!NODE_HOSTS[@]}"; do facts+=(--node "$(node_facts "$i")"); done
aiter_jit=$(timeout 300 docker run --rm --entrypoint python3 "$IMAGE" -c "import importlib.util as u, os
s = u.find_spec('aiter')
d = s and (os.path.dirname(s.origin) if s.origin else list(s.submodule_search_locations)[0])
print('AITERJIT=' + (os.path.join(d, 'jit') if d and os.path.isdir(os.path.join(d, 'jit')) else ''))" 2>/dev/null |
    sed -n 's/^AITERJIT=//p')
note "aiter JIT directory kept in a volume: ${aiter_jit:-none}"
py "$R/harness/plan.py" --config "$R/render/out/$config.json" --run-dir "$run" --label "$prefix" \
    "${facts[@]}" "${places[@]/#/--place=}" --port-base "$base" --model "$MODEL" --image "$IMAGE" --site "$R" \
    --hf-home "$LOCAL/hf_home" ${aiter_jit:+--aiter-jit "$aiter_jit"} || { fail plan; status=plan; }

cleanup() {
    while IFS=$'\t' read -r name node _ container _; do
        on "$node" "docker logs $container >$run/logs/$name.log 2>&1; docker rm -f $container >/dev/null 2>&1" || true
    done <"$run/manifest.tsv"
    for n in $nodes_used; do on "$n" "docker rm -f $prefix-vram >/dev/null 2>&1" || true; done
    docker rm -f "$prefix-agentx" >/dev/null 2>&1 || true
}

if [[ "$status" == ok ]]; then
    for n in $nodes_used; do
        wait_idle "$n" 300 || fail "node $n busy before start"
    done
fi
if [[ "$status" == ok ]]; then
    for n in $nodes_used; do
        ionic_counters "$n" >"$run/ionic-before-${NODE_HOSTS[$n]}.txt"
        on "$n" "docker run -d --name $prefix-vram --user $(id -u):$(id -g) -v $R:$R --entrypoint bash $IMAGE \
            $R/harness/vram_sampler.sh $run/vram-${NODE_HOSTS[$n]}.txt" >/dev/null
    done
    t_up=$SECONDS
    say "start etcd and workers"
    while IFS=$'\t' read -r name node kind _ _; do
        [[ "$kind" == router ]] && continue
        on "$node" "bash $run/proc/$name.sh" >/dev/null || fail "start $name"
        [[ "$kind" == etcd ]] && sleep 5
    done <"$run/manifest.tsv"
fi

# Wait for every worker's /health while its container runs; then the router and its registrations.
alive() { [[ "$(on "$1" "docker inspect -f '{{.State.Running}}' $2 2>/dev/null")" == true ]]; }
if [[ "$status" == ok ]]; then
    deadline=$((SECONDS + 3600))
    while IFS=$'\t' read -r name node kind container health; do
        [[ "$kind" == prefill || "$kind" == decode || "$kind" == etcd ]] || continue
        until curl -sf -m 5 -o /dev/null "$health"; do
            alive "$node" "$container" || { fail "$name exited before healthy"; break 2; }
            ((SECONDS < deadline)) || { fail "$name not healthy after 3600 s"; break 2; }
            sleep 20
        done
        say "$name healthy after $((SECONDS - t_up)) s"
    done <"$run/manifest.tsv"
fi
if [[ "$status" == ok ]]; then
    read -r _ node _ container health < <(awk -F'\t' '$3 == "router"' "$run/manifest.tsv")
    on "$node" "bash $run/proc/router.sh" >/dev/null || fail "start router"
    want_p=$(grep -c $'\tprefill\t' "$run/manifest.tsv") want_d=$(grep -c $'\tdecode\t' "$run/manifest.tsv")
    deadline=$((SECONDS + ${ROUTER_WAIT:-1200}))
    until [[ "$(curl -sf -m 10 "$health" | py -c 'import json, sys
w = json.load(sys.stdin).get("workers") or []
c = lambda m: sum(1 for x in w if x.get("disagg_mode") == m and x.get("status") == "active")
print(c("prefill"), c("decode"))' 2>/dev/null)" == "$want_p $want_d" ]]; do
        ((SECONDS < deadline)) || { fail "router: workers not registered after ${ROUTER_WAIT:-1200} s"; break; }
        alive "$node" "$container" || { fail "router exited"; break; }
        sleep 15
    done
    [[ "$status" == ok ]] && say "router: $want_p prefill + $want_d decode active, up in $((SECONDS - t_up)) s"
    curl -sf -m 10 "$health" >"$run/workers.json" || true
fi

# Every rank must be on RDMA (a provider that fails to load leaves TCP) with its KV pool in GEM BOs.
if [[ "$status" == ok ]]; then
    while IFS=$'\t' read -r name node kind container _; do
        [[ "$kind" == prefill || "$kind" == decode ]] || continue
        on "$node" "docker logs $container 2>&1" >"$run/logs/$name.startup.log"
        hcas=$(grep -aoE 'Found [0-9]+ HCAs' "$run/logs/$name.startup.log" | sort | uniq -c | tr '\n' ' ')
        tcp=$(grep -acE 'installTransport, type=tcp|Found 0 HCAs|couldn.t load driver' "$run/logs/$name.startup.log")
        gem=$(grep -ac 'amdgpu GEM BOs' "$run/logs/$name.startup.log")
        note "$name: $hcas; tcp/provider warnings $tcp; GEM pools $gem"
        ((tcp == 0)) || fail "$name is not on RDMA"
        ((gem > 0)) || fail "$name has no GEM KV pool"
    done <"$run/manifest.tsv"
fi

# Smoke: two short chat completions through the router.
if [[ "$status" == ok ]]; then
    router=$(awk -F'\t' '$3 == "router" {sub("/v1/workers", "", $5); print $5}' "$run/manifest.tsv")
    for q in "What is 17 * 23? Answer with the number." "Name the capital of France in one word."; do
        curl -sf -m 600 "$router/v1/chat/completions" -H 'Content-Type: application/json' -d "$(py -c '
import json, sys; print(json.dumps({"model": "amd/GLM-5.2-MXFP4", "max_tokens": 512, "temperature": 0,
"messages": [{"role": "user", "content": sys.argv[1]}]}))' "$q")" >>"$run/smoke.jsonl" || fail "smoke request failed"
        echo >>"$run/smoke.jsonl"
    done
    py -c 'import json, sys
for line in open(sys.argv[1]):
    if line.strip():
        m = json.loads(line)["choices"][0]["message"]
        print("smoke:", repr((m.get("content") or "")[-120:]))' "$run/smoke.jsonl" || fail "smoke output unreadable"
fi

# AgentX through InferenceX's own client. The guard stops it once an engine exits or reports a fatal error.
if [[ "$status" == ok && "$mode" != up ]]; then
    mkdir -p "$run/result" "$LOCAL/cache/agentx-tmp-$label"
    {
        cat "$run/bench.env"
        echo "CONC=$conc"
        echo "CONC_LIST=$conc"
        echo "DURATION=3600"
        echo "AIPERF_EXPERIMENTAL_FAST=$([[ "$mode" == fast ]] && echo 1 || echo 0)"
        echo "RESULT_DIR=$run/result"
        echo "RESULT_FILENAME=agentx_$label"
        echo "AGENTIC_OUTPUT_DIR=$run/result"
        echo "AIPERF_RUNTIME_DIR=$LOCAL/cache/aiperf-runtime"
        echo "AIPERF_DATASET_MMAP_CACHE_DIR=$LOCAL/cache/aiperf_mmap_cache"
        echo "HF_HOME=$LOCAL/cache/hf"
        echo "HOME=$LOCAL/cache/home"
        echo "TMPDIR=/ax-tmp"
        echo "XDG_CACHE_HOME=$LOCAL/cache/xdg"
    } >"$run/agentx.env"
    mkdir -p "$LOCAL/cache/home" "$LOCAL/cache/xdg"
    (
        while sleep 60; do
            while IFS=$'\t' read -r name node kind container _; do
                [[ "$kind" == etcd ]] && continue
                if ! alive "$node" "$container"; then
                    echo "$(date -u +%T) guard: $name exited" >>"$run/notes.txt"
                    docker stop "$prefix-agentx" >/dev/null 2>&1
                    exit 0
                fi
                bad=$(on "$node" "docker logs --since 70s $container 2>&1 | grep -aE 'Fatal Python error|HSA_STATUS_ERROR|Memory access fault|crashed with exit code' | head -1")
                if [[ -n "$bad" ]]; then
                    echo "$(date -u +%T) guard: $name: $bad" >>"$run/notes.txt"
                    docker stop "$prefix-agentx" >/dev/null 2>&1
                    exit 0
                fi
            done <"$run/manifest.tsv"
        done
    ) &
    guard=$!
    for n in $nodes_used; do ionic_counters "$n" >"$run/ionic-bench-start-${NODE_HOSTS[$n]}.txt"; done
    t_bench=$SECONDS
    say "AgentX C$conc ($mode) via $router"
    timeout $((3600 + 7200)) docker run --name "$prefix-agentx" --network host --ipc host --shm-size 32g \
        --user "$(id -u):$(id -g)" -v "$R/src/InferenceX/inferencex-e2e:/infmax-workspace:ro" -v "$MODEL:$MODEL:ro" \
        -v "$run:$run" -v "$LOCAL/cache:$LOCAL/cache" -v "$LOCAL/cache/agentx-tmp-$label:/ax-tmp" \
        --env-file "$run/agentx.env" --entrypoint bash "$IMAGE" /infmax-workspace/benchmarks/srt_agentic.sh \
        >"$run/agentx.log" 2>&1
    rc=$?
    kill "$guard" 2>/dev/null
    say "AgentX exited $rc after $((SECONDS - t_bench)) s"
    ((rc == 0)) || fail "agentx rc=$rc"
    for n in $nodes_used; do ionic_counters "$n" >"$run/ionic-after-${NODE_HOSTS[$n]}.txt"; done
    while IFS=$'\t' read -r name node kind _ health; do
        [[ "$kind" == prefill || "$kind" == decode ]] && curl -s -m 10 "${health%/health}/metrics" >"$run/logs/$name.prom"
    done <"$run/manifest.tsv"
fi

# ionic firmware rejections surface in the kernel log as "opcode ... error IONIC_RC_*" (the driver returns EINVAL).
for n in $nodes_used; do
    on "$n" "(dmesg 2>&1 || journalctl -k --no-pager 2>&1) | grep -aiE 'ionic|IONIC_RC|Operation not permitted' | tail -200" \
        >"$run/kernel-ionic-${NODE_HOSTS[$n]}.txt" || true
done
cleanup
# Mooncake connection setup failures per worker (748af1a5: notification QP/MR setup, QP RTR/RTS transitions).
for log in "$run"/logs/prefill*.log "$run"/logs/decode*.log; do
    [[ -e "$log" && "$log" != *.startup.log ]] || continue
    counts=$(for p in 'Notification QP initialization failed' 'ERR_MEMORY|-302' 'modify QP|modify_qp|ibv_modify_qp' \
        'RTR|RTS' 'Invalid argument' 'Failed to register memory|ibv_reg_mr' 'Sync batch data transfer timeout|transfer failed'; do
        echo -n "[$p]=$(grep -acE "$p" "$log") "
    done)
    note "mooncake $(basename "$log" .log): $counts"
    grep -aE 'Notification QP initialization failed|-302|modify QP|RTS|RTR|Invalid argument' "$log" | head -5 \
        >>"$run/mooncake-errors.txt"
done
for n in $nodes_used; do wait_idle "$n" 1800 || note "node $n still busy after teardown"; done
py "$R/harness/summarize.py" "$run" "$status" "$label" "$config" "$conc" "$mode" >>"$R/runs/points.jsonl" ||
    echo "{\"label\": \"$label\", \"status\": \"$status\", \"summary\": \"failed\"}" >>"$R/runs/points.jsonl"
say "point $label done: $status"
exit 0
