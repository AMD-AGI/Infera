# Host-side helpers for the holder items. Source after env.sh. Every item sets RUN (log dir name) first.
source /perf_apps/yihou/dsv41_kf_20261120/pd/scripts/env.sh
R="$W/logs/${RUN:?RUN}"
mkdir -p "$R" && chmod 1777 "$R"
say() { echo "$(ts) $RUN: $*" | tee -a "$W/logs/summary.log"; }

# The Docker Hub login in ~/.docker/config.json carries an expired token, which fails even public pulls; retry
# anonymously with an empty client config.
pull_image() {
    docker pull "$1" && return 0
    # anonymous retry with an empty client config (kept, never deleted: node-local yihou dir)
    mkdir -p /data/yihou/dsv41_kf_20261120/cache/docker-anon
    DOCKER_CONFIG=/data/yihou/dsv41_kf_20261120/cache/docker-anon docker pull "$1"
}
ensure_images() {
    docker image inspect "$IMAGE" >/dev/null 2>&1 || pull_image "$IMAGE" || return 1
    docker image inspect "$ETCD_IMAGE" >/dev/null 2>&1 || pull_image "$ETCD_IMAGE" || return 1
}

# Our containers (prefix yihou-dsv41kf-) are removed on exit (kf port: no slurm guard step; we run over ssh inside
# our pool job and stop our containers before release).
GUARDS=()
start_guard() { :; }
remove_ours() {
    local names n; names="$(docker ps -a --format '{{.Names}}' | grep "^$CTR_PREFIX" || true)"
    for n in $names; do case "$n" in yihou-dsv41kf-*) docker rm -f "$n" >/dev/null 2>&1 ;; esac; done
}
cleanup() { collect_logs; remove_ours; }

start_etcd() {
    docker rm -f "${CTR_PREFIX}etcd" >/dev/null 2>&1
    docker run -d --name "${CTR_PREFIX}etcd" --network host "$ETCD_IMAGE" etcd \
        --data-dir /tmp/etcd --advertise-client-urls "http://$(node_ip):$ETCD_PORT" \
        --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
        --listen-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
        --initial-advertise-peer-urls "http://127.0.0.1:$((ETCD_PORT + 1))" \
        --initial-cluster "default=http://127.0.0.1:$((ETCD_PORT + 1))" >/dev/null || return 1
    for _ in $(seq 60); do curl -sf "http://127.0.0.1:$ETCD_PORT/health" >/dev/null && return 0; sleep 1; done
    return 1
}

host_gids() { local g gid; for g in video render; do gid=$(getent group "$g" | cut -d: -f3); [ -n "$gid" ] && echo "--group-add $gid"; done; }

# How Mooncake registers VRAM on this node, by amdgpu version. On 6.16 a KFD BO pinned through peer memory
# (ibv_reg_mr) is charged once in the KFD limit and one pinned through dma-buf twice: register by peer memory
# ("peermem"). On 6.19 both paths charge it twice, so the KV pools go in amdgpu GEM BOs, which KFD charges once
# ("gem"; registered by dma-buf). A 6.16 node without a GPU peer-memory module can only use dma-buf: GEM as well.
rdma_mr_mode() {
    local v maj min
    v="$(cat /sys/module/amdgpu/version 2>/dev/null)"; maj=${v%%.*}; min=${v#*.}; min=${min%%.*}
    if [[ "$maj" =~ ^[0-9]+$ && "$min" =~ ^[0-9]+$ ]] && ((maj > 6 || (maj == 6 && min >= 19))); then
        echo gem
    elif grep -qE '^(ib_peer_mem|amdp2p) ' /proc/modules 2>/dev/null; then
        echo peermem
    else
        echo gem
    fi
}

# node_state TAG: driver, peer-memory modules, pinned host memory and KFD eviction warnings, before / after a run.
node_state() {
    {
        echo "== $(ts) $1 on $(hostname -s)"
        echo "amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null) kernel $(uname -r) mr_mode $(rdma_mr_mode)"
        grep -E '^(ib_peer_mem|amdp2p|amdgpu|ionic_rdma|ionic) ' /proc/modules | awk '{print "module", $1}'
        grep -E '^(MemTotal|MemAvailable|Shmem|Mlocked|HugePages_Total):' /proc/meminfo
        echo "kfd evictions in dmesg: $(dmesg 2>/dev/null | grep -c evict_process_worker)"
        rocm-smi --showmeminfo vram --csv 2>/dev/null | head -9
    } >> "$R/node-state.txt" 2>&1
}

# build_gem NAME: compile the GEM KV allocator (Infera dev/pd_opt/vllm_glm52_agentx, VMM import) inside the engine
# container, once per image, into $W/gem/. Sets GEM_SO.
build_gem() {
    local ctr="${CTR_PREFIX}$1"
    GEM_SO="$W/gem/libhip_gem_alloc-${IMAGE_DIGEST:7:12}.so"
    [ -s "$GEM_SO" ] && return 0
    docker exec "$ctr" bash -c "r=\${ROCM_PATH:-/opt/rocm}; gcc -O2 -Wall -shared -fPIC -D__HIP_PLATFORM_AMD__ -I\$r/include \
        $W/gem/hip_gem_alloc.c -o $GEM_SO.tmp -L\$r/lib -lamdhip64 -lhsa-runtime64 -libverbs -lpthread -ldl && mv $GEM_SO.tmp $GEM_SO"
}

# start_engine_ctr NAME: a long-lived container with every GPU and RDMA device; processes are started by docker exec.
start_engine_ctr() {
    local name="${CTR_PREFIX}$1"
    mkdir -p "$NODE_CACHE"
    docker rm -f "$name" >/dev/null 2>&1
    # The vLLM image has no render group: pass the host's numeric gids. The router shares this container and holds
    # client + prefill + decode connections per request, so nofile goes well above the 1024 default.
    docker run -d --name "$name" --network host --ipc host --shm-size 64g \
        --device /dev/kfd --device /dev/dri --device /dev/infiniband \
        $(host_gids) --cap-add SYS_PTRACE --cap-add IPC_LOCK \
        --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=1048576:1048576 \
        -v /perf_apps:/perf_apps -v "$NODE_CACHE:/root/.cache" -e HF_HUB_OFFLINE=1 \
        ${EXTRA_DOCKER_ARGS:-} --entrypoint "" "$IMAGE" sleep infinity >/dev/null || return 1
    docker exec "$name" bash "$W/scripts/ctr_install_infera.sh" >> "$R/install-$1.log" 2>&1
}

# exec_bg CTR NAME CMD...: run CMD in the container in the background, output to $R/NAME.log
exec_bg() {
    local ctr="${CTR_PREFIX}$1" log="$R/$2.log"; shift 2
    docker exec -d -e NODE_IP="$(node_ip)" "$ctr" bash -c "$(printf '%q ' "$@") > $log 2>&1"
}

# wait_worker LOGNAME PORT TIMEOUT_S: until the engine answers /health; fails early on a dead launcher.
wait_worker() {
    local log="$R/$1.log" port="$2" deadline=$((SECONDS + ${3:-7200}))
    while ((SECONDS < deadline)); do
        curl -sf -m 5 "http://127.0.0.1:$port/health" >/dev/null && grep -q "worker ready" "$log" 2>/dev/null && return 0
        if grep -qE "vllm engine failed to start|EngineCore failed|exited with code|RuntimeError: Engine core initialization failed" "$log" 2>/dev/null; then
            return 1
        fi
        sleep 20
    done
    return 1
}

wait_router() {
    for _ in $(seq 90); do curl -sf -m 5 "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null && return 0; sleep 2; done
    return 1
}

# chat PROMPT MAX_TOKENS [STREAM]: one temperature-0 chat completion through the router; prints the JSON.
chat() {
    curl -s -m 1800 "http://127.0.0.1:$ROUTER_PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d "$(python3 -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}], "max_tokens": int(sys.argv[3]), "temperature": 0, "stream": sys.argv[4] == "1"}))' \
            "$SERVED" "$1" "$2" "${3:-0}")"
}

# fix_libionic NAME: if the image's libionic cannot drive the host's ionic NICs (no active port in the container),
# copy the host library in and point libionic.so.1 / the libibverbs provider at it (as Infera's _pd_1p1d.sh does).
fix_libionic() {
    local ctr="${CTR_PREFIX}$1" n
    n="$(docker exec "$ctr" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')"
    echo "$(ts) $ctr active RDMA ports before: ${n:-0}"
    if [ "${n:-0}" = 0 ]; then
        local hl b
        hl="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1)"; b="$(basename "$hl")"
        docker cp "$hl" "$ctr:/usr/lib/x86_64-linux-gnu/$b"
        docker exec "$ctr" bash -c "cd /usr/lib/x86_64-linux-gnu && ln -sf $b libionic.so.1 && ln -sf libionic.so.1 libionic.so && \
            mkdir -p libibverbs && cd libibverbs && ln -sf ../$b libionic-rdmav34.so && ldconfig 2>/dev/null; true"
        n="$(docker exec "$ctr" bash -c 'ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE')"
        echo "$(ts) $ctr host $b copied in; active RDMA ports after: ${n:-0}"
    fi
    [ "${n:-0}" -gt 0 ]
}

# pd_deploy PCFG DCFG P_EXTRA_ENV D_EXTRA_ENV [K=V ...]: etcd, the engine container and the prefill (port 30000) /
# decode (30001) workers; K=V are config inputs (e.g. MC_TOPO=pergpu). Needs MR_MODE. 0 once both serve.
pd_deploy() {
    local pcfg=$1 dcfg=$2 pex=$3 dex=$4; shift 4
    start_etcd || { say "etcd failed"; return 1; }
    start_engine_ctr eng || { say "engine container / infera install failed: $(tail -3 "$R/install-eng.log" | tr '\n' ' ')"; return 1; }
    GEM_SO=""
    if [ "$MR_MODE" = gem ]; then
        build_gem eng > "$R/gem-build.log" 2>&1 || { say "GEM allocator build failed: $(tail -3 "$R/gem-build.log" | tr '\n' ' ')"; return 1; }
    fi
    cp "$pcfg" "$R/prefill-config.sh"; cp "$dcfg" "$R/decode-config.sh"; cp "$W/configs/pd_mooncake_common.sh" "$R/"
    echo "P_EXTRA=$pex | D_EXTRA=$dex | $* | MR_MODE=$MR_MODE GEM_SO=$GEM_SO" > "$R/deploy-vars.txt"
    local t0=$SECONDS wp wd rp rd
    exec_bg eng prefill env MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" EXTRA_WORKER_ENV="$pex" "$@" bash "$W/scripts/ctr_worker.sh" "$pcfg"
    sleep 30
    exec_bg eng decode env MR_MODE="$MR_MODE" GEM_SO="$GEM_SO" EXTRA_WORKER_ENV="$dex" "$@" bash "$W/scripts/ctr_worker.sh" "$dcfg"
    wait_worker prefill 30000 7200 & wp=$!
    wait_worker decode 30001 7200 & wd=$!
    wait $wp; rp=$?; wait $wd; rd=$?
    if ((rp || rd)); then
        say "workers failed after $((SECONDS - t0)) s (prefill rc=$rp decode rc=$rd)"
        say "prefill: $(grep -aE 'Error|error:|Traceback|assert' "$R/prefill.log" | tail -4 | tr '\n' ' ' | head -c 800)"
        say "decode: $(grep -aE 'Error|error:|Traceback|assert' "$R/decode.log" | tail -4 | tr '\n' ' ' | head -c 800)"
        return 1
    fi
    say "workers ready in $((SECONDS - t0)) s"
}

# router_up LOGNAME [K=V ...]: (re)start the router with ctr_router.sh inputs (ROUTER_BACKEND, ROUTER_POLICY, ...).
router_up() {
    local name=$1; shift
    docker exec "${CTR_PREFIX}eng" pkill -f "infera-router|infera.server" 2>/dev/null; sleep 3
    exec_bg eng "$name" env "$@" bash "$W/scripts/ctr_router.sh"
    wait_router && sleep 10
}

# probe_set LABEL: 16k needle / GSM8K (64 requests, concurrency 16) and GSM8K (64, concurrency 16) through the router;
# counts the decode's "Timeout waiting for P side ready" pulls in the window.
probe_set() {
    local l=$1 t0 n
    t0=$(grep -ac "Timeout waiting for P side ready" "$R/decode.log")
    docker exec "${CTR_PREFIX}eng" timeout 1500 python3 "$W/scripts/longctx_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
        "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/$l-longctx16k.jsonl" --ctx-tokens 16000 --n 64 --concurrency 16 > "$R/$l-longctx16k.log" 2>&1
    say "$l longctx 16k: rows $(wc -l < "$R/$l-longctx16k.jsonl") errors $(grep -c '"error"' "$R/$l-longctx16k.jsonl") | $(grep -aE '^(needle|gsm8k): n=' "$R/$l-longctx16k.log" | tr '\n' ' ')"
    docker exec "${CTR_PREFIX}eng" timeout 1500 python3 "$W/scripts/gsm8k_probe.py" "$W/data/gsm8k_probe_prompts.jsonl" \
        "http://127.0.0.1:$ROUTER_PORT" "$SERVED" "$R/$l-gsm8k.jsonl" --limit 64 --concurrency 16 --max-tokens 8192 > "$R/$l-gsm8k.log" 2>&1
    n=$(grep -ac "Timeout waiting for P side ready" "$R/decode.log")
    say "$l gsm8k: $(tail -1 "$R/$l-gsm8k.log"); decode P-side-ready timeouts in this set: $((n - t0))"
}

nic_tx() { for d in /sys/class/infiniband/ionic_*; do printf '%s=%s ' "$(basename "$d")" "$(cat "$d/ports/1/counters/port_xmit_data" 2>/dev/null)"; done; }

collect_logs() {
    for c in $(docker ps -a --format '{{.Names}}' | grep "^$CTR_PREFIX" || true); do
        docker logs "$c" > "$R/docker-${c#$CTR_PREFIX}.log" 2>&1
    done
}
