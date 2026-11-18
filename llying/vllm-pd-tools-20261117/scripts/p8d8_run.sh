#!/usr/bin/env bash
# One two-node P8D8 scoring round (the sglang recipe's P8D8 C208 shape): this node runs etcd, the TP1 x PP8 prefill,
# the Rust router (serial vLLM Mooncake) and AgentX; the DP8 decode runs on the node whose runner reads $R/$QD (an item
# dropped there, waited for through its .rc). Same engine settings as p4d4_run.sh (keep-alive 120 s, offload descriptor
# cap 512, cache-aware admission 128 / 0, renderer threads, decode overlays); FAULT logging on both nodes' engines.
#   p8d8_run.sh TAG       (run directory runs/p8d8-TAG; run as a queue item on the prefill node, via scripts/snap.sh)
# Env: QD (decode node queue dir under R, e.g. queue-n4), WAIT_RC, DURATION (1800), CONC (208), CPU_BYTES (1.6e12),
#      D_MODE (v3comm | v2 | stock), P_RENDER / D_RENDER (4), D_UTIL (0.95), P_UTIL (0.6), P_ARGS_X / D_ARGS_X,
#      P_ENV_X / D_ENV_X, ROUTER_BIN (infera-router-vllmserial), PP_LAYER_PARTITION (10,10,10,10,10,10,10,8)
set -uo pipefail
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
cd "$R" || exit 1
tag=${1:?tag}
run=p8d8-$tag
QD=${QD:?decode node queue dir, e.g. queue-n4}
V=/usr/local/lib/python3.12/dist-packages/vllm
IMAGE=${IMAGE:-infera-vllm:v0340-0c5e5e89}
MODEL=${MODEL:-/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4}
IP=$(ip -o -4 route get 8.8.8.8 | awk '{for (i = 1; i < NF; i++) if ($i == "src") print $(i + 1)}')
if [[ -n "${WAIT_RC:-}" ]]; then
    while ! compgen -G "$WAIT_RC" > /dev/null; do sleep 20; ls "$(dirname "$WAIT_RC")" > /dev/null; done
fi
RUN_DIR=$R/runs/$run
mkdir -p "$RUN_DIR"
stop=$RUN_DIR/samples.jsonl.stop
rm -f "$stop"
# Drop a command on the decode node and wait for it; prints its log.
on_d() {
    local id="$R/$QD/$(date -u +%m%d-%H%M%S)-$1"; shift
    { echo "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=$HOME USER=$USER R=$R NODE0=\$(hostname -s)"
      echo "cd $R; $*"; } > "$id.tmp" && mv "$id.tmp" "$id.sh"
    while [ ! -e "$id.rc" ]; do sleep 5; ls "$R/$QD" > /dev/null; done
    cat "$id.log"; return "$(cat "$id.rc")"
}
shm_gb=$(awk '/^Shmem:/ {print int($2 / 1048576)}' /proc/meminfo)
if (( shm_gb > ${SHMEM_MAX_GB:-200} )); then
    echo "$(date -u +%T) $run: NOT STARTED, Shmem ${shm_gb} GB on the prefill node" | tee -a logs/summary.out; exit 0
fi
docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1
d_ip=$(on_d dprep 'docker rm -f vpd-prefill vpd-decode vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1; awk "/^Shmem:/ {print \"shmem_gb=\" int(\$2 / 1048576)}" /proc/meminfo; ip -o -4 route get 8.8.8.8 | awk "{for (i = 1; i < NF; i++) if (\$i == \"src\") print \"ip=\" \$(i + 1)}"' | sed -n 's/^ip=//p')
[[ -n "$d_ip" ]] || { echo "no decode node IP"; exit 0; }
echo "$(date -u +%T) $run: prefill node $(hostname -s) $IP, decode node $d_ip ($QD)"
docker image inspect quay.io/coreos/etcd:v3.5.14 > /dev/null 2>&1 || docker pull -q quay.io/coreos/etcd:v3.5.14
docker run -d --name vpd-etcd --network host --entrypoint /usr/local/bin/etcd quay.io/coreos/etcd:v3.5.14 \
    --advertise-client-urls "http://$IP:24701" --listen-client-urls "http://0.0.0.0:24701" \
    --listen-peer-urls http://127.0.0.1:24702 --initial-advertise-peer-urls http://127.0.0.1:24702 \
    --initial-cluster default=http://127.0.0.1:24702 > /dev/null || exit 0
fh="PYTHONFAULTHANDLER=1"
case "${D_MODE:-v3comm}" in
    v3comm) O=$R/overlay-v3comm/vllm
        d_env="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1"
        d_mount="$O/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $O/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro $O/distributed/device_communicators/cuda_communicator.py:$V/distributed/device_communicators/cuda_communicator.py:ro" ;;
    v2) d_env="VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1 VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4"
        d_mount="$R/overlay/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:$V/v1/attention/ops/rocm_aiter_mla_sparse.py:ro $R/overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:$V/v1/attention/ops/rocm_gluon_paged_mqa_logits.py:ro" ;;
    *) d_env="" d_mount="" ;;
esac
src=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/infera
d_args="--block-size 64 --gpu-memory-utilization ${D_UTIL:-0.95} --api-server-count 1 --renderer-num-workers ${D_RENDER:-4} ${D_ARGS_X:-}"
p_args="--gpu-memory-utilization ${P_UTIL:-0.6} --block-size 64 --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0 --api-server-count 1 --renderer-num-workers ${P_RENDER:-4} ${P_ARGS_X:-}"
cpu=${CPU_BYTES:-1600000000000}
p_kv="{\"kv_connector\":\"OffloadingConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$cpu,\"eviction_policy\":\"lru\"}}"
sim=${SIM_ACCEPT_LEN:-2.99}
for c in vpd-prefill vpd-router; do
    ( until docker inspect "$c" > /dev/null 2>&1; do [ -e "$stop" ] && exit 0; sleep 5; done
      docker logs -f "$c" > "$RUN_DIR/$c.stream.log" 2>&1 ) &
done
d_cmd="INFERA_SRC=$src GEM=1 SIM_ACCEPT_LEN=$sim D_ARGS=$(printf %q "$d_args") D_ENV=$(printf %q "$d_env VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 $fh ${D_ENV_X:-}") D_MOUNT_EXTRA=$(printf %q "$d_mount") bash scripts/pd_role_up.sh decode $run $IP; (docker logs -f vpd-decode > $RUN_DIR/vpd-decode.stream.log 2>&1 &); klog() { docker run --rm --cap-add SYSLOG --network none --entrypoint bash $IMAGE -c 'dmesg -T | tail -60'; }; klog > $RUN_DIR/dmesg-before-decode.txt 2>&1; exit 0"
on_d "Ld-$tag" "$d_cmd" > "$RUN_DIR/up-decode.log" 2>&1 &
d_pid=$!
INFERA_SRC=$src GEM=0 SIM_ACCEPT_LEN=$sim P_ARGS="$p_args" P_KV_EXTRA="$p_kv" \
    P_ENV="VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 $fh ${P_ENV_X:-}" \
    bash scripts/pd_role_up.sh prefill "$run" "$IP" > "$RUN_DIR/up-prefill.log" 2>&1
p_rc=$?
wait $d_pid
tail -3 "$RUN_DIR/up-prefill.log" "$RUN_DIR/up-decode.log"
if (( p_rc != 0 )) || ! grep -q "vpd-decode ready" "$RUN_DIR/up-decode.log"; then
    echo "$(date -u +%m%d-%T) score $run: deployment failed (prefill rc=$p_rc)" | tee -a logs/summary.out
    touch "$stop"; docker rm -f vpd-prefill vpd-router vpd-etcd > /dev/null 2>&1; on_d dclean 'docker rm -f vpd-decode > /dev/null 2>&1'; exit 0
fi
docker run -d --name vpd-router --network host --ulimit nofile=1048576:1048576 --cap-add SYS_PTRACE -v "$MODEL:$MODEL:ro" \
    -v "$src:/opt/infera/infera:ro" -v "$R/bin/${ROUTER_BIN:-infera-router-vllmserial}:/usr/local/bin/infera-router:ro" \
    -e INFERA_VLLM_MOONCAKE_SERIAL=1 -e "$fh" --entrypoint python3 "$IMAGE" -m infera.server \
    --host 0.0.0.0 --port 28300 --router-backend rust --discovery-backend etcd --etcd-endpoint "$IP:24701" \
    --request-transport http --kv-event-transport zmq --router-tokenizer-path "$MODEL" --router-policy kv-aware \
    --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill \
    --pd-prefill-guard-release completion --decode-input-demand on > /dev/null || exit 0
for i in $(seq 120); do
    modes=$(curl -sf -m 10 "http://$IP:28300/v1/workers" | python3 -c 'import json, sys; print(*sorted(w["disagg_mode"] for w in json.load(sys.stdin)["workers"]))' 2>/dev/null)
    [[ "$modes" == "decode prefill" ]] && break
    sleep 10
done
echo "http://$IP:28300" > "$RUN_DIR/router_url"
curl -sf -m 10 "http://$IP:28300/v1/workers" > "$RUN_DIR/workers.json"
echo "$(date -u +%T) router workers: ${modes:-none}"
murls="http://$IP:27201 http://$d_ip:27301"
( python3 scripts/metrics_sampler.py "$RUN_DIR/samples.jsonl" 10 9000 $murls ) &
( bash scripts/hang_watch.sh "http://$IP:27201" vpd-prefill "$RUN_DIR/hang" 180 9000 ) &
rc=1
if [[ "$modes" == "decode prefill" ]]; then
    PREFILL_NUM_WORKERS=1 timeout 900 python3 scripts/probes/smoke.py "http://$IP:28300" glm5.2-mxfp4 "$RUN_DIR/smoke" > "$RUN_DIR/smoke.out" 2>&1
    if grep -q '"name": "long_prefix_repeat"' "$RUN_DIR/smoke.out"; then
        # shellcheck disable=SC2086
        python3 scripts/metrics_snap.py snap "$RUN_DIR/m0.json" $murls
        CONC=${CONC:-208} DURATION=${DURATION:-1800} SIM_ACCEPT_LEN=$sim PREFILL_TP=8 DECODE_TP=8 KV_OFFLOADING=dram \
            KV_OFFLOAD_BACKEND=vllm-native bash scripts/agentx.sh "$run"; rc=$?
        # shellcheck disable=SC2086
        python3 scripts/metrics_snap.py snap "$RUN_DIR/m1.json" $murls
        python3 scripts/metrics_snap.py diff "$RUN_DIR/m0.json" "$RUN_DIR/m1.json" > "$RUN_DIR/metrics-delta.json"
        out=$RUN_DIR/agentx-c${CONC:-208}; tmp=$(mktemp -d)
        ln -s "$out"/agentx_conc*.json "$tmp/"; ln -s "$out/aiperf_artifacts/profile_export_aiperf.json" "$tmp/"
        row=$(python3 scripts/summarize_points.py "$tmp" 2>&1 | tail -1); rm -rf "$tmp"
        echo "$(date -u +%m%d-%T) score $run C${CONC:-208} ${DURATION:-1800}s rc=$rc: $row; metrics $(cat "$RUN_DIR/metrics-delta.json")" | tee -a logs/summary.out
    else
        echo "$(date -u +%m%d-%T) score $run: smoke did not complete ($(tail -1 "$RUN_DIR/smoke.out"))" | tee -a logs/summary.out
    fi
fi
touch "$stop"
for c in vpd-prefill vpd-router; do
    docker inspect -f "$c {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}" "$c" 2>&1
done > "$RUN_DIR/container-state.txt"
docker run --rm --cap-add SYSLOG --network none --entrypoint bash "$IMAGE" -c 'dmesg -T | tail -300' > "$RUN_DIR/dmesg-after-prefill.txt" 2>&1
on_d dclean "docker inspect -f 'vpd-decode {{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' vpd-decode >> $RUN_DIR/container-state.txt 2>&1; docker run --rm --cap-add SYSLOG --network none --entrypoint bash $IMAGE -c 'dmesg -T | tail -300' > $RUN_DIR/dmesg-after-decode.txt 2>&1; docker rm -f vpd-decode > /dev/null 2>&1; sleep 20; for f in $RUN_DIR/vpd-decode.stream.log; do [ -s \$f ] && gzip -f \$f; done; exit 0" > /dev/null
docker rm -f vpd-prefill vpd-router vpd-etcd vpd-agentx > /dev/null 2>&1; sleep 30
for c in vpd-prefill vpd-router; do [ -s "$RUN_DIR/$c.stream.log" ] && gzip -f "$RUN_DIR/$c.stream.log"; done
exit 0
