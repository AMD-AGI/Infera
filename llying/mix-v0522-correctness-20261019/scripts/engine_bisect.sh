#!/usr/bin/env bash
# Bisect the garbled-output defect of the aggregated GLM-5.2 recipe under concurrency.
# Starts etcd + one engine with the recipe of examples/sglang_mix_glm5.2/engine/worker.sh,
# minus the features switched off below, then runs GSM8K (EVAL_LIMIT questions at
# EVAL_CONCURRENCY, straight to the engine) and counts garbled answers.
#
# Usage (on the GPU node): engine_bisect.sh NAME [FUSE_TOPK=0] [HICACHE=0] [MTP=0] [DPA=0]
#                          [GATHERV=0] [DELAYER=0] [FUSION=0]
# Env: SITE (file exporting NODE_IP MODEL IMAGE), GPUS, PORT_OFFSET (0 or 100), EVAL_LIMIT=64
set -euo pipefail
name="$1"; shift
FUSE_TOPK=1 HICACHE=1 MTP=1 DPA=1 GATHERV=1 DELAYER=1 FUSION=1
for kv in "$@"; do export "$kv"; done
KIT=/home/liyingli/bench_agentx/baseline/Infera-mix-v0522/examples/sglang_mix_glm5.2
OUT_ROOT=/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect
set -a; . "$SITE"; set +a
off=${PORT_OFFSET:-0}
export WORK_DIR="$OUT_ROOT/$name" CTR="bisect_$name"
export ETCD_PORT=$((24601 + off)) ETCD_PEER_PORT=$((24602 + off)) ENGINE_PORT=$((27101 + off))
export KV_EVENTS_PORT=$((27657 + off)) KV_SNAPSHOT_PORT=$((28951 + off)) ROUTER_PORT=$((27101 + off))
source "$KIT/engine/common.sh"
mkdir -p "$WORK_DIR/runs/$name"
echo "$WORK_DIR/runs/$name" > "$WORK_DIR/current_run"
printf '%s\n' "name=$name FUSE_TOPK=$FUSE_TOPK HICACHE=$HICACHE(${HICACHE_POLICY:-write_through},${HICACHE_IO:-kernel},${HICACHE_LAYOUT:-page_first}) MTP=$MTP DPA=$DPA GATHERV=$GATHERV DELAYER=$DELAYER FUSION=$FUSION IMAGE=$IMAGE GPUS=$GPUS" | tee "$WORK_DIR/runs/$name/variant.txt"

docker run -d --name "$CTR-etcd" --network host --entrypoint /usr/local/bin/etcd "$ETCD_IMAGE" \
    --advertise-client-urls "http://$NODE_IP:$ETCD_PORT" --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
    --listen-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" --initial-advertise-peer-urls "http://127.0.0.1:$ETCD_PEER_PORT" \
    --initial-cluster "default=http://127.0.0.1:$ETCD_PEER_PORT" >/dev/null
wait_health "http://$NODE_IP:$ETCD_PORT/version" "$CTR-etcd"

nic="$(ip -o -4 addr show | awk -v ip="$NODE_IP" '$4 ~ ("^" ip "/") {print $2; exit}')"
env=("HIP_VISIBLE_DEVICES=$GPUS" "SGLANG_HOST_IP=$NODE_IP" "HOST_IP=$NODE_IP"
    "SGLANG_LOCAL_IP_NIC=$nic" "GLOO_SOCKET_IFNAME=$nic" "NCCL_SOCKET_IFNAME=$nic"
    NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 AITER_USE_FLYDSL_MOE_SORTING=1 PYTHONHASHSEED=0
    SAFETENSORS_FAST_GPU=1 SGLANG_OPT_USE_TOPK_V2=false SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0
    SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512 SGLANG_TIMEOUT_KEEP_ALIVE=900 INFERA_ENGINE_READY_TIMEOUT=3600)
[[ "$GATHERV" == 1 ]] && env+=(SGLANG_DP_USE_GATHERV=1) || env+=(SGLANG_DP_USE_GATHERV=0)
[[ "$FUSE_TOPK" == 1 ]] || env+=(SGLANG_DSA_FUSE_TOPK=0)
env_args=(); for kv in "${env[@]}"; do env_args+=(-e "$kv"); done

args=(--model-path "$MODEL" --served-model-name "$SERVED_MODEL" --trust-remote-code
    --host "$NODE_IP" --port "$ENGINE_PORT" --advertise-host "$NODE_IP"
    --discovery-backend etcd --etcd-endpoint "$NODE_IP:$ETCD_PORT" --request-transport http --kv-event-transport zmq
    --enable-kv-events --kv-events on --kv-events-bind "tcp://0.0.0.0:$KV_EVENTS_PORT" --kv-snapshot-port "$KV_SNAPSHOT_PORT"
    --tp-size 4 --ep-size 1 --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton
    --mem-fraction-static 0.85 --chunked-prefill-size 32768 --max-prefill-tokens 8192
    --max-running-requests 128 --cuda-graph-max-bs-decode 64
    --json-model-override-args '{"index_share_for_mtp_iteration":false}'
    --reasoning-parser glm45 --tool-call-parser glm47 --enable-metrics --enable-cache-report --watchdog-timeout 3600)
[[ "$DPA" == 1 ]] && args+=(--dp-size 4 --enable-dp-attention --enable-dp-lm-head)
[[ "$DELAYER" == 1 ]] && args+=(--enable-prefill-delayer --prefill-delayer-max-delay-passes 60)
[[ "$MTP" == 1 ]] && args+=(--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6)
[[ "$HICACHE" == 1 ]] && args+=(--enable-hierarchical-cache --hicache-size 64 --hicache-ratio 1.5
    --hicache-write-policy "${HICACHE_POLICY:-write_through}" --hicache-io-backend "${HICACHE_IO:-kernel}"
    --hicache-mem-layout "${HICACHE_LAYOUT:-page_first}")
[[ "$FUSION" == 1 ]] && args+=(--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion)

image_id="$(docker image inspect -f '{{.Id}}' "$IMAGE")"
# Two engines sharing --ipc host collide on RCCL's pid-named shared memory (container
# pids start low); PID_OFFSET burns pids first so the second engine's ranks differ.
cmd=(python3 -m infera.engine.sglang "${args[@]}")
if [[ -n "${PID_OFFSET:-}" ]]; then
    cmd=(bash -c "i=0; while [ \$i -lt $PID_OFFSET ]; do ( : ); i=\$((i+1)); done; exec \"\$@\"" _ "${cmd[@]}")
fi
docker run -d --name "$CTR-engine" --network host --ipc host --shm-size 32g \
    --device /dev/kfd --device /dev/dri --device /dev/infiniband --group-add video --group-add render \
    --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
    --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
    -v "$MODEL:$MODEL:ro" -v "glm52_mix-cache-${image_id:7:12}:/root/.cache" \
    -v "glm52_mix-aiter-jit-${image_id:7:12}:/sgl-workspace/aiter/aiter/jit" \
    "${env_args[@]}" "$IMAGE" "${cmd[@]}" >/dev/null
wait_health "$ENGINE_URL/health" "$CTR-engine"
echo "engine healthy $(date -u +%T)"

EVAL_LIMIT="${EVAL_LIMIT:-64}" EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}" bash "$KIT/engine/eval.sh" > /dev/null 2>&1 || true
f="$(ls "$WORK_DIR"/runs/"$name"/gsm8k/*/samples_gsm8k_*.jsonl | tail -1)"
python3 - "$f" "$WORK_DIR/runs/$name/gsm8k/eval.log" <<'EOF' | tee "$WORK_DIR/runs/$name/verdict.txt"
import json, re, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
strict = [r for r in rows if r["filter"] == "strict-match"]
garbled = sum(1 for r in strict if len(r["resps"][0][0]) > 8000 or "</think>" in r["resps"][0][0])
acc = sum(r["exact_match"] for r in strict) / len(strict)
print(f"n={len(strict)} strict_acc={acc:.3f} garbled={garbled}")
EOF
docker logs --timestamps "$CTR-engine" > "$WORK_DIR/runs/$name/engine.log" 2>&1 || true
docker rm -f "$CTR-engine" "$CTR-etcd" >/dev/null
echo "removed $CTR-engine $(date -u +%T)"
