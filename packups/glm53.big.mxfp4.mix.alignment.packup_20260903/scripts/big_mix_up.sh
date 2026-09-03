#!/bin/bash
# GLM-5.3 "big" MIX bring-up on this 8xMI355X node:
#   teardown (OURS ONLY) -> container -> etcd -> mix worker -> kv-aware router.
#
# THIS ARM IS THE SECOND ARM ON A SHARED NODE. Everything that can collide is
# moved off the leader arm and off the foreign services:
#   GPUs        4-7            (leader: 0-3)
#   etcd        :22379         (leader: 12379; a FOREIGN etcd holds 2379)
#   engine      :30010         (leader: 30000)
#   router      :8110          (leader: 8100)
#   kv events   :5567 / :8811  (leader: 5557 / 8801)
#   container   glm53_big_mix  (leader: glm53_mix*)
#
# NOT RUN. Review before executing.
set -u
# No apostrophe in this message: a `'` inside ${VAR:?...} opens a quote that
# swallows the rest of the script, and the failure surfaces dozens of lines later
# as an unrelated "unbound variable".
MY_IP="${MY_IP:?MY_IP=IP of this node}"
IMAGE="${IMAGE:?IMAGE=infera engine image tag built from deploy/docker/Dockerfile.sglang}"

VARIANT="${VARIANT:-mxfp4}"
case "$VARIANT" in
  fp8)   DEF_MODEL=/apps/data/models/GLM-5.3;       DEF_SERVED=glm-5.3-fp8   ;;
  mxfp4) DEF_MODEL=/apps/data/models/GLM-5.3-MXFP4; DEF_SERVED=glm-5.3-mxfp4 ;;
  *) echo "VARIANT must be fp8 or mxfp4, got: $VARIANT" >&2; exit 2 ;;
esac
MODEL="${MODEL:-$DEF_MODEL}"
SERVED="${SERVED:-$DEF_SERVED}"
MODEL_MOUNT="${MODEL_MOUNT:-$(dirname "$MODEL")}"
# A SECOND bind, for when $MODEL is a symlink farm pointing at the real weights.
# /apps/data/models is its OWN mount on this host and does NOT propagate into the
# container when you bind its parent /apps -- inside, /apps/data/models is simply
# a different, empty-looking directory and every symlink into it is dangling.
# That failure surfaces far downstream as "Unrecognized processing class", because
# config.json is the one real file left and the tokenizer/processor files vanish.
WEIGHTS_MOUNT="${WEIGHTS_MOUNT:-}"

CTR="${CTR:-glm53_big_mix}"
ROUTER_PORT="${ROUTER_PORT:-8110}"
PORT="${PORT:-30010}"
ETCD_PORT="${ETCD_PORT:-22379}"
TP="${TP:-4}"
GPUS="${GPUS:-4,5,6,7}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESET="${RESET:-$SELF/../scripts/reset_gpus.sh}"

echo "===== 0. plan ====="
echo "  variant=$VARIANT model=$MODEL served=$SERVED"
echo "  gpus=$GPUS tp=$TP  engine=:$PORT router=:$ROUTER_PORT etcd=:$ETCD_PORT  ctr=$CTR"

echo "===== 1. teardown OUR OWN containers only ====="
# Named explicitly. Never a pattern, never a variable-driven sweep: this node
# carries another user work and the leader arm, and neither is ours to remove.
docker rm -f glm53_big_mix glm53_big_mix_etcd >/dev/null 2>&1
sleep 5

echo "===== 2. confirm the GPUs WE want are released ====="
# Hard gate, not a best-effort sleep: starting the worker while a previous round
# still holds VRAM aborts the distributed bootstrap with a misleading
# "memory capacity is unbalanced" error.
#
# OWN_CTR_RE is deliberately NARROW -- it matches ONLY this arm containers. The
# default in reset_gpus.sh (glm53_[a-z0-9_]*) would also match the leader
# glm53_mix, and killing that is exactly the failure mode this node cannot
# afford. reset_gpus.sh aborts rather than touching anything foreign; an abort
# here means the node is busy, which is the correct outcome, not a bug.
[ -x "$RESET" ] || { echo "  ABORT: reset script not found/executable: $RESET"; exit 1; }
OWN_CTR_RE='^(glm53_big_mix|glm53_big_mix_etcd)$' GPUS="$GPUS" bash "$RESET" \
  || { echo "  ABORT: GPUs $GPUS not available"; exit 1; }

echo "===== 3. fresh container ====="
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT":ro \
  ${WEIGHTS_MOUNT:+-v "$WEIGHTS_MOUNT":"$WEIGHTS_MOUNT":ro} \
  "$IMAGE" sleep infinity >/dev/null
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"
# Prove the weights are really visible inside, before spending minutes loading.
docker exec "$CTR" bash -c "test -f '$MODEL/config.json' && ls '$MODEL' | wc -l" \
  || { echo "  ABORT: $MODEL/config.json not visible inside $CTR -- check WEIGHTS_MOUNT"; exit 1; }

echo "===== 3.5 flag preflight (seconds, vs minutes wasted on a 704 GB load) ====="
# The most likely first failure in this family is a flag this engine build does not
# accept. The vendor card validated on sglang v0.5.16; this image is based on
# v0.5.18. --dsa-* vs --nsa-* is the known trap (GLM-5.2 used --nsa-*), and
# --mm-feature-transport / --tool-call-parser / --cuda-graph-max-bs are the next
# most likely to have moved. Catch it HERE, not after the weights are read.
# PREFLIGHT=0 skips, e.g. if --help is unavailable in a stripped image.
if [ "${PREFLIGHT:-1}" = "1" ]; then
  HELP=$(docker exec "$CTR" python3 -m sglang.launch_server --help 2>&1 || true)
  if [ -z "$HELP" ]; then
    echo "  WARN: could not read --help; skipping flag preflight"
  else
    MISSING=""
    for f in --dsa-prefill-backend --dsa-decode-backend --kv-cache-dtype \
             --moe-runner-backend --mem-fraction-static --context-length \
             --max-running-requests --cuda-graph-max-bs --chunked-prefill-size \
             --max-prefill-tokens --reasoning-parser --tool-call-parser \
             --mm-feature-transport --enable-cache-report --ep-size \
             --disable-custom-all-reduce --chat-template --quantization \
             --disable-shared-experts-fusion; do
      case "$HELP" in *"$f"*) : ;; *) MISSING="$MISSING $f" ;; esac
    done
    if [ -n "$MISSING" ]; then
      echo "  ABORT: this engine build does not accept:$MISSING"
      case "$HELP" in
        *--nsa-prefill-backend*) echo "  NOTE: it DOES accept --nsa-*-backend -- this build wants the GLM-5.2 flag names." ;;
      esac
      exit 1
    fi
    echo "  all recipe flags accepted by this build"
  fi
fi

echo "===== 4. etcd on :$ETCD_PORT ====="
# etcd v3.5.14 image has an empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd];
# passing `etcd` as argv[0] dumps usage and exits 2. Use an explicit --entrypoint.
# 2379 is held by a FOREIGN etcd and 12379 by the leader. Never free a port by
# killing whoever holds it -- move ours instead.
# The PEER port must move too: 2380 on this node is held by the foreign etcd
# (verified with ss), so the etcd default peer port is not available to us.
# But moving --listen-peer-urls ALONE makes etcd fail to start, with
#   --initial-cluster has default=http://localhost:2380 but missing from
#   --initial-advertise-peer-urls=http://<ip>:22380
# because the derived advertise URL no longer matches the DEFAULT initial-cluster,
# which still names localhost:2380. All three must move together for a
# single-node cluster. First-hand: this cost one bring-up here.
ETCD_PEER_PORT="${ETCD_PEER_PORT:-$((ETCD_PORT + 1))}"
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  quay.io/coreos/etcd:v3.5.14 \
  --advertise-client-urls "http://$MY_IP:$ETCD_PORT" \
  --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
  --listen-peer-urls "http://0.0.0.0:$ETCD_PEER_PORT" \
  --initial-advertise-peer-urls "http://$MY_IP:$ETCD_PEER_PORT" \
  --initial-cluster "default=http://$MY_IP:$ETCD_PEER_PORT" >/dev/null
sleep 5
curl -sf -m5 "http://$MY_IP:$ETCD_PORT/version" >/dev/null && echo "  etcd up" || echo "  ETCD FAILED"

echo "===== 5. stage + launch mix worker ====="
docker cp "$SELF/big_mix_worker.sh" "$CTR":/big_mix_worker.sh >/dev/null
docker exec -d "$CTR" env \
  MY_IP="$MY_IP" ETCD_IP="$MY_IP" ETCD_PORT="$ETCD_PORT" \
  VARIANT="$VARIANT" MODEL="$MODEL" SERVED="$SERVED" PORT="$PORT" \
  LOG=/tmp/glm53_big_mix.log TP="$TP" GPUS="$GPUS" \
  QUANT="${QUANT:-}" MOE_RUNNER="${MOE_RUNNER:-}" KV_DTYPE="${KV_DTYPE:-fp8_e4m3}" \
  GMU="${GMU:-0.80}" CTX="${CTX:-262144}" \
  MAX_RUNNING="${MAX_RUNNING:-32}" CHUNK="${CHUNK:-65536}" \
  MAX_PREFILL="${MAX_PREFILL:-16384}" CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-32}" \
  DPA="${DPA:-0}" MTP="${MTP:-0}" HICACHE="${HICACHE:-0}" \
  KVAWARE="${KVAWARE:-1}" CUSTOM_AR="${CUSTOM_AR:-0}" VENDOR_ENV="${VENDOR_ENV:-0}" \
  SHARED_FUSION="${SHARED_FUSION:-}" \
  FUSED_DECODE_MLA="${FUSED_DECODE_MLA:-0}" \
  KV_PUB_PORT="${KV_PUB_PORT:-5567}" KV_SNAP_PORT="${KV_SNAP_PORT:-8811}" \
  bash /big_mix_worker.sh
echo "  mix worker launching -> /tmp/glm53_big_mix.log in $CTR"

echo "===== 6. wait for worker /health ====="
# Cold start is MINUTES, and longer here than for Flash: 408 GB (mxfp4) or 704 GB
# (fp8) of weights to read, plus graph capture. Silence is not a hang.
for i in $(seq 1 360); do
  if docker exec "$CTR" bash -c "curl -sf -m3 http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  worker serving after $((i * 10))s"; break
  fi
  if [ $((i % 12)) -eq 0 ]; then
    echo "  ... $((i * 10))s: $(docker exec "$CTR" bash -c 'tail -1 /tmp/glm53_big_mix.log' 2>/dev/null | cut -c1-140)"
  fi
  sleep 10
done

echo "===== 7. router (kv-aware) on :$ROUTER_PORT ====="
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

echo "===== up. endpoint: http://$MY_IP:$ROUTER_PORT  (served-model-name: $SERVED) ====="
echo "===== next: MY_IP=$MY_IP VARIANT=$VARIANT bash $SELF/big_smoke.sh ====="
