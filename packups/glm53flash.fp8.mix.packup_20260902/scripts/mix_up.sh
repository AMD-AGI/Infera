#!/bin/bash
# GLM-5.3-Flash FP8 MIX bring-up, TP4 on GPUs 4-7 of a SHARED node.
#
# SAFETY, and it is the point of this file:
#   * Teardown touches ONLY names starting yihou_f8_. Nothing else on this box
#     is ours: turbo.jax.zhuang12, xiaoming-dev, primus.zhuang12.* and their KFD
#     processes belong to a colleague and are never killed, stopped or removed.
#   * scripts/reset_gpus.sh is deliberately NOT called. There is nothing of ours
#     to reclaim and it must not go near a colleague's processes.
#   * GPUs 4-7 only.
#   * --mem-fraction-static is a fraction of TOTAL GPU memory, not free memory,
#     so it is computed here from a live measurement rather than hard-coded.
#     The vendor's 0.80 would claim ~230 GB per GPU on top of whatever the
#     colleague holds and can OOM THEIR job.
set -u
MY_IP="${MY_IP:-$(hostname -I | awk '{print $1}')}"
IMAGE="${IMAGE:-infera/engine-sglang:glm53-c821c425}"
# /apps/data/models is a symlink to /perf_apps/data/models and /perf_apps is a
# SEPARATE NFS mount -- bind the realpath, resolved now, or the dir is empty
# inside the container.
MODEL_HOST="$(readlink -f "${MODEL_HOST:-/apps/data/models/GLM-5.3-Flash}")"
MODEL_MOUNT="$(dirname "$MODEL_HOST")"
MODEL="$MODEL_HOST"
CTR="${CTR:-yihou_f8_mix}"
PORT="${PORT:-31400}"
ETCD_PORT="${ETCD_PORT:-23795}"
ROUTER_PORT="${ROUTER_PORT:-18105}"
# SELF is where mix_worker.sh lives. It is NOT derived from BASH_SOURCE, because
# this script is deliberately run from a /tmp copy: editing a script on NFS while
# bash is still reading it corrupts the rest of the run ("error reading input
# file: Stale file handle"), which cost one bring-up at the router step.
SELF="${SELF:-/apps/yihou/glm53.series.workspace_20260901/flash-fp8-0529}"
GPUS="${GPUS:-4,5,6,7}"
TP="${TP:-4}"
export GPUS   # the VRAM-measuring python in step 2 reads it from the environment

echo "===== 1. teardown OUR OWN containers only (prefix yihou_f8_) ====="
for name in "$CTR" "${CTR}_etcd"; do
  case "$name" in
    yihou_f8_*) docker rm -f "$name" >/dev/null 2>&1 && echo "  removed $name" ;;
    *) echo "  REFUSING to touch non-yihou_f8_ name: $name"; exit 1 ;;
  esac
done
# Wait for OUR OWN VRAM to actually drain before measuring. `docker rm -f`
# returns long before KFD releases the pages: measured 100.1 GiB per card still
# held ~10 s after teardown, which made the next launch size itself against its
# own corpse and pick 0.50 instead of 0.60. This waits only on our memory going
# DOWN; it never touches anyone else's process.
echo "  waiting for our own VRAM to drain..."
prev=-1
for i in $(seq 1 30); do
  cur=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import sys, json, os
d = json.load(sys.stdin)
m = 0
for g in os.environ["GPUS"].split(","):
    e = d.get("card" + g.strip(), {})
    m = max(m, float(next(v for k, v in e.items() if "Total Used Memory" in k)))
print(int(m / 2**30))')
  echo "    t+$((i*5))s: worst card $cur GiB"
  [ "$cur" -le 2 ] && break
  [ "$cur" = "$prev" ] && [ "$i" -ge 4 ] && { echo "    plateaued at $cur GiB -- that is someone else, proceeding"; break; }
  prev="$cur"
  sleep 5
done

echo "===== 2. free-VRAM measurement on GPUs $GPUS -> mem-fraction-static ====="
# Headroom policy: our claim + their current usage must stay under CAP of total.
GMU="${GMU:-}"
if [ -z "$GMU" ]; then
  # CEIL is deliberately 0.60, not the vendor's 0.80. Measured on this node:
  # TP4 weights are 306 GB / 4 = ~71 GiB per GPU, so 0.60 (~173 GiB) already
  # leaves ~100 GiB of KV per GPU -- far more than correctness work needs --
  # while leaving 40% of each card to the neighbours. This node oscillates: a
  # foreign `infera-engine-*` container appeared and took 161 GiB on two cards
  # for ~5 minutes in the middle of this bring-up, so a claim sized to the
  # instantaneous free reading is not safe. Headroom is cheap here; an OOM in
  # someone else's job is not.
  GMU=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c '
import sys, json, os
CAP = 0.85
CEIL = 0.60
gpus = [g.strip() for g in os.environ["GPUS"].split(",")]
d = json.load(sys.stdin)
worst = 0.0
for g in gpus:
    k = "card" + g
    e = d.get(k) or d.get("card" + g.zfill(2)) or {}
    tot = float(next(v for kk, v in e.items() if "Total Memory" in kk))
    use = float(next(v for kk, v in e.items() if "Total Used Memory" in kk))
    frac = use / tot
    print(f"  card{g}: used {use/2**30:.1f} GiB / {tot/2**30:.1f} GiB = {frac*100:.1f}%", file=sys.stderr)
    worst = max(worst, frac)
gmu = min(CEIL, CAP - worst)
print(f"{gmu:.2f}")
' ) || { echo "  ABORT: could not measure VRAM"; exit 1; }
fi
echo "  chosen --mem-fraction-static = $GMU (cap 0.85 total, minus worst-case colleague usage, capped at 0.60 CEIL)"
case "$GMU" in 0.[0-9]*) ;; *) echo "  ABORT: bad GMU '$GMU'"; exit 1 ;; esac
python3 -c "import sys; sys.exit(0 if float('$GMU') >= 0.45 else 1)" || {
  echo "  ABORT: only $GMU of total is safely available -- not enough for a TP$TP FP8 load. The colleague is using the node; wait."; exit 1; }

echo "===== 3. fresh container ====="
docker run -d --name "$CTR" --network=host --ipc=host --shm-size=64G \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
  -v "$MODEL_MOUNT":"$MODEL_MOUNT":ro \
  "$IMAGE" sleep infinity >/dev/null
sleep 5
echo "  image: $(docker inspect "$CTR" --format '{{.Image}}')"
docker exec "$CTR" ls "$MODEL/config.json" >/dev/null || { echo "  ABORT: model not visible in container"; exit 1; }
echo "  model visible: $MODEL"

echo "===== 4. etcd on :$ETCD_PORT ====="
# v3.5.14 has an empty ENTRYPOINT and Cmd=[/usr/local/bin/etcd]; passing `etcd`
# as argv[0] dumps usage and exits 2.
docker run -d --name "${CTR}_etcd" --network=host --entrypoint /usr/local/bin/etcd \
  quay.io/coreos/etcd:v3.5.14 \
  --advertise-client-urls "http://$MY_IP:$ETCD_PORT" \
  --listen-client-urls "http://0.0.0.0:$ETCD_PORT" \
  --listen-peer-urls "http://0.0.0.0:$((ETCD_PORT + 1))" \
  --initial-advertise-peer-urls "http://$MY_IP:$((ETCD_PORT + 1))" \
  --initial-cluster "default=http://$MY_IP:$((ETCD_PORT + 1))" >/dev/null
  # All three peer flags move together or none do. Overriding only
  # --listen-peer-urls leaves --initial-cluster at its default
  # (default=http://localhost:2380) while --initial-advertise-peer-urls is
  # derived from the detected host, and etcd exits 1 with "--initial-cluster
  # has default=http://localhost:2380 but missing from
  # --initial-advertise-peer-urls". The reference script got away with omitting
  # all of them because it used the default 2380; here 2379/2380 are not free.
sleep 5
curl -sf -m5 "http://$MY_IP:$ETCD_PORT/version" >/dev/null && echo "  etcd up" || echo "  ETCD FAILED"

echo "===== 5. stage + launch mix worker (TP$TP, GPUs $GPUS) ====="
docker cp "$SELF/mix_worker.sh" "$CTR":/mix_worker.sh >/dev/null
docker exec -d "$CTR" env MY_IP="$MY_IP" ETCD_IP="$MY_IP" MODEL="$MODEL" \
  PORT="$PORT" ETCD_PORT="$ETCD_PORT" LOG=/tmp/glm53_f8_mix.log \
  TP="$TP" GPUS="$GPUS" GMU="$GMU" \
  CUDA_GRAPH="${CUDA_GRAPH:-0}" GRAPH_BS="${GRAPH_BS:-1 2 4 8 16 24 32 48 64 96 128}" \
  KV_DTYPE="${KV_DTYPE:-bfloat16}" MOE_RUNNER="${MOE_RUNNER:-triton}" \
  DISABLE_SEF="${DISABLE_SEF:-0}" CTX="${CTX:-262144}" \
  bash /mix_worker.sh
echo "  worker launching -> /tmp/glm53_f8_mix.log in $CTR"

echo "===== 6. wait for worker /health ====="
ok=0
for i in $(seq 1 240); do
  if docker exec "$CTR" bash -c "curl -sf -m3 http://$MY_IP:$PORT/health" >/dev/null 2>&1; then
    echo "  worker serving after $((i*10))s"; ok=1; break
  fi
  if ! docker exec "$CTR" pgrep -f infera.engine.sglang >/dev/null 2>&1; then
    echo "  worker process gone -- last 40 lines:"
    docker exec "$CTR" tail -40 /tmp/glm53_f8_mix.log; exit 1
  fi
  sleep 10
done
[ "$ok" = 1 ] || { echo "  worker never became healthy"; exit 1; }

echo "===== 7. router (kv-aware) on :$ROUTER_PORT ====="
docker exec "$CTR" bash -c "printf '%s\n' '#!/bin/bash' 'exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT --discovery-backend etcd --etcd-endpoint $MY_IP:$ETCD_PORT --request-transport http --kv-event-transport zmq --router-policy kv-aware --router-tokenizer-path $MODEL --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0' > /run_router.sh && chmod +x /run_router.sh"
docker exec -d "$CTR" bash -c 'nohup /run_router.sh > /tmp/router.log 2>&1'
sleep 20
docker exec "$CTR" bash -c "curl -sf -m5 http://$MY_IP:$ROUTER_PORT/health >/dev/null && echo '  router healthy' || { echo '  router not ready'; tail -20 /tmp/router.log; }"

echo "===== up. endpoint: http://$MY_IP:$ROUTER_PORT ====="
