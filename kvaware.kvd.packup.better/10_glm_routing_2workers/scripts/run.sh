#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 10 — does the kv-aware router actually ROUTE?
#
# Earlier runs could confirm the role weights were LOADED but never what they
# DID: with one prefill worker and one decode worker the scorer has no
# alternative to choose between. It can be observed but not tested.
#
# This adds a SECOND decode worker so a routing decision can exist, then runs
# the SAME workload under two policies and counts where the work landed.
#
#   prefill  chi2879  10.2.122.10  TP8 (GPU0-7)  :30000  gmu 0.88
#   decodeA  chi2867  10.2.122.44  TP4 (GPU0-3)  :30000  gmu 0.70
#                                   KV_PUB_PORT 5557  KV_SNAP_PORT 8801
#   decodeB  chi2867  10.2.122.44  TP4 (GPU4-7)  :32000  gmu 0.70
#                                   KV_PUB_PORT 5657  KV_SNAP_PORT 8802
#
# THREE THINGS THAT MUST DIFFER PER CO-LOCATED WORKER, and they are unrelated
# code paths — fixing one does nothing for the others:
#     PORT           30000 / 32000
#     KV_PUB_PORT    5557 / 5657     (--kv-events-bind)
#     KV_SNAP_PORT   8801 / 8802     (--kv-snapshot-port)
# The third is the nastiest: on a collision the worker logs "ready to roll" and
# THEN dies during etcd registration, so it looks healthy and simply never
# appears in /v1/workers. This script sets all three and verifies the worker
# count, which is the only way to catch it.
#
# GMU 0.70, NOT 0.85: TP4 doubles GLM-5.2's per-GPU weights to ~102 GB and 0.85
# OOMs on a ~390 MiB allocation.
#
# Cost: ~15 min (prefill + two TP4 decode cold starts) + ~8 min for two arms.
# ---------------------------------------------------------------------------
set -uo pipefail

JUMP="${JUMP:-root@149.28.124.225}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PREFILL_HOST="${PREFILL_HOST:-chi2879}"; PREFILL_IP="${PREFILL_IP:-10.2.122.10}"
DECODE_HOST="${DECODE_HOST:-chi2867}";   DECODE_IP="${DECODE_IP:-10.2.122.44}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; ETCD_PORT="${ETCD_PORT:-2379}"
HICACHE_GB="${HICACHE_GB:-16}"
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
DECODE_TP="${DECODE_TP:-4}"
DECODE_GMU="${DECODE_GMU:-0.70}"     # NOT 0.85 — see header
SESSIONS="${SESSIONS:-4}"
CTR="${CTR:-glm52_rt10}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_rt10}"
WAIT_MIN="${WAIT_MIN:-25}"
KEEP="${KEEP:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

# The measurement: how many decode batches each worker ran.
decode_batches(){ J "$PREFILL_HOST" "grep -ac 'Decode batch' $KIT/$1 2>/dev/null" | tr -dc 0-9; }

cleanup(){
  if [ "$KEEP" = "1" ]; then echo "[cleanup] KEEP=1 — deployment left up."; return; fi
  echo "[cleanup] removing $CTR on both nodes"
  J "$PREFILL_HOST" "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
  J "$DECODE_HOST"  "docker rm -f $CTR >/dev/null 2>&1"
}
trap cleanup EXIT

echo "############################################################"
echo "# 10 — kv-aware routing with TWO decode workers"
echo "############################################################"

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
echo
echo "===== PREFLIGHT ====="
fatal=0
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  img=$(J "$h" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null | head -c 24")
  [ -n "$img" ] && echo "  [ok]   $h image present" || { echo "  [FAIL] $h missing $IMAGE"; fatal=1; }
  m=$(J "$h" "test -d $MODEL && echo yes")
  [ "$m" = "yes" ] && echo "  [ok]   $h model visible" || { echo "  [FAIL] $h cannot see $MODEL"; fatal=1; }
done
g=$(J "$DECODE_HOST" "rocm-smi --showid --csv 2>/dev/null | grep -c '^card'")
echo "  [info] $DECODE_HOST GPUs visible: ${g:-?} (need >= $((DECODE_TP * 2)) for two TP$DECODE_TP workers)"
# TP-dependent memfrac guard. 0.85 is a TP8 number; at TP4 it OOMs.
if [ "$DECODE_TP" -le 4 ] && awk "BEGIN{exit !($DECODE_GMU > 0.75)}"; then
  echo "  [FAIL] DECODE_GMU=$DECODE_GMU at TP=$DECODE_TP will OOM."
  echo "         GLM-5.2 is 408 GB: TP8 = 51 GB/GPU of weights, TP4 = 102 GB/GPU."
  echo "         Use 0.70 at TP4. Override with DECODE_GMU= if you know better."
  fatal=1
else
  echo "  [ok]   DECODE_GMU=$DECODE_GMU appropriate for TP=$DECODE_TP"
fi
[ "$fatal" = "1" ] && { echo "PREFLIGHT FAILED"; exit 1; }
J "$PREFILL_HOST" "mkdir -p $KIT" >/dev/null

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
echo
echo "===== 0. containers + libionic ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host --ipc=host \
     --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
     --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
     --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
     -v /mnt/vast:/mnt/vast --entrypoint \"\" $IMAGE sleep infinity >/dev/null"
  J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); \
     docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B >/dev/null; \
     docker exec $CTR bash -c \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && \
       ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && \
       ldconfig 2>/dev/null; echo $h active_ports=\\\$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)\""
done

echo
echo "===== 1. stage scripts ====="
for f in glm52_leg.sh probe.py net_fixed.py run_kvd.sh kvdstats.sh prefix_reuse.py \
         run_router_weighted.sh run_router_roundrobin.sh; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f'" < "$HERE/$f"
done
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  # patch 0001 is MANDATORY here: two workers on ONE host with kvaware on.
  # Without it both get the same --kv-events-config base and decodeB dies.
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && \
          docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null && \
          docker cp $KIT/run_kvd.sh $CTR:/run_kvd.sh >/dev/null && \
          docker cp $KIT/kvdstats.sh $CTR:/kvdstats.sh >/dev/null && \
          docker cp $KIT/prefix_reuse.py $CTR:/tmp/prefix_reuse.py >/dev/null && \
          docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null && echo $h staged"
done
J "$PREFILL_HOST" "docker cp $KIT/run_router_weighted.sh $CTR:/run_router_weighted.sh >/dev/null; \
                   docker cp $KIT/run_router_roundrobin.sh $CTR:/run_router_roundrobin.sh >/dev/null"

echo
echo "===== 2. etcd ====="
J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd \
   --network=host quay.io/coreos/etcd:v3.5.14 etcd \
   --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT \
   --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 5; \
   curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version >/dev/null && echo etcd_up || echo ETCD_FAILED"

echo
echo "===== 3. kvd daemon on both nodes ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker exec -d $CTR bash /run_kvd.sh"; done
sleep 25
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  s=$(J "$h" "docker exec $CTR bash -c 'test -S $KVD_SOCK && echo ok || echo MISSING'")
  echo "  $h kvd socket: $s"
  [ "$s" = "ok" ] || { J "$h" "docker exec $CTR tail -20 /tmp/kvd.log"; echo "FATAL kvd"; exit 1; }
done

echo
echo "===== 4. prefill leg (TP8, whole node) ====="
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 TP=8 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
   HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_prefill.log bash /glm52_leg.sh"

echo
echo "===== 5. TWO decode workers on one host — ALL THREE ports differ ====="
sleep 20
# decodeA: GPU0-3, :30000, pub 5557, snap 8801
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 TP=$DECODE_TP BASE_GPU=0 GMU=$DECODE_GMU \
   KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
   KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_decodeA.log bash /glm52_leg.sh"
sleep 20
# decodeB: GPU4-7, :32000, pub 5657, snap 8802  <- every one of the three MUST differ
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=32000 TP=$DECODE_TP BASE_GPU=$DECODE_TP GMU=$DECODE_GMU \
   KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK HICACHE_GB=$HICACHE_GB \
   KV_PUB_PORT=5657 KV_SNAP_PORT=8802 \
   LOG=$KIT/pd_decodeB.log bash /glm52_leg.sh"

echo
echo "===== 6. router (kv-aware, role weights 20.0 / 2.0) ====="
sleep 10
J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router_weighted.sh"

echo
echo "===== 7. waiting for all THREE legs (~8-15 min; NOT a hang) ====="
ready=0
for i in $(seq 1 $((WAIT_MIN * 6))); do
  sleep 10
  p=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_prefill.log 2>/dev/null")
  a=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_decodeA.log 2>/dev/null")
  b=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_decodeB.log 2>/dev/null")
  echo "  [$((i*10))s] prefill=${p:-0} decodeA=${a:-0} decodeB=${b:-0}"
  [ "${p:-0}" = "1" ] && [ "${a:-0}" = "1" ] && [ "${b:-0}" = "1" ] && { ready=1; break; }
done
[ "$ready" = "1" ] && echo "  all three legs report ready" || echo "  !! not all legs ready"
sleep 30

# ---------------------------------------------------------------------------
# THE CHECK THAT CATCHES THE 8801 COLLISION
# 'ready to roll' is NOT sufficient. A worker that lost the --kv-snapshot-port
# race prints it and THEN dies during etcd registration.
# ---------------------------------------------------------------------------
echo
echo "===== 8. worker count — the ONLY way to catch the snapshot-port collision ====="
workers=$(J "$PREFILL_HOST" "docker exec $CTR curl -s -m10 http://$PREFILL_IP:$ROUTER_PORT/v1/workers")
echo "$workers"
n=$(echo "$workers" | grep -o '"url"' | wc -l)
echo "  router sees $n worker(s) — NEED 3 (1 prefill + 2 decode)"
if [ "${n:-0}" -lt 3 ]; then
  echo
  echo "  !! FEWER THAN 3 WORKERS. Check for the snapshot-port failure signature:"
  J "$DECODE_HOST" "docker exec $CTR bash -c \"grep -a 'address already in use\\|Errno 98\\|STARTUP_FAILURE' $KIT/pd_decodeA.log $KIT/pd_decodeB.log | tail -10\""
  echo "  A worker that logged 'ready to roll' and then hit [Errno 98] on 8801"
  echo "  is alive-looking and absent. Give it its own KV_SNAP_PORT."
  echo "  WITHOUT 2 decode workers there is NO routing decision to measure and"
  echo "  this experiment cannot produce a result."
fi

# ---------------------------------------------------------------------------
# run_arm <policy-label>
# ---------------------------------------------------------------------------
run_arm(){
  local label="$1"
  echo
  echo "############################################################"
  echo "# ARM: $label"
  echo "############################################################"
  local a0 b0 a1 b1
  a0=$(decode_batches pd_decodeA.log); b0=$(decode_batches pd_decodeB.log)
  echo "  decode-batch counters BEFORE: decodeA=${a0:-0} decodeB=${b0:-0}"

  {
    echo "# ===== ARM $label — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- router policy, from the router's own log ---"
    J "$PREFILL_HOST" "docker exec $CTR grep -a 'router-policy=' /tmp/router.log | tail -1"
    echo
    echo "--- decode-batch counters BEFORE: decodeA=${a0:-0} decodeB=${b0:-0} ---"
    echo
    echo "--- SAME workload as the other arm ---"
    J "$PREFILL_HOST" "docker exec $CTR timeout 1800 python3 /tmp/prefix_reuse.py \
        http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions $SESSIONS"
  } | tee "$OUT/routing_${label}.observed.txt"

  a1=$(decode_batches pd_decodeA.log); b1=$(decode_batches pd_decodeB.log)
  {
    echo
    echo "--- decode-batch counters AFTER: decodeA=${a1:-0} decodeB=${b1:-0} ---"
    echo "--- DISTRIBUTION THIS ARM: decodeA=$(( ${a1:-0} - ${a0:-0} ))  decodeB=$(( ${b1:-0} - ${b0:-0} ))"
  } | tee -a "$OUT/routing_${label}.observed.txt"
  echo "  distribution: decodeA=$(( ${a1:-0} - ${a0:-0} ))  decodeB=$(( ${b1:-0} - ${b0:-0} ))"
  eval "DIST_${label}_A=$(( ${a1:-0} - ${a0:-0} ))"
  eval "DIST_${label}_B=$(( ${b1:-0} - ${b0:-0} ))"
}

run_arm kvaware

echo
echo "===== switching the router to round-robin — ONLY the policy changes ====="
echo "===== (same legs, same workload, same warm caches) ====="
J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router_roundrobin.sh"
sleep 30

run_arm roundrobin

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
echo "  policy        decodeA   decodeB"
echo "  ------------  -------   -------"
printf "  kv-aware      %7s   %7s\n" "${DIST_kvaware_A:-?}" "${DIST_kvaware_B:-?}"
printf "  round-robin   %7s   %7s\n" "${DIST_roundrobin_A:-?}" "${DIST_roundrobin_B:-?}"
cat <<'EOF'

  Reference result (2026-07-30):
      kv-aware (prefill 20.0 / decode 2.0)    decodeA 17   decodeB 0
      round-robin                             decodeA +6   decodeB +8

  Same workload, same two workers, flip the policy and the distribution
  inverts. That differential is the result — the scorer is not merely
  instantiated, it is DECIDING.

  ALL-TO-ONE IS CORRECT HERE, not a load-balancing bug. Every request in this
  workload shares one prefix, decodeA holds it, and
      cost = w * (request_blocks - hits) + active_blocks
  with w=2.0 on the decode side makes the cache-locality term dominate the load
  term at this concurrency. Sticking to the worker that already has the prefix
  is exactly the point of the policy.

  WHAT IT DOES NOT SHOW:
    * the PREFILL weight (20.0) — still unmeasured, there is only one prefill
      worker;
    * CORRECT affinity — one prefix only, so this is affinity, not "each prefix
      lands on its own worker";
    * the balance point — requests were sequential, so the load term never
      pushed back against locality.

  Committed reference: results/step4_role_weights_routing.txt
EOF
