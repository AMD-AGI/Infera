#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 07 — GLM-5.2-MXFP4 two-node PD with kvaware ON + kvd ON.
#
# Same topology as the control, one thing changed: the features.
#
#   prefill  chi2879  10.2.122.10  TP8  :30000  gmu 0.88  KVAWARE=1 KVD=1
#   decode   chi2867  10.2.122.44  TP8  :30000  gmu 0.85  KVAWARE=1 KVD=1
#   DP-attention symmetric (dp8/ep8). infera router :8100 --router-policy kv-aware.
#   infera.kvd daemon on BOTH nodes (each engine talks to its LOCAL daemon).
#   etcd on the prefill node :2379.
#
# Cost: ~6 min cold start + ~1 min probe, 16 GPUs across 2 nodes.
# Expected: probe.py 4/4 (matching the switches-off run), 8 kvd adapters per
# leg, "KV plane up:" present, distinct kv-events endpoints, prefill allocating
# 8x 16.00 GB host memory --- AND kvd counters at ZERO. That last part is the
# honest result: wired and harmless, not useful. See doc/README.md.
#
# Traps handled automatically: libionic injection, patch-0001 net.py, the
# `docker exec -d ... bash -lc` pitfall (never used), --hicache-size bounding
# the host pool, and a poll long enough that a 6-min cold start is not read
# as a hang.
# ---------------------------------------------------------------------------
set -uo pipefail

JUMP="${JUMP:-root@149.28.124.225}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PREFILL_HOST="${PREFILL_HOST:-chi2879}"; PREFILL_IP="${PREFILL_IP:-10.2.122.10}"
DECODE_HOST="${DECODE_HOST:-chi2867}";   DECODE_IP="${DECODE_IP:-10.2.122.44}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; ETCD_PORT="${ETCD_PORT:-2379}"
POLICY="${POLICY:-kv-aware}"
HICACHE_GB="${HICACHE_GB:-16}"      # ABSOLUTE. The default ratio 2.0 blew up to
                                    # 355 GB/rank on a small model. Never rely on it.
KVD_SOCK="${KVD_SOCK:-/tmp/kvd/kvd.sock}"
CTR="${CTR:-glm52_kv07}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_kv07}"   # MUST be on the shared FS
TAG="${TAG:-kv}"
WAIT_MIN="${WAIT_MIN:-20}"
KEEP="${KEEP:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){
  if [ "$KEEP" = "1" ]; then
    echo "[cleanup] KEEP=1 — leaving $CTR up on both nodes (kvd daemon included)."
    echo "          Remove by hand: docker rm -f $CTR ${CTR}_etcd"
    return
  fi
  echo "[cleanup] removing $CTR on both nodes"
  J "$PREFILL_HOST" "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
  J "$DECODE_HOST"  "docker rm -f $CTR >/dev/null 2>&1"
}
trap cleanup EXIT

echo "############################################################"
echo "# 07 — kvaware ON / kvd ON / policy=$POLICY"
echo "#   prefill $PREFILL_HOST ($PREFILL_IP) TP8 gmu 0.88"
echo "#   decode  $DECODE_HOST ($DECODE_IP) TP8 gmu 0.85"
echo "############################################################"

# ---------------------------------------------------------------------------
# PREFLIGHT
# ---------------------------------------------------------------------------
echo
echo "===== PREFLIGHT ====="
fatal=0
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  img=$(J "$h" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null | head -c 24")
  [ -n "$img" ] && echo "  [ok]   $h image present ($img...)" \
                || { echo "  [FAIL] $h missing $IMAGE"; fatal=1; }
  m=$(J "$h" "test -d $MODEL && echo yes")
  [ "$m" = "yes" ] && echo "  [ok]   $h model path visible" \
                   || { echo "  [FAIL] $h cannot see $MODEL — is /mnt/vast mounted?"; fatal=1; }
  lo=$(J "$h" "cat /proc/sys/net/ipv4/ip_local_port_range | awk '{print \$1}'")
  [ "${lo:-0}" -ge 2048 ] 2>/dev/null && echo "  [ok]   $h ip_local_port_range low=$lo" \
                   || echo "  [WARN] $h ip_local_port_range low=$lo (<2048 has bitten this stack)"
done
[ "$fatal" = "1" ] && { echo "PREFLIGHT FAILED"; exit 1; }
J "$PREFILL_HOST" "mkdir -p $KIT" >/dev/null

# ---------------------------------------------------------------------------
# LAUNCH
# ---------------------------------------------------------------------------
echo
echo "===== 0. containers + libionic injection ====="
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
echo "===== 1. stage scripts (shared FS so both nodes see one copy) ====="
for f in glm52_leg.sh probe.py net_fixed.py run_kvd.sh kvdstats.sh run_router.sh; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f'" < "$HERE/$f"
done
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  # patch 0001. WITHOUT this the two legs get the SAME --kv-events-config base
  # (deterministically, not as a race) and the second one dies with
  # ZMQError: Address already in use. Mandatory once kvaware is on.
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && \
          docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null && \
          docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null && \
          docker cp $KIT/run_kvd.sh $CTR:/run_kvd.sh >/dev/null && \
          docker cp $KIT/kvdstats.sh $CTR:/kvdstats.sh >/dev/null && echo $h staged"
done
J "$PREFILL_HOST" "docker cp $KIT/run_router.sh $CTR:/run_router.sh >/dev/null"

echo
echo "===== 2. etcd on the prefill node ====="
J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd \
   --network=host quay.io/coreos/etcd:v3.5.14 etcd \
   --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT \
   --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 5; \
   curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version >/dev/null && echo etcd_up || echo ETCD_FAILED"

echo
echo "===== 3. kvd daemon on BOTH nodes (each engine talks to its LOCAL daemon) ====="
# TRAP: `docker exec -d $CTR bash -lc 'python3 -m infera.kvd ...'` silently does
# not persist. Run a STAGED SCRIPT FILE instead. This bit us twice.
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker exec -d $CTR bash /run_kvd.sh"
done
sleep 25
kvd_fatal=0
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  s=$(J "$h" "docker exec $CTR bash -c 'test -S $KVD_SOCK && echo ok || echo MISSING'")
  echo "  $h kvd socket: $s"
  [ "$s" = "ok" ] || { kvd_fatal=1; J "$h" "docker exec $CTR tail -20 /tmp/kvd.log"; }
done
[ "$kvd_fatal" = "1" ] && { echo "FATAL: kvd daemon did not come up"; exit 1; }

echo
echo "===== 4. legs — KVAWARE=1 KVD=1 (the configuration under test) ====="
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
   HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_prefill_$TAG.log bash /glm52_leg.sh"
sleep 20
# The legs are on DIFFERENT hosts, so 5557/8801 do not collide here. They would
# on one host — see doc/README.md.
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
   HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_decode_$TAG.log bash /glm52_leg.sh"

echo
echo "===== 5. router on the prefill node, policy=$POLICY ====="
sleep 10
# Staged script FILE, not `docker exec -d ... bash -lc '...'` — the detached
# login-shell form exits and takes the router with it, silently.
J "$PREFILL_HOST" "docker exec -d $CTR env POLICY=$POLICY ROUTER_PORT=$ROUTER_PORT \
   ETCD=$PREFILL_IP:$ETCD_PORT MODEL=$MODEL bash /run_router.sh"

# ---------------------------------------------------------------------------
# WAIT
# ---------------------------------------------------------------------------
echo
echo "===== 6. waiting for both legs (GLM-5.2 cold start ~6 min, NOT a hang) ====="
ready=0
for i in $(seq 1 $((WAIT_MIN * 6))); do
  sleep 10
  p=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_prefill_$TAG.log 2>/dev/null")
  d=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_decode_$TAG.log 2>/dev/null")
  pl=$(J "$PREFILL_HOST" "wc -l < $KIT/pd_prefill_$TAG.log 2>/dev/null")
  echo "  [$((i*10))s] prefill_ready=${p:-0} decode_ready=${d:-0} (prefill log ${pl:-0} lines — growing = loading)"
  [ "${p:-0}" = "1" ] && [ "${d:-0}" = "1" ] && { ready=1; echo "  both legs ready"; break; }
done
[ "$ready" = "1" ] || echo "  !! legs did not both report ready — evidence below will show why"
sleep 25

# ---------------------------------------------------------------------------
# MEASURE
# ---------------------------------------------------------------------------
echo
echo "===== EVIDENCE ====="
{
  echo "# Experiment 07 — kvaware ON / kvd ON, observed $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- the switches, confirmed FROM THE WIRE (kv_events_endpoint must be NON-null) ---"
  J "$PREFILL_HOST" "docker exec $CTR curl -s -m10 http://$PREFILL_IP:$ROUTER_PORT/v1/workers"
  echo
  echo "--- transport was REAL RDMA, not the MC_FORCE_TCP fallback ---"
  J "$PREFILL_HOST" "echo MC_FORCE_TCP_hits=\$(grep -ac 'MC_FORCE_TCP' $KIT/pd_prefill_$TAG.log); \
     echo HIP_dmabuf_disabled=\$(grep -ac 'HIP dmabuf disabled' $KIT/pd_prefill_$TAG.log)"
  echo "# expect 0 and 8"
  echo
  echo "--- PD + DP-attention symmetric on both legs ---"
  for f in pd_prefill_$TAG pd_decode_$TAG; do
    echo "  $f:"
    J "$PREFILL_HOST" "grep -aoE \"disaggregation_mode='[a-z]+'|disaggregation_transfer_backend='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8\" $KIT/$f.log | sort -u | sed 's/^/    /'"
  done
  echo
  echo "--- kvd wired on EVERY DP rank of BOTH legs (expect 8 and 8) ---"
  J "$PREFILL_HOST" "echo prefill_kvd_adapters=\$(grep -ac 'infera-kvd adapter connected' $KIT/pd_prefill_$TAG.log); \
     echo decode_kvd_adapters=\$(grep -ac 'infera-kvd adapter connected' $KIT/pd_decode_$TAG.log)"
  echo
  echo "--- infera's OWN KV probe plane attached (this line was once thought impossible) ---"
  J "$PREFILL_HOST" "grep -ah 'KV plane up:' $KIT/pd_prefill_$TAG.log $KIT/pd_decode_$TAG.log"
  echo
  echo "--- host pool bounded by --hicache-size (NOT the 2.0 ratio) ---"
  J "$PREFILL_HOST" "grep -ah 'Allocating .* host memory for hierarchical' $KIT/pd_prefill_$TAG.log"
  echo
  echo "--- the decode leg is only LEGAL because infera auto-appended the radix flag ---"
  J "$PREFILL_HOST" "echo decode_radix_flag=\$(grep -ac 'disaggregation-decode-enable-radix-cache' $KIT/pd_decode_$TAG.log); \
     echo decode_disable_radix=\$(grep -aoE 'disable_radix_cache=[A-Za-z]+' $KIT/pd_decode_$TAG.log | sort -u)"
  echo "# expect 1 and disable_radix_cache=False. Without the flag the decode leg"
  echo "# sets disable_radix_cache=True and hicache is rejected outright."
  echo
  echo "--- patch 0001 held: the two legs got DIFFERENT kv-events port bases ---"
  J "$PREFILL_HOST" "grep -aoE '\"endpoint\": \"tcp://\\*:[0-9]+\"' $KIT/pd_prefill_$TAG.log | sort -u"
  J "$PREFILL_HOST" "grep -aoE '\"endpoint\": \"tcp://\\*:[0-9]+\"' $KIT/pd_decode_$TAG.log | sort -u"
  echo "# identical numbers here = patch 0001 is NOT applied; the run is invalid."
  echo
  echo "--- kvd counters BEFORE the probe ---"
  for hn in "$PREFILL_HOST:$PREFILL_IP" "$DECODE_HOST:$DECODE_IP"; do
    h=${hn%%:*}; echo -n "  $h: "; J "$h" "docker exec $CTR bash /kvdstats.sh 2>/dev/null | tail -1"
  done
  echo
  echo "--- CORRECTNESS: 4 temp=0 factual prompts through the router (>=3/4 required) ---"
  J "$PREFILL_HOST" "docker exec $CTR timeout 600 python3 /tmp/probe.py \
      http://$PREFILL_IP:$ROUTER_PORT $SERVED"
  echo
  echo "--- kvd counters AFTER the probe ---"
  echo "# EXPECT THESE TO BE ZERO. Four short prefix-disjoint prompts give an"
  echo "# offload path nothing to do. That is the honest result of this run."
  for hn in "$PREFILL_HOST:$PREFILL_IP" "$DECODE_HOST:$DECODE_IP"; do
    h=${hn%%:*}; echo -n "  $h: "; J "$h" "docker exec $CTR bash /kvdstats.sh 2>/dev/null | tail -1"
  done
} | tee "$OUT/step1_kvaware_kvd_4of4.observed.txt"

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
obs="$OUT/step1_kvaware_kvd_4of4.observed.txt"
score=$(grep -oE '^[0-9]+/4 correct' "$obs" | head -1)
tcp=$(grep -oE 'MC_FORCE_TCP_hits=[0-9]+' "$obs" | head -1 | cut -d= -f2)
pa=$(grep -oE 'prefill_kvd_adapters=[0-9]+' "$obs" | head -1 | cut -d= -f2)
da=$(grep -oE 'decode_kvd_adapters=[0-9]+' "$obs" | head -1 | cut -d= -f2)
plane=$(grep -c 'KV plane up:' "$obs" || true)
eps=$(grep -oE '"endpoint": "tcp://\*:[0-9]+"' "$obs" | sort -u | wc -l)
echo "  correctness ............... ${score:-<not reached>}  (need >=3/4, and it should MATCH the switches-off run)"
echo "  MC_FORCE_TCP hits ......... ${tcp:-?}                (need 0)"
echo "  kvd adapters prefill/decode ${pa:-?} / ${da:-?}      (need 8 / 8 — one per DP rank)"
echo "  'KV plane up:' lines ...... ${plane:-0}              (need 2 — one per leg)"
echo "  distinct kv-events bases .. ${eps:-0}                (need 2 — patch 0001 holding)"
echo
n=$(echo "${score:-0}" | cut -d/ -f1)
if [ "${n:-0}" -ge 3 ] && [ "${tcp:-1}" = "0" ] && [ "${pa:-0}" = "8" ] && [ "${da:-0}" = "8" ]; then
  echo "  ==> PASS on correctness and wiring."
else
  echo "  ==> FAIL / INCONCLUSIVE. Compare with results/step1_kvaware_kvd_4of4.txt."
fi
cat <<'EOF'

  *** READ THIS BEFORE CITING THE RESULT ***

  A PASS here means "kvaware+kvd are wired correctly and do not break
  correctness". It does NOT mean kvd did anything useful. Check the two kvd
  counter blocks above: on the reference run they were identical and ALL ZERO
  (gets=0 sets=0 hits=0 entries=0). Four short prefix-disjoint prompts give a
  KV-offload path nothing to store and nothing to fetch.

  If your counters are also zero, that is the expected, correct outcome of THIS
  workload — not a failure, and not evidence of value either. Demonstrating that
  kvd serves needs a workload with a long shared prefix.
EOF
