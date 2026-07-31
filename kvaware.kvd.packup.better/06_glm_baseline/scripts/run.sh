#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 06 — GLM-5.2-MXFP4 two-node PD BASELINE. THE CONTROL.
#
#   kvaware OFF (--no-enable-kv-events)   kvd OFF   --router-policy round-robin
#
# This run exists to establish that the two-node mooncake-RDMA substrate is
# clean BEFORE any KV feature is switched on. Without a matched control, a
# later "it works with the features on" proves nothing — you cannot tell a
# working feature from a substrate that was fine all along.
#
#   prefill  chi2879  10.2.122.10  TP8  :30000  gmu 0.88
#   decode   chi2867  10.2.122.44  TP8  :30000  gmu 0.85
#   DP-attention symmetric on both legs (dp8/ep8). infera router :8100.
#   etcd on the prefill node :2379.
#
# Cost: ~6 min cold start + ~1 min probe, 16 GPUs across 2 nodes.
# Expected: probe.py 4/4, coherent text, MC_FORCE_TCP=0, "HIP dmabuf
# disabled" x8, and both workers advertising kv_events_endpoint=null.
#
# The script handles the traps automatically: libionic injection, the
# net.py port fix, `docker exec -d ... env ... bash /script` (never
# `bash -lc`), and a 6-minute-plus poll that does not mistake a cold start
# for a hang.
# ---------------------------------------------------------------------------
set -uo pipefail

JUMP="${JUMP:-root@149.28.124.225}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
PREFILL_HOST="${PREFILL_HOST:-chi2879}"; PREFILL_IP="${PREFILL_IP:-10.2.122.10}"
DECODE_HOST="${DECODE_HOST:-chi2867}";   DECODE_IP="${DECODE_IP:-10.2.122.44}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; ETCD_PORT="${ETCD_PORT:-2379}"
CTR="${CTR:-glm52_base06}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_base06}"   # MUST be on the shared FS
TAG="${TAG:-base}"
WAIT_MIN="${WAIT_MIN:-20}"          # cold start is ~6 min; poll generously
KEEP="${KEEP:-0}"                   # KEEP=1 leaves the deployment up
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){
  if [ "$KEEP" = "1" ]; then
    echo "[cleanup] KEEP=1 — leaving $CTR up on both nodes. Remove by hand:"
    echo "          docker rm -f $CTR ${CTR}_etcd"
    return
  fi
  echo "[cleanup] removing $CTR on both nodes"
  J "$PREFILL_HOST" "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
  J "$DECODE_HOST"  "docker rm -f $CTR >/dev/null 2>&1"
}
trap cleanup EXIT

echo "############################################################"
echo "# 06 BASELINE — kvaware OFF / kvd OFF / round-robin"
echo "#   prefill $PREFILL_HOST ($PREFILL_IP) TP8 gmu 0.88"
echo "#   decode  $DECODE_HOST ($DECODE_IP) TP8 gmu 0.85"
echo "############################################################"

# ---------------------------------------------------------------------------
# PREFLIGHT — fail loudly here rather than 6 minutes into a cold start.
# ---------------------------------------------------------------------------
echo
echo "===== PREFLIGHT ====="
fatal=0
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  img=$(J "$h" "docker image inspect $IMAGE --format '{{.Id}}' 2>/dev/null | head -c 24")
  [ -n "$img" ] && echo "  [ok]   $h image present ($img...)" \
                || { echo "  [FAIL] $h missing $IMAGE (docker save|ssh docker load it)"; fatal=1; }
  m=$(J "$h" "test -d $MODEL && echo yes")
  [ "$m" = "yes" ] && echo "  [ok]   $h model path visible" \
                   || { echo "  [FAIL] $h cannot see $MODEL — is /mnt/vast mounted?"; fatal=1; }
  # ip_local_port_range starting at 1024 breaks sglang's own port arithmetic.
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
# Without host libionic the RDMA path silently degrades to TCP and the whole
# "this ran on real RDMA" claim evaporates. Verified below: 8 PORT_ACTIVE.
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
for f in glm52_leg.sh probe.py net_fixed.py run_router.sh; do
  # Nested-ssh quoting mangles $f if you inline it; stage via stdin redirect
  # evaluated on the LOCAL shell, which is safe.
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f'" < "$HERE/$f"
done
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  # patch 0001: the image predates the randomised free_tcp_port_block. Harmless
  # here (kvaware is OFF so the function is unreachable) but applied anyway so
  # this control differs from the feature run in exactly the features.
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && \
          docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null && \
          docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null && echo $h staged"
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
echo "===== 3. legs — KVAWARE=0 KVD=0 (the control) ====="
# TRAP: `docker exec -d $CTR bash -lc '...'` does NOT persist. Use the
# `env VAR=... bash /script` form against a staged script file.
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=0 KVD=0 \
   LOG=$KIT/pd_prefill_$TAG.log bash /glm52_leg.sh"
sleep 20
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=0 KVD=0 \
   LOG=$KIT/pd_decode_$TAG.log bash /glm52_leg.sh"

echo
echo "===== 4. router on the prefill node, policy=round-robin ====="
sleep 10
# Staged script FILE, not `docker exec -d ... bash -lc '...'` — the detached
# login-shell form exits and takes the router with it, silently.
J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router.sh"

# ---------------------------------------------------------------------------
# WAIT — GLM-5.2 loads 408 GB. ~6 min is normal; this is NOT a hang.
# ---------------------------------------------------------------------------
echo
echo "===== 5. waiting for both legs (GLM-5.2 cold start ~6 min, NOT a hang) ====="
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
sleep 25   # let the router discover both workers via etcd

# ---------------------------------------------------------------------------
# MEASURE
# ---------------------------------------------------------------------------
echo
echo "===== EVIDENCE ====="
{
  echo "# Experiment 06 — BASELINE observed, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# kvaware OFF / kvd OFF / round-robin, GLM-5.2-MXFP4 two-node PD + DPA"
  echo
  echo "--- the switches, confirmed FROM THE WIRE (kv_events_endpoint must be null) ---"
  J "$PREFILL_HOST" "docker exec $CTR curl -s -m10 http://$PREFILL_IP:$ROUTER_PORT/v1/workers"
  echo
  echo "--- transport was REAL RDMA, not the MC_FORCE_TCP fallback ---"
  J "$PREFILL_HOST" "echo MC_FORCE_TCP_hits=\$(grep -ac 'MC_FORCE_TCP' $KIT/pd_prefill_$TAG.log); \
     echo HIP_dmabuf_disabled=\$(grep -ac 'HIP dmabuf disabled' $KIT/pd_prefill_$TAG.log)"
  echo "# expect MC_FORCE_TCP_hits=0 and HIP_dmabuf_disabled=8 (one per ionic rail)"
  echo
  echo "--- PD + DP-attention symmetric on both legs ---"
  for f in pd_prefill_$TAG pd_decode_$TAG; do
    echo "  $f:"
    J "$PREFILL_HOST" "grep -aoE \"disaggregation_mode='[a-z]+'|disaggregation_transfer_backend='[a-z]+'|enable_dp_attention=True|dp_size=8|ep_size=8\" $KIT/$f.log | sort -u | sed 's/^/    /'"
  done
  echo
  echo "--- the features really are OFF in the engine, not just on the command line ---"
  J "$PREFILL_HOST" "echo prefill_kvevents_cfg=\$(grep -ac 'kv-events-config' $KIT/pd_prefill_$TAG.log); \
     echo prefill_kvd_adapters=\$(grep -ac 'infera-kvd adapter connected' $KIT/pd_prefill_$TAG.log); \
     echo prefill_hier=\$(grep -aoE 'enable_hierarchical_cache=[A-Za-z]+' $KIT/pd_prefill_$TAG.log | sort -u)"
  echo "# expect 0 / 0 / enable_hierarchical_cache=False"
  echo
  echo "--- leg readiness ---"
  J "$PREFILL_HOST" "echo prefill_ready=\$(grep -ac 'ready to roll' $KIT/pd_prefill_$TAG.log); \
     echo decode_ready=\$(grep -ac 'ready to roll' $KIT/pd_decode_$TAG.log)"
  echo
  echo "--- CORRECTNESS: 4 temp=0 factual prompts through the router (>=3/4 required) ---"
  J "$PREFILL_HOST" "docker exec $CTR timeout 600 python3 /tmp/probe.py \
      http://$PREFILL_IP:$ROUTER_PORT $SERVED"
} | tee "$OUT/baseline_probe_4of4.observed.txt"

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
obs="$OUT/baseline_probe_4of4.observed.txt"
score=$(grep -oE '^[0-9]+/4 correct' "$obs" | head -1)
tcp=$(grep -oE 'MC_FORCE_TCP_hits=[0-9]+' "$obs" | head -1 | cut -d= -f2)
dma=$(grep -oE 'HIP_dmabuf_disabled=[0-9]+' "$obs" | head -1 | cut -d= -f2)
nulls=$(grep -oc '"kv_events_endpoint":null' "$obs" || true)
echo "  correctness ............ ${score:-<not reached>}   (need >=3/4)"
echo "  MC_FORCE_TCP hits ...... ${tcp:-?}                 (need 0 = real RDMA)"
echo "  HIP dmabuf disabled .... ${dma:-?}                 (need 8 = all ionic rails)"
echo "  kv_events_endpoint null  ${nulls:-0} worker(s)     (need 2 = both legs, kvaware off)"
echo
n=$(echo "${score:-0}" | cut -d/ -f1)
if [ "${n:-0}" -ge 3 ] && [ "${tcp:-1}" = "0" ] && [ "${dma:-0}" = "8" ]; then
  echo "  ==> PASS. The substrate is clean and it was real RDMA."
  echo "      Committed reference: results/baseline_probe_4of4.txt"
else
  echo "  ==> FAIL / INCONCLUSIVE. A control that did not come up clean is not a"
  echo "      control — do NOT proceed to a feature run against it. Compare with"
  echo "      results/baseline_probe_4of4.txt and results/transport_was_real_rdma.txt."
fi
