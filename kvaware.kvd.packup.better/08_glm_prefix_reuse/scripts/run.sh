#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 08 — does kvd actually SERVE, or does it only connect?
#
# A previous run had kvaware+kvd wired on every DP rank of both legs, scored
# 4/4 — and kvd's counters were all zero. Four short prefix-disjoint prompts
# give a KV-offload path nothing to do. This experiment supplies a workload
# with real prefix reuse and reads the daemon's own counters around it.
#
# Workload: scripts/prefix_reuse.py — 4 sessions x 4 turns, every turn sharing
# a ~6200-token system prefix, run twice (cold phase + reuse phase) = 32 reqs.
#
#   prefill  chi2879  10.2.122.10  TP8  :30000  gmu 0.88  KVAWARE=1 KVD=1
#   decode   chi2867  10.2.122.44  TP8  :30000  gmu 0.85  KVAWARE=1 KVD=1
#   DP-attention symmetric (dp8/ep8). infera router :8100.
#   infera.kvd daemon on BOTH nodes. etcd on the prefill node :2379.
#
# Two router arms, both run by default:
#   ARM=default   --router-policy kv-aware, role weights left at 1.0/1.0
#   ARM=weighted  + --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0
#
# Cost: ~6 min cold start + ~4 min per arm, 16 GPUs across 2 nodes.
#
# *** THE TRAP THIS EXPERIMENT EXISTS TO WARN ABOUT ***
# The warm run is ~2.7x faster. That speedup is the GPU radix cache, NOT kvd —
# kvd's counters do not move during it. The script prints the counters around
# EVERY phase precisely so the attribution cannot be fudged. Read the verdict.
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
ARMS="${ARMS:-both}"            # both | default | weighted
SESSIONS="${SESSIONS:-4}"
CTR="${CTR:-glm52_px08}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_px08}"
TAG="${TAG:-px}"
WAIT_MIN="${WAIT_MIN:-20}"
KEEP="${KEEP:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

case "$ARMS" in both|default|weighted) ;; *) echo "ARMS must be both|default|weighted"; exit 2 ;; esac

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

# kvd counters from the daemon itself — NOT from the engine log, which never
# contains them.
kvd_stats(){ J "$1" "docker exec $CTR bash /kvdstats.sh 2>/dev/null | tail -1"; }

snap(){  # snap <label>
  echo "  [$1] prefill(chi2879): $(kvd_stats "$PREFILL_HOST")"
  echo "  [$1] decode (chi2867): $(kvd_stats "$DECODE_HOST")"
}

cleanup(){
  if [ "$KEEP" = "1" ]; then
    echo "[cleanup] KEEP=1 — deployment left up (kvd daemons still hold their entries)."
    return
  fi
  echo "[cleanup] removing $CTR on both nodes"
  J "$PREFILL_HOST" "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
  J "$DECODE_HOST"  "docker rm -f $CTR >/dev/null 2>&1"
}
trap cleanup EXIT

echo "############################################################"
echo "# 08 — prefix-reuse workload, ARMS=$ARMS"
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
  lo=$(J "$h" "cat /proc/sys/net/ipv4/ip_local_port_range | awk '{print \$1}'")
  [ "${lo:-0}" -ge 2048 ] 2>/dev/null && echo "  [ok]   $h port_range low=$lo" \
                   || echo "  [WARN] $h port_range low=$lo (<2048 has bitten this stack)"
done
# The workload's prefix must clear the L3 prefetch threshold or the offload path
# is skipped outright. Check it locally before spending a cold start.
python3 - "$HERE/prefix_reuse.py" <<'PY'
import re, sys
src = open(sys.argv[1]).read()
ns = {}
exec(compile(src.split("def ask(")[0], "prefix", "exec"), ns)
n = len(ns["PREFIX"])
print(f"  [ok]   workload prefix = {n} chars (~{n//4} tokens est.)"
      if n // 4 > 256 else
      f"  [FAIL] workload prefix only ~{n//4} tokens — below the 256 default"
      " prefetch threshold; nothing will be offloaded")
PY
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
echo "===== 1. stage scripts ====="
for f in glm52_leg.sh probe.py net_fixed.py run_kvd.sh kvdstats.sh prefix_reuse.py \
         run_router.sh run_router_weighted.sh; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f'" < "$HERE/$f"
done
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && \
          docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null && \
          docker cp $KIT/run_kvd.sh $CTR:/run_kvd.sh >/dev/null && \
          docker cp $KIT/kvdstats.sh $CTR:/kvdstats.sh >/dev/null && \
          docker cp $KIT/prefix_reuse.py $CTR:/tmp/prefix_reuse.py >/dev/null && \
          docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null && echo $h staged"
done
J "$PREFILL_HOST" "docker cp $KIT/run_router.sh $CTR:/run_router.sh >/dev/null; \
                   docker cp $KIT/run_router_weighted.sh $CTR:/run_router_weighted.sh >/dev/null"

echo
echo "===== 2. etcd ====="
J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd \
   --network=host quay.io/coreos/etcd:v3.5.14 etcd \
   --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT \
   --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 5; \
   curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version >/dev/null && echo etcd_up || echo ETCD_FAILED"

echo
echo "===== 3. kvd daemon on BOTH nodes (staged FILE — 'bash -lc' does not persist) ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker exec -d $CTR bash /run_kvd.sh"; done
sleep 25
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  s=$(J "$h" "docker exec $CTR bash -c 'test -S $KVD_SOCK && echo ok || echo MISSING'")
  echo "  $h kvd socket: $s"
  [ "$s" = "ok" ] || { J "$h" "docker exec $CTR tail -20 /tmp/kvd.log"; echo "FATAL kvd"; exit 1; }
done

echo
echo "===== 4. legs — KVAWARE=1 KVD=1 ====="
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
   HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_prefill_$TAG.log bash /glm52_leg.sh"
sleep 20
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
   MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
   HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
   LOG=$KIT/pd_decode_$TAG.log bash /glm52_leg.sh"

echo
echo "===== 5. router (default weights first) ====="
sleep 10
J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router.sh"

echo
echo "===== 6. waiting for both legs (~6 min cold start, NOT a hang) ====="
ready=0
for i in $(seq 1 $((WAIT_MIN * 6))); do
  sleep 10
  p=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_prefill_$TAG.log 2>/dev/null")
  d=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_decode_$TAG.log 2>/dev/null")
  pl=$(J "$PREFILL_HOST" "wc -l < $KIT/pd_prefill_$TAG.log 2>/dev/null")
  echo "  [$((i*10))s] prefill=${p:-0} decode=${d:-0} (prefill log ${pl:-0} lines)"
  [ "${p:-0}" = "1" ] && [ "${d:-0}" = "1" ] && { ready=1; echo "  both legs ready"; break; }
done
[ "$ready" = "1" ] || echo "  !! legs not both ready — the evidence below will show why"
sleep 25

# ---------------------------------------------------------------------------
# run_arm <default|weighted>
# ---------------------------------------------------------------------------
run_arm(){
  local arm="$1"
  echo
  echo "############################################################"
  echo "# ARM: $arm"
  echo "############################################################"
  if [ "$arm" = "weighted" ]; then
    echo "== restarting ONLY the router with role weights 20.0 / 2.0"
    echo "== (the legs and the kvd daemons stay up — only the scorer changes)"
    J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router_weighted.sh"
    sleep 30
    echo "== the weights, confirmed from the router's own log:"
    J "$PREFILL_HOST" "docker exec $CTR grep -a 'router-policy=' /tmp/router.log | tail -1"
  fi

  {
    echo "# ===== ARM $arm — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "--- kvd counters BEFORE ---"
    snap BEFORE
    echo
    echo "--- workload: 4 sessions x 4 turns x 2 phases, ~6200-token shared prefix ---"
    J "$PREFILL_HOST" "docker exec $CTR timeout 1800 python3 /tmp/prefix_reuse.py \
        http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions $SESSIONS"
    echo
    echo "--- kvd counters AFTER ---"
    snap AFTER
  } | tee "$OUT/step2_prefix_reuse_${arm}.observed.txt"
}

if [ "$ARMS" = "both" ] || [ "$ARMS" = "default" ];  then run_arm default;  fi
if [ "$ARMS" = "both" ] || [ "$ARMS" = "weighted" ]; then run_arm weighted; fi

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
for arm in default weighted; do
  f="$OUT/step2_prefix_reuse_${arm}.observed.txt"
  [ -s "$f" ] || continue
  tot=$(grep -oE '^TOTAL [0-9]+/[0-9]+ correct' "$f" | head -1)
  lat=$(grep -oE 'median latency: phase1 [0-9.]+s -> phase2 [0-9.]+s' "$f" | head -1)
  echo "  arm $arm: ${tot:-<no total>}   ${lat:-}"
  echo "     kvd (prefill) before: $(grep -m1 '\[BEFORE\] prefill' "$f" | sed 's/.*prefill(chi2879)://')"
  echo "     kvd (prefill) after : $(grep -m1 '\[AFTER\] prefill'  "$f" | sed 's/.*prefill(chi2879)://')"
done
cat <<'EOF'

  *** HOW TO READ THIS — the attribution is the whole point ***

  1. CORRECTNESS is the gate: 32/32 per arm. A cache that returns fast WRONG
     text is the failure mode a KV cache introduces, so the workload checks
     every response, not just the timing.

  2. kvd SERVED if, in the FIRST arm, gets_total goes 0 -> >0 with hits>0.
     On the reference run: gets 0 -> 170, hits 170, misses 0, 340 entries,
     600837120 bytes (573 MB) resident on the prefill node.

  3. THE LATENCY DROP IS NOT KVD'S. The reference run went 0.71s -> 0.26s
     between arms (2.7x) while kvd's counters stayed EXACTLY the same across
     the second arm. The GPU radix cache was already warm from the first arm
     and requests never sank to the host tier. If your numbers look like that,
     the speedup belongs to the GPU cache. Do not attribute it to kvd, and do
     not attribute it to the role weights either.

  4. hits=170 / misses=0 is SAME-PROCESS reuse. Every lookup repeats something
     this same live deployment just stored. It does not show cross-restart or
     cross-engine reuse, which is kvd's actual selling point. That needs the
     engine killed and the daemon kept alive.

  Committed reference: results/step2_prefix_reuse.txt
EOF
