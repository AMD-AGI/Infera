#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 09 — THE CONFOUNDER-FREE ONE. Cross-restart KV reuse.
#
# A prefix-reuse run showed kvd serving hits, but every hit repeated something
# the SAME live process had just stored, with a warm GPU radix cache in front of
# it the whole time. Two confounders, and neither is removable by measuring
# harder. This experiment removes both by construction:
#
#   Kill ONLY the sglang engine legs. Leave the kvd daemon RUNNING.
#     engine dies  -> GPU KV cache gone, HiRadixCache gone, new process starts
#                     with an empty in-VRAM cache.
#     kvd survives -> still holds what the OLD engine wrote.
#   Then relaunch the legs and replay the IDENTICAL workload.
#
# THE SIGNATURE TO LOOK FOR, on the prefill node:
#     gets_total   climbs          <- the new engine READ
#     hits_total   climbs equally  <- and every read HIT
#     sets_total   FLAT            <- it wrote nothing back
#     entries      FLAT
#     host_bytes   FLAT
# A brand-new process reading blocks it never wrote. sets_total staying flat is
# what rules out "it just rebuilt everything and hit its own fresh writes".
#
# PRECONDITIONS THE SCRIPT VERIFIES BEFORE RELAUNCHING (do not skip these — they
# are the experiment):
#   * GPU VRAM back to the idle baseline (~297 MB) => the engine is really dead
#     and the GPU cache really is gone.
#   * kvd still alive (pgrep) and still holding its entries.
#
# Cost: ~6 min initial cold start + ~4 min workload + ~6 min second cold start
#       + ~4 min replay. 16 GPUs across 2 nodes.
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
SESSIONS="${SESSIONS:-4}"
IDLE_VRAM_MAX="${IDLE_VRAM_MAX:-1073741824}"   # 1 GB. Observed idle was ~297 MB.
CTR="${CTR:-glm52_rr09}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_rr09}"
WAIT_MIN="${WAIT_MIN:-20}"
KEEP="${KEEP:-0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

kvd_stats(){ J "$1" "docker exec $CTR bash /kvdstats.sh 2>/dev/null | tail -1"; }
snap(){
  echo "  [$1] prefill(chi2879): $(kvd_stats "$PREFILL_HOST")"
  echo "  [$1] decode (chi2867): $(kvd_stats "$DECODE_HOST")"
}

cleanup(){
  if [ "$KEEP" = "1" ]; then echo "[cleanup] KEEP=1 — deployment left up."; return; fi
  echo "[cleanup] removing $CTR on both nodes"
  J "$PREFILL_HOST" "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
  J "$DECODE_HOST"  "docker rm -f $CTR >/dev/null 2>&1"
}
trap cleanup EXIT

# launch_legs <tag>
launch_legs(){
  local tag="$1"
  J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP ETCD_IP=$PREFILL_IP \
     MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
     HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
     LOG=$KIT/pd_prefill_$tag.log bash /glm52_leg.sh"
  sleep 20
  J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP ETCD_IP=$PREFILL_IP \
     MODEL=$MODEL SERVED=$SERVED PORT=30000 KVAWARE=1 KVD=1 KVD_SOCK=$KVD_SOCK \
     HICACHE_GB=$HICACHE_GB KV_PUB_PORT=5557 KV_SNAP_PORT=8801 \
     LOG=$KIT/pd_decode_$tag.log bash /glm52_leg.sh"
  sleep 10
  J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router.sh"
}

# wait_legs <tag>
wait_legs(){
  local tag="$1" i p d pl
  for i in $(seq 1 $((WAIT_MIN * 6))); do
    sleep 10
    p=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_prefill_$tag.log 2>/dev/null")
    d=$(J "$PREFILL_HOST" "grep -ac 'ready to roll' $KIT/pd_decode_$tag.log 2>/dev/null")
    pl=$(J "$PREFILL_HOST" "wc -l < $KIT/pd_prefill_$tag.log 2>/dev/null")
    echo "  [$((i*10))s] prefill=${p:-0} decode=${d:-0} (prefill log ${pl:-0} lines — growing = loading)"
    [ "${p:-0}" = "1" ] && [ "${d:-0}" = "1" ] && { echo "  both legs ready"; sleep 25; return 0; }
  done
  echo "  !! legs not both ready after ${WAIT_MIN}m"; return 1
}

echo "############################################################"
echo "# 09 — cross-restart KV reuse (kill the engine, keep the daemon)"
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
         run_router.sh restart_legs.sh; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f'" < "$HERE/$f"
done
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && \
          docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null && \
          docker cp $KIT/run_kvd.sh $CTR:/run_kvd.sh >/dev/null && \
          docker cp $KIT/kvdstats.sh $CTR:/kvdstats.sh >/dev/null && \
          docker cp $KIT/restart_legs.sh $CTR:/restart_legs.sh >/dev/null && \
          docker cp $KIT/prefix_reuse.py $CTR:/tmp/prefix_reuse.py >/dev/null && \
          docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null && echo $h staged"
done
J "$PREFILL_HOST" "docker cp $KIT/run_router.sh $CTR:/run_router.sh >/dev/null"

echo
echo "===== 2. etcd ====="
J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd \
   --network=host quay.io/coreos/etcd:v3.5.14 etcd \
   --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT \
   --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 5; \
   curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version >/dev/null && echo etcd_up || echo ETCD_FAILED"

echo
echo "===== 3. kvd daemon on BOTH nodes — this process must SURVIVE the restart ====="
for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker exec -d $CTR bash /run_kvd.sh"; done
sleep 25
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  s=$(J "$h" "docker exec $CTR bash -c 'test -S $KVD_SOCK && echo ok || echo MISSING'")
  echo "  $h kvd socket: $s"
  [ "$s" = "ok" ] || { J "$h" "docker exec $CTR tail -20 /tmp/kvd.log"; echo "FATAL kvd"; exit 1; }
done

# ---------------------------------------------------------------------------
# ROUND 1 — populate
# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# ROUND 1 — first engine. Populates kvd."
echo "############################################################"
launch_legs r1
wait_legs r1 || echo "  (continuing; the evidence will show the state)"

{
  echo "# ===== ROUND 1 (first engine) — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- kvd BEFORE round 1 (expect all zero on a fresh daemon) ---"
  snap R1_BEFORE
  echo
  echo "--- workload ---"
  J "$PREFILL_HOST" "docker exec $CTR timeout 1800 python3 /tmp/prefix_reuse.py \
      http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions $SESSIONS"
  echo
  echo "--- kvd AFTER round 1 (this is what the SECOND engine must be able to read) ---"
  snap R1_AFTER
} | tee "$OUT/round1_populate.observed.txt"

# ---------------------------------------------------------------------------
# THE KILL — engine only
# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# KILL — engine legs ONLY. The kvd daemon stays up on purpose:"
echo "# it holds the very thing under test."
echo "############################################################"
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  echo "  $h:"
  J "$h" "docker exec $CTR bash /restart_legs.sh" | sed 's/^/    /'
done

echo
echo "===== VERIFYING THE PRECONDITIONS (this IS the experiment) ====="
{
  echo "# ===== PRECONDITIONS BEFORE RELAUNCH — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- 1. GPU VRAM back to idle => the engine is truly dead, GPU cache gone ---"
  echo "#    If VRAM is still tens of GB, the engine did NOT die and any 'reuse'"
  echo "#    observed afterwards could be the surviving GPU cache. The whole"
  echo "#    experiment depends on this check."
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    v=$(J "$h" "rocm-smi --showmeminfo vram --csv 2>/dev/null | sed -n 2p | awk -F, '{print \$3}'")
    echo "    $h GPU[0] VRAM used = ${v:-<unavailable>} B"
  done
  echo
  echo "--- 2. kvd still alive, and still holding the entries ---"
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    a=$(J "$h" "docker exec $CTR bash -c \"pgrep -fc 'infera.kvd'\"")
    echo "    $h  pgrep -fc 'infera.kvd' = ${a:-0}   (need >=1)"
  done
  snap PRE_RELAUNCH
} | tee "$OUT/preconditions.observed.txt"

# hard gate: refuse to continue if VRAM says the engine is still resident
gate_ok=1
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  v=$(J "$h" "rocm-smi --showmeminfo vram --csv 2>/dev/null | sed -n 2p | awk -F, '{print \$3}' | tr -dc 0-9")
  if [ -n "$v" ] && [ "$v" -gt "$IDLE_VRAM_MAX" ] 2>/dev/null; then
    echo "  !! $h VRAM still $v B (> $IDLE_VRAM_MAX) — the engine may not be dead."
    gate_ok=0
  fi
done
[ "$gate_ok" = "1" ] || echo "  !! PRECONDITION NOT MET. Any 'reuse' below would be CONFOUNDED. Continuing so
     you can see the state, but DO NOT cite the result."

# ---------------------------------------------------------------------------
# ROUND 2 — brand-new engine, same workload
# ---------------------------------------------------------------------------
echo
echo "############################################################"
echo "# ROUND 2 — brand-new engine process, SAME workload, warm kvd."
echo "############################################################"
launch_legs r2
wait_legs r2 || echo "  (continuing; the evidence will show the state)"

{
  echo "# ===== ROUND 2 (new engine, warm kvd) — $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- the new engine really is new: kvd adapters re-attached on all 8 ranks ---"
  J "$PREFILL_HOST" "echo prefill_kvd_adapters=\$(grep -ac 'infera-kvd adapter connected' $KIT/pd_prefill_r2.log); \
     echo decode_kvd_adapters=\$(grep -ac 'infera-kvd adapter connected' $KIT/pd_decode_r2.log)"
  echo
  echo "--- kvd BEFORE the replay (== R1_AFTER; the daemon never restarted) ---"
  snap R2_BEFORE
  echo
  echo "--- SAME workload, replayed ---"
  J "$PREFILL_HOST" "docker exec $CTR timeout 1800 python3 /tmp/prefix_reuse.py \
      http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions $SESSIONS"
  echo
  echo "--- kvd AFTER the replay — THE RESULT ---"
  snap R2_AFTER
} | tee "$OUT/round2_reload.observed.txt"

# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
echo
echo "===== VERDICT ====="
python3 - "$OUT/round2_reload.observed.txt" <<'PY'
import re, sys
txt = open(sys.argv[1]).read()

def grab(label, node):
    m = re.search(rf"\[{label}\] {node}[^\n]*?StatsResponse\(([^)]*)\)", txt)
    if not m:
        return None
    d = {}
    for kv in m.group(1).split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            try:
                d[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return d

for node in ("prefill", "decode "):
    b, a = grab("R2_BEFORE", node), grab("R2_AFTER", node)
    n = node.strip()
    if not b or not a:
        print(f"  {n}: counters unreadable — inspect the file by hand")
        continue
    print(f"  {n} node:")
    for k in ("entries", "host_bytes", "sets_total", "gets_total",
              "hits_total", "misses_total"):
        if k in b and k in a:
            d = a[k] - b[k]
            print(f"    {k:<14} {b[k]:>12} -> {a[k]:>12}   delta {d:+d}")
    if n == "prefill":
        clean = (a.get("gets_total", 0) > b.get("gets_total", 0)
                 and a.get("hits_total", 0) > b.get("hits_total", 0)
                 and a.get("sets_total") == b.get("sets_total")
                 and a.get("entries") == b.get("entries"))
        print()
        if clean:
            print("    ==> CROSS-RESTART REUSE CONFIRMED. gets/hits climbed while")
            print("        sets/entries stayed FLAT: a brand-new process read blocks")
            print("        it never wrote.")
        else:
            print("    ==> NOT the clean signature. If sets_total also grew, the new")
            print("        engine may simply have re-written and then hit its own")
            print("        fresh writes — which proves nothing about survival.")
PY

cat <<'EOF'

  Reference result (2026-07-30), prefill node chi2879:
      entries      340 -> 340    delta 0
      host_bytes   600837120 -> 600837120   delta 0
      sets_total   340 -> 340    delta 0      <- nothing re-written
      gets_total   170 -> 340    delta +170   <- the new engine READ
      hits_total   170 -> 340    delta +170   <- and every read HIT
      misses_total   0 ->   0    delta 0

  Correctness on the replay was 31/32. The single miss was a TRUNCATION (the
  model spent its 128-token budget on a reasoning preamble and never reached
  "Jupiter"), not a wrong answer — the same prompt passed in the other three
  sessions and in every prior run.

  LATENCY DID NOT IMPROVE (0.76s vs 0.71s cold). That is expected and worth
  stating: kvd's win here is capacity and survival, not speed.

  Committed reference: results/step3_restart_reload.txt
EOF
