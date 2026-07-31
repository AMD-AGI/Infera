#!/usr/bin/env bash
###############################################################################
# ONE COMMAND to reproduce the whole kvaware+kvd investigation.
#
#   bash run_all.sh                 # everything, in order
#   bash run_all.sh --phase step3   # just one phase (see PHASES below)
#   bash run_all.sh --list          # what the phases are
#   bash run_all.sh --teardown      # release the nodes
#
# Every trap this investigation hit is handled in here. You should NOT need to
# read notes.md to get a green run — that file is for understanding *why*, not
# for operating this script.
#
# Handled automatically:
#   - three kvaware ports that collide when workers share a host
#     (sglang kv-events block, --kv-events-bind 5557, --kv-snapshot-port 8801)
#   - `docker exec -d ... bash -lc '...'` silently not persisting
#   - --mem-fraction-static being TP-dependent (0.85 @TP8, 0.70 @TP4)
#   - --hicache-ratio sizing the host pool off the KV pool (355 GB/rank blowup)
#   - the ~6-12 min cold start that looks like a hang
#   - patching infera in-container (the image predates both fixes)
#
# Results land in ./run_<TS>/ : per-phase logs, a machine-readable summary.csv,
# and the engine logs pulled off the shared FS.
###############################################################################
set -uo pipefail

# ---------------------------------------------------------------- config ----
JUMP="${JUMP:-root@149.28.124.225}"
PREFILL_HOST="${PREFILL_HOST:-chi2879}"; PREFILL_IP="${PREFILL_IP:-10.2.122.10}"
DECODE_HOST="${DECODE_HOST:-chi2867}";   DECODE_IP="${DECODE_IP:-10.2.122.44}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/xiaobo/models/GLM-5.2-MXFP4}"
SERVED="${SERVED:-glm5.2-mxfp4}"
CTR="${CTR:-glm52_kvexp}"
KIT="${KIT:-/mnt/vast/c_huggingface/glm52_kvexp}"
ROUTER_PORT="${ROUTER_PORT:-8100}"; ETCD_PORT="${ETCD_PORT:-2379}"
KVD_SOCK=/tmp/kvd/kvd.sock
READY_TIMEOUT="${READY_TIMEOUT:-1200}"   # cold start is 6-12 min; not a hang

# This script lives in packup.try.all/ but sources scripts/ and patches/ from
# the packup root, one level up. SELF = where this file is; HERE = packup root.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HERE="$(cd "$SELF/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$SELF/run_$TS}"
CSV=""   # set in init_out

PHASES="env baseline step1 step2 step3 step4 step5"

# ------------------------------------------------------------- plumbing ----
c_ok(){ printf '\033[32m%s\033[0m\n' "$*"; }
c_bad(){ printf '\033[31m%s\033[0m\n' "$*"; }
c_hdr(){ printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }
log(){ echo "[$(date +%H:%M:%S)] $*"; }
die(){ c_bad "FATAL: $*"; exit 1; }

# Run a command on a node, through the jump host.
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
       "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $1 '$2'" 2>&1 | grep -v "^Warning: Permanently"; }

# Run a script FILE inside a container on a node. Use this instead of
# `docker exec -d $CTR bash -lc '...'` — the detached login-shell form does not
# persist (it exits and takes the child with it). This bit us twice.
J_script(){ local host="$1" script="$2" dest="${3:-/tmp/_run.sh}"
  scp -o StrictHostKeyChecking=no -q "$script" "$JUMP:/tmp/$(basename "$script")" 2>/dev/null
  ssh -o StrictHostKeyChecking=no "$JUMP" \
      "scp -o StrictHostKeyChecking=no -q /tmp/$(basename "$script") $host:/tmp/ 2>/dev/null" 2>/dev/null
  J "$host" "docker cp /tmp/$(basename "$script") $CTR:$dest >/dev/null"
}

# Append a row to summary.csv.
row(){ printf '%s,%s,%s,%s,%s\n' "$1" "$2" "$3" "$4" "$5" >> "$CSV"; }

init_out(){
  mkdir -p "$OUT"/{logs,phases}
  CSV="$OUT/summary.csv"
  [ -s "$CSV" ] || echo "phase,metric,value,expected,verdict" > "$CSV"
  log "output -> $OUT"
}

# Wait for both legs to print "ready to roll". Bounded; reports progress so a
# slow cold start is visibly *progressing* rather than looking wedged.
wait_ready(){ local pf="$1" df="$2" waited=0
  log "waiting for both legs (cold start 6-12 min; this is NOT a hang)"
  while [ $waited -lt $READY_TIMEOUT ]; do
    local p d
    p=$(J "$PREFILL_HOST" "docker exec $CTR grep -ac 'ready to roll' $KIT/$pf 2>/dev/null" | tr -dc '0-9')
    d=$(J "$DECODE_HOST"  "docker exec $CTR grep -ac 'ready to roll' $KIT/$df 2>/dev/null" | tr -dc '0-9')
    [ "${p:-0}" -ge 1 ] && [ "${d:-0}" -ge 1 ] && { c_ok "  both legs ready (${waited}s)"; return 0; }
    # fail fast on a dead leg instead of burning the full timeout
    local err
    err=$(J "$PREFILL_HOST" "docker exec $CTR grep -acE 'scheduler died|Address already in use|out of memory' $KIT/$pf 2>/dev/null" | tr -dc '0-9')
    [ "${err:-0}" -gt 0 ] && { c_bad "  prefill leg died — see $KIT/$pf"; return 1; }
    sleep 20; waited=$((waited+20))
    [ $((waited % 120)) -eq 0 ] && log "  ... ${waited}s (prefill=$p decode=$d)"
  done
  c_bad "  timeout after ${READY_TIMEOUT}s"; return 1
}

kvd_stats(){ J "$1" "docker exec $CTR timeout 30 bash /kvdstats.sh 2>/dev/null"; }
kvd_field(){ echo "$1" | grep -oE "$2=[0-9]+" | head -1 | cut -d= -f2; }

# ------------------------------------------------------------- phase: env --
ph_env(){ c_hdr "PHASE env — fabric preflight + container prep + in-container patching"
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    local ports; ports=$(J "$h" "ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE" | tr -dc '0-9')
    [ "${ports:-0}" -eq 8 ] || die "$h has ${ports:-0}/8 ionic ports ACTIVE — fix the fabric first"
    c_ok "  $h: 8/8 ionic PORT_ACTIVE"
  done
  J "$PREFILL_HOST" "ping -c2 -W2 $DECODE_IP >/dev/null && echo ok" | grep -q ok \
    || die "no data-plane route $PREFILL_IP -> $DECODE_IP"
  c_ok "  data-plane route ok"

  log "recreating containers (bind-mount for L3; see step5)"
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J "$h" "docker rm -f $CTR >/dev/null 2>&1; mkdir -p /mnt/nvme-raid/kvd-long; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast -v /mnt/nvme-raid/kvd-long:/kvd-long --entrypoint \"\" $IMAGE sleep infinity >/dev/null && echo started" | grep -q started \
      || die "container start failed on $h"
    # host libionic; without it RDMA silently degrades to TCP
    J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B; docker exec $CTR bash -lc \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && ldconfig 2>/dev/null\""
    local inner; inner=$(J "$h" "docker exec $CTR ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE" | tr -dc '0-9')
    [ "${inner:-0}" -eq 8 ] || die "$h: libionic injection failed (${inner:-0}/8 inside container)"
    c_ok "  $h: container up, libionic ok (8/8 inside)"
    # BOTH infera fixes — the image predates them
    J "$h" "docker cp $KIT/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null"
    J "$h" "docker cp $KIT/storage_classify_fixed.py $CTR:/opt/infera/infera/kvd/storage_classify.py >/dev/null 2>&1 || true"
    c_ok "  $h: infera patches applied in-container"
  done
  J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd --network=host quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$PREFILL_IP:$ETCD_PORT --listen-client-urls http://0.0.0.0:$ETCD_PORT >/dev/null; sleep 4"
  J "$PREFILL_HOST" "curl -sf -m5 http://$PREFILL_IP:$ETCD_PORT/version" | grep -q etcdserver || die "etcd did not come up"
  c_ok "  etcd up"
  row env fabric "8/8 both nodes" "8/8" PASS
}

# --------------------------------------------------------- leg launching ----
# $1 host  $2 role  $3 ip  $4 port  $5 kvaware  $6 kvd  $7 logname
# $8 tp  $9 base_gpu  $10 gmu  $11 kv_pub_port  $12 kv_snap_port
launch_leg(){
  J "$1" "docker exec -d $CTR env ROLE=$2 MY_IP=$3 ETCD_IP=$PREFILL_IP PORT=$4 \
     KVAWARE=$5 KVD=$6 TP=${8:-8} BASE_GPU=${9:-0} GMU=${10:-} HICACHE_GB=16 \
     KV_PUB_PORT=${11:-5557} KV_SNAP_PORT=${12:-8801} MODEL=$MODEL SERVED=$SERVED \
     LOG=$KIT/$7 bash /glm52_leg.sh"
}

start_router(){ local policy="$1" weights="${2:-}"
  cat > /tmp/_router.sh <<EOF
#!/bin/bash
pkill -9 -f "infera.server" 2>/dev/null; sleep 3
exec python3 -m infera.server --host 0.0.0.0 --port $ROUTER_PORT \\
  --discovery-backend etcd --etcd-endpoint $PREFILL_IP:$ETCD_PORT \\
  --request-transport http --kv-event-transport zmq \\
  --router-policy $policy --router-tokenizer-path $MODEL $weights \\
  > /tmp/router.log 2>&1
EOF
  J_script "$PREFILL_HOST" /tmp/_router.sh /run_router.sh
  J "$PREFILL_HOST" "docker exec -d $CTR bash /run_router.sh"; sleep 25
  J "$PREFILL_HOST" "docker exec $CTR curl -s -m10 http://$PREFILL_IP:$ROUTER_PORT/v1/workers" > "$OUT/phases/workers_$policy.json"
  local n; n=$(grep -o '"worker_id"' "$OUT/phases/workers_$policy.json" | wc -l)
  log "  router up (policy=$policy), $n worker(s) registered"
  echo "$n"
}

start_kvd(){ for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J_script "$h" "$HERE/scripts/run_kvd.sh" /run_kvd.sh
    J_script "$h" "$HERE/scripts/kvdstats.sh" /kvdstats.sh
    J "$h" "docker exec -d $CTR bash /run_kvd.sh"
  done; sleep 25
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J "$h" "docker exec $CTR test -S $KVD_SOCK && echo up" | grep -q up || die "kvd did not start on $h"
    c_ok "  kvd up on $h"
  done
}

probe(){ J "$PREFILL_HOST" "docker exec $CTR timeout 500 python3 /tmp/probe.py http://$PREFILL_IP:$ROUTER_PORT $SERVED"; }

stage_kit(){ log "staging kit onto the shared FS"
  J "$PREFILL_HOST" "mkdir -p $KIT"
  for f in glm52_leg.sh probe.py prefix_reuse.py net_fixed.py; do
    scp -o StrictHostKeyChecking=no -q "$HERE/scripts/$f" "$JUMP:/tmp/$f"
    ssh -o StrictHostKeyChecking=no "$JUMP" "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/$f' < /tmp/$f"
  done
  # patched storage_classify (fix #2) staged under its own name
  scp -o StrictHostKeyChecking=no -q "$SELF/storage_classify_fixed.py" "$JUMP:/tmp/" 2>/dev/null && \
    ssh -o StrictHostKeyChecking=no "$JUMP" "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cat > $KIT/storage_classify_fixed.py' < /tmp/storage_classify_fixed.py" 2>/dev/null
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J "$h" "docker cp $KIT/glm52_leg.sh $CTR:/glm52_leg.sh >/dev/null; docker cp $KIT/probe.py $CTR:/tmp/probe.py >/dev/null; docker cp $KIT/prefix_reuse.py $CTR:/tmp/prefix_reuse.py >/dev/null"
  done
  c_ok "  kit staged"
}

# ------------------------------------------------------- phase: baseline ----
ph_baseline(){ c_hdr "PHASE baseline — PD+DPA, kvaware OFF / kvd OFF (the control)"
  launch_leg "$PREFILL_HOST" prefill "$PREFILL_IP" 30000 0 0 pd_prefill_base.log
  sleep 20
  launch_leg "$DECODE_HOST"  decode  "$DECODE_IP"  30000 0 0 pd_decode_base.log
  wait_ready pd_prefill_base.log pd_decode_base.log || return 1
  start_router round-robin >/dev/null
  local out; out=$(probe); echo "$out" > "$OUT/phases/baseline_probe.txt"
  local score; score=$(echo "$out" | grep -oE '[0-9]+/4 correct' | head -1)
  echo "$out" | tail -3
  local tcp dma
  tcp=$(J "$PREFILL_HOST" "docker exec $CTR grep -ac MC_FORCE_TCP $KIT/pd_prefill_base.log" | tr -dc '0-9')
  dma=$(J "$PREFILL_HOST" "docker exec $CTR grep -ac 'HIP dmabuf disabled' $KIT/pd_prefill_base.log" | tr -dc '0-9')
  row baseline correctness "$score" ">=3/4" "$([ "${score%%/*}" -ge 3 ] 2>/dev/null && echo PASS || echo FAIL)"
  row baseline mc_force_tcp "$tcp" 0 "$([ "${tcp:-1}" -eq 0 ] && echo PASS || echo FAIL)"
  row baseline rdma_rails "$dma" 8 "$([ "${dma:-0}" -eq 8 ] && echo PASS || echo FAIL)"
  c_ok "  baseline: $score, real RDMA (tcp=$tcp rails=$dma)"
}

# ---------------------------------------------------------- phase: step1 ----
ph_step1(){ c_hdr "PHASE step1 — same topology, kvaware ON / kvd ON"
  J "$PREFILL_HOST" "docker exec $CTR bash /restart_legs.sh" >/dev/null 2>&1
  J "$DECODE_HOST"  "docker exec $CTR bash /restart_legs.sh" >/dev/null 2>&1
  sleep 30
  start_kvd
  launch_leg "$PREFILL_HOST" prefill "$PREFILL_IP" 30000 1 1 pd_prefill_kv.log
  sleep 20
  launch_leg "$DECODE_HOST"  decode  "$DECODE_IP"  30000 1 1 pd_decode_kv.log
  wait_ready pd_prefill_kv.log pd_decode_kv.log || return 1
  local n; n=$(start_router kv-aware "--kv-overlap-weight 1.0")
  local out; out=$(probe); echo "$out" > "$OUT/phases/step1_probe.txt"
  echo "$out" | tail -3
  local score kvd_ranks plane
  score=$(echo "$out" | grep -oE '[0-9]+/4 correct' | head -1)
  kvd_ranks=$(J "$PREFILL_HOST" "docker exec $CTR grep -ac 'infera-kvd adapter connected' $KIT/pd_prefill_kv.log" | tr -dc '0-9')
  plane=$(J "$PREFILL_HOST" "docker exec $CTR grep -ac 'KV plane up:' $KIT/pd_prefill_kv.log" | tr -dc '0-9')
  row step1 correctness "$score" ">=3/4" "$([ "${score%%/*}" -ge 3 ] 2>/dev/null && echo PASS || echo FAIL)"
  row step1 kvd_ranks_connected "$kvd_ranks" 8 "$([ "${kvd_ranks:-0}" -eq 8 ] && echo PASS || echo FAIL)"
  row step1 kv_plane_up "$plane" 1 "$([ "${plane:-0}" -ge 1 ] && echo PASS || echo FAIL)"
  c_ok "  step1: $score, kvd ranks=$kvd_ranks, KV plane=$plane"
}

# ---------------------------------------------------------- phase: step2 ----
ph_step2(){ c_hdr "PHASE step2 — prefix-reuse workload: does kvd actually serve?"
  local before after g_b g_a h_a s_b s_a
  before=$(kvd_stats "$PREFILL_HOST"); g_b=$(kvd_field "$before" gets_total); s_b=$(kvd_field "$before" sets_total)
  log "  kvd before: gets=$g_b sets=$s_b"
  local out; out=$(J "$PREFILL_HOST" "docker exec $CTR timeout 900 python3 /tmp/prefix_reuse.py http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions 4")
  echo "$out" > "$OUT/phases/step2_workload.txt"; echo "$out" | tail -4
  after=$(kvd_stats "$PREFILL_HOST"); g_a=$(kvd_field "$after" gets_total); h_a=$(kvd_field "$after" hits_total); s_a=$(kvd_field "$after" sets_total)
  log "  kvd after:  gets=$g_a hits=$h_a sets=$s_a"
  local score; score=$(echo "$out" | grep -oE 'TOTAL [0-9]+/[0-9]+' | awk '{print $2}')
  row step2 correctness "$score" 32/32 "$([ "$score" = "32/32" ] && echo PASS || echo FAIL)"
  row step2 kvd_gets_delta "$(( ${g_a:-0} - ${g_b:-0} ))" ">0" "$([ "$(( ${g_a:-0} - ${g_b:-0} ))" -gt 0 ] && echo PASS || echo FAIL)"
  row step2 kvd_sets_delta "$(( ${s_a:-0} - ${s_b:-0} ))" ">0" "$([ "$(( ${s_a:-0} - ${s_b:-0} ))" -gt 0 ] && echo PASS || echo FAIL)"
  c_ok "  step2: $score, kvd gets +$(( ${g_a:-0} - ${g_b:-0} )), sets +$(( ${s_a:-0} - ${s_b:-0} ))"
}

# ---------------------------------------------------------- phase: step3 ----
ph_step3(){ c_hdr "PHASE step3 — kill the ENGINE, keep kvd: cross-restart reuse"
  local before g_b s_b; before=$(kvd_stats "$PREFILL_HOST")
  g_b=$(kvd_field "$before" gets_total); s_b=$(kvd_field "$before" sets_total)
  log "  kvd before restart: gets=$g_b sets=$s_b"
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    J_script "$h" "$HERE/scripts/restart_legs.sh" /restart_legs.sh
    J "$h" "docker exec $CTR bash /restart_legs.sh"
  done
  sleep 40
  # THE precondition: GPU must be back to idle, else a warm GPU cache explains any hit
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    local vram; vram=$(J "$h" "rocm-smi --showmeminfo vram 2>/dev/null | grep -i used | head -1 | grep -oE '[0-9]+'" | tail -1)
    if [ "${vram:-999999999999}" -lt 1000000000 ]; then c_ok "  $h GPU released ($((vram/1048576)) MB) — GPU cache cannot explain later hits"
    else c_bad "  $h GPU still holds $((vram/1048576)) MB — result would be confounded"; fi
  done
  J "$PREFILL_HOST" "docker exec $CTR test -S $KVD_SOCK && echo alive" | grep -q alive || die "kvd died — the whole point is that it survives"
  c_ok "  kvd survived the engine restart"
  launch_leg "$PREFILL_HOST" prefill "$PREFILL_IP" 30000 1 1 pd_prefill_r2.log
  sleep 20
  launch_leg "$DECODE_HOST"  decode  "$DECODE_IP"  30000 1 1 pd_decode_r2.log
  wait_ready pd_prefill_r2.log pd_decode_r2.log || return 1
  start_router kv-aware "--kv-overlap-weight 1.0" >/dev/null
  local out; out=$(J "$PREFILL_HOST" "docker exec $CTR timeout 900 python3 /tmp/prefix_reuse.py http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions 4")
  echo "$out" > "$OUT/phases/step3_workload.txt"; echo "$out" | tail -4
  local after g_a h_a s_a; after=$(kvd_stats "$PREFILL_HOST")
  g_a=$(kvd_field "$after" gets_total); h_a=$(kvd_field "$after" hits_total); s_a=$(kvd_field "$after" sets_total)
  local dg=$(( ${g_a:-0} - ${g_b:-0} )) ds=$(( ${s_a:-0} - ${s_b:-0} ))
  log "  kvd after: gets=$g_a hits=$h_a sets=$s_a   (delta gets=+$dg sets=+$ds)"
  row step3 kvd_gets_delta "$dg" ">0" "$([ "$dg" -gt 0 ] && echo PASS || echo FAIL)"
  row step3 kvd_sets_delta "$ds" "0 (read-only reuse)" "$([ "$ds" -eq 0 ] && echo PASS || echo "PARTIAL")"
  c_ok "  step3: a FRESH engine read +$dg blocks it never wrote (new sets: $ds)"
}

# ---------------------------------------------------------- phase: step4 ----
ph_step4(){ c_hdr "PHASE step4 — 2 decode workers: does kv-aware actually route?"
  J "$DECODE_HOST" "docker exec $CTR bash /restart_legs.sh" >/dev/null 2>&1; sleep 40
  # TP4 x2 on the decode node. GMU 0.70 (not 0.85): TP4 doubles weights/GPU.
  # All three port vars must differ per worker or the second dies.
  launch_leg "$DECODE_HOST" decode "$DECODE_IP" 30000 1 1 pd_decodeA.log 4 0 0.70 5557 8801
  sleep 25
  launch_leg "$DECODE_HOST" decode "$DECODE_IP" 32000 1 1 pd_decodeB.log 4 4 0.70 5657 8802
  log "  waiting for both TP4 decode workers"
  local waited=0
  while [ $waited -lt $READY_TIMEOUT ]; do
    local a b
    a=$(J "$DECODE_HOST" "docker exec $CTR grep -ac 'ready to roll' $KIT/pd_decodeA.log 2>/dev/null" | tr -dc '0-9')
    b=$(J "$DECODE_HOST" "docker exec $CTR grep -ac 'ready to roll' $KIT/pd_decodeB.log 2>/dev/null" | tr -dc '0-9')
    [ "${a:-0}" -ge 1 ] && [ "${b:-0}" -ge 1 ] && break
    sleep 20; waited=$((waited+20))
  done
  local nw; nw=$(start_router kv-aware "--kv-overlap-weight 1.0 --kv-prefill-overlap-weight 20.0 --kv-decode-overlap-weight 2.0")
  [ "${nw:-0}" -ge 3 ] || c_bad "  only $nw workers registered (want 3) — check pd_decodeB.log for the 8801 snapshot-port collision"
  row step4 workers_registered "$nw" 3 "$([ "${nw:-0}" -ge 3 ] && echo PASS || echo FAIL)"

  cnt(){ J "$DECODE_HOST" "docker exec $CTR grep -ac 'Decode batch' $KIT/pd_decode$1.log 2>/dev/null" | tr -dc '0-9'; }
  local a0 b0 a1 b1 a2 b2
  a0=$(cnt A); b0=$(cnt B)
  J "$PREFILL_HOST" "docker exec $CTR timeout 900 python3 /tmp/prefix_reuse.py http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions 4" > "$OUT/phases/step4_kvaware.txt"
  a1=$(cnt A); b1=$(cnt B)
  log "  kv-aware  -> decodeA +$(( a1-a0 ))  decodeB +$(( b1-b0 ))"
  start_router round-robin >/dev/null
  J "$PREFILL_HOST" "docker exec $CTR timeout 900 python3 /tmp/prefix_reuse.py http://$PREFILL_IP:$ROUTER_PORT $SERVED --sessions 4" > "$OUT/phases/step4_roundrobin.txt"
  a2=$(cnt A); b2=$(cnt B)
  log "  round-robin -> decodeA +$(( a2-a1 ))  decodeB +$(( b2-b1 ))"
  row step4 kvaware_split "$(( a1-a0 ))/$(( b1-b0 ))" "skewed (affinity)" INFO
  row step4 roundrobin_split "$(( a2-a1 ))/$(( b2-b1 ))" "even (no affinity)" INFO
  # The claim is that the two DIFFER; that's the whole experiment.
  if [ "$(( b1-b0 ))" -eq 0 ] && [ "$(( b2-b1 ))" -gt 0 ]; then
    c_ok "  routing effect CONFIRMED: kv-aware pinned to one worker, round-robin spread"
    row step4 routing_effect "confirmed" "policies differ" PASS
  else
    c_bad "  distributions did not clearly differ — inspect $OUT/phases/step4_*.txt"
    row step4 routing_effect "unclear" "policies differ" FAIL
  fi
}

# ---------------------------------------------------------- phase: step5 ----
ph_step5(){ c_hdr "PHASE step5 — L3 on a real block device (storage classification)"
  local cls; cls=$(J "$PREFILL_HOST" "docker exec $CTR python3 -m infera.kvd classify /kvd-long 2>&1 | head -8")
  echo "$cls" > "$OUT/phases/step5_classify.txt"; echo "$cls"
  local dev; dev=$(echo "$cls" | grep -oE 'devices  = \[[^]]*\]' | head -1)
  if echo "$dev" | grep -q '(none)'; then
    c_bad "  device NOT resolved: $dev"
    c_bad "  -> is patch 0002 applied in-container? is /dev/md0 exposed (--device)?"
    row step5 device_resolved "none" "a real device" FAIL
  else
    c_ok "  device resolved: $dev"
    row step5 device_resolved "$dev" "a real device" PASS
  fi
  # NOTE: 'buffered' is the CORRECT verdict here — md0 is a raid1 of SATA SSDs.
  row step5 io_mode "$(echo "$cls" | grep -oE 'L3 io_mode: [A-Z]+' | awk '{print $3}')" "buffered on SATA" INFO
}

# ------------------------------------------------------------- collection --
collect(){ c_hdr "collecting engine logs"
  mkdir -p "$OUT/logs"/{glm_baseline,glm_step1_kvaware_kvd,glm_step3_restart,glm_step4_routing}
  ssh -o StrictHostKeyChecking=no "$JUMP" "ssh -o StrictHostKeyChecking=no $PREFILL_HOST 'cd $KIT && tar cf - pd_*.log 2>/dev/null'" > /tmp/_logs.tar 2>/dev/null
  local tmp; tmp=$(mktemp -d); tar xf /tmp/_logs.tar -C "$tmp" 2>/dev/null
  cp "$tmp"/pd_*base.log "$OUT/logs/glm_baseline/" 2>/dev/null
  cp "$tmp"/pd_*_kv.log  "$OUT/logs/glm_step1_kvaware_kvd/" 2>/dev/null
  cp "$tmp"/pd_*_r2.log  "$OUT/logs/glm_step3_restart/" 2>/dev/null
  cp "$tmp"/pd_decode[AB].log "$OUT/logs/glm_step4_routing/" 2>/dev/null
  # gzip anything over 4 MB, per the packup rule
  find "$OUT/logs" -name '*.log' -size +4M -exec gzip {} \; 2>/dev/null
  log "  logs -> $OUT/logs ($(du -sh "$OUT/logs" 2>/dev/null | cut -f1))"
}

teardown(){ c_hdr "teardown"
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do J "$h" "docker rm -f $CTR >/dev/null 2>&1; echo done" >/dev/null; done
  J "$PREFILL_HOST" "docker rm -f ${CTR}_etcd >/dev/null 2>&1" >/dev/null
  sleep 20
  for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    local v; v=$(J "$h" "rocm-smi --showmeminfo vram 2>/dev/null | grep -i used | head -1 | grep -oE '[0-9]+'" | tail -1)
    c_ok "  $h GPU0 now $(( ${v:-0} / 1048576 )) MB (idle baseline ~284 MB)"
  done
}

summary(){ c_hdr "SUMMARY  ($CSV)"; column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
  local f; f=$(grep -c ',FAIL$' "$CSV" 2>/dev/null || echo 0)
  echo; [ "$f" -eq 0 ] && c_ok "no FAIL rows" || c_bad "$f FAIL row(s) — see above"
}

# ------------------------------------------------------------------ main ----
case "${1:-}" in
  --list) echo "phases: $PHASES"; exit 0 ;;
  --teardown) teardown; exit 0 ;;
  --phase) [ -n "${2:-}" ] || die "--phase needs a name (one of: $PHASES)"
           init_out; "ph_$2" ; summary; exit $? ;;
  "") ;;
  *) die "unknown arg '$1' (try --list)" ;;
esac

init_out
log "full run: $PHASES"
stage_kit
ph_env      || die "env preflight failed"
ph_baseline || c_bad "baseline phase failed — later phases may be meaningless"
ph_step1
ph_step2
ph_step3
ph_step4
ph_step5
collect
summary
c_hdr "done — artifacts in $OUT"
echo "  summary.csv     machine-readable results"
echo "  phases/         per-phase raw output"
echo "  logs/           engine logs, per experiment"
echo
echo "release the nodes with:  bash run_all.sh --teardown"
