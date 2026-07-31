#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 05 — THE DIFFERENTIAL. Is the garbled output kvaware/kvd's fault?
#
# Two arms, on one node, over MC_FORCE_TCP, with exactly three variables
# differing between them and everything else held identical by construction:
#
#   arm A (round r4):  KVD=1 KVAWARE=1 POLICY=kv-aware
#   arm B (round r5):  KVD=0 KVAWARE=0 POLICY=round-robin
#
# Default ARMS=both runs A, tears it down, runs B, then compares. Running both
# in one invocation is the point — an arm measured on a different day, after a
# reboot, or beside a different neighbour is not a control.
#
#   ARMS=both (default)  ~18 min. The actual experiment.
#   ARMS=A / ARMS=B      ~9 min. One arm only, for re-running a half that failed
#                        to launch. A single arm PROVES NOTHING on its own.
#
# *** NEITHER ARM IS EXPECTED TO PRODUCE CORRECT OUTPUT. *** Both come back
# garbled. That is the observation, and the comparison is the result. This
# experiment ends in a NO-REGRESSION finding, not a correctness pass — see
# scripts/compare_arms.py, which refuses to print "PASS" for this case.
# ---------------------------------------------------------------------------
set -uo pipefail
ARMS="${ARMS:-both}"
JUMP="${JUMP:-root@149.28.124.225}"
NODE="${NODE:-chi2879}"
NODE_IP="${NODE_IP:-10.2.122.10}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
SERVED="${SERVED:-qwen3}"
CTR="${CTR:-diff05_$$}"              # unique name: never collide with another job
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

case "$ARMS" in both|A|B) ;; *) echo "ARMS must be both|A|B"; exit 2 ;; esac

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $NODE '$1'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){ echo "[cleanup] removing $CTR"; J "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"; }
trap cleanup EXIT

# ---------------------------------------------------------------------------
# one-time setup: container, libionic, net.py fix, etcd, staged scripts
# ---------------------------------------------------------------------------
echo "== 05 differential, ARMS=$ARMS on $NODE"

echo "== starting container $CTR"
J "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host --ipc=host \
   --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
   --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
   --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
   -v /mnt/vast:/mnt/vast --entrypoint '' $IMAGE sleep infinity >/dev/null && echo ok" \
  | grep -q ok || { echo "FATAL: container start failed"; exit 1; }

echo "== injecting host libionic"
J "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); \
   docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B >/dev/null; \
   docker exec $CTR bash -c \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && \
     ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && \
     ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)\""

echo "== applying the free_tcp_port_block fix (the image predates it)"
ssh -o StrictHostKeyChecking=no "$JUMP" \
  "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/net_fixed.py'" < "$HERE/net_fixed.py"
J "docker cp /tmp/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && echo net_fix_applied"

echo "== etcd"
J "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd --network=host \
   quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$NODE_IP:2379 \
   --listen-client-urls http://0.0.0.0:2379 >/dev/null; sleep 4; \
   curl -sf -m5 http://$NODE_IP:2379/version >/dev/null && echo etcd_up"

echo "== staging leg.sh / up.sh / probe.py"
for f in leg.sh up.sh probe.py; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/$f'" < "$HERE/$f"
  J "docker exec $CTR mkdir -p /work >/dev/null; docker cp /tmp/$f $CTR:/work/$f >/dev/null"
done

# ---------------------------------------------------------------------------
# run_arm <A|B>
# ---------------------------------------------------------------------------
run_arm(){
  local arm="$1" kvd kvaware policy tag
  case "$arm" in
    A) kvd=1; kvaware=1; policy=kv-aware;    tag=r4 ;;
    B) kvd=0; kvaware=0; policy=round-robin; tag=r5 ;;
  esac

  echo
  echo "############################################################"
  echo "# ARM $arm  (round $tag):  KVD=$kvd KVAWARE=$kvaware POLICY=$policy"
  echo "############################################################"

  J "docker exec -d $CTR env MY_IP=$NODE_IP MODEL=$MODEL KVD=$kvd KVAWARE=$kvaware \
       POLICY=$policy TAG=$tag HICACHE_GB=8 FORCE_TCP=1 bash /work/up.sh"

  echo "== cold start ~2 min. Polling 8 min for both legs."
  for i in $(seq 1 48); do
    sleep 10
    st=$(J "docker exec $CTR bash -c \"echo p=\\\$(grep -ac 'ready to roll' /tmp/$tag/prefill.log 2>/dev/null) \
          d=\\\$(grep -ac 'ready to roll' /tmp/$tag/decode.log 2>/dev/null)\"")
    echo "  [$((i*10))s] $st"
    echo "$st" | grep -q "p=1 d=1" && { echo "  both legs ready"; break; }
  done
  sleep 25   # let the router pick both workers up

  {
    echo "# ===== ARM $arm (round $tag) — KVD=$kvd KVAWARE=$kvaware POLICY=$policy"
    echo "# $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo
    echo "--- the switches, confirmed from the wire, not from the launch command ---"
    echo "# kv_events_endpoint is null when kvaware is OFF; that is the switch"
    echo "# confirming itself rather than being taken on trust."
    J "docker exec $CTR curl -s -m10 http://$NODE_IP:8100/v1/workers"
    echo
    echo "--- kvd adapter connections per leg (expect >0 in arm A, 0 in arm B) ---"
    J "docker exec $CTR bash -c \"echo prefill=\\\$(grep -ac 'infera-kvd adapter connected' /tmp/$tag/prefill.log 2>/dev/null); \
         echo decode=\\\$(grep -ac 'infera-kvd adapter connected' /tmp/$tag/decode.log 2>/dev/null)\""
    echo
    echo "--- transport (must be MC_FORCE_TCP in BOTH arms — the held-constant) ---"
    J "docker exec $CTR bash -c \"grep -ahc 'MC_FORCE_TCP' /tmp/$tag/prefill.log\""
    echo
    echo "--- leg readiness ---"
    J "docker exec $CTR bash -c \"echo prefill_ready=\\\$(grep -ac 'ready to roll' /tmp/$tag/prefill.log 2>/dev/null); \
         echo decode_ready=\\\$(grep -ac 'ready to roll' /tmp/$tag/decode.log 2>/dev/null)\""
    echo
    echo "--- THE OUTPUT (this is the result; read the text, not the score) ---"
    J "docker exec $CTR timeout 400 python3 /work/probe.py http://$NODE_IP:8100 $SERVED \
         --arm $arm --json /tmp/$tag/probe.json"
  } | tee "$OUT/arm${arm}_${tag}.observed.txt"

  # pull the structured result back for the comparison step
  J "docker cp $CTR:/tmp/$tag/probe.json /tmp/arm${arm}.json >/dev/null 2>&1"
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $NODE 'cat /tmp/arm${arm}.json'" > "$OUT/arm${arm}.json" 2>/dev/null \
    || echo "  (could not retrieve probe.json for arm $arm)"
}

if [ "$ARMS" = "both" ] || [ "$ARMS" = "A" ]; then run_arm A; fi
if [ "$ARMS" = "both" ] || [ "$ARMS" = "B" ]; then run_arm B; fi

# ---------------------------------------------------------------------------
# the comparison — the only step that attributes anything
# ---------------------------------------------------------------------------
if [ "$ARMS" = "both" ]; then
  echo
  echo "===== COMPARISON ====="
  if [ -s "$OUT/armA.json" ] && [ -s "$OUT/armB.json" ]; then
    python3 "$HERE/compare_arms.py" "$OUT/armA.json" "$OUT/armB.json" \
      | tee "$OUT/differential_verdict.observed.txt"
  else
    echo "one or both probe.json files are missing — compare"
    echo "  $OUT/armA_r4.observed.txt  and  $OUT/armB_r5.observed.txt"
    echo "by hand."
  fi
fi

echo
echo "===== WHAT TO LOOK FOR ====="
cat <<'EOF'
NEITHER ARM IS EXPECTED TO PRODUCE CORRECT OUTPUT. The observed 2026-07-30
result was:

  arm A (kvaware+kvd ON):
    "capital of France" -> 'v4 freddy\n\nAists. Log In andapace.a\n\nWho wouldin %%%%3...'
    "17*23"             -> 'S情辣梯neig治\n\n杖\n\n及格...'
  arm B (both OFF):
    "capital of France" -> 'v4ই\n\n脐猫\n\nument=""&gt;&lt; t-tcan you-...'

Both garbled. Note both open with the same 'v4' token.

The pass condition for this EXPERIMENT is not correct output — it is a valid
comparison. Check, in this order:

  1. Both arms reached prefill_ready=1 decode_ready=1. An arm that never came
     up is not a control.
  2. Both arms show MC_FORCE_TCP > 0. The transport must be held constant; it
     is a prime suspect and varying it would confound the attribution.
  3. Arm A's workers show a non-null kv_events_endpoint and >0 kvd adapter
     connections; arm B's show null and 0. That is the switches confirming
     themselves from the wire rather than being taken on trust.
  4. Both arms produced completions (no HTTP 500). If either errored, the round
     is INCONCLUSIVE — an arm that never answered says nothing about content.

Only then does the comparison mean anything.

VERDICT if both arms are equally garbled: the garbling belongs to the shared
substrate (same-host PD + MC_FORCE_TCP), NOT to kvaware or kvd.

  *** That is a NO-REGRESSION observation, NOT a correctness pass. ***

  Nothing here was correct. You may cite it as "the features did not cause the
  garbling". You may NOT cite it as "kvaware+kvd verified". For that, remove
  the substrate: two nodes, real RDMA.
EOF
echo
echo "Committed reference evidence: results/r4_r5_differential.txt"
