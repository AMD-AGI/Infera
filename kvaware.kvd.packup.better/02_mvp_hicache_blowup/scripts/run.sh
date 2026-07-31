#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 02 — the --hicache-ratio host-memory blow-up (Qwen3-1.7B MVP r1).
#
#   *** THIS ROUND IS SUPPOSED TO FAIL. ***
#
# It reproduces the two independent startup failures of MVP round r1 on ONE
# node: a derived-ZMQ-port collision that kills the prefill leg, and a hicache
# host pool sized off the KV pool that tries to reserve 354.94 GB PER DP RANK.
#
# MODE=broken (default) — r1 as it happened. Expect BOTH failures.
# MODE=fixed            — r1 with the two fixes applied (--hicache-size 16 and
#                         a 1000-port gap). Expect both legs to reach
#                         "ready to roll" and 8.00 GB / 16.00 GB host pools.
#
# WARNING (MODE=broken): the decode leg genuinely attempts a ~1.4 TB host
# allocation. On the 3023 GB chi287x boxes that gets refused / OOM-killed
# rather than wedging the node, which is what happened on 2026-07-30 — but do
# not run this on a smaller-RAM machine, and do not run it beside somebody
# else's job. The script kills its own legs on exit.
# ---------------------------------------------------------------------------
set -uo pipefail
JUMP="${JUMP:-root@149.28.124.225}"
NODE="${NODE:-chi2879}"
NODE_IP="${NODE_IP:-10.2.122.10}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
CTR="${CTR:-hicache02_$$}"          # unique name: never collide with another job
MODE="${MODE:-broken}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

case "$MODE" in
  broken) HICACHE_MODE=ratio; PORT_GAP=100;  HICACHE_GB=16; TAG=r1_broken ;;
  fixed)  HICACHE_MODE=size;  PORT_GAP=1000; HICACHE_GB=16; TAG=r1_fixed  ;;
  *) echo "MODE must be broken|fixed"; exit 2 ;;
esac

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $NODE '$1'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){
  echo "[cleanup] killing legs and removing $CTR"
  J "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"
}
trap cleanup EXIT

echo "== 02 hicache blow-up, MODE=$MODE (hicache=$HICACHE_MODE port_gap=$PORT_GAP) on $NODE"

echo "== starting container $CTR"
J "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host --ipc=host \
   --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
   --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
   --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
   -v /mnt/vast:/mnt/vast --entrypoint '' $IMAGE sleep infinity >/dev/null && echo ok" \
  | grep -q ok || { echo "FATAL: container start failed"; exit 1; }

echo "== injecting host libionic (without it RDMA silently degrades to TCP)"
J "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); \
   docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B >/dev/null; \
   docker exec $CTR bash -c \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && \
     ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && \
     ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)\""

echo "== etcd"
J "docker rm -f ${CTR}_etcd >/dev/null 2>&1; docker run -d --name ${CTR}_etcd --network=host \
   quay.io/coreos/etcd:v3.5.14 etcd --advertise-client-urls http://$NODE_IP:2379 \
   --listen-client-urls http://0.0.0.0:2379 >/dev/null; sleep 4; \
   curl -sf -m5 http://$NODE_IP:2379/version >/dev/null && echo etcd_up"

echo "== staging leg.sh / up.sh"
for f in leg.sh up.sh; do
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/$f'" < "$HERE/$f"
  J "docker exec $CTR mkdir -p /work >/dev/null; docker cp /tmp/$f $CTR:/work/$f >/dev/null"
done

echo "== launching 1P1D (this is the part that fails when MODE=broken)"
J "docker exec -d $CTR env MY_IP=$NODE_IP MODEL=$MODEL KVD=1 KVAWARE=1 TAG=$TAG \
     HICACHE_MODE=$HICACHE_MODE HICACHE_GB=$HICACHE_GB PORT_GAP=$PORT_GAP \
     bash /work/up.sh"

echo "== Qwen3-1.7B cold start is ~2 min. Polling for 6 min."
for i in $(seq 1 36); do
  sleep 10
  st=$(J "docker exec $CTR bash -c \"echo p=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null) \
        d=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\"")
  echo "  [$((i*10))s] $st"
  echo "$st" | grep -q "p=1 d=1" && break
done

echo
echo "===== EVIDENCE ====="
{
  echo "# Experiment 02 — observed, MODE=$MODE, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- host pool sizing (the headline number) ---"
  J "docker exec $CTR bash -c \"grep -ah 'Allocating .* host memory for hierarchical' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- KV pool size the ratio is computed FROM ---"
  J "docker exec $CTR bash -c \"grep -aho 'max_total_num_tokens=[0-9]*' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- any ZMQ address-in-use (the port collision) ---"
  J "docker exec $CTR bash -c \"grep -ah 'Address already in use' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- leg readiness ---"
  J "docker exec $CTR bash -c \"echo prefill_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null); \
       echo decode_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\""
} | tee "$OUT/r1_hicache_blowup.observed.txt"

echo
echo "===== WHAT TO LOOK FOR ====="
if [ "$MODE" = "broken" ]; then
cat <<'EOF'
MODE=broken is EXPECTED TO FAIL. A successful reproduction shows:

  1. "Allocating 354.94 GB host memory for hierarchical KV cache." x4 DP ranks
     (the decode leg; x4 = ~1.4 TB requested on a 3023 GB box)
  2. "max_total_num_tokens=1547424"   <- what hicache_ratio=2.0 multiplied
  3. a ZMQError "Address already in use" on a derived port (r1: 30235)
  4. prefill_ready=0

If instead both legs come up, your box either has a different
ip_local_port_range (so the derived blocks happen not to overlap) or enough
free RAM that the 1.4 TB request succeeded. Both are environment differences,
not a broken script — record them.
EOF
else
cat <<'EOF'
MODE=fixed is EXPECTED TO PASS the hicache half:

  1. "Allocating 8.00 GB host memory for hierarchical KV cache." (or the
     HICACHE_GB you set) — bounded, ratio no longer in play
  2. NO 'Address already in use' on a *derived* port from the 1000-gap
  3. prefill_ready=1

decode_ready may STILL be 0 — that is the NEXT bug (both legs' --kv-events-config
carrying the same base port). It is a different failure with a different fix and
is not what this experiment is about.
EOF
fi
echo
echo "Committed reference evidence: results/r1_hicache_blowup.txt"
