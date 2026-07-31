#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 04 — same-host PD over mooncake RDMA does not work (MVP round r3).
#
#   *** TRANSPORT=rdma IS SUPPOSED TO FAIL. ***
#
# This is the first round in which BOTH legs come up — the three earlier fixes
# (--hicache-size, port spacing, patched free_tcp_port_block) all hold, and the
# router sees two workers with DISTINCT kv-event endpoints. So it is also the
# first round that can take a request, which is what exposes the next layer:
# two legs on ONE host means the mooncake KV transfer must loop back across
# RDMA rails (ionic_0 -> ionic_4), and this ionic fabric will not do it.
#
#   TRANSPORT=rdma (default) — real mooncake RDMA. Legs ready, both register,
#                              every completion returns HTTP 500
#                              "Failed to get kvcache from prefill instance".
#   TRANSPORT=tcp            — MC_FORCE_TCP=1. Completions succeed.
#                              (Whether their CONTENT is right is a separate
#                              question with a separate experiment.)
#
# This limitation is ORTHOGONAL to kvaware/kvd. It is a property of putting a
# PD pair on one box, and it is the reason the real correctness work moved to
# two nodes.
# ---------------------------------------------------------------------------
set -uo pipefail
TRANSPORT="${TRANSPORT:-rdma}"
JUMP="${JUMP:-root@149.28.124.225}"
NODE="${NODE:-chi2879}"
NODE_IP="${NODE_IP:-10.2.122.10}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
SERVED="${SERVED:-qwen3}"
CTR="${CTR:-samehost04_$$}"          # unique name: never collide with another job
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

case "$TRANSPORT" in
  rdma) FORCE_TCP=0; TAG=r3_rdma ;;
  tcp)  FORCE_TCP=1; TAG=r3_tcp  ;;
  *) echo "TRANSPORT must be rdma|tcp"; exit 2 ;;
esac

J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $NODE '$1'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){ echo "[cleanup] removing $CTR"; J "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"; }
trap cleanup EXIT

echo "== 04 same-host RDMA, TRANSPORT=$TRANSPORT (FORCE_TCP=$FORCE_TCP) on $NODE"

echo "== starting container $CTR"
J "docker rm -f $CTR >/dev/null 2>&1; docker run -d --name $CTR --network=host --ipc=host \
   --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
   --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK \
   --security-opt seccomp=unconfined --ulimit memlock=-1:-1 \
   -v /mnt/vast:/mnt/vast --entrypoint '' $IMAGE sleep infinity >/dev/null && echo ok" \
  | grep -q ok || { echo "FATAL: container start failed"; exit 1; }

echo "== injecting host libionic — MANDATORY. Without it RDMA silently degrades"
echo "   to TCP and TRANSPORT=rdma would 'pass' for the wrong reason."
J "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); \
   docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B >/dev/null; \
   docker exec $CTR bash -c \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && \
     ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && \
     ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)\""
echo "   ^ that MUST say 8. If it says 0, stop: the experiment is invalid."

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

echo "== launching 1P1D + router"
J "docker exec -d $CTR env MY_IP=$NODE_IP MODEL=$MODEL KVD=1 KVAWARE=1 POLICY=kv-aware \
     TAG=$TAG HICACHE_GB=8 FORCE_TCP=$FORCE_TCP bash /work/up.sh"

echo "== Qwen3-1.7B cold start is ~2 min. Polling for 8 min for BOTH legs."
for i in $(seq 1 48); do
  sleep 10
  st=$(J "docker exec $CTR bash -c \"echo p=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null) \
        d=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\"")
  echo "  [$((i*10))s] $st"
  echo "$st" | grep -q "p=1 d=1" && { echo "  both legs ready"; break; }
done

echo "== letting the router pick both workers up"
sleep 25

echo
echo "===== EVIDENCE ====="
{
  echo "# Experiment 04 — observed, TRANSPORT=$TRANSPORT, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- RDMA rails visible in the container (must be 8) ---"
  J "docker exec $CTR bash -c \"ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE\""
  echo
  echo "--- was MC_FORCE_TCP actually in effect? ---"
  J "docker exec $CTR bash -c \"grep -ahc 'MC_FORCE_TCP' /tmp/$TAG/prefill.log\""
  echo "    (0 = real RDMA, >0 = the TCP workaround)"
  echo
  echo "--- both legs ready, and their DISTINCT kv-event endpoints ---"
  J "docker exec $CTR bash -c \"echo prefill_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null); \
       echo decode_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\""
  J "docker exec $CTR bash -c \"for f in /tmp/$TAG/prefill.log /tmp/$TAG/decode.log; do \
       echo -n \\\"\\\$f: \\\"; grep -aoE '\\\"endpoint\\\": \\\"tcp://[^\\\"]+\\\"' \\\$f | head -1; echo; done\""
  echo
  echo "--- router's view: both workers registered? ---"
  J "docker exec $CTR curl -s -m10 http://$NODE_IP:8100/v1/workers"
  echo
  echo "--- THE PROBE (this is the result) ---"
  J "docker exec $CTR timeout 400 python3 /work/probe.py http://$NODE_IP:8100 $SERVED"
  echo
  echo "--- mooncake transport errors in the prefill leg ---"
  J "docker exec $CTR bash -c \"grep -ah -E 'transport retry counter|received packet mismatch|worker_pool.cpp|rdma_endpoint.cpp' /tmp/$TAG/*.log | sort -u | head -20\""
} | tee "$OUT/r3_samehost_rdma.observed.txt"

echo
echo "===== WHAT TO LOOK FOR ====="
if [ "$TRANSPORT" = "rdma" ]; then
cat <<'EOF'
TRANSPORT=rdma is EXPECTED TO FAIL, and to fail in a very specific way.
A successful reproduction shows ALL of:

  1. active_ports = 8                    (RDMA really is available)
  2. MC_FORCE_TCP hits = 0               (we really are on RDMA)
  3. prefill_ready=1 decode_ready=1      (startup is fully fixed)
  4. two workers in /v1/workers, with DISTINCT kv_events_endpoint ports
  5. probe: 0/4 correct, 4 errored, every one HTTP 500
     "Failed to get kvcache from prefill instance"
  6. in the leg log:
       worker_pool.cpp:408 ... local_nic: ionic_0, peer_nic: ...@ionic_4:
                               transport retry counter exceeded
       rdma_endpoint.cpp:472  Invalid argument: received packet mismatch

Points 1-4 are what make this a TRANSPORT result and not a wiring result: every
layer above the transfer is demonstrably healthy. The failure is the KV moving.

If instead the probe succeeds, check point 2 — a nonzero MC_FORCE_TCP count
means libionic injection failed and RDMA silently degraded, so you measured
the TCP path by accident.
EOF
else
cat <<'EOF'
TRANSPORT=tcp is EXPECTED to get completions back:

  1. MC_FORCE_TCP hits > 0
  2. prefill_ready=1 decode_ready=1
  3. probe: NO HTTP 500s, no "Failed to get kvcache"

That is a TRANSPORT result only. The probe may still report 0/4 correct with
garbled content — completions arriving is a different claim from completions
being right, and the content question needs a differential run to attribute.
Do not read "no 500s" as "correct".
EOF
fi
echo
echo "Committed reference evidence: results/r3_samehost_rdma.txt"
