#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Experiment 03 — the free_tcp_port_block collision (MVP round r2). BUG #1.
#
# Two independent paths, and you should run §A first because it is free:
#
#   MODE=desk   (default)  no GPU, no cluster, ~1 second.
#                          Reproduces the bug against the pre-fix code, confirms
#                          the fix, and demonstrates why the two more-obvious
#                          fixes were rejected. This is the whole root cause.
#
#   MODE=live              one node, ~8 min, 8 GPUs. Brings up the actual 1P1D
#                          pair. NETPY=old reproduces r2's dead decode leg;
#                          NETPY=new shows both legs coming up with distinct
#                          kv-event endpoints.
#
# *** MODE=live NETPY=old IS SUPPOSED TO FAIL. *** The decode leg's Rank-0
# scheduler dies during init with ZMQError on tcp://*:32765. That failure is the
# result.
# ---------------------------------------------------------------------------
set -uo pipefail
MODE="${MODE:-desk}"
NETPY="${NETPY:-new}"                # old = pre-fix net.py, new = patched
JUMP="${JUMP:-root@149.28.124.225}"
NODE="${NODE:-chi2879}"
NODE_IP="${NODE_IP:-10.2.122.10}"
IMAGE="${IMAGE:-infera/engine-sglang:pd-unified}"
MODEL="${MODEL:-/mnt/vast/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
CTR="${CTR:-portcol03_$$}"           # unique name: never collide with another job
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$HERE/../results}"

# ===========================================================================
# §A — desk check. No cluster.
# ===========================================================================
if [ "$MODE" = "desk" ]; then
  echo "== 03 §A desk check: reproduce the bug, confirm the fix, reject the alternatives"
  echo
  python3 "$HERE/mvp_port_block.py" 2>&1 | tee "$OUT/mvp_port_block.observed.txt"
  rc=${PIPESTATUS[0]}
  echo
  echo '== section 2 needs the `infera` package importable. If it said SKIPPED, retry with:'
  echo "     PYTHONPATH=<infera repo> bash scripts/run.sh"
  echo
  echo "== regression test (install to tests/unit/common/ in the repo)"
  if python3 -c "import pytest" 2>/dev/null; then
    python3 -m pytest "$HERE/test_net_port_block.py" -q 2>&1 | tail -5
  else
    echo "   pytest not installed — skipping. The 4 tests are in scripts/test_net_port_block.py"
  fi
  echo
  echo "== committed reference: results/mvp_port_block.txt, results/r2_port_collision.txt"
  exit "$rc"
fi

if [ "$MODE" != "live" ]; then echo "MODE must be desk|live"; exit 2; fi

# ===========================================================================
# §B — live 1P1D on one node.
# ===========================================================================
TAG="r2_$NETPY"
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$JUMP" \
      "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 $NODE '$1'" 2>&1 \
      | grep -v "^Warning: Permanently"; }

cleanup(){ echo "[cleanup] removing $CTR"; J "docker rm -f $CTR ${CTR}_etcd >/dev/null 2>&1"; }
trap cleanup EXIT

echo "== 03 §B live, NETPY=$NETPY on $NODE"

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

# --- THE VARIABLE UNDER TEST -----------------------------------------------
# The image predates the fix, so its /opt/infera/infera/common/net.py is the
# BUGGY one. NETPY=new overwrites it; NETPY=old leaves it alone.
if [ "$NETPY" = "new" ]; then
  echo "== applying the fix: docker cp net_fixed.py -> /opt/infera/infera/common/net.py"
  ssh -o StrictHostKeyChecking=no "$JUMP" \
    "ssh -o StrictHostKeyChecking=no $NODE 'cat > /tmp/net_fixed.py'" < "$HERE/net_fixed.py"
  J "docker cp /tmp/net_fixed.py $CTR:/opt/infera/infera/common/net.py >/dev/null && echo net_fix_applied"
else
  echo "== leaving the stock (pre-fix) net.py in place — expect the decode leg to die"
fi

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

echo "== launching 1P1D (prefill :30000 GPU0-3 | decode :31000 GPU4-7)"
J "docker exec -d $CTR env MY_IP=$NODE_IP MODEL=$MODEL KVD=1 KVAWARE=1 TAG=$TAG \
     HICACHE_GB=8 bash /work/up.sh"

echo "== Qwen3-1.7B cold start is ~2 min. Polling for 8 min."
for i in $(seq 1 48); do
  sleep 10
  st=$(J "docker exec $CTR bash -c \"echo p=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null) \
        d=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\"")
  echo "  [$((i*10))s] $st"
  echo "$st" | grep -q "p=1 d=1" && break
done

echo
echo "===== EVIDENCE ====="
{
  echo "# Experiment 03 — observed, NETPY=$NETPY, $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  echo "--- what free_tcp_port_block returns in THIS container ---"
  J "docker exec $CTR cat /tmp/$TAG/netpy_probe.txt"
  echo
  echo "--- the kv-events endpoint each leg was actually given ---"
  J "docker exec $CTR bash -c \"for f in /tmp/$TAG/prefill.log /tmp/$TAG/decode.log; do \
       echo -n \\\"\\\$f: \\\"; grep -aoE '\\\"endpoint\\\": \\\"tcp://[^\\\"]+\\\"' \\\$f | head -1; echo; done\""
  echo
  echo "--- ZMQ address-in-use (the crash) ---"
  J "docker exec $CTR bash -c \"grep -ah 'Address already in use' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- scheduler death ---"
  J "docker exec $CTR bash -c \"grep -ah 'scheduler died during initialization' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- kvd adapter connected (should be fine either way — kvd is not the problem) ---"
  J "docker exec $CTR bash -c \"grep -ahc 'infera-kvd adapter connected' /tmp/$TAG/prefill.log\""
  echo
  echo "--- host pool sizing (round-02's fix still holding) ---"
  J "docker exec $CTR bash -c \"grep -ah 'Allocating .* host memory for hierarchical' /tmp/$TAG/*.log | sort -u\""
  echo
  echo "--- leg readiness ---"
  J "docker exec $CTR bash -c \"echo prefill_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/prefill.log 2>/dev/null); \
       echo decode_ready=\\\$(grep -ac 'ready to roll' /tmp/$TAG/decode.log 2>/dev/null)\""
} | tee "$OUT/r2_port_collision.observed.txt"

echo
echo "===== WHAT TO LOOK FOR ====="
if [ "$NETPY" = "old" ]; then
cat <<'EOF'
NETPY=old is EXPECTED TO FAIL. A successful reproduction shows:

  1. netpy_probe: free_tcp_port_block(4) x10 -> all 32764, distinct: 1
  2. BOTH legs' "endpoint" showing the SAME port, e.g. tcp://*:32764
  3. zmq.error.ZMQError: Address already in use (addr='tcp://*:32765')
     -- note 32765 = base+1, i.e. DP rank 1's publisher
  4. RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)
  5. prefill_ready=1, decode_ready=0

Note 32764 assumes ip_local_port_range starts at 32768. On a box with a
different range the base differs but the collision does not — that is the point
of the bug: whatever the value, BOTH callers get it.
EOF
else
cat <<'EOF'
NETPY=new is EXPECTED TO PASS:

  1. netpy_probe: 10 calls, distinct: 10 (or at least >1)
  2. the two legs' "endpoint" ports DIFFER (r3 observed 17213 and 31215)
  3. no 'Address already in use'
  4. prefill_ready=1, decode_ready=1

What this does NOT prove: that requests work. Both legs being up is a startup
result. Serving over same-host mooncake RDMA fails for an unrelated reason
(cross-rail loopback), which is a separate matter with a separate fix.
EOF
fi
echo
echo "Committed reference evidence: results/r2_port_collision.txt"
