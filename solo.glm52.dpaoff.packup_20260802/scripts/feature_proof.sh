#!/usr/bin/env bash
# Six-row feature-proof matrix. Runs ON the prefill node.
#
# Every row is a signal that would go RED if the feature were absent. A green
# run that proves nothing is the default outcome on this stack, so each row
# names what its failure looks like.
set -u
CTR="${CTR:-bench_run}"
P_IP="${P_IP:-10.2.122.10}"
D_IP="${D_IP:-10.2.122.44}"
R="${R:-http://$P_IP:8100}"
W=/mnt/vast/c_huggingface/bench_20260801
PLOG="$W/logs/${TAG:-p2}_prefill.log"

echo "===== 1. PD: both disagg modes registered and active ====="
docker exec "$CTR" curl -s -m10 "$R/v1/workers" \
  | python3 -c 'import sys,json
d=json.load(sys.stdin)["workers"]
for w in d: print(f"  {w[\"worker_id\"]:24s} {w[\"disagg_mode\"]:8s} {w[\"status\"]:8s} dp_size={w[\"dp_size\"]}")
modes={w["disagg_mode"] for w in d if w["status"]=="active"}
print(f"  -> {\"OK\" if modes>={\"prefill\",\"decode\"} else \"FAIL\"} (want both prefill and decode)")'

echo "===== 2. DPA: 8 live DP schedulers per node ====="
for h in "$P_IP prefill" "$D_IP decode"; do
  set -- $h
  n=$(docker exec "$CTR" bash -c "true")   # placeholder; counted on the node below
done
echo "  prefill: $(pgrep -fc 'sglang::scheduler_DP' || echo 0) sglang::scheduler_DP* (want 8)"

echo "===== 3. RDMA: real mooncake, not TCP ====="
mcfail=$(strings "$PLOG" | grep -c 'Mooncake Transfer Engine initialization failed' || true)
tcp=$(strings "$PLOG" | grep -c 'MC_FORCE_TCP' || true)
gid=$(strings "$PLOG" | grep -oE 'Using user-specified GID index: [0-9]+' | tail -1)
ion=$(strings "$PLOG" | grep -c 'ionic_' || true)
echo "  mooncake init failures: $mcfail (want 0)"
echo "  MC_FORCE_TCP mentions:  $tcp (want 0)"
echo "  $gid ; ionic mentions: $ion"

echo "===== 4. MTP: accept len on the decode leg ====="
echo "  (read after traffic; 2.1-2.6 healthy, 4.00 = repetition loop = BAD)"

echo "===== 5. kvd counters (baseline) ====="
docker exec "$CTR" python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock

echo "===== 6. kv-aware: per-DP-rank views ====="
echo "  Rust router has no /v1/admin/cache-view route (handlers.rs:33-38);"
echo "  the signal is the policy log line, read after traffic. See cache_view.sh."
