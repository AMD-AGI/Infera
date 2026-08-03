#!/bin/bash
# The discriminating feature proofs, in one pass. Each row here is chosen because
# it would go RED if the feature were absent -- a green run on its own proves
# very little on this stack.
#
# Usage: feature_evidence.sh [tag] [outfile]
set -u
TAG="${1:-g1}"
OUT="${2:-/shared_nfs/yihou_agbench_mtp/results/feature_evidence_${TAG}.txt}"
W=/shared_nfs/yihou_agbench_mtp
PJOB="${PJOB:-24300}"; PIP="${PIP:-10.245.157.89}"
DJOB="${DJOB:-24301}"; DIP="${DIP:-10.245.146.87}"
CTR=agbench_mtp
mkdir -p "$(dirname "$OUT")"

exec > >(tee "$OUT") 2>&1
echo "############ feature evidence  tag=$TAG  $(date -u +%FT%TZ)"

echo
echo "=== 1. PD + mooncake RDMA (would go red: silent TCP fallback) ==="
for pair in "prefill $PIP" "decode $DIP"; do
  set -- $pair; R=$1
  S=$(mktemp); strings "$W/logs/${TAG}_${R}.log" > "$S"
  printf '  %-8s MC_FORCE_TCP=%-4s  mlx5_0=%-5s  dmabuf_disable=%s\n' \
    "$R" "$(grep -c MC_FORCE_TCP "$S")" "$(grep -c mlx5_0 "$S")" \
    "$(grep -c 'MOONCAKE_DISABLE_HIP_DMABUF' "$S")"
  printf '  %-8s bootstrap/transfer lines: %s\n' "$R" \
    "$(grep -cE 'mooncake|Mooncake' "$S")"
  rm -f "$S"
done

echo
echo "=== 2. DP-attention (would go red: dp_size=1, 1 scheduler proc) ==="
for pair in "prefill $PJOB" "decode $DJOB"; do
  set -- $pair; R=$1; J=$2
  S=$(mktemp); strings "$W/logs/${TAG}_${R}.log" > "$S"
  printf '  %-8s enable_dp_attention=%s  dp_size=8 lines=%s\n' "$R" \
    "$(grep -o 'enable_dp_attention=[A-Za-z]*' "$S" | head -1)" \
    "$(grep -c 'dp_size=8' "$S")"
  rm -f "$S"
  spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR bash -c 'echo -n \"    live sglang::scheduler_DP procs: \"; ps aux | grep -c \"[s]glang::scheduler_DP\"'" 2>&1 | grep -v libtinfow
done

echo
echo "=== 3. MTP / EAGLE (would go red: no speculative args, no accept len) ==="
S=$(mktemp); strings "$W/logs/${TAG}_decode.log" > "$S"
printf '  decode  speculative_algorithm: %s\n' "$(grep -o "speculative_algorithm=[^,]*" "$S" | head -1)"
printf '  decode  num_steps/topk/draft:   %s %s %s\n' \
  "$(grep -o 'speculative_num_steps=[0-9]*' "$S" | head -1)" \
  "$(grep -o 'speculative_eagle_topk=[0-9]*' "$S" | head -1)" \
  "$(grep -o 'speculative_num_draft_tokens=[0-9]*' "$S" | head -1)"
echo   "  decode  accept len samples (want 2.1-2.6; 4.00 is BAD -- a predicted loop):"
grep -ao 'accept len: [0-9.]*' "$S" | tail -8 | sed 's/^/      /'
printf '  prefill speculative args (want none): %s\n' \
  "$(strings "$W/logs/${TAG}_prefill.log" | grep -c 'speculative_algorithm=EAGLE')"
rm -f "$S"
echo "  --- server-reported acceptance (the number bench_serving reads) ---"
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR bash -c 'curl -sf -m10 http://$DIP:30001/server_info 2>/dev/null | python3 -c \"
import json,sys
d=json.load(sys.stdin)
d=d.get(\\\"decode\\\",[d])[0] if \\\"decode\\\" in d else d
st=(d.get(\\\"internal_states\\\") or [{}])[0]
print(\\\"      avg_spec_accept_length =\\\", st.get(\\\"avg_spec_accept_length\\\"))
\" 2>/dev/null || echo \"      (server_info unavailable via this path)\"'" 2>&1 | grep -v libtinfow

echo
echo "=== 4. kv-aware routing (would go red: cache view 0, no kv-aware picks) ==="
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR bash -c '
    echo \"  --- workers ---\"
    curl -sf -m5 http://$PIP:8190/v1/workers 2>/dev/null | python3 -m json.tool 2>/dev/null | head -40
    echo \"  --- router kv-aware metrics ---\"
    curl -sf -m5 http://$PIP:8190/metrics 2>/dev/null | grep -E \"^infera_(router_picks_total|policy_cache_view_size|router_pick_cache_hits_sum|router_pick_request_blocks_sum|policy_active_blocks)\" | head -30
    echo \"  --- kv-aware pick log lines ---\"
    grep -c \"policy=kv-aware\" /tmp/router.log 2>/dev/null || echo 0
    grep -o \"policy=kv-aware role=[a-z]* .*cache_hits=[0-9]* request_blocks=[0-9]*\" /tmp/router.log 2>/dev/null | tail -5
  '" 2>&1 | grep -v libtinfow

echo
echo "=== 5. kvd (counters; the SERVING proof is restart_replay.sh) ==="
for pair in "prefill $PJOB" "decode $DJOB"; do
  set -- $pair; R=$1; J=$2
  echo "  --- $R ---"
  spur exec "$J" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock" 2>&1 | grep -v libtinfow | sed 's/^/    /'
  S=$(mktemp); strings "$W/logs/${TAG}_${R}.log" > "$S"
  printf '    hierarchical_cache=%s  hicache_storage_backend=%s\n' \
    "$(grep -o 'enable_hierarchical_cache=[A-Za-z]*' "$S" | head -1)" \
    "$(grep -o 'hicache_storage_backend=[^,]*' "$S" | head -1)"
  printf '    "not wiring infera-kvd" (decode should be 1): %s\n' "$(grep -c 'not wiring infera-kvd' "$S")"
  rm -f "$S"
done
echo
echo "############ end"
