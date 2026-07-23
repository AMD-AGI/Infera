#!/bin/bash
# Per-retry reset ritual: kill sglang legs+router on BOTH nodes, wait GPU VRAM
# back to idle. Run from LOCAL (drives both nodes via jump host).
set -u
JUMP="root@149.28.124.225"
PN="${PN:-chi2832}"; DN="${DN:-chi2878}"
# uses shared helper mtt_vram.sh (avoids fragile inline awk over nested ssh)
# NOTE: the native router process is named `sglang::router` (NOT sglang_router),
# so pkill -f sglang_router MISSES it. Kill both patterns. Also: after ANY leg
# restart the router must be restarted too, else its circuit-breaker keeps the
# old (dead) decode URL "open/unhealthy" and never re-registers the new worker
# (-> "No available decode workers"). Restart order: legs first, then router.
kill_node(){ ssh -o ConnectTimeout=20 "$JUMP" "ssh -o ConnectTimeout=20 $1 'docker exec mtt_pd bash -lc \"pkill -9 -f sglang.launch_server; pkill -9 -f launch_router; pkill -9 -f \\\"sglang::router\\\"; pkill -9 -f bench_serving\" 2>/dev/null; echo killed on $1'" 2>&1; }
vram_sum(){ ssh -o ConnectTimeout=20 "$JUMP" "ssh -o ConnectTimeout=20 $1 'docker exec mtt_pd bash /mnt/vast/c_huggingface/mtt_scripts/mtt_vram.sh'" 2>&1 | grep -oE 'vram_sum=[0-9.]+' | grep -oE '[0-9.]+'; }
echo "=== reset: kill legs+router on $PN,$DN ==="
kill_node "$PN"; kill_node "$DN"
echo "=== wait VRAM idle (sum<10GB) both nodes ==="
for i in $(seq 1 90); do
  p=$(vram_sum "$PN"); d=$(vram_sum "$DN")
  echo "  t=${i} chi_p_sum=${p}GB chi_d_sum=${d}GB"
  awk "BEGIN{exit !(${p:-999}<10 && ${d:-999}<10)}" && { echo "=== VRAM idle, reset done ==="; exit 0; }
  sleep 4
done
echo "=== WARN: VRAM did not fully drop in time ==="; exit 1
