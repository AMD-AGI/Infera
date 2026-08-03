#!/bin/bash
# Resume the kvd serving proof from step 3.
#
# Steps 0-2 already completed in the first attempt before it stalled on the
# pkill self-kill: L3 holds 22,048 entries from the warm phase, the engine is
# dead, and all 8 GPUs read VRAM 0%. Re-running step 0 would re-drive the corpus
# against a fresh engine and destroy exactly the state under test, so this
# resumes rather than restarts.
#
# Usage: restart_replay_resume.sh
set -eu
W=/shared_nfs/yihou_agbench_mtp
PJOB="${PJOB:-24300}"; PIP="${PIP:-10.245.157.89}"
CTR=agbench_mtp
R=$W/results
D="$(cd "$(dirname "$0")" && pwd)"

snap() {
  spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock" 2>&1 \
    | grep -v libtinfow > "$R/kvd_$1.json"
  echo "  [$1]"; sed 's/^/    /' "$R/kvd_$1.json"
}

echo "===== preflight: engine dead, kvd alive, GPUs drained ====="
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR bash -c 'echo -n \"    engine procs: \"; (ps aux | grep -E \"[l]aunch_server|[i]nfera[.]engine[.]sglang\" | wc -l); echo -n \"    kvd alive:    \"; (ps aux | grep \"[i]nfera[.]kvd\" | wc -l)'
  echo -n '    vram%: '; rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): [0-9]+' | grep -oE '[0-9]+$' | paste -sd," 2>&1 | grep -v libtinfow

echo
echo "===== 3. reboot the prefill leg (empty GPU cache, warm L3) ====="
bash "$D/boot.sh" prefill 262144 1 0 replay
echo "  waiting for ready..."
for i in $(seq 1 90); do
  H=$(spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR bash -c 'curl -sf -m5 http://$PIP:30000/health >/dev/null 2>&1 && echo 1 || echo 0'" 2>/dev/null | grep -v libtinfow | tr -d '[:space:]')
  [ "$H" = "1" ] && { echo "  ready after ~$((i*20))s"; break; }
  sleep 20
done

bash "$D/router.sh" 8190 2>&1 | tail -4

snap after_restart

echo
echo "===== 4. replay the IDENTICAL prompts ====="
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR python3 $W/scripts/replay_probe.py http://$PIP:8190 replay" 2>&1 | grep -v libtinfow

snap after_replay

echo
echo "===== VERDICT ====="
python3 - "$R" <<'EOF'
import json, sys, os
R = sys.argv[1]
def load(n):
    with open(os.path.join(R, f"kvd_{n}.json")) as f:
        return json.load(f)
a, b, c = load("before_replay"), load("after_restart"), load("after_replay")
print(f"  {'counter':<18} {'warm':>14} {'after restart':>14} {'after replay':>14}   delta(replay)")
for k in ("sets_total", "gets_total", "hits_total", "misses_total", "entries", "evictions_total"):
    d = c.get(k, 0) - b.get(k, 0)
    print(f"  {k:<18} {a.get(k,0):>14,} {b.get(k,0):>14,} {c.get(k,0):>14,}   {d:+,}")
gets = c.get("gets_total", 0) - b.get("gets_total", 0)
hits = c.get("hits_total", 0) - b.get("hits_total", 0)
sets = c.get("sets_total", 0) - b.get("sets_total", 0)
print()
print(f"  gets +{gets:,}   hits +{hits:,}   sets {sets:+,}")
print("  VERDICT:", "PASS - kvd SERVED the replay from L3 (reads, not re-writes)"
      if (gets > 0 and hits > 0 and sets == 0) else "FAIL - want gets>0, hits>0, sets==0")
EOF
