#!/bin/bash
# The kvd SERVING proof: restart the engine, keep the daemon, replay.
#
# WHY NOTHING ELSE WORKS. A latency win proves nothing about kvd -- SGLang's
# in-GPU radix cache serves a repeated prefix without ever consulting L3. That
# is exactly why the counters currently read thousands of `sets` and ZERO
# `gets`: the GPU tier answers first. The only clean attribution is to empty the
# GPU tier while leaving L3 alive:
#
#   1. kill the ENGINE only -- the kvd daemon and its L3 keep running
#   2. poll every GPU to VRAM 0% (release is ASYNCHRONOUS: after kill -9 the
#      processes go Z and rocm-smi still reads ~90% for a minute with no live
#      holder; booting before it drains OOMs)
#   3. boot the prefill leg again -- empty GPU radix cache, warm L3
#   4. replay byte-identical prompts
#
# WANT: gets_total and hits_total CLIMB while sets_total stays FLAT. `sets`
# staying put is the load-bearing part -- it means reads, not re-writes.
#
# Do NOT restart the container to do this: it would repopulate nothing.
#
# Usage: restart_replay.sh
set -eu
W=/shared_nfs/yihou_agbench_mtp
PJOB="${PJOB:-24300}"; PIP="${PIP:-10.245.157.89}"
CTR=agbench_mtp
R=$W/results

mkdir -p "$R"

snap() {  # $1 = label
  spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR python3 -m infera.kvd.statctl --socket /tmp/kvd/kvd.sock" 2>&1 \
    | grep -v libtinfow > "$R/kvd_$1.json"
  echo "  [$1]"; sed 's/^/    /' "$R/kvd_$1.json"
}

echo "===== 0. warm L3: drive the replay corpus once ====="
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR python3 $W/scripts/replay_probe.py http://$PIP:8190 warm" 2>&1 | grep -v libtinfow

snap before_replay

echo
echo "===== 1. kill the ENGINE only (kvd daemon SURVIVES) ====="
# The pattern is BRACKETED ([s]glang) on purpose. A bare
# `pkill -9 -f infera.engine.sglang` matches the `bash -c '...'` command string
# that CONTAINS that text -- i.e. this very shell -- so pkill kills itself and
# the step hangs forever with the engine already dead. Same trap router.sh
# documents for `infera.server`; it cost a stalled round here.
spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
  docker exec $CTR bash -c '
    pkill -9 -f \"[s]glang.launch_server\" 2>/dev/null
    pkill -9 -f \"[i]nfera.engine.sglang\" 2>/dev/null
    sleep 3
    echo -n \"    engine procs left: \"; ps aux | grep -cE \"[l]aunch_server|[i]nfera[.]engine[.]sglang\"
    echo -n \"    kvd daemon alive:  \"; ps aux | grep -c \"[i]nfera[.]kvd\"
  '" 2>&1 | grep -v libtinfow

echo
echo "===== 2. poll all 8 GPUs to VRAM 0% (release is asynchronous) ====="
for i in $(seq 1 60); do
  U=$(spur exec "$PJOB" bash -c "rocm-smi --showmemuse 2>/dev/null | grep -oE 'GPU Memory Allocated \(VRAM%\): [0-9]+' | grep -oE '[0-9]+$' | paste -sd,"  2>&1 | grep -v libtinfow | tr -d '[:space:]')
  MAX=$(echo "$U" | tr ',' '\n' | sort -n | tail -1)
  echo "  [${i}] vram% = $U"
  [ "${MAX:-99}" -le 1 ] && { echo "  all GPUs drained"; break; }
  sleep 10
done

echo
echo "===== 3. reboot the prefill leg (empty GPU cache, warm L3) ====="
bash "$(dirname "$0")/boot.sh" prefill 262144 1 0 replay
echo "  waiting for ready..."
for i in $(seq 1 90); do
  H=$(spur exec "$PJOB" bash -c "export DOCKER_CONFIG=/tmp/dockercfg
    docker exec $CTR bash -c 'curl -sf -m5 http://$PIP:30000/health >/dev/null 2>&1 && echo 1 || echo 0'" 2>/dev/null | grep -v libtinfow | tr -d '[:space:]')
  [ "$H" = "1" ] && { echo "  ready after $((i*20))s"; break; }
  sleep 20
done

# The router must rediscover the new worker instance.
bash "$(dirname "$0")/router.sh" 8190 2>&1 | tail -4

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
print(f"  {'counter':<18} {'before':>14} {'after restart':>14} {'after replay':>14}   delta(replay)")
for k in ("sets_total", "gets_total", "hits_total", "misses_total", "entries", "evictions_total"):
    d = c.get(k, 0) - b.get(k, 0)
    print(f"  {k:<18} {a.get(k,0):>14,} {b.get(k,0):>14,} {c.get(k,0):>14,}   {d:+,}")
gets  = c.get("gets_total", 0)  - b.get("gets_total", 0)
hits  = c.get("hits_total", 0)  - b.get("hits_total", 0)
sets  = c.get("sets_total", 0)  - b.get("sets_total", 0)
print()
ok = gets > 0 and hits > 0 and sets == 0
print(f"  gets +{gets:,}   hits +{hits:,}   sets {sets:+,}")
print("  VERDICT:", "PASS - kvd SERVED the replay from L3 (reads, not re-writes)"
      if ok else
      f"FAIL - want gets>0, hits>0, sets==0")
EOF
