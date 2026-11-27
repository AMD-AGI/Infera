#!/usr/bin/env bash
# Per-node queue runner. One copy runs on each node of an allocation (srun into job 37014, or holder.sbatch for the
# follow-up jobs). Items are bash scripts dropped into a queue; a node runs its own queue first, then the shared one.
#   queue:   $H/q-any/NN-label.sh  (either node)   $H/q-<node>/NN-label.sh  (that node only)
#   running: $H/running/<name>.sh (+ .node)       done: $H/done/<name>.sh + .rc, log in $H/logs/<name>.log
#   stop:    touch $H/STOP (every runner exits after its current item); touch $H/STOP-<node> (that node only)
# Items get NODE (short hostname) and ITEM in the environment and run from $W. Each node runs one item at a time:
# a P4D4 uses all 8 GPUs.
set -uo pipefail
W=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127
H=$W/holder
node=$(hostname -s)
mkdir -p "$H"/{q-any,q-$node,running,done,logs}
say() { echo "$(date -u +%FT%TZ) $node job ${SLURM_JOB_ID:-?}: $*" | tee -a "$H/holder.log"; }
# Our containers (vpd-*) are removed when the runner ends (job end / scancel), so the next job starts clean.
trap 'docker ps -a --format "{{.Names}}" | grep -E "^vpd-" | xargs -r docker rm -f > /dev/null 2>&1' EXIT TERM INT
say "runner up (amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null))"
# Items left in running/ by a dead runner of this node go back to the front of its queue.
for f in "$H"/running/*.sh; do
    [ -e "$f" ] || continue
    [[ "$(cat "${f%.sh}.node" 2>/dev/null)" == "$node" ]] || continue
    say "requeue $(basename "$f")"; rm -f "${f%.sh}.node"; mv "$f" "$H/q-$node/"
done
# Node check first (results/probe-<node>/: driver, NICs, GPU-NIC map, GEM / libionic decision), once per node.
if [ ! -e "$W/results/probe-$node/done" ]; then
    say "probe"; bash "$W/probe/probe_node.sh" > "$H/logs/probe-$node.log" 2>&1 < /dev/null
    say "probe done: $(tail -1 "$W/results/probe-$node/summary.txt" 2>/dev/null)"
fi
while [ ! -e "$H/STOP" ] && [ ! -e "$H/STOP-$node" ]; do
    item=$( (ls "$H/q-$node"/*.sh 2>/dev/null | sort; ls "$H/q-any"/*.sh 2>/dev/null | sort) | head -1)
    if [ -z "$item" ]; then touch "$H/hb-$node"; sleep 15; continue; fi
    name=$(basename "$item" .sh)
    mv "$item" "$H/running/" 2>/dev/null || continue      # the other node took it
    echo "$node" > "$H/running/$name.node"
    say "run $name"
    ( cd "$W" && NODE=$node ITEM=$name bash "$H/running/$name.sh" ) > "$H/logs/$name.log" 2>&1 < /dev/null
    rc=$?
    echo "$rc" > "$H/done/$name.rc"
    mv "$H/running/$name.sh" "$H/done/"; rm -f "$H/running/$name.node"
    say "done $name rc=$rc"
done
say "stop file found; exiting"
