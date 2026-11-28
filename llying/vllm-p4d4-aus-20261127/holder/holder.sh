#!/usr/bin/env bash
# Per-node queue runner. One copy runs on each node of an allocation (srun into job 37014, or holder.sbatch for the
# follow-up jobs). Items are bash scripts dropped into a queue; a node runs its own queue first, then the shared one.
#   queue:   $H/q-any/NN-label.sh  (either node)   $H/q-<node>/NN-label.sh  (that node only)
#   running: $H/running/<name>.sh (+ .node)       done: $H/done/<name>.sh + .rc, log in $H/logs/<name>.log
#   each:    $H/q-each/NN-label.sh  (every node runs it once, before its other items; done marks in $H/done-each-<node>/)
#   stop:    touch $H/STOP (every runner exits after its current item); touch $H/STOP-<node> (that node only)
# Items get NODE (short hostname) and ITEM in the environment and run from $W. Each node runs one item at a time:
# a P4D4 uses all 8 GPUs.
set -uo pipefail
W=/perf_apps/liyingli/bench_agentx/vllm-p4d4-aus-20261127
H=$W/holder
node=$(hostname -s)
mkdir -p "$H"/{q-any,q-each,q-$node,running,done,done-each-$node,logs}
say() { echo "$(date -u +%FT%TZ) $node job ${SLURM_JOB_ID:-?}: $*" | tee -a "$H/holder.log"; }
# Our containers (vpd-*) are removed when the runner ends (job end / scancel), so the next job starts clean.
trap 'docker ps -a --format "{{.Names}}" | grep -E "^vpd-" | xargs -r docker rm -f > /dev/null 2>&1' EXIT TERM INT
say "runner up (amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null))"
# A preempted / ended job can leave our containers behind on this node: start clean.
docker ps -a --format '{{.Names}}' | grep -E '^vpd-' | xargs -r docker rm -f > /dev/null 2>&1
# Nodes of an earlier allocation that this one does not have: their node queues and unfinished items go to q-any
# (the item scripts are node-agnostic; only the pairing intent is lost, which is logged).
nodes=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$node}" 2>/dev/null)
for q in "$H"/q-smci*; do
    [ -d "$q" ] || continue
    qn=${q##*/q-}
    grep -qx "$qn" <<< "$nodes" && continue
    for f in "$q"/*.sh; do [ -e "$f" ] && { say "node $qn gone: $(basename "$f") -> q-any"; mv "$f" "$H/q-any/"; }; done
done
for f in "$H"/running/*.sh; do
    [ -e "$f" ] || continue
    on=$(cat "${f%.sh}.node" 2>/dev/null)
    grep -qx "$on" <<< "$nodes" && continue
    say "item of node $on (not in this allocation): $(basename "$f") -> q-any"; rm -f "${f%.sh}.node"; mv "$f" "$H/q-any/"
done
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
    each=""
    for f in $(ls "$H/q-each"/*.sh 2>/dev/null | sort); do
        [ -e "$H/done-each-$node/$(basename "$f" .sh).rc" ] || { each=$f; break; }
    done
    if [ -n "$each" ]; then
        name=$(basename "$each" .sh)
        say "run (each) $name"
        ( cd "$W" && NODE=$node ITEM=$name bash "$each" ) > "$H/logs/$name-$node.log" 2>&1 < /dev/null
        rc=$?; echo "$rc" > "$H/done-each-$node/$name.rc"; say "done (each) $name rc=$rc"
        continue
    fi
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
