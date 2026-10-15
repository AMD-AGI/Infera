#!/usr/bin/env bash
# Capture a stalled engine instance for root-causing. For every scheduler process: py-spy
# (native) and rocgdb (GPU agents, HSA queues with their read/write indices, active dispatches,
# then backtraces of all threads, i.e. ROCr's native threads and the resident GPU waves), plus
# rocm-smi use / memory of the node. py-spy only sees Python threads; a queue blocked inside
# the runtime (e.g. waiting for scratch) shows up only in the rocgdb part.
# rocgdb attach stops the process for seconds to minutes: run this only on an instance that
# is already stalled (scripts/hang_watch.py does).
# Usage: hang_capture.sh INSTANCE=<prefill-N|decode-0> [ROUND=r2]
# Output: runtime/runs/<ROUND>/tune/hang-<INSTANCE>-<MMDD-HHMM>/
set -euo pipefail
for a in "$@"; do
    [[ "$a" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] || { echo "expected KEY=VALUE, got '$a'" >&2; exit 2; }
    export "$a"
done
: "${INSTANCE:?}"
export ROUND="${ROUND:-r2}"
HERE="$(cd "$(dirname "$0")" && pwd)"
R="$(cd "$HERE/.." && pwd)"
set -a; source "$R/config/config.3p1d.sh"; set +a
node="$(python3 "$BENCH_DIR/tools/topology.py" rows "$TOPOLOGY" | awk -v i="$INSTANCE" '$2 == i {print $4}')"
[[ -n "$node" ]] || { echo "unknown instance: $INSTANCE" >&2; exit 2; }
container="$CONTAINER_PREFIX-$INSTANCE"
out="$RUNS/tune/hang-$INSTANCE-$(date -u +%m%d-%H%M)"
mkdir -p "$out"
on() { ssh -o BatchMode=yes -o ConnectTimeout=10 -n "$node" "$@"; }

on "date -u +%FT%TZ; rocm-smi --showuse --showmemuse --showmeminfo vram --csv; free -g" > "$out/rocm-smi.txt" 2>&1 || true
pids="$(on "docker exec $container ps -eo pid=,comm=" | awk '$2 ~ /^sglang::schedul/ {print $1}')"
[[ -n "$pids" ]] || { echo "no scheduler processes in $container" | tee "$out/error.txt" >&2; exit 1; }
for p in $pids; do
    { echo "=== pid $p"; on "docker exec $container py-spy dump --native --pid $p" 2>&1 || true; } >> "$out/py-spy.txt"
done
# Pass 1 is quick (which kernels are in flight on which queue, for every rank); pass 2 can be
# long when many waves are resident. timeout runs inside the container so it reaches rocgdb.
gdb="/opt/rocm/bin/rocgdb -batch -nx -ex 'set pagination off'"
for p in $pids; do
    on "docker exec $container timeout -s KILL 300 $gdb -p $p -ex 'info agents' -ex 'info queues' -ex 'info dispatches' -ex detach" \
        > "$out/rocgdb-queues-$p.txt" 2>&1 || true
done
for p in $pids; do
    on "docker exec $container timeout -s KILL 900 $gdb -p $p -ex 'info threads' -ex 'thread apply all bt 20' -ex detach" \
        > "$out/rocgdb-threads-$p.txt" 2>&1 || true
done
echo "$out"
