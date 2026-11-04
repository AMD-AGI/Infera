#!/usr/bin/env bash
# Run the verification inside one 2-node Slurm allocation: the batch step runs on the first node (prefill), the second
# node is the decode. Partition / account / QoS / time go on the sbatch command line, e.g.
#   sbatch -p PART -A ACCT -q QOS -N 2 --gres=gpu:8 --exclusive -t 14:00:00 -o LOG_ROOT/slurm-%j.out \
#       scripts/slurm_verify.sbatch [STEPS...]
# Node values come from the allocation and override config/site.local.env. Decode commands use `ssh` when it works
# inside the job, otherwise tools/sshshim/ssh (srun --overlap into this job). On exit (end, cancel, preemption) the
# kit containers are removed on both nodes, so nothing is left holding GPUs on nodes handed to someone else.
#SBATCH -J lyl-vfix
#SBATCH -N 2
#SBATCH --exclusive
set -uo pipefail
PKG="${PKG:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
# sbatch copies the script to a spool dir; use the submit directory when the package is not found there
[[ -f "$PKG/config/recipe-e10.env" ]] || PKG="$(cd "${SLURM_SUBMIT_DIR:?}" && pwd)"
export PKG
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" 2>/dev/null || python3 - <<'EOF'
import os, re
for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?", os.environ["SLURM_JOB_NODELIST"]):
    m = re.match(r"(.*)\[(.*)\]$", part)
    if not m:
        print(part); continue
    for rng in m.group(2).split(","):
        a, _, b = rng.partition("-")
        for i in range(int(a), int(b or a) + 1):
            print(f"{m.group(1)}{i:0{len(a)}d}")
EOF
)
[[ ${#nodes[@]} -ge 2 ]] || { echo "need 2 nodes, got: ${nodes[*]}"; exit 1; }
p="$(hostname -s)"
done_mark="${PKG}/.vfix-job-${SLURM_JOB_ID}.done"
# Slurm runs the batch script once, on the first node. A scheduler that starts it on every node (spur: one task per
# node by default) must not get a second driver: every copy but the first node's waits for the driver and leaves.
if [[ "${p}" != "${nodes[0]%%.*}" || "${SLURM_PROCID:-0}" != 0 ]]; then
    echo "$(date -u +%T) $p: not the driver node (${nodes[0]}), waiting for it"
    while [[ ! -e "$done_mark" ]]; do sleep 30; done
    exit 0
fi
d="${nodes[1]}"
if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$d" true 2>/dev/null; then
    export SHIM_JOB="$SLURM_JOB_ID" PATH="$PKG/tools/sshshim:$PATH"
    echo "ssh to $d refused inside the job: using tools/sshshim (srun --overlap)"
fi
# The node's own name can map to 127.0.0.1 in /etc/hosts: take the source address of the default route on each node.
src_ip='ip -o -4 route get 1.1.1.1 | awk "{for (i = 1; i < NF; i++) if (\$i == \"src\") {print \$(i + 1); exit}}"'
export PREFILL_NODE_IP="${PREFILL_NODE_IP_OVERRIDE:-$(bash -c "$src_ip" | tr -d '[:space:]')}"
export DECODE_NODE_IP="${DECODE_NODE_IP_OVERRIDE:-$(ssh -o BatchMode=yes "$d" "$src_ip" | tail -1 | tr -d '[:space:]')}" DECODE_SSH="$d"
[[ "$PREFILL_NODE_IP" =~ ^[0-9.]+$ && "$PREFILL_NODE_IP" != 127.* && "$DECODE_NODE_IP" =~ ^[0-9.]+$ && "$DECODE_NODE_IP" != 127.* ]] ||
    { echo "could not find the node IPs (prefill '$PREFILL_NODE_IP', decode '$DECODE_NODE_IP'): set PREFILL_NODE_IP_OVERRIDE / DECODE_NODE_IP_OVERRIDE"; exit 1; }
set -a; . "$PKG/config/site.local.env"; set +a
mkdir -p "$LOG_ROOT"
cleanup() {
    echo "$(date -u +%T) job ending: removing kit containers"
    (set -a; . "$PKG/config/recipe-e10.env"; set +a; cd "$PKG/kit" && bash engine/down.sh) >/dev/null 2>&1
    touch "$done_mark"
}
trap cleanup EXIT TERM INT
echo "$(date -u +%T) job $SLURM_JOB_ID: prefill $p ($PREFILL_NODE_IP), decode $d ($DECODE_NODE_IP); LOG_ROOT=$LOG_ROOT"
STEPS="${*:-${STEPS:-}}" bash "$PKG/scripts/run_verify.sh" >> "$LOG_ROOT/verify.out" 2>&1
