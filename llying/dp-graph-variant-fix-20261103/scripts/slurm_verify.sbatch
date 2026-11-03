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
d="${nodes[0]}"; [[ "${nodes[0]%%.*}" == "$p" ]] && d="${nodes[1]}"
ip_of() { getent hosts "$1" | awk '{print $1; exit}'; }
export PREFILL_NODE_IP="${PREFILL_NODE_IP_OVERRIDE:-$(ip_of "$p")}" DECODE_NODE_IP="${DECODE_NODE_IP_OVERRIDE:-$(ip_of "$d")}" DECODE_SSH="$d"
if ! timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$d" true 2>/dev/null; then
    export SHIM_JOB="$SLURM_JOB_ID" PATH="$PKG/tools/sshshim:$PATH"
    echo "ssh to $d refused inside the job: using tools/sshshim (srun --overlap)"
fi
set -a; . "$PKG/config/site.local.env"; set +a
mkdir -p "$LOG_ROOT"
cleanup() {
    echo "$(date -u +%T) job ending: removing kit containers"
    (set -a; . "$PKG/config/recipe-e10.env"; set +a; cd "$PKG/kit" && bash engine/down.sh) >/dev/null 2>&1
}
trap cleanup EXIT TERM INT
echo "$(date -u +%T) job $SLURM_JOB_ID: prefill $p ($PREFILL_NODE_IP), decode $d ($DECODE_NODE_IP); LOG_ROOT=$LOG_ROOT"
STEPS="${*:-${STEPS:-}}" bash "$PKG/scripts/run_verify.sh" >> "$LOG_ROOT/verify.out" 2>&1
