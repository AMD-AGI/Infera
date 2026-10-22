#!/usr/bin/env bash
# E-dbg -- ROCm debug agent smoke (presence, kernel naming on a deliberate GPU fault
# eager + graph replay, steady-state overhead) on ONE GPU of an allocated node.
# usage: run_dbg.yihou.sh <node> <job> <gpu> [extra args for dbgagent_smoke.yihou.py]
#   IMG=infera-sglang:v0521-yihou-1009-dbgagent run_dbg.yihou.sh ...  # after building the thin layer
# Container flags mirror bench-harness/engine.sh (SYS_PTRACE, IPC_LOCK, seccomp=unconfined, ipc host)
# so a result here transfers to the engine containers.
set -uo pipefail
node=${1:?node}; job=${2-}; gpu=${3:?gpu}; shift 3
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
IMG=${IMG:-infera-sglang:v0521-yihou-1009}
ts=$(date -u +%Y%m%d-%H%M%S)
name="dbgagent-yihou-g${gpu}-${ts}"
outdir="$D/diag/out.dbg.yihou/$ts"
mkdir -p "$outdir"
# empty or "-" job -> plain ssh (node reachable directly); else spur exec shim
if [[ -z "$job" || "$job" == "-" ]]; then
    SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
    Routdir=/tmp/yihou_v0521dbg/dbg/$ts   # /shared_nfs can be read-only on such nodes
else
    export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
    export SPUR_SSH_JOBS="$node:$job"
    SSH=(ssh)
    Routdir=$outdir
fi

remote="mkdir -p $Routdir && docker run --rm --name $name --device /dev/kfd --device /dev/dri --group-add video \
 --memory 64g --network none --ipc host \
 --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
 -e HIP_VISIBLE_DEVICES=$gpu -e PYTHONUNBUFFERED=1 \
 -v $D:$D:ro -v $Routdir:$Routdir --entrypoint bash $IMG -c 'python3 $D/diag/dbgagent_smoke.yihou.py --outdir $Routdir $*'"
echo "[run_dbg] $node job=$job gpu=$gpu img=$IMG container=$name outdir=$outdir"
"${SSH[@]}" "$node" "$remote" 2>&1 | tee "$outdir/dbg.parent.yihou.log"
[[ "$Routdir" != "$outdir" ]] && "${SSH[@]}" "$node" "tar -C $Routdir -cf - ." | tar -C "$outdir" -xf -
echo "[run_dbg] done:"
grep -a -E '^(VERDICT|BENCH ratio|FAULT )' "$outdir/dbg.parent.yihou.log" || echo "VERDICT MISSING (see $outdir)"
