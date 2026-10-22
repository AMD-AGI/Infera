#!/usr/bin/env bash
# E0 -- H1 chunked-hipHostRegister micro-repro on ONE GPU of an allocated node.
# usage: run_e0.yihou.sh <node> <job> <gpu> [extra args for h1_chunked_register.yihou.py]
#   default: --size-gib 24 --chunk-gib 8 --touch on  (3 registrations, <=24 GiB pinned at a time)
# Probe-only (no GPU reads at all): append `--touch none`.
set -uo pipefail
node=${1:?node}; job=${2-}; gpu=${3:?gpu}; shift 3
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
IMG=${IMG:-infera-sglang:v0521-yihou-1009}
ts=$(date -u +%Y%m%d-%H%M%S)
name="e0-h1-yihou-g${gpu}-${ts}"
logdir="$D/diag/out.e0.yihou/$ts"
mkdir -p "$logdir"
# empty or "-" job -> plain ssh (node reachable directly); else spur exec shim
if [[ -z "$job" || "$job" == "-" ]]; then
    SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
    Rlogdir=/tmp/yihou_v0521dbg/e0/$ts   # /shared_nfs can be read-only on such nodes
else
    export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
    export SPUR_SSH_JOBS="$node:$job"
    SSH=(ssh)
    Rlogdir=$logdir
fi

args="--size-gib 24 --chunk-gib 8 --touch on --logdir $Rlogdir $*"
remote="mkdir -p $Rlogdir && docker run --rm --name $name --device /dev/kfd --device /dev/dri --group-add video \
 --memory 64g --ulimit memlock=-1:-1 --network none --ipc host \
 --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
 -e HIP_VISIBLE_DEVICES=$gpu -e PYTHONUNBUFFERED=1 \
 -v $D:$D:ro -v $Rlogdir:$Rlogdir --entrypoint bash $IMG -c 'python3 $D/diag/h1_chunked_register.yihou.py $args'"
echo "[run_e0] $node job=$job gpu=$gpu container=$name logdir=$logdir"
"${SSH[@]}" "$node" "$remote" 2>&1 | tee "$logdir/e0.parent.yihou.log"
[[ "$Rlogdir" != "$logdir" ]] && "${SSH[@]}" "$node" "tar -C $Rlogdir -cf - ." | tar -C "$logdir" -xf -
echo "[run_e0] done; VERDICT:"
grep -a '^VERDICT' "$logdir/e0.parent.yihou.log" || echo "VERDICT MISSING (see $logdir)"
