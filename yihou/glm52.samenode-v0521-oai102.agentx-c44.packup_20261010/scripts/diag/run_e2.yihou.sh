#!/usr/bin/env bash
# E2 -- production-size host pool registration on ONE GPU, direct child run.
# usage: run_e2.yihou.sh <node> <job|-> <gpu> <size_gib> <chunk_gib> <offs> [mem_limit e.g. 448g]
#   single : run_e2.yihou.sh crsuse2-m2m-138 - 7 420 512 0,200,300,end
#   default: run_e2.yihou.sh crsuse2-m2m-138 - 7 420 256 bnd-4096,bnd+4096   (2nd touch expected to fault)
set -uo pipefail
node=${1:?node}; job=${2-}; gpu=${3:?gpu}; size=${4:?size_gib}; chunk=${5:?chunk_gib}; offs=${6:?offs}; mem=${7:-448g}
D=/shared_nfs/yihou_glm52/v0521dbg_20261010
IMG=${IMG:-infera-sglang:v0521-yihou-1009}
ts=$(date -u +%Y%m%d-%H%M%S)
name="e2-yihou-g${gpu}-s${size}-c${chunk}-${ts}"
logdir="$D/diag/out.e2.yihou/${ts}.s${size}.c${chunk}"
mkdir -p "$logdir"
if [[ -z "$job" || "$job" == "-" ]]; then
    SSH=(ssh -o BatchMode=yes -o ClearAllForwardings=yes)
    Rlogdir=/tmp/yihou_v0521dbg/e2/$ts   # /shared_nfs can be read-only on such nodes
else
    export PATH=/shared_nfs/yihou_glm52/kvcap_20261008/scripts/shimbin:$PATH
    export SPUR_SSH_JOBS="$node:$job"
    SSH=(ssh); Rlogdir=$logdir
fi
remote="mkdir -p $Rlogdir && docker run --rm --name $name --device /dev/kfd --device /dev/dri --group-add video \
 --memory $mem --ulimit memlock=-1:-1 --network none --ipc host \
 --cap-add SYS_PTRACE --cap-add IPC_LOCK --security-opt seccomp=unconfined \
 -e HIP_VISIBLE_DEVICES=$gpu -e PYTHONUNBUFFERED=1 \
 -v $D:$D:ro -v $Rlogdir:$Rlogdir --entrypoint bash $IMG -c \
 'python3 $D/diag/h1_chunked_register.yihou.py --child --case e2 --size-gib $size --chunk-gib $chunk --offs $offs > $Rlogdir/e2.yihou.log 2>&1; echo CHILD_RC=\$? >> $Rlogdir/e2.yihou.log'"
echo "[run_e2] $node gpu=$gpu size=${size}GiB chunk=${chunk}GiB offs=$offs mem=$mem container=$name"
"${SSH[@]}" "$node" "$remote"
[[ "$Rlogdir" != "$logdir" ]] && "${SSH[@]}" "$node" "tar -C $Rlogdir -cf - ." | tar -C "$logdir" -xf -
grep -aE "^\[child\]|^\[touch\]|Memory access fault|CHILD_RC|Error|error" "$logdir/e2.yihou.log" | grep -av "^\[probe\]"
echo "[run_e2] log: $logdir/e2.yihou.log"
