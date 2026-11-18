#!/usr/bin/env bash
# Runs one round: several variants side by side on one node of an existing allocation (srun
# --overlap), each in its own pyxis container on its own 4 GPUs. Light; runs on the login node.
#   usage: JOB=<jobid> round_env.sh <name>@<squash>@<patchset>@<gpus>@<port>[@K=V,K=V] [...]
set -uo pipefail
H=/apps/liyingli/holder-dsa
JOB=${JOB:?set JOB}
NODE=$(squeue -h -j "$JOB" -o %N)
[ -n "$NODE" ] || { echo "job $JOB not running"; exit 1; }
pids=()
for spec in "$@"; do
    IFS=@ read -r name sq patch gpus port env <<< "$spec"
    echo "$(date -u +%FT%TZ) round: $name on $NODE gpus $gpus ($(basename "$sq"), patchset $patch, env ${env:-none})"
    srun --jobid "$JOB" --overlap -N1 -n1 -w "$NODE" \
        --container-image="$sq" --container-remap-root --container-writable \
        --container-mounts=/perf_apps/data/models/GLM-5.2-MXFP4:/model,$H:/work,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri \
        --container-workdir=/work \
        bash /work/in_container_env.sh "$name" "$patch" "$gpus" "$port" "${env:-}" > "$H/out/$name.srun.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -u +%FT%TZ) round done rc=$rc"
exit $rc
