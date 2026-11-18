#!/usr/bin/env bash
# Runs mla_bench.py in several images side by side, one GPU each, inside an existing allocation.
#   usage: JOB=<jobid> bench_round.sh <name>@<squash>@<gpu> [...]
set -uo pipefail
H=/apps/liyingli/holder-dsa
JOB=${JOB:?set JOB}
NODE=$(squeue -h -j "$JOB" -o %N)
[ -n "$NODE" ] || { echo "job $JOB not running"; exit 1; }
pids=()
for spec in "$@"; do
    IFS=@ read -r name sq gpu <<< "$spec"
    mkdir -p "$H/out/bench"
    echo "$(date -u +%FT%TZ) bench: $name on $NODE gpu $gpu ($(basename "$sq"))"
    srun --jobid "$JOB" --overlap -N1 -n1 -w "$NODE" \
        --container-image="$sq" --container-remap-root --container-writable \
        --container-mounts=$H:/work,/dev/kfd:/dev/kfd,/dev/dri:/dev/dri --container-workdir=/work \
        bash -c "export HIP_VISIBLE_DEVICES=$gpu HOME=/tmp/home-$name TRITON_CACHE_DIR=/work/out/bench/triton-$name; mkdir -p \$HOME;
                 for m in module off on; do
                     echo \"### async-copy \$m\"
                     python3 /work/mla_bench.py --async-copy \$m --out /work/out/bench/$name-\$m.json
                 done" > "$H/out/bench/$name.log" 2>&1 &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "$(date -u +%FT%TZ) bench round done rc=$rc"
exit $rc
