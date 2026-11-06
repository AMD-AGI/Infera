#!/usr/bin/env bash
# holder item 06-f3e-2p1d-c176, rendered by srtctl in /apps/liyingli/holder/render/f3e-2p1d-c176
set -euo pipefail
dst="/home/liyingli/code/bench_agentx/infx-aus-ws/f3-2p1d-c176/inferencex-e2e/srt-slurm-1793344195-1-81a3bb0511c9/outputs/holder-${SLURM_JOB_ID}-06-f3e-2p1d-c176"
mkdir -p "$dst/logs"
cp "/apps/liyingli/holder/items/06-f3e-2p1d-c176/config.yaml" "$dst/config.yaml"
sed 's|^OUTPUT_DIR=.*|OUTPUT_DIR="'"$dst"'"|' "/apps/liyingli/holder/items/06-f3e-2p1d-c176/sbatch_script.sh" > "$dst/sbatch_script.sh"
echo "item 06-f3e-2p1d-c176: output $dst"
if [ -f "/apps/liyingli/holder/items/06-f3e-2p1d-c176/env0" ]; then
    # Restore the submit-time environment, keeping this allocation's own Slurm variables.
    while IFS= read -r -d '' kv; do
        case "${kv%%=*}" in SLURM_*|SBATCH_*|SALLOC_*|SRUN_*|PWD|OLDPWD|SHLVL|_|HOSTNAME|BASH_FUNC_*) ;; *) export "$kv" ;; esac
    done < "/apps/liyingli/holder/items/06-f3e-2p1d-c176/env0"
fi
bash "$dst/sbatch_script.sh" > "$dst/logs/sweep_holder.log" 2>&1
