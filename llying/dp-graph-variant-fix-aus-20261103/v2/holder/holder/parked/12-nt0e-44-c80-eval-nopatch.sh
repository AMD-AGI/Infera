#!/usr/bin/env bash
# holder item 12-nt0e-44-c80-eval-nopatch, rendered by srtctl in /apps/liyingli/holder/render/nt0e-44-c80
set -euo pipefail
dst="/home/liyingli/code/bench_agentx/infx-aus-ws/nt0-44-c80/inferencex-e2e/srt-slurm-1793616100-1-b6811fc29363/outputs/holder-${SLURM_JOB_ID}-12-nt0e-44-c80-eval-nopatch"
mkdir -p "$dst/logs"
cp "/apps/liyingli/holder/items/12-nt0e-44-c80-eval-nopatch/config.yaml" "$dst/config.yaml"
sed 's|^OUTPUT_DIR=.*|OUTPUT_DIR="'"$dst"'"|' "/apps/liyingli/holder/items/12-nt0e-44-c80-eval-nopatch/sbatch_script.sh" > "$dst/sbatch_script.sh"
echo "item 12-nt0e-44-c80-eval-nopatch: output $dst"
if [ -f "/apps/liyingli/holder/items/12-nt0e-44-c80-eval-nopatch/env0" ]; then
    # Restore the submit-time environment, keeping this allocation's own Slurm variables.
    while IFS= read -r -d '' kv; do
        case "${kv%%=*}" in SLURM_*|SBATCH_*|SALLOC_*|SRUN_*|PWD|OLDPWD|SHLVL|_|HOSTNAME|BASH_FUNC_*) ;; *) export "$kv" ;; esac
    done < "/apps/liyingli/holder/items/12-nt0e-44-c80-eval-nopatch/env0"
fi
# TEST (AUS, NODE ORDER): n04-33 runs a Kubernetes control plane on 2379/2380, so put
# srtctl's head/infra node (etcd, router) on n10-29.
export SLURM_NODELIST=smci355-ccs-aus-n10-29,smci355-ccs-aus-n04-33 SLURM_JOB_NODELIST=smci355-ccs-aus-n10-29,smci355-ccs-aus-n04-33
bash "$dst/sbatch_script.sh" > "$dst/logs/sweep_holder.log" 2>&1
