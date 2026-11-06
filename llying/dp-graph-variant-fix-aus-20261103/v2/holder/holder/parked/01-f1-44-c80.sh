#!/usr/bin/env bash
# holder item 01-f1-44-c80, rendered by srtctl as /home/liyingli/code/bench_agentx/infx-aus-ws/f1-44-c80/inferencex-e2e/srt-slurm-1793331060-1-ea20c0ef7174/outputs/34968
set -euo pipefail
dst="/home/liyingli/code/bench_agentx/infx-aus-ws/f1-44-c80/inferencex-e2e/srt-slurm-1793331060-1-ea20c0ef7174/outputs/holder-${SLURM_JOB_ID}-01-f1-44-c80"
mkdir -p "$dst/logs"
cp "/home/liyingli/code/bench_agentx/infx-aus-ws/f1-44-c80/inferencex-e2e/srt-slurm-1793331060-1-ea20c0ef7174/outputs/34968/config.yaml" "$dst/config.yaml"
sed 's|^OUTPUT_DIR=.*|OUTPUT_DIR="'"$dst"'"|' "/home/liyingli/code/bench_agentx/infx-aus-ws/f1-44-c80/inferencex-e2e/srt-slurm-1793331060-1-ea20c0ef7174/outputs/34968/sbatch_script.sh" > "$dst/sbatch_script.sh"
echo "item 01-f1-44-c80: output $dst"
exec bash "$dst/sbatch_script.sh" > "$dst/logs/sweep_holder.log" 2>&1
