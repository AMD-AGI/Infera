#!/usr/bin/env bash
set -euo pipefail
ROOT=/perf_apps/liyingli/bench_agentx/router-capacity-20260928
[[ ! -e "$ROOT/extra-job-id.txt" ]] || { echo 'An extra-node request is already recorded; inspect it before another request.' >&2; exit 1; }
EXTRA_EXCLUDE=${EXTRA_EXCLUDE:-smci355-ccs-aus-n04-29,smci355-ccs-aus-n01-25,smci355-ccs-aus-n10-29,smci355-ccs-aus-n01-21}
begin_args=()
if [[ -n "${EXTRA_BEGIN:-}" ]]; then begin_args+=(--begin="$EXTRA_BEGIN"); fi
job=$(sbatch --parsable "${begin_args[@]}" --account=emad --partition=Compute-DCPT --qos=batch --nodes=1 --ntasks-per-node=1 --exclusive --gres=gpu:mi355x:8 --mem=3000000M --time=06:00:00 --no-requeue --exclude="$EXTRA_EXCLUDE" --job-name=llying-campaign-p2 --output="$ROOT/extra-%j.slurm.log" --wrap='sleep infinity')
job=${job%%;*}
[[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected submission result: $job" >&2; exit 1; }
printf '%s\n' "$job" > "$ROOT/extra-job-id.txt"
date -u --iso-8601=seconds > "$ROOT/extra-requested-at.txt"
printf '%s\n' "${EXTRA_BEGIN:-immediate eligibility}" > "$ROOT/extra-begin.txt"
echo "Requested extra node: job=$job"
python3 "$ROOT/scripts/prepare_extra_node.py" --job "$job" --isolated-smoke
