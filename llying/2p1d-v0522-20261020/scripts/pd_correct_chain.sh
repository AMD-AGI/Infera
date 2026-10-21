#!/usr/bin/env bash
# Correctness run on a kit, after both nodes' GPUs have released their VRAM: a fresh deployment with
# real MTP acceptance -> smoke.sh -> eval.sh (GSM8K) -> down.sh. Before it records who else is on the
# nodes into LOG_DIR/correct.env-check for manual review; it does not judge them.
# Usage: pd_correct_chain.sh KIT SITE_ENV LOG_DIR RUN_ID
set -uo pipefail
kit="$1" site="$2" log_dir="$3" run_id="$4"
set -a
# shellcheck source=/dev/null
. "$site"
set +a
mkdir -p "$log_dir"
nodes=(localhost "$DECODE_SSH")
on() { if [[ "$1" == localhost ]]; then bash -c "$2"; else ssh -o BatchMode=yes "$1" "$2"; fi; }
vram() { on "$1" "rocm-smi --showmeminfo vram --csv | awk -F, 'NR>1 && \$2 != \"\" {printf \"%d \", 100 * \$3 / \$2}'"; }

for n in "${nodes[@]}"; do
    until [[ -z "$(vram "$n" | tr ' ' '\n' | awk '$1 >= 1')" ]]; do sleep 60; done
done
{
    for n in "${nodes[@]}"; do
        echo "===== $n $(date -u +%FT%TZ)"
        on "$n" "rocm-smi --showmeminfo vram --csv; rocm-smi --showpids; docker stats --no-stream --format '{{.Name}} {{.CPUPerc}} {{.MemUsage}}'; uptime; ps -eo user,pcpu,etime,comm --sort=-pcpu | head -12"
    done
} >"$log_dir/$run_id.env-check" 2>&1
echo "$(date -u +%T) $run_id up"
(cd "$kit" && SIM_ACCEPT_LEN= RUN_ID="$run_id" bash engine/up.sh) >"$log_dir/$run_id.log" 2>&1 && {
    echo "$(date -u +%T) $run_id smoke"
    (cd "$kit" && bash engine/smoke.sh) >>"$log_dir/$run_id.log" 2>&1
    echo "$(date -u +%T) $run_id smoke exited $?"
    (cd "$kit" && bash engine/eval.sh) >>"$log_dir/$run_id.log" 2>&1
    echo "$(date -u +%T) $run_id eval exited $?"
}
(cd "$kit" && bash engine/down.sh) >>"$log_dir/$run_id.log" 2>&1
echo "$(date -u +%T) $run_id down"
