#!/usr/bin/env bash
# A correctness run (pd_correct_chain.sh: real acceptance; smoke, GSM8K, the 200-prompt probe and the long-context
# probes of LONGCTX_SPECS) and then the AgentX points of PLAN_FILE (perf_plan.sh) whatever the correctness result:
# the perf points are the baseline for prefill experiments, which do not depend on the decode being correct.
# Usage: [KEY=VALUE ...] correct_and_perf.sh KIT CORRECT_SITE LOG_DIR PLAN_FILE
set -uo pipefail
kit="$1" site="$2" log_dir="$3" plan="$4"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llying="$here/../.."
export PROBE_PROMPTS="${PROBE_PROMPTS:-/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/hicache0-n200/runs/hicache0-n200/gsm8k/glm5.2-mxfp4/samples_gsm8k_2026-10-19T19-22-22.773074.jsonl}"
export LONGCTX_SPECS="${LONGCTX_SPECS-16000:128 100000:32}"
run_id="correct-$(date -u +%m%dT%H%M)"
mkdir -p "$log_dir"
bash "$llying/2p1d-v0522-20261020/scripts/pd_correct_chain.sh" "$kit" "$site" "$log_dir" "$run_id"
log="$log_dir/$run_id.log"
echo "$(date -u +%T) correctness: $(grep -a 'checks passed' "$log" | tail -1)," \
    "GSM8K strict $(grep -a 'strict-match' "$log" | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')," \
    "probe: $(grep -a '^n=200 ' "$log" | tail -1)"
grep -aE '^(needle|gsm8k): n=' "$log" | sed "s/^/$(date -u +%T) longctx /"
unset PROBE_PROMPTS LONGCTX_SPECS
bash "$llying/prefill-car-20261023/scripts/perf_plan.sh" "$kit" "$log_dir" "$plan"
