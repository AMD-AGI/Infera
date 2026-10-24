#!/usr/bin/env bash
# A correctness run (../../2p1d-v0522-20261020/scripts/pd_correct_chain.sh with the 200-prompt probe; real
# acceptance) under the caller's overrides (e.g. PREFILL_EXTRA_ARGS), then, only if smoke is 6/6, GSM8K strict-match
# is at least 0.96 and the probe saw no garbled answer, the AgentX points of PLAN_FILE (scripts/perf_plan.sh).
# Usage: [KEY=VALUE ...] correct_then_plan.sh KIT CORRECT_SITE LOG_DIR PLAN_FILE
set -uo pipefail
kit="$1" site="$2" log_dir="$3" plan="$4"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROBE_PROMPTS="${PROBE_PROMPTS:-/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/hicache0-n200/runs/hicache0-n200/gsm8k/glm5.2-mxfp4/samples_gsm8k_2026-10-19T19-22-22.773074.jsonl}"
run_id="correct-$(date -u +%m%dT%H%M)"
mkdir -p "$log_dir"
bash "$here/../../2p1d-v0522-20261020/scripts/pd_correct_chain.sh" "$kit" "$site" "$log_dir" "$run_id"
log="$log_dir/$run_id.log"
smoke="$(grep -a 'checks passed' "$log" | tail -1)"
strict="$(grep -a 'strict-match' "$log" | tail -1 | awk -F'|' '{print $8}' | tr -d ' ')"
probe="$(grep -a '^n=200 ' "$log" | tail -1)"
echo "$(date -u +%T) correctness: ${smoke:-no smoke}, GSM8K strict ${strict:-none}, probe: ${probe:-none}"
if [[ "$smoke" == "6/6 checks passed" && "$probe" == *" garbled=0 "* ]] && awk -v s="${strict:-0}" 'BEGIN {exit !(s >= 0.96)}'; then
    unset PROBE_PROMPTS
    bash "$here/perf_plan.sh" "$kit" "$log_dir" "$plan"
else
    echo "$(date -u +%T) correctness gate failed; plan not run"
fi
