#!/usr/bin/env bash
# Artefact report for one run, in the order that keeps a green honest:
#   1. materials file count FIRST  -- a pass carries no reasons, so the count is
#      the only retrospective check on it (PRE-REGISTER-m2.md section 0)
#   2. verdicts with mtimes and validator identity
#   3. where each verdict landed
# Identity is never taken from `latest`, mtime or "newest dir" -- see bug record
# "two checks, one failure mode".
set -uo pipefail
R="${1:?usage: artefact_report.sh <run dir>}"
[ -d "$R" ] || { echo "no such run dir: $R" >&2; exit 2; }
P=/home/yihou/dev/git.16-19/infera/agent_sys/examples/llm_e2e_performance_optimization/e2e-flow

echo "run: $R"
echo "read at: $(date -u +%FT%TZ)  (RunTime=$(scontrol show job 29313 2>/dev/null | tr ' ' '\n' | grep RunTime= | cut -d= -f2))"
echo
echo "=== 1. were the zones shown anything? (before any attribution) ==="
bash "$P/assets/lib/refusal_saw_something.sh" "$R" 2>&1 | tail -6

echo
echo "=== 2. verdicts: mtime | validator | zone files | value ==="
n=0
while IFS='|' read -r t f; do
  [ -n "$t" ] || continue
  n=$((n+1)); z=$(dirname "$f")
  id=$(python3 -c "
import json
d=json.load(open('$z/args.json'))
print(d.get('layout') or ('check_environment' if 'compare_fixed_across_inputs' in d else ','.join(list(d)[:2])))" 2>/dev/null)
  [ -f "$z/probe_results.json" ] && id="$id (serves)"
  files=$(python3 -c "
import json,pathlib
m=json.load(open('$z/materials.json'))
print(sum(sum(1 for _ in pathlib.Path(p).rglob('*') if _.is_file()) for p in m.values()))" 2>/dev/null)
  printf '  %s  %-42s files=%-4s %s\n' "$t" "$id" "${files:-?}" "$(tr -d '\n ' < "$f")"
done < <(find "$R" -name 'verdict.json' -printf '%T+|%p\n' 2>/dev/null | sort)
echo "  total: $n"

echo
echo "=== 3. refusals, with the report line that names a file ==="
find "$R" -name 'verdict.json' | while read -r f; do
  grep -q 'false' "$f" || continue
  z=$(dirname "$f")
  echo "  zone $(basename "$z")"
  grep -m3 -E 'PROBLEM|REFUSED' "$z/validator_report.txt" 2>/dev/null | sed 's/^/    /' || echo "    (no validator_report.txt)"
done
