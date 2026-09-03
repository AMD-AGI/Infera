#!/bin/sh
# STEP 7 -- grade yourself with the code that will grade you.
#
# Runs `check_optimization_shape`'s OWN body over the handoff, in a throwaway
# zone built the way `validator/phase.py` builds one: `args.json`,
# `inputs.json`, `materials.json` in the cwd, and the body owes `verdict.json`.
# Same code, so a pass here is not a promise and a FAIL here is certain.
#
# It does NOT run `check_speedup_substantiated`'s measurement -- that is
# `cost: gpu_hours` and belongs to the phase. Its free half (the premise, the
# denominator, correctness) is already enforced by STEPs 2, 4 and 6.
#
# If it names a problem, **fix the problem**. Do not adjust a number until the
# check stops complaining.
set -eu

HANDOFF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --handoff) HANDOFF="$2"; shift 2 ;;
    *) echo "usage: 70_selfcheck.sh --handoff <content dir>" >&2; exit 1 ;;
  esac
done
[ -n "$HANDOFF" ] || { echo "--handoff is required" >&2; exit 1; }
[ -d "$HANDOFF" ] || { echo "no such directory: $HANDOFF" >&2; exit 1; }

PKG="${AGENT_SYS_TASK_PACKAGE:-${AGENT_SYS_DEMO_PACKAGE:?the runner exports one of these}}"
PY="${KFO_PYTHON:-python3}"
: "${KFO_SCRATCH_ROOT:?must be local disk inside a yihou/ directory}"

ZONE="$KFO_SCRATCH_ROOT/selfcheck"
rm -rf "$ZONE"        # named path, not a variable glob. `rm -rf "$d"/*` with $d
mkdir -p "$ZONE"      # unset is `rm -rf /*`, and that has happened on this host.

# The args the step yaml declares. Kept in step with `steps/m4_kernel_opt.yaml`
# by hand; a drift shows up as this check being kinder than the phase, which is
# the safe direction but is still worth noticing.
cat > "$ZONE/args.json" <<'EOF'
{
  "schema": "kernel_optimization",
  "require_apply_mode": ["overlay_files"],
  "require_fields": ["operator", "workset_ref", "apply", "evidence", "premise"],
  "min_content_lines": {"README.md": 12, "REPRODUCE.md": 8, "environment.md": 8, "notes.md": 8},
  "min_command_lines": 4,
  "required_evidence": [
    "results/kernel_optimization.json",
    "results/workset.snapshot.yaml",
    "results/workset.baseline_report.json",
    "results/optimized_kernel.py",
    "results/forge_result.json"
  ]
}
EOF
echo '["selfcheck"]' > "$ZONE/inputs.json"
"$PY" - "$ZONE" "$HANDOFF" <<'EOF'
import json, os, sys
zone, handoff = sys.argv[1], os.path.abspath(sys.argv[2])
json.dump({"selfcheck": os.path.relpath(handoff, zone)}, open(os.path.join(zone, "materials.json"), "w"), indent=2)
EOF

cd "$ZONE"
"$PY" "$PKG/assets/check_optimization_shape.validator/check.py"
"$PY" - <<'EOF'
import json, sys
verdict = json.load(open("verdict.json"))
if not all(verdict.values()):
    print("selfcheck: check_optimization_shape would FAIL this handoff", file=sys.stderr)
    sys.exit(1)
print("selfcheck: check_optimization_shape passes")
EOF
