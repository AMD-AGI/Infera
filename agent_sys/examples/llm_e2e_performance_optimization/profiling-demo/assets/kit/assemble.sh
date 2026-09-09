#!/usr/bin/env bash
# Assemble the five upstream handoffs into one packup a colleague can follow.
#
# Runs on the LOGIN NODE and touches no cluster: everything it needs was already
# published by the tasks before it. That is the point of the step -- if it needed
# the machine, the machine would be part of the deliverable.
#
# The layout is the experiment-result-packup skill's, in
# temp/claude_code_skill_used_by_human/experiment-result-packup/references/deliverable_layout.md:
# README, REPRODUCE, environment, scripts, results, notes, logs. `patches/` is
# omitted rather than shipped empty -- the skill says to omit what does not apply.
set -uo pipefail

PKG="${AGENT_SYS_TASK_PACKAGE:?}"
OUT="${PD_OUTPUT_PACKUP:?}"

say() { printf '[packup] %s\n' "$*"; }

# Every input is a staged content directory. Absent ones are named rather than
# skipped: a packup missing a round is a different artefact from one that has it.
need() {
  local var="$1" path="${!1:-}"
  if [ -z "$path" ] || [ ! -d "$path" ]; then
    say "ABORT: $var is unset or not a directory: ${path:-<unset>}"
    exit 1
  fi
  printf '%s' "$path"
}
DEP_BASE="$(need PD_INPUT_DEPLOYMENT_BASELINE)"
DEP_PROF="$(need PD_INPUT_DEPLOYMENT_PROFILED)"
AIP_BASE="$(need PD_INPUT_AIPERF_BASELINE)"
AIP_PROF="$(need PD_INPUT_AIPERF_PROFILED)"
KERNELS="$(need PD_INPUT_KERNEL_TABLE)"

STAMP="$(date +%Y%m%d)"
ROOT="$OUT/items/codes"
rm -rf "$OUT/items"; mkdir -p "$ROOT"/{scripts,results,logs}

say "assembling -> $ROOT"

# ---- scripts: verbatim, from the package rather than from a handoff ---------
# The skill says copy the real thing so byte-level flags survive. The handoffs
# carry `command` items with placeholders instead of the scripts, for the
# locality reason recorded in temp/bugs/002 -- so the scripts come from the
# package, which is where they actually live.
cp "$PKG/assets/serve"/*.sh "$ROOT/scripts/"
cp "$PKG/assets/load"/*.sh "$ROOT/scripts/"
cp "$PKG/assets/analyze"/*.sh "$ROOT/scripts/"
mkdir -p "$ROOT/scripts/pythonpath"
cp "$PKG/assets/load/pythonpath/sitecustomize.py" "$ROOT/scripts/pythonpath/"
# The `command` item of each handoff: the invocation as it actually ran.
for pair in "deployment_baseline:$DEP_BASE" "deployment_profiled:$DEP_PROF" \
            "aiperf_baseline:$AIP_BASE" "aiperf_profiled:$AIP_PROF" \
            "kernel_table:$KERNELS"; do
  name="${pair%%:*}"; dir="${pair#*:}"
  [ -f "$dir/items/command" ] && cp "$dir/items/command" "$ROOT/scripts/command.$name.sh"
done
chmod +x "$ROOT"/scripts/*.sh 2>/dev/null || true

# ---- results: the numbers that back the claim -------------------------------
cp "$AIP_BASE/items/result/summary.json"  "$ROOT/results/aiperf_baseline.summary.json" 2>/dev/null || true
cp "$AIP_PROF/items/result/summary.json"  "$ROOT/results/aiperf_profiled.summary.json" 2>/dev/null || true
cp "$AIP_BASE/items/result/profile_export_aiperf.csv" "$ROOT/results/aiperf_baseline.csv" 2>/dev/null || true
cp "$AIP_PROF/items/result/profile_export_aiperf.csv" "$ROOT/results/aiperf_profiled.csv" 2>/dev/null || true
cp "$KERNELS/items/result/gap_analysis/gap_analysis.csv" "$ROOT/results/" 2>/dev/null || true
cp "$KERNELS/items/result/top_kernels.json" "$ROOT/results/" 2>/dev/null || true
cp "$KERNELS/items/env/trace_manifest.json" "$ROOT/results/" 2>/dev/null || true
cp "$DEP_BASE/items/result/smoke.txt" "$ROOT/results/smoke.baseline.txt" 2>/dev/null || true

# ---- logs: gzipped, as the skill suggests -----------------------------------
for pair in "baseline:$DEP_BASE" "profiled:$DEP_PROF"; do
  name="${pair%%:*}"; dir="${pair#*:}"
  for f in "$dir/items/logs"/*.gz; do
    [ -f "$f" ] && cp "$f" "$ROOT/logs/$name.$(basename "$f")"
  done
done
cp "$KERNELS/items/logs"/*.gz "$ROOT/logs/" 2>/dev/null || true

# ---- environment.md ----------------------------------------------------------
python3 "$PKG/assets/kit/render.py" \
  --root "$ROOT" \
  --stamp "$STAMP" \
  --deployment-baseline "$DEP_BASE" \
  --deployment-profiled "$DEP_PROF" \
  --aiperf-baseline "$AIP_BASE" \
  --aiperf-profiled "$AIP_PROF" \
  --kernel-table "$KERNELS" || { say "ABORT: could not render the packup documents"; exit 1; }

# ---- the handoff's own README ------------------------------------------------
# content_type `code`, so the sections are Purpose / Interface / Boundary. The
# packup's own README.md lives inside items/codes/ and is a different document.
cat > "$OUT/README.md" <<EOF
# profile_packup

## Purpose

One directory a colleague can be handed: what was run, on what, what came out,
and how to do it again. Assembled from the five handoffs this pipeline published,
in the layout the experiment-result-packup skill defines.

## Interface

\`items/codes/\` is the packup. Read \`README.md\` inside it first; it carries the
result and a map of the rest. \`REPRODUCE.md\` is the file a reproducer executes,
\`environment.md\` the exhaustive record of what the numbers came from,
\`scripts/\` the scripts verbatim, \`results/\` the machine-readable evidence and
\`logs/\` the gzipped log tails.

Site paths are written as \`@NAME@\` throughout; \`REPRODUCE.md\` says what each one
was and what it should be on a new machine.

## Boundary

This packup does **not** carry the torch traces. They are about 462 MB per
capture and the ranking derived from them is here instead;
\`results/trace_manifest.json\` identifies them by SHA-256 so a copy held
elsewhere can be matched to this run.

It also does not carry \`patches/\`. Nothing in this pipeline needed a patch to the
engine, and the skill says to omit a folder rather than ship it empty.
EOF

# ---- make it publishable -----------------------------------------------------
say "redacting site-specific paths"
python3 "$PKG/assets/lib/redact.py" "$OUT" \
  "MODEL_MOUNT=$(dirname "${PD_MODEL:?}")" \
  "WORK_ROOT=${PD_WORK_ROOT:?}" \
  "MAGPIE_ROOT=${PD_MAGPIE_ROOT:?}" \
  "TASK_PACKAGE=$PKG" \
  "TRACE_DIR=$(dirname "${PD_AIPERF_TRACE:?}")" \
  "TMPDIR=/tmp" \
  "HOME=$HOME" || { say "ABORT: the packup still names local paths redact.py could not place"; exit 1; }

say "done: $(find "$ROOT" -type f | wc -l) file(s) in the packup"
exit 0
