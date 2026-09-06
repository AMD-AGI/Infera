#!/bin/bash
# One record per run, for a campaign whose goal is defects rather than greens.
#
#     bash assets/lib/collect_run.sh <run dir> [launcher script] [orchestrator log]
#
# **`bash`, not `sh`** — uses `set -o pipefail`, which dash rejects at runtime
# with `Illegal option -o pipefail`. Verified by running, not by `dash -n`,
# which accepts an invalid option because it is a syntax check.
#
# ## Why this exists
#
# **A run does not record the line that launched it.** Measured 2026-09-05: the
# staged copy of a step still reads `expect_ranks: '${expect_ranks:-8}'`, so
# whether a run passed 4 or fell back to 8 is unrecoverable afterwards — and at
# least four separate incidents traced to a launch variable nobody could check.
# The launcher is the only place the line exists, so this copies it beside the
# record while it still exists.
#
# ## The three questions it answers, and why these three
#
# 1. **Was any validator handed an empty directory?** A zero-file zone makes a
#    refusal say nothing about the artefact, and makes a pass say nothing at all.
#    Four instances on 2026-09-05, one of which killed a healthy run. **The rate
#    is an open question**: ~1-in-13 on real paths, 0-in-80 on a mock loop. Every
#    real run is a sample and the two numbers currently contradict each other, so
#    the counts are recorded even when nothing is wrong.
# 2. **What did each validator say?** Tallied by verdict, not by handoff state —
#    a failing sibling invalidates its passing siblings, so counting handoff
#    states overcounts refusals every time a stage has several outputs.
# 3. **Where did it stop?** A partial run is a sample, not a failure.
set -uo pipefail

RUN="${1:?usage: collect_run.sh <run dir> [launcher] [log]}"
LAUNCHER="${2:-}"
LOG="${3:-}"
[ -d "$RUN" ] || { echo "collect_run: no such run dir: $RUN"; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "# run record: $(basename "$RUN")"
echo "collected_at   $(date -u +%Y-%m-%dT%H:%M:%SZ)"

echo
echo "## where it ran"
# **Its own field, not a row buried in a sorted var list.** With one node this
# was implicit; with two it is a variable, and a run does not record it. If the
# empty-zone rate ever differs between hosts, that is the first structural clue
# anyone has about the mechanism -- and it is only visible if the node is written
# down per run.
if [ -n "$LAUNCHER" ] && [ -f "$LAUNCHER" ]; then
  echo "  node    $(grep -oE '\-\-var node=[^ \\]*' "$LAUNCHER" | head -1 | cut -d= -f2)"
  echo "  jobid   $(grep -oE '\-\-var jobid=[^ \\]*' "$LAUNCHER" | head -1 | cut -d= -f2)"
  echo "  cards   $(grep -oE '\-\-var gpu_devices=[^ \\]*' "$LAUNCHER" | head -1 | cut -d= -f2)"
  echo "  stages  $(grep -oE '\-\-var mock_stages=[^ \\]*' "$LAUNCHER" | head -1 | cut -d= -f2) mocked"
else
  echo "  NOT RECORDED"
fi

echo
echo "## launch line"
if [ -n "$LAUNCHER" ] && [ -f "$LAUNCHER" ]; then
  # The vars, one per line, in sorted order — so two runs can be diffed.
  grep -oE '\-\-var [a-z_]+=[^ \\]*' "$LAUNCHER" | sed 's/--var //' | sort
  echo "  (package)    $(grep -oE '\-\-package [^ \\]*' "$LAUNCHER" | head -1)"
  echo "  (timeout)    $(grep -oE '\-\-timeout [0-9]*' "$LAUNCHER" | head -1)"
else
  echo "  NOT RECORDED — pass the launcher path. The run itself does not keep it,"
  echo "  and that is the gap this section exists to close."
fi

echo
echo "## package provenance"
# **Prefer the LOG, which recorded it at launch, over GENERATED.txt, which
# records it now.** The package directory is regenerated in place between runs,
# so reading its stamp at collection time answers "what is there now", not "what
# did this run use" -- the same staleness trap that cost a round, one level up.
STAMP=$(grep -m1 'package built from' "${LOG:-/dev/null}" 2>/dev/null || true)
if [ -n "$STAMP" ]; then
  echo "  at launch:$STAMP"
else
  echo "  at launch: NOT RECORDED in the log (launcher had no freshness check)"
  PKG=$(grep -oE '\-\-package [^ \\]*' "${LAUNCHER:-/dev/null}" 2>/dev/null | awk '{print $2}' | head -1)
  case "$PKG" in
    /*) : ;;
    *)  REPO=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null)
        [ -n "$REPO" ] && PKG="$REPO/$PKG" ;;
  esac
  if [ -n "${PKG:-}" ] && [ -f "$PKG/GENERATED.txt" ]; then
    echo "  NOW (may differ from what the run used):"
    grep -E '^(generated_at|source_commit)' "$PKG/GENERATED.txt" | sed 's/^/    /'
  fi
fi

echo "## zones: was anything handed an empty directory?"
bash "$HERE/refusal_saw_something.sh" "$RUN" 2>&1 | grep -E '^(EMPTY|.*zone\(s\))' | sed 's/^/  /'

echo
echo "## verdicts (by verdict, NOT by handoff state)"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  grep -E '^ *verdict' "$LOG" | sed 's/^ *//' | sort | uniq -c | sort -rn | sed 's/^/  /'
else
  echo "  (no log given)"
fi

echo
echo "## where it stopped"
if [ -n "$LOG" ] && [ -f "$LOG" ]; then
  # tail -1 is not enough: the terminal line reads like an ordinary stop and the
  # cause is above it. Measured twice on 2026-09-05.
  tail -3 "$LOG" | sed 's/^/  /'
else
  echo "  (no log given)"
fi
exit 0
