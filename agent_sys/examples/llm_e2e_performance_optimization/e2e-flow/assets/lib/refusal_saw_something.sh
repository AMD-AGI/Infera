#!/bin/bash
# Did the validators in this run actually get shown anything?
#
#     bash assets/lib/refusal_saw_something.sh <run dir>
#
# **`bash`, not `sh`.** Both the shebang and this line say bash because the
# script uses `set -o pipefail` (and process substitution), which dash
# rejects at RUNTIME with `Illegal option -o pipefail`. `dash -n` accepts it
# — a syntax check cannot fail on an invalid option — so this was verified by
# running it, not by parsing it. Do not 'simplify' it back to `sh`.
#
# **Why.** 2026-09-05: `check_profiling_evidence` refused a handoff with eight
# `PROBLEM: items/... is missing` lines while every one of those files was
# present in the same run. The validator had been staged a version directory
# containing **zero files**; the content was in the sibling version, 47 files.
#
# The refusal is **accurate about what it was pointed at and false about the
# artefact**, and there is no way to tell from the report: it is well formed, it
# names eight specific paths, and every statement in it is true of the empty
# directory. **Nothing in the verdict, the report or the escalation mentions
# which version was staged.**
#
# ## What this checks, and what it does NOT
#
# **The file count under a validation zone's `materials/`.** That is the whole
# check. It is not the version number:
#
#     w17m1e  check_acceptance          materials/*/v0   97 files   fine
#     w17m1e  check_profiling_evidence  materials/*/v1   45 files   fine
#     w17m1g  check_profiling_evidence  materials/*/v0    0 files   THE FAULT
#
# **`v0` is a normal staged version and is often fully populated.** An earlier
# version of the bug record called this "the empty v0" and that framing sends a
# reader looking for `v0`, which they will find everywhere. **Count files.**
#
# ## What a zero does and does not prove
#
# A zero means **this refusal says nothing about the artefact** — re-run before
# attributing anything to a producer. It does **not** prove the artefact is
# good, and a non-zero count does **not** prove the refusal is right; it only
# means the validator was shown something. The stronger signal is the leader's
# criterion, which this cannot compute: **a refusal quoting a number from inside
# a file proves it read the file; a refusal that only says "X is missing" does
# not.**
set -uo pipefail

RUN="${1:?usage: refusal_saw_something.sh <run dir>}"
[ -d "$RUN" ] || { echo "refusal_saw_something: no such run dir: $RUN"; exit 2; }

bad=0
seen=0
while IFS= read -r z; do
  [ -d "$z/materials" ] || continue
  seen=$((seen + 1))
  n=$(find "$z/materials" -type f 2>/dev/null | wc -l | tr -d ' ')
  vers=$(find "$z/materials" -maxdepth 2 -mindepth 2 -type d 2>/dev/null \
         | sed 's|.*/||' | sort -u | tr '\n' ' ')
  # Name the validators in the zone, so a hit is actionable rather than a path.
  who=$(grep -ho '^# check_[a-z_]*' "$z"/validation-*/validator_report.txt 2>/dev/null \
        | sed 's/^# //' | sort -u | tr '\n' ' ')
  [ -n "$who" ] || who='(no report written — check_environment writes none)'
  if [ "$n" = 0 ]; then
    bad=$((bad + 1))
    echo "EMPTY  $(basename "$z")"
    echo "         versions: ${vers:-none}   files: 0   validators: $who"
  else
    echo "ok     $(basename "$z" | cut -c1-56)  files: $n  versions: ${vers:-none}"
  fi
done < <(find "$RUN" -type d -name 'validation.*' 2>/dev/null | sort)

echo
if [ "$seen" = 0 ]; then
  echo "no validation zones found — the run may not have reached validation"
  exit 0
fi
if [ "$bad" = 0 ]; then
  echo "$seen zone(s), none empty: every validator was shown something."
  echo "That is NOT a claim that any verdict is right — only that none of them"
  echo "judged an empty directory."
  exit 0
fi
echo "$bad of $seen zone(s) were handed ZERO files."
echo "Any refusal from those says nothing about the artefact. Re-run before"
echo "attributing it to a producer."
exit 1
