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
  # **Name the CLOSURE, not just the path.** The zone directory is
  # `validation.<TASK-ID>.<phase>.<hash>`, and the store maps task id -> closure,
  # so the owning stage is one lookup away and always was. Six separate reports
  # of this fault gave a COUNT ("1 of 13") and stopped; a count invites a rate,
  # and the rate we all wrote down -- "non-deterministic, roughly 1 in 13" -- was
  # wrong. It is `m2_profiling`'s zone every time. Reporting the identity instead
  # of the number is the difference between a hunt and a targeted read.
  tid=$(basename "$z" | sed 's/^validation\.//; s/\..*//')
  closure=$(python3 - "$RUN" "$tid" <<'PYEOF' 2>/dev/null || true
import json,glob,os,sys
run,tid=sys.argv[1],sys.argv[2]
for f in glob.glob(run+"/store/task/**/*", recursive=True):
    if not os.path.isfile(f): continue
    try: j=json.load(open(f))
    except Exception: continue
    for e in (j if isinstance(j,list) else [j]):
        if isinstance(e,dict) and e.get("id")==tid and e.get("closure"):
            print(e["closure"]); sys.exit(0)
PYEOF
)
  # Name the validators in the zone, so a hit is actionable rather than a path.
  who=$(grep -ho '^# check_[a-z_]*' "$z"/validation-*/validator_report.txt 2>/dev/null \
        | sed 's/^# //' | sort -u | tr '\n' ' ')
  [ -n "$who" ] || who='(no report written — check_environment writes none)'
  if [ "$n" = 0 ]; then
    bad=$((bad + 1))
    # **Report the KIND being validated, not just the closure.** The closure came
    # from mapping the zone name's task id through the store, which is an
    # inference; the kind comes from the handoff the zone actually lists in
    # inputs.json, which is what was being judged. On 2026-09-06 the two were
    # read differently by two people on the same zone and produced opposite
    # conclusions -- one "e2e_packup", one "m2_profiling" -- while the kind was
    # unambiguous and the same in all seven runs: profiling_evidence.
    kinds=$(python3 - "$RUN" "$z" <<'PYEOF' 2>/dev/null || true
import json,glob,os,sys
run,z=sys.argv[1],sys.argv[2]
inp=glob.glob(z+"/validation-*/inputs.json")
out=[]
if inp:
    for i in json.load(open(inp[0])):
        rd=sorted(glob.glob(run+f"/handoffs/{i}/v*/content/README.md"))
        out.append(open(rd[-1]).readline().strip().lstrip("# ") if rd else i[:8])
print(",".join(out))
PYEOF
)
    echo "EMPTY  kind=${kinds:-<unknown>}  closure=${closure:-<unresolved>}"
    echo "         zone: $(basename "$z" | cut -c1-52)"
    echo "         versions staged: ${vers:-none}   files: 0   validators: $who"
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
