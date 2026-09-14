#!/bin/bash
# Has this run stopped being able to make progress, and why?
#
# WHY THIS EXISTS: three runs died on 2026-09-06/07 and all three looked alive
# to a poll that reads the phase line. The cause was in store/event every time,
# written within a second of the death:
#
#   runs 7, 8   validation_failed -> root task (declares no agent, no recipient)
#               phase stays at `output_validating` FOREVER. The label is accurate
#               and means "validation finished and refused", not "validating".
#   run 10      output_absent + handling_failed: the body succeeded, the SEAL was
#               refused (attributes.seal_refused names the file and the reason),
#               and the framework's remedy - instruct the agent to continue - had
#               no loop, because the agent had already returned from mainloop.
#               Found 4h33m later. A longer stall threshold made that worse.
#   run 6       engine wedged; AIPerf's 900 s timeout bursts re-fed the stall
#               detector every cycle. NOT visible here - use the engine log's
#               "Stop profiling" line count after a stack window.
#
# It reads. It changes nothing.
#
# Usage:  bash deathwatch.sh <run dir>            one run
#         bash deathwatch.sh <run dir> --quiet    print only if something is wrong
set -uo pipefail

R="${1:-}"
QUIET="${2-}"
[ -n "$R" ] || { echo "usage: deathwatch.sh <run dir> [--quiet]" >&2; exit 2; }
[ -d "$R/store/event" ] || { echo "ABORT: no store/event under $R" >&2; exit 2; }

NOW=$(date -u +%H:%M:%SZ)

python3 - "$R" "$QUIET" "$NOW" <<'PY'
import json, pathlib, sys

run, quiet, now = pathlib.Path(sys.argv[1]), sys.argv[2] == "--quiet", sys.argv[3]

# Kinds that mean "nothing further can happen without a human". Listed rather
# than pattern-matched: a kind we have never seen should show up as unknown in
# the tail below, not be silently swallowed by a regex.
# STRONG = seen only on runs that stopped. WEAK = also seen on a run that went
# on to seal green, so a hit is a reason to look, not a verdict.
#
# The weak classification is measured, not cautious: run 4 (d9c7af) carries SEVEN
# `validation_failed` events at 16:42:36 and its `_on` sealed with four passing
# validators. The first version of this file called that "NOT going to progress"
# and would have raised a false alarm on the best run of the night. Caught by
# running it on a case whose answer was already known.
STRONG = {
    "output_absent":   "the body finished and the handoff was NOT sealed",
    "handling_failed": "the escalation had no loop to deliver to (agent already returned)",
}
WEAK = {
    "validation_failed": "a validator refused -- NOT terminal by itself; run 4 had 7 and sealed green",
    "escalated":         "a program body escalated -- may or may not have a recipient",
}
TERMINAL = {**STRONG, **WEAK}

events = []
for f in (run / "store" / "event").glob("*.json"):
    try:
        events.append(json.loads(f.read_text()))
    except Exception:
        continue
events.sort(key=lambda d: d.get("at") or "")

hits = [e for e in events if e.get("kind") in TERMINAL]
lines = []
for e in hits:
    a = e.get("attributes") or {}
    lines.append(f"  {(e.get('at') or '')[11:19]}  {e.get('kind')}")
    lines.append(f"      {TERMINAL[e['kind']]}")
    for key in ("message", "seal_refused", "detail", "exit_status"):
        if a.get(key):
            lines.append(f"      {key}: {str(a[key])[:300]}")
    if e.get("exception_message"):
        lines.append(f"      exception: {str(e['exception_message'])[:220]}")

strong = [e for e in hits if e.get("kind") in STRONG]

if not strong and quiet:
    sys.exit(0)

print(f"run:  {run}")
print(f"read: {now}   events: {len(events)}")
if hits:
    if strong:
        print(f"\n!! {len(strong)} STRONG signal(s): the body finished and nothing was sealed.")
        print("   A run in this state does not progress on its own, and the phase line")
        print("   keeps showing whatever it showed when this happened.")
    else:
        print(f"\n?  {len(hits)} weak signal(s) only -- worth reading, NOT a verdict.")
        print("   Run 4 carried seven of these and sealed green.")
    print("\n".join(lines))
    last = (hits[-1].get("at") or "")[11:19]
    print(f"\n   earliest: {(hits[0].get('at') or '')[11:19]}   latest: {last}")
else:
    print("\nno terminal event. That is NOT a claim the run is healthy --")
    print("a wedged engine produces none of these. Check the engine log's")
    print("'Stop profiling' count after any stack window.")
PY
