# SNAPSHOT of watch_for_verdict.sh taken 2026-09-06T07:28:24Z, sha256[0:16]=c1aa4ad85de43d23
#!/usr/bin/env bash
# Block until m1's run writes its first verdict, then exit. Read-only.
#
# **No `2>/dev/null` anywhere in here, deliberately.** The predicate that failed
# this morning failed LOUDLY (bfs errors on a relative -newermt) and a stderr
# redirect turned that into a silent zero. An error from this loop must be
# visible, because the whole point is that its zero is trusted.
set -uo pipefail
R="${1:?usage: watch_for_verdict.sh <run dir>}"
[ -d "$R" ] || { echo "watch: no such run dir: $R" >&2; exit 2; }
DEADLINE=$(( $(date +%s) + 3600 ))
while :; do
  # **find's STATUS, not just its output.** `find ... | wc -l` is 0 both when
  # there are no verdicts and when find itself failed — and this loop's zero is
  # the thing being trusted. bfs errors on a bad predicate; that must abort the
  # watch rather than be counted as "still waiting".
  out=$(find "$R" -name 'verdict.json' -type f); rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "watch: find failed (rc=$rc) — this zero is NOT 'no verdicts yet'" >&2
    exit 3
  fi
  n=$(printf '%s' "$out" | grep -c . )
  if [ "${n:-0}" -gt 0 ]; then
    echo "VERDICTS APPEARED: $n at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "watch: one hour elapsed, still zero verdicts at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 1
  fi
  sleep 20
done
