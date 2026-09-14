#!/usr/bin/env bash
# Negative and positive controls for the reset_gpus.sh ownership guard.
#
# **What these controls DO test:** the discriminator — given a pid, does the
# guard say ours / foreign / unknown, and does every error path land on spare.
#
# **What they do NOT test, stated rather than papered over:** a real KFD-holding
# process. I cannot manufacture a foreign GPU workload — the cards belong to a
# live chain and this host runs one GPU thing at a time. So the containers below
# hold no GPU; they exercise the ownership question, which is the only part the
# guard changes. The kill itself is `reset_gpus.sh`'s existing code, unmodified.
#
# The two REAL foreign containers on this host are used read-only as the
# genuine foreign case, which is better than any container I could fabricate.
set -uo pipefail
cd "$(dirname "$0")"
. ./reset_gpus.guard.sh

FAILED=0
case_() { # name expected_prefix actual
  if [[ "$3" == "$2"* ]]; then printf 'PASS  %-46s %s\n' "$1" "$3"
  else printf 'FAIL  %-46s got %s, wanted %s*\n' "$1" "$3" "$2"; FAILED=$((FAILED+1)); fi
}

MINE=yihou_m35_guardtest_ours_$$
UNLABELLED=yihou_m35_guardtest_unlabelled_$$
cleanup() {
  # Only containers this script created. Names are fixed strings, not variables
  # feeding a wildcard.
  docker rm -f "$MINE" >/dev/null 2>&1
  docker rm -f "$UNLABELLED" >/dev/null 2>&1
}
trap cleanup EXIT

echo "=== positive control: one of OURS must be killable ==="
docker run -d --name "$MINE" --label infera_e2e_run=yihou_m35_guardtest \
  ubuntu:24.04 sleep 120 >/dev/null || { echo "could not start $MINE"; exit 1; }
p=$(docker inspect -f '{{.State.Pid}}' "$MINE")
case_ "C1 our labelled container -> ours" "ours:" "$(owner_of "$p")"

echo
echo "=== negative control: an UNLABELLED container of ours must be spared ==="
echo "    (this is the shape mix_up.sh's own comment records as the 2026-09-04 hazard:"
echo "     an unlabelled, generically-named GPU container is indistinguishable from a stranger's)"
docker run -d --name "$UNLABELLED" ubuntu:24.04 sleep 120 >/dev/null || exit 1
p=$(docker inspect -f '{{.State.Pid}}' "$UNLABELLED")
case_ "C2 unlabelled container -> foreign (spared)" "foreign:" "$(owner_of "$p")"

echo
echo "=== the REAL foreign tenants, read-only, no interference ==="
for c in rc_26_7_902 xiaoming-dev; do
  p=$(docker inspect -f '{{.State.Pid}}' "$c" 2>/dev/null)
  if [ -n "${p:-}" ] && [ "$p" != "0" ]; then
    case_ "C3 $c -> foreign (spared)" "foreign:" "$(owner_of "$p")"
  else
    printf 'SKIP  %-46s not running\n' "C3 $c"
  fi
done

echo
echo "=== every failure path must land on 'unknown', i.e. spare ==="
case_ "C4 pid that does not exist"       "unknown:" "$(owner_of 999999)"
case_ "C5 pid 1 (host init, no docker cgroup)" "unknown:" "$(owner_of 1)"
case_ "C6 empty pid argument"            "unknown:" "$(owner_of '')"

echo
echo "=== docker unavailable must ALSO spare, not kill ==="
p=$(docker inspect -f '{{.State.Pid}}' "$MINE")
# **Absolute `/bin/bash`, and PATH broken INSIDE.** `PATH=/nonexistent bash -c`
# makes the SHELL unable to find bash, so nothing runs and the empty output is
# the absence of a test rather than its result. That is what the first version
# did, and a `2>/dev/null` hid the `command not found` — this file's own
# section 5a, in this file. No stderr redirect here, deliberately.
out=$(/bin/bash -c "PATH=/nonexistent; . ./reset_gpus.guard.sh; owner_of $p")
case_ "C7 docker not on PATH -> unknown (spared)" "unknown:" "$out"

echo
if [ "$FAILED" -ne 0 ]; then echo "$FAILED control(s) FAILED"; exit 1; fi
echo "all controls pass — the guard discriminates, and every error path spares"
