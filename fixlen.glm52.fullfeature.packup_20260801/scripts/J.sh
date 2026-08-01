#!/usr/bin/env bash
# Two-hop SSH to a cluster node, with a retry.
#
# The jump host is loaded and resets connections intermittently; without the
# retry a transient reset reads as a failed step on the node. Staging a script
# FILE and running it is the rule here -- nested `bash -c '...'` quoting through
# two SSH hops silently no-ops (feedback_nested_ssh_quoting).
#
#   J.sh <node> '<command>'
set -u
J=root@149.28.124.225
NODE="${1:?node}"; shift
CMD="$*"
for i in 1 2 3; do
  out=$(ssh -o ConnectTimeout=15 -o BatchMode=yes "$J" \
        "ssh -o ConnectTimeout=10 -o BatchMode=yes $NODE \"\$(cat <<'__EOF__'
$CMD
__EOF__
)\"" 2>&1 | grep -v '^Warning: Permanently added')
  rc=$?
  [ $rc -eq 0 ] && { printf '%s\n' "$out"; exit 0; }
  sleep $((i * 3))
done
printf '%s\n' "$out"
exit $rc
