#!/usr/bin/env bash
# Run a command on a cluster node through the jump host, retrying the SSH.
# The jump host is loaded (load 44, 22k zombies) and resets connections
# intermittently; without the retry a transient reset reads as a failed step on
# the node, which it is not. -T suppresses the PTY and its login banner.
set -u
NODE="${NODE:?}"
CMD="$*"
for a in 1 2 3 4 5; do
  out=$(timeout 90 ssh -T -o ConnectTimeout=15 -o StrictHostKeyChecking=no -o LogLevel=ERROR \
        root@149.28.124.225 "ssh -T -o StrictHostKeyChecking=no -o LogLevel=ERROR $NODE '$CMD'" 2>&1)
  rc=$?
  case "$out" in
    *"Connection reset"*|*"Connection closed"*|*"kex_exchange"*) sleep $((a*4)); continue;;
  esac
  echo "$out"; exit $rc
done
echo "SSH FAILED after 5 tries" >&2; exit 1
