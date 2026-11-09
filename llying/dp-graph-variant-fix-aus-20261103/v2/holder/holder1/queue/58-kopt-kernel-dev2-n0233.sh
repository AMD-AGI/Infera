#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 12: kernel development (after the kopt-tp4 gate and trace) holds n02-33. The GPUs are
# used by llying-kopt-* containers started by hand over ssh. Only on n02-33: picked up anywhere else, it goes back to
# the queue. Ends when /apps/liyingli/holder1/kopt.lease2 is removed or after LEASE_MAX seconds, and removes the
# llying-kopt-* containers on exit.
H=/apps/liyingli/holder1
if [[ "$(hostname -s)" != smci355-ccs-aus-n02-33 ]]; then
    echo "$(date -u +%FT%TZ) on $(hostname -s), not n02-33: back to the queue"; cp "$0" "$H/queue/"; sleep 90; exit 0
fi
L=$H/kopt.lease2
end=$(( $(date +%s) + ${LEASE_MAX:-17000} ))
cleanup() { docker ps -aq --filter name=llying-kopt | xargs -r docker rm -f; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) lease start on $(hostname)"
while [ -e "$L" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) lease end"
