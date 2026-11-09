#!/usr/bin/env bash
# KERNEL-PROGRESS-20261109 section 12: kernel development (after the acceptance attribution) holds the holder1 node (job 35370). The GPUs are
# used by llying-kopt-* containers started by hand over ssh. Ends when /apps/liyingli/holder1/kopt.lease5 is removed or after LEASE_MAX seconds, and removes the
# llying-kopt-* containers on exit.
H=/apps/liyingli/holder1
L=$H/kopt.lease5
end=$(( $(date +%s) + ${LEASE_MAX:-17000} ))
cleanup() { docker ps -aq --filter name=llying-kopt | xargs -r docker rm -f; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) lease start on $(hostname)"
while [ -e "$L" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) lease end"
