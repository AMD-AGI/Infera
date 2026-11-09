#!/usr/bin/env bash
# KERNEL-PLAN-20261109 phase 0: kernel micro-benchmarks hold this node. The GPUs are used by llying-kopt-* containers
# started by hand over ssh. Ends when /apps/liyingli/holder1/kopt.lease is removed or after LEASE_MAX seconds, and
# removes the llying-kopt-* containers on exit.
L=/apps/liyingli/holder1/kopt.lease
end=$(( $(date +%s) + ${LEASE_MAX:-9300} ))
cleanup() { docker ps -aq --filter name=llying-kopt | xargs -r docker rm -f; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) lease start on $(hostname)"
while [ -e "$L" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) lease end"
