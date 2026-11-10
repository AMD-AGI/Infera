#!/usr/bin/env bash
# REPORT-20261110 section 6: root cause of the start-up segfault on amdgpu 6.19.14 firmware-pilot nodes. Holds the node
# (pinned to a 6.19 node) for interactive debugging over ssh: Mooncake GPU-memory registration repro (v2-aus/mc_intranode.sh),
# HIP / HSA / Mooncake knobs, host vs image ROCm userspace. Ends when /apps/liyingli/holder1/driver-debug.lease is removed
# or after LEASE_MAX seconds; removes the llying-* containers on exit.
H=/apps/liyingli/holder1
L=$H/driver-debug.lease
touch "$L"
end=$(( $(date +%s) + ${LEASE_MAX:-12600} ))
cleanup() { docker ps -aq --filter name=llying- | xargs -r docker rm -f; }
trap cleanup EXIT
trap 'cleanup; exit 143' TERM
echo "$(date -u +%FT%TZ) lease start on $(hostname), amdgpu $(cat /sys/module/amdgpu/version 2>/dev/null)"
while [ -e "$L" ] && [ "$(date +%s)" -lt "$end" ]; do sleep 20; done
echo "$(date -u +%FT%TZ) lease end"
