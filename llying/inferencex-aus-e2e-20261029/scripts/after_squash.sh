#!/usr/bin/env bash
# Wait (detached) until the pinned SGLang squash is staged and valid, then launch one point.
#   after_squash.sh <label> <exp-name> [manual_launch options]
sq=/apps/liyingli/squash/lmsysorg_sglang-rocm_v0.5.22-rocm720-mi35x-20261018_sha256_8f7d8034546b17cc4ae57c63f3f65c584eb73e62378243d1416b8385aa7be286.sqsh
setsid nohup bash -c '
    until [ -r "$0" ] && unsquashfs -l "$0" >/dev/null 2>&1; do sleep 30; done
    exec bash /apps/liyingli/jobs/launch_point.sh "$@"
' "$sq" "$@" > "/apps/liyingli/runs/$1.wait.log" 2>&1 < /dev/null &
echo "waiter pid $!"
