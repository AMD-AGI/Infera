#!/usr/bin/env bash
# Run a snapshot of scripts/SCRIPT, so that editing the script later does not change a queued or running item.
#   snap.sh SCRIPT [ARGS...]
R=${R:-/home/liyingli/bench_agentx/vllm-pd-20261117}
s=$(mktemp "/tmp/snap-${1%.sh}.XXXXXX.sh")
cp "$R/scripts/$1" "$s" || exit 1
shift
bash "$s" "$@"
rc=$?
rm -f "$s"
exit $rc
