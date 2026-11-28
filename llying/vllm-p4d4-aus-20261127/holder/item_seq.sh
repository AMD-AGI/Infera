#!/usr/bin/env bash
# Holder item: several kit points one after the other on this node (pairs that must share a node).
#   item_seq.sh "VARIANT|TAG|K=V K=V" ...     each point: STEPS=score unless given, results/<TAG>-<time>
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
rc=0
for spec in "$@"; do
    IFS='|' read -r v tag envs <<< "$spec"
    echo "$(date -u +%FT%TZ) $NODE: $v as $tag ($envs)"
    env STEPS=score $envs bash "$here/item_kit.sh" "$v" "$tag-$(date -u +%m%d%H%M)" || rc=1
done
exit $rc
