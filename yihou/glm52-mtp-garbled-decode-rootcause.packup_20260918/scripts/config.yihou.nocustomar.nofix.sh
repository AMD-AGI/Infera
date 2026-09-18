#!/usr/bin/env bash
# Purpose: R06 -- R05 with the NextN fusion fix REMOVED, to find out whether the
#   fusion fix is actually required or whether disabling custom all-reduce alone
#   is the whole cure.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# R05 established that (fusion fix + no custom all-reduce) = correct output.
# R03 established that (fusion fix alone) = still garbled. The untested corner is
# (no fusion fix + no custom all-reduce). Without it we could only say "these two
# together work", which is not good enough to tell anyone what to ship.
#
# Single variable against R05: IMAGE back to the un-patched build.
: "${IMAGE:=infera-sglang:v0519-yihou-0917}"
: "${CONTAINER_PREFIX:=glm52-pd-yihou-nofix-nocustomar}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.nocustomar.sh"
