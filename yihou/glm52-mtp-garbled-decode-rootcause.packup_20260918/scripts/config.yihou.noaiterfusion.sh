#!/usr/bin/env bash
# Purpose: R07 -- the missing cell of the 2x2. Custom all-reduce back ON (the
#   default), AITER all-reduce fusion OFF. MTP on, real acceptance.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# THE QUESTION. R03 (custom AR on, aiter on) is garbled; R05 (custom AR off,
# aiter on) is correct. aiter fusion was held constant across that A/B, so custom
# all-reduce is the established discriminator -- but that does not say whether it
# misbehaves on its own or only in combination with the aiter fusion path. This
# round fills the cell that decides it.
#
#   correct  -> the fault needs BOTH; custom all-reduce alone is innocent, and
#               dropping --enable-aiter-allreduce-fusion is a narrower fix that
#               keeps custom all-reduce's optimisation.
#   garbled  -> custom all-reduce misbehaves regardless of the aiter path, and
#               --disable-custom-all-reduce stays the right fix.
#
# Single variable against R03: AITER_ALLREDUCE_FUSION only. Same nextnfix image,
# same everything else, and DECODE_EXTRA_ARGS left empty so custom all-reduce is
# enabled exactly as it was in R03.
: "${AITER_ALLREDUCE_FUSION:=0}"
: "${CONTAINER_PREFIX:=glm52-pd-yihou-noaiterfusion}"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.mtpfix.sh"
