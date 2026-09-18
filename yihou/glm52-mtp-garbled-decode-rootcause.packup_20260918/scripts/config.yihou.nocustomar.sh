#!/usr/bin/env bash
# Purpose: R05 -- R03 with custom all-reduce disabled on the DECODE leg only.
#   MTP is back ON; this is the first genuine behavioural delta against the
#   peer's known-good stack.
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# WHY THIS IS THE ONLY SURVIVING CANDIDATE OF THE FOUR.
# The peer's kit lists four "numerics / kernel selection" deltas. Three of them
# are NO-OPS on our image -- verified by reading our own image, not by trusting
# the list:
#
#   SGLANG_ROCM_FUSED_DECODE_MLA=0   environ.py:899 declares EnvBool(False), and
#                                    forward_mla_fused_rope_rocm.py:35 reads it
#                                    with default "false". Unset == 0.
#   SGLANG_OPT_USE_TILELANG_INDEXER=1
#                                    environ.py:1492 defaults False, but
#                                    arg_groups/model_hook.py:441-442 does
#                                    `if not ...is_set(): ...set(True)`.
#                                    Unset == 1.
#   SGLANG_OPT_USE_JIT_NORM=0        the string does not exist anywhere in our
#                                    image; setting it is inert.
#
# So only --disable-custom-all-reduce actually changes what our engine does.
#
# WHY IT CAN BE MTP-SPECIFIC EVEN THOUGH R04 SHOWED PLAIN DECODE IS FINE.
# custom_all_reduce.py gates on tensor size (`should_custom_ar`, `_MAX_CAR_SIZE`)
# and on world size, so different message sizes take different paths. The draft
# is a single-layer model and the verify step runs with num_draft_tokens=6, so
# the speculative path drives all-reduce at shapes the plain decode path never
# produces. A bug confined to one of those size branches would show up only with
# MTP on -- which is exactly the split we measured.
#
# DECODE LEG ONLY, deliberately: R04 established that prefill's output is
# correct, so restricting the change keeps the variable as narrow as possible.

: "${DECODE_EXTRA_ARGS:=--disable-custom-all-reduce}"
: "${CONTAINER_PREFIX:=glm52-pd-yihou-nocustomar}"

# Everything else identical to R03: nextnfix image, MTP on, real acceptance,
# same-rail on both halves, P4D4 on devices 2,3,4,5.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.mtpfix.sh"
