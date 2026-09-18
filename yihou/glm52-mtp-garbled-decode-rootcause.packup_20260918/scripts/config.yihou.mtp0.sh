#!/usr/bin/env bash
# Purpose: R04 -- R03 with speculative decoding OFF and nothing else changed.
#   Splits "the fault is inside the MTP path" from "the fault is in plain decode".
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.
#
# DIAGNOSTIC ONLY. The user lifted the earlier "never turn MTP off" rule for
# diagnosis in this round; this is not a delivery configuration.
#
# Reading the outcome:
#   coherent text  -> the fault is inside the MTP path (draft at TP4, the
#                     verify/reject path, or the speculative-only
#                     dsa_page_table_rows repeat_interleave).
#   still garbled  -> the fault is in the plain decode path: the target model at
#                     TP4, or the handed-off KV being mis-mapped once decode
#                     starts reading it. MTP would then be exonerated entirely.
#
# What makes this worth a round: R03 showed the corruption MODE is a
# deterministic function of the serving DP rank -- 16 identical sequential
# requests alternate with period 4 (= dp_size) between a repeated-filler mode
# and a word-salad mode. Whatever is rank-indexed is the thing to find, and this
# round says whether it lives in the speculative path at all.

: "${DECODE_MTP:=0}"
: "${CONTAINER_PREFIX:=glm52-pd-yihou-mtp0}"

# Everything else identical to R03, including the nextnfix image: keeping the
# fusion fix in place makes this single-variable against R03 rather than against
# the original baseline.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config.yihou.mtpfix.sh"
