#!/usr/bin/env bash
# Purpose: T2f -- the SAME full-mode sweep as config.yihou.full.sh, but on the
#   ONE cell verified coherent on TP4: index_share_for_mtp_iteration=false AND
#   --disable-custom-all-reduce. That is T1's decode cell.
#
# WHY THIS EXISTS. config.yihou.full.sh dropped --disable-custom-all-reduce (its
# line 52 sets DECODE_EXTRA_ARGS to empty, before sourcing base) to align byte
# for byte with C40. Combined with index_share=false + fused top-k ON, that is
# the KNOWN-BAD garbled cell (original root-cause kit rounds R03/R07): the t2e
# stack produced `1!...`-prefixed degenerate text for its entire 4.6 h life,
# including both "completed" points. Simulated acceptance (3.61) hid it because
# the acceptance gauge is forced, not measured.
#
# THE FIX. Set DECODE_EXTRA_ARGS BEFORE sourcing config.yihou.full.sh. Its
# line 52 is `: "${DECODE_EXTRA_ARGS=}"` (assign-if-unset), so a value set here
# survives; base.sh line 96 likewise keeps it. JSON_MODEL_OVERRIDE_ARGS is set
# at the END of config.yihou.full.sh as a plain assignment, so
# index_share=false is preserved unconditionally.
#
# Everything else -- full mode (HiCache on, max-running 128, 3600 s / warmup 10),
# simulated acceptance 3.61, base image, tilelang/sgl-kernel backends -- is
# inherited unchanged from config.yihou.full.sh. The ONLY delta vs t2e is
# --disable-custom-all-reduce back on the decode leg.
#
# Usage: CONFIG=<this file> TOPOLOGY=<workspace>/topology.yihou.tsv
# Artifacts: none; this file only defines shell variables.

_D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Restore the coherence fix. Plain assignment: this is not negotiable for this
# cell, so it overrides any command-line DECODE_EXTRA_ARGS as well.
DECODE_EXTRA_ARGS='--disable-custom-all-reduce'

source "$_D/config.yihou.full.sh"
