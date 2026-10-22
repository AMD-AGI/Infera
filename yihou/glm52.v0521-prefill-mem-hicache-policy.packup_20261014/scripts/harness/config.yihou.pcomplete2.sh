#!/usr/bin/env bash
# Overlay (task v0521cap2, 2026-10-14): config.yihou.pcomplete.sh + the YIHOU-PCOMPLETE-ATTNDP hook fix mounted on BOTH legs
# through YIHOU_MOUNTS (regbase rows + hook row) instead of {PREFILL,DECODE}_MOUNTS_ADD, so audit v3's role-only rule
# no longer flags it (pc48b's audit STOP was that false negative). Chain: PCOMPLETE_BASE_CONFIG via config.yihou.pcomplete.sh.
source "$(dirname "${BASH_SOURCE[0]}")/config.yihou.pcomplete.sh" || { return 2 2>/dev/null || exit 2; }
YIHOU_MOUNTS="$WS/harness/mounts.yihou.v2-regbase-attndp.tsv"
YIHOU_EXP="${YIHOU_EXP:-E}+attndp"
