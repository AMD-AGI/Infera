#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-13 ~16:00 "prefill侧expert砍半"): truncated-expert GLM-5.2 on ONE leg (EN_LEG=prefill,
# no MTP support): the leg loads only routed experts 0..127 (yihou_expertn.py + model_loader/loader.py hook) and an overlay
# config.json (n_routed_experts 128, yihou_synthetic_keep_experts 128) mounted over $MODEL/config.json. Routing is the model's
# own (sigmoid top-8 over 128). EP unchanged (1). Shared-expert fusion: sglang only allows it for 256/384 routed experts, so
# it is disabled on that leg; passed explicitly (--disable-shared-experts-fusion) so audit v3 can check it ({LEG}_EXPERT8=1
# is audit v3's switch for "fusion off on this leg"). Chain: EN_BASE_CONFIG (default config.yihou.mc4340n2.sh).
case "${EN_LEG:-}" in prefill) ;; *)
    echo "config.yihou.expertnleg.sh: EN_LEG=prefill only (the draft/MTP path is not supported)" >&2; return 2 2>/dev/null || exit 2 ;; esac
_en_base="${EN_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n2.sh}"
[[ -r "$_en_base" ]] || { echo "config.yihou.expertnleg.sh: base not readable: $_en_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_en_base"
[[ -z "${PREFILL_MOUNTS_ADD:-}" || "$PREFILL_MOUNTS_ADD" == "$WS/harness/mounts.yihou.expert128.tsv" ]] ||
    { echo "config.yihou.expertnleg.sh: PREFILL_MOUNTS_ADD already set by the base" >&2; return 2 2>/dev/null || exit 2; }
export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.expert128.tsv"
export PREFILL_EXPERT8=1 PREFILL_MAX_TOTAL_TOKENS=
[[ " $PREFILL_EXTRA_ARGS " == *" --disable-shared-experts-fusion "* ]] ||
    PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }--disable-shared-experts-fusion"
YIHOU_EXP="${YIHOU_EXP:-E}+expert128-prefill"
unset _en_base
