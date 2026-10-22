#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-13): synthetic 8-expert GLM-5.2 on ONE leg only, to see which side's
# KV capacity binds. E8_LEG=decode -> decode runs 8 routed experts (frees weight memory -> bigger decode KV pool);
# E8_LEG=prefill -> prefill does. The other leg keeps the real 256-expert checkpoint, unchanged.
# NON-FAITHFUL model on that leg (correctness already waived by SIMULATE_ACC_LEN); KV geometry is unchanged
# (attention untouched), so prefill->decode KV transfer is layout-compatible.
# Mechanism (role-only mounts, same --model-path string on both legs so agentx_env/audit/router agree):
#   - overlay config.json (n_routed_experts 8, ep_size 4, yihou_synthetic_expert8 true) mounted OVER
#     $MODEL/config.json in that leg's container only;
#   - yihou-expert8-nextn.patch (hicache-ceiling packup 2026-09-24, the 26k runs) applied to the v0.5.21 image
#     sources: models/yihou_expert8.py (new), models/deepseek_v2.py, model_loader/loader.py, layers/moe/topk.py.
#   - that leg: EP4 (2 local experts/rank, as in the 26k runs), --disable-shared-experts-fusion (patch requires it),
#     --max-total-tokens E8_MAX_TOTAL_TOKENS (default 3670016 = the 26k runs' cap; empty = no cap).
# Chain: E8_BASE_CONFIG (default config.yihou.mc4340n2.sh -> MC4340_BASE_CONFIG).
case "${E8_LEG:-}" in decode|prefill) ;; *)
    echo "config.yihou.expert8leg.sh: set E8_LEG=decode|prefill" >&2; return 2 2>/dev/null || exit 2 ;; esac
_e8_L="${E8_LEG^^}"
# Before the base: the := knobs pick these up (a YIHOU_OVERRIDES value still wins).
: "${E8_MAX_TOTAL_TOKENS=3670016}"
if [[ "$_e8_L" == DECODE ]]; then : "${DECODE_EP:=4}"; else : "${PREFILL_EP:=4}"; fi
_e8_base="${E8_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n2.sh}"
[[ -r "$_e8_base" ]] || { echo "config.yihou.expert8leg.sh: base not readable: $_e8_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_e8_base"
# (re-sourced by engine.sh/audit with the parent's exports: our own TSV is fine, anything else is a conflict)
_e8_cur="$(eval echo "\${${_e8_L}_MOUNTS_ADD:-}")"
[[ -z "$_e8_cur" || "$_e8_cur" == "$WS/harness/mounts.yihou.expert8.tsv" ]] ||
    { echo "config.yihou.expert8leg.sh: ${_e8_L}_MOUNTS_ADD already set by the base (merge the TSVs first)" >&2; return 2 2>/dev/null || exit 2; }
# export: audit_live.v3 reads these through `set -a; source; env`.
export "${_e8_L}_MOUNTS_ADD=$WS/harness/mounts.yihou.expert8.tsv"
export "${_e8_L}_EXPERT8=1"
export "${_e8_L}_MAX_TOTAL_TOKENS=$E8_MAX_TOTAL_TOKENS"
_e8_args="--disable-shared-experts-fusion${E8_MAX_TOTAL_TOKENS:+ --max-total-tokens $E8_MAX_TOTAL_TOKENS}"
if [[ "$_e8_L" == DECODE ]]; then DECODE_EXTRA_ARGS+=" $_e8_args"; else PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }$_e8_args"; fi
YIHOU_EXP="${YIHOU_EXP:-E}+expert8-${E8_LEG}"
unset _e8_L _e8_base _e8_args _e8_cur
