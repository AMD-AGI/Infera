#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-13 ~08:00): decode allocates KV only after prefill has finished computing
# the request: --disaggregation-decode-allocation-policy prefill_complete on BOTH legs (upstream PR #40703, in the
# image; a leg mismatch fails every request with "policy mismatch"). The policy REQUIRES prefill HiCache write_back
# (pd_disaggregation_hook.py:231-247 rejects write_through) -> PREFILL_HICACHE_WRITE_POLICY=write_back is forced here.
# prefill optimistic_prefill_attempts is set to 1 by sglang itself. Research: v0521cap2 workflow wf_837ba919-06f.
# Chain: PCOMPLETE_BASE_CONFIG (default config.yihou.mc4340n2.sh -> MC4340_BASE_CONFIG).
: "${PREFILL_HICACHE_WRITE_POLICY:=write_back}"   # before the base: its := keeps this
_pc_base="${PCOMPLETE_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n2.sh}"
[[ -r "$_pc_base" ]] || { echo "config.yihou.pcomplete.sh: base not readable: $_pc_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_pc_base"
[[ "$PREFILL_HICACHE_WRITE_POLICY" == write_back ]] ||
    { echo "config.yihou.pcomplete.sh: prefill_complete needs PREFILL_HICACHE_WRITE_POLICY=write_back" >&2; return 2 2>/dev/null || exit 2; }
# After the base built *_EXTRA_ARGS (re-sourced by engine.sh: append only once).
_pc_flag="--disaggregation-decode-allocation-policy prefill_complete"
[[ " $PREFILL_EXTRA_ARGS " == *" $_pc_flag "* ]] || PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }$_pc_flag"
[[ " $DECODE_EXTRA_ARGS " == *" $_pc_flag "* ]] || DECODE_EXTRA_ARGS+="${DECODE_EXTRA_ARGS:+ }$_pc_flag"
YIHOU_EXP="${YIHOU_EXP:-E}+pcomplete"
unset _pc_base _pc_flag
