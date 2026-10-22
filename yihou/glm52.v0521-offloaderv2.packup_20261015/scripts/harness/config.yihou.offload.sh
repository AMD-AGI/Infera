#!/usr/bin/env bash
# Overlay (task v0521cap2, user 2026-10-14 "开始"): sglang OffloaderV2 on the PREFILL leg only: routed-expert weights (w13/w2) of
# every OFF_G-th MoE layer group's last OFF_K layers live in pinned host memory and are streamed H2D per forward
# (--offload-group-size OFF_G --offload-num-in-group OFF_K --offload-prefetch-step OFF_STEP --offload-mode cpu).
# Patch YIHOU-OFFLOAD-ATTRS (offloader.py create_device_tensor carries is_shuffled) mounted on the prefill leg.
# Decode unchanged. Chain: OFF_BASE_CONFIG (default config.yihou.mc4340n.sh).
_of_base="${OFF_BASE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/config.yihou.mc4340n.sh}"
[[ -r "$_of_base" ]] || { echo "config.yihou.offload.sh: base not readable: $_of_base" >&2; return 2 2>/dev/null || exit 2; }
source "$_of_base"
export PREFILL_MOUNTS_ADD="$WS/harness/mounts.yihou.offload-attrs.tsv"
_of_args="--offload-group-size ${OFF_G:-8} --offload-num-in-group ${OFF_K:-1} --offload-prefetch-step ${OFF_STEP:-1} --offload-mode cpu"
[[ " $PREFILL_EXTRA_ARGS " == *" --offload-group-size "* ]] || PREFILL_EXTRA_ARGS+="${PREFILL_EXTRA_ARGS:+ }$_of_args"
YIHOU_EXP="${YIHOU_EXP:-E}+offload-g${OFF_G:-8}k${OFF_K:-1}s${OFF_STEP:-1}"
unset _of_base _of_args
