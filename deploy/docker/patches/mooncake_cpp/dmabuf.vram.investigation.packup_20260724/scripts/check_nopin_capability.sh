#!/bin/bash
# ============================================================================
# check_nopin_capability.sh — one-click: can this system do NO-PIN (dynamic
# attach) dma-buf RDMA, or is ibv_reg_dmabuf_mr forced to PIN (→ KFD double-
# counts the pinned GPU mem → reported available GPU memory shrinks)?
#
# Self-contained. Run on the HOST (best coverage: needs /boot, dmesg, lsmod).
# Also runnable in a container that has ibv_devinfo + libibverbs; host-only
# checks will degrade gracefully to kallsyms/other sources.
#
#   bash check_nopin_capability.sh [ionic_device]     # default: ionic_0
#
# Exit code 0 = NO-PIN possible (all 4 layers green). Non-zero = pin forced;
# the failing layer(s) are printed. A RED on Layer 1 (NIC ODP) is the usual
# and unfixable-by-config cause on ionic (AMD Pensando RoCE).
# ============================================================================
set -u
DEV="${1:-ionic_0}"
RED=0; L1=RED; L2=RED; L3="?"; L4=RED; PEER="RED"

hr(){ echo "------------------------------------------------------------"; }
echo "############################################################"
echo "# NO-PIN dma-buf capability — one-click check"
echo "# host=$(hostname)  dev=$DEV  date=$(date +%F_%T)"
echo "############################################################"

# ---------------------------------------------------------------------------
hr; echo "LAYER 1 — NIC ODP (On-Demand Paging)   [decisive]"
# no-pin needs ODP: without it the provider is forced to ib_umem_dmabuf_get_pinned
ODP=$(ibv_devinfo -d "$DEV" -v 2>/dev/null | grep -iA6 'odp_caps')
echo "$ODP" | sed 's/^/    /'
if echo "$ODP" | grep -qiE 'NO SUPPORT'; then
  L1=RED; echo "  => Layer1 RED: NIC declares NO ODP support -> pin FORCED."
elif echo "$ODP" | grep -qiE 'SEND|RECV|WRITE|READ|ATOMIC|SRQ'; then
  L1=GREEN; echo "  => Layer1 GREEN: ODP caps present."
else
  L1=RED; echo "  => Layer1 RED: ODP caps empty/absent -> pin FORCED."
fi

# ---------------------------------------------------------------------------
hr; echo "LAYER 1b — peermem (whether plain ibv_reg_mr GPU-direct is even an option)"
if lsmod 2>/dev/null | grep -qiE 'peermem|ib_peer'; then
  PEER=GREEN; echo "  peermem module LOADED -> ibv_reg_mr GPU-direct available (still pins via peermem)."
else
  PEER=RED; echo "  no peermem module loaded -> ibv_reg_mr GPU-direct NOT available"
  echo "                                 -> ibv_reg_dmabuf_mr is the ONLY path."
fi
dmesg 2>/dev/null | grep -iE 'peerdirect|peer_mem' | tail -1 | sed 's/^/    dmesg: /'

# ---------------------------------------------------------------------------
hr; echo "LAYER 2 — kernel CONFIG_DMABUF_MOVE_NOTIFY"
CFG=""
if [ -r /proc/config.gz ]; then CFG=$(zcat /proc/config.gz 2>/dev/null | grep -E 'DMABUF_MOVE_NOTIFY'); fi
[ -z "$CFG" ] && CFG=$(grep -E 'DMABUF_MOVE_NOTIFY' /boot/config-$(uname -r) 2>/dev/null)
SYM=$(grep -cE ' dma_buf_move_notify$' /proc/kallsyms 2>/dev/null)
echo "    config: ${CFG:-<not found via /proc or /boot>}"
echo "    kallsyms dma_buf_move_notify present: $([ "${SYM:-0}" -ge 1 ] && echo yes || echo no)"
if echo "$CFG" | grep -q '=y' || [ "${SYM:-0}" -ge 1 ]; then L2=GREEN; else L2=RED; fi
echo "  => Layer2 $L2"

# ---------------------------------------------------------------------------
hr; echo "LAYER 3 — amdgpu driver freshness (pin/notify support)"
VER=$(modinfo amdgpu 2>/dev/null | awk '/^version:/{print $2}')
FN=$(modinfo amdgpu 2>/dev/null | awk '/^filename:/{print $2}')
echo "    uname -r = $(uname -r)"
echo "    amdgpu version = ${VER:-<builtin/unknown>}   file=${FN:-n/a}"
dmesg 2>/dev/null | grep -i amdgpu | grep -iE 'version|peerdirect' | tail -2 | sed 's/^/    /'
if [ -n "$VER" ]; then L3=GREEN; else L3="AMBER(builtin?)"; fi
echo "  => Layer3 $L3"

# ---------------------------------------------------------------------------
hr; echo "LAYER 4 — user-space rdma-core / libibverbs dmabuf MR symbol"
LIB=$(ldconfig -p 2>/dev/null | grep -oP '/\S*libibverbs\.so\S*' | head -1)
echo "    libibverbs = ${LIB:-<not found>}"
if [ -n "$LIB" ] && strings "$LIB" 2>/dev/null | grep -qi 'ibv_reg_dmabuf_mr'; then
  L4=GREEN; echo "    symbol ibv_reg_dmabuf_mr: present"
else
  L4=RED; echo "    symbol ibv_reg_dmabuf_mr: ABSENT"
fi
echo "  => Layer4 $L4"

# ---------------------------------------------------------------------------
hr; echo "VERDICT"
printf "    Layer1 NIC-ODP        : %s\n" "$L1"
printf "    Layer1b peermem       : %s (affects ibv_reg_mr option, not no-pin)\n" "$PEER"
printf "    Layer2 MOVE_NOTIFY    : %s\n" "$L2"
printf "    Layer3 amdgpu         : %s\n" "$L3"
printf "    Layer4 rdma-core sym  : %s\n" "$L4"
hr
if [ "$L1" = GREEN ] && [ "$L2" = GREEN ] && [ "$L4" = GREEN ]; then
  echo "  RESULT: NO-PIN POSSIBLE — dynamic attach path available (zero-copy, no pin)."
  exit 0
else
  echo "  RESULT: PIN FORCED — ibv_reg_dmabuf_mr will PIN the GPU memory."
  echo "          Consequence: KFD double-counts the externally-pinned GPU mem,"
  echo "          so reported available GPU memory shrinks by ~the registered"
  echo "          size (reversible on dereg; no physical VRAM duplication)."
  [ "$L1" != GREEN ] && echo "          Blocking layer: L1 NIC has no ODP (ionic: unfixable by config)."
  exit 1
fi
