#!/bin/bash
# 4-layer diagnostic: can this system do dynamic-attach ("no-pin") dmabuf RDMA?
# Any layer RED -> falls back to pin -> the 2x/OOM behavior.
echo "############################################################"
echo "# NO-PIN (dynamic attach) capability — 4 layer diagnostic"
echo "# host=$(hostname) date=$(date +%F_%T)"
echo "############################################################"

echo
echo "===== LAYER 1: NIC ODP (On-Demand Paging) — THE decisive one ====="
echo "--- ionic_0 ODP caps via ibv_devinfo -v ---"
ibv_devinfo -d ionic_0 -v 2>/dev/null | grep -iA25 'on_demand_page\|odp_caps\|general_odp' | head -40
echo "--- explicit ODP grep (any nonzero SEND/RECV/WRITE/READ = supported) ---"
ibv_devinfo -d ionic_0 -v 2>/dev/null | grep -iE 'odp|on_demand|per_transport|SEND|RECV|WRITE|READ|ATOMIC|SRQ_RECV' | grep -iE 'odp|on_demand|per_transport'
echo "--- device list sanity ---"
ibv_devinfo 2>/dev/null | grep -iE 'hca_id|transport|state' | head

echo
echo "===== LAYER 2: kernel CONFIG_DMABUF_MOVE_NOTIFY ====="
( zcat /proc/config.gz 2>/dev/null | grep -i DMABUF_MOVE_NOTIFY ) || \
  grep -i DMABUF_MOVE_NOTIFY /boot/config-$(uname -r) 2>/dev/null || echo "  (config not found via /proc or /boot)"
grep -qE ' dma_buf_move_notify$' /proc/kallsyms 2>/dev/null && echo "  kallsyms: dma_buf_move_notify PRESENT" || echo "  kallsyms: dma_buf_move_notify ABSENT"

echo
echo "===== LAYER 3: amdgpu driver freshness (2025-04 pin/notify patch) ====="
echo "  uname -r = $(uname -r)"
modinfo amdgpu 2>/dev/null | grep -E '^version|^srcversion|^filename' | sed 's/^/  /'
dmesg 2>/dev/null | grep -i amdgpu | grep -iE 'version|peerdirect|kfd' | head -3 | sed 's/^/  /'
echo "  --- kfd dmabuf export capability ---"
dmesg 2>/dev/null | grep -i kfd | grep -iE 'dmabuf|peer|p2p' | head | sed 's/^/  /'

echo
echo "===== LAYER 4: user-space rdma-core / libibverbs dmabuf MR ====="
LIB=$(ldconfig -p 2>/dev/null | grep -oP '/\S*libibverbs\.so\S*' | head -1)
echo "  libibverbs = $LIB"
strings "$LIB" 2>/dev/null | grep -i reg_dmabuf | sed 's/^/  sym: /'
( dpkg -l 2>/dev/null | grep rdma-core || rpm -q rdma-core 2>/dev/null || echo "  (no pkg db)" ) | sed 's/^/  /'
echo "  --- ionic provider version ---"
ls -l /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null | sed 's/^/  /'

echo
echo "===== QUICK VERDICT ====="
echo "  Layer1 ODP  : (read caps above — all-zero = RED = forces pin)"
echo "  Layer2 CFG  : MOVE_NOTIFY above"
echo "  Layer3 amdgpu: version above"
echo "  Layer4 rdma : reg_dmabuf sym above"
echo "  => any RED  -> dynamic attach impossible -> pin path -> 2x budget / OOM"
