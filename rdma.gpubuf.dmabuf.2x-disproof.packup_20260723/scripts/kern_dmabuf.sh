echo "##### $(hostname) #####"
echo "=== 1. CONFIG_DMABUF_MOVE_NOTIFY / PCI_P2PDMA in kernel config ==="
f=/boot/config-$(uname -r)
[ -r "$f" ] && grep -E "CONFIG_DMABUF_MOVE_NOTIFY|CONFIG_PCI_P2PDMA|CONFIG_DMABUF_HEAPS|CONFIG_HSA_AMD_P2P" "$f"
grep -c dma_buf_move_notify /proc/kallsyms 2>/dev/null | sed 's/^/dma_buf_move_notify syms: /'
echo ""
echo "=== 2. ionic_rdma module version ==="
modinfo ionic_rdma 2>/dev/null | grep -iE "version|filename"
echo ""
echo "=== 3. amdgpu p2pdma / migrate evidence in dmesg ==="
dmesg 2>/dev/null | grep -iE "peer-to-peer DMA|p2pdma|amdgpu.*dmabuf|migrat" | tail -8 || echo "  (no dmesg perm)"
