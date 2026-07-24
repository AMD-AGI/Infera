echo "=== 1. uname -r ==="
uname -r
echo
echo "=== 2. zcat /proc/config.gz (may be absent) ==="
if [ -r /proc/config.gz ]; then zcat /proc/config.gz 2>/dev/null | grep -E 'DMABUF_MOVE_NOTIFY|PCI_P2PDMA' ; else echo "(no /proc/config.gz)"; fi
echo
echo "=== 3. /boot/config ==="
grep -E 'DMABUF_MOVE_NOTIFY|PCI_P2PDMA' /boot/config-$(uname -r) 2>/dev/null || echo "(no /boot/config)"
echo
echo "=== 4. kallsyms live symbols (kernel actually built with it) ==="
grep -cE ' dma_buf_move_notify$' /proc/kallsyms && echo "dma_buf_move_notify count above"
grep -E 'dma_buf_pin|dma_buf_dynamic_attach|dma_buf_move_notify' /proc/kallsyms | head
echo
echo "=== 5. amdgpu module version / srcversion + build date (2025-04 patch check) ==="
modinfo amdgpu 2>/dev/null | grep -iE '^version|^srcversion|^vermagic|^filename' | head
echo "--- amdgpu in-kernel? (built-in vs module) ---"
ls -l /sys/module/amdgpu/ 2>/dev/null | grep -iE 'version|srcversion' 
cat /sys/module/amdgpu/version 2>/dev/null || echo "(amdgpu builtin, no module version)"
echo
echo "=== 6. does amdgpu export move_notify path? (dmabuf dynamic attach symbol used by ib_core) ==="
grep -E 'ib_umem_dmabuf_get_pinned|ib_umem_dmabuf_get' /proc/kallsyms | head
echo
echo "=== 7. dmesg amdgpu build/version banner ==="
dmesg 2>/dev/null | grep -iE 'amdgpu.*(initial|version|kernel modesetting)' | head -3 || echo "(dmesg not readable)"
