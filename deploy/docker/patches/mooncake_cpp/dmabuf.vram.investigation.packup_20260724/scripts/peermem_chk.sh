echo "=== peermem module state (determines if ibv_reg_mr GPU-direct is even possible) ==="
lsmod 2>/dev/null | grep -iE 'peermem|ib_peer|amdgpu_peermem' || echo "  no *peermem* module loaded"
echo "--- available peermem modules ---"
find /lib/modules/$(uname -r) -iname '*peermem*' 2>/dev/null || echo "  none packaged"
echo "--- dmesg peermem ---"
dmesg 2>/dev/null | grep -iE 'peermem|peer_mem|peerdirect' | head
echo "=== ODP caps raw (ionic_0) ==="
ibv_devinfo -d ionic_0 -v 2>/dev/null | grep -iA6 'odp_caps' | head -30
