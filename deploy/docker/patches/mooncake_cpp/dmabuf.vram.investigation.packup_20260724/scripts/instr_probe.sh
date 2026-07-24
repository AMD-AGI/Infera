echo "=== A. dma_buf debugfs (per-buffer, exporter, size, refcount) ==="
ls -l /sys/kernel/debug/dma_buf/ 2>/dev/null && head -40 /sys/kernel/debug/dma_buf/bufinfo 2>/dev/null || echo "  MISSING/na"
echo
echo "=== B. KFD debugfs mem ==="
ls /sys/kernel/debug/kfd/ 2>/dev/null || echo "  MISSING/na"
echo
echo "=== C. amdgpu DRI debugfs for card1 (TTM buddy real occupancy) ==="
for d in /sys/kernel/debug/dri/*; do
  nm=$(cat $d/name 2>/dev/null)
  echo "  $d -> $nm"
done
echo
echo "=== D. amdgpu_vram_mm / gtt_mm / gem_info existence (pick the card1 dri node) ==="
for d in /sys/kernel/debug/dri/*; do
  if cat $d/name 2>/dev/null | grep -q '0000:75:00.0'; then
    echo "  card1 dri = $d"
    ls $d | grep -iE 'vram|gtt|gem|amdgpu_evict|amdgpu_gtt' 
  fi
done
echo
echo "=== E. BAR1 / visible VRAM used (mem_info_vis_vram_used already known; also vram_vis_total) ==="
for f in mem_info_vis_vram_used mem_info_vis_vram_total mem_info_vram_used mem_info_vram_total mem_info_gtt_used mem_info_gtt_total; do
  printf "  %s = %s\n" "$f" "$(cat /sys/class/drm/card1/device/$f 2>/dev/null)"
done
