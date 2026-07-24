echo "=== locate dri node for 0000:75:00.0 (card1) ==="
CARD1=""
for d in /sys/kernel/debug/dri/*/; do
  nm=$(cat ${d}name 2>/dev/null)
  echo "$d -> $nm" | grep -qi '75:00.0' && { CARD1=$d; echo "  MATCH: $d"; }
done
[ -z "$CARD1" ] && { echo "  no 75:00.0 match; dumping all names:"; for d in /sys/kernel/debug/dri/*/; do echo "$d -> $(cat ${d}name 2>/dev/null)"; done; exit 0; }
echo
echo "=== $CARD1 amdgpu_vram_mm (tail = totals) ==="
tail -8 ${CARD1}amdgpu_vram_mm 2>/dev/null
echo
echo "=== amdgpu_evict_vram / evict_gtt ==="
cat ${CARD1}amdgpu_evict_vram 2>/dev/null
cat ${CARD1}amdgpu_evict_gtt 2>/dev/null
echo
echo "=== amdgpu_gem_info line count (num BOs) ==="
wc -l ${CARD1}amdgpu_gem_info 2>/dev/null
