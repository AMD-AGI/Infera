#!/bin/bash
# sum VRAM used across 8 cards
echo "t=$(date +%H:%M:%S)"
rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 && $3 ~ /[0-9]/ {s+=$3; n++} END{printf "vram_sum=%.1fGB over %d cards\n", s/1073741824, n}'
