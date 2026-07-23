#!/bin/bash
# status helper: run INSIDE container. Args: <logfile> <label>
L="$1"; LB="${2:-log}"
echo "=== $LB ==="
grep -aiE "Memory pool end|DSV4 pool sizes|DSV4 memory calc|max_total_num_tokens=|fired up and ready|out of memory|Traceback|Aborted|Capturing" "$L" 2>/dev/null | grep -av server_args | tail -5
echo "--- raw tail ---"
tail -c 800 "$L" 2>/dev/null | tr '\r' '\n' | tail -4
echo "ALIVE=$(pgrep -f sglang.launch_server|wc -l) VRAM_c0=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR==2{printf "%.1fGB", $3/1073741824}')"
