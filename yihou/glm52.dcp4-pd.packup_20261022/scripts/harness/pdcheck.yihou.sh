#!/usr/bin/env bash
# Grep the live PD logs of a round for the hypotheses main asked to test per run (login host, read-only).
# Usage: pdcheck.yihou.sh ROUND
R=/shared_nfs/yihou_glm52/dcp_pd_20261021/pd/runs/$1/server-logs
P=$R/prefill-0.log; D=$(ls -t $R/decode-0.*.log 2>/dev/null | head -1)
g(){ strings "$1" | grep -a -E "$2" | cut -c1-260 | sort | uniq -c | sort -rn | head -${3:-6}; }
for f in "$P" "$D"; do
  echo "=================== $f"
  echo "-- DCP pack buffers";             g "$f" "PD DCP pack buffers allocated"
  echo "-- full-head Q weights";          g "$f" "prepared full-head Q weights"
  echo "-- dcp / server_args dcp_size";   g "$f" "dcp_size=[0-9]+|dcp_size': [0-9]+" 3
  echo "-- KV capacity";                  g "$f" "max_total_num_tokens=[0-9]+" 3
  echo "-- KV/draft pool alloc";          g "$f" "KV Cache is allocated|token_to_kv_pool|#tokens: [0-9]+" 6
  echo "-- mooncake HIP/registration";    g "$f" "HIP transport installed|Found [0-9]+ HCAs|Failed to install|register.*fail|hipIpc[A-Za-z]* failed|extends past its allocation" 8
  echo "-- transfer errors";              g "$f" "falling back|geometry differs|draft head slicing|item length mismatch|state layout|Rejecting state transfers|KVTransferError|transfer failed|Failed to send" 8
  echo "-- faults";                       g "$f" "Memory access fault|out of memory|OutOfMemory|Traceback|Error" 8
  echo "-- up";                           g "$f" "The server is fired up|Capture .*graph" 3
done
