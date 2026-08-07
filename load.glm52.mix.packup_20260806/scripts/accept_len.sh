#!/bin/bash
# Read the MTP acceptance-length DISTRIBUTION off an engine log.
# Median 2-3 is healthy; a median AT 4.00 means the draft model is predicting a
# repetition loop perfectly, i.e. the output has degenerated.
LOG="${1:-/tmp/glm52_mix_base.log}"
strings "$LOG" | grep -o 'accept len: [0-9.]*' | awk '{print $3}' | sort -n | awk '
  {a[NR]=$1; if ($1>=4) f++}
  END {if (!NR) {print "  (no accept-len lines)"; exit 1}
       printf "  n=%d  p10=%s  MEDIAN=%s  p90=%s  at-4.00=%d (%.1f%%)\n",
              NR, a[int(NR*0.1)+1], a[int(NR*0.5)+1], a[int(NR*0.9)+1], f, 100*f/NR}'
