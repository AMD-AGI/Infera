#!/bin/bash
# Dump one py-spy frame per DP scheduler. Sample twice ~8s apart:
# identical output means a hang, not slow progress.
# Runs INSIDE the container.
for p in $(ps -eo pid,args --no-headers | grep -oE "^ *[0-9]+ .*sglang::scheduler_DP[0-9]" | awk '{print $1}'); do
  N=$(ps -p $p -o args= | grep -oE "DP[0-9]")
  echo "$N: $(py-spy dump --pid $p 2>&1 | sed -n '5p' | xargs)"
done
