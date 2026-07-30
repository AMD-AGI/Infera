#!/bin/bash
# Kill all our vLLM engine workers inside the container, keep infera.server router.
for pat in "vllm.entrypoints.cli.main serve" "VLLM::EngineCore" "multiprocessing.spawn" "multiprocessing.resource_tracker" "pt_main_thread"; do
  pkill -9 -f "$pat" 2>/dev/null
done
sleep 1
for p in $(ps -eo pid,cmd | grep python3 | grep -v grep | grep -viE "infera.server|killworkers" | awk '{print $1}'); do
  kill -9 "$p" 2>/dev/null
done
sleep 2
echo "remaining engine procs:"
ps -eo pid,cmd | grep python3 | grep -v grep | grep -viE "infera.server" | head
echo "done"
