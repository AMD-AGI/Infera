#!/usr/bin/env bash
# Start vram_sampler detached (same pattern as node_go: setsid nohup bash -c in a script file).
setsid nohup bash -c "bash /shared_nfs/yihou_glm52/dcp_pd_20261021/gmu/tools/vram_sampler.yihou.sh $1 ${2:-6000}" > /dev/null 2>&1 < /dev/null &
echo "sampler started -> $1"
