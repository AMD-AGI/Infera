#!/bin/bash
for i in 1 2 3 4 5 6 7 8; do
  docker pull rocm/sgl-dev:sglang-0.5.13.post1-rocm720-mi35x-mori-0615 >> /home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg/pull_260.log 2>&1 && { echo "PULL_OK attempt $i" >> /home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg/pull_260.log; break; }
  echo "PULL_RETRY $i $(date)" >> /home/yihou/dev/git/infera.yihou.dev/temp/dsv4_sglang_spur/round3_pd_disagg/pull_260.log
  sleep 5
done
