#!/usr/bin/env bash
# run_t_pack2 (task dcpx10, ON 135): run t_pack2 on GPU 4 (unused by the round, which holds GPU 0-3) in a throwaway container.
D=/shared_nfs/yihou_glm52/dcpx_20261022/dcpx10; O=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-93034eba91/python
rocm-smi --showpids 2>/dev/null | grep -E "^[0-9]+ " | head
docker run --name yihou-packtest-$(date +%H%M%S) --network host --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=$O -v /shared_nfs:/shared_nfs:ro \
  infera-sglang:v0522-1020-yihou-dcp-mc4340-destpin python3 $D/t_pack2.yihou.py $D/dcp_pack.yihou-nowait2.py 2>&1 | grep -v Warning | tail -12
