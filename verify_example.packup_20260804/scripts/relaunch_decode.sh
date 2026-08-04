#!/usr/bin/env bash
# Reap the decode leg inside glm52_pd and relaunch it through the kit's own
# engine/leg.sh, so the relaunched leg is byte-identical to what up.sh started
# except for the GLM52_P1V3 source edit already applied in the container.
#
# Runs ON the decode node. Staged as a FILE deliberately: the same commands sent
# through nested `ssh ... "docker exec ... bash -c '...'"` silently no-op.
set -u
KIT=/mnt/vast/c_huggingface/glm52_example_verify/kit
CTR=glm52_pd

echo "===== reap decode engine ====="
docker exec "$CTR" bash -c '
  pkill -9 -f "infera\.engine\.sglang"  2>/dev/null
  pkill -9 -f "sglang\.launch_server"   2>/dev/null
  pkill -9 -f "multiprocessing\.spawn"  2>/dev/null
  true'
for i in $(seq 1 40); do
  n=$(docker exec "$CTR" bash -c 'ps -eo comm,args | grep -E "launch_server|infera\.engine|sglang::" | grep -v grep | wc -l' | tr -d '\r')
  [ "${n:-1}" -eq 0 ] && { echo "  engines gone after $((i*3))s"; break; }
  sleep 3
done
docker exec "$CTR" bash -c 'ps -eo args | grep -E "launch_server|sglang::" | grep -v grep | wc -l'

echo "===== wait for VRAM ====="
for i in $(seq 1 40); do
  n=$(rocm-smi --showpids 2>/dev/null | grep -cE '^[0-9]+' || true); n=${n:-0}
  [ "$n" -eq 0 ] && { echo "  GPUs idle after $((i*3))s"; break; }
  sleep 3
done
rocm-smi --csv --showmeminfo vram 2>/dev/null | tail -8 | awk -F, '{printf "  %s %.1f GB\n", $1, $3/1073741824}'

echo "===== confirm the patch is in the SOURCE the next import will read ====="
docker exec "$CTR" bash -c 'F=/sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
  echo -n "  GLM52_P1V3 in source: "; grep -c GLM52_P1V3 "$F"
  rm -f /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/__pycache__/dsa_indexer*.pyc
  echo "  stale .pyc removed (if any)"'

echo "===== relaunch through the kit ====="
# Same env up.sh passes for the decode leg.
CTR=$CTR \
INFERA_IMAGE=infera/engine-sglang:merged-e \
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4 \
MODEL_MOUNT=/mnt/vast \
SERVED=glm5.2-mxfp4 \
TOKENIZER=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4 \
ETCD_IP=10.2.122.78 ETCD_PORT=2379 \
ROUTER_PORT=8100 PREFILL_PORT=30000 DECODE_PORT=30001 \
BOOTSTRAP_PORT=8998 KVD_SOCK=/tmp/kvd/kvd.sock \
TP=8 CTX=262144 CHUNK=65536 KVAWARE=1 \
RDMA_IB_DEVICES=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_6,ionic_7 \
MC_GID_INDEX=1 MOONCAKE_DISABLE_HIP_DMABUF=1 RDMAV_FORK_SAFE=1 \
GMU_PREFILL=0.70 GMU_DECODE=0.85 \
ROLE=decode MY_IP=10.2.122.10 PORT=30001 DPA=1 MTP=1 KVD=0 \
  bash "$KIT/engine/leg.sh"
