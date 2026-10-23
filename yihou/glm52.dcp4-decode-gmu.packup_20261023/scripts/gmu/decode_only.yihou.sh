#!/usr/bin/env bash
# Decode-only start-up memory probe for the DCP4 decode leg (no prefill, no router, no etcd).
# Starts one sglang decode server (S2 argv, launch_server directly), waits until it is fired up or dies,
# extracts the memory lines into $OUT/<tag>.mem.txt, then docker-stops the container (never rm).
# Usage on the node (spur exec <job> ...):
#   NK=<nk> SLOT=a|b TAG=<tag> MEM=0.92 EXTRA="<extra sglang args>" ENVX="K=V K2=V2" bash decode_only.yihou.sh
#   SLOT a = GPUs 0-3 port 27311; SLOT b = GPUs 4-7 port 27321. KEEP=1 leaves the server up.
set -uo pipefail
: "${NK:?}"; : "${SLOT:?}"; : "${TAG:?}"; MEM=${MEM:-0.92}; EXTRA=${EXTRA:-}; ENVX="${ENVX:-} ${GLOBAL_ENVX:-}"  # GLOBAL_ENVX=HSA_ENABLE_SDMA=0 on burst v3 nodes
IMAGE=${IMAGE:-infera-sglang:v0522-d395c3b3-1020-mc4340}
MODEL=/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4
OVL=/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-ab7a898b77/python/sglang
OUT=/shared_nfs/yihou_glm52/dcp_pd_20261021/gmu
case $SLOT in a) GPUS=0,1,2,3; PORT=27311; IB='{"0":"ionic_0","1":"ionic_1","2":"ionic_2","3":"ionic_3"}';;
              b) GPUS=4,5,6,7; PORT=27321; IB='{"0":"ionic_4","1":"ionic_5","2":"ionic_6","3":"ionic_7"}';;
              *) echo "bad SLOT"; exit 2;; esac
# Slot b burns 30,000 PIDs first: both containers share /dev/shm (--ipc host) but not the PID namespace, and
# RCCL init failed in slot a ('HIP failure: invalid argument') when both started together; cause not confirmed.
[[ $SLOT == b ]] && BURN=30000 || BURN=0
CTR=yihou-dcp-pd-$NK-gmu-$TAG   # unique per run: containers are stopped, never removed
IP=$(ip -4 -o addr show ens3 | awk '{print $4}' | cut -d/ -f1)
mkdir -p "$OUT"; LOG=$OUT/$TAG.log
[[ -e $LOG ]] && { echo "log exists: $LOG"; exit 1; }
docker ps -a --format '{{.Names}}' | grep -qx "$CTR" && { echo "container $CTR exists; pick a new TAG"; exit 1; }
busy=$(docker ps --format '{{.Names}}' | grep -E "^yihou-dcp-pd-$NK-gmu-" | while read -r c; do docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$c" | grep -x "HIP_VISIBLE_DEVICES=$GPUS" >/dev/null && echo "$c"; done)
[[ -z "$busy" ]] || { echo "slot $SLOT busy: $busy"; exit 1; }
envargs=()
for kv in HIP_VISIBLE_DEVICES=$GPUS SGLANG_HOST_IP=$IP HOST_IP=$IP SGLANG_LOCAL_IP_NIC=ens3 GLOO_SOCKET_IFNAME=ens3 \
    NCCL_SOCKET_IFNAME=ens3 NCCL_IB_DISABLE=1 NCCL_IGNORE_CPU_AFFINITY=1 HSA_NO_SCRATCH_RECLAIM=1 HIP_FORCE_DEV_KERNARG=1 \
    AITER_USE_FLYDSL_MOE_SORTING=1 PYTHONHASHSEED=0 SAFETENSORS_FAST_GPU=1 SGLANG_OPT_USE_TOPK_V2=true \
    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 MC_TE_FILTERS=yihou_no_such_rdma_device MC_GID_INDEX=1 \
    MOONCAKE_DISABLE_HIP_DMABUF=0 MC_DISABLE_HIP_TRANSPORT=1 MC_ENABLE_DEST_DEVICE_AFFINITY=1 MC_ENABLE_DEST_LOCAL_RAIL=1 \
    MC_IB_TIMEOUT=18 RDMAV_FORK_SAFE=1 MC_MAX_MR_SIZE=1073741824 SGLANG_SIMULATE_ACC_LEN=3.61 \
    SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token SGLANG_DP_USE_GATHERV=0 \
    SGLANG_DSA_DCP_KEEP_KV_B_BF16=1 SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1 $ENVX; do
  envargs+=(-e "$kv"); done
args=(--model-path $MODEL --served-model-name glm5.2-mxfp4 --trust-remote-code --host "$IP" --port $PORT
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-ib-device "$IB"
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --chunked-prefill-size 32768
  --json-model-override-args '{"index_share_for_mtp_iteration":true}' --enable-fused-qk-norm-rope
  --enable-aiter-allreduce-fusion --reasoning-parser glm45 --tool-call-parser glm47 --enable-metrics
  --watchdog-timeout 3600 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 6 --tp-size 4 --ep-size 4 --dcp-size 4 --mem-fraction-static "$MEM"
  --cuda-graph-max-bs-decode 96 --max-running-requests 96)
# shellcheck disable=SC2206
extra=($EXTRA)
echo "$(date -u +%FT%TZ) $TAG NK=$NK SLOT=$SLOT GPUS=$GPUS MEM=$MEM EXTRA=[$EXTRA] ENVX=[$ENVX]" > "$LOG"
docker run -d --init --name "$CTR" --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --device /dev/infiniband --group-add video --group-add render --cap-add SYS_PTRACE --cap-add IPC_LOCK \
  --security-opt seccomp=unconfined --ulimit memlock=-1:-1 --ulimit nofile=65536:65536 \
  -v $MODEL:$MODEL:ro -v "$OVL":/sgl-workspace/sglang/python/sglang:ro -v "$OUT":"$OUT" \
  -v yihou-dcp-pd-gmu-cache-$SLOT:/root/.cache -v yihou-dcp-pd-gmu-aiter-$SLOT:/sgl-workspace/aiter/aiter/jit \
  "${envargs[@]}" --entrypoint bash "$IMAGE" -c 'i=0; while [ $i -lt '"$BURN"' ]; do ( : ); i=$((i + 1)); done; exec "$@" >> '"$LOG"' 2>&1' _ \
  python3 -m sglang.launch_server "${args[@]}" "${extra[@]}" >/dev/null || { echo "docker run failed"; exit 1; }
st=FAIL
for i in $(seq 1 180); do
  sleep 10
  grep -q "The server is fired up" "$LOG" && { st=UP; break; }
  docker ps --format '{{.Names}}' | grep -qx "$CTR" || { st=DIED; break; }
  grep -qE "Traceback|OutOfMemory|out of memory|HIP error" "$LOG" && sleep 30 && ! docker ps --format '{{.Names}}' | grep -qx "$CTR" && { st=DIED; break; }
done
{
  echo "## $TAG status=$st MEM=$MEM EXTRA=[$EXTRA] ENVX=[$ENVX] $(date -u +%FT%TZ)"
  grep -E "TP0" "$LOG" | grep -E "Load weight (begin|end)|KV Cache is allocated|Memory pool end|dcp_replicate_q_proj|Capture .* CUDA graph (begin|end)|max_total_num_tokens|workspace preallocated" | sed -E 's/^\[[^]]*\] //' | cut -c1-260
  grep -E "OutOfMemory|out of memory|HIP out|Error" "$LOG" | head -5 | cut -c1-300
  rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys,json;d=json.load(sys.stdin);print("vram_used_GiB", {k:round(int(v["VRAM Total Used Memory (B)"])/2**30,1) for k,v in d.items()})' 2>/dev/null
} > "$OUT/$TAG.mem.txt"
cat "$OUT/$TAG.mem.txt"
[[ "${KEEP:-0}" == 1 ]] || docker stop -t 30 "$CTR" >/dev/null
echo "$(date -u +%FT%TZ) done $TAG status=$st (container $( [[ ${KEEP:-0} == 1 ]] && echo kept || echo stopped))"
