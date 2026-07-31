#!/usr/bin/env bash
# GLM-5.2 mooncake PD, DP-attention + MTP COMBINED (the never-run combo). Reproduction of the
# reported "mtp+dpa+pd doesn't work". prefill=chi2835(10.2.122.78) decode=chi2867(10.2.122.44).
# Both legs symmetric-DPA; MTP (EAGLE) only on decode leg + patched deepseek_nextn.py mounted there.
# Uses sglang_router mini-LB (isolates DPA+MTP; NOT the kv-aware/kvd stack).
set -euo pipefail
IMAGE=infera/engine-sglang:pd-unified
MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
SERVED=glm5.2-mxfp4
PREFILL_HOST=chi2835; PREFILL_IP=10.2.122.78
DECODE_HOST=chi2879;  DECODE_IP=10.2.122.10
ROUTER_PORT=8002
MAX_RUNNING="${MAX_RUNNING:-2048}"
DMABUF="${DMABUF:-0}"
CTX="${CTX:-32768}"
MTP="${MTP:-1}"
CTR=pd_uni
KIT=/mnt/vast/c_huggingface/glm52_dpa_mtp
NEXTN=/mnt/vast/c_huggingface/glm52_nextn_patch_unified/deepseek_nextn.py
NEXTN_DST=/sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
J(){ ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 root@149.28.124.225 "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 $1 '$2'"; }

# 1. container + libionic inject on both nodes
prep(){ local h="$1"
  J "$h" "docker rm -f $CTR >/dev/null 2>&1 || true; docker run -d --name $CTR --network=host --ipc=host --shm-size=32G --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --group-add video --group-add render --cap-add=SYS_PTRACE --cap-add=IPC_LOCK --security-opt seccomp=unconfined --ulimit memlock=-1:-1 -v /mnt/vast:/mnt/vast --entrypoint \"\" $IMAGE sleep infinity >/dev/null"
  J "$h" "HL=\$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1); B=\$(basename \$HL); docker cp \$HL $CTR:/usr/lib/x86_64-linux-gnu/\$B; docker exec $CTR bash -lc \"cd /usr/lib/x86_64-linux-gnu && ln -sf \$B libionic.so.1 && ln -sf libionic.so.1 libionic.so && cd libibverbs && ln -sf ../\$B libionic-rdmav34.so && ldconfig 2>/dev/null; echo active_ports=\\\$(ibv_devinfo | grep -c PORT_ACTIVE)\""
}
echo "== 1. prep containers (libionic inject) =="; prep "$PREFILL_HOST"; prep "$DECODE_HOST"

# 2. stage merged leg script into both; mount patched nextn into DECODE (MTP eh_proj fix)
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
  J "$h" "docker cp $KIT/pd_leg_dpa_mtp.sh $CTR:/pd_leg_dpa_mtp.sh; docker cp $KIT/probe.py $CTR:/tmp/probe.py 2>/dev/null || true"
done
if [ "$MTP" = "1" ]; then
  echo "== 2b. mount patched deepseek_nextn.py into decode container =="
  J "$DECODE_HOST" "docker cp $NEXTN $CTR:$NEXTN_DST && docker exec $CTR grep -c 'eh_proj  # GLM-5.2' $NEXTN_DST"
fi

# 3. launch both legs (decode :30001, MTP on decode only)
echo "== 3. launch prefill DPA leg ($PREFILL_HOST :30000) =="
J "$PREFILL_HOST" "docker exec -d $CTR env ROLE=prefill MY_IP=$PREFILL_IP MODEL=$MODEL SERVED=$SERVED PORT=30000 DPA=1 MTP=0 DMABUF=$DMABUF CTX=$CTX MAX_RUNNING=$MAX_RUNNING bash /pd_leg_dpa_mtp.sh"
echo "== 4. launch decode DPA+MTP leg ($DECODE_HOST :30001) =="
J "$DECODE_HOST" "docker exec -d $CTR env ROLE=decode MY_IP=$DECODE_IP MODEL=$MODEL SERVED=$SERVED PORT=30001 DPA=1 MTP=$MTP DMABUF=$DMABUF CTX=$CTX MAX_RUNNING=$MAX_RUNNING bash /pd_leg_dpa_mtp.sh"

echo "== legs launching (mooncake RDMA + DPA + MTP=$MTP). DP+MTP cold start ~8-12min. =="
echo "   watch prefill: J $PREFILL_HOST \"grep -aE 'ready to roll|error|Traceback|CUDA|out of memory' $KIT/pd_prefill_30000_dpamtp.log | tail\""
echo "   watch decode:  J $DECODE_HOST  \"grep -aE 'ready to roll|error|Traceback|CUDA|out of memory' $KIT/pd_decode_30001_dpamtp.log | tail\""
echo "   After BOTH 'ready to roll', start router on $PREFILL_HOST:"
echo "   J $PREFILL_HOST \"docker exec -d $CTR bash -lc 'pkill -9 -f sglang_router; python3 -m sglang_router.launch_router --pd-disaggregation --prefill http://$PREFILL_IP:30000 8998 --decode http://$DECODE_IP:30001 --host 0.0.0.0 --port $ROUTER_PORT > /tmp/router.log 2>&1'\""
