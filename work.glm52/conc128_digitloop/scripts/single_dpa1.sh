#!/bin/bash
# Single-node colocated, DP-attention ON, NO disaggregation.
# The one-variable split against up_single_nodpa.sh: everything identical except DPA.
# chunk 65536 so the PER-RANK chunk is 65536/dp8 = 8192, matching both the no-DPA arm and the
# PD run -> same compute shape, only the parallelism layout differs.
#
# Runs INSIDE the pd_uni container. Staged as a FILE, not `docker exec ... bash -c "..."`:
# nested quoting made the redirect evaluate on the outer shell and the server never launched
# (log ended up containing only the ssh banner). Same class of bug as the REPRODUCE pitfalls.
export SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
export SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0 SGLANG_OPT_USE_JIT_NORM=0
export SGLANG_DP_USE_GATHERV=1
export SAFETENSORS_FAST_GPU=1 HIP_FORCE_DEV_KERNARG=1 HSA_NO_SCRATCH_RECLAIM=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG=/mnt/vast/c_huggingface/glm52_longctx_pd/single_dpa1_30000.log
exec python3 -m sglang.launch_server \
  --model-path /mnt/vast/xiaobo/models/GLM-5.2-MXFP4 --served-model-name glm5.2-mxfp4 \
  --host 0.0.0.0 --port 30000 --tp-size 8 --trust-remote-code \
  --nsa-prefill-backend tilelang --nsa-decode-backend tilelang \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.88 \
  --context-length 32768 --chunked-prefill-size 65536 \
  --max-running-requests 2048 --cuda-graph-max-bs 128 \
  --dp-size 8 --enable-dp-attention --ep-size 8 \
  --watchdog-timeout 3600 > "$LOG" 2>&1
