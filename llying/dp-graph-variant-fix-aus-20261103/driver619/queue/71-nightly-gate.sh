#!/bin/bash
# DRIVER619-20261113 formal delivery, SGLang nightly: one-node 4+4 gate of GLM-5.2 on the 6.19 node with the nightly stack
# (e1ab-sglang:1112-hipgem: SGLang src = main 44ad9d69f + e1ab patches + the AMD_GEM PR, Mooncake main + the
# HipGemAllocator PR) and SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM; otherwise e1ab's settings (MOONCAKE_DISABLE_HIP_DMABUF=1,
# MC_DISABLE_HIP=1). Reference: e1ab on 6.19 (n04-25, 11-12 08:34-09:13), same nightly with RDMA: prefill segfault 5/5.
srun --jobid=$JOB --overlap -N1 -n1 -w $NODE619 --exclude=$NODE616 env SLURM_JOB_NODELIST=$NODE619 MIN_EVICT=0 \
    KFD619_ENV_EXTRA="SGLANG_MOONCAKE_CUSTOM_MEM_POOL=AMD_GEM" PROBE_N=200 LONGCTX_SPECS="16000:64" EVAL=0 EST_MIN=45 \
    bash $K/nightly/nightly_item.sh nightly-gemkv-gate gate 1node
tail -12 $K/nightly/logs/nightly.out
