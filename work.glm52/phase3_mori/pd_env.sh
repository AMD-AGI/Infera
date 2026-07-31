# Phase 3 PD-mori config. Prefill=chi2878(10.2.122.3), Decode=chi2879(10.2.122.10).
export IMAGE=rocm/infera:sglang-v0.1.0-rc6
export MODEL=/mnt/vast/xiaobo/models/GLM-5.2-MXFP4
export SERVED=glm5.2-mxfp4
export PREFILL_HOST=chi2878
export DECODE_HOST=chi2879
export PREFILL_IP=10.2.122.3
export DECODE_IP=10.2.122.10
export ETCD_EP=10.2.122.3:2379
export NATS_PREFILL=nats://10.2.122.3:4222
export NATS_DECODE=nats://10.2.122.10:4222
export ROUTER_PORT=8100
