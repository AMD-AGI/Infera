# PD test env — DSv4 sglang mooncake/mori 1p1d, chi2879(prefill) + chi2865(decode)
export KIT_DIR=/mnt/vast/c_huggingface/infera-pd/examples/deepseek_v4
export INFERA_IMAGE=infera/engine-sglang:pd-test
export INFERA_MODEL_MOUNT=/mnt/vast
export INFERA_MODEL=/mnt/vast/d_huggingface/models/DeepSeek-V4-Pro-fixed
export PREFILL_NODE=chi2879  PREFILL_IP=10.2.122.10
export DECODE_NODE=chi2865   DECODE_IP=10.2.122.52
export TOPO=1p1d
export ROUTER_PORT=8100
export ETCD_PORT=2379
export GID_INDEX=1
export CONC=64
