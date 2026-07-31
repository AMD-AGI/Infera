export MY_IP=10.245.157.171 P_IP=10.245.159.138 ROLE=decode PORT=30001 DPA=1 MTP=1
export LOG=/shared_nfs/yihou_exp3way/e1/decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_exp3way/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_exp3way/triton_cache
export PREFILL_MTP=0
bash /shared_nfs/yihou_exp3way/pd_leg_exp.sh
