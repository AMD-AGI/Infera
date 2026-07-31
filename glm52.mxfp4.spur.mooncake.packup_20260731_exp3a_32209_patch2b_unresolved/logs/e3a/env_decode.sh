export MY_IP=10.245.147.58 P_IP=10.245.158.72 ROLE=decode PORT=30001 DPA=1 MTP=1
export LOG=/shared_nfs/yihou_exp3way/e3a/decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_exp3way/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_exp3way/triton_cache
export PREFILL_MTP=0
bash /shared_nfs/yihou_exp3way/pd_leg_exp.sh
