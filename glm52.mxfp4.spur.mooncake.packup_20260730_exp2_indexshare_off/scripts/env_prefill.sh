export MY_IP=10.245.152.243 P_IP=10.245.152.243 ROLE=prefill PORT=30000 DPA=1 MTP=1
export LOG=/shared_nfs/yihou_exp3way/e2/prefill.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_exp3way/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_exp3way/triton_cache
export PREFILL_MTP=1
export EXTRA_ARGS='--json-model-override-args {"index_share_for_mtp_iteration":false}'
bash /shared_nfs/yihou_exp3way/pd_leg_exp.sh
