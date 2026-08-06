export ROLE=prefill MY_IP=10.245.156.167 P_IP=10.245.156.167 ETCD_IP=10.245.156.167 PORT=30000
export CTX=262144 DPA=1 KVAWARE=1 KVD=1 MTP=0
export GMU=0.70
export LOG=/shared_nfs/yihou_final_pr/logs/armA2_prefill.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
bash /shared_nfs/yihou_final_pr/scripts/glm52_leg_spur_mtp.sh
