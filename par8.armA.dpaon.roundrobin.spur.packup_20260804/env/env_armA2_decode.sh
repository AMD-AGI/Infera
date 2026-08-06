export ROLE=decode MY_IP=10.245.152.164 P_IP=10.245.156.167 ETCD_IP=10.245.156.167 PORT=30001
export CTX=262144 DPA=1 KVAWARE=1 KVD=0 MTP=1
export LOG=/shared_nfs/yihou_final_pr/logs/armA2_decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
bash /shared_nfs/yihou_final_pr/scripts/glm52_leg_spur_mtp.sh
