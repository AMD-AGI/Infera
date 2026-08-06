export ROLE=decode MY_IP=10.245.151.18 P_IP=10.245.158.155 ETCD_IP=10.245.158.155 PORT=30001
export CTX=262144 DPA=1 KVAWARE=1 KVD=0 MTP=1
export LOG=/shared_nfs/yihou_final_pr/logs/armB_decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
bash /shared_nfs/yihou_final_pr/scripts/glm52_leg_spur_mtp.sh
