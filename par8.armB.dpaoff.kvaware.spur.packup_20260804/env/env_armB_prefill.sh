export ROLE=prefill MY_IP=10.245.158.155 P_IP=10.245.158.155 ETCD_IP=10.245.158.155 PORT=30000
export CTX=262144 DPA=0 KVAWARE=1 KVD=1 MTP=0
export CHUNK=65536
export GMU=0.70
export LOG=/shared_nfs/yihou_final_pr/logs/armB_prefill.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_final_pr/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_final_pr/triton_cache
bash /shared_nfs/yihou_final_pr/scripts/glm52_leg_spur_mtp.sh
