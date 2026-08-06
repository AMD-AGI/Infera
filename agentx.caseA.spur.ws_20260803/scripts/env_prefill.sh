export ROLE=prefill MY_IP=10.245.145.242 P_IP=10.245.145.242 ETCD_IP=10.245.145.242 PORT=30000
export CTX=262144 DPA=0 KVAWARE=1 KVD=1 MTP=0
export CHUNK=65536
export GMU=0.70
export LOG=/shared_nfs/yihou_agentx_caseA/logs/prefill.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/triton_cache
bash /shared_nfs/yihou_agentx_caseA/scripts/glm52_leg_spur_mtp.sh
