export ROLE=decode MY_IP=10.245.152.60 P_IP=10.245.145.242 ETCD_IP=10.245.145.242 PORT=30000
export CTX=262144 DPA=1 KVAWARE=1 KVD=1 MTP=1
export GMU=0.85
export LOG=/shared_nfs/yihou_agentx_caseA/logs/decode.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/triton_cache
bash /shared_nfs/yihou_agentx_caseA/scripts/glm52_leg_spur_mtp.sh
