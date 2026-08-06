# arm 3 -- decode leg with MTP OFF, which makes infera append
# --disaggregation-decode-enable-radix-cache (args.py:261-278).
#
# MTP=0 moves FOUR things at once in the leg script. Three are intended or
# already decoupled; ONE would silently ride along and is pinned here.
#
#   1. EAGLE spec-dec            on -> off      <- under test
#   2. decode radix cache        ChunkCache -> RadixCache   <- under test
#      (infera refuses to append the flag under --speculative-algorithm;
#       removing MTP is what legalises it. They are ONE switch, not two.)
#   3. --num-reserved-decode-tokens  256 -> 512  <- NOT intended.
#      RESERVED_TOK lives INSIDE the MTP_ARGS block (glm52_leg_spur_mtp.sh:198),
#      so dropping MTP drops the flag and sglang's default (512) takes over.
#      Pinned to 256 via EXTRA_ARGS so it is held fixed across the arms.
#   4. --disable-custom-all-reduce  unchanged. CUSTOM_AR is already independent
#      of MTP in the leg script (it was coupled once; that trap is fixed).
#
# GMU stays 0.85 -- the decode leg never OOM'd, and moving it would add a
# variable. Note the radix cache draws from the SAME static reservation, so if
# this arm retracts, RAISE GMU (decode direction), never lower it.
export ROLE=decode MY_IP=10.245.152.60 P_IP=10.245.145.242 ETCD_IP=10.245.145.242 PORT=30000
export CTX=262144 DPA=1 KVAWARE=1 KVD=1 MTP=0
export GMU=0.85
export EXTRA_ARGS="--num-reserved-decode-tokens 256"
export LOG=/shared_nfs/yihou_agentx_caseA/logs/decode_nomtp.log
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/inductor_cache
export TRITON_CACHE_DIR=/shared_nfs/yihou_agentx_caseA/triton_cache
bash /shared_nfs/yihou_agentx_caseA/scripts/glm52_leg_spur_mtp.sh
