# Source before EVERY run.v2.yihou.sh call for round e1a on 136 (plain ssh, /shared_nfs ro there ->
# NODE_RW_ROOT=/mnt/m2m_nobackup/yihou_v0521dbg by default). Sub-30000 port defaults (R015).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-136 YIHOU_IP=10.245.154.168 YIHOU_JOB= YIHOU_ROUND=e1a
export EXPECTED_IMAGE_ID=sha256:a3a450255b5fe8b70dee5ed0515f6d1bb8b3b525760fa299f7dd91e02e55c31d
export CONFIG=$H/config.yihou.E1-relaxed.sh
export YIHOU_OVERRIDES="DEBUG_AGENT=0"   # main 06:0x: agent off until the IPC matrix settles
