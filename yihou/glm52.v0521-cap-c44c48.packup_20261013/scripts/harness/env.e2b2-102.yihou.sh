# Source before EVERY run.v2.yihou.sh call for round e2b2 on 102: E2b with DEBUG_AGENT=0 (legacy IPC
# still on BOTH legs), new sub-30000 default ports (no stride override).
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2b2
export EXPECTED_IMAGE_ID=sha256:3f917e9bd052717538aafab7959bc1c7d109edc7867c2adb35e90fe552aaf503
export CONFIG=$H/config.yihou.E2b.sh
export YIHOU_OVERRIDES="DEBUG_AGENT=0"
