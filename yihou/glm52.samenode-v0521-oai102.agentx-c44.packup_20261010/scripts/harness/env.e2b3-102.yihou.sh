# Round e2b3 on 102 (R020): E2b config, plain v0521 1009 image, single change vs e2b2 = IPC mode:
# --pid host + default (dma-buf) IPC on both engines, legacy OFF, debug agent OFF. Sub-30000 ports.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2b3
export EXPECTED_IMAGE_ID=sha256:3f917e9bd052717538aafab7959bc1c7d109edc7867c2adb35e90fe552aaf503
export CONFIG=$H/config.yihou.E2b.sh
export YIHOU_OVERRIDES="ENGINE_PID_HOST=1 IPC_MODE_LEGACY=0 DEBUG_AGENT=0"
