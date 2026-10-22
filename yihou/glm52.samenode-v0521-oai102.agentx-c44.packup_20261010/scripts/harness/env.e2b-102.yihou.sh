# Source before EVERY run.v2.yihou.sh call for round e2b on 102 (up/census/audit/smoke/bench/stop).
# Port overrides: phase-1 c44e router+etcd are still Up on 102 holding 38100/24501/24502.
export H=/shared_nfs/yihou_glm52/v0521dbg_20261010/harness
export YIHOU_NODE=crsuse2-m2m-102 YIHOU_IP=10.245.147.130 YIHOU_JOB=202546 YIHOU_ROUND=e2b
export EXPECTED_IMAGE_ID=sha256:3f917e9bd052717538aafab7959bc1c7d109edc7867c2adb35e90fe552aaf503
export CONFIG=$H/config.yihou.E2b.sh
# ENGINE_PORT_STRIDE=22000 from decode a4 on: decode 61101 (derived block 61334-61344, outside the
# ephemeral range); a1-a3 died because prefill held a random listener on 39595 (decode's old block).
export YIHOU_OVERRIDES="ETCD_PORT=24601 ETCD_PEER_PORT=24602 ROUTER_PORT=38200 ENGINE_PORT_STRIDE=22000"
