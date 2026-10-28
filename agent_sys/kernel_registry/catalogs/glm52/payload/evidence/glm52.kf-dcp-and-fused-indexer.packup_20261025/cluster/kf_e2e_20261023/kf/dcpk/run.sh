#!/usr/bin/env bash
# kernel-level A/B of the DCP KF candidates in the v0522 image (compute-node-05): each task driver run with the
# pristine kernel file and with the candidate (correctness + CUDA-graph bench).
K=REDACTED_LOCAL_PATH D=$K/kf/dcpk
OVL=REDACTED_LOCAL_PATH
docker run --rm --name historical-container --network host --ipc host --shm-size 32g --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined -v $K:$K -v $OVL:REDACTED_LOCAL_PATH \
  -e HIP_VISIBLE_DEVICES=0 -e SGLANG_USE_AITER=1 --entrypoint bash historical-sglang-image-a -c \
  "for t in dcp_sparse dcp_indexer; do for v in prist kf; do echo \"=== \$t \$v\"; cd $D/\$t/\$v && python3 driver.py 2>&1 | grep -E 'case_snr|^SNR|allclose|Error|Traceback'; python3 driver.py --bench-mode 2>&1 | grep -E 'case_ms|median_ms|Error|Traceback'; done; done" > $K/dcpk.log 2>&1
echo "exit $?" >> $K/dcpk.log
