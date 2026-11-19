#!/usr/bin/env bash
# U12 diagnosis after v034-u12 failed (|diff| 8.3e-4 vs track A's 7.6e-6): the probe over four input settings, then
# track A's original function-level test (scripts/kf/test_overlay_logits.py, which passed on 11-18) on the branch code.
# key: git -C /home/liyingli/bench_agentx/upstream-20261119/wt/u12-gfx950-gluon-mqa-logits rev-parse HEAD
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/checks/u12_diff_probe.py
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready v034 || exit 1
A=/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52/llying/vllm-decode-trackA-20261118/scripts/kf
EXTRA_DOCKER="-v $A:/kf:ro" run_vllm up-u12-probe "$IMG_V034" "$(wt u12-gfx950-gluon-mqa-logits)" v0.34.0 \
    "python3 /checks/u12_diff_probe.py 2>&1 | grep -v Warning | tee /out/probe.jsonl;
     echo '== track A test_overlay_logits.py on the branch code'; cd /kf && python3 test_overlay_logits.py 2>&1 | grep -E 'batch|graph|PASS|FAIL' | tail -40"
