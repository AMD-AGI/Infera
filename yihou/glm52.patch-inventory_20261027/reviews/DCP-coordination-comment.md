Our GLM-5.2 PD audit found that the local split-indexer series (D14/D15/D18/D24/D30/D31) overlaps the local-top-k/all-gather/global-merge direction here. We will coordinate here rather than file a competing implementation.

Historical isolated `benchrun T15/T16` found a context-length crossover: versus the replicated indexer, step time was **0.5–1.5 ms slower at 34K**, **2.6–4.2 ms faster at 150K**, and **4.4/6.6/9.2 ms faster at 200K for N=16/24/32**. The fused merge was about 46 us/layer. The archived numerical gate passed 40/40, but these are results on that historical branch, not tests of your current head. Please retain short-context regression checks as well as long-context performance coverage.

We opened separate, draft current-main candidates for pieces outside this PR's transport scope:
- #46824: PD relayout for a DCP-sharded draft, with DP/PP host-byte fan-out regressions. Requires agreement on the final shared draft-layout policy from #44322/this work.
- #46826: constrained DCP1-prefill to ROCm-DCP-decode configuration gate; must follow the backend and transfer contract.
- #46828: default-off BF16 kv_b retention for replicated-Q projection; needs final-stack accuracy/memory/performance validation.

The local D26 change only suppresses an idle-rank `not supported forward_mode` warning in ROCm DCP prepare; it is a logging refinement to consider alongside backend enablement, not a separate performance optimization. We have not filed it as another standalone PR.

Known limits: DP2×DCP2 graph outputs in the archived stack were still garbled and unbisected. Historical serving throughput used simulated MTP acceptance; no stack throughput is claimed as evidence for these patches. The October 27 PP-overlap/custom-all-reduce bisect also prevents using later combined-stack speed as patch-correctness evidence.
