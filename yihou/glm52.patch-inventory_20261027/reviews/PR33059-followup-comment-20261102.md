Rebased onto main `acd6f5570`, including merged #46861, and pushed signed `ea88d74bf`. The production delta and regression are unchanged; **12 CPU tests pass** again and all applicable PR-scoped pre-commit hooks pass.

The old Qwen3-8B MI35x PD job's 0.01 GSM8K accuracy matches the shared fused-KV allocation regression investigated in #46861. Its ROCm allocation workaround is now in this branch. This is a concrete upstream baseline repair, not a claim that the new-head AMD job has already passed.

The DeepSeek-V3.2 MI300/MI355 DP accuracy and stage-c matrix requested by @HaiShaw is a separate remaining validation request. The historical GLM checks and CPU Aiter stub do not cover that matrix. Keeping the PR draft and updating its body accordingly.
