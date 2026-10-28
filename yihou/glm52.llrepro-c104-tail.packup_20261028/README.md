# GLM-5.2 PD 4+4 (135 -> 136): last runs on the C104 stack, status and next plan at machine release (2026-10-28)

Follows `yihou/glm52.llrepro-garblefree-sweep.packup_20261028`. Stack = llr37 (llying 4+4 kit + KF decode kernels + K3
EAGLE 3/1/4 + draft FP8 + 2 aiter BF16 GEMM csvs + decode mem 0.95 + prefill PP comm overlap + decode custom AR, C104).
AgentX 3600 s, A1, SIMULATED acceptance 2.99 (no correctness claim; user 10-28: performance phase ignores correctness).

| run | change vs llr37 | score | vs C104 mean | qps | step p50 ms | verdict |
|---|---|---|---|---|---|---|
| llr37 / llr37r / **llr37r2** | C104 x3 | 48,713 / 48,103 / **47,795** | mean **48,204** (±1 %) | 3.19-3.21 | 42.4 | base |
| llr50 | decode `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (ATOM/Dynamo P9) | 48,010 | -0.4 % | 3.209 | 42.8 | neutral (activation is debug-logged only) |
| **llr51** | decode mem 0.95 -> 0.96 | **48,806** (best single run) | +1.2 % | 3.208 | 42.2 | candidate: qps flat, ISL +1.5 %, prealloc-wait 0.58 -> 0.51; NOT replicated |
| llr52 | decode DP controller blocking poll instead of busy-poll (P18, `dpc_poll/`) | 47,402 | -1.7 % | 3.163 | 42.3 | neutral (controller left the top-CPU list) |

llr53 (prefill mem 0.91) was stopped at start and llr51r (replicate of llr51) not run: machines released to the user
2026-10-28 06:45Z. Site files for both are in `sites/`. Next plan: `OPT_LEDGER.md` section F.
