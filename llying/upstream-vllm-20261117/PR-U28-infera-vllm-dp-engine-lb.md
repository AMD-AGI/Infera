# PR U28（Infera）：`INFERA_VLLM_DP_ENGINE_LB=1` 让 vLLM 自己选 DP rank

- 编号：之前叫"DP 引擎均衡"，没有编号，11-19 记为 U28（已补进 `CANDIDATES.zh-CN.md`）。
- 目标：AMD-AGI/Infera main（`962845a65`）。分支 `pr/vllm-dp-engine-lb`（一个提交 `4885756d5`，liyingli@amd.com，DCO），fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，明天 review 后再开到 AMD-AGI/Infera。
  **fork PR：[llying-001/Infera#5](https://github.com/llying-001/Infera/pull/5)**（11-19 16:20 开，base = fork 的 main = AMD-AGI `962845a65`）。
- 查重（11-19）：没有重复。用户自己的 [AMD-AGI/Infera#196](https://github.com/AMD-AGI/Infera/pull/196)（每个 DP rank 绑定自己的 KV 事件端口）
  是另一回事。
- 与 U7（`pr/vllm-034-arg-parser`）都改 `infera/engine/vllm/args.py`，先合一个之后另一个要 rebase。
- 默认关，只在设了环境变量时生效。
- 状态：代码 + 测试完成；Python CI、vLLM 0.34 容器内测试、pre-commit 通过。证据有好有坏（见下），都是单次、节点不同，正文如实写。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| Python CI（3.10） | PR 3,058 通过 / main 3,054（新增 4 个），无新增失败 | J1b `infera-py`（作业 298969） |
| vLLM 0.34.0 容器内 | `tests/unit/engine/test_vllm_dp_identity.py` 4 / 4 通过 | J1b `infera-v034` |
| pre-commit | 通过 | J5 `infera-precommit` |
| 不开（m10，v3-006，3600 s） | 最忙 rank / 平均在跑数 p50 1.28、p90 1.52；60% 的时间里总有一个 rank KV > 95%，最空的 rank 中位 78% | 主线 P4D4 README 1q（m10 分析） |
| 开（f1，同一台 v3-006，3600 s，decode 显存目标 0.93） | 最忙 / 平均 p50 1.19、p90 1.40；72% 的采样有 rank KV > 95%，最空中位 0.78；48,173 tok/s/GPU 对 m10 48,572（−0.8%，但每 rank KV 少约 4%） | 主线 `results/f1/dp_balance.txt`、P4D4 README 1t |
| 开（f2，v3-124，3600 s，0.93） | 1.19 / 1.39；78% 的采样有 rank KV > 95%；44,691 | `results/f2/dp_balance.txt` |
| 开（m14，v3-124，1800 s，0.95） | 49,183 对 m10 前 1800 s 46,083（节点不同）；TTFT p90 8.61 对 10.97 s，ITL p90 17.9 对 20.6 ms，交互 p90 55.7 对 48.5 | P4D4 README 1s |
| 开（J3，v3-191，3600 s，0.95） | 1.19 / 1.40；71% 的采样有 rank KV > 95%；43,269（新节点，不归因） | `results/pr-298999/dp_balance.txt` |

- 结论：在跑请求数的均衡确实变好（1.28 → 1.19、1.52 → 1.40）；KV 占满的比例没有变好（decode 仍受 KV 容量限制）；吞吐结论不明
  （同节点 −0.8% 但 KV 少 4%，另一台节点前 1800 s +6.7%）。所以做成默认关的开关。

## PR description (English, Infera template)

**Title:** engine (vllm): INFERA_VLLM_DP_ENGINE_LB=1 lets vLLM pick the DP rank

# Description

A vLLM DP engine registers its `dp_size`, so the router fans it out into one target per rank and pins every request to a
rank with a header. For a PD decode without prefix caching the router's choice has no cache to aim at, and its booking
by input tokens does not follow output growth or request completion. On an 8-GPU GLM-5.2 PD deployment (decode DP4,
AgentX traffic) the busiest rank ran 1.28x the mean number of requests (p90 1.52x), and 60% of the time one rank's KV
was over 95% full while the least loaded rank sat at a median 78%.

With `INFERA_VLLM_DP_ENGINE_LB=1` the worker registers as a single target; the router sends no rank header and vLLM's
own DP load balancer picks a rank per request from the ranks' live waiting / running counts. Off by default.

## Type of change

- [x] New feature (non-breaking change which adds functionality)

## Changes

- `infera/engine/vllm/args.py`: when `INFERA_VLLM_DP_ENGINE_LB=1` (exact value), a DP engine registers no DP identity
  (one worker instead of `dp_size` rank targets).

## Validation

- Python unit tests (`pytest -m "not slow and not integration"`, Python 3.10): 3,058 passed (main 3,054), no new
  failures. New `tests/unit/engine/test_vllm_dp_identity.py` (4 cases: DP engine registers size and rank; single rank
  registers none; with the switch a DP engine registers as one worker; the switch needs the exact value); they also pass
  inside the `vllm/vllm-openai-rocm:v0.34.0` container.
- `pre-commit run --files <changed files>` passes.
- GLM-5.2 PD on 8x MI355X (prefill PP4, decode DP4), AgentX C104, single runs:

  | | busiest / mean running (p50 / p90) | samples with a rank > 95% KV | tok/s/GPU |
  |---|---|---|---|
  | off, 3600 s | 1.28 / 1.52 | 60% | 48,572 |
  | on, same node, 3600 s, decode `--gpu-memory-utilization 0.93` (~4% less KV) | 1.19 / 1.40 | 72% | 48,173 |
  | on, another node, first 1800 s | — | — | 49,183 (off on the first node: 46,083) |

  Requests spread more evenly across ranks and p90 latency improved on the 1800 s run (TTFT 8.61 vs 10.97 s, ITL 17.9 vs
  20.6 ms), but the decode stays KV-bound and the throughput effect is within the node-to-node spread we see (~7-10%), so
  this stays opt-in.

Notes: this touches `infera/engine/vllm/args.py`, like the vLLM >= 0.34 arg-parser fix; whichever lands second needs a
rebase. AI assistance was used to write this change and description; the submitter reviewed it.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
