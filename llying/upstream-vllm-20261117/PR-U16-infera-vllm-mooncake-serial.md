# PR U16（Infera）：vLLM Mooncake 串行下发（Python + Rust 路由，`INFERA_VLLM_MOONCAKE_SERIAL=1`）

- 目标：AMD-AGI/Infera main（`962845a65`，11-19 未变）。分支 `pr/vllm-mooncake-serial`（一个提交 `561b7feb7`，作者 liyingli@amd.com，已签 DCO），
  在仓库外 `/home/liyingli/bench_agentx/upstream-20261119/infera`，fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，明天用户 review，review 后再开到 AMD-AGI/Infera。
  **fork PR：[llying-001/Infera#1](https://github.com/llying-001/Infera/pull/1)**（11-19 16:20 开，base = fork 的 main = AMD-AGI `962845a65`）。
- 查重（11-19）：AMD-AGI/Infera 的 open PR 里没有串行下发相关的。
- 与硬件无关（路由逻辑）。
- 状态：代码 + 测试完成；Python / Rust CI、pre-commit 都通过；P4D4 自 s10 起每轮都用这条路径。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| Python CI（3.10，`pytest -m "not slow and not integration"` 同 CI） | PR 3,055 通过 / main 3,054，无新增失败 | J1b `infera-py`（作业 298969，v3-006） |
| Rust CI（`cargo fmt --check`、`clippy --locked --all-targets -D warnings`、`cargo test --locked`） | fmt / clippy 干净；370 通过 / main 369 | J1b `infera-rust`（首轮抓到多余的 `mut` 和一处格式，已修） |
| pre-commit（ruff format / check 等） | 通过 | J5 `infera-precommit`（作业 299267） |
| 现象（不串行时） | s2b：prefill 运行 0、排队 76、KV 99.7%、prompt 吞吐 0，AIPerf 返回数停住超过 1 min（decode 侧指标因多 API server 不可靠，机制是推断） | 主线 P4D4 README 1c |
| 串行之后 | s4 起所有轮（缓存感知准入 + 串行）没有再停住；用 Rust 串行路径的 P4D4 计分：s10 36,265、s13 42,724、m10 48,572（3600 s）、J3 43,269 tok/s/GPU | 主线 P4D4 README、`results/pr-298999/` |

- 开关目前是环境变量（与 Python 路由已有的开关一致），上游可能更希望做成 CLI 参数，PR 正文里提了。

## PR description (English, Infera template)

**Title:** router: serial vLLM Mooncake legs (INFERA_VLLM_MOONCAKE_SERIAL=1)

# Description

With vLLM's MooncakeConnector the router posts the prefill and decode legs of a PD request concurrently. The decode
reserves KV for a request as soon as its leg arrives, while a prefill that reorders its queue (cache-aware admission,
priority) can hold finished KV that the decode has not reserved yet. Under memory pressure each side then waits on the
other and the deployment stalls: on a GLM-5.2 PD deployment (8x MI355X, prefill PP4 with cache-aware admission, decode
DP4) the prefill sat at 0 running / 76 waiting / 99.7% KV with zero prompt throughput, and client responses stopped
for over a minute.

`INFERA_VLLM_MOONCAKE_SERIAL=1` posts the decode leg only after the prefill has answered, in both routers. The Python
router already had a serial-pull dispatcher; the Rust router only had a serial path for ATOM.

## Type of change

- [x] Bug fix (non-breaking change which fixes an issue)
- [x] New feature (non-breaking change which adds functionality)

## Changes

- Python: `VllmMooncakeProtocol.topology` becomes `serial-pull` when the variable is set (the existing serial
  dispatcher; the decode needs no handoff from the prefill reply, the transfer room is in both bodies).
- Rust: `vllm_serial_pull` posts the prefill, waits for its reply, then runs the decode leg through `serial_decode`,
  shared with ATOM's serial pull; a failed prefill is returned to the client and never reaches the decode.
- Default (unset) behavior is unchanged: concurrent legs.

## Validation

- Python unit tests (`pytest -m "not slow and not integration"`, Python 3.10): 3,055 passed (main 3,054), no new
  failures. New: the protocol switches topology and pairs the transfer ids.
- Rust (`cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`): clean, 370
  passed (main 369). New: `disagg::tests::vllm_serial_pull_posts_the_decode_after_the_prefill_reply` checks the decode
  is not posted while the prefill is running, and is never posted after a failed prefill.
- `pre-commit run --files <changed files>` passes.
- End to end: every GLM-5.2 PD run since we enabled it (cache-aware admission on the prefill, Rust router with this
  path, AgentX C104) completed without a stall, e.g. 48,572 tok/s/GPU over 3600 s (single runs).

Notes: the switch is an environment variable, like the Python router's existing one; happy to turn it into a router CLI
flag instead. AI assistance was used to write this change and description; the submitter reviewed it.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
