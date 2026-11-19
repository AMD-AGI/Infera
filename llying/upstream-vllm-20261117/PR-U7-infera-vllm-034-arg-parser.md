# PR U7（Infera）：vLLM ≥ 0.34 的 `make_arg_parser` 挪位后 vLLM worker 起不来

- 编号：CANDIDATES U7（Infera 适配 v0.34）的第一项。
- 目标：AMD-AGI/Infera main（`962845a65`）。分支 `pr/vllm-034-arg-parser`（一个提交 `907933373`，liyingli@amd.com，DCO），fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 16:08 改）：Infera 是公司内部仓库，直接在 AMD-AGI/Infera 开 draft PR。
  **draft PR：[AMD-AGI/Infera#251](https://github.com/AMD-AGI/Infera/pull/251)**（11-19 16:15 开，分支 `llying/vllm-034-arg-parser`，标题改成 conventional 风格；之前在 fork 开的 llying-001/Infera#6 已关闭）。
- 查重（11-19）：AMD-AGI/Infera 的 open PR 里没有（搜了 make_arg_parser / cli_args / vllm 0.34 / arg parser）。
- 与 U28（`pr/vllm-dp-engine-lb`）都改 `infera/engine/vllm/args.py`，先合一个之后另一个要 rebase。
- 状态：代码 + 测试完成；Python CI、vLLM 0.34 容器内测试、pre-commit 通过；我们所有 vLLM 0.34 上的 P4D4 运行都带着这个改动。

## 验证数据

| 项 | 结果 | 出处 |
|---|---|---|
| vLLM 0.34.0 容器内（`vllm/vllm-openai-rocm:v0.34.0`） | `tests/engine/vllm/test_disagg_allow_tcp_args.py`：Infera main 3 / 3 失败（`ModuleNotFoundError: No module named 'vllm.entrypoints.openai.cli_args'`），本分支 3 / 3 通过；新增 `tests/unit/engine/test_vllm_server_arg_parser.py` 2 / 2 通过 | J1b `infera-v034`（作业 298969，单次） |
| Python CI（3.10，无 vLLM） | PR 3,056 通过 / main 3,054（新增 2 个），无新增失败 | J1b `infera-py` |
| pre-commit | 通过 | J5 `infera-precommit` |
| 部署 | 11-17 起所有 vLLM 0.34 上的 P4D4 运行（Infera 实验镜像）都用这个改动 | 主线 P4D4 README |

## PR description (English, Infera template)

**Title:** fix(vllm): parse vllm serve args with vLLM >= 0.34

# Description

vLLM 0.34 moved `make_arg_parser` from `vllm.entrypoints.openai.cli_args` to `vllm.entrypoints.common.cli_args` and
expects its `FlexibleArgumentParser`, so Infera's vLLM worker launcher dies with an import error before it starts
`vllm serve`:

```
ModuleNotFoundError: No module named 'vllm.entrypoints.openai.cli_args'
```

`_vllm_server_arg_parser()` imports the new location first and falls back to the old one for earlier releases.

## Type of change

- [x] Bug fix (non-breaking change which fixes an issue)

## Changes

- `infera/engine/vllm/args.py`: `_vllm_server_arg_parser()` builds the parser from `vllm.entrypoints.common.cli_args`
  with vLLM's `FlexibleArgumentParser` when available, else from `vllm.entrypoints.openai.cli_args` with `argparse`.

## Validation

- In the `vllm/vllm-openai-rocm:v0.34.0` container, `pytest tests/engine/vllm/test_disagg_allow_tcp_args.py`: 3 failed on
  main (the `ModuleNotFoundError` above), 3 passed with this change.
- New `tests/unit/engine/test_vllm_server_arg_parser.py` (stub modules for both layouts, runs without vLLM): 2 passed,
  in that container and in the Python 3.10 CI environment (3,056 passed, main 3,054, no new failures).
- `pre-commit run --files <changed files>` passes.
- Our GLM-5.2 PD deployments on vLLM 0.34 have run with this change since 2026-11-17.

Notes: `infera/engine/vllm/args.py` is also touched by the `INFERA_VLLM_DP_ENGINE_LB` change; whichever lands second
needs a rebase. AI assistance was used to write this change and description.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
