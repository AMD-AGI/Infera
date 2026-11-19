# PR U14（Infera）：infera server 启动时把 `RLIMIT_NOFILE` 软上限提到硬上限

- 目标：AMD-AGI/Infera main（`962845a65`）。分支 `pr/server-nofile`（一个提交 `34b4b1bc0`，liyingli@amd.com，DCO），fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 16:08 改）：Infera 是公司内部仓库，直接在 AMD-AGI/Infera 开 draft PR。
  **draft PR：[AMD-AGI/Infera#248](https://github.com/AMD-AGI/Infera/pull/248)**（11-19 16:15 开，分支 `llying/server-nofile`，标题改成 conventional 风格；之前在 fork 开的 llying-001/Infera#3 已关闭）。
- 查重（11-19）：没有重复。相关的是用户自己的 [AMD-AGI/Infera#239](https://github.com/AMD-AGI/Infera/pull/239)（draft，在 engine worker 里提
  `RLIMIT_MEMLOCK`，改 `infera/engine/memlock.py` 等），文件不重叠，思路相同；正文里提一句。
- 与硬件无关。
- 状态：代码 + 测试完成；Python CI、pre-commit 通过。部署里一直用 docker `--ulimit nofile=1048576` 绕开（s7b 起），**本 PR 本身没有在部署里单独验证过**，正文如实写。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| Python CI（3.10） | PR 3,058 通过 / main 3,054（新增 4 个），无新增失败 | J1b `infera-py`（作业 298969） |
| pre-commit | 通过 | J5 `infera-precommit` |
| 修前的部署现象 | s7：预热第 18 min 路由 `OSError: [Errno 24] Too many open files` 2,048 次（`socket.accept()` 失败），随后 4 个请求同时 502 `prefill leg failed`（含根请求），AIPerf 中止；s5 的 502 同因。路由容器默认软上限 1024 | 主线 P4D4 README 1g、`results/s7-abort/` |
| 连接数 | s8 中路由描述符在 1,150–1,470 之间波动（AIPerf 长连接 + 每请求的 P / D 连接），不是泄漏 | 主线 P4D4 README 1i |

## PR description (English, Infera template)

**Title:** fix(server): raise the RLIMIT_NOFILE soft limit at start-up

# Description

Every proxied PD request holds a client, a prefill and a decode connection, so a container's default soft limit of 1024
open files runs out at a few hundred concurrent requests: `accept()` fails with EMFILE and the router answers 502. On a
GLM-5.2 PD deployment under AgentX traffic the router logged `OSError: [Errno 24] Too many open files` 2,048 times in
the warm-up, then four requests (one of them a root request) got `502 prefill leg failed` and the run aborted. The
router held 1,150-1,470 descriptors at peak under that load (not a leak). We had been working around it with
`docker run --ulimit nofile=1048576`.

This lifts the soft limit to the hard limit (1M when the hard limit is unlimited) before the backend is chosen, so the
Python router and the Rust router (which inherits it across exec) both get it. The same idea as #239 for
`RLIMIT_MEMLOCK` in the engine workers.

## Type of change

- [x] Bug fix (non-breaking change which fixes an issue)

## Changes

- `infera/server/__main__.py`: raise `RLIMIT_NOFILE`'s soft limit to the hard limit (capped at 1M when unlimited) at
  start-up; a refused `setrlimit` is logged and start-up continues.

## Validation

- Python unit tests (`pytest -m "not slow and not integration"`, Python 3.10): 3,058 passed (main 3,054), no new
  failures. New `tests/unit/server/test_nofile_limit.py`: raised to the hard limit, unlimited hard limit, already at the
  limit, `setrlimit` failure.
- `pre-commit run --files <changed files>` passes.
- Our deployments used `--ulimit nofile=1048576` (which raises both limits) since the failure above, so this change has
  not been exercised end to end on its own; it removes the need for that flag.

AI assistance was used to write this change and description.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
