# PR U15（Infera）：Python 路由的 kv-aware 策略把分词 / 哈希移出事件循环

- 目标：AMD-AGI/Infera main（`962845a65`）。分支 `pr/kv-aware-prepare`（一个提交 `20c4ba4c0`，liyingli@amd.com，DCO），fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，明天 review 后再开到 AMD-AGI/Infera。
  **fork PR：[llying-001/Infera#4](https://github.com/llying-001/Infera/pull/4)**（11-19 16:20 开，base = fork 的 main = AMD-AGI `962845a65`）。
- 查重（11-19）：没有重复。
- 与硬件无关。
- 状态：代码 + 测试完成；Python CI、pre-commit 通过。P4D4 从 s10 起改用 Rust 路由（分词在 Rust 里），**本 PR 没有在部署里做过前后对比**，正文如实写。
- 限制（正文写明）：单线程池让事件循环不再被堵，但分词本身仍是串行的；多线程需要每线程一份 tokenizer（共享 tokenizer 不能多线程同时用，
  另见 U21 的 tokenizers 并发崩溃嫌疑）。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| Python CI（3.10） | PR 3,057 通过 / main 3,054（新增 3 个），无新增失败 | J1b `infera-py`（作业 298969） |
| pre-commit | 通过 | J5 `infera-precommit` |
| 修前的部署现象（s8） | 路由进程单核 100%，路由 `/metrics` 60 s 不响应（prefill 的 `/metrics` 毫秒级）；py-spy 20 s：87% 在 `tokenizer.encode`（`KvEventAwarePolicy.pick_targets` → `BlockHasher.hash_for` → `token_ids_for`）；s7b 里路由测得的 22.8 s "prefill 段" 主要是路由自己的排队 | 主线 P4D4 README 1i、`results/s8-router-profile/` |

## PR description (English, Infera template)

**Title:** router (python): tokenize and hash for kv-aware routing off the event loop

# Description

`KvEventAwarePolicy.pick()` tokenized the whole prompt on the event loop (`BlockHasher` -> `tokenizer.encode`). For
agentic conversations of ~100K tokens that is 100-200 ms per request, during which the router serves nothing else. On a
GLM-5.2 PD deployment under AgentX traffic the router process sat at 100% of one core, its `/metrics` did not answer for
60 s, a 20 s py-spy profile showed 87% of the time in `tokenizer.encode`, and most of the time to first token was
queueing inside the router.

## Type of change

- [x] Bug fix (non-breaking change which fixes an issue)

## Changes

- `Policy.prepare()`: a new async hook that the routers await (`prepare_pick`) right before `pick()`, in the same task.
- `KvEventAwarePolicy.prepare()` tokenizes and hashes the request for every candidate's hash key on a one-thread
  executor (the tokenizer is shared and not safe to use from several threads) and leaves the hashes in a `ContextVar`
  keyed by the request object; `pick()` reuses them. Nothing is added to the forwarded body, and a `pick()` without
  `prepare()` hashes as before.
- `SessionAffinityPolicy` forwards `prepare()` to its inner policy; `disagg.py` and `mixed.py` call `prepare_pick`.

## Validation

- Python unit tests (`pytest -m "not slow and not integration"`, Python 3.10): 3,057 passed (main 3,054), no new
  failures. New in `tests/unit/router/test_kv_event_aware_policy.py`: `prepare` hashes off the event-loop thread and
  `pick` does not hash again; the prepared hashes only apply to the same request object; `pick` without `prepare` is
  unchanged.
- `pre-commit run --files <changed files>` passes.
- Not yet measured end to end: after finding this we moved that deployment to the Rust router (which tokenizes in
  Rust), so there is no before/after serving run with this change.

Limitation: the event loop is no longer blocked, but tokenization itself stays serial on one thread. Running it on
several threads would need one tokenizer copy per thread.

AI assistance was used to write this change and description; the submitter reviewed it.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
