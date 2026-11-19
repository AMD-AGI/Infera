# PR U24（Infera）：Rust 路由的上游连接池空闲超时短于引擎的 keep-alive

- 目标：AMD-AGI/Infera main（`962845a65`）。分支 `pr/router-pool-idle`（一个提交 `905c649e8`，liyingli@amd.com，DCO），fork `llying-001/Infera` 同名分支。
- 提交方式（用户 11-19 定）：先在 fork 里开 PR，明天 review 后再开到 AMD-AGI/Infera。
  **fork PR：[llying-001/Infera#2](https://github.com/llying-001/Infera/pull/2)**（11-19 16:20 开，base = fork 的 main = AMD-AGI `962845a65`）。
- 查重（11-19）：AMD-AGI/Infera 的 open PR 里没有 keep-alive / 连接池相关的。
- 与硬件无关。
- 状态：代码 + 测试完成；Rust CI、pre-commit 通过；J3（P4D4 3600 s，去掉 keep-alive 绕法）全程 0 次路由错误。
- 证据强度（如实写进正文）：机制复现只在 keep-alive 1 s 时出现 2 / 7,545 次错误，在 uvicorn 默认的 5 s 下本机复现不出；主要证据是实际部署（s10 / s11 的错误、J3 去掉绕法后 0 次），都是单次。

## 验证数据（都是单次）

| 项 | 结果 | 出处 |
|---|---|---|
| Rust CI | fmt / clippy 干净；370 通过 / main 369（新增 `pool_idle_defaults_below_uvicorn_keep_alive`） | J1b `infera-rust`（作业 298969） |
| Python CI | 3,054 / 3,054（本 PR 不改 Python） | J1b `infera-py` |
| pre-commit | 通过 | J5 `infera-precommit` |
| 修前的部署现象 | s11：AgentX 预热中一条根请求 502（`decode ... unreachable: error sending request`），整轮中止；s10 路由记了 2 次 5xx；之后用两边引擎 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120` 绕开（s12b 起） | 主线 P4D4 README 1k、CANDIDATES U24 |
| 本 PR 的部署（J3：P4D4 C104，路由用本补丁，**不加** keep-alive 绕法，uvicorn 默认 5 s） | 路由 "error sending request" 0、"unreachable" 0、5xx 0，覆盖 15 min 预热 + 3600 s | J3（作业 298999，v3-191），`results/pr-298999/pr-checks.txt` |
| 本机机制检查（uvicorn + 按路由同样方式构建的 reqwest，64 并发，客户端 1 个工作线程、每个响应后占 10 ms CPU，间隔落在 keep-alive 附近） | keep-alive 1 s：旧（池空闲 90 s）2 错误 / 7,545 请求，新（0.8 s）0 / 7,548；keep-alive 5 s：旧 0 / 3,857，新（4 s）0 / 3,858；客户端空闲的变体各设置都是 0 错误 | J1b `u24-race-busy` / `u24-race` |

## PR description (English, Infera template)

**Title:** rust router: drop idle upstream connections before the engine's keep-alive does

# Description

reqwest keeps an idle pooled connection for 90 s, while uvicorn (the API server of vLLM and SGLang) closes an idle
keep-alive connection after 5 s. A request the router writes to a connection that the engine is closing at that moment
fails with "error sending request" before the engine sees it, and the router answers 502. On a GLM-5.2 PD deployment
(AgentX agentic traffic) this showed up as occasional decode-leg 502s; when one hit a root request, the AgentX warm-up
aborted the whole run. We had been working around it by raising the engines' keep-alive
(`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`).

This sets the upstream client's pool idle timeout below uvicorn's default keep-alive, so the router drops an idle
connection before the engine can.

## Type of change

- [x] Bug fix (non-breaking change which fixes an issue)

## Changes

- `rust/router/src/proxy.rs`: the upstream `reqwest` client uses `pool_idle_timeout(INFERA_UPSTREAM_POOL_IDLE_S)`,
  default 4 s (below uvicorn's 5 s default keep-alive); invalid values fall back to the default.

## Validation

- `cargo fmt --check`, `cargo clippy --locked --all-targets -- -D warnings`, `cargo test --locked`: clean, 370 passed
  (main 369). New: `proxy::tests::pool_idle_defaults_below_uvicorn_keep_alive` (default, override, invalid values). The
  race itself is timing dependent, so there is no deterministic unit test for it.
- `pre-commit run --files rust/router/src/proxy.rs` passes.
- End to end (one run): GLM-5.2 PD on 8x MI355X, AgentX C104, 15 min warm-up + 3600 s, router with this change and the
  engines on uvicorn's default keep-alive (no workaround): 0 "error sending request", 0 "unreachable", 0 5xx in the
  router log. Before, without the workaround, runs logged 2 5xx (one run) and aborted on a root-request 502 (another).
- Loopback check (uvicorn app + a reqwest client built like the router's, 64 concurrent requests, client on one busy
  worker thread, request gaps around the keep-alive): with a 1 s keep-alive, 2 errors in 7,545 requests with a 90 s pool
  idle timeout vs 0 in 7,548 with 0.8 s; with uvicorn's 5 s default neither setting produced errors in this setup, so the
  production evidence above is the main support.

Notes: this does not add a retry for a connection that was closed before the write; that could be a follow-up. AI
assistance was used to write this change and description; the submitter reviewed it.

# Checklist:

- [x] The functionality is complete
- [x] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [x] My changes generate no new warnings
- [x] I have added tests that prove my fix is effective or that my feature works
- [x] New and existing unit tests pass locally with my changes
