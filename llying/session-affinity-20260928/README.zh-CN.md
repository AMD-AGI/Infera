# 会话亲和实现与验证

本版本实现单 Router 会话绑定；首次选择沿用 Infera 原评分，不是完整 Dynamo 成本模型复现。P weight=20，R1保留；权重40实验不运行。

- `INFERA_SESSION_AFFINITY=off|prefill|both`，默认 off。prefill 只绑定 PD 的 P；both 分别绑定 P/D，也覆盖聚合 worker。
- `INFERA_SESSION_AFFINITY_TTL_SECS=3600`，支持1..86400秒。活跃请求持有租约，最后一个租约结束后开始空闲计时。
- 显式接收 `X-Dynamo-Session-ID`；未提供时使用原策略。不把请求体的 prompt_cache_key 自动当作 session，不改变模型输入。
- 单 Router 内存表按模型、session、角色隔离，最多65536条；过期清理在后续请求中触发。表满且无法清理时使用正常选择并记录告警，不驱逐活跃绑定。
- 目标在可用候选中消失或登记信息改变会重选；请求失败使对应绑定失效。旧失败回调带代数检查，不能删除新绑定。不增加主动拥塞迁移，也不重放已经输出的请求。
- `/metrics` 提供按角色的 `infera_router_session_active`、`infera_router_session_hits_total`、`infera_router_session_selected_total`，日志记录命中/新建/过期/目标失效。不会输出 session ID。

离线验证已通过285项单元、26项HTTP功能、4项额外集成、14项render probe测试；4项原有外部服务测试未运行。证据见 offline-tests.txt。实机smoke待完成后补充。

实机工作目录：`/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928`。节点allocation31999；P=n10-29，D=n03-33；两端HiCache关闭、有效chunk=4K；使用相同已验证引擎镜像。不会释放用户allocation。
