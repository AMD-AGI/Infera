# 时间线（UTC，11-03 起）

- 13:20 接手（用户：在 AUS 两台节点上验证，再做后续优化）。读 `HANDOFF-20261103`、`REPRODUCE`。持有作业 35060（n06-25 / n06-33）到 22:43。
- 13:23 包拷到 `/perf_apps/liyingli/bench_agentx/vfix-20261103`，写 `site.local.env`；两台并行构建镜像（n06-33 上误起两次，停掉后重跑）。
- 13:29 两台镜像就绪，11/11 基线一致。13:30 验证队列项入 holder；preflight 通过，13:31 unit 26/26。
- 13:31–16:12 B1 N1a A1 B2 A2 B3 C0 C1 C2 N0 N1（每组约 15 分钟，显存约 1 分钟释放）。
- 16:12 B 生产；16:33–17:01 lm-eval（InferenceX + aiperf 子模块克隆约 15 分钟，GitHub 约 100 KB/s）。
- 16:26 用户问 B1 的错题；16:39 用户：查清是否我们的问题、记录并提交、开巡检、必要时自主申请节点、持续推进约 10 小时。
- 16:40 参照组 `ref-mxfp4-stock`（n06-33 GPU 0–3）、`ref-fp8-stock`（n06-25 GPU 0–3，挂载修正后 16:46 重起）；
  后续 holder 35103（`afterany:35060`）；巡检 `scripts/patrol.sh` 每 10 分钟。
- 16:50 router 按 fdbafb60 + 565461b3 重编（单元测试全过）；优化队列 30–33 入 holder（预测见 README 第 5 节）。
- 17:03 C 生产。17:04 `ref-mxfp4-fp8kv`、之后 `ref-mxfp4-stock-r2`（n06-33）。
- 17:16 参照组结论：错题是模型本身（RESULT 第 2 节）。
- 17:30 r2（原版原样重跑）0.966：同一配置两次跨 0.966–0.973，错题差异是 batch 不确定性下的临界题翻转。C 生产 0.969、0 / 0。
- 17:35 C 生产 lm-eval 0.9674 / 0.9697。17:37 起 scoreB。17:40 修法对到 SGLang main 1ea5f8c6（三类 variant + 单元测试 11/11），见 upstream/；未提 PR。
- 17:50 trace 项（C + packet capture 关，两次 40 步 decode profile）入队，排在 E10 之后（用 35060 最后约 50 分钟）。
- 19:08 scoreB 51,477（与 crusoe E10 51,374 同水平）；19:09 起 scoreC。
- 20:36 scoreC 53,609（同对 B +4.1%，新最高）；judge 0 FAIL。20:36 起 E10 同对基线（holder 项 30）。
- 22:05 E10 同对基线 50,797（B +1.3%、C +5.5%）；traceC 推迟到下一分配。22:16 镜像存为 /perf_apps/liyingli/images/infera-sglang-v0522-fdbafb60.tar（65 GB），opt_item.sh 在缺镜像时先从它加载。
- 23:36 traceC 完成：verify 前空闲 0.59 → 0.03 ms/步，剩余最多 1.9 ms/步（一次 ~30 ms 的结果 event 等待 + DP 同步 gather）。23:36 起 Cdpbal。
- 00:20 trace 复核：DP0 的 ~34–47 ms 空闲是 profile 启动后的第一步（采集开销）；去掉后每步空闲 0.52–0.83 ms，graph 间空闲方向不再投入。
