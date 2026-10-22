# 时间线（UTC）

## 10-21

- 17:32 用户：在 1P1D C88 上做 decode 侧负载均衡实验（参照 Dynamo），分到节点后先复现 1P1D C88，再在其上 A/B；顺带 review PR 210；
  周期巡检。用户 9 小时后回来 review。
- 17:39 作业 34439（AUS，QOS batch，可被抢占）提前开始：P = n02-29（10.235.192.61），D = n04-33（10.235.192.139）。campaign 的 2P1D
  模板改为空（`../2p1d-aus-v0521-20261019/operations/queue.next.2p1d-pmtp.tsv` 保留原模板）。
- 17:41–17:48 两台节点各自构建 PR 210 镜像（`aab4cc36`）：base 拉取约 1 分钟，Mooncake / infera / Rust router 层都成功，约 7 分钟。
- 17:47 decode 节点上发现 cyao 遗留的 7 个 ATOM 容器（`cyao-pr4045-k3-*`，14:38 起，占 8 卡各 263–288 GB 显存、约 1.8 TB 内存）；
  该节点由 Slurm 独占分给 34439。17:49 `docker stop`（未删除，容器与日志保留），显存归零。详见 `../2p1d-aus-v0521-20261019/operations/STATUS.zh-CN.md`。
- 17:20–17:55 router 改动（分支 `feat/decode-dp-request-balance`，`a8cf30bf`）：`--decode-request-weight`。在 base 镜像里编译并测试
  （`scripts/build_router.sh`）：Rust 354 + 46 + 4 + 14 通过、0 失败（含新增 3 个），`cargo fmt --check` 通过；Python 启动器测试 25 通过，
  ruff 通过。
- 权重取值：AUS 2P1D C176 两轮 decode 日志（v0.5.21，R2 开）拟合 `每步 ms ≈ 23 + 0.8·最忙 rank 请求数 + 4.2~4.9·最忙 rank 上下文(M)
  + 0.17~0.19·总请求 + 0.24~0.25·总上下文(M)`，R² 0.93–0.94；最忙 rank 多 1 个请求 ≈ 多 17–19 万 token → 先用 180,000。
  R2 下最忙 / 均值：中位 1.42、p90 1.67（C176）。v0.5.22 C88 的系数在复现轮的日志上重拟，必要时改 B 组权重。
- 17:56 `scripts/kit_prep.sh`：两台镜像就绪、kit 端口空闲、显存干净；prefill 节点 WORK_DIR 预置 InferenceX `918524ff` 与 AgentX 数据集（1.8 GB）。
- 17:57 队列开始：`kit_up.sh RUN=k6`（kit 的 `engine/up.sh` 原样）→ etcd、prefill0（GPU 4–7）、decode（经 srun 替身在 D 上启动）。
  `scripts/kit_patrol.sh` 巡检（容器、三个 /health、日志致命行、AgentX 进度）与原 `patrol.sh`（作业 / 队列）同时运行。
- 18:08 起栈完成（11 分钟）：**PR 210 kit 从分支本身以 `PREFILL_WORKERS=1` 跑通**（PR 的待办项之一）。router 参数与 PR 210 一致
  （`--decode-input-demand on`、guard completion、重叠权重 20 / 2、prefill 会话亲和）。18:08 `k6-repro` 开始：flush 成功，AgentX 载入 393 条 trace。
- **18:29:41 作业第 7 次被抢占**（`PreemptTime 18:29:41`、`EndTime 18:34:41`：Slurm 先给所有 step 发 SIGTERM，留 5 分钟宽限再结束作业
  并重新排队；本作业的 extern step 18:30:10 结束）。容器守护在 18:29:41 删除了两台节点上的全部容器；`k6-repro` 当时在预热
  （845 / 973），无结果。这次分配只有 51 分钟。
- 18:35 **调整顺序**（两天 7 次抢占，最短 51 分钟；原计划整套约 5.5 小时）：A/B 优先，部署时 prefill 就带 EAGLE 参数（kit 的
  `PREFILL_EXTRA_ARGS`，`kit_up.sh MTP=1`，不改 kit 文件，省去中途重启 prefill），同一部署上 w0 → w180k（prefill 开 MTP 后 decode
  不漂移，AUS 2P1D 已验证）；之后重新部署 kit 原样做 PR 210 C88 的严格复现。`kit_round.sh` 增加核对：B 组 router 必须是新二进制且
  `INFERA_DECODE_REQUEST_WEIGHT` 生效；`kit_prep.sh WAIT_VRAM=60` 在重新部署前等显存释放。
- 18:35 作业重新排队，Slurm 预计 10-22 00:57；**20:33:50 提前开始**（ROUND=a7，P = n04-33、D = n06-25，两台无外部容器、显存为 0）。
  20:39–20:43 n06-25 构建镜像（4 分钟；n04-33 已有）。20:43–20:50 `kit_up.sh RUN=ka7 MTP=1`：**prefill 带 EAGLE 参数在 v0.5.22 上正常
  起栈**（7 分钟，JIT 缓存在节点上）；prefill 两次 KV 分配：主 122.57 GB（2,758,976 token / rank）+ 草稿 1.82 GB。日志里的
  `post-warmup freeze_gc failed`（uvicorn 起来 2 ms 内发 POST 被拒）是 sglang 自身的无害时序竞争，进程继续。巡检在起栈期间误报三个
  /health（`current_run` 现在起栈一开始就写）→ 起栈 / 重启 prefill 期间放 `.starting` 标记，巡检跳过 /health。
- 20:50 `ka7-pmtp-w0` 开始（router = 新二进制，权重 0，核对通过）。预热进度与 `k6-repro` 逐点相同（开头 93 个冷请求约 14 分钟才陆续
  返回，之后才放出后续轮次），prefill MTP 没有拖慢预热。**21:04:36 作业第 8 次被抢占**（这次只有 31 分钟），无结果。
- 21:10 再调整（分配时长 51 / 31 分钟，而标准一轮至少 1.6 小时）：新策略（w180k）先跑、w0 后跑；A/B 两组测量窗口 1800 s
  （预热仍为每 lane 10 个），一轮约 1 小时；被抢占时 `kit_round.sh` 仍按已覆盖的测量窗口做 decode 均衡 / 步长分析（日志实时在登录节点上）。
  decode 每 rank 均衡统计与节点无关，A、B 即使落在不同分配上，机制层面也可比。严格复现（3600 s）放最后。
- 21:09 重新排队，**23:34:48 再次开始**（ROUND=a8，P = n02-29、D = n04-33，与 a6 同一对节点，两台已有镜像与 JIT 缓存）。
  23:35–23:41 `kit_up.sh RUN=ka8 MTP=1`（6 分钟；prefill 日志的 Traceback 又是无害的 `freeze_gc` 竞争）。
- 23:41 `ka8-pmtp-w180k` 开始（新 router，权重 180,000，核对通过），00:06:13 进入测量。**测量 12 分钟时的实时统计**：请求 2.7 req/s
  （PR 210 C88：2.75）、ITL p50 / p95 12 / 14 ms、interactivity p50 86（PR 210：49.1）、TTFT p50 4.7 s（4.9）。decode 每 rank 平均只有
  3.76 个在跑（PR 210 C88：6.7），最忙中位 6 / p90 8，每步中位 41 ms——**吞吐相同，交互接近翻倍，主要来自 prefill 带 MTP**（decode 草稿层
  不再读未初始化 KV，每步变快 → 同等请求速率下在跑数更少 → 又更快）；router 权重的作用要等同部署的 w0 对照。同期 router 视角的每 rank
  在途请求 6–11（共 66），比 decode 实际在跑（约 30）多一倍：一半以上的在途请求还在 prefill / 传输阶段，按在途请求数均衡 ≠ 按 decode
  当前在跑数均衡。
