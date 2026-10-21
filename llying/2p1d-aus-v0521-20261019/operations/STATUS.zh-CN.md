# 过程记录（UTC）

## 2026-10-19

- 09:13 资源：Slurm 34389（`salloc -N 2 --exclusive`，到 10-20 02:45），smci355-ccs-aus-n02-25 / n02-29，16 张 GPU 空闲，无其他容器。
- 09:52 拉基础镜像：节点上的 Docker Hub 凭据过期（"personal access token is expired"），改用空的 `DOCKER_CONFIG` 匿名拉取，09:54 完成，
  digest `sha256:6c2c0b0e…` 与 crsuse2 相同。
- 09:57 n02-25 上开始建镜像（`build/build_images.sh`，日志 `logs/build-images.log`）：Mooncake faae8dd4 + #4340、libionic 54.0-187、
  v0521 DSA 补丁在 bytecode 中校验通过。
- 09:58–10:05 运行目录：3P1D 的 `scripts/`、`patches/`、`config/config.3p1d.sh` 复制到 `/perf_apps/.../2p1d-aus-v0521-20261019`，
  改节点、路径、拓扑（`config/topology.2p1d.aus.tsv`）、端口段；`round.b1.sh`；客户端 HF 数据集缓存从
  `p8d8-tracing-aus-20260922/cache/agentx/hf/hub` 复制到 n02-25 的 `/scratch/liyingli_2p1d/cache/25/agentx/hf/`。
- 10:07 镜像完成：`infera-sglang:v0521-llying-destpin-aus1019` = `sha256:1eb737ab…`，两台 ID 一致（整个镜像步骤约 11 分钟）。
  DRY_RUN 在两台节点上渲染的命令核对无误。
- 10:08 队列第 1 步（bringup）第一次启动：preflight 通过；prefill-0 权重从 `/perf_apps` 读入用了 417 s。10:17:57–10:18:09
  prefill-0 的 DP1–3 三个 scheduler 在 `scheduler.py:1878`（读 `schedule_stream.cuda_stream`）段错误，原生栈
  `c10::hip::HIPStream::stream()`，日志无其他错误；DP0 存活但卡住，此后在 DP0 所在 GPU 上新开的进程（包括全新容器）一建 stream 就段错误，
  其余空闲 GPU 正常。10:27 停掉队列与容器，16 张卡逐卡测试（`scripts/gpu_stream_check.sh`）全部正常 → 硬件无问题，是起栈过程把 GPU 弄坏。
- 原因判断：3P1D 配置沿用了 crsuse2 的 `MOONCAKE_DISABLE_HIP_DMABUF=0`，而 AUS 历来用 1（`../b4-align-crsuse2-20261008/README.zh-CN.md`：
  两个集群网卡驱动不同）；段错误恰在内存注册之后。改为 1 后 10:29 重新起栈（失败那次的日志在运行目录 `runtime/failed/`）。
- 10:53 起栈完成（prefill-1 权重走页缓存 35 s，decode 冷读 `/perf_apps`），census 0 致命项；未再出现段错误 → DMABUF 判断成立。
  已知答案门禁：简单集 48/48；困难集 37 正确 + 11 检索错位 + 0 编造（crsuse2 r2 为 38 + 10 + 0），正确性与 crsuse2 一致。
  11:01 decode 切到模拟接受 3.61；传输冒烟 192/192、0 失败、8 个 decode rank 都有 batch。
- 11:02–12:12 预热轮 warm-b1（C176，30 分钟窗口，不计分）：24,958 tok/s/GPU，输出 216.0，P50 / P90 交互 41.8 / 37.5，TTFT p50 / p90 8.0 / 46.4 s；
  GPU 命中 54.7% + 主机 37.3%；两个 prefill 均衡（排队 5.5 / 6.3、缓存份额 92.2% / 92.4%，t5b 为 2.7 / 8.1）；decode 每 rank 在跑 8.7、
  KV 0.73、等 KV 传输 7.85/rank（t5b 6.95）。重启后首轮 decode 偏慢 + 窗口短，只作参考。
- 12:13 基线 b1-wt C176 开始。**13:04:06 prefill-0 GPU 内存访问错误退出**（"Memory access fault by GPU node-3 … on address 0xfffffffff000"，
  之前日志无异常，只有 aiter 未调优 GEMM 形状提示），随后 decode 侧成串传输失败；13:14 起 prefill-1 有 8 个请求停住（py-spy：scheduler 在正常
  事件循环里，不是 3P1D prefill-2 的 `synchronize` 挂起签名）。13:19 AgentX 因失败请求 10.07% > 10% 中止，**b1-wt 结果无效**（部分数据 16,606 不可用）。
  这是 prefill-0（`HSA_NO_SCRATCH_RECLAIM=0` 的那个实例），根因未查。
- 13:19 队列按设计停下；此后无人处理（我这边在等待中，没有及时发现）。
- **15:18–15:23 Slurm 作业 34389 被抢占（sacct: PREEMPTED）**，两台节点已不可登录（pam_slurm_adopt 拒绝），容器是否被清理无法确认。
  原始数据在 `/perf_apps/.../2p1d-aus-v0521-20261019/runtime/runs/b1/`；结果副本在本目录 `results/`。
- 16:20 用户决定：按原命令重新申请 2 个节点；两个 prefill 都改 `HSA_NO_SCRATCH_RECLAIM=1`（`config/round.b1n.sh`，ROUND=b1n）。
  `salloc` 排队时被我的 60 s 超时取消（作业 34438），改用同样资源的 `sbatch --wrap "sleep infinity"`：作业 34439，排队中，预计 20:58 开始。
- 16:45 warm-b1 与 t5b 的对比（用户问为什么差这么多）：
  - 吞吐差 13.6%（24,958 vs 28,900）几乎全部来自平均输入更短：105.1k vs 122.6k（−14.3%）；每秒完成请求 3.77 vs 3.74（持平）。
    30 分钟窗口紧接预热段，会话还在较早的轮次、上下文短；吞吐 99% 是输入 token，所以请求数不变、tok/s 也跟着降。3P1D 的
    warm-mem092b C144（30 分钟窗口）同样：输入 102.8k vs 正式轮 119.9k，吞吐 22,596 vs 25,269。
  - 另一个真实差异：同等负载（最忙 rank 在跑数相同）下 decode 每步比 t5b 慢约 20%（@8 / @10 / @12 / @14：50.1 / 63.1 / 77.9 / 80.6 ms
    vs 40.7 / 52.1 / 63.6 / 66.1）；第二轮 b1-wt 崩溃前 17 分钟为 44.0 / 62.5 / 71.6 / 76.5（仍慢 8%–20%）。输出 216 vs 230.7 tok/s/GPU，
    ITL p50 23.9 vs 18.3 ms。一部分是重启后首轮偏慢，其余原因未定（AUS 节点 / 驱动 6.16.13，或 `MOONCAKE_DISABLE_HIP_DMABUF=1`
    下 KV 写入 decode 显存的路径不同），b1n 里按轮次对比每步耗时再判断。
  - prefill 侧与 t5b 相当：缓存份额 92.3% vs 93.6%，新算 30.8k vs 29.7k tok/s，两实例排队 5.5 / 6.3（t5b 2.7 / 8.1）。
  - 更正：warm-b1 发布的服务端表按 60 分钟窗口算（`config.3p1d.sh` 把 `AGENTX_DURATION` 写死为 3600，覆盖了预热传入的 1800），
    后半段是空闲，prefill 新算速率被减半（15.7k → 实际 30.8k）。已改为 `${AGENTX_DURATION:-3600}` 并按 11:37:30–12:08:00 重新生成。
- 16:59 巡检：登录节点后台 `scripts/patrol.sh`（每 2 分钟：作业状态、队列失败/驱动退出/队列空转、引擎与 router 健康（连续两次才报）、
  服务端日志致命模式、decode 传输失败、AgentX 日志停滞），问题写入运行目录 `runtime/patrol/ALERT.log`；我用 `scripts/wait_event.sh`
  等待告警、队列步骤推进或超时后检查。报告 `RESULT.zh-CN.md` 建立，`tune_report.py` 改为按实例列（2 个 prefill）并加 decode @14 列。
- 19:17 作业 34439 开始：P = smci355-ccs-aus-n02-29（10.235.192.61，上次的 decode 节点，镜像还在），D = smci355-ccs-aus-n03-33（10.235.192.56，
  8 个 ionic 均 ACTIVE，驱动 6.16.13）。两个小问题：n03-33 的 ssh 主机密钥不在 known_hosts（已接受）；n02-29 上 `/scratch/liyingli_2p1d/cache/29`
  是上次容器以 root 建的，数据集缓存目录建不了 → prepare 加了用容器 chown 的步骤，重跑。19:24 镜像拷到 n03-33，16 卡 stream 检查通过。
- 19:25 队列开始（ROUND=b1n）。
- 19:48 起栈在等 decode 健康时失败：对两台节点的 ssh 都被 pam_slurm_adopt 拒绝（作业的 extern step 19:48:44 结束）；`srun --jobid --overlap`
  仍可进节点，prefill-0/1 健康、decode 在首次 JIT 编译。我先按"extern step 意外结束"处理：写了 ssh 替身 `tools/sshshim/ssh`（每条命令一个
  srun step）、把日志跟随改到登录节点（`engine.sh` 写 `<log>.follow` 标记 + `scripts/follow_supervisor.sh` + `bench-harness/follow_remote.sh`；
  修了 `grep -F` 自匹配），队列驱动 / sampler / hang 监视器改为在登录节点运行。
- ~19:52 发现**作业 34439 被抢占并自动重新排队**（`Restarts=1`，状态 PENDING，预计 20:22 再开始）——19:48 的 ssh 被拒其实是抢占的开始。
  账户可用的 QOS 里 batch 已是最高（优先级 500，会被 perf / dcgpu-* / sponsor / shared-medium / special-group 抢占），没有更不容易被抢占的选项。
- 20:05 为应对反复抢占改成 `scripts/campaign.sh`（登录节点后台）：作业每次（重新）开始 → ROUND = a<重启次数> → `prepare_alloc.sh` →
  把 `operations/current_stage` 指向的配置装成 `config/round.<ROUND>.sh` → 用 `operations/queue.next.tsv` 模板生成队列并启动；作业离开
  RUNNING 时停掉本轮登录节点进程、写告警、等下一次开始。每个实验结束后我更新模板和 current_stage，重启时从计划所在位置继续。
  同时：起栈改为 `bringup_fast.sh`（decode 直接模拟接受 3.61，跳过已在 AUS 通过的已知答案门禁和只重启 decode 的切换，decode 日志为
  decode-0.log），预热窗口缩到 20 分钟；配置拆成 `config/stage-b1-wt.sh`（基线）与 `config/stage-b2-wb.sh`（write_back）。
- 20:05 作业 34439 重新开始（restart 1，ROUND=a1）：P = smci355-ccs-aus-n01-29（10.235.192.54），D = smci355-ccs-aus-n02-33（10.235.192.133），
  两台都没有镜像 → 重建。第一次 build 失败：n01-29 上用空的 DOCKER_CONFIG 时找不到 buildx 插件，旧 builder 不认 `--progress`
  → 去掉该参数；同时 prepare 加了"先记录新节点 ssh 主机密钥"（节点间 docker save | ssh load 用的是真 ssh）。20:08 campaign 重跑 prepare。
- 20:16 镜像重建完成（`sha256:f38645fb…`，同一 recipe 重建 ID 会变），16 卡检查通过。20:17 `start_queue.sh` 静默退出：`set -euo pipefail` 下
  `grep` 无匹配使命令替换失败 → 加 `|| true`。另：srun step 默认拿不到 GPU（rocm-smi 看不到卡，preflight 的显存检查形同虚设）→ 替身加 `--gres=gpu:8`。
- 20:18–20:59 a1 起栈（两台新节点冷读模型 + 首次 JIT，37 分钟），传输冒烟 192/192（p50 179 s，首次编译所致）。
- 20:59–21:56 warm-a1（20 分钟窗口）= 22,011 tok/s/GPU；窗口结束后的 12 条 decode 传输失败是取消，正常。
- 21:56 基线 a1-wt 开始；**22:19 prefill-0 挂起**（GPU 100%、scheduler 停在 `prefill.py:910`，与 3P1D 相同），两个 prefill 都已是
  `HSA_NO_SCRATCH_RECLAIM=1`。22:29 hang 监视器自动抓取：TP0/1/3 正在执行 RCCL `ncclDevKernel_Generic_1`，TP2 正在执行
  `quickreduce::allreduce_prototype_twoshot`（INT8）→ 不同后端互相等待的死锁（RESULT §2.4，证据 `evidence/hang-a1-prefill-0/`）。
  22:30 停掉本轮（中途我的一条 `pkill -f` 因模式出现在自身命令行里把自己的 shell 杀了，改用精确匹配重做）。
- 22:32 新增 `config/stage-b1-wt-qr0.sh`（prefill `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`），只重启两个 prefill（decode 保持预热），22:43 就绪
  （GPU KV 2.50M token/rank，比开 QuickReduce 时多 2.8%）；队列改为 a1-wt2 → HiCache 对照检查 → write_back。
- 22:43–00:20 **a1-wt2 = 25,695 tok/s/GPU**，12,439 完成、0 错误，全程无挂起（RESULT §2.5）。窗口结束时 25 条 decode 传输失败（取消）触发一次巡检告警，属正常。
- 00:20 队列继续：hicache_check（write_through 对照）；重启模板改为从 write_back 开始（`operations/current_stage` = `config/stage-b2-wb.sh`）。
- 00:21 HiCache 检查（write_through）通过；00:32–00:41 只重启 prefill 改 write_back；00:43 HiCache 检查（write_back）通过（RESULT §2.6）。
- 00:53–02:33 **a1-wb = 30,659 tok/s/GPU**（+19.3%，RESULT §2.7）。02:33 队列进入我预先插入的 C144。
- 02:38–03:20 与用户讨论：我补做了 C176 的瓶颈分析（RESULT §2.8）：prefill 只忙约 65% 却排队 10.9 s，49% 的忙碌时间被单个长冷请求独占（FCFS
  排队头阻塞），38% 的时间有 rank 空闲而另一 rank 排队 ≥ 5。理论上 C144 不会超过 C176、prefill_complete 在 C176 上约 +2%，对准瓶颈的是 prefill
  调度策略。用户决定：02:55 停掉 C144（第一次停时 agentx_bench 卡在退出 trap 里，kill -9 收掉）；先做 `shortest-prefill-first`（hrrn 为后备）；
  1P(TP8)1D 暂不做；已知答案门禁在所有优化完成后统一跑。另答复：关 QuickReduce 与 `--disable-custom-all-reduce` 的区别（后者 custom AR 和
  QuickReduce 都不建）。新增 `scripts/prefill_rank_analysis.py`（每 rank 负载、独占段、逐秒不均衡），结果存到各轮 results 目录。
- 03:13–03:23 只重启 prefill（`config/stage-b4-spf.sh`），server_info 确认 `schedule_policy=shortest-prefill-first`；03:23:51 a1-spf C176 开始。
  重启模板改为从 SPF 开始（`operations/current_stage` = `config/stage-b4-spf.sh`）。
- 16:25 登录节点上后台运行 `scripts/prepare_alloc.sh`（日志 `logs/prepare.34439.log`）：作业开始后自动换节点名（`scripts/set_nodes.sh`）、
  镜像（已有则复用/节点间拷贝，否则重建）、清理旧轮容器、数据集缓存、16 卡 stream 检查，然后启动队列（b1n：起栈 → 预热 → C176 基线）。
- 05:42–07:49 a1-spf C192 = 33,698（−4.7%，decode KV 被等传输请求占满，RESULT §2.10）。07:55–08:10 全栈重启为 prefill_complete + write_back + SPF，
  08:22 HiCache 检查通过；08:22–10:30 a1-pcwb C192 = 32,339（decode 冷启动 + KV 传输进关键路径，RESULT §2.11）；10:40 深入分析（RESULT §2.12）。
- 10:53 a1-pcwb-r4 C192（router `INFERA_R4_PREFILL_WORK=on`，只影响新会话首轮放置）开始；11:49 进入正式测量。
- **11:57 作业 34439 第三次被抢占**（测量开始 8 分钟后，AgentX rc=143，无结果），12:02 campaign 停掉本轮登录节点进程并告警，作业自动重新排队
  （Restarts=2）。14:50 查看：排队中，Slurm 预计 10-21 02:53 才开始（分区 14 台节点全被占用）。campaign / patrol / follower supervisor 仍在运行，
  作业开始后自动按 `operations/current_stage` + `operations/queue.next.tsv` 续跑。
- R4 的残缺数据：AgentX 预热阶段两实例新算量在开 / 不开 R4 时都均衡（9.6k / 9.8k vs 9.5k / 9.7k）；测量前 8 分钟太短，无法判断。
- 预热漂移排查进展：decode 启动后第一轮显存每卡涨约 3.2 GiB（273.9 → 277.1），之后几轮共再涨约 1.3 GiB 后走平，与每步变慢的幅度同步
  （pcwb 重启后同样 273.9 → 277.0），支持"显存分配器扩容"的猜测，待在线 profile 验证。

## 10-21（作业 34439 反复抢占；SPF + early 计划）

- 10-20 20:56 作业再次开始（Restarts=2 → ROUND=a2，n03-33 + n04-25）：没有匹配镜像 → 重建 10 分钟；21:07 起栈，21:32 launch 失败，
  21:33 作业又被抢占（起栈失败就是抢占开始所致）。
- 00:26 再次开始（a3，n02-25 + n02-33）：镜像重建；**n02-33 上 a1 的 decode 容器（`…-a1-decode-0`、`…-a1-decode-0-pcwb`）还在**，
  prepare 将其删除；n02-25 GPU1 的 stream 检查崩溃 → prepare 失败，campaign 按设计不在坏卡上跑、原地等待；之后作业又被抢占。
- 02:17 再次开始（a4，n01-33 + n10-29）：镜像重建；02:28–03:22 起栈（55 分钟：两台新节点冷读模型 + 首次 JIT），census 0 致命行、
  冒烟通过、传输冒烟 192/192（p50 190 s，首次编译所致）；03:26 进入 HiCache 检查，之后按计划跑 C176 两轮（抓冷 / 热 decode profile）→ R4。
- **容器泄漏**：抢占时 Slurm 只杀 step 进程，docker 容器不归 Slurm 管，会留在节点上继续占 GPU 和锁页内存。a1 的 decode 在 n02-33 上
  留了 12.5 小时。可能仍有泄漏的节点：n01-29（a1 的 prefill-0/1、router、etcd）、n03-33 / n04-25（a2 起栈中途的容器）——三台现在都分给了
  guanchen 的作业 34506，我们无权登录，无法确认或清理（需联系对方或管理员执行 `docker rm -f $(docker ps -aq --filter name=glm52-llying-2p1d-)`）。
- 03:28 新增 `scripts/guard_containers.sh`：作业每台节点上一个 srun step；Slurm 抢占 / 结束作业时先给所有 step 发 SIGTERM（KillWait = 30 s
  后才 SIGKILL，GraceTime = 0），守护收到后并行 `docker rm -f` 本节点所有 `glm52-llying-2p1d-*` 容器。a4 已手动启动（03:28 一个顺序删除版、
  03:30 一个并行删除版，运行中的旧版不能重启——杀掉它本身就会触发清理）。`campaign.sh` 改为每次准备完成后自动启动守护，并新增
  `RESUME_ROUND`（作业运行期间重启 campaign 时接管当前轮、不重做准备）；03:29 以 `RESUME_ROUND=a4` 重启 campaign。
  取舍：守护依赖登录节点上的 srun 进程，若它被杀，step 结束同样会删除容器（即停掉实验）。
- 03:38–05:15 a4-spf C176 = 33,793（decode 冷）；05:15–07:16 a4-spf2 C176 = 34,862（热，换节点复现 a1-spf 的 98.5%）。两轮各在测量开始 15 分钟时
  自动抓了 decode profile。07:16 a4-r4 开始，**07:31 作业第 5 次被抢占**：守护在 07:31:02 收到 SIGTERM，两台节点并行删除全部容器（prefill 容器
  锁页内存大，07:36 仍在退出中，进程已收到 SIGKILL）。
- 07:42 作业再次开始（a5，n02-21 + n04-33），镜像重建，07:53 起栈（旧配置 stage-b4-spf）。
- 07:45–08:30 冷 / 热 profile 分析（RESULT §2.13）：差距在 MTP 草稿层 indexer 的 `k_scatter`（每次调用冷 0.9–3.2 ms / 热 14–183 µs）；
  从镜像取出 `topk_transform.cuh`：等于阈值档的候选逐个全局原子追加，K 为 0 → 打分并列为 0 → 候选暴增。prefill 的 `speculative_algorithm=None`
  → 草稿层 KV / index-K 从未传给 decode。InferenceX 官方 recipe 注释描述的正是这个坑；规则要求模拟接受（草稿质量不计分），官方每次新起服务 = 冷态。
- 08:00 用户：先修冷热差，R4 暂缓，方案须符合 InferenceX 规则。08:30 改动：`engine.sh` 加 `PREFILL_MTP=1`（prefill 用 decode 同一组 EAGLE 参数），
  `agentx_env.py` 的部署核对随之认 prefill 的 MTP，新 stage `config/stage-b6-spf-pmtp.sh`；a5 队列第 2 步起改为 只重启 prefill → HiCache 检查 →
  a5-pmtp / a5-pmtp2 C176（各抓 profile）；`operations/current_stage` 与 `queue.next.tsv` 改为新配置直接起栈。
