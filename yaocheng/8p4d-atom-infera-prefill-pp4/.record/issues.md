# 问题记录

每个问题记录现象、日志片段、原因、处理、状态。时间均为 UTC。

## 1. 两台节点的 GPU 被已结束作业的容器占用；停止后容器被自动删除

- 现象（2026-10-09 02:47）：n05-21 与 n05-33 的 16 张 GPU 显存为每卡 254-274 GiB，`gpu_busy_percent` 除 n05-33 的一张为 100% 外均为 0%。占用者为两台节点上各一个容器 `primus-training-32841`（镜像 `unifiedtrainingdockers.azurecr.io/utd/ci:primus_ci_3979e53_20260930`，2026-10-08 18:05:29 启动），运行 clairlee 的 8 节点 Qwen3-235B 预训练（`NNODES=8`，master 为 n01-21，`--train_iters 10`），已运行约 8 小时 40 分钟。
- 原因：所属 slurm 作业 32841（clairlee）已于 2026-10-08 22:05:37 以 `TIMEOUT` 结束，docker 容器不受 slurm 分配约束，留在节点上继续运行。
- 处理：用户授权在两台节点上 `docker stop`（不删除）。依次在 n05-21（02:55:43）与 n05-33（02:56:03）执行 `docker stop primus-training-32841`。
- 结果：docker events 为 `kill`（SIGTERM）、10 s 后 `kill`（SIGKILL）、`stop`、`die`、`destroy`，`destroy` 与 `die` 在同一秒（n05-21 02:55:57，n05-33 02:56:17），之后 `docker inspect` 返回 `no such object`。该容器以自动删除（`--rm`）方式启动，停止即被删除，无法再用 `docker start` 恢复；停止前没有检查这一属性。两台节点的 16 张 GPU 显存在停止后约 30 s 内降到 0。
- 状态：已处理。容器已删除，所属作业此前已结束。

## 2. 实验环境与上一组实验（DPA8 prefill）的差异

- 驱动：两台节点均为 amdgpu 6.16.13、ionic 26.07.9.001、host libionic 1.1.39。上一组正式测试（`../8p4d-atom-infera/results/`）在 amdgpu 6.14.14、ionic 26.03.3、libionic 1.1.54 的节点上完成。6.19.x 节点上 RDMA 注册的 KV 显存被重复计入（`../8p4d-atom-infera/.record/issues.md` 第 10 条），6.16.13 是否有同样的问题需要实测。
- 验证（03:12，`acc-c48`）：decode 以 0.94 启动并完成 KV 分配、RDMA 注册与 cudagraph 捕获，初始化后每卡显存约 273.6 GB；prefill 以 0.85 启动正常。6.16.13 下没有出现 6.19.x 的问题，显存比例按 recipe 取值。
- 状态：已验证。驱动、ionic 与 libionic 版本仍与上一组正式测试不同，比较结果时需要注明。

## 3. C48 预热在 03:32 后停滞，prefill0 于 03:42 退出

- 现象（`pp4-c48`，作业 32947）：C48 预热 03:25:31 开始，目标 531 个请求。返回数逐步上升（03:31 这一分钟完成 121 个），03:31:59 完成最后一个请求，03:32:03 起一直是 `returned=367/531 | sent=407 | in_flight=40 | errors=0`，到 04:29 没有变化。aiperf 日志中没有错误、超时或取消。
- 停滞前的请求（`aiperf_artifacts/profile_export.jsonl`，367 条，均为 1 token 输出的预热请求）：最后 15 个请求的延迟为 1.7-4.8 s，ISL 为 1.4 万至 69 万 token；全部 367 个请求的延迟 p50 31.1 s、p90 89.2 s、最大 139.1 s，ISL p50 12.2 万、p90 51.6 万、最大 70.0 万。停滞前没有延迟逐步变长的过程。
- 显存采样（`.tmp/logs/vram-n0533.log`）：prefill0 所在的 GPU 0-3 在 03:31:19-03:42:27 期间显存完全不变（260.9/261.4/261.3/249.5 GB），03:42:27-03:42:37 降到 290 MiB，即 prefill0 退出，距停滞开始约 600 s。prefill1 所在的 GPU 4-7 此后保持 253-265 GB；decode（`vram-n0521.log`）保持约 273.7 GB。
- 分析（未经日志确认）：
  - prefill0 的显存在停滞期间不变，约 600 s 后退出，与 PP stage 间通信挂起、在 NCCL/gloo 默认超时后退出的情况相符。
  - 停滞后 prefill1 上的请求同样没有返回，客户端也没有再发出请求。如果只有 prefill0 挂起，round-robin 分到 prefill1 的请求应能继续完成，因此卡住的环节可能还包括 decode 或路由，例如 decode 等待 prefill0 某个 stage 的 KV 写入时阻塞了后续请求的接收。
  - 上一组 C144 预热的停滞（`../8p4d-atom-infera/.record/issues.md` 第 14 条）表现相同：返回数突然停止、引擎没有退出、错误为 0，原因同样没有查明。
- 候选原因（源码核对，未经日志确认）：upstream `68e0df5`（09-25，fix(offload): fence dense LMCache saves）为 dense LMCache 保存加入了等待 KV 写完的栅栏，并说明 PP 下调度器的 `advance_on_schedule`（`scheduler.py` 第 616 行，PP 时为真，在前向之前推进 chunked prefill 进度）会破坏该栅栏依赖的前提；同一组修改还处理了 save 失败后请求停在 `_save_retry_blocked`、没有回收路径的情况。本镜像的 `d9f0720e` 没有这些修改。该问题更可能造成写入 CPU 缓存的 KV 不完整，是否导致挂起需要日志确认。
- 处理：`up.sh` 增加 `follow`，引擎与路由的日志实时写入运行目录，下次停滞或抢占时日志不会丢失。
- 状态：未解决。引擎日志在节点上的容器内，作业被抢占后无法读取（第 4 条）。需要 `docker logs` 保存 prefill0（已退出）、prefill1、decode、router 的日志后才能定位。

## 4. 作业 32947 被抢占，套件容器留在节点上

- 现象：04:27 SSH 到 n05-33、随后 n05-21 均返回 `Access denied by pam_slurm_adopt: you have no active jobs on this node`。`scontrol show job 32947`：`PreemptTime=2026-10-09T04:27:15`，`EndTime=04:32:15`（5 分钟宽限期），`Requeue=1`。宽限期内已不能 SSH。
- 抢占方：sirafati 的作业 32780（QOS batch，优先级 6016，本作业 2234），排定 04:32:15 在 n05-21 启动；n05-33 排给 xiaompen 的作业 32950（n05-[29,33]，07:27）。
- 影响：
  - 没能执行 `down.sh`，sweep 进程与显存采样随作业结束；C48 没有结果，C80、C120 没有开始。
  - 留在节点上的本套件容器：n05-21 的 `glm52-8p4d-pp4-{decode,router,etcd,agentx}`（decode 占用 GPU 0-3，每卡约 273 GB）；n05-33 的 `glm52-8p4d-pp4-prefill1`（GPU 4-7，每卡 253-265 GB，LMCache 锁页内存 640 GiB）与已退出的 `glm52-8p4d-pp4-prefill0`。AgentX 客户端容器带 `--rm`，aiperf 退出后会自动删除。
- 状态：未解决。需要在这两台节点上有权限的人（节点上的下一个作业的用户或管理员）先 `docker logs --timestamps <容器名>` 保存日志，再 `docker rm -f` 删除上述容器。

## 5. 作业每次重新运行约 30 分钟后即可被抢占

- 现象：作业 32947 第一次运行 02:28:30-04:32:37（约 2 小时，n05-[21,33]），第二次 04:34:57-05:10:37（36 分钟，n02-[25,29]），都以 `PREEMPTED` 结束。第二次在可抢占时刻（启动后 30 分钟）约 1 分钟后收到信号，抢占方为 radwived 的作业 32953（QOS `dcgpu-prod-r`，8 节点）。
- 影响：每次重新运行可能分到不同节点，需要重新检查节点、构建镜像、冷启动（无编译缓存时约 8.5 分钟）。单档 AgentX（预热约 10-20 分钟、测量 3600 s、指标收集与聚合约 20 分钟）需要约 80-90 分钟不被打断。
- 处理：`build_image.sh` 改为两台节点并行构建（约 2 分钟，替代约 4-7 分钟的镜像传输）；`up.sh` 的实时日志保证抢占时诊断信息不丢失。
- 第三次运行 05:24:20-06:16:37（52 分钟，n02-[25,29]），C48 测量进行到 16 分 47 秒时收到信号，抢占方为 xiaoqunw 的作业 32965（QOS `dcgpu-prod`）。三次运行时长为 2 小时 4 分、36 分、52 分。
- 状态：阻塞。QOS `batch` 可被 `dcgpu-*` 等 QOS 抢占（`../8p4d-atom-infera/.record/issues.md` 第 9 条），本账号可用的 QOS 均可被抢占。

## 6. 第三次抢占后套件容器留在 n02-[25,29]，占用作业 32965 的节点

- 现象：06:11:37 左右收到抢占信号后，作业进程（sweep、AgentX 的 docker 客户端与 `tee`、`docker logs -f`、显存采样）随即结束，节点不能 SSH。容器由 dockerd 管理，不受影响：06:17:55 AgentX 客户端容器仍在发送请求（`done=2,130, err=0`）。
- 留在节点上的容器：n02-25 的 `glm52-8p4d-pp4-{agentx,router,decode,etcd}`（decode 占用 GPU 0-3，每卡约 273.7 GB）；n02-29 的 `glm52-8p4d-pp4-prefill0`、`glm52-8p4d-pp4-prefill1`（8 张 GPU 每卡 254-266 GB，LMCache 锁页内存 1280 GiB）。这两台节点自 06:16:46 起属于作业 32965（xiaoqunw，16 小时）。
- 说明：AgentX 客户端容器带 `--rm`，按 3600 s 的测量时长约 06:55 停止发送，聚合完成后会把 `agentx_conc48.json` 写入 `.tmp/runs/pp4-n02-c48/agentx/` 并自动删除；引擎、路由与 etcd 容器会一直运行。06:16:46 之后的测量与作业 32965 共用节点，数据有效性需要另行判断。
- 状态：未解决。需要作业 32965 的用户或管理员用 `docker logs --timestamps <容器名>` 保存日志后，用 `docker rm -f` 删除上述容器。
