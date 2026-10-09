# 实验记录

时间均为 UTC。套件说明见 [../README.md](../README.md)，问题见 [issues.md](issues.md)。
上一组实验（prefill TP8 + DPA）见 [`../../8p4d-atom-infera`](../../8p4d-atom-infera/README.md)。

## 2026-10-09

- 02:28:30 作业 32947 开始（Compute-DCPT，QOS batch，2 节点，至 2026-10-10 02:28:30，02:58:30 起可被抢占），节点 `smci355-ccs-aus-n05-[21,33]`。
- 02:44 用户要求：在这两台节点上按 `../8p4d-atom-infera/analysis-new/pcp_and_alternatives.md` 的替代方案把 prefill 改为 PP4，测 C48、C80、C120；新建本目录，复制最小的依赖并记录实验。
- 02:47 节点检查：
  - 两台均为 amdgpu 6.16.13（n05-21 在 09-24 为 6.14.14）、ionic 26.07.9.001、host libionic 1.1.39、内核 6.8.0-138；每卡显存 288 GiB，GTT 上限 1511 GiB，主机内存 3023 GB。
  - n05-21：fenic `10.235.192.138`，`ionic_0-7` ACTIVE，GID 1 为 `192.168.{1..8}.6`；本账号在 docker 组；已有镜像 `infera-atom:nightly_202609221542`（`sha256:aef823184e20…`，09-24 07:58 构建，包含 `../8p4d-atom-infera/patch/` 的 4 个 patch，patch 文件此后没有修改）。
  - n05-33：fenic `10.235.192.58`，`ionic_0-7` ACTIVE，GID 1 为 `192.168.{1..8}.8`；docker socket 为 `srw-rw-rw-`，不需要 sudo；没有 ATOM 镜像。
  - 两台的 `ionic_i` 与 rail 的对应一致（例如 `ionic_2` 在 rail 4，`ionic_3` 在 rail 3）。本套件端口空闲，临时端口范围 32768-60999。
  - 两台的 16 张 GPU 被已结束作业 32841 的容器占用（issues.md 第 1 条）。
- 02:48 源码与镜像核对（ATOM `d9f0720e2f99`，源码在 `../8p4d-atom-infera/.tmp/cache/ATOM`；upstream `c3a88b5c` 的文件在 `../8p4d-atom-infera/analysis-new/upstream_parallel_review/`）：
  - `d9f0720e` 的 `recipes/Agentic-GLM-5.2.md` 已有 1P1D PD：prefill TP1×PP4（`VLLM_PP_LAYER_PARTITION=20,20,20,18`）、mooncake producer 加 LMCache。
  - upstream `c3a88b5c`（10-08）新增的设置在本镜像中都可用：`--timeout-keep-alive`（`api_server.py`，默认 5 s）；`LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3`（镜像内 LMCache 0.5.5rc3 解析为 `[0, 1, 2, 3]`，ATOM 按 PP×TP=4 计算 worker 数）；`AITER_REUSE_IDENTICAL_COMM_GROUPS`（镜像内 aiter `dist/parallel_state.py`）。
  - PP 下 LMCache CPU 预算：`scale_cpu_size_for_pp` 把 `LMCACHE_MAX_LOCAL_CPU_SIZE × pp_size` 按层数分给各 stage，最后一个 stage 计入 MTP 层。
  - mooncake producer 各 stage 的握手端口为 `handshake_port + pp_rank×dp×tp + dp_rank×tp + tp_rank`，prefill 响应带 `remote_pp_size`；Infera `atom_mooncake` 协议把 prefill 响应的 `kv_transfer_params` 原样转给 decode，路由不需要修改。
  - `patch/atom-dcp-kv-staging.diff` 写出时使用原代码按 `_start_layer` 映射得到的 consumer 地址，PP stage 的层偏移不受影响。
  - DPA 与 PP 不能同时开启（`llm_engine.py` 第 95 行）。
  - 同一节点上的两个 PP 实例：stage 之间的 ZMQ 端点为 uuid 命名的 ipc 路径，LMCache lookup 的 ipc 路径在容器内的 `/tmp/vllm_rpc`，torch 分布式端口由 `get_open_port()` 和 Infera 自动分配，没有固定端口冲突。
  - InferenceX `918524ff` 多节点模式的 GPU 数为 `workers × TP × PP × PCP`，读取 `PREFILL_PP_SIZE`。
  - Infera 路由策略只有 `round-robin` 与 `kv-aware`。
- 02:52 用户决定：
  - 授权 `docker stop`（不删除）两台节点上的 `primus-training-32841`，写入 `.record`。
  - prefill 使用 2 个 TP1×PP4 实例（n05-33 的 8 卡），decode 为 TP4×DCP4（n05-21 的 4 卡），共 12 卡；Infera 路由保持 round-robin，不修改代码。
  - 6.16.13 驱动下 decode 如果不能以 0.94 启动，降到能启动的最高显存比例，继续完成正式测试并注明。
- 02:55:43-02:56:17 两台节点 `docker stop primus-training-32841`。该容器以自动删除方式启动，停止后被 docker 删除（issues.md 第 1 条）。16 张 GPU 显存约 30 s 内降到 0。
- 02:56:56-03:00:26 镜像经 `docker save | ssh n05-33 docker load` 从 n05-21 同步到 n05-33，两台的 image ID 均为 `sha256:aef823184e20…`（`.tmp/logs/sync_image.log`）。
- 02:57-03:03 新建本目录：`config.sh` 与 `scripts/` 由 `../8p4d-atom-infera` 复制后修改（见 README.md），`.tmp/` 放入 `gsm8k.sh`、`needle_probe.sh`、`vram_sample.sh`。脚本语法检查通过。
- 03:03:30 两台节点启动显存采样（`.tmp/logs/vram-n0521.log`、`vram-n0533.log`），冷启动精度验证服务 `up.sh CONC=48 MTP_AL= RUN_ID=acc-c48`（日志 `.tmp/logs/up-acc-c48.log`）。
- 03:12 服务就绪（冷启动约 8.5 分钟，三个容器的编译缓存均为空）。`/v1/workers` 为 1 个 decode（19002）与 2 个 prefill（19001 握手 21301，19003 握手 21321，均为 `tp_size=1, dp_size=1`）。
  - decode（n05-21，amdgpu 6.16.13，0.94）：`budget=270.71GB, peak_torch=106.69GB, non_torch=10.53-10.78GB, available_for_kv=147.06-147.31GB, block_bytes=778752, num_kvcache_blocks=202760-203105`，即每 rank 约 324 万 token，DCP4 合计约 1298 万 token（上一组 0.95、block 64 为约 1150 万）。non_torch 由上一组的约 26 GB 降到约 10.7 GB（`AITER_REUSE_IDENTICAL_COMM_GROUPS=1`）。初始化后每卡显存约 273.4-273.8 GB，余量约 14 GB。6.16.13 下 decode 以 0.94 启动正常，没有出现 6.19.x 的 RDMA 显存重复计入问题。
  - prefill（n05-33，0.85）：4 个 stage 的 `peak_torch` 为 99.3-124.4 GB，最紧的 stage（含 MTP 层与 lm_head）`available_for_kv=104.05GB`；各 stage 统一为 598675 个 block，每个实例约 958 万 token（上一组每个 DP rank 约 289 万）。初始化后每卡显存 231-250 GB。
  - LMCache：每个实例 4 个 stage 的 CPU 预算为 162.03/162.03/162.03/153.92 GB（共 640 GB），每个 stage 输出 `lmcache lookup server started`，scheduler 的 lookup client 连接 rank 0-3。n05-33 主机内存已用 1624 GiB。
  - Mooncake：prefill 每个 stage 自动发现 matched rails（prefill0 主网卡 `ionic_0-3`，prefill1 为 `ionic_4-7`），`role=PRODUCER`；decode 每个 rank 只用本卡 `ionic_0-3`，`protocol=rdma`。
- 03:12 `smoke.sh` 通过：completions 返回 " Paris. It is the largest city in France and is known for its iconic landmarks"，流式 chat（17×23）返回 "391"（`.tmp/logs/smoke-acc-c48.log`）。
- 03:13-03:18 精度验证（`.tmp/logs/acc-validate.log`）：GSM8K 5-shot 200 题 exact_match 0.985 ± 0.0086（flexible 与 strict 相同）；6 万 token 大海捞针 3/3。GSM8K 用时约 5 分钟，其中 03:15:04-03:18:04 decode 只有 1 个请求在生成（约 63 tok/s，约 1.1 万 token，`max_gen_toks=16384`），其余请求在 03:14 前完成；两个 prefill 实例的 prompt 吞吐约 2600-2900 tok/s，没有排队。
- 03:19:10 正式测试开始：`sweep.sh SWEEP_ID=pp4`（`POINTS="48 80 120"`，每档冷启动，3600 s，每 lane 预热 10，MTP K3、forced acceptance 2.99），日志 `.tmp/logs/sweep-pp4.log`，运行目录 `.tmp/runs/pp4-c{48,80,120}`。sweep 开始时的 `down.sh` 把精度验证服务的日志保存到 `.tmp/runs/acc-c48/logs/`。
- 03:22 C48 服务就绪（编译缓存命中，冷启动约 2 分钟），AgentX 开始；`runtime.env` 为 `PREFILL_NUM_WORKERS=2 PREFILL_TP=1 PREFILL_PP_SIZE=4`，server metrics 包含路由、两个 prefill 与 decode。
- 03:25:31 C48 预热开始（目标 531 个请求）。03:31 时 prefill1 的 prompt 吞吐约 4 万 tok/s、排队 26 个，prefill0 排队 0 个。
- 03:32:03 起预热停滞在 `returned=367/531, in_flight=40, errors=0`；03:42:27 prefill0 退出（显存释放），见 issues.md 第 3 条。
- 04:27:15 作业 32947 收到抢占信号（04:32:15 结束，重新排队），两台节点随即不能 SSH，没能保存引擎日志和删除容器，见 issues.md 第 4 条。
- 04:32:37 作业 32947 结束（sacct：`PREEMPTED`），重新排队；n05-21 由作业 32780（sirafati）、n05-33 由作业 32940（jihhe）使用。
- 04:3x `up.sh` 增加 `follow`：每个容器启动后在其节点上运行 `docker logs -f`，日志实时写入 `<运行目录>/logs/<name>.live.log`。
- 04:34:57 作业 32947 重新运行（`Restarts=1`，05:04:57 起可被抢占），节点 `smci355-ccs-aus-n02-[25,29]`。用户要求在这两台节点上继续测试，节点上如有 docker 容器先停止。
- 04:43 新节点检查：两台均为 amdgpu 6.16.13、ionic 26.07.9.001、host libionic 1.1.39、内核 6.8.0-138；16 张 GPU 显存 0 GiB、利用率 0%；`docker ps -a` 为空，没有需要停止的容器；本账号在 docker 组；主机内存已用 109 GB。
  - n02-25：fenic `10.235.192.129`，`ionic_0-7` ACTIVE，GID 1 为 `192.168.{1..8}.42`；`/data` 剩余 868 GB。
  - n02-29：fenic `10.235.192.61`，`ionic_0-7` ACTIVE，GID 1 为 `192.168.{1..8}.44`；`/data` 剩余 267 GB。
  - 两台的 `ionic_i` 与 rail 的对应与 n05 节点相同；没有 ATOM 镜像；本套件端口空闲。
  - 分工：decode 与控制节点 n02-25，prefill n02-29；`config.sh` 默认值相应修改。
- 04:44:29-04:45:51 n02-25 拉取基础镜像（repo digest `sha256:8d7ebab3069a…`，与之前相同）。构建输入（`../8p4d-atom-infera/docker/Dockerfile`、`patch/`、Infera 源码）在 09-24 07:53 之后没有内容变化；commit `97c03e79` 只是把已有的 `infera-atom-multi-connector.diff` 加入 git。Dockerfile 中的 `pip install` 在构建时解析依赖，新镜像中部分包的版本可能与 `aef823184e20` 不同。
- 05:04:12 n02-25 上执行 `build_image.sh`（构建并同步到 n02-29，日志 `.tmp/logs/build_image_n0225.log`）。04:46-05:04 之间的约 18 分钟没有操作。
- 05:04:41 n02-25 构建完成，镜像 `sha256:e61dab71a5d1…`（09-24 在 n05-21 构建的为 `sha256:aef823184e20…`，构建输入相同）。`pip freeze` 保存在 `.tmp/logs/pip_freeze_e61dab71.txt`：ATOM `d9f0720e` 加 3 个 patch 文件，`grpcio-health-checking 1.81.1`、`protobuf 6.33.6`、`lmcache 0.5.5rc3`、`transformers 5.16.1`，与上一组记录的版本一致。随后开始 `docker save | ssh n02-29 docker load`。
- 05:10:37 作业 32947 第二次被抢占（sacct：`PREEMPTED`，本次运行 04:34:57-05:10:37），抢占方为 radwived 的作业 32953（QOS `dcgpu-prod-r`，8 节点，含 n02-[25,29]）。镜像同步中断（`client_loop: send disconnect: Broken pipe`），n02-29 上可能留下未完成的镜像层；两台节点上没有启动过引擎容器。
- 05:13 作业 32947 排队（`Restarts=2`，`Reason=Resources`），slurm 估计 08:10:51 在 `smci355-ccs-aus-n03-33`、`smci355-ccs-aus-n04-25` 启动。
- 05:15 `build_image.sh` 改为两台节点并行拉取基础镜像并构建（不再经 SSH 传输 50 GB 镜像），两台的 image ID 不同、内容相同。
- 05:20 用户要求暂停并汇报进展。
- 05:24:20 作业 32947 第三次运行（`Restarts=2`，05:54:20 起可被抢占），节点仍为 n02-[25,29]。
- 05:31 用户要求在这次运行上继续测试。两台节点 GPU 显存 0 GiB，没有容器；n02-25 有镜像 `e61dab71a5d1`，n02-29 没有本套件镜像。
- 05:31:53 两台节点启动显存采样（`.tmp/logs/vram-n0225.log`、`vram-n0229.log`）；n02-25 上依次执行 `build_image.sh` 与 `sweep.sh SWEEP_ID=pp4-n02`（参数与 `pp4` 相同），日志 `.tmp/logs/sweep-pp4-n02.log`。精度已在 n05 上用相同构建输入验证，本次只做 smoke。
- 05:32-05:33 镜像：n02-25 `sha256:e61dab71a5d1…`（命中缓存），n02-29 `sha256:81d922f320b5…`；两者 `pip freeze` 相同（`.tmp/logs/pip_freeze_{e61dab71,81d922f3}.txt`）。
- 05:41 C48 服务就绪（冷启动约 8 分钟，无编译缓存），AgentX 开始；四个容器的实时日志写入 `.tmp/runs/pp4-n02-c48/logs/*.live.log`。smoke：completions 返回重复的 "synthetic"（forced acceptance 2.99 开启时输出不代表精度）；decode 每 rank 202760-203105 个 block，与 n05 相同；每个 prefill 实例 4 个 stage 均完成 matched rails、PRODUCER 注册与 lookup server 启动。
- 05:43 本机运行 `.tmp/stall_watch.sh`：预热 `returned=` 或测量 `done=` 连续 5 分钟不变时报告。06:09 的一次报告为误报（计数超过 1000 后带千位分隔符，正则只取到 `done=1`），已修正。
- 05:46:45-05:55:04 C48 预热完成（531 个请求，0 错误，498.7 s）；05:55:04 测量开始（3600 s）。06:10 时 `done=1,527, err=0`；prefill 两个实例排队 0、KV 占用 0-2.4%、前缀命中率 80-84%；decode 运行 22-24 个、排队 0、KV 占用 33-40%、前缀命中率 96.5%。
- 约 06:11:37 作业 32947 第三次收到抢占信号：`runner.log` 最后一行为 06:11:52（`done=1,650, err=0`），实时日志与显存采样在 06:11:30-06:12:01 停止。06:16:37 作业结束（sacct：`PREEMPTED`，本次运行 05:24:20-06:16:37），抢占方为 xiaoqunw 的作业 32965（QOS `dcgpu-prod`，4 节点 n01-[21,29]、n02-[25,29]，时限 16 小时，06:16:46 启动）。两台节点不能 SSH，套件容器留在节点上（issues.md 第 6 条）。
- 被抢占前的实时统计（测量开始后 16:47，非最终聚合结果）：rps 均值 1.6，输入 198,551 tok/s，输出 1,362 tok/s，TTFT p50 / p95 1.009 / 3.750 s，ITL p50 14 ms，ISL p50 88,194。上一组 DPA8 C48 在 16:33 时为：rps 均值 1.6，输入 191,830 tok/s，输出 1,313 tok/s，TTFT p50 / p95 2.437 / 6.597 s，ITL p50 14 ms，ISL p50 86,415。
- 06:17:55 AgentX 客户端容器（不在作业的 cgroup 内）仍在运行，`benchmark.log` 为 `done=2,130, err=0`，即遗留的引擎仍在服务。作业 32947 再次排队（`Restarts=3`）。
