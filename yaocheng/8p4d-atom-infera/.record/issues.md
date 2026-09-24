# 问题记录

每个问题记录现象、日志片段、原因、处理、状态。

## 1. n10-29 的 GPU 被他人容器占用

- 现象：作业 31626 开始时，n10-29 的 8 张 GPU 显存占用 85-86%，`rocm-smi --showpids` 显示 `sglang::scheduler` 等进程。
- 原因：yihou 的 `glm52-pd-yihou-sn-p4d4-n1029-*` 容器在作业开始前启动，docker 容器不受 slurm 分配约束。
- 处理：用户授权在 GPU 步骤开始时 `docker stop`（不删除）这 3 个容器，以及 n02-33 上的 `sikl.jihhe`、`dev_primus_mxfp6_265`。
- 状态：已处理（10:02 停止，容器保留，可用 `docker start` 恢复）。

## 2. 后端网络按 rail 隔离

- 现象：`ping -I benic1p1 192.168.2.72` 等跨 rail 测试失败，同 rail 成功。
- 原因：两节点间只有同编号 rail 互通；Prefill 的 8 个 DP rank 向 Decode 的 4 个 TP rank 写 KV 时会遇到跨 rail 的 GPU 组合。
- 处理一：decode 的 kv-transfer-config 增加 `"ib_enable_alternate_hca": true, "ib_hca_count": 8`，两侧设置 `MC_ENABLE_DEST_DEVICE_AFFINITY=1`。
- 结果一：GSM8K（短提示）正常；AgentX 验证（12:19 起，长上下文）中 decode 侧 `pd_kv_transfer_seconds` 27 次累计 983.8 s，平均 36.4 s/次；10 s 采样窗口内 16 张 ionic 的 `tx/rx_rdma_ucast_bytes` 均为 0（`.tmp/rdma_rate.sh`），prefill 的 `ionic_1`、`ionic_5` 各有 8 次 `req_tx_retry_excd_err`。原因：decode 注册在 8 张 HCA 上时通告的目标网卡不唯一（connector 注释："An engine registered on multiple NICs can still choose an unreachable rail"），部分 RDMA WRITE 发往不可达 rail，重传超限后等超时。
- 处理二：decode 去掉 alternate HCA（每个 rank 只注册 `ionic_<gpu>`），prefill 设置 `ATOM_MOONCAKE_MATCHED_RAILS=auto`（每条 rail 一个单网卡引擎，按 consumer 通告的网卡选同 rail 引擎）。写入 `config.sh` 的 `PREFILL_ENV` 与 `scripts/up.sh`。
- 结果二（12:37，`check2-matchedrails`）：prefill 8 个 DP rank 均输出 `Auto-discovered Mooncake matched rails`；此后 `req_tx_retry_excd_err` 不再增长，decode 只在本卡 `ionic_0-3` 接收。AgentX 结果与处理一基本相同（TTFT p50 13.07 s），传输慢的主因见第 6 条。
- 状态：已解决（跨 rail 重传）。

## 6. DCP consumer 的 KV 传输按 token 写入

- 现象：AgentX 负载下 decode 侧 `pd_kv_transfer_seconds` 平均 36-56 s，31 次中 0 次低于 5 s；采样窗口内网卡速率 0.01-0.02 GB/s。空闲时 10 万 token 提示的 KV 传输 4.93-5.56 s（`.tmp/kv_transfer_probe.sh`）。
- 原因：GLM-5.2（DSA 索引缓存）+ MTP 的 PD 要求 DCP `interleave_size=1`（connector："Sharded preshuffled DSA index P/D requires dcp interleave_size=1"；`config.py` 对投机解码也有同样要求）。consumer rank 拥有每隔 4 个的 token，producer 对 MLA KV 区域按 token 生成 RDMA 描述符（每层每 token 一条，每 4096 条做一次同步写），10 万 token 约 790 万条。
- 尝试：`_MAX_RDMA_ENTRIES_PER_BATCH` 4096 → 262144（容器内修改后重启），传输变为 10.65-11.44 s，排除批次往返次数这一原因。
- 处理：`patch/atom-dcp-kv-staging.diff`（`mooncake_connector.py` + `aiter_mla.py`）：producer 为 MLA KV 区域建立 token 视图，按 8192 token 分块、每次 8 层在 GPU 上 `index_select` 收集到已注册的 staging 缓冲（16 槽，与默认 16 个发送线程一致，每槽独立 stream），合并连续目的地址后整页 RDMA；DSA 索引 staging 分块由 256 页改为 2048 页。`.tmp/patchwork/test_kv_staging_mapping.py` 在镜像内验证 staging 写出的字节与逐 token 路径一致（每块描述符 1024 → 约 60）。
- 结果（空闲，10 万 token）：第一版（4 槽、逐层写）传输 2.57-3.98 s；按 8 层合并与索引 2048 页后 1.86-2.10 s；16 槽版 1.57-2.87 s；端到端 9.1-9.8 s（基线 12.4-15 s）。6 万 token 大海捞针 3/3（三个版本均通过），GSM8K 200 题 0.99（第一版）、0.965 ± 0.013（16 槽版）。
- AgentX 负载对比（`check3-staging`，C16，600 s）：total 2732.42 tok/s/GPU，TTFT p50 14.46 s / p90 37.04 s，ITL p50 6.73 ms，错误 0；check2 为 2861.15、13.07 s / 34.01 s。负载下 TTFT 没有下降，主要耗时在 prefill 重算（第 8 条）。
- 状态：已解决（空闲传输时间约为原来的 40%，精度不变），patch 保留。

## 7. 服务端口被出站连接占用

- 现象：12:59 prefill 启动失败：`ERROR: [Errno 98] error while attempting to bind on address ('0.0.0.0', 39001): address already in use`，事后该端口空闲。
- 原因：两节点临时端口范围为 32768-60999，38000/39001/39002 位于其中，引擎启动时大量出站连接可能先占用同号端口；日志中 Mooncake 的随机 RPC 端口在 15000-17000，与握手端口 16301-16311 重叠。
- 处理：路由 18000、prefill 19001、decode 19002、握手端口 21301 起（prefill）与 21311（decode），`config.sh` 中注释说明约束。
- 状态：已解决。

## 8. prefill 各 DP rank 的 prefix cache 命中率低

- 现象：`check3-staging` 中 prefill 8 个 DP rank 的 prefix cache 命中率为 24%-41%，decode 为 78.4%，AgentX 数据集的理论命中率为 95%。
- 原因：ATOM 的 DPA 引擎在 `CoreManager` 内按负载为每个请求选择 DP rank，每个 rank 有独立的 prefix cache，同一会话的多轮请求分布到不同 rank，后续轮次需要重算前文。ATOM 提供 `ATOM_DP_SESSION_AFFINITY`：按请求头 `x-dynamo-session-id` 或 `x-correlation-id`（AIPerf 在同一会话的各轮保持不变）把新会话放到负载最低的 rank，之后各轮固定在该 rank。Infera PD 路由发给两个引擎的请求只带 `X-Request-Id` 与 DP rank 头，客户端的会话头没有转发。
- 处理：`patch/infera-forward-session-headers.diff`：`infera/server/app.py` 把 `x-correlation-id`、`x-dynamo-session-id`、`x-dynamo-parent-session-id` 存入私有字段 `_infera_session_headers`，`infera/router/disagg.py` 把它加到 prefill 与 decode 两条请求的请求头，`infera/router/mixed.py` 转发前删除该字段。Dockerfile 在 `pip install` 前对 `/opt/infera` 应用 `patch/infera-*.diff`。`config.sh` 的 `PREFILL_ENV` 增加 `ATOM_DP_SESSION_AFFINITY=1`（decode 为 TP4，只有一个 DP rank，不需要设置）。
- 结果（`check4-affinity`，C16，600 s，与 check3 相同参数）：prefill 路由计数 `atom:dp_affinity_new_total` 随会话增加，`dp_route_load_balanced_total` 与 `dp_route_explicit_total` 为 0；prefill 各 DP rank 命中率 49%-92%（7 个 rank 在 73% 以上），decode 85.0%。total 4888.7 tok/s/GPU（check3 为 2732.42），output 31.24（20.65），TTFT p50 1.50 s（14.46 s）、p90 9.24 s（37.04 s），ITL p50 8.36 ms（6.73 ms，decode 同时运行的请求增多），profiled 252（148），错误 0。
- 状态：已解决。

## 9. 作业 31626 被抢占后在其他节点重新运行

- 现象：15:03 SSH 到 n10-29 返回 `Access denied by pam_slurm_adopt: you have no active jobs on this node`。`sacct -D -j 31626`：第一次运行 09:28:39-14:55:33，状态 `PREEMPTED`；14:57:46 以同一编号在 `smci355-ccs-aus-n04-[25,29]` 重新运行。之后 n10-29 为 `allocated`（其他作业），n02-33 为 `completing`。
- 影响：C80 在预热阶段中断，数据不完整。原节点已无访问权限：本套件容器（n10-29 的 etcd、decode、router、agentx，n02-33 的 prefill）可能仍在运行；此前停止的他人容器（n10-29 的 `glm52-pd-yihou-sn-p4d4-n1029-{prefill-0,decode-0,etcd}`，n02-33 的 `sikl.jihhe`、`dev_primus_mxfp6_265`）仍为停止状态，需要节点所有者或管理员处理。
- 处理：套件改用新节点（prefill n04-25，decode 与控制节点 n04-29），重新拉取基础镜像、构建并同步，冷启动后重新执行 C80-C256。
- 第二次抢占：`scontrol show job 31626` 显示 `QOS=batch`、`Priority=2091`、`PreemptEligibleTime=15:27:46`（启动后 30 分钟）、`PreemptTime=15:28:01`、`EndTime=15:33:01`（5 分钟宽限期），`Requeue=1 Restarts=1`；分区 `Compute-DCPT` 中有 `dcgpu-te` QOS、优先级 6751 的作业等待资源。15:28 起两台 n04 节点的 SSH 均被 pam_slurm_adopt 拒绝。n04-25 上的 prefill 容器与 n04-29 上的 etcd 容器可能仍在运行，n04-29 的 `sikl.jihhe` 为停止状态。
- 抢占来源（15:40 查询）：
  - 提交命令（`sacct --format=SubmitLine`）：`sbatch --partition=Compute-DCPT --qos=batch --nodes=2 --ntasks-per-node=1 --cpus-per-task=256 --gres=gpu:mi355x:8 --mem=0 --exclusive --time=24:00:00 --job-name=cyao1002-2n --wrap="sleep infinity"`，09:17:54 提交。
  - 14:55 的抢占方为作业 31639（jabowden，账号 swammy，QOS `dcgpu-test`，优先级 6748，14:55:43 在 n02-[25,33]、n03-33、n04-33、n05-[29,33]、n06-[25,33] 启动，15:04:02 `NODE_FAIL` 结束）。n10-29 随后由作业 31640（yelkhamr，QOS `batch`）使用，该作业是在 31626 重新排队后分配到节点，没有抢占 31626。
  - 15:28 的抢占方为作业 31642（jabowden，账号 swammy，QOS `dcgpu-test`，优先级 6748，15:33:12 在 n01-[21,25,29,33]、n02-29、n04-[25,29]、n06-25 启动）。
  - 集群配置：`PreemptType=preempt/qos`，`PreemptMode=REQUEUE`，`PreemptExemptTime=00:30:00`，各 QOS 的 GraceTime 为 5 分钟。`dcgpu-test`（优先级 5000）可抢占 `batch,debug,long-run,normal,perf,shared-low,shared-medium,special-group`；不在其可抢占列表中的 QOS 为 `dcgpu-test`、`dcgpu-prod`、`dcgpu-fullpool`、`sponsor`。
  - 账号 emad 下 cyao1002 可用的 QOS 为 `batch,debug,normal,shared-low`（默认 `normal`），均可被 `dcgpu-test` 抢占。
  - 15:40 作业为 `PENDING`（`Reason=Priority`，`Restarts=2`），`SchedNodeList=smci355-ccs-aus-n04-[25,29]`，预计 16:14 启动。
- 状态：阻塞。作业每次启动 30 分钟后即可被抢占，单档 AgentX（冷启动、预热、3600 s 测量）需要约 75-90 分钟。用户要求：分配到问题节点（如 n04-29 这类新驱动节点）时暂停。

## 10. n04-29（amdgpu 6.19.16）上 decode 在 KV 分配后 OOM

- 现象：15:10 在 n04-29 冷启动 decode（与原节点相同的镜像和参数，`GPU_MEM_UTIL=0.85`），4 个 rank 在 `model_runner.allocate_kv_cache` 之后的 `torch.distributed.barrier()` 失败：`Failed to CUDA calloc 33554432 bytes`，`HIP failure: 'out of memory'`。同一镜像的 prefill 在 n04-25（amdgpu 6.14.14）正常就绪。
- 数据：ATOM 显存预算与原节点一致（`total_gpu=287.98GB`，`budget=244.79GB`，`peak_torch=110.32GB`，`non_torch≈26.6GB`，KV 101.5 GB）。`.tmp/decode_probe.sh` 在 n04-29 同时启动两个只含 decode 的实例（GPU 0-3 为 `MOONCAKE_DISABLE_HIP_DMABUF=1` 的 peer-mem 注册，GPU 4-7 为 `MOONCAKE_DISABLE_HIP_DMABUF=0` 的 dma-buf 注册），`.tmp/vram_sample.sh` 每 2 s 采样：两者都在显存用量约 243 GB（总量的 84.4%）时 OOM，此时仍有约 45 GB 未使用。两台节点 BAR 可见显存均为 287 GB；ionic 驱动 n04-25 为 26.03.3，n04-29 为 26.07.9，host libionic 分别为 1.1.54 与 1.1.39。
- 结论：与 RDMA 注册方式无关；n04-29 的驱动下可分配显存上限约为总量的 84%，低于 ATOM 0.85 预算加 RCCL 缓冲。
- 状态：未解决（节点已被抢占）。在该驱动版本的节点上，可行的处理有两种：decode 放在旧驱动节点上；或者降低 decode 的 `GPU_MEM_UTIL`，这会减少 KV 容量。

## 3. infera `[atom]` 依赖升级 protobuf

- 现象：`pip install ".[atom]"` 输出 `opentelemetry-proto 1.40.0 requires protobuf<7.0,>=5.0, but you have protobuf 7.36.2 which is incompatible.`
- 原因：`[atom]` 包含 gaie 依赖，pip 为 `grpcio-health-checking>=1.81.1` 选择了 1.84.0，该版本需要 protobuf 7，于是卸载了基础镜像的 protobuf 6.33.6。
- 处理：Dockerfile 改为 `pip install ".[atom]" "protobuf<7"`，pip 改选 `grpcio-health-checking 1.81.1`，protobuf 保持 6.33.6；`pip check` 与基础镜像一致。
- 状态：已解决。

## 4. 通过 SSH 前台运行的 up.sh 随会话退出

- 现象：10:05 从本机经 `ssh -J` 启动的 `up.sh` 在 10:07 结束，终端记录 exit code 未知；n10-29 上已无 `up.sh`，但 etcd、prefill、decode 容器仍在运行，路由未启动。
- 原因：脚本依附于本机的 SSH 会话，会话被中断时收到 SIGHUP。
- 处理：长时间脚本（up、agentx、sweep）改为在控制节点上用 `nohup setsid bash scripts/... > .tmp/logs/<name>.log 2>&1 &` 启动，只轮询日志。本次部署等待两个引擎就绪后手动执行 `up.sh` 末尾的路由启动命令。
- 状态：已规避。

## 5. DPA prefill 在 `--enforce-eager` 下 MoE 断言失败

- 现象：GSM8K 开始约 1 秒后（10:17:49，8 个 DP rank 各有约 8 个请求），prefill 的一个 ModelRunner 退出，其余 rank 随后因 gloo 连接断开退出；prefill 的 `/health` 仍返回 200，但请求不再完成。日志：
  ```text
  File "/app/ATOM/atom/model_ops/moe.py", line 386, in pad_for_all_gather
  AssertionError: MoE was handed 35 rows on a decode step expecting 40 (scheduled_tokens=35, running_tokens=40)
  RuntimeError: [.../gloo/transport/tcp/pair.cc:547] Connection closed by peer [127.0.1.1]:22963
  AsyncIOProcManager(ModelRunner): [ModelRunner0/1] proc died unexpectedly (exitcode=1), shutting down.
  ```
- 原因：DPA 的 prefill 引擎仍会执行 decode 形态的步骤（MTP，每序列 1+4 个 token）。`ForwardMode` 把 batch 按 capture 梯度填充（7 个序列按 8 计，`running_tokens=40`），而 `--enforce-eager` 使该步走非 cudagraph 路径，`input_ids` 保持实际的 35 行；DP MoE all-gather（`pad_for_all_gather`）在 decode 步要求行数等于 `running_tokens`，于是断言失败。`--enforce-eager` 来自 ATOM 1P1D PD recipe，那里的 prefill 为 TP1×PP4，没有 DPA；ATOM 的 TP8 + DPA recipe 不使用 `--enforce-eager`。
- 处理一：prefill 去掉 `--enforce-eager`，改为 `--max-num-seqs 64 --cudagraph-capture-sizes "[1,2,4,8,16,32,64]"`，让 decode 形态步骤走填充后的 graph 路径。
- 结果一（10:26，`.tmp/runs/bringup2-c16-noal`）：8 个 DP rank 完成 bs 64→1、`max_q_len=5` 的捕获后，全部 8 张 GPU 报 `Memory access fault by GPU node-N ... Reason: Unknown`，ModelRunner exitcode=-6，prefill 启动失败；decode 正常就绪。
- 补充事实：prefill 执行 decode 形态步骤的原因是 MTP。producer 在 `kv_transfer_params` 中返回 `draft_token_ids`（`mooncake_connector.py` 第 505-528 行），需要运行 drafter；connector 要求 producer 与 consumer 在投机解码上一致（第 1722-1725 行："agree on speculative decode (a draft KV layer widens every group)"），所以不能只在 decode 开 MTP。ATOM 的 GLM-5.2 recipe 中 TP8 + DPA 与 MTP 分别验证过，没有两者同时开启的配置。
- 用户决定（11:10）：decode 必须保留 MTP，prefill 侧尽量调通，允许在 `patch/` 下加 patch 并打进 Dockerfile。
- 处理二：`patch/atom-moe-eager-decode-pad.diff` 修改 `atom/model_ops/moe.py` 的 `pad_for_all_gather`：decode 步行数等于 `scheduled_tokens` 时（eager 未填充）按 prefill 方式填充到 `running_tokens`，填充行清零（该函数也用于 router logits），返回真实行数供 reduce-scatter 截回。只在该条件下生效，cudagraph 路径与 prefill 步逻辑不变；decode 节点 dp_size=1，不经过这段代码。Dockerfile 用 `git -C /app/ATOM apply` 打入，prefill 恢复 `--enforce-eager --max-num-seqs 512`。
- 结果二（`bringup3-c16-noal`）：prefill 运行正常，GSM8K 200 题 0.98，无报错。
- 性能关注（用户 11:20 提出）：eager 只影响 producer 每请求一次的 decode 形态前向（prefill 前向在任何模式下都不用 cudagraph）。探测显示 prefill 端每请求有约 200 ms 的固定开销。
- 尝试三（11:39，`bringup4-c16-piecewise`）：prefill `--cudagraph-mode PIECEWISE --cudagraph-capture-sizes [1,2,4,8,16,32,64]`，捕获完成后同样 8 张 GPU `Memory access fault`，与 FULL 模式相同，说明故障与 DPA + MTP producer 的 cudagraph 捕获或首次重放有关，与 graph 模式种类无关。
- graph 模式定位（11:52-12:10，prefill 捕获尺寸 1-64，均在捕获后 8 卡访存错误）：`ATOM_DP_DRAFT_ARGMAX=0` 无效；`ATOM_DRAFT_CUDAGRAPH=0` 无效；`ATOM_DP_LM_HEAD_MODE=default` 无效。`PYTHONFAULTHANDLER=1` 显示主线程停在 `spec_decode/draft_graph.py:303 warmup`（`torch.cuda.synchronize()`），调用链 `capture_cudagraph → drafter.warmup_draft_graphs`，即 drafter 在合成 context 上的预热 pass；eager 模式不调用 `capture_cudagraph`，真实请求中的 eager draft pass 工作正常（GSM8K 0.98，decode 接受率 72%）。
- 尝试四（12:05）：`.tmp/patchwork/atom-dp-eager-draft-skip-warmup.diff` 在 DP 且无 pass 需要捕获时跳过该预热，prefill 用 graph + `ATOM_DRAFT_CUDAGRAPH=0`，启动成功（镜像 `sha256:e8431489a94c…`）。延迟探测：prefix cache 命中时 prefill 调度到输出 219-231 ms，与 eager 的稳定值（约 218 ms）相同；各 rank 首次 1500 token 请求 605-635 ms（eager 为 250-265 ms），原因是跳过预热后首次出现的形状在服务中 JIT。
- 结论：每请求约 220 ms 的固定开销在 graph 模式下同样存在，eager 的 decode 形态前向没有带来可测量的差异。默认保留 eager + `patch/atom-moe-eager-decode-pad.diff`；第二个 patch 不打入镜像，保留在 `.tmp/patchwork/` 备查。两节点镜像标签指回只含第一个 patch 的 `sha256:cb8c9eee3ee2…`（即当前 Dockerfile 的构建结果）。
- 状态：已解决（eager + patch）。
