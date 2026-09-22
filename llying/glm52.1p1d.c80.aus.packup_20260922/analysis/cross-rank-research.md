# AUS P/D 非同 rank 传输只读调研（2026-09-22）

更正后的结论：**当前运行 router 已在进行非同 DP rank 调度，`INFERA_PD_DP_RANK_AFFINITY=true` 在本次构建的 binary 中没有对应实现，因此没有强制 same-rank。** 初稿把 yihou 归档 patch 的功能误当作本次镜像能力，现已撤回。C80 正常推进与非同 rank 调度并存，是实际可行性的正向证据；仍需关联请求完成/传输事件确认具体成功项及实际 HCA 路径，不能直接把路由日志等同于完整 8×8 GPU/RDMA 验证。

本轮只读查看仓库、已有 preflight 和运行容器源代码、主机接口及路由。未更改任何服务配置，未启动 GPU/RDMA 实验，未影响 C80。

## 三个必须分开的层次

1. **路由与协议**：当前源码 `Infera/rust/router/src/disagg.rs:64` 分别调用 P、D 的 `policy.pick`，没有 rank 约束；当前 binary 的 help、strings 均没有 affinity 字段。旧归档 patch 实现了可选约束，但没有进入本次构建。SGLang 当前镜像 `mooncake/conn.py:2511` 的 sender 接收 `dest_tp_ranks`。
2. **NIC 配置意图与真实路径**：配置将 GPU i 的 Mooncake engine 过滤到 `ionic_i`；若该参数确实生效，P i→D j 会走异 HCA。`MC_ENABLE_DEST_DEVICE_AFFINITY` 是 peer HCA 选择 hint，不会建立网络路由。当前已经出现异 rank picks，不能据配置推断“必然不通”；还须从实际 engine 初始化和传输证据确认 filter、目标 HCA 与成功请求。
3. **物理可达性**：当前主机路由确认按 rail 分离，但不足以证明交换网络完全不支持跨 rail。需要指定源/目的 HCA 的 RDMA 测试；普通 management ping 不成立，甚至 IP ping 成功也不是 RDMA 成功证据。

## AUS 现场证据

- P `smci355-ccs-aus-n01-33` / `10.235.192.136`：benic1..8 地址 `192.168.r.70/31`，每个 `192.168.r.0/24` 经对应 `192.168.r.71`，默认路由走 fenic。
- D `smci355-ccs-aus-n02-21` / `10.235.192.128`：benic1..8 地址 `192.168.r.40/31`，每个 `/24` 经对应 `.41`，默认路由走 fenic。
- 两边 RDMA map：ionic 0,1,2,3,4,5,6,7 分别对应 benic 1,2,4,3,5,6,8,7；不要假设所有 ionic index 都等于 benic index−1。
- `results/preflight/*.json`：各方向 8 个 GPU write 测试全部 byte-verified，带宽约 38.39–41.27 GB/s。
- 仓库 `infera/tools/preflight/network/mooncakeperf.py:634` 循环给 target/initiator 传同一个 `gpu_id`，故上述只有 8 对角线 GPU 路径（双向 16 项），没有 56 个非对角项。CPU/TCP 成功也不能补足 GPU 跨 rank 证据。

## 运行镜像源码证据

在 P 运行容器 `llying-aus-1p1d-prefill-0` 轻量读取：

- `/sgl-workspace/sglang/python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py:110`：使用 `get_ib_devices_for_gpu(ib_device, gpu_id)`，结果作为 initialize 的 device filter。
- `/sgl-workspace/Mooncake/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp:51`：`enable_dest_device_affinity` 将 local HCA 名作为 `selectDevice` 的 hint。
- 同目录 `rdma_transport.cpp:770`：按目标 buffer 所属位置及 hint 选择目标已公布的 device；失败再走 wildcard topology。
- 同文件 `:210`：memory registration 遍历 engine 的所有已安装 HCA contexts。这为“非同 GPU 使用同 rail”提供代码路径，但非本地 GPU 经某 HCA 注册/访问的硬件支持与速度仍需实测。
- `/sgl-workspace/Mooncake/mooncake-transfer-engine/src/config.cpp:424`：**DEST affinity 使用 getenv 存在性判断，设成 `0` 仍会开启**。若测试关闭，应 unset；`MC_ENABLE_HCA_PEER_AFFINITY` 与 DEST 同时打开会报错并同时关闭。HCA peer affinity/映射只是选择已有可达设备，并不能凭配置创造跨 rail 网络。

## 旧调研的适用范围

`Infera/bench/glm5p2_pd/issue.md:171` 记载旧 campaign 的 600 秒诊断：48 次真实失败全部跨 rank，其中 P7→D0 有 31 次，同 rank 0；错误为 ACK timeout/retry exceeded，后续 session blacklist 放大故障。旧 same-rail packup 与 yihou 实验主要在 crsuse2 节点，不能外推 AUS 已经失败。它解释了归档 baseline 为何使用 same-rank + 单 NIC map，但本次构建并未带入该 router patch，不能称当前运行 same-rank。更不能由旧失败逻辑上证明 AUS 跨 rank 永不可行。

## 建议最小后续验证（未执行）

等 C80 完成后另开短时独立测试，避免共享 GPU/NIC 带宽污染结果；保留全部原配置。每项记录指定 GPU/HCA、实际所选 local/peer HCA、GID、注册返回码、write 返回码、byte verification、NIC retry/error counter、带宽与时延。

1. 先用小 CPU buffer 做指定 HCA 0→0 正对照，再做 0→1、1→0，以当前 `MC_GID_INDEX=1` 判定代表性跨 rail 路径。只有成功后再扩 8×8 HCA 矩阵；两个样本失败不能证明所有组合失败。
2. 用小 GPU buffer 做 0→0 正对照，以及 GPU0→GPU1，双方都显式只用 ionic_0。这个实验将“跨 GPU/rank”与“跨 rail”分离，直接检验 GPU1 经非本地 NIC 的 MR/P2P 支持。再测试跨 NUMA 的 GPU0→GPU4；无数据校验不得记成功。
3. 若共同 HCA 方案成功，可以测试多 HCA 暴露并保留 DEST affinity，让源 GPU 使用本地 HCA、目的进程也暴露同 rail HCA访问 GPU j。代价可能是跨 PCIe/NUMA、更多注册资源；需测性能。更简易的全 rank 固定 ionic_0 是诊断基线，存在 NIC 争用，不宜直接作为最终吞吐配置。
4. 优先关联当前 C80 已发生的非同 rank 调度与成功请求/transfer 记录，作为真实 SGLang 证据。未来独立小请求实验强制指定 P i→D j 并记录目标 HCA；本次 binary 没有 affinity 开关可取消。验证 KV 完整性、生成正确性、失败恢复、TTFT/ITL。

若跨 rail 不通且 GPU 非本地 NIC 访问也不通，需要软件 staging/本机 GPU 拷贝中转或网络配置支持，不能通过单一环境变量解决。当前没有证据决定是否需要这一步。

## 09:35 UTC 运行 router 复核（纠正初稿）

- `docker inspect llying-aus-1p1d-router` 确认 env 有 `INFERA_PD_DP_RANK_AFFINITY=true`，容器命令为 `python3 -m infera.server --router-backend rust ...`。
- 容器 `ps` 确认 PID 7 实际为 `/usr/local/bin/infera-router ... --router-policy kv-aware ...`；无 affinity 参数。
- `infera-router --help` 完整输出没有 `--pd-dp-rank-affinity` 或对应 env 描述；`strings /usr/local/bin/infera-router | grep -E 'PD_DP_RANK_AFFINITY|pd-dp-rank-affinity|pd_dp_rank_affinity'` 无命中；Python router/server 源码也未命中该 env。
- `/opt/infera/rust/router/src` 不存在，符合 Dockerfile 编译后删除 Rust build/source 的步骤。base Dockerfile `:222` COPY 当前 rust，`:230` cargo build，`:231` 安装 binary；overlay 以 `infera-sglang:v0519-llying-aus-base-83e0f6c8` 为基底，仅应用 NextN/HiCache，`:29` 只做 router help 检查，没有应用 yihou router patch。
- 运行日志相邻 pick 示例（P/D 时间差几十微秒；当前源码两次 pick 间没有 await）：

| UTC | P rank | D rank |
|---|---:|---:|
| 09:35:39.073139 / .073171 | 2 | 3 |
| 09:35:39.226868 / .226901 | 2 | 4 |
| 09:35:40.265958 / .265994 | 6 | 5 |
| 09:35:40.306715 / .306744 | 5 | 6 |
| 09:35:41.087662 / .087697 | 5 | 0 |

这些是实际跨 rank 调度证据，足以排除“当前 router 已启用同 rank 约束”。日志本身没有 request ID，尚不替代请求完成与网络路径关联。主任务正在继续 C80，未因本调研改动或重启服务。

已另存 `router-rank-picks-20260922T0936Z.log`：80 条 pick（09:36:07–09:36:36 UTC），仅保留时间、角色、目标，无请求正文。当前路由日志没有 request ID，未把相邻 picks 宣称为逐项完成证据。两次 pick 同步、无 await 的代码见本地 `Infera/rust/router/src/disagg.rs:64`。主任务同期观察到 C80 662 完成、0 errors；该聚合数字支持运行总体正常，不证明样本内每对在截取时都已完成。

## Header → scheduler → KV receiver 代码链复核

进一步只读查看运行 P 容器源代码，并在 D 容器交叉核实实际 Chat Completions 路径和 DP controller。本轮实际 endpoint 是 `/v1/chat/completions`（`--endpoint-type chat`）；Responses 仅作旁证，不替代实际请求链。结论：**当前非同 rank pick 会传给引擎，代码没有把 D 隐式改回 P rank；对应 KV 接收目标属于选中的 D rank。** 这是实际执行路径的代码证据，不依赖 C80 聚合 0 errors 推断。

1. 当前 Rust `disagg.rs:106` 给 D body 写 `disagg_prefill_dp_rank=P`；`:161`、`:488` 的 streaming prefill drain 发 `X-Data-Parallel-Rank=P`；`:530`/`:577` 的 decode POST 发 `X-Data-Parallel-Rank=D`。`dp.rs:17` 仅使 room 对 P rank 取模对齐，没有改 D rank。
2. 实际 Chat Completions 路径：当前 D 容器 `/sgl-workspace/sglang/python/sglang/srt/entrypoints/openai/serving_chat.py:1146` 调用 `extract_routed_dp_rank_from_header(raw_request, request.routed_dp_rank)`；`:1172` 将结果写入 `GenerateReqInput.routed_dp_rank`，`:1173` 原样保留 `request.disagg_prefill_dp_rank`。前面的 `bootstrap_host/port/room` 也原样传递。共用 helper `serving_base.py:261` 优先取 HTTP rank header。因此本轮实际使用的 Chat 路径保留独立的 D 执行 rank 与远端 P rank。Responses 路径也有同逻辑，但不是本次 benchmark 的依据。
3. `managers/tokenizer_manager.py:1442–1443` 继续保留两个字段；`managers/data_parallel_controller.py:752` 遇到显式 `routed_dp_rank`，检查合法性后直接 `sock_send(self.workers[rank], req)` 并返回 True。round-robin 和 follow-bootstrap-room scheduler 都先检查这一显式分支，命中即 return。因此 bootstrap_room 不会把显式 D rank 覆盖成 P rank。这部分已在实际 D 容器确认。
4. D 的 `disaggregation/decode.py:711` 用 `disagg_prefill_dp_rank` 确定**远端 P**；`:1102` 调用 `decode_req.kv_receiver.init(prefill_dp_rank)`。它改变联系哪一个 P bootstrap，不迁移 D 执行 rank。
5. `disaggregation/mooncake/conn.py:2630` 的 receiver 注册 `self.kv_mgr.kv_args.kv_data_ptrs`、`:2655` 的自身 `engine_rank`、`:2692` 的自身 `session_id`，发给对应 P bootstrap。P 的 `_transfer_data` 在同文件 `:641–647` 使用接收方 session 与目的地址调用 `batch_transfer_sync`。因此跨 rank 请求的目的 KV buffer 是 D j 的 buffer，不是默认改写成 D i。

边界：上述证明路由参数被实际 engine 路径 honoring，以及传输目标属于相应 rank。没有新增请求，也没有启用额外 tracing；现有小样本无逐请求成功 session trace，故仍不把每个 sampled pick 声称为已完成传输，更不等同于全面异 HCA 矩阵验证。
