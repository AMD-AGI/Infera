# 踩坑 / 关键分析(what / why / how / context)

> 本轮最有价值的产出。核心是 dma-buf 开关的原理,以及一堆 PD/同机/router 的实测坑。

---

## ★分析 #1(核心):dma-buf 开关 —— 编译期 `USE_HIP_DMABUF`,不是运行时 env

**What**: 要让 Mooncake 对 GPU 显存走 `ibv_reg_dmabuf_mr`(而非裸 `ibv_reg_mr`),开关是
**编译期 CMake 定义 `USE_HIP_DMABUF`**,而且 base 镜像里它**被静默编译掉了**。

**Why**: base 镜像的 Mooncake(`/sgl-workspace/Mooncake` @ 01d1eb2a)注册逻辑在
`rdma_transport/rdma_context.cpp`:
```cpp
#elif defined(USE_HIP_DMABUF)        // AMD dmabuf 路径:hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr
#else                                // 退化:裸 ibv_reg_mr(无 peermem 时对 GPU 指针 EFAULT)
```
`USE_HIP_DMABUF` 在 `src/CMakeLists.txt` 默认 `ON`,但**只加到 `transfer_engine` target**;
而 `rdma_context.cpp` 属于**另一个 target `rdma_transport`(OBJECT lib)**,拿不到这个 define →
`#elif` 分支被预处理器切掉 → 落到 `#else`。出厂 `.so` 无 dmabuf 调用,也不链 libhsa-runtime64。

**How(Dockerfile 怎么修)**: `build_mooncake_dmabuf.sh`:
1. apply 仓内两 patch(HIP gate + auto-chunk)。
2. **往 `rdma_transport/CMakeLists.txt` 追加** `USE_HIP_DMABUF` + `find_package(hsa-runtime64)` +
   `target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)` + link hsa。
3. `cmake -DUSE_HIP_DMABUF=ON` 重编;**自校验** `nm rdma_context.cpp.o | grep ibv_reg_dmabuf_mr` 有符号。

**How(运行时怎么确保走这条)**: 编进去后,只要注册的是 GPU 显存就自动走。运行时配置是保证
**用对 NIC 且不退回 hip/tcp**:`--disaggregation-ib-device <nic>`、`MC_MS_AUTO_DISC=0 MC_MS_FILTERS=<nic>`、
`MC_GID_INDEX`、`NCCL_IB_DISABLE=1`、**不设** `MC_ENABLE_HIP_TRANSPORT`(跨节点必须 RDMA)。

**Context**: 验证 `.so` 有没有 dmabuf 不能 `strings | grep reg_dmabuf`(它是**外部调用符号**不是
字符串字面量,恒为空)——要 `nm rdma_context.cpp.o` 看 undefined 符号 + `.so` 链不链 libhsa-runtime64。
第一次重编就因为漏了 CMake 传播,cmake 打印 "enabled" 但符号没进对象文件,踩了一轮。

---

## ★分析 #2:那两个仓内 patch 到底是不是 no-op / 影不影响现有功能

**结论**:两个都 apply,和用户方向一致。
- `transfer_engine_impl.diff`(HIP-transport gate):把 `installTransport("hip")` gate 到
  `MC_ENABLE_HIP_TRANSPORT`(默认关)。**dmabuf 路线必需**——否则 selectTransport 优先 hip,跨节点
  peer 打不开 hip IPC segment。
- `rdma_auto_chunk_mr_2017.diff`(超 max_mr_size 分块):**不是**"只修 ibv_reg_mr"。它改的是注册
  **公共前置层**(dmabuf/bare 两条路都经过)。但**对 mlx5 是 no-op**(mlx5 `max_mr_size` 无限大 →
  `length > limit` 永远 false → 单块不分,行为不变);只在有限 max_mr_size 的卡(ionic ~2GiB)才分块。
  → 对我们无害,且更稳。
- 实测两个 diff 都 `git apply --check` **clean** 到 base 的 01d1eb2a(pin 的是 747003c 但上下文没漂)。
- **仓内 `apply_mooncake_cpp_patches.sh` 不含** CMake 传播(B.1 被 infera 主动 drop),所以那步必须在
  build 脚本里单独做——这是本 Dockerfile 与标准 `Dockerfile.sglang` 的唯一实质差异。

---

## ★坑 #3:hip transport 跨独立实例打不开 IPC handle(单机 TP4 失败根因)

**What**: 单机 TP4 P↔D 开 `MC_ENABLE_HIP_TRANSPORT=1`,server 起得来、warmup 200,但真实请求全 500。

**Why**: hip transport 用 `hipIpcOpenMemHandle`。P(GPU0-3)和 D(GPU4-7)是**两个独立 sglang 进程**,
一个进程**打不开另一个进程的 GPU 显存 IPC handle**。prefill 侧报
`hip_transport.cpp:815 Requested address 0x... not found!`,decode 侧
`KVTransferError: Failed to get kvcache from prefill instance`。数据面确实零网卡(mlx5 xmit delta=0),
但传输失败。和跨节点 "peer 打不开 hip IPC segment" 是同一类根因。

**How**: 同机 P↔D **不要开** hip,走 mlx5 RDMA loopback(见坑#4)。hip transport 的适用场景是
**同进程内** GPU P2P,不是两个独立 PD 实例之间。

**Context**: warmup 的 200 OK 是 prefill **本地自环 warmup**,不真正跨实例传 KV,曾误导以为通了。
判据必须是**跨实例 smoke**。

---

## ★坑 #4:同机 P↔D 应走 mlx5 RDMA loopback(可行),ionic 不稳

**What**: 单机 TP4,mlx5 RDMA loopback 稳定跑通(smoke 全对);ionic 第 1 次 OK 后 session 死。

**Why**: 即便同机,让 KV 走 mlx5 网卡 loopback 是稳的(RDMA 自环)。ionic 无 ODP,重复 dmabuf 传输后
mooncake session 不稳(`remote mooncake session not alive`)。同节点同脚本、只换 NIC → 坐实是 NIC 差异。

**How**: 同机 PD 用 `NIC_DEV=mlx5_0 GID=3`,不开 hip。避开 ionic。

**Context**: 这次 ionic 在**同机 TP4** 没在 KV 注册步 OOM/崩(KV 池才 45GB,pin 一份 243+45<288 塞得下);
失败后移到了**传输 session**。而**2 节点 TP8**(KV 池上百 GB)才会在 KV 注册步直接崩(见坑#7)。

---

## ★坑 #5:TP4 每卡权重 210GB,mem-fraction 必须 ≥0.76

**What**: TP4 起 server 报 `ValueError: Loaded weights leave no GPU memory for KV cache under
--mem-fraction-static=0.42. Raise above 0.758`。

**Why**: DSv4 806G 权重,TP4 每卡装 **210.74 GiB**(TP8 是 159)。mem-fraction 是 KV 池的**总预算**;
0.42×288≈121 < 210 → KV 预算算成负数。

**How**: TP4 用 `--mem-fraction-static 0.85`(≥0.758)。TP4 只剩 ~67GB 给 KV(vs TP8 的 117)。

---

## ★坑 #6:router 反复 `Address already in use` + PD worker 假熔断

**What**: (a) 重起 router 报 `Address already in use (os error 98)`;(b) 有时 router 判 prefill
`circuit open / No available prefill workers`,但 prefill `/health` 其实 200。

**Why**: (a) `--network=host` 下 router 绑 host 级端口 + **prometheus metrics 端口**,旧 router 崩了没
释放;(b) warmup 刚完 server 短暂 busy,router 探测早了触发熔断。

**How**: (a) 换**全新**的 `--port` **和** `--prometheus-port`,并彻底 `pkill -9 -f launch_router` 等
端口释放;(b) 等 server settle 后再起 router,或换端口重起让它重新探测。

---

## ★坑 #7:ionic + dmabuf 在 2 节点 TP8 会崩(反证,已独立复现)

**What/Why/How**: 见 evidence/ionic_2node_crash_node322.log。同 dmabuf 镜像强制 ionic,权重加载完
(159GB)后在 **KV 池 `ibv_reg_dmabuf_mr` 注册步**崩:node 215 = NODE_FAIL,node 322 = mooncake
`disaggregation/mooncake/conn.py` **SIGSEGV(exit -11)**。根因:ionic 无 ODP → dmabuf 强制 pin 整个
KV 池(上百 GB)→ 耗 KFD/资源。对照 mlx5(有 ODP)dynamic attach 不 pin,同一步顺利 ready。
**这是整个 mlx5 方案的价值所在**。集群 NODE_FAIL 抽风期,靠 `probe_and_repro.sh` cron 每 10min 抢节点
复现,判据严格区分"坏节点早挂"(inconclusive)vs "KV 注册步崩"(confirmed)。

---

## 其他有用事实

- **冷启动**:2 节点 TP8 ~22min(两阶段权重加载,第一阶段 ~715s);单机 TP4 双实例更慢(~1200s),
  aiter FileBaton 争锁但不死锁。
- **ionic 需 `RDMAV_FORK_SAFE=1`**,否则 `rdma_context setup failed: fork compatibility [22]`。
- **GID**:mlx5 用 idx3(IPv4-mapped RoCEv2,可路由);ionic 用 idx1(`fc01:` global)。idx0/link-local(fe80)会崩。
- **节点间传镜像走 NFS tar**(`docker save/load`),spur 禁 ssh 计算节点。
- **验证 KV 不翻倍**:decode 稳态 VRAM = 权重 + KV,不是 权重 + 2×KV。2节点 237/288、单机 mlx5 243/288。
