# 踩坑 / 关键分析(what / why / how / context)

> 本次最有价值的产出。核心是 **dmabuf 开关到底怎么回事**,以及 spur/PD 的实测潜规则。

---

## ★分析 #1(核心):Mooncake 的 `ibv_reg_dmabuf_mr` 开关

**What**: 用户要"运行时使用 `ibv_reg_dmabuf_mr` 的正确开关"。答案不是一个 env,而是一个
**编译期开关 `USE_HIP_DMABUF`**,而且 base 镜像里它**被静默编译掉了**。

**Why(为什么 base 镜像没有 dmabuf)**:
base 镜像 `lmsysorg/sglang:v0.5.15.post1` 打包的 Mooncake(源码 `/sgl-workspace/Mooncake`
@ 01d1eb2a, upstream #2682)里,注册逻辑在 `rdma_transport/rdma_context.cpp`:
```cpp
#if defined(USE_CUDA) ...            // NV 路径
#elif defined(USE_HIP_DMABUF)        // ← AMD dmabuf 路径:hsa_amd_portable_export_dmabuf
    ... ibv_reg_dmabuf_mr(...)        //    + ibv_reg_dmabuf_mr(不 pin,靠 ODP dynamic attach)
#else
    ... ibv_reg_mr(...)              // ← 退化:裸 reg_mr(在 ionic 上要 pin/翻倍)
#endif
```
`USE_HIP_DMABUF` 在 `src/CMakeLists.txt` 里默认 `ON`,但**只加到了 `transfer_engine` target**:
```cmake
target_compile_definitions(transfer_engine PRIVATE USE_HIP_DMABUF)  # ← 只给了这个 target
```
而 `rdma_context.cpp` 属于**另一个 target `rdma_transport`**(OBJECT lib),那个 target
**没拿到这个 define** → `#elif USE_HIP_DMABUF` 分支被预处理器切掉 → 落到 `#else` 的裸
`ibv_reg_mr`。所以出厂 `.so` 里根本没有 dmabuf 调用(`nm rdma_context.cpp.o` 无
`ibv_reg_dmabuf_mr` 符号,`.so` 也不链接 libhsa-runtime64)。

> 这正是 infera 仓 `deploy/docker/patches/mooncake_cpp/apply_mooncake_cpp_patches.sh`
> 注释里说的 **"B.1: propagate USE_HIP_DMABUF to rdma_transport"** —— 但 infera **主动把
> B.1 drop 了**,因为在 **ionic(无 ODP)** 上 dmabuf 会 pin 整个 KV 池(~156 GiB/GPU)→
> 耗尽 KFD 资源 → 之后每个 `hipModuleLoad` 失败(HIP-209)→ decode 崩。infera 因此只保留
> 裸 `ibv_reg_mr` + host-libionic 注入。

**How(怎么修回来)**: 见 `scripts/build_mc_dmabuf.sh`,三步:
1. 往 `rdma_transport/CMakeLists.txt` **追加**:`option(USE_HIP_DMABUF ON)` +
   `find_package(hsa-runtime64)` + `target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)`
   + `target_link_libraries(rdma_transport PRIVATE hsa-runtime64::hsa-runtime64 hip::host)`。
2. `cmake -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON ... && ninja engine.…so`,cover 掉 pip 装的 `.so`。
3. 顺便把 `installTransport("hip")` gate 到 `getenv("MC_ENABLE_HIP_TRANSPORT")`(默认关)。
4. 验证:`nm rdma_context.cpp.o | grep ibv_reg_dmabuf_mr` 有符号;`.so` 链接 libhsa-runtime64。

**Context(为什么这里能用 dmabuf 而 infera 不行)**:infera 针对 **ionic**(无 ODP,dmabuf→pin→
翻倍→崩)。本实验**强制 mlx5**(有 `ODP_SUPPORT`),dmabuf 走 **dynamic attach**——NIC 靠缺页/
move-notify 回调直接访问 GPU 显存,**不 pin、不复制第二份**。所以同一条 dmabuf 路径:ionic 上
是灾难,mlx5 上正是想要的"零 pin 不翻倍"。实测 decode 稳态 237/288 GiB(=159+78,未翻倍),
若 pin 则 159+2×78=315>288 必 OOM——没 OOM 即证明 dynamic attach 生效。

---

## ★分析 #2:为什么必须 gate 掉 HIP transport(`MC_ENABLE_HIP_TRANSPORT`)

**What**: 重编时把 `installTransport("hip")` 藏到 `getenv("MC_ENABLE_HIP_TRANSPORT")` 后(默认关)。

**Why**: upstream #2682 无条件装了个 "hip" transport,`selectTransport` 会**优先 hip(优先级4)
而非 rdma(优先级2)**。hip transport 用 `hipIpcOpenMemHandle` 做**同机** GPU P2P,但**跨节点**
peer 打不开对方的 IPC handle → `Corrupted segment descriptor` / KVTransferError → decode 拿不到 KV。

**How**: gate 成默认关,跨节点 PD 自动走 rdma;真要同机 P2P 才 `MC_ENABLE_HIP_TRANSPORT=1`。
evidence/decode.log 里能看到 `installTransport, type=rdma`(没有 hip)= gate 生效。

**Context**: 这条 infera 仓也修了(`patch_mooncake_sglang.sh` / B.2),我们照搬。

---

## ★坑 #3:强制单张 mlx5 的正确姿势(3 层保险)

**What**: 让全 8 个 GPU 的 KV 都走**同一张** mlx5,而不是默认自动发现去抓 8 张 ionic。

**How(三层)**:
1. sglang 官方 flag:`--disaggregation-ib-device mlx5_0`(支持 per-GPU JSON 映射,这里全用 mlx5_0)。
2. Mooncake 层双保险:`MC_MS_AUTO_DISC=0` + `MC_MS_FILTERS=mlx5_0`(关自动发现 + 白名单)。
3. 进程/gloo 层:`SGLANG_LOCAL_IP_NIC=ens3 GLOO_SOCKET_IFNAME=ens3`,`--host <mlx5 IP>`。

**Context**: sglang PD 脚本本来就用 `MY_IP` 反查网卡,而节点上**只有 mlx5(ens3)有 IP**、
8 张 ionic 全 `ip=none`,所以天然偏向 mlx5;上面三层是把它钉死。验证:decode.log 全程
`grep ionic` = **0**。

---

## ★坑 #4:镜像里 `sglang.bench_serving` 是坏的

**What**: `python3 -m sglang.bench_serving` 报 `ModuleNotFoundError: sglang.benchmark.serving`。

**Why**: 这个镜像里 `sglang/bench_serving.py` 是个 shim,import 了被移走的模块;真正的 bench
代码在 `/sgl-workspace/sglang/python/sglang/benchmark/serving.py`,但 pip 装的 sglang 不含
`benchmark` 子包。

**How**: 用 `PYTHONPATH=/sgl-workspace/sglang/python python3 -m sglang.benchmark.serving …`
(`scripts/bench_sweep_mlx5.sh` 已内置)。参数与 legacy 一致:`--backend sglang --dataset-name
random --random-range-ratio 1.0`。

---

## ★坑 #5:GID index 必须用 3(不是 1)

**What**: 跨节点 RoCE 用错 GID 会连不上/崩。

**Why**: mlx5_0 port1 的 GID:idx1 = `fe80::…`(**link-local**,跨子网不可路由);
idx3 = `::ffff:10.245.x.x`(**IPv4-mapped RoCEv2**,可路由)。两节点在 `10.245.x/20`,
必须用 idx3。`MC_GID_INDEX=3`。(legacy 在 ionic 上用的是 idx1,因为那套网不同——别照抄。)

**How**: `show_gids | grep mlx5_0 | grep v2` 找带 IPv4 的那条,取它的 index。

---

## ★坑 #6:router 端口冲突 + 必须经 router 才能 smoke

**What**: (a) router 起在 8100 报 `Address already in use`;(b) 直连 prefill 报
`Disaggregated request received without bootstrap room id`。

**Why**: (a) `--network=host` 下 8100 被节点上**别人的进程**占了(容器内 `pgrep` 看不到 host 进程);
(b) PD 分离下,请求必须经 router 注入 bootstrap room id 来配对 prefill↔decode,裸打 prefill 会被拒。

**How**: (a) 换端口 8200;(b) smoke/bench 一律打 router(`http://127.0.0.1:8200`),别直连 30000。

---

## 坑 #7:节点间传镜像不能 ssh(spur 禁)

**What**: `docker save | ssh <node> docker load` 流式传镜像失败。

**Why**: spur 集群 2026-07-22 起禁止普通用户 ssh 计算节点(`AllowUsers ubuntu root` 白名单,
报错伪装成 publickey)。

**How**: 走 NFS 中转:`docker save -o /home/$USER/…/img.tar`(NFS,两节点可见)→
另一节点 `docker load -i` 同一个 tar。27GB,慢但可靠。

---

## 弯路记录(err analysis)

1. **第一次重编 dmabuf 没生效**:cmake 打印了 "HIP dmabuf enabled"、`.so` 也链接了 hsa,但
   `nm rdma_context.cpp.o` 没有 `ibv_reg_dmabuf_mr`。定位:cmake 的 message 来自
   **transfer_engine** target 的 option,而 `rdma_context.cpp` 在 **rdma_transport** target,
   define 没传过去。→ 加坑#1 的 CMakeLists 传播才真正编进去(build.log = 失败,build2.log = 成功)。
2. **验证方法踩坑**:`strings engine.so | grep reg_dmabuf` 恒为空——因为 `ibv_reg_dmabuf_mr` 是
   **外部调用符号**,不是字符串字面量。正确判据是 `nm rdma_context.cpp.o | grep ibv_reg_dmabuf_mr`
   有 undefined 符号,以及 `.so` 链接 libhsa-runtime64。
3. **误判"conc≥64 会顶显存"**:拍脑袋结论。实算:KV 池按 token 总量,prefill 池 580 万 token,
   8k1k(9216 token/条)要 ~630 conc 才占满,conc≤128 只用 20%。真正的软上限是
   `max_running_requests=128`,不是显存。

---

## 有用事实

- **冷启动 ~22min**:权重两阶段加载 elapsed=1302s(evidence/prefill.log),之后 KV 分配 + warmup。
  判活看 VRAM 上涨,别急着判死。
- **cuda graph 是关的**(`disable_cuda_graph=True`),所以没有图捕获阶段。
- **KV 未翻倍的直接证据**:decode 稳态 237/288 GiB = 159 权重 + 78 KV;disagg warmup 阶段就已
  200 OK 跨节点传了 KV(evidence/decode.log)。
- **bench 软上限**:吞吐在 conc=128 到顶(~26.6k),conc=256 只堆 TTFT(4.2s→23s)不涨吞吐。
  要真吃 256 需重起 server 抬 `--max-running-requests`(显存够,KV 池装 256 条 8k1k 只 41%)。
