# GLM-5.2-FP8 1P1D on MI325X —— 部署、验证、基准与调优

本目录是 `tus1-p15-g43` / `tus1-p15-g61` 这对 MI325X 节点上做 GLM-5.2-FP8
prefill/decode 分离（1P1D）的完整记录：怎么部署、怎么证明它是对的、跑出来是多少、
以及低并发下怎么把单用户速度提上去。

两种部署方式都做了，并且做了逐 flag 的静态比对和同一负载的动态比对：

- **Docker/裸机臂** —— `examples/glm5.2_gfx942/` 里的 shell 脚本
- **Kubernetes 臂** —— `llying/dev/k8s_glm5p2_gfx942` 分支的 Infera operator

`results/` 是所有原始结果，`tools/` 是过程中写的、可复用到下一个集群的脚本。

---

## 0. 结论速览

| 问题 | 结论 |
| --- | --- |
| 1P1D 在这对 MI325X 上跑得起来吗 | 跑得起来。agentic 多轮负载 448/448 全成功，KV 缓存效率 **100.00%** |
| operator 部署和脚本部署一致吗 | **一致**。三个组件 0 处实质 flag 差异；同一负载 decode 侧指标差 ≤2% |
| 低并发单用户速度 | 默认配方 53.7 tok/s/user；**关掉 dp-attention 后中位 133–154** |
| 130–180 tok/s/user 的目标 | **达成**。在 50–80k 长上下文（这条 trace 的 p50=68k）中位 **153–163**；全体中位 133–154；均值 122–133 被短开场轮拉低（§10.7） |
| 关掉之后高并发的代价 | 聚合吞吐从 182.7 掉到 139.4 tok/s（−24%），TTFT 差 1.79× |
| 最大的坑 | RoCE GID index 选错不报错，只是慢 4–110×（§3） |

一句话：**dp-attention 是个并发档位开关，不是"打开总是更好"的优化**。

| 场景 | 配置 | 单用户（中位） | 聚合吞吐（conc=16） |
| --- | --- | --- | --- |
| 批量 / 高并发 | dp-attention **开**，`CHUNK=8192` | 62.5 tok/s/user | **182.7 tok/s** |
| 交互 / 低并发 | dp-attention **关**（`DP=1`），`CHUNK=2048` | **133–154 tok/s/user** | 139.4 tok/s |

同一份权重、同一套镜像，只差这一个开关。**这是本次最该交给客户的一条结论。**

调优侧另外三条路都试过并证伪了：custom all-reduce 会死锁（§10.3）、
树状 draft 被 DSA 后端拒绝（§10.4）、加深 draft 链净亏（§10.5）。
出厂的 MTP 形状 `5/1/6` 经实测已落在最优点。

---

## 1. 集群与前置条件

### 1.1 硬件拓扑

| 节点 | 角色 | 数据网 IP |
| --- | --- | --- |
| `tus1-p15-g43` | etcd + router + **prefill** leg | `10.115.43.101` |
| `tus1-p15-g61` | **decode** leg | `10.115.61.101` |

每节点 8× MI325X（gfx942），节点间 8 条 RoCE rail（HCA 命名 `rdma0..rdma7`，
不是常见的 `mlx5_N`）。

两个跟别的集群不一样、必须先知道的点：

1. **`g43` 同时是 RKE2 的 control-plane server 节点**，它的控制面 etcd 已经占了
   2379/2380。PD 自己的 etcd 必须挪开（本文用 12379）。
2. **管理网是 `eno0`/`eno1`，RoCE 是 `tw-ethN`。** 只能对外 advertise `tw-ethN`
   上的地址；advertise 管理网 IP 不会在启动时报错，会在第一次 KV 交接时才失败。

### 1.2 模型权重

`MODEL` 必须是本地目录（脚本以只读方式 bind-mount）：

```bash
export MODEL=/shared_silo/scratch/models/zai-org/GLM-5.2-FP8
```

### 1.3 镜像必须自己 build

节点上预置的 `amdsiloai/amd-internal-luma:infera-sglang-gfx942-glm52` **不能直接用**：
它是 2026-08-04 的，早于 sglang_rocm 的 hicache 补丁（08-11 合入），它的 sglang
树不是本仓库这一份。两个节点上各跑一次：

```bash
cd examples/glm5.2_gfx942 && bash build_image.sh
```

本次用的是 `infera:sglang-gfx942-glm52-5c9cb3f`（tag 带 commit，这样从跑着的容器
可以回溯到源码）。镜像 107 GB，build 约 40 分钟。

---

## 2. 适配集群：`cluster.env`

`env.sh` 拥有调好的配方，不要改它；集群相关的量全部放在 `cluster.env` 里，
在**宿主机和容器内**跑任何脚本之前先 source：

```bash
source cluster.env && bash launch/launch_prefill.sh
```

本集群的 `cluster.env`（已提交在 `examples/glm5.2_gfx942/cluster.env`）：

```bash
export PREFILL_NODE=tus1-p15-g43
export DECODE_NODE=tus1-p15-g61
export PREFILL_IP=10.115.43.101
export DECODE_IP=10.115.61.101

export ETCD_ENDPOINT=10.115.43.101:12379   # 避开 RKE2 控制面 etcd 的 2379

export IB_DEVICE=rdma0                     # 本集群 HCA 叫 rdma0，不是 mlx5_0
export MC_GID_INDEX=3                      # rdma0 上的 RoCE v2 IPv4 条目

export MODEL=/shared_silo/scratch/models/zai-org/GLM-5.2-FP8
export IMAGE=infera:sglang-gfx942-glm52-5c9cb3f
```

`launch_etcd.sh` 原本把 2379/2380 写死了，为了这个集群改成从 `ETCD_ENDPOINT`
推导（peer port = client port + 1）。这个改动是通用的，已一并提交。

---

## 3. RDMA 预检 —— 本次最大的坑在这里

跨节点 PD 每个请求都要把 KV 搬过去。**RoCE GID index 选错不会报任何错，只会慢。**
预检实测（`tools/` 外的 `preflight_rdma.sh`，结果在 `results/preflight/`）：

| 配置 | 单 rail 带宽 | 相对 |
| --- | --- | --- |
| `MC_GID_INDEX=3`（RoCE v2 IPv4） | **8.8 GB/s** | 基准 |
| 不设，让 Mooncake 自己选 | 0.5 – 2.4 GB/s | 慢 **4 – 18×** |
| `MC_FORCE_TCP=1`（退化到 TCP） | 0.08 GB/s | 慢 **110×** |

三种情况下服务都**正常应答、没有任何报错**。所以上线前必须实测，不能靠"能跑通"
来判断。

**而且这个值不能跨集群抄。** 同一批工作里的 MI300X 对照集群（tw041/tw044）上，
`MC_GID_INDEX=3` 指向的是**管理网卡**，照抄过去会让 Mooncake advertise 一个对端
根本到不了的地址。本集群上 3 恰好是 RoCE v2 条目，是查出来的，不是默认值。

找到正确 GID 的方法：

```bash
# 在 rdma0 上找 RoCE v2 且是 IPv4 映射地址的那个 index
for i in $(seq 0 7); do
  t=$(cat /sys/class/infiniband/rdma0/ports/1/gid_attrs/types/$i 2>/dev/null)
  g=$(cat /sys/class/infiniband/rdma0/ports/1/gids/$i 2>/dev/null)
  echo "$i  $t  $g"
done
# 本集群输出：3  RoCE v2  ::ffff:10.115.43.101   <- 就是它
```

跑完整预检（两节点各一个 task，共享 `DUMP_PATH`）：

```bash
bash preflight_rdma.sh
```

需要看到的：容器内可见的 active RDMA port 数等于节点实际的（本集群 8，**不能是 0**），
以及 mooncake 段 `status=OK`。

> 本集群 netperf 有 4 条 warn（g43 nic4–nic7 → g61 方向偏慢，最低 3.1 GB/s）。
> 不影响本次结论，因为配方只用 rail 0，实测 19.6–24.2 GB/s 正常。**KV 只用一条
> rail 是有意的**：实测把 KV 打散到全部 NIC 反而慢 11.9%，KV 只用掉单个
> 200 Gb/s 口的 4.5%。

---

## 4. Docker 臂部署

### 4.1 顺序

```text
g43 宿主机:   host_container.sh  ->  launch/launch_etcd.sh
g61 宿主机:   host_container.sh
g43 容器内:   launch/launch_prefill.sh  ->  launch/launch_router.sh
g61 容器内:   launch/launch_decode.sh
g43 容器内:   verify.sh
```

两条 leg 通过 etcd 互相发现，所以 decode 可以和 prefill 并行启动；router 只要在
接流量之前看到两条 leg 注册上即可。

### 4.2 命令

宿主机（两个节点都做）：

```bash
cd examples/glm5.2_gfx942 && source cluster.env
bash host_container.sh          # 建长期存活的引擎容器，顺带自检镜像/权重挂载
```

只在 g43：

```bash
bash launch/launch_etcd.sh
```

容器内（`docker exec -it infera-glm52-gfx942 bash`），g43：

```bash
cd <repo>/examples/glm5.2_gfx942 && source cluster.env
bash launch/launch_prefill.sh
bash launch/launch_router.sh
```

g61：

```bash
cd <repo>/examples/glm5.2_gfx942 && source cluster.env
bash launch/launch_decode.sh
```

**冷启动 15–25 分钟**（加载权重 + CUDA graph capture）。启动慢不代表卡住，不要 kill，
用 `tail -f logs/prefill.log` 跟。

### 4.3 生效的引擎配置

两条 leg 除了端口和 `--disaggregation-mode` 之外**完全一致**：

```text
--model-path <MODEL>
--tp-size 8 --dp-size 8 --enable-dp-attention
--kv-cache-dtype fp8_e4m3
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--mem-fraction-static 0.85 --max-running-requests 128 --chunked-prefill-size 8192
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--disaggregation-mode {prefill|decode} --disaggregation-transfer-backend mooncake
--disaggregation-ib-device rdma0
--json-model-override-args {"index_share_for_mtp_iteration":false}
```

### 4.4 验证

```bash
bash verify.sh
```

六项检查，每一项都对着一个**不报错的失败模式**：

1. 两条 leg 都在 etcd 里注册了
2. 带已知答案的长 prompt —— KV 交接坏掉不会返回 HTTP 错误，decode 会读到损坏的
   前缀，然后生成一段跟提问无关但很通顺的文本，只有对答案才能抓到
3. kv-aware 路由确实生效（router 日志里 prefill 选择带 `request_blocks > 0`）——
   没有 block hash 时它退化成按负载路由，而且看起来非常健康
4. MTP 生效（decode 的 `/metrics` 里有 `sglang:spec_accept_length`）—— 两条 leg
   的投机配置不一致时会被静默丢弃
5. kvd（默认 `KVD=0` 时跳过）
6. decode 日志里有 Mooncake 传输行

---

## 5. 负载与基准方法

### 5.1 agentic 多轮 trace

正式数字用的是 agentic trace（`cc_traces_100k.json`），不是随机数据集：

| 属性 | 值 |
| --- | --- |
| 会话数 / 轮数（N=60） | 60 / 448 |
| 每轮输入 token | p50 = 68,288，p90 = 90,240，p99 = 98,688，max = 99,904 |
| 理论缓存命中率 | 84.61% |

它是**长上下文 + 强前缀共享**的，这正是 kv-aware 路由和 PD 分离要解决的形态。
仓库里 `bench.sh` 用的 `--dataset-name random` 前缀不共享，缓存命中率天然接近 0，
**那是对的**，但它衡量的是裸吞吐，不能用来看缓存。

评分脚本 `score_agentic_trace.py` 输出的核心指标是**缓存效率**：

```text
efficiency = 实际命中 token / 理想命中 token
```

理想模型是"每轮的输入 = 上一轮的全部内容 + 新增"，即前缀单调增长。效率 100%
意味着**一次驱逐都没发生**，没有任何一轮需要重算本可以复用的前缀。

### 5.2 N 取多少

跑满 N=60 要 90 分钟，调参时太慢。按子集统计做了标定：

| N | 覆盖到的长尾 |
| --- | --- |
| 8 / 15 | p90 只到 ~70k，够不到长上下文尾巴 |
| **20** | p90 = 84,928（是 N=60 的 94%），**够用** |
| 60 | p90 = 90,240，正式数字 |

于是分三档用：

| 档 | 用途 | 耗时 |
| --- | --- | --- |
| random ISL≈68k，conc=1，N=5 | 快速探针，只看单步 decode 延迟 | ~3 min |
| agentic conc=1，N=20 | 参数筛选 | ~10–20 min |
| agentic conc=16，N=60 | 正式数字 | ~9 min（conc=16 并行度高） |

### 5.3 运行

```bash
CONC=16 NUM_PROMPTS=60 \
IMAGE=infera:sglang-gfx942-glm52-5c9cb3f \
MODEL=/shared_silo/scratch/models/zai-org/GLM-5.2-FP8 \
bash tools/run-agentic.sh docker-rust 10.115.43.101:8000 \
     http://10.115.43.101:30001 http://10.115.61.101:31501
```

第一个位置参数是 tag，第二个是 router，其余是跑之前要 `flush_cache` 的引擎 URL。
**flush 是必须的**：上一轮留下的 block 会让命中率虚高，而 `flush_cache` 在有请求
在飞的时候是 no-op，所以必须在空闲时做。

`NUM_PROMPTS` 在这里是**会话数**（60 个会话 = 448 轮 = 448 个请求），不是请求数。

> 用仓库里的 `bench.sh`（随机数据集）时是另一回事：`env.sh` 里
> `NUM_PROMPTS=$((CONC*4))` 是 source 时求值的，同一个 shell 里改 `CONC` 不会重算，
> 需要 `env -u NUM_PROMPTS CONC=32 bash bench.sh` 让它重新推导。

---

## 6. Docker 臂基线结果

`results/agentic/docker-rust_c16_n60.*`，concurrency 16，N=60：

| 指标 | 值 |
| --- | --- |
| 成功请求 | **448 / 448** |
| 缓存效率 | **100.00%**（实际命中 84.61% = 理想 84.61%） |
| 驱逐损失 | 0 token（0.00%） |
| 全部命中来自 | device 层 100% |
| 时长 | 539.6 s |
| 输出吞吐 | 182.7 tok/s |
| Mean / Median TPOT | 21.50 / 18.57 ms |
| Mean / Median TTFT | 12,411 / 3,494 ms |

和 MI300X 上的同一负载基线（`results/baseline-mi300x/`）对齐：缓存效率同为
100.00%，理想命中率 84.69%（数据集切分略有不同）。**MI325X 上没有出现回归。**

---

## 7. Kubernetes 臂部署

分支 `llying/dev/k8s_glm5p2_gfx942`，用 Infera operator 部署同一套 1P1D。

### 7.1 operator 镜像要自己 build

`deploy-k8s.sh` 默认拉 `docker.io/rocm/infera:operator-v0.1.0`，**这个 tag 不存在**
（`rocm/infera` 只有 atom/kvd/server/sglang/vllm 几个变体）。从源码 build：

```bash
docker build -f deploy/operator/Dockerfile -t infera-operator:local .
```

### 7.2 把镜像喂给 containerd

集群是 RKE2，用 containerd，不读 docker 的镜像库。而且这里的账号**没有免密 sudo**，
但**有 kubectl 的 cluster-admin**。所以用一个特权 pod 来做导入，
`tools/import-image-to-containerd.sh`：它挂载宿主机的 docker 和 ctr 两个二进制，
把 `docker save` 直接管道给 `ctr images import`，不落盘中间 tar。

两个前提，缺一个就失败：
- **docker 和 ctr 两个二进制都要挂**（只挂 ctr 时 `docker save` 无从执行）
- 镜像必须**先存在于那台节点的 docker 里**

### 7.3 kubelet 的镜像 GC 会把刚导进去的镜像删掉

107 GB 的引擎镜像导进 containerd、`IMPORT_OK` 之后**两分钟内消失了**。原因：
g43 磁盘 91% 占用，kubelet 的镜像 GC 阈值是 85%，**任何没有 pod 引用的镜像会被
立即回收**，最短存活保护只有 2 分钟。

清了 272 GB 构建缓存仍然到不了 85%（还差约 1.8 TB）。最终的做法是**抢在 GC 之前**：
先把 manifest 渲染好，导入一结束立刻 apply，让 pod 在 2 分钟保护窗口内引用上镜像。

```bash
python3 tools/render-deploy.py --combo disaggregated \
  --prefill-node tus1-p15-g43 --decode-node tus1-p15-g61 \
  --model-dir /shared_silo/scratch/models/zai-org/GLM-5.2-FP8 \
  --rail rdma0 --gid-index 3 --image infera:sglang-gfx942-glm52-5c9cb3f \
  > pd.yaml                      # 先渲染
bash tools/import-image-to-containerd.sh ... && kubectl apply -f pd.yaml   # 导完立刻 apply
```

> `render-deploy.py` 只接受一个 `--gid-index`。两个节点的正确 GID 可能不同，
> 这里恰好都是 3。如果不同，需要渲染后手工分别改 prefill/decode 的 `MC_GID_INDEX`。

### 7.4 服务暴露：ClusterIP 和 NodePort 在这个集群都不通

从宿主机访问 router 时：

- ClusterIP `10.43.x.x:8000` —— **不通**。RKE2 的 kube-proxy 不给 Service VIP
  建宿主机侧的转发规则。
- NodePort —— **也不通**。
- 直接用 **pod IP** —— 通（ClusterIP 的 endpoint 指向的就是这个 pod）。

引擎 pod 本身用 `hostNetwork: true`，所以在 `节点IP:端口` 上可达。

### 7.5 router 侧的连接失败：现象确凿，机理未定

同一个 agentic c16 n60 负载跑了三次才干净：

| 轮次 | 配置 | 结果 |
| --- | --- | --- |
| 1 | manifest 默认：router 8 CPU / 16 GB，pod 网络 | 448 中 **44 失败**，首个失败在第 48 轮 |
| 2 | router 提到 32 CPU / 32 GB | 448 中 **36 失败**，首个失败提前到第 24 轮 |
| 3 | router 改 `hostNetwork: true` | **448 / 448 全成功** |

**能下的结论只有一条**：在这个集群上，router 换到 `hostNetwork` 之后失败清零。除此
之外的因果都还没被证明，下面两条要按"候选原因"读，不是结论：

1. **CPU 配额。** manifest 把 router 写死在 8 CPU，docker 臂能用满 128 核；kv-aware
   路由要对 68k token 的 prompt 做 tokenize，CPU 打满会让 accept 队列堆积。机制上
   讲得通，但 44→36 这个差值和单次运行的抖动同量级，**A/B 本身没有证据力**，而且
   "8 CPU + hostNetwork" 这一格从来没测过。
2. **pod 网络。** 早期版本的本文档把原因写成"flannel overlay MTU 8950 vs 宿主机
   9000 的分片问题"。**这个解释是错的，已推翻**：用 `tools/check-router-hostnetwork.sh`
   实测，本集群 host→pod 走的是 Calico 的 cali veth，路由 MTU 就是 1450，DF ping
   扫描实测通过的最大包也是 1450，路径 MTU 自洽，不存在黑洞。真实原因落在 CNI
   数据面在持续大 body 负载下的某个环节，具体是什么没有查实。

所以有两点要对客户讲清楚：

- **这不是引擎的差异，是部署形态的差异。** 三轮的引擎侧 flag 完全一致。
- **不要把 `hostNetwork: true` 当成 recipe 的默认值。** 同一套 recipe 在一个
  MI300X 集群上没有复现这个问题，说明它跟集群的 CNI/MTU 环境相关。默认值保持中性，
  用下面的脚本按集群判定。

### 7.6 怎么判断自己的集群需不需要 hostNetwork

`tools/check-router-hostnetwork.sh`。因为机理没查实，脚本**不做配置层面的预测，只做
复现**——只有真正跑出失败才算数：

```bash
# 从"你的压测客户端实际所在的位置"运行，vantage point 就是问题本身
tools/check-router-hostnetwork.sh --router-url http://<ROUTER_POD_IP>:8000
```

必须用 **router 的 pod IP**，不能用 Service VIP 或节点 IP——那样就绕开了被测的那段路。

四步，只有第 4 步能定性：

| 步 | 做什么 | 能否定性 |
| --- | --- | --- |
| 1 | 判断客户端在 pod 内还是宿主机上。在 pod 内则整个问题不存在，直接退出 | 是（判"不需要"）|
| 2 | 列 pod veth / overlay 设备的 MTU，并区分目标 pod 是本机还是跨机 | 否，仅参考 |
| 3 | conntrack 水位 + DF ping 扫真实路径 MTU | 否，仅参考 |
| 4 | 往 router 打 16 KB→1 MB 的大 body，再打 16 并发 × 300 KB × 3 轮 | **是** |

第 4 步发的是**故意写坏的 JSON**并填充到指定大小：服务端必须把整个 Content-Length
读完才能报解析失败，所以拿到任何 HTTP 状态码（实测是 400）都说明 body 完整送达 ——
测的是网络不是 API。拿不到响应（curl `000`）才是复现。

退出码：`0` 保持默认，`2` 复现了、改 `hostNetwork`，`1` 没结论（通常是没给
`--router-url`）。

单发很少失败，失败是在持续并发下出现的，所以第 4 步的 b) 并发轮才是重点。脚本判
`2` 时也会提醒：如果 CNI 是你自己维护的，拿着第 2、3 步的信息去查根因，比给 router
开 `hostNetwork`（要占节点 8000 端口、且脱离 NetworkPolicy）更值。

---

## 8. A/B 一致性结论

### 8.1 静态：逐 flag 比对

`tools/compare-docker-vs-k8s.py` 的做法不是肉眼比 shell 脚本和 YAML —— 那种比法
正是错误结论能活下来的原因。它给 docker 侧套一个记录 argv 就退出的 `python3` shim，
**实际跑一遍 launch 脚本**，拿到引擎真正会收到的 argv（包含 `env.sh` 算出来的一切
和脚本走过的每个条件分支）；k8s 侧直接从 manifest 读。

完整输出在 `results/flag-diff.txt`：

| 组件 | 结果 |
| --- | --- |
| prefill | 8 个裸 flag + ~26 个带值 flag 一致，**0 处实质差异** |
| decode | 8 个裸 flag + ~24 个带值 flag 一致，**0 处实质差异** |
| router | ~9 个带值 flag 一致，**0 处实质差异** |

被单独归为"基底差异"、不算差异的项，每一条都附了原因：

| flag | docker | k8s | 为什么必须不同 |
| --- | --- | --- | --- |
| `--advertise-host` | 环境变量里的 IP | `$(POD_IP)` | k8s 用 downward API |
| `--discovery-backend` | `etcd` | `kubernetes` | 基底不同 |
| `--etcd-endpoint` | 有 | 无 | etcd 只存在于 docker 侧 |
| `--disaggregation-ib-device` | 按集群 | 按集群 | rail 名是集群硬件属性 |
| `--host` / `--port` / `--model-path` 等 | — | — | 端口分配和挂载路径 |

唯一实质的环境差异：k8s 侧在两条 leg 上各注入 5 个 Mooncake/NCCL 变量，docker 侧没有：

```text
MC_DISABLE_HIP_TRANSPORT=1   MC_MS_AUTO_DISC=0   MC_MS_FILTERS=rdma0
MOONCAKE_DISABLE_HIP_DMABUF=1   NCCL_IB_DISABLE=1
```

这些是 operator 为了把 Mooncake 钉在单 rail、绕开 HIP transport 而加的，
和 docker 侧靠 `--disaggregation-ib-device` + `MC_GID_INDEX` 达到的是同一个效果。

### 8.2 动态：同一负载

agentic concurrency 16 / N=60，两边都是干净的 448/448：

| 指标 | Docker | Kubernetes | 差异 |
| --- | --- | --- | --- |
| 成功请求 | 448 / 448 | 448 / 448 | — |
| **缓存效率** | **100.00%** | **100.00%** | **0** |
| 驱逐损失 | 0 token | 256 token（0.00%） | 可忽略 |
| 时长 | 539.6 s | 519.4 s | k8s 快 3.7% |
| 输出吞吐 | 182.7 tok/s | 189.8 tok/s | k8s 高 3.9% |
| **Mean TPOT** | 21.50 ms | 21.60 ms | **0.5%** |
| **Median TPOT** | 18.57 ms | 18.34 ms | **1.2%** |
| Mean TTFT | 12,411 ms | 11,762 ms | k8s 快 5.2% |

decode 侧指标（TPOT）差 ≤1.2%，落在运行间噪声里。时长/TTFT 上 k8s 略快，方向和
量级都符合"router 从 8 CPU 提到 32 CPU"带来的排队改善，不是引擎差异。

**结论：operator 部署出来的就是 shell 脚本部署出来的那套引擎。** 后续调优结论在
两种部署方式之间可以互相沿用。

---

## 9. dp-attention 消融

### 9.1 为什么要做这个实验

默认配方在两条 leg 上都开 `--enable-dp-attention`。这个选择是在
**concurrency 16** 上调出来的，当时纯 TP8 prefill 慢 25.9%，所以"并发赢过单请求延迟"。

但 concurrency 1 的画像完全不同，先看 profiling（`tools/profile-decode.sh` +
`tools/analyze-trace.py`，trace 在 `results/prof/`）：

| 观察 | 数值 | 说明 |
| --- | --- | --- |
| GPU busy | 79–86% | **不是空转，也不是启动开销** |
| GEMM 占比 | 54–58% | 时间主要在矩阵乘 |
| 最热 kernel | o_proj（MLA 输出投影），413 µs/次 × 1,185 次 | — |
| 该 kernel 的 CU 占用 | 304 个 CU 里只用了 48 个（**16%**） | 形状不匹配 |
| 集合通信占比 | 6.4%（rank 2）– 13.3%（rank 0） | **不是通信瓶颈** |
| attention 本身占比 | 0.6 – 0.7% | **完全不是 attention 的问题** |
| 上下文长度影响 | 1k → 68k，TPOT 只变 2.3% | **不是带宽瓶颈**，是固定的每步开销 |

完整输出在 `results/prof/dpon-analysis.txt`。两个 rank 的差别本身就是证据：
rank 0 比 rank 2 多出 73.8 ms 的 all-reduce，来自 45 次 1,639 µs 的 LL 归约 ——
这正是"有活的 rank 走 graph replay、空转的 rank 走 eager"造成的等待。

关键在于 **DP-attention 下每个 rank 跑的是完整未切分的 attention 权重**。
concurrency 1 + DP=8 时只有 1 个 rank 有真实请求，但 8 个 rank 都要跑完整的 78 层
前向来参与 MoE 的 all-reduce。o_proj 的 K=16384（完整，未切分）是没调过的形状，
而纯 TP8 下 K=2048 是调过的。CU 占用 16% 正是这么来的：batch 维只有 6，
而 kernel 按 M=128 的 block tile 切。

所以假设是：**concurrency 1 下 dp-attention 是净负担**。

> **本节只在 docker 臂上实测。** k8s 臂在做消融之前已经拆掉，以便把 16 张 GPU 交回
> docker 臂，重新部署一轮的代价（镜像重新导入 + 冷启动）超出了本次预留窗口。
>
> 结论可以沿用到 k8s 臂，依据是 §8：两臂 0 处实质 flag 差异，同一负载 decode 侧
> 指标差 ≤1.2%。dp-attention 是引擎侧参数，而两臂的引擎已被证明是同一套。
> 要在 k8s 上确认，把 manifest 里两条 leg 的 `--dp-size` 改成 `1`、去掉
> `--enable-dp-attention`、`--chunked-prefill-size` 改成 `2048`，跑同一条
> `tools/run-agentic.sh` 即可 —— 预期与 §10.6 的 docker 数字落在同一区间。

### 9.2 怎么做的

把 dp-attention 从写死改成跟随 `DP`：

```bash
# launch_prefill.sh / launch_decode.sh
ATTN_ARGS=(--tp-size "$TP" --dp-size "$DP")
[[ "$DP" -gt 1 ]] && ATTN_ARGS+=(--enable-dp-attention)
```

这样 `DP=1` 就是纯 TP8，`DP=8`（默认）行为不变。消融时两条 leg 都用 `DP=1` 重启。

### 9.3 结果

**随机定长探针**（`results/probe/`，concurrency 1，N=5）：

| 输入长度 | dp-attention 开 | 关 | 加速 |
| --- | --- | --- | --- |
| 1,024 | 22.77 ms → 43.9 tok/s/user | — | — |
| 4,096 | 22.73 ms → 44.0 tok/s/user | **9.39 ms → 106.5** | **2.42×** |
| 68,288 | 23.52 ms → 42.5 tok/s/user | **9.76 ms → 102.5** | **2.41×** |

注意开着 dp-attention 时 TPOT 几乎不随上下文变化（22.7 → 23.5 ms，1k 到 68k），
这本身就说明瓶颈是每步的固定开销，不是 KV 的读带宽。

**agentic 真实负载**（`results/agentic/docker-dp{on,off}_c1_n20.*`，concurrency 1，N=20）：

| 指标 | dp-attention 开 | 关 | 改善 |
| --- | --- | --- | --- |
| Mean TPOT | 18.63 ms | **8.18 ms** | **2.28×** |
| Median TPOT | 16.00 ms | **7.54 ms** | **2.12×** |
| P99 TPOT | 32.24 ms | **13.47 ms** | 2.39× |
| **Mean tok/s/user** | 53.7 | **122.2** | **2.28×** |
| **Median tok/s/user** | 62.5 | **132.6** | **2.12×** |
| Mean TTFT | 4,445 ms | **2,459 ms** | 1.81× |
| 时长 | 1,176.6 s | **586.8 s** | 2.00× |
| 缓存效率 | 99.98% | **100.00%** | 持平 |
| MTP 接受长度 | 3.65–3.71 | 3.61–3.68 | 持平 |

**中位 132.6 tok/s/user 已经进入 130–180 的目标区间**，均值 122.2 略低于下限。
缓存效率和 MTP 接受长度都没有退化，说明这是纯粹的算子效率收益，没有拿正确性换。

### 9.4 关掉 dp-attention 会连带改变 `CHUNK` 的含义

`CHUNK` 是**全局预算**，而 SGLang **只在 dp-attention 开着时**才把它按 `dp_size` 切分：

| 配置 | `CHUNK` | 每 rank 实际吃进的 token |
| --- | --- | --- |
| `DP=8` + dp-attention | 8192 | 1,024（被除以 8） |
| `DP=1`，dp-attention 关 | 8192 | **8,192**（不除） |
| `DP=1`，dp-attention 关 | 1024 | 1,024（全局预算被砍到 1/8） |

所以直接照搬 `CHUNK=8192` 到 `DP=1`，每个 rank 要一次扛下完整 8,192 token 的激活，
是原来的 8 倍。`MEM_FRAC=0.85` 只留了 15% 给激活，concurrency 1 时侥幸不炸，
**concurrency 16 的长 prefill 直接 OOM**：

```text
HSA_STATUS_ERROR_OUT_OF_RESOURCES
448 个请求里 417 个失败："no active mixed worker for model=..."
```

本次用 `CHUNK=1024` 修好了 OOM（448/448，见下），代价是全局 prefill 预算只剩 1/8，
TTFT 明显变差。**这是一个折中，不是最优值**：真正要调的是在不 OOM 的前提下把
`CHUNK` 尽量抬高，或者反过来降 `MEM_FRAC` 给激活腾地方。两者都会动 KV 池大小，
所以要连着缓存效率一起看。

引擎日志里那句 `chunked prefill size adjusted from ... to ...` 读起来像是设置被拒绝了，
其实只是被除了。上线前回读一次实际生效的值：

```bash
curl -s $PREFILL_URL/get_server_info | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d["chunked_prefill_size"], "per rank x", d["dp_size"], "ranks")'
```

### 9.5 高并发下的代价：dp-attention 必须留着

把关掉 dp-attention 的配置放回 concurrency 16 跑完整 N=60
（`results/agentic/final-dpoff_c16_n60.*`，`CHUNK` 用 §10.6 选定的 2048）：

| 指标 | dp-attention 开 | 关 | 变化 |
| --- | --- | --- | --- |
| 成功请求 | 448 / 448 | 448 / 448 | 都干净 |
| 缓存效率 | 100.00% | 100.00% | 持平 |
| 时长 | 539.6 s | 707.3 s | **慢 31.1%** |
| 输出吞吐 | 182.7 tok/s | 139.4 tok/s | **低 23.7%** |
| Mean TTFT | 12,412 ms | 22,161 ms | **差 1.79×** |
| Mean TPOT | 21.50 ms | 9.05 ms | **好 2.38×** |
| Median TPOT | 18.57 ms | 8.00 ms | 好 2.32× |

这是一个非常干净的权衡，也是本次最该讲给客户的一条：

- **要聚合吞吐（batch 场景）** → dp-attention 开着，`CHUNK=8192`
- **要单用户速度（交互场景）** → dp-attention 关掉，`CHUNK=2048`

注意 TTFT 那 1.79× 里有一部分是 `CHUNK` 从 8192 降到 2048 的账，不全是 dp-attention
的，所以这一列是"关掉 dp-attention 并把 `CHUNK` 调到不 OOM 的最大值之后"的综合结果，
不是 dp-attention 单变量的代价。

---

## 10. 低并发调优

目标是 concurrency 1 下 130–180 tok/s/user。起点是默认配方的 53.7，
§9 关掉 dp-attention 之后到 122.2（中位 132.6）。这一节是在那之后继续找的。

### 10.1 关掉 dp-attention 之后瓶颈换人了

同一套 profiling 工具，在 dp-off 的服务上重跑一遍（`results/prof/dpoff-analysis.txt`），
和 §9.1 的 dp-on 画像对比：

| | dp-attention 开 | 关 |
| --- | --- | --- |
| GPU busy | 79 – 86% | **63%** |
| GEMM 占比 | 54 – 58% | 28% |
| 最热 kernel | 单个 GEMM 413 µs × 1,185 次 | 无单点，最热的是 all-reduce 24.6 µs × 2,400 次 |
| 集合通信 | 6.4 – 13.3% | 14.7% |

原来那个 16% CU 占用的 o_proj 巨型 GEMM 消失了（K 从 16384 变回 8 路切分后的 2048），
**新瓶颈不再是某个算子，而是 GPU 有 1/3 的时间在空转**。

### 10.2 空转在哪里

`tools/analyze-gaps.py` 把每一段 GPU 空隙归因到覆盖它的 CPU 算子
（`results/prof/dpoff-gaps.txt`）：

```text
span 547.7 ms   busy 419.0 ms (76.5%)   idle 128.7 ms (23.5%)
gaps >= 20 us: 725, totalling 123.6 ms

        ms   count    us/gap  covering CPU op
     53.64      12    4470.1  hipGraphLaunch
     20.68     245      84.4  step[DRAFT_EXTEND_V2 bs=1]
      9.20       2    4602.2  step[TARGET_VERIFY bs=1]
```

两类，形状完全不同：

1. **每个 decode step 一次 ~4.5 ms 的停顿**（15 步、15 次），合计约 12% 的 span。
   PD 的 decode leg 每个调度迭代都要 `poll_and_all_reduce` —— 跨 8 个 rank 同步
   轮询 KV 传输队列的状态。这是 PD 分离架构固有的同步点，**没有开关**。
2. **draft 阶段每步十几次 ~84 µs 的小空隙**，合计 20.7 ms。draft 是 `bs=1` 的
   链式展开，5 次串行前向，每次之间都有一小段主机侧空档。

第 2 类指出了一条路：**GPU 既然空着 1/4 到 1/3，就应该让投机更宽而不是更深** ——
加宽（topk > 1 的树状 draft）用的是并行度，加深（更多 steps）用的是串行时间。

### 10.3 试过但走不通的：打开 custom all-reduce

profiling 里最热的 kernel 是 2,400 次小 all-reduce，而两条 leg 都带着
`--disable-custom-all-reduce`。看起来是个现成的靶子 —— 小消息、低并发，
正是 custom all-reduce 相对 RCCL 占便宜的区间。

**不能开。** 仓库里两处独立记录了原因：

> `--disable-custom-all-reduce` is on by default and **independent of MTP**, because
> the custom all-reduce kernel deadlocks on this architecture during speculative
> verify.
> —— `examples/sglang_1p1d_glm5.2/cluster/README.md`

> All runs used `--disable-custom-all-reduce` (required on gfx942/gfx950 for EAGLE),
> so the custom all-reduce path is unexercised.
> —— `deploy/docker/patches/sglang_dsa/README.md`

aiter 的 custom all-reduce kernel 在 gfx942/gfx950 上做 EAGLE verify 时会**死锁**。
开了不会报错，会挂住 —— 在冷启动要 15 分钟的场景里，这是最贵的一种失败。
只有当镜像带了对应修复时才可以试。

### 10.4 加宽投机：被 DSA 后端否掉

按 §10.2 的思路，第一个要试的是 `--speculative-eagle-topk 2`（树状 draft，
用并行度换接受长度）。**这条路在这个模型上走不通**，SGLang 启动时直接拒绝：

```text
ValueError: speculative_eagle_topk > 1 with page_size > 1 is only supported on
('flashinfer', 'fa3', 'triton'); got attention_backend='dsa'.
Use page_size == 1 or one of those backends.
```

GLM-5.2 走的是 DSA attention backend，本部署 `page_size=64`，两个条件都不满足。
换 attention backend 等于放弃 DSA，换 `page_size=1` 会重排 KV 布局，
都不是为了一个投机参数值得付的代价。

**所以 GLM-5.2 + DSA 上的投机只能是链式的（topk=1），只有深度一个自由度。**
好消息是这个失败很便宜：参数校验在启动后约 70 秒就报错，不用等完整冷启动。
`tools/tune-cycle.sh` 里的存活检查就是为此加的 —— 否则就绪轮询会在那里干等 30 分钟。

### 10.5 加深投机链

既然只剩深度，先从 §10.2 的 trace 里把成本量出来，而不是盲扫。给 profiler 的
标注按名字聚合：

```text
   398.80 ms   30 calls   13.293 ms/call   step[TARGET_VERIFY bs=1]
   136.41 ms   30 calls    4.547 ms/call   step[DRAFT_EXTEND_V2 bs=1]
```

每个 MTP step 里，target verify 13.29 ms，整条 5 步 draft 链 4.55 ms
（合 **0.91 ms/步**）。接受率从 decode 日志读：`accept len` 和 `accept rate` 满足
`accept_rate = (accept_len - 1) / steps`，实测 0.42–0.59。按几何接受模型反解，
逐 token 接受概率 q ≈ 0.79，能复现 `steps=5 → accept_len 3.61–3.71` 的实测值。

于是 `TPOT(k) = (固定开销 + 0.91k) / accept_len(k)`，而结论**完全取决于把"固定开销"
取多大**：

| 固定开销取值 | 依据 | 预测最优深度 | 相对 k=5 |
| --- | --- | --- | --- |
| 13.3 ms | 只算 profiler 标注（TARGET_VERIFY） | k = 5 | 0% |
| 29.4 ms | 用实测 ITL 33.9 ms 反推 | k ≈ 8–10 | +6 – 7% |

差别在于那 16 ms 的每步额外开销（§10.2 的 4.5 ms 轮询停顿 + 主机侧空档）里，
有多少是随深度增长的。**这个靠推算分不出来，只能实测**，所以跑了 `k=8`
（`MTP_STEPS=8 MTP_TOPK=1 MTP_DRAFT_TOKENS=9`），两组都在 `CHUNK=2048` 上，
agentic conc=1 N=20：

| | `k=5`（配方值） | `k=8` | 变化 |
| --- | --- | --- | --- |
| accept len | 4.32（上限 6） | 5.65（上限 9） | **+30.8%** |
| Mean ITL | 33.64 ms | 44.66 ms | **+32.8%** |
| **Mean TPOT** | **7.52 – 8.02 ms** | 8.48 ms | **差 6 – 13%** |

> accept len 取的是 decode 日志里 `#running-req: 1` 的批次的均值。**必须按并发过滤** ——
> 直接对整个日志求均值会把 concurrency 16 的批次混进来，得到的对比是两个变量的。

接受长度确实涨了，但每步时间涨得更多，净亏。3 步 draft 多花 11.1 ms，
**边际成本 3.70 ms/步** —— 而 profiler 的 `DRAFT_EXTEND_V2` 标注里只看得见
0.91 ms/步。差出来的约 2.8 ms/步是随深度增长的主机侧空档，所以上表两个假设里
**"固定开销 13.3 ms"那一行是错的**：真正不随深度变的部分只有 15.3 ms。

用两个实测端点重新拟合（`ITL(k) = 15.3 + 3.67k`，逐 token 接受概率 q ≈ 0.87 能同时
复现 `k=5 → 4.32` 和 `k=8 → 5.65`），最优点是一段以 `k=5` 为中心的平底：

| k | 预测 accept | 预测 TPOT | 相对 k=5 |
| --- | --- | --- | --- |
| 3 | 3.28 | 8.02 ms | −3.4% |
| 4 | 3.85 | 7.79 ms | −0.4% |
| **5** | **4.32** | **7.76 ms** | **最优** |
| 6 | 4.74 | 7.86 ms | −1.3% |
| 8 | 5.53 | 8.13 ms | −4.8%（实测 8.48，同向） |

**结论：配方里的 `5/1/6` 在 concurrency 1 上已经落在最优点，深度不是杠杆。**
根本原因是接受长度饱和得比每步成本增长得慢 —— 每多一步 draft 固定要 3.67 ms，
而它换回来的 token 越来越少。曲线在 `k=4..6` 之间很平（相差 <1.5%），
所以这个结论对模型细节不敏感。

### 10.6 最终配方与达成情况

调优只有一个开关是有效的，就是 §9 的 dp-attention。其余三条路都被证伪了：
custom all-reduce 会死锁（§10.3）、树状 draft 后端不支持（§10.4）、
加深 draft 链净亏（§10.5）。

交互场景的最终配方（在默认 `env.sh` 之上只改两个量）：

```bash
DP=1 CHUNK=2048 bash launch/launch_prefill.sh    # DP=1 时 dp-attention 自动关闭
DP=1 CHUNK=2048 bash launch/launch_decode.sh
```

`CHUNK` 取 2048 而不是 §9.4 里那个先修 OOM 用的 1024。在 concurrency 16 上
两者都不 OOM，2048 全面更好：

| `CHUNK`（`DP=1`） | 时长 | 输出吞吐 | Mean TTFT | Mean TPOT | 缓存效率 |
| --- | --- | --- | --- | --- | --- |
| 1024 | 841.2 s | 117.2 tok/s | 26,636 ms | 9.07 ms | 100.00% |
| **2048** | **707.3 s** | **139.4 tok/s** | **22,161 ms** | 9.05 ms | 100.00% |
| 8192 | OOM，417/448 失败 | — | — | — | — |

**最终达成（agentic conc=1，N=20）：**

| 配置 | Mean TPOT | **均值 tok/s/user** | Median TPOT | **中位 tok/s/user** |
| --- | --- | --- | --- | --- |
| 出厂配方（dp-attention 开） | 18.63 ms | 53.7 | 16.00 ms | 62.5 |
| `DP=1 CHUNK=8192` | 8.18 ms | 122.2 | 7.54 ms | 132.6 |
| `DP=1 CHUNK=2048` 第 1 次 | 7.52 ms | 133.0 | 6.48 ms | 154.3 |
| `DP=1 CHUNK=2048` 第 2 次 | 8.02 ms | 124.7 | 7.51 ms | 133.2 |
| `DP=1 CHUNK=2048` 第 3 次 | 7.30 ms | 137.0 | 6.29 ms | 159.0 |

最终配方跑了三次，**中位 133 – 159 tok/s/user，均值 125 – 137**，相对出厂配方
2.3 – 2.6×。缓存效率三次都是 100.00%，accept len 也没有退化（4.29 → 4.32），
说明收益不是拿正确性或缓存换来的。**在长上下文工作点上区间更窄也更高，见 §10.7。**

**为什么要报区间而不是单个数：** `CHUNK=2048` 的第一次测量看起来比 `CHUNK=8192`
好 8%（8.18 → 7.52 ms）。但 `CHUNK` 只影响 prefill leg 的分块，**理论上不应该影响
decode 侧的 TPOT** —— 正是这个矛盾促使补测。第二次 8.02 ms，第三次 7.30 ms，
那 8% 落回了噪声里。三次的 Mean ITL 几乎相同（33.55 / 33.72 / 33.75 ms），
accept len 汇总后 4.32、与 `CHUNK=8192` 的 4.29 一致，
**证实 `CHUNK` 对 decode 没有影响，理论和实测对上了**。

抖动来自 accept len 的 run 间波动。这条对客户很重要：
**在这个负载上单点 conc=1 测量有约 ±5% 的抖动，任何小于 10% 的调参结论都必须重复测量。**
本次能识破，靠的是先有"CHUNK 不该影响 decode"的理论预期，
再拿它去质疑一个看起来不错的结果 —— 而不是先看到好数字再去找解释。

### 10.7 长上下文反而更快 —— 而且原因很干净

均值和中位数差这么多（125–137 vs 133–159），值得拆开看。把每个请求按它这一轮的
上下文长度分桶（用 `cached_tokens` 作为上下文代理，`input_lens` 在多轮模式下是
会话级的、不可用）。下表是第 1 次测量：

| 上下文 | n | accept len | Mean ITL | **tok/s/user（中位）** |
| --- | --- | --- | --- | --- |
| < 20k | 20 | 3.31 | 33.1 ms | 98.7 |
| 20 – 50k | 16 | 4.08 | 33.0 ms | 124.6 |
| **50 – 80k** | **88** | **5.50** | **33.8 ms** | **162.5** |
| ≥ 80k | 14 | 5.72 | 34.5 ms | **165.0** |

**每步时间几乎不随上下文变化（33.1 → 34.5 ms，+4%），速度的全部变化都来自 MTP
接受长度（3.31 → 5.72，+73%）**，而 tok/s/user 几乎精确地跟着它走（+67%）。
三次测量的长上下文桶分别是：

| 桶 | 第 1 次 | 第 2 次 | 第 3 次 |
| --- | --- | --- | --- |
| ctx 50 – 80k（n=88） | 162.5 | 153.3 | 164.1 |
| ctx ≥ 80k（n=14） | 165.0 | 156.0 | 158.8 |

长上下文桶的抖动（153 – 164）比全体中位（133 – 159）小得多 —— 短开场轮的
accept 更不稳定，是 run 间波动的主要来源。

原因是内容而不是系统：长对话的后续大量是重复的工具调用 JSON，draft 模型猜得准，
一步就能提交 5–6 个 token；而会话的开场轮内容最"新"，接受长度只有 3.3。

三条推论：

1. **目标在客户真正关心的工作点上是达成的。** 这条 trace 的 p50 输入是 68k，
   落在 50–80k 桶里，中位 **153 – 163 tok/s/user**，稳稳在 130–180 区间内。
   拉低均值的是那些短的开场轮，不是长上下文。
2. **§9.3 的随机数据探针低估了性能。** 随机 token 无法预测，接受长度只有 3.6，
   量出来约 102 tok/s/user；同一套引擎跑真实长上下文文本是 162。
   **随机数据集适合测每步时间，不适合报 tok/s/user。**
3. **上下文长度不是 decode 的成本来源。** 68k 上下文和 4k 上下文的每步时间差
   4%，说明 KV 读带宽远没有成为瓶颈 —— 这和 §10.2 "瓶颈是空转不是带宽"一致。

---

## 11. 清理

容器内，两个节点：

```bash
bash stop.sh
```

宿主机：

```bash
bash host_container.sh --rm
docker rm -f infera-glm52-etcd          # 只在 g43
```

**重启前一定要先清掉引擎进程**，否则下一次启动会撞上上一次还占着的 VRAM 而 OOM。

k8s 臂：

```bash
kubectl -n infera delete inferadeployment glm52-fp8-pd
```

删除是异步的，等 `kubectl -n infera get pods` 空了再去起 docker 臂，否则两边抢 GPU。

---

## 12. 坑清单

1. **RoCE GID index 选错只是慢，不报错** —— 慢 4–110×，见 §3。上线前必须实测。
2. **`CHUNK` 是全局预算，且只在 dp-attention 开着时才按 `dp_size` 切** —— 见 §9.4。
   改 `DP` 而不改 `CHUNK`，concurrency 高时会 OOM。日志里的
   `adjusted from ... to ...` 是"被除了"，不是"被拒绝了"。
3. **不能用别人预置的镜像** —— 节点自带的那个早于 hicache 补丁，见 §1.3。
4. **`NUM_PROMPTS` 在 source 时就求值了** —— 同一 shell 里改 `CONC` 不会重算，
   每次跑之前 `env -u NUM_PROMPTS`，见 §5.3。
5. **advertise 必须是数据网 IP** —— 管理网地址不会在启动时报错，会在第一次 KV
   交接时才失败，而注册发生在权重加载**之后**，所以发现这个错误的代价是一整轮冷启动。
6. **`Ctrl-C` 掉 `tail -f` 不会停引擎** —— 启动脚本用 `nohup`，要用 `stop.sh`。
7. **kill benchmark 要连容器一起** —— `pkill -f run-agentic.sh` 只杀了外层 wrapper，
   `docker run` 和里面的 `bench_serving` 还在跑。用
   `docker ps --no-trunc --format '{{.Names}}|{{.Command}}'` 找到再 `docker rm -f`。
8. **扫并发时每档要换 seed 并清缓存** —— 固定 `SEED=42` 而 `NUM_PROMPTS` 随并发增长时，
   高并发档的 prompt 集合是低并发档的超集，缓存命中率会假高到 ~50%。
   每档 `SEED=$((1000+C))` 并在两条 leg 上 `flush_cache` 之后，命中率回到 1.3%。
9. **kubelet 会秒删刚导入的镜像** —— 见 §7.3。
10. **k8s router 在长上下文负载下出现过连接失败，换 `hostNetwork` 后清零** —— 见
    §7.5。机理没查实，且换个集群没复现，所以别改 recipe 默认值；用 §7.6 的
    `check-router-hostnetwork.sh` 按集群判定。
11. **RKE2 上 ClusterIP / NodePort 从宿主机不可达** —— 见 §7.4，用 pod IP。
12. **etcd 端口可能被控制面占了** —— g43 是 RKE2 server，2379/2380 已占用，见 §2。
13. **`--disable-custom-all-reduce` 不能关** —— gfx942/gfx950 上 aiter 的 custom
    all-reduce kernel 在 EAGLE verify 时**死锁**，见 §10.3。它是 profiling 里最显眼的
    靶子，也是最贵的一次误判：不报错，直接挂住。
14. **DSA backend 上 `--speculative-eagle-topk` 只能是 1** —— `page_size > 1` 时
    树状 verify 只支持 flashinfer/fa3/triton，见 §10.4。
15. **`flush_cache` 在有请求在飞时是 no-op** —— 两个 benchmark 撞在一起时，后一个的
    flush 静默失效，跑出来缓存效率 101%、开场轮就从 device 命中 25 万 token。
    效率**超过** 100% 就是这个信号，跑之前先确认队列是空的。
16. **`set -e` 的脚本里 `curl` 探活要加 `|| true`** —— 端口没起来时 curl 退出码 7，
    而 `p=$(curl ...)` 会让整个脚本静默退出。就绪轮询脚本最容易踩这个。
17. **rail 挂掉时报出来的是 GPU 显存错误，不是传输错误** —— 这条在 MI300X 对照集群上
    花了半小时。指定的 rail 在一侧 DOWN 时，decode leg 会正常加载权重、分配 KV、
    起 uvicorn，然后在 PD warmup 阶段死于 `Memory access fault by GPU node-3`。
    真正的原因在几千行之前的三行 info 日志里：

    ```text
    topology.cpp:93]  <rail>:1 is not active (state: 1)
    topology.cpp:135] Device <rail> has no active ports, skipping
    topology.cpp:224] Skipping unavailable device: <rail>
    ```

    Mooncake 把唯一给它的 rail 跳过了，最后没有可用设备，故障以显存错误的形式浮出来。
    选 rail 时要确认它在**两侧都 ACTIVE**。

---

## 13. 目录内容

```text
results/
  agentic/          agentic trace 的全部运行，每次三个文件（.log / .jsonl / .score.txt）
  probe/            concurrency 1 定长探针
  prof/             profiler 分析输出（原始 trace 太大，留在 temp/exec-logs/prof/）
  preflight/        RDMA 预检输出（含 HTML 报告）
  enginelogs/       各配置 decode leg 的 "Decode batch" 行，用于统计 accept len
  baseline-mi300x/  MI300X 对照基线
  flag-diff.txt     docker vs k8s 逐 flag 比对完整输出
  *.yaml            实际用过的 k8s manifest（含三轮 router 修改）
tools/
  run-agentic.sh                 跑 agentic trace（tag / router / flush URL）
  tune-cycle.sh                  一次完整调参迭代：重启两条 leg -> 等就绪 -> 打分
  wait-and-bench.sh              tune-cycle 的后半段，冷启动已在进行时接管
  probe-conc1.sh                 concurrency 1 定长延迟探针
  profile-decode.sh              抓 decode leg 的 profiler trace
  analyze-trace.py               按 kernel 类别拆解 trace（"是哪个算子"）
  analyze-gaps.py                把 GPU 空隙归因到 CPU 算子（"在等什么"）
  compare-docker-vs-k8s.py       逐 flag 静态比对
  render-deploy.py               渲染 k8s manifest
  import-image-to-containerd.sh  无 sudo 时把镜像导入 containerd
  check-router-hostnetwork.sh    判定本集群 router 要不要 hostNetwork（§7.6）
```

`results/agentic/` 里的 tag 命名对应本文各节：

| tag | 对应 |
| --- | --- |
| `docker-rust_c16_n60` | §6 docker 基线 |
| `k8s-rust*_c16_n60` | §7.5 三轮 k8s（`_c16_n60` / `-cpu32` / `-hostnet`） |
| `docker-dpon_c1_n20` / `docker-dpoff_c1_n20` | §9.3 dp-attention 消融 |
| `docker-dpoff_c16_n60` | §9.4 `CHUNK=8192` 的 OOM 现场 |
| `docker-dpoff-chunk1024_c16_n60` | §9.5 / §10.6 `CHUNK=1024` |
| `mtp8-chunk2048_c1_n20` | §10.5 加深 draft 链 |
| `final-dpoff_*` / `final-dpoff-rep2_*` | §10.6 最终配方与复测 |

### 复现整条调优链

```bash
# 一次迭代 = 重启两条 leg + 等就绪 + agentic 打分
TAG=myrun DP=1 CHUNK=2048 MTP_STEPS=5 MTP_TOPK=1 MTP_DRAFT_TOKENS=6 \
  CONC=1 NUM_PROMPTS=20 bash tools/tune-cycle.sh
```

冷启动首次约 16 分钟；权重进了 page cache 之后约 4 分钟加载 + 图捕获。
非法参数组合在约 70 秒内就会被 SGLang 拒绝，脚本的存活检查会立刻报出来。
