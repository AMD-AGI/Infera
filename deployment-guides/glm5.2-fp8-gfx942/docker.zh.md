# GLM-5.2-FP8 双节点 gfx942 —— Docker + shell 部署

在两台 gfx942 节点（MI300X / MI325X）上做 GLM-5.2-FP8 的 prefill/decode 分离
（1P1D）：SGLang TP8 + DP-attention、MTP 投机解码、fp8 KV 缓存、Mooncake RDMA 搬运
KV、Infera kv-aware 路由。部署方式是每个节点起一个长期存活的容器，用 shell 脚本拉起。

同一套部署的 Kubernetes 版本见 [`kubernetes.zh.md`](kubernetes.zh.md)。两者跑出来的
是同一个引擎——做过逐 flag 的静态比对和同一负载的动态比对——所以在任一边调出来的结论
可以直接沿用到另一边。

以下命令都在仓库的 `examples/glm5.2_gfx942/` 目录下执行。

---

## 0. 先看结论

在 2× MI325X 上用多轮 agentic 负载实测（60 个会话、448 个请求、输入中位 68k token）：

| | 批量 / 高并发 | 交互 / 低并发 |
|---|---|---|
| 配置 | 默认（`DP=8`，DP-attention 开，`CHUNK=8192`） | `DP=1 CHUNK=2048` |
| 并发 16 聚合吞吐 | **182.7 tok/s** | 139.4 tok/s |
| 并发 1 单用户速度（中位） | 62.5 tok/s/user | **133 – 159 tok/s/user** |
| 并发 16 平均 TTFT | 12.4 s | 22.2 s |
| 缓存效率 | 100.00% | 100.00% |

同一份权重、同一个镜像，只差一个开关。**DP-attention 是一个并发档位的取舍，不是"开着
总是更好"的优化**——切换方法见 §8。请先确定你要服务的是哪一列，再谈其他调优：本次实测
中没有任何其他参数的影响接近这个量级。

冷启动每个节点 15–25 分钟，加载权重期间日志几乎不输出。**启动慢不代表卡住，不要 kill。**

---

## 1. 前置条件

**硬件。** 两个节点，各 8× gfx942；prefill 节点（同时跑 etcd 和 router）需要约
272 GiB 主机内存。两节点之间需要 RoCE 网络：KV 交接走 RDMA，没有可用的 TCP 退路。

**先在两个节点上读主机驱动版本。** 它决定你用哪个 base 镜像，而且选错不会拒绝启动：

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'
```

容器自带 ROCm 用户态，但内核驱动只能用宿主机的——它通过 `/dev/kfd` 访问宿主机的
amdgpu，而 AMD 只保证一个有限窗口内的搭配：

| 主机驱动 | 支持的 ROCm 用户态 | 用哪个 base |
|---|---|---|
| 6.4.x | ≤ 7.2.x | `rocm720`，Dockerfile 默认值 |
| 6.3.x | ≤ 7.0.x | `rocm700`，**必须显式覆盖** |

搭配错了的表现是：镜像正常初始化、加载权重、捕获 graph，然后在负载下的某处报
`Memory access fault by GPU node-N`，而且每次落点都不一样。两行 `dpkg -l` 比重新
build 一次便宜得多。

**模型权重。** 两个节点上路径相同的本地目录，共享文件系统或各自下载都行。可以是
HuggingFace snapshot 路径或指向它的软链接——`host_container.sh` 会识别软链接并把仓库
根目录一起挂进去，这样 snapshot 里指向 `blobs/` 的内部链接才不会断。

```bash
hf download zai-org/GLM-5.2-FP8
```

---

## 2. Build 镜像

**不要用厂商预置的 GLM-5.2 镜像。** GLM-5.2 在 gfx942 的 SGLang base 上需要重新编译的
Mooncake 和四个 SGLang 源码补丁；预置镜像如果早于其中任何一个，失败现象都不会指向真正
的原因。在两个节点上各跑一次（或者 build 一次推到两边都能拉的 registry）：

```bash
bash build_image.sh                                        # 6.4.x 主机驱动
SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  bash build_image.sh                                      # 6.3.x 主机驱动
```

镜像约 107 GB，build 约 40 分钟。四个补丁里有三个会留下标记，这是判断"补丁真的打上了"
最省事的办法——补丁的锚点位置变动时会静默地打不上：

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/check_image.py:/check.py:ro" infera:sglang-gfx942-glm52 /check.py
```

三行都是 `YES` 才算过：

```text
sglang_rocm/host_alloc       srt/mem_cache/pool_host/common.py            YES
sglang_rocm/staged_wb        srt/mem_cache/pool_host/mla.py               YES
sglang_disagg/early_send     srt/disaggregation/mooncake/conn.py          YES
```

别人给你的镜像也用这条命令查。已验证节点上预置的那个 GLM-5.2 镜像，第一行是 `no`——
这正是本文让你自己 build 的原因。

如果你会 build 多次，tag 里带上源码 commit：这样从跑着的容器可以回溯到它是从哪棵树
build 出来的。

---

## 3. 配置

所有与集群相关的值都在一个文件里，**两个节点**都要做：

```bash
cp cluster.env.example cluster.env    # 然后编辑
```

`env.sh` 会先读它、再套自己的默认值，而每个脚本都 source `env.sh`，所以没有别的文件
需要改，调用时也不用记得先 source 什么。仓库在容器内挂载在同一路径，所以宿主机和容器
共用这一份。

| 变量 | 含义 |
|---|---|
| `PREFILL_NODE` / `DECODE_NODE` | 主机名。prefill 节点同时跑 etcd 和 router。 |
| `PREFILL_IP` / `DECODE_IP` | 它们在**数据网**上的地址，不是管理网卡。 |
| `ETCD_ENDPOINT` | PD 自己的 etcd 监听地址。没被占用就用 `2379`。 |
| `IB_DEVICE` | Mooncake 搬 KV 用的 RoCE rail —— `mlx5_0`、`rdma0` 等。 |
| `MC_GID_INDEX` | 该 rail 上的 RoCE v2 GID index（§4）。 |
| `MODEL` | 本地权重目录。 |
| `IMAGE` | §2 build 出来的镜像。 |

`cluster.env.example` 对每个值都给了"怎么查出来"的命令，而不是一个让你抄的值。其中
三个**错了不会报错**，值得各花两分钟确认：

- **`MC_GID_INDEX`** —— 见 §4。错了是慢，不是坏。
- **对外 advertise 的 IP** —— 填管理网地址在启动时不会失败：两条 leg 都能正常注册，
  漏洞要到第一次 KV 交接才出现，而那是在权重加载**之后**。发现这个错误的代价是一整轮
  冷启动。
- **`ETCD_ENDPOINT`** —— 如果同节点上的 Kubernetes 控制面已经占了 2379/2380，就得把
  PD 的 etcd 挪开。`launch_etcd.sh` 会按 client 端口 + 1 推导 peer 端口。

---

## 4. 检查网络

跨节点 PD 每个请求都要搬 KV，所以这一步值得认真做。在每个节点的宿主机 shell 上：

```bash
bash preflight_rdma.sh
```

容器内可见的 active RDMA port 数必须等于该节点实际的数量，**不能是 0**。为 0 意味着
`ibv_get_device_list()` 什么都没找到，Mooncake 会退化到 TCP，而部署照样起得来、照样
应答。

**然后要把 `MC_GID_INDEX` 查出来，而不是抄一个。** GID index 选错不报错，只是慢，而且
差距很大：

| 配置 | 单 rail 带宽 | 相对 |
|---|---|---|
| 正确的 RoCE v2 IPv4 index | 8.8 GB/s | 基准 |
| 不设，让 Mooncake 自己选 | 0.5 – 2.4 GB/s | 慢 **4 – 18×** |
| 强制退化到 TCP | 0.08 GB/s | 慢 **110×** |

三种情况下服务都正常应答，日志里什么都没有。这个值**也不能跨集群抄**——在一个集群上是
RoCE v2 条目的那个编号，在另一个集群上可能指向管理网卡。在你选定的 rail 上读出来：

```bash
for i in $(seq 0 7); do
  echo "$i $(cat /sys/class/infiniband/$IB_DEVICE/ports/1/gid_attrs/types/$i 2>/dev/null)" \
       "$(cat /sys/class/infiniband/$IB_DEVICE/ports/1/gids/$i 2>/dev/null)"
done
```

取类型为 `RoCE v2`、且 GID 是本节点数据网 IP 的 IPv4 映射形式（`::ffff:10.0.0.1`）的
那个 index。

**确认这条 rail 在两个节点上都是 ACTIVE。** 两条 leg 都钉在同一条 rail 上，而 rail 在
一侧 down 掉时不会报传输错误——decode leg 会正常加载权重、分配 KV、起服务，然后在 PD
warmup 阶段死于 `Memory access fault by GPU node-N`，而真正的原因是几千行之前的三行
info 日志。

```bash
for d in /sys/class/infiniband/*; do echo "$(basename "$d") $(cat "$d/ports/1/state")"; done
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX"                  # decode 节点
ib_write_bw -d "$IB_DEVICE" -x "$MC_GID_INDEX" "$DECODE_IP"     # prefill 节点
```

---

## 5. 拉起服务

```text
prefill 宿主机:  host_container.sh  →  launch/launch_etcd.sh
decode  宿主机:  host_container.sh
prefill 容器内:  launch/launch_prefill.sh  →  launch/launch_router.sh
decode  容器内:  launch/launch_decode.sh
prefill 容器内:  verify.sh
```

两条 leg 通过 etcd 互相发现，所以 decode 可以和 prefill 并行启动；router 只要在接流量
之前看到两条 leg 都注册上即可。

两个宿主机上都执行：

```bash
bash host_container.sh
```

它建立长期存活的引擎容器，并且在你花 20 分钟加载之前先确认镜像存在、以及引擎将来读
`config.json` 的那条路径在挂载之后确实可读。

只在 prefill 宿主机上：

```bash
bash launch/launch_etcd.sh
```

然后进容器（`docker exec -it infera-glm52-gfx942 bash`），prefill 节点上：

```bash
bash launch/launch_prefill.sh
bash launch/launch_router.sh
```

decode 节点上：

```bash
bash launch/launch_decode.sh
```

**冷启动 15–25 分钟**——加载权重，然后是内存池、JIT 和 CUDA graph 捕获，这些阶段几乎
不打印任何东西。用 `tail -f logs/prefill.log` 跟。对这个 tail 按 `Ctrl-C` 不会停掉引擎，
启动脚本用的是 `nohup`，要停请用 `stop.sh`。

两条 leg 除了端口和 `--disaggregation-mode` 之外 flag 完全一致：

```text
--tp-size 8 --dp-size 8 --enable-dp-attention
--kv-cache-dtype fp8_e4m3
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang
--mem-fraction-static 0.85 --max-running-requests 128 --chunked-prefill-size 8192
--speculative-algorithm EAGLE --speculative-num-steps 5
--speculative-eagle-topk 1 --speculative-num-draft-tokens 6
--disaggregation-transfer-backend mooncake
```

---

## 6. 验证

```bash
bash verify.sh
```

六项检查，每一项都对着一个**不会返回错误的失败模式**。任一断言失败则退出码非零。

1. **两条 leg 都在 etcd 里注册了**。
2. **正确性** —— 一个填充过、答案已知的 prompt。KV 交接坏掉时返回的是 HTTP 200：
   decode leg 读到损坏的前缀，然后生成一段通顺但与提问无关的文本。只有核对答案才抓
   得到。
3. **kv-aware 路由确实生效** —— router 日志里出现了 `request_blocks > 0` 的 prefill
   选择。没有 block hash 时它会静默退化成按负载路由，而且看起来非常健康。
4. **MTP 在被接受** —— decode leg 的 `/metrics` 里有 `sglang:spec_accept_length`。
   两条 leg 的投机配置不一致时，投机解码会被静默丢弃。
5. **kvd** —— 默认 `KVD=0` 时跳过。
6. **RDMA 交接** —— decode 日志里的 Mooncake 传输行。

---

## 7. 压测

两个负载，回答的是不同的问题。

### 7.1 用随机数据集做容量摸底

```bash
bash bench.sh                        # ISL=4096 OSL=1024 CONC=16
ISL=8192 OSL=512 CONC=32 bash bench.sh
bash run_sweep.sh                    # 并发 1..128
```

缓存命中那一行会接近 0，**这是对的**——随机 prompt 之间没有共享前缀，kv-aware 路由无
可复用。这个测的是裸吞吐。扫并发请用 `run_sweep.sh` 而不是自己套一层 `CONC` 循环：种子
固定时，高并发档的 prompt 集合是低并发档的超集，而缓存里还留着后者，测出来从并发 16
起每一档命中率都是 ~50%。

### 7.2 agentic 负载 —— 配方就是在它上面调出来的

真实的多轮 agent 流量，长上下文、强前缀共享，正是 PD 分离和 kv-aware 路由要解决的形态。
数据集只需构建一次，在 prefill 节点的容器内：

```bash
hf download semianalysisai/cc-traces-weka-062126-256k --repo-type dataset
SRC=$(ls "$HF_HOME"/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*/traces.jsonl)

python3 weka_to_agentic_trace.py "$SRC" -o "$TRACE" \
  --output-len "$OUTPUT_LEN" --min-turns 4 --max-context 100000 \
  --verify 20 --tokenizer "$MODEL"
```

原始语料只有每轮的 token 数和 KV block id、没有文本，所以转换脚本合成填充文本，同时
精确保留真正重要的东西：每轮长度和 block 级别的前缀复用。然后：

```bash
NUM_PROMPTS=60 CONC=16 bash run_agentic_trace.sh docker
```

`NUM_PROMPTS` 数的是**会话数**，不是请求数——60 个会话平均约 7.5 轮，合 448 个请求。

### 7.3 怎么读结果

`sglang.benchmark.serving` 在多轮模式下把输入侧统计错了：它对每一轮都沿用会话级的
`prompt_len`，所以它自己的汇总可能一边打印 `Total input tokens: 0`，一边给出超过 100%
的缓存命中率。而它记录的每请求 `cached_tokens` 来自服务端，是对的。因此
`score_agentic_trace.py` 用数据集里已校验的每轮长度重算，给出真正值得比较的那个数：

```text
  actual hit rate              84.61 %
  ideal  hit rate              84.61 %
  efficiency (a/i)            100.00 %
  tokens lost to evict              0 (0.00% of ideal)
```

**效率**＝实际命中 ÷ 一个从不驱逐的缓存本可以返回的量。100% 且无驱逐说明这一轮还没到
压力点，kv-aware 路由无从体现差异——把 `CONC` 加大到出现驱逐为止。效率**超过** 100%
说明 flush 没生效：`flush_cache` 在有请求在飞时是 no-op，而且照样返回成功。

含有任何失败请求的运行会被拒绝评分而不是照算——失败会被记为 `cached_tokens=0`，算进去
会让一个挂掉的 worker 看起来像缓存问题。

---

## 8. 两个工作点

默认配方是在并发 16 上调的，那里 DP-attention 是赢的：纯 TP8 的 prefill 慢 25.9%。到了
并发 1，结论反过来，而且幅度不小。

DP-attention 下每个 rank 携带的是**完整未切分**的 attention 权重。只有一个请求在飞时，
只有一个 rank 有真实工作，但 8 个 rank 都要跑完整的 78 层前向来参与 MoE 的 all-reduce
——而且跑在一个没人调优过的矩阵形状上。在并发 1 下对默认配置做 profiling：GPU 忙碌率
79–86%，其中一半以上耗在单个输出投影 GEMM 上，而它只占用了 304 个 CU 中的 48 个。
attention 本身占 0.7%，集合通信占 6–13%：瓶颈既不是网络也不是 KV 带宽。

关掉它：

```bash
DP=1 CHUNK=2048 bash launch/launch_prefill.sh     # DP=1 时 dp-attention 自动关闭
DP=1 CHUNK=2048 bash launch/launch_decode.sh
```

并发 1 下重复测三次，与默认配方对比：

| | 默认（`DP=8`） | `DP=1 CHUNK=2048` |
|---|---|---|
| 平均 TPOT | 18.63 ms | 7.30 – 8.02 ms |
| 单用户速度（中位） | 62.5 tok/s/user | **133 – 159** |
| 缓存效率 | 100.00% | 100.00% |
| MTP 接受长度 | 4.29 | 4.32 |

缓存效率和接受长度都没退化，说明这是纯粹的算子效率收益，没有拿正确性去换。

**`CHUNK` 必须跟着 `DP` 一起改。** `CHUNK` 是全局预算，而 SGLang **只在 DP-attention
开着时**才把它按 `dp_size` 切分。默认配置下 `8192` 是每 rank 1,024；到了 `DP=1`，同样的
`8192` 就是每 rank 8,192，激活内存是原来的 8 倍。并发 1 时侥幸不炸，并发 16 的长 prefill
会直接 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。`2048` 是实测下不 OOM 的最大值，并且在每项
指标上都优于 `1024`。启动后回读实际生效的值——日志里那句
`chunked prefill size adjusted from … to …` 是"被除了"，不是"被拒绝了"：

```bash
curl -s "$PREFILL_URL/get_server_info" | python3 -c \
  'import json,sys; d=json.load(sys.stdin); print(d["chunked_prefill_size"], "per rank x", d["dp_size"], "ranks")'
```

**高并发下的代价**，同一负载并发 16：

| | DP-attention 开 | 关 |
|---|---|---|
| 输出吞吐 | **182.7 tok/s** | 139.4 tok/s（−23.7%） |
| 平均 TTFT | **12.4 s** | 22.2 s（1.79×） |
| 平均 TPOT | 21.50 ms | **9.05 ms** |

TTFT 那 1.79× 里有一部分是 `CHUNK` 从 8192 降到 2048 的账，不全是 DP-attention 的。

**继续调优之前要知道两件事。** 并发 1 下的单点测量在这个负载上有约 ±5% 的运行间抖动，
所以任何小于 10% 的结论都必须重复测量。另外，单用户速度在这里是随上下文长度上升的——
50–80k 这一桶的中位是 153–163 tok/s/user，而会话开场的短轮次只有约 99——原因是长对话
后续大量重复的工具调用 JSON，draft 模型猜得准。请按你的真实流量所在的区间报数字。

另有三条调优方向已经试过并证伪：custom all-reduce 在 gfx942/gfx950 上做 EAGLE verify
时会**死锁**，所以 `--disable-custom-all-reduce` 必须留着；树状 draft
（`--speculative-eagle-topk > 1`）在 `page_size > 1` 时被 DSA attention 后端拒绝；加深
draft 链净亏——接受长度涨 31%，但每步耗时涨 33%。出厂的 `5/1/6` 已经在最优点上。

---

## 9. 停止

两个节点的容器内：

```bash
bash stop.sh
```

然后在宿主机上：

```bash
bash host_container.sh --rm
docker rm -f infera-glm52-etcd     # 只在 prefill 宿主机
```

**重启前一定要先清掉引擎进程**，否则下一次启动会撞上上一次还占着的 VRAM 而 OOM。

---

## 10. 排障

**什么都不应答，或者应答很慢，但哪里都没有报错。** 按 §4 逐项过：active port 数、GID
index、两个节点上的 rail 状态。这个栈里所有 RDMA 故障都是静默的。

**回答很通顺但答非所问。** KV 交接坏了，decode leg 读到的是损坏的前缀。`verify.sh` 的
第 2 项检查就是为此存在的。

**启动看起来一切正常，然后报 `Memory access fault by GPU node-N`。** 见过两个原因：
钉住的 rail 在一侧 down 了（往前翻几千行找 `topology.cpp … is not active`、
`has no active ports, skipping`），或者 base 镜像与主机驱动不匹配（§1）。

**`chunked prefill size adjusted from … to …`。** 不是被拒绝，是被 `dp_size` 除了。
见 §8。

**缓存效率超过 100%。** flush 没生效，因为还有请求在飞。`flush_cache` 在有请求运行时
是 no-op，并且照样返回成功。

**kill 压测没杀干净。** `pkill -f run_agentic_trace.sh` 只杀掉了外层 wrapper。找到容器
再删：

```bash
docker ps --no-trunc --format '{{.Names}}|{{.Command}}'
```

**启动失败后残留僵尸 python 进程。** `pkill -9` 收不掉，重启容器。

**两条 leg 的 MTP 形状必须一致。** SGLang 会拒绝投机配置不一致的 PD 对，而且是在权重
加载**之后**才拒绝，所以单边改一个参数要花一整轮冷启动才能发现。请改 `cluster.env`
里的 `MTP_STEPS` / `MTP_DRAFT_TOKENS`，两条 leg 都读它。

---

## 数字的出处

本文所有数字都是在一对 MI325X 节点上实测的。完整记录——原始压测输出、profiler trace、
与 Kubernetes 部署的逐 flag 比对、以及那些失败的调优尝试——在
[`mi325x-handoff/`](../../mi325x-handoff/README.md)。
