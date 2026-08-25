# GLM-5.2-FP8 gfx942 —— Kubernetes 部署

通过 Infera operator 在 gfx942（MI300X / MI325X）上服务 GLM-5.2-FP8：SGLang TP8 +
DP-attention、MTP 投机解码、fp8 KV 缓存、kv-aware 路由，以及可选的 Mooncake RDMA
prefill/decode 分离和向主机内存 + 本地 NVMe 的 KV 卸载。

引擎、router 和卸载层的 flag 全部原样取自 [`docker.zh.md`](docker.zh.md) 里的 shell
部署。这一点是验证过的而不是声称的：对两边真正产生的 argv 做逐 flag 比对，prefill、
decode、router 三个组件**零处实质差异**；同一负载下 decode 侧延迟相差在 1.2% 以内。
**调优结论在两种部署方式之间可以互相沿用。**

manifest 在 `examples/recipes/glm5.2-fp8-gfx942/`。

---

## 1. 选一个组合

| 组合 | 节点数 | KV 缓存 | MTP | manifest |
|---|---|---|---|---|
| `aggregated` | 1 | 仅 GPU | 有 | `aggregated/deploy.yaml` |
| `aggregated + kvd` | 1 | 加主机内存（L2）和本地 NVMe（L3） | **无** | `aggregated-kvd/deploy.yaml` |
| `disaggregated` | 2 | 仅 GPU | 有 | `disaggregated/deploy.yaml` |
| `disaggregated + kvd` | 2 | prefill leg 上加 kvd | 有 | `disaggregated-kvd/deploy.yaml` |

四个组合都是 TP8 / DP8 + DP-attention，四个都已验证。

**即使你的目标是 PD，也请先从 `aggregated` 起。** 它只需要一个节点，不碰 Mooncake、不碰
`/dev/infiniband`、不需要 GID index，因此完全不可能踩上 RDMA 那一整类静默故障。这也让
它成为"出了问题但还不知道在哪一层"时该拉起来的东西：它把"这个镜像能不能加载这份权重
并提供服务"和"KV 能不能过这张网"分开，而这两件事的失败方式截然不同。

**`aggregated + kvd` 是唯一没有 MTP 的一档。** 在同时做 prefill 和 decode 的 worker
上，MTP 和分层缓存会互相死锁。为了确认是 MTP 触发的而不是它恰好在场，试了六种配置；
最终发布的就是"关掉 MTP、其余不变"那一种。`disaggregated + kvd` 保留了 MTP，因为投机
只发生在它的 decode leg 上，所以这个组合永远不会出现在带 kvd 的那条 leg 上。

**除非有明确理由，否则不要开 `kvd`。** 缓存是 `write_through` 的，所以这一层吸收的每
一个字节都要在 prefill 路径上付出代价，无论以后有没有人读回来。在两节点、无复用的负载
上实测：输出吞吐掉 43%、TTFT 变成 3.45 倍，而读取次数为零。只有当请求之间有长前缀共享、
且复用量超出每 rank 约 54 GB 的显存池时，它才赚得回来。

对测量还有一个影响：`aggregated` **不是** `aggregated + kvd` 的 `KVD=0` 对照组，因为
它还多了 MTP。只有 disaggregated 那一对是仅差 kvd 一个变量的。

---

## 2. 前置条件

**硬件**，按组合区分。每个引擎 Pod 都要 8× gfx942 和 `cpu: 32`。

| 组合 | 节点数 | 最忙节点的主机内存 | 本地 NVMe | RoCE 网络 |
|---|---|---|---|---|
| `aggregated` | 1 | 约 272 GiB | 否 | 否 |
| `aggregated + kvd` | 1 | 约 670 GiB | 是，用于 L3 | 否 |
| `disaggregated` | 2 | 约 272 GiB（prefill） | 否 | **是** |
| `disaggregated + kvd` | 2 | 约 670 GiB（prefill） | 是，在 prefill 上 | **是** |

**集群。** 两个 kvd 组合需要 Kubernetes **1.29+**：kvd 守护进程是原生 sidecar
（`initContainers` 加 `restartPolicy: Always`），这才能保证它在引擎启动**之前**就达到
健康的 `startupProbe`。这个顺序是关键：引擎只用 5 秒超时探一次 kvd socket，没有应答就
拒绝启动，不重试。在更老的集群上 sidecar 只能是普通容器，顺序就变成了一场引擎输赢
各半的竞态。另外两个非 kvd 组合没有这个要求。

```bash
# 节点必须上报 amd.com/gpu
kubectl get nodes -o custom-columns=NODE:.metadata.name,GPU:.status.allocatable.'amd\.com/gpu'

# operator，提供 InferaDeployment CRD
helm install infera-operator deploy/operator/helm/infera-operator \
  -n infera-system --create-namespace
kubectl -n infera-system rollout status deploy/infera-operator
```

**权重。** 用 `hostPath` 而不是 PVC，且在这个部署用到的**每个节点上路径相同**。如果这
个路径是 HuggingFace 缓存的软链接，还要把链接指向的目录一起挂上——否则内部的相对链接
会悬空，`transformers` 会以 `Should have a model_type key in its config.json` 拒绝这个
模型，而且是在启动四分钟之后，离真正的原因很远。

**Build 之前先在每个节点上读主机驱动版本。** 它决定 base 镜像，而选错不会拒绝启动：

```bash
dpkg -l | grep -E 'amdgpu-dkms|rocm-core'
```

| 主机驱动 | 支持的 ROCm 用户态 | 用哪个 base |
|---|---|---|
| 6.4.x | ≤ 7.2.x | `rocm720`，Dockerfile 默认值 |
| 6.3.x | ≤ 7.0.x | `rocm700`，**必须显式覆盖** |

搭配错了的表现是：正常初始化、加载权重、捕获 graph，然后在负载下的某处报
`Memory access fault by GPU node-N`，每次落点还不一样。

---

## 3. Build 镜像并送进节点

**不要用厂商预置的 GLM-5.2 镜像。** 这个 recipe 跑的是 infera 自己 build 的引擎镜像，
而不是"厂商镜像 + 挂载 overlay"，因为 GLM-5.2 在这个 base 上需要重新编译的 Mooncake 和
四个 SGLang 源码补丁，而挂载式的载荷提供不了这些。Mooncake 的重编尤其不是可选项：base
自带的 Mooncake 会无条件安装 HIP IPC 传输并优先使用它，于是跨节点 PD 在第一个请求上就
死在 `hipIpcOpenMemHandle` 里——它打不开对端节点的 handle。

```bash
docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  -t infera:sglang-gfx942-glm52 .                            # 6.4.x 主机驱动

docker build -f deploy/docker/Dockerfile.sglang.gfx942 \
  --build-arg SGLANG_BASE_IMAGE=lmsysorg/sglang:v0.5.16-rocm700-mi30x \
  -t infera:sglang-gfx942-glm52 .                            # 6.3.x 主机驱动
```

镜像约 107 GB，build 约 40 分钟。信任它之前先确认补丁真的打上了——四个里有三个会留下
标记，这是抓"补丁因锚点移动而静默失效"最省事的办法：

```bash
docker run --rm --entrypoint python3 \
  -v "$PWD/examples/glm5.2_gfx942/check_image.py:/check.py:ro" \
  infera:sglang-gfx942-glm52 /check.py
```

三行都是 `YES` 才算过：

```text
sglang_rocm/host_alloc       srt/mem_cache/pool_host/common.py            YES
sglang_rocm/staged_wb        srt/mem_cache/pool_host/mla.py               YES
sglang_disagg/early_send     srt/disaggregation/mooncake/conn.py          YES
```

别人给你的镜像也用这条命令查。已验证节点上预置的那个 GLM-5.2 镜像，第一行是 `no`——
这正是本文让你自己 build 的原因。

### 3.1 镜像要进节点的容器运行时，不是 docker 的

manifest 用的是 `imagePullPolicy: IfNotPresent`，所以本地已有的镜像会被直接使用——但
"本地已有"指的是对 **kubelet 的运行时**可见。在基于 containerd 的集群（RKE2、k3s）上，
`docker build` 的产物对它是不可见的。推到集群能拉的 registry，或者直接导入：

```bash
docker save infera:sglang-gfx942-glm52 \
  | ctr -a /run/k3s/containerd/containerd.sock -n k8s.io images import -
```

如果节点上没有 root 但有 cluster-admin，可以用一个特权 Pod 做同样的事，把宿主机的
`docker` 和 `ctr` 两个二进制都挂进去——两个都要，只挂 `ctr` 的话 `docker save` 无从
执行。可用的实现见
[`mi325x-handoff/tools/import-image-to-containerd.sh`](../../mi325x-handoff/tools/import-image-to-containerd.sh)。

### 3.2 kubelet 可能在两分钟内把它删掉

导入后没有任何 Pod 引用的镜像，一旦节点超过镜像 GC 的高水位（默认磁盘 85%）就会被立即
回收，最短存活保护只有两分钟。一个 107 GB 的镜像在磁盘 90% 的节点上几乎是瞬间消失，而
且导入过程会先报成功。

导入前先看：

```bash
df -h /var/lib/rancher     # 或者你的运行时镜像库所在的位置
```

如果降不到水位以下，**先把 manifest 渲染好，导入一结束立刻 apply**，让 Pod 在两分钟的
保护窗口内引用上镜像：

```bash
sed -e "s|<PREFILL_NODE>|node-a|" ... disaggregated/deploy.yaml > pd.yaml
bash import-image-to-containerd.sh ... && kubectl apply -f pd.yaml
```

---

## 4. 适配你的网络

`aggregated` 系列可以跳过本节——它们不把 KV 搬出节点，因此完全没有 rail、GID index 和
Mooncake 配置。

disaggregated 的 manifest 需要三个网络相关的值。请从节点上读出来，而**不要从任何地方
抄，包括从本文抄**：

| 占位符 | 含义 |
|---|---|
| `<RDMA_IB_DEVICES>` | 允许 Mooncake 使用的 rail，逗号分隔，来自 `ibv_devices`。物理上 down 的 rail **不能**列进去。 |
| `<PREFILL_GID_INDEX>` / `<DECODE_GID_INDEX>` | 该 rail 上类型为 `RoCE v2` 的那个 index。有两个占位符，是因为这个 index 是**按节点**而不是按集群的——两台完全相同的机器给出不同的值是常事。 |

先在两个节点上从镜像内部检查网络，因为可见端口数为 0 正是 RDMA 在这里失败的方式——
Mooncake 会退化到 TCP，而部署照样起得来：

```bash
docker run --rm --network host --device=/dev/infiniband --cap-add=IPC_LOCK \
  --entrypoint bash infera:sglang-gfx942-glm52 -c 'ibv_devinfo | grep -c PORT_ACTIVE'
```

然后在每个节点上读 GID 表：

```bash
show_gids <rail>
# 或者：
for i in $(seq 0 7); do
  echo "$i $(cat /sys/class/infiniband/<rail>/ports/1/gid_attrs/types/$i 2>/dev/null)" \
       "$(cat /sys/class/infiniband/<rail>/ports/1/gids/$i 2>/dev/null)"
done
```

index 选错不会报错——它会把 KV 钉在一个从来不承载它的接口上，结果要么传输超时，要么慢
4–18 倍，而日志里什么都没有。

**已验证过两种网络形态，它们的答案不同。** 替换任何值之前先确认你属于哪一种：

**A. rail 上有 IPv4。** GID 表里存在一个 `RoCE v2` 条目，其地址是节点数据网 IP 的 IPv4
映射形式（`::ffff:10.115.43.101`）。取这个 index —— 在已验证的 MI325X 节点对上它是
`3`，那对机器的 HCA 叫 `rdma0..rdma7` 而不是 `mlx5_N`。这种情况下做一次占位符替换就够了。

**B. rail 上没有 IPv4。** 所有 GID 都是链路本地的 `fe80::`，`RoCE v2` 条目通常是 index
`1`——这种情况下 index 3 一般属于管理网卡，把 A 里的 `3` 抄过来会让 Mooncake advertise
一个对端根本到不了的地址。这种形态还需要多做一件事：既然每条 rail 看起来都在同一个
`fe80::/64` 子网里，Mooncake 就无法区分它们，可能把 prefill 节点的 rail A 和 decode
节点的 rail B 配对，而那样传输只会超时。把它钉在单条 rail 上：

```yaml
- {name: MC_MS_AUTO_DISC, value: "0"}
- {name: MC_MS_FILTERS,   value: "<rail>"}
```

[`mi325x-handoff/tools/render-deploy.py`](../../mi325x-handoff/tools/render-deploy.py)
就是针对这种情况渲染四个组合的现成实现。

**把 KV 钉在单条 rail 上不是妥协。** 在这个负载上把 KV 打散到全部 NIC 实测反而慢
11.9%，而 KV 只用掉单个 200 Gb/s 口的 4.5%。

---

## 5. 部署

每个 manifest 给的是占位符而不是默认值，所以漏替换的会大声失败——`kubectl` 会直接拒绝
`<NODE>`，而字面量 `<RDMA_IB_DEVICES>` 也不是 `ibv_open_device` 会接受的设备名。

```bash
kubectl create namespace infera --dry-run=client -o yaml | kubectl apply -f -
```

`aggregated`，单节点，两个占位符：

```bash
sed -e "s|<NODE>|node-a|" -e "s|<MODEL_DIR>|/mnt/models|" \
    examples/recipes/glm5.2-fp8-gfx942/aggregated/deploy.yaml | kubectl apply -f -
```

`disaggregated`，两节点，外加 §4 的网络值：

```bash
sed -e "s|<PREFILL_NODE>|node-a|"   -e "s|<DECODE_NODE>|node-b|" \
    -e "s|<MODEL_DIR>|/mnt/models|" -e "s|<RDMA_IB_DEVICES>|rdma0|" \
    -e "s|<PREFILL_GID_INDEX>|3|"   -e "s|<DECODE_GID_INDEX>|3|" \
    examples/recipes/glm5.2-fp8-gfx942/disaggregated/deploy.yaml | kubectl apply -f -
```

`kvd` 变体多一个 `<KVD_L3_DIR>`，它必须是**节点本地 NVMe**——任何共享存储（NFS、Weka）
都会被判定为 buffered I/O，重新加载的开销会落进 TTFT 预算里而不是被藏在它下面。

每个组合部署成各自的名字：

| 组合 | `InferaDeployment` | 引擎 service |
|---|---|---|
| `aggregated` | `glm52-fp8-mixed` | `worker` |
| `aggregated + kvd` | `glm52-fp8-mixed-kvd` | `worker` |
| `disaggregated` | `glm52-fp8-pd` | `prefill`、`decode` |
| `disaggregated + kvd` | `glm52-fp8-pd-kvd` | `prefill`、`decode` |

**冷启动：aggregated 10–20 分钟，disaggregated 15–25 分钟。** 日志会安静下来：权重约
3.5 分钟就加载完，之后的 draft 权重、内存池、`tilelang`/`aiter` JIT、graph 捕获会有十
分钟以上几乎不打印任何东西。所以 worker 用的是 90 分钟预算的 `startupProbe`，且没有
readiness probe。**不要 kill 一个加载慢的 Pod。**

```bash
CR=glm52-fp8-pd        # 或 glm52-fp8-mixed 等
kubectl -n infera get pods -w
kubectl -n infera logs -f -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=prefill
```

---

## 6. 访问 router

```bash
kubectl -n infera port-forward svc/$CR-server 8000:8000 &
```

`port-forward` 走 API server，在哪都能用。**不要想当然认为 Service VIP 在节点的宿主机
shell 上可用**——在 RKE2 上 ClusterIP 和 NodePort 都不通，因为 kube-proxy 不为 Service
VIP 在宿主机侧建转发规则。router 的 **Pod IP** 是通的；引擎 Pod 用的是
`hostNetwork: true`，所以它们在 `节点IP:端口` 上可达。

需要持续打压力的压测客户端请用 router 的 Pod IP，不要用 port-forward——后者是一条经
API server 代理的单流，自己会先成为瓶颈。

---

## 7. 验证

```bash
curl -s localhost:8000/v1/workers | jq     # 一个 mixed worker，或者一个 prefill + 一个 decode

curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"/models/GLM-5.2-FP8",
       "messages":[{"role":"user","content":"What is 127 * 31? Answer with the number only."}],
       "max_tokens":128,"temperature":0,
       "chat_template_kwargs":{"enable_thinking":false}}' | jq -r '.choices[0].message.content'
```

manifest 没有传 `--served-model-name`，所以对外的模型名**就是**模型路径，即上面的
`/models/GLM-5.2-FP8`。

答出 `3937` 是必要而不充分的。还有两项检查，针对的都是会返回 HTTP 200 的失败：

**disaggregated 组合要检查传输层。** RDMA 初始化失败不会让任何东西停下来；Mooncake 会
退化到 TCP，一切照常应答。每个 DP rank 应该有一行：

```bash
kubectl -n infera logs -c main \
  -l infera.amd.com/deployment=$CR,infera.amd.com/service=decode \
  | grep -aE 'GID index|installTransport'
```

八行 `installTransport, type=rdma` 才算通过。`type=tcp` 说明这一轮能跑，但它对网络什么
都没证明。

**每个组合都要发一个超过一个 chunk 的长 prompt。** 缺少 Mooncake early-send wait-event
补丁时，除最后一块之外的每个 prefill chunk 都会在前向还在写它的时候被读走，于是多 chunk
的 prompt 会**返回部分错误的内容，而且任何日志里都没有痕迹**。一个能答对的短 prompt 看
不出这个问题。请在一个长度为 `--chunked-prefill-size` 数倍的 prompt 的**头部、中部和
尾部**各埋一个独特的 needle，然后要求全部取回——只丢头部的情况下，如果你恰好只探尾部，
读起来就是"它是好的"。

---

## 8. 压测

router 就是一个普通的 OpenAI 兼容端点，不关心引擎是怎么起来的，所以用和 shell 部署完全
相同的客户端。这一点是有意的：tokenizer、数据集、并发限流器和评分器全都在客户端这一侧，
用两个不同的客户端等于把这些也放进了对比里。

数据集只需构建一次（见 [`docker.zh.md`](docker.zh.md) §7.2），然后在一台有镜像和权重的
主机上：

```bash
cd examples/glm5.2_gfx942
cp cluster.env.example cluster.env    # 这里只有 MODEL、IMAGE 和 DATA_DIR 起作用

NUM_PROMPTS=60 CONC=16 bash bench_client.sh k8s http://<router-pod-ip>:8000 \
  http://<prefill-node-ip>:30001 http://<decode-node-ip>:31501
```

后面那几个 URL 是开跑前要 flush 的引擎地址。flush 很重要：上一轮留下的 block 会让命中
率虚高，而 `flush_cache` 在有请求在飞时是 no-op 却照样返回成功。效率**超过** 100% 就是
它没生效的信号。

`NUM_PROMPTS` 数的是**会话数**而不是请求数：60 个会话平均约 7.5 轮，合 448 个请求。
怎么读评分见 `docker.zh.md` §7.3。

在已验证的 MI325X 节点对上，这套部署跑同一负载的结果与 shell 部署一致：

| | Kubernetes | Docker + shell |
|---|---|---|
| 成功请求 | 448 / 448 | 448 / 448 |
| 缓存效率 | 100.00% | 100.00% |
| 输出吞吐 | 189.8 tok/s | 182.7 tok/s |
| 平均 TPOT | 21.60 ms | 21.50 ms |
| 中位 TPOT | 18.34 ms | 18.57 ms |

---

## 9. 两个工作点

默认配方是在并发 16 上调的，那里 DP-attention 是赢的。到并发 1 结论会反过来，而且幅度
很大——实测数据在 [`docker.zh.md`](docker.zh.md) §8，之所以能沿用到这里，是因为开头那次
逐 flag 比对证明了两种部署跑的是同一个引擎。

| | 批量 / 高并发 | 交互 / 低并发 |
|---|---|---|
| 并发 16 聚合吞吐 | **182.7 tok/s** | 139.4 tok/s |
| 并发 1 单用户速度（中位） | 62.5 tok/s/user | **133 – 159 tok/s/user** |
| 并发 16 平均 TTFT | 12.4 s | 22.2 s |

要把一个部署切到交互工作点，在 manifest 里对**两条 leg**都改：

- 去掉 `--enable-dp-attention`
- `--dp-size` 改成 `1`
- `--chunked-prefill-size` 改成 `2048`

**第三条不是可选的。** `--chunked-prefill-size` 是全局预算，而 SGLang **只在
DP-attention 开着时**才按 `dp_size` 切分它。默认配置下 `8192` 是每 rank 1,024；关掉
DP-attention 之后同样的 `8192` 就是每 rank 8,192，激活内存是原来的 8 倍，并发 16 的
长 prefill 会直接 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`。

这个消融实验是在 shell 部署上做的，没有在 Kubernetes 上重跑。要在这里确认，按上面三条
改完再跑一遍 §8，预期落在 `docker.zh.md` §8 的区间内。

---

## 10. 清理

```bash
kubectl -n infera delete inferadeployment $CR
```

删除是异步的。等 `kubectl -n infera get pods` 返回空之后，再在这些 GPU 上启动别的东西，
否则两边会抢 GPU。

---

## 11. 排障

**Pod 一直 `Pending`。** 通常是 `nodeSelector` 的主机名或者 `amd.com/gpu` 的可分配量。
`kubectl -n infera describe pod` 会说是哪个。

**刚 build 好的镜像却报 `ErrImageNeverPull` / `ImagePullBackOff`。** kubelet 的运行时
看不到 docker 的镜像库，或者镜像在导入后被回收了——见 §3.1 和 §3.2。

**一切正常应答，但日志是 `installTransport, type=tcp`。** Mooncake 没拿到 RDMA。先从
镜像内部查可见端口数，再查 GID index——见 §4。

**网络看起来健康，但 KV 传输超时。** 如果你的 rail 上没有 IPv4，多半是 Mooncake 把两侧
不同的 rail 配对了。把它钉在单条 rail 上——见 §4 的情况 B。

**长 prompt 返回部分错误的内容，日志里没有任何痕迹。** 镜像缺 Mooncake early-send
wait-event 补丁。重新 build 并跑 `check_image.py`——见 §3。

**启动看起来正常，然后报 `Memory access fault by GPU node-N`。** 要么是 base 镜像与主机
驱动不匹配（§2），要么是钉住的 rail 在一侧 down 了。后者请往前翻几千行找
`topology.cpp … is not active`、`has no active ports, skipping`、
`Skipping unavailable device`：Mooncake 把你给它的唯一一条 rail 跳过了，故障要等到第一次
真实 KV 传输时才以显存错误的形式浮出来。

**kvd sidecar 在 graph 捕获之后立刻被 OOMKilled（exit 137）。** kvd 持有两份独立的预算，
按其中一份来设容器 limit 是不够的。用 manifest 里给的数值。

**单个请求正常，但持续压力下会有请求失败。** 在 RKE2 上遇到过一次：长上下文负载下
448 个请求失败了 44 个，把 router 换成 `hostNetwork: true` 之后失败清零。**机理一直没有
查实**——曾经怀疑是 MTU 问题，实测推翻了；同一套 recipe 在另一个集群上完全没有复现。
所以这不是推荐的默认值：`hostNetwork` 会占用节点的 8000 端口，并且脱离 NetworkPolicy。
把它当作"复现了这个症状之后可以试的缓解手段"，并优先在自己的 CNI 里找原因。

**`no active mixed worker for model="..."`。** 这是模型名不匹配，不是集群挂了——对外的
模型名就是模型路径（§7）。所有请求会在毫秒级失败，看起来很像一次全面宕机。

---

## 数字的出处

四个组合是在一对 MI300X 上验证的；§8 和 §9 里的 MI325X 数字、与 shell 部署的逐 flag
比对、以及调优记录，都在 [`mi325x-handoff/`](../../mi325x-handoff/README.md)。recipe
层面的细节——源码补丁、kvd 容量规划、每个组合各自验证了什么——在
[`examples/recipes/glm5.2-fp8-gfx942/`](../../examples/recipes/glm5.2-fp8-gfx942/README.md)。
