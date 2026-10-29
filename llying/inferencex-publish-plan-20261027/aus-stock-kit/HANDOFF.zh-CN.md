# AUS 实测 handoff：原样镜像 + 启动时安装 Infera 和打补丁（2026-10-29）

## 这次实测要回答什么

InferenceX 要求用上游原样镜像。PR #210 的数字是在我们自建的镜像上测的，那个镜像在构建时做了四件事：重编 Mooncake、换 libionic、打 SGLang 补丁、预装 Infera。InferenceX 上这四件事要么不做，要么改成作业启动时做。这次只验证这一个差别：

> 原样 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`，在容器启动时装 Infera、打 5 个 SGLang 补丁，能否起栈、结果正确、性能与 #210 一致。

补丁要不要，已经在镜像源码里逐个核对过（见文末附录），这次**不做**"逐个去掉补丁"的实验。

## kit 是什么

本目录是 #210 的 2P1D kit（`ex-8859b247`），只改了"镜像怎么来"这一层。recipe 参数和 #210、InferenceX 的 recipe 一致：

| 文件 | 改动 |
|---|---|
| `env.sh` | `IMAGE` 默认用原样镜像。`MOONCAKE_DISABLE_HIP_DMABUF` 默认 1（AUS 一向用 1）。新增 `MOUNT_HOST_IONIC`、`HOST_IONIC_PROVIDER`。`CTR` 默认 `glm52_stock` |
| `engine/stock_entry.sh`（新增） | 容器入口：先 `pip install --no-deps` 装 `artifacts/` 里的两个 wheel，再装 `infera-router`，然后跑 `patches/apply.sh`，最后 exec 原命令 |
| `patches/apply.sh`（新增） | 打 B1、B2、B6、B7、B8，并核对每个补丁都进了源码；可以重复运行。镜像里没有 GNU `patch`，B2 用 `git apply` |
| `engine/worker.sh` | 把 kit 只读挂到 `/kit`；命令前加 `stock_entry.sh`；按 `MOUNT_HOST_IONIC` 把宿主机的 ionic provider 挂到镜像里同名路径（和 InferenceX kimik3 recipe 的做法一样）；Mooncake 环境变量改成 recipe 那套（`MC_DISABLE_HIP=1`、`MC_ENABLE_DEST_DEVICE_AFFINITY=1`、`MC_MAX_MR_SIZE`），去掉原样 Mooncake 里不存在的 `MC_ENABLE_DEST_LOCAL_RAIL`、`MC_IB_TIMEOUT`、`MC_DISABLE_HIP_TRANSPORT` |
| `engine/up.sh` | router 容器也走 `stock_entry.sh`（`STOCK_PATCHES=0`） |
| `artifacts/` | `amd_infera-0.3.2+pr222-py3-none-any.whl`（v0.3.2 加 #222 三个提交）、`infera-router`（同一份源码，在原样镜像里编译）、`nats_py-2.16.0` |
| `README.kit-210.md` | 原 kit 的说明，用法和各项注意事项都还适用 |

已在 crsuse2-m2m-138 上不用 GPU 验证过：

- 原样镜像里 `stock_entry.sh` 能装好 wheel 和 router，5 个补丁全部打上，第二次运行显示已打过；
- `infera.engine.sglang`、`infera.server --help` 正常；
- router 带 `decode-request-weight`、`decode-input-demand`、`pd-prefill-guard-release` 三个参数；
- #222 的端口解析正常。

**还没上过 GPU。**

为什么不直接用官方 v0.3.2：它今天（10-29）已经发版，但 wheel 是 `py3-none-any`，**不带 router 二进制**（#223 还没合入），也不含 #222。InferenceX 的 setup 脚本会检查 `infera-router`，所以 InferenceX 要用下一版（需要同时包含 #222 和 #223）。

## 0. 准备（约 30 分钟）

1. 申请 2 台节点，用法同以前，例如 `salloc -N 2 --exclusive`。
2. 把 kit 放到两台节点都能看到的同一路径下，例如 `/perf_apps/liyingli/bench_agentx/aus-stock-kit`。打包在 crsuse2 的 `/home/liyingli/bench_agentx/baseline/aus-stock-kit.tar.gz`。
3. 两台节点上都执行 `docker pull lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`，然后用 `docker image inspect -f '{{.RepoDigests}}' <image>` 确认 digest 是 `sha256:8f7d8034546b17cc4ae57c63f3f65c584eb73e62378243d1416b8385aa7be286`。10-21 在 AUS 构建 #210 镜像时就用的这个 base，节点上可能已经有了。
4. 编辑 `env.sh`，以下数值都来自以前的 AUS 实验：

   | 变量 | 值 |
   |---|---|
   | `PREFILL_NODE_IP` / `DECODE_NODE_IP` / `DECODE_SSH` | 两台节点的 IP 和主机名 |
   | `MODEL` | `/perf_apps/data/models/GLM-5.2-MXFP4` |
   | `WORK_DIR` | prefill 节点本地的 scratch 目录，例如 `/scratch/liyingli_stock` |
   | `RDMA_DEVICES` | `ionic_0,...,ionic_7`（默认值即可） |
   | `MC_GID_INDEX` | `1` |
   | `MOONCAKE_DISABLE_HIP_DMABUF` | `1` |

5. 节点之间的 ssh：如果节点之间不能直接 ssh，沿用以前的 srun 替身：Infera 仓库 `llying/2p1d-aus-v0521-20261019/tools/sshshim/ssh`（设 `SHIM_JOB` 和 `PATH`，用法见 `llying/1p1d-decode-balance-aus-20261021/scripts/kit_env.sh`）。

## 1. 网卡预检（每台节点 1 分钟）

原样镜像里的 libionic 是 `54.0-149.g3304be71`，比 #210 镜像装的 `54.0-187` 旧。下面两条命令看它在 AUS 上认不认得出网卡，结果两种情况都要记下来：

```bash
P=/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so
ls -l $P /lib/x86_64-linux-gnu/libionic.so*        # 宿主机上 provider 在哪
IMG=lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018
docker run --rm --device /dev/infiniband --entrypoint ibv_devinfo $IMG -l 2>&1 | tail -3
docker run --rm --device /dev/infiniband -v $P:$P:ro --entrypoint ibv_devinfo $IMG -l 2>&1 | tail -3
```

- 期望：挂载宿主机 provider 后能列出 8 个 `ionic_*`。不挂载时如果报 "does not support the kernel ABI" 或者列出 0 个，就说明必须挂载；InferenceX 的 recipe 就照这个加 `extra_mount`。
- 如果宿主机上没有 `libibverbs/libionic-rdmav34.so`，就在 `env.sh` 里把 `HOST_IONIC_PROVIDER` 改成 `/lib/x86_64-linux-gnu/libionic.so`（它就是这个 provider 链接指向的真实文件）。

## 2. 起栈与正确性：4+4，真实接受率（约 1 小时）

```bash
PREFILL_WORKERS=1 PREFILL0_GPUS=4,5,6,7 DECODE_GPUS=4,5,6,7 RUN_ID=s1-44-real bash engine/up.sh
bash engine/smoke.sh
bash engine/eval.sh
```

起栈期间顺手看一下两台节点的 `docker logs`：

- 有 `[stock-patches] all five applied`；
- Mooncake 走的是 RDMA：没有 `Found 0 HCAs`，没有退回 TCP，没有 `EINVAL`；
- 没有 Triton/LLVM 的 `iota_range` 断言（B8 要防的就是这个）。

**通过标准：**

- `up.sh` 跑完，router 里有 1 个 prefill 和 1 个 decode；
- `smoke.sh` 6/6，其中包括 66k token 的 needle 两遍，第二遍走 HiCache；
- GSM8K 与 #210 的 0.9666 strict / 0.9674 flexible 相差不超过约 1 个点。

另外两项，B5 和 B4 去掉之后，看的就是这里：

- prefill 用的是 `write_back` HiCache，日志里不能有 write-back 报错，也不能有 scheduler 退出；
- 长 prompt（needle）的答案要正确。

## 3. 性能：三种形态各取关键点，模拟接受率 3.61（约 5 小时）

每个点都新起一次部署。`down.sh` 之后，等两台节点的显存都释放完再起下一个（`README.kit-210.md` 说要 10–40 分钟）。

```bash
# 4+4 C80
PREFILL_WORKERS=1 PREFILL0_GPUS=4,5,6,7 DECODE_GPUS=4,5,6,7 DECODE_REQUEST_WEIGHT=0 SIM_ACCEPT_LEN=3.61 RUN_ID=p-44-c80 bash engine/up.sh
CONC=80 bash engine/agentx/agentx.sh; bash engine/down.sh
# 1P1D C88
PREFILL_WORKERS=1 PREFILL0_GPUS=4,5,6,7 DECODE_REQUEST_WEIGHT=1000000 SIM_ACCEPT_LEN=3.61 RUN_ID=p-1p1d-c88 bash engine/up.sh
CONC=88 bash engine/agentx/agentx.sh; bash engine/down.sh
# 2P1D C176
DECODE_REQUEST_WEIGHT=0 SIM_ACCEPT_LEN=3.61 RUN_ID=p-2p1d-c176 bash engine/up.sh
CONC=176 bash engine/agentx/agentx.sh; bash engine/down.sh
```

- **对比基准**：
  - 最干净的对照：如果 AUS 节点上还有 `infera-sglang:pr210-aab4cc36`，在同一组节点上用原 kit 跑同一个点做 A/B。那个镜像要用原 kit 跑，本 kit 跑不了它。
  - 其次是 `README.kit-210.md` 里 crsuse2 的数字：4+4 C80 37,310，1P1D C88 31,088，2P1D C176 38,392 tok/s/GPU。集群不同，只能看大致是否一致。
  - AUS 上已有的 #210 镜像结果（Infera 仓库 `llying/1p1d-decode-balance-aus-20261021/`，1P1D C88）用的 decode 权重是 0 和 180,000，和这里的 1,000,000 不同，只能参考。
- **通过标准**：与同集群基准相差在约 3% 以内。原样镜像的 Mooncake（`78566d45`）比 #210 的 `faae8dd4` 新 305 个提交；去掉 B5 以后 MLA 的 write-back 走 JIT，可能更快。
- 时间够的话，再补 1P1D 的 C32、C48、C64 和 4+4 的 C64，把整条 Pareto 线对齐。

## 4. 拆栈后的释放时间（每次 `down.sh` 时顺手记）

InferenceX 每个作业结束后要释放节点，所以要记两个数：

- prefill 节点上 HiCache 的 1 TB 锁页内存多久回到 `free -g`；
- 两台节点的显存多久回到 0（看 `rocm-smi --showmemuse`）。

## 5. 可选项（各 20 分钟，有时间再做）

- `MOONCAKE_DISABLE_HIP_DMABUF=0` 跑一次 4+4 起栈。以前用旧 Mooncake 时一起栈就段错误；换成原样镜像的 Mooncake 后会怎样，记下来就行，不用调试。
- `MOUNT_HOST_IONIC=0` 跑一次 4+4 起栈。只有第 1 步里不挂载也能列出网卡时才需要做。

## 需要带回来的东西

- `$WORK_DIR/runs/` 下每个 run 目录：`engine-config.env`、`workers.json`、`logs/`、`smoke/`、`gsm8k/`、`agentx/`；
- 第 1 步预检的输出；
- 第 4 步记下的时间。

## 常见失败及含义

| 现象 | 含义 / 处理 |
|---|---|
| `apply.sh` 报 MISSING 或 anchor 不匹配 | 镜像不是 20261018 这一版，先查 digest |
| `Found 0 HCAs`、`does not support the kernel ABI`、KV 传输退回 TCP | ionic provider 不匹配，检查 `MOUNT_HOST_IONIC` 和 `HOST_IONIC_PROVIDER` |
| prefill 一起栈就段错误 | `MOONCAKE_DISABLE_HIP_DMABUF` 没设成 1 |
| `/health` 通了，但 router 里迟迟没有 prefill | infera 启动探测要先把 KV 送到每个 decode rank，失败就是 RDMA 不通，看 prefill 日志里 Mooncake 的报错 |
| `AttributeError ... embedding_dim` | B7 没有打上 |
| `Expected lengths.size(0) == B` | B1 没有打上 |
| decode 卡住，所有 DP rank 都不出步 | B2 没有打上 |

## 附录：补丁核对结论（在 20261018 镜像源码 `e9b22a08` 上读的代码）

| 补丁 | 结论 | 源码依据 |
|---|---|---|
| B1 indexer DP 补齐行 | **要** | `dsa_indexer.py:1366-1376` 的 aiter 分支把补齐后的 `q_fp8` 整个传进去（XPU 分支切到 `[:q_offset]`），`1438` 的恢复也排除了 HIP |
| B2 DSA DP 同步死锁 | **要** | `dsa_backend.py:1102` 的 `seq_lens.max().item()`，以及 `1172-1179` 两次 `.cpu()`，都在各 rank 各走各的分支上 |
| B3 page table 行数 | 不要 | `index_share_for_mtp_iteration:false` 时 `index_topk_share.py:79` 直接关掉共享，每个草稿步各算各的 top-k；补丁自己的记录也写着这个参数可以替代它，并在 2×8 MI355X 上验证过 |
| B4 非末块等待事件 | 不要 | 补丁只改 `prefill.py:1078-1082` 的 `if self.enable_overlap` 分支；PP prefill 用了 `--disable-overlap-schedule`，`scheduler.py:495` 会让 `enable_overlap=False`，被改的代码根本不执行 |
| B5 HiCache staged write-back | 不要 | MLA 池在 HIP 上开 JIT，DSA indexer 池不开，所以 group 的 `supports_per_pool_backup_indices=True`（`group.py:59-60`）；控制器按池分别搬索引，只把 JIT 池的 host 索引搬到 CPU（`hybrid_cache_controller.py:840-846, 1445-1448`）。v0.5.16 上那个崩溃的路径在这里走不到。MLA JIT write-back 在 ROCm 上实际跑，要靠第 2 步确认 |
| B6 KV metrics 阻塞发送 | **要** | `kv_events_publisher.py:89-90` 仍是阻塞发送，目标是 connect 到 `metrics_ipc_name` 的 PUSH（`ipc_channels.py:76`），SGLang 里没有任何地方 bind 它；prefill 开了 KV events。v0.5.21 上 py-spy 抓到过 scheduler 卡在这里（yihou R001），v0.5.22 这段代码没变。我用 SGLang 同样的 socket 参数做了一个简化测试，没有复现阻塞，所以具体触发条件还没定位；但代码没变，又有实测记录，保留 |
| B7 PP 中间级无 embedding | **要** | `deepseek_v2.py:2987, 3022` 读 `self.embed_tokens.embedding_dim`，中间 PP 级是 `PPMissingLayer`（`2923`） |
| B8 absorb-BMM 配置 | **要** | `forward_mla_rocm.py:156` 的 `_absorb_bmm_config` 还在；sglang#46829 在 10-27 用 Triton 3.7.0 / gfx950 复现过，原样镜像正是 Triton 3.7.0 |
| B9 Responses 工具参数、decode radix | 不要 | AgentX 走 `/v1/chat/completions`；decode radix 补丁在这个 base 上本来就是空操作 |

网卡和 Mooncake：

- 原样镜像带 libionic `54.0-149.g3304be71`、rdma-core 39.0，以及 Mooncake `78566d45`，它包含 #210 用的 `faae8dd4` 及其所需的上游修复。
- Mooncake 里，`MC_DISABLE_HIP`、`MOONCAKE_DISABLE_HIP_DMABUF`、`MC_ENABLE_DEST_DEVICE_AFFINITY`、`MC_MAX_MR_SIZE` 这几个变量都存在；`MC_ENABLE_DEST_LOCAL_RAIL`、`MC_IB_TIMEOUT` 不存在。
