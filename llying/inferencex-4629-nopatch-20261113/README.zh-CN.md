# InferenceX#4629：去掉 SGLang 补丁、换镜像、AUS 验证（11-13 起，持续更新）

目标：sglang#33059、#46830、#46820、#21194 合入，srt-slurm#630 更新之后，
[InferenceX#4629](https://github.com/SemiAnalysisAI/InferenceX/pull/4629)（`glm5.2-fp4-mi355x-infera-sglang-agentic-disagg`）
不再给 SGLang 打补丁，用原样 lmsysorg 镜像跑通 AgentX 和 GSM8K，并保持 v0.5.22 时的性能。
上一阶段（带补丁的版本、与 #210 的性能对齐）见 `../inferencex-aus-e2e-20261029/`。

## 0. 当前状态（随时更新）

- 11-18 01:30 UTC：镜像定为 `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261117`（digest `sha256:e016f334…`）。
  单节点已验证：不打补丁能启动，decode 每步 60.3 ms。2 节点验证（4+4 C80，AgentX fast + GSM8K eval）在 holder 36659 里排队，预计 03:57 UTC 开始。
- PR 分支本地提交 `08902ae6`（只改镜像和 perf-changelog 那条记录的镜像名），验证通过后合并 upstream main 再推送。
  PR 目前是关闭状态（reviewer 之前关的，用户同事建议不要我们自己 reopen），reopen 由用户和 chunfangamd 沟通。
- 还缺：Infera 0.3.4 发版（带 AMD-AGI/Infera#239 的 memlock 修复）之后，`router.version` 改 0.3.4，删掉安装脚本里的 `prlimit` 一段。
- Triton 3.8 编译退化的 issue 草稿在 `issue-drafts/`，未发布。

## 1. 结论汇总

| 问题 | 结论 | 处理 |
|---|---|---|
| 补丁 B1/B2/B6/B7 | 对应的 sglang PR 都已合入 nightly | PR 里删除 |
| B8（absorb-BMM 配置，Triton 3.7 上 `iota_range` 断言） | rocm10 镜像（Triton 3.8）不触发；sglang#49732 11-17 合入后 rocm720 nightly 也不再需要 | PR 里删除 |
| Mooncake 注册 KV 时 ENOMEM | RLIMIT_MEMLOCK 软限制（Slurm 传进来的 8227232 KB）加并发 1 GiB 注册，见第 3 节 | 安装脚本 `prlimit` 抬软限制（PR `fabd6a8e`）；Infera#239 在包装层做同样的事 |
| amdgpu 6.19.14 节点上注册后 hipErrorOutOfMemory | 6.19.14 对锁定显存重复计数（peer 和 dma-buf 都会），6.16.13 只有 dma-buf 会 | 端到端验证只用两台 6.16 节点；驱动侧另一条工作线（`../dp-graph-variant-fix-aus-20261103/driver619/`） |
| rocm10 镜像 AgentX 吞吐低约 10% | Triton 3.8 编译 DSA sparse MLA decode kernel 时寄存器溢出，加上 autotune 选配置不稳定，见第 4 节 | 换 rocm720-20261117；issue 草稿 |

## 2. rocm10-20261112 镜像的端到端结果（4+4 C80，两台 6.16 节点）

| 测试 | 时间 / 节点 | 结果 |
|---|---|---|
| AgentX fast（1200 s） | 11-16 20:04–21:10，n01-33 + n02-21 | 29,933 tok/s/GPU，TTFT p50 2.08 s / p90 4.06 s，ITL p50 24.8 ms，2,380 个请求、0 出错 |
| GSM8K（eval-only，真实接受率，1319 题） | 11-17 08:12–08:37，n06-25 + n06-33 | strict 0.9666 / flexible 0.9682 |

memlock 用的是 TEST-ONLY 的 Infera 包装层覆盖（#239 的改动叠加在 0.3.3 上）。两次 ENOMEM 都是 0。
同一点位 10-30 的 v0.5.22 加 5 个补丁是 33,121 tok/s/GPU（fast）；11-02 的 rocm720-1101 nightly 加 4 个补丁是同样水平。rocm10 低约 10%，原因见第 4 节。

## 3. ENOMEM 根因

- ionic 驱动的 `ib_umem_get_flags` 先走内核的 `ib_umem_get`（主机内存路径）：把页数计入 `pinned_vm`，没有 CAP_IPC_LOCK 时超过 RLIMIT_MEMLOCK 软限制就返回 `-ENOMEM`；只有返回 `-EFAULT` 才回退到 peer memory。
- pyxis 重映射出来的 root 没有 CAP_IPC_LOCK；软限制是 Slurm 传进来的 8227232 KB，硬限制是 unlimited。Mooncake 按 1 GiB 分块并发注册 KV，计数叠加超过软限制就失败。
- 修复：worker 启动前把软限制抬到硬限制。不需要集群配置改动，也不需要 srt-slurm 改动。
- 诊断脚本在 `scripts/enomem-diag/`（`kfd_pin_repro.py` 复现注册，`regdiag.c` 是 LD_PRELOAD 注册诊断，`vma_sampler2.sh` 采样 VMA，`memlock_ab.sh` 做 memlock A/B）。

## 4. rocm10 decode 慢 10% 的根因（全部在单节点上查）

方法：借用 1 节点（作业 36271 的 n02-25，用户同意），两个 4 卡 server 并行跑不同镜像。参数照 decode 角色（TP4、DP4 attention、EAGLE 5/1/6、合成接受率 3.61、fp8 KV），不走 PD。
固定负载：48 个 170k token 的随机请求，prefill 结束后测 3 分钟稳态，再抓 torch profile。脚本在 `scripts/decode-bisect/`（`round_env.sh` → `in_container_env.sh` → `load.py`）。

| 镜像 | 补丁 / 环境变量 | 每步 decode（12 请求/rank，约 2.1M KV token/rank） |
|---|---|---|
| rocm720-1101 | 4 个补丁 | 59.7 ms |
| rocm720-1113 | 只有 B8 | 60.8 ms |
| rocm720-1117 | 无 | 60.3 ms |
| rocm10-1112 | 无 | 78.6 ms（两个 rank 选到更差的配置，其余 rank 在 NCCL 里等） |
| rocm10-1112 | `SGLANG_TOPK_V2_MODE=0` | 71.6 ms（rank 间均衡；top-k 模式本身影响约 0.1 ms） |
| rocm10-1112 | `TRITON_HIP_USE_ASYNC_COPY=1` | 71.7 ms |

- 慢的是 `_sparse_mla_fwd_split_dim_kernel`：一个 rank 超过 64 行 query（11 个以上请求、每个 6 个 verify token、64 个头）时才走它，每步约 79 次。server profile 里单次调用 113 µs（Triton 3.7）对 267–361 µs（Triton 3.8）。
- 同一份 SGLang 源码（1113 和 1117 的 rocm720 镜像已经包含 11-12 之前的全部改动），aiter 是同一个提交 `285579b6`，所以原因在 Triton 3.8 这一层。
- microbenchmark（`scripts/decode-bisect/mla_bench.py`，数据在 `results/mla-bench/`）：同一配置 (BLOCK_N 64, warps 4, stages 2) 在 Triton 3.7 上 117.6 µs，在 3.8 上 191.4 µs；3.8 上最好的配置 (32, 4, 2) 也要 166 µs。索引局部还是分散对耗时没有影响，所以不是 cache 或 TLB 的问题。
- 编译产物（`results/mla-bench/vgpr-spills.txt`）：所有 `num_stages=2` 的配置在 3.8 下 VGPR 用满 512 并溢出到 scratch，例如 (64, 4, 2) 溢出 8 个变成 373 个；`num_stages=1` 的配置两版几乎相同。
- autotune 在启动时用占位数据、每个 rank 各自测：rocm720 上 4 个 rank 都选 (64, 4, 2)；rocm10 上选出 (32, 2, 2)、(64, 2, 1) 等 262–361 µs 的配置，rank 之间还可能不一样。
- 已排除：SGLang 代码改动（包括 #47960 默认 exact top-k）、DSA indexer（paged-MQA logits 255 对 252 µs）、MoE 和 FP4 GEMM（单次相同）、Infera memlock 修复。
- 更正：中途一度以为变慢随 KV 长度增长、出在 indexer 上。线上日志里的相关性来自高负载：请求多时才走这个 kernel，而那时上下文也长。

## 5. 换到 rocm720-20261117

- 镜像构建参数 `SETUPTOOLS_SCM_PRETEND_VERSION=0.5.23.dev20261117+g4469f9fc7f`，仓库 HEAD `703e763d`（比 #49732 的合入提交 `4469f9fc` 新 4 个提交），#33059 / #46830 / #46820 / #21194 都在 11-11 前合入。Triton 3.7.0。
- n02-25 的 docker 数据盘满了（都是别人的镜像），dockerd 导入不了。改用 `scripts/decode-bisect/pull_image.py`：匿名从 registry 下 52 层，按序解包并处理 whiteout，写入 enroot 会加的 `/etc/environment` 和 `/etc/rc`，再用 `mksquashfs -comp lz4` 打包。squash 在 `/apps/liyingli/squash/`。
- AUS 测试树（`~/code/bench_agentx/infx-aus2`）和 PR 对齐：撤掉 TEST-ONLY 的 Infera 覆盖（`c321e0f`），用 PR 的安装脚本（`0b09514`，带 `prlimit`），630 补丁按本树的 srt-slurm 固定提交 `acfb4dcc` 加 573 补丁重新生成（`9447741`，内容与 PR 一致），镜像换成 1117（`8be5c5e`）。
- holder 队列（`scripts/holder/`）：`88-require-6.16` 检查两台节点驱动，不是 6.16 就加进排除列表，由 `resubmit_watch.sh` 重新提交；然后清理残留容器 → fast → 清理 → eval → 失败才重跑 → `99-stop` 让 holder 退出。

## 6. 时间线

- 11-13 起：PR 去掉 B1/B2/B6/B7/B8、apply.sh 和 waiver；rocm10 镜像编译 absorb-BMM 测试通过；PR 换 rocm10-20261112 镜像。
- 11-14 至 11-16：rocm10 2 节点验证遇到 ENOMEM，在 6.16 和 6.19 节点上分别诊断，定位到 memlock 软限制；Infera#239（draft）和 PR 安装脚本 `prlimit`（`fabd6a8e`）；630 补丁更新到 `290bc0de`（`89594d7f`）。
- 11-16 20:04–21:10：B2 fast 通过。11-17 08:12–08:37：GSM8K 通过。PR 曾 reopen，按用户同事意见又关闭。
- 11-17 17:44–19:27：单节点定位 decode 慢 10% 的原因，确认 rocm720-1117 可用。
  17:43–17:52 另一条工作线（driver619）也在 36271 里通过 overlap 跑了 Mooncake 构建和测试，与我的第一轮启动重叠，两边都没有受影响（他们 rc=0；我的第一轮因 JIT 缓存路径问题重跑）。
- 11-18 01:30：AUS 测试树对齐 PR 并换 1117 镜像，2 节点 holder 36659 排队。
