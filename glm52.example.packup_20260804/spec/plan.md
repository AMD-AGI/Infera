# GLM-5.2 SGLang + Mooncake 1P1D + DPA + MTP — deliverable example

## Context

infera 已经在两个集群上把 GLM-5.2 的 PD 部署跑通并做了 agentic bench（infera 自身的
Optimus-AgenticBench，以及客户在 ROCm/MAD PR #173 提供的 AgentX Case-A）。这些成果目前散
落在 `../infera.merge.liying.kv.mtp/*.packup_*` 里 —— 每个 packup 都带着本地节点名、
spur job id、adhoc 构建的镜像 tag、以及调试过程。它们是实验记录，不是可交付物。

本任务把这些成果收敛成 `examples/` 下一个**可交付的正式部署样例**：一键/几键脚本、
标准发布镜像占位符、零本地信息、DPA 与 non-DPA 两套部署、RDMA preflight 使用说明，
以及一个 `results/` 目录记录 conc=8 下两种 bench 的实测数字与分析。

预期结果：客户拿到这个目录，改一个 wrapper 文件里的集群参数，就能把 1P1D + mooncake +
DPA + MTP 的 GLM-5.2 服务拉起来并自检。

## 已确认的决策

| 问题 | 决策 |
|---|---|
| 目录形态 | 仿 `examples/deepseek_v4`（`common.sh` + 子目录），不用扁平编号脚本 |
| transport | **两份 wrapper，不自动探测**：peermem 多网卡型 / dmabuf 单 ODP 网卡型 |
| 客户 bench | 只在 `results/` 与 README 引用上游，**不提供任何 aiperf 启动脚本** |
| 镜像 | 统一占位符 `inferaimage/infera-sglang:0.2.0`，README 标注待定 |
| results 内容 | vultr + spur 两个集群 × infera bench + 客户 bench，并给出跨集群对比分析 |

## 交付目录

`examples/sglang_1p1d_glm5.2/`（命名与既有 `sglang_1p2d_kimi2.6` 对齐）

```
README.md                     # 主文档：拓扑 / 前置 / preflight / 部署 / 自检 / 推荐配置 / 踩坑
common.sh                     # 参照 examples/deepseek_v4/common.sh：require_env / start_container
                              #   / start_etcd / start_router / wait_health / smoke / reap
preflight_rdma.sh             # 薄封装，跑 infera.tools.preflight[.mooncake_mode]（容器内）
cluster/
  README.md                   # 「要按你的集群改，就改这里」——唯一需要用户编辑的地方
  cluster.peermem.sh          # wrapper A：多 ionic 网卡 + peermem，dmabuf OFF
  cluster.dmabuf.sh           # wrapper B：单 mlx5 ODP 网卡、无 peermem，dmabuf ON + MC_MS_FILTERS
engine/
  leg.sh                      # 真实启动器：一条 PD leg，全部旋钮走 env，无任何本地默认值
  up.sh                       # 两节点一键：etcd + router + prefill + decode
  smoke.sh                    # /v1/workers + 一次 chat + 五项特性证据
  bench.sh                    # sglang 自带 bench_serving 参考压测
  down.sh                     # 拆除并等 VRAM 释放
results/
  README.md                   # 索引 + 两 bench 语义差异 + 跨集群（spur vs vultr）分析
  infera_agenticbench_conc8.md
  customer_agentx_caseA_conc8.md
```

## 关键设计

### 1. 本地化参数全部落在 `cluster/*.sh`（要求 7）

`engine/leg.sh` 不含任何默认 IP、路径、NIC 名、GID。wrapper 只负责 `export` 一组变量后
`exec` 到 `engine/up.sh`。两份 wrapper 的差异就是 transport 那一段：

| | `cluster.peermem.sh` | `cluster.dmabuf.sh` |
|---|---|---|
| 对应集群形态 | 多张 ionic RoCE 网卡，`amdgpu` peer-mem 已加载 | 单张 mlx5（有 ODP），无 peer-mem |
| `MOONCAKE_DISABLE_HIP_DMABUF` | `1` | `0` |
| `MC_MS_AUTO_DISC` / `MC_MS_FILTERS` | 不设 | `0` / `<ODP 网卡名>` |
| `MC_GID_INDEX` | 逐节点探测（首个非空且非 `fe80::` 的 GID） | 固定，由 preflight 报告读出 |
| `--disaggregation-ib-device` | 全部 `PORT_ACTIVE` 网卡 | 只那一张 ODP 网卡 |
| `RDMAV_FORK_SAFE` | `1` | 仅当节点上还有非 ODP 网卡 |

这张表的每一行都直接对应 `infera/tools/preflight/mooncake_mode.py` 里 `_eval_mode_a`
（模式 A）与 `_eval_mode_b`（模式 B）已经算好的 `env` + `launch_flags`——README 会明确
说明「preflight 输出哪个模式，就用哪个 wrapper」，而不是让用户自己拼。

### 2. DPA / non-DPA（要求 3、5）

`engine/leg.sh` 用 `DPA=0|1` 控制，且必须避开三个已被一线证据坐实的陷阱：

- **`--ep-size` 与 `--enable-dp-attention` 解耦**。两者是不同并行轴（专家 vs 注意力）；
  耦合在一个 `if` 里会让 `DPA=0` 顺带把 MoE 从 ep8 塌成 TP 默认，延迟差异就无法归因。
  `--ep-size $TP` 无条件下发。
- **`--chunked-prefill-size` 是全局预算，且只在 DPA 开时被 sglang 除以 `dp_size`**。
  脚本必须让 `CHUNK` 可以显式传入，默认不因 DPA 翻转而暗改 8×。
- **`--mem-fraction-static` 与 DPA / 路由策略耦合**。prefill 侧 activation OOM
  （`HSA_STATUS_ERROR_OUT_OF_RESOURCES` 且 token usage 极低）要**调低**而非调高，方向
  与 decode 侧 retract 修法相反。脚本按 role 分开取值并在注释里写清诊断规则。

README 的**推荐配置**基于 spur 上 armA/armB 这对实测（两者只差 DPA + 路由策略两个变量，
不能做单变量归因，README 会照实说明）：prefill DPA off + kv-aware，decode DPA on + MTP。

### 3. RDMA preflight 使用方式（要求 6）

不新写探测逻辑。README 用仓库已有的两件工具，说明「看什么、看到什么就改哪个 flag」：

- `python -m infera.tools.preflight.mooncake_mode` —— 逐网卡列出 vendor / 链路速率 /
  ODP / **PCI BDF** / NUMA / GID index，节点级列出 **peer-mem 是否加载**，并把 A/B/C
  三种模式各自「可用 / 不可用 + 原因」连同**现成的 env 与 launch flag**一起打出来。
  退出码 2 = 只剩需要给 KV 池设上限的路径，需要人工确认。
- `python -m infera.tools.preflight --netperf --mooncake`（多节点 srun）—— 跨节点 RoCE
  带宽矩阵，以及 mooncake KV 搬运在 rdma / tcp 下的**分别**读数。README 会点明这是唯一
  能把「静默退化成 TCP」变成一个可读数字的检查。

`preflight_rdma.sh` 只是把这两条命令包进正确的容器参数（`--device=/dev/infiniband`、
`--cap-add=IPC_LOCK`、不要用 `--entrypoint ""`）。

### 4. `results/`（要求 4）

两份 bench，各自成篇，明确写清**不可直接横向对比**（闭环 session 驱动 vs 开环冻结 trace
重放；集群不同）。数据来源全部是一线 packup / 原始 artifact：

| bench | vultr | spur |
|---|---|---|
| infera Optimus-AgenticBench（conc=8 = par8 workload） | `par8.glm52.dpaoff.packup_20260803` | `par8.armB.dpaoff.kvaware.spur.packup_20260804`（DPA off + kv-aware）、`par8.armA.dpaon.roundrobin.spur.packup_20260804`（DPA on + round-robin） |
| 客户 AgentX Case-A（c8） | `agentx.caseA.customer.packup_20260803/results/c8` | `/shared_nfs/yihou_agentx_caseA/bench/results/c8`（c16 亦已完成） |

spur 侧客户 bench 的百分位阶梯用 packup 自带的 `scripts/analyze.py` 从原始
`profile_export.jsonl` 重算，不抄 summary 行。已跑通一次，输出正常。

**spur 为何看起来更差**——README 会列出候选原因并标注各自的证据等级，不下定论：

1. **fabric**：spur 单张 mlx5 + dma-buf；vultr 8 张 ionic + peermem。按 preflight 自己的
   带宽口径是 ~200G vs ~3200G 聚合。**一线事实，但未做控制实验**。
2. **chunk 不同**：spur 全局 65,536 vs vultr 全局 16,384 → spur 每次 forward 的 prefill
   工作量是 8×。两个参考 kit 本身就不一致，armB 的 notes 记录了这个未解分歧。
3. **GMU 不同**：spur prefill 0.70 vs vultr 0.80 → KV 池更小（armA 实测 −19%）。
4. **MTP 与 decode radix cache 上游互斥** → `decode_prefix_len` 恒为 0，每一轮都要重传整个
   prompt 的 KV，于是单轨 fabric 落在每一轮的关键路径上。这条解释为什么 fabric 差异在这个
   workload 上会被放大。

四条里没有任何一条是被单变量实验证实的；README 会写明要证实它需要哪一次缺失的对照运行。

### 5. 严禁出现的内容（要求 1）

节点名（`chi28xx` / `crsuse2-m2m-xxx`）、spur job id、`/mnt/vast` 与 `/shared_nfs` 路径、
跳板机 IP、`infera/engine-sglang:merged-*` 这类 adhoc tag、分支名与调试过程 —— 一律不进
脚本。必要的避坑提醒改写成脚本注释与 README 的「Notes & gotchas」条目，只保留机制，不保留
现场。镜像统一 `inferaimage/infera-sglang:0.2.0`，README 注明待正式发布后替换。

## 验证方式

本任务不申请集群资源，验证限于静态与本地可执行的部分；需要真机验证的部分明确列为未验证。

1. `bash -n` 全部脚本；有 `shellcheck` 则一并跑。
2. **flag 存在性核对**：`engine/leg.sh` 与 router 用到的每个 infera flag，逐个 grep 本仓库
   `infera/engine/sglang/args.py` 与 `infera/server/args.py` 确认在 `main` 上存在。
   （已抽查：`--infera-kvd-socket` / `--kv-events-bind` / `--kv-snapshot-port` /
   `--kv-prefill-overlap-weight` / `--kvd-socket-path` / `--router-backend` 均在。）
3. `python -m infera.server --help` 与 `python -m infera.engine.sglang --help` 在本地跑通，
   确认 README 里引用的参数名拼写正确。
4. `results/` 中每个数字都能回溯到一个 packup 文件或原始 artifact；spur 客户 bench 的阶梯
   用 `analyze.py` 重算并与写入文档的值核对一致。
5. **未验证项**（README 的 Validation status 章节照实登记）：整套脚本没有在真机上端到端跑过
   ——它是从两个集群上已验证的 leg 脚本重构而来，重构本身未经运行验证。若需要，可另外申请一次
   spur 分配做一次冷启动 + smoke 验证。

## 工作流程（遵循用户级 CLAUDE.md）

1. 建子 workspace `work.glm52.example/`，所有临时产物留在其中。
2. 备份现有项目 `CLAUDE.md`（当前处于 deleted 状态，按 git HEAD 内容恢复后备份为
   `CLAUDE.<keyword>.<YYYYMMDD-HHMM>.md.bak`），再写新的 `CLAUDE.md`。
3. 先写 `common.sh` + `engine/leg.sh`（最核心），再 wrapper，再 up/down/smoke/bench。
4. 最后写 README 与 `results/`。
