# decode 集合通信的 graph 捕获拷入：修"开头坏"与 E11 乱码（141，11-03 起）

目的：
1. 验证 aiter custom all-reduce 在 CUDA graph 捕获时改为"拷进预注册 IPC 池"（`SGLANG_AITER_AR_CAPTURE_COPY_IN=1`）能否去掉 E10 栈里的"开头坏"，同时保留 custom all-reduce（CAR）。
2. 验证 graph 内 aiter all-gather / reduce-scatter 的拷入（镜像自带开关 `SGLANG_AITER_CAPTURE_COPY_IN=1`，默认关）能否让 E11（`SGLANG_USE_ROCM700A=0`）恢复正确。

背景见 `../best-4p4-51k-20261103/README.zh-CN.md` 4.1 节。

## 环境

- 节点 crsuse2-m2m-141（11-03 03:33 空闲：无容器、8 卡显存 0）。同机 PD：prefill GPU 0–3（TP1 × PP4），decode GPU 4–7（TP4 / DP4）。
  同机时 Mooncake 走 XGMI 上的 GPU IPC，不走 RDMA（yihou `glm52.samenode-p4d4.agentx-c40.packup_20260922`），KV 传输时序与跨机不同。
- 镜像：138 上的 `infera-sglang:v0522-fdbafb60`（sha `509b8427`，与 E10 相同），`docker save | ssh 141 docker load`。
- kit：`2p1d-kit/llr35-141` = `llr35` 的副本，只在 `engine/up.sh` 加了一处：同机时等 prefill `/health` 后再起 decode
  （两个 leg 同时初始化 RCCL 会报 `hipIpcGetMemHandle failed`，yihou 同机记录的第 3 个坑）。
- 配方：E10（`../prefill-opt-20261027/config/site-llr35-v3-evict512k-bq16.env`），**HiCache 关**（`PP4_HICACHE=0`），
  prefill 设备池限 30 万 token / 级（同限池门禁）。真实接受率。
- 日志：`/home/liyingli/bench_agentx/llr141/logs/`（队列 `q141-1103.out`，每个部署 `gate/<label>.log`、`.probe.jsonl`、`.longctx-16000.jsonl`），
  部署目录 `/home/liyingli/bench_agentx/llr141/work/runs/<label>/`。题目 = 138 上的 1,319 题 GSM8K（同 night31 / 33）。

| 配置 | 改动 |
|---|---|
| `config/site-141-ctl.env` | 对照：E10，CAR 开，镜像原行为 |
| `config/site-141-ctl-hc.env` | 对照，开 HiCache（只在关 HiCache 复现不出时用） |
| `config/site-141-arcopy.env` | + decode 挂 `../prefill-car-20261023/patches/custom_all_reduce.py`，`SGLANG_AITER_AR_CAPTURE_COPY_IN=1` |
| `config/site-141-e11.env` | E11 对照：decode `SGLANG_USE_ROCM700A=0` |
| `config/site-141-e11-copyin.env` | E11 + `SGLANG_AITER_CAPTURE_COPY_IN=1` + all-reduce 拷入 |

队列 `scripts/q141_1103.sh`：对照（400 题 + 16k × 64）→ 若开头坏 < 3 则开 HiCache 重做对照 → all-reduce 拷入（600 题 + 16k × 64）
→ E11 对照（200 题）→ E11 + 拷入（600 题 + 16k × 64）。

## 预测与判据（开跑前写下）

- **对照**：138 → 140 限池门禁开 CAR 时 8 次里 6 次开头坏 3–8 / 200。同机、关 HiCache 的条件不同，复现率未知；400 题里 ≥ 3 条才算复现，否则改开 HiCache。
- **all-reduce 拷入**：若"开头坏"来自 CAR 注册 graph 池输入，600 题应 ≤ 1 条（对照按 1.5–4% 折算约 9–24 条）。≥ 3 条则拷入不是解法，回到关 CAR 量代价。
  拷入只多一次每层的小拷贝，正确性上不应引入新错误：乱码 0、长上下文与对照同水平。
- **E11 对照**：140 上探测 0.670、乱码 44 / 200；141 上应同样大量乱码（只与 decode 的 graph 内通信有关）。
- **E11 + 拷入**：若乱码来自 graph 内 all-gather / reduce-scatter 的注册路径（镜像源码注释描述的情形），600 题乱码 0、正确率 ≥ 0.97、
  开头坏与 all-reduce 拷入同水平；否则 E11 的问题在别处（10-30 记录的候选：MAX_LEN 下 idle rank 的 pad 行、< 2048 token 的注意力路径）。

## 结论（11-03 06:40 更新）

1. **找到 E11 乱码的根因**：镜像在 HIP 上为 DSA 模型每个 batch size 捕获 dense（`max_kv_len ≤ 2048`）和 sparse 两套 decode graph，
   **每个 DP rank 按自己 batch 的最长上下文各选一套**（`graph_variants.py` `DsaGraphVariants.select`；没有请求的 idle rank 一律 sparse）。
   于是同一步里不同 rank 会回放不同的 graph，而 aiter collective 在捕获时注册的是"本 graph 的输入地址"，对端读到的是另一套 graph 的旧缓冲。
   两种修法都在 141 上验证有效：捕获时拷入（乱码 26 / 200 → 0 / 600），或把所有 rank 固定到 sparse graph、不拷入（→ 0 / 200）。
   这也解释了 10-30 记下的现象：短 prompt 坏、长上下文好（长上下文所有 rank 都是 sparse）、smoke 单请求就坏（一个 rank dense、三个 idle rank sparse）。
2. **E11 的乱码里有和"开头坏"一模一样的形状**（首 token 正确、紧接一串乱码、后面又回到正确答案）。所以 E10 的"开头坏"很可能是同一机制的低频版
   （E10 在 graph 内只有 all-reduce 一个注册路径的 collective，关 CAR 就消失）。但 141 上 E10 对照本身就是 0，E11 + 拷入的 0 不能证明拷入修好了 E10 的开头坏；
   要在跨机部署上验证，脚本 `scripts/q140_1103.sh` 已备好、未启动。
3. **固定 sparse 还能省 decode 的空闲**：`select()` 每步调一次 `seq_lens.max().item()`，是 10-30 decode trace 里 GPU 空闲的最大单一来源（约 1.0 ms/步，
   外加那次 39 ms 停顿），见 `../best-4p4-51k-20261103/` 6.5 节。AgentX 的请求上下文都远超 2048，本来就全是 sparse，固定后不多算。
4. **关 CAR 只掉 1.4%**（50,670 对 51,374，night35 实测），可以作为现成的正确配置兜底。
5. 同机 PD 需要两处 kit 改动（等 prefill 健康后再起 decode、decode 空转进程号避开 RCCL 共享内存撞名），还有偶发的 aux 数据发送失败（每次 0–13 个请求报 500）。

## 结果

### 起栈：同机 PD 的两个坑（03:42–04:01）

1. 第 1 次（`site-141-ctl-1103T0342`）：prefill 就绪后才起 decode（启动门禁生效），decode 仍在 TP 组 `ncclCommInitRank` 报 `HIP failure: 'invalid argument'`。
   原因：两个容器都是 `--ipc host`，RCCL 按进程号给共享内存命名，每个容器的进程号都从很小的数开始，同机的 prefill 和 decode 撞名。
   kit 原本只给同机的第二个 prefill 空转 3 万个进程号；`llr35-141/engine/worker.sh` 给同机的 decode 加了空转 6 万个。
2. 第 2 次（`site-141-ctl-1103T0349`）：decode 起来了，但 DP2 的一块 383 MB 缓冲 Mooncake 注册失败（`ibv_reg_mr` `Invalid argument`），
   infera 的 PD 启动探测连续 3 次失败，prefill 退出。与 10-29 在 140 上的偶发失败相同；队列改为起栈失败重试一次。
3. 第 3 次（`site-141-ctl-1103T0401`）正常，起栈约 7 分钟。

（第 1 次失败后，旧版队列把"起栈失败"当成"未复现"，转去开 HiCache 起了一次，已手动停掉，不计。）

### 对照（关 HiCache）：不复现

| 部署 | 探测 正确 / 乱码 / 开头坏 | 长上下文 16k × 64（捞针 / 长前缀 GSM8K） |
|---|---|---|
| `site-141-ctl-1103T0401`（E10，CAR 开，关 HiCache，限池 30 万） | 0.978 / 0 / **0**（400 题） | 32/32、31/32 |

与历史数据合起来，"开头坏"需要 prefill 开 HiCache **且** decode 开 CAR：

| prefill HiCache | decode CAR | 结果 | 来源 |
|---|---|---|---|
| 开 | 开 | 生产约 0.4%（1,319 题 4–6 条）；限池门禁 8 次里 6 次 3–8 / 200 | `prefill-opt` 10-28 至 11-02 |
| 开 | 关 | 限池门禁 3 次都是 0 / 200 | night31 / 33 |
| 关（带 v3） | 开 | 140 上 b3 + CAR、clean-k3 + CAR 各 0 / 1,319；141 上完整 E10 配方 0 / 400 | `prefill-opt` 10-27；本次 |

按出现率估算，HiCache 开、CAR 开时 400 题预计约 2–16 条，0 条的概率在 1% 以下（按 0.4% 的生产率算约 20%），所以关 HiCache 时大概率确实不出现。
这说明问题不只在 decode 的 CAR：prefill 侧的 HiCache（回载、驱逐写回、提前发送缓存前缀）也是必要条件。
"开头坏"的形状（首 token 由 prefill 算、正确，decode 从第 2 个 token 起坏）也更像 decode 收到的部分 prompt KV 不对；CAR 可能只是通过改变时序起作用。
all-reduce 拷入是否有效仍要实测：队列按预设分支改为开 HiCache 重做对照，复现后再测拷入。

### 对照（开 HiCache）：仍不复现；141 上无法检验 all-reduce 拷入

| 部署 | 探测 正确 / 乱码 / 开头坏 | 说明 |
|---|---|---|
| `site-141-ctl-hc-1103T0417`（E10，CAR 开，开 HiCache 250 GB / 级，限池 30 万） | 0.910 / 3 / **0**（成功的 387 条里） | 另有 13 条 HTTP 500 |

- `probe_badstart.py` 报 13 条，其实就是这 13 条 HTTP 500 的空输出（脚本把不足 25 字符的推理也算开头坏）；排除后开头坏 0 / 387。
  138 → 140 的限池门禁在同样条件下 8 次里 6 次是 3–8 / 200，按 1.5–4% 算 387 条应有 6–15 条。
- 13 条 500 来自同机传输：prefill PP2 `Failed to send aux data`，decode `Failed to get kvcache from prefill instance`。这是同机部署开 HiCache 时的问题，与本实验无关，另记。
- 解读：两次对照的差别只在 KV 传输路径。同机时 Mooncake 走 XGMI 上的 GPU IPC，跨机走 RDMA。"开头坏"很可能和跨机 RDMA 传输的完成时序有关，
  CAR 可能是通过改变 decode 侧的时序起作用。yihou 记录过同机强制走 RDMA（`MC_DISABLE_HIP=1`）的路径不传数据，所以 141 单机复现不了。
- 结论：**all-reduce 拷入对"开头坏"的效果只能在跨机部署上验证**。11-03 138 / 140 正在跑另一个会话的 `night35_1103.sh`（E10 关 CAR 计分 + Pareto 点），
  不能并行使用。旧队列 04:40 停掉，没有跑"开 HiCache + 拷入"那一步；改跑 `scripts/q141b_1103.sh`（只测 E11）。

### E11：graph 内 all-gather / reduce-scatter 拷入后输出正确（关 HiCache）

| 部署 | smoke | 探测 正确 / 乱码 / 开头坏 | 长上下文 16k × 64 |
|---|---|---|---|
| `site-141-e11-1103T0440`（E11 对照，`SGLANG_USE_ROCM700A=0`） | **4/6**（chat 输出 "39oss什么呢berman…"、stream 乱码） | 0.810 / **26** / 5（200 题，另有 3 条 500） | 未测 |
| `site-141-e11-copyin-1103T0459`（E11 + `SGLANG_AITER_CAPTURE_COPY_IN=1` + all-reduce 拷入） | **6/6**（chat "391"、stream "899"） | 0.978 / **0** / **0**（成功的 593 条里；另有 7 条 500） | 32/32、31/32 |
| 对照：`site-141-ctl-1103T0401`（E10，镜像原行为） | 6/6 | 0.978 / 0 / 0（400 题） | 32/32、31/32 |

- 容器内确认三个开关都生效（`docker exec env`），decode 日志 4 个 rank 都打出 `all-reduce capture copy-in` 和 `Using AiterCustomAllreduce`。
- 7 条 500 与开 HiCache 对照里的 13 条相同，都是同机传输 `Failed to send aux data`，请求直接失败，不是输出内容错。
- **结论：E11 的乱码来自 graph 内 aiter collective 的"注册 graph 池输入"路径**，与镜像源码注释描述的一致；改成捕获时拷入预注册 IPC 池后，
  smoke、600 题探测、长上下文都与对照同水平。按预先写的判据（乱码 0、正确率 ≥ 0.97、长上下文与对照同水平）通过。
  E11 这一步没有把两个拷入开关拆开测，所以不能说只靠 all-gather / reduce-scatter 那个开关就够。
- decode 步长粗比（探测负载，每 40 步一行日志，数据点少，只作参考；`analyze_pd_logs.py`）：

| 最忙 rank 请求数 | E10 对照 | E11 | E11 + 拷入 |
|---|---|---|---|
| 2 | 21 ms | 20 ms | 20 ms |
| 4 | 25 ms | 23 ms | 23 ms |
| 16 | 39 ms | 37 ms | 37 ms |

  E11 每步约快 1–2 ms，与 10-30 的预测（1.3–2.4 ms）一致；拷入没有看出额外开销。吞吐要在 138 → 140、开 HiCache 的 C104 上计分才算数。

### 两处统计口径的说明（11-03 06:10 用户问）

- **开 HiCache 对照的 3 条"乱码"不是开头坏**：都是生成到 16K 上限的长输出，前面推理正常，几千 token 之后才退化（doc 106 末尾 "( only ( ( ( (…"，
  doc 119 末尾一串数字，doc 100 是连贯地反复验算、被重复检测判为乱码）。这类"长生成后退化"在 138 → 140 的限池门禁里也有（开 CAR 时约 4 / 200，关 CAR 时 0），
  关 HiCache 的两次 141 部署里是 0。它和 graph 错配的关系还要验证：短 prompt 生成超过 2048 token 时，本 rank 从 dense 切到 sparse，正是错配的时刻之一（推断）。
  队列 `q141c` 的第 2 步（E10 + all-reduce 拷入，开 HiCache，600 题）检验它。
- **E11 对照的 5 条"开头坏"是 E11 自身的乱码**：例如 doc 0 "ThePOOLusta或少ledgearca0_cleanuposz…, she sells 16 - 3 - 4 = 9 eggs … #### 18"
  （最终答案正确）。E11 + 拷入里这 5 条消失，说明拷入修好了 E11；但 E10 对照在 141 上本来就是 0，不能据此说拷入修好了 E10 的开头坏。

### 根因验证：固定 sparse graph（11-03 06:14，`config/site-141-e11-sparse.env`，`patches/dsa-force-sparse/`）

补丁只改 `graph_variants.py`：`SGLANG_DSA_GRAPH_FORCE_SPARSE=1` 时 `select()` 直接返回 sparse（不设时与镜像相同；md5 基线 `c21985ce…` 与镜像一致）。
预测（开跑前）：若乱码来自 rank 之间回放不同的 graph，E11 不拷入、只固定 sparse 也应 smoke 6/6、乱码 0。

| 部署（141，关 HiCache） | smoke | 200 题 乱码 / 开头坏 | 正确（成功请求中） |
|---|---|---|---|
| E11 对照 | 4/6 | 26 / 5 | 162 / 197 |
| E11 + 拷入 | 6/6 | 0 / 0（600 题） | 580 / 593 |
| **E11 + 固定 sparse，不拷入** | **6/6** | **0 / 0** | **190 / 191** |

（9 条 500 是同机 aux 数据发送失败，prefill / decode 两侧各 9 条一一对应。）预测成立。

### 开 HiCache 时的"长生成后退化"：all-reduce 拷入后为 0（11-03 06:27，`config/site-141-arcopy-hc.env`）

| 141，开 HiCache，CAR 开，限池 30 万 | 乱码 | 开头坏 | 生成到 16K 上限 | 正确（成功请求中） | 长上下文 16k × 64 |
|---|---|---|---|---|---|
| 对照 `site-141-ctl-hc-1103T0417` | 3 / 387 | 0 | 18（4.7%） | 0.941 | 31/32、30/32 |
| + all-reduce 拷入 `site-141-arcopy-hc-1103T0627` | **0 / 599** | 0 | 12（2.0%） | 0.970 | 31/32、30/32 |

- 拷入组的 `probe_badstart.py` 报 2 条：1 条是 500 错误的空输出，1 条是以 "They used…" 开头的正常推理（正则 `The[^ \n's]` 把 "They" 也算进去），都不是开头坏。
- 方向与"graph 错配"一致：短 prompt 生成超过 2048 token 时，本 rank 从 dense 切到 sparse，与其他 rank 错开。但样本小：乱码只看 3 对 0，
  3 条恰好全落在对照组的概率约 6%；生成到上限的比例（4.7% 对 2.0%）也指向同一方向。**算"有利证据"，不算确认**；跨机门禁里再看。

### 下一步（跨机，未启动）

`scripts/q140_1103.sh`（138 上运行，先等 night35 结束；要用户同意再启动，因为 138 / 140 正被另一个会话使用）：
1. E10 + all-reduce 拷入（CAR 开）限池门禁 × 2（`config/site-140-arcopy-gate.env`）。
2. E10 + 固定 sparse（CAR 开、不拷入）限池门禁 × 2（`config/site-140-sparse-gate.env`）。
   判据（两项相同）：两次开头坏合计 ≤ 1 视为修好（开 CAR 的对照 8 次里 6 次 3–8 / 200，按 2.5% 算 400 题约 10 条）。
3. E11 + 固定 sparse 的限池门禁 × 1（`config/site-140-e11-sparse-gate.env`）。
4. C104 计分：E10 + 固定 sparse（预测 ≥ 51,900，来自去掉每步的同步 D2H）；E11 + 固定 sparse（预测 ≥ 52,400）。同一对上 E10 是 51,374。

`config/site-140-e11-copyin*.env` 留作备选：拷入不依赖 graph 选择方式，更稳；固定 sparse 更快（没有拷贝、还省掉同步）。

如果第 1 步不成立（拷入去不掉开头坏），发布配置就用关 CAR（−1.4%，见下）。E11 能否叠在关 CAR 上没有核实：
aiter 的 all-gather / reduce-scatter 可能依赖同一个 custom all-reduce 通信器，`--disable-custom-all-reduce` 后也许会退回 RCCL，要读代码确认。

### 关 CAR 的代价（另一会话 night35 的计分，138 → 140）

`../prefill-opt-20261027/results/run-llr37-140-e10-nocar-c104.txt`：E10 关 CAR 的 C104 = **50,670 tok/s/GPU**，开 CAR 的 E10 是 51,374，**−1.4%**（单次波动约 ±1%）。
decode 步长在最忙 rank 有 17 个请求时 46 ms，开 CAR 时 45 ms。关 CAR 的正确性证据是限池门禁 3 次 0 / 0，所以关 CAR 是代价约 1.4% 的现成正确配置。
