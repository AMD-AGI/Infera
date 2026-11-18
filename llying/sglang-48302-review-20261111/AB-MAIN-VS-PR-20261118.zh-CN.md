# sglang#48302 按 review 修改后，main 与 PR 的端到端 A/B（11-18）

- PR：[sgl-project/sglang#48302](https://github.com/sgl-project/sglang/pull/48302)；review 评估见同目录 `REVIEWER-JACOB0226-20261118.zh-CN.md`
- **结论**：最新 main 在 GLM-5.2 开 MTP + DP attention 时稳定出乱码，PR 修掉了，同时吞吐略有提升，正确性和性能都与提 PR 的初衷一致。
  - 正确性：两种配置各跑 3 轮，main 的 GSM8K 正确率是 0.854 和 0.900，共有 271 条乱码；PR 两种配置都是 0.978，乱码和开头坏都是 0。
  - 性能：同机、同一个压测客户端下，8 组对比里 PR 的输出吞吐有 7 组更高，平均 +1.9%；平均每 token 时间（TPOT）8 组全部更低，一般低约 1.8%。
- **状态**：代码改动只在本地提交（`959f88ab`），**还没有 push，PR 描述和给 reviewer 的回复也都没发**，等你确认。

## 一、代码改动

| 提交 | 内容 |
|---|---|
| `0407b339` | 合并最新 main（`b1a02a53`，11-18 03:14 UTC，比上次合并多 490 个提交），没有冲突。MLP sync 行仍是 14 列，PR 读的 `tp0_info_cpu[:, 13]` 依然正确 |
| `959f88ab` | 按 review 方案一：删掉 `Dsv41CandidateGraphVariants.select()` 开头读全局上界的 5 行和 `test_dsv41_candidates_follow_the_bound_in_decode_only`。DSV4.1 的行为恢复到和 main 一致；PR 相对 main 剩 9 个文件，+411 / −3 |

`ruff check`、`ruff format --check`、`isort --check-only` 都通过。

## 二、测试方法

- **镜像**：最新 nightly `lmsysorg/sglang-rocm:v0.5.23-rocm720-mi35x-20261117`（自带 sglang `703e763d`，比 main 落后 34 个提交，期间 sgl-kernel 和 ROCm 依赖都没变）。**镜像没有打任何补丁**，也没有 KF overlay 等私有改动。
- **换 sglang 的方式**：用 `git archive` 导出两份 `python/` 快照，通过 `PYTHONPATH` 指过去。A 组用 main `b1a02a53`，B 组用 PR `959f88ab`。两份快照只差 PR 改的 6 个 Python 文件。12 个服务的日志开头都记录了实际导入路径，没有一个用到镜像自带的 sglang。
- **机器**：作业 36659 的 n04-33 和 n06-25（MI355X）。
- **服务**：GLM-5.2-MXFP4 单机聚合，TP4 + DP4 attention，EAGLE 3/1/4，KV fp8_e4m3，DSA prefill/decode 都用 Triton，custom all-reduce 开（默认）。`index_share_for_mtp_iteration` 用 checkpoint 默认值（true）。完整参数见 `ab-20261118/server_args.txt`。
- **两种配置**：
  - **P**：镜像默认（`SGLANG_USE_ROCM700A=1`），decode 图里的 DP gather/combine 走图内的 custom all-reduce。这是用户开箱即用的配置。
  - **E**：E11（`SGLANG_USE_ROCM700A=0`），DP gather/combine 走图内的 aiter all-gather / reduce-scatter。
- **布局**：每台机器上 main 和 PR 并排各占 4 张卡，slot0 是 GPU 0–3 加 NUMA0 的 CPU，slot1 是 GPU 4–7 加 NUMA1 的 CPU，用 `taskset` 绑核。两组都设了 `SGLANG_SET_CPU_AFFINITY=0`，否则两个服务的 rank i 会被绑到同一批核上。P 固定在 n04-33，E 固定在 n06-25。第 1、3 轮 main 在 slot0，第 2 轮对调，用来抵消卡位差异。
- **每轮流程**（两个 slot 同步推进）：
  1. **正确性**：GSM8K 400 题（64 并发，temperature 0，max_tokens 16384）和 16k 长上下文 64 条（32 并发）同时跑，让各 DP rank 的序列落在 `index_topk`（2048）两侧。乱码和开头坏用的是 11-11 那套探测脚本的判据。
  2. **吞吐**：`python3 -m sglang.benchmark.serving`，用镜像自带的 sglang 做客户端，两组完全相同。`random` 数据集从本地 ShareGPT 取真实文本，range ratio 0.8，测前先 `flush_cache`：
     - p1k：输入 1024 / 输出 1024，768 条，64 并发；
     - p8k：输入 8192 / 输出 1024，192 条，32 并发。
- **第 1 轮的吞吐没测成**：脚本里写的是 `sglang.bench_serving`，镜像里已经没有这个模块了。改成 `sglang.benchmark.serving` 后加了第 3 轮补上（布局同第 1 轮），所以吞吐数据来自第 2、3 轮。

## 三、单测与机制核对

- 单测：在同一镜像里用 PR 快照跑，全部通过。`test_graph_variants_dp_consistency.py` 12 个，`test_tbo_filter_batch_marker.py` 11 个，`test_schedule_batch_out_of_place.py` 6 个，`test_dp_spec_prefill_coordination.py` 12 个，`test_aiter_megamoe_rank_sync.py` 1 个（`ab-20261118/unit.txt`）。
- 启动日志里的 `DSA dual-graph enabled`：main 每个服务 4 行，即每个 rank 的 target-verify 都抓了 dense/sparse 两套图；PR 每个服务 0 行，即 verify 只抓一套。和 PR 的设计一致。
- 12 个服务都在 210 秒内就绪，没有 Traceback，没有 NaN 或 HIP 报错。

## 四、正确性

GSM8K 400 题，每格是"正确数 / 开头坏 / 乱码 / 生成到上限"：

| 配置 | 轮次 | main | PR |
|---|---|---|---|
| P | 1 | 335（0.838）/ 4 / 58 / 59 | 389（0.973）/ 0 / 0 / 5 |
| P | 2 | 350（0.875）/ 1 / 40 / 48 | 391（0.978）/ 0 / 0 / 3 |
| P | 3 | 340（0.850）/ 1 / 55 / 52 | 394（0.985）/ 0 / 0 / 2 |
| E | 1 | 359（0.897）/ 2 / 41 / 34 | 391（0.978）/ 0 / 0 / 5 |
| E | 2 | 371（0.927）/ 3 / 36 / 21 | 392（0.980）/ 0 / 0 / 2 |
| E | 3 | 350（0.875）/ 4 / 41 / 47 | 391（0.978）/ 0 / 0 / 4 |

三轮合计：

| 配置 | 组 | GSM8K 1,200 题 | 开头坏 | 乱码 | 长上下文 needle（96） | 长前缀 GSM8K（96） / 其中乱码 |
|---|---|---|---|---|---|---|
| P | main | 1,025（0.854） | 6 | 153 | 96 | 86 / 7 |
| P | PR | 1,174（0.978） | 0 | 0 | 96 | 94 / 0 |
| E | main | 1,080（0.900） | 9 | 118 | 96 | 80 / 11 |
| E | PR | 1,174（0.978） | 0 | 0 | 96 | 92 / 0 |

- main 的乱码样本和以前的形状一样：推理开头正常，然后突然变成一长串 `!!!!!!`，或者出现思考标签之类的格式错误。生成到上限的条数（102–159）也比 PR（10–11）多一个数量级，这些就是退化后停不下来的输出。
- PR 的 0.978 和 11-11 在 main 上的审计（0.980 / 0.983），以及以前"原版 sglang + 官方 FP8 权重"的水平一致。
- **为什么 11-11 在 main 上没复现**：那次服务参数里有 `--disable-custom-all-reduce`。P 配置说明，只要用默认的 custom all-reduce，问题在 main 上就能稳定复现，不需要 PD、限池或 E11。

## 五、性能

输出吞吐（tok/s，单服务），括号里是 PR 相对 main：

| 配置 | 负载 | 第 2 轮（PR 在 slot0） | 第 3 轮（main 在 slot0） |
|---|---|---|---|
| P | p1k | main 3,149.5，PR 3,240.7（+2.90%） | main 3,185.3，PR 3,222.3（+1.16%） |
| P | p8k | main 947.7，PR 962.6（+1.57%） | main 964.6，PR 963.2（−0.15%） |
| E | p1k | main 3,257.3，PR 3,333.1（+2.33%） | main 3,188.0，PR 3,273.3（+2.68%） |
| E | p8k | main 957.4，PR 976.1（+1.95%） | main 966.0，PR 994.9（+2.99%） |

- 平均 TPOT 8 组全部是 PR 更低：7 组低 1.8%–3.1%，P p8k 第 3 轮持平（29.93 对 29.90 ms）。
- 两组的接受长度相近（p1k 3.74–3.86，p8k 3.80–4.00），没有系统性偏向哪一边。
- 方向和幅度都与 PR 描述里 PD 计分的 +1.3% 一致，来源也一致：verify 只抓一套图，replay 前不再做 `seq_lens.max().item()` 这次 GPU→CPU 同步。每组只测了一次，单次误差大约 ±1%，所以只能说"稳定略好"，不宜精确到某个百分比。
- PR 描述里 E11 那 +5.5% 是 PD 部署下 E11 相对 E10 的提升，这次单机测试没有重复那组对比。本次能确认的是：E 配置在 main 上不能用（0.900、118 条乱码），在 PR 上正确。

## 六、待你确认的公开操作

1. **push**：`git push fork fix/dsa-graph-variant-dp-consistency`（本地仓库 `/perf_apps/liyingli/bench_agentx/upstream/sglang-main`），把 `0407b339`、`959f88ab` 推到 PR。
2. **改 PR 描述**：
   - 删掉 NVIDIA 那一条；
   - "Covered"一条改为 DSV4.1 留给后续 PR；
   - 单测数量从 13 改成 12；
   - 用本文第四、五节替换 "On current `main`" 那段，去掉 "I did not rerun the failure reproduction on main"。
3. **回复 reviewer**：用 `REVIEWER-JACOB0226-20261118.zh-CN.md` 第四节的草稿，补一句"已合并最新 main 并在 nightly 镜像上 A/B"。

## 七、文件

| 路径 | 内容 |
|---|---|
| `ab-20261118/ab_node.sh` | 单机 A/B：两个服务并排，依次跑正确性和吞吐两个阶段 |
| `ab-20261118/driver.sh`、`driver_r3.sh` | 第 1–2 轮、第 3 轮的总控 |
| `ab-20261118/unit.sh`、`unit.txt` | 单测脚本与结果 |
| `ab-20261118/server_args.txt`、`check_args.py` | 服务参数，以及在两份快照上解析参数的检查脚本 |
| `ab-20261118/summary.txt` | 三轮的逐行结果 |
| `/perf_apps/liyingli/sgl48302-ab-20261118/runs/` | 每个服务的日志、探测原始输出（`*.gsm8k.jsonl`、`*.longctx.jsonl`）、乱码样本（`*.review.txt`）、压测结果（`*.p1k.jsonl`、`*.p8k.jsonl`） |
