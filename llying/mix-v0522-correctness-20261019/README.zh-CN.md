# GLM-5.2 aggregated TP4 + DP4 attention：并发下输出乱码（10-19）

## 现象

在升级 sglang v0.5.22 的 example（`examples/sglang_mix_glm5.2`，分支 `feat/sglang-mix-glm5.2-tp4-v0522`）做端到端验证时，
GSM8K（lm_eval 0.4.13，5-shot、200 题，经 router，真实 MTP 接受）只有 0.69 / 0.74，远低于 v0.5.19 PD 时的 0.97。
答错的题里很多是乱码：上万字符、夹 `</think>`、反复出现其他题目的数字片段，像读到了别的请求的 KV。

| 镜像 / 设置 | 并发 | 题数 | strict / flexible | 乱码题（>8000 字符或含 `</think>`） |
|---|---|---|---|---|
| v0.5.22（`infera-sglang:v0522-llying-mix-1019b`），example 配方，GPU 0-3 | 64 | 200 | 0.74 / 0.69 | 63 |
| v0.5.22，同上 | 1 | 24 | 0.958 / 0.958 | 0 |
| v0.5.21（`infera-sglang:v0521-llying-1012-mc4340-destpin`），同配方但 HiCache 64 GB、直连引擎，GPU 4-7 | 64 | 200 | 0.745 / 0.690 | 59 |

结论：**不是 v0.5.22 引入的回归**，v0.5.21 上同样存在；只在并发（批处理）时出现。smoke（顺序 6 个请求）和此前的
负载下 decode 检查（`decode_check.py`，答案只有 4-5 个 token）都没有暴露它。性能测试用模拟接受长度、输出长度由 trace 决定，
吞吐数据不受影响，但这个配方直接上线会输出错误内容。

## 二分

`scripts/engine_bisect.sh NAME [FUSE_TOPK=0] [HICACHE=0] [MTP=0] [DPA=0] [GATHERV=0] [DELAYER=0] [FUSION=0]`：在 138 上起
etcd + 引擎（example 配方去掉对应特性，HiCache 64 GB），GSM8K 64 题、并发 64、直连引擎，数乱码题数。结果在
`/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/<NAME>/runs/<NAME>/verdict.txt`（138 本地盘）。

| 变体 | 关掉的特性 | strict | 乱码题 / 64 |
|---|---|---|---|
| base | 无（example 配方，HiCache 64 GB） | 0.859 | 8 |
| fusetopk0 | DSA fused top-k（`SGLANG_DSA_FUSE_TOPK=0`） | 0.812 | 6 |
| hicache0 | HiCache | 0.891 | **0** |
| mtp0 | EAGLE MTP | 0.891 | 2 |
| wb | HiCache 改 `write_back`（其余同基线） | 0.844 | 7 |
| directio | HiCache I/O 改 `direct` + `page_first_direct` | 0.906 | 2 |
| b300like | B300 的组合：`write_back` + `direct` + `page_first_direct`，不覆盖 `index_share` | 0.859 | 1 |
| idxshare1 | 基线去掉 `index_share_for_mtp_iteration=false` 覆盖 | 0.859 | 3 |
| gtt | 基线 + `HSA_USERPTR_FOR_PAGED_MEM=0`（HiCache 主机池用 hipHostMalloc / GTT） | 0.906 | 1 |
| dpa0 | DP attention（纯 TP4，HiCache 开） | 0.953 | **0** |
| gatherv0 | `SGLANG_DP_USE_GATHERV=0` | 0.906 | 3 |
| delayer0 | prefill delayer | 0.859 | 3 |
| hicache0-n200 | HiCache，200 题复核 | 进行中 | |
| stock | 上游镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`，`sglang.launch_server`，不打 Infera 任何补丁，同配方 | 进行中 | |

剩下的"乱码"逐条看过，都是真乱码（`</think>` 碎片、成串的 `0:`、`2`），不是检测规则误判的长答案。

**基线要用同一套测法。** 最早写的"基线约 19 / 64"是从 200 题、经 router 的 E2E 结果换算的；同脚本（64 题同时发、直连引擎）
的基线实测是 8 / 64。按这个基线，1–3 题的那些变体（gatherv0、delayer0、idxshare1、directio、mtp0、gtt、b300like）只能说
"可能减轻"，64 题的样本分不出来；**站得住的只有两条：关 HiCache 为 0，关 DP attention 为 0**，即需要两者同时开。

**不需要真的发生主机传输。** wb 一项（`write_back`，只在驱逐时写主机）引擎日志里每个 rank 的 GPU KV 池 238 万 token，
64 题远装不满，没有驱逐也没有载回（prefill 的 `#cached-token` 全为 0），仍有 7 题乱码。两次启动日志（hicache0 对 wb）
除 HiCache 主机池的几行外完全相同，`server_args` 也只差 HiCache 自身的参数。所以问题在"挂上 HiCache 之后"走的不同代码
路径上，而不在数据拷贝本身；`HSA_USERPTR_FOR_PAGED_MEM=0`（gtt）也没有消除它。

读过但排除的点：prefill delayer 的否决发生在 `init_load_back` 之前（`schedule_policy.py` 1401 对 1459 行）；没有待载回数据时
`start_loading()` 返回 -1，前向不会等任何事件；HiCache 的跨 rank 同步用的是各自的 attention TP 组（DP attention 下大小为 1）。
上游相近的报告：sglang#30321（GLM-5.2 + MTP + HiCache 偶发乱码，关 HiCache 消失；修复 sglang#31427，维护者称 main 换
`UnifiedRadixTree` 后不再出现——v0.5.22 已是 `UnifiedRadixCache`，且对方是纯 TP4），sglang#22811（GLM-5 HiCache 下 indexer
读与层级载回竞争，已合入）。

下一步：stock 一项区分上游与我们自己的补丁（DSA 补丁集、`patch_hicache_rocm_staged_write_back.py` 等两个镜像都带）。

fusetopk0 关掉 fused top-k 后，decode 改走 `transform_index_page_table_decode`，也就是 `dsa_page_table_rows.diff` 生效的路径；
该补丁在 token 行数不是请求数整数倍时（DP 填充）会直接裁剪或补齐，真实 token 与请求的对应可能错位，剩下的 6 题可能来自这里。

**同节点两个引擎会撞 RCCL 共享内存名。** 两个容器都 `--ipc host`，容器内 PID 都从很小的数开始，后起的那个在
`ncclCommInitRank` 报 `HIP failure: 'invalid argument'` 退出（hicache0 前两次即此）。并行跑两个排查栈时，第二个先空转把 PID
推高（`PID_OFFSET=30000`，与 mixed 套件的 `PID_OFFSET_mixed_0` 同理）。
