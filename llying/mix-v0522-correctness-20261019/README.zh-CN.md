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
| fusetopk0 | DSA fused top-k（`SGLANG_DSA_FUSE_TOPK=0`） | 0.812 | 6（基线约 30%，即约 19） |
| hicache0 | HiCache | 进行中 | |
| mtp0 | EAGLE MTP | 进行中 | |

fusetopk0 关掉 fused top-k 后，decode 改走 `transform_index_page_table_decode`，也就是 `dsa_page_table_rows.diff` 生效的路径；
该补丁在 token 行数不是请求数整数倍时（DP 填充）会直接裁剪或补齐，真实 token 与请求的对应可能错位，剩下的 6 题可能来自这里。

**同节点两个引擎会撞 RCCL 共享内存名。** 两个容器都 `--ipc host`，容器内 PID 都从很小的数开始，后起的那个在
`ncclCommInitRank` 报 `HIP failure: 'invalid argument'` 退出（hicache0 前两次即此）。并行跑两个排查栈时，第二个先空转把 PID
推高（`PID_OFFSET=30000`，与 mixed 套件的 `PID_OFFSET_mixed_0` 同理）。
