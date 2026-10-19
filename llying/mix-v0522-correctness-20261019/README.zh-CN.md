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
etcd + 引擎（example 配方去掉对应特性，HiCache 64 GB），64 道 GSM8K 同时发、直连引擎。结果在
`/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/bisect/<NAME>/runs/<NAME>/`（138 本地盘）。

### 更正：HiCache 不是必要条件

前几轮用 lm-eval 的回答数"乱码"（>8000 字符或含 `</think>`），据此得出"关 HiCache 为 0、需要 HiCache 和 DP attention
同时开"。**这个结论是错的**：漏数了空回答。模型一直不闭合 thinking、生成到 `max_gen_toks=16384` 时，glm45 reasoning
parser 把全部文本放进 `reasoning_content`，lm-eval 看到的 `content` 是空串。把空回答算上（"坏"= 长/含 `</think>` + 空）：

| 变体 | 关掉的特性 | strict | 长或含 `</think>` | 空 | 坏 / 64 |
|---|---|---|---|---|---|
| base | 无（example 配方，HiCache 64 GB） | 0.859 | 8 | 2 | 10 |
| hicache0 | HiCache | 0.891 | 0 | 7 | 7 |
| **dpa0** | **DP attention（纯 TP4，HiCache 开）** | **0.953** | 0 | 2 | **2** |
| mtp0 | EAGLE MTP | 0.891 | 2 | 4 | 6 |
| fusetopk0 | DSA fused top-k（`SGLANG_DSA_FUSE_TOPK=0`） | 0.812 | 6 | 6 | 12 |
| wb | HiCache 改 `write_back` | 0.844 | 7 | 3 | 10 |
| directio | HiCache I/O 改 `direct` + `page_first_direct` | 0.906 | 2 | 4 | 6 |
| b300like | `write_back` + `direct` + `page_first_direct`，不覆盖 `index_share` | 0.859 | 1 | 8 | 9 |
| idxshare1 | 去掉 `index_share_for_mtp_iteration=false` 覆盖 | 0.859 | 3 | 6 | 9 |
| gtt | `HSA_USERPTR_FOR_PAGED_MEM=0`（HiCache 主机池用 GTT） | 0.906 | 1 | 4 | 5 |
| gatherv0 | `SGLANG_DP_USE_GATHERV=0` | 0.906 | 3 | 5 | 8 |
| delayer0 | prefill delayer | 0.859 | 3 | 6 | 9 |
| **stock** | **上游镜像 `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261018`，`sglang.launch_server`，不打任何 Infera 补丁** | 0.891 | 0 | 7 | **7** |

空回答是不是乱码，看引擎日志：收尾阶段只剩 1 个请求、已超过 6000 token 的 decode 行里，hicache0 四个 DP rank 的 MTP
平均接受长度 1.03–1.36、一直跑到 17k token（上限），stock 的 DP2 也是 1.31；正常 decode 是 3.5–4.5。接受长度掉到 1
说明 target 输出的已经不是 draft 能猜的正常文本——就是乱码，只是写在 thinking 里。dpa0 收尾时 2 个请求合计 5.5k token、
接受长度 4.4，没有这种长尾（它的 2 个空回答另查）。

所以：
- **需要的是 DP attention，不是 HiCache。** 只有 dpa0 没有乱码长尾；关 HiCache、换任何 HiCache 组合都还在。
- **上游无补丁镜像同样出现**，不是我们的 DSA 补丁集或 HiCache 补丁引入的；v0.5.21 也有（见上表）。
- 之前读 HiCache 代码排除的几点（delayer 否决在 `init_load_back` 之前、无载回时 `start_loading()` 返回 -1、HiCache
  同步用 attention TP 组）仍然成立，只是不再相关。上游 sglang#30321（GLM-5.2 + MTP + HiCache 乱码）对方是纯 TP4，与这里不同。
- 上游 CUDA 上 GLM-5.2-NVFP4 TP4/DP4 + EAGLE 的 GSM8K 是 0.948（sglang#32196 的验证），这里同构型只有 0.86–0.89，
  更可能是 ROCm 上 DP attention 路径（aiter / triton DSA / MXFP4）特有的问题。

### 换判据：`scripts/gsm8k_probe.py`

lm-eval 只看 `content`，判不了这个问题。`gsm8k_probe.py` 用 base 那一组的同一批 64 个 prompt 直连引擎（temperature 0，
`max_tokens` 16384），每个请求记 `finish_reason`、生成 token 数、`reasoning_content` 与 `content`；**被上限截断
（`finish_reason=length`）即判为失控**，与 parser 无关。之后各项都用它（`verdict.txt` 一行汇总，`probe.jsonl` 逐条）。

**同节点两个引擎会撞 RCCL 共享内存名。** 两个容器都 `--ipc host`，容器内 PID 都从很小的数开始，后起的那个在
`ncclCommInitRank` 报 `HIP failure: 'invalid argument'` 退出（hicache0 前两次即此）。并行跑两个排查栈时，第二个先空转把 PID
推高（`PID_OFFSET=30000`，与 mixed 套件的 `PID_OFFSET_mixed_0` 同理）。
