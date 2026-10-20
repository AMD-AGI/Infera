# GLM-5.2 aggregated TP4 + DP4 attention：并发下输出乱码（10-19）

## 10-20 03:22 更新：`--disable-custom-all-reduce` 消除乱码

**v0.5.21 镜像（`infera-sglang:v0521-llying-1012-mc4340-destpin`）基线配置（DP attention、MTP、HiCache 64 GB、delayer、融合
算子全开）只加 `--disable-custom-all-reduce`：200 题 197 对（0.985）、0 乱码、1 条截断**（doc 119，连贯的过度推敲，dpa0 也在这题
上跑满）；另 2 条答错是正常算错。140 GPU 4-7，`bisect/v0521-dcar-200/`。同一镜像不加此参数（10-19，138，lm-eval）0.745、59 条乱码；
v0.5.22 各变体 53-61 条乱码。

这正是已验证配置与这次配置的关键差异：09-24 完成完整 GSM8K（0.968 / 0.976）的 PD P4D4 部署，decode 引擎带着
`--disable-custom-all-reduce`；yihou 09-18 已查明"decode 上的 custom all-reduce 破坏推测路径"并以此修复（`yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`，
症状同为 `1!...` 开头的退化文本、MTP 接受长度约 1.25）。a3 / example 的聚合配方没有带这个参数，我在二分中也没有先与已验证配置
对比、没有测过它（只关过 aiter allreduce 融合），下面"当前结论"一节的排查方向因此绕了远路。与 yihou 的结论有一处不同：关 MTP
后照样乱码，说明受影响的不只是推测路径。a6b 的 decode profile 里 aiter custom all-reduce（`cross_device_reduce_2stage`）占 GPU 时间 4.7%；
yihou 测过的代价是 C40 +0.7%、C56 −4.2%（PD decode）。

**QuickReduce 默认就是开的，而且是 INT8 有损量化。** sglang 在 HIP 上除 aiter custom all-reduce（`ca_comm`）外还初始化
QuickReduce（`qr_comm`，`distributed/device_communicators/quick_all_reduce.py`），由 `ROCM_QUICK_REDUCE_QUANTIZATION` 控制，
代码默认 `NONE`（关）；但上游基础镜像设置了 `ROCM_QUICK_REDUCE_QUANTIZATION=INT8`，v0.5.21、v0.5.22 两个 Infera 镜像都继承了它
（`docker image inspect` 与运行中引擎的环境均为 INT8）。`--disable-custom-all-reduce` 会把两者一起关掉。DP attention 下 MoE 之后要对
gather 后的大缓冲区做 all-reduce，正是 QuickReduce 的适用范围，所以单独关 QuickReduce（`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`、保留
aiter custom all-reduce）是更精确的检验，10-20 03:30 起在 140（v0.5.21）和 138（v0.5.22）上各跑一组。

| 变体（200 题、并发 64，基线配置上改动） | 镜像 / 节点 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| v0521-dcar-200：`--disable-custom-all-reduce` | v0.5.21 / 140 | 0.985 | 0 | 1（连贯） |
| **v0522-dcar-200：`--disable-custom-all-reduce`** | v0.5.22 / 138 | **0.975** | **0** | 3（45、2、119，连贯，与 dpa0 同题） |
| **v0522-qrnone-200：只关 QuickReduce**（`ROCM_QUICK_REDUCE_QUANTIZATION=NONE`，aiter custom all-reduce 保留） | v0.5.22 / 138 | 157 / 198 | **44** | 44（03:42 手动停，结论已清楚） |
| v0521-qrnone-200：同上 | v0.5.21 / 140 | 起不来：启动正赶上同节点 b0r 栈回收显存，引擎退出、容器与日志没留下；v0.5.22 已回答，未重跑 | | |
| **v0522-dcar-topkv2-200：`--disable-custom-all-reduce` + `SGLANG_OPT_USE_TOPK_V2=true`** | v0.5.22 / 138 | **0.980** | **0** | 2（119、93，连贯） |
| **v0522-dcar-idxshare1-200：`--disable-custom-all-reduce` + 不覆盖 `index_share_for_mtp_iteration`** | v0.5.22 / 138 | **0.975** | **0** | 3（100、119、93，连贯） |

- **元凶是 aiter custom all-reduce 本身，不是 INT8 QuickReduce**：只关 QuickReduce 仍有 44 条乱码。
- **top-k v2、`index_share` 默认值在 GSM8K 上都没有引入乱码**（在关 custom all-reduce 的前提下）。但 GSM8K 的 prompt 只有约 1,000
  token，DSA 只有上下文超过 top-k（2048）才走稀疏选择；三组里总上下文超过约 2,048 的请求只有 11-17 条（全部无乱码）。这两项改的
  正是稀疏路径，要算验证充分，还需一个长上下文、多请求并发的检查（每个请求都走稀疏路径），再进性能 A/B。
- 性能代价（10-20 03:46 起）：138 上 example（v0.5.22）加 `ENGINE_EXTRA_ARGS=--disable-custom-all-reduce` 跑 AgentX C48，对照同节点
  example 基线 32,809；140 上 agg 套件 b5（b0r + 该参数），对照 b0r 32,776。

140 上的 b5（v0.5.21，b0r + 该参数）04:40 停掉：要交付的是 v0.5.22 的 example，决策只需要 138 那组；担心波动应在同镜像同节点
重复，换镜像换节点不降噪。140 改做长上下文正确性检查：v0.5.22 镜像 04:40-04:53 从 138 `docker save | ssh docker load`（限速
100 MB/s），04:53-04:55 再拷 AITER JIT 与 `/root/.cache` 两个缓存卷（6.5 GB，同样限速）——这段时间与 138 上 example 性能测试的计分
窗口（约 04:10-05:10）重叠，138 那组结果里注明。

**长上下文检查**（`scripts/longctx_probe.py`，`engine_bisect.sh PROBE_KIND=longctx`）：128 个请求、并发 32，每个约 16k token、各自
独立的随机填充文本（不共享前缀，每个都走 DSA 稀疏选择），一半是大海捞针（4 条"记录 X 的访问码"埋在随机深度，问其中一条，答案须
原样出现），一半是填充文本后接 GSM8K 题。先跑不加修复的基线做灵敏度检查（能抓到已知乱码，后面的"干净"才有意义），再跑加修复、
加修复 + top-k v2、加修复 + `index_share` 默认值。

**性能代价（138，v0.5.22，example 配方 C48，计分窗口 04:07-05:07，同节点对照 10-19 的 example 基线）**，产物在
`results/example-c48-dcar/`：

| | example | example + `--disable-custom-all-reduce` | 变化 |
|---|---|---|---|
| tok/s/GPU | 32,809 | 33,098 | +0.9% |
| 完成请求 / 每秒 | 3,494 / 0.964 | 3,530 / 0.974 | +1.1% |
| interactivity p50 / p90 | 40.8 / 22.5 | 40.3 / 19.1 | p90 −14.7% |
| TTFT p50 / p90 (s) | 4.5 / 14.0 | 3.7 / 10.1 | |
| ITL p50 / p90 (ms) | 24.5 / 44.5 | 24.8 / 52.2 | p90 +17% |
| decode ms/token | 30.3 | 31.3 | +3.1% |

吞吐不变，慢的一端 interactivity p90 降约 15%：custom all-reduce 关掉后 decode 的集合通信走 RCCL，尾部 decode 步变慢。两个
保留：窗口内 04:40-04:55 在从 138 往 140 拷镜像与缓存卷（限速 100 MB/s 的 `docker save`）；p90 本身两次运行间有几个百分点的波动。
方向可信，幅度要重复一次才能定。

**长上下文检查的灵敏度**：不加修复的基线（lc-nodcar，140，v0.5.22）捞针 64/64、长前缀 GSM8K 63/64（1 条截断）、0 乱码——这个
测法（并发 32、回答多数很短）抓不到 custom all-reduce 的乱码。它仍能检验 top-k v2 / `index_share` 会不会弄坏稀疏选择本身
（基线 128 个请求都走稀疏路径、捞针全对，若某项设置破坏 top-k 选择会直接失败），但说明不了它们与乱码在高并发长生成下的交互，
那部分只有 GSM8K 200 题 / 并发 64 的结果。

**长上下文四组（v0.5.22，128 请求 / 并发 32，prompt 中位约 13.9k token）：**

| 变体 | 节点 | 大海捞针 | 长前缀 GSM8K | 模板标签乱码 | 备注 |
|---|---|---|---|---|---|
| lc-nodcar：基线，不加修复 | 140 | 64/64 | 63/64 | 0 | 1 条截断 |
| lc-dcar：`--disable-custom-all-reduce` | 140 | 64/64 | 62/64 | 0 | 答错 1 条为 `26.00` 对 `26`（判分过严，实为对），1 条 12 对 13 |
| lc-dcar-topkv2：+ `SGLANG_OPT_USE_TOPK_V2=true` | 140 | 64/64 | 63/64 | 0 | 答错为 12 对 13 |
| lc-dcar-idxshare1：+ 不覆盖 `index_share` | 138 | 64/64 | 61/64 | 0 | 12 对 13；"2/5 times more" 歧义题；1 条在 `26` / `26.00` 之间反复（"I'll use 26.00. Let's use 26.00." 循环到 8192） |

结论：top-k v2 与 `index_share` 默认值都没有破坏长上下文下的稀疏选择（捞针每组 64/64）。`index_share` 那组的一条重复循环是连贯文本、
不是乱码，同题在 lc-dcar 正常结束；只有一例，不足为据，但留意。

**更精确的修复：上游代码里已有，但默认关闭。** 关 custom all-reduce 让 interactivity p90 降约 15%，于是找更窄的开关。v0.5.22
`distributed/parallel_state.py` 里，DP attention 的 gather / scatter 在 HIP 上走 aiter custom all-gather（`SGLANG_USE_AITER_AG`，
默认开）与 aiter custom reduce-scatter（`SGLANG_DP_USE_REDUCE_SCATTER`，HIP 默认开，只用于 `MAX_LEN` 即 graph decode 步），
两者在 v0.5.19、v0.5.21 里也有。第 87-91 行：

> Under CUDA-graph capture, stage aiter custom all-gather / reduce-scatter inputs through the pre-registered IPC pool instead of
> registering graph-pool tensors: the registered path makes peers read stale data over IPC on replay (garbage rows / NaN under DP
> attention + TP MoE on MI355X). Set to 0 to restore it.
> `_AITER_CAPTURE_COPY_IN = os.environ.get("SGLANG_AITER_CAPTURE_COPY_IN", "0") == "1"`

描述的正是这里的症状（乱码行、NaN 式的 `!`、只在 DP attention + graph 下），但默认值是 `"0"`（注释却写"设为 0 恢复"，两者不一致），
即默认走的是有问题的注册路径。10-20 05:35 起三组（v0.5.22，GSM8K 200 题 / 并发 64，custom all-reduce 保留）：
`SGLANG_AITER_CAPTURE_COPY_IN=1`（138 GPU 0-3）、只关 custom reduce-scatter `SGLANG_DP_USE_REDUCE_SCATTER=0`（138 GPU 4-7）、
只关 custom all-gather `SGLANG_USE_AITER_AG=0`（140 GPU 0-3）。上一轮我提议的"设 `SGLANG_DP_USE_REDUCE_SCATTER=1`"没有先核对
默认值——HIP 上它本来就是 1，设了等于没改。

| 变体（v0.5.22，custom all-reduce 保留） | 节点 | 题数 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|---|
| v0522-copyin-200：`SGLANG_AITER_CAPTURE_COPY_IN=1` | 138 | 200 | 159 | 43 | 41 |
| v0522-rs0-200：`SGLANG_DP_USE_REDUCE_SCATTER=0` | 138 | 200 | 139 | 50 | 61 |
| v0522-ag0-200：`SGLANG_USE_AITER_AG=0` | 140 | 187（06:02 时） | 135 | 50 | 49 |

**三个更窄的开关单独都不行**，与基线（53-61 条）同一水平；中途 110 题、0 乱码的部分统计是长尾未完时的假象。
`SGLANG_DP_USE_REDUCE_SCATTER=1` 没有单独测：它在 HIP 上默认就是 1（`environ.py` `EnvBool(_default_hip)`，未设置时检测到 ROCm
即为 True），之前所有乱码的基线都是这个设置。目前只有 `--disable-custom-all-reduce`（ca_comm 整体不初始化）能消除乱码。

**top-k v2 / `index_share` 的性能（140，10-20 06:13 起，`scripts/example_perf_chain.sh`，依次跑完、每点之间等 GPU 0-3 显存回收）：**
三个 AgentX C48 点，都在修复（`--disable-custom-all-reduce`）之上、都用 example 套件（v0.5.22，`SIM_ACCEPT_LEN=3.61`）：
`base`（只加修复，140 的同节点基线——140 比 138 快约 3.7%，不能直接用 138 的数）；`topkv2`（example 的本地副本，只把 `worker.sh`
里的 `SGLANG_OPT_USE_TOPK_V2=false` 改成 `true`，提交的 example 没有改）；`idxshare`（`ENGINE_EXTRA_ARGS` 追加
`--json-model-override-args {}`，覆盖掉 `index_share_for_mtp_iteration=false`）。日志在 140 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/perf140/`。

**base（140）与 138 的同配置对照**（example + `--disable-custom-all-reduce`，同镜像，产物 `results/example-c48-dcar-140/`）：

| | 138（04:07-05:07） | 140（06:3x-07:3x） | 差 |
|---|---|---|---|
| tok/s/GPU | 33,098 | 31,931 | −3.5% |
| 每秒请求 | 0.974 | 0.944 | −3.1% |
| interactivity p50 / p90 | 40.3 / 19.1 | 38.8 / 19.3 | |
| TTFT p90 (s) | 10.1 | 12.4 | |

完全相同的配置，两节点（或两次运行）差 3.5%。之前写的"140 比 138 快约 3.7%"来自跨天、跨镜像的比较（b0r 对 a3），方向是错的。
这说明 ±3% 以内的差异都在噪声里（包括修复的吞吐代价 +0.9%）；top-k v2 / `index_share` 只与 140 上的 base 比。

**调整（08:07）**：base 本不该在 140 重测（138 已有 33,098），机器分配应是 138 测性能、140 做 CI。改为：140 只跑完 topkv2
（07:53 开始，对比 140 base 31,931），已手工结束串行脚本的外层循环、不再接着跑 idxshare；idxshare 改到 138（08:07 开始，
对比 138 的 33,098，日志 138 `$W/perf138/`）。引擎日志确认生效的是 `json_model_override_args='{}'`（即用模型默认的 index_share）。
topkv2 的预热在 480-510 s 只剩 1 个在途，与 base 同期形态一致（base 同期返回 41/531，topkv2 50/531），不是卡住。

**结果**（都在修复之上，C48，3,600 s，与同节点 base 比；引擎 inspect 确认 140 base `SGLANG_OPT_USE_TOPK_V2=false`、
topkv2 `=true`，138 idxshare 为 `false`、`json_model_override_args='{}'`）：

| | tok/s/GPU | 每秒请求 | 成功/错误 | intvty p50 / p90 | TTFT p50 / p90 (s) | ITL p50 / p90 (ms) | E2E p90 (s) |
|---|---|---|---|---|---|---|---|
| 140 base | 31,931 | 0.943 | 3,418 / 2 | 38.8 / 19.3 | 4.09 / 12.35 | 25.8 / 51.7 | 85.2 |
| **140 topkv2** | **34,662（+8.6%）** | 1.020（+8.2%） | 3,699 / 1 | 42.3 / 22.7（+9.0% / +17.6%） | 3.78 / 10.62（−8% / −14%） | 23.6 / 44.1（−9% / −15%） | 74.7（−12%） |
| 138 base | 33,098 | 0.975 | 3,530 / 2 | 40.3 / 19.1 | 3.70 / 10.09 | 24.8 / 52.2 | 84.6 |
| 138 idxshare | 32,675（−1.3%） | 0.972 | 3,524 / 1 | 40.2 / 20.5（0 / +7%） | 3.96 / 11.64（+7% / +15%） | 24.9 / 48.9（0 / −6%） | 85.8（+1%） |

- **top-k v2：吞吐和延迟全面变好**，幅度远超同配置两节点之间 3.5% 的差异；绝对值也比 138 base 高 4.7%。正确性此前已验证
  （v0.5.22 GSM8K 0.980、0 乱码；长上下文 needle 64/64）。值得放进 example；按用户决定不在 138 上复测。
- **`index_share`：没有明确收益**，吞吐 −1.3% 在噪声内，intvty p90 略好、TTFT p90 略差，方向不一致。维持 example 现有的
  `index_share_for_mtp_iteration=false`。
- 产物：`results/example-c48-dcar-topkv2-140/`、`results/example-c48-dcar-idxshare1/`。

待做：topkv2、idxshare 两点；同时关 custom reduce-scatter 与 all-gather（看是否等价于关整个 custom all-reduce）；修复代价复测；之后
example 与 agg 配方补上修复。

## 当前结论（10-19 23:10，后面各节是按时间记的过程，其中"需要 HiCache""混合步触发"两个中间假设已被推翻）

- **现象**：MI355X 上 GLM-5.2-MXFP4 聚合部署 TP4 + DP4 attention，持续并发（GSM8K 200 题、并发 64）下约 30% 的回答坏掉：
  开头是连贯的推理，decode 中途先零星出现单个错 token（零宽空格、`!`、无关汉字），随后越来越乱，最后成串的数字 / 模板标签 /
  重复片段，常常跑满 `max_tokens`。并发 1 时正常（0.958）。
- **必要条件：DP attention + decode CUDA graph**。关 DP attention（纯 TP4，graph 开）0.980、0 乱码；保留 DP attention 但
  `--disable-cuda-graph` 197 / 199 对、0 乱码（eager decode 每请求约 8 tok/s，不能当绕过办法）。
- **已排除**（单独关掉或替换后仍乱码）：HiCache（含 write_back、direct I/O、GTT 主机池等所有写法）、EAGLE MTP、prefill delayer、
  混合步（delayer 上限 2000，prefill 步几乎全是 4 rank 一起）、overlap 调度、GatherV、融合 qk-norm-rope / aiter allreduce 融合、
  `--enable-dp-lm-head`、DSA decode 后端（triton / tilelang）、DSA top-k（fused / 非 fused / `torch.topk`）、flydsl MoE 排序、
  上游的 DP 填充行清零开关。`breakable` graph 后端与 KV canary 都不支持 GLM-5.2 的 DSA，用不上。
- **上游问题，且由来已久**：上游无补丁镜像 v0.5.19、v0.5.20、v0.5.22 都复现；v0.5.19 的 3P1D PD 部署正常（0.97），那里 decode
  节点只跑 decode 步、从不跑 EXTEND。规律：乱码条数与 EXTEND（prefill）步数大致相当（6 步 / 9 条，40-51 步 / 53-61 条）。
- **影响**：example 配方不能直接上线（example README note 6 已写明）；AgentX 回放用 `ignore_eos`，输出长度由 trace 决定，
  性能数据不受影响。
- **复现**：`scripts/engine_bisect.sh NAME [开关...]`（在 GPU 节点上起 etcd + 引擎）+ `scripts/gsm8k_probe.py`（200 题直连重放，
  数模板标签外泄 `garbled` 与截断 `runaway`；干净配置 runaway 只有 2 条连贯的过度推敲）。
- **下一步**：在引擎里插桩，每隔 N 个 decode 步把同一批再用 eager 算一遍，与 graph 回放的 logits 比对，定位第一处分歧的层与行；
  向上游报告（文末草稿，待 review 后再提）。

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

## CI 验证（10-20，138，分支 `feat/sglang-mix-glm5.2-tp4-v0522`）

按 `.github/workflows/ci.yml` 在 138 上复现与 sglang 镜像相关的 job（`scripts/ci_local.sh`，从本地仓库克隆分支，不写工作目录；
日志在 138 `/mnt/m2m_nobackup/liyingli_3p1d/mix-v0522/ci/`）：

| CI job | 做法 | 结果 |
|---|---|---|
| `lint` | `python:3.10` 容器，`pre-commit run --all-files`，`CI=1`；cargo-fmt / gofmt 跳过（分支未改 Rust、Go，容器无工具链） | 首次 `ruff format` 失败（两个补丁脚本的路径拼接写法），按 ruff 结果改好（`fe6cfa01`）后**通过** |
| `unit` | 同一容器，`pip install -e ".[dev,gaie,projection]"`，`pytest -m "not slow and not integration"` | **通过**：2,570 passed，137 skipped，25 deselected |
| `tests/unit`（镜像内） | v0.5.22 镜像，`tests/lib/container_pytest.sh` | **通过**：2,359 passed，81 skipped（10-19 的 15 个 Responses 相关失败已修好） |
| `engine`（sglang 那一半） | 按 `Dockerfile.sglang` 构建（标签 `infera/engine-sglang:llying-ci-v0522`，不占用共享的 `test-local`），逐文件跑 `tests/engine/sglang` | **通过**：8 个文件全部通过 |
| `e2e-mixed`（sglang） | `INFERA_E2E_LOCAL=1 tests/run_tests.sh e2e sglang mixed`；非独占模式（不设 `INFERA_E2E_EXCLUSIVE`、`INFERA_E2E_RECLAIM_FOREIGN_CONTAINERS`，不动别人的容器）；节点上原有的 `test-local`（09-03 构建）先另存、跑完恢复 | 首次 **3 个用例失败、2 个跳过**：worker 报 `ambiguous option: --cuda-graph-max-bs could match --cuda-graph-max-bs-decode, --cuda-graph-max-bs-prefill`。修复（`4403eda6`）后重跑**通过**：gpt-oss-120b-tp2-ep、Kimi-K2.6-MXFP4-tp4-ep-dpattn、GLM-5.1-FP8-tp4 PASSED，DeepSeek-V4-Flash-FP8-tp4-dpattn、GLM-5.2-FP8-tp8-dpattn 按矩阵跳过（与首次相同），30 分钟；`test-local` 标签已恢复 |
| `e2e-disag`（sglang） | 原样要 SLURM（`srun`），138/140 没有；改 harness 支持 ssh（见下）后在 138+140 上跑，网络设置同 ci.yml | **通过**（10:39-10:50，11 分钟）：gpt-oss-120b-tp2 PASSED（prefill 138 / decode 140，counting、capital、long-context 三项正确性通过，两侧 KV 传输均为 `rdma`）；GLM-5.2-FP8-tp8-dpattn、DSV4-Flash-FP8-tp4-dpattn 按矩阵在 gfx950 上跳过（与 CI 相同） |

`rust`、`unit-torch-cpu`、`agent_sys`、`engine` 的 vllm 那一半：分支未改相关代码，未跑。

**e2e-disag 改为可在 138+140 上跑（10-20）**：独立分支 `feat/e2e-disag-ssh-transport`（`a99fd8eb`，基于 main，已推送，未开 PR）。
`INFERA_E2E_TRANSPORT=ssh` 时，所有节点操作走 `ssh <node>` 而不是 `srun`（唯一出口 `cluster.run_on_node`），节点由
`INFERA_E2E_NODES` 指定；`disagg.sh` 在此模式下不抢占/不换节点/不重试，以非独占方式跑一次（只删本次带标签的容器，不做 GPU
清理）。不设这个变量时 SLURM 路径不变。`tests/unit/e2e_harness/` 168 passed（含新增 3 个），ruff 0.13.0 通过。

前提（已只读确认）：两节点 mlx5_0 的 GID 3 都是由管理网 IP 映射出的可路由 RoCEv2 地址（138 `10.245.157.237`，140 `10.245.159.30`），
都没有 ib_peer_mem，与 ci.yml 的 e2e-disag 网络设置（GID 3、`MC_TE_FILTERS=mlx5_0`、dma-buf）一致；`/home` 两节点共享，
`/mnt/m2m_nobackup` 是各自本地盘（checkout 必须放 `/home`）；138↔140 免密 ssh 双向可用。gfx950 上 sglang 矩阵只跑 gpt-oss-120b：
prefill TP2 在 138 GPU 0-1，decode TP2 在 140 GPU 0-1，etcd/router 在 138。

步骤（`scripts/ci_local.sh`，在 138 上手动执行，`OUT=/home/liyingli/bench_agentx/baseline/ci-disag-v0522`）：
`disag-prep`（clone v0522 分支并 cherry-pick `a99fd8eb`；OUT 下建 venv 装 pytest/pytest-asyncio/httpx，不动系统 python；140 缺
v0.5.22 base 镜像，`docker save | ssh docker load`，64.7 GB）→ 两节点并行 `disag-build`（`Dockerfile.sglang` 没有声明
`MOONCAKE_HIP_DMABUF`，这个构建参数对 sglang 镜像不起作用，只给 tag 加后缀 `test-local-mooncake_hip_dmabuf1`；138 可命中缓存，
140 要从头编 Mooncake）→ `disag`。

进度（10-20）：10:32 `disag-prep` 开始，cherry-pick 干净地落在 `4403eda6` 上（`25867ca8`），venv 已建好；`--collect-only`
收集到 3 个用例（gpt-oss-120b-tp2；GLM-5.2-FP8 和 DSV4-Flash 在 gfx950 上会运行时跳过）。138 `disag-build` 完成（命中缓存）。
base 镜像正在传往 140。`scripts/disag_after_prep.sh` 在 138 后台按顺序衔接：准备阶段结束 → 140 `disag-build` → `disag`，
每步只在上一步报告 rc 0 时继续；日志在 138 `$W/ci/disag-{prep,build-138,build-140}.log`、`disag.log`、`disag-chain.out`。

结果：base 镜像传输约 4 分钟（10:36 完成）；140 构建 2.5 分钟（Mooncake 实际编译了，66 s）；`disag` 10:39-10:50
`RESULT: PASS`（1 passed、2 skipped）。容器全部清理，两节点显存归零。worker 日志在
`/home/liyingli/bench_agentx/baseline/ci-disag-v0522/logs/llying-v0522-disag/`。至此 v0.5.22 分支涉及 sglang 的 CI 作业
（lint、unit、unit-image、engine、e2e-mixed、e2e-disag）在本地全部通过。

**example 已更新（10-20，`00da3cf3`，按用户决定不另跑）**：`worker.sh` 加 `--disable-custom-all-reduce`、
`SGLANG_OPT_USE_TOPK_V2=true`；README 的验证状态、参考数据（140 topkv2 那组：34,662 tok/s/GPU、intvty p90 22.7，命中 96.5%）
与 note 6 改写。smoke 6/6 与并发 1 的 GSM8K 0.958 是加这两项之前的结果，README 中以 * 标出；并发 64 的 GSM8K 0.980 来自
bisect 基线配置（与 example 只差主机池 64 GB 对 270 GB），一并注明。旧 README 写的"0 errors"实际是 3 条
`InvalidInferenceResultError`，新数据如实写 1 条。example 仍不合入 main；`feat/e2e-disag-ssh-transport` 等用户 review。

**e2e 暴露的升级回归：`--cuda-graph-max-bs`。** v0.5.22 把它拆成 `--cuda-graph-max-bs-decode` / `--cuda-graph-max-bs-prefill`，旧写法
在 argparse（默认允许缩写）里成了两者的有歧义前缀，Infera 入口用 sglang 的 `ServerArgs.add_cli_args` 解析剩余参数时直接报错，
worker 退出码 2。e2e 的 `tests/e2e/pd_mixed/sglang/conftest.py` 传 `--cuda-graph-max-bs 8`；仓库里还有十来处 example / 部署配方
（`deepseek_v4` 三套 engine 脚本、`k8s-deployments/glm5.2-sglang.yaml`、`recipes/glm5.2/*`、`recipes/glm5.3/*`、`sglang_1p1d_glm5.2`）
用它，凡跑在 `Dockerfile.sglang` 镜像上都会起不来。按用户决定在 `infera/engine/sglang/args.py` 加兼容转换
`_rename_cuda_graph_max_bs`：当前 sglang 没有 `--cuda-graph-max-bs` 而有 `-decode` 时，把参数（含 `=` 写法）改写成
`--cuda-graph-max-bs-decode`（旧参数本就只管 decode graph），改写后的参数列表同时用于解析和转发给 `sglang.launch_server`；旧版
sglang 上什么都不变。新增 `tests/engine/sglang/test_cuda_graph_max_bs_args.py`（镜像内 3 passed；CPU 环境按设计跳过）。
提交 `4403eda6` 后 `lint`、`unit`、`engine` 全部重跑通过。

## example 性能对齐（10-19，138 GPU 0-3）

按 example README 手动起：`SIM_ACCEPT_LEN=3.61 bash engine/up.sh`，`CONC=48 bash engine/agentx/agentx.sh`，`bash engine/down.sh`。
计分窗口 20:21:48-21:21:48；其中 20:26-20:43 GPU 4-7 在跑 topk-torch-200 排查（之后暂停）。产物在 `results/example-c48/`，
指标用 `agg-tp4dpa-138-v0521-20261015/scripts/plot_agg_rounds.py` 的同一套 `measured()` 算。

| | a3 C48（v0.5.21 镜像，138，10-15） | example C48（v0.5.22 镜像，138，10-19） |
|---|---|---|
| tok/s/GPU | 31,605 | **32,809**（+3.8%） |
| 完成请求 / 每秒 | 3,409 / 0.947 | 3,494 / 0.964（0 错误） |
| interactivity p50 / p90 | 40.4 / 20.8 | 40.8 / **22.5** |
| TTFT p50 / p90 (s) | 4.8 / 14.5 | 4.5 / 14.0 |
| ITL p50 / p90 (ms) | 24.8 / 48.1 | 24.5 / 44.5 |
| decode ms/token | 30.6 | 30.3 |
| AIPerf 计入命中 / trace 上限 | 96.1% / 97.1% | 96.3% / 97.2% |

性能对齐（略好）。v0.5.22 的服务端命中计数口径变了：AgentX 实时日志的 `srv prefix_cache_hit` 显示 48.1%、引擎计数器拆分为
GPU 40.8% / 主机 7.4%，但 AIPerf 按 API `cached_tokens` 算的命中是 96.3%，与 a3 相同，吞吐也对得上，所以是计数口径而非命中下降。

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

lm-eval 只看 `content`，判不了这个问题。`gsm8k_probe.py` 把 lm-eval 样本里的 GSM8K prompt 原样直连引擎重放（temperature 0，
`max_tokens` 16384），每个请求记 `finish_reason`、生成 token 数、`reasoning_content` 与 `content`，给两个数：
- **garbled**：正文（thinking + 回答）里出现 `</think>`、`</arg_value>` 等模板标签的字面文本。真的特殊 token 会被 parser
  剥掉，只有解码坏了才会把它们当普通文本吐出来；hicache0-n200 的 60 条长回答全部命中，dpa0 一条不中。
- **runaway**：被 `max_tokens` 截断。干净的模型也会在个别有歧义的题上反复推敲到上限（dpa0 的 2 条即此，逐字看过是连贯推理），
  所以只作参考。

**200 题、并发 64 比 64 题同时发灵敏得多。** 同样关 HiCache，64 题同时发只有 7 条空回答，200 题持续发有 60 条乱码。
64 题同时发时 prefill 全挤在开头，之后是纯 decode；持续发时新请求不断进来，总有"一个 DP rank 在 prefill、其余在 decode"
的混合步——这一步全局不能走 CUDA graph，decode 的 rank 走 eager，DP gather 用 `SUM_LEN` 布局（`dp_attention.py`
`get_dp_padding_mode`），纯 decode 步则是 `MAX_LEN` + CUDA graph。之后都用 200 题、并发 64（prompt 取 hicache0-n200 的样本）。

| 变体 | 设置 | 题数 | correct | garbled | runaway |
|---|---|---|---|---|---|
| base-p | 基线，64 题同时发 | 64 | 0.859 | 9 | 8 |
| **dpa0-200** | 关 DP attention（纯 TP4，HiCache + MTP 开） | 200 | **0.980** | **0** | 2（连贯的过度推敲） |
| cg0-200 | 基线 + `--disable-cuda-graph` | 200 | — | — | 引擎第一个 decode 步即崩溃 |
| **mtp0-200** | **关 MTP** | 200 | 0.685 | **61** | 59 |
| fusion0-200 | 关 `--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion` | 200 | 0.710 | 56 | 59 |
| mtp0cg0-200 | 关 MTP + `--disable-cuda-graph`（全部 eager，避开上面的 draft 崩溃） | 200 | 手动停止，无汇总 | | 见下 |
| lmhead0-200 | 关 `--enable-dp-lm-head` | 200 | 0.675 | 55 | 60 |
| dec-tilelang-200 | DSA decode 后端 `triton` → `tilelang` | 200 | 0.670 | 55 | 1 |
| topk-torch-200 | `--dsa-topk-backend torch --speculative-dsa-topk-backend torch` + `SGLANG_DSA_FUSE_TOPK=0`（DSA top-k 改用 `torch.topk`；torch 后端不支持 fused top-k，不关会在 graph 捕获时报错） | 152 / 200 | 135 / 152 | 12 | 8 |

**MTP 也不是必要条件。** 关 MTP 后同样 61 / 200 乱码。至此 HiCache、MTP、prefill delayer、GatherV、融合算子、
DP lm-head、HiCache 各种写法都排除，**唯一必要的是 DP attention**，且要在持续并发下（有混合步）。

**关 CUDA graph 后长尾少了一个数量级。** mtp0cg0-200 全部 eager，decode 每请求只有约 8 tok/s，跑满 16384 token 要半小时以上，
为了把 GPU 0-3 让给 example 的性能对齐，healthy 后 14 分钟手动停掉（`verdict.txt` 有记录）。停时只剩 6 个请求在跑
（各 rank 合计 3.6k-12.8k token）；对照开 graph 的 mtp0-200，healthy 后 4 分钟每个 rank 还有 13-16 个请求、平均各 1 万
token 以上。这 6 个是不是乱码没来得及看（之后用逐条落盘的新版 `gsm8k_probe.py` 补测）。

v0.5.19 的 3P1D PD 部署 GSM8K 是 0.97，那里 decode 节点同样是 DP attention + CUDA graph，但 decode 节点从不做 prefill，
没有混合步。合起来的工作假设：**聚合部署里 eager 的混合步（DP gather `SUM_LEN`）与 CUDA graph 的纯 decode 步
（`MAX_LEN`）交替时，有状态没有对齐**，graph 回放读到了错位的数据。尚未定位到具体代码。

**cg0-200 的崩溃。** 全部走 eager 后，第一个 decode 步在 MTP draft 的 `_draft_extend_for_decode` → NextN 层 DSA indexer →
`fast_topk_transform_fused` 报 `RuntimeError: lengths must have one entry per score row`，随后 GPU memory access fault。
即 DP attention 下 eager 的 draft-extend 路径，打分矩阵的行数与 `lengths` 条目数不一致（纯 decode 步 eager 时 DP gather 是
`MAX_LEN`，行数按全局最大 token 数填充）。平时纯 decode 走 CUDA graph 绕开了这里；混合步也走 eager，但 `SUM_LEN` 不填充，
行数凑巧对得上。draft 算错只会拉低接受率、不会让输出乱码（target 逐个验证），所以它不是乱码的直接原因，但说明
**DP attention 下 eager 路径的 DSA 元数据与 DP 填充对不齐**；target 的 verify / decode 若在同类路径上行错位而不报错，
就会按错的 top-k 去读 KV，正是乱码的样子。

**同节点两个引擎会撞 RCCL 共享内存名。** 两个容器都 `--ipc host`，容器内 PID 都从很小的数开始，后起的那个在
`ncclCommInitRank` 报 `HIP failure: 'invalid argument'` 退出（hicache0 前两次即此）。并行跑两个排查栈时，第二个先空转把 PID
推高（`PID_OFFSET=30000`，与 mixed 套件的 `PID_OFFSET_mixed_0` 同理）。

**换 DSA decode 后端（tilelang）同样乱码，但形态变了。** 55 条乱码里只有 1 条跑到上限，其余都以 `stop` 结束（500-4.8k token），
内容是成串的 `!!!!`（0 号 token）夹模板标签。`!` 连串是 logits 全 NaN / 全 0 时 argmax 的典型结果；triton 后端下则是数字和
标签碎片。两个后端读到的像是同一批无效数据，只是容错不同——它们共用的上游是 DSA indexer 给出的 top-k 索引，所以下一项把
top-k 换成 `torch.topk`（`--dsa-topk-backend torch`）。

**top-k 换成 `torch.topk` 也乱码。** topk-torch-200 跑到 152 题时已有 12 题乱码（另 48 题未完，多为长尾），为了不和 example 的性能
对齐（GPU 0-3，计分窗口 20:22-21:22）重叠太久，20:43 手动停掉（`verdict.txt` 记部分结果；20:26-20:43 与该窗口重叠，性能结果
里注明）。top-k kernel、DSA decode 后端都排除后，剩下的是 indexer 打分及更上游、DP gather/scatter 与 MoE、以及 overlap 调度。
example 计分窗口内 GPU 4-7 不再跑排查。之后：`--disable-overlap-schedule`；上游 v0.5.20 / v0.5.19 镜像的聚合 DP attention
（找出从哪个版本开始出现）。

**乱码是 decode 中途逐步出现的，不是 prefill 写坏。** mtp0-200 的 61 条乱码里，第一个模板标签出现在回答的第 721 / 2,049 / 8,033
个字符（p10 / p50 / p90），开头 200 字符内出现的只有 1 条。逐条看，开头都是连贯、正确的长推理，然后先零星冒出单个错 token——
`50,000 \u200b\u200b000`（零宽空格）、`17!02`、`7辩论`、`I's`——之后越来越密，最后崩成成串的数字、`!` 和模板标签。也就是说，
某些 decode 步里某一行的 logits 突然是错的，一个错 token 让模型偏离，错误再累积。`!` 是 0 号 token，通常意味着那一行那一步的
hidden state 是全 0 / NaN，而 DP 填充行正是用 0 填的：像是某些步里真实请求的行在 gather / scatter 或 graph 输入输出的切片上
错了位，读到了填充行。

相关的上游记录：sglang#20404（MI355X，PD + DP attention，DeepSeek-R1 输出退化成重复/空串，关 DP attention 94%；2026-03，
v0.5.9，仍 open）；sglang#37192 第 3 点（DP 下 batch 先补到全局 token 数，graph runner 再补到捕获桶，第二次补齐不进 DP 缓冲区
长度——说的是 prefill graph，但与这里怀疑的"补齐与切片错位"同类）。

**大块 HiCache 栈撤下后，同节点新引擎会卡住。** example 栈（4 × 270 GB 锁页主机池）21:27 `down.sh` 后，21:28 在 GPU 4-7 起的
stock0520 引擎进程停在 `D` 状态、没有任何日志，直到 21:42 GPU 0-3 的显存读数归零才继续——驱动回收前一个栈期间，新进程的 KFD
初始化被挡住。64 GB 主机池的排查栈之间没有这个现象。

### 上游旧版本：v0.5.19、v0.5.20 也有

为了找出从哪个版本开始，用上游镜像（`STOCK=1`，不打任何补丁）跑同一配方，关 MTP、关 HiCache（避开旧版缺 DSA 补丁时的崩溃；
v0.5.22 上这两项单独关掉都还乱码）：

| 变体 | 镜像 | correct | 模板标签乱码 | 截断（runaway） |
|---|---|---|---|---|
| stock0520 | `lmsysorg/sglang-rocm:v0.5.20-rocm720-mi35x-20260924` | 0.830 | 0 | 21 |
| stock0519 | `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917` | 0.870 | 0 | 26 |
| 对照 dpa0-200 | v0.5.22（我们的镜像），关 DP attention | 0.980 | 0 | 2（连贯） |

两个旧版本没有模板标签，但截断的回答逐条看都是坏的：v0.5.20 是 `tsetsetset…`、`0.0.0.0…`、`1111…`、`magic magic magic…`、
`field field field…`；v0.5.19 是 `0.8 * 60 = 48.` 无限重复、`,0,0,0…`、`1008 years8 years8…`，还有一条模型自己写
"I will not include the weird glitching text in the final output"。所以**这不是 v0.5.19 之后的回退**，ROCm 上聚合 DP
attention 一直有；版本之间只是表现不同（v0.5.21 起多了模板标签外泄）。判据上，`garbled`（标签）只抓到新版本的形态，
**`runaway` 在干净配置下只有 2 条连贯的过度推敲，作为第二指标更通用**（乱码时 20-60 条）。

注意部分统计会偏乐观：长尾总是最后结束，stock0519 跑到 174 题时还是 0 / 0，剩下的 26 题全是截断。

v0.5.19 的 3P1D PD 部署 GSM8K 0.97，那里 decode 节点同样是 DP attention + CUDA graph，区别是 decode 节点从不做 prefill。
下一步直接检验"混合步"：`--prefill-delayer-max-delay-passes 2000`（混合时最多等 2000 步，持续负载下各 rank 基本都等到
"全部可 prefill"再一起做，GSM8K 的 prompt 不分块，混合步基本消失）；同时测 `--disable-overlap-schedule`。

### 混合步不是触发条件

| 变体 | 设置 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| dly2000-200 | 基线 + `--prefill-delayer-max-delay-passes 2000` | 0.705 | 61 | 60 |

把引擎日志的 `Prefill batch` 行按 30 ms 间隔分组（同一前向步内各 rank 的日志相差几毫秒），看每个 prefill 步有几个 rank 在做：
mtp0-200 是 51 步里 17 步只有 1 个 rank（其余 rank 在 decode 或空转），dly2000-200 是 40 步里 36 步 4 个 rank 一起、只有 4 步不齐。
**混合步基本没了，乱码一条没少**，前面"eager 混合步与 graph 纯 decode 步交替"的工作假设不成立。

`--disable-overlap-schedule` 与 prefill delayer 不能同时开（`AssertionError: To use PrefillDelayer, disable_overlap_schedule must
be False`），ovl0-200 改为同时关 delayer 重跑（delayer0 已证明 delayer 本身不是原因）。

目前站得住的：只要 DP attention；关 CUDA graph 后长尾少一个数量级；持续到达（200 题 / 并发 64）比一次发完（64 题）严重得多。
上游在 v0.5.22 的 `models/deepseek_common/moe_transport.py` 里已经写明 "Pad rows are not refreshed during graph replay. Stale
bf16-max values can overflow the MoE and contaminate the combine output"，并在 HIP + DP attention + TP MoE 时把 DP gather 前的
填充行清零（`SGLANG_DP_MASK_PAD_ROWS`，默认开；`decode_cuda_graph_runner._set_dp_real_num_tokens` 每次回放写真实行数）。
清零后的填充行全部路由到同一组专家，MoE 负载极度倾斜——下一项把 `AITER_USE_FLYDSL_MOE_SORTING` 关掉（docker 对重复的
`-e` 取最后一个，容器内已确认为 0）。

| 变体 | 设置 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| flydsl0-200 | `AITER_USE_FLYDSL_MOE_SORTING=0` | 0.700 | 55 | 60 |
| ovl0-200 | `--disable-overlap-schedule` + 关 delayer | 200 题 | 53 | 53 |
| **mtp0cg0b-200** | **关 MTP + `--disable-cuda-graph`（重跑，逐条落盘）** | **197 / 199** | **0** | **0** |
| maskpad0-200 | `SGLANG_DP_MASK_PAD_ROWS=false`（关掉上游的填充行清零） | 0.705 | 52 | 61 |
| breakable-200 | `--cuda-graph-backend-decode breakable` | 起不来：DSA 不支持（`'NoneType' object has no attribute 'get_page_table_64'`） | | |

MoE 排序实现、overlap 调度也排除。一个规律：乱码条数与 EXTEND 步数大致相当——base-p（64 题一次发完）6 个 prefill 步、9 条乱码；
mtp0-200 51 步、61 条；dly2000-200 40 步、61 条——像是每个 EXTEND 步之后坏掉约一个请求，而全 eager 时不坏。EXTEND 步本身两种
模式下一样（总是 eager），不同的只是之后的 decode 走 graph 还是 eager；PD 的 decode 节点只有 decode 步，所以没事。工作假设改为：
**eager 的 EXTEND 步与随后的 decode graph 回放之间，有共享状态没对齐**（v0.5.22 的 decode 输入缓冲区注册表与 ForwardBatch 共享
存储，见 `cuda_graph_buffer_registry.py`）。

**关掉 decode CUDA graph 就干净。** mtp0cg0b-200（关 MTP、`--disable-cuda-graph`，其余同基线：DP attention、HiCache、delayer、
融合算子都开）跑到 199 题：197 对、0 乱码、0 截断（22:41 手动停，剩 1 题在 eager 下以约 8 tok/s 慢慢跑，未到 7k token）。
对照同配置开 graph 的 mtp0-200：61 乱码、59 截断。加上 dpa0（关 DP attention，开 graph）也干净，结论是：
**DP attention 下的 decode CUDA graph 路径出错**，与 MTP、HiCache 无关；eager decode 太慢，不能当绕过办法。
下一项 `--cuda-graph-backend-decode breakable`：graph 分段捕获、注意力层 eager——干净则问题在 graph 内的注意力（DSA indexer /
注意力对 DP 填充行的处理），仍乱码则在 graph 内的 MoE / DP 通信；若干净且性能可接受，也是一个绕过办法。

上游的填充行清零开关（`SGLANG_DP_MASK_PAD_ROWS`）开或关结果一样，不是它；`breakable` graph 后端不支持 DSA，没法用来区分注意力
与 MoE；元数据 glue graph（`SGLANG_ENABLE_METADATA_GLUE_GRAPH`）默认就是关的。下一项用 v0.5.22 自带的 KV canary
（`--kv-canary log --kv-canary-real-data PARTIAL --kv-canary-sweep-interval 50`）：写 KV 时记内容哈希、读和定期巡检时校验。
有违规说明 graph decode 里有写坏别人 KV 的（如填充行），没有则问题在 hidden state / logits 一侧。配置简化为关 MTP、关 HiCache
（v0.5.19 / v0.5.20 上这个配置同样乱码），并跑同配置不开 canary 的对照 mtp0hc0-200。

| 变体 | 设置 | correct | 模板标签乱码 | runaway |
|---|---|---|---|---|
| mtp0hc0-200 | 关 MTP、关 HiCache（graph 开） | 0.695 | 53 | 59 |
| canary-200 | mtp0hc0 + `--kv-canary log --kv-canary-real-data partial --kv-canary-sweep-interval 50` | 起不来：kv-canary 不支持 `DSATokenToKVPool` | | |

最简配置（只剩 DP attention + decode graph + delayer + 融合算子）在我们的镜像上同样乱码；KV canary 只支持 MHA / SWA / DSv4 的池，
GLM-5.2 用不了。现成的开关到此基本用完，再往下要在引擎里插桩（比如每隔 N 步把同一批用 eager 再算一遍、比对 graph 回放的 logits），
留作下一步。23:00 起 138 换去跑 b2（agg 套件，piggyback 的同节点对照），期间不跑排查。

## 上游 issue 草稿（未提交，待 review）

**[Bug][ROCm] GLM-5.2 aggregated TP4 + DP-attention: outputs drift into garbage under sustained concurrency; needs DP attention + decode CUDA graphs**

- Hardware / images: 4 × MI355X (gfx950), ROCm 7.2; `lmsysorg/sglang-rocm` v0.5.19-rocm720-mi35x-20260917, v0.5.20-rocm720-mi35x-20260924,
  v0.5.22-rocm720-mi35x-20261018 (unpatched, `python3 -m sglang.launch_server`); model `amd/GLM-5.2-MXFP4`.
- Launch, as run on the unpatched images: `--tp-size 4 --ep-size 1 --dp-size 4 --enable-dp-attention --enable-dp-lm-head
  --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton --mem-fraction-static 0.85
  --chunked-prefill-size 32768 --max-prefill-tokens 8192 --max-running-requests 128 --cuda-graph-max-bs-decode 64
  --enable-prefill-delayer --prefill-delayer-max-delay-passes 60 --enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion
  --json-model-override-args '{"index_share_for_mtp_iteration":false}' --reasoning-parser glm45 --tool-call-parser glm47`;
  env `SGLANG_DP_USE_GATHERV=1 SGLANG_OPT_USE_TOPK_V2=false SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=0 AITER_USE_FLYDSL_MOE_SORTING=1`.
  The prefill delayer, the two fusions, `--enable-dp-lm-head` and GatherV were each switched off separately on v0.5.22 without effect.
- Workload: 200 GSM8K 5-shot chat prompts, temperature 0, `max_tokens` 16384, 64 concurrent requests (continuous arrival).
- Result: ~25-30% of answers degrade mid-decode (coherent reasoning, then isolated wrong tokens such as zero-width spaces, `!`
  (token 0) or unrelated CJK tokens, then repetition / digit soup / chat-template tags as text), most running to `max_tokens`.
  Accuracy 0.68-0.87 depending on version. Concurrency 1: fine.
- Same engine with `--disable-cuda-graph`: 197/199 correct, 0 degraded. Same engine without DP attention (TP4, graphs on): 0.980, 0 degraded.
- Not affected by: hierarchical cache, EAGLE MTP, prefill delayer, overlap scheduling, `SGLANG_DP_USE_GATHERV`, `--enable-dp-lm-head`,
  fused qk-norm-rope / aiter all-reduce fusion, DSA decode backend (triton / tilelang), DSA top-k (sgl_kernel fused / unfused / torch),
  `AITER_USE_FLYDSL_MOE_SORTING`, `SGLANG_DP_MASK_PAD_ROWS`.
- The number of degraded requests tracks the number of prefill (EXTEND) steps in the run. A PD deployment whose decode node only runs
  decode steps (v0.5.19, DP attention + graphs) scored 0.97.
