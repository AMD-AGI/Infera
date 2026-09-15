# GLM-5.2 AgentX warmup1 并发扫描：P4D8 与 P8D8（137/138）

## 目标

1. 用统一的 warmup1 口径扫描 P4D8 与 P8D8 的常规并发点（C32/C48/C64/C96/C128）。
2. 在每个拓扑上找出使 `total_token/s/GPU` 最大的**最小**并发点。
3. 记录每个点的性能、瓶颈归因，以及过程中遇到的问题与解决方案。

后续（视时间与机器余量）：扩展 Prefill worker 数（2 节点、3 节点）后重复同样的扫描。

## 隔离边界

本报告只涉及 Track 1。与 fused DSA indexer 的 GPU fault 根因调查严格隔离：

| | Track 1（本报告） | Track 2（fused indexer bug） |
|---|---|---|
| 代码目录 | `/home/liyingli/bench_agentx` | `/home/liyingli/bench_agentx_perm_fault_debug` |
| 机器 | `crsuse2-m2m-137` / `crsuse2-m2m-138` | `crsuse2-m2m-136` / `crsuse2-m2m-139` |
| 镜像 | `glm52-v518-c29bd17-b02ab81`（pin `sha256:6ff85f4a…`） | `glm52-v518-402df1e-2c71811-pr38583-*` |
| 容器前缀 | `glm52-pd-*` | `glm52-permfault-debug-*` |
| RCA 文档 | 本文件 | `bench_agentx_perm_fault_debug/RCA.zh-CN.md` |

## 指标定义：tok/s/GPU 到底是什么

取值来自每个点的 `agentx_conc<C>.json`：

```
request_metrics.throughput.per_gpu.total_tput_tps
       = (throughput.input.tokens_per_second + throughput.output.tokens_per_second)
         / (num_prefill_gpu + num_decode_gpu)
```

分母是拓扑自适应的：P8D8 = 8+8 = 16，P4D8 = 4+8 = 12，P4D4 = 4+4 = 8。
`tools/collect_agentx.py` 会用 `num_prefill_gpu + num_decode_gpu` 独立重算一遍，
并与 JSON 自带字段做 `rel_tol=1e-5` 对账，不一致直接抛错，因此该字段不是盲信值。

使用这个指标时必须知道它的三个性质，否则"最大值"会被误读：

1. **它几乎全是 prefill 输入 token。** P8D8 C64 的 input 为 225,615 tok/s，
   output 仅 1,527 tok/s，input 占 99.33%。这实质是 prompt token 吞吐，
   不是 decode 吞吐。
2. **input 计的是完整 prompt 长度，含 prefix cache 命中部分。**
   P8D8 C64 的 `theoretical_cache_hit_rate = 0.9687`，即其中约 97% 的 token
   并未真正参与计算。该指标因此对 cache 命中率高度敏感。
3. **平均 prompt 长度随并发变化，不是常量。** P8D8 pinned 三点分别为
   114,291 / 129,454 / 136,174 tok/请求。C32→C64 的 +73.7% 增幅中，
   约 +45.9% 来自请求速率上升（QPS 1.138→1.660），
   约 +19.1% 来自固定 1200 s 窗口纳入了更长的请求。

**推论（本次扫描的比较纪律）**：只在同镜像、同 warmup、同 trace、同 duration、
同 mem_fraction 的点之间比较；每个点都记录 cache 命中率，命中率显著偏离的点
不进入曲线。例如历史 P8D8 C128 命中率仅 0.8077（对比 pinned 三点的 ~0.968），
属于不可比点。

## 最小并发点的判定规则

曲线不一定单调上升到平台，也可能出现峰值后回落（P4D8 上已经观察到回落）。
因此判定规则定为：

> 在所有**有效**点中取 `tok/s/GPU` 的最大值 \(M\)，
> 答案是满足 `tok/s/GPU ≥ 0.97 M` 的**最小**并发。

这条规则同时覆盖"平台"和"内部峰值"两种形状。早期使用的
"下一点增幅 < 3% 即停"只在单调上升到平台时成立，曲线掉头时会给出错误答案，
已废弃。

有效性门禁：出现 Prefill OOR、GPU fault，或 profiling 阶段错误率触发门禁的点
判为无效，停在该点，把上一个有效点作为结论并显式标注"受容量截断、非真平台"。

## 配置族的选择（重要）

P8D8 已完成的 pinned warmup1 三点（C32/C48/C64）实际运行在
**`mem_fraction_static = 0.85`**，配方是 `config.p8d8.sh` 默认值 + warmup1 +
每点 `max_running = graph_max_bs = conc`，即：

- `PREFILL_MEM_FRACTION = DECODE_MEM_FRACTION = 0.85`
- `PREFILL_HSA_NO_SCRATCH_RECLAIM = 0`（仅 Prefill 生效）
- **无**主动 GC（`PYTORCH_HIP_ALLOC_CONF` 为空）
- `PD_DP_RANK_AFFINITY = 1`、`ENABLE_KV_AWARE = 1`（均为默认值）

证据：`retest-pinned-6ff85f4a-warmup1/c{32,64}/launch/server-info/prefill-0.json`
中 `mem_fraction_static = 0.85`。

因此本次扫描的两个拓扑统一使用该配置族，只改并发。任何偏离（如为压制 OOR 而
降低 mem_fraction）都作为独立的归因实验单独记录，不混入主曲线。

## 每点门禁

每个点都是**全新 launch**，理由有两条：`max_running` 与 `cuda_graph_max_bs`
必须等于并发；同时保证 prefix cache 冷启动，避免点与点之间继承缓存状态。

`gate.py` 在 launch 之后、压测之前做 fail-closed 校验：

- Prefill/Decode 的 `tp_size` / `dp_size` 与拓扑一致（P4D8 = 4/4 与 8/8）
- 两端 `max_running_requests == cuda_graph_max_bs_decode == conc`
- 两端 `mem_fraction_static` 等于计划值
- 两端 `status == "ready"`
- `launch.log` 记录的两节点 image ID 均等于 pin `sha256:6ff85f4a…`

最后一条是因为本 campaign 曾被"同 tag 指向不同内容"的镜像污染坑过一次
（ITL 从 9.90 ms 劣化到 43–46 ms），所以镜像 pin 必须在每个点上强制复核。

## 结果

`index.tsv` 为机器可读的逐点索引；`results.csv` 由 `tools/collect_agentx.py` 汇总。

### 结论（数据截至 2026-09-15 09:05 UTC，扫描已结束）

扫描按拓扑分组进行，**P8D8 优先扫完**（重点关注的组合），之后回到 P4D8。
**已于 2026-09-15 09:05 UTC 按要求停止，两台机器已清空。**

| 拓扑 | 最小并发（答案） | tok/s/GPU | 门槛 0.97 M | ITL P90 | TTFT P90 | Interactivity P90 | 已测点 |
|---|---|---|---|---|---|---|---|
| **P8D8** | **C80** | **16,268.53** | 15,780.47 | 17.62 ms | 18.29 s | 56.76 tok/s | C32/C48/C56/C64/C72/C80/C88/C96/C128 |
| **P4D8** | **C32** | **8,670.86** | 8,410.73 | 11.47 ms | 19.63 s | 87.16 tok/s | C24/C32/C36/C40/C48/C64 |

两个拓扑的峰值都已被左右更低的实测点夹住（P8D8 为 C80 ± 8，P4D8 为 C32 ± 4），
因此 M 与门槛都是可用的，上面是有效的判定结果，而不是"当前最高点"。
各自还差一个更细的点未测（P8D8 的 C76、P4D8 的 C28），
它们只可能把答案再下移一个 4 并发的台阶，**不确定度上界就是这一个台阶**。

#### 瓶颈：Prefill 算力限制吞吐，Decode KV 容量限制并发上限

两个拓扑的瓶颈结构完全同构，而且是**两段式**的——
这一点很重要，因为它决定了扩容该往哪边加卡。

运行期采样（每 15 s 一次，只取 decode 真正在服务的样本，每点 78–91 个样本）：

| | P4D8 C24 | P4D8 C36 | P4D8 C40 | P4D8 C64 | P8D8 C56 | P8D8 C72 | P8D8 C80 | P8D8 C96 |
|---|---|---|---|---|---|---|---|---|
| prefill 队列峰值 | 7 | 17 | 32 | 56 | 20 | 25 | 35 | 70 |
| prefill KV 峰值 | 0.25 | 0.33 | 0.28 | 0.20 | 0.24 | 0.22 | 0.17 | 0.17 |
| decode 队列峰值 | 5 | 11 | 7 | 0 | 0 | 3 | 5 | 1 |
| decode KV 峰值 | 0.29 | 0.49 | 0.55 | **0.96** | 0.38 | 0.47 | 0.56 | 0.66 |
| decode 传输队列中位数 | 2 | 11 | 18 | **48** | 4 | 8 | 13 | **45** |
| decode 预占中位数 | 0 | 0 | 0 | **16** | 0 | 0 | 0 | 0 |

##### 这些指标从哪来、怎么自己查

六行全部来自引擎自带的 Prometheus `/metrics`，由 `sample_metrics.py` 每 15 s
同时抓 prefill 与 decode 两端，逐点写进该点目录下的 `metrics_sample.log`。
gauge 是按 DP rank 打标签的，所以计数类**在 rank 上求和**，
比率类**在非零 rank 上求平均**（这就是 `token_usage` 能代表整个 worker 的原因）。

| 表中行 | 采样行里的字段 | Prometheus gauge | 取的统计量 |
|---|---|---|---|
| prefill 队列峰值 | prefill 段 `q=` | `sglang:num_queue_reqs` | 峰值 |
| prefill KV 峰值 | prefill 段 `kv=` | `sglang:token_usage` | 峰值 |
| decode 队列峰值 | decode 段 `q=` | `sglang:num_queue_reqs` | 峰值 |
| decode KV 峰值 | decode 段 `kv=` | `sglang:token_usage` | 峰值 |
| decode 传输队列中位数 | decode 段 `xfer=` | `sglang:num_decode_transfer_queue_reqs` | 中位数 |
| decode 预占中位数 | decode 段 `prealloc=` | `sglang:num_decode_prealloc_queue_reqs` | 中位数 |

正文里还用到同一行的 `run=`（`sglang:num_running_reqs`），
即"在跑请求数"，用来验证 prefill 是否始终满载。

**为什么队列和 KV 取峰值、传输队列和预占取中位数**：前两者要回答
"有没有触到上限"，一次触顶就有意义；后两者要回答"积压是不是持续存在"，
单点尖峰不能说明问题，必须是中位数抬起来才算堵住。

**采样窗口的过滤**：只保留 `decode run > 0` 的样本。launch 后的加载、
warmup 和压测结束后的排空阶段两端都是 0，混进来会把峰值和中位数同时稀释。
过滤后每点剩 78–91 个样本（约 20–23 分钟），与 1200 s 的压测窗口吻合。

实时查看（压测进行中，IP 为各自的 data_ip，端口是 `ENGINE_PORT_BASE + 拓扑行号`）：

```bash
python3 sample_metrics.py \
  --prefill 10.245.153.247:19001 --decode 10.245.157.237:19002 \
  --interval 15 --out /tmp/live.log
tail -f /tmp/live.log
```

从已跑完的点复算本表（同一口径，可直接核对）：

```bash
python3 - <<'PY'
import re, statistics, pathlib
FIELD = re.compile(r"(\w+)=([-0-9.]+)")
rows = []
for line in pathlib.Path("p8d8/c80/metrics_sample.log").read_text().splitlines():
    if not line or line.startswith("#") or "<" in line:
        continue          # 抓取失败的样本形如 prefill=<URLError>
    parts = line.split(" ", 1)[1].split("|")
    if len(parts) != 2:
        continue
    p, d = ({k: float(v) for k, v in FIELD.findall(c)} for c in parts)
    if d["run"] > 0:      # 只保留 decode 真正在服务的样本
        rows.append((p, d))
print("样本数", len(rows))
print("prefill q peak", max(p["q"] for p, _ in rows), "kv peak", max(p["kv"] for p, _ in rows))
print("decode  q peak", max(d["q"] for _, d in rows), "kv peak", max(d["kv"] for _, d in rows))
print("xfer med", statistics.median(d["xfer"] for _, d in rows),
      "prealloc med", statistics.median(d["prealloc"] for _, d in rows))
PY
```

##### 怎么判断当前跑在哪一段

判据不是某一行单独看，而是**"prefill 在排队"与"decode 有没有开始反压"
这两组信号的组合**：

| 信号 | 第一段（峰值及以下） | 第二段（越过峰值） |
|---|---|---|
| prefill `run=` | 恒等于 prefill GPU 数，满载 | 同样满载（这一行两段不可区分） |
| prefill `q=` | 随并发单调增长 | 继续放大 |
| prefill `kv=` | 0.17–0.33，几乎空着 | 仍然空着（**不会**因为过载而上升） |
| decode `kv=` | 稳步上升但明显低于 0.9 | 持续抬升，极端时逼近 1.0 |
| decode `xfer=` 中位数 | 个位数到十几 | **跳到 40 以上** |
| decode `prealloc=` 中位数 | 恒为 0 | 出现非 0 |

**读表的顺序很重要，三个 decode 信号出现的时间不一样**：

1. **`xfer` 中位数是最早的信号**。P8D8 C80→C96 从 13 跳到 45，
   而同期 decode KV 只从 0.56 升到 0.66、`prealloc` 仍是 0。
   也就是说 C96 已经进入第二段，但只有传输队列这一行说得出来。
2. **decode `kv=` 是中期信号**，真正压到 0.9 以上要更晚
   （P8D8 要到 C128 才 0.94）。
3. **`prealloc` 非 0 是末期信号**，代表 decode 已经在为还没完成 prefill 的
   请求占槽位。全部实测点里只有 P4D8 C64 出现（中位数 16），
   它同时也是 decode KV 唯一到 0.96 的点。

**一条关键的反直觉判据**：`prefill kv=` 在两段都停在 0.2 左右。
所以"prefill 显存没涨"**不能**用来论证系统还没过载——恰恰相反，
prefill 排着长队而显存空着，本身就是"受限于算力而非容量"的证据。

**另一条容易误判的**：越过峰值后 ITL P90 可能不升反降
（C128 的 15.73 ms 优于 C96 的 17.35 ms），因为能挤进 decode 的请求变少了。
只看 ITL 会得出"系统很健康"的错误结论，必须与吞吐和 `xfer` 一起看。
判定阶段时，**吞吐相对峰值是否回落**始终是最终依据，上面这些
gauge 是用来解释"为什么回落"的。

**第一段（峰值及以下）：prefill 算力是吞吐的限制项。**

1. Prefill 的在跑请求数恒等于它的 DP 宽度（P8D8 为 8、P4D8 为 4，
   与各自的 prefill GPU 数相同），始终满载。
2. Prefill 队列随并发单调堆积，而 **prefill 自己的 KV 池几乎是空的**
   （峰值仅 0.17–0.33）。一个排着队却只用了两成显存的 worker，
   受限的只能是算力——本 trace 平均输入 125 k token，prefill 的实算量极大。
3. **ITL 几乎不随并发变化，TTFT 却呈指数增长。** P8D8 从 C32 到 C96，
   ITL P90 只从 11.63 ms 涨到 17.35 ms（+49%），
   同期 TTFT P90 从 7.1 s 涨到 65.8 s（**+827%**）。排队全在首 token 之前。

**第二段（越过峰值之后）：decode 的 KV 容量开始反压。**

decode KV 占用随并发近似线性增长（P8D8：C56 0.38 → C80 0.56 → C96 0.66
→ C128 0.94，见问题 10）。一旦进入 0.9 以上，
KV 传输队列就会爆炸式堆积（C96 中位数 45、C128 达 123），
已完成 prefill 的请求排队等 decode 腾出 KV 空间。
P4D8 C64 更极端：decode KV 0.96、**预占中位数 16**
（其余各点均为 0）——decode 为还没完成 prefill 的请求占住了槽位。

**两段的关系**：prefill 慢 → 请求积压 → 在途请求变多 → decode 需要同时
持有更多 KV → 容量耗尽 → 驱逐前缀 → 命中率下降 → prefill 实算量上升。
所以第二段是第一段的**后果**，不是独立原因。这也是为什么越过峰值后
不是缓慢劣化而是断崖。

decode 的**算力**则自始至终是空闲的：C80 点上 output 吞吐只有 2,007.81 tok/s，
而 input 是 258,288.73 tok/s，**相差 128 倍**，8 张 decode 卡每卡仅 251 tok/s 输出。
**decode 消耗的是容量，不是算力**——这个区分直接决定了下面的配比建议。

#### 推荐的 PD 配比：应该继续往 prefill 倾斜

把 prefill 从 4 卡加到 8 卡（decode 固定 8 卡）的收益是**超线性**的：

| 配比 | 总 GPU | 最优点总吞吐 | tok/s/GPU |
|---|---|---|---|
| P4D8（1:2） | 12 | 104,050 tok/s | 8,670.86 |
| P8D8（1:1） | 16 | 260,296 tok/s | 16,268.53 |

**GPU 数只增加 33%，总吞吐却增加到 2.50 倍**，每卡吞吐翻了 87.6%。
在一个瓶颈资源上加卡才会有这种超线性收益，这从另一个角度印证了 prefill 是瓶颈。

因此对这类**输入占 99%、前缀命中约 96%** 的 agent 负载：

- **已测范围内推荐 P8D8（1:1），并发取 C80。** 这是本次实测到的最优配置，
  延迟侧也支持这个点：TTFT P90 在 C80 之前都还在 18 s 上下，
  一过 C80 就跳到 40.7 s（C88）、65.8 s（C96），
  **吞吐峰值与延迟拐点恰好重合**。
- **扩容优先加 prefill 卡，保持 decode 不变**（P12D8、P16D8 方向）。
  prefill 是吞吐的限制项，而它的 KV 只用了 17%，加卡直接换吞吐。
- **不要简单地把 decode 卡挪给 prefill。** 这是一个容易犯的错误：
  decode 的**算力**确实闲着（每卡仅 251 tok/s 输出），
  但它的 **KV 容量是实打实被占用的**，且随并发线性增长——
  C80 点已用掉 0.56。把 decode 从 8 卡减到 4 卡会让容量减半，
  同样并发下 decode KV 直接超限，第二段瓶颈会提前触发。
  P8D4 若要成立，并发大约要退到 C40–C56 一档，**是否净赚必须实测**。
- **不推荐 P4D8**，除非只有 12 张卡可用。它的 4 张 prefill 卡在 C40
  就让 decode 的 KV 传输队列堆到 18（中位数），C48 起整条曲线坍塌。
- **这个建议绑定当前 trace 形态。** 若输出占比上升（更长的生成、更少的前缀复用），
  decode 的算力会从闲置转为瓶颈，上述比例需要重测。

### 已有的对照基线

P8D8，pinned warmup1，mem 0.85（本次扫描的上游基线）：

| 并发 | tok/s/GPU | 相对前一点 | ITL P50 | cache 命中 |
|---|---|---|---|---|
| C32 | 8,173.07 | — | 9.91 ms | 0.9649 |
| C48 | 11,485.48 | +40.5% | 10.47 ms | 0.9693 |
| C64 | 14,196.34 | +23.6% | 11.81 ms | 0.9687 |

C64 处仍以 +23.6% 上升，远未进入平台，故峰值在 C64 以上，需要 C96/C128。

P4D8 历史点（warmup10，非本配置族，仅作趋势参考）：

| 并发 | tok/s/GPU | cache 命中 | 来源 |
|---|---|---|---|
| C8 | 2,745.37 | 0.9779 | `20260911_p4dpa_d8dpa_c8-64_d1200_137_138` |
| C16 | 5,290.29 | 0.9672 | 同上 |
| C32 | 11,804.13 | 0.9643 | 同上 |

### 本次扫描逐点结果

统一口径：warmup1、`mem_fraction_static=0.85`、pinned 镜像、duration 1200 s、
每点全新 launch、`max_running = graph_max_bs = conc`。

#### P4D8（Prefill TP4/DPA4 @137 + Decode TP8/DPA8 @138，12 GPU）

吞吐与容量：

| 并发 | tok/s/GPU | input tok/s | 计分请求 | QPS 均值 | 服务端命中 | kv_usage | 判定 |
|---|---|---|---|---|---|---|---|
| C24 | 6,268.14 | 74,641 | 777 / 826 | 0.639 | 0.9367 | 0.65 | PASS（为最高点的 72.3%，**未达门槛**） |
| C32 | **8,670.86** | 103,256 | 1,189 / 1,255 | 0.971 | 0.9286 | 0.73 | PASS（最高点，门槛 100%） |
| C36 | 8,498.05 | 101,246 | 1,128 / 1,205 | 0.919 | 0.9202 | 0.75 | PASS（为最高点的 98.0%，**达门槛**） |
| C40 | 8,257.17 | 98,314 | 1,074 / 1,158 | 0.876 | 0.9056 | 0.90 | PASS（为最高点的 95.2%，**未达门槛**） |
| C48 | 3,274.57 | 39,009 | 400 / 499 | 0.327 | 0.6501 | 1.00 | PASS（已越过容量悬崖） |
| C64 | 2,059.90 | 24,546 | 208 / 339 | 0.169 | 0.3430 | 1.00 | PASS（重测成功，见问题 7） |

延迟（P50 与 P90 并列；interactivity 为每用户每秒输出 token 数，**越大越好**，
因此它的 P90 是**低分位**，代表最慢的那 10% 用户的体感）：

| 并发 | tok/s/GPU | ITL P50 | ITL P90 | TTFT P50 | TTFT P90 | Intvty P50 | Intvty P90 | E2E P90 |
|---|---|---|---|---|---|---|---|---|
| C24 | 6,268.14 | 9.06 ms | 10.31 ms | 4.2 s | 13.4 s | 110.3 | 97.0 | 28.2 s |
| C32 | **8,670.86** | 9.95 ms | **11.47 ms** | 5.3 s | **19.6 s** | 100.5 | **87.2** | 38.3 s |
| C36 | 8,498.05 | 9.90 ms | 11.33 ms | 8.7 s | 34.9 s | 101.1 | 88.2 | 54.4 s |
| C40 | 8,257.17 | 10.23 ms | 12.23 ms | 9.5 s | 51.1 s | 97.7 | 81.8 | 70.9 s |
| C48 | 3,274.57 | 9.23 ms | 11.36 ms | 3.7 s | 423.2 s | 108.4 | 88.0 | 429.2 s |
| C64 | 2,059.90 | 7.99 ms | 10.86 ms | 92.0 s | 696.0 s | 125.1 | 92.1 | 703.6 s |

##### 逐点瓶颈

**这张延迟表本身就是瓶颈定位的最强证据。** ITL 和 interactivity 在全部六个点上
几乎是平的（ITL P90 始终在 10.3–12.2 ms，intvty P90 始终在 82–97），
**即使在吞吐已经坍塌 76% 的 C64 上，ITL P90 反而是全场第二低的 10.86 ms**。
一个真正过载的 decode 不可能给出这种曲线。同期 TTFT P90 却从 13.4 s 一路涨到 696 s，
涨了 52 倍。排队全部发生在首 token 之前。

- **C24（欠载）**：prefill 队列峰值仅 7，4 个 DP 槽位有空转。
  命中率 0.9367 是全场最高，decode KV 只用 0.29，纯粹是并发不够、喂不饱 12 张卡。
- **C32（最优）**：prefill 满载（run 恒为 4）但队列还压得住（峰值 17），
  decode 的 KV 传输队列中位数仅 2。吞吐触顶而 TTFT P90 还在 20 s 以内。
- **C36 / C40（prefill 排队开始显性化）**：prefill 队列峰值升到 17 / 32，
  TTFT P90 翻到 34.9 s / 51.1 s。注意 **ITL 基本没动**——
  多出来的时间全部花在等待 prefill 上。C40 的 decode KV 传输队列
  中位数已堆到 18、KV 占用 0.55，是坍塌前的最后一个点。
- **C48 / C64（坍塌）**：这里发生的是一个自我强化的死循环。
  prefill 追不上 → 请求在 prefill 侧排队（C64 队列峰值 56）→
  decode 为尚未完成 prefill 的请求**预占** KV（C64 的 prealloc 中位数 16，
  其余各点均为 0）→ decode KV 顶到 0.96 → radix cache 被迫驱逐 →
  命中率从 0.93 崩到 0.6501 / 0.3430 → prefill 需要重算的 token 暴增 →
  prefill 更追不上。C64 的 TTFT P50 高达 92 s，说明**一半以上的请求**
  都卡在这个循环里。

结论：P4D8 全程由 4 张 prefill 卡的算力限制吞吐；
decode 的算力从未吃紧，但它的 KV 容量在 C48 之后被积压的在途请求撑满，
成为坍塌的放大环节。

C64 首次运行的数据丢失已通过重测补齐，实测 2,059.90，比 C48 更低，
与"越过悬崖后单调恶化"的判断一致。

##### P4D8 的判定

M = 8,670.86（C32），门槛 0.97 M = **8,410.73**。达标情况：

| 并发 | tok/s/GPU | 占 M | 是否达标 |
|---|---|---|---|
| C24 | 6,268.14 | 72.3% | 否 |
| C32 | 8,670.86 | 100% | 是 |
| C36 | 8,498.05 | 98.0% | 是 |
| C40 | 8,257.17 | 95.2% | 否 |

达标区间是 [C32, C36] 这个闭合窗口，两侧的 C24（72.3%）和 C40（95.2%）都掉出门槛。
要找的是**最小**并发，所以 **P4D8 的答案是 C32 = 8,670.86 tok/s/GPU**。
C28 未测，它只可能把答案再下移一档；不对它做插值估计——
C24 与 C32 只差 8 并发、吞吐却相差 27.7%，是全曲线最陡的一段，不足以支撑外推。

#### P8D8（Prefill TP8/DPA8 @137 + Decode TP8/DPA8 @138，16 GPU）

本轮新测两点，与上游 pinned 基线（C32/C48/C64）**同镜像、同 warmup1、同 mem 0.85**，
可直接并入同一条曲线（可比性验证见下）：

| 并发 | tok/s/GPU | 服务端命中 | kv_usage | ITL P50 | 来源 |
|---|---|---|---|---|---|
| C32 | 8,173.07 | 0.9502 | 0.50 | 9.91 ms | pinned 基线 |
| C48 | 11,485.48 | 0.9579 | 0.67 | 10.47 ms | pinned 基线 |
| C64 | 14,196.34 | 0.9563 | 0.96 | 11.81 ms | pinned 基线 |
| C96 | 13,423.83 | 0.9186 | 0.95 | 13.77 ms | **本轮重测，PASS** |
| C128 | 5,358.08 | 0.6908 | 1.00 | 12.01 ms | **本轮，PASS** |

这张表只用于验证本轮两点与 pinned 基线的口径一致性，不承担判定；
完整曲线（含 C56/C72/C80/C88）见下一节。

C96 明细：input 212,990 tok/s、计分 2,203 / 2,400、QPS 均值 1.794、
理论命中 0.9597、TTFT P50 9.6 s。

**可比性验证**：基线目录 `retest-pinned-6ff85f4a-warmup1/` 的 pin
（`sha256:6ff85f4a…`）与本轮每个点 `launch.log` 记录的镜像 ID 逐点一致；
两批的 warmup 丢弃数按并发一一对应（C32=66、C48=99、C64=131），
说明 warmup 口径相同。因此五个点构成一条口径统一的曲线。

##### P8D8 逐点结果

吞吐与容量：

| 并发 | tok/s/GPU | 占 M | 服务端命中 | kv_usage | 是否达标 |
|---|---|---|---|---|---|
| C32 | 8,173.07 | 50.2% | 0.9502 | 0.50 | 否 |
| C48 | 11,485.48 | 70.6% | 0.9579 | 0.67 | 否 |
| C56 | 12,861.91 | 79.1% | 0.9570 | 0.96 | 否 |
| C64 | 14,196.34 | 87.3% | 0.9563 | 0.96 | 否 |
| C72 | 15,625.18 | 96.0% | 0.9517 | 0.99 | 否（差 1.0%） |
| C80 | **16,268.53** | **100%** | 0.9489 | 0.99 | **是** |
| C88 | 15,253.66 | 93.8% | 0.9395 | 0.98 | 否 |
| C96 | 13,423.83 | 82.5% | 0.9186 | 0.95 | 否 |
| C128 | 5,358.08 | 32.9% | 0.6908 | 1.00 | 否 |

延迟（interactivity 越大越好，其 P90 为低分位，代表最慢的 10% 用户）：

| 并发 | tok/s/GPU | ITL P50 | ITL P90 | TTFT P50 | TTFT P90 | Intvty P50 | Intvty P90 | E2E P90 |
|---|---|---|---|---|---|---|---|---|
| C32 | 8,173.07 | 9.91 ms | 11.63 ms | 2.3 s | 7.1 s | 100.9 | 86.0 | 24.3 s |
| C48 | 11,485.48 | 10.47 ms | 12.50 ms | 2.4 s | 9.1 s | 95.5 | 80.0 | 27.9 s |
| C56 | 12,861.91 | 11.28 ms | 13.38 ms | 2.9 s | 11.9 s | 88.7 | 74.7 | 33.6 s |
| C64 | 14,196.34 | 11.81 ms | 15.11 ms | 3.2 s | 12.8 s | 84.7 | 66.2 | 37.3 s |
| C72 | 15,625.18 | 12.80 ms | 17.18 ms | 3.9 s | 17.5 s | 78.1 | 58.2 | 42.7 s |
| C80 | **16,268.53** | 13.60 ms | **17.62 ms** | 4.3 s | **18.3 s** | 73.5 | **56.8** | 45.3 s |
| C88 | 15,253.66 | 13.26 ms | 17.55 ms | 5.9 s | 40.7 s | 75.4 | 57.0 | 69.3 s |
| C96 | 13,423.83 | 13.77 ms | 17.35 ms | 9.6 s | 65.8 s | 72.6 | 57.6 | 86.9 s |
| C128 | 5,358.08 | 12.01 ms | 15.73 ms | 18.9 s | 383.0 s | 83.3 | 63.6 | 399.8 s |

C80 明细：input 258,289 tok/s、计分 2,519 / 2,683、QPS 均值 2.054、理论命中 0.9634。
C88 明细：input 242,141 tok/s、计分 2,327 / 2,508、QPS 均值 1.897、理论命中 0.9637。
C72 明细：input 248,311 tok/s、计分 2,287 / 2,435、QPS 均值 1.868、理论命中 0.9673。

##### P8D8 的判定

M = 16,268.53（C80），门槛 0.97 M = **15,780.47**，九个点中只有 C80 达标。
**因此 P8D8 的答案是 C80 = 16,268.53 tok/s/GPU。**
C72 差门槛 1.0%，C76 未测，它只可能把答案再下移一档。

##### 逐点瓶颈

与 P4D8 完全同构：**ITL P90 在 C32→C96 之间只从 11.63 ms 涨到 17.35 ms（+49%），
同期 TTFT P90 从 7.1 s 涨到 65.8 s（+827%）**，
prefill 的在跑请求数全程恒为 8（等于 prefill GPU 数），队列单调堆积。

- **C32 / C48（欠载）**：prefill 有空转，kv_usage 仅 0.50 / 0.67，
  TTFT P90 还在 10 s 以内，interactivity P90 高达 86 / 80。
  命中率在 C48 达到全场最高的 0.9579。
- **C56 / C64（进入满载）**：聚合 kv_usage 一步跳到 0.96 并停在那里，
  prefill 队列峰值 20。吞吐仍在爬升，因为命中率还稳在 0.956，
  prefill 真正要算的 token 没有增加。
- **C72 / C80（峰值区）**：聚合 kv_usage 0.99，命中率 0.9517 / 0.9489——
  **容量刚好用尽但驱逐尚未开始**，这是吞吐能冲到最高的原因。
  C80 的 prefill 队列峰值 35、decode KV 传输队列中位数 13，
  而 decode 自身的 KV 占用只有 0.56，仍有余量。
- **C88（越过拐点）**：TTFT P90 从 18.3 s 跳到 40.7 s（**翻倍**），
  而 ITL P90 纹丝不动（17.62 → 17.55 ms）。命中率掉到 0.9395，
  聚合 kv_usage 从 0.99 回落到 0.98——这个回落不是缓解，
  而是**驱逐开始丢弃前缀**，可复用的内容变少，于是 prefill 的实算量上升。
- **C96（下坡）**：命中率 0.9186，prefill 队列峰值 70，
  decode KV 传输队列中位数升到 45（C80 时是 13）、KV 占用升到 0.66，
  TTFT P90 达 65.8 s。第二段瓶颈（decode 容量）开始显性化。
- **C128（坍塌）**：命中率崩到 0.6908、聚合 kv_usage 打满 1.00，
  decode 自身 KV 达 0.94、传输队列堆到 123（详见问题 10），TTFT P90 383 s。
  注意 ITL P90 反而降到 15.73 ms、interactivity P90 回升到 63.6——
  **因为能挤进 decode 的请求变少了，少数幸存请求跑得很顺，
  大量请求则堵在 prefill 前**。这是典型的过载假象：
  只看 ITL 会误判为"系统很健康"。

一个容易误读的细节：C128 的 ITL/interactivity 比 C96 更好，
但它的吞吐只有 C96 的 40%。**在这类 prefill 受限的负载上，
ITL 和 interactivity 不能单独用来判断系统是否过载**，必须和 TTFT 一起看。

#### 跨点的单一解释变量：服务端**实际**命中率

把所有已完成点并排看，有一个量几乎单独解释了全部吞吐差异——
服务端的实际 prefix cache 命中率：

| 点 | tok/s/GPU | 服务端实际命中 | 客户端理论命中 | kv_usage | TTFT P50 |
|---|---|---|---|---|---|
| P4D8 C32 | 8,670.86 | **0.9286** | 0.9620 | 0.73 | 5.3 s |
| P4D8 C40 | 8,257.17 | **0.9056** | 0.9601 | 0.90 | 9.5 s |
| P4D8 C48 | 3,274.57 | **0.6501** | 0.9684 | 1.00 | 3.7 s |
| P4D8 C64 | 2,059.90 | **0.3430** | 0.9653 | 1.00 | 92.0 s |
| P8D8 C128 | 5,358.08 | **0.6908** | 0.9565 | 1.00 | 18.9 s |

三点读法：

1. **客户端理论命中率在所有点上都稳定在 0.956–0.968**。这是 trace 自身的可复用度，
   与并发无关。所以命中率的塌陷完全不是负载变了，而是**服务端留不住**。
2. **理论值与实际值之间的差就是纯粹的驱逐损失**。C32 只差 3.3 个点，
   C64 差了 62 个点——本来能命中的 token 被逐出后必须重算。
3. **`kv_usage` 从 C48 起就恒为 1.00**，已经饱和、失去分辨力；
   真正还能区分点位好坏的是命中率。因此判断一个点是否越过悬崖，
   应当看服务端实际命中率而不是 kv_usage。

C40 是这条曲线上最有价值的一点：它的 kv_usage 已经到 0.90、命中率掉到 0.9056，
说明**悬崖边缘就在 C32 与 C40 之间**，C40 已经踩在下坡上但还没坠落。

**C32 → C48 不是平台，是断崖：tok/s/GPU 下降 62.2%。**

这里有一个容易误读的地方：C48 的 ITL P50（9.23 ms）和 TTFT P50（3,733 ms）
都比 C32（9.95 ms / 5,308 ms）**更好**，但吞吐只有 C32 的 38%。
延迟更好而吞吐更差，说明问题不在"每个 token 变慢"，而在"完成的请求变少"——
同样 1200 s 窗口，C32 计分 1,189 条，C48 只有 400 条，
且 C48 有超过一半的 1 秒窗口是零完成（QPS p50 = 0.0）。

##### 瓶颈归因：Prefill 算力不足，打满的是 Decode 侧的 KV

运行期服务端指标把机制说得很清楚（每第 4 个采样点）：

| 阶段 | C32 | C48 |
|---|---|---|
| kv_usage（聚合） | 6%–56% 震荡下行 | **85%–98% 全程顶满** |
| 队列 running/waiting | 3–4r / 0–6w，能排空到 0w | **1r / 10–11w，从不排空** |
| srv prefix 命中 | 92%–97% 稳定 | **90.6% → 69.9% 单调下滑** |
| tput_in_srv | 141k → **204k/s 上升** | 169k → **78k/s 腰斩** |
| unique_in_srv | 17.1 M | **28.9 M** |

**这里要修正本报告先前的一处归因错误。** 上面这张表用的是聚合 kv_usage，
不区分 prefill 与 decode，我据此写过"P4D8 的约束是 Prefill 侧 KV 容量"。
后来按角色分开采样，结论正好相反：

| P4D8 C64 采样时刻 | prefill kv | prefill 队列 | decode kv | decode prealloc |
|---|---|---|---|---|
| 00:28 | 0.073 | 21 | 0.000 | 0 |
| 00:41 | 0.105 | 20 | 0.267 | 0 |
| 00:47 | 0.037 | 41 | **0.890** | **28** |
| 00:53 | 0.028 | 55 | **0.850** | 15 |

**Prefill 的 KV 池全程几乎是空的（0.03–0.15），队列却堆到 55。**
一个排着 55 个请求、显存只用了 3% 的 worker，不可能是被显存卡住的。
打满的是 decode 的池，而且填满它的是 `prealloc`——
decode 为那些**还没完成 prefill** 的请求预占的槽位。

修正后的因果链：

4 张 Prefill 卡算不过来（平均输入 125 k token）→ 请求在 prefill 侧排队
→ decode 为排队中的请求预占 KV（prealloc 升到 28）→ decode KV 打满
→ radix cache 被迫驱逐 → prefix 命中率从 90.6% 掉到 69.9%
→ 每条请求需要真正计算的 unique token 从 17.1 M 涨到 28.9 M（+69%）
→ prefill 更加追不上 → 队列进一步加长。**这是一个正反馈回路**，
所以 C48 之后不是缓慢劣化而是断崖。

注意 `kv_cache_pool_tokens` 两点都是 33,157,888，而 C48 的 48 条 lane
按平均 prompt 119 k 计只占 5.7 M（17%）。之所以仍会打满，是因为 radix cache
保留的是**所有会话的历史前缀**而不只是活跃上下文；agentic replay 的会话上下文
持续增长，48 条 lane 的历史总量远大于 32 条。

**因此 P4D8 的约束是 Prefill 算力**，KV 打满是这个约束的**后果**而非原因。
这也解释了为什么 ITL 反而更好：decode 侧真正在跑的请求变少了。

##### 曲线形态：两侧都下降，但原因完全不同

**C24 上升 → C32 触顶 → C36 微降 → C40 回落 → C48 坠崖 → C64 继续恶化。**
C32→C36 只掉 2.0%、C36→C40 再掉 2.8%，随后 C40→C48 直接掉 60.3%，
下坡是先缓后崩的，悬崖在 C40 与 C48 之间。

峰值两侧虽然都是下降，成因却相反，不能混为一谈：

- **左侧（C24）是欠载。** 它的服务端命中率有 0.9367，比 C32 的 0.9286 还高，
  kv_usage 只有 0.65，prefill 队列峰值仅 20——容量和算力都有余量，
  它慢只是因为并发不够、喂不饱这 12 张卡。
- **右侧（C40 起）是 prefill 追不上导致的连锁饱和**，机制见上一节。

区分这两者在实践上是有意义的：左侧可以靠加并发解决，右侧只能靠加 prefill 卡。

## 问题与解决方案

### 1. warmup 口径混用导致曲线不可比

- **现象**：早期点位混用 warmup10 与 warmup1，两者的 prefix cache 预热程度不同，
  而 tok/s/GPU 对 cache 命中率高度敏感（见指标定义第 2 条）。
- **解决**：本次扫描统一 warmup1（`AGENTX_WARMUP_REQUESTS_PER_LANE=1`），
  并在报告中显式标注历史 warmup10 点为"仅作趋势参考"，不与新点同表比较。
- **代价**：warmup1 缩短了 cache-pressure 阶段，降低了 Prefill OOR 的触发概率，
  因此 warmup1 通过不能反推 warmup10 也会通过。

### 2. 镜像污染（同 tag 不同内容）

- **现象**：`glm52-v518-c29bd17-b02ab81` 这个 tag 曾被覆盖为错误 commit
  （402df1e/2c71811），导致 ITL 从 9.90 ms 劣化到 43–46 ms。
- **解决**：固定 digest `sha256:6ff85f4a43ae…`（内含 SGLang `c29bd17`、
  AITER `b02ab81`），并在 `sweep.sh` 启动时与每个点的 `gate.py` 中双重复核
  两节点的实际 image ID。

### 3. 不重启服务端换客户端参数时，编排脚本的 trap 会误杀服务端

- **现象**：为把 C48 从 warmup10 改为 warmup1，需要只重启客户端。
  但外层编排循环带有 EXIT trap 调用 `stop.sh`，直接终止它会连服务端一起停掉。
- **解决**：对外层循环用 SIGKILL（trap 不触发），保住服务端；
  再处理内层 `agentx_bench.sh`。
- **二次现象**：对 `agentx_bench.sh` 发 SIGTERM 后进程不退出。
  原因是 bash 在前台子进程（`ssh` 跑客户端容器）运行期间会**延迟**执行 trap，
  必须等前台命令结束。
- **解决**：直接 `docker stop` 客户端容器，使前台 `ssh` 返回，
  其 cleanup trap 随即执行，脚本正常退出，服务端不受影响。

### 4. Prefill 的 `/flush_cache` 返回 HTTP 400

- **现象**：中止上一轮后立即 flush，Prefill 返回 400，Decode 正常。
- **原因**：SGLang 在存在 running/waiting 请求时拒绝 flush。
- **解决**：先轮询 `/metrics` 的 `sglang:num_running_reqs` 与
  `sglang:num_queue_reqs` 等待排空，再 flush；确认 Prefill 4 个 rank、
  Decode 8 个 rank 都输出 `Cache flushed successfully`。
- **本次扫描的改进**：改为每点全新 launch，从根上避免跨点缓存继承，
  不再依赖 flush 流程。

### 5. mem_fraction 偏离造成的 P4D8 假崩塌

- **现象**：P4D8 C48 在 `mem_fraction_static = 0.70` + 主动 GC 下测得
  **2,550.14 tok/s/GPU**，而 P4D8 C32（warmup10）为 11,804.13，
  P8D8 C48 为 11,485.48。看起来像是 C32→C48 崩塌了 4.6 倍。
- **归因**：运行期日志显示 `kv_usage` 达 95%，服务端 prefix 命中率从 85.8%
  一路掉到 73.6%，`unique_in_srv` 从 696 万涨到 1,553 万，队列积压 11 条。
  即 KV 容量不足 → radix cache 被驱逐 → 命中率下降 → 实际 prefill 计算量激增
  → 请求排队 → 固定窗口内完成数锐减 → 客户端测得吞吐崩塌。
- **根本原因**：`mem_fraction 0.70` 是为压制 Prefill OOR 引入的缓解措施，
  但它同时削减了 KV 容量；而 P8D8 的对照点跑在 0.85。两者不同口径，
  这个"崩塌"里混入了我自己引入的变量。
- **解决**：主曲线统一回到 mem 0.85（与 P8D8 对照点一致）。
  0.70 + 主动 GC 的那个点保留为"OOR 缓解措施的性能代价"归因样本，
  单独记录，不进主曲线。
- **后续修正（重要）**：在 mem 0.85 下重测 C48 得到 **3,274.57**，
  相比 0.70 的 2,550.14 只回升了 28.4%，仍比 C32 的 8,670.86 低 62.2%。
  所以**崩塌本身是真实的容量悬崖，mem_fraction 只解释了其中约四分之一**。
  我最初把整个崩塌归因于 mem 0.70 是错的；是这个受控点把两个变量拆开了。
  正确结论：mem 0.70 让悬崖更深，但悬崖在 mem 0.85 下同样存在，
  位置在 C32 与 C48 之间。教训是——改了缓解措施就必须重测对照点，
  否则缓解措施会被误当成根因。
- **遗留风险**：P4D8 在 mem 0.85 下历史上于 C48 连续 3 次触发 Prefill OOR
  （mem 0.85/0.80/0.70 各一次，均为 warmup10 且当时没有主动 GC 与 rank
  affinity）。本次在 mem 0.85 + warmup1 + affinity 下重测，
  若仍 OOR 则按有效性门禁停在该点。

### 6. Prefill OOR 只有缓解、没有根因修复

- **现状**：压住 OOR 的是三件叠加措施——Prefill-only `HSA_NO_SCRATCH_RECLAIM=0`、
  降低 `mem_fraction`、以及 `PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.8`
  的主动 GC。中途试过的 `HSA_SCRATCH_SINGLE_LIMIT_ASYNC=8G` 为 no-op 已回退，
  `expandable_segments` 在该 ROCm build 上不支持。
- **最强假设**：scratch 驻留叠加分配器碎片；未做到定位级修复。
- **对本次扫描的影响**：P4D8（Prefill 仅 4 卡，单 rank 长上下文峰值压力约为
  P8D8 的两倍）在高并发点仍可能触发；P4D4 还会多一层 decode 侧 KV 减半的风险。

### 7. 编排工具的两处缺陷，以及一次**被我误判**的失败归因

- **现象**：P4D8 C64 记为 `BENCH_FAILED`，其 `bench/` 目录里只留下
  `aiperf_artifacts/profile_export.jsonl`，没有 `runner.log`。
- **确实存在并已修复的工具缺陷（两处）**：
  1. `sweep.sh` 原先用 `while read ... done < points.tsv` 驱动主循环，
     而循环体里的 `launch.sh` / `agentx_bench.sh` 会调用 `ssh`，
     **ssh 读取 stdin**，把计划文件剩余的行吃光，扫描在第一个点之后就退出了。
     改为先 `mapfile` 读入数组，并给三处内层调用加 `</dev/null`。
  2. supervisor 原用 `pgrep -f "$CAMPAIGN/sweep.sh"`（绝对路径）判断是否
     已有扫描在跑，而在跑的那个是以 `./sweep.sh`（相对路径）启动的，
     匹配不上。已改为 `pgrep -f "sweep[.]sh"`，并追加一道
     "137/138 上还有 `glm52-pd` 容器就不启动新 pass"的门禁。
- **必须更正的归因**：我曾把 C64 的数据丢失写成"第二个并发 supervisor
  对同一目录执行 `rm -rf`"。**这个结论没有证据支持，是错的。**
  后续核查表明：进程表中始终只有一个真实的 `sweep.sh`（PID 610837），
  当时 `pgrep -af "sweep[.]sh"` 多出来的那一项是**同一脚本自己 fork 出的
  子 shell**（父进程即 610837），不是第二次扫描。也就是说缺陷 2 是真实的
  代码缺陷，但它造成"并发双扫描删库"的那段因果链并未发生。
- **C64 数据丢失的真实原因：证据不足，暂记为未确认。** 该点没有
  `runner.log`，无法还原客户端阶段状态。可以确定的只有它**不是容量失败**，
  因此不得按容量结论解读。同一失败签名
  （`profile_export_aiperf.json not found` + `退出码 1`）在 P8D8 C96 上
  复现并留下了完整日志，已定性为 warmup 排空超时（见问题 9）；
  C64 是否同因，需重测才能断言。
- **重测结果（已完成）**：C64 在 stdin 缺陷修复后重跑一次即成功，
  拿到完整数据 **2,059.90 tok/s/GPU**（原失败点归档为
  `p4d8/c64.attempt1-failed/`）。一次重跑即通过，说明首次失败是偶发的，
  不是该并发点的固有属性。首次失败的确切原因仍无日志可考，
  维持"未确认"不再追查——结论所需的数据已由重测提供。
- **教训**：失败归因必须以该次运行自己的日志为准。缺陷 2 在代码层面成立，
  但我把"代码有缺陷"直接当成了"缺陷导致了这次事故"，
  在没有进程证据的情况下补出了一条因果链。

### 8. 空响应（KV transfer 失败）已有验证过的修复

- **根因**：跨 rank 的 Mooncake KV transfer 失败。
- **修复**：`PD_DP_RANK_AFFINITY=1`。paired treatment 中 transfer 失败与
  profiling 阶段空响应同时归零；pinned 三点的 profiling 错误均为 0
  （C48 的 2 条、C64 的 1 条 `InvalidInferenceResultError` 全部落在 warmup 阶段）。
- **状态**：受控。该项默认开启，本次扫描沿用。

### 9. P8D8 C96：warmup 排空超时导致整点作废（已定性，待复测）

- **现象**：`profile_export_aiperf.json not found`、客户端退出码 1，
  该点没有任何 profiling 数据。
- **客户端事实链**（`p8d8/c96/bench/runner.log`）：
  - 21:03:04 `sending complete | sent=197, completed=101, in_flight=96`；
  - 此后 **30 分钟零推进**：21:18 与 21:33 两次进度日志都停在 `returned=101/197`，
    `errors=0`；
  - 21:33:04 `Accelerated warmup drain timed out after 1800.0s`，
    96 个在飞请求被取消，整轮 abort。
  - 该 1800s 来自 warmup 阶段的 `grace_period_sec`
    （`aiperf/timing/phase/runner.py` 的 `_wait_for_accelerated_warmup_handoff`）。
- **先排除一个看起来很像的错误解释**：日志里
  `BranchOrchestrator stats: spawned=172 completed=0` 很像"会话树自死锁"
  （parent 占满并发槽等 child，child 拿不到槽）。**但这是正常现象**：
  成功的 C32（`spawned=32 completed=0`）与 C48（`spawned=51 completed=0`）
  同样是 `completed=0`——子会话按设计挂起并移交给 profiling 阶段。
  因此 `completed=0` 不是故障特征。
- **真正的规律是收尾排空时间随并发急剧恶化**：

  | 并发 | 收尾在飞数 | 排空耗时 | 结果 |
  |---|---|---|---|
  | C32 | 32 | 130 s | 正常 |
  | C48 | 48 | 511 s | 正常 |
  | C96 | 96 | >1800 s | 超时作废 |

  并发从 32 到 48（1.5×），排空时间涨到 3.9×：尾部请求既变多、
  又因过饱和后 ITL 恶化而变慢，两个因素相乘。按此趋势 C96 需要的排空时间
  本就远超 1800s 预算。
- **1800s 这个数字来自我们自己的 harness**，不是 aiperf 默认值：
  `benchmark_command.txt` 里显式带着 `--warmup-grace-period 1800`。
  aiperf 侧若不指定，agentic warmup 的屏障是**无限**的
  （`_agentic_warmup_grace_period` 在无 cache-warmup-duration 时返回 `inf`）。
- **一条被我写错、现已更正的"异常"**：我曾把
  "decode 侧整轮没有调度行、`num_running_reqs` 恒为 0"当作 decode 未接手
  请求的证据。**这是错的，它是这套负载的常态。** 在随后健康运行的 C128 上
  实测：warmup 期间 decode 容器同样是 0 条调度行、1 秒间隔连采 20 次
  `num_running_reqs` 全为 0，而 decode 容器持续返回
  `POST /v1/chat/completions 200 OK`、客户端请求正常完成。
  原因是该 trace 约 99% 的 token 是 prefill 输入（见第 2 节），
  warmup 阶段 decode 占用接近零，`decode_log_interval=40` 自然打不出行。
  因此这条不构成 C96 的异常证据。
- **真正推翻"并发越高必然超时"的证据：C128 成功了。**
  `completed=270, cancelled=0, errors=0, elapsed=1914.89s`，排空正常结束。
  并发比 C96 更高的点反而通过，说明 C96 那 1800 秒**零推进**
  （21:03→21:33 始终 `returned=101`）是一次真实停滞，
  而不是"预算不足 + 尾部变慢"就能解释的。
  故 C96 不应仅通过调大 grace period 来"修"，需带运行期指标复测以定位停滞点。
- **保留的可疑线索**：21:02:53–21:03 有 **16 条** 真实 `KVTransferError`
  （`Decode transfer failed ... bootstrap_room=...`），紧接着就进入零推进；
  21:33 的另外 96 条是取消动作的产物，不是原因。
- **已采取的措施**：新增 `sample_metrics.py`，运行期直接轮询两端 `/metrics`。
  它在 C128 上立刻给出了此前 `docker logs` 看不到的瓶颈画面（见问题 10）。
- **复测结果：PASS，13,423.83 tok/s/GPU。** C96 于 01:03 带运行期采样重跑，
  02:28 完成，计分 2,203 / 2,400，profiling 阶段无错误。
  首次运行那段"30 分钟零推进"完全没有复现，首次缺失的 `runner.log` 也完整留存。
- **结论：首次 C96 失败是偶发停滞，不是该并发点的固有属性。**
  这与 C64 的情况一致——两个首次失败点在各自重跑一次后都直接通过。
  因此问题 9 里"C96 需要调大 grace period 才能测"的担心不成立；
  1800 s 预算对 C96 是够的，首次那次是真实停滞（伴随 16 条 `KVTransferError`），
  但不可复现，不阻塞结论。
- **一个值得记下的运行期观察**：本次 C96 的 kv_usage 在 warmup 期稳步爬升
  （76% → 82% → 86% → 91%），终值 0.95，与 C64 基线的 0.96 相当。
  它逼近饱和但**没有越过**——这正是 C96 仍能保持 0.9186 服务端命中率的原因
  （当时它是最高点的 94.6%；后续补测 C72/C80 抬高了 M，该占比已改写为 82.5%）。
  而 C128 的 kv_usage 顶到 1.00、命中率掉到 0.6908，于是坠崖。
  **P8D8 的悬崖边缘因此被夹在 C96 与 C128 之间。**
- **后续**：C56（12,861.91）与 C88（15,253.66）均已完成，峰值已夹住。
  最终结论见开头"结论"一节。

### 10. P8D8 C128 的瓶颈：decode 侧 KV 接近打满，传输队列积压

这是 `sample_metrics.py` 上线后拿到的第一段运行期时序，也是本次扫描里
第一次**直接看到**（而非从吞吐曲线反推）瓶颈位置。

profiling 阶段 22:34–22:35 的连续采样（每 15s，跨 rank 求和）：

| 时刻 | prefill run / queue / kv | decode run / xfer队列 / kv / gen |
|---|---|---|
| 22:34:17 | 8 / 95 / 0.017 | 12 / 105 / 0.848 / 131 |
| 22:34:32 | 8 / 98 / 0.029 | 11 / 106 / 0.854 / 150 |
| 22:34:47 | 8 / 116 / 0.019 | 12 / 111 / 0.883 / 154 |
| 22:35:02 | 8 / 120 / 0.029 | 11 / 123 / 0.939 / 158 |
| 22:35:17 | 10 / 117 / 0.062 | 12 / 123 / 0.940 / 170 |

读法：

1. **在 C128 这个点上，decode 侧 KV 确实是约束**：`token_usage` 从 0.848
   单调爬到 0.940，同时 `num_decode_transfer_queue_reqs` 从 105 涨到 123——
   有一百多个请求已完成 prefill、卡在等待 decode 侧 KV 空间来接收 KV 传输。
2. **prefill 的 `num_running_reqs` 被钉在 8**（DP8，每 rank 一个），
   队列从 95 涨到 120，而它自己的 KV 占用只有 0.02–0.06。
3. **一处需要修正的表述。** 本节原先据第 2 点写的是"prefill 不是瓶颈，
   是被反压的一方"。补齐全部并发点的采样后，这个说法只在 C128
   这类远超峰值的点上成立，作为整体结论是错的：
   在峰值及以下（P8D8 ≤ C80），decode KV 占用只有 0.38–0.56、
   传输队列中位数 4–13，根本不具备反压能力，
   而 prefill 队列已经在堆积、其 KV 仅占 0.17。
   **正确的表述是两段式的**：prefill 算力限制吞吐（第一段），
   在途请求随之增多把 decode KV 撑满（第二段），C128 处在第二段。
   两者是因果关系而非并列，详见"结论"一节的瓶颈分析。
4. **对"最小并发"的含义**：一旦 decode KV 进入 0.85+ 区间，
   继续加并发只会加长传输队列、拉高单请求延迟，不会提高 tok/s/GPU。
   因此峰值必然落在 decode KV 打满之前——实测也确实如此，
   P8D8 峰值 C80 处 decode KV 仅 0.56。

**方法论收获**：`docker logs` 在这套负载上会给出误导性的静默
（decode 无调度行 ≠ decode 空闲，见问题 9），而 `/metrics` 的
`num_decode_transfer_queue_reqs` + `token_usage` 两个量能直接定位反压方向。
后续所有点都应带运行期采样。

## 附：请求计数口径

`agentx_conc<C>.json` 中：

- `num_requests_total` 含 warmup + profiling 两个阶段的记录；
- `num_requests_successful` 仅为 profiling 窗口内成功计分的请求；
- `records_error_dropped` 单独统计 warmup 阶段被丢弃的错误记录。

因此"计分请求数 < 总请求数"是正常的，差额是 warmup 丢弃量，
不代表 profiling 阶段有失败。判断某点是否健康应看
`profiling 阶段错误数`，而不是两个总数之差。
