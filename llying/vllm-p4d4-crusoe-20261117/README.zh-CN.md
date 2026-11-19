# 单节点 P4D4：Infera + vLLM v0.34.0 + Mooncake 复现 GLM-5.2 AgentX（2026-11-17 起，crusoe v3-135 → 新节点，进行中）

对标：sglang 在本集群单节点 P4D4 C104 53,351 tok/s/GPU（`dev/pd_opt/glm_5.2_agentx` 分支 `llying/pd-repro-crusoe-20261116/`）。
方案见 `../vllm-pd-plan-20261117/PLAN.zh-CN.md` 5.3 节；脚本见 `../vllm-pd-tools-20261117/`。

## 0. 当前结论（11-19 07:45 更新；细节见 1l–1t）

- **（07:45）f1 / f2（m14 配置 + decode 显存目标 0.93，3600 s，两台各一次）都有效、无崩溃 / 卡死 / KFD 驱逐告警，但没有超过 m10**：f1 48,173（v3-006），
  f2 44,691（v3-124）。同节点 f1 比 m10 低 0.8%；v3-124 上 f2 的前 1800 s 比 m14（显存目标 0.95）低 8%，显存目标 0.93 少的约 4% KV 可能抵消了 DP 均衡的收益（单次，1t）。
  3600 s 有效成绩仍以 m10 = 48,572 最高。

- **（06:15）m14 = 49,183 tok/s/GPU（1800 s，v3-124，单次）**：m10 配置 + decode 用 vLLM 内部 DP 负载均衡（`INFERA_VLLM_DP_ENGINE_LB=1`）。
  TTFT p90 8.61 s（m10 10.97）、交互 p90 55.7（48.5）、ITL p90 17.9 ms（20.6）、3.89 req/s（3.30），尾部改善最明显（1s）。
  m13（路由请求权重 10 万）decode 崩溃，无效。f1 / f2 = m14 配置 + decode 显存目标 0.93，3600 s，两台各一次，进行中。
- **（06:15 更正）m9b / m12 / m13 的崩溃和卡死更可能是 decode 显存余量不够**（推断）：m13 与 m9b 同一崩溃签名，崩溃前也有 m12 式的变慢；
  1r 里"疑为精确锁页"、1p 里"`--max-num-batched-tokens 1024` 不安全"的归因都不成立。f1 / f2 用显存目标 0.93 验证（1s）。
- **P8D8 第一次尝试 smoke 失败**：跨节点 Mooncake 建连失败（通知 QP -302），下次试 `MC_RDMA_NOTIFY_ENABLED=0`（1s）。
- **机器规则（11-19 起）**：不再用 hold 作业占机器；f1 / f2 结束后两台节点自动释放，之后每次计分提交"跑完就退出"的作业。
- **（11-18 22:00）重复轮 m12 无效**：测量第 12–14 min decode 逐渐变慢直到卡死（引擎 RPC 超时），同时主机出现 amdgpu `evict_process_worker`（KFD 驱逐进程的 GPU 队列）告警
  （1r；当时以为只有这一轮有，后来 m13 期间同一节点上计数又从 64 涨到 128）。精确锁页已改为直接 `hipHostAlloc`（与 torch 锁页同一机制，只是不取整），之后各轮用新版本。m10 仍是唯一的 3600 s 有效成绩。
- **最好的有效计分（与 sglang 同口径 3600 s）：m10 = 48,572 tok/s/GPU，sglang 53,351 的 91%**（v3-006，单次；重复轮 m12 卡死无效，见 1r）。配置：
  s13 + decode #47897 覆盖层 v3 + aiter AG/RS 修正（A 线 v3comm）+ prefill / decode 各 `--renderer-num-workers 4` + prefill GEM 0.88（VMM 导入库）
  + CPU 层 1.9 TB（精确锁页覆盖层 `overlay-cpupin/`，U26）。TTFT p50 / 均值 2.88 / 5.20 s 已好于 sglang（4.20 / 5.65），差距在 decode
  （交互 59.4 对 69.9、ITL 16.9 对 14.3 ms）：decode KV 满、DP rank 之间不均（1q）。
- 之前：m6（上面去掉精确锁页 / 1.9 TB，CPU 层 1.25 TB）1500 s 43,555，稳态 48,624。
- **口径提醒**：每轮测量期开头 5 min 都只有约 22K（测量方式造成的瞬态），1800 s / 1500 s 的整窗成绩因此低估了 3600 s 同口径的数；
  同配置两台节点之间差约 10%（m1 对 s13，差在 decode），单次运行里小于约 10% 的差别不当结论。
- **现在的瓶颈**：decode（1q）。prefill 现算已降到 ISL 的 3.92%（比 sglang 的 16.6K tok/s 还少），渲染排队已用 renderer 线程解决；
  decode 对速度 / KV 容量很敏感（慢 8% 就让 decode 排队涨到 2.6 倍），decode `--max-num-batched-tokens 1024` 会崩（1p），下一步试 decode DP 均衡（m14）。

### 0.1 15:40 的结论（保留）

- **最好的有效计分：s13 = 42,724 tok/s/GPU（sglang 53,351 的 80%；C104，1800 s，11-18 11:07–11:37，v3-135，单次）**：
  s10（Rust 路由 + vLLM 串行，单 API server，1.25 TB CPU 层）+ decode #47897 覆盖层 v2（组大小 4）+ P / D 都设 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`；
  prefill 不用 GEM（`queue/s13.sh`）。

  | 轮 | tok/s/GPU | req/s | 完成 / 丢弃 | 交互 p50 / p90 | ITL p50 / p90（ms） | TTFT p50 / p90 / 均值（s） |
  |---|---|---|---|---|---|---|
  | s9（Python 路由 round-robin） | 36,923 | 3.001 | 5,489 / 3 | 52.2 / 42.0 | 19.2 / 23.8 | 6.79 / 31.77 / 12.37 |
  | s10（Rust 路由串行，kv-aware） | 36,265 | 2.952 | 5,401 / 6 | 49.9 / 35.8 | 20.1 / 27.9 | 6.02 / 23.19 / 11.27 |
  | **s13（s10 + decode #47897 + keep-alive）** | **42,724** | **3.391** | 6,202 / 1 | 57.9 / 52.3 | 17.3 / 19.1 | 5.62 / 18.95 / 9.16 |
  | sglang P4D4 C104（3600 s） | 53,351 | 3.469 | 12,582 / 1 | 69.9 / 60.8 | 约 14 | 4.20 / 9.23 / 5.65 |

  s13 比 s10 高 17.8%，ITL p50 低 14%。A 线在聚合模式（随机 token）下测得 #47897 每步只快 2.7–3.9%；AgentX 下各 DP rank 上下文不均，
  #47897 缩短的是最重 rank 的 indexer（关键路径，KF README 4.1 估每步 5–6%），收益可能被放大；keep-alive 只消除 502，不应改变吞吐。
  但 17.8% 里有多少是跑间波动还不知道（各轮都只跑了一次），新节点上先原样复跑 s13（队列项 `m1-s13rerun`）。
- **瓶颈在 decode**：decode KV 满（每 rank p50 0.85–0.90）、每步慢（ITL 17–19 ms 对 sglang 约 14 ms）；prefill 命中接近理论上限（现算约 5%），
  路由、KV 传输（0.19 s / 次）都不是瓶颈（1k）。
- **A 线（decode kernel / 通信）合并**：推荐 decode 用默认配置 + #47897 覆盖层 v3（组大小来自 indexer metadata）+ aiter 自定义 AG/RS 在 FULL 捕获时放行
  （U12 / U13）；聚合模式下与原版同时跑，纯 decode 每步 104 请求 −6.3%、64 / 32 请求 −5.3%，门禁 196/200。EP、MoRI 都不比默认快，不用。
  P4D4 端到端待测（`m2`：s13 + v3comm）。
- **B 线（prefill GEM / 多级 offload）合并**：
  - 1b 的 GEM + offload 故障**已解决**（U2：ROCm clr 导入外部内存不设 `deviceId`，批量拷贝跑到 device 0；GEM 分配器改用 HIP VMM 导入）。
    B 线 b1-r1（s7b 配置 + prefill GEM 0.88，v3-258）= 33,203（比 s7b +10.3%，一次）。
  - 1e 的 `TieringOffloadingSpec` PP 问题**已修**（U11）；但 InferenceX AgentX 只允许 CPU DRAM 级 offload，NVMe / FS 层不能计分，只作诊断（3600 s 34,350）。
  - vLLM 0.34.0 唯一的写回式 offload（`SimpleCPUOffloadConnector` lazy）在 P4D4 上更差（3600 s 28,696，CPU 层只找到 12% 的 GPU 未命中，U22）；
    1.25 TB 写穿在 1800 s 内命中已接近 trace 理论上限，容量暂不是瓶颈（U9）。
  - **主机侧崩溃（U21）**：B 线 7 轮里 3 轮在预热第 15–23 min 崩在 tokenizers / libtorch_cpu（prefill、decode、路由三个不同进程，一次指针被 ASCII 覆盖），
    倾向于 HF tokenizers 0.23.3 并发下的堆破坏。本线 s12 / s12b / s12d（prefill GEM 0.88）三次 prefill 崩溃可能同因；在查清之前组合配置不用 prefill GEM。
- **节点**：v3-135（294199）11-18 到期。新作业 296460 拿到的 v3-106 上 docker 坏了：NVMe 卷在 dockerd 启动后才挂到 `/var/lib/containerd`（→ `/mnt/m2m_nobackup/containerd`）上，
  把 containerd 的数据目录遮住，`images/json` 正常但拉取、构建、起新容器全部失败（`mkdir .../io.containerd.content.v1.content/ingest/...: no such file or directory`）。
  已排除该节点、释放作业，重新申请（hold 脚本的启动检查加了真实的 `docker pull` + `docker run`）。之后各轮用 `scripts/p4d4_main.sh`（参数化的 s13 模板，
  自动取节点 IP，默认每个容器开 `PYTHONFAULTHANDLER=1`、每 60 s 记录各容器进程列表、每轮前后抓内核日志和 `Shmem`），队列项名 `m<序号>-…`。

### 0.9 早期结论（11-17 – 11-18 上午，保留原文；其中"目前最好"等说法已被上面取代）

- **（已过时）目前最好的有效计分：s9 = 36,923 tok/s/GPU（C104，1800 s，2026-11-18 05:44–06:14，1.25 TB CPU 层，Python 路由 round-robin）**，为 sglang 53,351 的 69%（1j）。
  此前 s8 31,384、s7b 30,091；瓶颈曾是 Infera Python 路由的 tokenize（1i）。
  之前的 s4 = 26,541（1 TB）。
  | | vLLM s4 | sglang P4D4 C104 |
  |---|---|---|
  | tok/s/GPU | 26,541 | 53,351 |
  | 输出 tok/s/GPU / req/s | 245.5 / 2.272 | — / 3.469 |
  | 完成 / 丢弃 | 4,146 / 3（1800 s） | 12,582 / 1（3600 s） |
  | 交互 p50 / p90 | 62.6 / 49.7 | 69.9 / 60.8 |
  | ITL p50 / p90（ms） | 16.0 / 20.1 | — |
  | TTFT p50 / p90 / 均值（s） | 22.5 / 79.2 / 33.8 | 4.20 / 9.23 / 5.65 |
  | prefill 命中（采样器） | 87.5% | 96% |
  差距几乎全在 prefill：TTFT 高 5 倍多，decode 的 ITL / 交互度接近 sglang，decode KV 多数时间不到一半。详见各轮。
  （s4 的 AIPerf 回放本身通过了覆盖率检查；InferenceX 聚合因缺 `KV_OFFLOAD_BACKEND_METADATA` 失败，事后用同一份产物重跑聚合得到上面的数，
  `../vllm-pd-tools-20261117/scripts/agentx_post.sh`。）

- **PD 链路跑通，门禁干净（18:12）**：Infera rust router（与 sglang recipe 相同的参数）→ vLLM prefill TP1×PP4（GPU 0-3）→
  Mooncake RDMA 同节点回环（GEM KV）→ vLLM decode DP4（GPU 4-7，MoE 平铺 TP4）。smoke 6/6、GSM8K 196/200（乱码 0、开头坏 0）、
  16k 捞针 32/32 + 长前缀 31/32、真实接受长度 3.46。
- **GEM KV 在 vLLM 上可用**：每卡 150–163 GB 的 KV 由 `infera.engine.vllm.gem_mooncake`（`get_mem_pool_context()` 返回 GEM 内存池）分配，
  Mooncake 以 dma-buf 分 1 GiB 块注册，注册后没有 sglang 未修时那种"假 OOM"。
- **s1（prefill 只有设备缓存、FCFS）在预热中就颠簸，19:05 中止**：预热 36.5 min 只返回 498 / 1,155 个请求（约 8 个 / min，
  sglang 稳态约 208 个 / min），prefill 前缀命中 5.2%（命中 410 万 / 查询 7,841 万 token，平均每个请求现算约 16 万 token），
  prefill 一直满载（约 3.3 万 tok/s 现算，100 个请求因 "capacity" 排队），KV 池用量 2–4%。机制见第 2 节。
- **s2**：prefill 加 1 TB CPU KV 层（OffloadingConnector，对应 sglang 每 rank 250 GB 的 HiCache）+ 缓存感知准入
  （`--cache-aware-admission-window 128 --cache-aware-admission-threshold 0`，前缀已在缓存里的请求先进，接近 sglang 的 SPF），19:06 起。

- **s2（GEM KV + CPU 层 + 缓存感知准入）**：prefill 在第一次 offload 写出（66k 冷请求的第一个完整块）时 GPU 访存故障（1b 节）。
- **s2b（prefill 不用 GEM、显存目标 0.6 + CPU 层 + 缓存感知准入；decode GEM）**：smoke 6/6、offload 正常；预热 17 min 后 **PD 死锁**
  （1c 节），20:17 中止。
- **s2c（同 s2b，但 prefill 回到 FCFS；decode 显存目标 0.95）**：预热约 25 min 后自行走出过载（prefill 排队从 ~130 降到个位数）；
  测量 20:57 起，**21:32 prefill worker 静默卡住，21:36 引擎因 `RPC call to sample_tokens timed out` 退出**，此后请求全部 500，
  AIPerf 以"测量期指标覆盖率 < 95%（TTFT 77.9%）"拒收结果。卡住前的稳态（20:57–21:32，采样器）：

  | | vLLM s2c | sglang P4D4 C104 |
  |---|---|---|
  | prefill 输入（含已缓存） | 约 20.7 万 tok/s | 约 42.4 万 tok/s |
  | prefill 现算 | 约 2.3 万 tok/s | 1.66 万 tok/s |
  | 命中（本地 / CPU 层） | 38.9% / 50.0%，合计 89% | 96% |
  | prefill 有排队的时间 | 62%（排队 p50 7、p90 109） | 66% |
  | AIPerf 请求吞吐（含卡住后） | 1.32 req/s | 3.47 req/s |
  | 每卡网卡流量 | 4 TB / h（每轮都传完整 KV） | — |

  prefill 现算不比 sglang 慢，差距来自命中率：受 prefill 限时，输入吞吐 ≈ 现算 /（1 − 命中），89% → 96% 约翻倍。
  卡住的原因未定：卡住前 worker 没有任何报错；最后一级 PP worker 停在 gloo `irecv` 等上一级。最可疑的是某次 offload 的 `hipMemcpyBatchAsync`
  （vLLM 注释记录它在 ROCm 7.15 上描述符过多会出错，本镜像是 ROCm SDK 10.0），未证实。
- **s3（CPU 层 2 TB）起不来**：KV 初始化（锁定主机内存）6 min 后一个 PP worker 被 SIGKILL（退出码 -9）。（更正）ROCm 上 `CPUOffloadingSpec`
  不用 `/dev/shm` 共享区域（`_uses_shared_region()` 在 ROCm 上为 False），CPU 层是每个 rank 自己的锁页主机内存；2 TB 锁页加上其余进程
  接近 2.75 TB 物理内存，是主机 OOM。1.25 TB 时（s5）主机还剩约 1.2 TB 可用，实际上限在 1.3–1.5 TB 之间。
- **s4（1 TB CPU 层 + Python router 串行下发 + 缓存感知准入 128 / 0 + offload 批量拷贝描述符上限 512 + decode 单个 API server，
  测量 1800 s）**：22:17 起；另挂看门狗（prefill 停 180 s 就用 py-spy 抓全部栈）。预热比 s2c 快（1,200 s 时返回 779 对 453，测量提前约 90 s 开始）。
  测量前 11 min：prefill 输入 18.7 万 tok/s、现算 2.3 万 tok/s、命中 87.5%（本地 60.3% + CPU 层 27.2%）；decode KV p50 0.20 / p90 0.48、
  在跑合计 p50 26（最忙 rank 8）——**瓶颈完全在 prefill**，decode 大量空闲。差距的结构性原因见 U9（写穿 vs 写回）。
- **s5（最终，3600 s）**：s4 + CPU 层 1.25 TB（失败则退回 1 TB）+ decode 测量中段采 3 s trace。

## 1t. f1 / f2：m14 配置 + decode 显存目标 0.93，3600 s，两台各一次（11-19 06:31–07:31）

- 配置：m10 + decode `INFERA_VLLM_DP_ENGINE_LB=1` + decode `--gpu-memory-utilization 0.93`（每 rank KV 2,827,968 token，0.95 时 2,955,712）+ 精确锁页改用 `hipHostAlloc`
  （每 worker 474.5–474.9 GB，`Shmem` 1,805 GB）。结果和分析在 `results/f1/`、`results/f2/`（节点释放前在节点上算完）。

  | | f1（v3-006） | f2（v3-124） | m10（v3-006，0.95，无 DP 均衡） | m14（v3-124，0.95，DP 均衡，1800 s） |
  |---|---|---|---|---|
  | tok/s/GPU（3600 s） | 48,173 | 44,691 | 48,572 | — |
  | 前 1800 s | 46,867 | 45,122 | 46,083 | 49,183 |
  | 稳态（300 s 起） | 50,394 | 46,587 | 50,881 | — |
  | req/s / 完成 / 丢弃 | 3.258 / 11,816 / 8 | 3.083 / 11,185 / 2 | 3.297 / 11,951 / 8 | 3.893 / 7,109 / 4 |
  | 交互 p50 / p90 | 60.9 / 53.9 | 60.6 / 48.1 | 59.4 / 48.5 | 60.3 / 55.7 |
  | ITL p50 / p90（ms） | 16.4 / 18.6 | 16.5 / 20.8 | 16.9 / 20.6 | 16.6 / 17.9 |
  | TTFT p50 / p90 / 均值（s） | 3.17 / 13.00 / 5.88 | 3.36 / 15.84 / 6.95 | 2.88 / 10.97 / 5.20 | 2.74 / 8.61 / 4.33 |

- **稳定性**：两轮都没有 decode 崩溃、卡死检测、`evict_process_worker` 告警（m9b / m12 / m13 的那类故障）。两次不足以断定 0.93 比 0.95 稳。
- **吞吐**：没有超过 m10。同节点 f1 比 m10 低 0.8%（前 1800 s 高 1.7%）；v3-124 上 f2 前 1800 s 比 m14 低 8%。decode 仍受 KV 容量限制，0.93 少的约 4% KV
  可能抵消了 DP 均衡的收益；两台同配置差 7%（节点 / 跑间波动）。都是单次，结论待更多数据。
- 两轮结束后节点按规则释放（07:35–07:37）。

## 1s. m13 崩溃、m14 = 49,183（decode 内部 DP 均衡）、P8D8 跨节点建连失败（11-18 夜）

- **m14 = m10 配置 + `INFERA_VLLM_DP_ENGINE_LB=1`（路由不再指定 decode rank，由 vLLM 按各 rank 实时的排队和在跑数选），C104 1800 s，v3-124（第一次用这台），单次**：

  | | m14（1800 s） | m10（3600 s） | m10 前 1800 s |
  |---|---|---|---|
  | tok/s/GPU | **49,183** | 48,572 | 46,083 |
  | req/s | 3.893 | 3.297 | — |
  | 完成 / 丢弃 | 7,109 / 4 | 11,951 / 8 | — |
  | 交互 p50 / p90 | 60.3 / **55.7** | 59.4 / 48.5 | — |
  | ITL p50 / p90（ms） | 16.6 / **17.9** | 16.9 / 20.6 | — |
  | TTFT p50 / p90 / 均值（s） | 2.74 / **8.61** / 4.33 | 2.88 / 10.97 / 5.20 | — |

  p90 指标全面改善，符合"请求不再排在 KV 满的 rank 上、每步不再被最忙的 rank 拖住"的预期。但这是另一台节点上的 1800 s 单次（节点间差约 10%），
  要靠 f1 / f2（3600 s，两台）确认。
- **m13 = m10 配置 + 路由 `INFERA_DECODE_REQUEST_WEIGHT=100000`，1500 s，v3-006：42,159，完成 / 丢弃 4,543 / 511，无效**。22:33:23 一个 decode worker 在
  hipBLASLt（`m 2624 n 1694 k 6144`）上 `HIPBLAS_STATUS_INTERNAL_ERROR`，随后 `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / "too many resources requested for launch"，进程中止，
  EngineCore 退出。之前 22:29:41–22:30:21 生成速度掉到 30–67 tok/s 后恢复（m12 卡死前也是这样）。同节点内核日志的 `evict_process_worker` 计数在 m13 期间从 64 涨到 128。
- **更正 1p 和 1r 的归因（推断）**：m13 的崩溃签名与 m9b（decode `--max-num-batched-tokens 1024`）完全相同，而 m13 用的是默认 8192，所以 m9b 的崩溃不是 1024 造成的。
  三次故障（m9b 崩溃、m12 卡死、m13 崩溃）都发生在 decode 走 prefill 路径（抢占重算的混合批次）时，更像是 **decode 显存余量不够**：显存目标 0.95 只给 torch 池外留约 14 GB，
  scratch / hipBLASLt 等临时资源分配失败就崩，分配引发的驱逐就表现为变慢或卡死。用 v3comm（多了 aiter AG/RS 缓冲区）的 5 轮里坏了 3 轮（m9b、m12、m13），m6、m10 跑完。
  精确锁页大概率不是原因（改成 `hipHostAlloc` 也没有坏处）。f1 / f2 用 decode 显存目标 0.93（KV 约少 4%）验证。
- **P8D8 第一次尝试（`p8d8_run.sh`，n3 prefill PP8 + n4 decode DP8，C208）：smoke 失败**。chat / completion / stream 都正常，60K token 的冷长前缀请求在
  decode 拉 KV 时失败（`Mooncake transfer engine returned -1` → HTTP 500）。prefill 侧 `rdma_endpoint.cpp:159 Notification QP initialization failed: -302` 1,272 次，
  decode 侧 314 次"Re-establish connection … waiting ready ACK"后放弃；同节点回环的 m14 里这两类错误都是 0。
  - 单进程主机内存跨节点测试（`scripts/mc_xnode.sh` / `mc_xnode.py`，n3 写 n4，与 vLLM 方向相同）：单网卡默认设置正常（30 GB/s，内容校验通过）；
    关掉通知 QP（`MC_RDMA_NOTIFY_ENABLED=0`）也正常（29.8 GB/s）；8 张网卡同时开时正常但只有约 4 GB/s（待查，可能是主机内存所在 NUMA 与网卡的对应）。
  - 所以失败和 P8D8 的规模有关（8×8 worker 全互连、每对跨多张网卡），推断是通知 QP 的资源不够。vLLM 的连接器用自己的 ZMQ 通道通知完成，不需要 Mooncake 的通知 QP，
    下次 P8D8 两边加 `MC_RDMA_NOTIFY_ENABLED=0`。

## 1r. m12（m10 原样重复）测量第 12–14 min decode 卡死；精确锁页改用 `hipHostAlloc`

（11-19 更正：下面"触发源"里的第 1 条大概率不成立，见 1s；更可能是第 2 条 decode 显存余量。）

- **m12 = m10 原样（同节点 v3-006，3600 s）**，AIPerf 21:43 中止：26,502 tok/s/GPU，完成 / 丢弃 2,795 / 320，无效。卡死前的指标和 m10 同一水平
  （交互 p50 57.1、ITL p50 17.5 ms、TTFT p50 3.10 s）。测量 20:58:57 开始，前两个 5 min 窗口 23,179 / 46,790，与 m10 的 22,977 / 45,975 相差不到 2%
  （同节点同配置的前段可复现）。
- **经过**（`vpd-decode.stream.log`、`samples.jsonl`、内核日志；`scripts/decode_pressure.py`）：
  - 21:04–21:11 decode 各 rank KV 89–100%、排队 0–10，累计抢占 3 → 9 次。这和 m10 一样（m10 一小时抢占 22 次，KV 也经常 95–100%），不是 m12 独有的。
  - 21:11:24 四个 rank 的生成速度同时从 700–1,600 tok/s 掉到 210–375，之后反复；21:12:24 起持续下滑（DP0 39 → 13 tok/s），最后一步约在 21:13:23。
    21:12:17 DP0 在推理中 JIT 编译 `_localize_aiter_prefill_topk_kernel`（decode 走了 prefill 路径，应是抢占后重算）。
  - 21:14:23 起四个 EngineCore 每 60 s 报 `No available shared memory broadcast block found`（worker 不返回）；21:18:20 DP0 / DP2 / DP3 的 `execute_model`
    RPC 超时（300 s），EngineCore 退出，之后 220 次 "decode unreachable"。prefill 从 21:12:39 起空闲（在跑 0、排队 0：并发都卡在 decode 上），
    decode 退出后 prefill 积压 176 个、卡住检测器触发（py-spy 只抓到 API server，EngineCore 进程改了进程名没被 `pgrep -f python` 匹配到）。
  - 内核日志：`workqueue: evict_process_worker [amdgpu] hogged CPU for >10000us`，按第 4 / 8 / 16 / 32 / 64 次打印，时间是 21:13:23、21:17:56、21:17:58、
    21:19:22、21:42:58。也就是说**变慢期间就在反复做 KFD 进程驱逐**（停掉一个进程的 GPU 队列、搬走它的显存对象，再恢复）。m10 以及用 torch 锁页、抓了内核日志的
    s13 / m6 / m9b 都没有这条告警。
- **判断（推断，只出现一次）**：某个进程（从现象看是 decode）的 GPU 队列被反复驱逐、恢复，导致越来越慢直到不动。触发源没定：
  1. 精确锁页的第一版是"匿名 `MAP_PRIVATE` mmap + `hipHostRegister`"，1.9 TB 注册的用户内存是本轮与早先各轮相比唯一新的驱动层因素；
  2. decode 侧显存压力（KV 满时抢占重算的临时分配、aiter AG/RS 缓冲区）。
  一次无法区分。
- **处理**：精确锁页改为直接调用 `hipHostAlloc` 分配精确大小（torch 锁页底层就是这个分配，只是去掉了缓存分配器按 2 的幂取整）。单独测试：2 × 20 GB 只让
  `Shmem` 增加 37 GiB、`is_pinned()` 为真、GPU 往返正确、分配 6.3 s（torch 14.4 s）（`exact_pin_test2.py`，覆盖层模块本身用 `exact_pin_test3.py` 测过）。
  `overlay-cpupin/` 和 U26 草稿已更新。已经在跑的 m13 / m14 仍是第一版；P8D8 和之后的轮用新版。如果新版也出现驱逐告警或卡死，就转查 decode 侧
  （降 decode 显存目标或关 aiter AG/RS 对比）。

## 1q. m10：3600 s 同口径 48,572 tok/s/GPU（sglang 的 91%）；prefill 已不是瓶颈，瓶颈在 decode 的 KV 与 rank 间不均

- **m10 = m6 配置 + 精确锁页覆盖层 + CPU 层 1.9 TB，C104 3600 s，v3-006（agentx 节点），单次**（`results/m10/`）：

  | | m10 | sglang P4D4 C104（3600 s） |
  |---|---|---|
  | tok/s/GPU | **48,572（91%）** | 53,351 |
  | 输出 tok/s/GPU / req/s | 405.8 / 3.297 | 435.7 / 3.469 |
  | 完成 / 丢弃 | 11,951 / 8 | 12,582 / 1 |
  | 交互 p50 / p90 | 59.4 / 48.5 | 69.9 / 60.8 |
  | ITL p50 / p90（ms） | 16.9 / 20.6 | 14.3 / 16.4 |
  | TTFT p50 / p90 / 均值（s） | **2.88** / 10.97 / **5.20** | 4.20 / 9.23 / 5.65 |
  | prefill 现算（tok/s）/ 占 ISL | **15,170** / 3.92% | 16,631 / 3.9% |

  每 5 min：0–300 22,977；300–1800 为 46.0–57.8K；1800–2100 57,140；2100–3000 掉到 43.1–52.8K（这段平均 ISL 150–158K，会话变长）；3000–3600 54–56K。
  稳态（300 起）50,881；前 1800 s 46,083（可与 1800 s 的 s13 42,724 / m6 比）。
- **prefill 基本到位**：现算 3.92%（m6 6.66%），是"按会话理想值"的 0.56 倍、接近 AIPerf 理论值对应的 3.55%；长会话整段丢失多算的量从 m6 的 928 万降到很少
  （续轮多算合计 321 万 token，只占现算的 5.8%）。prefill 排队非空 45%（sglang 61%），prefill 引擎排队 1.20 s、TTFT 2.24 s，渲染 0.19 s。
- **瓶颈在 decode**：decode 引擎排队 2.24 s（m6 1.49 s），KV 用量 p50 0.87、p90 0.98。DP rank 之间不均（`scripts/dp_balance.py`）：最忙的 rank
  在跑请求数是平均的 1.28 倍（p90 1.52 倍）；**60% 的时间里总有一个 rank 的 KV 超过 95%，而最空的 rank 中位 78%**。路由在 prefill 之前按"已登记的输入 token"
  选 decode rank，不跟踪输出增长和请求结束，于是请求在满的 rank 上排队、每步又被最忙的 rank 拖住。下一步 m14：decode 让 vLLM 内部 DP 负载均衡选 rank
  （`INFERA_VLLM_DP_ENGINE_LB=1`，按各 rank 实时的等待 × 4 + 在跑数，自然避开 KV 满的 rank）；路由的 `INFERA_DECODE_REQUEST_WEIGHT` 也可以试。
- m12 = m10 原样重复（3600 s）：测量第 12–14 min decode 卡死，无效（1r）。

## 1p. decode `--max-num-batched-tokens 1024` 不安全（m9b，v3-190，burst）

（11-19 更正：m13 用默认 8192 也出现了完全相同的崩溃，所以崩溃不是 1024 造成的，更可能是 decode 显存余量不够，见 1s。1024 本身是否安全仍未验证。）

- 目的：decode 的激活 profiling 按 8192 token 预留约 6.7 GiB，而 decode 每步多数只有约 100 token，改成 1024 后每 rank KV 从 2,955,712 增加到
  3,134,784 token（+6.1%）。
- 结果：测量第 12 min（19:37）decode DP0 worker 崩溃：一个 **885 token** 的批次上 hipBLASLt `HIPBLAS_STATUS_INTERNAL_ERROR`，随后
  `HSA_STATUS_ERROR_OUT_OF_RESOURCES`；之后 120 个请求 decode 502、65 个空内容，AIPerf 因失败 10.0% 中止（23,327，1,639 / 190，无效）。
- 原因：PD 的 decode 每步除了 MTP 验证（约 100 token），还要为新到的请求补算 prompt 末尾未整块传来的部分，几个请求叠在一起就是几百 token；
  按 1024 预留的激活显存不够 hipBLASLt 在这类批次上的工作区。不用这个改法（单次，但原因明确）；扩大 decode KV 要另找办法（例如只把显存目标提高一点、
  并保留足够的非 torch 余量，需要验证）。

## 1o. CPU 层加大：1.6 TB 时主机 OOM（m8b），原因是 torch 锁页分配器按 2 的幂取整；改成精确锁页（m10）

- **m8b（m6 配置 + `cpu_bytes_to_use` 1.6 TB，v3-006）在部署后第 14 min 被 OOM killer 杀掉一个 PP worker**（18:49:43，`shmem-rss` 约 682 GB）：
  `MemAvailable` 从 2.27 TB 一路降到 13 GB，4 个 PP worker 合计持有 2,481 GiB 共享内存（1.25 TB 时 1,360 GiB）。部署前 `Shmem` 只有 8 GB，不是泄漏。
  OOM 后残留的约 1.3 TB `Shmem` 在 2.5 min 内逐步释放到 8 GB（**更正 U23 的推断**：这次不是永久泄漏，只是 KFD / 进程清理慢；B 线那次 5 min 不变，
  可能是另一种情况）。紧跟着的 m8c 被部署前的 `Shmem` 检查拦下（看到 1,975 GB），没有叠加 OOM。
- **原因**：ROCm 上 `CPUOffloadingSpec` 不用共享 mmap 区域，每个 KV 张量都是 `torch.zeros((num_chunks, page_bytes), pin_memory=True)`，
  torch 的锁页缓存分配器把每次分配取整到 2 的幂。每个 PP worker 24–27 个张量，1.25 TB 时单个约 15.6 GB → 16 GiB（实际 +17%），
  1.6 TB 时超过 16 GiB → 32 GiB（几乎翻倍）。最小测试（`scripts/exact_pin_test.py`，2 × 20 GB）：torch 锁页 `Shmem` +64 GiB；匿名 mmap +
  vLLM 自带的 `host_register`（`hipHostRegister`）+37 GiB，`is_pinned()` 为真，GPU 往返正确，耗时相同。
- **修法（覆盖层 `overlay-cpupin/`，上游候选 U26）**：私有张量改为精确大小的匿名 mmap，按 ≤ 64 GiB 分块注册，worker 关闭时注销；
  注册失败时回落原路径；`VLLM_KV_OFFLOAD_EXACT_PIN=0` 关闭。m10 中 1.9 TB 配置实际每 worker 474.5–474.9 GB、合计 1.90 TB（计入 `AnonPages`），
  `MemAvailable` 稳定在 809 GiB。CPU 层约 3,800 万 token（1.25 TB 时约 2,500 万），超过 sglang 写回式的约 3,200 万有效容量。
  另加 `scripts/memguard.sh`：`MemAvailable` 低于 120 GB 时优雅停止引擎，不让 OOM killer 动手。
  （22:00 补充）重复轮 m12 卡死时出现 KFD 进程驱逐告警（1r），覆盖层已改为直接 `hipHostAlloc` 精确大小分配，不再用"mmap + 注册"；上面的数据是第一版的。

## 1n. m6：v3comm + 两边 renderer 4 + prefill GEM 0.88（v3-162，burst，测 1500 s，跑完）；逐请求 prefill 命中

- **m6 = 43,555 tok/s/GPU**（1500 s，5,380 / 9，TTFT p50 / p90 / 均值 3.57 / 18.58 / 7.51 s，交互 58.5 / 51.5，ITL 17.1 / 19.4 ms；单次；
  `results/m6/`）。**prefill GEM 0.88 没有崩溃**（全程开 faulthandler、内核日志无 segfault / OOM；一次）。1500 s 的整窗受开头 5 min 瞬态的影响比 1800 s 大
  （1/5 对 1/6），按窗口比较：

  | 每 5 min tok/s/GPU | 0–300 | 300–600 | 600–900 | 900–1200 | 1200–1500 | 稳态（300 起） |
  |---|---|---|---|---|---|---|
  | s13（v3-135） | 22,458 | 40,155 | 47,044 | 47,683 | 50,940 | 46,699（到 1800） |
  | m1（v3-069，s13 配置） | 22,740 | 38,412 | 44,508 | 43,436 | 39,263 | 41,719（到 1800） |
  | **m6（v3-162）** | 22,806 | 43,166 | **50,560** | **50,431** | **50,974** | **48,624** |

  第 10–25 min 稳定在 50.4–51.0K（sglang 53,351 的 95%）。三项改动在各段的效果（测量期第 5 min 起，与 s13 / m1 比）：
  prefill 渲染 0.21 s（m1 同阶段 1.35 s）；prefill 本地命中 73.6%（s13 26%）、CPU 层 19.9%，排队非空 62%（s13 81%），prefill 引擎排队 3.89 s（s13 5.25 s）；
  decode 生成 3,332 tok/s（s13 3,426、m1 3,142），decode 排队 1.49–1.69 s。各项是一起加的，单项贡献只能按上面的分段指标判断。
- m5（v3comm + renderer 4，不开 GEM，v3-075）测量第 7 min 时 burst 作业被抢占，作废；m9（m6 + decode `--max-num-batched-tokens 1024`）部署中被抢占。
- **更正 1l 节的判断**：逐请求数据（`results/m6/prefill_usage.txt`，路由记录的每个 prefill 的 cached token）显示，子 agent 第一轮现算 50.3%，
  而回放感知的可共享前缀是 49%（`scripts/subagent_prefix.py`：只和本次运行里更早发出过的请求比，token 级公共前缀），即**子 agent 第一轮已接近理想**；
  中位命中为 0 的多是约 2,965 token 的 `:aux:` 单轮辅助请求。1l 节"冷请求只共享到约 15%、缺口在跨会话共享"是用 s13 汇总数推出来的，不成立。
- **真正多算的是长上下文会话在长间隔后整段丢失**：续轮请求里 59 个（每个多算 > 32K token）合计多算 928 万 token，占测量期全部现算（3,540 万）的 26%。
  例：会话 208d6e…（73–75 万 token）间隔 37–102 s 的轮次全部命中，间隔 593 s、323 s 的两轮命中为 0；会话 1f5f…（33–39 万）间隔 8–76 s 命中，
  间隔 293 s、336 s 命中为 0（`scripts/conv_timeline.py`）。vLLM 0.34 的 CPU 层 LRU 按"请求先后 + 块位置"排序（同一请求内后面的块先淘汰，
  GPU 命中的前缀在写入路径上会被重新访问），所以这不是前缀先于后缀被淘汰，而是容量：一个会话停顿 5 min，期间 prefill 处理约 1,000 个请求、
  新算约 700 万 token，整个会话变成最旧的请求被整段淘汰。sglang 写回式有效容量约 3,200 万 token，我们写穿约 2,500–2,900 万。
  因此 m8 起 CPU 层加到 1.6 TB（实际锁页约 1.87 TB，仍在 InferenceX 每台 ≤ 3 TB 内）。

## 1m. m1：s13 原样复跑（v3-069，burst 作业，测 3600 s，30.8 min 时被抢占）

- 节点 v3-069（burst 作业 296788，15:46 起，16:54 被抢占，运行 1 h 08 min）。配置与 s13 完全相同（`results/m1/engine-config.env`；AIPerf 参数
  除了时长和 IP 都一样），只是测 3600 s。测量 16:23:48 开始，被抢占前写下了 30 min 的逐请求记录，按 s13 的口径算（`results/m1/window_tput.txt`）：

  | | 整窗（前 1800 s） | 稳态（300–1800 s） | 每 5 min（第 2–6 个窗口） |
  |---|---|---|---|
  | s13（v3-135） | 42,724 | 46,699 | 40,155 / 47,044 / 47,683 / 50,940 / 48,253 |
  | m1（v3-069） | **38,582（−9.7%）** | **41,719（−10.7%）** | 38,412 / 44,508 / 43,436 / 39,263 / 42,654 |

- **差别全在 decode**（测量期 300–1800 s，采样器）：prefill 两轮几乎一样（现算占比 6.8% / 6.8%，排队 5.98 / 5.92 s，引擎 TTFT 7.64 / 7.56 s）；
  m1 的 decode 生成 3,142 tok/s（s13 3,426，−8%），在跑请求 p50 73（66），4 个 rank 的 KV 用量合计 p50 3.48（3.09），decode 排队 3.38 s（1.30 s），
  decode 引擎 TTFT 4.44 s（2.31 s）。decode 慢 8% 就让 KV 更满、排队翻两倍多、TTFT 多约 2 s。
- 含义：同一配置在两台节点 / 两次运行之间差约 10%，**单次运行里小于约 10% 的差别不能当结论**；decode 侧对节点（或 DP rank 负载分布）敏感，
  prefill 侧稳定。之后的对比尽量在同一节点上做 decode 侧的改动，并同时看各段指标（渲染、排队、现算、decode 生成速度）来归因。
- burst 节点约 1 h 就可能被抢占（B 线那次 1 h 17 min），之后的 burst 轮测 1500 s（部署 + 预热 + 测量约 55 min）。

## 1l. s13 复盘（11-18 下午，新主线 session；数据都来自 s13 的单次运行）

- **测量期开头有固定的启动瞬态，1800 s 成绩低估了与 sglang（3600 s）同口径的数**（`scripts/window_tput.py`，按请求完成时间每 5 min 切窗，
  用 AIPerf 逐请求记录；整窗结果与 AIPerf 一致）：

  | 每 5 min tok/s/GPU | 0–300 | 300–600 | 600–900 | 900–1200 | 1200–1500 | 1500–1800 | 前半 / 后半 | 整窗 |
  |---|---|---|---|---|---|---|---|---|
  | s7b | 21,246 | 30,656 | 33,314 | 30,659 | 33,170 | 30,700 | 28,187 / 31,995 | 30,091 |
  | s9 | 22,301 | 33,878 | 46,664 | 37,754 | 36,581 | 44,626 | 34,308 / 39,538 | 36,923 |
  | s10 | 22,222 | 35,818 | 45,635 | 47,077 | 35,787 | 30,763 | 34,513 / 38,018 | 36,265 |
  | **s13** | 22,458 | 40,155 | 47,044 | 47,683 | 50,940 | 48,253 | 36,518 / **48,931** | 42,724 |

  每一轮的第一个 5 min 都是约 22K（ISL 约 70K、2.4–2.5 req/s）：测量期开头只统计测量开始后发出、且已经完成的请求，偏向短请求，是测量方式本身的瞬态，
  与配置无关，sglang 的 3600 s 运行也有，只是被摊薄。s13 从第 10 min 起稳定在 47–51K（sglang 53,351 的 88–95%）。按此推算 s13 的 3600 s 成绩约 46K，
  m1（s13 原样复跑、测 3600 s）直接验证。之后比较配置时同时报告去掉前 5 min 的"稳态"值。

- **TTFT 的大头从 decode 排队转到了 prefill 排队**（`scripts/ttft_breakdown.py`，测量期均值）：

  | | s10 | s13 |
  |---|---|---|
  | prefill 引擎 TTFT / 排队 / 计算（s） | 5.13 / 3.75 / 0.91 | 6.81 / **5.25** / 1.09 |
  | decode 引擎 TTFT / 排队（s） | 5.86 / **5.08** | 2.08 / 1.18 |

  #47897 让 decode 变快、KV 释放更快，decode 排队从 5.08 s 降到 1.18 s；prefill 排队反而变长（吞吐更高）。两段引擎 TTFT 合计 8.9 s，
  与 AIPerf 的 TTFT 均值 9.16 s 只差 0.3 s，路由、渲染、传输都不是主要部分。
- **prefill 被计算量卡住，不是 KV**（采样器，测量期均值）：在跑 3.0 个请求，排队 17.6 个（`capacity` 13.8 = 每步 8192 token 预算用满，
  `deferred` 3.8 = 等 CPU 层读回），KV 用量 0.42；忙碌（排队非空）81% 的时间（sglang 61%）。现算 23,223 tok/s，**比 sglang 的 16,631 多 40%**；
  按忙碌比例折算，prefill 满载约 28.6K tok/s，sglang 满块 320 ms 约合 25.6K tok/s，计算能力相当，差在要算的 token 多。
- **多算的 token 在跨会话前缀共享上**（`scripts/ideal_hit.py`、`scripts/cold_break.py`，s13 测量期 6,202 个请求）：
  - 续轮请求（5,038 个，同一会话的上一轮在本次运行里出现过）的理想增量 = 相邻两轮 ISL 之差，合计占 ISL 的 2.61%；
  - "冷"请求（1,164 个：973 个子 agent 第一轮，平均 13.7K token；50 个主会话上下文变短（trace 里的压缩），平均 182K；其余 flat / 新会话首轮）
    若全算占 ISL 的 4.88%；
  - vLLM 实际现算 6.76%（计数器），是"续轮理想 + 冷请求全算"的 0.91 倍，即冷请求只从别的会话共享到约 15%；
  - AIPerf 理论命中 95.88%（现算 4.12%）。理论值按前缀语义、在整条 trace 上用一个共享的已见块集合算（`theoretical_prefix_cache.py`、
    `weka_metric_prepass.py`），同一 trace 的子 agent 与父会话共享的块文本相同（`hash_id_scope: local`），所以真实的前缀缓存应该能吃到大部分；
    sglang 实测 96.11%，理论 96.68%，只差 0.6 个百分点，我们差 2.6 个百分点。
  - 结论：prefill 多算的约 40% 主要在冷请求（子 agent 第一轮、压缩后的主会话）的跨会话共享上，不是续轮。原因待查；m2 起路由记录每个
    prefill 的 prompt / cached token（`INFERA_LOG_PREFILL_USAGE=1` + `--enable-prompt-tokens-details`，`scripts/prefill_usage.py`）逐请求核对。
    理论值有一处偏乐观：会话从轨迹 25–75% 处开始回放，起点之前的块服务端从没见过，却算作已见。
- **prefill API server 的 chat 模板渲染在排队**（m1 测量期开头 4 min，`scripts/render_delta.sh`，vLLM 自带的 `request_render_duration_seconds`）：
  prefill 每请求渲染 1.35 s、引擎排队 1.48 s、计算 0.64 s；decode 渲染 0.24 s。渲染发生在请求进入引擎之前、串行，在 TTFT 关键路径上。
  离线测（`scripts/render_cost.py`，AgentX 数据集里 9–17 万 token 的对话）：Jinja 几乎不花时间，**成本全在分词，每请求 100–180 ms**（Rust，释放 GIL）；
  vLLM 0.34 的 `--renderer-num-workers` 默认 1（每个线程有自己的 tokenizer 副本），3.4 req/s 时单线程利用率约 50–80%，请求成批到达时排队。
  m4 = m2 + 两边 `--renderer-num-workers 4`（之前 s11 / s12d 的前端扩展都被别的问题打断，没有干净数据）。
- **CPU 层实际锁页比配置多 17%**：`cpu_bytes_to_use` 1.25 TB 时 prefill 4 个 PP worker 的 `RssShmem` 为 350 / 342 / 342 / 326 GiB，合计 1,360 GiB（1.46 TB；
  每个 stage 按自己的每块字节数分配）；部署前 `Shmem` 8 GB、部署后 1.48 TB，`MemAvailable` 仍有 1.30 TB（主要是可回收的文件缓存）。
  这解释了 2 TB 时 OOM（U10，实际会锁约 2.3 TB）；仍在 InferenceX 每台 ≤ 3 TB 之内，按余量 CPU 层大约还能加到 1.6–1.7 TB，但上面的分析说明容量不是主因。
- **s12 / s12b / s12d 的 prefill 崩溃重新判断**（原始日志）：s12 崩的是 prefill 的 EngineCore 进程（`Segfault encountered`，只有主线程在
  `PyEval_RestoreThread` 处的 native 栈，没有 tokenizers 帧）；s12b、s12d 死的是 PP0 worker（`died unexpectedly (exit code: None)`）。worker 不分词，
  所以这两次不是 B 线说的 tokenizers 问题；`exit code: None` 与 B 线 b1-r6 的主机 OOM（OOM killer 杀 `VLLM::Worker_PP`）一致，而 s12b、s12d 都紧跟在一次
  崩溃之后，可能是"崩溃 → 锁页内存不归还（U23）→ 下一轮 OOM"的连锁（推测，当时没抓内核日志）。`p4d4_run.sh` 现在每轮前检查 `Shmem`（> 200 GB 不部署）。
- tokenizers 压测（B 线 `tokenizer_stress.py`，GLM-5.2 tokenizer、16 线程并发 encode + chat 模板、奇偶线程不同的 `split_special_tokens` / 截断设置，
  v3-069，10 min，约 5,200 次）**没有崩溃**（一次）。

## 1k. s10 之后的瓶颈：decode KV 满、每个请求在 decode 上停得久

- **不是等 KV 传输**（`scripts/mc_xfer.py`，s10 测量期）：prefill 侧 Mooncake 16,379 次传输，每次平均 0.19 s、1.55 GB（8.2 GB/s），合计 14.1 GB/s，
  0 失败、0 过期。
- **decode KV 满**：4 个 rank 的 KV 用量 p50 0.85–0.90、p90 0.99，因 capacity 排队 p90 每 rank 约 11，运行 p50 14–18 / rank。
- decode 每 rank 显存（DP0）：目标 273.6 GiB，权重 130.8 GiB，激活峰值 6.7 GiB，CUDA graph 1.1 GiB，KV 133.3 GiB = 295.6 万 token（48.4 KB / token，
  indexer 只在 1/4 层上）。按 checkpoint 头部统计（`scripts/ckpt_sizes.py`）：路由专家 MXFP4 358.6 GiB、注意力 BF16 24.0 GiB、MTP 专家 BF16 18.0 GiB、
  embed + lm_head 3.5 GiB；"路由专家（含 MTP）/ 4、其余每 rank 一份"= 125.5 GiB，实测只多约 5 GiB，**权重没有明显浪费**，大头是每 rank 一份的
  BF16 注意力（24 GiB）。
- 与 sglang 的 decode 容量**还不能直接比**：sglang 记录的"decode KV 409.5 万 token / rank"要看注意力是 DP 还是 TP（AUS 2P1D 的 decode 是
  `tp 8`、`dp_size 1`、`enable_dp_attention`，KV 在 TP 组内复制，`max_total_num_tokens` 即总量）；vLLM DP4 共 1,182 万个不重复 token。
- 请求在 decode 上停得更久：vLLM ITL p50 约 19 ms（s9 / s10），sglang 约 14 ms（交互 69.9）；decode 4 个 EngineCore 主线程约 99% CPU，
  草稿模型的注意力元数据每步重建（日志：`ROCM_AITER_MLA_SPARSE cannot update draft decode metadata in place`）。decode 每步是下一个主攻点（A 线）。
- s12 = s11 + prefill GEM 0.88（B 线）+ decode #47897（A 线）组合（08:02 起，日志确认 gluon kernel 生效）。
- **s11（s10 + P / D 各 4 API server + renderer 8 线程）预热第 16 min 中止**：根请求得到 Rust 路由的 502
  `decode 10.245.144.166:27301 unreachable: error sending request`，另有 9 条空内容。推断是连接复用竞争：reqwest 连接池空闲 90 s 才丢弃连接，
  uvicorn（vLLM API server）空闲 5 s 就关，路由会写到引擎正在关闭的连接上（s10 路由也记了 2 次 5xx；Python 路由为此关了 keep-alive）。
  s12b = s12 + P / D `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`（长于 reqwest 的 90 s），排在 s12 之后。
- **s12（组合配置）预热第 14 min 中止**：08:17:34 prefill EngineCore 意外退出，日志里一次 `Segfault encountered`（栈 `PyEval_RestoreThread`，
  不带进程名），之后 82 个请求得到 `prefill unreachable`。s8–s11 的 P / D 日志里都没有段错误；B 线 b1-r1（GEM 0.88、1 个 API server）
  跑完 1800 s 无故障。原因未定（候选：GEM + offload 与多 API server 的组合、偶发）。s12b 原样重跑（加 keep-alive），s12c = s12b 但 prefill
  回到 1 个 API server、无 renderer 线程池（decode 保持 4 + 8）。
- s12b：测量第 6 min prefill PP0 worker 被信号杀掉，无效。s12c：无崩溃，但 prefill 500 `Chat template rendering timed out after 30.0s`
  （`VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT` 默认 30 s，单 API server 单 renderer 线程）在预热中止。s12d（放宽到 600 s + prefill renderer 4 线程）：
  测量中 prefill 又挂，473 / 4,635 失败，无效。**prefill GEM 0.88 在本节点 3 次都挂**，与 API server 个数无关（主线交接第 5 节）。
- s13 = s10 + decode #47897 + keep-alive，prefill 不用 GEM（10:35 起）。上游候选：Rust 路由的连接池空闲时间短于引擎的 keep-alive，
  或对写入前已关闭的连接重试。

## 1j. s9（s7b + Python 路由 round-robin，不 tokenize）= 36,923 tok/s/GPU，目前最好

| | s7b | s8（4 API server） | **s9（round-robin）** | sglang |
|---|---|---|---|---|
| tok/s/GPU | 30,091 | 31,384 | **36,923** | 53,351 |
| req/s | 2.511 | 2.609 | 3.001 | 3.469 |
| 完成 / 丢弃 | 4,593 / 5 | 4,770 / 6 | 5,489 / 3 | 12,582 / 1（3600 s） |
| 交互 p50 / p90 | 59.4 / 42.3 | 57.7 / 43.4 | 52.2 / 42.0 | 69.9 / 60.8 |
| TTFT p50 / p90 / 均值（s） | 28.5 / 41.8 / 26.3 | 23.9 / 42.7 / 24.3 | **6.79 / 31.8 / 12.4** | 4.20 / 9.23 / 5.65 |

- 路由 `/metrics` 响应约 15 ms（s8 > 60 s）。路由测的 prefill 段 p50 3.37 s（s8 14.6 s）、均值 13.5 s；路由 TTFT p50 7.8 s、均值 19.7 s（`results/s9/ttft_breakdown.txt`）。
- 新的排队点：**decode 引擎排队均值 6.3 s**（s8 0.5 s）——round-robin 不看负载，decode 各 rank 的长上下文请求分布不均；prefill 引擎排队 3.9 s。
- 命中率不变（测量期现算约 5%）。为 sglang 的 69%。
- **s10（Rust 路由 + 串行 vLLM Mooncake，kv-aware 同 sglang 参数）= 36,265 tok/s/GPU**，与 s9 差 2%（波动内）；TTFT p50 / p90 / 均值
  6.02 / 23.2 / 11.3 s（长尾比 s9 好）、交互 49.9 / 35.8、5,401 / 6；路由 CPU 约 24%、`/metrics` 约 10 ms。路由不再是瓶颈。
- **更正**：s10 按负载选 decode rank，decode 引擎排队均值仍有 5.08 s（s9 6.3 s），所以 decode 排队不是 round-robin 不均造成的。
  候选：vLLM 把等 Mooncake 拉 KV（约 10 万 token、约 5 GB / 请求）计入排队；decode EngineCore 主线程约 99% CPU；decode 单个 API server
  最忙线程 91%。s11 先测前端扩展（4 API server + renderer 8 线程）。

## 1i. 更正：TTFT 的大头在 Infera Python 路由，不在 vLLM 的 API server

- s8（s7b + P / D 各 4 个 API server）测量期间：P 的 4 个 API server 在 py-spy 采样中全部空闲；路由的 `/metrics` 60 s 都不响应，而 P 的
  `/metrics`、`/health` 毫秒级返回；路由进程单核 100%。
- 路由 profile（旁挂容器 `--pid=container:vpd-router --cap-add SYS_PTRACE` 跑 py-spy，20 s，`results/s8-router-profile/`）：**87% 在
  `tokenizer.encode`**，调用链 `KvEventAwarePolicy.pick_targets` → `BlockHasher.hash_for` → `token_ids_for`：kv-aware 策略为算前缀块哈希，
  在事件循环里同步 tokenize 每个约 11 万 token 的 prompt，整个路由被堵住。s7b 里路由测得的 22.8 s "prefill 段"主要是路由自己的排队。
- 1h 里"单 API server 是队列"的判断不成立。s8（4 个 API server）= **31,384 tok/s/GPU**（s7b 30,091，+4.3%），TTFT p50 23.9 s（s7b 28.5）、
  交互 57.7 / 43.4、4,770 / 6；API server 不是瓶颈，这 4% 可看作同配置跑间波动的量级（加少量收益）。
- sglang recipe 用 Rust 路由（多线程），没有这个问题。P4D4 只有一个 prefill，kv-aware 只用来选 decode 的 DP rank。
- s9 = s7b + 路由策略 `round-robin`（不 tokenize；decode rank 轮转，不看负载）。`pd1_up.sh` 新增 `ROUTER_POLICY_ARGS`，路由容器加 `SYS_PTRACE`。
  s9 第一次起路由失败（`--router-tokenizer-path` 是路由的必填参数，被我一起去掉了），05:26 手动按同样参数加上 tokenizer 路径重起路由，
  引擎未重启，冒烟通过后计分（`engine-config.env` 里记录的路由参数因此不全，以本条为准）。`pd1_up.sh` 已改为总是传 tokenizer 路径，
  并在路由容器退出时立即失败。
- s10 = s7b + **Rust 路由**（sglang recipe 用的路由，tokenize 在 Rust 里做）+ 串行 vLLM Mooncake：Rust 路由原来只对 ATOM 有串行，
  这次加了 `INFERA_VLLM_MOONCAKE_SERIAL=1` 的 vLLM 串行路径（`rust/router/src/disagg.rs::vllm_serial_pull`，与 ATOM 共用 `serial_decode`），
  新二进制挂载覆盖 `/usr/local/bin/infera-router`（`pd1_up.sh` 新增 `ROUTER_MOUNT_EXTRA`）。
- 路由描述符（1g）：s8 中在 1,150–1,470 之间上下波动，不是泄漏，是连接数峰值（AIPerf 长连接 + 每请求的 P / D 连接）；需要的上限就是连接峰值。

## 1h. s7b（1.25 TB CPU 层，路由 nofile 已修，1800 s）= 30,091 tok/s/GPU，目前最好；瓶颈转到 API 服务

| | s4（1 TB） | **s7b（1.25 TB）** | sglang |
|---|---|---|---|
| tok/s/GPU | 26,541 | **30,091** | 53,351 |
| req/s | 2.272 | 2.511 | 3.469 |
| 完成 / 丢弃 | 4,146 / 3 | 4,593 / 5 | 12,582 / 1（3600 s） |
| 交互 p50 / p90 | 62.6 / 49.7 | 59.4 / 42.3 | 69.9 / 60.8 |
| TTFT p50 / p90 / 均值（s） | 22.5 / 79.2 / 33.8 | 28.5 / 41.8 / 26.3 | 4.20 / 9.23 / 5.65 |
| 测量期每 5 min 现算比例 | 0.08–0.15 | 0.04–0.08 | ≈ 0.04 |

- 多 25% 的 CPU 层让命中率接近 sglang（现算比例约 5%），吞吐 +13%。路由描述符 30 min 内涨到约 1,190 后放缓（`results/s7b/router-fds.log`），
  超过原 1,024 上限，修复有效；增长形态更像 AIPerf 的长连接池，而不是无限泄漏（待确认）。
- **TTFT 主要不在引擎里**：测量期 prefill 引擎平均首 token 4.49 s（排队 3.18 s + prefill 0.77 s），decode 引擎 1.25 s；而路由测的 prefill 段平均
  22.8 s、TTFT 平均 32.4 s。prefill 引擎大部分时间运行 0–2 个请求、队列为空。差额在两边各**只有 1 个 API server**（chat 模板 + tokenize
  约 11 万 token 的 prompt，串行），decode 还要对完整 prompt 再处理一遍。
- s8 = s7b + P / D 各 `--api-server-count 4`（多 API server 时 `/metrics` 不完整，只影响诊断）。

## 1g. s5 / s7 中止的根因：Infera Python 路由的文件描述符用完（已修）

- s7（1.25 TB，1800 s）在预热第 18 min 同样中止。这次路由日志在：`OSError: [Errno 24] Too many open files`（2,048 次，`socket.accept()`
  失败），随后 01:54:22 四个请求同时得到 502 `prefill leg failed`（路由向 prefill 建连失败），其中有根请求，AIPerf 中止。s5 的 502 同因。
- 路由容器没设 `--ulimit nofile`，容器默认软上限 1024（引擎容器是 65536）；Python 串行路由关了 keep-alive，每段请求新建连接。
  在途请求最多约 130、每请求约 3 个连接，到不了 1024，说明还有描述符泄漏（约每请求不到 1 个；s4 跑 55 min 没撞上，原因未明）。
- 修复：`pd1_up.sh` / `pd2_drive.sh` 的路由容器加 `--ulimit nofile=1048576:1048576`。s7b 重跑 1.25 TB，并每 30 s 记录路由进程的描述符数与类型
  （`router-fds.log`），用来定位泄漏。证据 `results/s7-abort/`。
- 非致命的空内容失败（每轮 3–6 条，固定在少数会话）仍待查。

## 1f. s6c（s4 + ARC 淘汰，1800 s）= 23,864 tok/s/GPU，比 LRU 差，保留 LRU

| | s4（LRU） | s6c（ARC） |
|---|---|---|
| tok/s/GPU | 26,541 | 23,864 |
| req/s | 2.272 | 2.073 |
| 完成 / 丢弃 | 4,146 / 3 | 3,790 / 5 |
| 交互 p50 / p90 | 62.6 / 49.7 | 65.5 / 53.7 |
| TTFT p50 / p90 / 均值（s） | 22.5 / 79.2 / 33.8 | 15.3 / 111.8 / 40.8 |

- 测量期每 5 min 的现算比例 0.10–0.18（s4 0.08–0.15，`results/s*/hit_timeline.txt`）。单次运行，未量化跑间波动；测量第 25 min 抓了 13 s decode trace。
- **decode trace**（`results/s6c/trace/`，4 个 DP rank，约 68 步、每步约 43 ms）：MoE GEMM 31%、**MoE 分发 / 合并通信 26%**
  （每层 2 次 RCCL `ncclDevKernel_Generic`，平均 71.5 µs）、其他 18%、sparse MLA 6%、indexer 5%。各 rank 通信 411–758 ms 不等，
  差额是等最慢的 rank：dp3 的 indexer 是 dp0 的 2.4 倍（上下文更长），它是关键路径。vLLM 有给 DP 组的 aiter 自定义 AG/RS 路径，
  这里没走上，原因待查（队列项 `decode-comm`：默认 / EP / EP + MoRI 对比）。

## 1d. s5（1.25 TB CPU 层，3600 s）：预热第 18 min 一条根请求 502，AIPerf 中止整轮

- 23:56:04 会话 `0a279af1` 第 27 轮（根请求，agent_depth 0）得到 HTTP 502 `{"error":"prefill leg failed"}`：Infera Python 串行路由
  POST prefill 时出现 httpx 网络层异常（`_dispatch_serial` 的 `prefill_unreachable` 分支）。AgentX 预热阶段根请求失败即中止整轮，没有计分。
- 另有一类**非致命**失败：`InvalidInferenceResultError`（流里没有任何内容），s4 预热 3 条、s5 预热 4 条，固定落在少数会话
  （`11310dce` 第 264 / 266 轮在两轮都失败），输入长度 7.7 万–68 万不等（成功请求最长 75 万），与长度无关，原因待查（候选：生成内容被
  reasoning 解析剥空）。s4 能计分是因为这类失败不是根请求失败。
- 中止前最后一次采样（23:56:26）：prefill 运行 0、因 capacity 排队 52、GPU KV 99.1%；decode 4 个 rank 全部空闲。
- **证据不全**：引擎日志被下一轮部署覆盖（`logs/engines/*.log` 每轮重写），路由日志从未保存。之后各轮把三个容器的日志实时写进各自的 run
  目录（队列项 `queue/s6c7.sh`），502 与空内容再出现时可以直接对上。

## 1e. s6（vLLM FS 二级层）：起栈失败，vLLM 的 bug（B 线已修，U11）

- **补充（11-18，B 线）**：修法是 `KVCacheConfig.max_worker_bytes_per_block` 跨 worker 取最大、按它分槽（3 个文件，`../vllm-prefill-trackB-20261118/` 2 节）。
  另外 InferenceX AgentX 只允许 CPU DRAM 级 KV offload，FS（NVMe）层不能用于计分，只能作诊断。

- `TieringOffloadingSpec` 下 PP1 / PP2 等待 `/dev/shm/vllm_offload_*.mmap` 达到 1,249,999,527,936 字节超时（30 s），而 PP0 已建好"1250.00 GB"的文件；
  等待条件是 `>=`，说明各 PP stage 算出的区域大小不一致（各 stage 层数 20/20/20/18）。ROCm 上 `CPUOffloadingSpec` 不用共享区域（每个 rank 自己的锁页内存），
  所以 s4 / s5 不受影响。原始日志 `results/s6-fs-fail/prefill-fail.log.gz`。列入上游候选。
- 随后改为两轮 1800 s 的单变量对照：s6c（ARC 淘汰）、s7（1.25 TB）。

## 1b. GEM KV + OffloadingConnector：第一次写出就 GPU 访存故障（已由 B 线解决，见 U2）

- **更正（11-18，B 线）**：根因在 ROCm clr：`hipExternalMemoryGetMappedBuffer` 导入的 GEM BO 不设 `deviceId`（默认 0），`hipMemcpyBatchAsync` 按它把
  PP rank k > 0 的拷贝放到 device 0 上执行而故障。GEM 分配器改用 HIP VMM 导入后修复（`../vllm-prefill-trackB-20261118/` 1 节）。下面列的"尚未排除"的
  嫌疑项（150 GB BO、`is_src_access_order_any`、`/dev/shm`）都不是原因。以下为当时的记录。

- 现象：prefill 某个 PP worker 在第一次 offload 写出时 `HSA_STATUS_ERROR_MEMORY_FAULT`（之前的请求都不足一个块，不触发 offload）。
- 隔离（`../vllm-pd-tools-20261117/scripts/swap_fault_test.py` 与聚合复现）：
  - offload 的拷贝 `ops.swap_blocks_batch`（ROCm 上 `hipMemcpyBatchAsync`）在普通 / GEM 显存 × 锁页 / `hipHostRegister` mmap 的所有组合上正常，
    包括 8192 个描述符铺在 64 GB 缓冲区上；
  - 聚合 PP4 + offload（无 GEM、无 Mooncake）：100 GB 正常；1 TB 第一个请求卡 > 5 min（锁定主机内存，另见下）；
  - 聚合 PP4 + MultiConnector[GEM Mooncake + offload 100 GB]：**复现故障**。所以是 GEM 与 offload 的组合，与 PD、容量无关。
- 尚未排除：150 GB 级单个 GEM BO（测试只到 64 GB）、写出时的"源端访问顺序任意"属性（`is_src_access_order_any`）、
  `/dev/shm` 背后的共享 offload 区。绕法：prefill 不用 GEM（用较低显存目标容纳 dma-buf 重复记账），decode 用 GEM。

## 1c. PD 停顿：prefill 侧重排 + decode 到达即预留（推断，证据不全）

- 现象（s2b，20:16）：prefill 运行中 0、排队 76、KV 99.7%，prompt 吞吐 0；AIPerf 的返回数停在 198 超过 1 min。
  **更正（20:42）**：当时读到的 decode "4 个 rank 运行中 0、KV 84–97%" 不可靠——decode 开了 4 个 API server（vLLM 按 DP4 自动设置），
  每次抓 `/metrics` 只落到其中一个进程，计数与 gauge 都是局部的（s2c 中 decode 收到 296 个请求，抓到的成功数每个 rank 只有 1）。
  所以 s2b 能确定的只有 prefill 侧停顿；是循环等待还是 decode KV 满造成的反压，需要 decode 用单个 API server 重测。
- 机制：vLLM Mooncake 是并发下发，decode 收到请求就为整段 prompt 预留 KV 等 prefill 推送；prefill 算完的请求占着 KV 等 decode 来拉
  （租约 `VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT`）。prefill 开了缓存感知准入后与 decode 的到达顺序不一致：decode 预留的是 prefill 还没算的请求，
  prefill 占着的是 decode 还没预留的请求，两边都满时循环等待。两侧都 FCFS 时顺序一致，不会死锁（s1 没有死锁）。
- 含义：在 vLLM PD 下，prefill 侧任何重排（缓存感知准入、优先级、SPF）都需要配合 decode 侧"按 prefill 完成顺序预留"或"prefill 完成后再下发 decode"
  （串行拉取，如 NIXL）才安全。这是上游候选（Mooncake connector / router 的下发方式），也是 sglang 的 SPF 在 vLLM 上不能直接照搬的原因：
  sglang 的 prefill 要等 decode 预分配完成才开始算。
- 另：1 TB CPU 层的锁定内存使起栈慢（472 s）或第一个请求慢（77 s 至 > 5 min）。

## 1a. s1 的颠簸（为什么 prefill 只有设备缓存不行）

- AgentX 的会话很长：s1 里 prompt 平均约 16 万 token，104 条会话的工作集约 1,500 万 token 以上，prefill PP4 设备 KV 池 1,315 万 token。
- 预热开头 115 条会话各发一个冷启动的长请求（约 1,700 万 token）。FCFS 下每条会话的下一轮要在 prefill 排队约 100 个请求、约 450 s，
  这期间新算了约 1,500 万 token，它的前缀已被 LRU 换出；命中率塌到 5%，计算量更大、排队更长，自我强化。
- chat template 不是原因：用 `/tokenize` 比较带推理内容和工具调用的两轮 agent 对话，下一轮完全包含上一轮的 token（公共前缀 100%）。
- sglang 用 HiCache（容量）+ SPF（后续轮未命中 token 少，插到冷请求前面，趁缓存还在算完）避开了这个问题。vLLM 有 CPU 层
  （OffloadingConnector）和缓存感知准入窗口（只看 GPU 前缀缓存、二分"已缓存 / 未缓存"，不是按未命中 token 排序）；
  按未命中 token 排序、且把 connector（CPU 层）命中算进去的 SPF 是上游候选。

## 1. 部署

| 项 | prefill | decode |
|---|---|---|
| 并行 | TP1 × PP4，`VLLM_PP_LAYER_PARTITION=20,20,20,18` | `-dp 4 -tp 1`（MoE 平铺 TP4），内置 DP 均衡关，router 用 `X-Data-Parallel-Rank` 选 rank |
| 批 / 缓存 | `--max-num-batched-tokens 8192`（chunk 8192）、前缀缓存开、调度 fcfs（vLLM 没有 SPF） | `--max-num-seqs 64`、`--stream-interval 30`、前缀缓存关（与 sglang 的 decode 相同） |
| KV | FP8 e4m3；GEM 内存池；MultiConnector[Mooncake producer] | FP8 e4m3；GEM 内存池；Mooncake consumer |
| MTP | `{"method":"mtp","num_speculative_tokens":3}`（P 也要带，生成草稿层 KV） | 同左；计分时加 `rejection_sample_method: synthetic, synthetic_acceptance_length: 2.99` |
| KV 池 | 每卡 KV 缓冲 150–163 GB（权重 107–119 GiB） | 每 rank 2,764,032 token（sglang TP4 decode 关 PTPC 时 3,043,072，少 9%） |
| 起栈 | 151 s | 682 s（每 rank 读全部 shard，约 2 s / shard） |

Mooncake：镜像 `infera-vllm:v0340-0c5e5e89`（v0.3.14 + #4540 + #4541），`MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18
MC_MAX_MR_SIZE=1073741824 MC_GID_INDEX=1`，每个容器一个拓扑文件（容器内 `cuda:i` → 该 GPU 的 ionic，无备选，`results/gate-p4d4-r1/config/`）。
router：`--router-backend rust --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 --session-affinity prefill
--pd-prefill-guard-release completion --decode-input-demand on`。

## 2. 待查

- **KV 流量分布**：门禁期间 8 张 ionic 各约 5.4 GB/min、ACK 超时 0、重传 0；按"写入走目的端首选网卡"的设置，预期只在 ionic_4-7 上。
  单节点回环不受影响，两节点 P8D8 前要用 `MC_LOG_RDMA_SLICE_AFFINITY=1` 确认实际选卡。
- decode KV 池比 sglang 少 9%：显存目标（vLLM 默认 0.92，sglang 0.95）与 DP 下 dense 层每 rank 一份。
- prefill 的 aiter BF16 GEMM 多个形状无调优表（日志 "not found tuned config"），sglang recipe 带了两份 GLM-5.2 调优表。
