# GLM-5.2 P4D4（Infera + vLLM）最优 recipe（a25 配置，约 56.4k tok/s/GPU）：用到的修复 / 功能 / 优化与上游对照

截至 2026-11-25 12:00 UTC。数据来源：分支 `dev/pd_opt/vllm_glm52_agentx` 的 `llying/vllm-pd-opt-20261123/`（`SUMMARY-20261125`、`ITEMS-STATUS`、`README`、`HANDOFF`、
`once/`、`deploy/`）和 `llying/upstream-vllm-20261117/`（`CANDIDATES`、`REPORT-upstream`）。上游状态是 11-25 用 `gh` 查的。成绩若无说明都是**单次运行**。
本次只读文档、脚本和上游代码，没有跑任何作业。

## 0. 摘要

- **最优 recipe = a25 配置**：v3-037 上两次 56,342 / 56,448（平均 **56,395 tok/s/GPU**），比同节点 a8 配置 +5.2%，比 sglang P4D4（53,351）**+5.7%**；第二台节点 v3-239 成对 56,280（+4.9%）。
- recipe 里一共用到 **约 30 项**修复 / 功能 / 优化 / 配置（第 3 节）。其中**要改代码的 14 项**里：上游已合入 1 项（Mooncake #4540），其余都还是**覆盖层、fork 或我们的 open PR**。
  InferenceX 规则不允许引擎补丁，所以要提交正式成绩，这些都得先进上游（第 5 节）。
- **对已提 PR 的影响**（第 6 节）：
  - **结论被后续实验改变的**：U28（Infera #250）——当时"结论不明、默认关"，后来同节点成对 **+1.1%**，并成了 recipe 的基础；PR 正文要更新。U29（观察记录）——"显存余量不够"的推断被约 30 轮 0.95 显存目标的稳定运行进一步削弱。
  - **实现要改的**：只有 U13（main 上的 #68006 已修一半，recipe 用的 v0.34.0 版本在 main 上要换成改写版）。另外 FP4 dispatch 的收益**依赖 U13**，U13 的价值比当初测的更大。
  - **结论不变、但 recipe 没用到的**：U17（recipe 的 decode 不开 EP）、U19（recipe 不用 Tiering 共享区）、U11（recipe 的 PP 切分恰好让 stage 0 最大，默认模式不越界）。
  - **新发现的耦合**：U17 与 FP4 dispatch 改同两个文件；U19 与 #69254；U26（#60909）在 main 上可能被 #69254 改变行为；#61985（a33 在测）会绕开 U12。
- **最值得先推的上游动作**（第 5 节）：#58710 留言（sync skip，**+9.1%**，PR 已 16 天没更新）、写 U32 KV 感知 DP 均衡和 U33 FP4 dispatch 的 PR、Triton top-k（sglang #47872 已由 @dorado269 提出，vLLM 侧没有）、Mooncake #5584 合入 VMM 导入。

---

## 1. 背景

### 1.1 目标与口径
- **任务**：在 1 台 MI355X（crusoe，8 卡，主机内存 2,752 GiB，8 块 ionic 网卡）上跑 GLM-5.2-MXFP4 的 PD 分离推理，负载是 InferenceX 的 **AgentX 智能体回放**：
  104 个并发会话（C104），上下文 1 万–95 万 token（平均约 11.7 万），测量 3600 s，指标是 **tok/s/GPU**，同时看交互速度（interactivity）、ITL、TTFT。
- **MTP 接受长度**：按 InferenceX 智能体规则用合成接受 **2.99**（`synthetic_acceptance_length`）；正确性另用"门禁"（真实接受、GSM8K 200 题、16K needle / 长前缀）检查。
- **对标**：sglang 同口径 P4D4 = **53,351**。
- **InferenceX 规则**（`ITEMS-STATUS` 第 9 节）：引擎必须原样运行（不允许 `.patch`、覆盖容器文件、monkey-patch）；生产用到的参数要写进**已合入**的 vLLM recipe；
  目标模型降精度在 eval 通过时允许，草稿（MTP）不能降精度或减 FLOPs。**所以我们现在所有的覆盖层，最终都必须变成上游代码。**

### 1.2 拓扑（P4D4，1P1D）
```
GPU 0-3: prefill 引擎  TP1 × PP4（层切分 20,20,20,18），CPU 层 KV offload 1.9 TB
GPU 4-7: decode 引擎   DP4 × TP1（MoE 不开 EP，走 naive AG/RS 分发），MTP 3
KV 传输: Mooncake RDMA 本机回环（prefill → decode），8 块网卡
路由:    Infera Rust 路由（kv-aware 策略，vLLM Mooncake 串行下发）
```

### 1.3 演进
| 阶段 | 配置 | tok/s/GPU | 相对 sglang |
|---|---|---|---|
| 主线 m10（11-18） | v0.34.0 + U12 / U13 / U26 等覆盖层 + DP 均衡关 | 48,572 | 91% |
| a8（11-24） | m10 + sync skip + decode 前缀缓存 + DP 引擎均衡 + KV 感知均衡 + GEMM 表 + FP4 dispatch + decode mnbt 2048 | 53,615（三次平均） | 100.5% |
| **a25（11-24 / 25）** | a8 + Triton decode top-k + prefill 分块 4096 / 阈值 2048 | **56,395（两次平均）** | **105.7%** |

各项单独的贡献（成对比较，各单次）：sync skip **+9.1%**（a1）、decode 前缀缓存 +4.2%（a2）、DP 引擎均衡 +1.1%（a3）、KV 感知均衡 + GEMM 表 +1.2%（a5）、FP4 dispatch +0.6%（a7）、
decode mnbt 2048 +1.0%（a6）、Triton top-k +1.4%（a18）、prefill 阈值 4096 +1.6%（a19）、prefill 分块 4096 +0.8%（a20）；Triton top-k 与 prefill 阈值叠加 +4.5%（a22，多于各自之和）。

### 1.4 a25 的成绩（a28 A，v3-037）
| | a25 配置 | a8 配置 | sglang P4D4 |
|---|---|---|---|
| tok/s/GPU | **56,448** | 53,610 | 53,351 |
| 交互 p50 / p90 | 70.0 / 64.9 | 65.0 / 60.7 | 69.9 / 60.8 |
| ITL p50 / p90（ms） | 14.3 / 15.4 | 15.4 / 16.5 | 14.3 / 16.4 |
| TTFT p50 / p90 / 均值（s） | 2.01 / 5.22 / 2.81 | 2.50 / 6.85 / 3.87 | 4.20 / 9.23 / 5.65 |

---

## 2. recipe 的软件版本与启动命令

### 2.1 版本
| 组件 | 版本 | 说明 |
|---|---|---|
| 镜像 | `infera-vllm:v0340-0c5e5e89` | 由 Infera 提交 `0c5e5e89` 构建：**vLLM v0.34.0**（逐字节核对过）+ Mooncake **v0.3.14 自编（加我们的 #4540、#4541 和一个 include 修复）** + GEM 分配器 + Infera core |
| Infera 源码 | 分支 `dev/pd_opt/vllm_glm52_agentx` 的 `infera/` | 挂到 `/opt/infera/infera` 覆盖镜像里的；含 U7、U14、U16、U28 等改动和 `infera/engine/vllm/gem_mooncake.py` |
| Rust 路由 | 预编译 `bin/infera-router-vllmserial-usage` | 含 U16 串行下发 + prefill 用量日志；是否含 U24 文档里没写，recipe 两侧仍保留 keep-alive 120 的绕法 |
| GEM 分配器 | `bin/libhip_gem_alloc-vmm.so` | U2 的 VMM 导入版（prefill、decode 都用，`LD_PRELOAD`） |
| 覆盖层 | 作业开头从镜像原文件生成并单测 | 见 2.3 |
| 模型 | GLM-5.2-MXFP4 | |

### 2.2 启动命令（a25 B 侧，变量已展开；`$IP` 是节点 IP，`$M` 是模型路径）

**etcd**：`quay.io/coreos/etcd:v3.5.14`，客户端端口 24701。

**prefill（GPU 0–3，容器 `vpd-prefill`）**
```bash
# 环境变量
HIP_VISIBLE_DEVICES=0,1,2,3 PYTHONHASHSEED=0 SAFETENSORS_FAST_GPU=1 HSA_NO_SCRATCH_RECLAIM=1
VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 NCCL_IB_DISABLE=1 RDMAV_FORK_SAFE=1
VLLM_ENGINE_READY_TIMEOUT_S=3600 INFERA_ENGINE_READY_TIMEOUT=3600            # U6 的绕法（DP 起栈慢）
MC_GID_INDEX=1 MC_DISABLE_HIP=1 MC_RDMA_WRITE_PREFER_DEST_NIC=1 MC_IB_TIMEOUT=18 MC_MAX_MR_SIZE=1073741824
MOONCAKE_DISABLE_HIP_DMABUF=0 MC_HIP_KFD_PIN_FIX=0 MC_CUSTOM_TOPO_JSON=<每卡一块网卡的拓扑>
VLLM_HOST_IP=$IP VLLM_MOONCAKE_BOOTSTRAP_PORT=28398 VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT=1800
INFERA_AMD_GEM_ALLOCATOR=/opt/infera-gem/libhip_gem_alloc.so LD_PRELOAD=/opt/infera-gem/libhip_gem_alloc.so
VLLM_PP_LAYER_PARTITION=20,20,20,18
VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS=512 VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600
# 挂载：libhip_gem_alloc-vmm.so、overlay-cpupin/gpu_worker.py（U26 精确锁页）、Infera 源码
# docker：--network host --ipc host --shm-size 32g --ulimit memlock=-1 --ulimit nofile=65536 --cap-add IPC_LOCK --device /dev/infiniband

python3 -m infera.engine.vllm --discovery-backend etcd --etcd-endpoint $IP:24701 --request-transport http \
  --kv-event-transport zmq --advertise-host $IP \
  --model $M --served-model-name glm5.2-mxfp4 --trust-remote-code --host 0.0.0.0 --port 27201 \
  --kv-cache-dtype fp8_e4m3 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --tensor-parallel-size 1 --pipeline-parallel-size 4 --enable-prefix-caching --max-num-seqs 64 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[
      {"kv_connector":"MooncakeConnector","kv_connector_module_path":"infera.engine.vllm.gem_mooncake","kv_role":"kv_producer",
       "kv_connector_extra_config":{"num_workers":10,"device_name":"ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"}},
      {"kv_connector":"OffloadingConnector","kv_role":"kv_both",
       "kv_connector_extra_config":{"cpu_bytes_to_use":1900000000000,"eviction_policy":"lru"}}]}}' \
  --gpu-memory-utilization 0.88 --block-size 64 \
  --cache-aware-admission-window 128 --cache-aware-admission-threshold 0.0 \
  --api-server-count 1 --renderer-num-workers 4 --enable-prompt-tokens-details \
  --max-num-batched-tokens 4096 --long-prefill-token-threshold 2048      # a25 新增（覆盖前面的 8192）
```

**decode（GPU 4–7，容器 `vpd-decode`）**
```bash
# 环境变量：同 prefill 的基础 / Mooncake / GEM 部分（MC_CUSTOM_TOPO_JSON 换成 decode 的），再加
VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1                     # U12（覆盖层里的开关）
VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120 VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT=600
INFERA_VLLM_DP_ENGINE_LB=1                                 # U28：路由不选 rank，交给 vLLM
VPD_DP_LB_KV_WEIGHT=14 VPD_DP_LB_KV_CAP_TOKENS=3141120     # U32 KV 感知 DP 均衡（覆盖层）
VPD_FP4_DISPATCH=1                                         # U33 FP4 dispatch（覆盖层）
VPD_TOPK_IMPL=triton                                       # Triton decode top-k（覆盖层）
# 挂载（覆盖 vLLM v0.34.0 的文件）：
#   v1/attention/backends/mla/rocm_aiter_mla_sparse.py   ← sync skip（U30）
#   v1/engine/core_client.py                              ← KV 感知 DP 均衡（U32）
#   model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py + experts/rocm_aiter_moe.py  ← FP4 dispatch（U33）
#   v1/attention/ops/rocm_aiter_mla_sparse.py + vpd_triton_topk_decode.py  ← Triton top-k（基于 v3comm 版，含 U12 的调用）
#   v1/attention/ops/rocm_gluon_paged_mqa_logits.py       ← U12 gluon logits
#   distributed/device_communicators/cuda_communicator.py ← U13（v0.34.0 版）
#   aiter/configs/model_configs/glm52_bf16_tuned_gemm.csv ← decode BF16 GEMM 调优表

python3 -m infera.engine.vllm --discovery-backend etcd --etcd-endpoint $IP:24701 --request-transport http \
  --kv-event-transport zmq --advertise-host $IP --no-enable-kv-events --kv-events off \
  --model $M --served-model-name glm5.2-mxfp4 --trust-remote-code --host 0.0.0.0 --port 27301 \
  --kv-cache-dtype fp8_e4m3 --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --data-parallel-size 4 --tensor-parallel-size 1 --data-parallel-rpc-port 29620 \
  --enable-prefix-caching --max-num-seqs 64 --stream-interval 30 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"rejection_sample_method":"synthetic","synthetic_acceptance_length":2.99}' \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_connector_module_path":"infera.engine.vllm.gem_mooncake","kv_role":"kv_consumer",
      "kv_connector_extra_config":{"num_workers":10,"device_name":"ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"}}' \
  --block-size 64 --gpu-memory-utilization 0.95 --api-server-count 1 --renderer-num-workers 4 \
  --max-num-batched-tokens 2048
```
（计分时 decode 还开了 `--profiler-config` 抓 3 s trace、挂了逐步日志插桩 `vpd_steplog`，都只用于测量，不属于 recipe。）

**路由（容器 `vpd-router`）**
```bash
# docker：--ulimit nofile=1048576:1048576；Rust 路由二进制挂到 /usr/local/bin/infera-router
INFERA_VLLM_MOONCAKE_SERIAL=1      # U16 串行下发
INFERA_LOG_PREFILL_USAGE=1         # 只用于测量
python3 -m infera.server --host 0.0.0.0 --port 28300 --router-backend rust \
  --discovery-backend etcd --etcd-endpoint $IP:24701 --request-transport http --kv-event-transport zmq \
  --router-tokenizer-path $M --router-policy kv-aware --kv-prefill-overlap-weight 20 --kv-decode-overlap-weight 2 \
  --session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on
```

**客户端**：AgentX C104、3600 s，先跑 smoke；脚本 `scripts/score_p4d4.sh` → `scripts/agentx.sh`。
**复现**：`SUMMARY-20261125.zh-CN.md` 第 2 节（`once/p4d4-pair3.sh` + `once/score-a25-B.env`）。

### 2.3 哪些是"引擎补丁"（InferenceX 不允许）
decode：sync skip、KV 感知 DP 均衡、FP4 dispatch、Triton top-k、U12、U13、GEMM 表（7 个覆盖层）；prefill：U26 精确锁页（1 个）；
两侧：GEM 分配器（`LD_PRELOAD`）+ Infera 的 `gem_mooncake` 连接器模块；镜像里的 Mooncake 带未合入的 #4541。

---

## 3. 总表一：recipe 里用到的项

状态判定（11-25）：**已合入**；**活跃** = 7 天内有更新；**放缓** = 7–14 天没更新；**停滞** = 超过 14 天没更新；括号里是 draft。"u" 是最后更新日期。
"形态"：配置 = 只是参数；覆盖层 = 替换镜像文件；镜像 = 构建进镜像；Infera 分支 = 只在 `dev/pd_opt/vllm_glm52_agentx`。

### 3.1 RDMA / KV 传输（Mooncake）

| # | 项 | recipe 里的作用 / 收益 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| R1 | 目的端网卡写（`MC_RDMA_WRITE_PREFER_DEST_NIC=1` + 每卡一块网卡的拓扑） | 本机回环时 KV 走 decode GPU 对应的网卡 | 镜像 + 配置 | [Mooncake#4540](https://github.com/kvcache-ai/Mooncake/pull/4540)（我们的） | **已合入** 11-19 | 升级到含它的 Mooncake 版本后去掉自编 |
| R2 | RDMA ACK 超时可配（`MC_IB_TIMEOUT=18`） | 避免长时间卡在重传；仍有少量 ack 超时（每次约 1 s） | 镜像 + 配置 | [Mooncake#4541](https://github.com/kvcache-ai/Mooncake/pull/4541)（我们的） | open，**活跃**（u 11-23） | 跟进 review，合入后去掉自编 |
| R3 | KV 池放在 GEM BO 里（`libhip_gem_alloc-vmm.so` + `gem_mooncake` 连接器）：U1 / U2 | 避免 KFD 把 dma-buf 锁住的显存记两次账；VMM 导入修掉非 0 号卡访存故障（U2） | Infera 分支 + 预编译库 | [Mooncake#5584](https://github.com/kvcache-ai/Mooncake/pull/5584) HipGemAllocator（我们的）、[#5587](https://github.com/kvcache-ai/Mooncake/pull/5587)（我们的）；vLLM 侧 U1 没提；Infera main 没有 `gem_mooncake.py` | #5584 open **活跃**（u 11-18，有人在 review）；#5587 (draft) 活跃 | **#5584 改用 VMM 导入**（U2 的修法，`REPORT-upstream` 6.2 节，尚未改）；之后 Infera 提 `gem_mooncake` 连接器的 PR；ROCm clr issue（U2 根因）仍缺 SAML 授权没开 |
| R4 | vLLM Mooncake 串行下发（U16） | 防 prefill 重排时与 decode 预留互相等待 | Infera 分支（Rust 二进制 + Python） | [Infera#246](https://github.com/AMD-AGI/Infera/pull/246) | ready，等 review | 催 review |
| R5 | Mooncake 不走 HIP（`MC_DISABLE_HIP=1`） | 同节点时 Mooncake 默认优先 HIP；HIP 在发送端有计算时只有 0.36–0.4 GB/s | 配置 | — | — | 写进 recipe |

### 3.2 HiCache / CPU 层 KV offload

| # | 项 | 作用 / 收益 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| H1 | prefill CPU 层 1.9 TB（`OffloadingConnector`，LRU） | prefill 前缀命中与 sglang 持平 | 配置（v0.34.0 自带） | — | — | 写进 recipe |
| H2 | ROCm 上 CPU 层私有张量**精确大小锁页**（U26） | torch 锁页分配器按 2 的幂取整，1.25 TB 实锁 +17%、1.6 TB 主机 OOM；1.9 TB 只靠它才放得下 | 覆盖层（prefill） | [#60909](https://github.com/vllm-project/vllm/pull/60909)（@lobo235） | open，**活跃**（u 11-25） | 发留言稿（`PR-U26-…md` 末尾，附 1.9 TB 数据、提出帮测）；注意第 6.6 节 #69254 的影响 |
| H3 | CPU→GPU 读回每批描述符上限 512（`VLLM_KV_OFFLOAD_MAX_BATCH_DESCRIPTORS`） | 读回不长时间占住线程 | 配置（v0.34.0 自带） | — | — | 写进 recipe |
| H4 | 缓存感知准入（`--cache-aware-admission-window 128 --cache-aware-admission-threshold 0`） | 前缀已在缓存里的请求先进（接近 sglang 的 SPF） | 配置 | [#54625](https://github.com/vllm-project/vllm/pull/54625) | **已合入**（11-04，v0.34.0 里有） | 写进 recipe |

### 3.3 调度（prefill / decode / 路由）

| # | 项 | 作用 / 收益 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| S1 | prefill `--long-prefill-token-threshold 2048` + `--max-num-batched-tokens 4096` | 超长 prefill 不饿死小请求、PP4 流水线更短；阈值 4096 +1.6%（a19）、分块 4096 +0.8%（a20）；与 K3 一起的 a25 比 a8 +5.2% | 配置（v0.34.0 自带） | 相关：[#68560](https://github.com/vllm-project/vllm/pull/68560)（单个长 prefill 也切块）、[#68614](https://github.com/vllm-project/vllm/pull/68614)（PP 空步 / 按积压切分） | #68560 open 活跃；#68614 **已合入** 11-25 | 写进 recipe；#68560 + #68614 的回移植 a32 在测 |
| S2 | decode 前缀缓存（`--enable-prefix-caching`），U31 | 被抢占后不再整段重算：重算超出时间 10.8% → 0.2%，**+4.2%**（a2） | 配置 | 无代码 PR（部署问题） | — | 写进 recipe；PD 部署文档可以提一条建议（PD decode 开前缀缓存） |
| S3 | decode `--max-num-batched-tokens 2048` | KV 容量 +6.3%，**+1.0%**（a6） | 配置 | — | — | 写进 recipe |
| S4 | DP 引擎均衡：路由不选 rank，交给 vLLM（U28） | 最忙 / 平均 1.28 → 1.13，**+1.1%**（a3） | Infera 分支 | [Infera#250](https://github.com/AMD-AGI/Infera/pull/250) | ready，等 review | **更新 PR 正文**（第 6.7 节） |
| S5 | **KV 感知 DP 均衡**（U32，W = 14） | KV 满的时间 53% → 47%、抢占减半，**+1.2%**（a5，与 GEMM 表一起加） | 覆盖层（vLLM `core_client.py`） | **没有**（按 DP 负载均衡关键字查过） | — | **新开 vLLM PR**（第 5 节） |
| S6 | 路由 kv-aware 策略、`--session-affinity prefill`、`--pd-prefill-guard-release completion`、`--decode-input-demand on` | sglang recipe 的路由参数 | 配置（Infera main 自带） | — | — | 写进 recipe |
| S7 | `--renderer-num-workers 4`、模板渲染超时 600 s（U25 的绕法） | 引擎重复渲染 chat 模板的开销分到 4 个线程 | 配置 | 根本修法（路由直接传 token id）没做 | — | 暂不提 |
| S8 | 两侧 keep-alive 120 s（U24 的绕法） | 防路由复用正在关闭的连接（502） | 配置 | [Infera#247](https://github.com/AMD-AGI/Infera/pull/247)（U24 根本修法） | ready，等 review | recipe 没用 U24，见第 6.7 节 |

### 3.4 Kernel / 算子

| # | 项 | 作用 / 收益 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| K1 | **sync skip**：ROCm sparse MLA 元数据去掉 `seq_lens.cpu()` 和重算后的 `synchronize()`（U30） | GPU 忙碌 83–92% → 99.7%，**+9.1%**（a1，单项最大） | 覆盖层 | 第一部分 [#58710](https://github.com/vllm-project/vllm/pull/58710)（@NickLucche）；第二部分 [#66400](https://github.com/vllm-project/vllm/pull/66400)；相关 [#67232](https://github.com/vllm-project/vllm/pull/67232)、[#61504](https://github.com/vllm-project/vllm/pull/61504) | #58710 open **停滞**（u 11-09，作者写明缺 ROCm 硬件验证）；#66400 **已合入** 11-09；#67232 open 放缓；#61504 (draft) 放缓 | **发 #58710 的留言稿**（`COMMENT-vllm-58710.md`，附 MI355X 数据）——它正缺 ROCm 验证 |
| K2 | gluon paged-MQA logits（U12） | indexer logits 2.24–2.54 倍；每步 −2.7% 至 −3.9%（v0.34.0） | 覆盖层 | [#68596](https://github.com/vllm-project/vllm/pull/68596)（@RuibinCheung）；sglang 版 [sglang#47897](https://github.com/sgl-project/sglang/pull/47897)（@dorado269） | #68596 open **活跃**（u 11-25）；sglang#47897 open 活跃 | 发留言稿（`PR-U12-…md` 末尾）；注意 #61985 会绕开它（第 6.4 节） |
| K3 | **Triton decode top-k 第三版**（indexer 长上下文 top-k；新编 U34） | 每次 105–186 → 约 70 µs，**+1.4%**（a18），与 S1 叠加 +4.5%；生产 716 万行逐行精确 | 覆盖层 | vLLM：**没有**（[#69700](https://github.com/vllm-project/vllm/pull/69700) 是 gfx942 的 FlyDSL top-k (draft)；[#63847](https://github.com/vllm-project/vllm/pull/63847) persistent top-k 已合入，不同实现）；sglang 原版 [sglang#47872](https://github.com/sgl-project/sglang/pull/47872)（@dorado269） | sglang#47872 open（u 11-19） | **新开 vLLM PR**（第 5 节），署名 @dorado269 |
| K4 | decode BF16 dense GEMM 调优表（141 个形状，FlyDSL；新编 U35） | 与 S5 一起 +1.2%（a5）；单独没有干净的成对数据（a4 作废，卡死前 +1.4%） | 覆盖层（aiter 配置表） | aiter 仓库的模型配置表；`ROCm/aiter` 要 SAML 授权，**没查到** | 未知 | 提到 aiter 的 `configs/model_configs/`；先拿到 ROCm 组织授权 |

### 3.5 通信与并行策略

| # | 项 | 作用 / 收益 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| P1 | P4D4 拓扑：prefill TP1 × PP4（切分 20,20,20,18）、decode DP4 × TP1、MTP 3、fp8 KV、block 64 | 基础形态（EP / MoRI / DBO 单节点都更慢，DCP 不能用） | 配置 | — | — | 写进 recipe |
| P2 | FULL graph 捕获时放行 AITER 自定义 AG/RS（U13） | v0.34.0：每步 −2.7%；main：−2.0% 至 −3.1%（J8）；**FP4 dispatch 的收益也靠它**（第 6.1 节） | 覆盖层（v0.34.0 版） | fork [llying-001/vllm#1](https://github.com/llying-001/vllm/pull/1)；main 上 [#68006](https://github.com/vllm-project/vllm/pull/68006) 修了一半 | fork；#68006 **已合入** 11-20 | 按改写补丁重测后开上游（`BRIEFING` 第 4 节） |
| P3 | **FP4 dispatch**：naive AG/RS 分发前先量化成 MXFP4（U33） | hidden all-gather 20–29 → 5–8 µs，**+0.6%**（a7） | 覆盖层 | **没有**（MoRI 路径有：[#59596](https://github.com/vllm-project/vllm/pull/59596) 已合入、[#63029](https://github.com/vllm-project/vllm/pull/63029) open） | — | **新开 vLLM PR**（第 5 节），与 U17 协调 |

### 3.6 部署 / 稳定性（Infera）

| # | 项 | 作用 | 形态 | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|---|
| D1 | Infera 适配 vLLM 0.34 的参数解析（U7） | 没有它 Infera 的 vLLM worker 起不来 | Infera 分支 | [Infera#251](https://github.com/AMD-AGI/Infera/pull/251) | ready，等 review | 催 review（与 #250 有文本冲突） |
| D2 | 引擎就绪超时 3600 s（U6 的绕法） | DP 每个 rank 读全部权重分片，起栈约 11 min | 配置 | 根本修法没做 | — | 暂不提 |
| D3 | nofile（U14） | recipe 里由 docker `--ulimit` 提供，U14 的代码在但不起作用 | Infera 分支 + 配置 | [Infera#248](https://github.com/AMD-AGI/Infera/pull/248) | ready | 不受影响 |
| D4 | 部署前检查 Shmem（U23） | 防上一轮残留的锁页内存让 1.9 TB 层 OOM | 部署脚本 | — | — | — |

---

## 4. 总表二：过程中解决或尝试过、但不在 recipe 里的项

| 类别 | 项 | 为什么不在 recipe | 上游对应 | 上游状态 | 下一步 |
|---|---|---|---|---|---|
| RDMA | Mooncake 完成信号间隔（`MC_RDMA_SIGNAL_INTERVAL=32 MC_MAX_WR=4096`，a31） | **在 ionic 网卡上让 prefill 发送反复 30 s 超时**，失败前也没加速（163.0 对 161.2 ms） | [Mooncake#4542](https://github.com/kvcache-ai/Mooncake/pull/4542)、RFC [#63006](https://github.com/vllm-project/vllm/issues/63006) | #4542 已合入 10-10 | **可以给 Mooncake 提 issue**（ionic 上这组参数导致发送卡住，有 a31 成对数据） |
| RDMA | 通知 QP 跨节点建不起（U27，P8D8） | 只在 P8D8 出现，单节点不用 | — | — | 多进程复现后再提 |
| RDMA | 同节点 HIP IPC（XGMI）传输（a30） | 发送端 GPU 有计算时只有 0.36–0.4 GB/s；部署两次没起来 | RFC Mooncake#5041 | RFC | 不做 |
| RDMA | Mooncake 并行提交 / 数组接口（执行表 4） | 待做 | [Mooncake#5680](https://github.com/kvcache-ai/Mooncake/pull/5680)、[#67656](https://github.com/vllm-project/vllm/pull/67656) | 都已合入（11-24、11-16） | 随 recipe 上 main / 新 Mooncake 自然得到 |
| RDMA | 按层 / 按分块边算边发 | 待做（大） | [#65613](https://github.com/vllm-project/vllm/pull/65613) | open 活跃 | 跟进 |
| HiCache | Tiering 共享区只注册本 worker 的槽（U19） | recipe 不用 `TieringOffloadingSpec` | [#69617](https://github.com/vllm-project/vllm/pull/69617)（我们的） | open，无 review | 第 6.3 节 |
| HiCache | PP 各 stage 槽宽统一（U11） | recipe 的切分让 stage 0 最大，默认模式恰好不出错 | [#69348](https://github.com/vllm-project/vllm/pull/69348)（同修法）、[#50653](https://github.com/vllm-project/vllm/pull/50653) | #69348 (draft) 活跃；#50653 **停滞**（u 10-29） | 第 6.5 节 |
| HiCache | Simple CPU offload 按页对齐（U20） | recipe 用 OffloadingConnector | [#66594](https://github.com/vllm-project/vllm/pull/66594) | open 放缓（u 11-17） | 留言稿待发 |
| HiCache | CPU LRU 去掉 O(n) 堆重建、合并相邻命中（执行表 5） | 覆盖层就绪，未上 P4D4 | [#69540](https://github.com/vllm-project/vllm/pull/69540)（@dorado269）、[#68845](https://github.com/vllm-project/vllm/pull/68845) | 都是 (draft)，活跃 | 上 P4D4 后把数据回馈给 PR |
| HiCache | 读回描述符每步上限 | 默认 8192 会把我们一次约 4.8 万描述符的读回拆到约 6 步 | [#65519](https://github.com/vllm-project/vllm/pull/65519) | open 活跃 | 可以在 PR 下说明大块读回的情形 |
| HiCache | 精确锁页第一版（mmap + 分块注册） | 一次卡死时出现 KFD 驱逐告警，改用 `hipHostAlloc` | 同 H2 | | |
| HiCache | nightly 清理崩溃 | 上游已修 | [#68710](https://github.com/vllm-project/vllm/pull/68710) | 已合入 | — |
| 调度 | PP4 prefill 调度（#68614 + #68560 回移植，a32） | **进行中** | 见 S1 | | 看 a32 |
| 调度 | 小请求优先 / 准入算上 CPU 层命中（执行表 6） | 待做 | [#66705](https://github.com/vllm-project/vllm/pull/66705) (draft) 活跃、[#66558](https://github.com/vllm-project/vllm/pull/66558) (draft) **停滞**、[#66361](https://github.com/vllm-project/vllm/pull/66361) 活跃、[#65339](https://github.com/vllm-project/vllm/pull/65339) **停滞** | | 做完 A/B 后把数据给对应 PR |
| 调度 | decode `--watermark 0.03` | 与前缀缓存效果相同，选了前缀缓存 | — | | |
| 调度 | decode `--gpu-memory-utilization 0.97`（a27） | +0.2%，波动内 | — | | |
| 调度 | 抢占时只缺草稿块就不带草稿 | 只有模拟器数据 | [#63079](https://github.com/vllm-project/vllm/pull/63079) | open 放缓 | 不做 |
| Kernel | MXFP4 indexer K cache（a33） | **进行中**；改数值，只能用于目标模型层 | [#61985](https://github.com/vllm-project/vllm/pull/61985) | open 活跃（u 11-19） | 看 a33；草稿层要拆开 |
| Kernel | 目标模型 dense 层在线 FP8（执行表 9b） | 准备中；改数值 | 机制已在 v0.34.0（[#51392](https://github.com/vllm-project/vllm/pull/51392)）；相关 [#67381](https://github.com/vllm-project/vllm/pull/67381) | #51392 已合入 | 看 eval 与成对数据 |
| Kernel | MoE 填充行跳过（执行表 7） | 待做 | [#66799](https://github.com/vllm-project/vllm/pull/66799) | 已合入 11-14 | **核对过**：它的屏蔽不经过 AITER 融合共享专家的路由，要单独看 aiter 路径 |
| Kernel | vLLM 自带 top-k（a15） | 不如 Triton | — | | |
| Kernel | hipBLASLt 参与 GEMM 重调（a12） | 只再快约 0.1 ms | — | | |
| Kernel | KF sparse MLA / 融合 indexer、采样 kernel、KF MoE gemm1 | 慢于现有实现或收益 < 0.3% | — | | 不做 |
| Kernel | 草稿稀疏 LM head | 违反 InferenceX Check 13 | [#67897](https://github.com/vllm-project/vllm/pull/67897) | open | 不做 |
| 并行 | DP + EP + 融合共享专家结果错误（U17） | recipe 不开 EP（单节点 EP 更慢） | [#69616](https://github.com/vllm-project/vllm/pull/69616)（我们的） | open，无 review | 第 6.2 节 |
| 并行 | MoRI 接收缓冲区切片（U18） | recipe 不用 MoRI | [#63412](https://github.com/vllm-project/vllm/pull/63412) | open **停滞**（u 11-11） | 留言稿待发 |
| 并行 | EP / MoRI / DBO（单节点） | 都比默认慢 | — | | 不做 |
| 并行 | decode DCP | DP4 × TP1 不能开；公开数据慢 2–9% | [#65685](https://github.com/vllm-project/vllm/pull/65685) 已合入、[#68935](https://github.com/vllm-project/vllm/pull/68935) (draft) | | 不做 |
| 并行 | 三次 all-gather 合一（topk ids / weights 写进 FP4 量化缓冲区，执行表 8） | 待做 | 无 | | 并进 U33 的 PR 或其后续 |
| Infera | Python 路由分词移出事件循环（U15） | recipe 用 Rust 路由 | [Infera#249](https://github.com/AMD-AGI/Infera/pull/249) | ready | 不受影响 |
| Infera | 路由连接池空闲超时（U24） | recipe 仍用 keep-alive 120 绕法 | [Infera#247](https://github.com/AMD-AGI/Infera/pull/247) | ready | 第 6.7 节 |
| 稳定性 | decode 0.95 下偶发崩溃（U29） | 观察记录 | — | | 第 6.8 节 |
| 稳定性 | 锁页 CPU 层超过物理内存时被 OOM 杀（U10）、主机侧偶发崩溃（U21） | 部署上控制 / 未定位 | — | | 暂不提 |

---

## 5. 上游没有的项：怎么提

### 5.1 必须先进上游、才能用于正式 InferenceX 成绩的（按收益排序）

| 优先 | 项 | 提到哪 | 怎么提 | 已有证据 | 还缺什么 |
|---|---|---|---|---|---|
| 1 | K1 sync skip（+9.1%） | vLLM | **不新开**：#58710 已做第一部分，在它下面留言附 MI355X 数据（留言稿已有），请作者 / reviewer 推进；第二部分 main 已有 | a1 成对 +9.1%、GPU 忙碌 99.7%、门禁干净 | 你决定发不发；最好在 main 上用 #58710 的分支跑一次单测 + 小规模 A/B，留言更有分量 |
| 2 | S5 KV 感知 DP 均衡（U32） | vLLM `[Core][DP]` | 新 PR：在 `DPLBAsyncMPClient` 的打分里加 `W × kv_cache_usage`，选中后在下一份快照前把该 rank 的用量预加上；默认关、由参数打开（vLLM 偏好参数而非环境变量，正文要说明为什么不能自动） | a5 +1.2%（与 GEMM 表一起）；KV 满的时间 53% → 47%、抢占减半 | **单独的成对 A/B**（现在和 GEMM 表混在一起）；上游会要通用负载（非 PD）的数据或说明 |
| 3 | K3 Triton decode top-k（新编 U34） | vLLM `[ROCm][Perf]` | 新 PR：DSA indexer decode 路径加 Triton top-k，ROCm 默认或开关；正文引用 sglang#47872，**署名 @dorado269**（Co-authored-by），并说明与 #69700（gfx942）的关系 | a18 +1.4%、生产 716 万行影子比对 0 个不精确、kernel 基准与单测 | 异常节点上比 aiter 慢（v3-071），需要回退或说明；在 main 上重测；与 #67044（确定性 top-k）不冲突的说明 |
| 4 | P3 FP4 dispatch（U33） | vLLM `[ROCm][Perf][MoE]` | 新 PR：naive DP 分发在 MXFP4 W4A4 专家时先量化再 all-gather（照 MoRI #59596 的做法）；把"128 < token ≤ 2048"的硬阈值改成由 aiter 报告该 M 的 kernel 是否接受预量化输入 | a7 +0.6%、门禁干净、hidden AG 20–29 → 5–8 µs | **与 U17（#69616）改同两个文件**，先等 U17 合入或在正文里说明；依赖 U13 才走 AITER AG（第 6.1 节） |
| 5 | P2 U13 | vLLM | 按 `BRIEFING` 第 4 节：改写版重测后从 fork 开上游 | J8 −2.0% 至 −3.1% | 在 main + a25 类配置上重测 |
| 6 | H2 U26 精确锁页 | vLLM | **不新开**：#60909 已做同一件事，留言附 1.9 TB 数据、提出帮测（留言稿已有） | m10 / f1 / f2 / J3 / J3b / a 轮全部用它 | 你决定发不发；先核对 #60909 是否覆盖 OffloadingConnector 的私有张量 |
| 7 | K2 U12 | vLLM | **不新开**：#68596 活跃，留言附数据（留言稿已有） | v0.34.0 每步 −2.7% 至 −3.9% | 若 a33（#61985）被采用，U12 就不在 recipe 路径上了，留言的意义下降 |
| 8 | K4 decode GEMM 调优表（新编 U35） | aiter | 向 aiter 提 GLM-5.2 decode 形状的调优表 | a5 里 | **要先给 `gh` 做 ROCm 组织的 SAML 授权**，才能查重和提交 |
| 9 | R3 GEM 分配器 + `gem_mooncake` 连接器（U1 / U2） | Mooncake → vLLM / Infera | 先让 Mooncake#5584 合入（建议改成 VMM 导入）；再在 Infera 提 `gem_mooncake` 连接器的 PR；vLLM 侧 U1（MooncakeConnector 支持自定义 KV 池）视需要 | 所有 P4D4 轮都用它 | `LD_PRELOAD` 一个分配器算不算"引擎补丁"要和 InferenceX 确认 |
| 10 | R2 Mooncake#4541 | Mooncake | 已开，跟进 | | |

### 5.2 只是配置、要写进 vLLM recipe 的
S1（prefill 分块 / 阈值）、S2（decode 前缀缓存）、S3（decode mnbt 2048）、S6（路由参数）、S7、S8、H1、H3、H4、P1、R5、D2：
InferenceX 要求"所有生产有用的参数写进已合入的 vLLM recipe"。建议等第 5.1 节的代码项进了 main 之后，一次性提一个 GLM-5.2 MI355X PD recipe（vLLM 的 recipes / docs，以及 InferenceX 的配置）。

### 5.3 不必单独提、但可以把数据回馈给现有 PR 的
#68560 / #68614（a32 结果）、#69540 / #68845（上 P4D4 后）、#65519（大块读回会被拆步）、#61985（a33 结果；草稿层要拆开）、#66705 / #66558（做完 A/B 后）、Mooncake#4542（ionic 上发送卡住，可以提 issue）。

---

## 6. 最优 recipe 对已提 PR 的影响

### 6.1 U13（fork #1）：**实现要改；价值比当初测的更大；FP4 dispatch 依赖它**
- recipe 跑在 v0.34.0 上，用的是 **U13 原版**（"`NONE` 且正在捕获就当 FULL"）。v0.34.0 捕获时模式确实是 `NONE`，所以原版在 recipe 里是对的、生效的。
- 上 main 时要换成**改写版**（main 的 #68006 已让捕获时模式为 FULL，只剩去掉 `uniform` 要求那一半），这一点与 `BRIEFING` 第 4 节一致。
- **新发现的依赖**：FP4 dispatch（U33）把 FP4 数据和 scale 以 BF16 视图做 all-gather，"仍走 aiter 自定义 all-gather"（`README` a7 设计一节）。没有 U13，FULL graph 里这条 all-gather 走 RCCL，
  FP4 dispatch 测到的 +0.6% 就不一定成立。所以 **U33 的 PR 要写明依赖 U13**，或者在 U13 合入后再测。
- **价值变大**：U13 当初在每步约 52 ms 的配置上测的（−2–3%）。现在每步约 42 ms，通信时间的占比更高，U13 的相对收益可能更大。重测应该用 a25 类配置（DP4 decode + FP4 dispatch + Triton top-k），不再用 J8 的配置。

### 6.2 U17（#69616）：**结论不变；recipe 不触发；与 FP4 dispatch 改同两个文件**
- recipe 的 decode **不开 EP**（`--enable-expert-parallel` 没设），融合共享专家的 meta 缓冲区 `is_EP=False`，U17 的函数直接返回，不起作用。单节点 EP 已判断更慢，recipe 短期不会开 EP。
- 所以 recipe 既不证明也不否定 U17；U17 的证据（J8：0 / 200 → 196 / 200）仍是在 DP4 + EP 下测的，结论不变。
- **耦合**：FP4 dispatch 覆盖层改的正是 `naive_dp_ep.py` 的 `prepare` 和 `rocm_aiter_moe.py`。两者语义互不影响（一个改 hidden 的量化，一个改 `topk_ids` 的共享专家列），但上游会有文本冲突；先提的那个合入后，另一个要 rebase。
- 核对过：main 上已合入的 #66799（DP 填充行的 expert id 设 −1）**不经过** AITER 融合共享专家的路由（`aiter_shared_routed_fused_moe_router.py` 不用 padding mask），与 U17 无交互。

### 6.3 U19（#69617）：**结论不变；recipe 不触发**
- recipe 的 CPU 层是 `OffloadingConnector` 默认的 `CPUOffloadingSpec`。v0.34.0 在 ROCm 上**不用共享区**（`_uses_shared_region()` 在 ROCm 返回 False），每个 worker 用私有张量 + U26 精确锁页。
  U19 改的 `pin_mmap_region` 只在 `TieringOffloadingSpec`（ROCm 上唯一用共享区的）里走到。U19 的证据来自 B 线的 Tiering 实验，结论不受影响。
- #69254 若合入，ROCm 上 `CPUOffloadingSpec` 会在容量够时改用共享区，但那是**每个 rank 一块自己的区域**，不走"只注册本 worker 的槽"的逻辑，所以对 U19 仍无影响（影响的是 U26，见 6.6）。

### 6.4 U12（#68596）：**recipe 在用；可能被 #61985 绕开**
- recipe 的 decode 用 U12 的 gluon logits（覆盖层）。正在测的 a33（#61985 MXFP4 indexer K cache）走 aiter 的 `paged_mxfp4_mqa_logits`，**绕开 U12 和 Triton top-k**
  （覆盖层已把 Triton top-k 接到 MXFP4 路径上）。如果 a33 被采用，U12 就不在 recipe 路径上了；给 #68596 的留言要等 a33 出结果再决定措辞。

### 6.5 U11（fork #4）：**recipe 不需要；更支持关闭 fork #4**
- recipe 用默认 spec + PP4，切分 20,20,20,18 让 **stage 0 的块最大**（796,416 字节）。默认模式下调度器按 stage 0 的大小分配 chunk 编号，其他 stage 的块更小、能放的 chunk 更多，
  所以编号不会越界——**恰好没出 #69348 描述的问题**。如果改成让后面的 stage 更大（比如层数或 MTP 层的位置变了），默认模式就会越界。
- #69348 合入后所有 stage 都按最大的（就是 stage 0）计算，我们的有效容量不变。
- 结论：U11 / #69348 对当前 recipe 没有影响，与"关闭 fork #4、在 #69348 留言"的方向一致。

### 6.6 U26（#60909）：**recipe 在用；上 main 后可能被 #69254 改变行为**
- recipe 的 1.9 TB CPU 层靠精确锁页才放得下（torch 锁页分配器按 2 的幂取整：1.25 TB 实锁 +17%，1.6 TB 时已涨到 2.6 TB 以上、主机 OOM）。
- 若 #69254 合入：ROCm 上 `CPUOffloadingSpec` 在注册量不超过 amdkfd 上限时会**改用共享区**（每 rank 一块 mmap + `host_register`），不再走私有张量。按它的公式，我们 1.9 TB（约 1,769 GiB）低于预算约 2,667 GiB（2,752 GiB 主机的上限减去每 worker 10 GiB 余量），能放下，
  所以 recipe 上 main 后会自动换路径——这条路径就是我们当初"第一版 mmap + 分块注册"的同类做法，那时一次卡死时出现过 KFD 驱逐告警（单次，因果未证实）。**上 main 时要专门验证这一点。**

### 6.7 Infera 六项（#246–#251）
| PR | recipe 里 | 结论是否变化 |
|---|---|---|
| #246 U16 串行下发 | **在用**，是基础 | 不变；之后所有 P4D4 都没有再停住 |
| #247 U24 池空闲超时 | **没用**：recipe 两侧仍是 keep-alive 120 绕法，路由二进制是否含 U24 没写 | 不变；U24 的证据仍只有 J3 / J3b（C 级） |
| #248 U14 nofile | 代码在，但 docker `--ulimit` 已给足，不起作用 | 不变 |
| #249 U15 分词移出事件循环 | **没用**（Rust 路由） | 不变 |
| #250 U28 DP 引擎均衡 | **在用**，是 KV 感知均衡（U32）的前提 | **变了**：PR 正文写"结论不明（f1 −0.8%），默认关"；之后 a3 同节点成对 **+1.1%**（最忙 / 平均 1.28 → 1.13），并且所有更好的配置都开着它。**建议更新正文**，加上 a3 和 a5 的数据；默认值可以继续保持关（只对 PD decode 有利），但不再是"结论不明" |
| #251 U7 参数解析 | **在用**（镜像是 v0.34.0） | 不变 |

### 6.8 观察记录 U29（decode 0.95 下偶发崩溃）：**原推断进一步被削弱**
- 原推断是"显存余量不够"。之后 a 系列二十多轮 3600 s 计分都在 decode 0.95 + AITER AG/RS + `hipHostAlloc` 精确锁页下运行，文档里没有记录 hipBLASLt / `OUT_OF_RESOURCES` 崩溃
  （记录到的只有 a4 prefill 的 `hipMemcpyBatchAsync` 卡死，和 a31 的 Mooncake 参数问题）。"第一版锁页引发 KFD 驱逐"的可能性继续上升。这是从文档推断，没有逐轮查日志。

### 6.9 其他
- **U2**：recipe 两侧都用 VMM 导入的 GEM 分配器，U2 的修法已在 recipe 里实际验证了所有轮次，建议把它移进 Mooncake#5584 的理由更充分了。
- **U30（sync skip）的上游**：#58710 停滞 16 天且作者写明缺 ROCm 验证，我们的 +9.1% 正好补这个缺口，是收益最大的一条。

---

## 7. 待你决定
1. 第 5.1 节的四份留言稿（#58710、#60909、#68596、#66594 / #63412）发不发；#58710 优先。
2. 新开三个 vLLM PR（U32 KV 感知 DP 均衡、U34 Triton top-k、U33 FP4 dispatch）：做不做、顺序（建议 U34 → U32 → U33，U33 等 U17 / U13）；每个都需要先在 main 上补单独的成对数据。
3. 更新 Infera#250 的正文（加 a3 / a5 数据）。
4. 给 `gh` 做 ROCm 组织的 SAML 授权（aiter 的 GEMM 表、ROCm clr issue 都卡在这里）。
5. Mooncake：给 #4542 的参数在 ionic 上卡住的现象提 issue；#5584 改成 VMM 导入。
6. 本报告是否提交到分支（现在只在本机）。

## 附录：状态判定和数据来源
- 状态按 11-25 的 `updatedAt`：活跃 ≤ 7 天，放缓 7–14 天，停滞 > 14 天。只看 PR 的更新时间，不代表有没有 reviewer 在看。
- 未查到的：`ROCm/aiter`（SAML 授权）；路由二进制 `infera-router-vllmserial-usage` 的构建提交。
- 编号：U1–U33 见 `CANDIDATES.zh-CN.md`；U34（Triton decode top-k）、U35（decode GEMM 调优表）是本报告新编，尚未写进 `CANDIDATES`。
