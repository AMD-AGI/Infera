# U27 观察记录：Mooncake 跨节点在 P8D8 规模下建不起通知 QP（先不提 issue）

- 类型：观察记录，**不提 issue**（只出现过一次，没有最小复现）。目标仓库（以后提时）：kvcache-ai/Mooncake。
- 查重（11-20）：Mooncake 仓库没有 "Notification QP initialization failed" / `-302` 的 issue；相关的只有
  [kvcache-ai/Mooncake#4061](https://github.com/kvcache-ai/Mooncake/issues/4061)（RFC：RDMA 通知通道的容错）。
- 版本：实验镜像 `infera-vllm:v0340-0c5e5e89` 里的 Mooncake 由 `llying/vllm-image-v0340-20261117/build_mooncake.sh` 构建，脚本默认
  `MC_REF=v0.3.14`；镜像构建时是否覆盖了 `MC_REF` **待核实**（要在作业里进镜像看 `[mc-build] Mooncake <commit>` 或库的版本）。主线 README 原记"比 `faae8dd4` 新"。

## 现象（11-18 22:46，第一次 P8D8，单次）

- 配置：两台节点，prefill PP8（n3）+ decode DP8（n4），vLLM MooncakeConnector（Infera GEM 子类），8 张 ionic 网卡，C208。
- smoke：chat / completion / stream 正常；60K token 的冷长前缀请求在 decode 拉 KV 时失败（`Mooncake transfer engine returned -1` → HTTP 500）。
- prefill 侧日志：`rdma_endpoint.cpp:159 Notification QP initialization failed: -302` **1,272 次**（`-302` 是 `ERR_MEMORY`，
  `mooncake-transfer-engine/include/error.h`）；decode 侧 **314 次** "Re-establish connection … waiting ready ACK" 后放弃。
- 同节点回环（P4D4，m14 等）这两类错误都是 0。
- 原始日志：主线 P4D4 的 11-18 22:46 那轮（README 1s）。

## 隔离（单次）

| 测试 | 结果 |
|---|---|
| 单进程主机内存跨节点（`../vllm-pd-tools-20261117/scripts/mc_xnode.sh` / `mc_xnode.py`，n3 写 n4，与 vLLM 方向相同），1 张网卡，默认设置 | 正常，30 GB/s，内容校验通过 |
| 同上，关掉通知 QP（`MC_RDMA_NOTIFY_ENABLED=0`） | 正常，29.8 GB/s |
| 同上，8 张网卡同时开 | 正常但只有约 4 GB/s（未查，可能是主机内存所在 NUMA 与网卡的对应） |
| P8D8 两边 `MC_RDMA_NOTIFY_ENABLED=0`（作业 298394，v3-207 prefill PP8 + v3-161 decode DP8，11-19） | smoke 通过（含 60K 冷长前缀），AgentX 预热 2,302 个请求全部完成、0 错误；测量开始 9 min 后作业被外部 SIGKILL（调度层，与本问题无关） |

## 判断（推断）

- 失败与规模有关：P8D8 是 8 × 8 个 worker 全互连、每对跨多张网卡，单进程测试没有这么多端点。推测"端点数 × 每个端点的通知 QP 资源"超过了某个上限，
  在 Mooncake 分配通知 QP 的缓冲区 / MR 时返回 `ERR_MEMORY`。具体是哪一步分配失败、上限来自 HCA（QP / CQ / MR 数）还是锁页内存，**未确认**。
- vLLM 的 MooncakeConnector 用自己的 ZMQ 通道通知传输完成，不依赖 Mooncake 的通知 QP，所以关掉它是安全的绕法（已在 P8D8 验证）。

## 提 issue 之前要做的

1. 在作业里确认镜像里 Mooncake 的确切提交，读该版本 `rdma_endpoint.cpp` 里通知 QP 的初始化，找出 `-302` 的来源。
2. 做一个多进程复现：两台节点各起 8 个进程、每个进程开 8 张网卡，全互连传一次，看通知 QP 的失败数随进程数 / 网卡数怎么变；同时记录 `ibv_devinfo -v` 的
   `max_qp` / `max_cq` / `max_mr` 和 `ulimit -l`。
3. 复现后开 Mooncake issue（附版本、复现脚本、失败数随规模的变化、`MC_RDMA_NOTIFY_ENABLED=0` 的绕法）；vLLM 侧可以建议 MooncakeConnector 默认关掉通知 QP
   （它用不到），或在文档里写明。
