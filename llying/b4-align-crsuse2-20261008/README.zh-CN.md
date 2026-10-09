# B4 在 crsuse2 上的性能对齐

目标：在本集群用 1P1D C80 复现 aus 集群的 B4（"原评分；P完成释放；P/D绑定；D radix开；
Triton"），看性能能否对齐。B4 的配置与结果见
`../router-capacity-campaign-20260928/b4/`，汇总指标见 `../experiment-summary-20261008/`。

## 与 B4 的关系

软件按 B4 的固定输入重建，配方参数逐项与 B4 展开后的环境一致；与 B4 不同的只有：

| 项 | B4（aus） | 本集群 | 原因 |
|---|---|---|---|
| Mooncake | `faae8dd4` | `faae8dd4` + v3 修复（目的端固定 rail、固定 HCA、ACK 超时 1.07 s） | B4 关闭 DP rank 亲和，有跨 rank 传输；本集群原版 Mooncake 会 transfer failure（`../cross-rank-rca-20260921/`） |
| `MOONCAKE_DISABLE_HIP_DMABUF` | 1 | 0 | 两个集群网卡驱动不同 |
| `--disaggregation-ib-device` | 每 rank 一张：`{"k":"ionic_k"}` | 每 rank 共享 `ionic_0,…,ionic_7` | crsuse2 rail 隔离（`ionic_k` 只通 `ionic_k`），单网卡下 i≠j 不可达（第二轮实测 CQE12） |
| 节点 | 物理机 n04-33 / n05-21 | 虚拟机 crsuse2（默认 P 136、D 138） | |
| router 二进制 | `19a6c1d2`，sha256 `71750540…` | 同一 commit 重新编译，`4077025d…` | 编译环境不同 |
| SSH 选项 | — | 加 `LogLevel=ERROR` | 首次连接的主机密钥提示会混进解析的输出 |

数据集（Weka `23f152f6…` 离线缓存）与 B4 记录的清单逐文件 SHA256 一致。客户端设置由 aus 的
校验脚本把握：基线是 B4 自己的 `runtime.env`（`build/make_reference.sh` 只改写 `MODEL`、
`HF_HOME`、`INFMAX_CONTAINER_WORKSPACE`、`UV_CONSTRAINT` 四个路径），其余任何键与 B4 不同，
运行在加压前就会失败。

## 镜像链

`build/build_images.sh` 在一台节点上依次构建（tag 沿用 aus 的名字）：

1. `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260916@sha256:eef7b70e…` + Infera
   `83e0f6c8`（`deploy/docker/Dockerfile.sglang`）→ `v0519-llying-aus-base-83e0f6c8`
2. yihou overlay（NextN 修复、PR #37152）→ `v0519-llying-aus-0922-nextnfix-hicache`
3. tracing（`aus_diag`）→ `aus-0922-reqtrace`
4. p01/p02（MTP 下允许 D radix）→ `aus-campaign-radix-20260928`
5. Mooncake v3（`../cross-rank-rca-20260921/fix/Dockerfile`）→
   `aus-campaign-radix-20260928-mcdestpin-ibto18`

本集群原有的 `v0519-yihou-0917-nextnfix-hicache` 与此只差 SGLang 一天的 nightly（63 个上游
提交，其中只有 #39050 "HiCache write-through 备份按步合并提交"作用于 B4 的 Prefill
HiCache），所以按 aus 配方重建而不复用它。

## 目录

- `build/`：镜像链、router（`build_router.sh`）、`runtime/` 组装（`assemble_runtime.sh`）、
  D 端前缀诊断文件（`make_decode_diag.sh`）、客户端校验基线（`make_reference.sh`）
- `runtime/`：B4 的 `TRACE_RUNTIME` 布局；脚本是 aus 套件的原样副本，来源与 SHA256 见
  `runtime/MANIFEST.tsv`；`cache/`、`runs/` 不进 git
- `config/b4.crsuse2.sh`：B4 展开后的环境，本集群绑定项集中在文件开头
- `scripts/run_b4.sh`：按 B4 的流程运行；只替换了 aus 依赖 Slurm 的启动脚本
- `artifacts/`：router 二进制（不进 git）
- `operations/STATUS.zh-CN.md`：工作状态

## 运行

两台节点都需要最终镜像（ID 见 `config/b4.crsuse2.sh` 的 `EXPECTED_IMAGE_ID`）且 GPU 空闲；
`runtime/cache/agentx/hf/hub` 需有 Weka `23f152f6…` 快照（从节点的
`/tmp/inferencex-agentic-liyingli/hf/hub` 整体复制）。
在 Prefill 节点上无人值守运行：

```bash
B4=/home/liyingli/bench_agentx/baseline/Infera/llying/b4-align-crsuse2-20261008
B4_PREFILL_NODE=crsuse2-m2m-136 B4_DECODE_NODE=crsuse2-m2m-138 \
  setsid nohup bash "$B4/scripts/run_b4.sh" all > "$B4/operations/<name>.log" 2>&1 < /dev/null &
# 结束或失败后（日志第一行给出 B4_RUN_ID）只删除本实验容器：
B4_RUN_ID=<id> B4_PREFILL_NODE=... B4_DECODE_NODE=... bash "$B4/scripts/run_b4.sh" stop
```

`all` 依次为：prepare（空闲、镜像、端口检查）→ launch（etcd、collector、P、真实接受率的
D、router）→ gate（已知答案的长前缀请求，直到覆盖全部 P/D rank）→ switch（D 改为模拟接受长度 3.61，
重启 router）→ preflight（`preflight_placement.py`，KV 容量与 B4 偏差 ≤ 0.1%）→ measure
（采样器 + AgentX C80，3600 秒，warmup 每 lane 10）。失败时保留服务供检查。
