# GLM-5.2 MXFP4：2P1D Triton DSA 性能测试（2026-09-22）

从 [`../2p1d-sweep/`](../2p1d-sweep/README.md) 复制运行配置、拓扑、完整测试脚本和
补丁资料，用于测试 Triton DSA backend。所有 Prefill、Decode worker 默认传入：

```text
--dsa-prefill-backend triton --dsa-decode-backend triton
```

默认值在 [`config.sh`](config.sh) 中定义，由启动脚本传递给 worker，
AgentX 会核对实际 worker 参数。本轮已于 2026-09-22 12:48 UTC 启动，运行编号
`2p1d-triton-dsa-20260922T124849Z`；当前进度见 [`.tmp/active_run.json`](.tmp/active_run.json)
及该轮 `status.txt`。`.tmp/` 和 `results/` 从空目录开始，沿用源目录的访问 ACL。
旧实验的结果和日志保留在原套件中。
镜像和模型复用原环境，不随目录复制。

## 默认配置

| 配置 | 默认值 |
|---|---|
| Prefill | crsuse2-m2m-137、crsuse2-m2m-138 |
| Decode | crsuse2-m2m-136 |
| 控制节点 / SSH 账号 | crsuse2-m2m-137 / xiaobche |
| GPU | 每个 worker TP8/DP8/DPA，共 24 张 MI355X |
| DSA Prefill / Decode | triton / triton |
| 镜像 | infera-sglang:v0519-yihou-0917-nextnfix-hicache，沿用原镜像 ID 校验 |
| Prefill HiCache / Decode HiCache | 开 / 关 |
| Decode MTP | EAGLE，5 steps / 6 draft tokens / topk 1 |
| 模拟接受长度 | 3.61；用于性能比较 |
| 并发档位 | 80、112、144、192、256 |
| 每档测量 / 预热 | 3600 秒 / 10 requests per lane |
| 容器前缀 | glm52-pd-yaocheng-2p1d-triton-dsa-20260922 |
| 运行日志、缓存、原始数据 | 本目录 `.tmp/` |
| 最终结果 | 本目录 `results/` |

节点、端口及其他性能参数沿用原套件。两套测试使用相同 GPU 和端口，需错开运行。
本轮复用旧实验固定版本的 InferenceX、AgentX 下载缓存及 AITER 编译缓存，
在本目录 `.tmp/cache/` 保存独立副本；AgentX 的虚拟环境重新创建。
Triton 编译缓存按节点和镜像隔离，保存在 `.tmp/cache/triton/`。

## 运行

```bash
cd /shared_nfs/yaoc/work/GLM-5.2-pd-opt/Infera-isl-debug/yaocheng/2p1d-sweep-triton-dsa-20260922

# 本地检查配置与拓扑，输出两项 DSA backend 参数。
bash scripts/run_full_sweep.sh DRY_RUN=1
```

原环境的计算节点将 `/shared_nfs` 挂载为只读，原套件的可写别名
`/mnt/m2m_nobackup/xiaobche/2p1d-sweep/shared-kit` 指向旧目录。
2026-09-22 已在 136/137/138 为本套件准备好下方的独立可写入口，并验证写入。
它通过 Docker NFS 数据卷和任务目录的共享挂载传播建立；`/shared_nfs` 的全局
只读设置保持原样。准备过程记录在 [`.tmp/setup_rw_mount.sh`](.tmp/setup_rw_mount.sh)。
挂载当前有效；若节点重启，应重新检查并按需执行该脚本。

```bash
# 此路径已在三个计算节点准备为本套件的共享可写挂载。
RW_KIT=/mnt/m2m_nobackup/xiaobche/2p1d-sweep-triton-dsa-20260922/shared-kit

# 检查 SSH、GPU、镜像、端口、RDMA 和 PD 补丁。
bash scripts/check.sh \
  "SWEEP_TMP_DIR=$RW_KIT/.tmp" "RESULTS_DIR=$RW_KIT/results"

# 完整五档测试，启动参数默认已经包含两个 Triton 开关。
bash scripts/run_full_sweep.sh \
  "RUN_ID=2p1d-triton-dsa-$(date -u +%Y%m%dT%H%M%SZ)" \
  RUN_PREFLIGHT=0 \
  "SWEEP_TMP_DIR=$RW_KIT/.tmp" "RESULTS_DIR=$RW_KIT/results"

# 短跑验证链路时，可使用新的 RUN_ID 并追加：
# 'SWEEP_POINTS=8' DURATION=120 AGENTX_WARMUP_REQUESTS_PER_LANE=1
```

若节点已能写本套件的 `/shared_nfs` 路径，可直接运行 `bash scripts/run_full_sweep.sh`。
所有入口接受 `KEY=VALUE` 覆盖；显式指定 backend 的写法为：

```bash
bash scripts/run_full_sweep.sh DRY_RUN=1 \
  DSA_PREFILL_BACKEND=triton DSA_DECODE_BACKEND=triton
```

每轮配置冻结在 `.tmp/results/<RUN_ID>/config.resolved.sh`。
原始数据位于 `.tmp/raw/<RUN_ID>/`，通过校验的档位发布到 `results/`。
默认整轮结束后停止服务。完整运行、恢复与指标口径见
[原套件说明](../2p1d-sweep/README.md)；使用其中的命令时，应换成本目录及本轮配置。

TileLang 基线结果见 [`../2p1d-sweep/results/`](../2p1d-sweep/results/README.md)，
PD 修复来源见 [原补丁报告](../2p1d-sweep/PD_FIXES_REPORT.md)，
补丁文件保存在本目录 [`patches/`](patches/README.md)。
