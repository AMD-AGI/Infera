# GLM-5.2 MXFP4 8P8D：原生 ATOM + Infera

1 个 Prefill（TP8 + DPA，8 卡）与 1 个 Decode（TP8 + DCP8，EP1，DPA 关闭，8 卡）的 PD 分离部署。
与 [`../8p4d-atom-infera`](../8p4d-atom-infera/README.md) 的区别只在 decode：8P4D 的 decode（TP4 + DCP4）
在 AgentX C80 下 KV 容量不足（见 `../8p4d-atom-infera/results/README.md`），本套件把 decode 扩展到 8 卡。
镜像与 patch 与 8P4D 相同，由 `../8p4d-atom-infera/scripts/build_image.sh` 构建。过程与问题见 `.record/`。

## 目录

- `config.sh`：节点、IP、镜像、模型、端口、引擎与 AgentX 参数；所有脚本支持用 `KEY=VALUE` 覆盖。
- `scripts/`：`up.sh`、`down.sh`、`smoke.sh`、`agentx.sh`、`sweep.sh`、`summarize.py`，用法与 8P4D 相同。
- `results/`：通过的 AgentX 聚合 JSON 与汇总表。
- `.tmp/`（不入库）：运行目录 `runs/<RUN_ID>/`、缓存与验证脚本（`gsm8k.sh`、`needle_probe.sh`、
  `offload_probe.sh`、`kv_transfer_probe.sh`、`decode_probe.sh`、`vram_sample.sh`）。

## 默认配置

- Prefill：GPU 0-7，`-tp 8 --enable-dp-attention --enforce-eager --max-num-seqs 512`，与 8P4D 相同：
  会话亲和（`ATOM_DP_SESSION_AFFINITY=1`）、matched rails、LMCache CPU 卸载（每个 DP rank `PREFILL_OFFLOAD_GB=160` GiB）。
- Decode：GPU 0-7，`-tp 8 --decode-context-parallel-size 8`，`--max-num-seqs` 与 cudagraph 尺寸按 `2*CONC`，
  每个 rank 只用本卡 `ionic_<gpu>`。
- 共 16 张 GPU；MTP、端口、AgentX 参数与 8P4D 相同，容器名前缀为 `glm52-8p8d-atom`。
- 显存比例：`GPU_MEM_UTIL=0.85`，可分别用 `PREFILL_MEM_UTIL`、`DECODE_MEM_UTIL` 设置。amdgpu 6.19.x 的节点上，
  RDMA 注册后固定的 KV 显存会被重复计入，需要降低比例（见 `../8p4d-atom-infera/.record/issues.md` 第 10 条）。

## 运行

脚本在控制节点（decode 节点）执行；长时间任务用 `nohup setsid` 脱离 SSH 会话：

```bash
cd /apps/tas/yaoc/research/topic/glm-5.2-pd-opt/Infera-glm-5.2-2p1d/yaocheng/8p8d-atom-infera

bash scripts/up.sh CONC=80 MTP_AL=           # 关闭 forced acceptance，用于精度检查
bash scripts/smoke.sh
bash .tmp/gsm8k.sh LIMIT=200
bash scripts/down.sh

# AgentX：每档冷启动，结束后写 results/summary.{md,csv}
nohup setsid bash scripts/sweep.sh POINTS=80 > .tmp/logs/sweep.log 2>&1 < /dev/null &
```
