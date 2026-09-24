# GLM-5.2 MXFP4 8P4D：原生 ATOM + Infera

1 个 Prefill（TP8 + DPA，8 卡）与 1 个 Decode（TP4 + DCP4，EP1，DPA 关闭，4 卡）的 PD 分离部署。
KV 由 ATOM 的 mooncake connector 经 RDMA 传输，Infera Python 路由按 `atom-mooncake` 协议配对请求；
AgentX 测试使用与 [`../2p1d-sweep-triton-dsa-20260922`](../2p1d-sweep-triton-dsa-20260922/README.md)
相同的 InferenceX commit。计划见 [`plan/plan.md`](plan/plan.md)，过程与问题见 `.record/`。

## 目录

- `docker/Dockerfile`：基于 `rocm/atom-dev:nightly_202609221542` 的 Infera ATOM 镜像，构建时打入 `patch/`。
- `patch/atom-moe-eager-decode-pad.diff`：ATOM 的 DP MoE all-gather 在 eager decode 步按实际行数填充，
  使 prefill（DPA + MTP + `--enforce-eager`）不再触发 `MoE was handed N rows` 断言，见 `.record/issues.md` 第 5 条。
- `patch/atom-dcp-kv-staging.diff`：producer 向 DCP consumer 发送 MLA KV 时先在 GPU 上收集每个 rank 的 token，
  整页 RDMA 写出，替代逐 token 描述符；10 万 token 传输由约 5.3 s 降到约 2 s，见 `.record/issues.md` 第 6 条。
- `patch/infera-forward-session-headers.diff`：Infera PD 路由把客户端的 `X-Correlation-ID` 等会话头转发给
  prefill 与 decode，供 ATOM 的 DP 会话亲和使用，见 `.record/issues.md` 第 8 条。
  `atom-*.diff` 应用到 `/app/ATOM`，`infera-*.diff` 应用到 `/opt/infera`。
- `config.sh`：节点、IP、镜像、模型、端口、引擎与 AgentX 参数；所有脚本接受 `KEY=VALUE` 覆盖。
- `scripts/`：`build_image.sh`、`up.sh`、`down.sh`、`smoke.sh`、`agentx.sh`、`sweep.sh`、`summarize.py`。
- `results/`：通过的 AgentX 聚合 JSON 与汇总表。
- `.tmp/`（不入库）：运行目录 `runs/<RUN_ID>/`（容器日志、`docker inspect`、AgentX 原始数据）、缓存、临时脚本。

## 默认配置

- Prefill：`smci355-ccs-aus-n04-25`，GPU 0-7，`-tp 8 --enable-dp-attention --enforce-eager --max-num-seqs 512`，
  kv_producer，握手端口 21301-21308（每个 DP rank 一个）。prefill 前向本身不用 cudagraph；
  graph 模式的对比测试显示每请求延迟与 eager 相同，默认保留 eager。
  `ATOM_DP_SESSION_AFFINITY=1`：同一 AgentX 会话的各轮固定在一个 DP rank，复用该 rank 的 prefix cache。
- Decode：`smci355-ccs-aus-n04-29`，GPU 0-3，`-tp 4 --decode-context-parallel-size 4`，
  `--max-num-seqs` 与 cudagraph 尺寸按 `2*CONC`；kv_consumer 每个 rank 只用本卡 `ionic_<gpu>`。
- 两节点后端网络按 rail 隔离，prefill 设置 `ATOM_MOONCAKE_MATCHED_RAILS=auto`，按 decode rank 的 rail 写入。
- 两侧：FP8 KV、block 16、prefix caching、MTP（CONC < 48 为 K4、acceptance 3.33，否则 K3、2.99；
  `MTP_AL=` 关闭 forced acceptance）、在线 PTPC FP8 量化（层 0-77 的专家与 MTP 层 78 保持原精度）。
- etcd（23379）、路由（18000）、AgentX 客户端与镜像构建在 n04-29；prefill 19001、decode 19002，
  端口均在临时端口范围（32768-60999）和 Mooncake 随机端口（15000-17000）之外。
- 模型 `/apps/data/models/GLM-5.2-MXFP4`。

## 运行

脚本在控制节点 n04-29 执行；长时间任务用 `nohup setsid` 脱离 SSH 会话：

```bash
ssh -J cyao1002@dccs-1334-slurm.prov.aus.ccs.cpe.ice.amd.com smci355-ccs-aus-n04-29
cd /apps/tas/yaoc/research/topic/glm-5.2-pd-opt/Infera-glm-5.2-2p1d/yaocheng/8p4d-atom-infera

bash scripts/build_image.sh                  # 构建镜像并同步到 prefill 节点
bash scripts/up.sh CONC=16                   # 冷启动约 8 分钟
bash scripts/smoke.sh                        # completions 与流式 chat，打印 Mooncake RDMA 初始化日志
bash scripts/agentx.sh CONC=16 DURATION=600  # 单档 AgentX，短于 900 秒时带 --unsafe-override
bash scripts/down.sh                         # 保存日志并删除容器

# 完整测试：并发档位与 2p1d 参考相同（80 112 144 192 256），每档冷启动，结束后写 results/summary.{md,csv}
nohup setsid bash scripts/sweep.sh > .tmp/logs/sweep.log 2>&1 < /dev/null &
```
