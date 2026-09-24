# 实验记录

时间均为 UTC。之前的过程见 [../../8p4d-atom-infera/.record/progress.md](../../8p4d-atom-infera/.record/progress.md)。

## 2026-09-24

- 04:48 用户要求：8P4D 的 decode KV 容量不足，decode 改为 TP8 + DCP8 后在 C80 重新测试，环境放在 `yaocheng/8p8d-atom-infera`。
- 04:51 由 `../8p4d-atom-infera` 复制 `config.sh`、`scripts/` 与 `.tmp/` 下的验证脚本，改动：decode `-tp 8 --decode-context-parallel-size 8`、GPU 0-7；容器前缀 `glm52-8p8d-atom`；AgentX 元数据 `DECODE_TP=8`、`DECODE_DCP_SIZE=8`；汇总表标题。镜像与 patch 沿用 8P4D（`infera-atom:nightly_202609221542`，`sha256:b709d3d3fc3f…`，两节点已就绪）。
- 节点（作业 31690）：prefill `smci355-ccs-aus-n10-29`（amdgpu 6.14.14，0.85，LMCache 每 rank 160 GiB）；decode 与控制节点 `smci355-ccs-aus-n04-21`（amdgpu 6.19.14，`DECODE_MEM_UTIL` 需要降低）。
- 04:56 8P4D 的 AgentX 聚合缺少 `KV_OFFLOAD_BACKEND_METADATA`（8P4D issues.md 第 12 条），本套件的 `agentx.sh` 与 `config.sh`（`LMCACHE_VERSION`）同步修改。
- 05:01 精度验证：`up.sh CONC=80 MTP_AL= DECODE_MEM_UTIL=0.62 RUN_ID=acc-c80`（`.tmp/logs/acc-validate.log`）。
  - decode（TP8 + DCP8，0.62）05:04 就绪：每 rank `peak_torch=60.60GB`、`non_torch` 31.2-32.6 GB，KV 79.0-80.5 GB（108947-110948 个 block），DCP8 合计约 1400 万 token，为 8P4D decode（0.74）的 2.4 倍；42 个 cudagraph，内存池 0.85 GB。按新驱动的重复计入模型，余量约 34 GB。
  - 05:07 服务就绪；smoke 通过；GSM8K 200 题 0.985 ± 0.009；大海捞针 3/3。
- 05:08 C80 正式测试开始：`sweep.sh POINTS=80 SWEEP_ID=c80-lmcache DECODE_MEM_UTIL=0.62`（3600 s，每 lane 预热 10，MTP K3、forced acceptance 2.99），日志 `.tmp/logs/sweep-c80-lmcache.log`。
- 05:15-05:30 AgentX 首次安装（InferenceX 克隆、aiperf venv、数据集）后开始预热；05:30:54 预热完成（884 个请求，错误 0，用时 902 s），测量开始。
- 05:33 测量开始后 2 分钟：TTFT p50 2.67 s，ITL p50 8 ms；decode 运行 13-17 个请求、等待 0-5 个，KV 占用 4.8%-10.6%，prefix 命中率 81%-83%。
- 05:55:40 测量 24:45 时的实时指标：4013 个请求全部成功，最近窗口 TTFT p50 5.08 s，ITL p50 12 ms，intvty p50 86。
- 05:55 作业 31690 收到抢占信号（issues.md 第 1 条），`sweep.sh` 与 `agentx.sh` 在节点上被终止，06:00:48 作业结束（sacct：`PREEMPTED`，运行 3:41:52），随后重新排队。服务容器运行到约 06:01:50，之后 AgentX 无法连通路由（16 个错误）。测量未完成 3600 s，没有聚合 JSON。
- 06:25 按测量开始后前 30 分钟计算部分指标（`.tmp/window_metrics.py`，8P4D 取同一窗口，方法对 8P4D 完整一小时的结果与聚合 JSON 一致）：8P8D 16 卡完成 4809 个请求，total 19056 tok/s/GPU，output 161.7，TTFT p50 5.45 s / p90 13.15 s，ITL p50 11.45 ms；8P4D 12 卡为 3508、17778、156.2、17.95 s / 30.97 s、16.12 ms。写入 `results/README.md`。
- 06:30 初步瓶颈分析与 yihou P8D8（SGLang，16 卡）C80 对比（`results/README.md`）：
  - 8P8D 的 total/output tok/s/GPU 为 19056 / 161.7，对方为 19513 / 157.7；ITL p50/p90 为 11.45 / 12.51 ms，对方为 14.89 / 21.28 ms；TTFT p50 相近（5.45 对 5.69 s）。
  - `.tmp/ttft_breakdown.py`：新增 token 少于 16k 的请求（92%）TTFT p50 约 5 s，与新增量无关，说明 TTFT 主要是等待。
  - 8P4D C80 的 prefill 日志：各 rank 平均 prompt 吞吐 1.4k-1.9k tok/s，13%-21% 的采样点有排队（最多 6-13 个）。
  - 初步判断瓶颈在 prefill 端（8 卡 DPA）：同为 8 卡 prefill 的 SGLang P8D8 TTFT p50 也约 5.7 s，16 卡 prefill 的 2p1d 参考为 2.4 s。
