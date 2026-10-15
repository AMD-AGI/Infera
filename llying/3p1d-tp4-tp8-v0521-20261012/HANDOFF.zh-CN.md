# 交接（2026-10-15 03:45 UTC）—— 新会话先读这个

## 目标与当前结论
- 目标：升级镜像（sglang v0.5.21 + Mooncake 跨 rank 修复）后，3P(TP4)1D(TP8) GLM-5.2-MXFP4 在 AgentX 上的最高吞吐（tok/s/GPU）；
  用户口径的 interactivity 约束是 P50 ∈ [50,100]，但 10-14 起用户要求"以推高吞吐为目标"。
- **目前最高：3P1D C176 = 30,562 tok/s/GPU**（t4d-tok4；P50/P90 interactivity 48.9/44.7）。详见 `RESULT.zh-CN.md` §0/§3（表格由
  `scripts/tune_report.py` 生成）。最优配置 = router R1 + P 会话亲和 + R2；prefill mem 0.88 + HiCache 400/200 GB/rank；decode
  `--tokenizer-worker-num 4`（`config/best.sh`，复现：`REPRODUCE.zh-CN.md`）。
- 2P(TP4)1D C176 = 28,900 tok/s/GPU（16 卡，−5.4%，prefill 成瓶颈 + 实例间失衡）。C192（3P1D）因挂起无有效结果（部分数据未超过 C176）。

## 栈当前状态（ROUND=r2，容器前缀 glm52-llying-3p1d-r2-）
- decode-0（138，mem **0.92**，tok4，日志名 `decode-0.after-mem092.log`）、prefill-0（135，400 GB HiCache）、prefill-1（137 **card4–7**）在运行；
  **prefill-2（137 card0–3）已因第三次挂起退出**；router 在 137。`config/round.r2.sh` = `config/stage-mem092.sh`（GPU 组已对调）。
- 队列已停（`operations/tune_queue.tsv`，进度 `runtime/runs/r2/tune/queue.done`）。sampler 在 137 后台运行（`runtime/runs/r2/tune/samples.jsonl`）。
- **NFS server 日志从 10-14 21:59 起未写入**：继续前先 `bash scripts/log_follow.sh ROUND=r2`（续接日志；看 `/home` 余量）。

## 未完成 / 下一步（按优先级）
1. **prefill-2 挂起根因**（3 次，签名相同：4 个 scheduler 卡在 `process_batch_result_disagg_prefill` → `stream.synchronize()`，GPU 100%；
   py-spy 在 `runtime/runs/r2/tune/hang-prefill-2-*/`）。第三次时 prefill-2 已在 card0–3 → **跟实例走，不跟 GPU 走**。与 prefill-1 的差别：
   同节点第二个实例、`PID_OFFSET_prefill_2=60000`（prefill-1 为 30000，ROCr IPC 用 `\0hsa<pid>` 抽象 socket，长时间运行后 PID 是否会交叠？）、端口、启动顺序。
   建议：对调 PID_OFFSET / 换端口做对照；挂起时抓 RCCL 调试日志或 ROCm debug agent。
2. decode mem 0.92 的 3P1D C176 / C192（`config/stage-mem092.sh` 已就位，需先拉起 prefill-2：`restart_prefills.sh INSTANCES=prefill-2`，
   但它可能再挂）→ 看 decode KV 扩大后峰值能否超过 30,562。
3. prefill_complete（需 prefill HiCache 改 write_back；可释放 decode 中"等 KV 传输"请求占的 7–12% KV；先看 mem 0.92 有没有收益再决定，
   理论分析见 `RESULT.zh-CN.md` §4.7）；`index_share_for_mtp_iteration=true`（抬 decode 每 rank 上限，需真实接受率门禁验证正确性）；
   1P(TP8)1D（每 rank GPU KV 约 +29%，但需 137 整机）。
4. **按 InferenceX 官方协议复测最优配置**（`RESULT.zh-CN.md` §5）：官方要求每个并发点重新部署、不 flush、不复用在线服务；我们的 30,562 是在复用的栈上、
   tok4 重启后第三轮测得的，而重启后首轮 decode 每步慢 3–17%。做法：整栈重启 → 直接测 C176（只用 AgentX 自带预热），客户端最好换成官方的
   `agentx-harness==1.0.9`。另外，官方 recipe 的 decode 用 `stream-interval` 20–30（我们是 1），值得作为单独一轮测试。

## 关键数据口径（易错）
- interactivity = 1 / aiperf Full-Response ITL（每请求平均 token 间隔）的 P50/P90。
- decode 容量：每个在跑请求实际占 KV ≈ 18–21 万 token（按停留步数加权，长请求输出也长），不是平均输入 12.5 万；
  "等 KV 传输"的请求占 decode KV 7–12%（随 prefill 延迟增长），"等预分配"的请求不占 KV。
- 每 rank decode 输出在在跑 8–14 个时平在 650–720 tok/s；ITL≈20 ms 对应每 rank 约 10–11 个请求（与 P8D4 一致）。
- TTFT 分段：`scripts/sample_window.py samples.jsonl START END`；prefill 服务不报 TTFT，decode 的 `time_to_first_token` 约等于客户端 TTFT。

## 操作要点与踩过的坑
- 一轮 = `scripts/tune_round.sh`（flush → router 带开关 → AgentX → `tune_post.sh` 发布）；队列 `scripts/start_queue.sh ROUND=r2`（幂等）。
- 引擎重启顺序：**先 decode 后 prefill**（只重启 decode 会让部分传输通道失效）；重启后必须跑 `scripts/transfer_smoke.py`（后台跑再轮询）。
  重启后第一轮 decode 偏慢 3–17%，先跑 `warmup_round.sh`。HiCache 栈停后显存回收 10–30 分钟（135 的 400 GB 栈最慢）。
- 巡检：`scripts/tune_watch.py r2 10`、`scripts/hang_check.py r2 180`；census 要区分窗口内与窗口结束（取消）条目。
- **不要在命令里用 `pgrep -f` 匹配自己命令行里出现的字符串**（会自匹配）；工具层面偶尔会把同一条命令执行两次 → 启动类操作必须幂等、
  单独先检查再启动；**长时间阻塞的命令会被中断并导致会话停摆** → 长任务一律在节点后台执行、本地只用短命令轮询。
- 运行中的脚本不要原地编辑（bash 增量读取；NFS 上 mv 替换会 stale handle）。
- `/home` 是共享 10 TB 卷，曾写满导致 follower/编排退出；拷回已排除 `server_metrics_export.json`，每轮要求 ≥ 20 GB。
- 只动名字含 `llying` 且前缀匹配的容器；135 GPU1、138/136 上有其他租户。
- 过程记录：`operations/STATUS.zh-CN.md`（时间线最全）；计划与协议：`PLAN-TUNING.zh-CN.md`。
