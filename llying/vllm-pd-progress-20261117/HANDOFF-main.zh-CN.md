# 交接：GLM-5.2 P4D4 on Infera + vLLM v0.34.0（主线，2026-11-18）

目标：在 crusoe 上用 Infera + vLLM（ROCm）复现并超过 infera + sglang 的 GLM-5.2 PD recipe（P4D4 C104 53,351 tok/s/GPU、P8D8 C208 54,468），
vLLM 缺的优化走 vLLM / Infera / Mooncake / ROCm 上游。本文件是主线（P4D4 计分、总体协调）的交接；A 线（decode kernel / 通信，v3-240）和
B 线（prefill GEM / 多级 offload，v3-258）各有自己的 session 与目录。

## 1. 现在在哪

- **最好的有效成绩：s9 = 36,923 tok/s/GPU（sglang 的 69%）**，s10 = 36,265（同水平，TTFT 长尾更好）。详细表见 `../vllm-p4d4-crusoe-20261117/README.zh-CN.md` 第 0、1h–1k 节。
- **当前瓶颈在 decode**：decode KV 满（每 rank p50 0.85–0.90）、每步慢（ITL 约 19 ms 对 sglang 约 14 ms）。prefill 命中已接近理论上限（现算约 5%），
  路由、KV 传输（0.19 s / 次）都不是瓶颈（README 1k）。
- 组合配置 s12b / s12c 的结果见第 4 节（本文件最后更新时补）。

## 2. 环境

- 仓库 `/home/liyingli/bench_agentx/baseline/Infera-vllm-glm52`，分支 `dev/pd_opt/vllm_glm52_agentx`（三个 session 共用，push 前 `git pull --rebase`）。
  本分支上的代码改动：`infera/engine/vllm/gem_mooncake.py`（GEM 内存池）、`infera/engine/vllm/args.py`（v0.34 适配）、
  `infera/router/disagg_protocols/vllm_mooncake.py`（Python 路由串行开关）、`rust/router/src/disagg.rs`（Rust 路由 vLLM 串行，U16）。
- 工作目录 `/home/liyingli/bench_agentx/vllm-pd-20261117/`：`scripts/`（最新脚本，分支里的副本在 `../vllm-pd-tools-20261117/scripts/`）、
  `queue/`（v3-135 的 runner）、`queue-n2/`（v3-258，**B 线专用**）、`queue-n3/`（新节点，`hold_n3.sbatch` / `q3.sh`）、`runs/<run>/`（每轮原始数据，
  含实时落盘的 `vpd-prefill/decode/router.log`）、`logs/summary.out`（每轮一行成绩）、`bin/`（`infera-router-vllmserial`、`libhip_gem_alloc-vmm.so`）、
  `overlay/`（#47897 覆盖层 v2）。
- 节点 / 作业：294199（v3-135，约 11-18 12:00 到期）；**296460（`hold_n3.sbatch`，24 h，排队中）**，起来后批处理脚本里自带 runner，`q3.sh` 投递；
  294784（v3-258，B 线）；295840（v3-240，A 线）。
- 新节点准备（参考 `queue-n2/1118-040712-Lprep-n2.run`）：从提交 `0c5e5e89` 用 `git archive` 构建 `infera-vllm:v0340-0c5e5e89`
  （Dockerfile `../vllm-image-v0340-20261117/`），`scripts/stage_model.sh /shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4 /mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4`。
  AgentX 第一次运行会在节点本地 clone InferenceX（`918524ff`）。`pd1_up.sh` 自动取节点 IP，但**队列项模板（`s9.sh`、`s10.sh` 等）里写死了 v3-135 的 IP `10.245.144.166`**（采样、监控的 URL），换节点要改。
- **登录节点只跑监控**（`scripts/monitor.sh`，打印 `AGENT_LOOP_WAKE_vpd` 事件）；分析、打包、读大文件一律经 `q.sh` / `q3.sh` 在计算节点执行
  （曾因在登录节点读几百 MB 的 JSON 卡死会话）。等待时匹配 monitor 的 `"event":"rc"` 事件，别只匹配某一个队列项。
- spur：每节点一个 exec-into-job 会话；作业步骤非零退出即结束作业；不注入宿主机 libionic；不要改正在执行的 bash 脚本（`pd1_up.sh` 每轮执行一次）。

## 3. 跑一轮 P4D4

- 以 `../vllm-p4d4-crusoe-20261117/queue/s10.sh`（Rust 路由串行，单 API server）或 `s9.sh`（Python 路由 round-robin）为模板复制成新队列项，
  只改要比较的变量；名字以 `L` 开头。部署（P TP1×PP4 GPU 0–3、D DP4 GPU 4–7、etcd、路由）由 `scripts/pd1_up.sh` 完成，开关见文件头：
  `P_GEM / D_GEM`、`P_ARGS / D_ARGS`、`P_ENV / D_ENV`、`P_MOUNT_EXTRA / D_MOUNT_EXTRA`、`P_KV_EXTRA`、`ROUTER_BACKEND`、`ROUTER_ENV`、
  `ROUTER_POLICY_ARGS`、`ROUTER_MOUNT_EXTRA`。计分 `scripts/score_p4d4.sh RUN`（`SMOKE=1 DURATION=1800`），一行成绩写 `logs/summary.out`。
- 一轮约 70 min：部署 12–14 min，AgentX 预热 24–35 min，测量 30 min。**AgentX 预热阶段任何根请求失败都会中止整轮**；叠加没验证过稳定性的
  改动前先做 10 min 左右的稳定性检查。
- 分析工具（计算节点上跑）：`ttft_breakdown.py RUN`（引擎内 vs 路由的 TTFT / 排队）、`hit_timeline.py`（prefill 命中随时间）、`mc_xfer.py`
  （Mooncake 传输）、`samples_summary.py`、`collapse_top.py`（py-spy raw）、`trace_decode.sh` + `trace_cat.py`（decode torch trace 分类）、
  `ckpt_sizes.py`（checkpoint 各类权重大小）。多 API server 时 vLLM `/metrics` 只来自一个进程，`samples.jsonl` 的引擎数字不可信，看路由与 AIPerf 导出。

## 4. 成绩（C104，1800 s；sglang 53,351）

| 轮 | 相对上一轮的变化 | tok/s/GPU | TTFT p50 / 均值 s | 备注 |
|---|---|---|---|---|
| s4 | Python 串行路由 + 缓存感知准入 + 1 TB CPU 层 | 26,541 | 22.5 / 33.8 | |
| s6c | ARC 淘汰 | 23,864 | 15.3 / 40.8 | 不用 ARC |
| s7b | 1.25 TB CPU 层 + 路由 nofile | 30,091 | 28.5 / 26.3 | |
| s8 | P / D 4 API server | 31,384 | 23.9 / 24.3 | API server 不是瓶颈 |
| **s9** | 路由 round-robin（不 tokenize） | **36,923** | 6.8 / 12.4 | 最好 |
| s10 | Rust 路由 + vLLM 串行（kv-aware） | 36,265 | 6.0 / 11.3 | 同 s9 |
| s11 | s10 + 4 API server + renderer 8 | 中止 | | Rust 路由 → decode 502（keep-alive 竞争） |
| s12 | s11 + prefill GEM 0.88 + decode #47897 | 中止 | | prefill EngineCore 段错误 |
| s12b | s12 + keep-alive 120 s | （待补） | | |
| s12c | s12b，prefill 单 API server | （待补） | | |

## 5. 已知问题（按影响排序）

1. Infera Python 路由 kv-aware 在事件循环里同步 tokenize（U15）：用 Rust 路由（`ROUTER_BACKEND=rust ROUTER_ENV=INFERA_VLLM_MOONCAKE_SERIAL=1
   ROUTER_MOUNT_EXTRA="$R/bin/infera-router-vllmserial:/usr/local/bin/infera-router:ro"`），或 Python 路由 `ROUTER_POLICY_ARGS="--router-policy round-robin"`。
2. Rust 路由（reqwest，空闲连接 90 s）与 vLLM uvicorn（keep-alive 5 s）的连接复用竞争 → 502：两边引擎加 `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=120`（s12b 起）。
3. 路由描述符：连接峰值 1,150–1,470，`pd1_up.sh` 已给路由容器 `--ulimit nofile=1048576`（U14）。
4. s12 的 prefill EngineCore 段错误（`PyEval_RestoreThread`），原因未定；s12c 用来区分 prefill 多 API server。
5. GEM + offload 故障已由 B 线定位并绕过（U2：clr 的外部内存没有 `deviceId`；`bin/libhip_gem_alloc-vmm.so` 改用 VMM 导入，挂到
   `/opt/infera-gem/libhip_gem_alloc.so`，`P_GEM=1`，prefill 显存目标 0.88）。
6. 每轮 3–9 条"空内容"失败（固定在少数会话，与长度无关），原因未查；落在根请求上就会中止 AgentX 预热。
7. `TieringOffloadingSpec` 在 PP>1 起不来（U11，B 线在修），修好后可试 NVMe 二级层（3600 s 计分时工作集更大）。

## 6. 下一步（建议顺序）

1. 看 s12b / s12c：选一个稳定的组合配置（GEM 0.88 + #47897 + keep-alive，前端扩展是否保留）。
2. **decode 每步**（主攻）：A 线的 aiter 自定义 AG/RS 修正（`../vllm-decode-trackA-20261118/overlay-comm/`，U13；每步 MoE 通信 15–26%）接进 P4D4；
   decode EngineCore 主线程约 99% CPU（py-spy：`queue/1118-082247-Lpyspy-s12b` 若已采到，结果在 `runs/<s12b>/pyspy/`）；草稿注意力元数据每步重建
   （`ROCM_AITER_MLA_SPARSE cannot update draft decode metadata in place`）。
3. decode 容量：`--gpu-memory-utilization 0.97`、减小 decode 的 `--max-num-batched-tokens`（激活峰值 6.7 GiB）。
4. 用最好配置跑 3600 s（与 sglang 同口径），并重复一次量化波动（目前都是单次）。
5. P8D8（两节点：`scripts/pd_role_up.sh`、`scripts/pd2_drive.sh`，未测）。
6. 上游：`../upstream-vllm-20261117/CANDIDATES.zh-CN.md`（U1–U16）；优先 U12（#47897，草稿 `PR-U12-*.md` 与 A 线的 v3 改法）、U13、U15 / U16 / U14（Infera）、
   U2（Infera GEM VMM 导入 + ROCm clr）、U11。

## 7. 与 A / B 线的协作

- A / B 线只写自己的目录（`../vllm-decode-trackA-20261118/`、`../vllm-prefill-trackB-20261118/`），不改共享文件；主线负责把结论合并进
  `PROGRESS.zh-CN.md`、`CANDIDATES.zh-CN.md`、P4D4 README、KF README。合并前看它们最新的 README 与提交（`git log -- llying/vllm-*-track*`）。
- B 线交接 `../vllm-p4d4-crusoe-20261117/HANDOFF-trackB.zh-CN.md`；A 线交接 `../vllm-kf-port-20261117/HANDOFF-trackA.zh-CN.md`。
