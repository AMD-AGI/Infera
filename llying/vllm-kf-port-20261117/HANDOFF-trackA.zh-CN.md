# 交接：A 线（GLM-5.2 decode 侧 kernel / 通信，vLLM v0.34.0，聚合模式）

2026-11-18 03:30 起由新 session 在 v3-240（作业 295840）上做。P4D4 计分线继续在 v3-135（294199）上，两边不共用工作目录和队列。

## 1. 不要碰的东西

- `/home/liyingli/bench_agentx/vllm-pd-20261117/`，尤其 `queue/`（v3-135 的 runner）和 `queue-n2/`（v3-258 的 runner）：两个 runner 会执行
  那里出现的任何 `*.sh`。新 session 用自己的工作目录（例如 `/home/liyingli/bench_agentx/vllm-trackA-20261118/`）和自己的 runner。
- v3-135（294199）、v3-258（294784）上的容器（`vpd-*`）；这两台由主 session 用于 P4D4 / P8D8。

## 2. 准备

- 集群约束（crusoe spur）：任何 step 非零退出都会结束作业；每个节点只能有一个 exec-into-job 会话（所以要一个常驻的 runner 执行队列）；
  不要注入主机的 libionic。做法参考 `../vllm-pd-tools-20261117/`（`onnode.sh`、`runner.sh`、`q.sh`）。
- 镜像：按 `../vllm-image-v0340-20261117/` 构建 `infera-vllm:v0340-<hash>`（约 30–40 min），或在 v3-135 上 `docker save` 后到 v3-240 `docker load`（约 37 GB）。
  A 线只用聚合模式，原版 `vllm/vllm-openai-rocm:v0.34.0` 也够；覆盖层文件按 bind mount 挂进去。
- 模型拷到本地 NVMe：`/mnt/m2m_nobackup/liyingli_vpd/models/GLM-5.2-MXFP4`（v3-135 上从 `/mnt/hf_cache` 拷约 82 s）。
- 聚合 decode 配置（与 P4D4 的 decode 一致）：`--data-parallel-size 4 --tensor-parallel-size 1 --kv-cache-dtype fp8_e4m3 --enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' --max-num-seqs 64 --max-num-batched-tokens 8192 --block-size 64
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice`，环境 `VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1`
  （`../vllm-pd-tools-20261117/scripts/vllm_up.sh`）。8 卡可同时起两个（0–3 / 4–7）做 A/B。
- 正确性门禁：`../vllm-pd-tools-20261117/scripts/gate.sh`（GSM8K 200、长上下文 16K、MTP 接受长度）。原版基线 195/200。

## 3. 任务

### A1 #47897（gfx950 gluon paged MQA logits）端到端

- 覆盖层：`overlay/vllm/v1/attention/ops/rocm_gluon_paged_mqa_logits.py`（新文件）、`overlay/rocm_aiter_mla_sparse.py.diff`（对 v0.34.0 的改动；
  完整文件 `overlay/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`）。开关 `VLLM_ROCM_DSA_GLUON_PAGED_MQA_LOGITS=1`。
- 函数级测试通过（`scripts/test_overlay_logits.py` + `scripts/run_overlay_test.sh`，含 vLLM 在 ROCm 上的拉平 decode 形式与 CUDA graph 重放）。
  单算子对 vLLM 实际的拉平 aiter 调用快 2.24–2.54 倍（README 4.0）。
- **两次聚合 A/B 都没在引擎里用上新 kernel**（README 4.0 / 4.0b）。第二次的推测原因：worker 运行时 `get_current_vllm_config()` 取不到 MTP 配置，
  `_gluon_lpt_group_rows()` 为 0。下一次：
  1. 加 `VLLM_ROCM_DSA_GLUON_LPT_GROUP_ROWS=4`；起来后先确认日志里有 `Using the gfx950 gluon paged-MQA logits kernel for decode.`，没有就先查
     `_use_gluon_paged_mqa_logits` 哪个条件不成立（next_n、block size、`gluon.supported`）。上游 PR 里应在初始化时从配置取组大小，而不是运行时取。
  2. bench 改成 decode 主导：例如一波 104 个请求、输入 64K、输出 4096、`--ignore-eos`，或者直接用引擎 torch trace 比较 indexer kernel 时间
     （`../vllm-pd-tools-20261117/scripts/trace_decode.sh`、`trace_cat.py`）。之前的 `--num-prompts 208 --max-concurrency 104 --random-output-len 1024`
     被 prefill 主导（TPOT ~200 ms），测不准 decode kernel。
- PR 草稿：`../upstream-vllm-20261117/PR-U12-gfx950-gluon-paged-mqa-logits.md`（kernel 文件保留 AMD 的 MIT 头，署名 sglang #47897，需确认许可处理）。

### A2 decode MoE 通信（U13）

- P4D4 s6c 的 decode trace（`../vllm-p4d4-crusoe-20261117/results/s6c/trace/`）：每层 2 次 RCCL `ncclDevKernel_Generic`（平均 71.5 µs），约占每步 26%
  （含等最慢 rank）。vLLM 有给 DP 组的 aiter 自定义 AG/RS（`distributed/device_communicators/cuda_communicator.py::_can_use_aiter_ag_rs`），
  aiter 侧条件（world size 4、`fully_connected` 写死 True、128 MB 上限）都满足，仍没走上。待查 vLLM 侧：batch descriptor 的 `uniform`、
  FULL graph 捕获时的选择、AG 输入是不是 list（量化 dispatch）。
- 实验脚本：`scripts/decode_comm_ab.sh`（原 v3-135 队列项 decomm3，路径需改成新工作目录）：默认 / `--enable-expert-parallel` /
  EP + `--all2all-backend mori`，各跑 bench 并抓 trace。

### A3 #47872（单遍 Triton top-k）/ #47276（融合 indexer + top-k）

- diff 与 PR head 见 `scripts/upstream-src/sglang-pr-heads.txt`；decode top-k 在 vLLM 的位置：`v1/attention/ops/rocm_aiter_mla_sparse.py` decode 段
  （aiter `top_k_per_row_decode` / `torch.ops._C.top_k_per_row_decode`）。s6c trace 里 top-k 每次约 47 µs、每步约 20 次。

## 4. 结果放哪

- 结果放新目录 `llying/vllm-decode-trackA-20261118/`（README.zh-CN.md + scripts + results），及时提交到 `dev/pd_opt/vllm_glm52_agentx`
  （`git commit -s`，push 前 `git pull --rebase`；建议用自己的 `git worktree`）。
- **不改主 session 维护的共享文件**：`../vllm-pd-progress-20261117/`、`../upstream-vllm-20261117/CANDIDATES.zh-CN.md`、本目录的 `README.zh-CN.md`。
  结论（含对 U12 / U13 的更新）写在自己的 README 里，由主 session 合并，避免两个 session 改同一文件冲突。
