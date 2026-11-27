# GLM-5.2-MXFP4 P4D4（Infera + vLLM，单节点 8 × MI355X）复现 kit

一个节点：prefill TP1 × PP4 在 GPU 0–3，decode DP4（TP1）在 GPU 4–7，Infera Rust 路由；KV 经 Mooncake RDMA 回环（同节点也不走 HIP），
每个 GPU 只用自己的网卡。客户端和 eval 直接用 InferenceX main 的命令，口径与 InferenceX 相同。

## 1. 准备

| 步骤 | 命令 | 说明 |
|---|---|---|
| 节点检查 | `bash ../probe/probe_node.sh` | 驱动、peer memory、150 GiB 显存注册测试 → `results/probe-<节点>/{gem,libionic,gpu-nic.txt}`，`env.sh` 自动读取 |
| 镜像 | `bash ../docker/build.sh series-C1.txt infera-vllm:nightly-de7cf7f8-C1` | vLLM nightly de7cf7f8 + `../patches/series-C1.txt`（每行一个 PR 的原样 diff，见 `../patches/PATCHES.md`）+ Mooncake（v0.3.14 + #4540 / #4541）+ GEM 分配器 + Infera |
| InferenceX | `git worktree add <dir> d490919c` | `env.sh` 里的 `IXM` |

站点相关的设置都在 `env.sh` 的 site 段：模型路径、InferenceX 路径、节点检查结果目录。GEM 的选择：

| 驱动 | 显存注册 | GEM |
|---|---|---|
| amdgpu 6.19（有无 peer memory 都一样） | dma-buf 和 peer memory 都会把锁住的显存记两次账 | 用（`GEM=1`） |
| amdgpu 6.16 + peer memory | peer memory 正常 | 不用（`GEM=0`，`MOONCAKE_DISABLE_HIP_DMABUF=1`） |
| amdgpu 6.16，无 peer memory | dma-buf 记两次账 | 用 |

## 2. 跑一个点

```bash
bash run.sh <变体> <输出目录>              # 默认：eval（真实接受）→ 计分（合成接受），C152
CONC=152 STEPS=score FAST=1 bash run.sh C1 /tmp/c1-fast   # 只计分、InferenceX fast 模式（1200 s，只用于跑通）
```

- `eval`：全新部署、真实 MTP 接受 → InferenceX `infx.bench eval`（lm-eval GSM8K 全量 1,319 题、5-shot、并发 = 计分并发，阈值 0.90）
  + `infx.evals.validate_scores` + AL 门禁（`infx.bench al-gate`，golden 3.33 的 0.95–1.05）。
- `score`：全新部署、合成接受 3.33（InferenceX golden，草稿 4）→ InferenceX `infx.bench agentic`（agentx-harness，3600 s、每条通道 10 个预热）。
  吞吐 = (输入 + 输出 token) / 时间跨度 / 8 卡（prefill 4 + decode 4）。
- 结果：`<输出目录>/{eval,score}/`，`score/agentx/score.txt` 是 InferenceX 汇总的要点，`engine-config.txt` 是两个引擎的完整命令。

## 3. 变体（`variants/*.env`）

| 变体 | 内容 |
|---|---|
| `none` | 原版 nightly + Infera，无 vLLM 补丁 |
| `C0` | nightly + 9 个已有的上游 PR / fork PR |
| `C1` | C0 + 新开的 fork PR #15（FP4 dispatch）、#16（每层 MoE 一次 all-gather）、#17（Triton decode top-k） |
| `D` | C1 + 不能上游的本地项：THP-off（`local/thp_off.py`）、decode BF16 GEMM 调优表 |
| `E` | D 去掉 GEMM 调优表（集群必需的 GEM、THP-off 保留） |
| `F-B0` | C1，decode DP rank 改由 Infera 路由选（按 rank 注册，`--decode-input-demand on --decode-request-weight W`） |
| `F-B1` | F-B0 + 路由把 decode 的流式输出计入 rank 的记账（`--decode-output-demand on`，路由来自 `llying-001/Infera:llying/router-dp-on-vllm-glm52`） |

共同的配置（`engine/up.sh`）：
- prefill：TP1 × PP4（层 20 / 20 / 20 / 18），前缀缓存，分块 4096、长 prefill 阈值 2048，CPU KV 层 1.9 TB（LRU），缓存感知准入 128 / 0，显存 0.88；
- decode：DP4 × TP1，前缀缓存，分块 2048，`--stream-interval 30`，显存 0.95；
- 两边：MTP 4，FP8 KV cache，块 64，dense 层在线 FP8（0–77 层，草稿层 78 不动）+ MXFP4 indexer K cache（目标层，草稿层保持 FP8）；
- 路由：kv-aware（prefill 重叠 20、decode 2），prefill 会话亲和，Mooncake 串行下发。

## 4. 结果（aus MI355X，C152，草稿 4，3600 s；详见 `../results/`）

| 配置 | 节点 | tok/s/GPU | 交互 p50 / p90 | TTFT 均值（s） | GSM8K | AL |
|---|---|---|---|---|---|---|
| crusoe a49（v0.34 + 覆盖层，参考） | crusoe v3-017 | 71,061 | 72.2 / 61.5 | 8.18 | — | — |
| a49 原样复现（`../replay/`） | n04-29 | 69,682 | 71.5 / 59.4 | 8.40 | 0.9689 | — |
| C1 | n04-25 | 70,672 | 70.1 / 59.6 | 8.36 | 0.9682 | 3.254 |
| D | n04-29 | 69,744 | 69.5 / 59.6 | 8.55 | 0.9727 | 3.277 |
