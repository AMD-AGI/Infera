# GLM-5.2-MXFP4 每步计算 backend 统计(runtime 实测)

来源:exp06(PD mooncake MTP)decode 腿运行时 log
`/mnt/vast/c_huggingface/glm52_p2b/pd_decode_30001.log`,GLM-5.2-MXFP4 on gfx950 / pd-unified
镜像(sglang 0.5.15.post1)。这是 **runtime 实际解析后**跑的 kernel,不是 CLI 字面的 `auto`/`None`。

## 一、注意力(MLA + DSA 稀疏)

| 计算步骤 | 实际 backend | 证据 |
|---|---|---|
| 注意力总框架 | **dsa** (`attention_backend='dsa'`) | `Use dsa attention backend for DeepSeek with DSA` |
| DSA 主 kernel — prefill | **tilelang** | `Set DSA backends for fp8_e4m3 KV Cache: prefill=tilelang` |
| DSA 主 kernel — decode | **tilelang** | `...decode=tilelang` + `Loading tilelang libs from /opt/tilelang/build` |
| DSA paged MQA-logits | **aiter**(`auto`→ROCm=aiter) | help 文档 + `module_aiter_core` import |
| DSA top-k 选择 | **sgl-kernel** | `dsa_topk_backend='sgl-kernel'` |
| QK-norm + RoPE + KV量化(融合) | **aiter** | `module_fused_qk_norm_rope_cache_quant_shuffle.so` |
| RoPE(带 cached positions) | **aiter** | `module_rope_2c_cached_positions_fwd` |

> 注:`--nsa-prefill/decode-backend` 已 deprecated,新名 `--dsa-prefill/decode-backend`。
> DSA_CHOICES = [flashmla_sparse, flashmla_kv, flashmla_auto, fa3, tilelang, aiter, trtllm]
> → 主 kernel **有 aiter 实现**,我们手动选了 tilelang(gfx950 保守配方,性能不佳,待验证 aiter)。

## 二、MoE(256 专家,MXFP4)

| 计算步骤 | 实际 backend | 证据 |
|---|---|---|
| MoE runner 框架 | `auto` → **aiter fused_moe** | `moe_runner_backend=auto, quant_method=QuarkFusedMoEMethod` |
| FP4 专家 GEMM(量化层) | **aiter FlyDSL**(启发式 fallback) | `flydsl_moe1_afp4_wfp4_bf16_t32x128x256`(afp4/wfp4=MXFP4) |
| BF16 专家 GEMM(dense 层 0-2) | **aiter 2stage default** | `using 2stage default ... QuantType.No, bfloat16` |
| 专家排序/dispatch | **aiter** | `module_moe_sorting_opus` + `module_moe_asm` |
| TRTLLM MoE deferred finalize | **禁用** | `deferred finalize is disabled` |

> ⚠️ FP4 MoE 走 **无 tuned config 的 FlyDSL 启发式**(`no tuned FlyDSL config for gfx950`,
> shape 256E/6144 hidden)→ 功能正确但性能非最优。

## 三、Norm / GEMM / 通信 / 采样

| 计算步骤 | 实际 backend | 证据 |
|---|---|---|
| RMSNorm | **aiter** | `module_norm` |
| RMSNorm + 量化融合 | **aiter** | `module_rmsnorm_quant.so` |
| 通用量化 kernel | **aiter** | `module_quant` |
| 稠密层 FP8 GEMM | `fp8_gemm_runner_backend='auto'`(ROCm→aiter/hipblaslt) | server_args |
| BF16 GEMM | `bf16_gemm_backend='auto'` | tuned csv(glm5_bf16_tuned_gemm.csv) |
| All-Reduce(TP8) | **NCCL/RCCL**(custom AR 禁用) | `[AR] All-reduce call path: NCCL (custom AR disabled)` |
| KV cache dtype | **fp8_e4m3** | `kv_cache_dtype='fp8_e4m3'` |
| Sampling | **pytorch** | `sampling_backend='pytorch'` |
| Grammar(约束解码) | **xgrammar** | `grammar_backend='xgrammar'` |
| Mamba | triton(未用,GLM 无 mamba) | `mamba_backend='triton'` |

## 四、MTP / EAGLE draft(仅 decode 腿)

| 计算步骤 | 实际 backend | 证据 |
|---|---|---|
| draft 注意力 | **继承 target = dsa+tilelang** | `speculative_draft_attention_backend=None` |
| draft 注意力模式 | **prefill 模式** | `speculative_attention_mode='prefill'` |
| draft MoE runner | `auto`→aiter | `speculative_moe_runner_backend='auto'` |
| draft CUDA graph | **full backend, eager compiler** | `Capture draft decode CUDA graph begin. backend=full` |
| DP-MLP sync(spec×DP 耦合点) | **False**(exp06 无 DPA) | `speculative_skip_dp_mlp_sync=False` |

## 五、CUDA graph / 调度

| 项 | 值 |
|---|---|
| decode CUDA graph | `backend=full`, max_bs=64, eager compiler |
| prefill CUDA graph | `disabled` |
| radix cache | **禁用**(`disable_radix_cache=True`,PD decode 腿) |

## 一句话总结

GLM-5.2 = **aiter 为主 + tilelang 专精 DSA 主 kernel + sgl-kernel 管 topk** 的混合栈:
- **aiter** 承担绝大多数算子:MoE(FP4 FlyDSL + BF16 2stage)、RMSNorm、RoPE、QK-norm、量化、DSA 的 MQA-logits。
- **tilelang** 只负责 DSA 稀疏注意力的 prefill/decode **主 kernel**(aiter 那版在 gfx950 未验证)。
- **sgl-kernel** 管 DSA top-k;**NCCL** 管 TP8 通信;**pytorch** 管采样。

"aiter vs tilelang" 不是对立——同一次 DSA 计算里 tilelang(主 kernel)+ aiter(MQA-logits)+
sgl-kernel(topk)分工协同,aiter 还独揽 MoE 和所有 norm/rope。

## 已验证:aiter DSA 主 backend 在 gfx950/GLM-5.2 上 GPU FAULT,不可用(2026-07-29)

诉求:主 attention 用 tilelang 太慢 → 试 aiter。单机 chi2835,iterative-debug-loop 剥了 5 轮:

| 轮 | 单变量改动 | 结果 | 学到 |
|---|---|---|---|
| A1 | `--dsa-*-backend aiter`,stock CSV | GPU fault @ warmup | 崩在 aiter flydsl GEMM |
| A2 | scrub base csv 5 条 flydsl,mount | 仍 fault | mount 错文件:runtime 读合并产物 `/tmp/aiter_configs/` |
| A3 | scrub glm5 model_config 65 条 + mount 双文件 | 仍 fault,flydsl_picks=16 | 合并会 glob **所有** model_configs(dsv4=415/gptoss=78/kimik2=79…共 728 条 flydsl) |
| A4 | env `AITER_CONFIG_GEMM_BF16=<单 scrub csv>` 绕过合并,flydsl_picks=**0** | **仍 8 卡 fault** | GEMM 全 torch 也崩 → **FlyDSL GEMM 不是根因** |
| A5 | aiter DSA + `--disable-cuda-graph`(eager) | **仍 fault @ warmup forward** | 非 cuda-graph capture 问题 |

**最终根因:aiter 的 DSA 稀疏注意力 kernel 在 gfx950 + GLM-5.2 上本身 GPU memory fault**——
GEMM 配置全干净(torch,0 flydsl)、eager 无 capture,warmup 前向照崩(A4 崩在
`decode_cuda_graph_runner.py:698`,A5 eager 崩在 `_execute_server_warmup`)。

**结论:tilelang 是此栈唯一正确的 DSA 主 backend。** jiejing 用 tilelang 不是保守,是 aiter DSA
kernel 在这里真会 fault。想给 attention 提速,不能简单切 aiter;可选(未验证):其他
`DSA_CHOICES`(flashmla_sparse/flashmla_kv/fa3/trtllm)、等 aiter 修 gfx950 DSA kernel、或先
profile 确认 tilelang 的真实瓶颈是否在 attention。

> 两个独立问题叠加:FlyDSL 那 728 条坏 GEMM config 是真实地雷(会先崩),但即使全绕开,底层
> aiter DSA attention 仍不可用。之前误把前者当唯一根因。
> 产物:`/mnt/vast/c_huggingface/glm52_dsa_test/*.noflydsl.csv`;绕过 GEMM 合并的开关 = env
> `AITER_CONFIG_GEMM_BF16=<单文件>`(否则 runtime 会 merge 所有 model_configs 的 flydsl 行)。

---

# 交接:下一个 agent 从这里接手(HANDOFF, 2026-07-29)

## 任务背景(为什么在查这个)
主线任务是 GLM-5.2-MXFP4 在 sglang 上的 bring-up(见 CLAUDE.md / mission.glm5.2.md)。用户当前诉求:
**"DSA 主 attention 用 tilelang 性能不可接受,想换 aiter 提速。"** 上面 §已验证 部分已证明**直接切
`--dsa-*-backend aiter` 会 GPU fault,不可用**。交给你的是:**在 aiter 不可用的前提下,找到一条
能提速 attention 的正确路径**(或证明 tilelang 已是最优、瓶颈不在 attention)。

## 环境与资源(全部已就绪)
- **节点**:prefill=chi2835(10.2.122.78),decode=chi2879(10.2.122.10)。都归我,GPU 空
  (baseline 297MB/卡)。chi2867(10.2.122.44)**别用**——根盘 98% 满,装不下 78GB 镜像,且上面
  `pd_etcd`/`kimi_pd_debug` 是别人的 DSv4 残留。
- **镜像**:`infera/engine-sglang:pd-unified`(sglang 0.5.15.post1),chi2835 id=`05967248b58f`,
  chi2879 id=`f8ec2d627392`。两台都有。
- **模型**:`/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`(=tokenizer dir)。head_dim 192,78L,256E。
- **访问**:jump host `root@149.28.124.225` → `ssh <node>`。
- **单机测试脚手架**:`/mnt/vast/c_huggingface/glm52_dsa_test/`(scrub 后的 csv 都在这)。

## 已确立的事实(别再重复验证,已实测)
1. **aiter DSA 主 kernel(prefill+decode)在 gfx950/GLM-5.2 会 GPU memory fault**——eager 也崩、
   GEMM 全 torch 也崩,是 attention kernel 本身的问题。证据:§已验证 A1–A5。
2. **FlyDSL bf16 GEMM 是独立的第二个地雷**:aiter 合并 `/tmp/aiter_configs/bf16_tuned_gemm.csv` 时
   会 glob **所有** model_configs 的 `*bf16_tuned_gemm*.csv`(dsv4=415/gptoss=78/kimik2=79/glm5=65…
   共 728 条 flydsl),这些 flydsl kernel 在 gfx950 也 fault。绕过法:env
   `AITER_CONFIG_GEMM_BF16=<单个干净 csv>`(单文件→不触发 merge)。**注意:即便这样绕开,§事实1 仍崩。**
3. **当前生产配方(tilelang)是对的、能跑**:exp01/02/03/06/07 全 PASS,DSA 主 kernel=tilelang。
   见 `01_single_node_mix/`、results_summary.csv。
4. GLM-5.2 全栈每步 backend 归属见本文件 §一~§五(exp06 log 实测)。

## 精确崩溃证据(py-spy 实测,复现用)
- A4(cuda-graph capture 阶段):`decode_cuda_graph_runner.py:698 capture` →
  `full_cuda_graph_backend.py:90 capture_one` → `torch.cuda.synchronize` → GPU fault。
- A5(eager,禁 cuda-graph):`_execute_server_warmup`(http_server.py:2091)阶段 warmup 前向 →
  8 卡 `Memory access fault ... Reason: Unknown/Write to read-only page`。
- 复现命令模板见 `/tmp/launch_aiter{4,5}.sh`(已 scp 到 chi2835:/tmp/,也可从本文件重建)。

## 建议的下一步(按性价比排序,均未验证)
1. **先 profile,别急着换 backend**。用户说 tilelang "太慢",但没数据证明瓶颈在 attention。
   先在**已知能跑的 tilelang 单机**上跑 `sglang.bench_serving` + 打开 `--show-time-cost` 或
   torch profiler,确认 decode 时间到底花在 DSA attention 还是 MoE/GEMM/通信。若瓶颈不在 attention,
   换 backend 白费力。**这是最该先做的一步。**
2. **试其他 `DSA_CHOICES`**:`flashmla_sparse` / `flashmla_kv` / `fa3` / `trtllm`。单机、单变量、
   照 §已验证 的 A1 方法(改 `--dsa-prefill/decode-backend <x>`,ctx 32768,先看能否过 warmup + probe
   连贯)。**注意**:每个都可能像 aiter 一样在 gfx950 上崩,MVP 快验、别直接上 PD。flashmla 需要
   flash-attention 后端,先 grep 镜像里有没有对应 so。
3. **查 aiter DSA kernel 崩的深层原因**:head_dim 192 是 GLM-5.2 特有(DSv4 是 64/128)。aiter 的
   DSA kernel 可能 hardcode 了 head_dim 或 page_size 假设。可 grep `aiter/.../mla` / `dsa` kernel 源码,
   或去 aiter GitHub 查 gfx950 + head_dim 192 的 issue。若能加个 tuned config / 传对参数,或许能救。
4. **等上游修**:确认 pd-unified 镜像的 aiter 版本,查 aiter release note 有没有 gfx950 DSA 修复。

## 关键陷阱(踩过,别再踩)
- **嵌套 ssh + 引号会炸**:python 一行 / 复杂命令务必**写成脚本文件 scp 过去再执行**,别内联进
  三层 ssh(会 SyntaxError / shell 展开错乱)。本 session 所有 diag 都是 `/tmp/*.sh` 传过去跑的。
- **aiter GEMM config 不是你 mount 的那个文件**:runtime 读的是合并产物 `/tmp/aiter_configs/`,
  mount `configs/*.csv` 无效。用 env `AITER_CONFIG_GEMM_BF16` 才能真正指定。
- **冷启动 ~8-12min**(权重 282 shard + warmup + capture),"卡住不动"先用 py-spy 抓栈区分
  "慢编译/加载" vs "真死锁",别急着 kill。判据:GPU util、log 行数是否增长、栈是否推进。
- **崩了要清理**:`docker rm -f <name>`,等 3s,`rocm-smi --showmeminfo vram` 确认回 baseline
  再起下一轮(RDMA/GPU reset ritual)。
- **iterative-debug-loop**:一次一个变量、对拍已知能跑的 tilelang 配置、每轮记 working_process。

## 相关文件清单
- 本文件:全栈 backend 归属 + aiter 验证全过程。
- `working_process.dpamtp.md`:本 session 的调试流水(含 aiter side-quest A1-A5)。
- `01_single_node_mix/scripts/launch.sh`:已知能跑的 tilelang 单机基线(对拍用)。
- chi2835 `/tmp/launch_aiter{4,5}.sh`、`/tmp/*.sh`:各轮启动/诊断脚本。
- `/mnt/vast/c_huggingface/glm52_dsa_test/`:scrub 后的 csv + log。
- exp06 log `/mnt/vast/c_huggingface/glm52_p2b/pd_decode_30001.log`:§一~五 backend 归属的证据源。
