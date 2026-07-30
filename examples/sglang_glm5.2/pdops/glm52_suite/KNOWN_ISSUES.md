# 已修复：长上下文冷 prefill 下的偶发 token 损坏（token id 0）

**状态：已定位并修复（2026-07-27）。**

- **根因**：`--enable-aiter-allreduce-fusion`。开启后长上下文冷 prefill 时
  `next_token_logits` 会整行变成 NaN，采样退化为 token id 0（解码出 `!`）。
- **修复**：`run_sglang.sh` 里默认关闭（`ENABLE_AITER_ALLREDUCE_FUSION=0`）。
- **验证**：关闭后 120/120 冷缓存请求 0 失败、0 NaN 告警；完整正确性套件全部通过。

下面保留完整排查过程，一是留证据，二是万一复发可以照着走。

---

## 1. 环境

| 项 | 值 |
|---|---|
| 机器 | AMD Instinct MI325X × 8，gfx942，304 CU，256 GiB/GPU |
| SGLang | 0.5.16（`/sgl-workspace/sglang`） |
| torch | 2.9.1+rocm7.2.0，HIP 7.2 |
| tilelang | 0.1.7.post3 |
| 模型 | `/wekafs/models/GLM-5.2-FP8`，141 shard / 704 GiB，`index_topk=2048`，78 层 + 1 nextn |

运行时实际生效：KV cache `torch.float8_e4m3fnuz`、2310976 tokens、118.86 GiB/GPU；
attention backend `dsa`；allreduce `AiterCustomAllreduce`。

> 附带发现：`ROCM_QUICK_REDUCE_QUANTIZATION=INT4` 在当前配置下是**无效配置**——日志确认走的是
> AiterCustomAllreduce，QuickAllReduce 从未启用。这个环境变量是从 MI355X cookbook 继承来的，
> 排查时不要被它误导。

## 2. 现象

长上下文检索任务（needle）偶发返回被截断的错误答案。样本（`want` 为埋入 context 的正确值）：

| want | got | completion_tokens |
|---|---|---|
| `5338298` | `'5!'` | 3 |
| `6000049` | `'600!'` | 3 |
| `9843397` | `'98!'` | 3 |
| `4291863` | `'4!291863'` | 7 |

## 3. 定位过程

**(a) 不是检索失败。** `want=4291863 got='4!291863'` 这一例正确数字被完整取回，只是中间插进一个
多余 token。sparse indexer 找 needle 是正确的。

**(b) 那个 `!` 是 token id 0**（`tokenizer.encode('!') == [0]`）。这是 logits 行退化的典型特征
——全零 / 全相等 / NaN 时 argmax 返回索引 0。不是模型"想"输出感叹号。

**(c) 损坏发生在首个 decode step。** 对照 token 序列：

```
5338298 -> [20, 100702, 23, 100104, 23]   正确输出 completion_tokens=6（5 token + eos）
'5!'    -> [20, 0, eos]                    completion_tokens=3
6000049 -> [103306, 98503, 19, 24]        '600!' = [103306, 0, eos]
```

第一个 token 一律正确（来自 prefill 最后位置的 logits），紧接的第二个变成 id 0
（来自首个 decode step）。`'4!291863'` 表明之后能自行恢复，是单步瞬时损坏。

**(d) 只在冷 prefix cache 下出现。** 每次请求前 `POST /flush_cache`：36 次 4 次失败（11%）。
prefix cache 预热后：同一 prompt 连续 5 次全对，完整 needle 检查连跑 3 轮 27/27 通过。
同样的逻辑输入，KV 复用时正确、重算时损坏——排除了"模型本来就这么答"。

**(e) NaN 确认。** 保持配置不变、只加观测开关 `SGLANG_SANITIZE_NAN_LOGITS=1` 重启，
8 个 TP rank **全部**打出：

```
NaN detected in sampler: next_token_logits; values were sanitized before sampling.
```

且损坏仍复现（3/36）。sanitize 会把 NaN 换成 `-1e30`；若只有部分元素是 NaN，argmax 会挑到真正的
最大值、输出反而被"修好"。实际仍采样到 id 0，说明**整行 logits 都是 NaN**。

**(f) MTP / MTP+DP 配置从不复现**，这是指向 allreduce 的关键线索。两者都带
`--disable-custom-all-reduce`、且 aiter 融合 allreduce 为关：MTP-only 36/36、MTP+DP 60/60 全过。

## 4. 根因确认实验

两次重启，**除 allreduce fusion 外配置完全相同**，同一份 `repro_token_corruption.py --iters 6`：

| 实验 | `enable_aiter_allreduce_fusion` | 结果 | NaN 告警 |
|---|---|---|---|
| dbg1 (`tmp/dbg1_nan_sanitize.log`) | True | **3/36 失败（8.3%）** | 8 个 rank 全部告警 |
| dbg2 (`tmp/dbg2_no_aiter_ar.log`) | False | 36/36 通过 | 0 |
| dbg2 加大样本 | False | **120/120 通过** | 0 |

按 8.3% 的发生率，120 次全过的偶然概率约 3×10⁻⁵，因此可以定论。

注意两次实验里 `disable_custom_all_reduce` 都是 False，`AiterCustomAllreduce` 都在用。
唯一变量是 **layernorm + allreduce 的融合**，所以问题出在这个融合 kernel，而非 allreduce 本身。

### 上游旁证

SGLang 自己已经把这个融合对 DSA 家族的自动启用**注释掉了**，只留下一行误导人的日志：

```
srt/server_args.py:4521
    if not self._resolved().enable_dp_attention and self.nnodes == 1:
        # TODO (Hubert): Put this back later
        # self.enable_aiter_allreduce_fusion = True
        logger.info("Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM")
```

所以日志里出现 "Enable Aiter AllReduce Fusion for DeepseekV3ForCausalLM" **不代表它真的启用了**，
判断请以 `server_args` 里的 `enable_aiter_allreduce_fusion=` 为准。MI355X cookbook 里的
`--enable-aiter-allreduce-fusion` 等于把上游刻意撤掉的东西又打开了。

## 5. 修复与验证

`run_sglang.sh` 中默认关闭：

```bash
ENABLE_AITER_ALLREDUCE_FUSION="${ENABLE_AITER_ALLREDUCE_FUSION:-0}"
```

仍可用 `ENABLE_AITER_ALLREDUCE_FUSION=1 ./run_sglang.sh` 复现旧行为。

修复后完整套件（`tmp/verify_fixed_baseline.json`）：

```
PASS weights 2/2   PASS basic 7/7        PASS determinism 1/1   PASS idle 3/3
PASS needle 9/9    PASS humaneval 20/20  INFO humaneval-long 19/20
PASS code-retrieval 2/2                  PASS deep-api 3/3
结论: 全部通过
```

`humaneval-long` 那 1 题（HumanEval/17）在所有配置下都失败，是模型能力问题，非本 bug。

**代价**：失去 layernorm + allreduce 的融合优化。`AiterCustomAllreduce` 本身仍在使用，
未测量具体性能差异。正确性优先。

## 6. 复现方法（若复发）

```bash
./repro_token_corruption.py                 # 24 次冷缓存请求
./repro_token_corruption.py --iters 20      # 120 次，用于验证修复
./repro_token_corruption.py --no-flush      # 对照：暖缓存，预期不复现
```

发生率只有百分之几，**样本量小的时候可能一次都不出现**。判定"修好了"至少要 100 次以上无失败
（`--iters 20`），这是排查此问题最主要的困难。

## 7. 若复发时的后续排查方向

第 1 步（确认 NaN）与第 2 步（allreduce）已完成。若关掉融合后问题仍在，继续二分，
每次只改一项并跑 `--iters 20`：

| 改动 | 验证的假设 |
|---|---|
| `--kv-cache-dtype bfloat16` | FP8 `e4m3fnuz`（gfx942 用的格式，与 gfx950 的 `e4m3fn` 指数偏置不同）溢出 |
| 去掉 `SGLANG_DSA_TRITON_PREFILL=1` | triton indexer prefill 在 gfx942 上的问题 |
| `--dsa-prefill-backend aiter --dsa-decode-backend aiter` | tilelang 非 gfx950 分支（304 CU 那条）的问题 |
| `TP_SIZE=4` | allreduce / 跨 8 卡规约规模因素 |

更深的手段：`SGLANG_ENABLE_ASYNC_ASSERT=1` 遇 NaN 直接崩并带上下文（注意它会短路
`maybe_warn_nan`，两者互斥；且会中断服务）；`--debug-tensor-dump-output-folder` 配合
`--debug-tensor-dump-layers` dump 首个 decode step 的中间张量，定位 NaN 最早出现在哪一层。
