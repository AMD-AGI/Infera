# DSv4-Pro 1P1D — KV cache 真实用量 + max-total-tokens 优雅回退配置

**Ran:** 2026-07-24（max-total-tokens 扫描）→ 2026-07-26（prefill/decode 真实用量 + 优雅回退）
**Author:** yihou
**Status:** ✅ PASS — 两套交付均达成；含一条重要机制发现（DSv4 无优雅 retract）。

## Goal

以 `lmsysorg/sglang-rocm:v0.5.13-rocm720-mi35x-20260612` 跑 DSv4-Pro 1P1D，实测真实 KV cache 利用率，
据此用 `--max-total-tokens` 把 KV 占用压到最小合理值；并搞清 KV 不够时如何配置才能**优雅回退不崩**。

**两套最终交付**：
1. **KV 真实用量公式** — 给定 (conc, ISL, OSL) + 1P1D TP8+DP8，服务不掉性能时 prefill/decode 各自真实
   KV 绝对字节。见 `DELIVERABLE_1_kv_usage_formula.md`。
2. **优雅回退配置法** — 池不够时排队/等待而非崩溃的配置方法 + 参数计算关系。
   见 `DELIVERABLE_2_graceful_degradation_config.md`。

## Result（结论一览）

**KV 真实用量（1P1D TP8+DP8 @ conc=128, 8k/1k，吞吐无损）：**

| 腿 | 默认池 | 真实峰值用量 | **真实 KV/卡** | 浪费 |
|----|--------|------------|--------------|------|
| prefill | 105 GiB | 2%（瞬态）| **~2.1 GiB** | ~50× |
| decode | 119 GiB | (running16+transfer1)×10900 | **~5.6 GiB** | ~23× |

**max-total-tokens 扫描（decode, conc=128）：**

| max-total-tokens | KV池/卡 | 峰值利用 | retract | 吞吐 tok/s | 判定 |
|------|------|------|------|------|------|
| 默认(3.9M) | 118.9 GiB | 4% | 0 | 29,418 | 基线 |
| 262144 | 8.0 GiB | 61% | 0 | 29,634 | ✅ 稳 |
| **163840** | **5.0 GiB** | 93% | 0 | 29,618 | ✅ **推荐地板** |
| 147456 | 4.5 GiB | 98% | 0 | 27,972 | ⚠️ 边缘(掉81请求) |
| 131072 | 4.0 GiB | 100% | — | **崩** | ❌ NotImplementedError |

**核心机制发现**：
- **DSv4 没有优雅 retract**：池撑满 → `retract_decode() NotImplementedError` → scheduler 崩。
  所以池必须永不触 100%，"容忍轻微 retract 换省显存"在 DSv4 上不成立。
- **8k/1k 1P1D 是 prefill-bound**：decode KV 从不是瓶颈（conc=512 过载 decode 仍只 20% cap）。
  两腿池都可缩到 <10 GiB/卡而吞吐零损失。

## How to reproduce

见 `REPRODUCE.md`。TL;DR：两节点起 0.5.13 容器（libionic 修复）→ RDMA MVP → 起 1P1D（mooncake）→
扫 max-total-tokens / 测 5态队列 → 用 `extract_pd_stats.py` 出表。

## Folder map

- **`DELIVERABLE_1_kv_usage_formula.md`** — KV 真实用量公式（交付1）
- **`DELIVERABLE_2_graceful_degradation_config.md`** — 优雅回退配置法（交付2）
- `REPRODUCE.md` — 从零复现的有序命令
- `environment.md` — HW/SW 精确环境（镜像 digest、sglang commit、RDMA rail）
- `scripts/` — 启动/router/bench/reset + 分析脚本（verbatim）
- `results/` — `master_results.csv`（扫描主表）+ `pd_5state_analysis.md`（5态队列分析）
- `notes.md` — 坑/弯路/纠错（含 R3→G1→G2→G3 机制修正链、环境事故教训）
- `logs/` — 全部原始 launch/bench log（gzip）
- `working_process.md`（父目录）— 完整调试叙事，含每轮 hypothesis/命令/结果
