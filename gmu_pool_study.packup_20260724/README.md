# DSv4-Pro SGLang — 每卡 KV cache 内存 × gmu(mem-fraction-static) 研究

**Ran:** 2026-07-24（单节点 chi2879, 8×MI355X）
**Author:** c_huggingface
**Status:** ✅ PASS — 机制、实测曲线、运行时验证、源码调用链全部闭合。

## Goal

搞清 DSv4-Pro sglang 1P1D 下**每卡 KV cache 实际占多少显存**、它与 **gmu ratio
(`--mem-fraction-static`)** 的关系和机制、以及**手动调节方法**；并追出 KV pool 从 launch
入口到 `torch` malloc 的**真实源码调用链**。

## Result（结论一览）

**每卡 KV pool = gmu 的严格线性函数**（5 点实测 R²=1.000）：

```
KVpool(GiB) ≈ 276.9 × gmu − 130.3       每 +0.01 gmu → +2.77 GB / +216,470 token
```

| gmu | 每卡 KV pool | full_token/卡 | 占卡 VRAM |
|-----|-------------|--------------|----------|
| 0.80 | 91.24 GB | 7,053,056 | 31.7% |
| 0.85 | 105.08 GB | 8,135,424 | 36.5% |
| 0.90 | 118.93 GB | 9,217,792 | 41.3% |
| 0.92 | 124.47 GB | 9,650,944 | 43.2% |

- **机制**：gmu 启动时圈定 `weights+KV pool` 占卡 VRAM 的比例；`KVpool ≈ gmu×total − weights`，
  一次性静态分配、**运行时永不增长**（不够则驱逐请求 retract，不再 malloc）。
- **P/D 无关**：同 gmu 下 prefill==decode 池大小逐字节一致；role 只改默认 gmu（P0.85/D0.90）。
- **运行时验证**：真跑 8k/1k conc64×128 → 峰值 KV 占用 1–5%、**retract=0**，远在池内。
- **反算权重** ≈ 11.1 GiB/卡（斜率 288−276.9），与 DSv4 fp8 ÷TP8 吻合。

## 两个常见追问（详见 `FAQ_allocation_and_capacity.md`）

- **Q1 一次性还是增量？** → 完全一次性 static，运行时只在固定 buffer 内切页；不够=驱逐非补分配。
- **Q2 容量受什么影响？** → 决定池大小的：gmu / GPU总显存 / 权重 / kv-dtype / swa-ratio / page-size。
  不改池的（只做消费/上限）：max-running-requests / batch / context-len / ISL / OSL。
  （唯一例外：unified 路径下 max-running 经 num_slots 撑大 SWA 区物理页。）

## How to reproduce

见 `REPRODUCE.md`。TL;DR：单机起 decode 单腿扫 gmu∈{0.80..0.92} 读 `DSV4 memory calculation`
日志行 → 建 gmu→pool 曲线 → mix 真跑验证 retract=0。全单机，~30min。

## Folder map

- **`REPORT.md`** — 完整分析报告（结论/机制/实测表/调节方法/environment）。
- **`CALLCHAIN.md`** — KV pool 从 launch 入口到 `torch.zeros` 真实调用链（容器实机源码, 逐跳行号）。
- **`FAQ_allocation_and_capacity.md`** — Q1(分配时机) + Q2(容量影响因素) 源码级解答。
- `REPRODUCE.md` — 从零复现的有序命令。
- `environment.md` — HW/SW 精确环境（镜像 digest、sglang commit、模型校验）。
- `scripts/` — 扫描 + 真跑验证脚本（verbatim）。
- `results/` — CSV + 解析脚本 + 主表。
- `logs/` — 8 份原始 launch log（gzip；KV pool 分配行的原始证据）。
- `src_refs/` — CALLCHAIN 引用的 4 个 sglang 源文件（从容器原样拷出, v0.5.15.post1）。
- `notes.md` — 坑/弯路/纠错（含"真实 malloc 是 unified 路径"这条对初版的修正）。
