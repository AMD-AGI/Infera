# 交付1：DSv4-Pro 1P1D KV cache 真实用量公式

**问题**：给定 (conc, ISL, OSL) + 部署方案（1P1D TP8+DP8），在承载请求且**服务不掉性能**时，
prefill 和 decode 各自**真实使用**的 KV cache 绝对内存是多少？

**镜像/模型**：sglang 0.5.13, DeepSeek-V4-Pro (MLA+DSA, fp8_e4m3), MI355X 288GiB/卡。
**每 full-token KV 成本 = 32,781.44 B**（0.5.13 DSv4 实测，含 swa/c4/c128/state 加权）。

---

## 结论公式

### decode 腿（KV 驻留主体）

```
decode 真实峰值 tokens/rank = (峰值#running + 峰值#transfer-in) × per_req_cost
  per_req_cost ≈ ISL + OSL + page对齐  ≈ 10,900 tok  (8k/1k 实测 131072/12=10923)
  峰值#running  ≈ min(conc/dp, decode服务率上限)
  峰值#transfer-in ≈ P→D 传输管道深度（随 conc 增长）
decode KV 字节/卡 = 真实峰值 tokens × 32,781 B
```

**实测锚点（8k/1k, DP8）**：
| conc | 峰值 running/rank | 峰值 transfer-in | 真实 tokens/rank | **真实 KV/卡** | 默认池 | 浪费 |
|------|------|------|------|------|------|------|
| 128 | 16 | 1 | (16+1)×10900=185K | **5.6 GiB** | 118.9 GiB | 23× |
| 512 | 25 | 27 | (25+27)×10900=567K | 17.3 GiB | 118.9 GiB | 6.8× |

### prefill 腿（KV 瞬态，不驻留）

```
prefill KV 是瞬态的：算一个 chunk 的 KV → mooncake 传给 decode → 立即释放，从不累积。
prefill 真实 KV ≈ chunk 工作集，与 conc 弱相关，远小于 decode。
```

**实测（conc 流量, gmu0.85, 池 3,441,920=105 GiB）**：
- 峰值 full token usage = **2%** → ~69K tok → **~2.1 GiB/卡**。
- 峰值 #queue-req = 20（请求堆在 prefill 门口 = prefill-bound）；#inflight-req = 6（正传给 decode）。

---

## 一句话结论

**1P1D TP8+DP8 @ conc=128, 8k/1k，服务不掉性能时：**
- **prefill 真实 KV ≈ 2.1 GiB/卡**（瞬态，池 105 GiB 是 50× 浪费）
- **decode 真实 KV ≈ 5.6 GiB/卡**（驻留 running+transfer，池 119 GiB 是 23× 浪费）
- 两腿池都可缩到 **<10 GiB/卡** 而**吞吐零损失**（因 8k/1k 1P1D 是 prefill-bound，KV 非瓶颈）。

## 如何套用到其它 (conc, ISL, OSL)

1. 定 `per_req_cost ≈ ISL + OSL`，再加 page-256 对齐余量（实测 8k/1k 约 +18%，取 1.2×）。
2. 起一次大池 baseline，读 decode 日志 `#running` / `#transfer-req` 峰值（用 `scripts/extract_pd_stats.py`）。
3. `decode KV/卡 = (峰值running + 峰值transfer) × per_req_cost × 32781 B`。
4. prefill 侧读 `full token usage` 峰值 ×池大小（8k/1k 下仅 2%，瞬态）。
5. **注意 P/D bound**：若 decode running 远低于 cap（如本例 24%），说明 prefill-bound，decode 池可大缩；
   若 decode running 贴近 cap，则 decode-bound，池要按 running 峰值精算。

## 证据

- decode 5态实测：`results/pd_5state_analysis.md` + `logs/retry2_decode.log.gz`（baseline conc128）、
  `logs/g3_decode_bigpool.log.gz`（conc512）。
- prefill 实测：`logs/g1_prefill.log.gz`（真实流量 full usage 2%、queue 20）。
- per_req_cost=10923 推导：`logs/g1_decode.log.gz`（131072/12 running 崩点）。
