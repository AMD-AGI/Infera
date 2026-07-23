# 交付2：DSv4-Pro KV cache 配置法 + 优雅回退（不崩）

**问题**：如何配置 KV cache 大小，保证服务在 KV 不够用时**性能正常回退、排队、等待，而不是崩溃**？
参数之间的计算关系是什么？

---

## ⚠️ 首要事实：DSv4 没有优雅 retract

DSv4 unified KV pool **没实现 retract**。当 decode 池撑满、scheduler 尝试驱逐请求时：
```
retract_decode() @ managers/schedule_batch.py:2423  →  raise NotImplementedError()
→ scheduler 进程崩 → worker 掉线 → router 大量请求 ERROR
```
**证据**：R3/G1（池 131072 撑到 100%）decode scheduler 抛 NotImplementedError 崩，router 报 7023 个 ERROR。

**推论**：sglang 常规引擎"KV 不够就 retract 排队"的优雅回退，**在 DSv4 上不存在**。
→ **必须把池 sized 到运行时永不触 100%**，让多余负载堵在**准入队列（池外）**而非池内 retract。

---

## 核心自洽约束

```
池(max-total-tokens) ≥ (max_running/dp) × per_req_cost × 安全系数(1.2)
  ⟺  自然running ≤ max_running/dp ≤ 池 / per_req_cost
  per_req_cost ≈ ISL + OSL + page对齐 (8k/1k ≈ 10,900)
```

`max_running/dp` 必须落在 **[自然running数, 池/per_req]** 这个**中间带**：
- **太小**（< 自然 running）→ 掐死 decode（见"塌"）
- **太大**（> 池/per_req）→ 池撑爆触 retract → NotImplementedError 崩

## 三态行为（实测）

| 态 | 条件 | 现象 | 实测 |
|----|------|------|------|
| **稳** | 池 ≥ max_running×per_req | running 满速、retract=0、吞吐满 | baseline/R1/R2 |
| **崩** | 池 < running 实际需求 | retract → **NotImplementedError 进程死** | R3(131072/64), G1(131072/12) |
| **塌** | max_running << 自然 running | decode 被掐死、transfer 管道占满池、吞吐崩（不崩进程）| G2(131072/8): 吞吐 8-174 vs 3268 |

## 参数联动表

| 参数 | 作用 | 设置原则 |
|------|------|---------|
| `--max-total-tokens` | KV 池大小(token) | ≥ (max_running/dp)×per_req×1.2；物理显存 = tok × 32,781 B |
| `--max-running-requests` | decode 并发硬闸(global, ÷dp 生效) | **设在 [自然running, 池/per_req] 中间带**。别掐死(塌)、别撑爆(崩)|
| `--context-length` | 单请求最大 token | 决定 per_req_cost 上界；压到实际 ISL+OSL 可省池 |
| `--num-reserved-decode-tokens` | 每活跃请求预留解码步数(默认512) | 调小→准入激进(池利用高但抖动风险)；调大→保守。**回退行为第二旋钮** |

## 优雅排队的正解

1. 池 sized 到 `(峰值running + 峰值transfer) × per_req × 1.2` 覆盖目标 conc。
2. `max_running` 设 ≥ 自然 running 数（别掐死），同时 ≤ 池/per_req（硬限防 admission 触 retract）。
3. 超出负载堵在 **state3 admission 队列**（池外等待，`#queue-req` 涨）→ retract=0、不崩、请求排队。

## ⚠️ 本 workload 的重要限定

**8k/1k @ 1P1D 是 prefill-bound**：conc 拉到 512（4×）decode running 仍只 20% cap、池 6%、retract=0。
真实过载**先触发 router circuit-breaker 拒绝**（prefill 算力跟不上），decode KV 层的优雅排队**在此
workload 下不是主约束**。

→ **decode KV 优雅排队要成为主回退点，需 decode-bound workload**：短 ISL 长 OSL（如 512/4096）、
高 P:D 比（如 2P1D+）。那时 decode admission 队列(state3)才会堆积成为主回退点。

## 推荐配置（8k/1k conc=128 生产）

```
decode: --max-total-tokens 163840 (5.0 GiB/卡, 峰值93%, 零 retract, 吞吐满)
        --max-running-requests 512 (=64/rank, 自然running16 << 64 << 池/per_req≈15... 见下注)
prefill: 池可缩到 <10 GiB（瞬态 KV），--max-total-tokens 留余量即可
```
> **注**：163840/10900 ≈ 15，而 64/rank cap 看似 > 15 会撑爆——但实际全局 conc=128÷dp8=16 running
> 才是真闸（max_running=64 从不触及），16×10900=174K > 163840 会擦边。R2 实测峰值 93% retract=0 稳，
> 因池自调节 running 卡在 cap 下。**若要更保守留余量，用 262144(8GiB) 池 + conc≤128**。

## 证据

- 崩：`logs/r3_decode_mtt131072.log.gz`（retract_decode NotImplementedError 时序）。
- 塌：`logs/g2_decode.log.gz`（running 2.8、transfer 9.2、吞吐塌）。
- 稳：`logs/r2_decode_mtt163840.log.gz`（93%、retract=0、29.6K tok/s）。
- 扫描主表：`results/master_results.csv`；5态分析：`results/pd_5state_analysis.md`。
