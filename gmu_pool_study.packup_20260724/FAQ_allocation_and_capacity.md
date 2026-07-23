# FAQ：KV cache 分配时机 & 容量影响因素（源码级）

两问都从容器实机源码（`/sgl-workspace/sglang` @ v0.5.15.post1, commit 0b3bb0c）+ 本轮实测得出。
行号引用见同目录 `CALLCHAIN.md` 与 `src_refs/`。

---

## Q1：KV cache 是一次性分配，还是可增量补充？

**答：完全一次性（static），运行时永不增长、永不再 malloc。**

1. **唯一分配点在启动**：`init_memory_pool`（mixin:1323）→ 每 effective 层一次
   `torch.zeros(...)`（`deepseek_v4_memory_pool.py:430`，unified 路径）。这就是
   `--mem-fraction-static` 里 **static** 的字面含义：weights + KV pool 启动时一次性圈定、常驻。

2. **运行时只在这块固定 buffer 内切页**：allocator 的
   `available_size = len(free_pages) × page_size`（`mem_cache/allocator/base.py:57`），
   `alloc`/`free`（:105/:109）只是 page index 的记账（游标切分）。**全程没有任何运行时
   KV 显存 grow / resize_ / torch.cat 出新 buffer**。free-list 张量的 `torch.empty`（base.py:82）
   只是重建索引数组，不分配 KV。

3. **不够用时是"驱逐"，不是"补分配"**：`ScheduleBatch.retract_decode`
   （`managers/schedule_batch.py:2471`，注释 "Retract the decoding requests when there is not
   enough memory"）把 decode 请求踢回 waiting queue 重排，等页释放再续跑——**而非**向 GPU 再要显存。
   调度日志里的 `#retracted-req` 计数就是这个动作（本轮实测全程 = 0）。

4. **想让池变大只能降 gmu 重启**：那是重走一遍启动分配，不是运行时增量。
   → 实践含义：pool 大小必须在**启动前**用 gmu 定对；跑起来之后再想扩 KV，只能 kill→改 gmu→重启。

**实测佐证**（Step 3，mix gmu=0.90, 真跑 8k/1k conc64×128）：
peak full-attn usage 1% / swa usage 5% / #running 13 / **retract=0** —— 池启动即锁定、运行时远在池内。

---

## Q2：KV cache 容量受哪些因素影响？

分两类：**决定池物理字节的**（改这些才改显存） vs **只做消费/上限、不改池的**。

### A. 决定池物理大小的因素（改这些 → 改 KV 显存）

| 因素 | 机制 | 代码锚点 |
|------|------|---------|
| **gmu (`--mem-fraction-static`)** | **主旋钮，线性**。`available_bytes = free_now − pre_load×(1−gmu)`；`full_token = available/13735` | mixin:115 / pc:752 |
| **GPU total mem** | `pre_load_free`≈整卡；`gmu×total` 圈定 static 上限。实测斜率 276.9 ≈ 288−权重 | mixin:104-108 |
| **模型权重大小** | 权重先占，剩下才是 KV：`KVpool ≈ gmu×total − weights`（实测 weights≈11.1GB/卡）。换量化/换模型直接改 | mixin:115 |
| **kv-cache-dtype** | fp8_e4m3 vs bf16 决定每 token 字节；DSv4 复合 `kv_bytes=qk_nope+qk_rope×2+8=584 B/token/层` | pc:610 |
| **`--swa-full-tokens-ratio`** | swa 池 token 数 = `full×ratio`（本轮 0.15）；改比例改 swa 池占比 | pc:653 |
| **`sliding_window` + `--max-running-requests`** | ⚠️**仅 unified(MI355X ROCm) 路径**：swa 物理页 `swa_pages = num_slots × swa_ring_size = (max_run(+extra)+1) × (window+spec)`。SWA 区物理大小随这两者走 | dmp:416 |
| **`--max-total-tokens`（用户 cap）** | 只能**向下**砍池：`min(profiled, user_limit)`（填得比 profiled 大会被忽略并警告） | mixin:1202-1211 |
| **`--page-size`** | 池容量向下取整到 page 倍数（须为 128 的倍数）；影响对齐损耗 | pc:652 |

### B. 不影响池大小的因素（是池的消费者/上限，不是决定者）

| 因素 | 为什么不改池 |
|------|------------|
| **`--max-running-requests`** | 从池容量**反推**的并发上限：`min(requested/dp, token_capacity//2)`（mixin:1235），是池的消费者。⚠️唯一例外：unified 路径下它经 `num_slots` 撑大 SWA 区物理页（见 A 表） |
| **batch size** | 运行时概念，在固定池内切页；池不随 batch 变 |
| **`--context-length`** | 只用于**估算** max_num_reqs（mixin:1229 `token_capacity/context_len×512`），不改池字节。决定"池能装几个满上下文请求"（gmu0.90: full 9.22M ÷ 9472 ≈ 973 个/rank） |
| **ISL（输入长度）** | 请求级：每 token 占页，ISL 大 → 单请求吃更多页 → 池能并发的请求数少。这是**运行时占用**，不是池大小 |
| **OSL（输出长度）** | 同理请求级：decode 每步 +1 token 占 1 页，OSL 大 → 单请求生命周期内累积占更多页。不改池总量 |

### 一句话总结

> **池大小 = f(gmu, GPU总显存, 模型权重, kv-dtype, swa-ratio, page-size)**，启动一次性定死。
> `max-running / context-len / batch / ISL / OSL` 都是在这块固定池里**怎么用**，不改池本身。
> **唯一例外**：unified(MI355X) 路径下 `max-running-requests` 会经 num_slots 撑大 SWA 区物理页。

### 与本研究主结论的关系
主结论 `KVpool_GB ≈ 276.9×gmu − 130.3`（线性）成立的前提是 A 表其他因素（total mem/权重/dtype/
swa-ratio/page-size）固定、只扫 gmu。要预测任意配置的 KV pool：先按 A 表定 `bytes_per_full_token`
（DSv4 fp8=13735.04），再 `full_token = (gmu×total − weights − c128_state_fixed) / bytes_per_full_token`。
