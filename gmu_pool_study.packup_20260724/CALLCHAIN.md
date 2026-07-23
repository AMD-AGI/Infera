# DSv4-Pro KV pool 真实调用链：从 launch 入口到 torch malloc

**来源**：容器 `dsv4_pd_sgl` 实机源码 `/sgl-workspace/sglang`（editable install，
`git rev 0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` = **tag v0.5.15.post1**）。
**全部行号为容器真实源码**（非 WebFetch），已逐跳 Read 核对。
运行 env 关键：`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` + ROCm → **走 unified 分支**（见 §4）。

---

## 一图看全（调用栈）

```
ModelRunner.init_memory_pool(pre_model_load_memory)                     [mixin:1323]
│  └─ _resolve_memory_pool_config(pre_model_load_memory)                [mixin:1295]
│     ├─ available_bytes = _profile_available_bytes(pre_model_load)     [mixin:104]
│     │     └─ get_available_gpu_memory(...)  ← torch.cuda.mem_get_info [utils/common]
│     │        rest = free_now − pre_load_free × (1 − mem_fraction_static)   [mixin:115]
│     │        return int(rest * 2^30)  # bytes                         [mixin:139]
│     ├─ configurator = create_memory_pool_configurator(self)          [pool_configurator:774]
│     │     └─ is_deepseek_v4 & is_hybrid_swa → DSV4PoolConfigurator    [pool_configurator:778]
│     ├─ config = configurator.calculate_pool_sizes(available_bytes,page_size) [pc:734]
│     │     ├─ full_token = int(avail_for_tokens / bytes_per_full_token)      [pc:752]
│     │     │     bytes_per_full_token = _get_bytes_per_full_token()          [pc:609]  (=13735.04)
│     │     ├─ sizes = _compute_dsv4_sizes(full_token, page_size)            [pc:651]
│     │     │     swa=full×0.15 ; c4=full//4 ; c128=full//128                [pc:653-658]
│     │     └─ logger.info("DSV4 memory calculation: ... full_token=N")      [pc:755] ★实测抓的行
│     ├─ constrained = _apply_token_constraints(...)  # user cap/page align  [mixin:1310]
│     └─ finalize_with_max_running_requests(config)                          [mixin:1319]
└─ _apply_memory_pool_config(config)                                    [mixin:1264,1333]
   └─ (在 pool 创建段) pool_cls = DeepSeekV4TokenToKVPool                [mixin:686]
      self.token_to_kv_pool = pool_cls(swa_size=…, c4_size=…, c128_size=…)   [mixin:690]
      └─ DeepSeekV4TokenToKVPool.__init__                              [dsv4_mem_pool:451]
         ├─ logger.info("Initialize DeepSeekV4TokenToKVPool with swa_size=…") [dmp:490] ★实测抓的行
         ├─ stage_layer_num=len(ratios); c4_layer_num=Σ(r==4); c128=Σ(r==128) [dmp:556-558]
         └─ if is_unified_kv_triton():   ← 我们的 ROCm+env 命中             [dmp:566-568]
            └─ DeepSeekV4UnifiedKVPool(num_slots, num_blocks=c128_size, …)   [dmp:578]
               └─ __init__: for ratio in stage_ratios:                       [dmp:427]
                  compress_pages = num_blocks × K_PER_BLOCK[ratio]           [dmp:428]
                  ★★ torch.zeros(swa_pages+compress_pages, head_dim=512,     [dmp:430] ← 真·malloc
                                 dtype=bfloat16, device=cuda)
                  （每 effective layer 一个 buffer，在 torch.cuda.use_mem_pool 上下文内）
```

---

## 逐跳详解

### 1. gmu → 可用字节预算（真实测量点）
`model_runner_kv_cache_mixin.py:104 _profile_available_bytes`：
```python
available_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id,
                          distributed=world_size>1, cpu_group=...)   # :108  torch.cuda.mem_get_info + TP MIN all-reduce
rest_memory = available_gpu_memory - pre_model_load_memory * (1 - self.mem_fraction_static)  # :115
...
return int(rest_memory * (1 << 30))   # bytes                        # :139
```
- `available_gpu_memory` = **加载权重后**的当前空闲（已扣权重）。
- `pre_model_load_memory` = 加载权重**前**的空闲（≈整卡）。
- 语义：`rest = free_now − pre_load × (1−gmu)`；`pre_load×(1−gmu)` 是给 activation+cudagraph 的动态预留，
  权重占用天然被算进（因为 free_now 已扣权重）。rest≤0 时报错并建议最小 gmu（:122-137）。

### 2. 选配置器（DSv4 专用）
`pool_configurator.py:774 create_memory_pool_configurator` → `:778 is_deepseek_v4 & is_hybrid_swa → DSV4PoolConfigurator`。

### 3. 字节预算 → full_token → 多池 token 数
`pool_configurator.py:734 calculate_pool_sizes`：
```python
full_token = int(available_bytes / self.bytes_per_full_token)              # :746 (预估，为算 c128_state_fixed)
available_bytes_for_tokens = max(available_bytes - c128_state_fixed_bytes, 0)  # :751 先扣 1.01GB 请求态
full_token = int(available_bytes_for_tokens / self.bytes_per_full_token)   # :752 ★核心除法
sizes = self._compute_dsv4_sizes(full_token, page_size)                    # :754
logger.info("DSV4 memory calculation: bytes_per_full_token=…, available_bytes=… GB, full_token=…")  # :755
```
`bytes_per_full_token`（`:609 _get_bytes_per_full_token`，=13735.04）：
```python
kv_bytes = qk_nope_head_dim + qk_rope_head_dim*2 + 8      # :610  = 584  ← MLA cell/token/层
indexer_bytes = indexer_head_dim + indexer_head_dim//128*4  # :613  = 132
return (swa_ratio*kv_bytes*num_layers_total                # :639  swa 覆盖全部层
      + (1/4)*kv_bytes*num_layers_ca4                       # :640  c4 = ratio-4 层
      + (1/128)*kv_bytes*num_layers_ca128                   # :641  c128 = ratio-128 层
      + (1/4)*indexer_bytes*num_layers_ca4                  # :642  c4 indexer
      + swa_ratio*c4_state_ratio*c4_state_bytes*num_layers_ca4   # :643 c4 attn-state
      + 0*c128_state_bytes*num_layers_ca128                 # :644 c128_state_ratio=0（请求态，后置）
      + swa_ratio*c4_state_ratio*c4_indexer_state_bytes*num_layers_ca4)  # :645 c4 indexer-state
```
`_compute_dsv4_sizes`（`:651`）固定比例劈池：
```python
swa  = int(full_token * swa_ratio) // page_size * page_size   # :653  swa_ratio=0.15
c4   = full_token // (4 * c4_shrink_factor)                    # :657  =full//4
c128 = full_token // 128                                       # :658
```

### 4. 建池 → 真正 malloc（unified 分支）
`mixin:686` `pool_cls=DeepSeekV4TokenToKVPool` → `:690` 传 `swa_size/c4_size/c128_size`（来自 config）。
`deepseek_v4_memory_pool.py:451 __init__` → `:490` 打 `Initialize DeepSeekV4TokenToKVPool with…`（实测第二锚点）。
分层（`:556-558`）：`stage_layer_num=全部层`、`c4_layer_num=Σ(ratio==4)`、`c128_layer_num=Σ(ratio==128)`。

**关键分叉**（`:566-568`）：`is_unified_kv_triton() = is_hip() and env==unified_kv_triton`
（`layers/attention/dsv4/unified_kv_kernels/env_gate.py:10`）。我们 **MI355X(ROCm) + env 命中 → True**：
```python
self.unified_kv_pool = DeepSeekV4UnifiedKVPool(num_slots=num_req_slots,
                          num_blocks=c128_size, swa_ring_size=sliding_window+spec_extra, …)  # dmp:578
```
`DeepSeekV4UnifiedKVPool.__init__`（`:400`）**真·malloc**：
```python
self.head_dim = qk_nope_head_dim + qk_rope_head_dim          # :414  =512
self.swa_pages = num_slots * self.swa_ring_size              # :416
for ratio in stage_ratios:                                   # :427  每 effective 层
    compress_pages = self.num_blocks * self.k_per_block[ratio]   # :428
    bufs.append(torch.zeros(self.swa_pages + compress_pages,     # :430 ★★★ 显存在此分配
                            self.head_dim, dtype=torch.bfloat16, device=device))
```
在 `memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE)` + 可选 `torch.cuda.use_mem_pool` 上下文内（`:421-426`）。

> **对上一份报告的修正**：MI355X 实走 **unified** 路径，每层是一块 **`[swa_pages+compress_pages, 512] bf16`** 统一 buffer
> （SWA 环 + 压缩区拼在一起，row-partitioned）。三独立子池 `DeepSeekV4SingleKVPool`（`:51`，
> 布局 584B/token/uint8、page pad 到 576，`create_buffer:107→torch.zeros:119`）是**非 unified/CUDA 路径**才走。
> 两条路径的 **token 容量规划完全相同**（都由 §3 的 configurator 算 swa/c4/c128），只是**物理排布**不同：
> unified 把三部分拼成一块 bf16 buffer；非 unified 分三块 uint8 buffer。gmu→pool 的线性关系（报告主结论）不受影响。

### 5. 收尾
`_apply_memory_pool_config`（`mixin:1264`）把 config 各字段赋到 `self.*` 后，`init_memory_pool` 末尾（`:1335`）打
`Memory pool end. avail mem=… GB`（实测第三锚点，=建完池后的剩余显存）。

---

## 实测锚点 ↔ 代码行对应（gmu=0.90 per-rank）
| 实测日志行 | 代码位置 | 值 |
|-----------|---------|----|
| `DSV4 memory calculation: bytes_per_full_token=13735.04, available_bytes=118.93 GB, c128_state_fixed=1.01 GB, full_token=9217792` | `pool_configurator.py:755` | 核心预算+除法 |
| `DSV4 pool sizes: full=9217792, swa=1382656, c4=2304448, c128=72014, ...` | `pool_configurator.py:_to_config:703` | 分池 token 数 |
| `Initialize DeepSeekV4TokenToKVPool with swa_size=1382656 c4_size=2304448 c4_logical_size=2304448 c128_size=72014 ...` | `deepseek_v4_memory_pool.py:490` | 池构造入参 |
| `Memory pool end. avail mem=44.02 GB` | `model_runner_kv_cache_mixin.py:1335` | 建池后剩余 |

真·显存分配 = `deepseek_v4_memory_pool.py:430 torch.zeros(...)`（unified，每层一块 bf16 buffer）。
```

