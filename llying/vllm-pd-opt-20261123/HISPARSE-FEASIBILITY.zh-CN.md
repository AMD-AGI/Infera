# HiSparse 移植 ROCm 可行性（只读代码研究，2026-11-27）

源码：v0.34.0（`S=vllm-trackA-20261118/src/vllm`，与镜像一致）。下文路径都相对 `S/`。

## 1. v0.34.0 的 HiSparse 怎么工作

- **三层存储**：完整 MLA KV 放在主机内存的 pinned 池里（source 组，`host_resident=True`，`v1/hisparse/layout.py:243-255`）；GPU 上有两块：resident 页（write-back 缓存，主机副本写完后就 unpin，块池可以随时收回，`coordinator.py:88-98`），以及每个请求一块 hot LRU 缓冲。**indexer K cache 整个留在 GPU**（单独一个 `HISPARSE_INDEXER` 组，`layout.py:145-152`）。
- **什么时候取 miss**：在 forward 里逐层取。index group 的 leader 等本层 top-k 算完（`logical_topk_ready`）后，在 copy stream 上跑 `hisparse_resolve_residency`（LDS 哈希 + int16 LRU），再跑 `hisparse_gather_compact` 把 miss 的行拷进 hot 缓冲；attention 要等本层的 event（`runtime.py:943-1001`、`1003-1048`）。所以搬运基本**在关键路径上**，只有共享 indexer 的 follower 层能被 leader 提前预取（`runtime.py:978-980`）。
- **粒度**：从主机取是**逐 token 行**（16 B 向量逐行拷）；D2H 的镜像和 spill 按 64-token 页（`types.py:11-27`）。hot 缓冲默认 `(max_query_len+1)×top_k` 行，MTP4 时是 6×2048=12288 行/请求/层（`runtime.py:131-137`）。
- **支持范围**：只支持 DSA 模型（必须有 `index_topk`，`config/vllm.py:1835-1840`），不支持 DSV4（`layout.py:101`）。只能用 MRV2，不能 PP / DCP / `cudagraph_mode=FULL`，必须开 HMA 和 `--scheduler-reserve-full-isl`（`config/vllm.py:1812-1834,2217-2225`）。后端只有 FlashMLA sparse / FlashInfer sparse 走通用路径（`sparse_mla_attention.py:823-900`）。平台只有 CUDA（`platforms/cuda.py:274`，基类返回 None，`interface.py:200`）。
- **和其他组件的关系**：MTP 已考虑在内（verify 批也会预取，`runtime.py:54-62,1179`）。HiSparse 本身是一个 connector，要和别的 connector 一起用就得套 MultiConnector，不能和 `--kv-offloading-size` 同时开（`config/vllm.py:2133-2141`）。**PD 导入的请求一律要分配 hot 区**（`single_type_kv_cache_manager.py:2442`）。v0.34 的 Mooncake PD connector 完全不认识 host 组（grep `host_resident|hisparse` 是 0 处）；上游 #66502（Mooncake HiSparse 主机池导入）今天才合并，需要 backport。

## 2. ROCm 后端要做什么

| `HiSparseBackend` 方法 | CUDA 实现 | ROCm |
|---|---|---|
| `check_available` / `create_cache_ops` | 4 个 `_C_cache_ops.hisparse_*` 自定义 kernel（`backend.py:86-96`） | CMake 只给 CUDA 编；要 hipify：wave64 里用 32-lane 逻辑 warp、`__ldcg` 换成 nontemporal、主机池地址用 `hipHostGetDevicePointer` 拿（kernel 直接读主机映射内存，零拷贝，不走 DMA） |
| `pin/unpin_tensor` | `cudaHostRegister`（`utils/host_memory.py:13-25`） | 有 `hipHostRegister`（注意 prefill 1.9 TB 层碰到过的 THP 迁移问题） |
| IPC event ×3 | 跨 TP 共享主机池时才用 | ROCm 的跨进程 event 不保证异步 D2H 拷贝的顺序，所以关掉共享池；我们是 DP4、每个 rank TP1，用不到 |
| `swap_blocks_batch` / `is_current_stream_capturing` / `gather_resident_rows` | — | 现成可用 |

**attention 一侧必须改**：`ROCMAiterMLASparseImpl.forward_mqa` 自己做 index 映射，不进 index group（`rocm_aiter_mla_sparse.py:872-875,1156-1229`），AMD 模型的 attention 也没接 hot cache。另外 hot 缓冲是整个多层 slab 上的一个视图，行号会超出 ASM kernel 的 int32 偏移，所以上游方案**放弃 `mla_a8w8_qh16_qseqlen4`，改用 aiter Triton sparse MLA**。要保住 ASM kernel，就得在 aiter 里改成 64 位偏移，或者按层传基址。

## 3. 上游进展

**#60358 "[ROCm] Enable HiSparse on ROCm"**（tjtanaa，open，+476/-68）把上面这些全做了：`ROCmHiSparseBackend`、HIP kernel、AMD DSV3.2/GLM-5.x 的接线、Triton 的 `_forward_mqa_kernel`。限制：只支持 gfx950，模型要 bf16，KV 只能是 `auto`/`bf16`/`fp8`（不支持 `fp8_ds_mla`），不能有 sink。GLM-5.3 FP8 在 TP4、bf16 KV、128k 上下文、并发 16-24 时吞吐约是不开 offload 的 6 倍，并发 1-8 时慢约 10%。维护者 11-19 给了初审，要求补任务级精度评测，**还没合并**。另有 XPU 版 #68941，以及 indexer 下放到主机的 #68771。

## 4. 粗算（每 rank 每步）

代码里的行宽是：我们用 `fp8_e4m3`（per-tensor scale），**每 token 每层 576 B**；656 B 是 `fp8_ds_mla` 的格式，ROCm 后端会直接拒绝。按题目的口径：76 请求 × 5 个 query token × 2048 × 79 层 = 6148 万行/步，全部 miss 时是 35.4 GB。

| miss 率 m | 字节/步 | 50 GB/s | 64 GB/s | 29 GB/s（本机 CPU 层实测批量读回） |
|---|---|---|---|---|
| 1% | 0.35 GB | 7.1 ms | 5.5 ms | 12 ms |
| 5% | 1.77 GB | 35 ms | 28 ms | 61 ms |
| 20% | 7.08 GB | 142 ms | 111 ms | 244 ms |

搬运在关键路径上，所以这些时间基本直接加到 42 ms 的步长上。要把步长增幅控制在约 10%（约 4 ms），按这个口径需要 **m ≲ 0.6%**。实际 5 个 query 的 top-k 高度重叠，并集远小于 5×2048，阈值可能放宽到 2–3%，但需要实测。另有两项固定开销：每层一次 resolve，以及 Triton 替换 ASM（现在 ASM 约 64 µs/层，Triton 只要慢 50%，每步就多 2.5 ms）。

**显存账**：hot 缓冲是 76 × 12288 × 576 B × 79 ≈ **42.5 GB/rank**（最小可以压到 10240 行，约 35 GB）。今天 MLA KV 约 159 GB、indexer 约 19 GB。hot 缓冲固定吃掉约 1/4 的 KV 预算，换来的上下文容量受主机内存限制：每 token 需要 45.5 KB 主机内存，且 prefill 已经占了 1.25–1.9 TB。

## 5. 1 天可行性测试（只测组件，先不跑 e2e）

1. **真实 miss 率**：从现有 decode（不开 HiSparse）抽 1 个 rank，在若干层上 dump 每步的逻辑 top-k（可以挂一个只读 hook），离线回放 LRU（10240 / 12288 行）。统计：每步 host miss 率、5 个 query 的并集大小、新请求的冷启动量。同时看 GLM-5.2 的 `indexer_types` 里有没有 `shared` 层（决定能不能预取）。
2. **MI355X 上散列 576 B 行的 H2D 吞吐**：单卡，行数取 7.8k / 39k / 156k 每层，对比 (a) PR 的零拷贝 kernel，(b) `swap_blocks_batch` 的 SDMA 逐行拷，(c) CPU 先 gather 再一次 DMA。绑本地 NUMA，并记下 prefill CPU 层并发时的 DRAM 争用。
3. **kernel 对比**：在 decode 形状（380 token、top-k 2048、fp8）下测 Triton sparse MLA 和 ASM `qseqlen4` 的耗时。
4. **主机内存**：在计算节点上看可用内存，以及 prefill 池之外还能给 decode 多少 pinned 内存。

## 结论

**技术上可以移植**：上游 #60358 已经在 MI355X 上跑通。我们还需要 backport #66502（Mooncake）、核对 MXFP4 indexer overlay 和 MTP 的 FP8 indexer 是否满足 layout 的要求，并接受改用 Triton kernel，或者自己改 aiter 的 ASM。估计 1–2 周属实。

但对现在的 C152 吞吐**收益很小，可能是负的**。decode KV 的上限只比今天的负载高约 10–15%，HiSparse 最多把这个上限挪开，而它带来的成本有三项：每步的 H2D 搬运（m=1% 就是 +5–12 ms，即 +13–30%）、Triton 换 ASM 的差价、以及 hot 缓冲固定占用的约 42 GB。只有 miss 率实测 ≲1% 且 Triton 不比 ASM 慢时，才可能拿到约 +5–10%。它真正的价值在于支持更长的上下文、更多的长会话，不在于提高当前负载下的吞吐。**建议先做第 5 节的第 1、3 项**；m > 3% 或 Triton 慢 30% 以上就放弃。
