# U29 观察记录：decode 在高显存目标下偶发崩溃 / 卡死（先不提 issue）

- 编号：之前叫"decode 显存余量崩溃"，没有编号，11-20 记为 U29（已补进 `CANDIDATES.zh-CN.md`）。
- 类型：观察记录，**不提 issue**（3 次故障、原因没定，没有最小复现）。可能的目标：vLLM（显存预留）/ ROCm（hipBLASLt、KFD 驱逐）。

## 现象（P4D4，decode DP4 TP1 MTP3，GLM-5.2 MXFP4，都是单次）

| 轮次 | 配置差异 | 结果 | 签名 |
|---|---|---|---|
| m9b（v3-190） | decode `--max-num-batched-tokens 1024`，显存目标 0.95，aiter AG/RS 覆盖层（v3comm），CPU 层 torch 锁页 | 测量第 12 min DP0 worker 崩溃 | 一个 885 token 的批次上 hipBLASLt `HIPBLAS_STATUS_INTERNAL_ERROR`，随后 `HSA_STATUS_ERROR_OUT_OF_RESOURCES`；AIPerf 因失败 10% 中止 |
| m12（v3-006） | m10 原样重复：0.95、v3comm、精确锁页第一版（匿名 mmap + `hipHostRegister`） | 测量第 12–14 min decode 逐渐变慢直到卡死，`execute_model` RPC 超时 300 s | 主机内核日志 `evict_process_worker [amdgpu] hogged CPU`（第 4 / 8 / 16 / 32 / 64 次）；之前 DP0 在推理中 JIT 编译 `_localize_aiter_prefill_topk_kernel`（decode 走了抢占重算的 prefill 路径） |
| m13（v3-006） | m10 + 路由 `INFERA_DECODE_REQUEST_WEIGHT=100000`，默认 8192，0.95、v3comm、精确锁页第一版 | 一个 decode worker 崩溃，EngineCore 退出 | hipBLASLt（`m 2624 n 1694 k 6144`）`HIPBLAS_STATUS_INTERNAL_ERROR` → `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / "too many resources requested for launch"；崩溃前 40 s 生成速度掉到 30–67 tok/s 后恢复；`evict_process_worker` 计数 64 → 128 |

没出问题的对照：m6、m10（0.95、v3comm，torch 锁页 / 精确锁页第一版）；f1、f2（0.93、v3comm、`hipHostAlloc` 精确锁页，3600 s，各一次）；
**J3（11-19，0.95、U13 = AITER AG/RS 生效、`hipHostAlloc` 精确锁页，3600 s，一次）也没出问题**，decode `fatal` / `out_of_resources` 都是 0。

## 判断（推断，原因未定）

- 共同点：三次故障都发生在 decode 走 prefill 路径（KV 满时抢占重算的混合批次）时，签名是 hipBLASLt 内部错误 / `OUT_OF_RESOURCES`，或伴随 KFD
  进程驱逐的变慢卡死。显存目标 0.95 只给 torch 池外留约 14 GB，scratch / hipBLASLt workspace 等临时分配失败就崩、分配引发驱逐就变慢——这是主线 11-19
  的推断（README 1s）。
- **11-20 补充（与上面的推断相反的数据）**：J3 同样是 0.95 + AITER AG/RS，跑满 3600 s 没出问题（它的 CPU 层用 `hipHostAlloc`）。m12 / m13 用的是精确锁页第一版
  （1.9 TB 匿名 mmap + `hipHostRegister`），m9b 用 torch 锁页但 `--max-num-batched-tokens 1024`。所以"第一版锁页（大块注册的用户内存）引发 KFD 驱逐"
  这个之前被降级的假设又不能排除。每种组合都只有一两次，现在**区分不了**。
- 用 0.93 的 f1 / f2 两次都正常，但 KV 少约 4%。

## 要定原因还需要（都要 GPU，等用户安排）

1. 同一节点、同一配置（0.95、AITER AG/RS、`hipHostAlloc`）连跑 2–3 轮 3600 s，看是否还出现；出现时同时抓内核日志、`amd-smi` 显存、hipBLASLt 日志
   （`HIPBLASLT_LOG_LEVEL`）。
2. 如果复现：对照 0.93 与 0.95，以及关 AITER AG/RS，各一轮。
3. 结论出来后再决定是提 vLLM（decode 侧给临时分配多留余量 / 抢占重算批次的显存估算）还是 ROCm（hipBLASLt 在显存紧张时报 INTERNAL_ERROR 而不是可恢复的 OOM）。
