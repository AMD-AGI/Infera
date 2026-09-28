# 偶发约 11 秒的请求延迟：prefill 现场编译 tilelang 内核

日期：2026-09-24。现象最早出现在 C 组"从主机加载"专项测试的复用阶段（[DECODE-HICACHE.zh-CN.md](DECODE-HICACHE.zh-CN.md)）：8 个请求中有 2 个耗时约 12 秒，其余约 1 秒。

## 结论

**原因：** prefill 使用 tilelang 做 DSA 稀疏注意力。当某一批的 query token 数第一次落进一个新的档位时，prefill 会现场 JIT 编译两个内核，耗时约 11 秒。

**影响：** 编译期间，同一 prefill 的其他 DP rank 也会停在 DP attention 的同步上，所以整个 prefill 都被卡住。

**与本任务改动无关：** 这不是 decode radix 或 decode HiCache 引入的。B 组（decode 未开 HiCache）也出现过一次；decode 在运行期间从未编译。

**触发条件：**
- 只在上下文超过 topk（2048 token）、走稀疏注意力路径时才会触发。GSM8K 这类短上下文不会触发。
- 每个档位在一个容器里只编译一次。多个 DP rank 共用同一份磁盘缓存，谁先遇到谁编译。
- 缓存位于容器内的 `~/.tilelang/cache`，容器重启后失效。

## 证据

**1. 延迟发生在 prefill 开始计算之前。** 从 Router 派发到 prefill 的 `Prefill batch` 日志：

| 场景 | 慢请求 | 同组其他请求 |
|---|---|---|
| C 组复用阶段 | 11.39 秒、11.19 秒 | 0.19–0.85 秒 |
| B 组探测 | 11.56 秒 | — |

几个慢请求生成的 token 数正常，prompt 长度和 prefill 缓存命中与同组其他请求相同，并且是串行发送的，没有排队。

**2. 每次卡顿的时间窗口里，prefill 都在编译。** 日志先出现 `[fused_moe] using 2stage ... for ('gfx950', 256, M, ...)`，约 5 秒后出现 C++ 编译告警：

```
tmpXXXX.cpp:364: if ((sumexp[0] == 0.000000e+00f))
```

`sumexp` 就是稀疏注意力内核中的变量。全部运行期间：
- prefill 的运行时编译只出现在这几个卡顿窗口里；
- decode 只在启动（抓 CUDA graph）时编译；
- 同一时刻，容器中新写入了 `/root/.tilelang/cache/<hash>/kernel_lib.so`、`/tmp/tmp*.cpp`，以及一个 Triton 内核（`/root/.cache/sglang/triton/`）。

**3. 可以按需复现。** 脚本：[stall_probe.py](../scripts/stall_probe.py)，结果在 `evidence/stall/`。

| 运行 | 做法 | 结果 |
|---|---|---|
| run1 | 60 个串行长前缀请求，新 token 约 30–45 个，落在已编译过的档位 | 0 次卡顿；冷请求约 3.5 秒，热请求约 0.8 秒 |
| run2 | 加长问题，使新 token 约 100 个，落入新档位（inner_iter=4） | 第 1 个 11.22 秒，后 3 个 1.3–1.4 秒 |
| run3 | 新 token 约 230 个（inner_iter=8） | 第 1 个 11.06 秒，后 3 个 1.3–1.8 秒 |

**4. 卡顿期间的调用栈**（`evidence/stall/run3-bucket256/stacks-000-prefill.txt`，py-spy `--nonblocking`）：

```
sglang::scheduler_DP2_TP2  (正在编译)
  subprocess.communicate <- tilelang/jit/adapter/libgen.py compile_lib
  <- tilelang/jit/kernel.py <- tilelang/cache/kernel_cache.py cached
  <- tilelang_sparse_fwd (sglang/kernels/ops/attention/dsa/tilelang_kernel.py:1350)
sglang::scheduler_DP1_TP1 / DP3_TP3  (等待中)
  all_gather_into_tensor <- dp_attn.prepare_mlp_sync_batch <- get_next_disagg_prefill_batch_to_run
sglang::scheduler_DP0_TP0  (空闲，等待请求广播)
```

## 档位是怎么定的

`tilelang_sparse_fwd` 在 ROCm fp8 上会编译两个内核：`sparse_mla_fwd_decode_partial_fp8`（partial）和 `sparse_mla_fwd_decode_combine`（combine）。它们按 `inner_iter` 特化：

```
inner_iter = _pick_inner_iter(q.shape[0], ni = topk/64 = 32, cu = 256, block_per_cu = 2)
```

其中 `q.shape[0]` 是这一批在该 rank 上的 query token 总数。在 gfx950 上，档位与本次观察到的触发如下：

| query token 数 | inner_iter | 本次观察 |
|---|---|---|
| < 32 | 1 | 29 个 token（C 组第二次卡顿） |
| 32–63 | 2 | 40 个 token（C 组第一次卡顿） |
| 64–127 | 4 | 72 个 token（B 组）；run2 |
| 128–255 | 8 | run3 |
| 256–511 | 16 | 本次未触发 |
| ≥ 512 | 32 | 冷启动时 8192 token 的大块 prefill（C 组写入阶段的编译，按时间推断，未抓调用栈） |

最多 6 档。每档编译 partial 和 combine 两个内核，约 11 秒。所有档位都编译过之后，这个容器里就不会再卡。

## 影响

- **只影响长上下文负载。** 例如 AgentX 这类上下文超过 2048 token 的请求。一个新容器在开始的几分钟里，最多卡 6 次，每次约 11 秒，而且每次整个 prefill 都会停下来，影响期间所有在 prefill 中的请求。
- **测试时的影响：** AgentX 有较长的 warmup，一般能吸收掉这几次卡顿。但每次重启容器都会重新发生。如果 warmup 很短，或者直接测单个请求的 TTFT，结果就会受影响。
- **与 decode radix、decode HiCache 无关。** 之所以在本任务中暴露，是因为前缀复用测试会产生"长前缀 + 很短新增部分"的 prefill，恰好落到没编译过的小档位上。

## 修复建议（尚未实施）

1. **把 tilelang 缓存放到宿主机上（首选，改动最小）。** 在 `engine.sh` 中，参照 AITER 缓存的做法，按镜像 ID 挂载一个宿主机目录，并设置 `TILELANG_CACHE_DIR`：
   ```
   -v /tmp/tilelang-cache-<uid>/<image-id>:/tilelang-cache -e TILELANG_CACHE_DIR=/tilelang-cache
   ```
   这样同一镜像只在第一次遇到某个档位时编译一次。Triton 缓存（`/root/.cache/sglang/triton`）也可以同样处理。
2. **启动后预热。** 在 launch 完成后、开始测试前，对 prefill 发一组请求：先发一个超过 2048 token 的前缀，再用不同长度的后缀（约 16、48、96、192、384、600 token），把 6 个档位都触发一遍。约 1 分钟。可以和第 1 条一起使用。
3. **上游修复（长期）。** 在服务启动时预编译这几个变体，或者让 `inner_iter` 不再成为编译期常量。
