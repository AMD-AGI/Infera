# Decode 侧无法开启 KV-aware 的原因

日期：2026-09-24。适用栈：GLM-5.2-MXFP4，PD 分离（1P1D，各 TP8/DP8，DP attention），decode 使用 EAGLE MTP（steps 5 / topk 1 / draft 6），Mooncake 传输，镜像 `infera-sglang:v0519-yihou-0917-nextnfix-hicache`（SGLang `0.5.19.dev20260917+ga9fb1c3238`）。

## 结论

decode 打不开 KV-aware 的根本原因是 **SGLang 不允许 decode radix cache 与投机解码同时开启**。只要 decode 开了 MTP（EAGLE），decode 就只能使用 `ChunkCache`。`ChunkCache` 没有 radix 树，不产生任何 KV 事件，所以 Router 对 decode 没有可用的 KV 视图。Infera 和测试脚本都知道这条限制，并在各自的层上提前绕开，结果是每一层都不会报错，只是悄悄关掉。

prefill 能开，是因为 prefill 没有启用投机解码（`speculative_algorithm=None`），默认就有 radix cache（UnifiedRadixCache + HiCache），因此能正常发布事件。

SGLang 的这条限制是 decode radix cache 初版 PR（[sgl-project/sglang#19746](https://github.com/sgl-project/sglang/pull/19746)）加入的一刀切保守限制。PR 描述和评审中都没有给出技术原因；PR 下多次有人追问（2026-06、07），至今没有答复。上游目前只有针对 DeepSeek-V4 的 MTP 支持 PR（[#27831](https://github.com/sgl-project/sglang/pull/27831)、[#31097](https://github.com/sgl-project/sglang/pull/31097)，均未合入），ROCm 的 DSV4 PR（[#30929](https://github.com/sgl-project/sglang/pull/30929)）也明确排除了投机解码。通用模型没有现成方案。

## Decode 侧“KV-aware”包含两件事

在 Infera 中，对 decode 打开 `--enable-kv-events` 实际上同时要求：

1. **引擎侧 decode radix cache**：`--disaggregation-decode-enable-radix-cache`。decode 保留已完成请求的前缀 KV，新请求只向 prefill 请求未命中的增量（`decode_prefix_len`）。
2. **路由侧 decode KV 视图**：decode 各 DP rank 发布 `BlockStored/BlockRemoved`，Router 以 `--kv-decode-overlap-weight` 为 decode 打分。

第 2 项完全依赖第 1 项。没有 radix 树就没有事件，因此真正的阻断点在第 1 项。

## 四层阻断链

### 第 1 层：测试脚本固定关闭 decode 事件

最近 llying 运行使用的 `p8d8-c80-c112-tracing-aus-20260922/scripts/bench-harness/engine.sh` 只给 prefill 传 KV events，decode 分支固定为关闭：

```188:201:llying/p8d8-c80-c112-tracing-aus-20260922/scripts/bench-harness/engine.sh
if [[ "$role" == prefill ]]; then
    engine_args+=(--disaggregation-bootstrap-port "$bootstrap_port")
    if [[ "$ENABLE_KV_AWARE" == 1 ]]; then
        engine_args+=(
            --enable-kv-events --kv-events on
            --kv-events-bind "tcp://0.0.0.0:$kv_event_port"
            --kv-snapshot-port "$snapshot_port"
        )
    else
        engine_args+=(--no-enable-kv-events --kv-events off)
    fi
else
    engine_args+=(--no-enable-kv-events --kv-events off)
fi
```

同一脚本第 76-79 行还拒绝 decode HiCache 与 MTP 同时开启。yihou、yaocheng 的同名脚本也是同样写法。

### 第 2 层：Infera 在 MTP 下不追加 decode radix 开关

`infera/engine/sglang/args.py` 在 decode 打开 KV events 且后端为 mooncake 时，会自动追加 `--disaggregation-decode-enable-radix-cache`。但设置了 `--speculative-algorithm` 时会跳过，只输出一条 INFO：

```301:318:infera/engine/sglang/args.py
    if (
        known.enable_kv_events
        and sglang_parsed.disaggregation_mode == "decode"
        and getattr(sglang_parsed, "disaggregation_transfer_backend", None) == "mooncake"
        and _DECODE_RADIX_CACHE_FLAG not in remaining
    ):
        # SGLang rejects this flag under speculative decoding, so appending it
        # kills an EAGLE/MTP decode leg at parse time. Skipping it costs only the
        # decode-side KV view; prefix-aware routing runs on the prefill one.
        if getattr(sglang_parsed, "speculative_algorithm", None) is not None:
            logger.info(
                "kv-events on, but --disaggregation-decode-enable-radix-cache is "
                "incompatible with --speculative-algorithm %s; not appending it. "
                "The decode leg will use SGLang's chunk cache and contribute "
                "little to the router KV view; prefix-aware routing runs on the "
                "prefill-side view.",
                sglang_parsed.speculative_algorithm,
            )
```

此时 decode 仍会注册 `kv_events_endpoint` 和 `kv_block_size`（`infera/engine/sglang/worker.py` 第 102-115、146-156 行），但 SGLang 用的是 `ChunkCache`，其 `reset()` 为空，也不产生事件。Router 虽然会订阅 decode 的各 rank，视图却始终为空。

### 第 3 层：SGLang 在参数解析阶段直接拒绝

如果手动传入该开关，SGLang 在 PD 参数钩子中直接抛出 `ValueError`。这一步发生在权重加载之前，所以失败很快：

```77:106:(image) /sgl-workspace/sglang/python/sglang/srt/arg_groups/pd_disaggregation_hook.py
    if cfg.disaggregation_mode == "decode":
        if cfg.disaggregation_decode_enable_radix_cache:
            if cfg.enable_hisparse:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with --enable-hisparse"
                )
            if cfg.disaggregation_transfer_backend == "fake":
                raise ValueError(...)
            if cfg.speculative_algorithm is not None:
                raise ValueError(
                    "--disaggregation-decode-enable-radix-cache is incompatible "
                    "with speculative decoding "
                    f"(--speculative-algorithm {cfg.speculative_algorithm})"
                )
            if resolved_view(server_args).enable_dp_attention:
                logger.warning(
                    "EXPERIMENTAL: Decode radix cache with DP attention. "
                    "Requires prefix-aware DP rank routing for optimal cache hits."
                )
            declare_resolution(..., disable_radix_cache=False)
        else:
            declare_resolution(..., disable_radix_cache=True)
            logger.warning("KV cache is forced as chunk cache for decode server")
```

没有开关时，radix cache 被强制关闭，树缓存工厂返回 `ChunkCache`（`mem_cache/registry.py` 第 91-95 行）。

`git log -S` 显示这条投机解码检查来自 `5b7ce417d0`（#19746，2026-05-01），此后只有文件搬迁，逻辑没有变化。同一 PR 后来陆续放开了 mooncake 后端（#26227）和 SWA 模型（#27770）限制，唯独投机解码一直保留。

### 第 4 层：Router 默认认为 decode 不参与 KV-aware

Rust Router 对没有 `kv_block_size` 的 decode worker 只打 debug 日志，认为这是正常情况：

```257:271:rust/router/src/kv_event.rs
    // Severity depends on the role, not on the symptom. A PD decode leg is
    // SUPPOSED to arrive without a block size: it runs --no-enable-kv-events, so
    // the engine leaves kv_block_size unset by design, and kv-aware routing never
    // applies to the decode pool anyway (prefix affinity is decided on the
    // prefill side; disagg dispatch picks decode by load). ...
    if w.disagg_mode == DisaggMode::Decode {
        tracing::debug!(
```

此外，镜像中的 Router 带有 `--pd-dp-rank-affinity`（来自 `yihou/glm52-1p1d-samerail-c32-c40.packup_20260918/patches/01-router-pd-dp-rank-affinity.patch`，本仓库 `rust/router/src` 中没有这部分代码）。测试脚本默认开启它（`PD_DP_RANK_AFFINITY=1`），decode 固定使用 prefill 选中的 DP rank。当前拓扑是 1 个 prefill worker 和 1 个 decode worker（各 DP8），因此即使 decode 有 KV 视图，Router 在 decode 上也只剩一个候选，没有可选择的空间。

这一层不影响“能否开启”，但会影响开启后收益来自哪里，见 [SOLUTION.zh-CN.md](SOLUTION.zh-CN.md)。

## 复现

在 crsuse2-m2m-138 上，使用同一镜像、`--network none`、不挂 GPU 设备，运行 Infera 的参数解析，再单独调用 SGLang 的 `handle_pd_disaggregation`。脚本为 [scripts/repro_decode_args.py](../scripts/repro_decode_args.py) 和 [scripts/run_repro.sh](../scripts/run_repro.sh)。原始输出在 [evidence/](evidence/)。

| 情况 | 转给 SGLang 的开关 | SGLang PD 钩子 | decode 缓存 |
|---|---|---|---|
| current：MTP + `--enable-kv-events` | 未追加 | 通过，并打印 `KV cache is forced as chunk cache for decode server` | ChunkCache |
| forced：再显式加 `--disaggregation-decode-enable-radix-cache` | 追加 | `ValueError: ... incompatible with speculative decoding (--speculative-algorithm EAGLE)` | 启动失败 |
| patched：应用补丁 01、02，并设置 `SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC=1` | 追加 | 通过，打印三条 EXPERIMENTAL（spec、DP attention、decode radix） | radix（参数层面） |
| patched，但未设置环境变量 | 未追加 | 与 current 相同 | ChunkCache |

完整参数解析（`resolve_once()`）在没有加速器的容器里会报 `No accelerator ... is available`，所以只单独执行了 PD 钩子。这正是决定 ChunkCache 或 radix 的那一步。

## 其他发现

1. **Infera 的模型兼容性检查在 SGLang 0.5.19 上已失效。** `_decode_radix_cache_unsupported_reason` 调用 `server_args.get_model_config()`，而该方法在此版本中已不存在（`AttributeError`）。异常被捕获后按“无法判断、照常追加”处理，所以对 SWA/SSM 模型的提前拦截实际上不起作用。GLM-5.2 是 DSA 模型，不属于 SWA/SSM，本来就应该追加，因此不影响本任务。
2. **Infera 注释中关于解析时机的描述已过时。** `no_clear_event_reason` 的注释说 `pd_disaggregation_hook` 在 `from_cli_args` 后就已把 decode 的 `disable_radix_cache` 设为 True。在 0.5.19 中构造出的记录是原始值，钩子要到 `resolve_once()` 才运行，复现中该字段读到的是 False。Infera 读取转发 argv 的做法仍然正确，只是注释与此版本不符。

   在未打补丁的镜像中运行 `tests/engine/sglang/test_decode_radix_cache_guard.py`，结果为 18 通过、2 失败。失败的正是上述两项：`test_the_guard_still_speaks_sglangs_own_api`（第 1 点）和 `test_a_decode_leg_is_read_off_the_argv_not_the_resolved_attribute`（第 2 点）。输出见 [evidence/infera-guard-tests-stock.txt](evidence/infera-guard-tests-stock.txt)。两者都早于本任务的改动。
3. **当前部署中 prefill 没有 draft 模型。** prefill 的 `speculative_algorithm=None`。decode 的 EAGLE 起步状态（hidden states、top-k、DSA seed）由 prefill 写入哨兵值，decode 在 draft extend 时自行生成。yihou 的记录中，这种配置下真实接受长度约 2.9。这意味着 prompt 位置的 draft KV 本来就不是从 prefill 传来的，radix 复用不会改变这一点（见 SOLUTION 第 2 节）。
