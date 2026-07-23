# Notes — 坑、弯路、纠错（what / why / how / context）

## 1. `--no-enable-kv-events` 在裸 sglang 0.5.15 是非法旗标

- **what**：初版扫描脚本给 decode 腿加了 `--no-enable-kv-events`，sglang 报
  `error: unrecognized arguments: --no-enable-kv-events`，进程秒退。
- **why**：`--no-enable-kv-events` 是 **infera wrapper** 的旗标（阻止 infera 自动追加
  `--disaggregation-decode-enable-radix-cache`，那个对 SWA 模型非法）。裸 sglang（`python -m
  sglang.launch_server`）没有这个旗标，且裸 decode 单腿本就不会自动追加 radix-cache。
- **how**：去掉它。裸 sglang 用 `--kv-events-config`（默认关），无 SWA 冲突。
- **context**：本研究直接调 `sglang.launch_server`（不经 infera wrapper），所以凡是 infera-only
  旗标都要剔除。REPRODUCE 里的脚本已不含。

## 2. sglang 0.5.15 的 DSv4 KV pool 是多池，无单一 `max_total_num_tokens`

- **what**：参考运行（0.5.13）日志有 `max_total_num_tokens=3895296`；0.5.15 grep 不到这行。
- **why**：0.5.15 对 DSv4 换成 `DeepSeekV4TokenToKVPool` 多池结构（swa/c4/c128 + state），
  分池打印 `DSV4 pool sizes: full=... swa=... c4=... c128=...`，不再有单一 max_total。
- **how**：改抓三行 —— `DSV4 memory calculation`（含 available_bytes/bytes_per_full_token/full_token）、
  `DSV4 pool sizes`（各池 token 数）、`Initialize DeepSeekV4TokenToKVPool`（池构造入参）。
- **context**：老 `max_total` ≈ 新 `c4` 口径不同，勿直接比。要"总容量"用 `full_token`。

## 3. 绝对数字 0.5.15 ≠ 0.5.13，必须实测勿套旧数

- **what**：同 gmu=0.90，参考运行 avail mem≈34.6GB，本轮 44.0GB，差 ~10GB。
- **why**：sglang 版本（KV 布局、activation 预留）+ 权重加载差异。
- **how**：每次实测，别拿旧报告的绝对数字当当前值。线性**关系**稳定，绝对数字随版本漂。

## 4. ⚠️ 真实 malloc 路径 = unified，不是三独立池（对第一版报告的修正）

- **what**：第一版报告写 KV buffer 是 584B/token/uint8 布局（`DeepSeekV4SingleKVPool`）。追容器源码
  发现 MI355X 实际不走那条。
- **why**：env `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton` + ROCm →
  `is_unified_kv_triton()=True`（`env_gate.py:10`）→ 走 `DeepSeekV4UnifiedKVPool`，每层是一块
  **bf16 `[swa_pages+compress_pages, 512]`** 统一 buffer（`deepseek_v4_memory_pool.py:430`），
  SWA 环 + 压缩区 row-partition 拼在一起。584B/uint8 是**非 unified（CUDA）**路径才走。
- **how**：CALLCHAIN.md 已按 unified 真实路径修正。两路 **token 容量规划相同**（同 configurator），
  仅**物理排布**不同 → gmu→pool 线性主结论不受影响。
- **context**：教训——引用调用链要落到**运行时实际命中的分支**，不能只看默认路径。env 决定分支。

## 5. `docker exec -d` 保后台进程；`nohup &` 在双 ssh 下会被 SIGHUP 杀

- **what**：`ssh chi2879 'docker exec ... nohup bash x.sh &'` 起的扫描进程，docker exec 返回时进程没了。
- **why**：双层 ssh + docker exec 的 `&` 后台化在父命令返回时收到 SIGHUP。
- **how**：用 `docker exec -d dsv4_pd_sgl bash /path/wrapper.sh`（-d = detach，最稳），wrapper
  是落盘的独立脚本（无内联引号，避免嵌套引号解析炸）。

## 6. 扫描脚本读数时机：DSV4 行在 "Memory pool end" 之前打印

- **what**：脚本等 `grep "Memory pool end"` 后 sleep 3 抓数即可——三条 DSV4/pool 行时间戳都在
  Memory pool end 同秒或之前，不会漏。CUDA-graph capture 在其后，可安全 kill（池已定）。
- **why**：KV pool 在 `init_memory_pool` 里一次性建好并打日志，之后才 capture graph。
- **context**：所以"纯读数"不必等 server ready（省 ~4min/点 cuda-graph 时间）。Step 3 真跑才需 ready。

## 7. 换 gmu 点之间必须等 VRAM 回落

- **what**：不等上个 server 的 VRAM 释放就起下一个 → 叠加 OOM。
- **how**：kill 后轮询 `rocm-smi --showmeminfo vram` 到 MX<5GB 再起下一点。脚本已内置。
- **context**：MI355X 288GiB/卡，gmu0.90 一个 server 吃 ~245GB，残留没清就起第二个必炸。

## 8. `bytes_per_full_token=13735.04` 是复合值，不是单池 per-token

- **what**：13735 不是某一池每 token 字节，是"1 个 full-context token"摊到 swa+c4+c128+state 各层的**加权总成本**。
- **why**：`_get_bytes_per_full_token`（pool_configurator.py:609）= swa_ratio×584×L_total + (1/4)×584×L_ca4
  + (1/128)×584×L_ca128 + indexer + state 项。`full_token = available_bytes / 13735`。
- **context**：所以"每 token KV"要看是问哪个池；单池 MLA cell 是 584 B/token/层，复合是 13735 B/full-token。
