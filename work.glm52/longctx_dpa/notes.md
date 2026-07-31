# GLM-5.2-MXFP4 单机 DPA 长文本（65K input）正确性测试 — chi2867

日期 2026-07-29。目标：DPA 打开的情况下，验证 ~65K token 输入的输出正确性。

## 配置

- 节点 chi2867（8× MI355X gfx950），镜像 `rocm/infera:sglang-v0.1.0-rc6`（现拉，节点原本没有）。
- 单机 colocated（非 PD），`launch_dpa_longctx.sh`：
  - DPA: `--dp-size 8 --enable-dp-attention --ep-size 8` + `SGLANG_DP_USE_GATHERV=1`（exp07 配方）
  - GLM DSA 基础配方：`SGLANG_OPT_USE_TILELANG_INDEXER=1 SGLANG_OPT_USE_TOPK_V2=0
    SGLANG_OPT_USE_JIT_NORM=0 SGLANG_USE_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0`
    + `--nsa-prefill/decode-backend tilelang --kv-cache-dtype fp8_e4m3`
  - `--context-length 131072 --chunked-prefill-size 16384 --mem-fraction-static 0.85`
- 启动耗时 ~4min，`max_total_num_tokens=3099136`（每 DP rank KV 159.4GB），ready 09:09:45。
  注意 sglang 把 chunked_prefill_size 自己降到 2048（DPA 下按 rank 分片后的实际值），
  max_running_requests 被降到 8 —— 长 ctx=131072 下的自动容量约束。

## 结果 — PASS

needle-in-a-haystack，filler 为编号维护日志行，temp=0，prompt_tokens 由 server usage 实测。

| case | 埋点深度 | prompt_tokens | 延迟 | 结果 |
|---|---|---|---|---|
| 密码短语 | 5% | 64974 | 13.2s | ✅ crimson-lantern-4417 |
| 校准常数 | 50% | 64971 | 13.0s | ✅ 82931 |
| 建筑名 | 95% | 64975 | 13.2s | ✅ Kestrel-Nine |
| 三针同时 | 5/50/95% | 65037 | 19.1s | ✅ 3/3（max_tokens=1024）|

单针 3/3，多针 3/3。65K 输入 ~13s 出结果，输出连贯、无乱码、无重复。

坑：GLM-5.2 默认走 thinking 模式，`max_tokens=160` 时三针 case 的推理过程就把预算吃完，
表面看只命中 2/3 —— 那是截断不是错误。长上下文检索题要给 >=512 的 max_tokens。

## 文件

- `launch_dpa_longctx.sh` — 启动脚本（DPA=1 可关）
- `longctx_probe.py` — needle 探针（会先做 2 轮 line-count 校准命中目标 token 数 ±2%）
- `longctx_65k.json` — 原始结果
