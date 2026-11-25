# DeepSeek-V41-Flash prefill profile

2026-11-25 15:03 UTC

## Summary

| metric       | value        | note                                                                  |
|--------------|--------------|-----------------------------------------------------------------------|
| TTFT         | 246.6 ms     | 2 chunks, one request per rank                                        |
| roll-up      | 244.9 ms     | sum of the per-op leaf rows below                                     |
| ratio        | 0.993        | roll-up / synthetic TTFT; agreement is not proof of complete coverage |
| computed     | 8,192 tokens | of 16,384; the other 8,192 came from cache                            |
| prompt rate  | 66,446 tok/s | the whole prompt over TTFT, per request                               |
| compute rate | 33,223 tok/s | only the tokens forwarded; the same number at a 0% hit                |

## Config

| item                | value                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model               | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                |
| layers              | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36) |
| parallel            | TP 4, attn-DP 4, attn-TP 1, MoE-EP 4                                                                                                                                                                                                                                |
| context             | ISL 16,384 per request                                                                                                                                                                                                                                              |
| prefix cache        | 50% hit, rounded down to the 256-token page                                                                                                                                                                                                                         |
| chunk               | --chunked-prefill-size 16,384 / 4 DP ranks = 4,096 new tokens per forward                                                                                                                                                                                           |
| batch               | 1 request per attention rank, 4 in total                                                                                                                                                                                                                            |
| indexer             | top 512 per query (C4 / DSv4.1 index-source layers)                                                                                                                                                                                                                 |
| late-layer SWA tail | layers >= 20 run on the tail rows                                                                                                                                                                                                                                   |
| routing             | seeded balanced (seed 1234) on every MoE layer                                                                                                                                                                                                                      |
| kv dtype            | `fp8_e4m3`                                                                                                                                                                                                                                                          |
| weights             | `auto` load, fp8 block 32x32 quantized, experts fp4, bfloat16 elsewhere                                                                                                                                                                                             |
| measurement         | 3 graph replays per op; the full forward eager, the way the server runs a chunk, max over ranks (EP)                                                                                                                                                                |

## Versions

| component | version                                         |
|-----------|-------------------------------------------------|
| bench     | no git files-sha256 04e14c4e56d3                |
| sglang    | 0.5.23.dev20261111+g6be0d75ae0 6be0d75ae0+dirty |
| aiter     | 285579b62+dirty                                 |
| torch     | 2.11.0+rocm10.0.0                               |
| rocm      | 7.15.26333                                      |
| triton    | 3.8.0                                           |
| flydsl    | 0.4.0                                           |
| tilelang  | 0.1.7.post3+cuda.gita55a8230                    |
| aklr      | absent                                          |

## Memory, per rank

| component                              | buffers | B/tok/buffer | B/tok | GB     |
|----------------------------------------|---------|--------------|-------|--------|
| swa_kv_pool                            | 40      | 88           | 3519  | 104.09 |
| kv_pools[1]                            | 1       | 585          | 585   | 17.31  |
| kv_pools[2]                            | 3       | 293          | 878   | 25.96  |
| index_pools[1]                         | 2       | 34           | 68    | 2.01   |
| index_pools[2]                         | 6       | 17           | 102   | 3.02   |
| **kv total**                           |         |              | 5151  | 152.39 |
| weights                                |         |              |       | 83.25  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 127,051,776 full-KV tokens cluster-wide** = 31,762,944 per rank x 4 DP ranks = 7,754 requests at ISL 16384
- native pools: full KV 31,762,944, SWA 4,776,192, request slots 64; the B/tok columns divide every buffer by the full-KV token capacity (SWA holds its own, smaller count)
- `--mem-fraction-static 0.85` caps weights + kv at 245 GB of the 288 GB HBM; this run lands at 236 GB
- this run uses 32,768 context tokens, 0.1% of this rank's full-KV pool
- EP 4: this rank holds 96 routed experts at full intermediate width

## Chunk timeline

| chunk | prefix | new   | tail rows | ms     | roll-up | ratio |
|-------|--------|-------|-----------|--------|---------|-------|
| 1     | 8,192  | 4,096 | 128       | 123.02 | 122.40  | 0.995 |
| 2     | 12,288 | 4,096 | 128       | 123.56 | 122.54  | 0.992 |

`prefix` is the synthetic context already in KV. `tail rows` is how many rows the late layers ran on (DSv4.1 decoder SWA bounded replay); `-` means the tail was off. `ratio` compares graph-timed operators with an eager full forward, whose inter-kernel launch costs remain.

## Hotspots -- one TTFT, all layers

| op                                                                                     | 8,192 | 12,288 | total  | share |
|----------------------------------------------------------------------------------------|-------|--------|--------|-------|
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 50.82 | 50.59  | 101.41 | 41.4% |
| `mlp.experts`                                                                          | 30.60 | 30.54  | 61.14  | 25.0% |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`               | 23.90 | 23.65  | 47.55  | 19.4% |
| `mlp.shared_experts`                                                                   | 4.86  | 5.12   | 9.98   | 4.1%  |
| `attn.wo_b`                                                                            | 4.87  | 4.91   | 9.78   | 4.0%  |
| `attn.wq_b`                                                                            | 4.46  | 4.73   | 9.19   | 3.8%  |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                       | 2.80  | 2.79   | 5.59   | 2.3%  |
| `mlp.gate`                                                                             | 1.71  | 1.81   | 3.52   | 1.4%  |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)`   | 1.27  | 1.45   | 2.72   | 1.1%  |
| `attn.wqkv_a`                                                                          | 1.16  | 1.19   | 2.36   | 1.0%  |
| `mlp.topk`                                                                             | 0.60  | 0.62   | 1.22   | 0.5%  |
| `logits`                                                                               | 0.21  | 0.21   | 0.42   | 0.2%  |
| `engram_hasher`                                                                        | 0.04  | 0.04   | 0.09   | 0.0%  |
| `embed_tokens`                                                                         | 0.02  | 0.02   | 0.03   | 0.0%  |
| `hc_head :: final mHC combine`                                                         | 0.00  | 0.01   | 0.01   | 0.0%  |
| `norm`                                                                                 | 0.00  | 0.00   | 0.01   | 0.0%  |
| `mlp.<rest> :: shared+routed add, post-expert collective`                              | -4.92 | -5.14  | -10.07 | -4.1% |

ms per chunk, columns labelled by the prefix each chunk starts on. Leaf rows only, so a column sums to that chunk's roll-up and `share` adds to 100%. A zero is an op that never ran in that chunk. Rows that climb across the table are reading the context; rows that stay flat only ever see the chunk. Attention is expected among the flat ones past the indexer's top 512: beyond that it scores a fixed budget per query however deep the context goes, leaving the indexer's own scan to grow.

## Per layer kind -- deepest chunk, prefix 12,288

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 4.751   | 4.751    |
| `  attn`                                                                                 | 1.081   | 1.081    |
| `    attn.wqkv_a`                                                                        | 0.047   | 0.047    |
| `    attn.wq_b`                                                                          | 0.211   | 0.211    |
| `    attn.wo_b`                                                                          | 0.234   | 0.234    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.589   | 0.589    |
| `  mlp`                                                                                  | 1.388   | 1.388    |
| `    mlp.gate`                                                                           | 0.075   | 0.075    |
| `    mlp.topk`                                                                           | 0.021   | 0.021    |
| `    mlp.experts`                                                                        | 1.229   | 1.229    |
| `    mlp.shared_experts`                                                                 | 0.213   | 0.213    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.148  | -0.148   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.281   | 2.281    |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 4.719   | 4.719    |
| `  attn`                                                                                 | 1.068   | 1.068    |
| `    attn.wqkv_a`                                                                        | 0.046   | 0.046    |
| `    attn.wq_b`                                                                          | 0.212   | 0.212    |
| `    attn.wo_b`                                                                          | 0.219   | 0.219    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.591   | 0.591    |
| `  mlp`                                                                                  | 1.323   | 1.323    |
| `    mlp.gate`                                                                           | 0.069   | 0.069    |
| `    mlp.topk`                                                                           | 0.024   | 0.024    |
| `    mlp.experts`                                                                        | 1.241   | 1.241    |
| `    mlp.shared_experts`                                                                 | 0.215   | 0.215    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.227  | -0.227   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.328   | 2.328    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 1.396   | 1.396    |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.800   | 11.599   |
| `  attn`                                                                                 | 2.128   | 4.256    |
| `    attn.wqkv_a`                                                                        | 0.044   | 0.089    |
| `    attn.wq_b`                                                                          | 0.210   | 0.420    |
| `    attn.wo_b`                                                                          | 0.219   | 0.439    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.385   | 0.769    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.270   | 2.539    |
| `  mlp`                                                                                  | 1.309   | 2.618    |
| `    mlp.gate`                                                                           | 0.073   | 0.146    |
| `    mlp.topk`                                                                           | 0.024   | 0.048    |
| `    mlp.experts`                                                                        | 1.234   | 2.468    |
| `    mlp.shared_experts`                                                                 | 0.211   | 0.423    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.234  | -0.467   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.363   | 4.726    |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.328   | 79.923   |
| `  attn`                                                                                 | 1.649   | 24.739   |
| `    attn.wqkv_a`                                                                        | 0.042   | 0.635    |
| `    attn.wq_b`                                                                          | 0.216   | 3.246    |
| `    attn.wo_b`                                                                          | 0.216   | 3.244    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.174   | 17.615   |
| `  mlp`                                                                                  | 1.328   | 19.913   |
| `    mlp.gate`                                                                           | 0.072   | 1.083    |
| `    mlp.topk`                                                                           | 0.020   | 0.301    |
| `    mlp.experts`                                                                        | 1.224   | 18.360   |
| `    mlp.shared_experts`                                                                 | 0.213   | 3.190    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.201  | -3.021   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.351   | 35.270   |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.749   | 5.749    |
| `  attn`                                                                                 | 2.130   | 2.130    |
| `    attn.wqkv_a`                                                                        | 0.046   | 0.046    |
| `    attn.wq_b`                                                                          | 0.209   | 0.209    |
| `    attn.wo_b`                                                                          | 0.215   | 0.215    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.383   | 0.383    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.277   | 1.277    |
| `  mlp`                                                                                  | 1.322   | 1.322    |
| `    mlp.gate`                                                                           | 0.071   | 0.071    |
| `    mlp.topk`                                                                           | 0.022   | 0.022    |
| `    mlp.experts`                                                                        | 1.231   | 1.231    |
| `    mlp.shared_experts`                                                                 | 0.225   | 0.225    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.227  | -0.227   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.297   | 2.297    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 1.398   | 1.398    |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.695   | 0.695    |
| `  attn`                                                                                 | 0.210   | 0.210    |
| `    attn.wqkv_a`                                                                        | 0.015   | 0.015    |
| `    attn.wq_b`                                                                          | 0.026   | 0.026    |
| `    attn.wo_b`                                                                          | 0.032   | 0.032    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.078   | 0.078    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.060   | 0.060    |
| `  mlp`                                                                                  | 0.325   | 0.325    |
| `    mlp.gate`                                                                           | 0.019   | 0.019    |
| `    mlp.topk`                                                                           | 0.010   | 0.010    |
| `    mlp.experts`                                                                        | 0.299   | 0.299    |
| `    mlp.shared_experts`                                                                 | 0.036   | 0.036    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.039  | -0.039   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.160   | 0.160    |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.620   | 9.294    |
| `  attn`                                                                                 | 0.118   | 1.773    |
| `    attn.wqkv_a`                                                                        | 0.015   | 0.225    |
| `    attn.wq_b`                                                                          | 0.021   | 0.322    |
| `    attn.wo_b`                                                                          | 0.028   | 0.417    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.054   | 0.808    |
| `  mlp`                                                                                  | 0.318   | 4.765    |
| `    mlp.gate`                                                                           | 0.013   | 0.199    |
| `    mlp.topk`                                                                           | 0.010   | 0.151    |
| `    mlp.experts`                                                                        | 0.300   | 4.507    |
| `    mlp.shared_experts`                                                                 | 0.045   | 0.680    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.051  | -0.772   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.184   | 2.756    |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.683   | 2.734    |
| `  attn`                                                                                 | 0.169   | 0.677    |
| `    attn.wqkv_a`                                                                        | 0.023   | 0.091    |
| `    attn.wq_b`                                                                          | 0.022   | 0.088    |
| `    attn.wo_b`                                                                          | 0.028   | 0.114    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.054   | 0.217    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.042   | 0.167    |
| `  mlp`                                                                                  | 0.321   | 1.283    |
| `    mlp.gate`                                                                           | 0.036   | 0.144    |
| `    mlp.topk`                                                                           | 0.010   | 0.040    |
| `    mlp.experts`                                                                        | 0.301   | 1.205    |
| `    mlp.shared_experts`                                                                 | 0.034   | 0.136    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.061  | -0.243   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.194   | 0.774    |

## Coverage

| layer kind                  | hook not called          | possible containing row                            |
|-----------------------------|--------------------------|----------------------------------------------------|
| moe+c0+engram               | attn.wo_a                | attn.<rest>                                        |
| moe+c0+engram               | input_layernorm          | layer.<rest>                                       |
| moe+c0+engram               | post_attention_layernorm | layer.<rest>                                       |
| moe+c0                      | attn.wo_a                | attn.<rest>                                        |
| moe+c0                      | input_layernorm          | layer.<rest>                                       |
| moe+c0                      | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+indexer+late         | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c1+indexer+late         | attn.wo_a                | attn.<rest>                                        |
| moe+c1+indexer+late         | input_layernorm          | layer.<rest>                                       |
| moe+c1+indexer+late         | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+kvsrc+indexer+late   | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c1+kvsrc+indexer+late   | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c1+kvsrc+indexer+late   | attn.wo_a                | attn.<rest>                                        |
| moe+c1+kvsrc+indexer+late   | input_layernorm          | layer.<rest>                                       |
| moe+c1+kvsrc+indexer+late   | post_attention_layernorm | layer.<rest>                                       |
| moe+c1+late                 | attn.wo_a                | attn.<rest>                                        |
| moe+c1+late                 | input_layernorm          | layer.<rest>                                       |
| moe+c1+late                 | post_attention_layernorm | layer.<rest>                                       |
| moe+c2+kvsrc+indexer+engram | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer+engram | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer+engram | attn.wo_a                | attn.<rest>                                        |
| moe+c2+kvsrc+indexer+engram | input_layernorm          | layer.<rest>                                       |
| moe+c2+kvsrc+indexer+engram | post_attention_layernorm | layer.<rest>                                       |
| moe+c2+kvsrc+indexer        | attn.compressor          | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer        | attn.indexer             | attn.sources (timed through the backend call site) |
| moe+c2+kvsrc+indexer        | attn.wo_a                | attn.<rest>                                        |
| moe+c2+kvsrc+indexer        | input_layernorm          | layer.<rest>                                       |
| moe+c2+kvsrc+indexer        | post_attention_layernorm | layer.<rest>                                       |
| moe+c2                      | attn.wo_a                | attn.<rest>                                        |
| moe+c2                      | input_layernorm          | layer.<rest>                                       |
| moe+c2                      | post_attention_layernorm | layer.<rest>                                       |

Hooked but never called during the recording forward. A name is missing either because this layer kind does not run that op, or because a fused kernel reads the module's weight directly (DSv4.1 on gfx950: the mHC boundary kernel applies `input_layernorm` / `post_attention_layernorm` itself). A parent residual can contain this work; missing globals are not covered by layer residuals.
