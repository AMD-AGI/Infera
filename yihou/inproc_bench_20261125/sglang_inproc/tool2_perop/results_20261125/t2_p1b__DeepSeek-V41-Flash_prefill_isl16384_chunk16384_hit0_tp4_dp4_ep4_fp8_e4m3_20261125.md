# DeepSeek-V41-Flash prefill profile

2026-11-25 14:52 UTC

## Summary

| metric       | value         | note                                                                  |
|--------------|---------------|-----------------------------------------------------------------------|
| TTFT         | 491.3 ms      | 4 chunks, one request per rank                                        |
| roll-up      | 488.7 ms      | sum of the per-op leaf rows below                                     |
| ratio        | 0.995         | roll-up / synthetic TTFT; agreement is not proof of complete coverage |
| computed     | 16,384 tokens | of 16,384; the other 0 came from cache                                |
| prompt rate  | 33,350 tok/s  | the whole prompt over TTFT, per request                               |
| compute rate | 33,350 tok/s  | only the tokens forwarded; the same number at a 0% hit                |

## Config

| item                | value                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model               | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                |
| layers              | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36) |
| parallel            | TP 4, attn-DP 4, attn-TP 1, MoE-EP 4                                                                                                                                                                                                                                |
| context             | ISL 16,384 per request                                                                                                                                                                                                                                              |
| prefix cache        | 0% hit, rounded down to the 256-token page                                                                                                                                                                                                                          |
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
| kv_pools[1]                            | 1       | 585          | 585   | 17.30  |
| kv_pools[2]                            | 3       | 293          | 878   | 25.96  |
| index_pools[1]                         | 2       | 34           | 68    | 2.01   |
| index_pools[2]                         | 6       | 17           | 102   | 3.02   |
| **kv total**                           |         |              | 5152  | 152.38 |
| weights                                |         |              |       | 83.25  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 127,042,560 full-KV tokens cluster-wide** = 31,760,640 per rank x 4 DP ranks = 7,754 requests at ISL 16384
- native pools: full KV 31,760,640, SWA 4,776,192, request slots 64; the B/tok columns divide every buffer by the full-KV token capacity (SWA holds its own, smaller count)
- `--mem-fraction-static 0.85` caps weights + kv at 245 GB of the 288 GB HBM; this run lands at 236 GB
- this run uses 32,768 context tokens, 0.1% of this rank's full-KV pool
- EP 4: this rank holds 96 routed experts at full intermediate width

## Chunk timeline

| chunk | prefix | new   | tail rows | ms     | roll-up | ratio |
|-------|--------|-------|-----------|--------|---------|-------|
| 1     | 0      | 4,096 | 128       | 121.86 | 120.88  | 0.992 |
| 2     | 4,096  | 4,096 | 128       | 122.90 | 122.39  | 0.996 |
| 3     | 8,192  | 4,096 | 128       | 123.15 | 122.63  | 0.996 |
| 4     | 12,288 | 4,096 | 128       | 123.36 | 122.77  | 0.995 |

`prefix` is the synthetic context already in KV. `tail rows` is how many rows the late layers ran on (DSv4.1 decoder SWA bounded replay); `-` means the tail was off. `ratio` compares graph-timed operators with an eager full forward, whose inter-kernel launch costs remain.

## Hotspots -- one TTFT, all layers

| op                                                                                     | 0     | 4,096 | 8,192 | 12,288 | total  | share |
|----------------------------------------------------------------------------------------|-------|-------|-------|--------|--------|-------|
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 50.74 | 50.90 | 50.93 | 50.58  | 203.15 | 41.6% |
| `mlp.experts`                                                                          | 30.64 | 30.76 | 30.75 | 30.71  | 122.86 | 25.1% |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`               | 22.22 | 23.36 | 23.58 | 23.63  | 92.79  | 19.0% |
| `attn.wo_b`                                                                            | 5.06  | 4.98  | 5.03  | 5.06   | 20.12  | 4.1%  |
| `mlp.shared_experts`                                                                   | 5.07  | 4.99  | 4.93  | 4.99   | 19.98  | 4.1%  |
| `attn.wq_b`                                                                            | 4.71  | 4.75  | 4.56  | 4.64   | 18.65  | 3.8%  |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                       | 2.79  | 2.79  | 2.78  | 2.80   | 11.16  | 2.3%  |
| `mlp.gate`                                                                             | 1.73  | 1.78  | 1.78  | 1.92   | 7.20   | 1.5%  |
| `attn.wqkv_a`                                                                          | 1.29  | 1.33  | 1.27  | 1.29   | 5.18   | 1.1%  |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)`   | 1.02  | 1.16  | 1.30  | 1.45   | 4.93   | 1.0%  |
| `mlp.topk`                                                                             | 0.75  | 0.72  | 0.74  | 0.63   | 2.84   | 0.6%  |
| `logits`                                                                               | 0.21  | 0.21  | 0.21  | 0.21   | 0.83   | 0.2%  |
| `engram_hasher`                                                                        | 0.04  | 0.04  | 0.05  | 0.05   | 0.18   | 0.0%  |
| `embed_tokens`                                                                         | 0.02  | 0.02  | 0.02  | 0.02   | 0.08   | 0.0%  |
| `hc_head :: final mHC combine`                                                         | 0.01  | 0.01  | 0.01  | 0.01   | 0.03   | 0.0%  |
| `norm`                                                                                 | 0.01  | 0.01  | 0.01  | 0.00   | 0.02   | 0.0%  |
| `mlp.<rest> :: shared+routed add, post-expert collective`                              | -5.44 | -5.40 | -5.31 | -5.21  | -21.36 | -4.4% |

ms per chunk, columns labelled by the prefix each chunk starts on. Leaf rows only, so a column sums to that chunk's roll-up and `share` adds to 100%. A zero is an op that never ran in that chunk. Rows that climb across the table are reading the context; rows that stay flat only ever see the chunk. Attention is expected among the flat ones past the indexer's top 512: beyond that it scores a fixed budget per query however deep the context goes, leaving the indexer's own scan to grow.

## Per layer kind -- deepest chunk, prefix 12,288

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 4.738   | 4.738    |
| `  attn`                                                                                 | 1.080   | 1.080    |
| `    attn.wqkv_a`                                                                        | 0.045   | 0.045    |
| `    attn.wq_b`                                                                          | 0.208   | 0.208    |
| `    attn.wo_b`                                                                          | 0.227   | 0.227    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.600   | 0.600    |
| `  mlp`                                                                                  | 1.321   | 1.321    |
| `    mlp.gate`                                                                           | 0.074   | 0.074    |
| `    mlp.topk`                                                                           | 0.020   | 0.020    |
| `    mlp.experts`                                                                        | 1.220   | 1.220    |
| `    mlp.shared_experts`                                                                 | 0.219   | 0.219    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.211  | -0.211   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.337   | 2.337    |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 4.722   | 4.722    |
| `  attn`                                                                                 | 1.076   | 1.076    |
| `    attn.wqkv_a`                                                                        | 0.046   | 0.046    |
| `    attn.wq_b`                                                                          | 0.208   | 0.208    |
| `    attn.wo_b`                                                                          | 0.221   | 0.221    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.601   | 0.601    |
| `  mlp`                                                                                  | 1.320   | 1.320    |
| `    mlp.gate`                                                                           | 0.072   | 0.072    |
| `    mlp.topk`                                                                           | 0.020   | 0.020    |
| `    mlp.experts`                                                                        | 1.227   | 1.227    |
| `    mlp.shared_experts`                                                                 | 0.216   | 0.216    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.216  | -0.216   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.327   | 2.327    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 1.399   | 1.399    |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.800   | 11.600   |
| `  attn`                                                                                 | 2.131   | 4.261    |
| `    attn.wqkv_a`                                                                        | 0.042   | 0.084    |
| `    attn.wq_b`                                                                          | 0.208   | 0.415    |
| `    attn.wo_b`                                                                          | 0.208   | 0.416    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.383   | 0.766    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.290   | 2.580    |
| `  mlp`                                                                                  | 1.326   | 2.652    |
| `    mlp.gate`                                                                           | 0.072   | 0.145    |
| `    mlp.topk`                                                                           | 0.024   | 0.048    |
| `    mlp.experts`                                                                        | 1.232   | 2.465    |
| `    mlp.shared_experts`                                                                 | 0.222   | 0.445    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.225  | -0.450   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.343   | 4.686    |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.327   | 79.905   |
| `  attn`                                                                                 | 1.660   | 24.898   |
| `    attn.wqkv_a`                                                                        | 0.044   | 0.655    |
| `    attn.wq_b`                                                                          | 0.205   | 3.080    |
| `    attn.wo_b`                                                                          | 0.224   | 3.356    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.187   | 17.807   |
| `  mlp`                                                                                  | 1.329   | 19.940   |
| `    mlp.gate`                                                                           | 0.074   | 1.117    |
| `    mlp.topk`                                                                           | 0.021   | 0.314    |
| `    mlp.experts`                                                                        | 1.227   | 18.405   |
| `    mlp.shared_experts`                                                                 | 0.207   | 3.109    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.200  | -3.006   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.338   | 35.067   |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 5.747   | 5.747    |
| `  attn`                                                                                 | 2.137   | 2.137    |
| `    attn.wqkv_a`                                                                        | 0.045   | 0.045    |
| `    attn.wq_b`                                                                          | 0.218   | 0.218    |
| `    attn.wo_b`                                                                          | 0.215   | 0.215    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.384   | 0.384    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 1.275   | 1.275    |
| `  mlp`                                                                                  | 1.325   | 1.325    |
| `    mlp.gate`                                                                           | 0.071   | 0.071    |
| `    mlp.topk`                                                                           | 0.024   | 0.024    |
| `    mlp.experts`                                                                        | 1.230   | 1.230    |
| `    mlp.shared_experts`                                                                 | 0.216   | 0.216    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.216  | -0.216   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 2.285   | 2.285    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 1.400   | 1.400    |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.697   | 0.697    |
| `  attn`                                                                                 | 0.191   | 0.191    |
| `    attn.wqkv_a`                                                                        | 0.026   | 0.026    |
| `    attn.wq_b`                                                                          | 0.018   | 0.018    |
| `    attn.wo_b`                                                                          | 0.033   | 0.033    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.074   | 0.074    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.039   | 0.039    |
| `  mlp`                                                                                  | 0.322   | 0.322    |
| `    mlp.gate`                                                                           | 0.020   | 0.020    |
| `    mlp.topk`                                                                           | 0.011   | 0.011    |
| `    mlp.experts`                                                                        | 0.314   | 0.314    |
| `    mlp.shared_experts`                                                                 | 0.035   | 0.035    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.058  | -0.058   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.184   | 0.184    |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.635   | 9.528    |
| `  attn`                                                                                 | 0.115   | 1.729    |
| `    attn.wqkv_a`                                                                        | 0.020   | 0.303    |
| `    attn.wq_b`                                                                          | 0.027   | 0.412    |
| `    attn.wo_b`                                                                          | 0.031   | 0.472    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.036   | 0.543    |
| `  mlp`                                                                                  | 0.322   | 4.834    |
| `    mlp.gate`                                                                           | 0.022   | 0.330    |
| `    mlp.topk`                                                                           | 0.010   | 0.151    |
| `    mlp.experts`                                                                        | 0.305   | 4.572    |
| `    mlp.shared_experts`                                                                 | 0.039   | 0.578    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.053  | -0.797   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.198   | 2.964    |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.687   | 2.747    |
| `  attn`                                                                                 | 0.174   | 0.694    |
| `    attn.wqkv_a`                                                                        | 0.021   | 0.083    |
| `    attn.wq_b`                                                                          | 0.019   | 0.076    |
| `    attn.wo_b`                                                                          | 0.031   | 0.124    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.056   | 0.225    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.047   | 0.187    |
| `  mlp`                                                                                  | 0.331   | 1.323    |
| `    mlp.gate`                                                                           | 0.023   | 0.092    |
| `    mlp.topk`                                                                           | 0.010   | 0.041    |
| `    mlp.experts`                                                                        | 0.320   | 1.280    |
| `    mlp.shared_experts`                                                                 | 0.042   | 0.169    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.065  | -0.258   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.183   | 0.730    |

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
