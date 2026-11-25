# DeepSeek-V41-Flash prefill profile

2026-11-25 15:16 UTC

## Summary

| metric       | value         | note                                                                  |
|--------------|---------------|-----------------------------------------------------------------------|
| TTFT         | 874.6 ms      | 2 chunks, one request per rank                                        |
| roll-up      | 872.6 ms      | sum of the per-op leaf rows below                                     |
| ratio        | 0.998         | roll-up / synthetic TTFT; agreement is not proof of complete coverage |
| computed     | 32,768 tokens | of 32,768; the other 0 came from cache                                |
| prompt rate  | 37,465 tok/s  | the whole prompt over TTFT, per request                               |
| compute rate | 37,465 tok/s  | only the tokens forwarded; the same number at a 0% hit                |

## Config

| item                | value                                                                                                                                                                                                                                                               |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| model               | `DeepseekV4ForCausalLM`  /models/DeepSeek-V4.1-Flash                                                                                                                                                                                                                |
| layers              | 40 layers -- moe+c0 1 (0), moe+c0+engram 1 (1), moe+c2+kvsrc+indexer 2 (2,8), moe+c2 15 (3-7,9-13,15-19), moe+c2+kvsrc+indexer+engram 1 (14), moe+c1+kvsrc+indexer+late 1 (20), moe+c1+late 15 (21-23,25-27,29-31,33-35,37-39), moe+c1+indexer+late 4 (24,28,32,36) |
| parallel            | TP 4, attn-DP 4, attn-TP 1, MoE-EP 4                                                                                                                                                                                                                                |
| context             | ISL 32,768 per request                                                                                                                                                                                                                                              |
| prefix cache        | 0% hit, rounded down to the 256-token page                                                                                                                                                                                                                          |
| chunk               | --chunked-prefill-size 65,536 / 4 DP ranks = 16,384 new tokens per forward                                                                                                                                                                                          |
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
| swa_kv_pool                            | 40      | 708          | 28325 | 104.63 |
| kv_pools[1]                            | 1       | 585          | 585   | 2.16   |
| kv_pools[2]                            | 3       | 293          | 878   | 3.24   |
| index_pools[1]                         | 2       | 34           | 68    | 0.25   |
| index_pools[2]                         | 6       | 17           | 102   | 0.38   |
| **kv total**                           |         |              | 29958 | 110.66 |
| weights                                |         |              |       | 83.25  |
| **HBM**                                |         |              |       | 287.98 |
| host-resident params/buffers (not HBM) |         |              |       | 188.83 |

- **capacity 15,864,832 full-KV tokens cluster-wide** = 3,966,208 per rank x 4 DP ranks = 484 requests at ISL 32768
- native pools: full KV 3,966,208, SWA 4,800,768, request slots 64; the B/tok columns divide every buffer by the full-KV token capacity (SWA holds its own, smaller count)
- `--mem-fraction-static 0.7` caps weights + kv at 202 GB of the 288 GB HBM; this run lands at 194 GB
- this run uses 65,536 context tokens, 1.7% of this rank's full-KV pool
- EP 4: this rank holds 96 routed experts at full intermediate width

## Chunk timeline

| chunk | prefix | new    | tail rows | ms     | roll-up | ratio |
|-------|--------|--------|-----------|--------|---------|-------|
| 1     | 0      | 16,384 | 128       | 436.05 | 435.46  | 0.999 |
| 2     | 16,384 | 16,384 | 128       | 438.59 | 437.09  | 0.997 |

`prefix` is the synthetic context already in KV. `tail rows` is how many rows the late layers ran on (DSv4.1 decoder SWA bounded replay); `-` means the tail was off. `ratio` compares graph-timed operators with an eager full forward, whose inter-kernel launch costs remain.

## Hotspots -- one TTFT, all layers

| op                                                                                     | 0      | 16,384 | total  | share |
|----------------------------------------------------------------------------------------|--------|--------|--------|-------|
| `layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 186.02 | 185.14 | 371.16 | 42.5% |
| `mlp.experts`                                                                          | 106.86 | 106.63 | 213.49 | 24.5% |
| `attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`               | 82.06  | 84.55  | 166.61 | 19.1% |
| `attn.wq_b`                                                                            | 18.77  | 18.78  | 37.55  | 4.3%  |
| `attn.wo_b`                                                                            | 16.29  | 15.93  | 32.22  | 3.7%  |
| `mlp.shared_experts`                                                                   | 14.60  | 15.19  | 29.78  | 3.4%  |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                       | 10.96  | 10.93  | 21.89  | 2.5%  |
| `attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)`   | 4.16   | 5.78   | 9.95   | 1.1%  |
| `mlp.gate`                                                                             | 4.84   | 4.88   | 9.72   | 1.1%  |
| `attn.wqkv_a`                                                                          | 4.12   | 3.42   | 7.54   | 0.9%  |
| `mlp.topk`                                                                             | 2.58   | 1.89   | 4.48   | 0.5%  |
| `logits`                                                                               | 0.22   | 0.22   | 0.44   | 0.1%  |
| `engram_hasher`                                                                        | 0.08   | 0.07   | 0.15   | 0.0%  |
| `embed_tokens`                                                                         | 0.08   | 0.07   | 0.15   | 0.0%  |
| `hc_head :: final mHC combine`                                                         | 0.06   | 0.02   | 0.08   | 0.0%  |
| `norm`                                                                                 | 0.03   | 0.03   | 0.06   | 0.0%  |
| `mlp.<rest> :: shared+routed add, post-expert collective`                              | -16.28 | -16.43 | -32.72 | -3.7% |

ms per chunk, columns labelled by the prefix each chunk starts on. Leaf rows only, so a column sums to that chunk's roll-up and `share` adds to 100%. A zero is an op that never ran in that chunk. Rows that climb across the table are reading the context; rows that stay flat only ever see the chunk. Attention is expected among the flat ones past the indexer's top 512: beyond that it scores a fixed budget per query however deep the context goes, leaving the indexer's own scan to grow.

## Per layer kind -- deepest chunk, prefix 16,384

### moe+c0 -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 18.380  | 18.380   |
| `  attn`                                                                                 | 3.994   | 3.994    |
| `    attn.wqkv_a`                                                                        | 0.145   | 0.145    |
| `    attn.wq_b`                                                                          | 0.884   | 0.884    |
| `    attn.wo_b`                                                                          | 0.766   | 0.766    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 2.199   | 2.199    |
| `  mlp`                                                                                  | 5.274   | 5.274    |
| `    mlp.gate`                                                                           | 0.212   | 0.212    |
| `    mlp.topk`                                                                           | 0.074   | 0.074    |
| `    mlp.experts`                                                                        | 5.006   | 5.006    |
| `    mlp.shared_experts`                                                                 | 0.709   | 0.709    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.726  | -0.726   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 9.113   | 9.113    |

### moe+c0+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 18.216  | 18.216   |
| `  attn`                                                                                 | 4.000   | 4.000    |
| `    attn.wqkv_a`                                                                        | 0.141   | 0.141    |
| `    attn.wq_b`                                                                          | 0.914   | 0.914    |
| `    attn.wo_b`                                                                          | 0.785   | 0.785    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 2.160   | 2.160    |
| `  mlp`                                                                                  | 5.290   | 5.290    |
| `    mlp.gate`                                                                           | 0.225   | 0.225    |
| `    mlp.topk`                                                                           | 0.074   | 0.074    |
| `    mlp.experts`                                                                        | 4.983   | 4.983    |
| `    mlp.shared_experts`                                                                 | 0.726   | 0.726    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.718  | -0.718   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 8.926   | 8.926    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 5.470   | 5.470    |

### moe+c2+kvsrc+indexer -- x2 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 22.456  | 44.912   |
| `  attn`                                                                                 | 8.052   | 16.105   |
| `    attn.wqkv_a`                                                                        | 0.147   | 0.295    |
| `    attn.wq_b`                                                                          | 0.904   | 1.809    |
| `    attn.wo_b`                                                                          | 0.761   | 1.522    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 1.768   | 3.536    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 4.472   | 8.943    |
| `  mlp`                                                                                  | 5.262   | 10.524   |
| `    mlp.gate`                                                                           | 0.211   | 0.423    |
| `    mlp.topk`                                                                           | 0.073   | 0.147    |
| `    mlp.experts`                                                                        | 4.989   | 9.977    |
| `    mlp.shared_experts`                                                                 | 0.731   | 1.462    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.742  | -1.485   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 9.142   | 18.283   |

### moe+c2 -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 20.571  | 308.567  |
| `  attn`                                                                                 | 6.218   | 93.276   |
| `    attn.wqkv_a`                                                                        | 0.138   | 2.066    |
| `    attn.wq_b`                                                                          | 0.898   | 13.472   |
| `    attn.wo_b`                                                                          | 0.752   | 11.275   |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 4.431   | 66.463   |
| `  mlp`                                                                                  | 5.282   | 79.233   |
| `    mlp.gate`                                                                           | 0.222   | 3.335    |
| `    mlp.topk`                                                                           | 0.074   | 1.114    |
| `    mlp.experts`                                                                        | 5.007   | 75.112   |
| `    mlp.shared_experts`                                                                 | 0.700   | 10.498   |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.722  | -10.825  |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 9.070   | 136.057  |

### moe+c2+kvsrc+indexer+engram -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 22.404  | 22.404   |
| `  attn`                                                                                 | 8.150   | 8.150    |
| `    attn.wqkv_a`                                                                        | 0.139   | 0.139    |
| `    attn.wq_b`                                                                          | 0.882   | 0.882    |
| `    attn.wo_b`                                                                          | 0.746   | 0.746    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 1.749   | 1.749    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 4.633   | 4.633    |
| `  mlp`                                                                                  | 5.297   | 5.297    |
| `    mlp.gate`                                                                           | 0.218   | 0.218    |
| `    mlp.topk`                                                                           | 0.075   | 0.075    |
| `    mlp.experts`                                                                        | 4.957   | 4.957    |
| `    mlp.shared_experts`                                                                 | 0.700   | 0.700    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.653  | -0.653   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 8.957   | 8.957    |
| `engram :: Engram n-gram embedding gate (runs before the layer)`                         | 5.456   | 5.456    |

### moe+c1+kvsrc+indexer+late -- x1 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.735   | 0.735    |
| `  attn`                                                                                 | 0.220   | 0.220    |
| `    attn.wqkv_a`                                                                        | 0.037   | 0.037    |
| `    attn.wq_b`                                                                          | 0.036   | 0.036    |
| `    attn.wo_b`                                                                          | 0.068   | 0.068    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.164   | 0.164    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | -0.085  | -0.085   |
| `  mlp`                                                                                  | 0.323   | 0.323    |
| `    mlp.gate`                                                                           | 0.022   | 0.022    |
| `    mlp.topk`                                                                           | 0.017   | 0.017    |
| `    mlp.experts`                                                                        | 0.305   | 0.305    |
| `    mlp.shared_experts`                                                                 | 0.079   | 0.079    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.099  | -0.099   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.192   | 0.192    |

### moe+c1+late -- x15 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.646   | 9.688    |
| `  attn`                                                                                 | 0.127   | 1.899    |
| `    attn.wqkv_a`                                                                        | 0.025   | 0.380    |
| `    attn.wq_b`                                                                          | 0.041   | 0.612    |
| `    attn.wo_b`                                                                          | 0.034   | 0.517    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | 0.026   | 0.390    |
| `  mlp`                                                                                  | 0.327   | 4.907    |
| `    mlp.gate`                                                                           | 0.024   | 0.361    |
| `    mlp.topk`                                                                           | 0.019   | 0.290    |
| `    mlp.experts`                                                                        | 0.337   | 5.056    |
| `    mlp.shared_experts`                                                                 | 0.054   | 0.807    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.107  | -1.607   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.192   | 2.882    |

### moe+c1+indexer+late -- x4 layers

| op                                                                                       | ms/call | ms/chunk |
|------------------------------------------------------------------------------------------|---------|----------|
| `layer`                                                                                  | 0.714   | 2.855    |
| `  attn`                                                                                 | 0.205   | 0.820    |
| `    attn.wqkv_a`                                                                        | 0.054   | 0.215    |
| `    attn.wq_b`                                                                          | 0.044   | 0.175    |
| `    attn.wo_b`                                                                          | 0.062   | 0.247    |
| `    attn.sources :: kv-source compressor + indexer (backend forward_low_ratio_sources)` | 0.084   | 0.335    |
| `    attn.<rest> :: q/kv norm+rope+wo_a (fused)+kv write+sparse attn kernel`             | -0.038  | -0.152   |
| `  mlp`                                                                                  | 0.326   | 1.302    |
| `    mlp.gate`                                                                           | 0.020   | 0.081    |
| `    mlp.topk`                                                                           | 0.026   | 0.105    |
| `    mlp.experts`                                                                        | 0.308   | 1.232    |
| `    mlp.shared_experts`                                                                 | 0.051   | 0.206    |
| `    mlp.<rest> :: shared+routed add, post-expert collective`                            | -0.080  | -0.321   |
| `  layer.<rest> :: mHC boundary (hc_post+hc_pre+norms, fused)+row transport+collectives` | 0.183   | 0.733    |

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
