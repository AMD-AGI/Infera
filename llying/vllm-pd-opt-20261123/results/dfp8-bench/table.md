# dense BF16 (decode today) vs online fp8_per_channel (PTPC), us/call median, TP1, eager, rotated weights
| shape | N x K | M | bf16 branch | bf16 us | fp8 us | fp8/bf16 | quant us | ck us | bf16 TB/s | fp8 TB/s | err bf16 | err fp8 | err fp8 outl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fused_qkv_a | 2624x6144 | 1 | wvSplitK | 7.4 | 8.8 | 1.18 | 2.2 | - | 4.33 | 1.84 | 1.7e-03 | 3.8e-02 | 3.8e-02 |
| fused_qkv_a | 2624x6144 | 4 | wvSplitK | 8.9 | 8.6 | 0.96 | 2.3 | - | 3.60 | 1.89 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 8 | aiter_tgemm | 8.1 | 8.6 | 1.06 | 2.4 | - | 3.96 | 1.88 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 16 | aiter_tgemm | 8.5 | 8.8 | 1.03 | 2.4 | - | 3.78 | 1.83 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 32 | aiter_tgemm | 8.9 | 9.5 | 1.07 | 2.5 | - | 3.64 | 1.70 | 3.6e-03 | 3.7e-02 | 3.6e-02 |
| fused_qkv_a | 2624x6144 | 56 | aiter_tgemm | 9.5 | 10.6 | 1.11 | 2.5 | - | 3.38 | 1.53 | 3.6e-03 | 3.7e-02 | 3.6e-02 |
| fused_qkv_a | 2624x6144 | 64 | aiter_tgemm | 9.9 | 10.8 | 1.09 | 2.5 | - | 3.26 | 1.50 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 96 | aiter_tgemm | 13.9 | 13.1 | 0.94 | 2.5 | - | 2.32 | 1.23 | 2.9e-03 | 3.7e-02 | 3.6e-02 |
| fused_qkv_a | 2624x6144 | 128 | aiter_tgemm | 15.0 | 13.6 | 0.91 | 2.6 | - | 2.15 | 1.19 | 2.9e-03 | 3.8e-02 | 3.6e-02 |
| fused_qkv_a | 2624x6144 | 192 | aiter_tgemm | 22.9 | 14.7 | 0.64 | 2.8 | - | 1.41 | 1.10 | 3.2e-03 | 3.7e-02 | 3.6e-02 |
| fused_qkv_a | 2624x6144 | 256 | aiter_tgemm | 21.6 | 18.5 | 0.86 | 3.0 | - | 1.49 | 0.87 | 3.2e-03 | 3.8e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 2048 | aiter_tgemm | 85.6 | 46.5 | 0.54 | 8.0 | - | 0.38 | 0.35 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| fused_qkv_a | 2624x6144 | 4096 | aiter_tgemm | 117.8 | 79.3 | 0.67 | 13.0 | - | 0.27 | 0.20 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| q_b | 16384x2048 | 1 | wvSplitK | 13.6 | 10.8 | 0.79 | 2.1 | - | 4.93 | 3.12 | 1.7e-03 | 3.8e-02 | 3.9e-02 |
| q_b | 16384x2048 | 4 | wvSplitK | 14.5 | 10.9 | 0.75 | 2.1 | - | 4.62 | 3.09 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 8 | aiter_tgemm | 13.9 | 10.9 | 0.79 | 2.2 | - | 4.84 | 3.07 | 2.4e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 16 | aiter_tgemm | 15.8 | 10.9 | 0.69 | 2.1 | - | 4.25 | 3.08 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| q_b | 16384x2048 | 32 | aiter_tgemm | 17.5 | 11.4 | 0.65 | 2.1 | - | 3.85 | 2.95 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 56 | aiter_tgemm | 16.7 | 11.0 | 0.66 | 2.2 | - | 4.02 | 3.05 | 2.4e-03 | 3.8e-02 | 3.5e-02 |
| q_b | 16384x2048 | 64 | aiter_tgemm | 16.9 | 11.2 | 0.67 | 2.1 | - | 3.98 | 2.99 | 2.5e-03 | 3.7e-02 | 3.6e-02 |
| q_b | 16384x2048 | 96 | aiter_tgemm | 18.8 | 12.8 | 0.68 | 2.1 | - | 3.57 | 2.61 | 2.4e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 128 | aiter_tgemm | 19.9 | 13.6 | 0.68 | 2.1 | - | 3.37 | 2.47 | 2.5e-03 | 3.7e-02 | 3.6e-02 |
| q_b | 16384x2048 | 192 | aiter_tgemm | 31.2 | 18.0 | 0.58 | 2.2 | - | 2.15 | 1.87 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 256 | aiter_tgemm | 32.1 | 18.5 | 0.58 | 2.1 | - | 2.09 | 1.81 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 2048 | aiter_tgemm | 118.1 | 80.5 | 0.68 | 4.0 | - | 0.57 | 0.42 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| q_b | 16384x2048 | 4096 | aiter_tgemm | 347.2 | 160.5 | 0.46 | 6.1 | - | 0.19 | 0.21 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| o_proj | 6144x16384 | 1 | wvSplitK | 35.6 | 26.4 | 0.74 | 3.1 | - | 5.66 | 3.81 | 1.7e-03 | 3.8e-02 | 3.5e-02 |
| o_proj | 6144x16384 | 4 | wvSplitK | 36.8 | 27.0 | 0.73 | 3.2 | - | 5.48 | 3.73 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 8 | aiter_tgemm | 36.6 | 27.3 | 0.75 | 3.3 | - | 5.51 | 3.68 | 2.4e-03 | 3.8e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 16 | aiter_tgemm | 38.6 | 28.7 | 0.74 | 3.4 | - | 5.21 | 3.51 | 3.2e-03 | 3.8e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 32 | aiter_tgemm | 40.3 | 29.8 | 0.74 | 3.4 | - | 4.99 | 3.38 | 3.0e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 56 | aiter_tgemm | 40.9 | 32.0 | 0.78 | 3.5 | - | 4.93 | 3.15 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 64 | aiter_tgemm | 41.5 | 32.1 | 0.77 | 3.5 | - | 4.85 | 3.13 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 96 | aiter_tgemm | 47.5 | 37.0 | 0.78 | 3.7 | - | 4.24 | 2.72 | 3.2e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 128 | aiter_tgemm | 50.8 | 39.4 | 0.78 | 4.1 | - | 3.97 | 2.55 | 3.2e-03 | 3.8e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 192 | aiter_tgemm | 62.0 | 46.3 | 0.75 | 4.4 | - | 3.25 | 2.17 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 256 | aiter_tgemm | 78.8 | 48.1 | 0.61 | 4.7 | - | 2.56 | 2.09 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 2048 | aiter_tgemm | 340.2 | 203.4 | 0.60 | 23.6 | - | 0.59 | 0.49 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 4096 | aiter_tgemm | 625.1 | 389.4 | 0.62 | 47.1 | - | 0.32 | 0.26 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| indexer.wq_b | 4096x2048 | 1 | wvSplitK | 5.7 | 5.1 | 0.90 | 2.2 | - | 2.92 | 1.63 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 4 | wvSplitK | 6.3 | 5.0 | 0.80 | 2.1 | - | 2.67 | 1.67 | 1.7e-03 | 3.7e-02 | 3.4e-02 |
| indexer.wq_b | 4096x2048 | 8 | aiter_tgemm | 5.3 | 5.1 | 0.96 | 2.2 | - | 3.16 | 1.65 | 2.5e-03 | 3.8e-02 | 3.4e-02 |
| indexer.wq_b | 4096x2048 | 16 | wvSplitKrc | 7.3 | 5.2 | 0.71 | 2.2 | - | 2.29 | 1.62 | 1.7e-03 | 3.8e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 32 | aiter_tgemm | 6.0 | 6.2 | 1.04 | 2.2 | - | 2.79 | 1.34 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| indexer.wq_b | 4096x2048 | 56 | aiter_tgemm | 6.6 | 6.3 | 0.95 | 2.2 | - | 2.53 | 1.33 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| indexer.wq_b | 4096x2048 | 64 | aiter_tgemm | 6.7 | 6.3 | 0.94 | 2.2 | - | 2.49 | 1.33 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 96 | aiter_tgemm | 8.5 | 6.7 | 0.79 | 2.2 | - | 1.98 | 1.25 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 128 | aiter_tgemm | 8.8 | 7.0 | 0.80 | 2.2 | - | 1.91 | 1.20 | 1.7e-03 | 3.8e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 192 | aiter_tgemm | 10.2 | 8.8 | 0.86 | 2.2 | - | 1.64 | 0.96 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| indexer.wq_b | 4096x2048 | 256 | aiter_tgemm | 10.6 | 9.0 | 0.84 | 2.1 | - | 1.58 | 0.94 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 2048 | aiter_tgemm | 32.5 | 27.5 | 0.85 | 4.1 | - | 0.52 | 0.30 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| indexer.wq_b | 4096x2048 | 4096 | aiter_tgemm | 53.4 | 39.1 | 0.73 | 6.1 | - | 0.31 | 0.21 | 1.7e-03 | 3.7e-02 | 3.5e-02 |
| mlp.gate_up | 24576x6144 | 1 | wvSplitK | 48.0 | 31.6 | 0.66 | 2.2 | - | 6.30 | 4.78 | 1.6e-03 | 3.7e-02 | 3.9e-02 |
| mlp.gate_up | 24576x6144 | 4 | wvSplitK | 53.1 | 31.7 | 0.60 | 2.3 | - | 5.68 | 4.76 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 8 | aiter_tgemm | 53.6 | 31.9 | 0.60 | 2.4 | - | 5.64 | 4.73 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| mlp.gate_up | 24576x6144 | 16 | aiter_tgemm | 54.6 | 32.1 | 0.59 | 2.4 | - | 5.54 | 4.70 | 1.7e-03 | 3.8e-02 | 3.6e-02 |
| mlp.gate_up | 24576x6144 | 32 | aiter_tgemm | 49.4 | 29.9 | 0.61 | 2.5 | - | 6.11 | 5.04 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 56 | aiter_tgemm | 50.8 | 31.8 | 0.63 | 2.6 | - | 5.95 | 4.76 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 64 | aiter_tgemm | 51.0 | 31.6 | 0.62 | 2.5 | - | 5.92 | 4.78 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 96 | aiter_tgemm | 66.5 | 34.6 | 0.52 | 2.5 | - | 4.54 | 4.37 | 2.5e-03 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 128 | aiter_tgemm | 63.1 | 39.1 | 0.62 | 2.5 | - | 4.78 | 3.87 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| mlp.gate_up | 24576x6144 | 192 | aiter_tgemm | 83.4 | 50.7 | 0.61 | 2.6 | - | 3.62 | 2.98 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| mlp.gate_up | 24576x6144 | 256 | aiter_tgemm | 101.7 | 57.8 | 0.57 | 2.7 | - | 2.97 | 2.61 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 2048 | aiter_tgemm | 476.0 | 301.7 | 0.63 | 8.0 | - | 0.63 | 0.50 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 4096 | aiter_tgemm | 943.2 | 546.9 | 0.58 | 12.9 | - | 0.32 | 0.28 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 1 | wvSplitK | 26.1 | 18.0 | 0.69 | 2.7 | - | 5.79 | 4.19 | 1.6e-03 | 3.8e-02 | 3.5e-02 |
| mlp.down | 6144x12288 | 4 | wvSplitK | 26.9 | 19.3 | 0.72 | 2.8 | - | 5.62 | 3.92 | 1.7e-03 | 3.7e-02 | 3.6e-02 |
| mlp.down | 6144x12288 | 8 | aiter_tgemm | 27.7 | 18.4 | 0.67 | 2.9 | - | 5.45 | 4.09 | 3.2e-03 | 3.8e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 16 | aiter_tgemm | 28.2 | 19.1 | 0.68 | 2.9 | - | 5.36 | 3.95 | 3.2e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 32 | aiter_tgemm | 28.9 | 21.2 | 0.73 | 3.0 | - | 5.23 | 3.56 | 3.2e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 56 | aiter_tgemm | 37.7 | 26.0 | 0.69 | 3.0 | - | 4.00 | 2.91 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 64 | aiter_tgemm | 38.8 | 26.7 | 0.69 | 3.0 | - | 3.89 | 2.83 | 3.6e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 96 | aiter_tgemm | 50.4 | 38.0 | 0.75 | 3.3 | - | 2.99 | 1.99 | 2.4e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 128 | aiter_tgemm | 54.5 | 34.5 | 0.63 | 3.4 | - | 2.77 | 2.19 | 2.4e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 192 | aiter_tgemm | 50.0 | 43.8 | 0.88 | 3.8 | - | 3.02 | 1.72 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 256 | aiter_tgemm | 55.9 | 42.8 | 0.77 | 4.0 | - | 2.70 | 1.76 | 1.7e-03 | 3.8e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 2048 | aiter_tgemm | 255.5 | 157.4 | 0.62 | 16.7 | - | 0.59 | 0.48 | 1.7e-03 | 3.7e-02 | 3.7e-02 |
| mlp.down | 6144x12288 | 4096 | aiter_tgemm | 472.9 | 288.4 | 0.61 | 33.2 | - | 0.32 | 0.26 | 1.7e-03 | 3.7e-02 | 3.7e-02 |

## FP8 kernel selected by v0.34.0 per (N, K)
- fused_qkv_a 2624x6144: AiterPreshuffledPerTokenFp8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: selected kernel already is CK bpreshuffle (tuned)
- q_b 16384x2048: AiterPreshuffledPerTokenFp8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: selected kernel already is CK bpreshuffle (tuned)
- o_proj 6144x16384: RowWiseTorchFP8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: AssertionError: 
- indexer.wq_b 4096x2048: AiterPreshuffledPerTokenFp8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: selected kernel already is CK bpreshuffle (tuned)
- mlp.gate_up 24576x6144: RowWiseTorchFP8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: AssertionError: 
- mlp.down 6144x12288: AiterPreshuffledPerTokenFp8ScaledMMLinearKernel (build direct (loader: AssertionError: (torch.float8_e4m3fn, device(type='cpu')))); ck: selected kernel already is CK bpreshuffle (tuned)

## one decode step, target dense GEMMs only, M = 56 (sum over layer counts; MTP layer not included)
- fused_qkv_a x78: bf16 0.744 ms, fp8 0.824 ms (of which act quant 0.196 ms)
- q_b x78: bf16 1.303 ms, fp8 0.858 ms (of which act quant 0.172 ms)
- o_proj x78: bf16 3.187 ms, fp8 2.493 ms (of which act quant 0.269 ms)
- indexer.wq_b x21: bf16 0.139 ms, fp8 0.132 ms (of which act quant 0.047 ms)
- mlp.gate_up x3: bf16 0.152 ms, fp8 0.095 ms (of which act quant 0.008 ms)
- mlp.down x3: bf16 0.113 ms, fp8 0.078 ms (of which act quant 0.009 ms)

**M=56: bf16 5.64 ms, fp8 4.48 ms (delta -1.16 ms; act quant 0.70 ms), fp8 with CK where faster 4.48 ms**

Notes: eager calls (production decode is a FULL CUDA graph and may compile/fuse the activation quant into the
preceding RMSNorm, so 'quant us' is an upper bound); random weights/activations (error on real activations with
outliers is closer to the 'outl' column).
