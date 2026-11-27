# online FP8 PTPC, tuned CK bpreshuffle (selected now) vs hipBLASLt (production today), us/call median, TP1, eager, rotated weights
merged table /tmp/aiter_configs/a8w8_bpreshuffle_tuned_gemm.csv: 24 rows for these two (N, K)

| shape | N x K | M | selected us | hipBLASLt us | sel/hipb | tuner us | row hit | err sel | err hipb |
|---|---|---|---|---|---|---|---|---|---|
| o_proj | 6144x16384 | 1 | 37.9 | 26.7 | 1.42 | - | default (no row) | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 2 | 37.9 | 26.8 | 1.42 | - | default (no row) | 3.8e-02 | 3.8e-02 |
| o_proj | 6144x16384 | 4 | 38.0 | 27.2 | 1.40 | - | default (no row) | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 8 | 30.1 | 27.6 | 1.09 | 24.7 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.8e-02 | 3.8e-02 |
| o_proj | 6144x16384 | 16 | 29.1 | 28.6 | 1.02 | 23.7 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 24 | 36.6 | 29.1 | 1.26 | 27.0 | ck:a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 32 | 33.0 | 29.9 | 1.10 | 27.0 | ck:a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x1 splitK=0 | 3.8e-02 | 3.8e-02 |
| o_proj | 6144x16384 | 40 | 40.8 | 31.2 | 1.31 | 36.2 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 48 | 41.6 | 31.5 | 1.32 | 36.2 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 52 | 41.9 | 31.8 | 1.32 | 37.2 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 56 | 42.2 | 32.0 | 1.32 | 37.2 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 64 | 42.7 | 32.0 | 1.34 | 37.2 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 72 | 65.9 | 34.7 | 1.90 | 42.1 | ck:a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1 splitK=0 | 3.8e-02 | 3.8e-02 |
| o_proj | 6144x16384 | 80 | 66.6 | 34.9 | 1.91 | 42.1 | ck:a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 96 | 48.0 | 36.9 | 1.30 | 41.8 | ck:a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1 splitK=0 | 3.8e-02 | 3.8e-02 |
| o_proj | 6144x16384 | 104 | 69.2 | 39.3 | 1.76 | 42.1 | ck:a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 128 | 48.7 | 40.4 | 1.20 | 42.1 | ck:a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 160 | 71.1 | 44.1 | 1.61 | 53.5 | ck:a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 192 | 57.1 | 47.4 | 1.20 | 50.6 | ck:a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 256 | 57.0 | 49.1 | 1.16 | 53.5 | ck:a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 2048 | 195.9 | 202.4 | 0.97 | 162.6 | ck:a8w8_bpreshuffle_256x256x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 4096 | 373.4 | 384.5 | 0.97 | 305.8 | ck:a8w8_bpreshuffle_256x256x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| o_proj | 6144x16384 | 8192 | 826.2 | 723.6 | 1.14 | - | default (no row) | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 1 | 30.4 | 31.5 | 0.97 | - | default (no row) | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 2 | 30.5 | 31.6 | 0.97 | - | default (no row) | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 4 | 30.7 | 31.7 | 0.97 | - | default (no row) | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 8 | 29.8 | 32.0 | 0.93 | 26.3 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 16 | 29.8 | 32.2 | 0.93 | 26.5 | ck:a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x1 splitK=0 | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 24 | 34.7 | 30.2 | 1.15 | 28.8 | ck:a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 32 | 31.1 | 30.2 | 1.03 | 28.8 | ck:a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 40 | 36.2 | 35.8 | 1.01 | 31.2 | ck:a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 48 | 32.2 | 31.5 | 1.02 | 31.2 | ck:a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 52 | 38.6 | 31.6 | 1.22 | 33.5 | ck:a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 56 | 38.7 | 31.9 | 1.21 | 33.5 | ck:a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 64 | 34.7 | 31.7 | 1.09 | 33.5 | ck:a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 72 | 45.3 | 34.5 | 1.31 | 40.9 | ck:a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 80 | 45.5 | 35.0 | 1.30 | 40.9 | ck:a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1 splitK=0 | 3.8e-02 | 3.8e-02 |
| mlp.gate_up | 24576x6144 | 96 | 39.1 | 34.8 | 1.12 | 37.9 | ck:a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 104 | 46.3 | 40.2 | 1.15 | 40.9 | ck:a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 128 | 41.8 | 39.5 | 1.06 | 40.9 | ck:a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 160 | 77.5 | 46.1 | 1.68 | 54.7 | ck:a8w8_bpreshuffle_256x128x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 192 | 49.9 | 50.9 | 0.98 | 47.5 | ck:a8w8_bpreshuffle_256x96x192x256_16x16_16x16_16x16x1_16x16x1_ splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 256 | 56.1 | 57.7 | 0.97 | 54.7 | ck:a8w8_bpreshuffle_256x128x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 2048 | 294.8 | 297.5 | 0.99 | 258.7 | ck:a8w8_bpreshuffle_256x256x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 4096 | 553.5 | 540.6 | 1.02 | 500.4 | ck:a8w8_bpreshuffle_256x256x192x128_16x16_16x16_8x32x1_8x32x1_1 splitK=0 | 3.7e-02 | 3.7e-02 |
| mlp.gate_up | 24576x6144 | 8192 | 1217.5 | 1009.4 | 1.21 | - | default (no row) | 3.7e-02 | 3.7e-02 |

## kernel per (N, K)
- o_proj 6144x16384: tuned check True; selected AiterPreshuffledPerTokenFp8ScaledMMLinearKernel; production baseline RowWiseTorchFP8ScaledMMLinearKernel
- mlp.gate_up 24576x6144: tuned check True; selected AiterPreshuffledPerTokenFp8ScaledMMLinearKernel; production baseline RowWiseTorchFP8ScaledMMLinearKernel

**PASS: M=56 per decode step (o_proj x78 + gate_up x3): CK 3.411 ms vs hipBLASLt 2.588 ms (delta +0.823 ms)**

Notes: eager calls; production decode is a FULL CUDA graph (launch overhead hidden there). 'row hit' is aiter's
runtime lookup (exact M, then M padded to 16/32/64/128, then next power of two); 'default' = untuned CK kernel.
