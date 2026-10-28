> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# VALIDATION — tasks/indexer (pristine, first-hand)

Node compute-node-01, container historical-container (image v0.5.21-rocm720-mi35x-20261011), HIP_VISIBLE_DEVICES=2, 2026-10-20.
Kernel file md5 d4d115639ca5dbefc5d6ae41d4bc5e4b (= dsa_indexer_kernel.pristine.py). graph_harness.py md5 a86ce366... (= sparse_mla copy).

## 1. Pristine copy == production (equivalence script REDACTED_LOCAL_PATH
Copied gluon kernel + wrapper vs the image's own sglang aiter_paged_mqa_logits / sgl_kernel.fast_topk_transform_fused, same inputs:
```
pool build s 13.5
10 logits bitwise equal over [0,L): True ninf in range: 0
10 topk sets equal: True order-identical rerun: False sets rerun: True
12 logits bitwise equal over [0,L): True ninf in range: 0
12 topk sets equal: True order-identical rerun: False sets rerun: True
```
(run on the file before a docstring-only edit). Top-k output ORDER is not deterministic across calls; sets are. The bench capture guard therefore compares row-sorted outputs.

## 2. python3 driver.py (DRIVER_EXIT=0)
```
d4d115639ca5dbefc5d6ae41d4bc5e4b  dsa_indexer_kernel.py
case_snr: T48_bs8_ctx0 137.61
# T48_bs8_ctx0 invalid_rows=0 topk_set_recall=0.999959
case_snr: T48_bs8_ctx1 146.16
# T48_bs8_ctx1 invalid_rows=0 topk_set_recall=0.999990
case_snr: T60_bs10_ctx0 153.83
# T60_bs10_ctx0 invalid_rows=0 topk_set_recall=0.999992
case_snr: T60_bs10_ctx1 150.30
# T60_bs10_ctx1 invalid_rows=0 topk_set_recall=0.999984
case_snr: T72_bs12_ctx0 154.80
# T72_bs12_ctx0 invalid_rows=0 topk_set_recall=0.999980
case_snr: T72_bs12_ctx1 148.90
# T72_bs12_ctx1 invalid_rows=0 topk_set_recall=0.999966
case_snr: T84_bs14_ctx0 100.00
# T84_bs14_ctx0 invalid_rows=0 topk_set_recall=1.000000
case_snr: T84_bs14_ctx1 143.95
# T84_bs14_ctx1 invalid_rows=0 topk_set_recall=0.999971
SNR: 100.00 dB
allclose: True
```

## 3. python3 driver.py --bench-mode, run A and run B (case/median lines; 30 wall_ms samples per case in REDACTED_LOCAL_PATH bench.4.txt)
```
DRIVER_EXIT=0
# T48_bs8 mode: cudagraph
case_ms: T48_bs8 0.376482 unscored
# T60_bs10 mode: cudagraph
case_ms: T60_bs10 0.364913
# T72_bs12 mode: cudagraph
case_ms: T72_bs12 0.488164
# T84_bs14 mode: cudagraph
case_ms: T84_bs14 0.471003 unscored
median_ms: 0.432564
```
```
DRIVER_EXIT=0
# T48_bs8 mode: cudagraph
case_ms: T48_bs8 0.377443 unscored
# T60_bs10 mode: cudagraph
case_ms: T60_bs10 0.365972
# T72_bs12 mode: cudagraph
case_ms: T72_bs12 0.487903
# T84_bs14 mode: cudagraph
case_ms: T84_bs14 0.470923 unscored
median_ms: 0.432900
```
Spread A vs B: median_ms 0.432564 vs 0.432900 (0.08 %); bs10 0.29 %, bs12 0.05 %. Two earlier runs (docstring-only-different file): median_ms 0.432330 / 0.431912.
Per-sample min-max within run A: T48_bs8 0.3752-0.3775 (0.6 %); T60_bs10 0.3604-0.3674 (1.9 %); T72_bs12 0.4821-0.4946 (2.6 %); T84_bs14 0.4648-0.4775 (2.7 %).

## 4. python3 driver.py --profile-run
DRIVER_EXIT=0, no stdout. rocprofv3 --kernel-trace --stats of a profile run (target kernels; the rest is pool/case setup):
```
"Name","Calls","TotalDurationNs","AverageNs"
"_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle",6,1885933,314322.166667
"void sgl::dsa_topk::coop_topk_kernel<2048u, 12u, 4096u, 1024u>(...)",6,1117567,186261.166667
```

## 5. Anchor comparison (REDACTED_LOCAL_PATH 4 pools, CUDA graph, per-call µs)
```
bs8 ctx0         mean_L   271.2K max_L   461.9K sumL/6  2.17M | full   377.8 us  logits   287.7  topk   84.2  | anchor 280 us | modes cudagraph,cudagraph,cudagraph
bs8 uniform230K  mean_L   230.0K max_L   230.0K sumL/6  1.84M | full   271.6 us  logits   210.2  topk   62.1  | anchor 280 us | modes cudagraph,cudagraph,cudagraph
bs10 ctx0         mean_L   216.9K max_L   384.9K sumL/6  2.17M | full   363.8 us  logits   273.8  topk   85.8  | anchor 337 us | modes cudagraph,cudagraph,cudagraph
bs10 uniform210K  mean_L   210.0K max_L   210.0K sumL/6  2.10M | full   341.6 us  logits   274.5  topk   66.5  | anchor 337 us | modes cudagraph,cudagraph,cudagraph
bs12 ctx0         mean_L   180.8K max_L   432.9K sumL/6  2.17M | full   489.3 us  logits   307.8  topk  174.1  | anchor 334 us | modes cudagraph,cudagraph,cudagraph
bs12 uniform180K  mean_L   180.0K max_L   180.0K sumL/6  2.16M | full   339.7 us  logits   246.3  topk   87.2  | anchor 334 us | modes cudagraph,cudagraph,cudagraph
bs14 ctx0         mean_L   154.9K max_L   429.4K sumL/6  2.17M | full   472.8 us  logits   292.8  topk  173.5  | anchor 340 us | modes cudagraph,cudagraph,cudagraph
bs14 uniform155K  mean_L   155.0K max_L   155.0K sumL/6  2.17M | full   337.8 us  logits   251.5  topk   78.1  | anchor 340 us | modes cudagraph,cudagraph,cudagraph
```
Uniform contexts at the anchors' ISLs reproduce the in-model anchors within 3 %. The workset ctx0 contexts (heterogeneous, max 385–462 K) are slower; the bs12/bs14 top-k doubles (174 vs 87 µs) — mechanism not established.

## 6. Metric calibration (REDACTED_LOCAL_PATH bs12 ctx0, perturbed selections)
```
noise1e-4      SNR   123.65 dB  recall 0.9999
noise1e-3      SNR    97.22 dB  recall 0.9990
noise1e-2      SNR    68.33 dB  recall 0.9900
noise3e-2      SNR    54.47 dB  recall 0.9715
noise1e-1      SNR    38.48 dB  recall 0.9046
noise3e-1      SNR    23.76 dB  recall 0.7237
bf16           SNR    73.90 dB  recall 0.9949
fp16           SNR   102.58 dB  recall 0.9995
drop_last_100  SNR    26.28 dB  recall 0.9512
random         SNR    -0.18 dB  recall 0.0220
```
