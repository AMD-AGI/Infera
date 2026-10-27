# Notes (what / why / how)

- **Spread**: llr35 vs replicate llr35r differ by 0.2 %; llr33/35/35r/36 lie within 47.8-48.0 k. Treat < 0.5 % deltas as
  neutral; earlier stacks showed ±0.9 %.
- **Graph cap 80 (llr31)**: no OOM at K3 (9.5 GB free after capture) but no gain: per-rank running mean 12 / p90 18 at
  C96, so the cap of 64 never binds. User asked for it because 64 looked small; the data shows the batch is KV-limited.
- **Decode mem**: 0.95 gave +1.1 % (KV +4.2 %); 0.96 gave KV +2 % but no score change. 0.95 at K5 (llr8) had failed with
  HSA out of resources; at K3 + FP8 draft there was room. llr17 (0.93 + chunk 4096) had also crashed earlier.
- **GEMM retune**: the image had no tuned config for the decode o_proj (N=6144 K=16384) and, with the FP8 draft, for
  N=1024 K=6144 / N=6144 K=512 shapes; aiter fell back to torch. Tuned rows mix hipBLASLt, FlyDSL and asm kernels.
  Shapes depend on the verify width (K3 vs K5 changes M) and on draft quantization, so re-collect after such changes.
- **MXFP4 draft (llr34)**: as fast as FP8 within noise; real acceptance measured lower (190, previous pack-up). Not used.
- **Queue race**: `queue_runs.yihou.sh` launches the next site within 30 s of RUN_LLR_DONE; edit a queued site file
  before the previous run finishes (llr34 started on the llr32 base because of this; still one variable vs llr32).
- **Correctness** (unchanged from the previous pack-up): the kit decode fused flags + custom AR produced garbage on 190;
  e2e runs use simulated acceptance and carry these flags.
