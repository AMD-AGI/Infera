# Estimate: keep kv_b bf16 under DCP so q-rep removes the per-layer Q all-gather (2026-10-21)

Every number below is an estimate from shapes plus benchrun's T1 all-gather curve. None of them is measured on this
option. The decisive measurement is in section 5.

## 1. What blocks q-rep today (code read)

- **Why q-rep is off for the target.** `_prepare_replicated_q_proj` (`model_executor/model_runner.py:1120-1157`) skips any layer whose `w_kc` is not bf16/fp16. On gfx950, the GLM bf16 `kv_b_proj` is quantized to per-tensor e4m3fn (`deepseek_common/deepseek_weight_loader.py:724-731`). Result: "prepared ... for 0 MLA layers" for the target. The NextN layer keeps bf16 and is already q-rep.
- **What the checkpoint has.** `q_b_proj` and `kv_b_proj` are both in the quark `exclude` list, so both are bf16 in the checkpoint (`config.json`).
- **The option.** Skip that fp8 conversion when `attn_dcp_size > 1` (and q-rep is on). q-rep then runs the existing ROCm branch (`forward_mla_rocm.py:508-513, 636-641, 685-689`):
  - a full-head `q_b_proj` GEMM,
  - a full-head `torch.bmm` with `w_kc_qrep`,
  - no Q all-gather.
  - It also disables `fuse_bmm_rope_cache` (`forward_mla_rocm.py:676-680`).

Shapes are per rank: TP4 = DCP4, 16 local heads, 64 total. q_lora_rank 2048, kv_lora_rank 512, qk_nope 192, rope 64, v 256, 78 target layers.

## 2. Memory delta per GPU

| Item | Per layer | x78 |
|---|---|---|
| kv_b local fp8 -> bf16 (16 x 448 x 512) | +3.7 MB | +0.29 GB |
| `q_b_proj_qrep_weight`, full heads bf16 (64 x 256 x 2048); the local q_b stays, for extend modes | +67.1 MB | +5.23 GB |
| `w_kc_qrep`, full heads bf16 (64 x 192 x 512) | +12.6 MB | +0.98 GB |
| **Total** | **+83 MB** | **about +6.5 GB per GPU** |

- **Cost in KV tokens.** I assume about 23 KB per logical token per GPU: latent 656 B / 4 + index K 132 B replicated, x 78 layers. Treat this as an estimate; the pool byte size is in the server log. On that basis, 6.5 GB is roughly 280K logical tokens, about 3% of the 8.9M pool. r2 KV usage was 0.43, so this does not bind at C40-C56.
- **The 6.5 GB is dominated by the replicated `q_b_proj`, not by kv_b.** The fp8 alternative (q-rep with gathered and re-quantized fp8 `w_kc`) would save only about 0.3-1 GB of it.

## 3. Compute delta per target verify forward

Assumed shape: T = bs x 6 rows (bs 24 -> T 144). These kernels are memory-bound at decode T; I assume about 4-5 TB/s effective HBM.

| Change | Per layer | x78 |
|---|---|---|
| `q_b` full GEMM (67 MB read, 9.7 GFLOP at T 144) instead of local (16.8 MB) | +10-13 us | +0.8-1.0 ms |
| absorb `torch.bmm` bf16, 64 heads (12.6 MB), instead of aiter fp8, 16 heads | +1-3 us | +0.1-0.2 ms |
| `fuse_bmm_rope_cache` off (separate rope + cache-write launches) | +5-10 us (guess) | +0.4-0.8 ms |
| `w_vc` bf16 `torch.bmm` (16 heads) instead of fp8 | +1-3 us | +0.1-0.2 ms |
| **Total added** | **about +17-29 us** | **about +1.4-2.2 ms** |

## 4. RCCL saving: the Q all-gather

- **Payload.** The Q all-gather moves [T, 16 heads, 576] bf16 per rank, i.e. 18.4 KB x T: 1.77 MB at bs 16, 2.65 MB at bs 24, 3.5 MB at bs 32. That assumes Q is gathered in bf16; I did not check whether it is fp8 at that point.
- **Latency.** Read off benchrun T1 (same GPUs, 4 ranks, graph replay; 1.3 MB 44 us, 2.6 MB 70 us, 3.9 MB 94 us), that is about 52 / 70 / 87 us per layer.
- **Per verify step, x78:** about 4.1 / 5.5 / 6.8 ms at bs 16 / 24 / 32.
- **Cross-check.** The r2 verify trace has RCCL at 17.8 ms of 70.6 ms (bs 24). That is 897 `ncclDevKernel_Generic_1` kernels in the whole trace at 51 us average. The kernel name does not distinguish the collective type, so the Q all-gather share cannot be read from the trace. 5.5 ms would be about 31% of the RCCL time.
- **Draft forwards.** Unchanged: the NextN layer is already q-rep.

**Net estimate: about 2-5 ms saved per decode step at bs 16-32, roughly 3-8% of the 59-66 ms steps measured.** The uncertainty is mostly the q_b GEMM time and the lost fused rope/cache kernel.

## 5. Code change and how to decide

- **(a) The bf16 option.** About 3 lines in `deepseek_weight_loader.py:724-731`: add `and get_parallel().attn_dcp_size == 1`, or condition on `dcp_replicate_q_proj`. Plus one CPU test for the dtype decision. Effects:
  - PD prefill (DCP1): unaffected.
  - Mixed DCP runs: their prefill absorb bmm also becomes bf16 `torch.bmm`, a prefill-side cost that is unmeasured.
- **(b) The fp8 alternative.** At prep, dequantize each rank's `w_kc` with its own scale, gather, and re-quantize with one scale. Then call the aiter fp8 bmm in the q-rep branch: about 25 lines. This saves only the bf16-bmm and kv_b items above (about 0.3 GB, about 0.2-0.4 ms); the replicated q_b dominates either way.
- **Decisive measurement.** mixrun's decodebench (34K and 150K, N = 8/16/24/32, swapped halves), with (a) on vs off, sim 3.61. Then one correctness gate (math/passkey, sim off). Here bf16 kv_b is the more exact weight; the fp8 path is the approximation.
