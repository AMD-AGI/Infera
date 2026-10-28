### Scope / current blocker

Historical ROCm DSA DCP-PD report with measured allocator growth and HIP aborts. **A reproduction on current upstream main is still pending**: the measured stack used local DSA DCP/PD enablement, now tracked through #44322/#42618 and the PD drafts #46824/#46826. This report preserves the evidence and requests root-cause investigation; it does not claim that unmodified main supports or reproduces that exact configuration.

The local S12 workaround periodically calls `empty_cache()` on all decode ranks. I am filing an issue rather than a PR for that hard-coded policy: the allocation source and latency cost are not isolated.

### Environment and workload

- GLM-5.2-MXFP4, 8×MI355X in same-node P4D4 PD; decode TP4/EP4/DCP4, FP8-e4m3 KV.
- SGLang local branch commit `ab7a898b77`, based on the v0.5.22-era DCP work; image `infera-sglang:v0522-d395c3b3-1020-mc4340`.
- PyTorch 2.9.1+ROCm 7.2.0; observed across amdgpu 6.14.14 / 6.16.13 nodes (the latter needed SDMA disabled).
- EAGLE steps/topk/draft-tokens = 5/1/6, index sharing enabled, replicated-Q/BF16 and split-indexer options in that local DCP branch, decode custom all-reduce enabled.
- AgentX C80, 1,800 seconds, 10 warmup requests/lane, seed 42; InferenceX `918524ff`, trace dataset `semianalysis_cc_traces_weka_062126`.
- `SGLANG_SIMULATE_ACC_LEN=3.61` was enabled. This workload is not model-correctness evidence. No throughput claims are made.

### Observed failure

Changing decode `--mem-fraction-static` from 0.85 to 0.87/0.88/0.92 left less runtime headroom and repeatedly ended with:

```text
rocdevice.cpp:3582 ... Aborting with error:
HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources.
Fatal Python error: Aborted
```

In `S2e-20261022T203420Z` (0.88), live Torch allocations grew about **0.8 GiB** while Torch reserved memory grew **15.7 GiB**. Device free memory fell **16.36 -> 0.92 GiB**, with the abort about two seconds later. The link to the particular failing HIP allocation is inferred from timing, not traced. Cached unused blocks growing is established; a specific fragmentation or leak mechanism is not.

Controls/mitigations:

- 0.85 completed (`S2`, `S2r`) with enough headroom for comparable reserve growth.
- `expandable_segments:True` (`S2h`) and `garbage_collection_threshold:0.9` (`S2i`) still aborted.
- Graph trimming at 0.92 gained startup headroom but still aborted (`S2a`, `S2b`).
- TP0-only cache release (`S2k`) restored TP0 to 10.7 GiB free, while other decode ranks stayed near full and aborted.
- All-rank release below 6 GiB free, every eight decode iterations, allowed single runs at 0.88 (`S2n`) and 0.92 (`S2m`) to finish. These builds also retained diagnostics and the earlier TP0 hook; the final all-rank hunk alone was not measured.

### Reproduction / investigation needed

The historical trigger is a bursty C80 workload on the above DCP-PD topology, comparing 0.85/0.88/0.92 after graph capture. The exact branch/configuration is material: this is not a standalone current-main repro command. Evidence is archived in `glm52.dcp4-decode-gmu.packup_20261023` (run IDs above, memory logs and abort excerpts).

Once the final DCP/PD dependencies are available, repeat with real acceptance and log **every rank's** `memory_allocated`, `memory_reserved`, `memory_stats` and `mem_get_info`, alongside transfer/admission batch shapes. Capture an allocator snapshot/trace and identify the failing non-Torch allocation; measure any cache-release pause before proposing a policy. Recheck current allocator/graph behavior instead of carrying the old unconditional workaround forward.

The later October 27 GLM bisect implicated PP communication overlap and decode custom all-reduce in garbled output. It does not prove the cause of this memory failure, and the simulated-acceptance stack's throughput cannot validate a memory workaround. Related #45824 concerns a different persistent-buffer/graph correctness investigation and is not assumed to fix this report.
