# GLM-5.2 DCP4 decode: raising mem-fraction-static above 0.85 (2026-10-22 15:20Z -> 10-23 02:45Z UTC)

**Question.** The DCP4 decode leg (TP4 EP4 DCP4, same-node P4D4 PD on MI355X, llying v0522 config) ran at
`--mem-fraction-static 0.85`. Can it run at 0.92 (user target) for more KV?

**Answer (measured).**
- Without changes, 0.87 / 0.88 / 0.92 all abort under AgentX C80 load (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`).
  Cause found with an instrumented build: the torch caching allocator's reserve grows ~15-17 GiB above live tensors
  until the device is full.
- Fix: `torch.cuda.empty_cache()` on every decode rank when device free < 6 GiB (`patches/03-ec2-allrank.decode.diff`).
- With the fix: 0.88 completes at **27,685 tok/s/chip** (S2n; 0.85 runs: 25,941 / 25,433), 0.92 completes at
  22,540 (S2m, same node as S2n). Single runs; simulated acceptance (no correctness claim).

## Read in this order
1. `OPTIMIZATION.md` — what to apply, results, what did not work, open items, references. **Start here.**
2. `PROCESS.md` — step-by-step how it was found (code map -> start-up probes -> load runs -> instrumentation -> fix).
3. `REPRODUCE.md` — commands; `environment.md` — nodes, image, code SHAs.
4. `results/RESULTS.md` — the running lab log with every run; `results/memory_map.md` — the code-read memory map.

## Layout
- `patches/` 01 memlog (diagnostic print), 02 empty_cache at TP0-only log point (superseded, shows the failure
  mode), 03 empty_cache on every rank (the fix). Repo-relative to sglang `ab7a898b77`; `git apply --check` passes
  for 01 and 03 on that commit.
- `scripts/stages/S2*.env` — every load-run config (stage files for the llying kit); `scripts/gmu/` — decode-only
  probe, series files, launchers, samplers, overlay builders, collectors.
- `results/runs/<RUN_ID>/` — analysis.txt, AgentX JSON, engine-config, TP0 start-up memory lines, abort excerpts;
  `results/memlog/` — TP0 memory per Decode-batch line; `results/vram/` — rocm-smi per-GPU samples (15 s);
  `results/startup/` — decode-only probe summaries.

Related: `../glm52.llying-dcp-s1s3.packup_20261022/` (base config S1-S3, kit, image recipe).
Not included (size): full container logs and aiperf raw exports, under
`/shared_nfs/yihou_glm52/llying_dcp_20261022/work/<stage>-<NK>/runs/<RUN_ID>/`.
