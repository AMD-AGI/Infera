# GLM-5.2 PD 4+4 (135 -> 136): MTP K3 = new best 46,565; draft FP8, graph cap 80, MLA bmm configs, real MTP acceptance (2026-10-26)

Follows `yihou/glm52.llrepro-c96-bottleneck.packup_20261026` (llr24, C96, 44,432). All e2e runs: prefill
crsuse2-m2m-135 GPU4-7 (PP4xTP1), decode crsuse2-m2m-136 GPU4-7 (TP4/DP4 DPA), llying 4+4 kit + our overlay, AgentX
3600 s at CONC=96, A1 only. Goal = throughput only (user 10-26). Acceptance is SIMULATED in every e2e run
(`SGLANG_SIMULATE_ACC_LEN`): **no correctness claim**. Master table of all optimizations: `OPT_LEDGER.md`.

## E2E results (one variable vs llr24 unless noted; run-to-run spread about ±0.9 %)

| run | change | score | vs llr24 | qps | TPOT ms | decode step p50/p90 ms | verdict |
|---|---|---|---|---|---|---|---|
| llr24 | base (previous pack-up) | 44,432 | — | 2.942 | 17.6 | 55.3 / — | base |
| llr25 | decode per-rank CUDA graph cap 64 -> 80 | — | — | — | — | 57.2 / 79 (13 min) | FAILED: HSA out of resources 28 min into profiling; 5.85 GB free after capture |
| llr26 | MTP draft MoE experts online FP8 (`SGLANG_GLM_NEXTN_MOE_PTPC=1`) | 45,066 | +1.4 % | 2.963 | 17.2 | 54.2 / 83 | kept |
| **llr27** | **MTP K3: EAGLE 3/1/4, sim acceptance 2.99 (ATOM value)** | **46,565** | **+4.8 %** | **3.041** | **15.4** | **44.6 / 54** | **new best** |
| llr28 | PR #209 MLA bmm per-M Triton configs (3 JSONs mounted) | 44,847 | +0.9 % | 2.957 | 17.8 | 55.4 / 86 | neutral (inside spread; not carried) |

K3 also frees decode memory: verify-graph memory 10.5 -> 7.5 GB, 13.5 GB free after capture (vs 5.85 GB at K5 with cap 80).
In progress at pack-up time: llr29 = llr27 + draft FP8; queued llr31 = llr27 + graph cap 80 (retry of llr25 at K3).
The K3 vs K5 comparison rests on the simulated acceptance values 3.61 / 2.99; see the next section for real values.

## Real MTP acceptance (pool node crsuse2-m2m-190, single TP4/DP4 server, bf16 draft, no simulation)
`acceptance/results/summary.md`. Same requests to K5 and K3; tokens / verify steps.

| requests | K5 greedy / sampled | K3 greedy / sampled |
|---|---|---|
| AgentX-style synthetic (40; Shakespeare filler as aiperf does, ignore_eos) | 2.81 / 2.67 | 2.61 / 2.32 |
| real coding prompts (16; sglang source + task) | 3.92 / 3.81 | 3.24 / 3.12 |
| value simulated in our e2e runs | 3.61 | 2.99 |

MXFP4 draft (online, our patch `mxfp4_draft/`), K5, 12 generic prompts: 3.16-3.42 vs bf16 3.55; draft memory 6.92 -> 5.07 GB.

## Side work (offline, pool node 190)
- `mla_bmm/`: PR #209 MLA bmm table regenerated for our image + microbench (`r1.log`): v-up verify x1.2-2.9 faster,
  fused q-up x0.93-1.25. E2E (llr28) showed no measurable gain.
- `aiter_retune/`: a16w16 BF16 GEMM retune of the decode shapes that hit "not found tuned config -> torch"
  (o_proj N=6144 K=16384 etc.): 65 shapes from the K5 run (`c96/`), 10 more M values that only K3 hits (`k3/`). 0 key
  overlap with the image CSVs. Not yet run e2e. The BF16 MoE of the draft (E=257, no quant) has no tuner candidates.
- `mxfp4_draft/`: sglang patch enabling `--speculative-draft-model-quantization quark_mxfp4` for the GLM MTP draft.

## Correctness note (not fixed; user 10-24: performance first)
On 190, the kit decode flags `--enable-fused-qk-norm-rope --enable-aiter-allreduce-fusion` together with custom
all-reduce produced garbage output (bf16 and MXFP4 draft); without them and with `--disable-custom-all-reduce` output
was sane. The responsible flag was not isolated. All e2e runs here carry these flags.

## Layout
`results/llr/<run>/` (agentx json, hist, bneck, qseries, isl_series, key log lines, guard/run logs), `sites/` (site
files llr24-llr31), `acceptance/`, `mla_bmm/`, `aiter_retune/`, `mxfp4_draft/`, `scripts/`, `OPT_LEDGER.md`,
`REPRODUCE.md`, `notes.md`, `environment.md`.
