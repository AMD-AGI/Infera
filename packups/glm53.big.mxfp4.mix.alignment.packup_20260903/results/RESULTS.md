# Results — the three arms and what each isolates

Header on every CSV:
`arm,isl,osl,conc,completed,req_s,out_tok_s,total_tok_s,ttft_p50_ms,ttft_p99_ms,tpot_mean_ms,e2e_p50_ms`

| file | arm | TP | DPA | MTP | kvd | run by |
|---|---|---|---|---|---|---|
| `matched_tp8.csv` | matched | 8 | dp8 | EAGLE 3/1/4 | off | **team lead** |
| `tp4_control.csv` | control | 4 | off | off | off | this author |
| `mix_tp8_featoff_isolator.csv` | isolator | 8 | off | off | off | this author |
| `bigmxfp4_smoke.log` | mxfp4 smoke | 4 | off | off | off | this author |
| `bigfp8_smoke.log` | fp8 smoke, **no throughput numbers** | 4 | off | off | off | this author |

GLM-5.2 baseline (external): `fixlen.glm52.mix.packup_20260806/results/summary.csv`.

## Headline — matched vs baseline

| arm | conc | ours | baseline | ratio |
|---|---:|---:|---:|---:|
| p50 | 1 / 8 / 16 / 24 | 76.26 / 417.63 / 606.33 / 825.07 | 82.56 / 395.58 / 679.82 / 746.75 | 0.92 / **1.06** / 0.89 / **1.10** |
| p90 | 1 / 8 / 16 / 24 | 111.07 / 624.79 / 995.35 / 1346.45 | 112.18 / 606.55 / 1020.36 / 1331.73 | 0.99 / **1.03** / 0.98 / **1.01** |

**0.89–1.11× overall; p90 is 0.98–1.03 — parity.**

## What the confounds were worth — control vs matched

| arm | conc | TP4 control | ratio to baseline | matched TP8 | ratio |
|---|---:|---:|---:|---:|---:|
| p50 | 1 | 59.86 | 0.725 | 76.26 | **0.92** |
| p50 | 8 | 267.08 | 0.675 | 417.63 | **1.06** |
| p50 | 16 | 397.26 | **0.584** | 606.33 | **0.89** |
| p50 | 24 | 622.18 | 0.833 | 825.07 | **1.10** |
| p90 | 1 | 67.63 | 0.603 | 111.07 | **0.99** |
| p90 | 8 | 410.07 | 0.676 | 624.79 | **1.03** |
| p90 | 16 | 678.43 | 0.665 | 995.35 | **0.98** |
| p90 | 24 | 780.20 | **0.586** | 1346.45 | **1.01** |

**TP4 range 0.58–0.83; matched 0.89–1.11.** The gap was configuration.

## The isolator — TP8 with features off

`60.63 / 320.09 / 585.33 / 640.89` at conc 1/8/16/24.

Two uses. It is the **PD reference** (see the PD packup), and against the TP4
control it says **TP8 vs TP4 with no DPA and no MTP is 1.01 / 1.20 / 1.47 / 1.03**
— doubling the GPUs buys ~1 % at conc 1 and ~3 % at conc 24, and only helps in
the middle.

**Open, one line, not chased:** whether that is a property of an osl-320
decode-light shape or a sign something else binds. A throughput measurement at
osl 320 may say more about the workload than the topology.

## Smoke evidence (not throughput)

`bigmxfp4_smoke.log` — all blocks green, including block 6b:
`VERDICT: PASS -- AITER native FP4 MoE, not dequantised to BF16`
(`float4_e2m1fn_x2` ×48, `QuantType.per_1x32` ×72). `max_total_num_tokens=2877376`,
`available_gpu_mem=55.52 GB`.

`bigfp8_smoke.log` — all blocks green, `max_total_num_tokens=1148288`,
`available_gpu_mem=55.44 GB`. **This arm has no throughput numbers**; it is here
as a control showing the fp8 checkpoint runs with shared-experts fusion
**enabled** and no shape mismatch, which is what makes the MXFP4 fusion story a
quantization-mismatch story rather than "fusion is broken on gfx950".
