# Results — every arm, with what each one isolates

All arms: GLM-5.3-MXFP4, single-node 1P1D, TP4 prefill + TP4 decode, MTP off,
`isl 7400 / osl 320` unless noted, benched through the router.

| file | arm | node | decode DPA | hip |
|---|---|---|---|---|
| `pd_run1_dpa_on_hip_on_n0133.csv` | PD run 1, full curve | n01-33 | **on** (dp4) | on |
| `pd_dpa_off_conc24_n0133.csv` | the isolator, conc 24 | n01-33 | **off** | on |
| `pd_dpa_off_c1_c8_c16_n0121.csv` | isolator, rest of curve | n01-21 | **off** | on |
| `ab_hip_on_n0121.csv` | A/B reference | n01-21 | on | **on** |
| `ab_hip_off_n0121.csv` | A/B treatment | n01-21 | on | **off** |
| `repetition_armA_p90.csv` | osl 3300 quality arm | n01-21 | on | off |
| `repetition_armA_generations.jsonl` | 10 generations, loopcheck-ready | | | |

MIX reference for the PD/MIX comparison lives in the sibling packup:
`glm53.big.mxfp4.mix.alignment.packup_20260903/results/mix_tp8_featoff_isolator.csv`.

## The PD/MIX question — feature-matched

| conc | PD (DPA off) | MIX TP8 (DPA off) | PD / MIX |
|---:|---:|---:|---:|
| 1 | 66.45 | 60.63 | **1.10** |
| 8 | 404.28 | 320.09 | **1.26** |
| 16 | 511.46 | 585.33 | 0.87 |
| 24 | 554.10 | 640.89 | 0.87 |

**Ahead at low concurrency, behind above it.** Conc 24 is the only same-node
point; the rest are cross-node and carry the measured ~5 % node delta.

### What DP-attention is worth, and why the unmatched number misled

| conc 24 arm | out tok/s | ttft_p50 | tpot |
|---|---:|---:|---:|
| PD, decode DPA **on** | 734.92 | 657 ms | 21.42 |
| PD, decode DPA **off** | 554.10 | 6584 ms | 21.24 |
| MIX TP8, DPA off | 640.89 | 2548 ms | 23.72 |

**DPA on the PD decode leg is worth 1.33× throughput and 10× TTFT, and nothing on
TPOT.** PD-with-DPA vs MIX-without said **1.15× ahead**; matched it is **0.87×**.
The difference was DP-attention, not disaggregation.

## The hip A/B — same node, same image, one variable

| conc | hip ON | hip OFF | ratio | TTFT on → off |
|---:|---:|---:|---:|---:|
| 1 | 44.52 | 46.07 | 1.03 | 193 → 198 ms |
| 8 | 274.12 | **336.44** | **1.23** | 1977 → **589 ms** |

**Noise floor: ~5 %**, measured — the same configuration on two nodes gave 1.01×
and 1.05× (`ab_hip_on_n0121.csv` conc 1/8 vs `pd_run1_dpa_on_hip_on_n0133.csv`
conc 1/8: 44.52 vs 44.07, 274.12 vs 261.57). The 23 % is 4–5× the floor. **The
3 % at conc 1 is inside it and is noise, not a small effect.**

## Repetition arm — `isl 15500 / osl 3300`, conc 1

```
10 generations, looping = some 10-gram repeats >= 5x
  ALL   n=10  looping=6 (60.0%)  tokens-in-looping=60.0%  worst=x905
  worst: req 1 max_repeat=x905 unique_word_ratio=0.070
```

Chat template applied (server-side, via `/v1/chat/completions`) **and** MTP off,
so `temperature 1.0 / top_p 0.95` genuinely reached token selection. **Both
leading causes eliminated.** Score it yourself:

```bash
python3 scripts/loopcheck.py results/repetition_armA_generations.jsonl
```

That JSONL is already converted to the format `loopcheck.py` expects. Feeding it
the raw `--output-details` file instead reports `looping = 0 (0.0%)` — see
`notes.md` §4.

## How to re-derive any ratio

Every CSV is machine-readable with the header
`arm,isl,osl,conc,completed,req_s,out_tok_s,total_tok_s,ttft_p50_ms,ttft_p99_ms,tpot_mean_ms,e2e_p50_ms`.
All ratios above are `out_tok_s` divided by `out_tok_s`. **`ttft_p99` is not a
latency percentile on these arms** — it measures a whole-wave stall; see
`notes.md` §7.
