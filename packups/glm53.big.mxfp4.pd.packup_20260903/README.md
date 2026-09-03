# GLM-5.3-MXFP4 (big) — single-node 1P1D disaggregation

**Ran:** 2026-09-02 (`n01-33`) and 2026-09-03 (`n01-21`), 8×MI355X.
**Status: FIRST RUN.** Mission running-configurations 5 and 6 had never been
executed. They now have.

## Goal

**Spec:** the GLM-5.3-series integration task, `mission.md`, running
configurations **5 and 6** (PD-disaggregated). Distilled form in the repo at
`.claude/CLAUDE.md`, branch `yihou.dev.glm53.expr`, commit `46e79746`. The kit's
own contract is `examples/sglang_1p1d_glm5.3/README.md`.

**Success criterion:** none numeric — these configurations had never been run, so
the bar is that the shape comes up, serves a real completion, and produces
numbers that can be *attributed*.

Bring up PD-disaggregated GLM-5.3-MXFP4 on a single 8-GPU node (TP4 prefill on
GPUs 0-3, TP4 decode on 4-7, KV over mooncake), and answer two questions:
**what does disaggregation cost against an aggregated deployment on the same
hardware**, and **does the intra-node HIP/XGMI transport actually carry KV**.

## Result — two headlines, and **both inverted a conclusion drawn one arm too early**

### 1. "PD costs 14 %" is true only at high concurrency

Feature-matched (both MTP off, both DPA off), against the MIX TP8 reference:

| conc | PD 1P1D | MIX TP8 | **PD / MIX** |
|---:|---:|---:|---:|
| 1 | 66.45 | 60.63 | **1.10** |
| 8 | 404.28 | 320.09 | **1.26** |
| 16 | 511.46 | 585.33 | 0.87 |
| 24 | 554.10 | 640.89 | 0.87 |

**PD is 10–26 % ahead at conc 1–8 and 13 % behind at 16–24.** The first version
of this result was a single conc-24 point and read as a flat "PD costs 14 %".

Before that, an unmatched comparison (PD *with* decode DP-attention vs MIX
*without*) said PD was **15 % ahead**. Separating them: **decode-side
DP-attention is worth 1.33× on the PD arm** — more than disaggregation costs —
which is why the unmatched number looked like a win. `734.92 → 554.10` on one
variable.

### 2. The HIP "fast path" is the slow path on this shape

Both arms same node, same image, one variable:

| conc | hip **ON** | hip **OFF** | ratio | TTFT p50 on → off |
|---:|---:|---:|---:|---:|
| 1 | 44.52 | 46.07 | 1.03 | 193 → 198 ms |
| 8 | 274.12 | **336.44** | **1.23** | 1977 → **589 ms** |

**Disabling the intra-node HIP transport is 23 % faster at conc 8 with 3.4×
better TTFT.** The kit's README described hip as the fast path
(*"GPU-to-GPU across XGMI, no NIC in the path"*) and loopback RDMA as the
degraded fallback. For this workload that is backwards.

**Read against a measured noise floor**, not an assumed one: the same
configuration on two nodes gave 1.01× and 1.05×, so ~5 %. The 23 % is 4–5× the
floor; the 3 % at conc 1 is inside it and is called noise.

This also answers **"capability, not choice"**: if `selectTransport` had not been
routing KV over hip, disabling it could not have changed anything.

## What it took: six silent wrapper defects

`bash cluster.singlenode.sh up` could not work as shipped. **One design error
with six faces** — the GLM-5.3 wrappers used single-knob names against a strictly
per-leg interface, so each knob silently did nothing and fell back to a plausible
default. **None errored. Two would have produced a believable benchmark number
against a configuration the wrapper file claims is different.** See `notes.md` §1
and `patches/`.

## Folder map

- `REPRODUCE.md` — ordered reproduction, including the verification that works
- `environment.md` — HW/SW, image digest, the mooncake tree-vs-binary trap
- `scripts/` — the wrapper and sweep script, verbatim
- `patches/` — the five commits that made the shape runnable, with rationale
- `results/` — every arm's CSV, plus the repetition arm's generations
- `notes.md` — the defects, the transport chain, and four instances of one trap

## Evidence is carried here, not pointed at

**This packup deliberately carries heavy evidence** (4.9 MB). Everything below was
archived before its deployment was torn down; a packup that points at a scratch
directory is a packup whose evidence nobody has promised to keep.

| `logs/` | compressed | what |
|---|---:|---|
| `pd_run1_artifacts.tar.gz` | 2.2 M | run-1 JSONL + leg logs + router log; the DPA-off conc-24 arm; the first failed bring-up |
| `n21_pd_dpa0_artifacts.tar.gz` | 814 K | DPA-off curve JSONL + leg logs |
| `n21_ab_hip_artifacts.tar.gz` | 588 K | **both sides of the hip A/B** — JSONL + leg logs |
| `n21_repetition_artifacts.tar.gz` | 664 K | repetition arm, raw and converted generations |
| `nccl_crash_20260902.tar.gz` | 436 K | the NCCL-watchdog death; **only record, and it nearly died with its container** |
| `deadknob_evidence.tar.gz` | 30 K | **the only artifact supporting the dead-name retraction** — leg logs with `MC_DISABLE_HIP_TRANSPORT=1` present and hip installed anyway |
| `console_logs.tar.gz` | 2.5 K | every bring-up and sweep console log |

```bash
tar xzf logs/n21_ab_hip_artifacts.tar.gz -C <dest>
```

Originals remain untouched at
`/apps/yihou/glm53.series.workspace_20260901/bigmodel/`.

## Cross-cutting analyses — `analysis/`

Copied in rather than referenced. Both are duplicated in the alignment packup; a
packup must be readable alone.

- `ttft_wave_stall.md` — `ttft_p99` here is **not a latency percentile**; it
  measures a discrete whole-wave stall (PD conc 8: requests 8-15, exactly the
  second wave, stalled ~17.9 s with a 0.10 s spread while nothing else exceeded
  3.9 s). Present on aggregated MIX too, so **not caused by disaggregation**, but
  PD's version is ~3× longer and ~19× tighter. Its 2026-09-03 update identifies
  **decode-side DP-attention** as the axis for the conc-1 tail — 211 ms p99 with
  it off against 10,197 ms with it on, while hip moved it 0.6 % — **and keeps the
  tension unresolved**: the MIX arm with the same signature had DPA *off*.
  Mechanism **not established** in either.
- `METHOD_the_check_that_lies.md` — the eleven-instance table of observables that
  return the same value regardless of the world. `notes.md` §3 lists this
  packup's four; that file is the full set.
