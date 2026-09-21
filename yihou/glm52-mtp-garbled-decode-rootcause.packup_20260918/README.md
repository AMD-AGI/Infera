# GLM-5.2 MTP garbled decode on P4D4 — root cause — 2026-09-18

A five-round A/B that found why GLM-5.2 MXFP4 produced **garbled decode output**
under EAGLE MTP in a 1-Prefill/1-Decode disaggregated deployment, and fixed it.

**Finding: custom all-reduce on the decode leg corrupts the speculative path.**
Passing `--disable-custom-all-reduce` to the decode engine restores correct text
and healthy MTP acceptance. Nothing else was needed.

> **Scope correction, added 2026-09-19.** This finding is **conditional on two
> things this pack-up did not vary**, and the headline above states it too
> broadly.
>
> 1. **All five rounds ran with `index_share_for_mtp_iteration=false`** —
>    verified after the fact from each round's own
>    `results/r0*/server-info/decode-0.json`, every one of which carries
>    `json_model_override_args='{"index_share_for_mtp_iteration":false}'`.
>    GLM-5.2's `config.json` defaults that flag to **true**, so this is a
>    non-default setting inherited from `config.sh:82`, not a neutral baseline.
>    **TP4 with IndexShare ON was never tested for correctness here.**
> 2. **All five rounds were TP4/DP4.** A parallel session running the same image
>    at **TP8/DP8** with IndexShare ON and custom all-reduce **ON** measured
>    coherent output over 16/16 temperature-0 probes at real
>    `spec_accept_length` **4.69** — i.e. the failure this pack-up documents did
>    not appear there.
>
> Two variables differ between those results, so neither refutes the other. The
> defensible statement is: *with IndexShare off, on TP4/DP4, custom all-reduce on
> the decode leg corrupts output, and disabling it fixes both the text and the
> acceptance rate.* Do not quote the unconditional form.
>
> Related and also worth knowing before reusing this kit: the same stack has a
> separate, unfixed **GPU memory access fault** in the DSA indexer path that
> appears only when IndexShare is ON — four faults in five runs, upstream
> #39517/#37648 open. See `yihou/dsa.topk.indexer.bug.analysis.md`.

Run on **crsuse2-m2m-135** (prefill) and **crsuse2-m2m-138** (decode),
MI355X ×4 each (devices 2,3,4,5), 2026-09-18 09:05–10:00 UTC.

## The symptom

With MTP on and simulated acceptance **off**, the first token was correct and
every later token degenerated. 16 identical requests, `temperature=0`:

```
1!!!!!!!!!!!
1engaalog8效专用curaereziTlutuga8
1reis0obuf Erect bufferreisangan legacyu
```

Measured MTP acceptance was `accept len 1.25 / accept rate 0.05` against a
healthy ~3.

## The result

| round | custom all-reduce | AITER fusion | NextN fusion fix | MTP | coherent | `spec_accept_length` per rank | verdict |
|---|---|---|---|---|---|---|---|
| R03 | on | on | in | EAGLE | **0/16** | 1.00 / 1.00 / **2.81** / 1.00 | FAIL |
| R04 | on | on | in | **off** | 16/16 | n/a | PASS |
| R05 | **off** | on | in | EAGLE | **8/8** | 2.89 / 3.05 / 2.87 / 2.86 | PASS |
| R06 | **off** | on | **out** | EAGLE | **8/8** | 2.80 / 2.81 / 2.55 / 2.63 | PASS |
| R07 | on | **off** | in | EAGLE | **0/16** | 2.51 / 3.65 / 1.00 / 1.00 | FAIL |

Machine-readable in `results/matrix.csv`. Custom all-reduce is the discriminator
in every row.

Acceptance criteria and results:

| criterion | result |
|---|---|
| `"What is 2+2? …"` does not degenerate | **PASS** — `1` / `1. ` / `1.  **Analyze the Request` |
| `"Reply with exactly: BASELINE_OK_<nonce>"` reproduces the nonce | **PASS** — exact |
| `spec_accept_length >= 2.0` | **PASS** — min 2.86 (R05), 2.55 (R06) |
| same-rail not regressed | **PASS** — 0 Mooncake failures, 0 router affinity 503s |

## What is established, and what is not

**Established** — a single-variable A/B, run twice with the fusion fix in and
out. On this stack (sglang `0.5.19.dev20260917+ga9fb1c3238` plus the repo's DSA
patch set, P4D4 with DP attention, EAGLE 5 steps / topk 1 / 6 draft tokens),
custom all-reduce on the decode leg corrupts output, and disabling it fixes both
the text and the acceptance rate.

**Established — custom all-reduce misbehaves on its own.** `engine.sh` also
passes `--enable-aiter-allreduce-fusion` unconditionally, so the first four
rounds could not tell "custom all-reduce is broken" from "the two interact".
R07 closes that: with the AITER fusion path **off** and custom all-reduce back
**on**, the failure reproduces exactly — 0/16, same two modes, same period-4
alternation, `spec_accept_length` min 1.0. The 2×2:

| custom all-reduce | AITER fusion | output |
|---|---|---|
| on | on | garbled (R03) |
| on | **off** | **garbled (R07)** |
| off | on | correct (R05, R06) |

The fourth cell is unnecessary: custom all-reduce off already passes with AITER
fusion on, and custom all-reduce on fails either way. So disabling the AITER
path is neither necessary nor sufficient, and **there is no narrower knob on
this axis** — `--disable-custom-all-reduce` is the fix.

**Not established — the mechanism.** We have the A/B, not the mechanism. A
plausible shape is the size-gated branching in `custom_all_reduce.py`
(`should_custom_ar`, `_MAX_CAR_SIZE`): the draft is a single-layer model and
verify runs at `num_draft_tokens=6`, so speculative decoding drives all-reduce at
message sizes the plain decode path never produces — which would explain R04
being clean with the same all-reduce enabled. **This is a hypothesis. Do not
write it up as the cause.**

**Unmeasured — the throughput cost.** Custom all-reduce is an optimisation and
turning it off gives it up. Performance re-runs were out of scope. **No number in
the sibling packup `glm52-1p1d-samerail-c32-c40.packup_20260918` describes this
configuration** — those were all taken with custom all-reduce ON *and* simulated
acceptance ON.

## The lead that was wrong, and why it is still in here

The investigation started from a colleague's finding that
`GlmMoeDsaForCausalLMNextN` never overrides `fused_shared_experts_architecture`,
so the MTP **draft** silently loses shared-experts fusion (fork commit
`4350d37c5b`). That defect is **real and present on this stack** — confirmed by
code read and by the log signature — and R03 fixed it, verified three ways.

It did not cure the bug. R06 then showed the cure works without it. So the fusion
defect is a genuine, independent defect worth carrying upstream, and **not** the
cause of this failure. The patch and its verification are kept in `patches/` and
`scripts/` because reproducing R03 and R06 needs both arms.

## Navigation

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction of all five rounds |
| `notes.md` | wrong turns, ruled-out hypotheses, open questions — **the most re-read file** |
| `environment.md` | hardware, fabric, image digests, git SHA, in-image versions |
| `scripts/` | all six configs, topology, probe, Dockerfile, applier, env collector — verbatim |
| `patches/` | the two patches, each with what / why / how / context |
| `results/` | `matrix.csv`, `flags-by-round.txt`, then per-round raw evidence |
| `spec/` | mission, full `working_process.md` debug log, node preflight |
| `env/` | raw `collect_env.sh` output per node |
| `logs/` | all five rounds, both legs, gzipped |

## Provenance

- Repo `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD
  `5a342acffe10e09729662ff40e81b22a4367fd75`, plus the uncommitted changes in
  `patches/`.
- Base image `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, digest
  `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32`.
- In-image sglang `0.5.19.dev20260917+ga9fb1c3238`.
- Original working directory, untouched and still on disk:
  `bench/glm5p2_pd/results/yihou-mtp-fusion-fix/` (gitignored).

## Credit

The differential comparison that made this tractable came from a parallel
session's kit at
`/home/yihou/dev/git/infera.yihou.mtp.debug/work/glm52-1p1d-mtp-correctness.packup_20260918/`
— a known-good P4D4 stack on base 20260916. `--disable-custom-all-reduce` was one
row in their delta table. See `notes.md` §5.
