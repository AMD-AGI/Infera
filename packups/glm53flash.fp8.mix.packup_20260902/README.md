# GLM-5.3-Flash **FP8** — SGLang MIX, TP4, on a borrowed node

Bring-up and correctness validation of the **original FP8** GLM-5.3-Flash
checkpoint (not MXFP4) as an aggregated single-worker deployment through the
infera stack, TP4 on **GPUs 4-7 of a node we do not own**.

Run 2026-09-02, 04:52-11:50 UTC, on
`smci355-ccs-aus-n05-29.prov.aus.ccs.cpe.ice.amd.com` (8×MI355X, gfx950).

## Result against the success criteria

| criterion | result | evidence |
|---|---|---|
| serves `glm5_next` FP8 at all | **PASS** — 62/62 shards, healthy in 820 s | `logs/mix_up_r0.log.gz` |
| exactly one worker, `disagg_mode: mixed` | **PASS** | `results/verify_round0.txt` §A |
| `/v1/models` = `glm5.3-flash` | **PASS** | `results/verify_round0.txt` §B |
| `17 * 23` → `391`, reasoning split from content | **PASS**, both rounds | `results/verify_round{0,1}.txt` §C |
| coherent open-ended answer, no repetition loop | **PASS**, both rounds | `results/verify_round{0,1}.txt` §D |
| **8 AITER mHC lines** (2/rank × 4 ranks) | **PASS** — 4 + 4, both rounds | `results/verify_round{0,1}.txt` §E |
| decode lines carry `full token usage` **and** `mamba usage` | **PASS** — 10/10, both rounds | `results/verify_round{0,1}.txt` §F |
| fault scan `memory access fault\|HIP error\|Traceback` | **PASS** — 0 raw, 0 excluded | `results/verify_round{0,1}.txt` §H |
| decode CUDA graphs on, re-verified | **PASS** — round 1, capture 33.47 s | `results/verify_round1.txt` |
| throughput measured | **not by this operator, deliberately** — the lead's two-point sweep is carried and attributed | `results/sweep_f8_by_lead.txt`, `notes.md` §4 |

Two rounds were run: **round 0** decode CUDA graphs off, **round 1** graphs on.
Everything above passed in both.

## The headline: this is the CONTROL ARM for the MXFP4 fusion root cause

The most valuable thing here is not that FP8 serves. It is **what it proves
about the MXFP4 failure documented in
`../glm53flash.mxfp4.mix.packup_20260902/results/root_cause.md`.**

Flash-MXFP4 fails to load with shared-experts fusion enabled and loads with it
disabled. On its own that is equally consistent with *"the gfx950 fusion path is
broken"* — which would make the one-line quantization guard the wrong fix and a
revert the right one.

This run settles it. **Same `glm5_next` architecture, same `c821c425` image,
same TP4 topology, fusion left ENABLED — loads clean and answers correctly.**

```
'disable_shared_experts_fusion': False              # resolved server_args
[2026-09-02 05:07:28 TP0] Shared experts fusion optimization enabled.
```

So the gfx950 fusion path itself **works**. What sglang PR #36607 shipped
unguarded is the *decision* to use it, and that decision is wrong only on a
mixed-precision checkpoint. The one-line
`quant_blocks_shared_experts_fusion(quant_config)` guard is therefore the
correct upstream fix, and it **preserves this FP8 case as fused-and-profitable**
rather than disabling a working optimization.

**The outcome was predicted before launch from the checkpoint's own index**, not
discovered by trying it: `model.safetensors.index.json` carries **129
`mlp.shared_experts.*.weight` against 129 matching `.weight_scale_inv`** — a
strict 1:1 pairing, i.e. uniformly block-FP8. The MXFP4 precondition (shared
experts at a *higher* precision than the routed experts) is simply absent.

### Ruling out silent corruption specifically

"It loaded" is not "it is correct" — upstream #25261 is the case where this
class of mismatch produced **silently wrong output** instead of crashing. So the
correctness bar here is deliberately higher than `391`. Round 1, verbatim:

> Sunlight contains all colors of light, and as it passes through the
> atmosphere, shorter wavelengths like blue are scattered in all directions by
> gas molecules far more strongly than longer wavelengths like red—a phenomenon
> called Rayleigh scattering. This scattered blue light reaches our eyes from
> every direction we look, making the whole sky appear blue (**violet actually
> scatters even more, but the sun emits less of it and our eyes are less
> sensitive to it**).

The parenthetical is a correct, non-trivial physical refinement. A degenerate or
corrupted decoder does not produce it. That sentence — not the arithmetic — is
what rules out the #25261 mode.

## Second finding: 8/8 mHC lines on a second checkpoint

The `c821c425` pin was chosen over Infera PR #143's `9e692c92` precisely because
the earlier commit lacks the AITER mHC enablement (`77a46694`), whose absence
costs 4.3-5.4× with **nothing in any log saying so**. Until this run that was
verified on MXFP4 only. It now holds on **FP8, on a third node**:

```
[05:18:52 TP0..TP3] Using AITER gfx950 mHC pre/post kernels          x4
[05:19:17 TP0..TP3] Using fused AITER mHC attention-to-FFN boundary  x4
```

## Third finding: HIP IPC works across disjoint `HIP_VISIBLE_DEVICES`

Run at the end of this session on the same image, and it unblocks single-node
PD. Full detail and caveats: **`results/ipc_probe.md`**. Short form: a process
that **cannot see** the exporter's physical GPU imported its handle and read
back the correct byte pattern, **including across two separate containers**,
which is the shape PD actually runs. So `cluster.singlenode.sh`'s disjoint
device split does **not** need changing.

## This operator published no throughput number, and that is a result

A two-point sweep does exist here — **measured by the team lead** on this
deployment, and carried in `results/sweep_f8_by_lead.txt` /
`results/fixlen_f8_by_lead.csv` under their name. It gives the first direct
FP8-vs-MXFP4 comparison at fixed topology: **MXFP4 is 1.11× faster at
concurrency 1 and 1.23× at concurrency 8**. Its limits — contention not sampled
during either arm, different nodes, single run, cache confound, greedy-verified
sampling — are in `results/README.md`.

The attribution split is deliberate and is kept at the lead's instruction. The
reason no number of *this operator's* appears: the node oscillates. Measured over **22
polls at 7-minute cadence**, the neighbour's footprint cycled between 0 and
~30 GiB per card on a **7-15 minute period**; three fully-idle windows were
observed, the longest **~14 minutes**. No window on this node is long enough to
hold a sweep worth quoting, and the failure would be **silent** — a straddled
transition does not appear in the numbers. `notes.md` §5 records the method.

The single-stream figures that do appear anywhere here — **~12-15 tok/s graphs
off, ~110 tok/s graphs on** — are cited **only as evidence that the CUDA-graph
path engaged** (a 7.5× step, matching the TP8 packup's 15.3 → 106.85). They are
explicitly **not** a performance result.

## Scope — what this does NOT cover

- **PD for Flash**: not run. Only the IPC precondition was probed.
- **Multimodal**: not exercised. Text only, as upstream's own validation is.
- **GSM8K or any accuracy benchmark**: not run. Correctness here is smoke-level
  plus the anti-corruption check above.
- **MTP / speculative decoding**: off, deliberately — unvalidated for this model
  on ROCm.
- **Any throughput claim by this packup's operator**: none. The two-point sweep
  that is here is the lead's, attributed to them, with its limits stated.
- **Repeats or variance on any performance number**: none taken.

## Folder map

| path | what |
|---|---|
| `README.md` | this file |
| `REPRODUCE.md` | ordered, copy-pasteable, cold-start reproduction |
| `environment.md` | node, GPUs, driver, image digests, pinned SHAs, external paths, secrets by name, gaps |
| `notes.md` | **corrections, dead ends, unknowns, and the operational findings** — the most re-read file |
| `scripts/` | the three scripts that ran, verbatim, + what not to trust |
| `results/` | verify transcripts, the IPC probe, and the lead's two-point sweep |
| `logs/` | gzipped worker/router/build/bring-up logs + a map of what each proves |
| `PLAN.md`, `spec.mission.md` | the plan and the originating spec, copied in |

There is no `patches/` directory: **this run needed no code fix.** The image is
the repo's `Dockerfile.sglang.glm53` unmodified. Omitting the directory is
deliberate rather than an oversight.
