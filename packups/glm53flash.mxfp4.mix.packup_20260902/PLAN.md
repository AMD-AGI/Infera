# Plan — GLM-5.3 series integration into infera

Written 2026-09-01, after the research phase and after three failed bring-up
rounds. This is the step that was skipped the first time; the ladder in §3 exists
because skipping it produced three rounds that each moved more than one variable.

## 0. Where we actually are

**Done and verified first-hand**

| item | evidence |
|---|---|
| Image `infera/engine-sglang:glm53-c821c425` builds | overlay HEAD `c821c425`, `quark.py` 1172 lines, `glm5_next.py` 1942 lines, all checked inside the image |
| `Dockerfile.sglang` base v0.5.17 → v0.5.18 | DSA set applies with one anchor port; full apply + bytecode verification passes in-image |
| `reset_gpus.sh` no longer kills foreign processes | rewritten; ownership resolved per PID from `/proc/<pid>/cgroup` |
| etcd port collision | 2379 held by a **host** etcd that is not ours → ours moved to 12379 |
| broken symlink farm | `/apps/data/models` is its own mount and does not propagate when binding `/apps`; needs its own `-v` |

**Blocked on**

MoE weight load, `_load_w2`: `size of tensor a (256) must match tensor b (512)`.
256 = 512/2 = the MXFP4-packed width, so a packed buffer is being fed an
unpacked BF16 tensor.

**A hypothesis that was tested and did NOT hold.** `config.json`'s `exclude`
omits the routed experts of layers 3 and 5, while
`mixed_precision_correction.json` records `bf16_expert_layers: [3, 5, 6]`. That
inconsistency is real (verified against the tensor index: those layers' experts
carry no `weight_scale`). But adding the missing entries changed **nothing** —
the same error at the same shard. So the metadata gap is not, by itself, the
cause. Most likely reason it cannot be: `FusedMoE` is one module for all 288
experts with prefix `...mlp.experts`, while every exclude entry is deeper
(`...experts.0.down_proj`), so the exclusion cannot match at the granularity the
loader asks at. **Unverified** — the probe that would settle it is in §4.

**The framing that was missed.** Round 1 was already the vendor recipe verbatim
(original model dir, `--quantization quark`, `--moe-runner-backend aiter`, TP4,
graphs off, `SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`, the same ctx/gmu/chunk values)
and it failed identically. So the remaining deltas against the vendor are only:

1. **base image** — vendor validated on `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822`
   (**rocm724**); we used `lmsysorg/sglang:v0.5.18-rocm720-mi35x` (**rocm720**),
   which mission rule 4 names. The MXFP4 MoE kernels come from aiter, and the two
   images need not carry the same aiter.
2. **infera wrapper** — vendor runs bare `sglang.launch_server`; we run through
   `python3 -m infera.engine.sglang`.
3. **weights** — our local copy vs the HF repo the vendor pointed at.

Config surgery was the wrong next move while three untested deltas remained.

## 1. Objective

`mission.md`. Deliverables: support the GLM-5.3 cluster; update the infera sglang
image + patches; e2e tests **disabled by default in CI**; `examples/` for both
families, original and MXFP4. **Priority: GLM-5.3-Flash-MXFP4 first.**

## 2. Two independent tracks

**Track F — Flash family (`glm5_next`).** Needs the build-time source overlay.
This is where the current blocker is, and it owns the GPUs.

**Track B — big family (`glm_moe_dsa`).** GLM-5.3 and GLM-5.3-MXFP4 are
field-for-field identical to GLM-5.2 except `transformers_version`; stock sglang
already serves them via `glm4_moe.py`, and the GLM-5.3-MXFP4 card states it is a
drop-in on the stock recipe with no overlay. So Track B is mostly *reuse the
GLM-5.2 recipe and prove it*, and it needs no engine work. It can be prepared
while Track F is blocked, and run on GPUs 4-7 while Track F holds 0-3.

## 3. Track F: the incremental ladder — ONE variable per rung

Start from the configuration most likely to work (the vendor's), then walk each
delta back toward the one we must ship. Stop at the first rung that fails; that
rung names the cause.

| rung | image | engine source | launcher | model dir | isolates |
|---|---|---|---|---|---|
| **0** | vendor `sglang-rocm:v0.5.18-rocm724-mi35x-20260822` | bind-mount `c821c425` checkout | bare `launch_server` | original | is the vendor recipe reproducible here at all |
| **1** | `sglang:v0.5.18-rocm720-mi35x` | bind-mount, same | bare | original | **the base image / aiter** |
| **2** | our `glm53-c821c425` | baked overlay | bare | original | the baked overlay vs bind-mount |
| **3** | our image | baked | `infera.engine.sglang` | original | **infera's wrapper** |
| **4** | our image | baked | infera + etcd + kv-aware router | original | the MIX topology |
| **5** | our image | baked | full MIX | original | decode CUDA graphs on |

Rung 0 first, and no rung is skipped on the way back down. If rung 0 fails, the
delta is the weights or the host, and §4 applies instead.

TP4 throughout (what the vendor validated), GPUs 0-3, so a Track-B arm can use
4-7.

## 4. Cheap probes to run alongside, not instead

Each is minutes and needs no full bring-up. Run them concurrently with the
ladder, not in place of it.

- **P1 — exclusion granularity.** Call `should_ignore_layer` with the real
  `FusedMoE` prefix (`model.layers.3.mlp.experts`) against the checkpoint's
  exclude list, with and without `fused_mapping`. Settles whether per-expert
  entries can ever match. *This is the probe that was interrupted.*
- **P2 — which layer fails.** Map shards 1-4 of 120 to layer indices from the
  weight map. Confirms whether the failure is on layer 3/5 (the BF16-expert
  layers) or somewhere else entirely.
- **P3 — aiter delta.** Compare the aiter revision and the MXFP4 MoE kernel
  presence between the rocm720 and rocm724 images. Predicts rung 1's outcome.
- **P4 — upstream.** Search sglang issues/PRs for this exact traceback
  (`_load_w2`, quark, glm5_next, MXFP4). Mission rule: research the bug before
  debugging it.
- **P5 — weights identity.** Compare our local snapshot against the HF repo's
  file list and sizes. Rules the weights in or out as a delta.

## 5. Track B steps

1. Bring up GLM-5.3 (fp8) MIX on the existing `Dockerfile.sglang` image, reusing
   the GLM-5.2 recipe. Expect it to work unchanged.
2. Same for GLM-5.3-MXFP4, following the stock InferenceX recipe from its card —
   but **not** its published `--cuda-graph-max-bs 2 --max-running-requests 2`,
   which caps concurrency at 2 and is not a throughput configuration.
3. fixlen sweep against the GLM-5.2 baselines in
   `infera.glm52.mix.experiment/fixlen.glm52.mix.packup_20260806/`.
4. Optimus-AgenticBench Case A `.fix.yaml`.
5. 1P1D for GLM-5.3-MXFP4, reusing `examples/sglang_1p1d_glm5.2/`.

## 6. Repo deliverables (after the tracks produce evidence)

- `examples/` for Flash and big, original and MXFP4, PD and MIX as the mission
  lists them.
- One `enable=False` row per model in `tests/e2e/pd_mixed/sglang/matrix.py` (and
  `pd_disag` where applicable) — that flag *is* "disabled by default in CI".
- Keep `deploy/docker/Dockerfile.sglang.glm53` in step with what actually ran,
  including the pin rationale already written into its header.

## 7. Team

Leader polls every 10 minutes. First time a teammate looks off-track or
under-informed, **record only**; intervene on the second consecutive poll if it
has not self-corrected.

- **leader (me)** — the GPU ladder in §3. GPUs are one resource; the ladder is
  sequential and stays with one owner.
- **flash-debug-research** — P1/P3/P4 above. Code reads and upstream search, no
  GPU.
- **bigmodel-track** — §5 steps 1-2 preparation: scripts and recipe, ready to run
  on GPUs 4-7.

## 8. Rules carried in

Work in English, report in Chinese. Everything temporary stays in this workspace.
Containers, not the host. Never kill a process that is not ours — this node
carries another user's multi-day job and a foreign etcd. Before an expensive e2e
run, validate with a cheap probe; design each run to test several independent
hypotheses. Suspend rather than conclude: state what was measured, and name the
measurement that would answer what was not.
