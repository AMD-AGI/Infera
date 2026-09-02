# GLM-5.3 series — SGLang MIX (aggregated) on MI355X

Runnable deployment kit for the **four GLM-5.3 checkpoints** served the infera
way on a single 8×MI355X (gfx950) node: one aggregated worker, prefix caching
on, fronted by the **infera kv-aware router**. No PD, no RDMA, no second node.

One file is site-specific. Fill it in and the deployment is three commands.

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # container -> etcd -> worker -> router
bash engine/smoke.sh           # prove each feature is actually live
```

## Read this first: GLM-5.3 is two unrelated architectures

They share a product name and almost nothing else. Which one you are serving
decides **which engine image you need**, and getting it wrong fails at config
load rather than at inference.

| | `GLM-5.3`, `GLM-5.3-MXFP4` | `GLM-5.3-Flash`, `GLM-5.3-Flash-MXFP4` |
|---|---|---|
| `model_type` | `glm_moe_dsa` | `glm5_next` |
| hidden / layers | 6144 / 78 | 4096 / 45 |
| routed experts | 256 | 288 |
| attention | uniform MLA + DSA | **hybrid**: KDA linear attention + DSA |
| also carries | — | mHC, no-RoPE MLA, compressed indexer k-pool, a vision encoder |
| memory pools | paged KV only | paged KV **plus a KDA state pool** |
| **image** | `Dockerfile.sglang` | **`Dockerfile.sglang.glm53`** |

**The big pair is GLM-5.2 with different weights.** Their `config.json` is
identical to GLM-5.2's field for field except `transformers_version`, so the
released engine already serves them through `glm4_moe.py`.

**The Flash pair exists in no released sglang.** `glm5_next.py` is absent from
`lmsysorg/sglang:v0.5.18-rocm720-mi35x` and from the vendor-validated
`lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260822`. `Dockerfile.sglang.glm53`
brings the engine source in at build time, pinned to a SHA. Point a Flash
variant at the ordinary image and you get:

```
ValueError: The checkpoint you are trying to load has model type `glm5_next`
but Transformers does not recognize this architecture.
```

That message names `transformers`, which invites the wrong fix. Upgrading
transformers does not help — the branch registers its own `Glm5NextConfig` and
pins the same version. The missing component is sglang.

`glm5_next` is the **model body**, not an MTP head: `glm5_next.py` exports
`EntryClass = [Glm5NextForConditionalGeneration]` and MTP lives separately in
`glm5_next_nextn.py`. Turning speculative decoding off does not remove the need
for the overlay.

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — variant, IP, weights, image, shape, ports |
| `engine/worker.sh` | the real launcher; carries the tuned recipe for all four variants, no site values |
| `engine/up.sh` | container → etcd → worker → router, waiting on health at each step |
| `engine/smoke.sh` | six blocks, each red when a specific feature is *silently* absent |
| `engine/bench.sh` | reference fixed-length sweep via sglang's own `bench_serving` |
| `engine/down.sh` | tear down, then **wait** for VRAM to actually drain |

## Validation status

Stated plainly rather than implied. All on 8×MI355X, gfx950, ROCm 7.2, driver
6.14.14, TP4, decode CUDA graphs on unless noted.

| variant | status |
|---|---|
| `flash-mxfp4` | **validated end to end** — full infera stack, reproduced on two separate nodes, plus a fixed-length sweep |
| `big-fp8` | **validated** — all smoke blocks green, `max_total_num_tokens=1148288` |
| `big-mxfp4` | **validated, with numbers** — all smoke blocks green; AITER FP4 path confirmed dispatching (`torch.float4_e2m1fn_x2`, `per_1x32`) rather than dequantising to BF16; TP8+DPA+MTP fixed-length sweep lands at **0.92 / 1.06 / 0.89 / 1.10 ×** the GLM-5.2 MIX baseline at concurrency 1/8/16/24 |
| `flash-fp8` | **loads and serves correctly** — brought up on a third node, 62/62 shards, coherent answers, 8/8 AITER mHC lines. Throughput not yet measured in this kit |
| PD (1P1D) for any variant | **not covered by this kit.** For the big pair the shape is the same as [`sglang_1p1d_glm5.2`](../sglang_1p1d_glm5.2/), which is validated for GLM-5.2 |

## The one flag you must not drop: `--disable-shared-experts-fusion`

On `flash-mxfp4` this is load-bearing, not tuning. sglang PR #36607 opened the
gfx950 branch of `glm5_next`'s shared-experts fusion gate
(`glm5_next.py:1414`) **without** carrying the
`quant_blocks_shared_experts_fusion(quant_config)` guard that
`deepseek_v2.py:3069` has. `QuarkConfig.can_fuse_shared_expert()` computes the
correct answer and is never consulted. The checkpoint keeps its shared experts
in BF16 while its routed experts are MXFP4, so the shared expert gets renamed
into routed slot 288 of a packed FusedMoE and weight load dies:

```
RuntimeError: The size of tensor a (256) must match the size of tensor b (512)
at non-singleton dimension 1        # fused_moe_triton/layer.py::_load_w2
```

256 is the MXFP4-packed width; 512 is the same tensor unpacked and TP4-sharded.
Upstream issue **#37268** is the identical failure on NVFP4/NVIDIA with the same
accepted workaround.

Two consequences worth carrying:

- **The health signal is a line that must be ABSENT.** Grep the worker log for
  `Shared experts fusion optimization enabled.` — present means broken.
- **On `flash-fp8` the flag is NOT needed, and this is measured rather than
  assumed.** That checkpoint was brought up on a third node with fusion left
  **enabled** — `disable_shared_experts_fusion: False` in the resolved args and
  `[TP0] Shared experts fusion optimization enabled.` in the log — and it loaded
  all 62 shards with no `_load_w2` mismatch and answered correctly. The reason it
  is safe was predicted before launch from the checkpoint's own index: **129
  `.weight` / 129 `.weight_scale_inv`**, a strict 1:1 pairing, i.e. uniformly
  block-FP8. The MXFP4 precondition — shared experts at a *higher* precision than
  the routed experts — is simply absent, so `quant_blocks_shared_experts_fusion`
  returns False and fusion is legitimate.

  This is the control arm that makes the story a quantization mismatch rather
  than "fusion is broken on gfx950". **The gfx950 fusion path itself works.** What
  #36607 shipped unguarded is the *decision* to use it, and that decision is only
  wrong when the checkpoint is mixed-precision — which is why the one-line guard
  is the right upstream fix rather than reverting the feature.

- **On `big-mxfp4` the same flag is insurance, not a fix.** That checkpoint's
  shared experts are themselves MXFP4, so the precondition is absent. It stays
  on by default because upstream #25261 shows this class of mismatch failing
  *silently with wrong output* rather than crashing when the shapes happen to
  line up. Set `SHARED_EXPERT_FUSION=1` for a clean single-variable perf round.

## Notes and gotchas

**1. The AITER mHC lines are the real health check, not the absence of errors.**
On the Flash variants, grep the worker log for two `AITER gfx950 mHC` lines per
rank. Without them the server still starts, still answers correctly, and is
**4.3–5.4× slower**, with nothing in any log calling it out. They are gated on
HIP + gfx95 + `SGLANG_USE_AITER=1`; miss the env and you silently get the slow
path. `smoke.sh` block 5 counts them.

**2. The Flash family keeps TWO memory pools.** Decode lines must show
`full token usage` *and* `mamba usage`. If `max_running_requests` is clamped at
startup, suspect the KDA state pool first: raise `--mamba-full-memory-ratio`
(default 0.9) or pin `--max-mamba-cache-size`. Do **not** override
`linear_lower_bound` through `--json-model-override-args`.

**3. DSA flags are `--dsa-*`, not GLM-5.2's `--nsa-*`.** Both spellings exist in
v0.5.18; the `--nsa-*` ones are not what this model wants.

**4. The DSA-on-ROCm env block is mandatory for the big pair.** Without it the
model serves, returns 200s, and returns garbage — the sparse-attention indexer
takes a path not ported to gfx950. `worker.sh` sets it; `infera.engine.sglang`
also defaults `SGLANG_OPT_USE_TOPK_V2` off on ROCm.

**5. Do not copy the vendor card's `--cuda-graph-max-bs 2
--max-running-requests 2`.** Those appear in the published GLM-5.3-MXFP4 recipe
and cap the server at two concurrent requests. That is an accuracy
configuration, not a throughput one.

**6. Benchmark with `bench_serving`, not a shell loop.** A bash fan-out of
concurrent `curl`s becomes the bottleneck before the engine does — at
concurrency 32 one measured 350 output tok/s while the engine's own log reported
2398 tok/s with an empty queue.

**7. Resolve the weights symlink yourself.** Where the models path crosses an
NFS mount boundary, bind-mounting the symlink's parent gives the container an
empty directory. The failure appears minutes later as `Unrecognized processing
class`, because `config.json` is the one file that still resolves. `up.sh` binds
`realpath` output.

**8. MTP/EAGLE is off for both families, and that is unresolved rather than
decided.** Upstream's GLM-5.3 cookbook disables speculative decoding on AMD
because the gfx950 draft kernel is unvalidated; the OneNexus big-MXFP4 card runs
EAGLE at `--speculative-num-steps 3`. Both statements are recorded here; neither
has been tested in this kit.

**9. `Ctrl-C` on a log tail does not stop anything.** Use `engine/down.sh`, and
check `docker ps`.

## Reference numbers

`flash-mxfp4`, TP4 (four GPUs), decode graphs on, `bench_serving` at
ISL 7400 / OSL 320:

| concurrency | output tok/s | TTFT p50 |
|---:|---:|---:|
| 1 | 111.0 | 255 ms |
| 8 | 561.0 | 1065 ms |
| 16 | 962.6 | 744 ms |
| 24 | 1391.2 | 619 ms |

These are a reference point on one configuration, not a claim about the family.
In particular they are **not** comparable to GLM-5.2 numbers taken at TP8 with
DP-attention, MTP and kvd enabled: different model, different parallelism,
different feature set.

## Source

[`examples/sglang_mix_glm5.3/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [GLM-5.2 1P1D kit](../sglang_1p1d_glm5.2/) · [PD disaggregation concepts](../../manual/features/pd_disaggregation.md)
