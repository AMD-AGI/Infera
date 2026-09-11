# GLM-5.3 series — SGLang MIX (aggregated) on MI355X

Runnable deployment kit for **`GLM-5.3` and `GLM-5.3-MXFP4`** served the infera
way on a single 8×MI355X (gfx950) node: one aggregated worker, prefix caching
on, fronted by the **infera kv-aware router**. No PD, no RDMA, no second node.

One file is site-specific. Fill it in and the deployment is three commands.

```bash
$EDITOR env.sh                 # the only file you edit
bash engine/up.sh              # container -> etcd -> worker -> router
bash engine/smoke.sh           # prove each feature is actually live
```

## Read this first: the big pair is GLM-5.2 with different weights

`GLM-5.3` and `GLM-5.3-MXFP4` carry `model_type: glm_moe_dsa`, and their
`config.json` is identical to GLM-5.2's field for field except
`transformers_version`. The released engine therefore already serves them through
`glm4_moe.py`, and the ordinary `Dockerfile.sglang` image is the only one this kit
needs.

| | `GLM-5.3`, `GLM-5.3-MXFP4` |
|---|---|
| `model_type` | `glm_moe_dsa` |
| hidden / layers | 6144 / 78 |
| routed experts | 256 |
| attention | uniform MLA + DSA |
| memory pools | paged KV only |
| image | `Dockerfile.sglang` |

## Contents

| path | what |
|---|---|
| [`env.sh`](env.sh) | **the only file you edit** — variant, IP, weights, image, shape, ports |
| `engine/worker.sh` | the real launcher; carries the tuned recipe for both big variants, no site values |
| `engine/up.sh` | container → etcd → worker → router, waiting on health at each step |
| `engine/smoke.sh` | six blocks, each red when a specific feature is *silently* absent |
| `engine/bench.sh` | reference fixed-length sweep via sglang's own `bench_serving` |
| `engine/down.sh` | tear down, then **wait** for VRAM to actually drain |

## Validation status

Stated plainly rather than implied. All on 8×MI355X, gfx950, ROCm 7.2, driver
6.14.14, TP4, decode CUDA graphs on unless noted.

| variant | status |
|---|---|
| `big-fp8` | **validated** — all smoke blocks green, `max_total_num_tokens=1148288` |
| `big-mxfp4` | **validated, with numbers** — all smoke blocks green; AITER FP4 path confirmed dispatching (`torch.float4_e2m1fn_x2`, `per_1x32`) rather than dequantising to BF16; TP8+DPA+MTP fixed-length sweep lands at **0.92 / 1.06 / 0.89 / 1.10 ×** the GLM-5.2 MIX baseline at concurrency 1/8/16/24 |
| PD (1P1D) | **not covered by this kit.** The shape is the same as [`sglang_1p1d_glm5.2`](../sglang_1p1d_glm5.2/); [`sglang_1p1d_glm5.3`](../sglang_1p1d_glm5.3/) wraps it for these checkpoints |

## `--disable-shared-experts-fusion` here is insurance, not a fix

On `big-mxfp4` the checkpoint's shared experts are **themselves MXFP4**, so the
precondition for a mis-load is absent: `glm4_moe.py`'s fusion gate only
special-cases `w4afp8` and would fuse under `quark`, but there is no
higher-precision shared expert to rename into a packed routed slot.

It is on by default anyway, because upstream **#25261** shows this class of
mismatch failing *silently with wrong output* rather than crashing when the
shapes happen to line up. Set `SHARED_EXPERT_FUSION=1` for a clean
single-variable performance round.

The health signal is a line that must be **ABSENT** from the worker log:
`Shared experts fusion optimization enabled.`

## Notes and gotchas

**1. A startup line that reads like a clamp is a division.** Under DP-attention
the engine prints **per-rank** values while `/get_server_info` reports the
global ones. Ask for `--max-running-requests 256 --chunked-prefill-size 65536`
at dp8 and the startup line says 32 and 8192 — that is 256/8 and 65536/8. Never
compare a requested value against a resolved one; the GLM-5.2 baseline's log
shows both numbers too, so this is not new to GLM-5.3.

**2. DSA flags are `--dsa-*`, not GLM-5.2's `--nsa-*`.** Both spellings exist in
v0.5.18; the `--nsa-*` ones are not what this model wants.

**3. The DSA-on-ROCm env block is mandatory.** Without it the
model serves, returns 200s, and returns garbage — the sparse-attention indexer
takes a path not ported to gfx950. `worker.sh` sets it; `infera.engine.sglang`
also defaults `SGLANG_OPT_USE_TOPK_V2` off on ROCm.

**4. Do not copy the vendor card's `--cuda-graph-max-bs 2
--max-running-requests 2`.** Those appear in the published GLM-5.3-MXFP4 recipe
and cap the server at two concurrent requests. That is an accuracy
configuration, not a throughput one.

**5. Benchmark with `bench_serving`, not a shell loop.** A bash fan-out of
concurrent `curl`s becomes the bottleneck before the engine does — at
concurrency 32 one measured 350 output tok/s while the engine's own log reported
2398 tok/s with an empty queue.

**6. Resolve the weights symlink yourself.** Where the models path crosses an
NFS mount boundary, bind-mounting the symlink's parent gives the container an
empty directory. The failure appears minutes later as `Unrecognized processing
class`, because `config.json` is the one file that still resolves. `up.sh` binds
`realpath` output.

**7. MTP/EAGLE is off here, but the question it used to raise is settled.** This
note previously recorded a contradiction — upstream's GLM-5.3 cookbook disables
speculative decoding on AMD, the OneNexus big-MXFP4 card runs EAGLE at three
steps — as unresolved. It is resolved for the **PD** shape: EAGLE MTP(3,1,4) is
validated on this checkpoint in
[`../sglang_1p1d_glm5.3/`](../sglang_1p1d_glm5.3/), so the cookbook's AMD line
does not govern this architecture. What remains untested is MTP in *this*
kit — an aggregated, non-PD shape — so it stays off here for want of a
measurement, not for want of a decision.

**8. `Ctrl-C` on a log tail does not stop anything.** Use `engine/down.sh`, and
check `docker ps`.

## Reference numbers

`big-mxfp4`, measured with `bench_serving` at fixed lengths against the GLM-5.2
MIX baseline. Output tok/s, p50 of the sweep:

| concurrency | 1 | 8 | 16 | 24 |
|---|---:|---:|---:|---:|
| TP8, DP-attention, EAGLE MTP | 76.26 | 417.63 | 606.33 | 825.07 |
| GLM-5.2 MIX baseline | 82.56 | 395.58 | 679.82 | 746.75 |
| ratio | 0.92 | **1.06** | 0.89 | **1.10** |

At p90 the same arms give 0.99 / 1.03 / 0.98 / 1.01 — parity. The overall band
is **0.89–1.11×**.

**The parallelism is part of the result, not a detail.** The same checkpoint at
TP4 with DP-attention and MTP off reaches only 0.58–0.83× of the same baseline.
Comparing a TP4 arm against a TP8+DPA+MTP baseline measures the configuration,
not the model.

These are a reference point on one configuration, not a claim about the family.

## What this kit exposes on the network, and why you should care on a shared node

These scripts run with `--network=host` and bind on **all interfaces**. That is
deliberate for a single-tenant benchmark box and is **wrong for a shared one**,
where other users can be logged into the same node and claim its GPUs without
notice.

| what | where | exposed as |
|---|---|---|
| **etcd**, unauthenticated | `up.sh` | `0.0.0.0:$ETCD_PORT` and peer port |
| router | `up.sh` | `0.0.0.0:$ROUTER_PORT` |
| KV event / snapshot sockets | `worker.sh` | `tcp://0.0.0.0:$KV_PUB_PORT` |

**etcd is the one that matters.** It holds worker discovery, it has no auth, and
anyone who can reach the port can mutate the keys that tell the router where to
send traffic. The engine and KV ports are lower stakes but still raw.

The scripts are left as they were validated rather than quietly re-bound, because
changing a bind changes a recipe that was measured. **If you run this on a shared
or reachable host, change them yourself:** the router and workers are colocated,
so `127.0.0.1` works for etcd and the KV sockets, and only the router port needs
to be reachable — put something that authenticates in front of it.

**`--trust-remote-code` is passed.** The risk it carries is in the checkpoint
directory, not in the flag: where the weights live on a shared NFS mount,
**anyone who can write there can execute code in your container.** If that matters
to you, copy the checkpoint to a directory you own and verify permissions before
launching.

## Source

[`examples/sglang_mix_glm5.3/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [GLM-5.2 1P1D kit](../sglang_1p1d_glm5.2/) · [PD disaggregation concepts](../../manual/features/pd_disaggregation.md)
