# GLM-5.3 (big) — SGLang 1P1D

Prefill/decode-disaggregated deployment for **GLM-5.3** and **GLM-5.3-MXFP4**,
in two shapes: the usual **two-node** pair, and a **single-node** pair that
splits one 8-GPU box into TP4 prefill + TP4 decode.

## This kit does not fork the GLM-5.2 kit, and that is deliberate

GLM-5.3 (big) is `glm_moe_dsa` / `GlmMoeDsaForCausalLM`. Its `config.json` is
identical to GLM-5.2's field for field except `transformers_version` — same
hidden size, same layer count, same expert count, same attention. So the engine
recipe is not *similar* to the GLM-5.2 one, it **is** the GLM-5.2 one.

[`examples/sglang_1p1d_glm5.2/`](../sglang_1p1d_glm5.2/) already carries that
recipe in `engine/leg.sh`, validated end to end on two clusters and both RDMA
fabric types. Copying those ~600 lines here to change a model path would create
a second source of truth that drifts the first time either is fixed. So this kit
ships **wrappers only** and points `KIT_DIR` at the GLM-5.2 kit, exactly as that
kit's own `cluster/*.sh` files do.

If you find yourself editing an engine script to serve GLM-5.3, something is
wrong — say so, it is a bug in this arrangement.

**GLM-5.3-Flash is NOT covered here.** It is `glm5_next`, a different
architecture, and PD for it is a separate question this kit makes no claim
about. For Flash, see [`sglang_mix_glm5.3`](../sglang_mix_glm5.3/) (aggregated).

## Contents

| path | what |
|---|---|
| [`cluster.2node.sh`](cluster.2node.sh) | two-node pair — the validated shape. Fill in and run |
| [`cluster.singlenode.sh`](cluster.singlenode.sh) | one 8-GPU node split TP4 + TP4 |
| everything else | comes from [`../sglang_1p1d_glm5.2/`](../sglang_1p1d_glm5.2/) — `common.sh`, `engine/*.sh`, `preflight_rdma.sh` |

## Validation status

Stated plainly, because the honest answer is short.

| shape | status |
|---|---|
| the **deployment shape** (1P1D + mooncake + DPA + MTP + kvd + kv-aware) | validated for **GLM-5.2** on two clusters, both fabric types |
| **GLM-5.3 weights** through that shape, two-node | **not yet run.** Same architecture, so expected to work; expectation is not evidence |
| **single-node** TP4+TP4 | **not yet run, and it carries a real unknown** — see below |

## The single-node unknown: mooncake between two legs on one host

Both legs are on the same machine, so the KV handoff is a loopback RDMA
transfer. Whether mooncake does that, and at what speed, **is not established
here**. Do not assume it degrades gracefully — the failure mode this stack
specialises in is silent: mooncake falls back to a transport that works and is
merely 5-20x slower, and nothing raises.

What we *have* verified on the reference node, first-hand:

- 8 ionic RDMA devices on the host, all `PORT_ACTIVE`.
- `ib_peer_mem` **loaded**, so registration **mode A** (bare `ibv_reg_mr` +
  peer-mem: nothing pinned, KV pool not duplicated, every rail usable) is the
  mode to expect. That is the best of the three.
- Inside the engine image, `libionic 54.0-187-1` (ABI 4) and `ibv_devinfo`
  reporting **8 HCAs**.

**That last check has a trap worth knowing before you repeat it.** The container
must be started with `--device=/dev/infiniband`. Without it `ibv_devinfo` reports
zero HCAs from inside a container on a host that has eight — which is
indistinguishable from a libionic ABI mismatch, and is the same reading that
means "RDMA has silently degraded to TCP". `common.sh` passes it; an ad-hoc
`docker run` will not unless you remember.

Run `preflight_rdma.sh mode` before the first bring-up either way. Believe its
verdict from inside the container over the host's view: only the vendor provider
libraries the image ships can open a card.

## Quick start

```bash
# 1. which registration mode does this fabric support?
IMAGE=<infera-sglang-image> bash ../sglang_1p1d_glm5.2/preflight_rdma.sh mode

# 2. fill in ONE wrapper, then
bash cluster.singlenode.sh up      # or cluster.2node.sh
bash cluster.singlenode.sh smoke
bash cluster.singlenode.sh down
```

`smoke` is the GLM-5.2 kit's, and it checks each feature with a signal that goes
red when the feature is silently absent — including **`MC_FORCE_TCP` and
`GID is NULL` counts of 0 in both leg logs**, which is the check that catches a
pair that paired successfully and is moving KV over TCP.

## Notes carried over from the GLM-5.2 kit that still apply

These cost real debugging time there and the architecture has not changed:

1. **`--ep-size` and `--enable-dp-attention` are different axes.** Gate both on
   one condition and turning DPA off silently collapses the MoE from ep8 to the
   TP default, after which no latency delta is attributable to either.
2. **`--chunked-prefill-size` is a GLOBAL budget** that SGLang divides by
   `dp_size` only when DP-attention is on. One value serves both modes;
   hardcoding the per-rank number in a DPA-off branch cuts it 8x.
3. **Prefill activation OOM is fixed by LOWERING `--mem-fraction-static`**, the
   opposite of the decode-side fix. Diagnose by phase: decode retract → raise;
   prefill `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at *low* token usage → lower.
   Low token usage at the abort is the tell that it was never KV exhaustion.
4. **`SGLANG_OPT_USE_TOPK_V2=0` is mandatory on gfx950.** Without it the model
   serves, returns 200s, and returns garbage.
5. **MTP and decode-side radix cache are mutually exclusive upstream**, so
   `decode_prefix_len` is always 0 and every turn re-transfers the whole prompt
   KV. A prefill-side cache hit saves compute, not bytes — which is why fabric
   bandwidth matters on long-prompt agentic workloads even at a high hit rate.
6. **An MTP acceptance length of a steady 4.00 is bad news**, not a good result:
   the draft is predicting a repetition loop perfectly. 2-3 is healthy.

## Two GLM-5.3-specific things to decide before you run

**MTP.** The GLM-5.2 kit runs EAGLE MTP on the decode leg and that is its
validated configuration. For GLM-5.3 the evidence conflicts: upstream's GLM-5.3
cookbook says MTP/EAGLE is **disabled on AMD** because the gfx950 draft kernel is
unvalidated, while the OneNexus GLM-5.3-MXFP4 model card runs EAGLE at
`--speculative-num-steps 3` and lists it as validated. Both wrappers here
default `MTP=0`. That is a choice to avoid an unvalidated variable on the first
run, not a finding — resolve it deliberately rather than inheriting it.

**Shared-experts fusion.** Not a concern for the big MXFP4 checkpoint: its
shared experts are themselves MXFP4 (76 `.weight` / 75 `.weight_scale`, the odd
one being the BF16 MTP layer 78, which is not loaded while MTP is off), so the
precondition for the mismatch does not hold. It *is* a concern for Flash-MXFP4 —
see [`sglang_mix_glm5.3`](../sglang_mix_glm5.3/) and upstream issue #37268.
`glm4_moe.py`'s fusion gate only special-cases `w4afp8` and would fuse under
`quark`, so the wrappers pass `--disable-shared-experts-fusion` as insurance;
upstream #25261 shows this class failing *silently with wrong output* rather
than crashing when the shapes happen to line up.

## Source

[`examples/sglang_1p1d_glm5.3/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [the GLM-5.2 kit this drives](../sglang_1p1d_glm5.2/)
· [aggregated MIX kit for all four GLM-5.3 checkpoints](../sglang_mix_glm5.3/)
· [PD disaggregation concepts](../../manual/features/pd_disaggregation.md)
