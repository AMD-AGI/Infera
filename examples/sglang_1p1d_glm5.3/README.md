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

## The single-node path: HIP IPC over XGMI, not loopback RDMA

**An earlier version of this section said the same-host KV handoff is a loopback
RDMA transfer and that the risk is silent slowness. Both were wrong**, and the
correction changes what you check. Established by reading the mooncake tree and
build cache inside the shipped image:

The pinned mooncake commit is `01d1eb2a` (2026-07-01), *"[TE] Support rdma+hip
multi-protocol segments for single-node disaggregation (#2682)"* — literally the
single-node disaggregation commit, whose own message reports validation of
single-node 1P1D on MI355X over the rdma+hip path.

The image builds with `USE_HIP=ON`, `ENABLE_MULTI_PROTOCOL=ON`. On init,
`auto_discover` installs `rdma` (HCAs present, `MC_FORCE_TCP` unset) and then
**composes** `hip` on top — the local segment advertises `"rdma,hip"`.
Registration fans out to every installed transport, so device KV gets both a HIP
IPC buffer and an RDMA buffer, while host aux buffers land on rdma only.
`MultiTransport::selectTransport` then routes **per request** by fixed priority
`hip 4 > cxl 3 > rdma 2 > tcp 1`, so for KV **hip wins**: `hipIpcGetMemHandle` on
the exporter, `hipIpcOpenMemHandle` on the importer, `hipMemcpyAsync` over
enabled peer access. **GPU-to-GPU across XGMI, no NIC in the path.**

One trap that makes the config lie: sglang passes `protocol="rdma"` into
`engine.initialize()` (`MOONCAKE_PROTOCOL` defaults to `"rdma"`). On this build
that argument **does not choose the transport** — outside the EFA/CXI paths it
only feeds `initMemoryAllocator()`. Setting `MOONCAKE_PROTOCOL` will not disable
hip, and seeing `rdma` in the config does not mean KV moves over RDMA.

### The real risk: the two legs cannot see each other's GPUs

`cluster.singlenode.sh` sets `PREFILL_GPUS=0,1,2,3` and `DECODE_GPUS=4,5,6,7`,
applied as `HIP_VISIBLE_DEVICES` (`../sglang_1p1d_glm5.2/engine/leg.sh:153`). The
two legs therefore have **disjoint visible device sets**, each seeing 4 devices
renumbered 0-3, and `setupP2PAccess()` only iterates visible devices — so peer
access is enabled *within* each leg and never between them.

**ANSWERED — it works.** Measured on gfx950 / ROCm 7.2 with the shipped engine
image, two processes in one container, `--ipc=host`:

```
exporter  HIP_VISIBLE_DEVICES=0,1   writes pattern 7,3,9,1,4,1,5,9 to cuda:0
importer  HIP_VISIBLE_DEVICES=2,3   imports the handle, reads back
  -> READ BACK: [7, 3, 9, 1, 4, 1, 5, 9]   MATCH
```

The importer **cannot see the exporter's physical GPU** and still mapped its
memory and read the correct bytes. A bare "import succeeded" would not have
proved this — the handle records device index 0 and the importer's own ordinal 0
is a *different* physical GPU, so the import could plausibly have mapped local
memory instead. The data pattern is what rules that out; **check the bytes, not
the return code**, if you repeat this.

Caveat on scope: measured with a 4-GPU visible set split 0,1 / 2,3 (physical
4,5 / 6,7 of that host) rather than the 0-3 / 4-7 split this kit uses. Same node,
same XGMI fabric. Strong evidence, not proof, for the exact split.

Note `torch.cuda.cudart()` does **not** expose `cudaIpcGetMemHandle` in this
build — use PyTorch's storage IPC path (`untyped_storage()._share_cuda_()` /
`torch.UntypedStorage._new_shared_cuda(*info)`), which is what actually carries
HIP IPC handles here.

Two source reads that failed to answer this before the measurement, recorded so
nobody repeats them:

Two attempts to close it from source, both negative, recorded so nobody repeats
them:

- **The ROCm 7.2 header** (`hip_runtime_api.h:2535-2545`) says
  `hipIpcOpenMemHandle` *"can attempt to enable peer access between the devices as
  if the user called hipDeviceEnablePeerAccess"*, and points at
  `hipDeviceCanAccessPeer` to test it. Suggestive, not decisive:
  `hipDeviceCanAccessPeer` takes **visible** ordinals, and under disjoint
  `HIP_VISIBLE_DEVICES` the importer cannot name the exporter's device at all. The
  doc does not say what happens then.
- **Mooncake's own HIP tests do not cover it.** All three harnesses
  (`tests/hip_transport_test.cpp`, `mooncake-wheel/tests/test_transfer_on_hip.py`,
  `tent/tests/hip_bandwidth_bench.cpp`) are single-process and single-device, and
  `grep -rn HIP_VISIBLE_DEVICES` over the whole repo returns nothing. So the
  pinned commit's *"prefill GPU0 / decode GPU1"* validation is not reproducible
  from the tree, and its test suite does not exercise two processes with disjoint
  visible devices — which is exactly what this kit configures.

That is the argument for running the probe below rather than reasoning further.

What *is* established is the shape of each outcome:

- **If it works:** KV moves over XGMI, and the only positive evidence is the
  install line plus the absence of hip errors.
- **If it fails, it fails LOUDLY at transfer time, not silently.** Registration
  still succeeds (`hipIpcGetMemHandle` is local), the segment still advertises
  `"rdma,hip"`, `selectTransport` still picks hip, and then
  `hipIpcOpenMemHandle failed` is logged and the transfer returns
  `"device memory not registered"` — surfacing as *"Failed to get kvcache from
  prefill instance"*, exactly the pre-fix symptom the pinned commit quotes.

So **the single-node failure mode is a broken PD, not a slow one** — provided hip
is installed. The silent-slow path exists only if hip is *absent*.

If it fails, the fix is a topology change — give both legs all 8 GPUs and split
with `--base-gpu-id` so each process can see its peer's cards — not a mooncake
debug session.

### Settle it in seconds, before loading any weights

Two processes with the kit's own disjoint split, exchanging one IPC handle. No
model, no server:

```bash
# exporter — the prefill leg's GPUs
docker exec -e HIP_VISIBLE_DEVICES=0,1,2,3 <prefill-ctr> python - <<'EOF'
import torch
t = torch.zeros(1<<20, dtype=torch.uint8, device='cuda:0')
h = torch.cuda.cudart().cudaIpcGetMemHandle(t.data_ptr())
open('/dev/shm/ipc.h','wb').write(bytes(h)); print("exported, holding"); input()
EOF

# importer — the decode leg's GPUs
docker exec -e HIP_VISIBLE_DEVICES=4,5,6,7 <decode-ctr> python - <<'EOF'
import torch
torch.zeros(1, device='cuda:0')                        # init the HIP context first
h = open('/dev/shm/ipc.h','rb').read()
print(torch.cuda.cudart().cudaIpcOpenMemHandle(h, 1))  # 1 = LazyEnablePeerAccess
EOF
```

`--ipc=host` is already passed to both containers (`../sglang_1p1d_glm5.2/common.sh:46`),
which HIP IPC across processes requires.

### What to grep, and the two lines nobody was checking

| outcome | line |
|---|---|
| HIP transport installed | `HIP transport installed for intra-node GPU P2P` |
| HIP install failed | `Failed to install HIP transport (intra-node GPU P2P unavailable)` |
| RDMA installed | `installTransport, type=rdma` |
| KV not IPC-exportable | `HipTransport: hipIpcGetMemHandle failed` |
| peer's KV not importable | `HipTransport: hipIpcOpenMemHandle failed` |
| two GPUs cannot reach each other | `HipTransport: P2P access not available between device i and device j` |
| TCP forced | `MC_FORCE_TCP is set, using TCP transport only` |
| **TCP fallback (no HCAs)** | **nothing — see below** |

Two properties of this table matter more than the table:

1. **The TCP fallback is silent.** TCP is installed with no success log where the
   RDMA branch logs `installTransport, type=rdma`. Grep for the *positive* rdma
   line and require it; there is no tcp line to find.
2. **No log line says which transport a given transfer used.** `selectTransport`
   chooses silently per request, and the two routing `LOG(ERROR)` calls in
   `multi_transport.cpp` are commented out in this tree. The install lines tell
   you the *capability*, never the *choice*. `MC_LOG_LEVEL=TRACE` adds
   per-buffer registration lines — still not per-transfer routing.

**The existing `MC_FORCE_TCP` / `GID is NULL` checks do not cover this case.**
`MC_FORCE_TCP` is an env var we would have to set ourselves, so counting it only
confirms we did not force TCP by accident. (It does catch one real disaster: if
set, init returns early *before* auto-discover, hip is never installed, and every
KV byte goes over TCP loopback — that genuinely is the 5-20× case.) `GID is NULL`
is per-RDMA-device rail health and is a **cross-host** signal; with the hip path
live the single-node KV transfer never touches a GID, so a count of 0 tells you
nothing about it.

**So add one line to the single-node smoke: require `HIP transport installed for
intra-node GPU P2P` in BOTH leg logs**, plus zero `hipIpcOpenMemHandle failed`.
Without the first, the segment is `"rdma"` only and KV silently takes loopback
RDMA with nothing raised anywhere.

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
