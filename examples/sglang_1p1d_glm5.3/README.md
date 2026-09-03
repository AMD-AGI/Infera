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

**GLM-5.3-Flash is NOT covered by these wrappers**, but the question of whether
PD is *possible* for it now has an answer — see below. For Flash today, use
[`sglang_mix_glm5.3`](../sglang_mix_glm5.3/) (aggregated).

### Flash PD: feasible, OUT OF SCOPE, never run

**Deliberately not pursued.** What follows establishes that the shape is
*possible* — it is a source read, not a validation, and no Flash PD deployment
has ever been brought up. It is recorded so that whoever picks this up starts
from the code reads rather than repeating them, and so that "we did not do this"
is not mistaken for "this does not work".

The two checks at the end of this section are what that person should run first.

The obvious reason to expect Flash PD to be broken is that `glm5_next` keeps
**two** pools — the paged KV pool *and* a KDA recurrent-state pool (logged as
`mamba usage`) — while PD hands off after prefill. If the KDA state were not
transferred, the decode leg would start its linear-attention layers from a zero
state, and that fails as **subtly wrong output**, not as a crash.

**The premise does not hold.** sglang's PD path is not KV-only: it carries a
generic *state component* mechanism, `StateType.MAMBA` is one of its members, and
`glm5_next` resolves onto exactly the pool that mechanism reads. Verified link by
link against the pinned ref `c821c425`:

| link | where |
|---|---|
| `glm5_next` recognised as recurrent-state | `hybrid_arch.py:114` |
| folded into the generic predicate | `hybrid_arch.py:126` `mambaish_config()` |
| declared to keep SSM state | `kv_cache_builder.py:100` `uses_ssm_state()` |
| KDA state modelled as a mamba2 cache | `configs/glm5_next.py:264` |
| engine builds a hybrid req pool | `kv_cache_configurator.py:896` |
| pool exposes state buffers | `memory_pool.py:1436` `get_state_buf_infos` |
| **PD registers them as transferable** | `disaggregation/utils.py` — duck-typed on `get_state_buf_infos()` → `append_state_component(..., StateType.MAMBA, ...)` |
| prefill builds the payload | `prefill.py:1183` `_mamba_payload()` |
| decode has the matching pool | `decode.py:223` `HybridMambaDecodeReqToTokenPool` |
| transport is generic over components | `mooncake/conn.py:1241` iterates `state_types` |

**No guard refuses it** — grepping `NotImplementedError|not supported` across
`srt/disaggregation/*.py` for mamba/hybrid/linear/kda/glm5 returns nothing. The
guards that exist are narrow and unrelated.

**So the worry relocates rather than disappearing: from "not transferred" to
"transferred, but unverified".** Every link above is a code read; nobody has run
it, and whether the KDA conv/ssm buffers survive the mooncake round-trip
bit-exact on gfx950 is untested. That is still the failure mode that produces
subtly wrong output with nothing logged.

**Two checks, in order, before believing any Flash PD result:**

1. *Ten minutes, at bring-up.* Confirm at runtime that `StateType.MAMBA` actually
   lands in `kv_args.state_types` for `glm5_next` — one log line at
   `append_state_component`. That converts the table above from "the code says it
   should" into "it did".
2. *The decisive one.* Send the same prompt to a Flash **MIX** deployment and a
   Flash **PD** deployment, greedy, and **diff the token sequences**. Intact state
   → identical; zeroed decode-leg state → divergence, with nothing logged. The
   reference side already exists: `flash-mxfp4` MIX is validated end to end.

One INFERRED caveat: `mori/conn.py:1041` refuses loudly when `state_types` is
empty; the mooncake path iterates `state_types` without an equivalent check *at
that site*, so an empty list there could be a silent no-op. Whether mooncake
checks elsewhere is UNKNOWN.

Upstream has three OPEN items in this exact area — **#36651** (adding PD state
transfer for another Flash-class model, so the wiring is not automatic
everywhere), **#37276** (PD + mamba + speculative decoding, which is why running
Flash PD with MTP off sidesteps a known bug), and **#33457** (hybrid-linear KV
transfer under prefill pipeline parallelism). The machinery is live and being
repaired, not absent.

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

### Everything in this section is single-node-specific, and that is by design

**Two-node PD never uses hip, regardless of any setting.** `selectTransport`
calls `isHipReachableTarget()` and skips hip buffers whenever the target is on
another host. The in-source rationale, at the build commit:

> *"This makes the intra-node fast path (hip) and the cross-node path (rdma) work
> automatically from a single multi-protocol segment, without requiring the
> operator to set `MC_DISABLE_HIP`."*

So the multi-protocol segment is not a configuration problem to be solved — it
resolves itself by target. **Every hip question in this kit is a single-node
question**, and none of it applies to `cluster.2node.sh`.

### Installation is unconditional; *selection* is what the knob controls

`transfer_engine_impl.cpp:402-414` installs hip under a bare `#ifdef USE_HIP`
with no runtime condition. The gate is one stage later, at
`multi_transport.cpp:489`:

```cpp
if (p == "hip")  return std::getenv("MC_DISABLE_HIP") ? 0 : 4;
if (p == "rdma") return 2;
```

`MC_DISABLE_HIP` demotes hip from priority 4 to 0, so rdma wins for the device KV
pool — which is registered under both. **hip stays installed and stops being
used.**

**This is why the obvious check is useless.** `HIP transport installed for
intra-node GPU P2P` is an **install-time** log, and the variable gates
**selection**. It reads 4/4 whether hip is carrying KV or demoted to zero, in
every state, forever. Verifying a hip-off arm requires `MC_DISABLE_HIP=1` present
in `/proc/<pid>/environ` on both legs **plus** the source read above — there is
no log line that will confirm it, and treating the non-flip as evidence of
anything is how a correct hip-off deployment got discarded unmeasured.

Two related names are **absent from the binary entirely** and do nothing:
`MC_DISABLE_HIP_TRANSPORT` (which `leg.sh:60` sets) and
`MC_ENABLE_HIP_TRANSPORT`. So: **two dead names, one live name whose effect is
invisible to the natural check.**

One trap that makes the config lie: sglang passes `protocol="rdma"` into
`engine.initialize()` (`MOONCAKE_PROTOCOL` defaults to `"rdma"`). On this build
that argument **does not choose the transport** — outside the EFA/CXI paths it
only feeds `initMemoryAllocator()`. Setting `MOONCAKE_PROTOCOL` will not disable
hip, and seeing `rdma` in the config does not mean KV moves over RDMA.

### The real risk: the two legs cannot see each other's GPUs

`cluster.singlenode.sh` sets `PREFILL_GPUS=0,1,2,3` and `DECODE_GPUS=4,5,6,7`,
applied as `HIP_VISIBLE_DEVICES` (`../sglang_1p1d_glm5.2/engine/leg.sh:159`). The
two legs therefore have **disjoint visible device sets**, each seeing 4 devices
renumbered 0-3, and `setupP2PAccess()` only iterates visible devices — so peer
access is enabled *within* each leg and never between them.

> **This was not true as originally shipped, and the failure is worth knowing.**
> `up.sh` forwards a fixed list of per-leg variables through `on()` — which runs a
> fresh remote shell — and `GPUS` was not among them. Both legs therefore fell
> through to `leg.sh:26`'s default, `seq 0..TP-1`, and **landed on the same four
> cards**. Fixed by forwarding `${PREFILL_GPUS:+GPUS=$PREFILL_GPUS}` per leg;
> conditional expansion, so an unset variable injects nothing and the two-node
> path is unchanged.
>
> **The way it failed is the instructive part.** The prefill leg died with
> `Loaded weights leave no GPU memory for the KV cache under
> --mem-fraction-static=0.7. Raise --mem-fraction-static above 0.773` — a number
> that is arithmetically correct and diagnostically wrong. Taking the engine's
> advice would have let two legs coexist on four cards and produced a deployment
> that *ran*, with every subsequent number meaningless and nothing saying so.
> **Before trusting any memory error, check that both halves of the box are
> loaded** — `rocm-smi --showmeminfo vram` should show weights on GPUs 0-3 *and*
> 4-7 (~408 GB / TP4 ≈ 102 GB per card for GLM-5.3-MXFP4). That distinguishes a
> tuning problem from a topology problem.
>
> **Do not use `base_gpu_id` for this.** It is tempting and it does not work:
> `HIP_VISIBLE_DEVICES=4,5,6,7` renumbers the decode leg's devices to 0-3, so
> `base_gpu_id` is an index into the *visible* set, not the physical one. It reads
> `0` on both legs when the split is broken **and** `0` on both legs when it is
> correct — it does not discriminate at all. The VRAM read is more expensive and
> it is the only unambiguous check here.

**ANSWERED — it works.** Measured on gfx950 / ROCm 7.2 with the shipped engine
image, two processes in one container, `--ipc=host`:

```
exporter  HIP_VISIBLE_DEVICES=0,1   writes pattern 7,3,9,1,4,1,5,9 to cuda:0
importer  HIP_VISIBLE_DEVICES=2,3   imports the handle, reads back
  -> READ BACK: [7, 3, 9, 1, 4, 1, 5, 9]   MATCH
```

Repeated **across two separate containers** (importer started with `--ipc=host`),
which is the shape PD actually runs — same disjoint split, same pattern:

```
CROSS-CONTAINER IMPORT OK, bytes= 1048576
READ BACK: [7, 3, 9, 1, 4, 1, 5, 9]   MATCH
```

The importer **cannot see the exporter's physical GPU** and still mapped its
memory and read the correct bytes. The single-container run alone would have
left the container boundary as an untested variable; it is closed. A bare "import succeeded" would not have
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

> **And on this build hip cannot be turned off by the variable anyone would
> reach for.** `leg.sh:60` exports `MC_DISABLE_HIP_TRANSPORT=1` and unsets
> `MC_ENABLE_HIP_TRANSPORT`. **Neither name exists in the shipped mooncake
> binary.** Exact-match against `mooncake/engine.*.so`:
>
> | env name | matches |
> |---|---:|
> | `MC_DISABLE_HIP` | **1** |
> | `MC_USE_HIP_IPC` | 1 |
> | `MC_FORCE_TCP` | 1 |
> | `MC_DISABLE_HIP_TRANSPORT` | **0** |
> | `MC_ENABLE_HIP_TRANSPORT` | **0** |
>
> Confirmed behaviourally as well as by inspection: a run launched with
> `MC_DISABLE_HIP_TRANSPORT=1` — verified present in the process environment via
> `/proc` — still logged `HIP transport installed for intra-node GPU P2P` **4×
> per leg**, identical to a run without it.
>
> Two consequences. **`leg.sh:60` has never had an effect**, in either the
> two-node or single-node path, so it is not evidence that anyone deliberately
> disabled hip — which is likely why no reason for it could be established
> (INFERRED: the author may have intended to disable hip and used a name that
> does not exist). And **any A/B that varies hip must set `MC_DISABLE_HIP`**;
> using the `_TRANSPORT` spelling produces a guaranteed-zero differential that
> reads as a null result rather than as a broken experiment.
>
> Before trusting any such A/B, confirm the discriminator actually flipped:
> `HIP transport installed for intra-node GPU P2P` must go **4/4 → 0/0**.

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

   **3a. A third form, and the engine's own advice is a trap in it.** On the
   single-node shape a leg can abort at startup with

   ```
   ValueError: Loaded weights leave no GPU memory for the KV cache under
   --mem-fraction-static=0.7. Raise --mem-fraction-static above 0.773
   ```

   That number is arithmetically correct and **diagnostically wrong**. It is
   computed from the memory actually free at that moment, and the reason there
   is none is usually that **the other leg is on the same cards** — measured
   once as GPUs 0-3 at 263.8 GB each with 4-7 at 0.3 GB. Raising the fraction
   as instructed produces a *working* deployment on the wrong topology: two
   legs sharing half the node, every subsequent number meaningless, nothing
   logged.

   **Check the cards before you touch the knob.** `rocm-smi --showmeminfo vram`
   must show load on *both* halves. Do not use `--showmemuse`'s `VRAM%` (it
   does not fall when memory is released) and do not use each leg's
   `base_gpu_id` (it is an index into the leg's **visible** set, so
   `HIP_VISIBLE_DEVICES=4,5,6,7` renumbers the decode leg to 0-3 and it reads
   `base_gpu_id=0` on both legs whether the split is broken or correct).

   Distinguishing the three: **this** form aborts during startup profiling with
   weights already loaded and no request served; the classic prefill form
   aborts under load at low token usage; the decode form retracts under load at
   high token usage.
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
