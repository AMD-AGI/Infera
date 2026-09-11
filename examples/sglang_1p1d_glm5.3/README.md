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

## Contents

| path | what |
|---|---|
| [`cluster.2node.sh`](cluster.2node.sh) | two-node pair — the validated shape. Fill in and run |
| [`cluster.singlenode.sh`](cluster.singlenode.sh) | one 8-GPU node split TP4 + TP4 |
| everything else | comes from [`../sglang_1p1d_glm5.2/`](../sglang_1p1d_glm5.2/) — `common.sh`, `engine/*.sh`, `preflight_rdma.sh` |

## The engine base is part of the configuration, not a detail

**Verified base: `lmsysorg/sglang:v0.5.18-rocm720-mi35x` (sglang v0.5.18).** Both
the baked image and the overlay were validated against it and nothing else.

This is a harder constraint than a usual "tested with" note. `deploy/docker/patches/sglang_dsa/`
is `--fuzz=0` diffs cut against that one release, and GNU patch has no atomicity
across files: on a different sglang the set can land some files and reject
others, leaving the engine to start and serve on a **half-patched tree** with
nothing in any log saying so. Measured on a 0.5.15 base — six of seven files
applied, all seven hunks of `dp_attn.py` rejected.

The baked image catches this at build time, because bumping its pinned base fails
the build. **The overlay does not** — it patches whatever base it is mounted
over. So with the overlay, pin the base yourself. See the TODO in
`deploy/overlay/Dockerfile.payload`.

## Validation status

Stated plainly, because the honest answer is short.

| shape | status |
|---|---|
| the **deployment shape** (1P1D + mooncake + DPA + MTP + kvd + kv-aware) | validated for **GLM-5.2** on two clusters, both fabric types |
| **GLM-5.3-MXFP4**, two-node TP4+TP4, **symmetric DP-attention** | **validated** on 2×MI355X (gfx950), mooncake over RoCE, all 8 `ionic` rails, mode A. 6/6 correct including a 7845-token needle recovered verbatim, and **0/32 degenerate at concurrency 8** |
| **GLM-5.3-MXFP4**, two-node, **asymmetric** DPA (prefill dp1 → decode dp4) | **BROKEN — corrupt output.** This is the GLM-5.2 kit's default shape. See below |
| **single-node** TP4+TP4, `DECODE_DPA=0` | **validated** on one 8-GPU node — 7845-token needle recovered, both legs' HIP transport installed 4/4, zero IPC failures |
| **single-node** TP4+TP4, `DECODE_DPA=1` | **BROKEN — corrupt output.** Same signature as the asymmetric two-node case |
| **EAGLE MTP(3,1,4)** on the decode leg, two-node symmetric DPA | **validated** on 2×MI355X. 6/6 correct, **0/32 degenerate** at concurrency 8, 7845-token needle verbatim — token counts and `finish_reason` identical to the same probe with MTP off. Acceptance length over 509 samples: median **2.35**, p90 2.73, only **0.2 %** at the 4.00 ceiling |
| **EAGLE MTP** on the **single-node** shape | **not run.** That shape runs dp off, so it does not exercise the same draft path |

### The DP-attention failure, because it is silent and it will cost you a day

On GLM-5.3-MXFP4, DP-attention **plus** the PD path returns garbage: streams of
repeated `</think>` interleaved with digit noise, every request finishing on
`length` and never on `stop`. Measured:

| arm | topology | prefill | decode | result |
|---|---|---|---|---|
| 1 | one node | dp1 | dp4 | first completion clean, **then every one garbage** |
| 2 | one node | dp1 | dp1 | clean, 7845-token needle verbatim |
| 3 | **two nodes** | **dp4** | **dp4** | **clean**, 6/6 + 0/32 degenerate at conc 8 |
| 4 | aggregated, no PD | — | dp4 | clean, 5/5 including a repeat |

Four things about it that each cost time to establish:

- **It is silent.** Zero memory-access faults, zero HIP errors, zero non-dynamo
  tracebacks, `MC_FORCE_TCP` 0, GID-is-NULL 0, decode batches rolling with cuda
  graphs on. No log line anywhere says the tokens are wrong.
- **It is not a long-prompt or long-output problem.** 19-token prompts fail as
  readily as 7845-token ones; `max_tokens` 128, 1024 and 2048 all fail. The first
  instinct — KV corruption on a big transfer — is wrong.
- **It develops.** The first completion after bring-up was clean and every later
  one was not, *including a repeat of that same prompt*. **A single clean answer
  is not evidence.** Send a repeat and a small load round before believing a
  green result.
- **DP-attention alone is fine** (arm 4). It needs DP-attention and PD together.

Cause **not diagnosed**. Arms 1 and 3 differ in two variables at once — DPA
symmetry and topology — so arm 3 is a *working configuration*, not an isolation
of the cause. Upstream search found nothing matching; sgl-project/sglang#30033
is a different failure (a lockstep freeze, not corrupted output).

The wrappers now default to what was measured clean: `cluster.2node.sh` is
symmetric (`PREFILL_DPA=1`), `cluster.singlenode.sh` has DP-attention off.

## The single-node path: HIP IPC over XGMI, not loopback RDMA

The same-host KV handoff is **not** a loopback RDMA transfer, and the risk is
**not** silent slowness — which changes what you check. Established by reading the
mooncake tree and build cache inside the shipped image:

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
no log line that will confirm it. Treating the non-flip as evidence of anything
will get a correctly configured hip-off deployment discarded unmeasured.

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

**It works.** Measured on gfx950 / ROCm 7.2 with the shipped engine image, two
processes in one container, `--ipc=host`:

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
memory and read the correct bytes. A bare "import succeeded" would not have proved
this — the handle records device index 0 and the importer's own ordinal 0 is a
*different* physical GPU, so the import could plausibly have mapped local memory
instead. The data pattern is what rules that out; **check the bytes, not the
return code**, if you repeat this. The cross-container run closes the container
boundary as a variable.

Caveat on scope: measured with a 4-GPU visible set split 0,1 / 2,3 (physical
4,5 / 6,7 of that host) rather than the 0-3 / 4-7 split this kit uses. Same node,
same XGMI fabric. Strong evidence, not proof, for the exact split.

Note `torch.cuda.cudart()` does **not** expose `cudaIpcGetMemHandle` in this
build — use PyTorch's storage IPC path (`untyped_storage()._share_cuda_()` /
`torch.UntypedStorage._new_shared_cuda(*info)`), which is what actually carries
HIP IPC handles here.

Two attempts to close this from source, both negative, recorded so nobody repeats
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

### What to grep, and the two lines that are easy to miss

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
`MC_FORCE_TCP` is an env var the operator has to set, so counting it only
confirms TCP was not forced by accident. (It does catch one real disaster: if
set, init returns early *before* auto-discover, hip is never installed, and every
KV byte goes over TCP loopback — that genuinely is the 5-20× case.) `GID is NULL`
is per-RDMA-device rail health and is a **cross-host** signal; with the hip path
live the single-node KV transfer never touches a GID, so a count of 0 tells you
nothing about it.

**So add one line to the single-node smoke: require `HIP transport installed for
intra-node GPU P2P` in BOTH leg logs**, plus zero `hipIpcOpenMemHandle failed`.
Without the first, the segment is `"rdma"` only and KV silently takes loopback
RDMA with nothing raised anywhere.

What a healthy node looks like, verified first-hand on the validated hardware:

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

**MTP.** The two-node wrapper defaults `DECODE_MTP=1`, which is the GLM-5.2 kit's
split (prefill off, decode on) and is measured here. Upstream's GLM-5.3 cookbook
says MTP/EAGLE is disabled on AMD because the gfx950 draft kernel is unvalidated,
while the OneNexus model card runs EAGLE at `--speculative-num-steps 3`; the
measurement above settles it for this deployment and the cookbook line does not
govern.

It should not have been a surprise. `deploy/docker/patches/sglang_dsa/` exists
specifically to make PD + DP-attention + EAGLE MTP work on this architecture:
patch 02b fixes an assert that only fires under MTP, patch 04 is entirely about
the **draft** CUDA graph on the PD decode leg, and patch 01's `GLM52_P1V3` half
exists for the MTP draft-extend path on an idle DP rank. GLM-5.3 big is the same
`glm_moe_dsa` architecture as GLM-5.2, so the set applies unchanged — and turning
MTP off was leaving four of that set's seven bytecode markers unexercised.

**What it buys.** Single-variable A/B on the same deployment, same bench (random,
ISL 8192 / OSL 1024, concurrency 8, 80 prompts), only `DECODE_MTP` changed:

| | MTP=0 | MTP=1 |
|---|---|---|
| output token throughput | 343.5 tok/s | **466.8 tok/s** (+36 %) |
| mean TPOT | 20.62 ms | **14.51 ms** (−30 %) |
| mean TTFT | 1988 ms | 2139 ms (+8 %) |
| `ITL / TPOT` | **1.00** | **2.38** |

TTFT moves the wrong way, which is the expected shape: speculation is a
decode-side mechanism and adds nothing to prefill, while the decode leg's extra
draft work pushes the tail out. That last row is the cleanest identification of
the feature there is — with speculation off a decode step emits exactly one
token, with it on it emits 2.38, which is the measured acceptance length.

The parameters are **(3, 1, 4)** and you cannot change them from a wrapper:
`engine/up.sh`'s `COMMON_ENV` does not forward `SPEC_STEPS`/`SPEC_DRAFT`, so
`engine/leg.sh`'s defaults are what land. They are also what sglang v0.5.18 picks
for this architecture on its own.

**Read the acceptance-length median, never the last few lines.** A few percent of
batches sit at the 4.00 ceiling even on a healthy leg, so `tail -5` routinely
lands on them and misreads as degenerate — `engine/smoke.sh` reports the
distribution for exactly this reason. Two more traps: `decode_log_interval`
defaults to 40, so a leg that has served fewer than 40 decode iterations prints
**no** `accept len:` lines even with MTP working perfectly; and with dp on, each
DP rank's scheduler appends to the same log, so the sample count is ~4× a
single-rank leg and is not comparable across a DPA change. The median is.

**Shared-experts fusion.** Not a concern for the big MXFP4 checkpoint: its
shared experts are themselves MXFP4 (76 `.weight` / 75 `.weight_scale`, the odd
one being the BF16 MTP layer 78, which the decode leg now loads as its draft), so
the precondition for the mismatch does not hold. Even so, `glm4_moe.py`'s fusion gate only special-cases `w4afp8` and would fuse under
`quark`, so the wrappers pass `--disable-shared-experts-fusion` as insurance;
upstream #25261 shows this class failing *silently with wrong output* rather
than crashing when the shapes happen to line up.

## Source

[`examples/sglang_1p1d_glm5.3/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [the GLM-5.2 kit this drives](../sglang_1p1d_glm5.2/)
· [aggregated MIX kit for the GLM-5.3 checkpoints](../sglang_mix_glm5.3/)
· [PD disaggregation concepts](../../manual/features/pd_disaggregation.md)
