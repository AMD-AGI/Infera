# Running the e2e matrix on gfx942 (MI325X)

```{admonition} One-pager
:class: tip
**What:** let the existing e2e suite run on gfx942 with the *same* case table it
runs on gfx950 — same rows, same tp/ep/dp-attn, same pytest ids — while allowing
the launch knobs that genuinely differ per architecture to be written down next
to the case. **Why:** CI stays on gfx950; the gfx942 arm exists to freeze a
known-good model configuration so a deployment check on the local MI300X fleet
is one command, not an afternoon of rediscovering flags. **How:** the shell
*declares* the target arch (`INFERA_E2E_GFX_ARCH`, default `gfx950`), the
container *verifies* it against the GPU actually present, and each case row
carries an optional per-arch delta.
```

```{admonition} The two fleets
:class: note
GitHub CI runs on gfx950 (MI355X, 288 GB). The local SLURM cluster
(`amd-arad`, `amd-arad-burst`, `amd-rccl`) is **MI300X** — gfx942, but 192 GB
per card, not the 256 GB of an MI325X. Both are gfx942 as far as kernels and
images are concerned; the difference shows up only in how much fits, which is
why memory knobs are the first thing an overlay tends to carry.
```

## Running it

On the MI300X fleet, from a checkout and nothing else:

```bash
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e all mixed
INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e all disag
```

The profile supplies the model directory, the partition, the apt mirror and the
NIC pin; the arch is detected. Both tiers go through the scheduler and wait for
nodes. To reuse a pair you are already holding — `salloc`, or an `sbatch` that
sleeps — export its job id and the tiers attach to it instead of queueing:

```bash
SLURM_JOB_ID=<jobid> INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh e2e all disag
```

On the gfx950 CI fleet nothing changes: no profile is named, so every value comes
from `ci.yml` as before.

## Scope

- In scope: both tiers — PD-mixed (`tests/e2e/pd_mixed/`) and PD-disaggregated
  (`tests/e2e/pd_disag/`) — with gpt-oss-120b as the one model validated so far.
  The disagg tier was deferred while the mixed arm was being fitted, on the
  grounds that its orchestrator runs pytest on a GPU-less host and so has nothing
  local to probe; that is now solved by a per-node probe (`cluster.node_arch()`)
  and the tier is no longer refused off gfx950.
- Also in scope, and not anticipated when this was written: everything that
  stands between "the matrix is right" and "the suite runs here". A second fleet
  turns out to differ in apt reachability, in what "an idle node" means to the
  scheduler, and in a NIC rail — none of it about gfx942 at all. See
  [Running on a second cluster](#running-on-a-second-cluster).
- CI is unchanged. With `INFERA_E2E_GFX_ARCH` unset the target resolves to
  `gfx950` and every overlay is inert, so the collected case set and pytest ids
  are byte-identical to today.

## Base-image survey

Whether one image serves both architectures decides how much of the change is
Dockerfile work versus test-harness work. It differs per engine.

| Engine | Base image | Same image on gfx942 and gfx950? |
|---|---|---|
| vLLM | `vllm/vllm-openai-rocm:v0.25.1` | **Yes.** Upstream ships this tag for MI300X / MI325X / MI350X / MI355X. |
| ATOM | `rocm/atom:rocm7.2.4_…_atom0.1.4_20260612` | **Very likely yes.** ATOM lists gfx942 and gfx950 as fully supported (gfx950 is its primary CI target) and builds with a multi-arch `GPU_ARCHS`; the tag carries no arch marker. Verify on the box. |
| SGLang | `lmsysorg/sglang:v0.5.17-rocm720-mi35x` | **No.** The vendor tags are arch-split (`mi35x` / `mi30x`) and the gfx942 arm already has its own `Dockerfile.sglang.gfx942` on a v0.5.16 `mi30x` base. |

The infera layers we add on top are arch-neutral for vLLM and ATOM:

- aiter is built under the AITER RULE in `build_aiter_rocm.sh` — `GPU_ARCHS=native`
  with `PREBUILD_KERNELS` unset, so no kernel is compiled at build time and aiter
  JITs against the live GPU at first import. One image runs on any Instinct arch.
- `build_mooncake_rocm.sh` (used by `Dockerfile.vllm` and `Dockerfile.atom`) pins
  no GPU arch at all. Only `build_mooncake_sglang.sh` does, defaulting to gfx950
  and overridden to gfx942 by `Dockerfile.sglang.gfx942`.

Consequences:

1. Only SGLang needs a different Dockerfile per arch. vLLM and ATOM keep one
   Dockerfile **and one image tag** — the build output is identical, so a
   separate tag would only cost a redundant rebuild.
2. aiter's first-import JIT (~20 min) is paid again on a gfx942 node with a cold
   cache. Expect the first run's `server_ready_timeout` to need headroom.

## Can gpt-oss-120b even run on gfx942?

It is an MXFP4 checkpoint, and MXFP4 is where gfx942 and gfx950 genuinely
differ, so this was worth settling before writing overlays. **No model swap is
needed** — and there is no alternative checkpoint to swap to anyway.

**Checkpoint formats.** OpenAI post-trained gpt-oss-120b *natively* in MXFP4 for
the MoE linear weights, with every other tensor in BF16. There is **no official
FP8 or BF16 release**, and the model was trained directly at 4 bits, so one is
unlikely. Community dequantised BF16 copies exist (`lmsys/gpt-oss-120b-bf16`,
`unsloth/gpt-oss-120b-BF16`) but they are just the MXFP4 weights upcast: ~234 GB
instead of ~65 GB, and they need TP4 rather than TP2.

**What gfx942 actually does with MXFP4.** It is *emulated*, not unsupported:
the compute dequantises to BF16 inside the matmul while the weights stay
compressed in VRAM. So the memory footprint stays at the MXFP4 figure — the
reason a BF16 copy is not needed. (AMD's "MXFP4 is gfx950-only" notes are about
*native* MXFP4 arithmetic; they read as a flat "no" but are not one.)

| Engine | gpt-oss-120b on gfx942 | Evidence |
|---|---|---|
| ATOM | **On paper yes, in practice no** | Upstream `recipes/GPT-OSS.md`: "fits on a single MI300X/MI355X GPU"; nightly CI runs `gpt-oss-120b --kv_cache_dtype fp8` with `ATOM_GPT_OSS_MODEL=1` — the env var our gfx950 row already sets. Measured on MI300X, three published tags all fail before the first token: [gfx942-atom-gpt-oss](gfx942-atom-gpt-oss.md). |
| vLLM | **Yes** | The ROCm image ships gfx942 and gfx950 kernels; MI300X load figures are published (65 GB reported, 68.7 GB actual). Needs `VLLM_ROCM_USE_AITER_FP4BMM=0` — vLLM #34641 segfaults during warmup on gfx942 without it. |
| SGLang | **Blocked by a one-line gate** | See below. |

```{admonition} SGLang refuses MXFP4 on gfx942 at our pinned versions
:class: warning
`mxfp_supported()` in `python/sglang/srt/utils/common.py` matches `gfx95` only,
in **both** v0.5.16 (the mi30x base) and v0.5.17 (the mi35x base), and
`Mxfp4Config.from_config` turns that into a hard
`ValueError: Current platform gfx942 not support mxfp4 computation` at load
time. Not a fallback — a raise.

The kernels themselves are reported to work: upstream discussion #13611
concludes the check "is just too strict" and that the MXFP4/FP8 path is already
functional on gfx94x. The PR that would have flipped it,
[sgl-project/sglang#13929](https://github.com/sgl-project/sglang/pull/13929)
("a simple toggle", +1 −1), was **closed as stale in Aug 2026 without merging**.

**Decided: patch the gate** in `Dockerfile.sglang.gfx942`, alongside the three
SGLang patch sets the image already applies. It keeps the row, the model id and
the TP exactly as they are on gfx950, which is the whole point of the exercise.
The risk it carries is that upstream never merged the toggle and never posted
accuracy numbers for it, so the patch has to be justified by a correctness run
on the box, not by the PR — if the kernels turn out to be wrong rather than
merely un-gated, fall back to `skip`.

The two alternatives, for the record: `skip` the row on gfx942 with that
reason; or move to `lmsys/gpt-oss-120b-bf16` at TP4, which SGLang themselves
benchmarked on MI300X (`SGLANG_USE_AITER=0 --attention-backend triton --tp 4`)
but which changes both the model id and the TP, and so gives up the
identical-matrix property.
```

## Design

### 1. Arch resolution — ask the card, let a declaration override, verify the override

Three tiers, in order:

1. **A declaration.** `INFERA_E2E_GFX_ARCH` (`gfx950` | `gfx942`) — an *intent*,
   which overrides what the hardware says.
2. **The live GPU.** `run_tests.sh` probes the host it is about to build and run
   on; the test process probes again inside the container. Both sit on the same
   node and see the same card, so they agree without being told.
3. **`gfx950`**, for a host with no GPU to ask.

Detection is the normal path and is almost always right, so a run on an MI300X
node needs no ceremony. Tier 1 exists for the one case detection cannot serve:
the **PD-disagg orchestrator** runs pytest on a GPU-less login host and drives
containers onto two nodes SLURM picks *after* the image was chosen, so it has
nothing local to probe. It is also the escape hatch for forcing an arch on
purpose.

A declaration is **exported** and a probe result is **not**, and the difference
carries the design. `srun` re-runs `run_tests.sh` on the compute node, and that
node is the one that actually builds and runs; letting it probe for itself keeps
the image, the knobs and the hardware consistent there, whereas exporting the
login host's answer would pin the remote leg to gfx950 before it ever reached
the node that knows better. An intent, by contrast, has to travel. The same rule
governs what gets passed into the container.

Because a declaration overrides the hardware, it is the only thing that can be
*wrong* — which is what `resources.require_arch()` checks, and why an
auto-detected arch is never checked against anything (it agrees with the card by
definition). Verification **fails** rather than auto-corrects: the image was
already built for the declared arch, so switching the launch config while
keeping that image only moves the failure somewhere less legible.

```{admonition} Rejected: declaration-only, with no detection
:class: note
An earlier draft dropped tier 2 on the grounds that detection would make the
declaration agree with the hardware by construction, leaving the verification
unable to ever fire. That is circular: the check exists to catch a wrong
declaration, so having nothing to catch is the good outcome, not a cost. It also
overstated the risk — `srun` re-executes this script on the compute node, so the
node that builds is the node that probes, and the two halves cannot drift.
```

No new detection logic is written: `infera/common/arch.py` already maps compute
capability to a gfx name and exposes `python -m infera.common.arch`.

### 2. Per-arch config — an overlay inside `opts`, not a second table

The requirement is that the gfx942 matrix be *the same matrix*. Two rejected
alternatives:

- A separate `CASES_GFX942` table duplicates every row and drifts on the first
  change that touches only one copy.
- An `arch` axis alongside `tp`/`ep` has the wrong semantics: arch is a property
  of the machine, not a dimension to enumerate on one machine.

Instead, a row keeps its single entry and its `opts` dict gains an optional key
named after the arch, holding only the delta:

```python
[
    True,
    GPT_OSS,
    2,
    True,
    False,
    {
        "env": {"SGLANG_USE_AITER": "1"},
        "server_ready_timeout": 1800,
        "args": ["--attention-backend", "triton"],
        "gfx942": {
            "args": ["--attention-backend", "aiter"],
            "env": {"HSA_NO_SCRATCH_RECLAIM": "1"},
            "server_ready_timeout": 2400,
        },
    },
]
```

Merge semantics, fixed:

| Key | Rule | Why |
|---|---|---|
| `args` | **Replace** the whole list | Engines disagree on how a repeated flag resolves, so an append-merge is not predictable. These lists are short; writing them out keeps the delta readable as a config, not as a patch. |
| `setup` | **Replace** | Same reasoning. |
| `server_ready_timeout` | Replace | An older arch (and a cold aiter JIT cache) loads slower. |
| `env` | **Merge per key**; a value of `None` deletes the base key | Env is almost always additive; `None` is the escape hatch for the rare removal. |
| `skip` | Non-empty string skips the case on that arch, with the string as the reason | Keeps the case *visible* in the report. A silently absent case is not "the same matrix". |

Unknown keys — in the base `opts` or in an overlay — raise at collection time.
A typo like `"gfx940"` or `"arg"` would otherwise do nothing at all, quietly.

### 3. Image selection

One table maps `(engine, arch)` to `(tag, dockerfile)`, consumed by both the
PD-disagg conftests (which hardcode the pair today) and — through the same
values — `run_tests.sh`. Only the SGLang gfx942 entry differs from its gfx950
sibling; the rest exist so that adding an arch-specific variant later is a
one-line change rather than a refactor.

### 4. Guards

| Guard | Where | Fails on |
|---|---|---|
| Declared arch vs live GPU | `resources.require_arch()`, called first in `run_mixed_case` | Declaring gfx950 and landing on a gfx942 node, or the reverse (nothing to check when the arch was detected) |
| Unknown `opts` / overlay key | `apply_arch_overlay()` at collection | A typo'd overlay that would silently no-op |
| Unsupported `INFERA_E2E_GFX_ARCH` value | `target_arch()` | `gfx90a`, `GFX942`, a typo |
| Declared arch vs the GPU present | `resources.require_arch()`, from `mixed_suite` | A declaration that contradicts the card. **Mixed tier only** — see the gap below |
| Unknown variable in a site profile | `tests/unit/e2e_harness/test_site_profiles.py` | A profile setting something no code reads, which is silent at runtime |

## Change list

| File | Change |
|---|---|
| `tests/e2e/harness/arch.py` | New. `target_arch()`, `probe_arch()`, `assert_local_arch_matches()`. |
| `tests/e2e/harness/images.py` | New. `engine_image(engine, arch=None) -> (tag, dockerfile)`. |
| `tests/e2e/harness/matrix.py` | New `apply_arch_overlay()`; `expand_cases()` applies it; docstring documents the overlay. |
| `tests/e2e/harness/params.py` | `EngineParams.skip_reason: str = ""`. |
| `tests/e2e/harness/resources.py` | `require_supported()` honours `skip_reason`; new `require_arch()`. |
| `tests/e2e/harness/mixed_suite.py` | Calls `require_arch()` before the other guards. |
| `tests/e2e/conftest.py` | The `[e2e param]` line reports the target arch. |
| `tests/e2e/pd_disag/{sglang,vllm,atom}/conftest.py` | `IMAGE` / `DOCKERFILE` come from `engine_image()`. |
| `tests/run_tests.sh` | Resolve the arch (declaration, else `rocminfo`, else gfx950); forward a declaration to the remote leg and the container; pick the SGLang image/Dockerfile by arch. Plus the portability work below: site profiles, `_run_here()`, a corrected `_node_free()`, and reuse of an inherited allocation. |
| `tests/unit/e2e_harness/test_arch_overlay.py` | New. Overlay merge semantics, arch resolution, the image table, and the same-ids-on-both-arches invariant — pure logic. |
| `tests/unit/e2e_harness/test_site_profiles.py` | New. Static checks on `tests/sites/*.env`: grammar, that every variable set is one some code reads, K=V parsing. |
| `tests/sites/mi300x-rccl.env` | New. The MI300X fleet's measured values. |
| `deploy/docker/scripts/apt_mirror.sh` + `Dockerfile.{vllm,atom,sglang.gfx942}` | New `APT_MIRROR` / `APT_SECURITY_MIRROR` build args, no-op by default. Only the Dockerfiles a gfx942 run actually builds carry it; `Dockerfile.sglang` is gfx950-only, and that fleet reaches `archive.ubuntu.com`. |
| `deploy/docker/Dockerfile.sglang.gfx942` | Applies `patches/sglang_gfx942/`, which widens `mxfp_supported()` to `gfx94`. |

## Phasing

**Phase 1 — plumbing, no new behaviour.** Everything in the change list above,
with no `gfx942` overlay defined anywhere. On gfx950 nothing resolves
differently at all: the overlay merge is the identity and the image table
returns today's tags. On a gfx942 node the collected cases and their ids are
likewise unchanged (there are no overlays to apply yet); the one thing that does
change is that the SGLang tier now builds `Dockerfile.sglang.gfx942` instead of
the mi35x `Dockerfile.sglang` — which is a fix, since the latter cannot run
there. Acceptance: the unit tier passes (including the new tests) and a
collect-only run reports the same ids and counts as before, on both arches.

Landing alongside it, and *not* a zero-change edit: `pd_mixed/vllm` gains a
gpt-oss-120b row carrying the `pd_disag/vllm` row's knobs (tp2,
`--gpu-memory-utilization 0.9`) with expert parallelism on, matching what the
sglang and atom mixed grids do for this model. Single-node serving is the
precondition for reading a cross-node KV-transfer failure, and vLLM was the one
engine covering gpt-oss in disagg but not in mixed. Collection: 12 → 13 cases.

Also landing here, and the reason the gfx942 arm is worth trusting at all: two
more correctness probes in `harness/correctness.py`. A ~2.3k-token ledger with a
4-digit code buried 55% in has to be retrieved, and generated quicksort has to
survive execution against `sorted()` on duplicates, negatives and the empty list.
The old counting/capital pair passes on any model still producing fluent text,
which is exactly what MXFP4 emulated on gfx942 would produce if its kernels were
subtly wrong. The ledger is also the only probe that fills enough KV blocks to
make the PD tier's prefill→decode hop transfer a cache worth measuring. Gate: at
least one liveness probe passes, and every depth probe that *could* run must.

**Phase 2 — gpt-oss-120b on gfx942, PD-mixed.** ATOM first, since upstream
already validates that combination in nightly CI; SGLang after, since it also
needs the `mxfp_supported()` patch above. Add `gfx942` overlays to the gpt-oss
rows in `pd_mixed/{atom,vllm,sglang}`, iterate on a local node with
`INFERA_E2E_GFX_ARCH=gfx942 tests/run_tests.sh e2e atom mixed` until green, and
freeze the measured values into the overlay. Data (and one Dockerfile patch),
no harness logic. Known starting points, all needing validation on the box:

- `VLLM_ROCM_USE_AITER_FP4BMM=0` — vLLM #34641, a warmup segfault on gfx942.
  ATOM reads the same variable, so its gpt-oss row wants it too.
- `HSA_NO_SCRATCH_RECLAIM=1` — recorded in `manual/wip/mi325-deepseek-v4.md` as
  a gfx942 firmware requirement; distributed init aborts without it.
- `SGLANG_USE_ROCM700A=0` — selects the gfx942-correct ROCm path.
- The gfx950 rows force `--attention-backend triton` *only* because the aiter in
  the v0.5.17 mi35x base lacks a CK batch_prefill instance for that case. The
  v0.5.16 mi30x base may not have that gap, so the default backend is worth
  trying first on gfx942.
- Memory fraction: MI300X has 192 GB per card against MI355X's 288 GB, a third
  less, so `--mem-fraction-static` / `--gpu-memory-utilization` are the first
  knobs to drop while keeping TP fixed at 2.
- A generous `server_ready_timeout` for the first run, to absorb aiter's JIT.

**Phase 3 — PD-disaggregated. Done.** The per-node arch probe
(`cluster.node_arch()`, called from `disagg_fixtures._require_node_arch()` — one
`rocminfo` per node, which is what stops a 40-minute Mooncake build of the wrong
image), the gfx942 sglang image on both nodes, and the removal of the
`run_tests.sh` refusal. One correction fell out of it that is not arch-specific:
the tier's per-step `srun`s must carry `SLURM_JOB_ID` of the holder job, or on an
`OverSubscribe=NO` partition each step re-allocates and queues behind the tier's
own hold.

**Phase 4 (optional) — CI.** A `workflow_dispatch` input on `ci.yml` to run the
gfx942 arm on demand against gfx942 nodes. The scheduled path stays gfx950.

## Running on a second cluster

The matrix work above assumed the only thing separating the two fleets was the
GPU. Standing the suite up on the MI300X cluster said otherwise: most of what
blocked a clean `tests/run_tests.sh` run there was not about gfx942, and none of
it would have been visible from the gfx950 fleet, where every one of these
happens to be true by luck.

| What differs | Symptom before | Now |
|---|---|---|
| Nodes cannot route to `archive.ubuntu.com` | Every image build dies mid-way in Mooncake's `dependencies.sh` with a cmake "Could not find yaml-cpp" | `APT_MIRROR` / `APT_SECURITY_MIRROR` build args (`deploy/docker/scripts/apt_mirror.sh`), no-op unless set, and asserted: `apt-get update` exits 0 for a mirror that does not resolve, so the script checks that an index was actually fetched |
| Site values lived outside the repo | A run depended on a shell script only its author had | `INFERA_E2E_SITE` names a profile in `tests/sites/`; a profile only fills in what the caller left unset |
| Nodes carry background load, and idle-looking ones are drained | `_node_free()` read CPUAlloc, so it offered `IDLE+DRAIN` nodes and nodes whose GPUs or memory were already taken; the hold then sat PENDING until the tier gave up | It asks about the resources the hold requests — state, GPUs, memory — and keeps CPUAlloc only as the fallback where a scheduler does not report them |
| Submit hosts are themselves partition members with 8 GPUs | The mixed tier ran in place on a shared box whose cards were somebody else's job | `INFERA_E2E_LOCAL=0` (`_run_here()`) means "always go through the scheduler". The old `-n` test made `0` mean yes |
| A pair may already be held by hand or by an outer job | The tier asked SLURM for nodes it had already been given, and queued behind itself forever | `_inherited_pair()` reuses a RUNNING `SLURM_JOB_ID`'s nodes, and `_release_hold` will not cancel an allocation it did not create |
| Ten mlx5 rails per node | The two PD legs auto-discovered different rails, the QP never reached RTR, and decode reported "Failed to get kvcache from prefill instance" | `INFERA_E2E_WORKER_ENV=MC_TE_FILTERS=mlx5_0`, the channel ci.yml already uses on the gfx950 fleet — no new mechanism, it simply had never been set here |

The through-line: each of these was already parameterised, and the parameter was
being supplied out-of-band on one fleet and not at all on the other. The fix is
mostly to give the parameters a home in the tree rather than to invent knobs.

```{admonition} The NIC pin needs no CLI flag
:class: note
While the SGLang disagg leg was being debugged it was made to pass by adding
`--disaggregation-ib-device mlx5_0` to the case row. That flag is **not** in the
matrix, and should not be: it is a property of the cluster, not of the case, and
SGLang is the only engine that has one.

Measured instead with `MC_TE_FILTERS` supplied by the site profile and no flag at
all — SGLang's own `disaggregation_ib_device` stayed `None` — and Mooncake logged
`topology.cpp:168] IB device whitelist: mlx5_0` on both legs with
`protocol: rdma`, i.e. the rail was pinned and the transport did not quietly
degrade to TCP. One environment variable covers all three engines, because
`cluster.kv_transport_env()` injects it into every worker and ATOM's conftest
additionally reads it as its `ib_device`.
```

### Measured, 2026-08-26, gfx942 (MI300X)

Whole sweep through `INFERA_E2E_SITE=mi300x-rccl tests/run_tests.sh`, no other
environment, both tiers, on a two-node allocation. Run twice — once mid-way, and
again after a full image rebuild on both nodes from the final tree — with the
same result each time:

| Tier | vLLM | SGLang | ATOM |
|---|---|---|---|
| PD-mixed | 2 passed, 2 skipped (10m00s) | 1 passed, 2 skipped (2m28s) | 3 skipped (3s) |
| PD-disag | 1 passed (4m41s) | 1 passed (2m32s) | 1 skipped |

Of the twelve gfx942 cases, four run and eight skip; every skip is one of the
`gfx942` overlays and is reported with its reason rather than silently dropped.
The four that run are all gpt-oss-120b, and each ran the full set of correctness
probes. (vLLM's mixed tier shows one pass above its single matrix case because
that suite also carries a non-matrix check.)

## Open questions

1. ~~How to unblock SGLang's `mxfp_supported()` gate.~~ **Settled: patched.**
   `deploy/docker/patches/sglang_gfx942/patch_mxfp_supported_gate.py` widens the
   match to `gfx94`, and the row is justified by the correctness probes passing
   on the box rather than by the upstream PR (which was closed as stale). If a
   depth probe ever fails there, the fallback is still `skip`.
2. ~~Is the ATOM base image actually built with `gfx942` in its `GPU_ARCHS`?~~
   **Settled: no usable path.** Three published tags all fail before the first
   token — see [gfx942-atom-gpt-oss](gfx942-atom-gpt-oss.md). ATOM's gpt-oss row
   carries a `skip` on gfx942 rather than a second image entry, because the
   problem is upstream rather than in how we build it.
3. ~~Might 192 GB per card force a larger TP than the gfx950 row uses?~~
   **Settled in principle, unmeasured in fact.** TP stays a row axis, never an
   overlay key: raising it on one arch changes the pytest id and gives up the
   identical-matrix property. A case that will not fit gets `skip` with a reason,
   which is what Kimi-K2.6 and GLM-5.1 carry on gfx942 today. Whether they
   actually need a larger TP, or merely a memory-fraction overlay, is one run
   each — and those runs need the checkpoints staged, which they are not.
4. **Open, and newly found: the disagg tier verifies no arch at all.** The guard
   this design specified — `disagg_fixtures._require_node_arch()`, one probe per
   PD node before the build — does not fire. `cluster.node_arch()` probes via
   `cluster.srun_argv()`, which requests no GRES, so under GPU cgroup isolation
   the step sees no devices: `rocminfo` lists only CPU agents, `parse_rocminfo`
   returns None, and the check skips on `actual is None`. Measured on amd-rccl —
   the same step with `--gres=gpu:8` reports `gfx942` immediately. Nor does
   `resources.require_arch()` cover the gap: it is called from `mixed_suite` only.

   So on the disagg tier a wrong arch is caught by neither guard, and surfaces as
   a model-load failure after the image is built. The narrow fix is to let the
   probe ask for GPUs when it is attaching to an allocation we already hold (the
   `--jobid` path), where the GRES is ours and the step is instant; the Spur
   branch must be left alone, since there the probe would have to allocate and
   could queue. Not done, because it is a pre-existing gap rather than one this
   work introduced, and it wants its own change.

5. **Open.** Only gpt-oss-120b is validated on gfx942: of the twelve cases, four
   run and eight are `skip` — six for Kimi-K2.6 and GLM-5.1 across the three
   engines, two for ATOM's broken MXFP4 path.

   The six are blocked on which checkpoints are staged, not on anything
   measured. The MI300X fleet's model directory holds real weights for
   `gpt-oss-120b`, `GLM-5.2-FP8` and `DeepSeek-V4-Pro`, but only the first is
   what a row asks for: the matrix names `zai-org/GLM-5.1-FP8`, a different point
   release, there is no Kimi at all, and the two extra directories sit at the top
   level rather than under the `org/` prefix a model id resolves to. So retiring
   one of these skips starts with a decision — stage the checkpoint the row
   names, or move the row to one that is here — and only then is it: drop the
   overlay, run that engine's tier, write the measured knobs back.
