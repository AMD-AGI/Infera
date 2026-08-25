# GLM-5.2-MXFP4 — SGLang 1P1D + DP-attention + MTP on MI355X

Runnable deployment kit for **GLM-5.2-MXFP4** served the infera way: one **prefill**
node and one **decode** node, KV moved between them over **Mooncake RDMA**, fronted
by the **infera router**, with **DP-attention**, **MTP** (EAGLE speculative decoding)
and the **kvd** cache tiers all on.

Two files in [`cluster/`](cluster/) hold everything site-specific. Fill in one of
them and the deployment is three commands.

```bash
bash preflight_rdma.sh mode                  # which wrapper do I need?
bash cluster/cluster.peermem.sh up           # (or cluster.dmabuf.sh) — bring it up
bash cluster/cluster.peermem.sh smoke        # prove it works
```

This exact shape has been measured under two independent agentic benchmarks on two
different fabrics; those numbers are not published with this kit.

## Contents

| path | what |
|---|---|
| [`cluster/`](cluster/README.md) | **the only files you edit.** Two wrappers, one per RDMA fabric type |
| `common.sh` | shared helpers — container, etcd, router, health polling, teardown |
| `engine/leg.sh` | the real launcher for one PD leg. The tuned recipe lives here; no site values do |
| `engine/up.sh` | bring up both nodes: containers → etcd + kvd → both legs → router |
| `engine/smoke.sh` | service check **plus** positive evidence for each of the five features |
| `engine/bench.sh` | reference throughput sweep using SGLang's own `bench_serving` |
| `engine/capture.sh` | take one torch trace per PD role out of a running load (opt-in) |
| `engine/down.sh` | tear down and wait for VRAM to actually free |
| `preflight_rdma.sh` | RDMA preflight: registration-mode probe + cross-node fabric measurement |

## Topology

```
node-P                                        node-D
├─ etcd                    ◀── discovery ──▶
├─ infera router  :8100    ◀── clients
├─ kvd daemon
└─ prefill leg    :30000   ══ KV over RDMA ══▶ decode leg :30001
   TP8, DPA off,                                TP8, DPA on (dp8),
   kvd L2/L3                                    MTP (EAGLE)
```

The router discovers both legs through etcd and pairs them — there is no static
worker list to maintain. Clients only ever talk to `:8100`.

## 1. Prerequisites

**Hardware.** Two nodes, 8× MI355X (gfx950) each, on a **mutually routable RoCE
fabric**. The KV handoff is RDMA and there is no TCP fallback worth having — over TCP
the pair is slower than a single node.

**Weights.** GLM-5.2-MXFP4 (~400 GB) on **each** node.

**Image.** An infera-sglang build **newer than 0.2.0**. `<infera-sglang-image>` is a
placeholder wherever it appears below — substitute that tag. Both nodes must run the
same image.

**Docker with GPU + RDMA access**, and a way to run a command on each node
(`ssh` by default — see [`cluster/README.md`](cluster/README.md#schedulers-where-ssh-node-does-not-work)
if your scheduler blocks it).

## 2. Verify the RDMA fabric first

PD moves the KV cache across the fabric on **every request**, and every way that can
go wrong is silent. A container that cannot see the RDMA devices does not raise —
Mooncake falls back to a transport that works and is merely 5–20× slower. Run the
preflight before the bring-up, not after the numbers disappoint you.

### 2.1 Registration mode — this also picks your wrapper

```bash
IMAGE=<infera-sglang-image> bash preflight_rdma.sh mode
```

Per RDMA NIC it reports **vendor, link speed, ODP, PCI BDF, NUMA node and GID
index**; per node it reports whether a **peer-memory module** is loaded and whether
the image's Mooncake engine has dma-buf compiled in. It then enumerates the three
registration modes, each marked viable or blocked **with the reason**, and prints the
exact env and launch flags for the one it picks.

| what it reports | which wrapper | why |
|---|---|---|
| `peermem: present` → **mode A** | [`cluster/cluster.peermem.sh`](cluster/cluster.peermem.sh) | bare `ibv_reg_mr` hands the NIC the GPU pages directly: nothing pinned, KV pool not duplicated, **every** rail carries KV |
| `peermem: absent` + an ODP NIC → **mode B** | [`cluster/cluster.dmabuf.sh`](cluster/cluster.dmabuf.sh) | dma-buf is the only GPUDirect path without peer-mem, and it is only safe on an ODP NIC — so KV gets locked to that one NIC. A fallback, not a target: see below |
| only **mode C**, or nothing viable | **stop and decide** | mode C pins and **doubles** the KV pool; it needs an explicit cap computed for your model/TP/VRAM. Preflight exits `2` here on purpose — it is a human decision, not a default |

Then copy the reported `MC_GID_INDEX`, device list and dma-buf setting into the
wrapper. The report is the source; do not guess these.

**Mode B's single NIC is a constraint, not a design choice.** Locking KV to one
card is what makes dma-buf *safe* without peer-mem — a non-ODP rail would pin the
registration and duplicate the KV pool in VRAM. But it costs whatever bandwidth
the node's other rails would have carried, and on a node with fast non-ODP rails
that cost is large: preflight prints an explicit
`*** PERFORMANCE REGRESSION ***` line naming both numbers when it picks this mode.
Expect to see it, and read it as the price of the mode rather than as a
misconfiguration.

**If mode A is available, prefer it** — every rail carries KV, nothing is pinned.
Mode B is what you run when no peer-memory module is loaded and loading one is not
an option. Both wrappers configure the *same* deployment shape; only the transport
differs, so switching later means editing the wrapper, not the recipe.

Two of these values bite in ways worth knowing in advance:

- **`MC_GID_INDEX` is per node, not per cluster.** Two identical machines routinely
  expose the routable GID at different indices, because an empty slot on one shifts
  everything after it. A wrong index fails loudly (`GID is NULL`, on every DP rank,
  at init) — cheap to catch, expensive to assume. The link-local `fe80::` GID is
  never the answer; it is not routable across the fabric.
- **A down rail must not be listed.** Enumerate what is actually `PORT_ACTIVE` on
  **each** node; the two can legitimately differ.
- **A rail visible in `/sys` is not necessarily usable by the engine** — only the
  vendor provider libraries the image ships can open one, so `ibv_devinfo` inside
  the container may report far fewer cards than the host lists. Preflight runs
  there for that reason; believe its verdict over the host's view.

### 2.2 Cross-node fabric — the check that catches silent TCP

Device visibility does not prove the two nodes can move KV between them. This does,
one task per node into a shared directory:

```bash
export IMAGE=<infera-sglang-image> DUMP_PATH=<fresh-shared-dir>
srun --nodelist=<node-P>,<node-D> -N2 --ntasks-per-node=1 bash preflight_rdma.sh fabric
```

It produces a cross-node RoCE bandwidth matrix (`ib_write_bw`) and — the rows that
matter for PD — **Mooncake KV-transfer bandwidth measured separately over `rdma` and
over `tcp`**. A fabric that will silently serve at TCP speed therefore appears as a
number here, rather than as a deployment that is inexplicably slow later.

Use a **fresh** `DUMP_PATH` each run: rank 0 decides everyone has reported by counting
`*.json` files, so a stale file makes it render early.

## 3. Configure

Open the wrapper the preflight pointed you at and fill in the four blocks: nodes,
image and weights, transport, deployment shape. [`cluster/README.md`](cluster/README.md)
walks through every field.

**Everything site-specific is in that one file.** `common.sh` and `engine/*.sh`
contain no addresses, paths, NIC names or GID indices at all. If you find yourself
editing an engine script to adapt to your cluster, something is wrong — say so, it is
a bug in this kit.

## 4. Deploy

```bash
bash cluster/cluster.peermem.sh up
```

That runs, in order: containers on both nodes → etcd and kvd on the prefill node →
both legs concurrently → wait for both to serve → the router.

**Cold start is minutes** — ~400 GB of weights plus CUDA-graph capture. Silence is
not a hang; `up.sh` polls `/health` patiently and reports elapsed time as it goes.
Both legs are launched before either is awaited, so they load in parallel.

## 5. Verify

```bash
bash cluster/cluster.peermem.sh smoke
```

A green `/health` proves the process is alive — not that PD paired, that KV moved
over RDMA, or that speculative decoding is doing anything. `smoke` checks each of
those with a signal that would go **red** if the feature were silently absent:

| feature | what is checked | healthy reading |
|---|---|---|
| **router + PD pairing** | `/v1/workers` | two workers, one prefill and one decode |
| **serving + DSA correctness** | one chat completion | a correct, coherent answer |
| **mooncake over RDMA** | `MC_FORCE_TCP` and `GID is NULL` counts in both leg logs | **0** and **0** |
| **DP-attention** | resolved `dp_size` / `enable_dp_attention` per leg | matches what you configured |
| **MTP** | `accept len` on the decode leg | roughly **2–3** |
| **kv-aware / kvd** | router policy line; kvd adapter count and counters | one adapter per DP rank on the kvd leg |

Two of those readings are counter-intuitive:

- **Garbage or repeated tokens from the chat completion** is not a sampling problem.
  It is the signature of the DSA-on-ROCm env block not taking effect — the model
  serves, it just computes sparse attention on a path that is not ported to this
  architecture.
- **An MTP acceptance length of a steady 4.00 is bad news, not a good result.** It
  means the draft model is predicting a repetition loop perfectly, i.e. the output has
  degenerated. 2–3 is the healthy band.

The completion check asks a one-line question but sends `max_tokens: 512`, which is
deliberate. GLM-5.2 is a thinking model and the leg passes `--reasoning-parser glm45`,
so the chain of thought lands in `reasoning_content` — but it is billed against the
**same** budget. At a small value the model spends every token thinking, `content`
comes back empty with `finish_reason: "length"`, and the check reads as a failure on a
deployment that is serving correctly.

### Reference sweep

```bash
bash cluster/cluster.peermem.sh bench 8 16 32
```

This uses SGLang's own `bench_serving`, which ships inside the image. Three of its
flags are load-bearing in ways that are not obvious from the flag name:

- **`--random-range-ratio 1.0`** pins every prompt to exactly `ISL`. The default draws
  uniformly, and the percentiles then mix request sizes — a fixed-length sweep wants a
  delta, not a distribution.
- **`--temperature 1.0 --top-p 0.95`** are the checkpoint's own `generation_config`
  defaults, and are deliberately *not* greedy. At temperature 0 this reasoning model
  falls into repetition on a long prompt, MTP then predicts the loop perfectly,
  acceptance length pins at 4.00, and the run reads like KV corruption.
- **`--cache-report`** needs the server's `--enable-cache-report` (`engine/leg.sh`
  passes it). Its column is nonetheless meaningless on this dataset: `--dataset-name
  random` builds every prompt independently, so there is **no shared prefix by
  construction** and any nonzero value is residue from the previous round. Prefix reuse
  is an agentic-workload property.

`--num-prompts` is recomputed per concurrency (`10 × C`), so each arm of a sweep gets
enough requests to reach steady state.

**This kit ships no agentic benchmark client**, by design. Point the customer's own
harness at the router endpoint.

### Profiling

Off unless asked for. Two switches, `PROFILE_PREFILL` and `PROFILE_DECODE`, and the **same
values must be given to `up` and to `capture`** — `up` decides whether the control plane
exists at all, `capture` reads them to pick which roles to sample.

**Decode.** The common case, and the expensive one:

```bash
PROFILE_DECODE=1 bash cluster/cluster.peermem.sh up
nohup bash cluster/cluster.peermem.sh bench 64 &          # load, in the background
PROFILE_DECODE=1 bash cluster/cluster.peermem.sh capture  # 20s window out of it
```

**Prefill.** Same three commands, one switch changed:

```bash
PROFILE_PREFILL=1 bash cluster/cluster.peermem.sh up
nohup bash cluster/cluster.peermem.sh bench 64 &
PROFILE_PREFILL=1 bash cluster/cluster.peermem.sh capture
```

Cheaper than the decode case in a way worth knowing: the prefill leg runs its **normal**
configuration. There is no graph to turn off, because the extend path does not use one, so
the only distortion is the python router — the leg itself is the same one you benchmarked.
Measured on this stack, a prefill capture came back with 31 steps and 31 of them carrying
GPU operators (coverage 1.00) with nothing disabled.

What you do have to think about is whether there is enough prefill work in the window. At
steady state most of the machine is decoding; prefill only runs when a new request arrives,
so a long `OSL` starves it. If the step count comes back low, shorten the output
(`OSL=128`) or raise the concurrency, and re-check the histogram.

**Both at once.** Supported, and the only way to get a P and a D trace from the *same*
window — the two starts are issued together from one shell, so the windows line up:

```bash
PROFILE_PREFILL=1 PROFILE_DECODE=1 bash cluster/cluster.peermem.sh up
nohup bash cluster/cluster.peermem.sh bench 64 &
PROFILE_PREFILL=1 PROFILE_DECODE=1 bash cluster/cluster.peermem.sh capture
```

Note the asymmetry this creates. `PROFILE_DECODE=1` turns the decode graphs off, and
`bench_serving` is closed-loop: a request holds its concurrency slot until its last token, so
a decode leg running ~4× slower completes requests ~4× slower and **prefill is fed ~4× less
often**. What that does and does not damage:

- **Survives.** The prefill kernels themselves. The two legs are different GPUs on different
  nodes, and what an extend step costs is set by its batch shape and the model, not by why it
  got scheduled. "Which operators dominate prefill" — the usual reason to profile it — holds.
- **Thins out.** The step count in the window. A 20 s window may come back with single digits.
- **Skews.** The batch-shape mix. Sparser arrivals mean fewer requests queued when the
  scheduler forms an extend batch, so it biases toward `bs=1`, and `bs` drives the MoE expert
  distribution and the GEMM shapes. Shape-dependent conclusions are taken from a thinner
  batch distribution than real steady state.
- **Breaks.** Anything with wall-clock in the denominator — occupancy, inter-step gaps, "how
  busy is prefill", throughput — and all queueing and TTFT figures.

So: **profile prefill on its own if you want its operators** — it needs no graph disabled, so
the extra round is nearly free. Use both roles together for the one thing only it can give
you, a P and D pair from the same window, where the decode distortion is already priced in.

(The mechanism above is reasoned from the closed-loop setup, not measured; the 35.6 → 137.2 ms
decode figure is.)

`bench` is deliberately not given a detach flag — `nohup … &` already does it, and the one
thing a flag would buy (surviving an ssh drop mid-run) is already caught by `capture`, which
refuses to start, and warns again after warm-up, if no `bench_serving` is running. Give the
load enough requests to outlast `WARMUP_S + WINDOW_S`: `N=<count>` overrides the `10 × C`
default.

`capture` waits `WARMUP_S` (default 60) for the load to reach steady state, opens a
`WINDOW_S` (default 20) window, starts both roles from **one** shell inside the container
so their windows line up, stops with a single call, waits for the per-rank files to stop
growing in the shared `TRACE_OUT` mount, and returns the output directory. Trace inspection
is a separate operation; no tar, `docker cp`, or SSH transfer is involved.

Three things about this are easy to get wrong and are handled for you:

- **The rust router cannot do it.** `--enable-profiling` makes `launch_rust.py` raise
  `SystemExit` rather than degrade, so the router simply never comes up. `start_router`
  switches the backend to `python` when it sees the flag. The python router forwards more
  slowly, so **throughput from a profiling run is not comparable to a normal one** — say so
  in whatever you write up.
- **Decode CUDA graphs hide the kernels you are profiling for.** A graph replays as one
  opaque launch: the trace keeps the per-step annotation and loses everything inside it.
  Measured here: 134 decode steps in a window, **1** of them with GPU operators, and the
  resulting kernel ranking was not merely imprecise but *reordered*. So `PROFILE_DECODE=1`
  also sets `CUDA_GRAPH=0` on the decode leg — which costs real throughput (decode TPOT
  measured 35.6 → 137.2 ms), which is why it is not a default. Prefill needs no equivalent;
  its extend path does not run through a graph. `--enable-profile-cuda-graph` is **not** a
  substitute: it instruments graph *capture*, which happens once at start-up.
  Set `DECODE_CUDA_GRAPH=1` to profile with graphs on anyway — that is the run you compare
  against to show how much the graph was hiding.

The traces are ordinary torch/chrome traces — open one in Perfetto, or point an analyzer at
the output directory. This kit does not ship one.

## 6. Tear down

```bash
bash cluster/cluster.peermem.sh down
```

It removes the containers and then **waits** for VRAM to drain, printing
`rocm-smi --showpids` so you can see it happen. Relaunching before that completes
OOMs on a box that looks idle — the wait is the point, not the kill.

## Recommended configuration

The shipped defaults, and what each one is for:

| knob | prefill | decode | why |
|---|---|---|---|
| `--tp-size` | 8 | 8 | one node each |
| DP-attention | **off** | **on** (dp8) | see below |
| MTP (EAGLE) | off | **on** | decode-only is the validated configuration |
| `--ep-size` | 8 | 8 | **always**, independent of DP-attention |
| `--mem-fraction-static` | **0.70** | 0.85 | see below |
| `--chunked-prefill-size` | 65,536 global | — | a global budget, not per-rank |
| `--context-length` | 262,144 | 262,144 | covers a 260K-token input clamp |
| kvd (L2 host RAM + L3) | **on** | off | off on decode by design |
| router policy | `kv-aware` (pw 20.0 / dw 2.0) | | |
| custom all-reduce | **disabled** | **disabled** | see below |

**Prefill DP-attention off.** In the measured pair, the arm with prefill DPA off
carried 25 % more throughput at lower latency across every percentile. That
comparison moved two variables at once (DPA *and* router policy) and cannot be split
— but a separate single-variable comparison at concurrency 1 found DP-attention
costing 1.65–1.93× on TTFT, and the direction agrees. Decode keeps DPA on: it is
where the 8 attention ranks earn their keep.

**Router policy and prefill memory are coupled**, which is documented nowhere else.
Under `round-robin`, 4–5 DP ranks prefill concurrently — each holding its own chunk's
activations — where `kv-aware` concentrates on 1–2. The round-robin arm **would not
boot** at `--mem-fraction-static 0.80` and needed 0.70. **If you switch to
`round-robin`, lower `GMU_PREFILL`.** That is why 0.70 is the shipped default rather
than the higher value a kv-aware-only deployment could sustain.

## Notes & gotchas

**1. `--ep-size` and `--enable-dp-attention` are different parallelism axes.** Expert
parallelism vs attention parallelism. Gating both on one condition means turning
DP-attention off also collapses the MoE from ep8 to the TP default — so a deployment
billed as "DPA off" differs in the expert-dispatch collective too, and no latency
delta is attributable to either. `engine/leg.sh` emits `--ep-size` unconditionally.

**2. `--chunked-prefill-size` is a GLOBAL budget, and DP-attention divides it.**
SGLang divides it by `dp_size` **only** when DP-attention is on — a division, not a
clamp, though the engine's own warning text reads like a clamp. At dp8 a requested
65,536 resolves to 8,192 per rank while staying 65,536 machine-wide. One value serves
both modes; hardcoding the per-rank number in a DPA-off branch cuts the global budget
8×.

**3. Prefill activation OOM is fixed by LOWERING `--mem-fraction-static`.** This is
the opposite of the decode-side fix and it costs real debugging time to rediscover.
Diagnose by phase:

| phase | symptom | direction |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **raise** |
| prefill | `HSA_STATUS_ERROR_OUT_OF_RESOURCES` / `Aborted` at **low** token usage | **lower** |

Low token usage at the moment of the abort is the tell: the KV pool is nearly empty,
so it was never KV exhaustion — it is activation memory, which lives in the
`1 - mem_fraction_static` remainder.

**4. Custom all-reduce is disabled, and independently of MTP.** The aiter custom
all-reduce kernel deadlocks on this architecture during speculative verify. Letting
the switch follow MTP would make any "MTP on vs off" comparison a two-variable one.

**5. `SGLANG_OPT_USE_TOPK_V2=0` is mandatory on gfx950.** Without it the model
still serves and still returns 200s — it just returns garbage, because SGLang's
`topk_v2` JIT kernel is CUDA-only. No launcher sets it: `infera.engine.sglang`
defaults it off on ROCm (`infera/engine/rocm_dsa_env.py`), and `smoke` catches it
if it did not take.

**6. MTP and decode-side radix cache are mutually exclusive upstream.** SGLang raises
on `--disaggregation-decode-enable-radix-cache` together with
`--speculative-algorithm`. Consequence: `decode_prefix_len` is always 0, so **every
turn re-transfers the entire prompt KV**, and a prefill-side cache hit saves *compute*,
not *bytes*. This is why fabric bandwidth matters on long-prompt agentic workloads
even at a 89 % cache-hit rate.

**7. kvd's sizes are absolute on purpose.** `--hicache-size` is in GB, not a ratio:
SGLang's ratio-based default sizes the host pool off `max_total_num_tokens` and can
compute to hundreds of GB **per DP rank**, and a TB-scale pinned host allocation can
wedge a node at kernel level. Likewise kvd's L3 `--long-bytes` — it writes to a
container-local path, so an oversized budget fills the node's root filesystem, after
which every `docker exec` fails with `no space left on device` and the node reads as
broken rather than full.

**8. `Ctrl-C` on a log tail does not stop anything.** Use `down`, and check
`docker ps`.

## Validation status

Stated plainly rather than implied.

| what | status |
|---|---|
| the deployment **shape** this kit encodes (1P1D + mooncake + DPA + MTP + kvd + kv-aware) | **validated end-to-end on two clusters**, both fabric types, under a real agentic workload |
| the tuned values (GMU, chunk, ctx, EAGLE settings, DSA env, router weights) | **validated** — each is carried over from a run that completed cleanly |
| the three traps in Notes 1–3 | **first-hand**, each found by a run that failed or silently mis-measured |
| **these scripts as written** | **validated** — `preflight_rdma.sh mode` → `up` → `smoke` → `bench` → `down` on a 2-node MI355X mode-B cluster, with no edits outside `cluster/cluster.dmabuf.sh`. Long context checked separately (needle, to 238K tokens) and under a real agentic workload at concurrency 8 |
| `preflight_rdma.sh` | `mode` validated on both nodes and its verdict followed. `fabric` not exercised |
| `cluster.peermem.sh`, `round-robin` routing | **not validated** — no peer-mem cluster was available, and the shipped `kv-aware` default is what ran |

If you run this kit and it does not come up, that is worth reporting.

## Source

[`examples/sglang_1p1d_glm5.2/`](.) in [AMD-AGI/Infera](https://github.com/AMD-AGI/Infera)
· [Kubernetes recipes for the same model](../recipes/glm5.2/README.md)
· [PD disaggregation concepts](../../manual/features/pd_disaggregation.md)
· [preflight reference](../../manual/reference/preflight.md)
