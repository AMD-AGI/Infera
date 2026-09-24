# Working process — same-node 1P1D P4D4 on crsuse2-m2m-276

Task of record: `spec/mission.md`. One row per round; each round owns a
directory under `rounds/`.

## Round index

| # | dir | purpose / hypothesis | outcome |
|---|---|---|---|
| 0 | `rounds/000-recon/` | Survey: is same-node P4D4 reachable on 276 at all? | **done** — 4 blockers found (see below); user authorised clearing 276's load |
| 1 | `analysis/hardware_prep.yihou.md` | Free 276's GPUs; verify the GPU/NIC/NUMA split the plan assumes | **done** — GPUs at idle baseline; 0-3/4-7 split confirmed NUMA-clean, no dead rail |
| 2 | `rounds/002-rdma-loopback/` | Cheapest possible check of the Round-0 open question: does RDMA work between two *different* rails on the *same* host? | **done** — NO for different devices, YES for same device; reproduced on 276 and 137. Rail split withdrawn |
| 3 | `rounds/003-mooncake-mode/` | User-prompted: read `mooncake_mode.py`; does mooncake have a single-machine mode? | **done** — yes: HIP IPC + multi-protocol locality routing, both compiled into our image. `MC_DISABLE_HIP_TRANSPORT` is a dead name / no-op. Launch held |
| 4 | `analysis/mooncake_samehost.yihou.md` | What does `MC_USE_HIP_IPC` default to, and what does `mp_selectTransport` key locality on? | **done** — same-host KV takes HIP GPU-IPC over XGMI, NOT RDMA. Round 2 is moot for the KV path. Mechanism (VMM vs IPC) still being verified live |
| 5 | `analysis/upstream_samenode_research.yihou.md` | Upstream/official: Mooncake same-host transport, SGLang colocated PD, ionic same-host RoCE limits, ionic dma-buf/ODP | **done** — contradicts Round 4; leader reconciled: sglang passes `MOONCAKE_PROTOCOL` default `"rdma"`, so hip may never be advertised |
| 6 | `rounds/006-bringup-1/` | First same-node bring-up | **failed** — SGLang internal port block derived from `--port` overlaps between legs. Fixed with `ENGINE_PORT_STRIDE=256` |
| 7 | `rounds/007-bringup-2/` | Bring-up with the port stride | **partial** — port fix holds; **HIP transport confirmed installed on BOTH legs live**; `Found 1 HCAs` (N>0); decode loading weights; prefill dies in RCCL `ncclCommInitRank` with `HIP failure: invalid argument` |
| 8 | `rounds/008-nccl-debug/` | Re-run with `NCCL_DEBUG=INFO` to diagnose the RCCL init failure | **done** — `p2p.cc:256 hipIpcGetMemHandle failed : invalid argument`; prefix(0-3) 1 warning, decode(4-7) 0 |
| 9 | `rounds/009-gpu-swap/` | Differential: swap the GPU sets between legs | **NCCL cleared on both**; then **all 4 schedulers segfault on BOTH legs** at event-loop start. Variable still NOT isolated (N=1) |
| 10 | `rounds/010-prefill-only/` | Decomposition: bring up the prefill leg ALONE — is the segfault intrinsic, or a co-location interaction? | next |

## Round 0 — recon (2026-09-22, leader)

First-hand findings, all from commands run this session:

1. `crsuse2-m2m-276` is a **spur** node: `ssh` refused (`AllowUsers ubuntu root`),
   reachable only via `spur exec 165913`. That path runs as `yihou`, `pwd=/`,
   and `docker` works.
2. **All 8 GPUs were occupied** — container `hy4-nomtp` (vLLM, up 10 h) owned
   KFD PIDs 494054-494061, ~285 GB of 309 GB per card. `yzhou_model` holds no
   GPU memory. User authorised clearing the load.
3. `tools/topology.py:load()` rejects two rows on one node: it raises on
   `node in nodes or data_ip in ips`. 276 has exactly one physical IPv4,
   `10.245.152.249` on `ens3`. **This is the one change co-location forces.**
4. Ports are already safe: `launch.sh:rows()` adds the topology row index to
   every port base, so the two legs get distinct ports and container names.
5. The run image is **not** on 276; it is on 137 (66 GB).
6. Hardware is right: 8× MI355X, `ionic_0..7` all ACTIVE, `libionic.so` present,
   model visible at `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`,
   26 TB free on `/mnt/m2m_nobackup`.

Decisions taken: vendor the harness into this workspace and patch the copy
(user's choice); prefill = GPUs 0-3 + `ionic_0-3`, decode = GPUs 4-7 +
`ionic_4-7` so the legs never share a NIC.

Open questions carried forward:

- Does mooncake establish a KV session between two containers on the **same
  host** across different NICs? Untested.
- `MC_ENABLE_DEST_DEVICE_AFFINITY=1` + `PD_DP_RANK_AFFINITY=1` assume same-name
  rails; the 0-3/4-7 split breaks that symmetry. First hypothesis if KV
  transfer fails to establish.

## Round 1 — node prep (2026-09-22, node-prep)

`analysis/hardware_prep.yihou.md`. **Outcome: PASS, the 0-3/4-7 plan holds.**

- Stopped (not removed) container `hy4-nomtp` (`vllm-rocm-hy4-stacked:test`, up
  ~10 h, created 2026-09-21T20:22:28Z), which held ~285 GB on all 8 cards via
  KFD PIDs 494054-494061 (`VLLM::Worker_TP`). `yzhou_model` left running — it
  holds zero GPU memory. No file deleted.
- All 8 GPUs settled to 297,766,912 B (~284 MiB idle baseline), no KFD PIDs.
  VRAM release lagged the stop (41-185 GB mid-release on the first re-poll)
  before clearing — the expected settle behaviour, not a leak.
- GPU<->NIC<->NUMA verified two ways (`rocm-smi --showtoponuma` and sysfs
  `numa_node`), in agreement:
  - prefill GPUs 0-3 + `ionic_0-3` = NUMA0 / PCI domain 0002, rails 08/07/05/06
  - decode GPUs 4-7 + `ionic_4-7` = NUMA1 / PCI domain 0003, rails 04/03/01/02
  Disjoint sockets, disjoint NICs. All 8 ionic ports ACTIVE, 400 Gb/s, live
  netdev, non-zero GID. **No defective rail** (unlike 135's `ionic_7`).
  Topologically identical to 137/136.
- Host: `ens3` = 10.245.152.249/20; `libionic.so` 1.0.54.0-149.g3304be71;
  RAM 2751 GB total / 2718 GB available; `/mnt/m2m_nobackup` 26 TB free;
  model 408 GB / 282 shards, readable.

Two flags carried forward:

1. **`/shared_nfs` is 100% full, ~1.9 TB free.** Stage nothing large there;
   use `/mnt/m2m_nobackup`. Relayed to image-xfer.
2. **Same-node split means prefill rank i and decode rank i sit on different
   rails** — inherent to crossing the two NUMA/NIC groups on one box. Each leg
   is still NUMA-local to its own NICs. This is the concrete form of the
   `MC_ENABLE_DEST_DEVICE_AFFINITY` / `PD_DP_RANK_AFFINITY` open question from
   Round 0; still UNSETTLED, still the first hypothesis if KV transfer fails.

## Team poll — 2026-09-22 ~06:51Z (leader)

`ListAgents`: `node-prep` running (21 m), `upstream-research` running (4 m),
`image-xfer` idle/complete, `harness` idle/complete. Both live teammates polled
for a first-hand status, a blocker, and a next action.

**No new problem observed this poll.** Recording the state only, per the poll
rule (first sighting records, second sighting intervenes).

Carried items already handled earlier this session, not re-opened here:

- `harness` shipped a config that still held the withdrawn 0-3/4-7 rail split
  after being told to drop it. Caught by reading the file rather than trusting
  the report; the leader applied the fix directly. `harness` subsequently
  verified the on-disk state rather than re-writing it, so the per-role AITER
  cache addition survived. Closed.
- `harness` raised a genuine new watch-item: both legs share the dynamic
  `INFERA_NODEPORT_RANGE=30000-32767`, and whether infera de-conflicts it via
  etcd could not be verified first-hand. **Suspended, not a demonstrated
  collision.** Documented mitigation if it bites at bind time: split the range
  per role (prefill 30000-31383 / decode 31384-32767).
- `harness` also surveyed the shared-host surface: the AITER JIT cache is the
  **only** writable shared host path (`engine.sh` bind-mounts only `$MODEL:ro`,
  `$HOST_RDMA_LIB:ro`, and the JIT cache rw), so every other `/tmp` write is
  container-private. `--ipc host` shares SysV IPC and `/dev/shm` — a theoretical
  surface, unchanged from the cross-node shape, watch-item only.

Deliverables landed: `analysis/hardware_prep.yihou.md`,
`analysis/image_transfer.yihou.md`, `analysis/harness_samenode.yihou.md`,
`scripts/bench-harness/` (patched), `scripts/config.yihou.sn.p4d4.sh`,
`scripts/topology.yihou.tsv`, `bin/ssh`, `patches/`.

**Launch remains HELD** on mission obstacle 9 (which transport a same-host KV
target actually takes). Launching now would measure a configuration we cannot
describe.

## Round 4 — mooncake same-host transport (node-prep) — THE CENTRAL FINDING

`analysis/mooncake_samehost.yihou.md`. Evidence tiered by the teammate as
`[binary]` (strings/symbols from the exact `.so` we run), `[run]` (engine
actually initialized in our image), `[upstream]` (github main, which has
**drifted** — some binary strings are absent upstream, so upstream is
corroborating design only, cross-checked against binary symbols).

**Same-host KV transfer does not use RDMA. It uses GPU-IPC over XGMI.**

1. `[run]` The intra-node HIP GPU-P2P transport **installs by default** in our
   image, with our exact env: `transfer_engine_impl.cpp:412 HIP transport
   installed for intra-node GPU P2P`. It is **not** gated behind
   `MC_USE_HIP_IPC`.
2. `[binary]` The GPU KV pool is registered under **both** transports; the
   segment is advertised `"rdma,hip"`.
3. `[binary]` Locality key: `isHipReachableTarget(segmentName, localServerName)`
   + `segmentHost(segmentName)`, which **strips the port and compares the
   host**. Both legs advertise `10.245.152.249` → `hip_reachable=true`. Our
   same-IP/different-port topology does **not** defeat it. Log field
   `hip_reachable=`.
4. `[binary]` `protocol_priority`: `hip` = 4, `rdma` = 2
   (`getenv("MC_DISABLE_HIP") ? 0 : 4`), so a same-host peer takes hip;
   cross-host has a different host → hip unreachable → rdma.
5. `[run]` Physically available: every GPU pair reports `True`/`XGMI`, including
   cross-NUMA 0↔4 — exactly our prefill 0-3 ↔ decode 4-7 layout.

**This makes Round 2 moot for the KV path.** The ionic cross-device same-host
RDMA failure is real and reproduced, but KV never reaches the NIC on this shape.
Round 2 was not wasted — it is what forced the question, and it still governs any
cross-host deployment — but it must not be quoted as a constraint on same-node KV.

### Operational consequences, all mandatory

- **Never set `MC_DISABLE_HIP=1`.** It would force the same-host legs back onto
  the cross-device RDMA path Round 2 proved moves zero traffic. Our
  `config.sh:89` says `MC_DISABLE_HIP_TRANSPORT=1`, which is a **dead name and a
  no-op** — do not "fix" it into the real spelling.
- **Keep `MOONCAKE_DISABLE_HIP_DMABUF=0`** — a real variable, keeping GPU
  registration on for the `rdma` half of `"rdma,hip"`.
- The rail configuration (`ionic_0-3` on both legs) is now **inert for the
  same-node KV path**. Leaving it as-is; it is harmless and changing it would
  add a variable for no benefit.

### Open, carried to bring-up

- The HIP transport has two GPU-sharing mechanisms in the binary: fabric/VMM
  (`hipMemExportToShareableHandle`) and IPC, plus a `"falling back to IPC mode"`
  string. `MC_USE_HIP_IPC` forces IPC. Whether the **default** mechanism
  completes a copy is being verified live by a 2-process same-host GPU transfer.
  Routing is unaffected either way.
- **`Found N HCAs` must be N>0** in the real `engine.sh` container's startup log.
  A bare probe container reported `Driver ionic does not support kernel ABI 4` →
  `Found 0 HCAs`; almost certainly an artifact of probing without `engine.sh`'s
  host-`libionic` mount (the proven cross-node runs used this image with RDMA
  working), but it is undetermined and cheap to check.
- `MC_ENABLE_DEST_DEVICE_AFFINITY` governs only the RDMA path → moot here. The
  exact peer-device tie-break (name vs index) is **suspended**: upstream no
  longer has `selectPeerDevice`, so the teammate declined to guess.

## Round 5 — upstream research, and a teammate conflict the leader had to adjudicate

`analysis/upstream_samenode_research.yihou.md` (upstream-research). Its Q1/Q2
**contradict** Round 4's Q1.

- Round 4 (`node-prep`, `[run]` on our image): HIP transport installs, segment
  advertised `"rdma,hip"`, same-host takes hip.
- Round 5 (`upstream-research`, `[upstream]`): the `"rdma,hip"` multi-protocol
  segment is built by Mooncake PR #2682 + #2753 and gated on the CMake option
  **`-DENABLE_MULTI_PROTOCOL=ON`, which defaults OFF and our build never
  passes**. Separately, **SGLang initializes mooncake with a SINGLE protocol
  string** and never requests `"rdma,hip"`. It also found
  `MC_ENABLE_HIP_TRANSPORT` / `MC_DISABLE_HIP_TRANSPORT` have **zero upstream
  hits** — they are infera-invented names, which is consistent with their
  absence from our `.so`.

### Leader's first-hand reconciliation

Read in our image, not inferred:

- `srt/distributed/device_communicators/mooncake_transfer_engine.py:205-219` —
  sglang calls `engine.initialize(hostname, metadata, protocol, device_name)`
  with `protocol = envs.MOONCAKE_PROTOCOL.get()`.
- `srt/environ.py:803` — `MOONCAKE_PROTOCOL = EnvStr("rdma")`. Default
  **`"rdma"`**; and it is a free-form `EnvStr`, **not** a constrained choice
  list, so `"rdma,hip"` is settable.
- `srt/disaggregation/mooncake/conn.py:57,299` — the PD path takes the **same**
  shared engine via `get_mooncake_transfer_engine()`; `conn.py` has no
  `protocol` or `initialize` of its own.

**Hypothesis:** both teammates are half right. The HIP transport installs
regardless (Round 4's observation), but the **segment advertisement follows the
protocol string sglang passes**. At the default `"rdma"`, our segments are
rdma-only, `selectTransport` has no hip protocol to choose, and same-host KV
falls back to RDMA — straight onto the cross-device path Round 2 proved moves
zero traffic.

The suspected flaw in Round 4: the engine was likely initialized **directly**
rather than through sglang's wrapper, so its `"rdma,hip"` observation may not
describe what our deployment does. Asked `node-prep` to confirm how it
initialized.

**This would have cost a 30-minute bring-up.** Launch stays held.

### Experiment commissioned (differential, one variable)

2-process same-host GPU transfer, reproducing sglang's init path:
**A** `MOONCAKE_PROTOCOL` unset (`"rdma"`) vs **B** `MOONCAKE_PROTOCOL="rdma,hip"`.
Capture the actual `initialize(...)` args, whether the hip transport installs,
**what the segment is advertised as**, the `selectTransport route=` /
`hip_reachable=` lines, and whether the transfer completes. Folds in the open
fabric/VMM-vs-IPC mechanism question (`MC_USE_HIP_IPC=1`).

### Plan B, already measured, needs no new variable

Round 2 measured same-**device** RDMA loopback at ~40 GB/s, and
`infera/tools/preflight/mooncake_mode.py:613` independently prescribes, for
single-node, "**pin ONE device on both legs instead**". Pinning every rank on
both legs to `ionic_0` is therefore a measured-good rdma-only path. Costs: all 8
ranks funnel through one 400 Gb/s NIC, and it is **untested for GPU memory**
(Round 2 was host memory).

### Round 5's other answers (not in conflict)

- **Q3 ionic same-host cross-device RDMA:** no ionic-specific documentation. Best
  candidate is a NIC-agnostic rdma-core issue — a **withdrawn** June-2026
  rdma-next series describing exactly our symptom: for a destination on a
  different netdev of the same host, `addr_resolve_neigh()` copies the *source*
  NIC's MAC, and `validate_ipv6_net_dev()` rejects because `rt6_lookup()` of a
  same-host v6 destination **collapses to `lo`**. Both calls return success and
  no completion arrives. It fits our data: same-device loopback works,
  cross-device does not, and our ionic GID is a **global IPv6** while `mlx5_0`
  uses an IPv4-mapped GID and resolves differently. The author withdrew the patch,
  concluding the fix is **configuration: VRF-per-NIC**. Caveat: the series is
  about `rdma_cm`, and our non-`-R` run failing too is the teammate's inference,
  not established. Confidence MEDIUM on mechanism, HIGH that it is undocumented
  for ionic.
- **Q4 ionic GPUDirect (HIGH):** confirmed from kernel source — ionic **does**
  implement `ibv_reg_dmabuf_mr` (`.reg_user_mr_dmabuf`), but via
  `ib_umem_dmabuf_get_pinned`, the **pinned** helper, and `device_cap_flags`
  carries **no ODP**. So dma-buf registration pins VRAM with no dynamic-attach
  path — corroborating our own detector. The "doubles the pool" *magnitude*
  remains the detector's runtime claim, not stated in source.

## Round 6 — first same-node bring-up: SGLang internal port-block collision

`rounds/006-bringup-1/`. Launched with the vendored harness + spur ssh shim,
`MC_LOG_LEVEL=INFO` on both legs.

**Everything upstream of the engines worked.** etcd came up, the shim drove both
`engine.sh` invocations on 276, and the emitted argv was exactly as designed:
prefill GPUs 0-3 / decode GPUs 4-7, both legs `ionic_0-3`, decode carrying
`--disable-custom-all-reduce`, per-role AITER cache, `SGLANG_SIMULATE_ACC_LEN=3.61`.

**prefill-0 died: `exited exit=143 oom=False`.** Root cause, first-hand:

```
zmq.error.ZMQError: Address already in use (addr='tcp://127.0.0.1:29236')
  DetokenizerManager.init_ipc_channels -> get_zmq_socket -> socket.bind
[...] Received sigquit from a child process
RuntimeError: sglang subprocess exited with code -9 before reporting ready
```

SGLang derives an **internal** port block from `--port`
(`server_args.py:836` offsets by a fixed `ZMQ_TCP_PORT_DELTA`; `:846-861`
reserve `port_base+0..NUM_DERIVED_PORTS-1`, where `NUM_DERIVED_PORTS` is 6, or
`6+dp_size` on the rust path). `launch.sh:rows()` hands out
`ENGINE_PORT_BASE+index`, so the two legs got 29001 and 29002 and their derived
blocks **overlapped by 9 of 10 ports**.

This is exactly the hazard `examples/sglang_1p1d_glm5.2/engine/up.sh:73-77`
warns about ("the second leg dies at bind with *port_base at N is not
available*"). Round 3 saw that warning and **mis-scoped it** to the KV-event
ports, because the kit's own defaults are 30000/30001 — one apart — and it
therefore looked as though the engine port could not be the problem. The lesson
is that the kit's `PREFILL_PORT`/`DECODE_PORT` spacing is not a safe precedent.

**Fix (vendored only):** `ENGINE_PORT_STRIDE`, applied to the **engine port
only** in `launch.sh:rows()` and mirrored in `tools/agentx_env.py:load_topology`
(which independently re-derives the URL from `ENGINE_PORT_BASE`). Defaults to 1,
so every cross-node deployment is unchanged. Set to **256** in
`config.yihou.sn.p4d4.sh` → prefill 29001, decode 29257. Verified: no listener in
either derived block on 276.

## Round 7 — second bring-up: port fix holds, HIP path CONFIRMED LIVE, RCCL fails

`rounds/007-bringup-2/`. Same command plus the stride.

**Two wins, both first-hand in the real deployment:**

1. **No ZMQ collision.** Round 6's fix holds.
2. **`transfer_engine_impl.cpp:412 HIP transport installed for intra-node GPU
   P2P` printed on BOTH legs.** This is the live confirmation that Round 4's
   claim survives contact with the actual sglang-wired deployment, and it
   settles the Round-5 conflict in Round 4's favour on the *installation*
   question. Decode also logged
   `transfer_engine_impl.cpp:274 Topology discovery complete. Found 1 HCAs` —
   so `Found N HCAs` is **N>0**, clearing mission obstacle 11's worry. (Why 1
   and not 4, with `MC_TE_FILTERS=ionic_0-3`, is **not** established — plausibly
   per-rank selection. Noted, not concluded.)

**Decode got much further** — all four DP ranks reached
`Load weight begin. avail mem≈285 GB`.

**prefill-0 died differently:**

```
RuntimeError: NCCL error: unhandled cuda error
crsuse2-m2m-276:717:1687 [0] [FATAL ERROR]: HIP failure: 'invalid argument'
  initialize_model_parallel -> init_model_parallel_group -> GroupCoordinator
  -> PyNcclCommunicator -> ncclCommInitRank
```

i.e. RCCL communicator init for the TP group, ~40 s after start. Checked and
ruled out as the immediate cause: both legs report `nccl_port: None` and
`dist_init_addr: None` in `server_args` (derived at runtime, and the stride now
separates the derived blocks), and a port clash would surface as a bind/connect
error, not `HIP failure: invalid argument`.

**Not yet diagnosed.** Next: re-run with `NCCL_DEBUG=INFO` (the error text itself
asks for it). The failure is ~40 s in, so this iteration is cheap — far cheaper
than the ~30 min CUDA-graph phase further on.

## Round 8 — NCCL_DEBUG: the failure is RCCL's P2P HIP-IPC call

`rounds/008-nccl-debug/`. Re-ran round 7 with `NCCL_DEBUG=INFO`,
`NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV`. Cheap: the failure lands ~40 s in.

Exact line, first-hand:

```
rccl/hipify/src/transport/p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument
[FATAL ERROR]: HIP failure: 'invalid argument'
```

RCCL builds its rings/trees fine and logs `Check P2P Type isAllDirectP2p 1`,
then fails inside the **P2P transport** when taking a HIP IPC memory handle.

**Why this matters beyond RCCL:** `hipIpcGetMemHandle` is the *same primitive*
mooncake's HIP transport uses (`HipTransport: hipIpcGetMemHandle failed` is in
the binary). A host-level problem with HIP IPC would break the same-node KV path
too, not just collective init.

Clean asymmetry in the same run, same host, same second:

| leg | GPUs | `hipIpcGetMemHandle` warnings | progress |
|---|---|---|---|
| prefill | 0,1,2,3 | **1** | died at `ncclCommInitRank` |
| decode | 4,5,6,7 | **0** | reached `Load weight begin` on all 4 DP ranks |

Ruled out as the immediate cause: a port clash (both legs report
`nccl_port: None` / `dist_init_addr: None`; the stride separates the derived
blocks; and a clash would be a bind/connect error, not `invalid argument`).

## Round 9 — differential: swap the GPU sets between the legs

`rounds/009-gpu-swap/`. One variable: which physical GPUs each leg gets. Roles,
env, ports, rails all unchanged.

`PREFILL_GPU_DEVICES=4,5,6,7 DECODE_GPU_DEVICES=0,1,2,3`

**Result: BOTH legs cleared NCCL init and reached `Load weight begin`, with zero
`hipIpcGetMemHandle` warnings on either.**

| round | prefill | decode | outcome |
|---|---|---|---|
| 007 / 008 | GPUs 0-3 (started first) | GPUs 4-7 | prefill fails in RCCL P2P |
| 009 | GPUs 4-7 (started first) | GPUs 0-3 | **both pass** |

**DO NOT read this as "GPUs 0-3 are bad".** The experiment does **not** isolate
the variable:

- GPUs 0-3 carried the **decode** leg successfully in round 9, so the GPU set
  alone is not the discriminator.
- GPUs 4-7 carried the **prefill** leg successfully in round 9, so the role
  alone is not the discriminator either.
- Only the (prefill, GPUs 0-3, started-first) combination has ever failed, and
  the swap is **N=1**.

Live candidates, all unresolved: start ordering / a race between the two
containers' RCCL inits; something specific to the first communicator created on
the host; or plain run-to-run variance. A confirmation re-run of the original
GPU assignment is worth one cycle before this is written up — but it is deferred
behind the Phase-A deliverable, which this round is now on track to produce.

Recorded here so the packup cannot later claim a clean cause.

## Round 9 (continued) — both legs init fully, then ALL schedulers segfault

Round 9 went much further than any prior round, then failed in a new place.

**Reached, on both legs:** weights loaded, KV cache allocated
(`117.39 GB` per rank, `max_total_num_tokens≈2.28M`, ~41.7 GB free),
decode completed CUDA-graph capture, both HTTP servers came up.

**Then both legs died the same way.** `docker exec ps` shows, in **each**
container, all four `sglang::schedul` processes as `<defunct>` zombies with the
`data_parallel_controller` spinning at 87% CPU and the parent waiting forever:

```
Fatal Python error: Segmentation fault          (x4 per leg, both legs)
Current thread ...:
  File ".../managers/scheduler.py", line 1894 in run_event_loop
  File ".../managers/scheduler.py", line 5985 in run_scheduler_process
```

So the segfault is native (the Python frame is merely where the loop was), and
it lands **at the very start of the scheduler event loop** — prefill 07:04:38,
decode 07:13:25, i.e. each leg ~1 s into its own event loop, not at a shared
wall-clock moment.

**Consequence:** `/health` stays `503` forever and `Start of pd disaggregation
warmup` never completes — 23 min with no progress. That presented as a hang; it
was not one. Evidence that settled it, gathered live rather than by waiting for
the 1800 s timeout:

- prefill's `transfer_worker` threads sit idle in `utils.py:87 get` — no work
  queued, so nothing was stuck *in* a transfer;
- the bootstrap server listens on `10.245.152.249:28998` with **no established
  connections**, and no mooncake RPC connections exist;
- **zero** `POST /generate` completed on either leg.

**Ruled out (cheaply, with evidence):**

- *The failed-session probe.* `conn.py:2479` waits `interval` (30 s) **before**
  the first pass, and the crash is at +0.4 s. Not it.
- *The KV-events ZMQ publisher.* Prefill has `--enable-kv-events`; decode has
  them **off** and segfaults identically.
- *A pairing failure.* `_send_disaggregation_warmup_requests`
  (`http_server.py:2168`) posts to the engine's **own** URL with
  `bootstrap_host=FAKE_BOOTSTRAP_HOST` — warmup is self-contained and needs no
  peer. The hang is downstream of the dead schedulers, not a pairing problem.
- *A carry-over from earlier rounds.* Rounds 7/8 never reached KV-cache
  allocation (torn down first), so this is the **next** failure in the
  sequence, not something that was happening all along.

**Not yet diagnosed.** `dmesg` is not readable through `spur exec`, so the
faulting library is unknown. Leading hypothesis to test, from
`infera/tools/preflight/mooncake_mode.py`: this deployment is **mode C** —
`ibv_reg_dmabuf_mr` on ionic, which has **no ODP**, so the driver **pins** the
registered region; the tool predicts exactly `SIGSEGV`/HIP-209 on a large pool,
and our pool is 117 GB/rank with 41.7 GB spare. Against that hypothesis: the
proven cross-node runs used the same registration mode at the same
`mem_fraction 0.85`. What is new here is that **one host now carries both legs'
registrations at once**.

**Next round: decomposition.** Bring up the **prefill leg alone** (no decode
container). If it still segfaults, the fault is intrinsic to the configuration
and co-location is exonerated; if it survives, the fault is an interaction
between the two co-located engines. One variable, ~5 minutes.

Also queued for the next launch: restore `MC_LOG_LEVEL=INFO` (dropped in round
9 in favour of `NCCL_DEBUG`), so the `selectTransport route=` / `hip_reachable=`
lines are available if we get as far as a transfer.

## Round 5/4 conflict — RESOLVED by experiment; the leader's hypothesis was wrong

`analysis/mooncake_samehost.yihou.md` §15, `analysis/upstream_samenode_research.yihou.md` §Q1a.

My reconciliation hypothesis — that sglang's default `MOONCAKE_PROTOCOL="rdma"`
would leave segments advertised rdma-only, so the hip path could never be
chosen — is **falsified**.

**The discriminator experiment (first-hand, in our image):** a server registered
a GPU buffer under the default `protocol="rdma"`; a client with **zero RDMA
transport installed** (`Found 0 HCAs`, hip only) **read it successfully**. An
rdma-only segment would have failed that client with "transport rdma not
installed". Therefore `register_memory` on **GPU** memory advertises the segment
under **every installed transport that can serve GPU memory**, independent of
the init protocol string. Separately, `protocol="hip"` was shown to move GPU
data end-to-end GPU0↔GPU4 **cross-NUMA**, which also closes Round 4's
outstanding "byte movement unproven" item.

Settled, and now recorded in `spec/mission.md`:

- **Do NOT set `MOONCAKE_PROTOCOL="rdma,hip"`** — unnecessary, and a deviation
  from the proven cross-node config.
- **`MC_USE_HIP_IPC` defaults to IPC mode** (`hipIpcGetMemHandle` /
  `hipIpcOpenMemHandle`); `=0` opts into fabric/VMM mode (upstream #1344).
- **`MC_DISABLE_HIP` defaults to hip enabled**; setting it disqualifies hip even
  same-host (#2753). Never set it here.
- Locality keys purely on the **host substring of the segment name**
  (`segmentHost` → `hostEquals`), port stripped — not GID, not subnet, not a
  probe.
- **The routing decision is not logged even at `GLOG_v=5`.** Do not expect a
  "chose hip" line at bring-up. (This corrects the earlier suggestion that
  `MC_LOG_LEVEL=TRACE` would surface it.)

Two honest corrections from the teammates, both recorded: `node-prep`'s original
`"rdma,hip"` claim was upstream inference cross-checked only against symbol
presence, not a live segment dump — its probe container had **0 HCAs** because a
bare container skips `engine.sh:106-110`'s host-`libionic` mount, so no rdma
transport ever installed. And `upstream-research`'s "compiled out" framing was
wrong: the `MC_*_HIP_TRANSPORT` gate it quoted targets the **vLLM** image line,
while our sglang build is on the `ENABLE_MULTI_PROTOCOL=ON` line.

Bonus operational finding: **137 has a node-IP hairpin block** (a plain listener
on `0.0.0.0` is unreachable via the node's own primary IP; `127.0.0.1` and the
RoCE IP work) — host iptables, not mooncake. **276 does not have it**, verified.
So co-location's same-host handshake to `HOST_IP` is clear on our target node.

### The mode-C / OOM hypothesis is weakened

`upstream-research` read the kernel path ionic's `.reg_user_mr_dmabuf` →
`ib_umem_dmabuf_get_pinned` actually takes: `umem_dmabuf.c` sets
`.allow_peer2peer = true` and calls `dma_buf_pin`, which pins the KV-pool BO
**in place, in VRAM** under PCI P2PDMA — no second copy, no host migration. So
the detector's "pins" is right but its "doubles the pool" is not borne out on
this stack, and the cross-node 0.85 runs are expected behaviour rather than
luck. Confidence HIGH on pin≠copy; the in-VRAM-vs-migrate half is MEDIUM and
would be closed by a `rocm-smi` VRAM check across `reg_dmabuf`.

## Round 10 (planned) — `MC_USE_HIP_IPC=0`

The settled facts above connect straight to the live segfault:

- `MC_USE_HIP_IPC` defaults to **IPC**, so mooncake's HIP transport takes
  `hipIpcGetMemHandle` when it registers the KV pool;
- registration happens exactly when the scheduler event loop starts — where all
  8 schedulers segfault, ~1 s in;
- and Round 8 already caught **RCCL** failing on the *same primitive*:
  `hipIpcGetMemHandle failed : invalid argument`.

One primitive, two independent failures, on this host. Hypothesis:
`hipIpcGetMemHandle` is unreliable here, plausibly above some buffer size — our
pool is **117 GiB per rank**, with 8 ranks registering on one host.

**Cheapest test first.** A full bring-up costs ~25 min to reach the segfault; a
standalone 2-process probe costs minutes and answers the same question. So the
launch is **held** and `node-prep` has exclusive use of all 8 GPUs on 276 to
sweep `hipIpcGetMemHandle` by buffer size (4 MiB → ~117 GiB), same-NUMA and
cross-NUMA, with and without `MC_USE_HIP_IPC=0`.

(Leader's own scheduling error, recorded: the first version of that task told
the teammate the GPUs were free *and* that I was launching in parallel. Caught
and corrected before either started.)

## Team poll — 2026-09-22 ~07:51Z (leader)

`ListAgents`: `node-prep` running (1 h) on the `hipIpcGetMemHandle` sweep with
exclusive use of 276's 8 GPUs; `image-xfer`, `harness`, `upstream-research` all
idle with their deliverables landed.

**No new problem observed this poll.** Recording only, per the poll rule.

Polled `node-prep` for a first-hand status and re-stated the priority order —
(1) does the primitive work at all on 276, (2) the size threshold, (3) whether
`MC_USE_HIP_IPC=0` avoids it — and added one refinement: since the segfault hits
**every** rank on **both** legs within ~1 s, if the per-buffer sweep comes back
clean then the concurrency/aggregate dimension (8 processes x ~117 GiB on one
host) is the interesting one rather than a bonus.

Launch remains **held** behind this probe, deliberately: a bring-up costs ~25
min to reach the segfault, the probe costs minutes and answers the same
question.

## Rounds 11-14 — the segfault is NOT about co-location. It is node 276.

Four rounds, each isolating one variable. This supersedes the same-node framing
of rounds 6-10 **for this failure** (the same-node harness work itself stands).

| # | where | what changed | segfault? |
|---|---|---|---|
| 11 | 276, both legs | + `MC_LOG_LEVEL=INFO`, `GLOG_v=2` | decode **yes** (x4); prefill died differently (`Rank 0 scheduler died during initialization (exit code: -3)`, SIGQUIT) |
| 12 | 276, **decode alone**, no co-tenant engine | co-location removed | **yes** (x4) |
| 13 | 276, decode alone, **JIT cache wiped** | cache freshness | **yes** (x4) |
| 14 | **137**, decode alone, same harness/config/image | the node | **NO — `The server is fired up and ready to roll!`** |

Three exonerations, in order:

1. **Co-location is not the cause.** Round 12: one leg, alone on the box, still
   segfaults on all four ranks.
2. **JIT-cache corruption is not the cause.** Round 13 wiped
   `/tmp/aiter-jit-yihou-sn-*` (as root, via a throwaway container — the
   user-level `rm` could not remove root-owned `.cuda.o` files, which would have
   left round 13 ambiguous had I not noticed) and it still segfaults.
3. **Mooncake is not the cause.** Round 11's instrumented log puts the last
   mooncake line (`transfer_engine_impl.cpp:412 HIP transport installed`) at
   07:40:50 and the segfault at 07:44:28 — 3.6 min apart, nothing in between.

**The remaining variable is the node.** Identical vendored harness, config,
image and shape: 137 serves, 276 segfaults.

Environment compared and found **identical** on both: kernel `6.8.0-107-generic`,
amdgpu `6.14.14`, `amdgpu-dkms 1:6.14.14.30100100-2212064.24.04`,
`hsa-rocr 1.18.0.70001-42~24.04`, `rocm-core 7.0.1.70001-42~24.04`. So this is
**machine state, not software version**. And 276's GPU stack is not broadly
broken — `rocm-smi`, `perftest`, and cross-process torch CUDA IPC all work there.

Also settled along the way: **`Found 1 HCAs` is correct**, not a defect —
`topology.cpp:127 Device ionic_3 port 1 is available` in DP rank 3's process,
i.e. one HCA **per rank**, exactly as the `RDMA_DEVICE` map prescribes. Mission
obstacle 11 is closed.

**Not diagnosed:** *why* 276 fails. Candidates, none tested: residual KFD/driver
state from the other tenant's vLLM that was killed at Round 1; the still-running
co-tenant `yzhou_model`; something in 276's k8s/flannel/spur host stack.

**Decision put to the user**, because it changes which machine the deliverable
comes from: move the same-node P4D4 to 137 (free, 8 GPUs, image present, and
just demonstrated to run this exact configuration), keep debugging 276, or clear
276's remaining tenant and retry once.

## Rounds 15-17 — same-node 1P1D P4D4 is UP on 137, and the real same-node bug is a RCCL init RACE

**Round 15** (`rounds/015-samenode-137/`): full same-node bring-up on 137,
config default GPUs (prefill 0-3 / decode 4-7).

- **prefill-0: HEALTHY.** `End of disaggregation warmup` → `The server is fired
  up and ready to roll!`, `/health` 200. No segfault. So the round-12/13
  segfault really was node 276 and nothing else.
- **decode-0: FAILED** with the round-7/8 `NCCL error: unhandled cuda error`.

### The discriminator, finally

| round | node | prefill | decode | NCCL failure hits |
|---|---|---|---|---|
| 7 / 8 | 276 | GPUs 0-3 (1st) | GPUs 4-7 | **prefill** |
| 9 | 276 | GPUs 4-7 (1st) | GPUs 0-3 | neither |
| 15 | 137 | GPUs 0-3 (1st) | GPUs 4-7 | **decode** |

Rounds 7/8 and 15 have the **identical** GPU assignment yet a **different leg**
fails. So it is neither the GPU set nor the role — **it is a race**: two 4-GPU
RCCL communicators initialising concurrently on one host trip over each other,
which is exactly the shape of `p2p.cc:256 hipIpcGetMemHandle failed : invalid
argument`. Cross-node this cannot happen, because the two legs' RCCL inits are
on different machines. **This is a genuine same-node-specific defect** — unlike
the segfault, which was not.

This also retires the round-9 "GPU swap fixed it" reading, which I had already
refused to bank: the swap did not fix anything, it just lost the race
differently. Recording that the caution was warranted.

**Round 16** (`rounds/016-stagger-137/`): start decode only **after** prefill is
healthy, so the two RCCL inits cannot overlap. **Zero segfaults, zero NCCL
errors.** Both legs reached `The server is fired up and ready to roll!`; both
logged `HIP transport installed for intra-node GPU P2P` and `Found 1 HCAs`
(correct: one per rank).

**Deployment live on one node:** router `/health` 200, `/v1/workers` lists two
active workers — prefill `10.245.153.247:29001` and decode
`10.245.153.247:29257`, dp_size 4 each, same IP, same machine.

**End-to-end smoke test passed:** a chat completion went prefill → decode →
response. Output text is garbled (`1!Sans!Sans!…`) — **expected, not a bug**:
`SGLANG_SIMULATE_ACC_LEN=3.61` forces the accept count by construction, the same
waiver the 20260920 packups carry.

One incidental bug worth remembering: the router first died with
`HFValidationError … not a local path`. Cause was **zsh**, not docker — in
`-v $M:$M:ro`, zsh applied its history modifier `:r` (strip extension) to `$M`,
turning `/…/GLM-5.2-MXFP4` into `/…/GLM-5` + `o`. Quote it: `-v "${M}:${M}:ro"`.

**Round 17** (`rounds/017-agentx-c40/`): the Phase-A deliverable, AgentX
CONC=40 fast, simulated acceptance ON (3.61). Running; warmup returning with
`errors=0`.

### Fix to fold into the harness

`launch.sh` starts both legs back-to-back, which is what loses the race. The
harness needs an `ENGINE_START_STAGGER_S` between instances (default 0, so
cross-node is unchanged) — or, better, wait for instance *i* to answer `/health`
before starting *i+1* when two instances share a node. Round 16 achieved the
stagger by hand; it must be codified before this is reproducible.

## Round 17 — DELIVERABLE, and pack-up

**AgentX CONC=40 fast passed** on `crsuse2-m2m-137`, both legs on one node.
`AgentX passed: rounds/017-agentx-c40/out/agentx_conc40.json`.

843 requests profiled, **0/843 aiperf request errors**, 2
`InvalidInferenceResultError` dropped. **82,291 tok/s total, 10,286 tok/s/chip**
over 1,226.8 s. TTFT p50 15.03 s / p90 64.73 s; ITL p50 10.97 ms;
interactivity p50 91.2; server GPU cache hit 87.7 %.

`spec_accept_length` on all four decode DP ranks: **3.725 / 3.575 / 3.450 /
3.625**, mean **3.59** — consistent with the forced 3.61, and **every rank
moved**, so none is the idle-gauge artifact.

Full write-up with scope limits: `results/RESULT.md`.

**Packed to** `yihou/glm52.samenode-p4d4.agentx-c40.packup_20260922/` — 590 KB,
60 files, README / REPRODUCE / environment / notes / spec / scripts / patches /
results / analysis / logs. Verified self-contained: every path `REPRODUCE.md`
names exists inside the kit.

Per the user's decision the kit is **lean**: the 576 MB
`server_metrics_export.json`, the 28 MB timeslices CSV and the full per-round
server logs stay in this (gitignored) workspace; `logs/key-excerpts.md` carries
the load-bearing lines instead.

`diff -rq` confirms the vendored harness differs from the tracked repo in
**exactly three** files — `tools/topology.py`, `launch.sh`, `tools/agentx_env.py`
— one per same-node defect, each a no-op cross-node. The tracked repo was never
modified.

### Still running

The deployment is **still up on 137** (prefill-0, decode-0, etcd, router) holding
all 8 GPUs, in case Phase B or a sweep follows. Tear down with `stop.sh` when
done, then verify the idle baseline before reusing the node.

## Team poll — 2026-09-22 ~10:55Z (leader)

`ListAgents`: **all five teammates idle**, every deliverable landed —
`node-prep` (`analysis/hardware_prep.yihou.md`, `mooncake_samehost.yihou.md`,
`hip_ipc_limits.yihou.md`), `image-xfer` (`image_transfer.yihou.md`),
`harness` (`harness_samenode.yihou.md`, the vendored patches),
`upstream-research` (`upstream_samenode_research.yihou.md`), `env-collect`
(`environment.md`). Nothing in flight, so the poll is a no-op beyond this record.

**No problem observed.** No new work dispatched: Phase A is complete and packed,
and Phase B (prefill HiCache) is explicitly gated on the user in `spec/mission.md`.

Two open items were put to the user and are awaiting a decision, not a teammate:

1. The deployment is **still up on 137**, holding all 8 GPUs, in case Phase B or
   a concurrency sweep follows.
2. **276's root cause is not diagnosed.** Localised to `scheduler.py:1894` (a
   native HIP stream-handle read, i.e. the signature of prior context
   corruption); the spur job on 276 is retained for that work. `node-prep` has a
   ready next discriminator (`--disable-overlap-schedule`) if we resume it.

**Team poll, ~11:05Z** — unchanged from the 10:55Z poll: all five teammates idle
with deliverables landed, nothing in flight, no problem to record. Second
consecutive quiet poll. The idleness is by design, not a stall: Phase A is
complete and packed, and Phase B is gated on the user by `spec/mission.md`. The
137 deployment is still up (8 GPUs held) pending that decision.

## A unified theory, proposed and refused (2026-09-22, ~11:10Z)

`node-prep` proposed collapsing every failure into one root cause: the RCCL
start race, with `scheduler.py:1894` as its downstream surface — concluding that
276 is fine and would come up clean under the start gate, and that 137 "works
because of the gate, not because it is a healthier box".

**Refused, on our own evidence.** Rounds 12 and 13 were **single-leg** runs on
276 — one `engine.sh decode` started by hand after `stop.sh`, etcd plus one
engine, nothing else on the box. **There was no second communicator to race
with**, and they segfaulted anyway, all four ranks, both times. Round 14 was the
same single-leg run on 137 and served. The start gate is irrelevant to all three:
with one leg there is nothing to gate against.

So the honest split is **two** findings, not one:

1. **The RCCL start race is real and fixed.** Start-order dependent, impossible
   cross-node, resolved by the gate. Explains rounds 7/8 and 15, and plausibly
   the concurrent-start segfaults of rounds 9 and 11 via exactly `node-prep`'s
   mechanism — a failed `hipIpcGetMemHandle` poisons the HIP context and the
   other leg dies at the next bare native touch.
2. **276's solo-leg segfault is unexplained.** The `1894` analysis still fits as
   the *mechanism*, but the *cause* of the corruption in a process with no peer
   is not the race and is not identified.

Identical software stacks on 137 and 276 do not settle this — identical versions
is what makes it interesting, because whatever differs is **state**, not version.
That is why `environment.md` leads with the comparison table.

No run was spent on 276: Phase A is delivered and the user has not authorised
more time there. Asked `node-prep` to re-read rounds 12/13/14 for any way a race
could reach a solo leg, and otherwise to narrow the synthesis in
`analysis/hip_ipc_limits.yihou.md` to the concurrent-start rounds and record
12/13 as unexplained. The decisive future test, if 276 time is ever authorised,
is a **gated** two-leg run there — under this reading it is predicted to fail,
so it discriminates in both directions.

## Session paused — 2026-09-22 ~11:20Z (user decision)

User chose: tear down 137, and stop here pending their wake-up. Phase B is **not**
started — `spec/mission.md` gates it on the user and they did not authorise it.

**Teardown verified, per mission rule 8:** `stop.sh` removed the router, both
legs and etcd; `docker ps` shows no `yihou` containers on 137; GPUs settled to
**2,272 MB total (≈284 MiB/card) with no KFD processes**. 276 is likewise clean
— no leftover containers from this session's runs, same 2,272 MB baseline. The
spur job 165913 on 276 is retained as the placeholder the user asked for.

The 10-minute mission re-injection and 20-minute team-poll cron jobs were
cancelled, since both would keep firing against a paused task.

**State at pause.** Phase A delivered and packed. Phase B not started. Two
findings stand, one fixed and one open (see `notes.md`). The five teammates are
idle with every deliverable landed.
