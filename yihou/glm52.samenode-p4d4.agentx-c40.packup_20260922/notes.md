# notes — gotchas, wrong turns, and retractions

The ordered narrative is `working_process.md`. This file is the part worth
re-reading: what bit us, what we got wrong, and what is still open.

---

## Retracted conclusions — read these first

Four things were believed at some point in this session and are **not true**.
They are recorded because a reader of the round log will meet them on the way.

### 1. "The rail split is the point of the same-node shape" — WRONG

The first plan gave prefill `ionic_0-3` (NUMA0) and decode `ionic_4-7` (NUMA1) so
the legs never shared a NIC. Then a `perftest` differential on **two independent
hosts** showed same-host RDMA between two **different** ionic devices moves zero
traffic — QPs connect and exchange QPN/PSN/RKey/GID, then no completion ever
arrives and `ib_write_bw` exits 1 with no bandwidth row — while same-**device**
loopback runs at ~40 GB/s and the cross-host control at 42 GB/s with the
identical command. `ionic_0 <- ionic_1` (same NUMA0, same PCI domain) fails too,
so the discriminator is **different device**, not different socket.

**And then that whole line turned out to be moot**: same-node KV never touches
the NIC (see below). The rail configuration in the shipped config is inert for
this shape. It is left as-is because changing it would add a variable for no
benefit.

### 2. "The GPU swap fixed the NCCL failure" — WRONG, and retracted

Round 9 swapped the GPU sets between the legs and both cleared NCCL init, which
looked like a fix. It was not. Round 15 on a **different node**, with the
**identical** GPU assignment as rounds 7/8, failed on the **other leg**:

| round | node | prefill | decode | NCCL failure hits |
|---|---|---|---|---|
| 7 / 8 | 276 | GPUs 0-3 (started 1st) | GPUs 4-7 | **prefill** |
| 9 | 276 | GPUs 4-7 (started 1st) | GPUs 0-3 | neither |
| 15 | 137 | GPUs 0-3 (started 1st) | GPUs 4-7 | **decode** |

Same assignment, different leg ⇒ neither the GPU set nor the role. It is a
**race** between two concurrent 4-GPU RCCL communicator inits on one host. The
swap did not fix it; it lost it differently. The round-9 note refused to bank the
result at the time, and that caution was right.

### 3. "`hipIpcGetMemHandle` has a size ceiling" — FALSIFIED

The 117 GiB KV pool and RCCL's `hipIpcGetMemHandle failed : invalid argument`
suggested a size limit. Direct testing on 276: `rc=0` at 4 MiB, 1, 16, 32, 64,
100, 110, **117**, 130 and 200 GiB. No ceiling. Also falsified: that the
registered pointers might be VMM/pool offsets — they are per-layer torch
`data_ptr()` bases, and both the VMM post-capture path and the custom mem pool
are off for this deployment.

### 4. "Segments are advertised rdma-only, so hip can never be chosen" — FALSIFIED

This was the lead's own hypothesis, from reading that sglang initialises mooncake
with `MOONCAKE_PROTOCOL` defaulting to `"rdma"`. A discriminator experiment
killed it: a server registered a GPU buffer under `protocol="rdma"`, and a client
with **zero RDMA transport installed** read it successfully — impossible unless
the segment was advertised `"rdma,hip"`. `register_memory` on **GPU** memory
advertises under every installed transport that can serve it, independent of the
protocol string.

---

## Mooncake on one host — the rules, and why

Same-host KV rides **GPU-IPC over XGMI**, not RDMA. `isHipReachableTarget` compares
the **port-stripped host** of the segment name, so two legs on one IP differing
only in port both resolve to the same host; `hip` has priority 4 against `rdma`'s
2, so hip wins. Physically available: every GPU pair reports `True`/`XGMI`,
including cross-NUMA 0↔4.

Three hard rules follow. Each one is a way to break a working deployment:

- **Never set `MC_DISABLE_HIP=1`.** It disqualifies hip even same-host and forces
  the legs onto the cross-device RDMA path that moves zero traffic.
- **Never set `MC_USE_HIP_IPC=0`.** Fabric/VMM mode **segfaults `register_memory`
  at every buffer size** on this stack — 4 MiB included. It is not size-dependent;
  the fabric export path needs `hipMemCreate`/`hipMemMap` memory and the KV pool
  is plain torch/hipMalloc. It also strands VRAM: one such crash left 263 GiB
  held on GPU0 with no KFD PIDs, self-reclaiming only ~20 s after the container
  was removed.
- **`MC_DISABLE_HIP_TRANSPORT=1` in `config.sh:89` is a dead name** — the string
  is absent from the shipped `engine.so`, so it is a no-op. Do **not** "fix" it
  into the real spelling `MC_DISABLE_HIP`; the no-op is why the local path works.

Also: the routing decision is **not logged**, even at `GLOG_v=5`. Do not expect a
"chose hip" line at bring-up.

---

## Gotchas that cost real time

**`Found 1 HCAs` is correct, not a defect.** With `GLOG_v=2` the log reads
`topology.cpp:127 Device ionic_3 port 1 is available` → `Found 1 HCAs` in DP rank
3's process. The `RDMA_DEVICE` JSON maps each local rank to its own device, so
one HCA **per rank** is exactly right. This was carried as a suspected fault for
several rounds before the instrumented run settled it.

**A hung warmup was not a hang.** `Start of pd disaggregation warmup` sat for 23
minutes with `/health` at 503. The schedulers had all **segfaulted**; the parent
was waiting on dead children. `docker exec ... ps` showed four `<defunct>`
zombies and a `data_parallel_controller` spinning at 87 % CPU. Diagnosing this
live — idle `transfer_worker` threads, a bootstrap port listening with **no**
connections, zero completed `POST /generate` — beat waiting out the 1,800 s
timeout. Check for zombies before believing a hang.

**PD warmup does not need a peer.** `_send_disaggregation_warmup_requests`
(`http_server.py:2168`) posts to the engine's **own** URL with
`bootstrap_host=FAKE_BOOTSTRAP_HOST`. So a stuck warmup is never a pairing
problem — look downstream.

**zsh ate a docker mount path.** The router died with
`HFValidationError … not a local path`. Cause: in `-v $M:$M:ro`, **zsh** applied
its history modifier `:r` (strip extension) to `$M`, turning
`/…/GLM-5.2-MXFP4` into `/…/GLM-5` + `o`. Always quote: `-v "${M}:${M}:ro"`.

**Root-owned files in the JIT cache defeat a user-level wipe.** `rm -rf` as
`yihou` cannot remove the `.cuda.o` files the container wrote as root, and it
fails *partially* and quietly. Round 13 would have been ambiguous if that had
gone unnoticed. Wipe through a throwaway root container.

**`pkill -f launch.sh` killed its own shell** — the parent `bash -c` had the
pattern in its command line. Do not pattern-kill from a shell whose command line
contains the pattern.

**The kit's own port spacing is not a safe precedent.** A warning in
`examples/sglang_1p1d_glm5.2/engine/up.sh:73-77` about
`port_base at N is not available` was read as being about the KV-event ports,
because that kit's defaults are `PREFILL_PORT=30000` / `DECODE_PORT=30001` — one
apart. That mis-scoping cost a bring-up. The engine port **is** load-bearing on
one host; what that kit actually has to split is `KV_PUB_PORT`/`KV_SNAP_PORT`,
which our harness already offsets by row index.

---

## The registration-mode contradiction — resolved, and why the config was left alone

`infera/tools/preflight/mooncake_mode.py` warns that on a NIC without ODP (it
names AMD Pensando ionic) `ibv_reg_dmabuf_mr` **pins and doubles** the KV pool
and risks SIGSEGV/HIP-209. Our fleet has no peer-mem module and ODP only on
`mlx5_0`, so by that table we should be running mode B on `mlx5_0` at 200 Gb/s.
We instead run ionic rails with dma-buf on — nominally mode C, uncapped.

That contradiction is resolved in the kernel: `umem_dmabuf.c` sets
`.allow_peer2peer = true` and calls `dma_buf_pin`, which pins the KV-pool BO **in
place, in VRAM** under PCI P2PDMA. **Pinning is not copying.** The detector's
"pins" is right; its "doubles" is not borne out on this stack, which is why the
cross-node runs held at `mem_fraction 0.85` without OOM. Confidence is high on
pin≠copy from the kernel path; the in-VRAM-vs-migrate half is medium and would
be closed by a `rocm-smi` VRAM check across `reg_dmabuf`.

**The bench config was deliberately not changed to match the kit's table.** The
bench config has working runs behind it on this fleet; the table does not.

---

## Still open

- **Why 276 segfaults.** Localised to `scheduler.py:1894`,
  `self.schedule_stream.cuda_stream == self.forward_stream.cuda_stream` — a bare
  native HIP stream-handle read, which is the signature of an already-corrupted
  HIP context surfacing at the next native touch, not a bug in that line. It fits
  the non-determinism (round 9 both legs SIGSEGV; round 11 decode SIGSEGV but
  prefill SIGQUIT/-3) and the fact that the identical binary runs clean on 137.
  **Not diagnosed.** Untested candidates: residual KFD/driver state from the
  other tenant's vLLM that was killed at the start of the session; the
  still-running co-tenant container; 276's k8s/flannel/spur host stack. A cheap
  next discriminator: `--disable-overlap-schedule`, which removes the dual-stream
  path and line 1894 entirely — if it still crashes elsewhere, the corruption is
  upstream of overlap.
- **Whether the RCCL race has a root cause beyond ordering.** The start gate
  avoids it; nothing explains *why* two concurrent inits collide.
- **A paired cross-node control at this exact config.** 136 and 138 were occupied.
- **Peak throughput.** One point at CONC=40. No sweep.
- **Phase B (prefill HiCache on).** Not started.
- **A low-confidence lead, untested:** torch's caching allocator may pack several
  per-layer KV buffers into one segment, so their `data_ptr()`s are distinct
  offsets into the same block. `_registerable_regions` dedups only exact
  `(ptr,len)`, not overlaps, so mooncake could register overlapping regions
  resolving to one IPC block. Whether that misbehaves is unknown.

---

## TWO findings, not one — the distinction the record must keep

Late in the session a unified theory was proposed: that the RCCL start race is
the single root cause, with `scheduler.py:1894` as its downstream surface, so
276 is fine and would come up clean under the start gate. **It was refused, and
then retracted by its author.** The record keeps the split because collapsing it
would make the kit assert something the evidence does not support.

**Refutation, from our own rounds:** rounds 12 and 13 were **single-leg** runs on
276 — one `engine.sh decode` after a full `stop.sh`, etcd plus one engine,
nothing else on the box. **There was no second communicator to race with**, and
they segfaulted anyway, all four ranks, both times. Round 14 was the same
single-leg run on 137 and served. The start gate is irrelevant to all three:
with one leg there is nothing to gate against, so "137 works because of the
gate" is false.

One refinement worth keeping, contributed during the retraction: a solo leg
*does* form RCCL communicators among its own 4 DP ranks — but that intra-leg
init is **identical on both nodes**, so it cannot be the discriminator either.
If intra-leg init were the race, round 14 on 137 would have lost it too.

So:

| # | finding | status |
|---|---|---|
| 1 | **RCCL start race** between two co-located legs. Start-order dependent, impossible cross-node. | **Fixed** by the same-node start gate. Explains rounds 7/8 and 15, and plausibly the concurrent-start segfaults of 9 and 11 via the `1894` mechanism. |
| 2 | **Solo-leg segfault on 276.** No peer, so no race. | **Unexplained.** `1894` still fits as the *mechanism* — a corrupted HIP context surfacing at the next bare native stream-handle read — but the *cause* is unidentified. |

Identical software stacks on 137 and 276 do **not** settle finding 2. Identical
versions is what makes it interesting: whatever differs is **state**, not
version. Untested candidates: residual KFD/driver state from the other tenant's
vLLM that was killed at the start of the session, the still-running co-tenant
container, or 276's k8s/flannel/spur host stack.

**The decisive future test** — worth running only if 276 time is authorised — is
a **gated two-leg run on 276**. Under this reading it is *predicted to still
fail* at `1894`, because finding 2 is node state and the gate only addresses
finding 1. It therefore discriminates in both directions: success would mean the
unified theory was right after all and rounds 12/13 need re-explaining.
