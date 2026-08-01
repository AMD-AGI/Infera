# Notes — the traps, in the order they are likely to bite

Written as what / why / how / context. §3 and §4 are the two that actually broke this
run; the rest are inherited traps that were avoided because a previous run paid for them.

---

## 1. `temperature: 0` + MTP is indistinguishable from KV corruption

**What.** Use the model's own `generation_config.json` — **temperature 1.0, top_p 0.95**.
`fixlen_round.sh` passes them explicitly.

**Why.** Greedy decoding sends this reasoning model into repetition on a long prompt, and
EAGLE/MTP *amplifies* it: the draft model predicts the loop perfectly, so `accept len`
pins at its maximum (4.00) and the response runs to `max_tokens`.

**Context.** On a previous run this produced a needle test reading 3/5 with `</think>`
×1264 — exactly the mooncake early-send signature — while every check said the fix was
in. It was settled by control, not argument: MTP off → 5/5; MTP on with official
sampling → 5/5.

**The lesson that generalises:** `accept len: 4.00` is **bad news**, not good. It is a
symptom of a loop the draft model predicts perfectly, not evidence MTP is healthy.
Healthy here was 1.52–2.67.

---

## 2. Cold start is 4–9 minutes, and never grep the log for readiness

**What.** Poll the HTTP endpoint. Do not `grep "ready to roll"`.

**Why.** The leg logs are **appended** across runs, so a grep matches a *previous* run's
line within seconds and the caller proceeds against an engine still loading weights.

**Context.** Prefill ≈ 4 min; **decode ≈ 8 min** because MTP adds a second CUDA-graph
capture pass for the EAGLE draft model. During that pass the log shows `[aiter]
[fused_moe] using 2stage default…` repeatedly with no progress bar — that is normal, not
a hang. Observed sequence: weights → tilelang JIT → `Capturing batches (bs=N)` → draft
capture → `registered with etcd`.

---

## 3. `MC_GID_INDEX` is node-dependent — the bug that killed the first boot

**What.** The kit hardcoded `export MC_GID_INDEX=1`. It is correct on chi2879 and
**wrong on chi2867**. `patches/0001` discovers it instead.

**Why.** Both nodes expose two RoCE v2 GIDs per ionic port — link-local `fe80::` and
routable `fd93::` — at **different indices**:

| node | idx0 | idx1 | idx2 |
|---|---|---|---|
| chi2879 | `fe80::` | **`fd93::`** | empty |
| chi2867 | `fe80::` | **empty** | **`fd93::`** |

**How diagnosed.** The asymmetry was the whole diagnosis: the prefill leg on chi2879 had
**zero** mooncake errors while the decode leg on chi2867 failed on all 8 devices. Anything
that were a code or image defect would have hit both.

    rdma_context.cpp:1132  GID is NULL, please check your GID index by specifying MC_GID_INDEX
    rdma_context.cpp:200   Failed to open device ionic_4 on port with GID 1
    RuntimeError: Mooncake Transfer Engine initialization failed.
    RuntimeError: Rank 0 scheduler died during initialization (exit code: -3)

**Do not "fix" it with idx0.** The link-local GID is not routable across the fabric and
crashes MoRI at `ionic.cpp:414`.

**Why earlier kits never saw it.** Every prior cross-node run used chi2879 as prefill and
either never got decode past init on chi2867, or paired hosts where idx1 happened to be
routable. A latent node-dependency, not a regression.

---

## 4. Prefill HSA OOM at long ISL is an **activation** problem — lower the memfrac

**What.** At ISL 155,000 × conc 32 the prefill leg aborted:

    rocdevice.cpp:3582  HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 1203306 MB
    Fatal Python error: Aborted

Fix: prefill `mem-fraction-static` **0.88 → 0.80** (`patches/0002`).

**Why it is not KV exhaustion.** The scheduler lines right before the abort read
`token usage: 0.01–0.05` — the KV pool was essentially **empty**. What was large was
`#pending-token: 2,486,426` at `#queue-req: 16`. With DP-attention at dp8 each rank holds
its own 8192-token chunk activations and a 155K prompt is 19 chunks; under the
prefill-delayer's batching the transient activation peak exceeds what
`1 − mem_fraction_static` leaves outside the static reservation.

**The direction is the counter-intuitive part** — and the phase tells you which way:

| phase | symptom | fix |
|---|---|---|
| decode | retract / `get_cpu_copy NotImplementedError` | **raise** mem-fraction-static |
| prefill | `HSA_STATUS_ERROR` / `Aborted` | **lower** it |

**Context.** Cost KV pool 3,260,672 → 2,829,952 tokens/rank (−13 %) for activation
headroom 32.5 → 54.6 GB (+68 %). Afterwards p90/c32 ran 32/32 clean and c64/c128 followed
with **zero** `HSA_STATUS_ERROR`.

**Read the cascade correctly.** The decode leg also died — but its last healthy log line
(`token usage 0.20`, `accept len 3.4–3.7`) is timestamped **after** the prefill abort. It
died in the cascade. Only prefill's memfrac was changed; treating both as OOM would have
made the fix a two-variable change with no evidence for half of it.

---

## 5. kvd's L3 tier fills the node's **root disk**, not `/mnt/vast`

**What.** After round 8 every `docker exec` failed:

    OCI runtime exec failed: write /tmp/runc-processNNN: no space left on device

`/` was at **100 %** of 838 GB. Fix: `--long-bytes` 512G → 64G (`patches/0003`).

**Why.** `--long-path /tmp/kvd-long` is **inside the container**, so the L3 spill tier
lands in the container's writable layer on the node's root disk. `bench_run`'s layer had
reached **263 GB** against a budget larger than the disk.

**How to reclaim safely on a shared node.** Verify ownership before deleting anything:
243 GB was our `bench_run` tier and 20 GB our exited `merged_run`, both confirmed via
`docker inspect` / overlay2 path mapping. Nothing belonging to another user was touched.
`docker ps -s` shows per-container writable size; `du -sh /var/lib/docker/overlay2/*/diff/tmp/kvd-long`
maps it to a layer.

**Note kvd already knows.** Its self-check logs
`storage_classify: lsblk returned no devices for source='overlay'` and
`ssd region long on overlay — file-tier reads would compat-fallback under hipFile`. It has
the information to warn about an oversized budget on an overlay filesystem, and doesn't.

**Better fix not taken here:** point `--long-path` at `/mnt/vast`. Skipped deliberately —
it changes the storage medium under L3 (NFS vs local NVMe) and would make these numbers
non-comparable to every prior validated run.

---

## 6. The router KV view is per-process — measure it *after* driving traffic

**What.** Always: restart router → drive traffic → *then* read the view.

**Why.** The view lives in the router **process**. A freshly restarted router reports
nothing for every worker — which looks exactly like the kv-aware/bigram fix having
failed, since that count is its discriminator.

---

## 7. Router module name, and the Rust backend's missing routes

**What.** It is `python -m infera.server`. It is **not** `infera.router` — that is a
package with no `__main__` and fails with a message that reads like a missing dependency.

**Rust-specific.** `--router-backend rust` execs `/usr/local/bin/infera-router`, which
routes only `/health`, `/v1/workers`, `/v1/models`, `/metrics`, `/v1/{chat/,}completions`
(`rust/router/src/handlers.rs:33-38`). Consequences:

- **No `/v1/admin/cache-view/<w>?dp_rank=N`.** `total_blocks()` exists but is unrouted,
  and `/metrics` emits only `active_workers` + `uptime`. The kv-aware signal must come
  from the policy log line (`policy.rs:314`) — `scripts/cache_view.py` parses it.
- **No `--kvd-socket-path`** (`launch_rust.py` builds argv explicitly), so
  `POST /v1/cache/prewarm` returns 503. Neither bench uses prewarm.

Neither is a defect for this experiment; both are capability deltas worth knowing before
you go looking for an endpoint that isn't there.

---

## 8. kv-aware picks *look* broken under a single-prefix probe

**What.** During bring-up all 36 picks landed on `#dp0` with a single stable prefix.
Under sweep traffic (1,120 picks) **all 8 ranks** were used on both legs.

**Why.** With one shared prefix, every request's best-overlap rank legitimately *is* the
same one, and cost ties break to the first candidate. Concurrency is what makes
`active_blocks` diverge and the scorer spread.

**How to read the spread.** It is asymmetric on purpose:

    prefill (w=20.0):  dp0 175 · dp1 106 · dp2 64 · dp3 35 · dp4 41 · dp5 39 · dp6 59 · dp7 41
    decode  (w=2.0):   dp0 141 · dp1  59 · dp2 61 · dp3 62 · dp4 60 · dp5 59 · dp6 58 · dp7 60

Prefill is **skewed** (chasing cache locality at w=20.0); decode is **near-uniform**
(routing by load at w=2.0). That contrast is the weights doing their job. Prefill
`cache_hits` reached 2,422 blocks — real reuse, not blind balancing. Decode `cache_hits`
= 0 is by design: the decode radix cache is off under MTP.

**Do not run a live two-binary A/B to "prove" the bigram fix.** It has already produced a
confidently wrong answer once — the unpatched leg read `cache_hits=51` because that
prefill leg had no MTP, so it emitted plain ints and never exercised the bigram path.
Use a revert/restore control in unit tests instead.

---

## 9. A frozen tqdm bar is not a hung run

**What.** At conc ≥ 32 the bench's progress bar can sit at `1/32` for five minutes.

**Why.** `bench_serving` ticks tqdm on **completion**. With 32 requests of 155K tokens all
dispatched at once, nothing completes until the first one finishes end-to-end.

**How to tell the difference.** Look at the engine, not the client:

    strings <prefill.log> | grep -oE '#running-req: [0-9]+|#queue-req: [0-9]+' | tail -4

Movement in `#queue-req` / `#pending-token` means it is working. This run also had a real
crash hiding behind a frozen bar (§4) — so check the engine log for `HSA_STATUS_ERROR`
before assuming either.

---

## 10. Editing a script on `/mnt/vast` does not change what the container runs

**What.** `reset_node.sh` `docker cp`s the leg script into the container **once, at
container creation**. Editing the shared-fs copy between rounds silently runs the *old*
script. `start_leg.sh` now re-copies on every launch.

**Context.** This cost a full cold start: after fixing `MC_GID_INDEX` and re-staging, the
log still read `GID index: 1`. The fix was correct; the container was running the
pre-fix copy.

**Same class as the stale-`__pycache__` trap** that has invalidated a full experiment on
this stack before: *the artifact you edited is not the artifact that ran.* Verify the
thing that executes — bytecode, or in this case the in-container copy:

    docker exec bench_run grep -c _gid_discover /glm52_leg.sh    # want 2

---

## 11. What this phase does **not** establish

- **No kvd performance claim.** No kvd-off A/B was run. The restart-replay proves kvd
  *serves* (gets 102 / hits 102 / sets flat), not that it makes anything faster.
- **The cache-hit column is not a cache result.** `--dataset-name random` synthesises each
  prompt independently — there is no shared prefix by construction. The 0–50 % values are
  the previous round's radix residue, confirmed by `host_cached_tokens ≈ 0` in every
  round (`results/fixlen_summary.csv`): the hits are GPU-radix, not kvd L2/L3. The 88–90 %
  target belongs to Case A.
- **p50 and p90 are not comparable to each other.** p50 ran at prefill memfrac 0.88, p90
  at 0.80 (13 % less KV pool). Within-pair comparisons are clean.
- **The server was never reset between rounds** — deliberate (one frozen deployment), but
  each round inherits the previous round's radix tree. Per-round kvd snapshots bound it.
- **The kvaware weight sweep (w ∈ {1.0, 20.0}) was not run** — the OOM debugging consumed
  its budget. The weights' *effect* is still visible in §8's pick distributions.
- **`max_running_requests` is 2048 in the args but the engine reports 256 effective.** Not
  investigated; relevant to any conc=128 interpretation.

---

## 12. Cluster hygiene

The nodes are **shared** and their slurm hold belongs to `yeandy-debug`, not us.

- Kill only our own containers (`bench_run`, and the earlier `merged_run`) and our own
  sglang/router/kvd processes. **Never `scancel`** — dropping a hold has previously got a
  node marked down and it was costly to recover.
- Do not prune images or wipe `/tmp` wholesale. Prove a container is yours via
  `docker inspect` (Binds / Env / Created) before removing it.
- Do not fill the disks (§5). `df -h /` before a long run.
