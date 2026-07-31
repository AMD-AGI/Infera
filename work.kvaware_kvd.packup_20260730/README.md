# kvaware + kvd on the sglang engine — support survey, a bug fix, and a PD+DPA baseline

**Ran:** 2026-07-30
**Author:** yihou workspace
**Engine:** sglang 0.5.15.post1 via `infera/engine-sglang:pd-unified`
**Nodes:** chi2879 (prefill) + chi2867 (decode), 8× MI355X each, ionic RoCE v2
**Status:** **PASS** — all four deliverables landed, including the headline
correctness run for *kvaware+kvd together on GLM-5.2* (4/4 on real RDMA).

## Goal

Answer two questions the user asked: (1) what does infera actually support for
**kv-aware routing** and **kvd** on the **sglang** engine — and which parts are
sglang's, which are infera's own, which are infera's sglang adapter; (2) run a
**PD + DP-attention** deployment with both switched on and check correctness.

**Success criteria.** For (1): a source-verified support matrix with file:line
provenance. For (2): `scripts/probe.py` scoring **≥3/4** on temp=0 factual
prompts through the router, on real RDMA.

## Result

| # | Deliverable | Criterion | Actual | Verdict |
|---|---|---|---|---|
| 1 | Support survey, 3-layer split | source-verified | `results/support_matrix.md`, every claim file:line'd | ✅ |
| 2 | `free_tcp_port_block` collision bug | found → fixed → regression-tested | patch + 4 passing tests; fails on pre-fix code | ✅ |
| 3 | GLM-5.2 2-node PD+DPA **baseline** (kvaware/kvd OFF) | ≥3/4 | **4/4**, real mooncake RDMA (`MC_FORCE_TCP`=0) | ✅ |
| 4 | GLM-5.2 2-node PD+DPA **with kvaware+kvd ON** | ≥3/4 | **4/4**, real RDMA, kvaware+kvd both verified live | ✅ |

Deliverables 3 and 4 are a matched pair: same nodes, same transport, same probe,
only the two switches differ — and both score 4/4. That is the differential that
makes the claim meaningful. (The earlier single-node MVP produced garbled output,
but its baseline was *equally* garbled, so the garbling belonged to same-host PD
+ `MC_FORCE_TCP`, not to the features.)

| 5 | **Step 2** — prefix-reuse workload on the same deployment | kvd serves; correctness holds | **32/32**, kvd `gets 0→170, hits=170, misses=0`, 573 MB | ✅ |
| 6 | **Step 3** — cross-restart reuse (kill engine, keep kvd) | new engine reads old engine's blocks | **+170 gets / +170 hits / 0 new sets**, GPU cache empty | ✅ |
| 7 | **Step 4** — routing effect with 2 decode workers | policy changes where requests land | kv-aware **17/0**, round-robin **+6/+8** | ✅ |
| 8 | **Step 5** — real block-device L3 | O_DIRECT on a block device | **partial** — found+fixed bug #2; verdict correctly stays buffered on SATA md0 | ◐ |

Step 1 left kvd wired but idle (`gets=0 sets=0`) — four short prefix-disjoint
prompts give an offload path nothing to do. Step 2 fed it a ~6200-token shared
prefix across 4 sessions × 4 turns and kvd went to **170 gets / 170 hits / 0
misses**, 573 MB resident. That closes the "connected vs. actually serving"
question. See `results/step2_prefix_reuse.txt`.

Steps 3 and 4 then removed the two confounders step 2 left behind:

- **Step 3** killed the engine but kept kvd. GPU VRAM back to idle, so the GPU
  radix cache cannot explain anything — yet the fresh engine scored **+170 gets
  / +170 hits / 0 misses with `sets_total` perfectly flat**. It read 170 blocks
  it never wrote, from a store its predecessor left behind. That is kvd's actual
  selling point, demonstrated.
- **Step 4** added a second decode worker so a routing decision could exist.
  Same workload, same workers, only the policy differs: **kv-aware sent all 32
  requests to decodeA (17/0); round-robin split them (+6/+8)**. The scorer is
  deciding, not just instantiated.

**Still not claimed:** the 2.7× latency drop is **GPU radix-cache reuse, not
kvd** (kvd's counters are flat across the warm run). And the *prefill* weight
(20.0) remains unmeasured — there is only one prefill worker.

**Deliverable 8 turned up a second real bug.** Bind-mounting a host directory in
so kvd's L3 sits on a block device *still* reported `devices = [(none)]`:
`findmnt` prints a bind mount's subpath in brackets (`/dev/md0[/mnt/...]`) and
`_findmnt()` handed that whole string to `lsblk`, which rejects it. So **any**
bind-mounted L3 silently got buffered I/O. Fixed — see `patches/0002-note.md` —
A/B-verified on the node, 2 new tests. Two things it does *not* settle:

- The verdict on this box is still `buffered`, **correctly** — md0 is a raid1 of
  two SATA SSDs, and the classifier prefers buffered for SATA (readahead win).
- A stock container can't see `/dev/md0` at all; bind-mounting the *path* does
  not expose the *device*. You also need `--device=/dev/md0`. That is a
  deployment-recipe trap, not an infera defect.

True NVMe O_DIRECT remains untested: this node's eight 7 TB NVMe drives are
unmounted, and one already holds another team's 120 GB kvd store — not mine to
touch on a shared cluster.

### The finding most likely to bite someone

A decode leg sets `disable_radix_cache=True` on its own, and sglang forbids
`enable-hierarchical-cache` alongside it — so **kvd is illegal on a decode leg by
default**. What legalises it is `--disaggregation-decode-enable-radix-cache`,
which infera auto-appends *only when kv-events are enabled* and the transfer
backend is mooncake.

**So turning kvaware off silently disables kvd on the decode leg.** Verified
against sglang's own arg validator — matrix in `results/support_matrix.md`.

## How to reproduce

See `REPRODUCE.md`. §A is a desk check of the bug+fix (no cluster); §B is the
sglang arg matrix (one node, no model load); §C is the 4/4 two-node baseline
(~6 min cold start); §D turns the switches on; §E–§G are the prefix-reuse,
cross-restart and routing experiments.

## Folder map

- `MILESTONES.md` — the full arc in order, how each step removed the previous
  step's confounder, and a blunt list of what is still not established
- `REPRODUCE.md` — ordered, copy-pasteable steps for all sections
- `environment.md` — exact HW/SW, fabric, image digest, git SHA, external paths
- `notes.md` — **what is NOT proven**, gotchas, wrong turns. Read before quoting.
- `scripts/` — every script that ran, verbatim
- `patches/` — **two** infera fixes, each with a what/why/how note and tests.
  `patches/apply_all.sh <repo>` installs both (patches alone miss the new test
  file, which is untracked upstream).
- `results/` — the support matrix and the run evidence
- `logs/` — raw engine logs, one dir per experiment, both PD legs each. See
  `logs/README.md` for grep recipes that re-derive every claim in `results/`.
- `spec/` — the originating task, as the user posed it

## Caveats in one place

- Deliverable 4 is open; **there is no correctness result for kvaware+kvd on GLM-5.2.**
- Whether infera's own KV-event probe plane attaches at all is **unverified** — it
  looks for the RadixCache in the wrapper process while sglang runs in a subprocess.
- kvd was shown *connected*, never shown to *serve a hit*.
- The patch in `patches/` is **not committed** — it lives in the working tree of
  branch `yihou.dev.glm5.2.mxfp4.experiment`.
- `manual/features/kv_cache_offload.md:14-18` still claims kvd is "vLLM only" —
  stale, reported, not fixed here.
- GLM-5.2 engine logs are packed (`logs/`, 2.2 MB, per-experiment dirs). The
  **Qwen3 MVP** logs are gone — that container was removed before the GLM runs;
  excerpts survive in `results/kvaware_kvd_activation_evidence.txt`. kvd-daemon
  and router logs were container-local and also lost at teardown; the numbers
  they carried are quoted verbatim in the step 2–4 result files.
