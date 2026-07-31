# GLM-5.2-MXFP4 long-context correctness on sglang — single-node PASS, PD **BUG FOUND**

**Ran:** 2026-07-29 · **Nodes:** chi2867 (10.2.122.44) + chi2879 (10.2.122.10), 8× MI355X gfx950
**Engine:** sglang 0.5.15.post1 · **Model:** `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4`

Goal: verify **long-context (65K+) output correctness**, with DP-attention, on (a) single node and
(b) mooncake-RDMA PD disaggregation.

## Headline

| # | Config | Correctness | Verdict |
|---|--------|-------------|---------|
| A | single-node, DPA=1, rc6 image, 65K / 119K needle | 3/3 + 3/3 | ✅ PASS |
| B | single-node, DPA=1, conc=32 ISL=32K stress | 64/64, 0 err / 0 retract / 0 OOM | ✅ PASS |
| C | single-node, pd-unified image, DPA=0, 18 novel cold shapes | 0 corrupt | ✅ PASS |
| D | single-node, pd-unified image, DPA=1, 18 novel cold shapes | 0 corrupt | ✅ PASS |
| E | **PD mooncake RDMA, DPA=0 and DPA=1, long context** | **corrupt** | ❌ **BUG** |

## ✅ FIXED — Infera commit `854ebf70` (`mooncake_early_send_wait_event.diff`, bundled here)

Root cause: `prefill.py` records a CUDA event as a write barrier before handing pages to the
transfer worker, **but only the `mori` backend ever read it**. `mooncake/conn.py` had no
`synchronize()`, and the overlap-scheduling path that moves non-final chunks did not even record
the event. The transfer worker reads device memory outside the CUDA stream → it races the forward
still writing those pages. The last chunk is always correct because the sampling path already
synchronizes.

Verified on our stack (gfx950, v0.5.15.post1, MXFP4 — the patch targets v0.5.16 but applies with
offsets), at the **exact baseline config** (chunk 2048, overlap ON, mooncake RDMA):

| arm | corrupt reasoning |
|---|---|
| unpatched baseline | 2-4 / 10 |
| **patched, canary ×20** | **0 / 20** |
| **patched, the 6 baseline lengths** | **0 / 6** (was 5/6 corrupt), incl. pt=51389 = 25 chunks |

The `CHUNK=262144` workaround below is therefore obsolete — chunk size is a pure performance knob
again. Everything below documents the independent investigation that located the same bug.

## The bug, in one line

> **Under PD disaggregation, a prompt that needs MORE THAN ONE prefill chunk gets corrupted KV.
> Single-chunk prefill is always correct. Single-node is immune.**

Output degenerates into token salad while the transport looks perfectly healthy — decode receives
the right token COUNT, `#retracted-req: 0`, no `KVTransferError`, no error in either leg's log.

```
'The1. The maintenance log2</think>The2.0</think>The3</think></think></think>'
```

**The corruption boundary is exactly `chunked_prefill_size`:**

| chunk (per rank) | last clean | first corrupt |
|---|---|---|
| 2048  (DPA=1, default) | 2693  | 2801  |
| 16384 (DPA=0, default) | 31845 | 32548 |
| 32768 (DPA=1, `CHUNK=262144`) | 32599 | 34885 |

Every "length threshold" we measured was just *where the prompt stops fitting in one chunk*.

⚠️ **Under DPA, sglang divides `chunked-prefill-size` by `dp_size`** (262144 → 32768/rank). That
silently shrinks the safe single-chunk window 8×, which is why the DPA leg looked like it broke
"much earlier".

## Verified mitigation (workaround, not a fix)

Set `--chunked-prefill-size` ≥ `longest_prompt × dp_size` so prefill never chunks:

| test | default chunk | big chunk |
|---|---|---|
| identical canary ×10 | 4/10 corrupt | **0/10 corrupt** |
| 6 baseline lengths | 5/6 corrupt | **1/6** (the one > 32768 chunk) |

Costs KV memory and hurts high-concurrency batching. The real fix belongs in how the PD path
assembles/transfers KV across prefill chunks.

## How we got there (and two wrong turns worth knowing)

The failure *looks* like a length threshold, then like a per-shape first-touch bug. Both are
artifacts. What actually discriminates is **latency on the identical prompt**:

- ~2.2 s → always correct — prefill log: `#new-token: 64, #cached-token: 30400` (radix hit, 1 chunk)
- ~6.4 s → always corrupt — prefill log: `#new-token: 2048 × 15, #cached-token: 0` (multi-chunk)

Same canary sent 10×: `OK, GIB, OK, OK, MISS, OK, GIB, OK, GIB, OK`. Prefix-cache hits were
masking the failures and made it look self-healing.

**Wrong turn 1 — "length threshold".** Killed by the DPA=0 control arm: the threshold *moved*
with the chunk size.
**Wrong turn 2 — "per-shape first-touch".** Killed by the canary test: a byte-identical prompt
flips verdict run to run, so it is neither per-shape nor monotonically degrading.

Full evidence chain in `notes.md`.

## ⚠️ Gotcha: never probe a PD leg directly

Sending a request straight to `:30000` / `:30001` **kills the leg**:

```
AssertionError: req.bootstrap_room should not be None. Do not send requests directly to
prefill or decode instances; send to the router instead.
```

The DP controller SIGQUITs every scheduler. Always go through the router (`:8002`). This cost us
one full relaunch; `scripts/prefill_logprob_test.py` is kept **only** as a record of the trap.

## Folder map

- `REPRODUCE.md` — exact steps to reproduce both the bug and the mitigation
- `notes.md` — full debug log, every hypothesis and what killed it
- `scripts/` — launchers + probes (see REPRODUCE for which does what)
- `results/` — raw JSON from the final (big-chunk) runs
- `logs/` — trimmed leg logs (py-spy dumps and JIT noise stripped)

## Localisation: it is the PREFILL side, not the wire

Identical canary ×10, all at per-rank chunk 2048:

| arm | corrupt |
|---|---|
| mooncake RDMA (baseline) | 4/10 |
| mooncake RDMA + `--disable-chunked-prefix-cache` | 7/10 |
| **TCP transport (`MC_FORCE_TCP=1`)** | **6/10** |
| mooncake RDMA, per-rank chunk 32768 (single-chunk) | **0/10** |

- **Transport exonerated**: TCP shares no code with the RDMA path (no dmabuf, no peermem, no
  ionic) yet corrupts at the same rate with the same bimodal latency signature.
- **Chunked prefill compute exonerated**: the single-node control arm ran the same per-rank
  chunk 2048 and served up to pt=91386 (~45 chunks) with 0 corruption.

What remains is **PD-mode prefill KV bookkeeping** — how the prefill leg stages/indexes KV across
chunks when `disaggregation_mode=prefill`. Not the wire, not chunking itself, but the intersection.

## Open questions

1. Which specific step: KV written into the transfer-staging layout, sender-side chunk indexing,
   or the per-chunk completion/commit signal? (needs a source read + per-chunk KV dump)
2. mori transport and MTP untested; only mooncake + TCP covered here.
3. Is the DPA `chunk / dp_size` division intended?
