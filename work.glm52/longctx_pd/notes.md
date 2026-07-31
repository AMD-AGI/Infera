# GLM-5.2-MXFP4 PD long-context correctness — chi2867(P) + chi2879(D), mooncake RDMA

2026-07-29. Goal: verify long-context (65K+) output correctness under PD disaggregation,
DPA on and off. Result: **long-context output is intermittently corrupted under PD**, root
cause is a **first-touch / cold-shape bug**, NOT context length, NOT DPA, NOT chunked prefill.

## Setup

- prefill chi2867 (10.2.122.44), decode chi2879 (10.2.122.10), `infera/engine-sglang:pd-unified`
  (streamed 2879→2867, 78GB). Both legs symmetric, `--context-length 131072`.
- mooncake RDMA confirmed: `MC_FORCE_TCP` 0 hits, `HIP dmabuf disabled` ×8 (bare ibv_reg_mr+peermem).
- Router = sglang_router mini-LB on :8002. Short probe 4/4 on both DPA arms.

## The failure

Long-context output degenerates into token salad, e.g.
`'The1. The maintenance log2</think>The2.0</think>The3</think></think></think>'`.
Decode leg logs show KV **count** correct (65088 tokens arrive), `#retracted-req: 0`,
no KVTransferError — the KV is delivered but the generation is wrong.

## Debug path — what each pass ruled out

| pass | change | result | conclusion |
|---|---|---|---|
| 1 | 65K needle, DPA=1 | 0/3 + 0/3 garbage | bug reproduces |
| 2 | length sweep DPA=1 | OK ≤2693, bad ≥2801 | looked like a ~2.8K threshold |
| 3 | **DPA=0** (control) | OK ≤31845, bad ≥32548 | threshold MOVED → not a fixed length, not DPA |
| 4 | same prompt ×5, no flush | 1st garbage, next 4 OK @1.4s | **non-deterministic**; warm repeats always pass |
| 5 | same prompt ×8, flush both legs each time | 8/8 OK | not a stale-cache bug |
| 6 | unique (non-shared) prefixes, 9 lengths | 5/9 garbage | not prefix-cache reuse |
| 7 | identical to 6 but re-run later | 0/9 garbage | **intermittent** — same command, different outcome |
| 8 | 6 rounds × 6 lengths back-to-back | round1 4/6 bad, rounds 2-6 **0/6** | self-heals after first pass |
| 9 | novel unseen lengths on warm server | 5/6 garbage | **novel shape re-triggers it** |
| 10 | immediately repeat those same lengths | 0/6 garbage | shape now warm → correct |

## Root cause

**Per-shape first-touch.** A request whose sequence shape has not been seen before returns
corrupted output; every subsequent request at that same shape is correct. Passes 9→10 are the
clean proof: 6 brand-new lengths on an otherwise-warm server → 5/6 garbage; the identical 6
lengths immediately after → 6/6 correct.

This lines up with the tilelang DSA JIT: prefill log has 692 `compiling for gfx950` events, and
the round-1 corruption window (pass 8) sits inside the compile window (last compile 10:02:13).
The strong hypothesis is that the long-seq DSA kernel is used before its JIT compile/warm has
completed, so the first invocation at a new shape reads a not-yet-valid kernel/buffer.

Why the earlier "length thresholds" were an artifact: sweeps walk lengths in increasing order,
so every point is a *new* shape. The apparent threshold was just where the shape-space stopped
overlapping what warmup had already covered — which is why it moved from 2.8K (DPA=1, chunk 2048)
to 32.5K (DPA=0, chunk 16384).

## CONTROL ARM: single-node, same image → the bug is PD-SPECIFIC

`single_unified_coldtest.sh`: fresh container, **same `pd-unified` image**, same TP8 / DPA=0 /
ctx=131072 / chunk=16384 / gmu=0.88 / DSA env as the PD prefill leg. Only disaggregation removed.
Resolved to the identical `max_total_num_tokens=3713728, chunked_prefill_size=16384`.

First requests after `ready to roll` were lengths never used anywhere in this session, so
guaranteed-novel shapes:

| pass | shapes | result |
|---|---|---|
| A (cold, 6 novel lengths 40K–83K) | never seen | **0/6 gibberish** |
| B (same 6, warm) | now seen | 0/6 gibberish |
| C (12 more novel lengths 33K–92K, back-to-back) | never seen | **0/12 gibberish** |
| needle probe 65K | — | single 3/3, multi 3/3 (the 2/3 print is max_tokens truncation) |

JIT definitely runs on this arm — latency decays 18.2s (first, compiling) → 3.7-6.3s → 1.4s warm —
but it never corrupts. **18 novel cold shapes, zero corruption.**

Conclusion: **the corruption is PD-specific**, not the tilelang DSA JIT per se. Single-node
colocated on the identical image handles novel long shapes correctly. Something in the
disaggregated path — prefill-side KV being read/transferred before the first-touch long-seq
kernel has finished producing it, or the transfer racing the compile — corrupts the KV that
reaches decode. That is consistent with decode seeing the right token COUNT but wrong CONTENT.

## ROOT CAUSE (supersedes the "first-touch" reading above)

The first-touch model was WRONG. It was an artifact of prefix-cache hits masking the failure.

**Multi-chunk prefill under PD produces corrupt KV. Single-chunk prefill is always correct.**

Evidence chain:

1. **Canary test** (`degrade_test.py`): the SAME prompt (identical length, identical salt, so
   identical shape and content) sent 10× gives `OK,GIB,OK,OK,MISS,OK,GIB,OK,GIB,OK`. Not
   first-touch (shape was warm), not progressive degradation (it recovers). Random per-request.
2. **Latency is perfectly bimodal and predicts the verdict**: ~2.1-2.9 s → 6/6 OK;
   ~6.2-6.5 s → 4/4 corrupt. Same prompt.
3. **Prefill log explains the two populations**:
   - fast/correct: `#new-token: 64, #cached-token: 30400` → radix-cache hit, ONE tiny chunk
   - slow/corrupt: `#new-token: 2048 × 15, #cached-token: 0` → full multi-chunk prefill
4. **Sub-chunk lengths are clean**: at chunk 2048, pt=1634/1983 (1 chunk) OK; pt≥2108 (≥2
   chunks) degraded.
5. **Decisive fix test**: relaunch with `CHUNK=262144` → per-rank `chunked_prefill_size=32768`
   (sglang divides chunk by dp_size under DPA, 262144/8).
   - canary ×10: **0 GIBBERISH** (was 4/10), including the slow 21.1 s cold ones
   - the 6 baseline lengths that gave 5/6 corrupt → **1/6**, and that one was pt=36678 > 32768
   - boundary sweep: 30412 / 30758 / 32599 OK; **34885 / 38783 / 42940 / 69149 corrupt**

The corruption boundary moves exactly with `chunked_prefill_size`. Earlier "thresholds"
(2.8 K at chunk 2048, 32.5 K at chunk 16384, now 32.7 K at chunk 32768) were always just
"where the prompt stops fitting in one chunk".

**Mitigation available now**: set `chunked-prefill-size` ≥ the longest prompt × dp_size, so
prefill never chunks. Costs memory and hurts high-concurrency batching, so it is a workaround,
not a fix. The real fix is in how the PD path assembles/transfers KV across prefill chunks.

Single-node is unaffected because it consumes the KV in place — the chunked KV is only wrong
once it goes through the disaggregated transfer.

## Gotcha: never probe a PD leg directly

Sending a request straight to `:30000`/`:30001` **kills the leg**:
`AssertionError: req.bootstrap_room should not be None. Do not send requests directly to
prefill or decode instances; send to the router instead.` → DP controller SIGQUITs everything.
Cost one full relaunch. Always go through the router (`:8002`).

## Hypothesis 2 TESTED: `--disable-chunked-prefix-cache` does NOT help

`NO_CHUNKED_PREFIX=1`, default chunk (per-rank 2048), confirmed active in the leg log
(`disable_chunked_prefix_cache=True`).

Canary ×10: **7/10 GIBBERISH** (baseline with the flag off was 4/10). Same bimodal signature —
fast ~2.8-3.3 s runs correct, slow ~8.0-8.7 s runs corrupt. So the chunked *prefix cache* is not
the culprit; the corruption is in the chunked *prefill compute or its KV transfer*, and the flag
if anything removes a path that was accidentally masking some failures.

Ruled out. Next: localise prefill-side vs transfer-side.

## Localisation: the TRANSPORT is exonerated — bad KV is produced on the PREFILL side

Two independent facts:

1. **Chunked prefill COMPUTE is fine.** The single-node DPA=1 control arm also ran per-rank
   chunk 2048 and served prompts up to pt=91386 — that is ~45 chunks — with **0 corruption**
   (18 novel cold shapes). So chunking itself does not corrupt the KV.
2. **`MC_FORCE_TCP=1` still corrupts: canary ×10 → 6/10 GIBBERISH**, same bimodal split
   (fast ~2.5 s OK, slow ~7.9-8.4 s corrupt). TCP is a completely different data path — no RDMA,
   no dmabuf, no peermem, no ionic. If the RDMA transport were mangling bytes, TCP would be clean.

Corruption rate by arm, identical canary ×10 (all per-rank chunk 2048):

| arm | corrupt |
|---|---|
| mooncake RDMA (baseline) | 4/10 |
| mooncake RDMA + `--disable-chunked-prefix-cache` | 7/10 |
| **TCP transport** | **6/10** |
| mooncake RDMA, per-rank chunk 32768 (single-chunk) | **0/10** |

Conclusion: the fault is in **how the prefill leg produces / stages KV across chunks when running
in `disaggregation_mode=prefill`** — not in the wire. The difference between the clean single-node
arm and the broken PD arm, with chunking held constant, is PD-mode prefill bookkeeping (KV written
into the transfer-staging layout, sender-side chunk indexing, or the per-chunk completion signal).

## FIX VERIFIED — Infera commit 854ebf70 (`mooncake_early_send_wait_event.diff`)

Independent fix by liyingli@amd.com, found after our root-cause. Same bug, and it names the
mechanism more precisely than we had:

`prefill.py` already recorded a CUDA event as a barrier before handing over pages that may still
be under write — **but only the `mori` backend ever read it**. `mooncake/conn.py` had no
`wait_event`/`synchronize()` at all, so on mooncake the barrier never took effect. Worse, the
overlap-scheduling path that actually moves non-final chunks (our suspect L715 /
their L752) did not even *record* an event. The transfer worker reads device memory outside the
CUDA stream → it races the forward still writing those pages. The final chunk is always correct
because it goes through the sampling path, which already has a real `copy_done.synchronize()`.

Not DSA-specific: any PD + chunked prefill + mooncake + overlap deployment is affected. DSA's
sparse retrieval just makes it loud instead of a quiet quality drop.

### Applying to our stack

Patch targets sglang v0.5.16; we run v0.5.15.post1. `git apply` succeeds with offsets
(-40 to -65 lines), all 3 files, and `self.forward_stream` exists in 0.5.15. Baked into
`infera/engine-sglang:pd-unified-waitevent` on both nodes (`docker commit`).

### Result — baseline config, chunk 2048, overlap ON, mooncake RDMA (only the patch differs)

Scored with `score_outputs.py`, which separates the two failure modes the original binary
detector conflated:
- `CORRUPT_REASONING` — the reasoning text is token salad ⇒ the KV read was wrong (the real bug)
- `TAIL_REPEAT` — reasoning coherent + needle correct, only the post-`</think>` tail loops
- `NO_NEEDLE` — coherent but truncated at `max_tokens=96` mid-reasoning (a probe artifact)

| arm | CORRUPT_REASONING | coherent + needle |
|---|---|---|
| baseline (unpatched, TCP) | 2/10, plus 5/10 NO_NEEDLE (many salad) | 3/10 |
| **patched, canary ×10** | 1/10 | 9/10 |
| **patched, canary ×20** | **0/20** | 16/20 (the 4 misses are all 96-token truncation) |
| **patched, the 6 baseline lengths** | **0/6** | **6/6** (was 5/6 corrupt) |

The 6-length arm is the strongest single comparison: identical lengths, 5/6 corrupt before,
**6/6 correct after — including pt=51389 (25 chunks)**.

Before vs after, same failing prompt:

    unpatched: 'The in022.</think>829</think>The8292</think>The8.</think></think>82</think>'
    patched  : 'The user wants to know the calibration constant ... Record SECRET-B: ...
                is exactly 82931. The number is 82931.</think>82931</think></think>82931'

i.e. the reasoning and retrieval are now correct; only trailing `</think>` repetition remains.

### Stress test on the patched build (conc=32, ISL=32K, OSL=256)

ISL=32768 at per-rank chunk 2048 is ~16 prefill chunks per request, so every request in the run
exercises the multi-chunk path the fix targets.

| metric | value |
|---|---|
| successful requests | **64/64**, 0 errors |
| retracts / KVTransferError / OOM | **0** (the one prefill "retract" grep hit is the startup warmup echo, `num_retractions: 0`) |
| duration | 77.1 s |
| input tok/s | 23886 |
| total tok/s | 24073 |
| median TTFT | 23541 ms (P99 61023) |
| median TPOT | 20.7 ms (P99 24.1) |
| peak concurrent | 39 |

Correctness re-checked immediately after the stress, same 6 lengths: **6/6 coherent, 0 corrupt
reasoning** (1 `NO_NEEDLE` = 96-token truncation, 4 `TAIL_REPEAT`). The fix holds under load.

Perf note vs the single-node arm (not a like-for-like comparison — different topology, and the
single-node number was ISL=32K on one node): single-node did 26622 in-tok/s with median TPOT
73.7 ms; PD here does 23886 in-tok/s with median TPOT 20.7 ms. PD's much better TPOT is expected
(decode leg is dedicated); TTFT is much worse (23.5 s vs 19.7 s median, and P99 61 s) because
prefill now also pays the `wait_event.synchronize()` barrier plus the KV transfer. **The isolated
cost of the fix was NOT measured** — that needs patched-vs-unpatched at identical config, which
is only meaningful on prompts short enough that the unpatched build is still correct.

### Caveats

- `TAIL_REPEAT` persists (4/10, 1/20). Reasoning and needle are right, so the KV is right — this
  looks like a separate stop/EOS artifact, not the KV race. Not chased.
- Our earlier "TCP also corrupts 6/10" is consistent with this root cause and worth reporting
  upstream: TCP shares the same `transfer_worker` queue, so it was missing the same barrier.
  It rules out "the RDMA wire mangles bytes" independently of the fix.
- Cost of the fix (the author flags it too): `wait_event.synchronize()` blocks the transfer
  worker, trading some transfer/compute overlap for correctness. Not measured here.
- Verified on gfx950 / v0.5.15.post1 / GLM-5.2-**MXFP4**; the author verified gfx942 / v0.5.16 /
  GLM-5.2-**FP8**.

**The `CHUNK=262144` workaround is no longer needed** — chunk size goes back to a pure
performance knob, and overlap scheduling does not have to be sacrificed.

## Not yet established

- Which side is at fault: prefill writing bad KV for chunks 2..N, or the transfer/reassembly
  losing chunk boundaries. Next probe: compare per-chunk KV on prefill vs what decode receives.
- Whether `--disable-chunked-prefix-cache` or `page_size` interacts (page_size=64 here).
- Whether MTP / mori transport show the same behaviour (only mooncake tested here).
- Whether the DPA chunk-division (chunk/dp_size) is intended — it silently made the usable
  single-chunk window 8× smaller on the DPA legs.

## Files

- `up_dpa_longctx.sh` — brings up both PD legs; `DPA=0|1`, `TAG=` for per-arm logs
- `run_router.sh` — sglang_router in the prefill container
- `pd_leg_dpa_longctx.sh` — leg launcher (unchanged from packup exp07)
- `len_sweep.py` — length sweep + gibberish detector
- `repeat_cold.py` — same prompt N times with optional per-request flush
- `prefix_test.py` — shared vs unique prefix arms
