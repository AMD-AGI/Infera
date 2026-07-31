# conc=128 "digit loop" reproduction — DPA + PD mooncake, GLM-5.2-MXFP4

Goal: reproduce the user-reported failure — under **conc=128 stress on the DPA + mooncake-PD
path, SOME requests' outputs degenerate into a "digit loop"** (numbers repeating endlessly).
Distinguish it from the already-FIXED multi-chunk KV race (Infera 854ebf70).

Nodes: prefill = **chi2867** (10.2.122.44), decode = **chi2879** (10.2.122.10).
Image: `infera/engine-sglang:pd-unified-waitevent` (854ebf70 wait_event patch already in).
Decision (user): patched image only, workload **ISL=1024 / OSL=1024** (exp07 recipe),
verdict = **capture every output** and classify (not just bench_serving completion rate).

## Why this is a meaningful reproduction

- exp07 (2026-07-27) ran this exact shape at conc=128 and reported **512/512 PASS** — but it only
  checked *completion*, plus a 4/4 short probe. A digit loop completes normally and would not
  have been caught.
- The longctx packup (2026-07-29) found and fixed the multi-chunk KV race, and noted a
  **residual `TAIL_REPEAT`** mode: reasoning coherent + needle correct, only the post-`</think>`
  tail loops. It was explicitly "not chased". The reported digit loop is plausibly that mode,
  amplified by concurrency.
- Running on the **patched** image separates the two: any corruption seen here is NOT the
  fixed KV race.

## Prior state found on the nodes (before this round)

Both nodes already had a live `pd_uni` container on `pd-unified-waitevent`, started 12:04,
prefill args: ctx=131072 chunk=16384 **max-running-requests=64 cuda-graph-max-bs=64**, DPA=1,
mooncake, 8 ionic NICs; router alive on :8002. **max-running 64 caps the batch at 64**, so
conc=128 would just queue — must relaunch with the exp07 capacity knobs.

## Round 1 — relaunch at exp07 1k/1k capacity, then conc=1 baseline + conc=128 full capture

Hypothesis: the digit loop is concurrency-triggered (batch/CUDA-graph/DP-shard dependent), not
prompt-shape dependent, so the same prompts that are clean at conc=1 will corrupt at conc=128.

Change (single): relaunch both legs with ctx=32768, chunk=65536 (8192×TP8 → 8192/rank),
max-running=2048, cgbs=128 — i.e. exp07's config, only the image differs (patched).

Verdict method: `stress_capture.py` sends N temp=0 prompts with a verifiable answer at a given
concurrency, saves **every** output, and classifies:
- `DIGIT_LOOP`  — a short numeric/near-numeric span repeated many times (the reported bug)
- `CORRUPT_REASONING` — token salad in the reasoning (the old KV race; should be 0 on patched)
- `TAIL_REPEAT` — coherent + correct, only the post-`</think>` tail loops
- `TRUNCATED` / `CLEAN`

Arms: conc=1 (baseline, same prompt set) → conc=128. Same prompts, same seeds, temp=0, so any
delta is purely concurrency.

Gotcha hit: `up_conc128.sh`'s kill step used `bash -c '...'` nested inside `J()`'s own `'...'`
→ quoting collapsed, old legs survived, new legs died instantly on
`port_base at 30234 is not available`. Also a stale host-side `sglang::router` (pid 984817) held
:8002 and `pkill -f sglang_router` did not match it (process name is `sglang::router`). Both fixed.

### Result — REPRODUCED, and it is concurrency-triggered

| arm | conc | n | duration | CLEAN | DIGIT_LOOP | CORRUPT_REASONING | TAIL_REPEAT | TRUNCATED |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 1 | 32 | 109.5 s | **32** | 0 | 0 | 0 | 0 |
| **stress A** | **128** | 512 | 34.4 s | 506 | **4** | **1** | 1 | 0 |
| replay of A's 6 failures, 4× each | 1 | 24 | 69.1 s | **24** | 0 | 0 | 0 | 0 |
| **stress B** (fresh salt, all-novel prompts) | **128** | 512 | 44.7 s | 502 | **4** | **3** | 2 | 1 |

**~1.2–2% of requests degenerate at conc=128; 0% at conc=1 on byte-identical prompts.**

### The signature is unusually clean

- **Every** bad output — and **only** the bad outputs — has `finish_reason=length`
  (`completion_tokens` = 1024, the cap). All 1008 good outputs are `finish=stop`, median 155
  tokens, `</think>` count ≤ 1.
- All bad outputs **contain the correct needle** (`needle=True` for 15/16). The model finds the
  right answer, emits a coherent first paragraph, then **fails to stop** and spins on
  `</think>` / the answer digits until it hits the token cap:
  `'...is exactly 45203.</think></th' … '33333333333…'`
  `'...</think>68209682096820968209…'`
- `</think>` repeats 27–839× per bad output.
- So this is **not** corrupted KV. Reasoning and retrieval are correct; the **stop/EOS decision**
  fails under concurrency. It is the `TAIL_REPEAT` mode the longctx packup saw post-patch and
  explicitly did not chase — here it is, amplified by concurrency and running long enough that
  the loop drifts into digit salad, which is what the user described as "数字循环".
- 854ebf70 (mooncake wait_event) is present in this image (`conn.py:1233 wait_event.synchronize()`)
  and the multi-chunk KV race is NOT what we are seeing: prompts are ISL=1024 = single chunk.

### Where the failures land

| run | bad idx | in the first 128 (the truly-saturated wave) |
|---|---|---|
| A | 71, 84, 101, 112, 119, 121 | **6/6** |
| B | 64, 67, 76, 100, 104, 106, 116, 120, 124, **422** | **9/10** |

15/16 land in `idx<128` — the only window where all 128 requests are genuinely in flight
together (afterwards clients refill one at a time and the effective batch is decode-limited).
**The trigger scales with the concurrent decode batch size, not with prompt content or position
in the stream.**

### Classifier note (false positive found and fixed)

First pass flagged idx=460 as DIGIT_LOOP: a legitimate 1484-char chain-of-thought quoting its
answer 12×. Rule (c) now also requires the repeated literal to occupy >25 % of the output. The
7-case regression suite passes and the fix was applied retroactively to the stored results
(`verdict_v2`). Counts above are post-fix.

## Round 2 — CONTROL: single-node mix, no DPA, no PD, conc=128

Hypothesis under test: is the trigger "GLM-5.2 decode at batch 128" (would reproduce anywhere) or
does it need PD and/or DP-attention?

Change: same patched image, same container, same GLM DSA env, same TP8 / ctx 32768 /
**per-rank** chunk 8192 (= the PD run's 65536÷dp8, so the compute shape matches) /
cuda-graph-max-bs 128 / max-running 2048 / gmu 0.88. Removed exactly two variables:
disaggregation and DP-attention. Client hits the server directly (no router — it is not a PD leg).
Same prompts, same salts, same classifier.

### Result — does NOT reproduce on single-node

| topology | DPA | arm | conc | n | dur | CLEAN | DIGIT_LOOP | CORRUPT | TAIL_REP | finish=length |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PD mooncake | 1 | baseline | 1 | 32 | 109.5 s | 32 | 0 | 0 | 0 | 0 |
| PD mooncake | 1 | replay of failures ×4 | 1 | 24 | 69.1 s | 24 | 0 | 0 | 0 | 0 |
| **PD mooncake** | 1 | **conc128 A** | 128 | 512 | 34.4 s | 506 | **4** | **1** | 1 | **6** |
| **PD mooncake** | 1 | **conc128 B** (fresh salt) | 128 | 512 | 44.7 s | 502 | **4** | **3** | 2 | **10** |
| single-node mix | 0 | baseline | 1 | 32 | 83.5 s | 32 | 0 | 0 | 0 | 0 |
| **single-node mix** | 0 | **conc128 A** | 128 | 512 | 60.7 s | **512** | **0** | **0** | **0** | **0** |
| **single-node mix** | 0 | **conc128 B** (fresh salt) | 128 | 512 | 63.9 s | **511** | **0** | **0** | 0 | 1* |

\* the single `finish=length` on single-node B is `idx=422`, classified `TRUNCATED` — the same idx
also went TRUNCATED on PD run B. That prompt genuinely reasons past 1024 tokens; it is a probe
artifact, not the bug (no loop, no repeated digits, needle absent because it never got there).

**1024 single-node requests at conc=128 → 0 digit loops. 1024 PD requests at the same conc,
same prompts → 16 `finish=length`, of which 15 are degenerate loops.**

So the failure needs **PD and/or DP-attention**; plain GLM-5.2 decode at batch 128 is fine, and
`cuda-graph-max-bs=128` is exonerated as a standalone cause (same value on both arms).

## Round 3 — single-node with DPA=1 (the one-variable split)

Round 2 removed disaggregation AND DP-attention together. This arm puts DPA back on, still
single-node: `--dp-size 8 --enable-dp-attention --ep-size 8`, `SGLANG_DP_USE_GATHERV=1`,
chunk 65536 (→ per-rank 8192, same as the other two arms). Everything else unchanged.
Simple experiment only, no debugging.

Gotcha (again): `docker exec -d $CTR bash -c "... > $LOG"` through nested ssh let the redirect
evaluate on the OUTER shell — the log ended up containing only the ssh banner (76 bytes) and the
server never launched. Fixed by staging `single_dpa1.sh` as a FILE and running `bash /script`,
the same remedy the exp07 notes prescribe.

### Result — still does NOT reproduce

| topology | DPA | arm | conc | n | CLEAN | DIGIT_LOOP | CORRUPT | TAIL_REP | finish=length |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **PD mooncake** | 1 | conc128 A | 128 | 512 | 506 | **4** | **1** | 1 | **6** |
| **PD mooncake** | 1 | conc128 B | 128 | 512 | 502 | **4** | **3** | 2 | **10** |
| single-node | 0 | conc128 A | 128 | 512 | 512 | 0 | 0 | 0 | 0 |
| single-node | 0 | conc128 B | 128 | 512 | 511 | 0 | 0 | 0 | 1\* |
| **single-node** | **1** | **conc128 A** | 128 | 512 | **512** | **0** | **0** | **0** | **0** |
| **single-node** | **1** | **conc128 B** | 128 | 512 | **511** | **0** | **0** | **0** | 1\* |

\* both are `idx=422`, `TRUNCATED` — the same prompt that also went TRUNCATED on PD run B. It
genuinely reasons past 1024 tokens: no loop, no repeated digits. Probe artifact, appears in every
arm, ignore.

**DP-attention alone does not cause it.** 1024 more stressed requests, 0 digit loops. Combined
with round 2: **2048 single-node requests at conc=128 across both DPA settings → 0 failures**,
vs 16 `finish=length` / 15 loops in 1024 PD requests on the same prompts.

## Where this leaves it

The trigger requires **PD disaggregation**. Neither high decode batch, nor `cuda-graph-max-bs=128`,
nor DP-attention reproduces it on a single node.

Not yet run (deliberately — user asked for simple experiments, not a debug session):
- **PD with DPA=0 at conc=128** — would say whether DPA is even needed once PD is present, or
  whether plain PD at high concurrency is enough.
- If PD+DPA=0 also reproduces, the suspect narrows to the decode leg's sampling/stop path for
  transferred requests. `</think>` repeating 27–839× with correct reasoning and a correctly
  retrieved needle points at the EOS/stop check, not at the logits.
