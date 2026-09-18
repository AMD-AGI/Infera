# Debug loop — garbled MTP decode on P4D4 (continued)

## Target (pass/fail)

`temperature=0`, MTP on, simulated acceptance OFF:

1. `"What is 2+2? Answer with a single digit."` does not degenerate into a
   repeated filler token.
2. `"Reply with exactly: BASELINE_OK_<nonce>"` reproduces the nonce.
3. `spec_accept_length >= 2.0`.

Current state: **FAIL** (carried over from R01).

`accept_length` and `accept_rate` are different metrics; the bar is on length.
Ours measured len 1.25 / rate 0.05.

## Configuration under test

GLM-5.2 MXFP4, 1P1D, prefill crsuse2-m2m-135 / decode crsuse2-m2m-138, GPUs
2,3,4,5 on each, **P4+DPA / D4+DPA (TP4/DP4)**. Same-rail on both halves
(`PD_DP_RANK_AFFINITY=1` + per-GPU `RDMA_DEVICE` map).
`index_share_for_mtp_iteration=false`. Simulated acceptance OFF.

## Materials

1. **Known-bad reference (对拍基准)**: `../yihou-1p1d-c64/debug/r01-first-token/`
   — the recorded degenerate outputs at `max_tokens` 1 / 3 / 8.
2. `/home/yihou/dev/git/infera.glm52.view/glm52.mtp.nextn.fusion.fix.packup_20260918/`
   — fork commit `4350d37c5b`. A lead, **not** a diagnosis: the pack itself
   states it does not explain the P4D4 garbled-decode signature.
3. `bench/glm5p2_pd/issue.md` — prior campaign history (§2.4, Appendix A both
   document garbled output; §2.4's trigger is absent here).
4. No known-good reference program. The user decided **not** to build one (no
   TP8 run, no single-node non-PD run).

## Rounds

| # | Hypothesis | Change | Result |
|---|---|---|---|
| R01 | Is the first token already wrong? | none — probe only | First token OK, all later tokens degenerate. *(old workspace)* |
| R02 | MTP draft/verify is the fault | `DECODE_MTP=0` | **aborted, never completed** — user waived correctness at the time |
| R03 | The draft runs unfused, so its logits are rubbish | `IMAGE` → `...-nextnfix` | **fix took effect; output still garbled.** Acceptance went bimodal *per DP rank* |
| R04 | Fault is inside the MTP path vs plain decode | `DECODE_MTP=0` | **16/16 coherent, nonce reproduced.** Plain decode is clean; the fault is inside the MTP path |
| R05 | An MTP-path kernel/graph differs on our base | their env deltas, one at a time | next |

---

### R03 — apply the NextN shared-experts-fusion fix (`r03-fusion-fix/`)

**Single variable: `IMAGE`.** Everything else identical to the C32 baseline
except simulated acceptance, which must be off for the target to be measurable
at all.

#### Why this round, stated honestly

Established first-hand, in our own image and our own logs:

- The defect **fired on our decode leg**. `run4-full.decode-0.log.gz`: the
  target (`GlmMoeDsaForCausalLM`) logs `Shared experts fusion optimization
  enabled` at 06:47:31; the draft (`GlmMoeDsaForCausalLMNextN`) logs `Config
  does not support fused shared expert(s)` at 06:48:11, the same second as its
  own `Load weight begin`. The prefill leg carries no draft and shows the
  enabled line with no disabled line — a control obtained for free.
- Among the four NextN classes in the image, only GLM DSA's lacks
  `fused_shared_experts_architecture` (`deepseek_nextn.py:246`,
  `dots3_common/nextn.py:139`, `glm4_moe_lite_nextn.py:134` all have it), and
  upstream's own comment at `deepseek_v2.py:2967` names "the NextN drafts, GLM's
  DSA variant" as the cases that must override it.
- `model_config.py::_config_draft_model` rewrites **only** `architectures[0]`,
  leaving `n_routed_experts` / `n_shared_experts` untouched. The target passed
  the whole gate on the same config, so the name compare is the only term that
  can have failed. The argument closes.

**What this does and does not buy.** It cleanly explains `accept len 1.25 /
rate 0.05` — the draft loads its weights under the unfused layout. It does
**not** explain the garbling: under EAGLE a rubbish draft should be *rejected*
and fall back to the target's own token, giving correct-but-slow output. That
link is still missing and is left open on purpose.

**A tempting hypothesis, ruled out before spending a run on it.**
`install_shared_experts_fusion_decision` writes a *global*
(`get_flags().moe.disable_shared_experts_fusion`), so "the draft's decision
clobbers the target's and corrupts the target forward" would have explained
garbling, low acceptance and the correct-first-token all at once. It is wrong:
`eagle_worker_v2.py:281` builds the draft inside `draft_model_build_scope()`,
whose `finally` restores the global, and `moe/utils.py:533` carries an explicit
assertion that the flag is never read inside a forward — so readers are
build-time only and the target's already-built layers are unaffected.

#### Both outcomes are informative

- Output correct → target met.
- Output still garbled but accept length clearly up → the draft is now healthy,
  so the fault is in verify / target-decode. The search space is halved and R04
  becomes sharper, rather than the round being wasted.

Evidence sources this round: component logs (both legs), source read (the gate,
the build scope, the flag readers), program output (probes), plus `/metrics`.

#### Counter-evidence received mid-round, and why the round still ran

A parallel session (`mtp-repeat-debug-eb`, kit at
`/home/yihou/dev/git/infera.yihou.mtp.debug/work/glm52-1p1d-mtp-correctness.packup_20260918/`)
reported the **same** draft-side `Config does not support fused shared expert(s)`
line on a P4D4 run that nonetheless produces correct text and healthy
acceptance. Verified first-hand in their `logs/green-decode-0.log` rather than
taken on trust: both lines present with the same ~40 s gap,
`grep -c SIMULATE_ACC_LEN` = 0 so acceptance is real, `accept len` samples
running 4.99-5.54 at the tail. Their own later correction across all 201
telemetry samples: len min 2.33 / p50 3.86 / max 5.54.

So **a draft without shared-experts fusion is not sufficient, by itself, to
produce our failure.** That is now established.

It does not, however, refute the lead for *our* base, because base image is one
of their own delta rows. Confirmed on both sides: their in-image sglang is
`0.5.19.dev20260916+ge7f7447333`, ours is `0.5.19.dev20260917+ga9fb1c3238` —
different commits. Lining up the four stacks that now exist:

| stack | sglang | shape | draft fusion | outcome |
|---|---|---|---|---|
| ours | `ga9fb1c3238` (0917) | P4D4 | missing | accept 0.05, garbled |
| reference kit | `ga9fb1c3238` (0917) | TP8/DP8 | missing -> fixed | degenerate -> clean |
| peer | `ge7f7447333` (0916) | P4D4 | missing | accept ~3.9, correct |
| the fork's own | (its env) | — | missing -> fixed | entirely benign |

The reading consistent with all four is that the missing fusion is harmful **on
the 20260917 nightly specifically** and benign elsewhere — which is what the
reference kit itself says ("The two environments disagree on impact, and both
facts are true"). A *reading*, not a conclusion: neither we nor they have the
A/B. R03 is that A/B, on our base. The peer has amended their kit to scope their
claim to 20260916.

Unplanned benefit: this gives the differential-comparison reference the user
declined to *build*. It cost no cluster time because someone else already ran it.

#### A build failure worth recording — mine, not the fix's

The first `Dockerfile.yihou.nextnfix` verified the fix **inside** the build by
importing the class. That cannot work: importing `sglang.srt.models.glm4_moe`
pulls in `sglang.kernels...fp8_kernel`, which at module-import time calls
`torch.cuda.get_device_properties(0)` (`srt/utils/common.py:1082`), and
`docker build` has no GPU — `RuntimeError: No HIP GPUs are available`. The
applier itself had already succeeded; only my verification step failed, and it
failed loudly, which is the good case.

Fixed by moving the import check **after** the build into a GPU-enabled
`docker run`, which is strictly better anyway: it tests the image as it will
actually be run. Both nodes then verified `draft NextN:
GlmMoeDsaForCausalLMNextN`, target unchanged, sglang still `ga9fb1c3238`.

#### R03 result so far (launched 09:05 UTC)

Acceptance check 2 **passes**:

- decode: `Shared experts fusion optimization enabled` now appears **twice**
  (09:06:29 target, 09:07:08 draft, the latter immediately preceding the
  `GlmMoeDsaForCausalLMNextN` weight load), and `Config does not support fused
  shared expert(s)` is **gone**.
- prefill: still once. The control holds.
- Corroboration nobody asked for: draft weight footprint moved **6.90 -> 6.92
  GB**. The fused layout is not just logged, it is allocated.

Awaiting CUDA graph capture for checks 3-5.

#### Prepared for R04, if needed

The peer's `patches/0001-engine-sh-spur-fixes.patch` adds role-scoped
`{PREFILL,DECODE}_EXTRA_ARGS` / `_EXTRA_ENV` escape hatches to `engine.sh`
(default empty, so a no-op when unused). That makes bisecting their env deltas —
`SGLANG_ROCM_FUSED_DECODE_MLA=0`, `SGLANG_OPT_USE_TILELANG_INDEXER=1`,
`SGLANG_OPT_USE_JIT_NORM=0`, `--disable-custom-all-reduce` — a config-only
change with **no rebuild**. Cheapest first, since base 20260916 is on neither
node (a 66 GB pull plus a full DSA-patch rebuild).

`SGLANG_ROCM_FUSED_DECODE_MLA=0` is the one to try first on symptom match: it
names a **decode-path** kernel, and our fault is decode-side while prefill's
first token is correct.

#### R03 verdict

| check | result |
|---|---|
| 1. image (import the class, both nodes) | **PASS** |
| 2. startup log: enabled ×2, disabled gone | **PASS** |
| 3. output correctness | **FAIL** — `1!!!!!!!` reproduces |
| 4. `spec_accept_length >= 2.0` | **FAIL** overall — but see below |
| 5. same-rail: 0 Mooncake failures, 0 affinity 503 | **PASS** |

Prefill log: 0 real errors. The 51 matches for `Traceback|CRITICAL|fault` are all
the substring inside `default` / `defaults` / `defaulting` — checked, not assumed.

**The fix demonstrably took effect and demonstrably did not cure the bug.** That
settles the open question from the plan: the missing fusion was real, but it is
not what makes this stack produce wrong text. Combined with the peer's run, the
fusion defect is now closed as *a* defect worth carrying, not *the* defect.

#### What R03 actually bought — the failure is indexed by DP rank

Acceptance stopped being uniform. Before the fix every rank sat at ~1.25/0.05;
after it, `/metrics` under real traffic reads:

| dp_rank | accept_length | accept_rate |
|---|---|---|
| 0 | 1.0 | 0.00 |
| 1 | 1.0 | 0.00 |
| 2 | **2.8125** | **0.3625** |
| 3 | 1.0 | 0.00 |

A rate of *exactly* 0.00 is not "accepts rarely", it is "accepts nothing" — a
structural, not a numerical, signal. And rank 2's 2.81 is in the same range as
the peer's healthy stack (p50 3.86). So one rank is fine and three are not.

The decode-log series says the same thing: mostly `accept len: 1.00, accept
rate: 0.00`, punctuated by healthy `2.14 / 2.81 / 2.96 / 3.08`.

Then the decisive shape, from 16 **identical** sequential requests
(`temperature=0`, `max_tokens=12`, `rank-test/`). All 16 are wrong, but the
*mode* of wrongness alternates with period **4 = dp_size**:

| requests | failure mode |
|---|---|
| 2,3, 6,7, 10,11, 14,15 | repeated filler — `1!!!!!!!!!!!` |
| 1,4,5,8, 9,12,13,16 | word salad — `1reis legacyogos71reis…`, `1owl灵魂Sp_compat…` |

Sequential requests round-robin over the four DP ranks, and the corruption mode
tracks that rotation exactly. **The corruption is a deterministic function of
which DP rank serves the request.** That is a far sharper target than "MTP is
broken", and it is the first structural handle this bug has offered.

One more fact that does not fit a simple "3 ranks always broken" story and is
therefore recorded rather than explained: under 8-way concurrency with longer
prompts, **2 of 8 requests came back completely coherent for a full 400 tokens**
(`probe/load-1.json`, `load-7.json`). So a correct path through decode does
exist and is reachable; whatever is wrong is not a permanently corrupted weight
shard on three ranks, or nothing would ever be right. Left open.

---

### R04 — MTP off (`r04-no-mtp/`)

**Single variable against R03: `DECODE_MTP=0`.** Same nextnfix image, same
same-rail config, same real acceptance. Confirmed from the runner log that
`--speculative-algorithm` is absent and that decode loaded only
`GlmMoeDsaForCausalLM` (no `...NextN`).

**Result: 16/16 of the same identical requests are coherent**, against 0/16 in
R03. The two target probes pass outright:

- `"What is 2+2? …"` -> `1.  **Analyze the Request:** The user is asking for the
  result of 2+2. …`
- `"Reply with exactly: BASELINE_OK_YIHOU092824"` -> content
  `BASELINE_OK_YIHOU092824`, exact.

#### What this exonerates

Everything outside speculative decoding, on this exact hardware and config:

- the target model at TP4/DP4 with DP attention,
- the PD KV handoff, including the same-rail per-GPU NIC pinning and the router
  DP-rank affinity — both were active in this round,
- the plain decode attention path,
- the `index_share_for_mtp_iteration=false` override (still applied here).

**The fault is inside the MTP path.** That is the cleanest cut this bug has
taken, and it means the previous task's same-rail work is not implicated.

#### Where that leaves the search

Differential comparison now has a sharp shape. The peer runs the *same* EAGLE
configuration (5 steps / topk 1 / 6 draft tokens, P4D4, DP attention,
`index_share_for_mtp_iteration=false`, real acceptance) and is correct, on
sglang `ge7f7447333` (base 20260916). We are wrong on `ga9fb1c3238` (20260917).
So the MTP-path fault is specific to our base — either a 20260917 regression in
the speculative path, or one of our DSA patches interacting badly with it.

That second possibility deserves weight: the DSA patch set is cut against
**20260916**, and `draft_cuda_graph_dp_vote` had to be **re-cut** for this base
because upstream took the DP-sync slot it used (it now rides slot 8). Our own
landmine list calls it the one patch that can fail *silently*. I re-read all
four files of it by hand this session — the appended column is read at index 8
while its two neighbours read 6 and 7, defaults are permissive, the reduction is
`min()` — so it is self-consistent. Self-consistent is not the same as correct,
and it remains a suspect precisely because it is **indexed by DP rank**.

Caution against over-reading R03's period-4 signature: every one of those 16
requests was wrong, so the rank dependence shows up in the *mode* of failure,
not in pass/fail. A symptom that varies by rank does not prove the *cause* is
rank-indexed — different ranks hold different shards and different KV, so they
can land on different garbage from a single shared cause. Recorded as a clue,
not as a localisation.

Ranked next steps, cheapest first. All of the peer's numerics/kernel deltas are
env-or-flag only, so they need no rebuild:

1. `SGLANG_ROCM_FUSED_DECODE_MLA=0` — best symptom match now. R04 proves the
   *plain* decode path is fine, but MTP exercises different attention: the
   target's verify step runs in `speculative_attention_mode=prefill` and the
   draft runs its own decode. A kernel that is only wrong in the speculative
   modes would produce exactly this split.
2. `--disable-custom-all-reduce`.
3. `SGLANG_OPT_USE_TILELANG_INDEXER=1`, `SGLANG_OPT_USE_JIT_NORM=0`.
4. Draft CUDA graph off, to test the re-cut DP-vote patch by bypassing it.
5. Last and most expensive: our config on base 20260916 (66 GB pull + full
   patched rebuild on both nodes), which isolates the base image in one shot.

To run 1-4 at all, `engine.sh` needs a way to pass one-off env/flags. The peer's
`patches/0001-engine-sh-spur-fixes.patch` already adds exactly that —
role-scoped `{PREFILL,DECODE}_EXTRA_ARGS` / `_EXTRA_ENV`, defaulting to empty so
it is a no-op when unused.

---

### R05 — custom all-reduce off on the decode leg (`r05-nocustomar/`)

**Single variable against R03: `DECODE_EXTRA_ARGS=--disable-custom-all-reduce`.**
Same nextnfix image, MTP on, real acceptance, same-rail on both halves, P4D4.

#### How the candidate list got to one entry without spending a round

The peer's kit lists four "numerics / kernel selection" deltas. Read out of our
own image rather than trusted, three are **no-ops here**:

| delta | why it does nothing on our image |
|---|---|
| `SGLANG_ROCM_FUSED_DECODE_MLA=0` | `environ.py:899` declares `EnvBool(False)`; `forward_mla_fused_rope_rocm.py:35` reads it with default `"false"`. Unset == 0. |
| `SGLANG_OPT_USE_TILELANG_INDEXER=1` | `environ.py:1492` defaults False, but `arg_groups/model_hook.py:441-442` does `if not ...is_set(): ...set(True)`. Unset == 1. |
| `SGLANG_OPT_USE_JIT_NORM=0` | the string `jit_norm` does not occur anywhere under `python/sglang`, case-insensitive. Inert. |

The peer independently confirmed all three on their own base. Four greps
replaced three launches.

#### Result — target met

| check | result |
|---|---|
| ladder | `1` / `1. ` / `1.  **Analyze the Request` — R01 signature gone |
| nonce | **reproduced exactly**, `BASELINE_OK_YIHOU094206` |
| `spec_accept_length`, all 4 ranks | 2.8625 / 2.8684 / 3.05 / 2.8875 — **min 2.86, bar 2.0, PASS** |
| `spec_accept_rate` | 0.3725-0.41, same band as the peer and the reference kit |
| decode-log series | 2.66-3.76 |

**With MTP on and acceptance real, decode is correct.** The stated goal is met.

R03's per-rank bimodality is gone: all four ranks are now healthy and equal,
where before one read 2.81 and three read exactly 1.00/0.00. So the period-4
signature was ranks landing on different garbage from one shared cause — the
caution recorded in R04 was the right one, and the rank indexing was a symptom,
not a localisation.

#### A hypothesis falsified, cheaply, before it cost a round

The peer proposed that `draft_cuda_graph_dp_vote` might be mis-anchored on
20260917 — the patch the repo itself documents as able to fail silently.
Checked: `_get_local_tensor` packs exactly nine elements, index 7 is
`prefill_cuda_graph_max_prefix_len` and index 8 is
`int(can_run_draft_cuda_graph)`, the unpack reads `[:, 7].max()` and
`[:, 8].min()`, and the fallback tensor carries the permissive `1` in the same
ninth slot. **The re-anchor is sound.**

Worth recording against my own work: earlier this session I read that diff and
called the column arithmetic self-consistent, but I had only checked the read
side against its neighbours 6/7/8 — I never verified the packed list's absolute
length and offsets. The peer's one-line command closed a real gap in my
verification. It exonerated the patch rather than convicting it, which is the
less interesting outcome and the more useful one.

#### What is established, and what is not

**Established (A/B, single variable):** on this stack — sglang
`0.5.19.dev20260917+ga9fb1c3238` plus our DSA patch set, P4D4 with DP attention,
EAGLE 5/1/6 — custom all-reduce on the decode leg corrupts output, and disabling
it fixes both the text and the acceptance rate.

**Not established:** the mechanism. A plausible shape is the size-gated
branching in `custom_all_reduce.py` (`should_custom_ar`, `_MAX_CAR_SIZE`): the
draft is a single-layer model and verify runs at `num_draft_tokens=6`, so the
speculative path drives all-reduce at message sizes the plain decode path never
produces — which would explain why R04's MTP-off round was clean with the same
all-reduce enabled. **That is a hypothesis. We have the A/B, not the mechanism.**
Do not write it up as the cause.

Also not established: whether the fusion fix is needed at all — R05 carries
both. R06 tests the missing corner.

---

### R06 — the untested corner: no fusion fix + no custom all-reduce (`r06-nofix-nocustomar/`)

**Single variable against R05: `IMAGE` back to the un-patched
`infera-sglang:v0519-yihou-0917`.** Confirmed from the decode log that the
fusion loss is genuinely back — `Shared experts fusion optimization enabled`
once, `Config does not support fused shared expert(s)` once, i.e. the original
broken draft state.

**Result: 8/8 coherent, nonce reproduced, no degeneration, `spec_accept_length`
min 2.55 over active ranks — PASS.**

So **`--disable-custom-all-reduce` alone is the whole cure.** The NextN
shared-experts-fusion fix is *not* required for correct output on this stack.

#### Does the fusion fix buy anything measurable here?

| | per-rank `spec_accept_length` | decode-log series |
|---|---|---|
| R05, fix in | 2.8625 / 2.8684 / 3.05 / 2.8875 | 2.66 - 3.76 |
| R06, fix out | 2.80 / 2.8125 / 2.625 / 2.55 | 2.17 - 3.46 |

R05 is nominally higher (mean ~2.92 vs ~2.70), but the two runs' own log series
overlap substantially, and this is one sample of each under different scheduling.
**Not enough to claim the fusion fix improves acceptance.** Left open; it would
take repeated runs to separate from noise. What can be said: the fusion defect is
real and upstream-correct to fix, and it is independent of the bug that was
actually causing this failure.

#### Final state of the matrix

| custom all-reduce | fusion fix | output | acceptance |
|---|---|---|---|
| on | out (original) | garbled | 0.05 rate — R01/R02 baseline |
| on | in | garbled | bimodal per rank: one 2.81, three exactly 1.00/0.00 — R03 |
| on, MTP off | in | correct | n/a — R04 |
| **off** | in | **correct** | min 2.86 — R05 |
| **off** | out | **correct** | min 2.55 — R06 |

Custom all-reduce is the discriminator in every row.

#### Open, deliberately not concluded

1. **The mechanism.** We have the A/B, not the mechanism. The size-gated
   branching in `custom_all_reduce.py` (`should_custom_ar`, `_MAX_CAR_SIZE`) is a
   plausible shape — the draft is single-layer and verify runs at
   `num_draft_tokens=6`, so speculative decoding drives all-reduce at message
   sizes plain decode never produces, which fits R04 being clean with the same
   all-reduce enabled. Not demonstrated.
2. **Whether a narrower fix exists.** `engine.sh` also passes
   `--enable-aiter-allreduce-fusion` unconditionally. The culprit could be that
   path, or the interaction, rather than custom all-reduce as such. Disabling
   custom all-reduce is a blunt instrument and gives up an optimisation; a
   narrower knob might cost less throughput. Untested.
3. **The throughput cost of the fix.** Unmeasured — performance re-runs were
   explicitly out of scope for this task. Every number in the previous
   packup was taken with custom all-reduce ON and simulated acceptance ON, so
   none of them describes this configuration.
