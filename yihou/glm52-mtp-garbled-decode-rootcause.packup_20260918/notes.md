# Notes — wrong turns, ruled-out hypotheses, open questions

The full round-by-round log is `spec/working_process.md`. This file is the
distilled version: what cost time, what was ruled out and how, and what is still
open. Read §7 before quoting any number.

---

## 1. The starting lead was real, and wrong

**What.** The investigation opened on a colleague's finding: sglang's
`GlmMoeDsaForCausalLMNextN` never overrides the class attribute
`fused_shared_experts_architecture`, so the name compare in
`deepseek_v2.py::shared_experts_fusion_disable_reason` can never match and the
MTP **draft** silently loses shared-experts fusion. Fork commit `4350d37c5b`.

**Why it looked decisive.** Four independent confirmations, all first-hand:

- It fired on our decode leg. The target (`GlmMoeDsaForCausalLM`) logs
  `Shared experts fusion optimization enabled`; the draft
  (`GlmMoeDsaForCausalLMNextN`) logs `Config does not support fused shared
  expert(s)` in the same second as its own `Load weight begin`, ~40 s later. The
  prefill leg has no draft and shows the enabled line with no disabled line — a
  control obtained for free.
- Of the four NextN classes in the image, only GLM DSA's lacks the attribute
  (`deepseek_nextn.py:246`, `dots3_common/nextn.py:139`,
  `glm4_moe_lite_nextn.py:134` all have it).
- Upstream's own comment at `deepseek_v2.py:2967` names the cases that must
  override it: *"the NextN drafts, GLM's DSA variant"* — and GLM DSA's NextN is
  the intersection.
- `model_config.py::_config_draft_model` rewrites **only** `architectures[0]`,
  leaving `n_routed_experts`/`n_shared_experts` untouched, and the target passed
  the same gate on the same config. So the name compare is the only term that
  could have failed. The argument closes.

It also explained the measured `accept len 1.25 / rate 0.05` perfectly.

**How it fell.** R03 applied the fix, verified three ways (import check on both
nodes; `enabled` ×2 with `disabled` gone in the decode log; draft weight
footprint moved 6.90 → 6.92 GB, so the fused layout is genuinely allocated) — and
the output was still `1!!!!!!!`. R06 later showed the cure works with the fix
removed entirely.

**Context.** The gap was visible from the start and was written down before the
round ran: under EAGLE, a rubbish draft should be *rejected* and fall back to the
target's own token — slow, not wrong. That known gap is the reason R03 was framed
as a bisection step rather than a predicted cure, and it is why R04 was already
planned. **Carrying an unexplained gap explicitly is what kept the round cheap.**

The fusion defect is real, upstream-correct to fix, and independent of this bug.

---

## 2. A build that could never have worked — my error

**What.** The first `Dockerfile.yihou.nextnfix` verified the patch *inside* the
build by importing the patched class. Both nodes failed identically with
`RuntimeError: No HIP GPUs are available`.

**Why.** Importing `sglang.srt.models.glm4_moe` transitively imports
`sglang.kernels.ops.quantization.fp8_kernel`, which at **module import time**
calls `is_gfx1250_supported()` → `torch.cuda.get_device_properties(0)` →
`torch._C._cuda_init()` (`srt/utils/common.py:1082`). `docker build` has no GPU.

**How fixed.** Move the import check out of the build and into a post-build
`docker run --device /dev/kfd --device /dev/dri`. Strictly better anyway: it
tests the image the way it will actually run. The build keeps only the applier,
which already exits non-zero if its marker is absent — so a silent source miss
still fails the build.

**Context.** The failure was loud and the applier had already printed
`[patch] nextn-fusion: applied`, so nothing was ambiguous. Worth keeping the
import check at all: Python caches bytecode in `__pycache__` keyed on mtime, so
grepping the source can pass while the loaded class is stale. Importing the class
proves what will run.

---

## 3. Delegation cost more than it saved here

**What.** Two teammates were spawned. `bench-prep` did its job. `image-fix`
reached *exactly* the same diagnosis as §2 independently — and then stalled,
because its brief did not authorise it to modify the scaffolding it had been
handed, so it correctly came back to ask. Its messages arrived ~1 h late.

**Why it mattered.** The image build was the critical path. By the time the
question surfaced, the build had already been redone by hand, duplicating work
`image-fix` had finished.

**How to avoid.** A teammate brief must say (a) which files it may modify, and
(b) what to do when blocked. The agent's caution was correct behaviour; the brief
was the defect. Also: `image-fix` did one thing better than the leader — it ran
the same import against the **base** image with a GPU and got
`DeepseekV3ForCausalLMNextN`, the un-patched inherited value. That is direct
positive confirmation that the patch was needed, stronger than a source read.

---

## 4. A good hypothesis, cheaply falsified

**What.** `draft_cuda_graph_dp_vote` — the one patch the repo documents as able
to fail *silently* ("a mechanical re-anchor compiles **and passes the bytecode
markers** while reading a prefix length as a boolean") — governs the draft CUDA
graph across DP ranks, was re-cut for this base, and is DP-rank-indexed. R03's
period-4 failure signature (below) is DP-sync-shaped. It fit everything.

**How falsified.** One command, no GPU. `_get_local_tensor` in
`scheduler_components/dp_attn.py` packs exactly nine elements:

```
0 num_tokens                     5 local_forward_mode
1 num_tokens_for_logprob         6 int(can_run_prefill_cuda_graph)
2 int(can_run_decode_cuda_graph) 7 prefill_cuda_graph_max_prefix_len
3 int(is_extend_in_batch)        8 int(can_run_draft_cuda_graph)
4 int(local_can_run_tbo)
```

and the unpack reads `[:, 7].max()` as int and `[:, 8].min()` as bool, with the
fallback tensor carrying a permissive `1` in the same ninth slot. The re-anchor
is sound.

**Context, and a correction to my own earlier claim.** Earlier in the session I
read this diff and called the column arithmetic "self-consistent". That was based
only on the read side matching its neighbours 6/7/8 — **I never verified the
packed list's absolute length and offsets.** The peer's one-line command closed a
real gap in my verification. It exonerated the patch rather than convicting it,
which is the less interesting outcome and the more useful one.

---

## 5. What actually found it: someone else's known-good run

**What.** A parallel session had a **known-good** GLM-5.2 1P1D stack on the same
P4+DPA / D4+DPA shape, same EAGLE settings, same
`index_share_for_mtp_iteration=false`, real acceptance — correct output, accept
len p50 3.86 — on base **20260916**. Kit at
`/home/yihou/dev/git/infera.yihou.mtp.debug/work/glm52-1p1d-mtp-correctness.packup_20260918/`.

**Why it mattered.** The user had declined to *build* a differential reference
(no TP8 run, no single-node run) on cost grounds. This one already existed and
cost nothing. It supplied the delta table that `--disable-custom-all-reduce` came
out of.

**How it was used — and not taken on trust.** Their first message reported the
same draft-side fusion-disabled line with healthy output, framed as "a draft
without shared-experts fusion is not sufficient to cause the garbling". Verified
against their raw `logs/green-decode-0.log` rather than their summary: both lines
present with the same ~40 s gap, `grep -c SIMULATE_ACC_LEN` = 0 so acceptance was
real, tail `accept len` samples 4.99-5.54. The claim held.

But the inference over-reached one row of their own table: **base image** was
itself a delta (`ge7f7447333` vs our `ga9fb1c3238`). Pushing back on that is what
kept the fusion arm alive long enough to be tested properly rather than dropped.
They amended their kit; we later showed the fusion fix is neither necessary nor
sufficient here, which settled it for both bases.

**The step that actually saved cluster time.** Their delta table had four
"numerics / kernel selection" rows. Three are **no-ops on our image** — read out
of our own image, not trusted:

| delta | why it does nothing here |
|---|---|
| `SGLANG_ROCM_FUSED_DECODE_MLA=0` | `environ.py:899` declares `EnvBool(False)`; `forward_mla_fused_rope_rocm.py:35` reads it with default `"false"`. Unset == 0. |
| `SGLANG_OPT_USE_TILELANG_INDEXER=1` | `environ.py:1492` defaults False, but `arg_groups/model_hook.py:441-442` does `if not ...is_set(): ...set(True)`. Unset == 1. |
| `SGLANG_OPT_USE_JIT_NORM=0` | the string `jit_norm` does not occur anywhere under `python/sglang`, case-insensitive. Inert. |

The peer independently confirmed all three on their base. **Four greps replaced
three launches**, and the one surviving row was the answer.

---

## 6. The period-4 clue, and why it was not a localisation

**What.** In R03, 16 identical sequential requests were all wrong, but the *mode*
of wrongness alternated with period **4 = dp_size**: half repeated-filler
`1!!!!!!!!!!!`, half word-salad. Acceptance was bimodal per rank in the same
round — one rank at 2.81, three at exactly 1.00/0.00.

**Why it was tempting.** A rate of *exactly* 0.00 is structural, not numerical.
Period 4 over four DP ranks looks like a rank-indexed bug, which pointed straight
at §4.

**Why it was recorded as a clue, not a finding.** All 16 were wrong, so the rank
dependence lived in the *mode*, not in pass/fail. Different ranks hold different
shards and different KV, so **one shared cause can surface as different garbage
per rank**. That caution was written down at the time — and it was right: with
custom all-reduce disabled the bimodality vanished entirely and all four ranks
read 2.55-3.05.

**Context.** The genuinely odd datum from the same round, recorded and never
explained: under 8-way concurrency with longer prompts, 2 of 8 requests came back
coherent for a full 400 tokens (`results/r03-fusion-fix/probe/load-{1,7}.json`).
So a correct path through decode existed and was reachable even in the broken
configuration. Not explained; left open.

---

## 7. Open — do not close these by assumption

### 7a. The mechanism is unknown

We have the A/B, **not** the mechanism. A plausible shape is the size-gated
branching in `custom_all_reduce.py` (`should_custom_ar`, `_MAX_CAR_SIZE`): the
draft is a single-layer model and verify runs at `num_draft_tokens=6`, so
speculative decoding drives all-reduce at message sizes the plain decode path
never produces — which would explain R04 being clean with the same all-reduce
enabled. **Hypothesis. Not demonstrated.**

### 7b. This may not be the narrowest fix

`engine.sh` passes `--enable-aiter-allreduce-fusion` unconditionally, and it was
`True` in all four rounds (`results/flags-by-round.txt`). Held constant, so it
cannot be the discriminator — but the culprit could be custom all-reduce alone,
or only its interaction with the aiter fusion path. A narrower knob might cost
less throughput. Untested.

### 7c. The throughput cost is unmeasured

Disabling custom all-reduce gives up an optimisation. Performance re-runs were
explicitly out of scope. **No number in the sibling packup
`glm52-1p1d-samerail-c32-c40.packup_20260918` describes this configuration** —
every number there was taken with custom all-reduce ON *and* simulated acceptance
ON (`DECODE_SIMULATE_ACC_LEN=3.61`, which forces the acceptance length). Do not
compare across the two packups without re-measuring.

### 7d. Whether the fusion fix helps acceptance at all

R05 (fix in) read 2.8625 / 2.8684 / 3.05 / 2.8875; R06 (fix out) read 2.80 /
2.8125 / 2.625 / 2.55. Nominally higher with the fix, but the two rounds' own
decode-log series overlap (2.66-3.76 vs 2.17-3.46) and this is **one sample each
under different scheduling**. Not enough to claim an improvement, and this kit
does not claim one. Separating it from noise needs repeated runs.

### 7e. 2 of 8 correct under concurrency in the broken configuration

See §6. Unexplained.

---

## 8. Smaller traps worth knowing

- **`accept_length` and `accept_rate` are different metrics.** The bar in this
  work is on *length* (`>= 2.0`). The failing baseline read length 1.25 / rate
  0.05. Quoting one for the other misleads by more than an order of magnitude.
- **The idle-gauge trap.** A DP rank that served no decode tokens reports
  `spec_accept_length 0.0`, indistinguishable from "accepts nothing". Generate
  real decode traffic first, then read, and trust only ranks whose counters
  moved. `scripts/probe.yihou.sh` deliberately snapshots the idle gauge first so
  the trap is visible in the evidence rather than hidden.
- **`config.sh` uses `${DECODE_SIMULATE_ACC_LEN-3.61}` — a single dash.** An
  empty-but-set value therefore survives and disables the simulation; `:-` would
  silently restore 3.61. A sibling campaign lost a variable to exactly this
  (`DSA_TOPK_BACKEND=""` before a source yielding `aiter` again).
- **A bare closing brace inside `${VAR:=...}` truncates JSON.** The per-GPU
  `RDMA_DEVICE` map has to be assigned with an explicit `if [[ -z ... ]]` test.
- **Grepping logs for `fault` matches `default`.** A round here showed 51 hits
  for `Traceback|CRITICAL|fault` in the prefill log; all 51 were the substring
  inside `default` / `defaults` / `defaulting`. Check before reporting.
- **`launch.sh`, `stop.sh`, `preflight.sh` refuse a pre-existing `OUT_DIR`**, and
  `CONTROL_NODE` must appear in the topology file.
- **The repo's `topology.tsv` lists 136/137/140**, not the nodes used here. Pass
  `TOPOLOGY=` rather than editing it.
- **Don't machine-classify "is this output coherent".** A first attempt at an
  automatic classifier for this kit mis-scored the truncated-but-correct replies
  (`max_tokens=12` cuts `1.  **Analyze the Request:**` mid-sentence), and one
  garbled mode is pure ASCII word-salad (`1reis0obuf Erect bufferreisangan
  legacyu`) that no simple rule separates from prose. The counts in
  `results/matrix.csv` are **human-classified**, with every reply dumped to
  `results/<round>/rank-test-replies.txt` so a reader can check all 48 by eye.
