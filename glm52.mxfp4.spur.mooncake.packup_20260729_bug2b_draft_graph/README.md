# Bug 2b — PD + DPA + MTP draft-CUDA-graph deadlock: root-caused and fixed

**Date:** 2026-07-29 (17:20–19:00 UTC)
**Status:** ✅ **fixed and verified**, with a same-node differential control.
**Not pushed.** The user submits PRs.

---

## 1. One-paragraph summary

On the PD decode leg with DP-attention(8) + EAGLE MTP, the decision *"replay the draft
CUDA graph, or run the draft eagerly?"* was made **independently on each DP rank** from
rank-dependent inputs. The two paths do not issue the same host-side collective sequence,
so the moment one rank chose differently from the others the group desynchronized and
deadlocked — permanently, on the first routed request. The fix makes that one decision a
**DP-group vote**: each rank contributes whether *it* needs eager, the group takes the OR,
and every rank acts on the group's answer. The draft graph is still used on 98.4% of
iterations, so unlike the previous workaround this keeps its speedup.

---

## 2. What was wrong

`python/sglang/srt/speculative/eagle_worker_v2.py::draft()`:

```python
if (can_cuda_graph
    and not forward_batch.forward_mode.is_idle()      # rank-dependent (occupancy)
    and self.seed_dsa_topk_from_draft_extend
    and draft_input.dsa_topk_indices is None):        # rank-dependent (see below)
    can_cuda_graph = False
```

Two of the four terms differ across ranks **by construction**:

* `is_idle()` — under DP-attention some ranks have work and some do not, every step.
* `dsa_topk_indices is None` — on the **PD decode leg** this field is built from
  RDMA-shipped per-request payloads (`eagle_disaggregation.py:54-59`), so it is `None`
  for a freshly-arrived request and non-`None` once `_draft_extend_for_decode` has
  seeded it. It is therefore a function of *which requests this rank happens to hold*.

**Why single-node mix never hit this** (unexplained for the whole prior investigation):
in mix there is no disaggregation path, so the seed is always produced locally by the same
code on every rank, and term 4 never flips asymmetrically.

## 3. The measurement

Instrumented every guard term, per rank, per iteration, and ran to the hang.

**R1, the iteration that froze (it=9):**

| rank | mode | t1 can_graph | t2 not_idle | t3 seed | t4 topk_is_None | decision |
|---|---|---|---|---|---|---|
| **dp2** | DECODE | True | **True** | True | **True** | **EAGER** |
| dp0,1,3,4,5,6,7 | IDLE | True | False | True | True | GRAPH |

`gnt=[0,0,1,0,0,0,0,0]` — dp2 was the only rank with work. py-spy at that moment:

```
DP2: init_forward_metadata (dsa_backend.py:785)   <-- the eager path
others: all_gather_into_tensor / broadcast
```

**The rank whose decision diverged is exactly the rank stuck on the eager path.** Sampled
twice 10 s apart, byte-identical → hard deadlock.

Over the whole run: `t2` diverged on 5 of 9 iterations and `t4` on 4, but the *decision*
diverged on **exactly 1** — the one that deadlocked. Divergence in the terms is routine
and usually cancels; it is fatal only when it reaches the final decision. That is why
warmup passes and why the bug is racy.

**R4 (differential control) reproduced it independently** with a different victim rank
(dp3), same signature, same agreement between the logged decision and the py-spy stack.

## 4. The fix

`patches/bug2b_uniform_draft_graph.patch`

```python
_needs_eager_local = (can_cuda_graph and not is_idle
                      and self.seed_dsa_topk_from_draft_extend
                      and draft_input.dsa_topk_indices is None)
if can_cuda_graph and self.seed_dsa_topk_from_draft_extend:
    _vote = torch.tensor([1 if _needs_eager_local else 0], dtype=torch.int32)
    torch.distributed.all_reduce(_vote, op=MAX, group=get_tp_group().cpu_group)
    _needs_eager_local = bool(_vote.item())
if _needs_eager_local:
    can_cuda_graph = False
```

Three details that matter:

* **`get_tp_group()`, not `get_attention_tp_group()`.** Under DPA8 on tp8 the attention
  TP group is `attn_tp_size = tp/dp = 1` rank wide, so voting on it is a silent no-op.
  The group that must agree is the full TP group — the same one the scheduler's MLP-sync
  all-gather spans (`dp_attn.py:91`).
* **`cpu_group` (gloo), not the device group.** A 1-element host all-reduce adds no GPU
  sync and cannot serialize the compute stream.
* **A collective is safe at this line.** Instrumentation showed all 8 ranks entering
  `draft()` an equal number of times every iteration, idle ones included. The vote sits
  on a branch every rank takes — precisely the property the buggy code lacked.

An idle rank contributes `False`, so it can never drag the group to eager.

### Why the earlier attempt failed, now explained

A previous fix removed only the `not is_idle()` term. Replaying the measured iterations:

* it=9 → all ranks then have t1∧t3∧t4 → all eager → uniform ✓
* it=5 → the idle rank has t4=True → eager, busy ranks t4=False → graph → **divergent** ✗

It fixes one iteration and breaks another, which matches the observed behaviour exactly
(the failure moved *earlier*, into warmup). The prior attempt was right that term 2
mattered and wrong about the direction.

## 5. Verification

| # | criterion | result |
|---|---|---|
| 1 | PD warmup (8 concurrent, one per DP rank) | ✅ passed |
| 2 | 4 × 24-token sequential | ✅ 4/4, `acc_len` 1.71–3.00 |
| 3 | 1 × 512-token | ✅ 512/512 tokens |
| 4 | conc 1/2/4/8/16/64/128/256 × 512 tok | ✅ **1384/1384**, 0 failures, 0 KVTransferError |
| 5 | **draft graph actually used** | ✅ **98.4%** of iterations |
| 6 | single-node **mix** unregressed (shared code path) | ✅ **132/132** |

**Cumulative: 1516/1516 requests passed with the fix; 0/4 without it.**

Criterion 5 is what separates this from the earlier Variant-B workaround, which reached
1–4 by disabling the draft graph outright.

**Mechanism verified, not just the outcome** — over 2992 iterations with all 8 ranks logging:

```
iterations where the LOCAL decision diverges (the latent bug) :   38
iterations where the ACTED-ON decision diverges (must be 0)   :    0
all-ranks-GRAPH iterations                                    : 2943  (98.4%)
vote changed a rank's mind                                    :  190  (= 190 averted deadlocks)
```

The bug is still latent (LOCAL diverges 38 times) — the fix does not hide it by timing;
it converts each divergence into a uniform choice.

### Differential control (对拍) — mandatory, and it holds

Same node, same container, same config, same router, same traffic; **only the fix reverted**
(verified 0 in bytecode):

| | control (no fix) | fixed |
|---|---|---|
| 4 × 24-token | **0/4 — deadlock, 120 s timeout on request 1** | 4/4 |
| concurrency sweep | not reached | 1384/1384 |
| decision uniform | **no** — diverged on the frozen iteration | yes, 2992/2992 |
| victim rank | dp3 (R1 saw dp2) | — |

## 6. Layout

```
README.md                     this file
PLAN.md                       goal, hypotheses (H1/H2/H3), method
working_process.md            round-by-round index
REPRODUCE.md                  exact commands
PITFALLS.md                   wrong turns, with what/why/how/context
patches/                      the shippable patch
scripts/                      fix, probes, analyzers, launchers, stress
evidence/                     per-round RESULT.md, environment, raw guard records
results/                      stress jsonl (per-request outcomes)
```

## 7. Environment (captured live, not from notes)

MI355X ×8 (gfx950) · ROCm/HIP 7.2.26015 · torch 2.9.1+rocm7.2.0 · sglang 0.5.15.post1
@ `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` · kernel 6.8.0-107-generic · 236 cores ·
2751 GiB RAM · mlx5_0 fw 28.43.3608 (KV transport) + ionic_0 fw 1.117.1-a-63
Image `infera.yihou.sglang.1.0` ← `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`.
Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. **No credentials required.**

Jobs: 11428 → crsuse2-m2m-029 (10.245.146.21, prefill) · 11429 → crsuse2-m2m-084
(10.245.148.109, decode).

## 8. Prerequisite patches

This fix does not stand alone. The decode leg also needs, in this order:
Bug 1 (`apply_fix.py`), Bug 5 (`fix_bug5_page_table_rows.py`),
Bug 6 (`fix_bug6_idle_qoffset.py`), and the nextn `eh_proj` patch — without the last one
the server dies at load with a 3072-vs-6144 shape mismatch.

## 9. Still open

* **`dp_padding_mode` was `None` on all 3200+ records**, so hypothesis H3 (captured
  graphs fix MAX_LEN while the eager path may pick SUM_LEN → `all_gather` vs `all_reduce`)
  could be neither confirmed nor refuted here. It is not *needed* to explain this
  deadlock, but it remains a plausible second defect on a different path.
* The **seventh padded-vs-real-rows crash** seen once in 500+ requests during the previous
  session is unrelated to this fix and still unexplained.
