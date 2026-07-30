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
| 7 | sustained durability, 8 × conc=128 × 512 back to back | ✅ **1024/1024**, `acc_len` flat at ~3.0 |

**Cumulative: 2540/2540 requests passed with the fix; 0/4 without it.**

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

## 6. The complete patch set

This fix does **not** stand alone. A working PD + DPA + MTP decode leg needs all four,
and `patches/` contains all four so this kit is self-contained:

| # | patch | fixes | without it |
|---|---|---|---|
| 1 | `dsa_indexer_hip_dp_padded_rows.diff` | Bug 1 + Bug 6 — HIP sizes `logits` from DP-padded rows while `lengths` is real rows; the slice guard's `0 <` lower bound also broke DP-**idle** ranks | `RuntimeError: Expected lengths.size(0) == B` |
| 2 | `dsa_backend_dp_sync_and_page_table_rows.diff` | Bug 5 — page-table rows (per request) vs top-k rows (per token) under MTP; plus removes rank-divergent `.item()` host syncs | `assert page_table.shape[0] == topk_indices.shape[0]` under concurrency |
| 3 | `deepseek_nextn_glm52_mtp_bf16.diff` | GLM-5.2's nextn layer is bf16 while the model is quantized | dies at weight load, `size of tensor a (3072) must match ... b (6144)` |
| 4 | **`eagle_worker_v2_uniform_draft_graph.diff`** | **Bug 2b — this round's fix** | first routed request hard-deadlocks |

Apply in that order against sglang `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd`:

```bash
cd /sgl-workspace/sglang
for d in dsa_indexer_hip_dp_padded_rows.diff \
         dsa_backend_dp_sync_and_page_table_rows.diff \
         deepseek_nextn_glm52_mtp_bf16.diff \
         eagle_worker_v2_uniform_draft_graph.diff; do
  patch -p1 --fuzz=0 < patches/$d
done
```

Verify (do not trust `--dry-run`; it fuzzes):

```bash
bash scripts/verify_patches.sh /sgl-workspace/sglang
# -> OK for each diff, then ALL PATCHES VERIFIED
```

`patches/eagle_worker_v2_uniform_draft_graph.diff` was checked against
`pristine/eagle_worker_v2.py.upstream` (a copy of upstream pulled from the image before
any edit): it applies at **fuzz=0**, byte-compiles, and the result is **byte-identical**
to what the fix script produced on the node that passed 2540/2540.

## 7. Layout

```
README.md                     this file
TASK_SPEC.md                  the task as issued, verbatim + the goal set up front
PLAN.md                       goal, hypotheses (H1/H2/H3), method
working_process.md            round-by-round index
REPRODUCE.md                  exact commands
PITFALLS.md                   wrong turns, with what/why/how/context
patches/                      all four diffs (this fix + the three prerequisites)
pristine/                     upstream eagle_worker_v2.py, for patch verification
scripts/                      fix, probes, analyzers, launchers, stress, verify_patches.sh
evidence/                     per-round RESULT.md, environment, raw guard records
results/                      stress jsonl (per-request outcomes)
```

## 8. Environment (captured live, not from notes)

MI355X ×8 (gfx950) · ROCm/HIP 7.2.26015 · torch 2.9.1+rocm7.2.0 · sglang 0.5.15.post1
@ `0b3bb0cbe31873994c9f989fddfe2f87ca839fdd` · kernel 6.8.0-107-generic · 236 cores ·
2751 GiB RAM · mlx5_0 fw 28.43.3608 (KV transport) + ionic_0 fw 1.117.1-a-63
Image `infera.yihou.sglang.1.0` ← `lmsysorg/sglang:v0.5.15.post1-rocm720-mi35x`.
Model `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4`. **No credentials required.**

Jobs: 11428 → crsuse2-m2m-029 (10.245.146.21, prefill) · 11429 → crsuse2-m2m-084
(10.245.148.109, decode).

## 9. Dependencies, uncommitted files, credentials

**Credentials: none.** No token, key, or login is needed to reproduce this. Model and
image are already on the cluster; `DOCKER_CONFIG=/tmp/dockercfg` is a local docker-29
workaround, not a credential.

**Absolute paths this kit depends on** (none are in git):

| path | what | size |
|---|---|---|
| `/shared_nfs/huggingface_models/amd/GLM-5.2-MXFP4` | model weights | 408 GB |
| `/home/yihou/infera.yihou.sglang.1.0.tar` | container image tarball | ~30 GB |
| `/sgl-workspace/sglang` | editable sglang inside the image | — |
| `/home/yihou/glm52_fix/` | working dir, bind-mounted into the container | — |
| `/home/yihou/glm52_fix/{inductor_cache,triton_cache}` | persistent JIT caches; halve boot time | — |
| `/home/yihou/glm52_fix/bug2b/` | **raw round-by-round workspace, incl. all server logs** | 15 MB |

**Files on the filesystem that are NOT committed** (deliberately — the kit carries the
distilled evidence, and the rule is not to delete originals):

* `/home/yihou/glm52_fix/bug2b/r0*/{decode,prefill,mix,router}.log` — ~15 MB of raw server
  logs, including the 6.3 MB `r03_verify/decode.log` that holds all 23 936 vote records.
  The kit's `evidence/` carries the extracted records and the analyzer output instead.
  Ask if you want the logs added.
* `/home/yihou/glm52_fix/src_spec/` — upstream source pulled from the image for reading.
  The one file needed for patch verification is committed as `pristine/`.
* `/home/yihou/glm52_fix/bug2b/r0*/fix_*.py`, `probe_*.py` — all copied into `scripts/`.

**Job/node identifiers are dead**: spur evicted 11428/11429 shortly after the final run
(normal, undocumented, and expected). Everything above lives on NFS and survived; nothing
in this kit requires those jobs to still exist.

## 10. Still open

* **`dp_padding_mode` was `None` on all 3200+ records**, so hypothesis H3 (captured
  graphs fix MAX_LEN while the eager path may pick SUM_LEN → `all_gather` vs `all_reduce`)
  could be neither confirmed nor refuted here. It is not *needed* to explain this
  deadlock, but it remains a plausible second defect on a different path.
* The **seventh padded-vs-real-rows crash** seen once in 500+ requests during the previous
  session is unrelated to this fix and still unexplained.
