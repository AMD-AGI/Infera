# R3 — the fix passes all five exit criteria

Date 2026-07-29 18:05–18:20. Jobs 11428 (prefill) / 11429 (decode).
PD + DPA8 + MTP(steps=3, topk=1), **draft CUDA graph enabled**, fix
`GLM52_BUG2B_UNIFORM` + both probes live. All patches verified in **bytecode**.

## Exit criteria

| # | criterion | result |
|---|---|---|
| 1 | PD warmup (8 concurrent, one per DP rank) | **PASSED**, `ready to roll` |
| 2 | 4 × 24-token sequential | **4/4**, 0.80–0.97 s, `acc_len` 1.71–3.00 |
| 3 | 1 × 512-token | **512/512 tokens**, 45.1 s, `acc_len` 2.36 |
| 4 | conc=128 × 512 | **128/128 ok**, 46.8 s, all 8 dp ranks, `acc_len` mean 2.86 (1.97–3.97) |
| 5 | **the graph path is actually taken** | **98.0%** of iterations all-ranks-GRAPH |

Criterion 5 is the one that distinguishes this from Variant B, which reached 1–4 by
disabling the draft graph outright.

(Criterion 4: 119/128 produced the full 512 tokens; the other 9 stopped on EOS under
greedy decoding, which is normal termination, not truncation. 0 failures, 0
`KVTransferError`.)

## The measurement that proves the mechanism, not just the outcome

505 iterations, all 8 ranks logging on every one:

```
(A) iterations where LOCAL diverges (the latent bug) :   8
(A) iterations where VOTED diverges (must be 0)     :   0
(B) all-ranks-GRAPH iterations                       : 495
(B) all-ranks-EAGER iterations                       :  10
vote changed a rank's decision                       :  49 times
```

This is the whole argument in four numbers:

* **LOCAL still diverges 8 times.** The underlying rank-divergence is untouched and real
  — the fix does not paper over it by accident of timing.
* **VOTED never diverges.** The value the code acts on is uniform on every single
  iteration. That is the invariant whose violation deadlocked the group.
* **495 all-graph iterations.** The draft graph is genuinely used, so the speedup is kept.
* **49 mind-changes.** Each is an iteration where a rank locally wanted the graph while
  another needed eager — i.e. 49 would-be deadlocks, converted into a uniform choice.

Contrast with R1 (no fix): `final` diverged on exactly 1 iteration out of 9 — it=9 — and
that one iteration deadlocked the server permanently.

## Before / after on the same hardware, same config

| | R1 (no fix) | R3 (fix) |
|---|---|---|
| warmup | passed | passed |
| 4 × 24-token | **HUNG** (health stopped answering, router opened circuit) | 4/4 |
| py-spy | 3 distinct blocking sites, frozen across 10 s | n/a |
| decision uniform | no — diverged on the frozen iteration | **yes, 505/505** |
| draft graph used | yes, but only on the ranks that diverged into it | **yes, 98.0%** |

## What the fix is

`eagle_worker_v2.py::draft()` — vote on the *local need for eager*, act on the group answer:

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

Notes that mattered:

* **`get_tp_group()`, not `get_attention_tp_group()`.** Under DPA8 on tp8 the attention TP
  group is `attn_tp_size = tp/dp = 1` rank wide, so voting on it is a no-op. The group that
  must agree is the full TP group — the same one the scheduler's MLP-sync all-gather spans.
* **`cpu_group` (gloo), not the device group.** A 1-element host all-reduce adds no GPU
  sync and cannot serialize the compute stream.
* **Safe to put a collective here**: R1 measured all 8 ranks entering `draft()` an equal
  number of times, including the idle ones. The collective sits on a branch every rank
  takes — exactly the property the buggy code lacked.
* An idle rank contributes `False`, so it can never drag the group to eager. That is the
  failure mode of the earlier "drop `not is_idle()`" attempt, which this avoids.
