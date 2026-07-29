# R1 — the divergence is measured, and the mechanism is now closed

Date 2026-07-29 17:35–17:42. Jobs 11428 (prefill, m2m-029) / 11429 (decode, m2m-084).
PD + DPA8 + MTP(steps=3, topk=1), draft CUDA graph **enabled**, Bug 1/5/6 + nextn fixes
applied, `GLM52_R1_PROBE=1`. All four patches verified present in **bytecode**, not just
source (`verify_pyc.sh`; eagle_worker_v2 marker count 3).

## What happened

Boot clean, warmup **passed**, `health=200`. The 4-request probe then hung: the decode
leg stopped answering even its own internal health check ("Server couldn't get a response
from detokenizer for last 20 seconds"), so the router opened the circuit and returned 503.
The 503 is a *consequence*, not the failure — the failure is the decode-side deadlock.

py-spy, sampled twice 10 s apart, **byte-identical** → hard deadlock, not slow progress:

```
DP0: broadcast                (distributed_c10d.py:2841)
DP1: all_gather_into_tensor   (distributed_c10d.py:4056)
DP2: init_forward_metadata    (dsa_backend.py:785)      <-- the eager metadata path
DP3: all_gather_into_tensor
DP4: all_gather_into_tensor
DP5: all_gather_into_tensor
DP6: broadcast
DP7: all_gather_into_tensor
```

## The guard records — the decisive evidence

72 probe records, 0 probe errors. Per-rank draft-call totals are **equal** (9 each), so
no rank was starved of iterations; the divergence is in the *decision*, not the count.

**it=9, the iteration everything froze on:**

| rank | mode | bs | nreq | t1 can_graph | t2 not_idle | t3 seed | t4 topk_is_None | **final** |
|---|---|---|---|---|---|---|---|---|
| dp2 | DECODE | 1 | 1 | True | **True** | True | **True** | **False → EAGER** |
| dp0,1,3,4,5,6,7 | IDLE | 0 | 0 | True | False | True | True | **True → GRAPH** |

`gnt=[0,0,1,0,0,0,0,0]` — dp2 is the only rank with work. It takes eager; the seven idle
ranks take the graph. **The rank that diverged (dp2) is exactly the rank py-spy found in
the eager `init_forward_metadata`.** Decision and stack agree.

**it=5, an earlier iteration that did NOT hang:**

| rank | mode | t2 not_idle | t4 topk_is_None | final |
|---|---|---|---|---|
| dp6 | IDLE | False | True | True → graph |
| all others | DECODE | True | **False** | True → graph |

Here t2 and t4 both diverge but they **cancel**: every rank still lands on graph. That is
why warmup passes and why the failure is racy — divergence in the terms is routine, and
only becomes fatal when it reaches `final`.

Summary over the run: `t2` diverges on 5 of 9 iterations, `t4` on 4, but **`final`
diverges on exactly 1 — it=9, the one that deadlocked.**

## Why term 4 flips, and why this is a PD-only bug

`t4` is `draft_input.dsa_topk_indices is None`. On the PD decode leg the draft input for a
freshly-arrived request is built by `eagle_disaggregation.py:54-59` from
`req.output_dsa_topk_indices` shipped over RDMA — so it is `None` for a request that just
landed, and non-`None` afterwards once `_draft_extend_for_decode` seeds it
(`eagle_worker_v2.py:1040`). it=5 is a steady-state iteration (seed present, t4 False);
it=9 is the first iteration of a new request (no seed, t4 True).

In **mix** there is no disagg path — the seed is always produced locally by the same code
on every rank, so t4 never flips asymmetrically. **This explains the mix-passes /
PD-hangs asymmetry that has been unexplained for the whole investigation.**

## H1/H2/H3 verdict

* **H2 refuted.** `t1` (`can_run_graph`) never diverges; `can_dp_cg=True` on every rank
  every iteration. The all-gathered vote at `dp_attn.py:111` is working correctly.
* **H3 not applicable as stated.** `padmode` is `None` on all 72 records — `dp_padding_mode`
  is not populated on the draft forward batch at this point, so the captured-MAX_LEN vs
  eager-SUM_LEN mismatch cannot be read here. Neither confirmed nor refuted; it is not
  *needed* to explain the hang, since the graph/eager split alone accounts for it.
* **H1 confirmed and sharpened.** Term 4 is the flipping term, but **term 2 is what makes
  the flip asymmetric**: the idle ranks short-circuit at t2 and never look at t4.

## Why removing `not is_idle()` alone failed — now explained by measurement

Drop t2 and re-evaluate the two iterations:

* it=9 → every rank has t1∧t3∧t4 = True → all eager → **uniform, fixed**.
* it=5 → dp6 (idle) has t4=True → eager; the busy ranks have t4=False → graph →
  **divergent, newly broken**.

So that change trades one failing iteration for another, which is precisely the observed
behaviour: the failure moved *earlier*, into warmup (WARMUP_MATRIX runs 5/6). The prior
attempt was not wrong about term 2 mattering; it was wrong about which direction to fix it.

## The fix this implies

The graph/eager choice must be a **DP-group** decision, and the quantity to vote on is the
*local need for eager*, not the raw terms:

```python
needs_eager_local = (can_cuda_graph and not is_idle and seed and topk_is_None)
needs_eager_group = OR over the TP group        # one 1-element all-reduce (MAX)
can_cuda_graph    = can_cuda_graph and not needs_eager_group
```

Checked against both measured iterations:

* it=9: dp2 local=True, others local=False → OR=True → **all eager** → uniform ✓
* it=5: dp6 idle → local=False; busy ranks t4=False → local=False → OR=False →
  **all graph** → uniform ✓ (and the graph is still used, unlike Variant B)

Idle ranks contribute `False`, so an idle rank can never drag the group to eager — which
is the failure mode of the naive "drop t2" fix. Cost is one 1-element all-reduce per
`draft()` call; all 8 ranks provably reach this line (idle ranks logged records at it=9),
so a collective here is safe.

Files: `decode.log`, `prefill.log`, `router.log`, `probe_guard.py`, `analyze.py`.
