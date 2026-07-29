# Bug 2b — plan

## Goal (binary; this is the exit condition)

PD decode leg, DPA8 + MTP(steps=3, topk=1), **draft CUDA graph ENABLED**:

1. PD warmup completes (it is an 8-way concurrent, one-per-DP-rank burst — the sharpest test);
2. 4 × 24-token sequential → 4/4 HTTP 200 with `spec_accept_length > 1`;
3. 1 × 512-token → 512/512 tokens;
4. conc=128 × 512 tokens → 0 hangs, 0 `KVTransferError`;
5. the graph path is **actually taken** (verified by a marker count > 0, not by absence of a hang).

Anything less than 5 is not a fix — Variant B already gets 1–4 by disabling the graph.

## Control (对拍) — mandatory, both arms every round

| arm | config | expected |
|---|---|---|
| **B** (control, known-good) | `can_cuda_graph=False` forced in `draft()` | passes |
| **G** (subject) | stock guard, draft graph live | hangs |

The eager pair from the previous session is gone (jobs released). The control is now
Variant B on the *same* node, which is a better control anyway: it differs from the
subject by **one line**, not by three graphs.

## What is already established (do not re-measure)

* Draft graph specifically — not target-decode, not draft-extend (`RESULT_variant_B_draft_graph.md`).
* Dropping only `not is_idle()` does NOT fix it and moves the failure into warmup.
* Python-visible collective counts on the graph path are a **blind spot**, not a measurement.
* Single-node **mix** with the identical DPA+MTP+graph config **passes 256/256**. Only PD hangs.

## Hypotheses, ranked by what the source actually says

### H1 — `dsa_topk_indices is None` is rank-divergent in PD *by construction*, and mix never hits it

`eagle_disaggregation.py:54-59`, on the decode leg:

```python
dsa_topk_indices = None
dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
if dsa_indices_list and all(t is not None for t in dsa_indices_list):
    dsa_topk_indices = torch.stack(dsa_indices_list, dim=0).to(batch.device)
    if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
        dsa_topk_indices = None
```

This is computed **per rank from RDMA-received per-request payloads**. Two ranks holding
different requests get different answers — and a rank with an *empty* `batch.reqs` gets
`dsa_indices_list == []`, so `all(...)` is vacuously True but `dsa_indices_list` is falsy
→ `None`. So term 4 of the guard is a function of *which requests this rank received*.

In **mix**, `dsa_topk_indices` is seeded locally by `_draft_extend_for_decode`
(`eagle_worker_v2.py:1040`) — same code on every rank, driven by the same local loop. That
is the structural difference that explains why mix passes and PD hangs, and it is the
first thing this investigation has that *predicts* the mix/PD asymmetry instead of
merely restating it.

**Test:** log all four guard terms per rank per iteration on the PD decode leg. If term 4
diverges while terms 1–3 agree, H1 is confirmed and the fix is to make term 4 a DP-group
decision.

### H2 — the divergence is in `can_run_graph`'s bs computation, not the guard

`eagle_draft_cuda_graph_runner.py:290-310`:

```python
cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // self.num_tokens_per_bs
is_bs_supported = cuda_graph_bs <= self.max_bs      # disable_padding False
if self.require_mlp_sync:
    is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
```

`global_num_tokens_cpu` is all-gathered, so `max(...)` **is** rank-uniform, and
`can_run_dp_cuda_graph` comes from `tp0_info[:,2].min()` — an explicit DP-group AND
(`dp_attn.py:111`). So `can_run_graph` *should* be uniform. If measurement shows it is
not, the all-gather is being skipped or stale on some path — a different and more serious
bug. **This is the falsifier for H1**: if term 1 also diverges, H1 is incomplete.

### H3 — both paths are uniform but issue different collective *sequences*

The graph replay executes whatever was captured at `capture_one_shape`, which builds its
forward batch with `global_num_tokens_cpu = [num_tokens] * dp_size` (uniform, line 353)
and `DpPaddingMode.get_default_mode_in_cuda_graph()` = **MAX_LEN**
(`dp_attention.py:94-100`, since `SGLANG_USE_ROCM700A` is not set).

The eager path computes its mode at runtime via `get_dp_padding_mode`
(`dp_attention.py:67-91`) which for a decode batch returns MAX_LEN or SUM_LEN depending
on `sum_len*2 >= max_len*dp_size` — i.e. **occupancy-dependent**. MAX_LEN uses
`all_gather_into_tensor`; SUM_LEN uses `all_reduce`. **These are different collectives.**

So a graph rank and an eager rank in the same step can issue `all_gather` vs `all_reduce`
— which is a genuine RCCL mismatch, and unlike the retracted "count" argument this one
survives the graph-replay blind spot, because the collective *type* is fixed at capture
time and is visible in the source. Under PD, one rank busy + seven idle gives
`sum_len*2 < max_len*dp_size` → SUM_LEN eager, vs MAX_LEN captured. Under mix, all ranks
are typically busy → sum≈max*dp → MAX_LEN both ways → no mismatch. **H3 also predicts the
mix/PD asymmetry**, independently of H1.

H1 and H3 are not exclusive: H1 explains why the paths differ per rank, H3 explains why
that difference deadlocks. A complete fix may need both.

## Method, cheapest first

* **R1 — instrument, no fix.** Log per rank per iteration: the 4 guard terms,
  `can_run_graph`'s inputs, the resolved `dp_padding_mode`, and `global_num_tokens_cpu`.
  Run PD to the hang, then diff across ranks. This decides H1 vs H2 vs H3 in one boot.
  Probe must be written to fire on the *eager* path only (graph replay runs no Python) and
  to log the decision *before* it is acted on.
* **R2 — targeted fix** for whichever hypothesis survives; both arms.
* **R3 — the guard's own terms**, only if R1 shows term 4 is the sole divergence:
  all-reduce term 4 across the DP group (one small bool collective per draft call).
* **R4 — scale**: conc=128 × 512 and the 5-point exit condition.

Binary search is held in reserve: the problem domain is not yet closed enough for it to
beat instrumentation, since one PD boot is ~8 min and a bisect over guard terms was
already tried and failed (it removed one term and the split stayed divergent).

## Traps to respect (carried from CLAUDE.md, each already cost a cycle)

* `.pyc` staleness silently reverts a patch → `os.utime` + delete `__pycache__`, then
  `strings ...pyc | grep -c MARKER`.
* Never background a long docker client inside `spur exec`.
* Kill by explicit PID; a broad `pkill -f` matches your own shell.
* Logs contain binary bytes → `strings` or `grep -a`, never plain `grep -c`.
* A single passing run proves nothing about a race.
