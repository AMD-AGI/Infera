# Notes — Exp 2

## 1. The mechanism — why turning IndexShare off can replace patches 2 and 4

**What the deadlock is.** On the PD decode leg with DPA8 + MTP, the draft graph/eager
decision in `eagle_worker_v2.py::draft()` is made **per rank** from rank-dependent inputs.
When ranks disagree on that decision, some replay a captured graph while others run eager
— and the two paths issue *different collectives*. The ranks then wait on each other
forever.

**Why IndexShare is the source of the divergence.** One term of that guard is
`draft_input.dsa_topk_indices is None`. On the PD decode leg this field is seeded from
RDMA-shipped per-request payloads (`eagle_disaggregation.py:54-59`):

```python
dsa_topk_indices = None
dsa_indices_list = [req.output_dsa_topk_indices for req in batch.reqs]
if dsa_indices_list and all(t is not None for t in dsa_indices_list):
    dsa_topk_indices = torch.stack(dsa_indices_list, dim=0).to(batch.device)
    if torch.any(torch.all(dsa_topk_indices < 0, dim=1)).item():
        dsa_topk_indices = None
```

It is therefore a function of **which requests this rank happens to hold**. A rank with an
empty `batch.reqs` gets `dsa_indices_list == []` → falsy → `None`, while a busy rank gets a
tensor. That is rank divergence by construction.

In single-node **mix**, the same field is seeded locally by `_draft_extend_for_decode`, by
the same code on every rank — which is why mix never hangs and only PD does.

**So there are two ways out:** make the decision uniform (our patch 4: vote it over the TP
group), or remove the seed so the term stops diverging (this arm: IndexShare off). This
kit measures the second.

**Context.** Upstream issue **#32527** (2026-07-27, 8× B30Z / GLM-5.2-FP8 — a different
platform from both ours and llying's) reports the same defect with the same localization,
independently. It proposes a *third* strategy: install a dummy all-zeros seed so the guard's
fourth term is false and all ranks stay on the graph path.

## 2. Prefill MTP is not optional in this arm

**What.** `boot.sh` sets `PREFILL_MTP=1` for arm `e2` only. Exps 1 and 3 run MTP on the
decode leg alone.

**Why.** Without MTP on the prefill leg, the prefill worker never runs
`_draft_extend_for_prefill`, never fills `req.output_dsa_topk_indices`, and never registers
the draft KV pool for RDMA. The IndexShare seed then **cannot reach the decode leg at all**
— so "IndexShare off" would be indistinguishable from "IndexShare was never on", and the
arm would prove nothing.

**How to confirm it took effect.** `strings logs/prefill.log | grep speculative_algorithm`
must show `EAGLE`. This kit's `logs/prefill.log` does.

**Context.** This also means Exp 2 differs from Exp 1 in **two** variables, not one
(IndexShare *and* prefill MTP). That is unavoidable — llying's recipe couples them — but it
is why the ~5 % accept-length difference between the arms cannot be attributed to
IndexShare. See README, "Comparison with Exp 1".

## 3. The cost of this workaround, and its expiry date

**Today it is close to free.** Under PD, IndexShare's *consumer* is already disabled by
`should_use_dsa_fused_topk` — the seed is produced and then not used by fused top-k. So
switching it off costs approximately nothing right now. llying measure accept length
3.78/4 with IndexShare on and off; a reproducer on #32209 reports 3.239 vs 3.24. (Both
second-hand here; we did not run an IndexShare-on control on this node pair.)

**But upstream PR #31477 exists to delete that TODO.** `[Spec][PD] Enable fused TopK for
GLM-5.2 MTP IndexShare` (+93/−4, 3 files) adds
`should_remap_pd_dsa_seed_to_local_slots()` and materializes the RDMA-shipped
request-relative positions into decode-local physical slots before the seed enters the
draft loop. When it lands, IndexShare becomes genuinely useful under PD and this override
starts costing (~3 % TPOT, per llying — not measured by us).

**Status as of 2026-07-30** (checked with `gh` this session): #31477 is **open**, with
`reviewDecision = REVIEW_REQUIRED` — i.e. **no approval**. CI status not checked. Its
timing is therefore **unknown**; do not plan around it landing soon, and do not plan around
it not landing.

**Implication.** IndexShare-off is a good answer *today* and a dated one. If IndexShare is
wanted, patch 4 (or #32209's vote, or #32527's dummy seed) is the durable fix.

## 4. Traps hit or avoided this run

### 4.1 The override could have been silently ignored (avoided by checking)

`--json-model-override-args` takes JSON containing double quotes and braces, passed
through host → `spur exec` → `docker exec` → `sh`. A quoting mistake anywhere yields a
server that starts fine with the override **dropped**, and the arm would then measure the
default build while looking like a pass.

`boot.sh` writes the environment to a file and sources it inside the container instead of
interpolating it through that chain. `REPRODUCE.md` step 5 then reads the value back out
of the server's own startup banner. Both were done here; `logs/decode.log` contains:

```
json_model_override_args='{"index_share_for_mtp_iteration":false}'
```

### 4.2 `/home` NFS was 100 % full — caches had to move (hit)

Mid-setup, writes to `/home` began failing with `EDQUOT`; the 10 TB export was at 100 %.
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` defaulted under `/home`, and a failed JIT
cache write is **silent** — it surfaces only as a slower boot, which on this stack looks
like normal (cold start is already ~8 min). The workspace moved to
`/shared_nfs/yihou_exp3way` and `boot.sh` exports both cache dirs there. Nothing was
deleted; only the write target changed.

### 4.3 The `.pyc` staleness trap (avoided by construction)

A stale `.pyc` silently reverts a patch — the source shows the fix, the runtime does not.
This has already invalidated one full experiment on this stack. `apply_arm.sh` purges
`__pycache__`, recompiles, and greps the **bytecode**, using *identifiers* rather than `#`
comments (the compiler discards comments, so a comment marker is a guaranteed false
negative).

This arm depends on the check in **both** directions: markers present for patches 1 and 3,
and anti-markers absent for patches 2 and 4.

### 4.4 Server logs contain binary bytes (hit repeatedly)

Plain `grep` reports "binary file matches" and `grep -c` returns **0** — which reads as
"no errors" when it means "grep gave up". Always `strings <log> | grep` or `grep -a`.

### 4.5 Wrong patch source, caught before it cost a run (hit)

The first `apply_arm.sh` sourced the baseline patches from `~/glm52_fix/fix_bug*.py`. Those
are from an earlier round and encode a **different** patch-2a fix than the verified kit
(an empty-batch `.max()` guard plus a `base_spec_worker.py` edit, versus
`max_seqlen_k = req_to_token.shape[1]` in `dsa_backend.py` only). The kit diffs are the
2540/2540-verified artifacts; the scripts are not. Fixed to use the kit diffs.

### 4.6 A 503 that is not a server failure (hit on the sibling arm)

A router whose circuit breaker is still open returns HTTP 503 in ~0.4 s. That looks exactly
like a dead backend. The tell is **latency**: a real backend failure takes seconds to
surface; a tripped breaker answers immediately. Restart the router and re-probe before
concluding anything.

Each arm got its own router port (E2: 8120) so one arm's open breaker can never be mistaken
for another arm's failure.

## 5. Reading the results honestly

- **`full tok` < request count is not a failure.** `max_new_tokens=512` is a cap; greedy
  decoding hits EOS earlier on some prompts. The criterion is the `ok` column.
- **`acc_len` must be > 1**, or MTP was silently bypassed.
- **`dp ranks: [0..7]`** confirms the whole DP group served traffic.
- **No negative control was run in this arm.** We did not remove the IndexShare override on
  this node pair to show the hang returns. So this kit shows the configuration *works*; it
  does not independently re-demonstrate that the bug exists here.
- **conc=128 was not run.** Criterion was conc=32; 64 was added as headroom.
