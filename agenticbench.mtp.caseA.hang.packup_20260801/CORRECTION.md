# ⚠️ CORRECTION — read this before anything else in this kit

**Superseded by `../agenticbench.mtp.caseA.packup_20260801/`**, which ran the
same workload to completion. This kit documents the *failure*; its diagnosis was
**wrong on two counts**. The artifacts (logs, configs, timings) remain valid —
the *conclusions* do not.

## Wrong claim 1 — "prefill `TCPStore recvValue failed` is the first failure"

Stated in `README.md`, `notes.md`, `environment.md` and `REPRODUCE.md`.

**It is not.** The first failure is prefill **DP0 aborting on a GPU OOM**, three
seconds earlier, at `/shared_nfs/yihou_agbench_mtp/logs/probe_prefill.log:41399`:

```
rocdevice.cpp:3582 ... HSA_STATUS_ERROR_OUT_OF_RESOURCES ... Available Free mem : 78 MB
Fatal Python error: Aborted
```

Causal chain, every link logged:

1. 12:22:03 prefill DP0 logs its last `Prefill batch`, then goes silent
2. ~12:22:05 DP0 GPU down to 78 MB; HSA allocation fails; the process aborts
3. 12:22:06 prefill rank 5/6/7 `TCPStore recvValue failed` — **a consequence**
4. 12:22:08 decode's 8 ranks go silent (upstream gone)
5. 13:49:28 prefill DP1–7 hit `watchdog_timeout=3600` — hung 87 min

**Why the kit got it wrong:** the analysis read `g1_prefill.log` (a gate round)
instead of `probe_prefill.log`, which is what this attempt actually wrote. The
claims of "no traceback, GPU fault, or exception anywhere" are artifacts of
reading the wrong file.

## Wrong claim 2 — "the direct cause is the missing `--disable-custom-all-reduce`"

This was the correction *to* claim 1, and it is also wrong.

The flag **was** genuinely missing on the prefill leg, via a real two-variable
trap in the leg script:

```bash
CUSTOM_AR="${CUSTOM_AR:-$([ "$MTP" = "1" ] && echo 0 || echo 1)}"
#   prefill has MTP=0  ->  CUSTOM_AR=1  ->  the flag is never passed
```

Fixing it is correct and worth keeping — it freed 0.85 GB/rank. **But it is not
the cause:** the very next attempt (`logs/armA2_prefill.log` in the superseding
kit) crashed with an identical HSA OOM while
`disable_custom_all_reduce=True` was verified on both legs and
`AiterCustomAllreduce` counted **0**.

## The actual cause

**Prefill `--mem-fraction-static` was 0.88.** At Case A's prompt lengths, dp8
DP-attention activation memory exceeds what `1 − mem_fraction_static` leaves
outside the static reservation. It is not KV exhaustion — `token usage` reads
0.01–0.05 at the abort, i.e. the KV pool is nearly empty.

Lowering it to **0.80** raised per-rank free memory from 33.8 GB to **284 GB**
and the same workload then ran the full 4,007 s window with **zero faults**.

The counter-intuitive direction is the crux: **prefill activation OOM is fixed by
*lowering* mem-fraction-static** — the opposite of the decode-side retract fix.

This was already documented, before this kit was written, in
`../caseA.glm52.fullfeature.packup_20260801/` (the vultr sibling run of the same
branch and image), patch `0002`.

Full mechanism, including a third discarded hypothesis (the `fp8_mqa_logits`
quadratic buffer — refuted because the Phase-1 sweep passed 8/8 with a *larger*
buffer than the crash had), in
`../agenticbench.mtp.caseA.packup_20260801/notes/notes.config.md`.
