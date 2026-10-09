# notes — gotchas, corrections, and what is still open

The ordered narrative is `working_process.md`. This is the part worth re-reading.

---

## The one thing that will bite you: the RCCL start race

Two legs on one host initialise their 4-GPU RCCL communicators concurrently and
one of them loses:

```
rccl .../p2p.cc:256 NCCL WARN hipIpcGetMemHandle failed : invalid argument
RuntimeError: NCCL error: unhandled cuda error
Rank 0 scheduler died during initialization (exit code: -3)
```

The previous experiment added a **same-node start gate** to `launch.sh` — wait
for the already-started leg's `/health` before starting the next. **It is not
sufficient.** This run fired the gate and decode died anyway. That is now **N=2**,
so "the gate fixes it" is refuted rather than merely doubted.

What works, twice: leave prefill up, wait, start decode by hand. Settle times
**~9 min** (this run) and **~17 min** (the previous one).

**STILL UNSETTLED — do not write "wait N minutes" into the harness.** Both
successes are equally consistent with (a) a real settling threshold and (b) a
probabilistic race that a retry simply wins, and the two settle times differ by a
factor of two. Turning this into a fix on the present evidence would repeat the
"the GPU swap fixed it" error from the earlier same-node work, which had to be
retracted.

**The experiment that would distinguish them:** retry repeatedly at a *fixed,
short* settle — say 1 minute. If a 1-minute retry also succeeds several times in
a row, it is a race and the wait is irrelevant; if it reliably fails at 1 min and
reliably succeeds at 9, there is a threshold worth encoding.

---

## Three corrections made during the config audit

All three were the **same failure mode: evidence that looks like evidence but is
not.** Recorded because the next person will meet the same traps.

### 1. `AutoConfig.from_pretrained(...)` cannot prove the override

The first attempt to prove `index_share_for_mtp_iteration=true` was live ran
`AutoConfig.from_pretrained(model_path, trust_remote_code=True)` and got `True`.
That call reads the **checkpoint's own `config.json`**, which already carries
`True`, and never applies `--json-model-override-args`. It returns `True` whether
the override is honoured or silently ignored — the exact ambiguity the check was
meant to resolve.

**The conclusion survives on a different argument:** the checkpoint default and
our override agree, and nothing else sets the value `false`, so **both branches
resolve to `True`**. The engine runs with `index_share = True` regardless — but
the override *mechanism* is **unproven**. The discriminating test (drive the same
machinery with `false`, expect `False`) was **not run**.

This is the first run in the project where the two branches coincide; every
earlier run overrode the value to `false`, which is why the ambiguity is new.

### 2. `[aiter] import [module_aiter_core]` is AITER, not triton

An early draft cited that line as evidence triton was in use. AITER is enabled in
**every** run in this project, including all the `tilelang` ones, so it says
nothing about the DSA backend — and it was read from the **round-001** decode
log, the attempt that died in the RCCL race and never served a request.

Replaced with running-engine evidence: `libtriton.so` mapped into the live decode
scheduler process, an sglang triton cache at `/root/.cache/sglang/triton/<hash>/`
also mapped into that process, **175 compiled `.hsaco` kernels**, and
`TRITON_CACHE_DIR` unset with `HOME=/root`.

**Stated limit:** triton being loaded does not separate DSA-triton from sglang's
other triton users (`mamba_backend`, `linear_attn_backend` both default to
triton). DSA-triton execution is inference. Hence item 10 is `pass with limit`.

### 3. The HiCache evidence cited the wrong allocation

Prefill makes **two** distinct host-memory allocations per rank, and they are easy
to confuse:

- `Allocating kv hierarchical KV host pool: 3423488 tokens, 153.81 GB host memory`
  — the **HiCache** pool. This is the evidence for HiCache being on, and the
  figure matches the 20260920 cross-node reference exactly.
- `Allocating 35.25 GB host memory for DSA indexer (layout=page_first)` — a
  **separate** DSA indexer allocation, unrelated to HiCache.

Four of each, one per DP rank ⇒ ~189 GB/rank, ~756 GB total, consistent with the
~757 GB RSS observed on the prefill container.

---

## `index_share` is not cosmetic — it changes the PD wire schema

`disaggregation/utils.py:70-78`'s `get_dsa_seed_metadata_dim()` returns non-zero
**iff** the resolved flag is truthy *and* the model is DeepSeek-DSA (GLM-5.2 is).
Its only consumer is `scheduler.py:1518`, which passes it as
`output_dsa_topk_indices_dim` into the decode leg's `MetadataBuffers(...)`. So the
flag genuinely alters the PD metadata format.

**But it is consumed and never printed.** Two candidate observables were checked
against a real A/B log pair (this run `true` vs the previous workspace's
`019-hicache-decode-retry` `false`, same shape and image): no
`MetadataBuffers`/`output_dsa_topk_indices_dim` line in either — the only "seed"
hits are `random_seed` in the `server_args` dump — and no mooncake
registration-count difference at `MC_LOG_LEVEL=INFO`.

**There is no running-engine observable for this value.** Recorded so nobody
repeats the search.

---

## Other gotchas

**Pass the overrides to `agentx_bench.sh` too, not just `launch.sh`.** The
benchmark validates the live deployment against the config and refuses to start
on a mismatch: `agentx_env: ERROR: prefill-0: live hicache=True, config expects
False`. Putting the deltas in the config **file** makes both scripts agree. The
guard is a feature — it caught a run that would otherwise have been mislabelled.

**Triton compiles cold every bring-up.** `engine.sh` has no triton cache handling,
so the cache lives at `/root/.cache/sglang/triton/` **inside** the container and
is not bind-mounted. No same-node race, but slow first use that can look like a
hang.

**zsh eats `-v $M:$M:ro`.** `:r` is a zsh history modifier that strips the
extension, turning `/…/GLM-5.2-MXFP4` into `/…/GLM-5` + `o`. The router then dies
with `HFValidationError … not a local path`. Always quote: `-v "${M}:${M}:ro"`.

**Never `pkill -f <pattern>`** from a shell whose own command line contains the
pattern — it kills the calling shell. This happened twice across the two
experiments.

**Teammate status messages carry the state of the moment they were written.**
Several arrived minutes stale and caused two wrong readings, one of which briefly
recorded a teammate as negligent when it was actually blocked on a dead decode
leg. Check the timestamp before acting on a report.

**The Bash tool caps sleeps at 10 minutes.** Longer waits are silently truncated.

---

## Still open

- **Whether the start-settle is causal or the retry just wins a race.** N=2 both
  ways; the distinguishing experiment is above.
- **Whether the override mechanism works.** Unproven; irrelevant to this run's
  correctness because both branches land on `True`, but it would matter the next
  time someone wants to override the value *away* from the checkpoint default.
- **IndexShare stability.** The reference fault at ~1 h 26 m of sustained load was
  never reached — this run's profiling ended at ~1 hour with zero faults. The
  question is untouched, not answered.
- **triton vs index_share, separately.** They moved together. The ~12.5 % gain
  over the same-node tilelang/false baseline belongs to the pair.
- **Why the previous attempt's client died at 89 %.** The container was `--rm`, so
  no exit code survived. Not OOM. `ServerAliveInterval` was added as a targeted
  mitigation and this run completed, but that is one observation, not a diagnosis.
