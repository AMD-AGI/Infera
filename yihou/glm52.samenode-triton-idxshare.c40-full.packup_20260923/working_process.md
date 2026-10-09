# Working process — same-node P4D4, triton DSA + index_share ON, CONC=40 full

Task of record: `spec/mission.md`. One row per round.

## Round index

| # | dir | purpose | outcome |
|---|---|---|---|
| 0 | (research) | Is `triton` an accepted DSA backend here? What does "index_share on" mean? | **done** — see below |
| 1 | `rounds/001-bringup-triton/` | Bring up with triton + index_share=true + prefill HiCache | **prefill healthy; decode died in the RCCL start race DESPITE the gate** |
| 2 | `rounds/002-decode-retry/` | Manual decode retry ~9 min after prefill went healthy | **clean** — both legs healthy |
| 3 | `rounds/003-agentx-c40/` | The deliverable: AgentX CONC=40, full mode | **PASSED** — full 3,613.6 s window, 4,157 reqs, **0 errors**, 168,260 tok/s (21,032/chip). See `results/RESULT.md` |
| — | `analysis/config_audit.yihou.md` | Prove the four deltas are live before spending 2.5 h | **done** — 9 pass, item 10 `pass with limit` |

## Round 0 — research before touching anything

All first-hand, against this image / this model:

1. **`triton` IS accepted.** `python3 -m sglang.launch_server --help` gives
   `--dsa-prefill-backend {flashmla_sparse,flashmla_sparse_q8,flashmla_kv,flashmla_auto,flashinfer_sparse_mla,fa3,tilelang,triton,aiter,trtllm}`
   and the same list for `--dsa-decode-backend`. This also **confirms two inherited
   claims**: `--dsa-topk-backend` accepts only `{sgl-kernel,torch,flashinfer}` (no
   `aiter`), and `flydsl` appears in neither list.
2. **"index_share on" is the MODEL DEFAULT.** The checkpoint's own `config.json`
   carries `index_share_for_mtp_iteration = True`, so every previous run in this
   project was *overriding it off*. Set explicitly to `true` anyway, so the value is
   visible in `/get_server_info` instead of inferred from an absent flag.
3. **`engine.sh` has no triton cache handling** (grep: no `TRITON` anywhere). The
   reference kit manages `TRITON_CACHE_ROOT` in its own `common.sh`. Our containers
   mount only the model, libionic and the AITER cache, so the triton JIT cache is
   **container-private**: no same-node race, but a cold compile every bring-up. Slow
   first use is not a hang.
4. Reference kit read: `yaocheng/2p1d-sweep-triton-dsa-20260922/config.sh:87-89` —
   `triton`/`triton` with `DSA_TOPK_BACKEND` cleared. Its shape is 2P1D TP8/DP8 over
   three nodes, so only the DSA setting is borrowed, not the topology.

Config dry-run resolved for both roles before launch: triton/triton, topk empty,
`{"index_share_for_mtp_iteration":true}`, prefill HiCache 1 / decode 0, 3600 s,
warmup 10/lane, stride 256, GPUs 0-3 / 4-7, decode `--disable-custom-all-reduce`,
per-role AITER roots, sim acc 3.61, both legs `ionic_0-3`.

## Monitoring set up before the run, not after

The previous attempt died at 89 % and went unnoticed for hours. This run has:

- **`watchdog`** — polls every ~3 min for segfaults, `NCCL error`,
  `hipIpcGetMemHandle failed`, **`Memory access fault`**, `<defunct>` schedulers,
  `/health` regressions, container exits, benchmark-log staleness > 10 min (by
  **mtime**, which is how the last failure would have been caught), and the client
  container vanishing. Escalates immediately; heartbeats every ~20 min. Explicitly
  told to observe and report, never to fix.
- **`config-auditor`** — proves the four intended deltas at **two layers** (container
  argv *and* live `/get_server_info`) before the benchmark is allowed to consume
  2.5 hours, including closing the gap the leader left last time: whether the
  override is actually absorbed into the `hf_config` that
  `srt/disaggregation/utils.py:72` reads.
- Cron: mission re-injection every 10 min, team poll every 20 min, with the leader
  spot-checking independently rather than trusting the watchdog alone.

## Carried risk, taken deliberately

`glm52.p8d8.agentx-sweep.packup_20260920`: this shape with IndexShare **ON** faulted
at **1 h 26 m**; with it **off** it ran **13 h 33 m** fault-free. The user directed it
on. If a memory fault occurs it is a **finding**, to be captured and reported — not a
setup error to be quietly reverted.

## Round 1 — the start gate is insufficient. N=2, not N=1.

prefill came up healthy. The same-node start gate fired as designed
(`same-node gate: waiting for the already-started leg(s) ... to answer /health`)
and **decode died anyway**:

```
RuntimeError: NCCL error: unhandled cuda error
Rank 0 scheduler died during initialization (exit code: -3)
container stopped: status=exited exit=143
```

This is the **second** time the gate fired and decode still lost the RCCL race
(the first was the previous workspace's HiCache run). So "the gate fixes the
race" is **refuted**, not merely doubtful. The gate's release point —
`/health` 200 — is too early.

`watchdog` escalated within minutes and, importantly, reported the **negative**:
*"No `Memory access fault` is present."* That negative is what separates this
known RCCL race from the IndexShare fault the mission warns about, and it is why
the escalation was actionable rather than just alarming.

## Round 2 — manual retry, clean again. Also N=2.

Timeline, recorded because the mission asks for it:

| event | time (UTC) |
|---|---|
| decode died | 08:13:11 |
| prefill `/health` 200 since | ~08:12 |
| manual decode retry started | 08:21:19 (**~9 min** after prefill healthy) |
| decode `fired up and ready to roll` | 08:24:51 |

Zero NCCL errors, zero segfaults, zero memory faults. Previous workspace's retry
was ~17 min after; this one ~9 min. **Two-for-two on both halves** — gate
insufficient, manual retry works.

**STILL UNSETTLED, and must not be written up as solved:** whether the settle is
*causal* or the retry merely wins a probabilistic race. Both successful retries
are consistent with either. Turning "wait ~10 min" into a harness fix on this
evidence would repeat the Round-9 "the GPU swap fixed it" error, which was
retracted once already.

## Round 3 — the deliverable, running

Deployment verified in the **live** `/get_server_info`, both legs, before
spending 2.5 hours on it:

| setting | prefill | decode |
|---|---|---|
| `dsa_prefill_backend` / `dsa_decode_backend` | triton / triton | triton / triton |
| `dsa_topk_backend` | sgl-kernel (flag dropped → default) | same |
| `json_model_override_args` | `{"index_share_for_mtp_iteration":true}` | same |
| `enable_hierarchical_cache` / ratio | **True** / 1.5 | False / — |
| `disable_custom_all_reduce` | False | **True** |
| `speculative_algorithm` | None | **EAGLE** |
| TP / DP | 4 / 4 | 4 / 4 |

Smoke test returned a completion (garbled — simulated acceptance, expected).
Benchmark launched with `DURATION=3600`, `WARMUP_REQUESTS_PER_LANE=10` confirmed
in `runtime.env`.

Hardening against the previous run's death (client container died at 89 %, log
went silent, driver hung on a dead pipe):

1. `ServerAliveInterval=30 ServerAliveCountMax=1000` added to `SSH_OPTS` — the
   long-lived ssh channel carrying `docker run` output is the suspected cause.
2. `watchdog` checks log staleness by **mtime**, not content — content-based
   checks cannot see that failure.
3. `watchdog` watches for the `agentx-client` container vanishing while the
   driver script is still alive.

Now entering the window where the prior packup records IndexShare ON faulting
(~1 h 26 m of sustained load). That is the signal to watch.

## Team poll — 2026-09-23 ~08:53Z

Independent spot-check by the leader **before** reading any teammate report, per
the poll rule: bench log age **0 min** (by `stat -c %Y`, not by content), all
three endpoints **200**, 5 containers up, and **zero** `Memory access fault` /
`Segmentation fault` in both the round-002 decode log and the round-001 prefill
log. Run is healthy and in its profiling phase.

`watchdog` — running, has escalated correctly once and heartbeated since. Polled
for a status, and deliberately asked for the log age **by mtime** and a plain
yes/no on `Memory access fault`, with my own readings supplied so a disagreement
between us would surface rather than be echoed.

`config-auditor` — **FIRST SIGHTING OF A PROBLEM, RECORDED ONLY.** Spawned ~23
min ago, asked to report items 1-3 immediately because a 2.5 h benchmark was
waiting on them; it has sent **no message** and `analysis/config_audit.yihou.md`
does not exist. Per the poll rule I am not intervening this round — only
recording. It did not block anything: the leader verified items 1-4 directly
against the live `/get_server_info` on both legs and started the benchmark.

Polled it with its remaining useful scope narrowed to the one gap the leader did
**not** close: whether the override is actually absorbed into the `hf_config`
that `srt/disaggregation/utils.py:72` reads. That gap matters specifically
because the checkpoint's own `config.json` already carries `True`, so
"absorbed" and "ignored" look identical from outside — it has to be proven, not
inferred. Asked it to state whether it proves this in the **running** engine or
reproduces it in a **fresh** container, since those are different claims.

**If `config-auditor` is still silent at the next poll, intervene** per the rule.

Risk window note: profiling started ~08:31 UTC; the prior packup's IndexShare
fault came at ~1 h 26 m of sustained load, so ~09:55 UTC is when to watch
hardest.

### Correction to the poll above — `config-auditor` was not negligent

Its report and idle notice are timestamped **08:14:54** and predate both the
decode retry (08:21) and the benchmark start (08:31). At that moment decode
genuinely was absent — it had died at 08:13:11 in the RCCL race — so its
"decode container is not present yet" was **accurate for the state it could
see**. It was blocked on a dead leg, not silent through negligence. The
"first sighting of a problem" entry above is withdrawn; the real fault was
mine, for not pushing the state change to it when decode came back.

Re-briefed it with the current state, pointed it at the **round-002** decode log
(round-001 is the failed attempt and must not be audited), told it which items I
had already closed myself, and narrowed it to the single open gap.

Same lesson as the watchdog's stale idle notice at 08:18: **teammate messages
carry the state of the moment they were written.** Check the timestamp before
treating a report as current.

## Team poll — 2026-09-23 ~08:38Z

Leader's independent spot-check first, before reading any teammate report:
bench log age **0 min** (by `stat -c %Y`), `29001/29257/28000` all **200**, five
containers up including `agentx-client`, warmup at **33/444 with errors=0**, and
**zero** `Memory access fault` / `Segmentation fault` / `NCCL error` in both the
round-002 decode log and the round-001 prefill log.

Both teammates **running**. No new problem this poll.

`config-auditor` — the previous poll's "first sighting" was **withdrawn** (it was
blocked on a dead decode leg, not negligent). It has since independently
confirmed items 1, 2 and 4-8 against live `argv` and `/server_info`, matching the
leader's own reading **including the negative** that neither `/server_info`'s top
level nor `internal_states` carries `index_share_for_mtp_iteration`. Its
deliverable file is still absent but it reported progress a minute before the
poll and stated its next action, so this is work in progress, not a stall — **no
intervention.**

### Leader correction — the IndexShare risk window was wrong

I had told `watchdog` to watch hardest from ~09:55 UTC. That was measured from
the *benchmark* start, but **warmup is not sustained load**. Profiling had not
begun at that point; warmup was 33/444 at 08:37 and runs ~20 min. Profiling
therefore starts ~08:53, and the prior packup's fault at ~1 h 26 m of sustained
load maps to roughly **10:20 UTC**. Corrected with `watchdog`, with the added
caveat that 1 h 26 m is **one observation, not a deadline** — keep watching past
it.

### Finding while waiting: `index_share` changes the PD wire schema, but is unobservable at runtime

Traced for `config-auditor`'s open gap:
`disaggregation/utils.py:70-78` `get_dsa_seed_metadata_dim()` returns non-zero
**iff** the resolved `index_share_for_mtp_iteration` is truthy *and* the model is
DeepSeek-DSA (GLM-5.2 is). Its only consumer is `scheduler.py:1518`, which passes
it as `output_dsa_topk_indices_dim` into the decode leg's `MetadataBuffers(...)`.
So the flag genuinely alters the PD metadata wire schema — it is not cosmetic.

**But it is consumed and never printed.** Checked two candidate observables
against a real A/B log pair (this run `true` vs
`yihou-samenode-p4d4/rounds/019-hicache-decode-retry` `false`, same shape and
image): no `MetadataBuffers`/`output_dsa_topk_indices_dim` line in either (the
"seed" hits are `random_seed` in the `server_args` dump), and no mooncake
registration-count difference (only `rdma_transport.cpp:159` relaxed-ordering
lines appear at `MC_LOG_LEVEL=INFO`).

**Consequence:** the deeper half of item 3 can only be closed by a
fresh-container reproduction, which is a **weaker claim** than reading the
running engine, and must be labelled as such. Recorded here so the next person
does not repeat the search.

## Config audit — done, after three rounds of correction

`analysis/config_audit.yihou.md`: **9 items pass, item 10 `pass with limit`.**
All four intended deltas are confirmed live on both legs.

Three corrections were needed, and all three were the **same failure mode —
evidence that looks like evidence but is not**. Worth recording because the next
person will meet the same traps:

1. **`AutoConfig.from_pretrained(...)` → `True` does not prove the override.**
   It reads the checkpoint's `config.json`, which already carries `True`, and
   never applies `--json-model-override-args` — so it returns `True` whether the
   override is honoured or silently ignored. This is exactly the ambiguity that
   made item 3 worth checking in the first place.
   **The conclusion survives on a different argument:** the checkpoint default
   and our override agree, and nothing else sets the value `false`, so **both
   branches resolve to `True`**. The engine runs with `index_share = True`
   regardless — but the override *mechanism* is **unproven**. The discriminating
   test (apply the same machinery with `false`, expect `False`) is **not run**.
2. **`[aiter] import [module_aiter_core]` is AITER, not triton.** AITER is
   enabled in every run in this project, including all the `tilelang` ones, so it
   says nothing about the DSA backend. It was also read from the **round-001**
   decode log — the attempt that died in the RCCL race at 08:13 and never served
   a request; the live leg is the round-002 restart.
   Replaced with running-engine evidence: `libtriton.so` mapped into the live
   decode scheduler, an sglang triton cache at `/root/.cache/sglang/triton/<hash>/`
   also mapped into that process, **175 compiled `.hsaco` kernels**, and
   `TRITON_CACHE_DIR` unset with `HOME=/root` — which independently confirms the
   mission's "container-private cache, cold compile every bring-up" note.
   **Stated limit:** triton being loaded does not separate DSA-triton from
   sglang's other triton users (`mamba_backend`, `linear_attn_backend`), so
   DSA-triton execution is inference. Hence `pass with limit`, not a plain pass.
3. **The HiCache evidence cited the wrong allocation** (fixed by the leader
   rather than sent back a third time). Prefill makes **two** distinct
   host-memory allocations per rank: `Allocating kv hierarchical KV host pool:
   3423488 tokens, 153.81 GB` (HiCache — the correct evidence, and it matches the
   20260920 cross-node reference exactly) and `Allocating 35.25 GB host memory
   for DSA indexer (layout=page_first)` (unrelated). Four of each ⇒ ~189 GB/rank,
   ~756 GB total, consistent with the ~757 GB RSS observed on the prefill
   container.

## Interim server-side counters (NOT the result)

At ~5 min into profiling: `prefix_cache_hit` 91.3 → 91.8 %, `cpu_kv_usage` 100 %
(HiCache host tier saturated), `tput_in_srv` ~178-179 k/s. The previous
`tilelang` + `index_share=false` run read ~95.6 % hit at a comparable point.
**Do not read anything into this yet** — the cache is still warming and the
previous run's interim counters differed materially from its final computed
figures.

## Timing and a limit that must go in the result

Profiling 08:48:34 → ~09:48, aggregate ~10:00-10:10. The prior packup's
IndexShare fault came at ~1 h 26 m of sustained load ⇒ ~**10:15**, which is
**after** this run ends. So a clean finish here **cannot** be read as
"IndexShare ON is stable" — we never reach the point where the reference failed.

## Team poll — 2026-09-23 ~08:57Z

Leader's independent spot-check first: bench log age **0 min** (by mtime),
`29001/29257/28000` all **200**, 5 containers up, and **zero**
`Memory access fault` / `Segmentation fault` / `NCCL error` in both the
round-002 decode log and the round-001 prefill log.

Server-side counters climbing as the cache warms: `prefix_cache_hit`
92.9 → 93.2 %, `cpu_kv_usage` ~99.9 %, `tput_in_srv` 251 k → 256 k/s. Still
**interim counters, not the result** — the previous run's interim readings
differed materially from its final computed figures.

`watchdog` — running, polled. `config-auditor` — finished and stood down.
**No problem this poll.**

Told `watchdog` the two things that matter for the remainder:

1. The previous attempt died at **89 %** of its window. In this run's terms that
   is around **09:40**, shortly before the 09:48 end — so the highest-risk
   moment is near the finish, not the middle. Stale-log escalation matters most
   there.
2. `Memory access fault` stays top priority, with the calibration stated in both
   directions: the reference fault at ~1 h 26 m maps to ~10:15, i.e. **after**
   this run ends, so a clean finish proves nothing about IndexShare stability —
   and equally, 1 h 26 m is one observation, not a rule, so a fault could come
   earlier and the watch must not relax before then.

## Round 3 — DELIVERED

`AgentX passed`. Full **3,613.6 s** window (the previous attempt died at 89 %),
**4,157** requests, **0/4157** aiperf errors, 4 `InvalidInferenceResultError`
dropped. **168,260 tok/s total, 21,032 tok/s/chip.** `spec_accept_length` on all
four decode ranks: 3.575 / 3.500 / 3.525 / 3.650, mean **3.5625** — every rank
moved, none is the idle-gauge artefact. Zero `Memory access fault` throughout.

Versus the cross-node `t2f` reference: **+6.2 %** throughput, TTFT p95 **-25 %**,
with server GPU cache hit within 0.3 points (94.22 % vs 93.91 %) — so unlike the
earlier 0.52x comparison, this difference is **not** a cache-hit artefact.

Full numbers and the five scope limits: `results/RESULT.md`.

## Team poll — 2026-09-23 ~10:05Z

Leader's independent verification first: the deliverable exists
(`results/agentx_conc40.json`, 4,157 profiled, 3,613.6 s, 168,260 tok/s), all
three endpoints still **200**, 4 containers up (the `agentx-client` is gone, as
expected — the benchmark finished), and **zero** `Memory access fault` /
`Segmentation fault` in both engine logs.

Both teammates **idle with their work complete and accepted**:
`config-auditor` delivered `analysis/config_audit.yihou.md` (9 pass, item 10
`pass with limit`) and stood down; `watchdog` monitored to completion and stood
down. **No problem this poll**, and no new work dispatched — the mission's
deliverable is met and what follows is the user's call.

Note carried for future runs: several teammate idle notifications arrived
describing state minutes out of date, which caused two double-takes (once I
wrongly recorded `config-auditor` as negligent when it was blocked on a dead
decode leg). **Check the timestamp before treating a teammate report as
current.** Asked `watchdog` to put an explicit timestamp in the body of future
status messages rather than relying on metadata.

## Open, for the user to decide

1. **Tear down** (8 GPUs held) and pack up.
2. **Single-variable decomposition** — the +6.2 % cannot be attributed to
   co-location, triton, or index_share individually; that needs one run per
   variable.
3. **Extend the run past ~1 h 26 m of sustained load** — the only way to
   actually test the IndexShare stability question this run could not answer.
