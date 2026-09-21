# T2 AgentX sweep driver log

One section per ladder rung. Each records: the config delta from the rung above,
the launch command, the outcome, and per-point results. This file plus the
per-point `t2e-agentx-c<N>/agentx_conc*.json` are the deliverable.

Ladder (as of 2026-09-19, latest), climbed on each GPU memory access fault:

| rung | change from rung above |
|---|---|
| 1 | `JSON_MODEL_OVERRIDE_ARGS='{"index_share_for_mtp_iteration":false}'` |
| 2 | rung 1 + `DECODE_EXTRA_ENV="SGLANG_DSA_FUSE_TOPK=0"` (**untested**, ~2.92x decode TPOT per #36714) |
| 3 | rung 2 + max-running/graph-max-bs 64 on both legs (the T1 never-faulted shape) |
| 4 | stop, report ladder exhausted |

**Rung 2 (`SGLANG_DSA_FUSE_TOPK=0`) status changed TWICE on external evidence —
recorded so a later reader sees it was not quietly shuffled:**

1. **Original:** rung 2 = `SGLANG_DSA_FUSE_TOPK=0`.
2. **Struck out (2026-09-19, earlier):** a parallel P8D8/TP8 session reported that
   applying `SGLANG_DSA_FUSE_TOPK=0` stopped their §3.3 memory fault (two full
   3600 s points) but produced **garbled text** (`1!au!au!...`). Their source
   argument: with fusion off, `_kpool_fused_topk_mapping` returns `(None,None,None)`
   (`dsa_indexer_kpool.py:748`) and the kpool kernel runs `page_table=None`, while
   PR #36714's PD seed remap lives on the fused path. On that basis rung 2 was
   removed and marked do-not-reinstate.
3. **Reinstated as UNTESTED (2026-09-19, later):** the SAME peer **falsified their
   own claim.** They relaunched with fused top-k fully RESTORED
   (`SGLANG_DSA_FUSE_TOPK` unset, verified absent from the decode container) and
   IndexShare off — and the garbling PERSISTED, same `1!au!au!...` signature. So
   the indexer/fusion flag was never the cause, and the #36714 source argument did
   not predict reality. What actually tracks garbling across their three
   deployments is **simulated acceptance** (matches `spec_utils.py:401-434`). Rung
   2 is therefore restored, but as **untested**, not validated.

**Mechanism note the peer supplied (kills a tempting piece of reasoning).** "Real
acceptance is 4.69, we force only 3.61, so forcing is conservative and cannot
corrupt" is **WRONG.** `generate_simulated_accept_index` with
`match-expected`/`real-draft-token` constructs an accept set of a given **size**
(how many), not **which** tokens. Force-accepting a token the target would have
rejected sends topk=1 EAGLE down the draft's own branch and the divergence
compounds. A forced count below the real mean is NOT safe.

Fault detection: `grep -c "Memory access fault" <launchdir>/server-logs/decode-0.log`.
Liveness (NOT the parked `Phase warmup progress` counter): `tot in=/out=`,
`unique_in_srv`, `tput_in_srv` climbing.

---

## Rung 1 — `index_share_for_mtp_iteration:false`

**Config:** `config.yihou.full.sh` (full-mode: prefill HiCache on, max-running
128, simulated acceptance 3.61), with `JSON_MODEL_OVERRIDE_ARGS` set to
`{"index_share_for_mtp_iteration":false}` **after** the `source` line.

**Delta from prior faulting runs (t2..t2d):** those runs had `index_share` ON
(model default true, empty override). This rung turns it OFF — remedy #2 in
`dsa.topk.indexer.bug.analysis.md` §8. T1 ran ~40 min of real MTP with this knob
and never faulted, but T1 was a different shape (max-running 64, HiCache off),
so this is the first test of the workaround under full-mode.

### Bring-up

Launched by a prior driver instance into `t2e-launch/`; router ready 07:01Z.
Knob verified live: all 4 decode ranks' `server-info/decode-0.json` carry
`index_share_for_mtp_iteration":false`; prefill and decode command lines both
carry `--json-model-override-args {"index_share_for_mtp_iteration":false}`.
Image `infera-sglang:v0519-yihou-0917` (base, no thin layer — matches C40).

### Sweep

**Found the stack UP but IDLE** at 07:03Z: bring-up had completed at 07:01 but no
`sweep.yihou.sh` / `agentx_bench.sh` process was running and no `t2e-agentx-c*`
dir existed — the sweep loop was never started after bring-up. Stack was healthy
(router `active_workers:2`, both workers healthy, 0 faults in either log), so I
started the sweep against the live stack rather than tearing down.

Launch command (from the workspace dir):
```
OUT_PREFIX=t2e nohup ./sweep.yihou.sh 40 56 72 96 128 > t2e-sweep.log 2>&1 &
```

CONC=40 started 07:04:46Z into `t2e-agentx-c40/`.

### Sweep interruption — false-negative result check (points 40, 56 both valid)

The first sweep ran points 40 and 56, then **stopped after 2 of 5** at 10:16:47Z
on a FALSE failure. `agentx_bench.sh:105` tests `[[ -s "$result" ]]` immediately
after the client container exits; the container's EXIT trap chowns the output
tree, so the JSON can still be settling when that test runs. CONC=56 printed
"AgentX result is missing" and exited 1 — but the run had **succeeded**
(`Validated aiperf request error rate: 0/3309 = 0.000%`, `agentx_conc56.json`
5781 B on disk). The old sweep believed the exit code and took the `break`.

`main` fixed `sweep.yihou.sh`: a non-zero bench exit is now only believed if the
result file is still absent 20 s later (the `elif rc=$?; sleep 20; [[ -s ... ]]`
branch, lines 32-45). Points 40 and 56 are DONE and valid — not re-run.

**Resumed** the remaining points against the still-live, still-healthy stack
(verified: router `active_workers:2`, 3 containers on 135 + 1 on 138, 0 faults,
stack up >3.4 h with zero faults):
```
OUT_PREFIX=t2e nohup ./sweep.yihou.sh 72 96 128 > t2e-sweep-resume.log 2>&1 &
```
CONC=72 started 10:26:09Z. Watcher restarted against it.

### Rung-1 results

Both completed points ran full-mode config (HiCache on, max-running 128,
simulated acceptance 3.61), 3600 s profiling window each.

| metric | CONC=40 | CONC=56 |
|---|---|---|
| requests total / profiled / warmup-dropped / error-dropped | 4461 / 4017 / 444 / **3** | 3928 / 3309 / 619 / **3** |
| throughput input / output / total (tok/s) | 156,207 / 1,155 / 157,362 | 121,663 / 917 / 122,579 |
| per-GPU total / output (tok/s, 8 GPU) | 19,670 / 144.3 | 15,322 / 114.6 |
| TTFT mean / p50 / p90 / p95 (s) | 8.36 / 4.19 / 17.27 / 30.15 | **37.94** / 17.77 / 98.06 / 135.55 |
| E2E mean / p50 / p90 (s) | 21.33 / 11.66 / 48.57 | 50.56 / 30.34 / 121.80 |
| ITL/TPOT mean / p95 (ms) | 12.9 / 18.3 | 13.5 / 18.3 |
| interactivity p50 (tok/s/user) | 80.3 | 78.6 |
| output_actual / output_expected mean (tok) | 1042.5 / 1321.6 | 1004.5 / 1352.0 |

**KV-capacity limited, not clean scaling.** Total throughput *drops* 157k -> 123k
going 40 -> 56 concurrency, and TTFT mean balloons 8.4 s -> 37.9 s. At CONC=56 the
server sat at `kv_usage` 92-99 % with a persistent 9-15-request wait queue
(observed by `main`). The system is already saturated at these concurrencies, so
72/96/128 will be KV-capacity-limited too — their numbers must be read as
saturation behaviour, not as headroom scaling.

**Error-dropped records (both points: 3 each).** All `InvalidInferenceResultError`:
_"No responses with actual content were received from the server (only
usage/metadata, null/empty data, or [DONE] markers)."_ 3/4461 = 0.067 % at
CONC=40, well under the 10 % bar; dropped from profiling, not run failures. At
CONC=40 two of the three are turns 25 and 27 of a single conversation
(`0a279af1...`), the third is turn 85 of another.

**NOT caused by `index_share=false`** (correction — an earlier note here wrongly
attributed them to the workaround). `main` checked `request_accounting` across
runs: the 2026-09-18 C40 run with `index_share` **ON** (model default) also had
**exactly 3** `records_error_dropped`, as do both t2e points with it **OFF**. So
these are an independent, pre-existing phenomenon of this stack, not a cost of the
workaround. The count being stubbornly **3** across runs with different configs
and different request totals is itself odd and worth flagging — but it is not yet
explained, so it is left open, not concluded.

**Fault status through rung 1: ZERO memory access faults** across points 40 and
56 (>3.4 h continuous load), vs 4 faults in 5 runs with `index_share` ON. This is
the central positive signal for the workaround.

**Outcome:** points 40, 56 completed; CONC=72 was in progress when the stack was
stopped (see below). The c72 partial is preserved as-is.

### The t2e points are VALID timing data (a mid-task false alarm, corrected)

**A garbled-text probe of t2e on 2026-09-19 ~11:4x was briefly read as an
invalidation and this stack was torn down. That inference was wrong.** Under
`SGLANG_SIMULATE_ACC_LEN=3.61` garbled text is the DESIGNED behaviour, not a
symptom, so it does not bear on the timing numbers. The teardown was an
avoidable loss of a healthy run, recorded here so the reasoning is not repeated.

Why garbled text is expected under forced acceptance
(`python/sglang/srt/speculative/spec_utils.py:401-434`): the simulator FORCES
the accept index to the first `simulate_acc_len` positions regardless of what the
target would accept, and under `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`
writes the draft's UNVERIFIED candidate tokens straight into `predict` (in the
`fixed` mode it emits token id 100 repeatedly). Degenerate output is what the
simulation does by construction. The sibling C32/C40 pack-up says the same about
its own numbers: "timing measurements under simulated MTP acceptance, not
correctness evidence."

**Therefore the t2e CONC=40 and CONC=56 tables above are VALID** timing
measurements — exactly what T2 was asked to produce — and may be quoted as such.
They additionally document that the decode-side GPU memory access fault does not
occur in this cell across >4.6 h of load.

**One honest limitation to carry into any write-up** (applies to every simulated
run here and in the earlier pack-ups): under forced acceptance the emitted text
is unverified draft output, so **OSL and every token-count-derived figure
describe the DRAFT's behaviour, not the target's.** Input-side throughput and
TTFT are prefill-side and unaffected.

**Do not gate simulated-acceptance points on text coherence.** The
`coherence_gate.yihou.sh` is kept and is correct — but for REAL-acceptance runs
like T1, where garbled text IS a fault. Running it as a pre-condition for a T2
point tests a property the simulation is designed to violate.

---

## Rung 1f — coherent cell: `index_share=false` AND `--disable-custom-all-reduce`

**The fix.** Restore `--disable-custom-all-reduce` on the decode leg (T1's cell,
which produced coherent text at real `spec_accept_length` 2.15 and never
faulted). New config `config.yihou.full.dcar.sh`: sets
`DECODE_EXTRA_ARGS=--disable-custom-all-reduce` then sources
`config.yihou.full.sh` (unedited — it is t2e's evidence). Resolved and verified:
the ONLY delta vs t2e is the restored flag; everything else (full mode, HiCache
on, max-running 128, simulated 3.61, base image, tilelang/sgl-kernel) is
identical.

**Teardown of t2e.** Sweep + bench killed by PID; orphaned AgentX client ssh and
its remote container removed; `stop.sh` clean; both nodes verified at VRAM
baseline (~298 MB on all GPUs 2-5) after HiCache release.

**Coherence gate added.** `coherence_gate.yihou.sh` runs 3 temperature-0 probes
and fails on a leading `1!` or any short substring repeated 5+ times.
`sweep.gated.yihou.sh` runs it BEFORE the sweep and after EVERY point, aborting
rather than burning hours on garbage. Credit: the parallel TP8 session, which
lost two 3600 s points to exactly this.

**Prefix `t2f`, fresh OUT_DIRs.** Bring-up healthy 11:52:46Z, 0 faults, decode
`disable_custom_all_reduce=true` + `index_share_for_mtp_iteration=false` confirmed.

### Pre-sweep gate garbled — EXPECTED under simulation, NOT a stop (resolved)

The t2f pre-sweep coherence gate degenerated on all three probes
(`notes/coherence-t2f-pre-fixed.txt`, max_tokens 512, temp 0):

```
"What is 2+2?..."          -> '1!1The 1s!1The 1s!1The 1s!...'
"List the first 10 primes" -> '1!1. 1, 1, 1, 1, 1, 1, ...'
"Reply with exactly:..."   -> '1!1. 1.  **. 1. 1. 1. ...'
```

This was briefly treated as a coherence failure of the "safe" cell. **It is not
a failure — it is the designed behaviour of simulated acceptance** (see the t2e
section above and `spec_utils.py:401-434`). `SGLANG_SIMULATE_ACC_LEN=3.61` forces
acceptance and emits unverified draft tokens, so degenerate text is expected on
ANY simulated cell, including this one. The gate must NOT be a pre-condition for
a T2 (simulated) point; it is retained for real-acceptance runs like T1.

**The gate bug-fix stands on its own merit.** GLM-5.2 is a reasoning model: at
short `max_tokens` the whole reply lands in `reasoning_content` with `content`
empty, so the first gate (checking only `content`) falsely passed. Fixed to
inspect BOTH fields, use 512 tokens, and fail on an all-empty reply. This matters
for real-acceptance runs, where a genuinely garbled reply would otherwise slip
through. Unit cases retained in-band (`bash -n` clean; 8/8 detector cases pass).

**Resolution.** No teardown, no differential, no ladder climb. The t2f stack is
the better arm than t2e — it carries `--disable-custom-all-reduce`, the user's
stated "corrected MTP config", differing from t2e only by that flag. Proceeding
to run the ungated sweep on the live t2f stack.

### t2f sweep

Ungated sweep (`sweep.yihou.dcar.sh`, = the fixed `sweep.yihou.sh` late-result
guard but pointed at `config.yihou.full.dcar.sh` so the bench references the
config the stack actually ran) started against the live t2f stack.

**Outcome: 4/5 points obtained (40, 56, 72, 96). CONC=128 not obtained.**

Points 40/56/72/96 completed and are valid timing data on the t2f cell
(`index_share=false` + `--disable-custom-all-reduce`, full mode, simulated 3.61).
Result JSONs on disk: `t2f-agentx-c{40,56,72,96}/agentx_conc{N}.json`. Zero GPU
memory access faults across the whole sweep (confirmed by live `docker logs`
scan, see the watcher caveat below).

**CONC=128 failed TWICE, for two unrelated non-GPU reasons — recorded as "not
obtained", cause of the 2nd not established. Not a ladder trigger either time
(no `Memory access fault`), so no rung was changed.**

1. **1st attempt — shared /home NFS volume hit 0 bytes** (`10T 10T 0 100%`),
   external, not ours (our workspace is 16 G of a 10 TB volume shared by hundreds
   of users). The artifact tree died writing at ~23:42; the whole client + log
   follower process tree went with it. Partial preserved:
   `t2f-agentx-c128-nfsfull/`. Resumed after the volume recovered to ~700 G free.
2. **2nd attempt — decode-leg HANG** (PD KV-transfer wedge), ~00:53:59-01:14.
   Decode stopped completing requests at 00:53:59 (client frozen at
   returned≈346/1422, in_flight=149), then logged Mooncake `Decode handshake
   failed` / `Decode transfer failed`, including for the `/health` probe (so
   decode `/health` -> 000 timeout while it still answered `GET /metrics` 200).
   GPU VRAM held (~269 G) but idle; **0 GPU faults on both legs** (live
   `docker logs`); NCCL watchdog silent; `/home` fine at 665 G. Evidence:
   `t2f-c128-hang/` (HANG-SUMMARY.txt, gzipped docker-log tails both legs, client
   log, rocm-smi, df, health). Partial preserved: `t2f-agentx-c128-hang-partial/`.

**Decision (owner's call, agreeing with `main`'s read after first-hand
verification): stop at 4/5 and pack up.** Rationale: the 4 obtained points define
the saturation curve; CONC=96 already shows capacity collapse (throughput ~15 %
of CONC=40, server prefix-cache hit ~22 % vs ~96 % theoretical), so CONC=128 sits
deeper in the same saturated regime with low marginal value; it has now failed
twice for unrelated infra/transport reasons at ~1 h each; and the task book lists
stopping at the 128 point as an acceptable end. A third attempt was judged not
worth the ~1 h + recurrence risk.

**Teardown.** Sweep/bench/watcher killed by PID; orphan client ssh + its remote
container removed; `stop.sh` clean; decode node (138) GPUs 2-5 back to baseline
(~298 MB); prefill node (135) draining the HiCache pool (~282 G -> baseline, the
expected ~90 s+ release lag — not relaunching, so left to drain). Nothing deleted.

**Watcher caveat (a real monitoring gap, caught by `main`).** `notes/watch.sh`
scanned `t2f-launch/server-logs/decode-0.log` for faults — but that file's writer
died in the 1st CONC=128 NFS event and its mtime froze at 23:42:39. So every
`faults=0` the watcher logged AFTER the ~23:42 resume was read from a DEAD file,
i.e. unverified, for ~90 min. The true fault count was re-confirmed 0 by scanning
live `docker logs` on both containers. Lesson for any future run: scan
`docker logs <container>` directly, not the follower's on-disk file, which can
silently stop being written.
