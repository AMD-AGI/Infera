# Case A on the full-feature GLM-5.2 deployment — a reproducible engine hang

> # ⚠️ THIS KIT'S DIAGNOSIS IS WRONG — see [`CORRECTION.md`](CORRECTION.md)
>
> The first failure was **not** prefill's `TCPStore recvValue failed`; it was
> prefill DP0 aborting on a GPU OOM 3 s earlier. The cause was **not** the
> missing `--disable-custom-all-reduce`; it was `--mem-fraction-static 0.88` on
> the prefill leg. Superseded by
> [`../agenticbench.mtp.caseA.packup_20260801/`](../agenticbench.mtp.caseA.packup_20260801/),
> which ran the same workload to completion. Artifacts here remain valid;
> conclusions do not.


**Ran:** 2026-08-01, 11:54 – 13:40 UTC
**Author:** yihou
**Nodes:** `crsuse2-m2m-253` (prefill, job 24300) + `crsuse2-m2m-236` (decode, job 24301)
**Status:** **BLOCKED** — Case A did not complete. Three attempts, three hangs.
The deployment is healthy on shorter workloads; it dies under sustained Case A.

## The one-paragraph version

Optimus-AgenticBench Case A was run against the merged-branch deployment
(kvaware + kvd + **MTP** + PD + DPA). A 900 s calibration probe passed cleanly —
509 requests, 3 errors, in-flight 12–17 against a cap of 48. The 4,000 s
deliverable run then ran **healthily for 556 s** — 401 requests completed,
in-flight oscillating 8–29, cache hit 0.8897 against a configured 0.89 — and
then **froze**: completions stopped at exactly 401 and never advanced, while
in-flight ratcheted monotonically to the 48 cap and stuck there.

No traceback on either leg. No `Memory access fault`. No
`Scheduler hit an exception`. The engines were **not** saturated — the prefill
leg averaged **3.1 % KV usage** with a queue depth of 0.7 while the client
reported 48 requests in flight.

## What actually failed, in causal order

This ordering is load-bearing and it **reverses** the first diagnosis. The
earliest failure is on the **prefill** leg, not decode:

| time | event |
|---|---|
| 12:22:03–05 | prefill still emitting normal `Prefill batch` lines |
| 12:21:57 | decode **DP5** logs `#token: 0, #running-req: 1` — drained while still holding a request |
| **12:22:06** | **prefill rank5 + rank6: `TCPStore recvValue failed` inside `ProcessGroupNCCL::HeartbeatMonitor::runLoop`** ← first failure |
| 12:22:08 | decode's 8 ranks all stop logging, mid-decode |
| ~12:22 | driver's completion count freezes at 401 |

The TCPStore peers are `crsuse2-m2m-253:43517` and its slurm alias — i.e. the
**prefill leg's own 8 ranks**, not a cross-node store. So this is an *intra-leg*
c10d store failure on prefill, and decode's `all_gather` stall is downstream of
it.

### The stalled state, captured with py-spy on all 8 decode ranks

`results/hang_stacks/` holds the full dumps. Summary:

| ranks | top of stack |
|---|---|
| DP0, 2, 3, 4, 5, 6, 7 (**7**) | `prepare_mlp_sync_batch → dp_attn.py:98 all_gather → all_gather_into_tensor` |
| **DP1** | `process_decode_queue → decode.py:1095 pop_preallocated → mooncake/conn.py:1923 send_metadata → zmq send` |

Seven ranks waiting in the DP-attention MLP-sync collective; one rank never
arrives because it is blocked in a mooncake ZMQ `send_metadata`. That is a
ragged collective — the classic PD+DPA+MTP deadlock *shape* — but the trigger
here sits in the mooncake metadata path, **not** in the draft-graph decision the
branch's patch 3 fixes.

While hung: `/health` times out, all 8 schedulers spin at ~105 % CPU, GPUs hold
88 % memory and do no work. A hang, not a crash.

## Three attempts

| # | rate | outcome |
|---|---:|---|
| 1 | 0.15 | **healthy 0–556 s** (401 completed, in-flight 8–29, 0 ticks at cap), then froze |
| 2 | 0.115 | pinned from t=32 s — **the decode leg was already dead** from attempt 1; this run is void |
| 3 | 0.15 | after rebooting decode: **0 completions from the start**; prefill now `DEAD`, decode healthy |

Attempt 3 exposed a second trap of my own making: rebooting **one** leg while
the other keeps running orphans the survivor's c10d state. Both legs must be
restarted together.

## What this is NOT

Ruled out by measurement, not by argument:

* **Not offered-load saturation.** Attempt 1 sat at in-flight 8–29 against a cap
  of 48, with **zero** ticks at the cap, and completions were still growing at
  t=556 s. A saturated server completes requests slowly; it does not stop
  completing them while the engines idle at 3 % KV usage.
* **Not the client.** All 130 errors are the client's hardcoded 240 s timeout,
  and an independent `bench_serving` run against the same router — a completely
  different client — stalled identically.
* **Not the router.** It kept returning `200 OK` and its CPU was 7.9 %.
* **Not OOM, not a GPU fault, not an exception.** All zero on both legs.
* **Not the kvd/ROCm hicache bug.** `Memory access fault` is 0 throughout.

## The comparison that makes MTP the leading suspect

The operator's question, and it is the sharpest evidence here:

> the previous experiment, same config but **MTP off**, passed — and its
> in-flight never reached max either.

`agenticbench.glm52.spur.packup_20260731` ran the **same cluster, model,
`--context-length 262144`, kvaware + kvd, and the same Case A profile** for a
full 67 minutes / 2,919 requests with **0 GPU faults and 0 scheduler
exceptions**. The only deployment difference here is **MTP on the decode leg**.

That is an observation, not a controlled result — the two runs also used
different images. The MTP-off control arm that would settle it is specified in
`REPRODUCE.md` §6 and was **not** run before this kit was written.

### Why the existing "MTP is fixed" kits do not refute this

Four kits validate the DSA/DPA/MTP patch set
(`glm52.mxfp4.spur.mooncake.packup_20260730_exp{1,2,3,3b}`). All four PASS — and
all four ran **~96 requests, 512-token outputs, conc ≤ 64, ctx 32768**, with
their own notes recording `conc=128 was not run`. This run is 401+ requests,
74K–235K-token inputs, multi-turn sessions, and it survived 556 s before dying.
**The regime is untested by those kits**, so their green results neither
contradict nor explain this hang.

## A finding about `--disable-custom-all-reduce`

Checked because the operator asked whether it is still needed post-patch.
Across all four kits, **8 of 8 legs ran `disable_custom_all_reduce=True`** — every
"MTP is fixed" PASS depends on it being on, and `main_converged` states outright
that the custom all-reduce path `remains unexercised`. It guards an *aiter kernel*
defect on gfx942/gfx950 (sglang #28815 / #31071 / PR #31478), which is a different
defect class from the rank-divergence bugs the patches fix.

**A real bug was found and fixed in this kit's own scripts because of that
question**: `CUSTOM_AR` defaulted to *follow* MTP, so the MTP-off arm would have
silently re-enabled the known-broken kernel — turning the planned A/B into a
two-variable comparison. `scripts/glm52_leg_spur_mtp.sh` now passes
`--disable-custom-all-reduce` on **both** arms. The converged kit had removed the
same trap from its own leg script; I had reintroduced it.

## An untried mitigation, from a sanctioned kit

`exp2_indexshare_off` reproduced liying's configuration — PD + DPA + MTP working
**without** patches 2 and 4, by disabling GLM-5.2's MTP IndexShare:

    --json-model-override-args '{"index_share_for_mtp_iteration":false}'

This deployment runs `index_share_for_mtp_iteration = True` (the model default,
confirmed: `json_model_override_args='{}'`). So the IndexShare-off route is an
untested third arm here — a candidate mitigation, not a diagnosis.

## Folder map

| path | what |
|---|---|
| `REPRODUCE.md` | how to reproduce the hang, and the MTP-off control arm that was not run |
| `notes.md` | the wrong turns — three of them were mine, and each cost a round |
| `results/hang_stacks/` | py-spy dumps, all 8 decode ranks, taken while hung |
| `results/*_metrics.jsonl.gz` | per-tick time series; the freeze is visible at t=556 s |
| `workloads/` | the probe and full Case A configs, with the load-solving rationale |
| `scripts/` | bring-up + leg + bench driver, verbatim |
| `logs/` | engine and driver logs, gzipped |

## Related

- `agenticbench.mtp.sweep.packup_20260801/` — **the successful half**: the same
  deployment passing an 8-point `bench_serving` sweep with zero faults, plus all
  five feature proofs. Its bring-up is this kit's prerequisite.
- `../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/` — the
  MTP-off Case A that completed.
- `../infera.glm5.2.experiment/glm52.mxfp4.spur.mooncake.packup_20260730_exp2_indexshare_off/`
  — the IndexShare-off route.
