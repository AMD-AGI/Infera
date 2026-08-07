# Notes — GLM-5.2 MIX Task 3 agentic stress (Case-A, closed-loop)

## The method (what / why)

A **closed-loop** agentic stress run using the INFERA driver
`agent.agent_throughput` (`--mode realistic`, forced by the yaml's `profile:`
block). It runs on the **node host** (via the staged venv
`/shared_nfs/yihou_agentbench/venv`) against the router `:8100`. This is
deliberately **not** the customer/AgentX bench — mission rule 5 forbids it.

- **Workload:** `scripts/stress_caseA.yaml` — a copy of `glm52_crxx_caseA.fix.yaml`
  with the request-shape block unchanged (Case-A ISL/OSL percentiles 74K/155K/235K
  & 320/3300/17000, turns/session 3/20/103, inter-turn delay 4/31/240 s,
  `cache_hit_rate 0.89`) and the **load** block overridden to the operator's
  stress knobs.
- **Load knobs (mission + re-solve):** `initial_sessions: 8`, `max_inflight: 16`,
  `max_sessions: 24` (all mission-specified), `new_session_rate: 0.20` (re-solved,
  below).
- **Window:** `ramp_duration: 400` (a warmup *exclusion* window, not a load ramp)
  + `sustain_duration: 3600` (the honest number; a p99 session is 103 turns ×
  ~33 s ≈ 3400 s, so anything shorter truncates the calibrated tail).

### Closed-loop semantics — offered load is set by live sessions, not QPS
In `--mode realistic` each session issues **one request at a time and waits** for
the response before its inter-turn delay. So the offered load is a **consequence**
of the live-session population, never a QPS input. `initial_qps` / `max_qps` do
nothing here and are omitted (the driver warns if present). By Little's law the
steady-state population is:

```
N = new_session_rate × E[turns] × (E2E + E[inter_turn_delay])
```

`E2E` is a **measurement**, not a constant — which is exactly why the birth rate
had to be re-solved after seeing the achieved E2E (next section).

## The `new_session_rate` re-solve — run 1 was a calibration

**Two runs were made. Only run 2 is the reported result.**

- **Run 1 (calibration, `new_session_rate: 0.05`):** under-loaded. The steady
  live-session population settled at only **~4**, far below the caps
  (`max_inflight 16`, `max_sessions 24`). At N~4 the in-flight cap never binds, so
  the server is nowhere near saturation and the numbers describe an idle box, not
  a stress point. (Archived by the operator under
  `results/stress_run1_calib`, per the summary note; not carried into this packup.)
- **Re-solve (CASE_AB_GUIDE Step-3 method):** target a steady live-N ≈ 16 (= the
  in-flight cap, so backpressure actually binds), keeping `initial_sessions: 8`
  fixed per the mission. Solving `N = rate × E[turns] × (E2E + delay)` with the
  E2E **measured in run 1** yields `new_session_rate ≈ 0.20`.
- **Run 2 (reported, `new_session_rate: 0.20`):** live sessions climb to and hold
  at the **session cap (~23–24)**, and in-flight holds at the **in-flight cap
  (~15–16)**. The cap binding is the signal that this is a genuine saturation
  point — not an arbitrary QPS pick.

`metadata.json` records the resolved knobs that actually took
(`new_session_rate: 0.2`, `max_inflight: 16`, `max_sessions: 24`,
`num_initial_sessions: 8`, `sustain_duration_secs: 3600`).

## Results — how to read them (the sustain phase)

Read the **`sustain`** entry in `summary.json['phases']` — the ramp is warmup
(excluded) and the drain is the ~2 s tail after births stop.

| metric | sustain value |
|---|---|
| requests completed | 2092 (over 3600 s) |
| offered QPS | 0.581 req/s |
| success rate (whole run) | 98.2% (31 err / 2364 sent) |
| cache-hit rate | 0.889 (ideal 0.890) — Case-A 88–90% ✅ |
| TTFT p50 / p90 / p99 | 1460 / 3610 / 6196 ms |
| TPOT p50 / p90 / p99 | 19.0 / 32.6 / 74.8 ms |
| input TPM | 2,988,618 |
| uncached TPM / GPU | 41,330 (330,640 / 8) — the real prefill work |
| gen (visible) TPM / GPU | ~3,381 (27,047 / 8) |

- **Cache-hit 89% is honored** — the workload's shared-prefix model gives Case-A's
  target prefix reuse; `efficiency 0.9997`, eviction ~0 (one hot radix path).
- **In-flight cap binds ⇒ genuine saturation.** Per-window `in_flight` in
  `metrics.jsonl` sits at ~15–16 and `num_sessions_active` at ~23–24 through the
  sustain window — both at their caps. That is the whole point of the re-solve:
  the server is admission-limited, so these latencies are the real
  under-load numbers, not idle-box numbers.
- **Agreement with the reference kit:** TPOT p50 19 ms (ref ~14–16 ms) and
  cache-hit 89% (ref ~88%) line up — the stress result is consistent with prior
  characterizations of this stack.
- **TTFT (1.46 s p50) is much higher than Task 2's conc=1 (~0.4 s)** because here
  16 requests contend for prefill under an 89%-but-not-100% cache, whereas Task 2
  measured a single request against a fully warmed prefix.

## Gotchas / caveats (what / why / how / context)

### Driver runs on the HOST, not in the container (unlike Task 2)
**What:** `agent.agent_throughput` is launched from
`/shared_nfs/yihou_agentbench/venv/bin/python` on the node host, not via
`docker exec`. **Why:** the engine image has **no `agent` module**; the agentic
driver and its deps live only in the staged venv. **How:** point `--server` at the
router on the node's data-plane IP (`http://10.245.148.191:8100`), not
`127.0.0.1` — the host reaches the router over the node IP. **Context:** this is
the one structural difference from the Task 2 latency packup, whose driver ran
inside the container against `127.0.0.1:8100`.

### A closed-loop run cannot be "set to a QPS" — you tune the birth rate
**What:** there is no QPS knob that works in `--mode realistic`. **Why:** load is
emergent from the live-session count (closed loop). **How:** to hit a target
in-flight, solve `new_session_rate` from Little's law using the **measured** E2E
(that's what run 1 was for). **Context:** `runner.py --auto-search` bisects on QPS
and has **no effect** in this mode; sweeping capacity means sweeping
`new_session_rate` (and session length).

### `max_sessions` hit is a saturation signal, by design
The yaml comments treat hitting `max_sessions` as "sessions dying slower than
born ⇒ saturated ⇒ a failure signal." **Here it is intentional** — the stress run
*wants* the caps to bind so it measures the saturation point. Read it as "we
reached the operating limit," not "the run broke." Success (98.2%) confirms the
server held that limit without collapsing.

### The tokenizer path in the yaml is a REPLACE placeholder
`stress_caseA.yaml` carries `tokenizer: /shared_nfs/GLM-5.2-MXFP4`. `run_stress.sh`
also passes `--tokenizer /shared_nfs/GLM-5.2-MXFP4` on the CLI. On a different
machine both must point at the real weights dir or the run aborts.

### DSA env is mandatory on gfx950 or the model serves garbage
Without the DSA-ROCm env (`SGLANG_OPT_USE_TILELANG_INDEXER=1`, `TOPK_V2=0`,
`JIT_NORM=0`, `USE_AITER=1`, `ROCM_FUSED_DECODE_MLA=0`) the worker emits garbage.
All set in `scripts/mix_worker.sh`; the smoke's "coherent answer" block is the
live check.

### MTP requires `--disable-custom-all-reduce` on gfx950
aiter custom all-reduce **deadlocks** in EAGLE verify on gfx950.
`mix_worker.sh` auto-adds `--disable-custom-all-reduce` whenever MTP is on.
Accept-len **median ~3 is healthy; 4.00 means degeneration.**

### DSA indexer P1V3 reversed-IDLE-rank fix (baked in the image)
Under DP-attention on an **IDLE** rank during MTP draft-extend, the `GLM52_P1V2`
guard's inequality inverts (fewer query rows than lengths entries) →
`RuntimeError: Expected lengths.size(0) == B`. Fix: reconcile both sides to
`_p1v2_rows = min(real, padded)`. Applied at build by
`deploy/docker/patches/sglang_dsa/patch_dsa_indexer_hip_dp_padded_rows.py`;
**`_p1v2_rows` verified in the bytecode** (identifier marker — a stale
`__pycache__` has voided a run on this stack twice).

### ROCm hicache host-alloc fix (baked in the image)
gfx950 hard-aborts with `Memory access fault by GPU node-N on address <host VA>`
when hicache stores raw host `data_ptr()`s that a GPU kernel dereferences
(`hipHostRegister` maps host pages at a *different* device VA; gfx950 is `xnack-`,
no page-migration fallback). Fix: route `ALLOC_MEMORY_FUNCS` to `pin_memory`
(`hipHostMalloc`) — `deploy/docker/patches/sglang_rocm/patch_hicache_rocm_host_alloc.py`.

### Cluster / infra gotchas
- **`/tmp` is root-owned on crsuse** under `spur exec` (runs as `yihou`). Use
  `/var/tmp` — `DOCKER_CONFIG=/var/tmp/dockercfg_yihou`.
- **etcd v3.5.14 needs `--entrypoint /usr/local/bin/etcd`** (empty ENTRYPOINT).
- **Never background a long docker client inside `spur exec`** — teardown kills it.
  The stress driver runs on the host (not `docker exec`), so this bites only the
  bring-up steps.
- **Server logs contain binary bytes** — grep through `strings`.

## Feature-proof evidence (from the smoke, before measuring)

Same frozen server family as Tasks 1 & 2: 1 worker `disagg_mode=mixed`; coherent
answer (DSA live); MTP accept-len median ~3; ~8 kvd adapters + entries>0; router
`kv-aware`, tokenizer loaded.

## Known gaps in this packup

- Base-image `sha256` digest not captured at build (floating tag only) — see
  `environment.md` for the one-liner to resolve it on the node.
- DSA/ROCm patches are referenced by in-tree path at the pinned SHA, not copied
  in, since the image bakes them from the repo at `d1a97b2`.
- Run 1 (the `new_session_rate: 0.05` calibration) is documented but its raw
  output is not carried here — it was an under-loaded calibration, not a result;
  the operator archived it under `results/stress_run1_calib` on the node.
- `metrics.jsonl` is 4.1 MB (per-window samples over the whole run); included at
  the operator's request as the raw evidence behind the sustain table.
