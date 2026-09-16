# Routing landscape: Infera vs SGLang vs Dynamo

Status: research note (2026-09-16). Not user documentation — this is a
comparison of where request routing happens in the three stacks, what level
each component operates at, and what Infera does not have yet.

Evidence rule used throughout: every claim is tied to a file path (and line
where it is load-bearing). Infera and SGLang claims were read first-hand in
this tree and in `../sglang`; Dynamo claims were read from `ai-dynamo/dynamo`
`main` via the GitHub API plus the official docs, and are marked where they
are documentation-only or version-sensitive.

---

## 1. The layer map

| Stack | Component | Level | Decides | Load signal |
|---|---|---|---|---|
| **Infera** | `infera/gaie/endpoint_picker.py` (GAIE EPP / ext_proc) | k8s Gateway | endpoint; under PD returns decode as the primary endpoint plus `x-prefill-instance-id` | same policy as below |
| | `infera/router/auto.py` `AutoRouter` | fleet | PD vs mixed topology | pool contents |
| | `infera/router/disagg.py` `DisaggRouter` | **(worker, dp_rank)** | the prefill target and the decode target | policy |
| | `infera/router/mixed.py` `MixedRouter` | (worker, dp_rank) | single worker, plus in-generation migration | policy |
| | `infera/router/policy/kv_event_aware.py` | (worker, dp_rank) | the score | **router-local**: distinct in-flight blocks + decayed recent dispatches |
| | `infera/router/dp_routing.py` | **DP rank** | `X-Data-Parallel-Rank`, `bootstrap_room` residue, `disagg_prefill_dp_rank` | — |
| | `infera/router/breaker.py` | worker | health exclusion, applied *before* every pick | HTTP status of each leg |
| | `infera/planner/` `SlaPlanner` | cluster (off the request path) | **a recommendation only** | `/metrics` window deltas + `profile.json` |
| **SGLang** | `sgl-model-gateway/` (Rust) | worker; with `--dp-aware` the worker URL is `{base}@{dp_rank}`, so **DP rank** | one worker, or a PD pair with separate policies | polls each worker's `/v1/loads?include=core` |
| | `managers/data_parallel_controller.py` | **DP rank inside one engine** | which DP scheduler runs the request | shm snapshot: running+waiting reqs, total tokens |
| | `managers/schedule_policy.py` (lpm / fcfs / dfs-weight / priority) | **inside a batch** | which requests run this step | local queue — scheduling, not routing |
| **Dynamo** | Frontend + KV Router | **(worker_id, dp_rank)** | decode / general-pool target | global radix index + active blocks + busy thresholds |
| | internal prefill router | same | prefill target, with active-block tracking off | overlap-dominated |
| | KvIndexer (ZMQ or NATS Core) | block, tiered | the global prefix index across device/host/disk/shared | worker-published KV events |
| | SLA planner (and a load-based planner) | cluster | **actually scales** P/D replicas | ARIMA / Prophet / Kalman / constant + NPZ interpolation |
| | Migration operator | request | continue a generation elsewhere | `--migration-limit` |

**Where the three overlap.** Infera's `KvEventAwarePolicy`, SGLang's
`CacheAwarePolicy` and Dynamo's KV Router are three independent
implementations of the same thing: cluster-level selection by prefix-cache
locality traded against load. Nothing is shared between them.

**Where they do not.** SGLang's in-batch scheduler is orthogonal to every
router. And Infera's `dp_routing.py` is not a duplicate of SGLang's
`DataParallelController` — it decides the rank *outside* and then overrides
the engine's own choice on the wire.

---

## 2. PD disaggregation: who picks P, who picks D

| | Infera | SGLang | Dynamo |
|---|---|---|---|
| Prefill | `DisaggRouter.dispatch` → `policy.pick(role_hint="prefill")` (`infera/router/disagg.py:138`) | `select_pd_pair` → `prefill_policy` (`sgl-model-gateway/src/routers/http/pd_router.rs:1013-1021`) | internal prefill router; `router_track_active_blocks = false` (`prefill_router/activation.rs:348`) |
| Decode | same policy, `role_hint="decode"` (`infera/router/disagg.py:139`); pools and circuit breaker are filtered per role (`:121-127`) | `decode_policy` (`pd_router.rs:1031`); config forbids `bucket` for decode (`config/validation.rs:598-600`) | main KV Router, separate decode branch of the cost function |
| Handshake | `align_room_to_prefill_rank` + `disagg_prefill_dp_rank` + a per-connector protocol (mooncake / moriio / sglang bootstrap) | injects `bootstrap_host` / `bootstrap_port` / `bootstrap_room` (`pd_router.rs:242`) | NIXL metadata returned by prefill |

All three converge on the same asymmetry:

- **Prefill wants cache locality.** A prefix hit removes the entire prefill
  pass. Infera's documented production value is
  `--kv-prefill-overlap-weight 20.0`; SGLang typically runs
  `prefill_policy=cache_aware`; Dynamo's prefill router does not even track
  decode load.
- **Decode wants capacity.** Decode is memory-bound on KV residency and
  concurrent sequences, and a prefill-time cache hit buys it little. Infera
  uses `--kv-decode-overlap-weight 2.0`; SGLang typically runs
  `decode_policy=power_of_two`; Dynamo's decode routers normally force
  `overlap_score_credit = 0` to keep decode routing load-only, relaxing that
  only under conditional disaggregation.
- **Prefill carries a hard protocol constraint decode does not.** SGLang's
  default `follow_bootstrap_room` balancer derives the prefill rank from
  `bootstrap_room % dp_size` and calls `record_failure` when the request
  landed on a different rank
  (`python/sglang/srt/disaggregation/common/conn.py:1438-1450`, unless
  `SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK=1`; the check is gated on
  the request not already carrying an explicit `disagg_prefill_dp_rank`,
  `:1434`). Infera's `align_room_to_prefill_rank` exists precisely to satisfy
  it without an engine-side env var.

---

## 3. Infera's routing granularity

**Both legs are DP-rank level, not worker level.**

`RouteTarget` is a `(worker, dp_rank)` pair (`infera/router/policy/target.py:30-54`).
`expand_targets` fans a "rank-multiplexed" worker — one endpoint fronting
`dp_size` internal ranks, i.e. SGLang native `--dp-size` — into one candidate
per rank, with `route_key = worker_id#dpN` so every rank is scored and
book-kept separately. `KvEventAwarePolicy.pick` calls it on its first line
(`infera/router/policy/kv_event_aware.py:250`); the Rust router does the same
(`rust/router/src/policy.rs:491`).

Three mechanisms carry that choice onto the wire (`infera/router/dp_routing.py`):

1. `X-Data-Parallel-Rank` (`:22`) — honoured by SGLang's
   `DataParallelController` (as `routed_dp_rank`) and vLLM's
   `_get_data_parallel_rank`.
2. `align_room_to_prefill_rank` (`:44-57`) — rewrites the low residue so
   `room % dp_size == dp_rank`, keeping the high bits random.
3. `inject_disagg_prefill_dp_rank` (`:34-41`) — tells the decode worker which
   prefill rank holds its KV (consumed at
   `python/sglang/srt/disaggregation/decode.py:703-704`).

The decode leg gets its own rank header too: `_leg_headers`
(`infera/router/disagg.py:57-63`) is applied to both legs.

There is **no TP-rank routing** anywhere; every `tp_rank` reference in the
tree is engine-side KV-transfer dedup, never a routing pick.

**Decode routing exists** — not as a separate process, but as a second
`policy.pick` over the `DisaggMode.DECODE` pool inside the same dispatch, with
its own circuit-breaker filtering ("Independently per role",
`infera/router/disagg.py:121-127`).

---

## 4. What Infera implements today

The policy registry has exactly two entries
(`infera/router/policy/factory.py:87-90`; the Rust router validates the same
two at `rust/router/src/config.rs:160`):

1. **`round-robin`** — stateless spread.
2. **`kv-aware`** (default) — `infera/router/policy/kv_event_aware.py:327-344`:

```
cost(t) = -w_overlap · cache_hits(t)
        +  w_mm      · mm_miss(t)
        +  active_blocks(t) + recent_blocks(t)

tie-break: lower load
w_overlap  = base_weight(role) × retention_amplifier(cache_control)
multimodal requests: w_overlap = 0
```

Hits are credited rather than misses charged, because per-worker render
variants make block-list lengths differ and charging misses would penalise a
worker merely for rendering a longer preamble.

Supporting machinery that is part of the routing behaviour, not decoration:

- per-role weights `--kv-overlap-weight` / `--kv-prefill-overlap-weight` /
  `--kv-decode-overlap-weight` (`infera/server/args.py:146-176`), each falling
  back to the global one;
- multimodal image affinity — an XXH3 key over the image *reference* and a
  bounded per-worker LRU; the text-overlap term is dropped entirely, because
  the image placeholder token id is identical across images and trusting text
  overlap would serve one image's KV for another. Note the affinity term is
  **not** prefill-exclusive: it runs on the decode pick too;
- `cache_control` retention → `infera/router/engine_priority.py`, which
  becomes SGLang `priority` (radix eviction order) or vLLM
  `kv_transfer_params`;
- the circuit breaker, applied *before* every pick, scored per worker per leg
  (`_score_leg`);
- `--request-max-retries` (alternate worker before the first token) and
  in-generation migration (`infera/router/migration.py`, wired only into
  `MixedRouter`);
- `--router-mode direct`, which hands selection to an upstream GAIE EPP via
  `x-worker-instance-id` / `x-prefill-instance-id`;
- **render parity** (`infera/router/kv_event/render_probe.py`) — a correctness
  mechanism, not telemetry. On registration the router renders probe bodies
  both ways and compares token ids against the worker's own `/v1/tokenize`. A
  diverged worker keeps serving but silently routes on load alone, which is
  the exact failure this catches;
- **KV self-heal** (`rust/router/src/kv_selfheal.rs`) — when a worker's
  kv-event chain never anchored, the router asks it to flush its prefix cache
  so it re-emits `AllBlocksCleared`. The router is not purely observational;
- NATS-transport admission: either leg at its in-NATS backlog limit returns
  429 (`infera/router/disagg.py:331-344`), i.e. load shedding after a target
  has already been picked.

Not implemented: power-of-two, consistent hashing, prefix-hash, bucket,
manual/header routing, temperature sampling. SGLang's gateway registers all
eight of its policies at `sgl-model-gateway/src/policies/factory.rs:79-88`.

One maintenance hazard: Python and Rust are **two independent
implementations** of the same policy, with constants duplicated
(`RECENT_DECAY = 0.97`, `MM_IMAGE_BLOCK_WEIGHT = 48.0`, retention 2.0 / 0.1).
Any behavioural claim has to be checked against both, and any change made in
both.

---

## 5. What is actually hard in this area

1. **Keeping the router's render identical to the engine's.** The router
   tokenises and hashes on its own; the engine merges server-side template
   defaults the router is never told about. When they diverge nothing errors —
   every lookup misses, the policy degrades to load balancing, and readiness,
   kv-event flow and cache-view size all stay green. This has been found here
   days late, which is why the render probe exists.
2. **KV-event chain anchoring.** Events are chained on `parent_block_hash` and
   only the radix root reports `None`. On ZMQ the router subscribes after the
   worker has already warmed, so the anchor was broadcast to nobody and every
   later event is dropped. Infera asks the worker to flush; Dynamo rebuilds by
   querying worker-local indexers on restart.
3. **The cache-locality feedback loop.** The cache-hot worker keeps winning
   until it saturates. Dynamo has two brakes: `router_temperature > 0` switches
   selection to softmax sampling over cost logits, and
   `overlap_score_credit_decay` shrinks the overlap credit as that worker's
   normalised prefill load rises. SGLang switches to shortest-queue when
   `balance_abs_threshold` / `balance_rel_threshold` say the fleet is
   imbalanced. **Infera has neither.**
4. **Where the load number comes from.** Router-local accounting (Infera) is
   free but blind to the engine's real queue and KV residency; polling the
   engine (SGLang `/v1/loads`, Dynamo busy thresholds and active blocks) is
   accurate but lagged.
5. **Multi-replica routers.** A shared broker can converge the *prefix index*;
   the *active / in-flight* state has to be synced explicitly. Dynamo does
   this with three event types — `AddRequest`, `MarkPrefillCompleted`, `Free`
   (`lib/kv-router/src/protocols.rs:1191-1202`). SGLang has the `smg_mesh`
   crate (`MeshSyncManager`, remote inserts into the cache-aware tree at
   `policies/cache_aware.rs:284`).
6. **PD's two-leg semantics.** Two workers with independent health, so each
   leg must be scored from its own response. A registered `bootstrap_room`
   whose prefill then dies leaves decode blocked on `KVPoll` until its
   transfer timeout.
7. **Rank-level protocol alignment.** Room residue, header and
   `disagg_prefill_dp_rank` all have to agree with the engine's
   `load_balance_method`, or the KV transfer fails outright.
8. **Multi-tier cache in the cost function.** A block may sit in GPU, host,
   disk or shared storage, each worth a different amount. Dynamo already
   weights them (`host_cache_hit_weight` 0.75, `disk_cache_hit_weight` 0.25,
   a shared multiplier).
9. Plus the usual: priority and multi-tenant QoS, admission and rate limiting,
   draining on scale-down, and observability good enough to explain a pick.

---

## 6. Gaps against Dynamo

Already aligned: a global prefix index (with JetStream ordered replay and a KV
bucket snapshot for cold start, so the index survives a router restart and
converges across replicas — `rust/router/src/kv_event_nats.rs:13-27`,
`infera/router/kv_event/nats_client.py:167-176`; this holds under the default
NATS transport, not under ZMQ), a role-separated cost function, **DP-rank
routing**, circuit breaking, request migration for mixed, an SLA planner
skeleton, a Rust data plane, k8s discovery, NATS request transport with
admission control.

Infera-only, relative to Dynamo: multimodal image affinity, the render-parity
probe, `cache_control` plumbed through to engine eviction priority, and the
GAIE EPP direct mode.

Missing, in rough priority order:

1. **The planner recommends but does not act, and does not forecast.**
   `SlaPlanner.plan` returns a `ScalingDecision`, `on_decision` defaults to
   `None`, and there is no Kubernetes call anywhere under `infera/planner/`.
   It is window-delta measurement plus profile interpolation — no ARIMA /
   Prophet / Kalman. Dynamo's planner calls `update_graph_replicas()` on the
   DynamoGraphDeployment and carries four predictors plus online correction,
   and ships a second reactive load-based planner.
2. **No engine-reported load reaches routing at all.**
   `infera/common/engine_metrics.py:inflight_from_metrics` is imported only by
   `infera/engine/drain.py`; the Rust router polls no worker load endpoint and
   says so (`rust/router/src/policy.rs:229`). The decode pick therefore cannot
   see the target's KV utilisation or waiting queue — exactly the signal
   decode routing should be using. Dynamo exposes runtime-tunable
   `active_decode_blocks_threshold` / `active_prefill_tokens_threshold` over a
   `/busy_threshold` endpoint (529 on exceed).
3. **Nothing breaks the cache feedback loop** — no temperature, no credit
   decay, no imbalance threshold.
4. **Active / in-flight state is not shared across router replicas.**
   `_active_block_refs`, `_recent_blocks` and `_mm_affinity` are per-process.
   The prefix index is fine (see above); this is the part that is not. Both
   Dynamo and SGLang have an explicit mechanism.
5. **No cost or latency model.** "Load" is a block count with no relation to
   TTFT or ITL. Dynamo's `--router-prefill-load-model aic` predicts prefill
   duration from effective ISL and cached prefix length.
6. **No request migration under PD** — `migration_limit` reaches only
   `MixedRouter` (`infera/router/auto.py:44-62`), because the second leg's
   state would have to move too.
7. **The kvd tier is invisible to routing.** Blocks retrievable from kvd
   should be worth something in the cost function; Dynamo weights its tiers.
8. **No priority / multi-tenant QoS.** Dynamo has WSPT class queues with
   per-class busy thresholds and priority retry; SGLang's gateway has
   `core/job_queue.rs` and `core/token_bucket.rs`. Infera has one global 429.
9. **No conditional disaggregation.** `AutoRouter` decides PD vs mixed purely
   from pool membership; its own comment says a per-request decision belongs in
   the policy layer. Dynamo decides per request from effective ISL and busy
   thresholds.
10. **A narrow policy shelf** — two policies, with no session affinity, no
    power-of-two, no header-directed routing.

In one line: Infera's individual decision is already competitive — finer
granularity than worker level, multimodal affinity, render verification, a
durable prefix index — and the gap is the **closed loop**: real load flowing
back into the decision, scaling that executes, in-flight state shared across
replicas, and something that breaks the cache feedback cycle.

---

## Appendix: Dynamo facts that are version-sensitive

Read on `main`, 2026-09. Older releases behave differently, and several public
docs still describe the old behaviour.

- **Cost function** (`lib/kv-router/src/scheduling/selector/default.rs:341-343`)
  is three-term with overlap as a *credit that is subtracted and clamped at
  zero*, not the older two-term additive logit:

  ```
  logit = prefill_load_scale · max(0, raw_prefill_blocks − overlap_credit_blocks)
        + decode_cost_blocks
        + decode_active_request_weight · active_requests
  ```

  `overlap_credit_blocks` sums device, host, disk and shared-tier overlap with
  separate weights. `decode_active_request_weight` defaults to 0.
  `kv_overlap_score_weight` / `overlap_score_weight` was renamed
  `prefill_load_scale`. The decode-router branch (`:317-339`) is
  `max(0, decode_cost_blocks − overlap_credit_blocks) + active_request_cost_blocks`.
- **DP-rank routing exists.** The router keys overlap, load and selection by
  `WorkerWithDpRank` and returns a specific rank
  (`lib/kv-router/src/scheduling/overlap.rs:9,31-46`); vLLM publishes the DP
  topology. This is the direct equivalent of SGLang `--dp-aware` and Infera's
  `X-Data-Parallel-Rank`.
- **JetStream persistence is gone.** `router_snapshot_threshold` and
  `router_reset_states` are `_legacy_` compat fields; the current router
  rebuilds prefix state by querying workers' local indexers on restart. The
  event plane is ZMQ (examples' default) or NATS Core, not JetStream.
- **The automatic prefill-router path covers SGLang too** on `main`; the
  "SGLang needs a standalone prefill router" caveat is older-release
  behaviour.
- `--no-kv-events` predictive mode retains router-written blocks with a TTL,
  default 120s.
- Not found: any speculative-decoding awareness in the router.
