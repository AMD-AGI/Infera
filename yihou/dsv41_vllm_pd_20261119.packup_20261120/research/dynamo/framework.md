# Dynamo framework features in the NVIDIA DSv4.1-Flash AgentX PD runs, and their MI355X equivalents

Date: 2026-11-19. Research only (no GPU, no docker). Dimension: the **framework layer** (router, disagg flow,
KV tiers, planner, transport, frontend, session routing) of the NVIDIA Dynamo + vLLM PD submissions. Engine-side
levers (kernels, MegaMoE, EPLB, graph sizes) are only touched where they belong to the PD flow; see
`research/dynamo/vllm_side.md` (sibling file) for those.

Tags: **[FH]** first-hand (file:line, API row, PR read by me today); **[2H]** second-hand (PR/issue author claim,
doc claim not exercised); **[INF]** my inference, unverified. Suspend rule applies: no mechanism is asserted without a
code read, PR or measurement.

Sources and pins:
- `IXm` = InferenceX `origin/main` 6ed3774 (2026-11-19 01:49 PST), fetched into `source/InferenceX` (worktree still db3fc79).
- `RUN-GB300` = InferenceX API row 445572: GB300, dynamo-vllm, 12P DEP2 + 1D DEP16 = 40 GPUs, c9888, **1,011,514 tok/s/GPU,
  p90 interactivity 75.5**, image `vllm/vllm-openai:cu134-nightly-aarch64-dev-20261019`, Dynamo `1.6.0.dev20261010`, run
  https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39207089292/attempts/12, 2026-10-20 [FH: `notes/dynamo/reproduce_445572.json`;
  exact config as launched: `notes/dynamo/gb300v_445572_config.yaml` from `/api/v1/server-log?id=445572&file=config.yaml`].
- `RUN-B300` = API row 448173: B300, dynamo-vllm, 5P DEP2 + 1D DEP8 = 18 GPUs, c3584, **828,209 tok/s/GPU, p90i 83.8**, image
  `vllm/vllm-openai:cu134-nightly-96710245ece8...`, Dynamo `1.6.0.dev20261010`, 2026-11-17 [FH: `notes/dynamo/reproduce_448173.json`,
  `notes/dynamo/b300v_448173_config.yaml`].
- `RUN-GB300-SGL` = row 448095: GB300 dynamo-sglang 9P DEP2 + 1D DEP16, 34 GPUs, c7480, 1,071,944, p90i 105.6, Dynamo `1.7.0.dev20261110`
  [FH: API row; recipe `IXm:inferencex-e2e/benchmarks/multi_node/srt-slurm-recipes/dsv41flash/sglang/gb300-fp4/agentx/disagg-dep16-hicache-variants.yaml`].
- `DYN` = ai-dynamo/dynamo main **beb1a41 (2026-11-19)**, shallow clone `source/dynamo`. NOTE: this is not the 1.6.0.dev20261010
  wheel the vLLM runs used; behaviour of that wheel is inferred from the flags the run passed, code reads are of main.
  (The clone was later switched to sparse checkout by another process; I read files with `git show HEAD:<path>`.)
- `VR` = vllm-project/router main 3920aad (2026-11-19), clone `source/vllm-router`; tags read via GitHub API.
- `V` = vLLM main 7a62ca0666 (`source/vllm`). `INF` = this repo's infera router (`/home/yihou/dev/git.16-19/infera.dsv41`, last
  router commit c018271e5, 2026-10-26).
- MI355X bar for reference: 427,451 tok/s/GPU, vLLM colocated TP1xDP2 + UMBP, c384, p90i 15.8, router `vllm-router 0.1.14
  --policy consistent_hash --intra-node-data-parallel-size 2` [FH: API row 448242, `notes/ix/reproduce_448242.json`].

---

## 0. Bottom line

1. **The Dynamo features actually exercised in the winning vLLM runs are few**: Rust frontend (tokenize/template/parse once,
   fastokens + 8 GB prefix tokenizer cache), KV-aware router (`router-mode kv`, temperature 0, all busy/queue thresholds
   disabled) choosing **(worker, DP rank)** for prefill from engine KV events, session affinity TTL 14400 s keyed by the
   AIPerf session id, and (GB300 only) decode routing by **active request count** (`DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=1e7`).
   **Not used**: KVBM, planner/autoscaling, conditional disaggregation, router replica sync, multiple frontends
   [FH: both run config.yaml files contain no kvbm/planner/conditional keys; `enable_multiple_frontends: false`].
2. **KV tiers are vLLM's, not Dynamo's**: prefill = `MultiConnector[NixlConnector + SimpleCPUOffloadConnector(lazy)]` with
   80 GiB (GB300 10-20 run) / 70 GiB (B300) host DRAM per 2-GPU prefill worker; **decode runs with prefix caching OFF**
   [FH: gb300v_445572_config.yaml:97,173; b300v_448173_config.yaml:99,177]. `kv_offload_backend: vllm-simple` in the API config [FH].
3. **Every one of those framework functions has an AMD-reachable equivalent today**, at different maturity:
   vllm-router **0.1.17** (2026-11-04) added `kv_aware` (KV events, DP-rank granularity, host-tier credit, session affinity,
   PD over NIXL or Mooncake) [FH: VR release notes; `VR/src/main.rs:124-226,424`]; the infera router has a KV-event-aware
   DP-rank policy, session affinity, and vLLM-Mooncake / vLLM-MoRIIO PD protocols [FH: `INF/rust/router/src/{policy,dp,disagg}.rs`,
   `INF/infera/router/disagg_protocols/vllm_{mooncake,moriio}.py`]; Dynamo itself ran on MI355X in an AMD PoC (vLLM + RIXL
   2-node disagg, MiniMax-M2.5) [2H: andyluo7/dynamo `amd-mi355x-poc/README.md`, 2026-05], but has **no official ROCm image**
   (ROCm container PR ai-dynamo/dynamo#12537 closed stale 2026-11-11; a community fork carries it) [FH: gh].
4. **The MI355X bar's router is the weakest piece relative to Dynamo**: `consistent_hash` on vllm-router 0.1.14 hashes the
   `x-correlation-id` header (AIPerf's per-session id) to a DP rank - pure hard affinity, no load term, no KV-overlap
   signal [FH: vllm-router v0.1.14 `src/policies/hash_key.rs:10-17`; aiperf 0.13.0 `common/models/record_models.py:183`].
   Whether this costs throughput at c384 is **not measured**.
5. What made NVIDIA PD fast is not established by this file. The framework knobs above are the ones they set; the
   performance contribution of each is unpublished except the vLLM-blog CPU-path items (C6 in SYNTHESIS) [FH: no ablation found].

---

## 1. What the NVIDIA runs configured at the framework layer (as launched)

| Knob | RUN-GB300 (1.01M, 10-20) | RUN-B300 (828k, 11-17) | IXm GB300 recipe now (not yet re-published) | Source |
|---|---|---|---|---|
| Dynamo version | 1.6.0.dev20261010 wheel, request plane tcp, event plane zmq, etcd service, `sidecar: true` | same | same (SGLang recipe: 1.7.0.dev20261110) | [FH: config.yaml:13-27 / :14-27] |
| Frontend placement | dedicated node, numa_bind | `first_decode` node, numa_bind | dedicated; comment "Sharing prefill hosts exhausted RAM at c5120 with Engram and host KV offload" | [FH: gb300 cfg:47-48; b300 cfg:50-51; IXm gb300 yaml:43-45] |
| router-mode | kv | kv | kv | [FH: cfg:50 / :53] |
| Session affinity | TTL 14400 s (mode default) | TTL 14400 s | TTL 14400 s + `DYN_ROUTER_SESSION_AFFINITY_MODE=soft` | [FH: cfg:51 / :54; IXm yaml:48,55] |
| Busy thresholds | `active-decode-blocks-threshold`, `active-prefill-tokens-threshold(-frac)` = None | same | same, comment "Keep prefill off the lockstep decode ranks" | [FH: cfg:52-54 / :55-57] |
| Router queue | `DYN_ROUTER_QUEUE_THRESHOLD=None` | same | same | [FH: cfg:57 / :60] |
| Temperature | 0 (deterministic argmin) | 0 | 0 | [FH] |
| Decode cost | `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT=10000000` | not set (default 0) | 1e7 + `worker_selection.decode: dynamo-two-tier-cost-fn {cache_threshold: 1.0}` | [FH: gb300 cfg:63; IXm yaml:63-72] |
| KV events to router | prefill `kv_events: true`, decode none | same | prefill `kv_events: true` but frontend `no-kv-events: true` (approximate mode) | [FH: cfg:73 / :75; IXm yaml:53,81] |
| Active-request expiry | 3600 s | 3600 s | 3600 s | [FH] |
| Tokenizer | `DYN_TOKENIZER=fastokens`, `DYN_TOKENIZER_CACHE=1`, 8,000,000,000 B | same | same | [FH: cfg:59-61 / :62-64] |
| Parsing | `--dyn-tool-call-parser/--dyn-reasoning-parser deepseek_v41` on sidecar; workers `reasoning-parser: None` ("Parse once in the Dynamo frontend") | same | same | [FH: cfg:21-27,101] |
| Token transport | `VLLM_RS_TOKEN_IDS_AS_BYTES=1` | not set | set | [FH: gb300 cfg:95] |
| Client session header | `AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true`, `AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING=0` | same + `AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS=14400` | same | [FH: cfg:195-196 / :202-204] |
| P transport | NIXL UCX (`cuda_copy,cuda_ipc,tcp`, MNNVL), module `nixl.push_as_nixl` | NIXL push, LIBFABRIC/EFA | NIXL push | [FH: cfg:88-97 / :99] |
| P host tier | SimpleCPUOffloadConnector 85,899,345,920 B (80 GiB) lazy | 75,161,927,680 B (70 GiB) lazy | 64,424,509,440 B (60 GiB) | [FH: grep cpu_bytes_to_use] |
| D prefix cache | `no-enable-prefix-caching` | same | same | [FH: cfg:173 / :177] |
| D async KV admits | `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP=6` | 4 | 6 | [FH: cfg:150 / :153] |
| KVBM / planner / conditional disagg | absent | absent | absent | [FH: grep count 0] |

`VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` is not in vLLM main; main has the merged equivalent `max_async_kv_loads_per_step`
(#61725) [FH: grep V/vllm; `V/vllm/config/scheduler.py:90`]. `push_as_nixl` is not in vLLM main; main has
`NixlPushConnector` (push_scheduler.py/push_worker.py) [FH: `V/.../kv_connector/v1/nixl/connector.py:9-11`].

---

## 2. How each Dynamo feature works (code/doc read on DYN main) and what we have on AMD

### 2.1 KV-aware router
- Cost per candidate (worker, DP rank) [FH: `DYN/docs/.../router/router-design.md:55-95`; `offloading-support-matrix.md`
  quoting `lib/router/scheduler/src/selector/reference.rs`]:
  `adjusted_prefill_blocks = max(0, prefill_blocks - credit*device_overlap - 0.75*host_overlap - 0.25*disk_overlap)`;
  `logit = prefill_load_scale*adjusted_prefill_blocks + potential_decode_blocks + decode_active_request_weight*active_requests`;
  lowest logit wins (temperature 0). Prefill blocks include the worker's active prompt-side load.
- Cached blocks come from a global radix tree built from engine KV events (stored/removed, per DP rank, gap detection +
  replay from the worker's local indexer) [FH: router-design.md:115-235]. With `--no-router-kv-events` the router predicts cache
  state from its own decisions with TTL (default 120 s) or LRU [FH: `DYN/components/src/dynamo/common/configuration/groups/kv_router_args.py:478`, help text].
- In disagg, the frontend builds an internal prefill router with `track_active_blocks=false`; the decode stage disables overlap
  scoring, KV-reuse assumption, and prefill-token tracking [FH: `DYN/docs/.../router/disaggregated-serving.md:21,27-29`;
  `DYN/lib/llm/src/discovery/watcher.rs:666-690`]. So with `decode_active_request_weight=1e7` the GB300 decode pick is, in
  effect, least active requests per DP rank of the single DEP16 worker [INF from the formula].
- The DP rank is part of the routing result and is forwarded to vLLM as `data_parallel_rank=` [FH: `DYN/components/src/dynamo/vllm/handlers.py:4089,4179,4418`].
- Host tier credit applies only if the engine publishes CPU-medium events. vLLM `SimpleCPUOffloadConnector` emits CPU `BlockStored`
  on vLLM main [FH: `V/vllm/v1/simple_kv_offload/manager.py:1206-1225`]; whether the 2026-10-19 image did is not checked. Dynamo's
  own matrix lists vLLM CPU-tier routing as validated for `OffloadingConnector` (vLLM >= 0.24) [2H: doc].
- Dynamo indexes only "main attention" groups (full/MLA/sink) [FH: `DYN/lib/kv-events/src/zmq_wire/filter.rs:26-80`]; which kind
  vLLM reports for DSv4.1's compressed / SWA groups on ROCm is not checked (vLLM maps by manager class,
  `V/vllm/v1/core/single_type_kv_cache_manager.py:3412-3436`).

**AMD equivalents**
| Option | KV-overlap scoring | DP-rank granularity | Load term | Host-tier credit | PD | Status on MI355X |
|---|---|---|---|---|---|---|
| vllm-router **0.1.14** `consistent_hash` (our bar) | no | yes (`--intra-node-data-parallel-size`) | no | no | yes | in production [FH: reproduce_448242] |
| vllm-router **0.1.17** `kv_aware` | yes, from engine ZMQ KV events (`--kv-event-source`, `--pd-kv-events`; discovery needs `VLLM_SERVER_DEV_MODE=1`) | yes, `(worker, rank)` [FH: `VR/src/policies/kv_aware.rs:366`] | yes: cost = (new prefill tokens + active prefill tokens) x (active requests + 1) (multiplicative) [FH: kv_aware.rs:794-900] | `--kv-penalty-host 0.8`, external 0.6, disk 0.3 [FH: main.rs:220-226] | `--vllm-pd-disaggregation --kv-connector {nixl,mooncake}`, `--prefill-policy/--decode-policy kv_aware` [FH: main.rs:136-170,424] | released 2026-11-04; not run on MI355X by us; `kv_aware` absent in v0.1.14/v0.1.16 [FH: GitHub contents API per tag] |
| infera router (`--router-backend rust`) | yes, `cost(w) = -overlap_weight*hits(w) + load(w)` with decayed recent-miss load [FH: `INF/infera/router/policy/kv_event_aware.py:30-101`] | yes, per-rank ZMQ subscribers; vLLM rank via `X-Data-Parallel-Rank` [FH: `INF/rust/router/src/kv_event.rs:6-16`, `dp.rs:6-15`] | yes | not read | vLLM Mooncake (concurrent dual dispatch, Rust) and vLLM MoRIIO (Python only) [FH: `INF/rust/README.md`; `disagg.rs:6-16`] | used on other models (GLM-5.2, Kimi); DSv4.1 vLLM untested [INF] |
| Dynamo frontend on ROCm | as above | yes | yes | yes | NIXL (RIXL), Mooncake protocol classes exist [FH: `DYN/components/src/dynamo/vllm/kv_connector_protocols.py:25-115`] | PoC only: vLLM+RIXL disagg MiniMax-M2.5 730 tok/s @ c32 on 2 MI355X nodes; DSR1 vLLM+RIXL startup `ibv_reg_mr` failure unresolved [2H: andyluo7 PoC README, 2026-05]; no official ROCm image [FH: #12537 closed] |

### 2.2 Disaggregated flow
- Dynamo vLLM path: frontend -> prefill router -> prefill worker -> returns `kv_transfer_params` (NIXL) -> decode request built
  from the prefill result -> decode streams [FH: disaggregated-serving.md request-flow diagram; `DYN/lib/llm/src/kv_router/prefill_router/mod.rs:129-175`
  (`PrefillOutcome::Completed` for vLLM vs `Bootstrap` for SGLang)]. So for vLLM NIXL the decode leg is dispatched after prefill
  completes [FH-derived]; with push mode the D side registers its blocks and P WRITEs into them [FH: `V/.../nixl/push_scheduler.py:1-22`].
- No prefill queue component in these runs: the prefill router has busy thresholds disabled and router queueing off; queueing
  happens inside the vLLM prefill engines (max-num-seqs 64) [FH: §1 table]. Dynamo 1.7 adds backend admission limits, which the SGLang
  recipe raises to 10,000 because "Dynamo 1.7 backend admission rejects turns with 529 once a DP rank is full" [FH: IXm SGLang yaml].
- Conditional disagg (`--router-conditional-disagg`, decode may skip remote prefill when it has overlap) exists, experimental,
  default off, not used [FH: kv_router_args.py:663; run configs].
- **AMD equivalents**: vllm-router PD mode (two-stage, NIXL or Mooncake); infera router dual-dispatch (P and D posted concurrently for
  vLLM Mooncake) [FH: `INF/rust/router/src/disagg.rs:6-16`]. TTFT difference between sequential and concurrent dispatch: not measured.

### 2.3 KV tiers (KVBM / host / disk)
- KVBM (G1 GPU -> G2 host -> G3 disk -> G4 object) exists in Dynamo [FH: `DYN/docs/.../kvbm-configuration.mdx`]; **not used** in the
  DSv4.1 runs; they used vLLM's `SimpleCPUOffloadConnector` inside `MultiConnector` on prefill only [FH: §1].
- KVBM on ROCm: excluded in the ROCm container PR ("KVBM — the CUDA path links nccl ... neither is wired for ROCm") [FH: #12537 body].
- **AMD equivalent**: UMBPStoreConnector (MoRI host DRAM pool; the bar uses 449 GB per 2 GPUs) - keeps the AITER sparse-MLA fast
  path only when it is the sole connector or every MultiConnector child is UMBP [FH: SYNTHESIS C1; `V/.../multi_connector.py:274-280`].
  `SimpleCPUOffloadConnector` does not override `only_restores_canonical_kv` [FH: grep V], so copying NVIDIA's prefill
  `MultiConnector[NIXL + SimpleCPUOffload]` onto ROCm turns the fast path off (as any PD connector does).
  UMBP worker emits CPU-medium `BlockRemoved` events when `kv_events_config.enable_kv_cache_events` [FH: `V/.../umbp/worker.py:373-395`,
  `umbp/connector.py:183-187`]; matching `BlockStored` emission not located in this pass.

### 2.4 Planner / autoscaling
- Not used: P:D fixed per point by sweep (2P..16P DEP2/DEP4 x 1D DEP8/16/32) [FH: recipes/API]. Not a lever for a fixed-GPU benchmark [INF].

### 2.5 NIXL transport
- GB300: UCX over cuda_ipc/MNNVL; B300: LIBFABRIC over EFA; push mode; `kv_lease_duration 120`; `kv_load_failure_policy fail` [FH: §1].
- AMD: vLLM main ROCm Dockerfile builds a `nixl-rocm` wheel + UCX [FH: SYNTHESIS C3]; vLLM resolves `nixl_rocm` on ROCm since
  vllm#49251 [2H: #12537 comment]; Mooncake and MoRIIO are our ranked connectors (SYNTHESIS 4.3). DSv4.1 packed-row NIXL transfer
  vllm#61121 still OPEN [FH: gh 2026-11-19].

### 2.6 Frontend / tokenization
- Rust frontend tokenizes with `fastokens` (PCRE2 + rayon) and an L1 prefix tokenizer cache, applies the template, parses reasoning
  and tool calls once; workers receive token ids [FH: `DYN/.agents/skills/dynamo-frontend-benchmark/SKILL.md:79-80`; §1].
- GB300-SGL needed `DYN_SGL_GATEWAY_WORKERS=16`: "One gateway process saturates a CPU core serving 16 DP ranks" [FH: IXm SGLang yaml].
- **AMD equivalents**: `VLLM_USE_RUST_FRONTEND=1` is already in every MI355X CI vLLM run [FH: SYNTHESIS 3.1]; vllm-router 0.1.17
  adds a Rust gRPC frontend with L0/L1 encoding caches and gRPC P/D (needs vLLM >= 0.30.0 workers) [FH: VR v0.1.17 release notes].
  A community W7900 test reports Dynamo-frontend TTFT 65 ms vs 108 ms for plain vLLM (Qwen3-0.6B, c32), throughput within 0.4% [2H:
  #12537 comment, not MI355X]. Whether frontend CPU binds us at c384+ is not measured.

### 2.7 Multi-turn AgentX session routing
- AIPerf sends `X-Correlation-ID` (stable per conversation) by default and, when told, the same value as `X-Dynamo-Session-ID`
  [FH: `source/pypi_agentx/x/aiperf/common/models/record_models.py:183`, `common/environment.py:854-871`].
- Dynamo binds the session to (worker, DP rank) on first dispatch; `hard` keeps it, `soft` passes it to the selector as a preference
  [FH: `DYN/docs/.../use-cases/agents/session-affinity.md`]. Affinity TTL is wired into both the prefill router and the decode routing
  stage [FH: watcher.rs:666-690, 740-753]. With prefill DEP2 workers, the binding that matters for prefix hits is the prefill (worker, rank) [INF].
- vllm-router 0.1.17: "In P/D mode, affinity applies only to decode ... prefill is selected independently for each request" (prefill
  relies on KV-overlap scoring) [FH: `VR/docs/load_balancing/README.md:284`].
- Our bar: `consistent_hash` on `x-correlation-id` -> fixed DP rank per session, no load term [FH: §0.4].

---

## 3. Feature x (Dynamo / AMD equivalent / gap) table

| Feature | What Dynamo does in these runs | Our AMD equivalent | Gap / risk |
|---|---|---|---|
| KV-aware prefill routing | radix index from prefill KV events per DP rank; additive cost overlap vs load; temp 0 | vllm-router 0.1.17 `kv_aware` (multiplicative cost); infera `kv_event_aware`; bar uses 0.1.14 `consistent_hash` | not run on DSv4.1 ROCm; KV-event correctness for DSv4.1 hybrid groups on ROCm unchecked; benefit unmeasured |
| Session affinity | TTL 14400 s, (worker, rank) binding, hard (10-20) / soft (IXm now) | `consistent_hash` (hard, no expiry); 0.1.17 `--kv-aware-session-affinity-ttl-secs` (decode-only in PD); infera `session_affinity.rs` | P-side affinity semantics differ across routers |
| Decode DP-rank balancing | active-request weight 1e7 (GB300) -> least active requests | 0.1.17 decode = minimum occupancy; vLLM internal DP LB if no rank given | decode DEP width on MI355X not yet chosen; lockstep DP sync cost on ROCm unmeasured |
| Busy/queue thresholds | disabled; engine queues | vllm-router `--disable-retries`, request timeout 14400 (bar) | check our router never 503/429s at high c |
| Disagg flow | prefill first, then decode with kv_transfer_params (vLLM); push WRITE | vllm-router PD two-stage; infera concurrent dual dispatch | TTFT impact of sequential vs concurrent: unknown |
| Prefill host KV tier | vLLM SimpleCPUOffload 60-80 GiB/worker, lazy | UMBP (449 GB/2 GPUs in bar) | MultiConnector with a transfer connector drops AITER fast path on ROCm (SYNTHESIS C1) |
| Decode prefix cache | off | default on | NVIDIA keeps all prefix reuse on prefill; our SYNTHESIS idea "D pulls only uncached suffix" is the opposite choice - test both |
| Router host-tier credit | 0.75 weight, needs CPU-medium events | 0.1.17 `--kv-penalty-host 0.8` | UMBP `BlockStored` CPU events not located |
| KVBM | not used | UMBP / MoRI | n/a |
| Planner | not used | n/a | n/a |
| Conditional disagg | not used | none | idea only |
| NIXL | UCX/MNNVL or LIBFABRIC, push | nixl-rocm/RIXL in vLLM ROCm image; Mooncake; MoRIIO | DSv4.1 ROCm PD connector not proven (SYNTHESIS B1-B3) |
| Frontend | Rust, fastokens, 8 GB tokenizer cache, parse once | `VLLM_USE_RUST_FRONTEND=1`; 0.1.17 gRPC frontend | CPU headroom at high c unmeasured |
| Frontend placement | dedicated node / NUMA-bound (host RAM pressure) | router on the server node | host DRAM shared with UMBP 449 GB + Engram |
| Observability | `/metrics` `kv_cache_events_applied`, request trace with worker/cached_tokens per session | vllm-router Prometheus (`vllm_router_pd_affinity_decisions_total`), infera metrics | need per-rank hit + routing logs in E3/E8 |
| ROCm support of Dynamo itself | - | PoC (May), import fixes merged (#9929), container PR closed; community fork | building Dynamo for ROCm is our own work; RIXL DSR1 startup failure unresolved [2H] |

---

## 4. Transferable levers (framework layer) for MI355X vLLM

Each is a hypothesis to measure, not a predicted gain.

- **F1 KV-aware, DP-rank-granular prefill routing.** Replace `consistent_hash` with vllm-router 0.1.17 `kv_aware` (engine
  `--kv-events-config` publishing per rank) or the infera router's kv_event_aware policy. First on the colocated DP2/UMBP bar
  (E2/E3), then on the PD prefill pool. Discriminator: GPU+CPU prefix hit and per-rank new-token prefill balance vs consistent_hash.
- **F2 Session affinity with idle TTL 14400 s**, bound to (worker, rank), hard vs soft. AIPerf already sends the stable session id.
- **F3 Decode routed by active request count** across decode DP ranks (Dynamo weight 1e7; 0.1.17 decode minimum occupancy).
- **F4 Router never queues or rejects**: thresholds off, temperature 0, retries off, long timeouts; queue in the engine.
- **F5 Decode prefix caching off vs on** (NVIDIA: off). Run both at E7/E8; measure decode KV capacity and transfer bytes.
- **F6 Prefill host tier per prefill worker** with router host-tier credit; on ROCm use UMBP and measure the fast-path loss when it is
  combined with Mooncake/MoRIIO (E4 already planned).
- **F7 Frontend CPU path**: Rust frontend / tokenizer prefix cache / parse once / token ids to workers; place router on a NUMA node or
  host not starved by UMBP + Engram; measure router CPU at c384-c3000.
- **F8 Async KV admission cap on decode** (`max_async_kv_loads_per_step`, NVIDIA 4-6) - PD flow knob on vLLM main.
- **F9 Many small prefill DP2 instances as separate routing targets** with one wide decode - the shape, not a framework feature;
  routing granularity = 2 x N prefill ranks.
- **F10 Offline router tuning**: Dynamo ships a router simulator / mocker replay (`components/src/dynamo/router/simulation`,
  `lib/mocker`) [FH: paths exist]; could replay AgentX traces to compare policies without GPUs [INF, untested].
- **F11 (watch)** NIXL push + packed DSv4.1 rows (vllm#61121, OPEN) as the transfer path if nixl-rocm works for DSv4.1.

## 5. Open questions
- O1: Contribution of each router knob to NVIDIA throughput - no ablation published (searched IX PRs/recipes comments only).
- O2: Does vLLM ROCm DSv4.1 emit KV events whose groups Dynamo/vllm-router treat as main attention?
- O3: Does UMBP emit CPU `BlockStored` events usable for host-tier credit?
- O4: Is consistent_hash leaving prefill DP ranks imbalanced at c384 on the bar? (needs per-rank logs from E2.)
- O5: vllm-router 0.1.17 `kv_aware` with `--intra-node-data-parallel-size` on one 2-GPU DP server: works as expected? (read only).

---

## Verification (adversarial)

Verifier pass 2026-11-19. Every source re-opened by the verifier; Dynamo `components/src/dynamo/vllm/handlers.py` and
`lib/llm/src/discovery/watcher.rs` are not in the sparse clone and were fetched at `beb1a41` via the GitHub contents API into
`notes/dynamo/verify/{handlers.py,watcher.rs}.beb1a41`.

| # | Claim (short) | Verdict | Correct fact / evidence |
|---|---|---|---|
| V1 | GB300 1,011,514, 12P DEP2 + 1D DEP16, 40 GPUs, c9888, p90i 75.5, 2026-10-20, image cu134-nightly-aarch64-dev-20261019, Dynamo 1.6.0.dev20261010; frontend knobs | **Confirmed** | Row 445572 in `notes/ix/bench_dsv41_agentic_20261119.json`: prefill_num_workers 12, prefill_tp/ep 2, decode_tp/ep 16, 24+16 GPU, conc 9888, p90_intvty 75.47; `reproduce_445572.json` date 2026-10-20, run 39207089292. `gb300v_445572_config.yaml:5,13,43-65`: `placement.node: dedicated`, `numa_bind: true`, router-mode kv, TTL 14400, three active-* thresholds None, `DYN_ROUTER_QUEUE_THRESHOLD: None`, temperature 0, fastokens, cache 8,000,000,000 B (decimal 8 GB), `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT: 10000000`. Also present and not in the claim: `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS 3600`, `DYN_TCP_REQUEST_TIMEOUT 60`, `DYN_TCP_CHANNEL_BUFFER 128`, request_plane tcp / event_plane zmq (cfg:17-18). [FH] |
| V2 | B300 828,209, 5P DEP2 + 1D DEP8, 18 GPUs, c3584, p90i 83.8, 2026-11-17; same frontend minus decode weight; frontend on first_decode | **Confirmed** | Row 448173: 5 workers x tp/ep 2, decode tp/ep 8, conc 3584, p90_intvty 83.82; `reproduce_448173.json` date 2026-11-17, image `cu134-nightly-96710245ece8...@sha256:6743719343...`. `b300v_448173_config.yaml:50-65` identical keys except no `DYN_ROUTER_DECODE_ACTIVE_REQUEST_WEIGHT`, `placement.node: first_decode`. [FH] |
| V3 | Prefill MultiConnector[NIXL push + SimpleCPUOffload lazy] 85,899,345,920 / 75,161,927,680 B "per 2-GPU worker"; decode no prefix caching; admits 6/4; no KVBM/planner/conditional | **Corrected** (sizing unit) | Connector strings, decode `no-enable-prefix-caching: true`, `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` 6/4 and grep 0 for kvbm/planner/conditional all confirmed (gb300 cfg:97,150,173; b300 cfg:99,153,177). Transport differs: GB300 uses module `nixl.push_as_nixl` + `backends:[UCX]`; B300 uses `transfer_mode: push` + `backends:[LIBFABRIC]`. Sizing: on vLLM main 7a62ca0 `simple_cpu_offload_connector.py:101-107` treats `cpu_bytes_to_use` as server-wide divided by `parallel_config.world_size`, and `config/parallel.py:1182-1186` sets world_size = PP x TP x PCP (DP excluded); prefill is TP1 DP2 (cfg `tensor-parallel-size: 1`, `data-parallel-size: 2`), so under main semantics each DP rank gets the full 80 GiB, i.e. ~160 GiB (GB300) / ~140 GiB (B300) per 2-GPU worker. The images are Oct/Nov nightlies whose source was not available (shallow clone), so the semantics in the run are [INF]; "80 GiB per 2-GPU worker" (framework.md:43) should read "80 GiB per DP rank under main semantics, unverified for the run image". `VLLM_MAX_ASYNC_KV_ADMITS_PER_STEP` is absent from main; `max_async_kv_loads_per_step` exists (`config/scheduler.py:90`) but equivalence to the env var is not verified. [FH code / INF image] |
| V4 | Dynamo router cost formula; decode stage disables overlap, KV-reuse assumption, prefill-token tracking | **Confirmed, incomplete** | `router-design.md:66-85` (beb1a41). The full formula also multiplies device credit by `overlap_score_credit_decay_factor = 1/(1+decay*normalized_excess_prefill)` and subtracts `shared_cache_multiplier*(shared_beyond_blocks+fetchable_blocks)` (fetchable non-zero only under the `kvcr` policy). Host 0.75 / disk 0.25 are defaults (`configuration-and-tuning.md:74-75`), not hard-coded; the NVIDIA configs do not set them. Decode-stage list: `disaggregated-serving.md:27-29`; also line 21 prefill router `track_active_blocks=false`, which `watcher.rs:670` implements (`router_track_active_blocks = false`). Docs are main, not the 1.6.0.dev20261010 wheel. [FH] |
| V5 | Dynamo passes DP rank to vLLM as `data_parallel_rank`; session TTL wired into prefill router and decode stage | **Confirmed** (main only) | `handlers.py.beb1a41:4089,4179,4418` `data_parallel_rank=dp_rank`; `watcher.rs.beb1a41:666-690` PrefillRouter gets `session_affinity_ttl_secs`/`mode`; `:740-753` `build_preprocessed_routing_with_session_affinity_mode(..., session_affinity_ttl_secs, ...)`. Not checked against the 1.6.0.dev20261010 wheel. [FH] |
| V6 | vLLM NIXL -> `PrefillOutcome::Completed`; SGLang -> `Bootstrap` | **Confirmed, citation corrected** | Lines 129-175 only define the enum and its conversion. The selection is at `prefill_router/mod.rs:640-676`: `Bootstrap` iff the prefill endpoint registered `bootstrap_host/port` (`:820-838`) or the prefill result carries bootstrap params; otherwise the prefill stream is consumed to completion and `Completed` is used (decode leg starts after prefill finishes). The vLLM handler at beb1a41 has no bootstrap registration (grep `bootstrap` in handlers.py hits only elastic-EP lines 1191-1193), so vLLM takes the `Completed` path [FH code read; not observed in a run log]. |
| V7 | No official ROCm image; #12537 closed for inactivity 2026-11-11, KVBM scoped off; community fork; #9929 merged 2026-05-31 | **Confirmed** | `gh pr view`: #12537 (andyluo7, "feat(container): add rocm device for vllm and sglang") created 2026-08-01, closed 2026-11-11T09:35Z unmerged, github-actions stale-close comment; body lists KVBM as not wired for ROCm. #9929 merged 2026-05-31T06:23Z. The `openinferencefabric/dynamo` `rocm` fork is cited only in a PR comment (manmohanbrahma) - fork contents not opened [2H]. |
| V8 | AMD MI355X PoC: Dynamo vLLM+RIXL 2-node MiniMax-M2.5 730.4 tok/s c32; DSR1 vLLM+RIXL `ibv_reg_mr` EINVAL unresolved | **Confirmed, two qualifiers** | README copy lines 66-67,76: 730.4 tok/s, 320/320, TP4 each node, c32; 730 is aggregate across the 2-node deployment, not per GPU. DSR1 TP8 startup failure attributed by the author to vLLM-internal (lines 32-50), listed as open question. Last commits on the branch path are 2026-05-13 UTC (`107a113d`), not 05-14 (timezone). [2H: third-party fork] |
| V9 | vllm-router `kv_aware` only from v0.1.17 (2026-11-04); cost (new+active prefill tokens) x (active+1); host credit 0.8; affinity decode-only in PD; PD nixl\|mooncake | **Confirmed** | Contents API `src/policies/kv_aware.rs`: 404 at v0.1.14 and v0.1.16, present at v0.1.17; release published 2026-11-04T02:25Z. main 3920aad `kv_aware.rs:855-861,885-887` (ptoken x (batchsize+1)), `main.rs:221-222` `kv_penalty_host` default 0.8 (also external 0.6, disk 0.3), `docs/load_balancing/README.md:284` "In P/D mode, affinity applies only to decode", `main.rs:423-424` nixl default / mooncake. [FH] |
| V10 | 427,451 bar uses vllm-router 0.1.14 `consistent_hash` on x-correlation-id -> DP rank, no load/KV term | **Confirmed, nuance** | `reproduce_448242.json`: `vllm-router ... --intra-node-data-parallel-size 2 --policy consistent_hash ... --request-timeout-secs 14400 --disable-retries`, router_version 0.1.14. v0.1.14 `hash_key.rs:11-18` header order x-session-id, x-user-id, x-tenant-id, x-correlation-id, ...; aiperf 0.13.0 `record_models.py:181-184` X-Correlation-ID is per session; `X_SESSION_ID_FROM_CORRELATION_ID` default False (`environment.py:852-853`), so x-correlation-id is the key unless that env is set - either way the key is the stable session id. v0.1.14 `consistent_hash.rs:316-389` picks a (url, dp_rank) target. That the ring has no load term was not separately line-verified beyond the policy being a hash ring. [FH] |
| V11 | infera router: DP-rank KV-event routing (X-Data-Parallel-Rank for vLLM), session affinity, vLLM Mooncake PD concurrent dual dispatch, MoRIIO in Python backend | **Confirmed, caveat** | `rust/router/src/dp.rs:6-15` header honoured by SGLang and vLLM; `disagg.rs:6-16` concurrent topology for SGLang bootstrap and vLLM Mooncake; `rust/README.md:7-9` MoRIIO served by Python (`infera/router/disagg_protocols/vllm_moriio.py` exists); `config.rs:67-71` session affinity + TTL; `kv_event.rs` decodes vLLM's map wire format (lines 735, 1006, 1034). Caveat: the per-rank-port subscription text in `kv_event.rs:9-13` describes SGLang; the vLLM DP per-rank event path and any DSv4.1 run were not verified here. [FH code; no run] |
| V12 | SimpleCPUOffload emits CPU `BlockStored`, does not override `only_restores_canonical_kv`, so NVIDIA prefill MultiConnector on ROCm disables the AITER sparse-MLA fast path; UMBP emits CPU `BlockRemoved` | **Confirmed, attribution widened** | `manager.py:1206-1225` `BlockStored(medium=self.kv_event_medium)`, `:182` medium = CPU unless disk capacity > 0. Only `base.py:237` (returns False), `multi_connector.py:274-283` (all children must be canonical) and `umbp/connector.py:110-117` define it. NixlConnector does not override it either, so the MultiConnector is non-canonical because of BOTH children, and the decode-side plain NixlConnector alone also returns foreign-KV=True. Consumer: `models/deepseek_v41/amd/rocm.py:383-384` `_aiter_sparse_mla_enabled(kv_connector_may_load_foreign_kv(...))` and `:597`. So any NIXL-based DSv4.1 PD on ROCm (prefill or decode) loses this fast path on main, not only the CPU-offload combination. UMBP `worker.py:373-395` emits `BlockRemoved(medium="CPU")`; UMBP `BlockStored` CPU events not found by this verifier either (consistent with O3). [FH] |

Net effect on section 4: F6/E4 must assume the AITER sparse-MLA fast-path loss applies to NIXL on
decode too, not just to the prefill host tier (MoRIIO/Mooncake connectors not checked); and the prefill host-tier budget to copy is per DP rank (~80 GiB) under main semantics.
