# Run plan: llying best -> same-node P4D4 (S1) -> DCP4 decode (S2) -> + OffloaderV2 G4K1 on PP prefill (S3)

Written 2026-10-22, read-only analysis; nothing launched. Inputs: `design/tmp_ll/llying_best.md`, the same-node /
DCP-compat / offloader analyses of workflow wn0hnfjbd, `design/llying.analysis.user.md`.
Abbreviations: `LL` = `~/dev/git/infera.glm52.pd/llying/2p1d-v0522-20261020`; `BR` = Infera
`origin/feat/sglang-mix-glm5.2-tp4-v0522` @ `d395c3b3`; `K` = `BR:examples/sglang_2p1d_glm5.2`; `PP4D` =
`LL/config/kit-v6-pp4.diff`; `PK` = `~/dev/git/infera.glm52.pd/yihou/glm52.dcp4-pd.packup_20261022`;
`SG` = `/home/yihou/dev/git.16-19/sglang.pd.pp.dcp`; `WS` = `SG/yihou_ws/glm52_dcp_pd_20261021`.

## 0. Directives this plan applies

| Directive | Source | Effect on plan |
|---|---|---|
| Each experiment runs once, 30 min, no A1/A2 | user 07:54Z, `WS/poll.md:493` | AgentX `--benchmark-duration 1800`, warm-up 10/lane kept (not scored); one full-stack restart per stage |
| OAI nodes only (burst/v3 not used) | task context | amdgpu 6.14, no `HSA_ENABLE_SDMA=0` (`PK/environment.md:31-34`) |
| Custom AR ON, ignore correctness for now | user 10:12Z, `WS/poll.md:553` | Remove `--disable-custom-all-reduce` (`K/engine/worker.sh:67`) on both legs in S1-S3. No-op on the TP1 prefill. On S1's TP4/DPA4 decode this is the config llying measured garbling 53-61/200 (`LL/../mix-v0522-correctness-20261019/README.zh-CN.md:3-33`) |
| index_share (shared index) ON | user 10:09Z, `WS/poll.md:552` | `{"index_share_for_mtp_iteration":true}` on both legs in S1-S3 (llying ran false, `WK:68-71`) |

Both overrides are applied in all three stages so that S1->S2->S3 each change one thing. Consequence: S1 is NOT
llying's exact best (custom AR off, index_share false); it is "llying best + the two user overrides, same-node".

## 1. Common to all stages

| Item | Choice | Evidence / reason |
|---|---|---|
| Node | ONE OAI node for all three stages (8 x MI355X; P = GPU 0-3, D = GPU 4-7). Candidates as last seen: 151 (job 232116, idle after mixrun stop), 105 (231196, r7), 268 (231959, r8). Availability UNVERIFIED: crusoe login nodes unreachable since ~10:08Z (`WS/poll.md` 10:13Z, 10:17Z) | Same node removes the node variable between stages |
| Image (one for all stages) | NEW `infera-sglang:v0522-d395c3b3-1020-mc4340`: Infera `d395c3b3` `deploy/docker/Dockerfile.sglang`, base switched to `lmsysorg/sglang-rocm:v0.5.22-rocm720-mi35x-20261020` (our base), Mooncake `faae8dd4` + `rdma_dest_local_rail.diff` + `rdma_ack_timeout.diff` + our #4340 port (`PK22/patches/image-recipe/files/mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff`, added to `build_mooncake_sglang.sh:34-35`). `docker save` to NFS | (a) Our image's Infera `28121cb4` has no R1 / R2 / session-affinity flags (`git show 28121cb4:infera/server/args.py`); BR has them (`args.py:116,130,138`); router is compiled rust, so no source mount. (b) Same-node HIP IPC needed #4340 on faae8dd4 (`PK22/patches/README.md:16`); llying's build lacks it. (c) Base 20261020 is the only base our overlay has run on (`PK/RUN_BEST_DCP.md` §4) |
| Baked sglang patches | Skip/allow-fail the `--fuzz=0` sglang patch step in the build copy | They target base 20261018 (`Dockerfile.sglang:5-8,113-195`) and are fully replaced by our overlay, which carries each one (DCP-compat table (g): `e75479a189`, `9118404a4a`, `344765b3bc`, `e11f97495d`, `7abe9a8d5e`, `3c733c7f5b`, `36bc86517d`, `4d1d6a647e`). Whether the Dockerfile tolerates skipping: UNVERIFIED |
| sglang source | Overlay of `dev/yihou/glm52_dcp_pd` = `93034eba91` + ONE new commit (PP fix: `self.embed_tokens.embedding_dim` -> `config.hidden_size` at `SG/python/sglang/srt/models/deepseek_v2.py:3003,3038`), copied to `/shared_nfs/yihou_glm52/dcp_pd_20261021/src/sglang-<newsha>`, mounted ro over `/sgl-workspace/sglang/python/sglang` on BOTH legs | llying's whole-file mount (`PP4D:88-113,136-138`) is a 20261018 file and would hide our `20107f85b8` (sharded draft) changes; port the two lines instead |
| Harness | Copy of `K` @ `d395c3b3` + `PP4D` (`patch -p1`) in `/shared_nfs/yihou_glm52/<new>/kit`, with edits below | Our harness lacks PP, the three router flags, prefill EAGLE and `--tokenizer-worker-num` wiring (`PK/scripts/harness/bench-harness/engine.sh:257,273`; `launch.sh:196-197`) |
| Kit edits | `engine/common.sh:34` `on_decode` runs locally when `DECODE_SSH=local`; `worker.sh:45` `MC_TE_FILTERS=${MC_TE_FILTERS_OVERRIDE:-$RDMA_DEVICES}` with override `yihou_no_such_rdma_device`, plus `--disaggregation-ib-device '{"0":"ionic_0",...,"3":"ionic_3"}'` as r5b; `worker.sh:110-115` 30,000-PID burn also for `role==decode`; `worker.sh:67` drop `--disable-custom-all-reduce`; `index_share` override -> true; `engine/down.sh:20,27` `docker rm -f` -> `docker stop --time 60`; container names `yihou-dcp-ll-<stage>`; overlay mount on both legs; PP4D's `deepseek_v2.py` mount removed | Same-node facts: r5b decode command uses the PID burn, the bogus filter and the per-GPU ib map (`PK/scripts/harness/dryrun.pdBeperf-r5b-060.yihou.txt:27`); llying's kit only burns PIDs for prefill 1 (`worker.sh:110-115`) and starts decode over ssh (`up.sh:38`); project rule 1 (stop, not rm) |
| Prefill (all stages) | 1 x TP1 PP4 on GPU 0-3, `--disable-overlap-schedule`, `SGLANG_PP_LAYER_PARTITION=20,20,20,18`, chunk / max-prefill 8192, max running 256, mem 0.88, SPF, HiCache 250 GB/stage write_back ratio 1.5 kernel page_first, EAGLE 5/1/6 (prefill MTP), triton DSA, fp8 KV, kv-events on | `PP4D:41-80`, `WK:69-91` |
| Router | infera rust router, `kv-aware`, overlap 20/2, `--session-affinity prefill --pd-prefill-guard-release completion --decode-input-demand on`, `ROUTER_EXTRA_ARGS` empty | `K/engine/up.sh:46-52`; `LL/results/pp4-ck8k-d4-c80/engine-config.env:18` |
| Client | InferenceX `918524ff`, `--scenario inferencex-agentx-mvp --concurrency 80 --benchmark-duration 1800 --warmup-requests-per-lane 10 --warmup-grace-period 1800 --random-seed 42 --num-dataset-entries 393 --use-server-token-count`, `AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true`, sim 3.61 on decode | `LL/results/pp4-ck8k-d4-c80/benchmark_command.txt:1`, `runtime.env:19`; duration per directive |
| Concurrency | **C80 in all three stages** | llying's 8-GPU frontier best is C80: 37,310 tok/s/GPU @ intvty p90 44.3; C64 = 32,532 @ 48.6 (`LL/README.zh-CN.md:654-668,705`). At C80 the 4+4 decode KV is binding (usage mean/p90/max 76/95/99 %, up to 25 waiting for prealloc) while step time is still on its plateau (66-70 -> 71-78 ms), and llying puts the 4+4 limit at about C80 (`README:667-668,701`). That binding constraint is what S2 (single DCP pool) and S3 (more prefill KV) touch, and one C keeps the stages comparable. Alternative if the user prefers the base in `design/llying.analysis.user.md` (4+4 C64 32,532): use C64 for all three |
| Scoring label fix (optional) | `agentx.sh:66-72` hardcodes prefill TP4, pp/dcp 1 -> metadata only | `llying_best.md` §9 |
| Per-stage time | image build once; bring-up 10-65 min with HiCache pinning (`LL/README:440`), debug bring-up + gate ~25 min, warm-up ~20 min, scored 30 min, down. About 1.5-2 h per stage | estimates from cited durations, UNVERIFIED on our node |

Pre-flight once per stage (read-only): `rocm-smi` GPUs 0-7 empty; `ss -lnt` for 27201(+233..243), 27301, 28398, 27757,
29051, 28300, 24701 (`K/env.sh:81-88`); host free memory >= 1.1 TB (S3: >= 1.2 TB; HiCache 4 x 250 GB + indexer
cache, `LL/README:442-445`); etcd image `quay.io/coreos/etcd:v3.5.14`; `/lib/x86_64-linux-gnu/libionic.so`. Render
commands and diff against the previous stage's rendered commands: only the deltas in the stage table may differ.

## 2. Stages

### S1 — llying best, same-node P4D4 (reference)

| Item | Value |
|---|---|
| Deltas vs llying C80 (`LL/results/pp4-ck8k-d4-c80/`) | same node instead of 138->140; image/source as §1 (base 20261020 + our overlay instead of `v0522-fdbafb60` on 20261018); HIP IPC transport (`MC_TE_FILTERS` bogus, 0 HCAs) instead of RDMA; decode PID burn; custom AR ON both legs; index_share true; 1800 s instead of 3600 s |
| Decode | TP4 `--dp-size 4 --enable-dp-attention --ep-size 1` on GPU 4-7, mem 0.92, graph bs 256, max running 256, `--tokenizer-worker-num 4`, EAGLE 5/1/6, sim 3.61 / match-expected / real-draft-token, kv-events off (`PP4D:80`, `WK:95-101`) |
| Correctness gate (informational per 10:12Z; blocks only on crash / hang / all-garbled) | Debug bring-up `PP4_HICACHE=0`, sim OFF: Infera log "verified real KV transfer" (never seen on HIP transport, UNVERIFIED); `engine/smoke.sh` (llying 6/6, `README:431`); record real acceptance length (llying PP4 median 4.55, `README:437`, measured with index_share false) |
| Measure | tok/s/chip (`per_gpu.total_tput_tps`), TPOT mean/p50/p95, ITL p50/p90, intvty p50/p90, TTFT mean/p50/p90/p95, completed/invalid; decode ms/step and running distribution, per DP rank (busiest rank), KV usage mean/p90/max, prealloc wait, transfer-in; prefill queue mean and % time queued, per-stage scheduler stage seconds (`/metrics` snapshot as `prefill-*.prom`); KV pool tokens per leg from startup log |
| Reference numbers | llying 4+4 C80 37,310 @ intvty p90 44.3, ITL p50 20.3 ms, TTFT p50/p90 3.75/10.83 s, decode 71-78 ms/step @ running median 52 (`README:654-668`) |
| Stop condition | Bring-up: KV probe fails, OOM, PP stage AttributeError, or `hipErrorCapturedEvent`/graph-capture failure -> stop, report, at most one config fix. Run: scheduler crash, > 1 % invalid requests, or no new prefill/decode batch for 5 min -> stop. Gate to S2: S1 completes and scores. If S1 < 0.9 x 37,310, pause and diff rendered config against llying's before S2 (a difference vs llying is a finding, not a blocker; listed deltas make it non-attributable) |

### S2 = S1 with the decode leg swapped to DCP4 (r6b knobs)

| Item | Delta vs S1 (decode leg only; prefill, router, client, image, overlay unchanged) |
|---|---|
| Parallelism | `--tp-size 4 --ep-size 4 --dcp-size 4`; remove `--dp-size 4 --enable-dp-attention` (`worker.sh:108`) |
| Env | `SGLANG_DSA_DCP_KEEP_KV_B_BF16=1 SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1 SGLANG_LOG_DECODE_SEQ_LENS=1`; `SGLANG_DP_USE_GATHERV=0` (r5b rendered decode, `PK/RUN_BEST_DCP.md` §3; r5b dryrun `:27`) |
| Memory | mem 0.85 (r6b) instead of 0.92 |
| Graph / running | r6b has 64/64 (`dryrun:27`). C80 exceeds 64, so set `--cuda-graph-max-bs-decode 96 --max-running-requests 96` (deviation from r6b, forced by C80). DCP4 capture at bs 96 is unrun -> check capture log |
| Custom AR | ON (unchanged from S1 under the user override; also r6b) |
| Router | unchanged flags; R2 and decode-side choice become no-ops with one DP1 decode target (`BR rust/router/src/pool.rs:137-139`). BR has no `INFERA_PD_DP_RANK_AFFINITY` |
| Seed path | index_share true -> seed width > 0 (`SG/.../disaggregation/utils.py:130-137`), prefill sends `output_dsa_topk_indices` (`prefill.py:917-923,1014-1022`); remap is off under DCP (`layers/attention/dsa/utils.py:173-181`). Never run in PD+DCP (PD decodes r2-r6b ran false, `WS/poll.md` 09:50Z). Fallback switch: `SGLANG_DSA_DCP_PD_DROP_MTP_SEED=1` (`ae529c96bf`, `eagle_disaggregation.py:111-116`) |
| Correctness gate | As S1, plus confirm "prepared full-head Q weights for 78 MLA layers" x4 and `required_dst_info_num=4` path (`common/conn.py:1368-1378`); real acceptance length vs S1. A PP-last-stage draft entry with `draft_sharded=True` onto DCP4 has no CPU test (`646dc699f3` covers draft only for index-K); first real exercise is this gate |
| Measure | As S1; decode step p50 by running bucket (1-8 ... 49-56, 57-64, ...) via `PK/tools/decode_stats.dcp.yihou.py`; decode KV pool tokens; prealloc / transfer-in queues |
| Stop condition | As S1, plus: graph capture fails at 96 -> fall back to 64/64 (r6b) once and record; garbled-only output is not a stop (10:12Z) |

### S3 = S2 with OffloaderV2 G4K1 on the PP4 prefill

| Item | Delta vs S2 (prefill leg only) |
|---|---|
| Flags | `--offload-group-size 4 --offload-num-in-group 1 --offload-prefetch-step 1 --offload-mode cpu`; no `--cpu-offload-gb` (`offloader.py:73-76`); prefill CUDA graph backend stays off |
| Code | No change: is_shuffled fix `c9394c114f` is an ancestor of HEAD (`offloader.py:429-435`); `SGLANG_RUN_ID` set in `entrypoints/engine.py:1903-1905`, read at `offloader.py:219` |
| Expected placement | Per stage, local indices 3,7,11,15,19 -> global 3-19 / 23-39 / 43-59 / 63-75 = 5/5/5/4 layers (`common.py:1621-1657`, `offloader.py:255-257`); frees ~22.6/22.6/22.6/18.1 GiB, pins the same in host RAM |
| Correctness gate | Log `[offloader] offload module_index=...` with 5/5/5/4 per stage; per-stage weight memory and KV pool vs S2; smoke sim off with real acceptance vs S2 (prefill MTP + offloader never run together) |
| Measure | As S2, plus per-stage prefill forward time for 8192-token chunks (scheduler stage seconds per pp rank), prefill KV pool tokens (limited by the tightest stage, `kv_cache_configurator.py:2799-2806`), prefill queue / TTFT |
| Stop condition | As S1, plus: offloader warm-up hang on aiter JIT locks (seen before, `...offloaderv2-highconc.packup_20261015/README.md:51-54`) > 20 min -> stop |

## 3. Risks (evidence; no outcome predicted)

| # | Stage | Risk | Evidence |
|---|---|---|---|
| 1 | all | New image build: patch apply and skip of baked sglang patches | #4340 diff and RDMA diffs touch disjoint files (clean apply UNVERIFIED); `Dockerfile.sglang:5-8` `--fuzz=0` |
| 2 | all | New Infera KV probe / Mooncake preflight with 0 HCAs on HIP transport | probe `29d6fa74`, preflight `ac380139`; never run on HIP path (UNVERIFIED) |
| 3 | S1 | Custom AR ON on TP4/DPA4 decode garbles outputs | llying 53-61/200 garbled with AR on (`mix-v0522-correctness-20261019/README.zh-CN.md:3-33`); user accepts (10:12Z). Scores remain sim-acceptance scores |
| 4 | S2 | DCP4 decode step time at ~50 running is unmeasured | r6b step p50 by running 1-8/9-16/17-24/25-32/33-40 = 32.9/48.5/59.1/69.7/78.1 ms (`WS/ledger.pdrun.md:408-411`); llying DP4 at C80: 71-78 ms @ running median 52 (`README:662-664`). Our PD decode never exceeded ~52 running (r4 C52 max 52, `PK/results/RESULT.md:81`); mixed DCP4 C48 with max running 96 dropped to 13,012 (`RESULT.md` §3). gfx950 fused indexer declines above 48 rows = bs 8 x 6 (`WS/poll.md` 09:57Z) |
| 5 | S2 | Decode KV capacity lower than S1 | DCP4 mixed 8,894,208 tokens at mem 0.85 (`PK/results/RESULT.md:109`) vs llying DP4 3,001,920 x 4 = 12.0M at 0.92 (`README:621`); whether one shared pool offsets this at C80 is unmeasured |
| 6 | S2 | index_share seed path under PD+DCP, and from a PP last stage | never run (§S2 seed row); fallback env exists |
| 7 | S2 | Graph capture at bs 96 under DCP4 | capture only run at 64 (`dryrun:27`); DP2xDCP2 capture failed with AR off (`PK/notes.md:111-128`, different layout) |
| 8 | S3 | H2D bound on PP4 TP1 | ~24.3 GB per stage forward; measured 4-GPU pinned H2D 34.6-46.3 GB/s (`...offloaderv2.packup_20261015/results/h2d/h2d_bench.089.json`, other node) vs ~0.62 s/stage step from llying C80 prom; copy-hide window ~124 ms vs 105-140 ms per copy. Contention with HiCache write_back unmeasured |
| 9 | S3 | KV gain may not bind | PP4 prefill has 52-67 GB free/GPU already (`LL/README:433`); llying's PP4 limits are two-pass KV release and ~3.0 req/s per prefill (`README:499-500,697`) |
| 10 | S3 | 9 GiB transient per stage outside mem-fraction | prefetch 1 keeps two full layers resident (`offv2_cost.md:152`) |
| 11 | all | Same-node RCCL shm name collision | handled by decode PID burn as in r5b (`dryrun:27`); prefill is a single process group set |
| 12 | all | Comparison to llying's 37,310 is non-attributable | deltas listed in S1; 1800 s vs 3600 s; n=1 each |

## 4. Deliberately NOT carried over

| Item | Why |
|---|---|
| Decode request-count balancing (`--decode-request-weight`, `feat/decode-dp-request-balance` `a8cf30bf`) | Not merged; would need another router build; no-op for the DP1 decode in S2/S3; would add a variable to S1 only |
| llying's `deepseek_v2.py` file mount | 20261018 file; hides our `20107f85b8`; replaced by a 2-line commit on our branch |
| llying image's baked sglang patches | Superseded by our overlay, which carries each one (DCP-compat table (g)) |
| `--disable-custom-all-reduce` | User override 10:12Z (AR ON); r6b decode is AR on (-3.5..-3.9 ms/step vs off, task context) |
| `index_share_for_mtp_iteration=false` | User override 10:09Z |
| RDMA cross-rank Mooncake env as the transport (`MC_ENABLE_DEST_LOCAL_RAIL`, `MC_IB_TIMEOUT`, `MC_TE_FILTERS=ionic_*`) | Same node uses HIP IPC; RDMA patches are inert there. Env left set except the filter. Whether `MC_MAX_MR_SIZE` touches HIP transport: not checked |
| Decode mem 0.92 / graph 256 / max running 256 in S2-S3 | r6b knobs are 0.85 and 64; raised only to 96 to admit C80 |
| `SGLANG_OPT_USE_TOPK_V2=true` as a decode knob in S2-S3 | Not read on the decode path with the split knobs (`PK/notes.md:130-137`); left set, inert |
| 2P / TP8 decode layouts, C176 | Out of scope (P4D4 on one node) |
| A1 -> flush -> A2 protocol; 3600 s windows | User directive 07:54Z |
| `HSA_ENABLE_SDMA=0`, ROCm debug agent | OAI nodes are amdgpu 6.14; debug agent off for perf (`PK/RUN_BEST_DCP.md` §3) |
| DP2 x DCP2 decode | Graph capture failure with AR off and garbling in graph mode (`PK/notes.md:111-128`) |

## 5. Open items before launch

1. Node availability (crusoe login outage since 10:08Z).
2. Image build feasibility on base 20261020 with Infera `d395c3b3` (risk 1).
3. Commit the PP fix on `dev/yihou/glm52_dcp_pd` and build the new overlay directory.
4. User choice: C80 (plan default) or C64 (base named in `design/llying.analysis.user.md`).
