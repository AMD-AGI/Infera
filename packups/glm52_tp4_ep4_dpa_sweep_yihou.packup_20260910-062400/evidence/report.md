# TP4/EP4 internal decode DPA sweep — 2026-09-10

## Result
All ten full-output points completed with exit0 and correct global useful-output counts. Four MI355X GPUs on node056/job130737, ISL70000, OSL10000/request, EAGLE5/6/topk1, expected accept length3.61. Each run executed2768 speculative iterations, realized accept length3.613439306. Every rank recorded2768 target,draft,draft-extension graph executes. Global concurrency is C, not per-DP concurrency.

| Global C | DPA off TPOT ms | DPA on TPOT ms | Off output tok/s | On output tok/s | Off decode s | On decode s | Off launch-wall s | On launch-wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4|6.150486|7.199074|650.355|555.627|61.505|71.991|153|153|
|8|7.999084|8.870461|1000.115|901.870|79.991|88.705|163|168|
|16|10.286111|10.980857|1555.496|1457.081|102.861|109.809|185|191|
|20|11.667059|11.807833|1714.228|1693.791|116.671|118.078|208|198|
|24|12.632195|12.276834|1899.907|1954.901|126.322|122.768|217|214|

Per-GPU throughput is the aggregate value divided by4. TPOT is synchronized complete-loop elapsed/10000 useful output tokens per user, not client streaming ITL or a percentile. Global useful count is C*10000; terminal excess is excluded. Actual raw terminal state may extend beyond80000, with useful/accounting state clipped to80000 as in the baseline.

Observed: DPAoff is faster atC4–20; DPAon slightly faster atC24. This is one run per condition, not statistical significance or a pure DPA overhead measurement.

## Critical backend qualification
The configured backend is FlyDSL, but runtime fallback depends on topology and shape:
- DPAoff uses attentionTP4 with16 local Q heads. C4/8/16 have no FlyDSL sparse MLA decline. C20/C24 verify shapes have120/144 rows, exceeding supported1..96, and fall back for that phase. Other phases still emit decode engagement markers.
- DPAon uses DP4/attentionTP1 with64 local Q heads. **All five DPAon points decline FlyDSL sparse MLA**: logs require `(seq,8 or16,576)` but receive `(6/12/24/30/36,64,576)` at verification; draft shapes also decline. Zero FlyDSL decode-engagement markers were observed for these runs. The pinned stack's fallback executes inside CUDA Graph normally.
- Fused gfx950 DSA indexer engagement remains logged on all4 ranks in all ten points. Some MoE shapes use heuristic FlyDSL configurations rather than tuned configurations.

Therefore these results compare the same pinned stack and its default dispatch/fallback policy, **not identical attention kernels with only DPA changed**. No core kernel changes were made mid-sweep. Backend evidence is in verification_yihou.json and each runtime.log.

## Method and implementation
Authoritative Git repo: /home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script, branch dev.yihou.sglang.bench.fast.script, baseline4d982fa. Benchmark lives under glm52_decode_internal_yihou_20260909_1057. Code changes remain uncommitted as requested scope did not include a commit/push; each run captures HEAD,diff,hashes and a full bench source snapshot, including new untracked topology.py.

No Scheduler/HTTP/PD deployment. Real target/draft workers and real weights/routing; physically initialized synthetic KV/index prefix. DPAoff: TP4EP4DP1, localC replicated across attentionTP ranks, count rank0 once. DPAon: TP4EP4DP4, localC/4=[1,2,4,5,6], count each DP replica once. Base DP token metadata is passed to ForwardBatch, which scales it per speculative phase. All ranks share acceptance RNG seed1234 so collective iteration/completion stays synchronized. Cross-rank seven-integer progress checks run every warmup/measured iteration and are included in timing. This extra validation collective was not in the old TP8 benchmark, so do not attribute cross-experiment differences solely to topology.

Ten warmup iterations and exact-ISL reset precede timing. Measured loop includes actual draft,verify,draft-extension,synchronization,validation and bookkeeping; excludes load,pool initialization,capture,bootstrap,warmup,result serialization. Graph counters count actual execution, not presence. Per-rank reports stay local; aggregate result is explicitly global.

## Runtime and duration
- Image sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d.
- SGLang402df1e1e453e1e85ec0f5ac4052d36598cc691a; AITER2c71811b32c8ce2e1266aedaec199df7d90f597d.
- Torch2.9.1+rocm7.2.0.git7e1940d4, driver6.14.14, four visible MI355X via HIP_VISIBLE_DEVICES=0,1,2,3.
- Model /shared_nfs/models/GLM-5.2-MXFP4 read-only; config/index hashes in iterations/000_environment/environment_summary.json.
- Static memory fraction0.85,FP8 KV,AITER top-k,MoE a2a none (still communicates),fused allreduce and QK norm/rope. Original optimization environment preserved.

First cold TP4EP4off smoke:891s total (~14.9min),load/pool/capture802.27s. Following DPAon smoke:102s. Full sweep points:153–217s each (~2.6–3.6min); summed launch-wall time1850s (~30min50s), excluding setup and smoke. Actual decode61.5–126.3s. These are measured values, not a guarantee for a cold node/new kernel shape.

## Reproduce
Use an already authorized allocation; never submit/cancel jobs. Current ownership was056/job130737. Peer changed from249 to036; this task never operated either. Revalidate ownership and coordinate future node changes before any command. Unique owned container must exist or be created with scripts/create_container_yihou.sh; stale job IDs are not permanent allocations.

```bash
ROOT=/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/glm52_decode_internal_yihou_20260909_1057
export JOB_ID=130737 NODE=crsuse2-m2m-056 CONTAINER=yihou-glm52-tp4ep4-sweep-20260910
export OUTPUT_ROOT="$ROOT/sweeps/new_sweep_yihou/iterations"
# Start this task's retained stopped container only after resource revalidation,
# or create a fresh uniquely named container with create_container_yihou.sh.
bash "$ROOT/scripts/run_tp4_ep4_sweep_yihou.sh"
python3 "$ROOT/scripts/collect_sweep_yihou.py" "$OUTPUT_ROOT" --output "$ROOT/sweeps/new_sweep_yihou/summary.csv"
```

The launcher refuses existing iteration directories and forbidden/peer nodes. Each point records launch_status.json and live runtime.log; failure stops the sweep. `result_yihou.json`, local rank reports and step logs retain full evidence. `summary.csv` is machine-readable output; `working_process.md` records research, cleanup authorization, smoke and checkpoints.

## Verification and limits
All ten global counts,final useful contexts,per-replica aggregation,all-rank graph counts and frozen-source snapshots verified.23 CPU tests and5 launcher tests passed before execution; final verification reruns them. No numerical/text-quality evaluation or per-token client ITL distribution was measured. Synthetic positive FP8 prefix values are not realistic KV statistics. Expected acceptance3.61 is simulated and not a probability; real-draft-token mode does not make acceptance correctness-valid. No statistically controlled repetitions were requested; no speedup significance claim.

User explicitly authorized cleanup of non-our-party GPU workloads on056. Two identified GPU containers were stopped after mount/PID inspection; collectors retained. No files/containers deleted, no host reset, no allocation requested/cancelled. Other shared users can start workloads on the node; continuous GPU exclusivity was not instrumented, so this remains a shared-cluster measurement. Final workers exited and GPU memory returned0% before stopping only this task's container.
