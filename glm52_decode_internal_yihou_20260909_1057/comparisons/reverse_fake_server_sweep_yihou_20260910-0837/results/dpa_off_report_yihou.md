# DPA-off reverse fake-server sweep — completed

## Result
All five requested DPA-off configurations completed: TP4/EP4/DP1, C4/8/16/20/24, ISL70000/OSL10000. Every selected run has client exit0,128 successful measured requests,1280000 output tokens, and128 exact input/output lengths70000/10000. Total640 measured requests/6400000 outputs. DPA-on was explicitly paused by the user and remains unresolved; none of its diagnostic runs is included.

**Performance agreement is batch-size dependent:** mean TPOT atC16/20/24 is within1% of the internal baseline; atC4/C8 the server is23.7%/22.5% slower. These measurements do not establish token/state correctness or equivalence of the two methods. No kernel changes or timing-fitting parameters were applied to the selected DPA-off runs.

## Paired measurements
Delta=(server/internal-1)*100. TPOT in milliseconds; throughput in output tokens/second.

| C | Internal mean TPOT | Server mean TPOT | Server P50 | Server P90 | TPOT delta | Internal throughput | Server throughput | Throughput delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|4|6.1505|7.6103|7.5073|8.0660|+23.74%|650.36|525.30|-19.23%|
|8|7.9991|9.8014|9.8431|9.9846|+22.53%|1000.11|815.14|-18.49%|
|16|10.2861|10.3368|10.3162|10.4317|+0.49%|1555.50|1544.22|-0.73%|
|20|11.6671|11.5842|11.6802|11.7251|-0.71%|1714.23|1595.51|-6.93%|
|24|12.6322|12.5123|12.7675|12.8635|-0.95%|1899.91|1759.40|-7.40%|

| C | Server measured duration s | Effective concurrency (native client) | Realized acceptance length |
|---:|---:|---:|---:|
|4|2436.6974|3.9998|3.609725|
|8|1570.2742|7.9980|3.611754|
|16|828.8993|15.9873|3.613355|
|20|802.2508|18.5019|3.612245|
|24|727.5199|22.0266|3.605649|

The five successful measured client durations sum to6365.64s (106.09min), excluding startup,warmup and interrupted attempts. First selected server started2026-09-10T09:09:41Z; final C4 container stopped2026-09-10T12:09:23Z. Experiments were sequential on node271; the earlier internal baseline used node056.

## Interpretation
- Server TPOT is the native per-request `(latency-TTFT)/(output_len-1)` statistic. Internal TPOT is full synchronous decode-loop elapsed/OSL for one fixed batch. Server P50/P90 are request quantiles,not internal wave statistics.
- Native server runs128 rolling requests after16 warmups(output32). Internal baseline runs one fixed batch and step warmup. C20/C24 do not divide128; their measured effective concurrency is18.50/22.03 rather than20/24. This provides evidence for a tail/occupancy contribution to the6.9%/7.4% aggregate throughput gap, despite close meanTPOT. It is not a controlled causal decomposition.
- C4/C8 effective concurrency is essentially full and acceptance is near3.61,so those two observations alone do not explain their22–24% TPOT gap. Root cause remains open; no extra tuning or numerical debugging was performed.
- Fake handoff supplies the first output token;10000 outputtokens include that token. PrefixKV/proposal state and recycled-page history differ from the internal randomized synthetic prefix and bootstrap. Both use simulated acceptance,not measured production acceptance or real P-to-D KV transport.
- Both configuredFlyDSL. Selected C20/C24 logs explicitly show sparseMLA declining verify rows120/144 (supported1..96),consistent with the internal baseline's fallback. MoE heuristicFlyDSL configs also appear; do not describe the runs as fullytuned or all-FlyDSL-phase execution.

## C20 startup caveat
An unrelated eight-GPU ATOM workload appeared duringC20startup and was stopped after exactidentityverification under userauthorization. Its memory release affected free-memory-delta accounting:server_info reportsweight=-13.166GB (an accounting artifact) and KVcapacity3246912 tokens versus2860160 in the other four selectedruns. Effective runningcap remained20; maximum loggedretractions is0 for allfivepoints. No competingcontainer was observed during the subsequentC20measurement checks, but monitoring was periodic,not continuous. Preserve thisconfigurationconfound; C20 is an execution-successful result,not an exact memory-layout-controlled comparison.

## Evidence and configuration
- [Exact CSV](dpa_off_final_yihou.csv)
- [Verification audit and raw-result SHA256](dpa_off_verification_yihou.json)
- Internal source:[reference/internal_summary.csv](../reference/internal_summary.csv), from `/home/yihou/dev/git/infera.dev.yihou.sglang.bench.fast.script/packups/glm52_tp4_ep4_dpa_sweep_yihou.packup_20260910-062400`.
- Server method:`/shared_nfs/yihou/packups/glm52_fake_tp4ep4_10k500_c16_c32.packup_20260910-055810`.
- Image:`sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d`; SGLang402df1e1e453e1e85ec0f5ac4052d36598cc691a; AITER2c71811b32c8ce2e1266aedaec199df7d90f597d; model`/shared_nfs/models/GLM-5.2-MXFP4`.
- Three fake-source overlays match the reference package byte-for-byte. Selected runs use original packagedlaunch/bench scripts with workload/destination parameters; fixed EAGLE5/6/topk1,FP8e4m3KV,mem0.85,AITERtopk/fusions,simaccept3.61/match-expected/real-draft-token,overlapenabled. No synchronousdiagnostic included.

Selected round directories under`../rounds/`:
- `original_off_c4_retry5_yihou`
- `original_off_c8_retry4_yihou`
- `original_off_c16_retry1_yihou`
- `original_off_c20_retry5_yihou`
- `original_off_c24_retry5_yihou`

Each contains exactserver/clientcommands,containerinspect,readiness/serverinfo,rawbenchmarkJSONL,clientexitcode,clientlog,and fullserverlog. Interruptedrounds retained separately and excluded: notably C4retry2 had observedconcurrentGPUinterference and was externallykilled. Do not use earlier autogenerated`results/report.md` or`results/summary.csv` from the abandoned framework; this report and`dpa_off_final_yihou.csv` are authoritative.

## Completion
Final original-script loop b1qram9my exited0. Final owned server stoppedexit0; no files or containers deleted,noallocationcancelled,no commits/pushes. Seven-minute reminder02ab19ee cancelled after allfivepoints verified. DPA-on remains paused; no further experiments scheduled.
