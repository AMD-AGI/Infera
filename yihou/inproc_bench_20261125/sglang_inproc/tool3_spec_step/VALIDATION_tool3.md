# VALIDATION_tool3.md — GPU evidence for the tool-3 port (2026-11-25)

Host smci355-ccs-aus-n03-33 (8x MI355X), **GPUs 0-3 only**, container `yihou-inproc-t3` (image
`lmsysorg/sglang-rocm:v0.5.23-rocm10-mi35x-20261111`, sha256:6a0b7952..., sglang 6be0d75ae, label owner=yihou,
`--pid host`), model DeepSeek-V4.1-Flash. All points: `/data/yihou/inproc_bench_20261125/work_t3/iterations/<name>/`,
each with the run_decode.sh provenance set (command/git_head/code.diff/code_hashes/bench_snapshot/container-inspect
(redacted)/command.json/env.json/versions.json/kit_hashes/launch_started/launch_status/runtime.log) and `ARG CHECK: ok`.
Engine flags = `presets/dsv41_ix5034.args` (#5034) + DSPARK block 5 + `SGLANG_SIMULATE_ACC_LEN=3.51` match-expected.
Topology for every point: TP4 / DPA4 (attn_dp 4, attn_tp 1) / EP4 unless stated; resolved parallel state matched the
topology on every rank (the bench raises otherwise). Every number is a single run; no noise floor was measured.

## Points

| # | iteration | setup | key numbers | gate |
|---|---|---|---|---|
| a | `t3_e3_eager` | eager (`--disable-cuda-graph`), C16 (4/rank), ISL 1024, OSL 64, warmup 2, `--max-steps 8` | 8 iterations, accept seq 4,4,3,...; graph executes 0/0 (eager); prefill bootstrap + warmup + release + re-prefill ran | smoke (complete=false by design); `--allow-eager` gate flags only complete / C*OSL |
| b | `t3_g1_graph` | graph, C16, ISL 1024, OSL 64, warmup 3, full OSL | 18 iterations, TPOT 4.651 ms, realized accept 3.6667, graph executes target 18 / draft 18 on every rank, target_graph_iterations 18 | `verify_point.py --expect-input-len 1024 --expect-output-len 64` **PASS** |
| c | `tp4_ep4_dpa_on_c64_yihou` (sweeps/run_tp4_ep4_dpa.sh 64) | C64 (16/rank), ISL 8192, OSL 1024, warmup 10 | 288 iterations, **TPOT 6.3005 ms**, 10158.0 tok/s (2539.5 /GPU), realized accept **3.5625** (histogram 3:8064, 4:10368 request-iterations), terminal extra 128 tokens, graph executes 288/288 on all 4 ranks, captured decode bs target [16] / draft [16], backends DeepseekV4HipRadixBackend (target and draft), verify mode static, tree cache = server `build_kv_cache`, load+capture 190.7 s, bootstrap prefill 21.0 s, decode 6.45 s, peak alloc 271.4 GB/rank | `verify_point.py` **PASS**; `collect_sweep.py --prefix tp4_ep4_dpa_on_c` exit 0 |
| c' | `t3_c4_c64_isl8k_osl4k` | as c, OSL 4096 (for the server comparison) | 1164 iterations, **TPOT 6.2396 ms**, 2564.2 tok/s/GPU, realized accept 3.520619 | `--expect-output-len 4096 --expect-accept 3.520618556701031` PASS (graph 1164/1164) |
| d | `t3_p1_profile` (run_profile.sh defaults) | C64, ISL 8192, OSL 1024, warmup 10, max-steps 40, window iterations 20-25, rank 0, graph ON | torch.profiler trace `decode-...-TP-0-EP-0.trace.json.gz`; 130 distinct device kernels visible inside the replayed graphs, kernel self time 111.1 ms over 5 iterations; top-3: `mfma_moe1_silu_mul_afp8_wfp4...` 18.95 ms, `ncclDevKernel_Generic_1` 14.62 ms, `gemm2_a4w4...` 10.41 ms; DeviceTimer by runner over 40 iterations: target `target_verify` 0.788 s (19.7 ms/iter), draft 0.0758 s (1.9 ms/iter; the draft graph runner reports its forward mode as `target_verify`); `profile_meta` is_performance_measurement false; launch_status carries `is_performance_measurement:false` | profiled run: verify_point rejects it as a performance point by design |
| e | `t3_h1_hetero` | `--input-len-spec bimodal:2048,8192,0.25` (per rank: 4 x 2048, 12 x 8192), C64, OSL 1024 | 288 iterations, TPOT 6.3523 ms, realized accept **3.5625** (bit-identical to c), context_min/max logged 2401/8545 at iteration 100, graph 288/288 | `verify_point.py --expect-heterogeneous` **PASS** |
| - | `tp4_noep_dpa_on_c64_yihou` (sweeps/run_tp4_noep_dpa.sh 64) | as c with EP1 | TPOT 5.9960 ms, 10673.9 tok/s, realized 3.5625, graph 288/288 | `collect_sweep.py --prefix tp4_noep_dpa_on_c --expect-ep 1` exit 0 |
| - | `t3_x1_paths` (check_compare_paths.py) | C16, ISL 1024, 2 iterations per mode from a fresh prefill | accept lens and new seq lens equal in every pair (graph/graph_repeat, eager/eager_repeat, graph/eager); emitted ids, bonus tokens and verify-logit argmax differ in every pair incl. the repeats; logits rel-L2 0.46-1.00 | measured; cause not investigated (GLM-5.2 had the same pattern, open there too) |
| - | `t3_eq1_noprofile_{patched,pristine}_yihou` (equivalence_smoke.sh, PRISTINE = `t3_g1_graph/bench_snapshot`) | C16, ISL 2048, OSL 128, max-steps 20 | verify_iterations / realized accept / histogram / useful tokens / graph counts / config key set identical; TPOT 4.4299 vs 4.4430 ms | `EQUIVALENCE: ok` |
| - | `t3_w1_waves_c64_osl4k` (compare_server.py, server-like) | 16 warmup requests, 1 wave of 64, ISL 8192, OSL 4096 (prefill emits token 1) | decode TPOT 6.2378 ms, realized accept 3.520619, 1164 iterations all on the target graph | consistent with c' (6.2396 ms) |
| - | `t3_v1_graph_probe` (graph_visibility_probe.py, GPU 0) | eager vs hipGraph replay | eager 13 GPU entries, graph 7 entries incl. the GEMM, reduce and elementwise kernels: torch.profiler sees kernels inside a replayed graph on this image | info |

Determinism: realized accept 3.5625 / 288 iterations is bit-identical across c, d-free uniform, e (heterogeneous) and the EP1
point (OSL 1024, seed 1234). The value differs from the configured 3.51 because it is the mean of 288 match-expected
draws (fraction of 4s 0.5625 vs 0.51 expected); at OSL 4096 (1164 draws) it is 3.5206.

## (f) Equivalence vs the real #5034 SGLang server (same GPUs 0-3, same image and engine flags)

Server: `scripts/server_reference.sh` = `s2_ix5034_c256.sh` argv/env (overlap schedule ON, radix cache ON, prefill
delayer, DSPARK block 5, SIMULATE_ACC 3.51), router dropped, client `sglang.benchmark.serving` random dataset,
`--random-range-ratio 1.0 --tokenize-prompt` (exact ISL), ignore_eos (default), max concurrency 64.
Summary script: `scripts/server_tpot.py <server point> --internal <internal point> --local-batch 16`
(output in `t3_s4_server_c64_osl4k/server_vs_internal.json`).

| server point | shape | client TPOT mean / median (ms) | server steady decode TPOT (DP intervals with 16 running, queue 0) | internal | delta (server vs internal) |
|---|---|---|---|---|---|
| `t3_s4_server_c64_osl4k` | 64 requests (one wave), ISL 8192, OSL 4096, 16 warmup | 7.24 / 7.31 (client accept length 3.52) | median 7.14, mean 7.22 over 42 intervals (DP1, DP2) | `t3_c4_c64_isl8k_osl4k` 6.2396 ms (`t3_w1_waves_c64_osl4k` 6.2378) | client mean +16.1 %, median +17.1 %; steady median +14.5 %, mean +15.7 % |
| `t3_s3_server_c64_onewave` | 64 requests, ISL 8192, OSL 1024 | 7.75 / 8.24 | no interval with 16 running and queue 0 (requests finished before the last were admitted) | c: 6.3005 ms | +23.0 % / +30.8 % |
| `t3_s2_server_c64` | 256 requests at concurrency 64, OSL 1024 | 9.40 / 9.91 (P99 ITL 99.8 ms) | 2 intervals only, both right after admission (2 requests queued per DP rank otherwise); not used | c: 6.3005 ms | +49.1 % / +57.2 % |

Observed confounds (measured, not corrected): the server admits about 2 requests per DP rank per ~1 s interval
(chunked prefill 16384 = two 8k prompts) and interleaves those prefills with decode, so OSL 1024 points never reach a
decode-only steady state; in s4 the steady intervals ran DP0/DP1/DP2/DP3 = 17/16/16/15 requests (21-22 intervals each; the bench
runs exactly 16 per rank, and the steady-decode view uses only the DP1/DP2 intervals with 16); the server runs the overlap scheduler and radix cache (the bench: neither); prefix KV comes
from different random prompts; both sides simulate acceptance 3.51 from different generator states (server log accept
len mean 3.508 in the steady intervals). The +14.5 % steady-decode gap at OSL 4096 is not decomposed; I did not measure
which of these (or anything else) produces it. The next measurement that would localize it: a per-iteration server
trace (scheduler + overlap timing) at the same shape next to `steps_yihou.jsonl`, and the server with
`--disable-overlap-schedule` at the same shape.

Status: **OPEN**. The +14.5 % (steady) / +16.1 % (client mean) gap at C64 ISL 8192 OSL 4096 is recorded, not closed.

Reference from the GLM-5.2 tool (KB handoff `kbk-tool-inprocess-step-bench`, second-hand for this port): its reverse
validation was against a fake-input decode server, TP4 / EP4 / DPA **off**, ISL 70000 / OSL 10000, accept 3.61; mean TPOT
server vs internal C4 7.6103 vs 6.1505 ms (+23.74 %), C8 9.8014 vs 7.9991 (+22.53 %), C16 +0.49 %, C20 -0.71 %, C24 -0.95 %
(C4/C8 root cause open there). Its DPA-on server comparison never completed (server warmup stall). So this port's
comparison is the first DPA-on one for this tool, and it is against the real #5034 server, not a fake-input server.

## Bugs found and fixed during this validation

| symptom | iteration | cause (first-hand) | fix |
|---|---|---|---|
| check rejected a correct DPA config ("resolved (4,4,1,False)") | CPU probe in container | 6be0d75ae resolves `--enable-dp-attention --dp-size 4` to `attn_dp_size=4, dp_size=1` | check `attn_dp_size` |
| `DSpark with dp attention requires --enable-dp-lm-head` | CPU probe | engine check in `_handle_dspark` | preset `dsv41_ix5034.args` |
| `'NoneType' object has no attribute 'spec_algo'` at the first decode | `t3_e1_eager` | `resolve_forward_inputs(batch, None)` dereferences the FutureMap on decode | call it only for prefill staging |
| `int(batch.forward_iter)` TypeError in DSPARK observers | `t3_e2_eager` | the Scheduler stamps `batch.forward_iter` (scheduler.py:4961) | bench stamps a forward counter |
| `struct.error: 'i' format requires ... <= 2147483647` in the FlyDSL MXFP8 GEMM during bootstrap | `failed1_tp4_ep4_dpa_on_c64_int32_prefill` | one 131072-token prefill extend (16 x 8192) | group prefill by the resolved `chunked_prefill_size` (16384) |
| client `No module named sglang.benchmark.serving` | `t3_s1_server_c64` | `docker exec` default workdir of the image shadows the package | `docker exec -w /` |

## Not validated on GPU (code paths only)

`--profile-activities MEM,CUDA_PROFILER`, `--profile-graph-capture-trace` / `--enable-profile-cuda-graph` (CAPTURE_PROFILE=1),
`--dp-metadata static`, `compare_server.py --initial-state sikl`, `--input-len-spec normal:/list:` on GPU (CPU-tested),
DPA-off topologies, ISL above `chunked_prefill_size` per request (bootstrap would then run one request larger than a
server chunk; not exercised).

## Open items

- Server-vs-internal gap at the #5034 shape (above): +14.5 % steady-decode, +16.1 % client mean, not decomposed.
- Graph vs eager / repeat divergence in `t3_x1_paths` (tokens and logits differ even between repeats of one mode).
- Code paths not exercised on GPU (previous section).
- No same-node repeat pair measured, so no noise floor for deltas such as EP1 vs EP4 (5.996 vs 6.300 ms).
