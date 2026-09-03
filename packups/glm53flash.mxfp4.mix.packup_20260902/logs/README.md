# Logs

All gzipped. Split by the two nodes — see `../environment.md` for why there are
two.

**Watch out:** the engine logs contain a ~20 KB single-line `server_args` dump.
Always pipe through `grep` / `cut` / `head`; `cat`-ing one into a terminal or an
agent context is a mistake.

## `n0133/` — the node the packaged numbers came from

| file | what |
|---|---|
| `worker_glm53_mix.log.gz` | **the primary evidence.** The infera MIX worker log for the run that produced `results/fixlen_*.csv`: 8 AITER mHC lines, 0 `Shared experts fusion optimization enabled.`, 4818 decode lines carrying both `full token usage` and `mamba usage`, 0 faults, 0 tracebacks. |
| `router.log` | the kv-aware router (uncompressed, 1.2 KB) |
| `mix_up_graphs.log.gz` | the bring-up transcript — `worker serving after 650s`, `router healthy` |
| `fixlen_p50.log.gz` | the `sglang.bench_serving` output behind `results/fixlen_p50.csv` |
| `fixlen_p90.log.gz` | ditto for `results/fixlen_p90.csv` |
| `build_glm53.log.gz` | the `Dockerfile.sglang.glm53` build (`writing image sha256:fde2855…`) |
| `build_sglang.log.gz` | the `Dockerfile.sglang` (v0.5.18 base bump) build |
| `pull_base.log.gz` | base image pull |

## `n0433/` — the ladder, and the failure

| file | what |
|---|---|
| `rung0.log.gz` | **the failing run.** Vendor image, vendor flags verbatim, no `--disable-shared-experts-fusion`. Contains `Shared experts fusion optimization enabled.` (×1) and the `256 must match 512` traceback with the full `_load_w2` call path. |
| `rung0_wtrace.log.gz` | the instrumented re-run that produced the `[WTRACE] … eid=288 param=(289, 4096, 256) loaded=(4096, 2048)` line — the tensor-level proof. 1.1 MB raw; grep it, do not read it. |
| `rung0_nosef.log.gz` | rung 0 **with** the flag — the first pass |
| `rung1_nosef.log.gz`, `rung2_nosef.log.gz` | rungs 1 and 2 with the flag; both pass |
| `rung0_auto.log.gz`, `rung0_triton.log.gz`, `rung0_instr.log.gz` | one-axis-at-a-time variants tried before the cause was found: `--quantization` omitted (auto-detect), `--moe-runner-backend triton`, and an early instrumented run. All three log `Shared experts fusion optimization enabled.` and then fail with the **identical** `256 must match 512` — which is the evidence that none of those axes is the cause. |
| `rung0_qmx.log.gz` | the one variant that fails **differently**: `--quantization quark_mxfp4` → `NotImplementedError: Requantization into quark_mxfp4 is not supported, from the original quant_method=quark and activation_scheme=None`. It never reaches the fusion gate (`Shared experts fusion optimization enabled.` count 0), so it says nothing about the bug — only that `quark_mxfp4` is the wrong value for this checkpoint. Use `--quantization quark`. |
| `mix_infera_pass.log.gz` | rungs 3+4: the full infera MIX stack passing on this node |
| `router_infera_pass.log.gz` | its router |
| `mix_up_r1..r4.log.gz`, `mix_up_graphs.log.gz` | bring-up transcripts for the earlier rounds |
| `build_glm53.log.gz`, `build_sglang_v0518.log.gz` | this node's image builds (`sha256:6ecbeec…`) |
