> Redacted historical snapshot. Measurements and correctness statements refer to the original run, not these published bytes. Revalidate the published source and adapters on the target stack before reuse.

# Validation — moe_mxfp4 task (pristine kernel), 2026-10-20

Node compute-node-01, container `historical-container` (image v0.5.21-rocm720-mi35x-20261011,
sglang 11bfd868, aiter e7d2453), `HIP_VISIBLE_DEVICES=1`. Command form:

    ssh $NODE "docker exec -e HIP_VISIBLE_DEVICES=1 historical-container bash -c \
        'cd REDACTED_LOCAL_PATH && python3 driver.py [--bench-mode|--profile-run]'"

Raw stdout/stderr: `_val/{correct,bench1,bench2,profile}.{stdout,stderr}`.

## 1. `python3 driver.py` (02:18:42Z, 9.7 s wall)

```
case_snr: T48_bs8_x0 46.30
case_snr: T48_bs8_x1 46.29
case_snr: T60_bs10_x0 46.29
case_snr: T60_bs10_x1 46.29
case_snr: T72_bs12_x0 50.63
case_snr: T72_bs12_x1 50.61
case_snr: T84_bs14_x0 50.55
case_snr: T84_bs14_x1 50.71
SNR: 46.29 dB
allclose: True
```
Pristine passes the 30 dB gate (min 46.29 dB).

## 2. `python3 driver.py --bench-mode` ×2 (02:19:02Z and 02:19:09Z; all cases `mode: cudagraph`)

| case | run 1 case_ms | run 2 case_ms | Δ | run 1 wall_ms min / max | run 2 wall_ms min / max |
|---|---|---|---|---|---|
| T48_bs8 (unscored) | 0.236331 | 0.235872 | −0.19 % | 0.2345 / 0.2448 | 0.2312 / 0.2429 |
| T60_bs10 | 0.234012 | 0.233202 | −0.35 % | 0.2324 / 0.2378 | 0.2314 / 0.2349 |
| T72_bs12 | 0.236171 | 0.234782 | −0.59 % | 0.2343 / 0.2389 | 0.2333 / 0.2359 |
| T84_bs14 (unscored) | 0.236551 | 0.236462 | −0.04 % | 0.2346 / 0.2390 | 0.2354 / 0.2377 |
| **median_ms** | **0.235286** | **0.234251** | −0.44 % | | |

30 samples per case, 4 weight sets per replay (per-call = replay / 4). Additional observation from a
separate process (`_src/breakdown.log`, same GPU, a few minutes later): graph bs10 233.8 µs,
bs12 229.9 µs for the identical configuration, i.e. up to ~2.7 % below the driver runs.

## 3. `python3 driver.py --profile-run` (02:20:20Z)

Exit 0, 6.0 s wall, no stdout. Runs weight-set setup, then the bs12 op 6×.

## 4. Supporting measurements (scripts and logs in `_src/`)

- `probe_path.py` / `probe_bs12.log`: `get_2stage_cfgs` metadata, sorting path (`_moe_sorting_impl`,
  opus) and the 6 kernel names of the production call.
- `merge_fmoe_cfg.py`: fresh aiter merge of `tuned_fmoe.csv` = image `REDACTED_LOCAL_PATH`
  as row sets (2870 rows each, 0 rows differ; byte order differs).
- `probe_quant.py`, `probe_quant2.py`: quantizer rules (HIP activation quant = ceil(log2(amax/6)) + RNE,
  100 % scale match, 0 element mismatches; triton weight quant = Quark even + RNE, 0 mismatches).
- `oracle_variants.py` / `oracle_variants.log`: end-to-end SNR per oracle convention (table in program.md).
- `breakdown.py` / `breakdown.log`: per-kernel times, warm vs cold, dispatch policy 2 vs 0.
