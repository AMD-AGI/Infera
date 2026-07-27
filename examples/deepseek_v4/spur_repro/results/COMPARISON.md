# 对拍结果 — spur (本次) vs legacy (chi2865, 2026-07-07)

两者:同镜像 mori-0615、同 R4 no-DP recipe、同 workload(ISL/OSL=8192/1024, random-range 1.0, num-prompts=10*conc)。

## conc=2 (no-DP)

| metric | spur | legacy | ratio |
|--------|------|--------|-------|
| completed | 20.00 | 20.00 | 100.0% |
| output_throughput | 136.87 | 137.05 | 99.9% |
| total_throughput | 1231.87 | 1233.47 | 99.9% |
| median_ttft_ms | 628.71 | 621.19 | 101.2% |
| median_tpot_ms | 13.96 | 13.96 | 100.0% |
| median_itl_ms | 13.78 | 13.72 | 100.5% |
| **tot/GPU** | **154.0** | **154.2** | — |

conc=2 全部指标差异 <1%,复现与 legacy 完全一致。

## conc=32 (no-DP)

| metric | spur | legacy | ratio |
|--------|------|--------|-------|
| completed | 320.00 | 320.00 | 100.0% |
| output_throughput | 1046.56 | 1048.83 | 99.8% |
| total_throughput | 9419.01 | 9439.50 | 99.8% |
| median_ttft_ms | 5401.70 | 5301.69 | 101.9% |
| median_tpot_ms | 25.27 | 25.31 | 99.8% |
| median_itl_ms | 20.13 | 20.13 | 100.0% |
| **tot/GPU** | **1177.4** | **1179.9** | — |

conc=32 差异同样 <1%(tot/GPU 1177.4 vs 1179.9)。**两个并发点对拍全部通过。**
