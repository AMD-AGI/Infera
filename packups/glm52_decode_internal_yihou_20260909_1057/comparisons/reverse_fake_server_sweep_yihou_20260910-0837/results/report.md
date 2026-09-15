# Reverse fake-server sweep comparison

Native server:128 rolling requests,70K/10K,16 short warmups. Internal:one fixed batch.
TPOT delta=(server mean/internal mean-1). Throughput delta=(server/internal-1).

| DPA | C | Status | Internal TPOT ms | Server mean / P50 / P90 ms | TPOT delta | Throughput internal / server | Throughput delta |
|---|---:|---|---:|---|---:|---|---:|
| off | 4 | not_run | 6.150 | — | — | 650.4 / — | — |
| off | 8 | not_run | 7.999 | — | — | 1000.1 / — | — |
| off | 16 | starting | 10.286 | — | — | 1555.5 / — | — |
| off | 20 | not_run | 11.667 | — | — | 1714.2 / — | — |
| off | 24 | not_run | 12.632 | — | — | 1899.9 / — | — |
| on | 4 | not_run | 7.199 | — | — | 555.6 / — | — |
| on | 8 | not_run | 8.870 | — | — | 901.9 / — | — |
| on | 16 | not_run | 10.981 | — | — | 1457.1 / — | — |
| on | 20 | not_run | 11.808 | — | — | 1693.8 / — | — |
| on | 24 | not_run | 12.277 | — | — | 1954.9 / — | — |

## Scope and caveats
- Close timings do not validate tokens,hidden states,real P-to-D transfer,or production acceptance.
- Server TPOT=(latency-TTFT)/(OSL-1); internal TPOT=complete decode loop/OSL.
- Server overlap,rolling refill/tail and recycled fake prefix differ from synchronous internal random prefix.
-128 requests retain non-full tails atC20/C24; effective concurrency is recorded in CSV.
- Hardware node271 differs from internal node056. Cold startup is excluded from native measurements.
- FlyDSL configured backend does not guarantee every phase uses it. DPAon64Qheads and offC20/C24verify rows exceed the pinned FlyDSL gate; fallback remains unchanged.
- Earlier eager/graph differences remain unresolved; user stopped that investigation.
