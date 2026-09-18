# Results

`results.csv` and `pareto.png` hold **both** points. Per-run raw artefacts are
under `c32/` and `c40/`.

## Both points

| metric | C32 | C40 |
|---|---|---|
| config base | `config.sh` | `config.full.sh` minus 2 unported items |
| concurrency / duration | 32 / 1200 s | 40 / 3600 s |
| token throughput / chip | 12,448.1 | **20,711.1** tok/s/chip |
| output throughput / chip | 89.59 | 149.14 tok/s |
| input throughput / chip | 12,358.5 | 20,561.9 tok/s |
| P90 interactivity | 74.18 | 63.33 tok/s/user |
| median ITL | 0.01115 s | 0.01238 s |
| P90 ITL | 0.01348 s | 0.01579 s |
| median TTFT | 5.5678 s | **3.97312 s** |
| e2e latency p50 / p90 | 10.01 / 48.16 s | 10.92 / 44.58 s |
| profiled / total | 1150 / 1216 | 4128 / 4572 |
| **error-dropped** | **0** | **3** (`InvalidInferenceResultError`) |

The profiled-vs-total gap is warmup, dropped by design (66 for C32, 444 for C40
— C40 uses 10 warmup requests per lane instead of 1).

**`pareto.png` plots both against the InferenceX public reference, but they are
not a controlled series** — six settings differ between them. See the README's
comparison table before drawing any curve through the two points.

## Per-run directories

`c32/` and `c40/` each hold:

- `agentx_conc<N>.json` — the full AIPerf aggregate (QPS, TTFT/ITL/TPOT/e2el
  distributions, request accounting)
- `runtime.env` — the exact environment the client container ran with
- `benchmark_command.txt` — the exact invocation
- `service/` — the live-service snapshot the adapter captured at benchmark time:
  `workers.json`, `hardware.json`, per-instance server-info and container
  inspect. Use these to confirm each run's shape rather than trusting this file.

## `preflight/` — the same-rail evidence

Shared by both runs (the fabric and pinning did not change between them).
Naming is `<round>--<host>.json`; `sections.mooncake` is the interesting part.

| round | what it shows |
|---|---|
| `run1--*` | **stock** shared device list, NIC chosen by auto-discovery. GPUs 0-3 pass at 45.1-45.9 GB/s; **GPUs 4-7 `transfer_failed`, both directions.** |
| `pin-ionic4-b--*` | both ends pinned to `ionic_4`. **All 8 GPUs pass, both directions**, byte-verified — including the four that had just failed. |
| `pin-ionic_2--*`, `pin-ionic_3--*`, `pin-ionic_5--*` | the other three rails these runs use, each 16/16 verified. |

Read `run1` against `pin-ionic4-b` and the conclusion is immediate: the rails
were never broken; auto-discovery was letting the two ends land on different
NICs. Full reasoning in `../notes.md` §1.

Also visible in every round: `rdma-default` warns *"auto-selected GID is
link-local, not routable"* — the reason `MC_GID_INDEX=1` is mandatory here.
