# GLM-5.2 1P1D same-rail AgentX — C32 and C40 — 2026-09-18

Two AgentX benchmarks of GLM-5.2 MXFP4 in a 1-Prefill / 1-Decode disaggregated
deployment, with KV transfer constrained to the **same RDMA rail** so the
cross-DP-rank Mooncake failure of `bench/glm5p2_pd/issue.md` §3.1 cannot occur.

Run on **crsuse2-m2m-135** (prefill) and **crsuse2-m2m-138** (decode),
MI355X ×4 each (devices 2,3,4,5), 2026-09-18 03:40–08:15 UTC.

| run | config base | concurrency | duration | when (UTC) |
|---|---|---|---|---|
| **C32** | `config.sh` (validated baseline) | 32 | 1200 s | 05:54–06:26 |
| **C40** | `config.full.sh` **minus 2 unported items** | 40 | 3600 s | 06:51–08:11 |

## Headline results

| metric | C32 | C40 |
|---|---|---|
| token throughput / chip | 12,448 tok/s/chip | **20,711** tok/s/chip |
| output throughput / chip | 89.6 tok/s | 149.1 tok/s |
| P90 interactivity | 74.2 tok/s/user | 63.3 tok/s/user |
| median ITL | 11.15 ms | 12.38 ms |
| P90 ITL | 13.48 ms | 15.79 ms |
| median TTFT | 5.57 s | **3.97 s** |
| requests profiled | 1150 | 4128 |
| AgentX error rate | 0/1150 | 0/4128 |
| **router affinity 503s** | **0** | **0** |
| **cross-rail Mooncake failures** | **0** | **0** |

Median ITL for both sits inside the 9.9–11.8 ms band `issue.md` §3.2 records for
a good image (C40 marginally above), i.e. neither is the 43–47 ms regressed
state.

### These two points are NOT a concurrency sweep

Six things changed between them, not one. Do not read C32→C40 as the effect of
concurrency, and do not attribute the TTFT improvement to any single knob:

| | C32 | C40 |
|---|---|---|
| concurrency | 32 | 40 |
| Prefill HiCache | off | **on** (ratio 1.5) |
| max-running / graph max BS | 64 | **128** |
| JIT grouped-topk | off | **on** |
| IndexShare workaround | kept | **removed** |
| warmup requests / lane | 1 | **10** |
| profiling duration | 1200 s | **3600 s** |

They share a plot (`results/pareto.png`) because they share a topology, not
because they form a controlled series.

## The finding that mattered

The stated goal was same-rail KV transfer. Getting there turned up a cause that
was **not** what the initial evidence suggested.

Preflight first reported per-GPU VRAM transfers failing on GPUs 4-7 in *both*
directions while 0-3 passed cleanly — a symmetric, cleanly-grouped failure that
reads exactly like dead hardware. It is not. **Mooncake's auto-discovery lets
the two ends independently pick different HCAs from the NUMA-local pool, and
these rails are physically isolated (`ionic_i` reaches only `ionic_i`), so a
mismatch is *unreachable* rather than merely slow.**

| NIC selection | per-GPU VRAM result, 135 ↔ 138 |
|---|---|
| auto (shared device list) | GPU 4-7 `transfer_failed`, both directions |
| pinned `ionic_2` / `_3` / `_4` / `_5` | **16/16 verified each**, both directions |

The fix uses a format sglang already supports — a per-GPU JSON map for
`--disaggregation-ib-device` — and required **no repository code change**. See
`notes.md` §1.

Two knobs are needed and neither is sufficient alone: the **router** one makes
Prefill rank *i* hand off to Decode rank *i*; the **Mooncake** one makes rank
*i* then transfer over the rail that rank *i*'s GPU owns. Both held across both
runs — zero 503s and zero cross-rail failures in each.

## Scope limits — read before quoting the numbers

**1. These are timing measurements under simulated MTP acceptance, not
correctness evidence — and the configuration they were taken in is now known to
decode incorrectly.** Both runs used `DECODE_SIMULATE_ACC_LEN=3.61`, which
*forces* the acceptance length. With simulation off, this P4D4 shape produces
**garbled decode output** (first token correct, all later tokens degenerate;
measured real accept rate 0.05). Correctness was explicitly waived by the user.
Valid for comparison against other runs using the same simulation — the bench's
own alignment point — and nothing more.

> **Root-caused after this packup was written.** The cause is **custom
> all-reduce on the decode leg**, which corrupts the speculative path;
> `--disable-custom-all-reduce` fixes it. Every number in this packup was taken
> with custom all-reduce **on**, so none of them describes the corrected
> configuration, and the fix's throughput cost is unmeasured. Evidence and
> reproduction: `yihou/glm52-mtp-garbled-decode-rootcause.packup_20260918/`.
> See `notes.md` §3.

**2. C40 is "full minus two items", not full.** `config.full.sh` requests
`flydsl` DSA backends and an `aiter` fused top-k that **the pinned nightly does
not accept** — verified first-hand against the image's own argparse choices.
Substituted with `tilelang` and the default `sgl-kernel`. See `notes.md` §7.

**3. C40 dropped 3 real errors.** `records_error_dropped: 3`,
`InvalidInferenceResultError`. AgentX's headline 0/4128 counts completed
requests, but these three are real. C40 is also the run that **removed** the
IndexShare workaround, which exists to suppress garbled output — so a link is
plausible. **Not established**; recorded, not concluded. See `notes.md` §8.

**4. P4D4 is not "half of P8D8."** TP4 and TP8 shard differently and the
InferenceX reference curve is grouped by full P/D shape, so any comparison
against the existing TP8/DP8 baseline is indicative only.

## Navigation

| path | what |
|---|---|
| `REPRODUCE.md` | ordered, copy-pasteable reproduction of both runs |
| `environment.md` | hardware, fabric, image digests, git SHA |
| `notes.md` | gotchas, wrong turns, error analysis — **the most re-read file** |
| `scripts/` | both configs, topology, and the pinned-NIC probe, verbatim |
| `patches/` | the two patches both runs need, with what/why/how/context |
| `results/` | `results.csv` + `pareto.png` (both points), then `c32/` and `c40/` |
| `results/preflight/` | the same-rail evidence: auto vs pinned |
| `spec/` | mission, debug log, poll log, GPU-clearing notice |
| `env/` | raw `collect_env.sh` output per node |
| `logs/` | both benchmark rounds' worker logs + runners (gzipped) |

## Provenance

- Repo `AMD-AGI/Infera`, branch `dev/pd_opt/glm_5.2_agentx`, HEAD
  `5a342acffe10e09729662ff40e81b22a4367fd75`, **plus the two patches in
  `patches/`** (uncommitted at pack-up time).
- Image `infera-sglang:v0519-yihou-0917`, id
  `sha256:4190c3a99d0ea8b195580008e7f37fb2f1dfdbe6254b019884d4bef5721041cd`.
- Base `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`, digest
  `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32`.
- In-image sglang `0.5.19.dev20260917+ga9fb1c3238`; aiter source
  `4ad998328` (`v0.1.21.dev0-48-g4ad998328-dirty`).

Original working directory, untouched and still on disk:
`bench/glm5p2_pd/results/yihou-1p1d-c64/` (gitignored). Raw AgentX output stays
on crsuse2-m2m-135 under `/mnt/m2m_nobackup/yihou/agentx-results/`:
`c32-1p1d-20260918T055136Z` and `c40-full-1p1d-20260918T065059Z`.
