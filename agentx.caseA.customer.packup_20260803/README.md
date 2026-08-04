# AgentX Case-A (customer bench) against the infera GLM-5.2 PD deployment

**Ran:** 2026-08-03 11:59 – 12:33 UTC (single session)
**Status:** **PASS** — the bench ran unmodified and produced a first-round result.

The customer's agentic benchmark from
[ROCm/MAD PR #173](https://github.com/ROCm/MAD/pull/173) (`scripts/AgentX_CaseA/`),
replayed **unmodified** against the same live deployment that produced
`../par8.glm52.dpaoff.packup_20260803`.

## Goal and success criteria

**Spec:** [`spec/external.agentic.bench.md`](spec/external.agentic.bench.md).
Its stated goal, verbatim:

> 在 yihou.dev.glm52.merged.experiment 分支上,结合 infera 自身的部署经验。成功运行
> https://github.com/ROCm/MAD/pull/173/changes 中的 glm5.2 的 agentic bench,
> 拿到结果并分析报告。
> 1. vultr 好像有我们起好的服务,在我们已有的服务上直接运行 bench, 拿到首轮结果。

**The spec sets no numeric performance bar** — it is a "get it running and
analyse it" task, so the criteria below are its four explicit deliverables, not
invented thresholds. Performance is judged separately against par8's bars (see
`analysis/README.md` § Verdict).

| # | criterion (from the spec) | actual | verdict |
|---|---|---|---|
| 1 | successfully run the PR's GLM-5.2 agentic bench | 554 requests over 2 concurrency points, **0 errors** | **PASS** |
| 2 | run it on the **already-running** service | deployment untouched — no restart, no reconfig; cmdlines in `env/node_snapshots.txt` match par8 byte-for-byte | **PASS** |
| 3 | get a first-round result | c8 + c16 ladders, machine-readable in `results/` | **PASS** |
| 4 | analyse; flag what's worth learning from the customer's deployment method | `analysis/` (4 files), recommendations in `customer_method_review.md` | **PASS** |

Operator-set scope for this round: **concurrency 8 and 16 only**, aiperf in a
container, **zero modification** to customer code, deployment left as-is.
All four honoured.

## The result in one table

| | AgentX c8 | AgentX c16 | par8 (our own bench) |
|---|---|---|---|
| driver | aiperf, **open-loop** frozen trace | same | closed-loop session driver |
| measured requests | 231 | 323 | 2,671 |
| window | 901 s | 922 s | 3,600 s |
| **errors** | **0** | **0** | 15 (0.52 %) |
| in-flight max / mean | 8 / 5.1 | 16 / 11.1 | 22 / 12.1 |
| **TTFT p50 / p90** | 5,146 / 19,780 ms | 14,394 / 30,567 ms | 1,365 / 4,903 ms |
| **ITL / TPOT p50** | **13.8 ms** | **14.7 ms** | **14.8 ms** |
| E2E p50 | 12.6 s | 21.6 s | 7.4 s |
| **cache hit p50 (server-reported)** | **88.1 %** | **88.1 %** | 88.9 % |

**Two numbers agree across independent benches: ITL ≈ 14 ms and cache ≈ 88 %.**
**One does not: TTFT.** The TTFT gap is a load-model artifact, not a regression —
see the warning below and `analysis/README.md`.

## ⚠ Do not quote the TTFT comparison without this

The two benches ran against **the same server processes in the same hour**.
Nothing about the deployment changed. The TTFT difference comes from how the two
drivers apply load:

- **par8 is closed-loop** — in-flight 12.1 is *emergent*; a slower server gets
  relief because sessions issue fewer turns.
- **AgentX is open-loop with pinned lanes** — in-flight sat at **exactly 8** and
  **exactly 16**; a slower server gets no relief.

Measured proof it is queueing: at c16, TTFT in the 0–50K input bucket is already
**11,949 ms**, and the whole 0–300K span spreads only **2.3×** (at c8 it spreads
**10.0×**, the prefill-shaped curve par8 also measured). `http_req_sending` is
0.2 ms — the wait is entirely server-side.

**At c8 the offered load is *lower* than par8's (5.1 vs 12.1 in-flight) and TTFT
is still 3.8× worse. That residual is NOT explained by this data.** Four
candidates are listed in `analysis/vs_infera_bench.md`; none is confirmed.

## What this run establishes

1. **The customer's bench runs on our stack with zero code modification** — env
   vars only. `md5sum replay_caseA.sh` unchanged in `logs/run_caseA.log`.
2. **554 requests, 0 errors, 0 cancellations, 0 context overflows.**
3. **Our prefix cache independently confirmed at 88.1 %**, measured from the
   *server's* `usage_prompt_cache_read_tokens` through a third-party driver on a
   third-party trace.
4. **Decode is not the bottleneck** — ITL p50 moves +6 % when concurrency
   doubles while TTFT triples. Same conclusion par8 reached.
5. **The prefix cache is worth 2.5× on TTFT** — first-turn 8,981 ms vs
   cached-turn 3,568 ms at matched input size (c8). par8 could not measure this;
   its driver does not tag turn index.

## Two defects found in the customer's kit

- **Its README misstates its own cache metric.** It claims the reported figure is
  "the endpoint's realized server-side prefix hit"; the code
  (`theoretical_prefix_cache.py:22-30`) computes it from the *trace file's*
  `hash_ids` and never asks the server. The number is invariant to the
  deployment under test.
- **`replay_caseA.sh` loses results when `OUT` is set outside `$HERE`** — the
  container mounts only `$HERE`, so artifacts are written inside the container
  namespace and destroyed on exit. The sweep then prints `FAILED` for a run that
  actually succeeded. We hit this and recovered via `scripts/rescue_artifacts.sh`
  rather than modify their script.

Both are worth reporting upstream. Detail in `analysis/customer_method_review.md`.

## Navigate

| file | what |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | exact ordered steps, copy-pasteable |
| [`environment.md`](environment.md) | hardware, images, digests, the exact engine cmdlines |
| [`notes.md`](notes.md) | traps, wrong turns, what this run could not answer |
| [`analysis/README.md`](analysis/README.md) | headline, the confound, the verdict |
| [`analysis/sli_percentiles.md`](analysis/sli_percentiles.md) | full ladders, TTFT-by-size, first-turn vs cached |
| [`analysis/vs_infera_bench.md`](analysis/vs_infera_bench.md) | the two benches side by side |
| [`analysis/customer_method_review.md`](analysis/customer_method_review.md) | method + deployment-recipe review, recommendations |
| `results/c{8,16}/profile_export.jsonl.gz` | **the primary artifact** — one JSON per request |
| `scripts/` | our wrappers + `Dockerfile.aiperf` (the customer's script is in `spec/`, unmodified) |
| [`patches/README.md`](patches/README.md) | **N/A** — nothing was patched; records why, and the two adaptations done outside their code |
| `spec/` | the customer's kit verbatim + the task brief |
| `env/node_snapshots.txt` | both legs' full resolved engine cmdlines, captured with them still live |
| `logs/` | aiperf logs (gzipped) + the driver log |

## Deployment under test (unchanged from par8)

```
router   http://10.2.122.78:8100     served-model-name glm5.2-mxfp4
prefill  chi2835  TP8, DPA off, chunk 16384, ctx 262144, kvd + hicache
decode   chi2879  TP8, DPA 8, MTP (EAGLE, 4 draft), ctx 262144
```
