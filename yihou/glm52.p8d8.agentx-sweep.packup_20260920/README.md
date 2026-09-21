# P8D8 AgentX concurrency sweep — GLM-5.2 MXFP4 1P1D

Five complete 3,600 s AgentX points at concurrency 80 / 112 / 144 / 192 / 256,
all measured against **one** deployment. GLM-5.2 MXFP4, P8D8 (TP8/DP8 on both
legs), prefill on `crsuse2-m2m-137`, decode on `crsuse2-m2m-136`, MI355X x8 each.
MTP EAGLE (5 steps / 6 draft / topk 1) on decode only, with **simulated
acceptance `DECODE_SIMULATE_ACC_LEN=3.61`**. Deployment up 2026-09-19 12:14:08
UTC, driver finished 2026-09-20 01:47:15 UTC.

| conc | tok/s/chip | TTFT p50 | TTFT p90 | ITL p50 | intvty p50 | profiled | errors | rails |
|---|---|---|---|---|---|---|---|---|
| 80  | 19,513 | 5.69 s   | 22.1 s  | 14.89 ms | 67.2 | 9,483 | 3   | 0/0 |
| 112 | 19,014 | 11.30 s  | 64.5 s  | 15.92 ms | 62.8 | 9,536 | 5   | 0/0 |
| 144 | 12,555 | 52.45 s  | 209.3 s | 14.40 ms | 69.4 | 6,862 | 32  | 0/0 |
| 192 | 7,409  | 132.82 s | 443.8 s | 11.23 ms | 89.0 | 4,339 | 185 | 0/0 |
| 256 | 5,329  | 159.51 s | 510.1 s | 10.18 ms | 98.2 | 3,659 | 487 | 0/0 |

`tok/s/chip` is per-GPU throughput. `profiled` is completed (profiled) requests.
`errors` = `records_error_dropped`, all `InvalidInferenceResultError`. `rails` =
`transport retry counter exceeded` + `wqe is not posted`, counted before/after
each point. Full running record: `results/sweep_results.yihou.md`.

## The saturation cliff, and where the peak really is

Throughput is flat from 80 to 112 (19,513 → 19,014, **-2.6 %**) and then halves
twice: 112 → 144 loses **34 %**, 144 → 192 loses **41 %**. The knee is between
112 and 144 — closer to a cliff than a curve. Adding concurrency past 112 buys no
throughput and multiplies tail latency; extra requests queue rather than execute.

**Peak throughput is at or below concurrency 80, the sweep's lowest point.** 80
and 112 are within 2.6 % of each other and everything above collapses, so the
maximum is **outside the measured range**. Locating it would need points below 80
(e.g. 40, 56). This is not the `max_running_requests` ceiling: that was raised to
256 precisely to rule the scheduler cap out, and 112 is well inside it.

## ITL and interactivity invert as the system collapses

The trap in the table: ITL and interactivity IMPROVE monotonically toward the
worst point. ITL p50 runs 14.89 → 15.92 → 14.40 → 11.23 → **10.18 ms**;
interactivity 67.2 → 62.8 → 69.4 → 89.0 → **98.2**. The best readings of both
belong to concurrency 256, the collapsed point.

Mechanism: in PD disaggregation the queue forms on the prefill / KV-transfer
side, so decode is progressively starved. The few requests whose KV arrives
decode against an almost empty decode batch and therefore enjoy excellent
inter-token latency. **These metrics measure the survivors, not the system.**
Throughput per chip, TTFT p90 and the completed-request count are the honest
signals here; ITL and interactivity actively mislead. Do not use them as a
saturation signal in a PD deployment.

## Stability: 13 h 33 m fault-free

Deployment up 2026-09-19 12:14:08, driver finished 2026-09-20 01:47:15 — **13 h
33 m**, with **zero `Memory access fault`, zero `Fatal Python error`, zero RDMA
rail faults** across all five points. This ran with
`index_share_for_mtp_iteration=false` (the issue.md 3.3 mitigation) and custom
all-reduce on. For contrast, the same shape with **IndexShare on** faulted at
**1 h 26 m** — a clean single-variable fault contrast. The c080 throughput delta
between the two runs is 0.3 % (19,575 → 19,513), inside run-to-run noise; but
those runs differ in *two* settings at once (run 1: IndexShare on + fused-topk
off; run 2: IndexShare off + fused-topk on), so this figure **bounds** rather
than **isolates** the mitigation's throughput cost.

The end state at the top of the sweep is **queueing collapse, not out of
memory**. The mission anticipated OOM at 256; it never came. All five points
completed their full 3,600 s window; what degrades is service — at 256 TTFT p90
is 8.5 minutes, errors reach 487, and completed requests fall to 3,659 (38 % of
what concurrency 112 achieved).

## SCOPE LIMITS — read before quoting any number

1. **Correctness is waived by construction.** Simulated acceptance forces the
   accept *count* (~3.61 draft tokens per verify step), not which tokens are
   right, so the deployment emits garbled text. The reference c32/c40 kit states
   the same waiver. The acceptance gauge reading ~3.6 carries **no** correctness
   information — it reports the value it was told to report. These are timing
   numbers under forced acceptance, comparable to that reference kit, not to a
   correctness run.
2. **One non-reference setting is carried:** `index_share_for_mtp_iteration=false`,
   needed to avoid the issue.md 3.3 memory fault. The reference does not carry it.
3. **Peak throughput is outside the measured range** — at or below concurrency 80.
   Do not read 80 as the maximum; read it as the lowest point still on the flat.
4. **No OOM end state was reached.** The real limit is queueing collapse.
5. **A withdrawn workaround and a falsified hypothesis.** An earlier deployment
   used `SGLANG_DSA_FUSE_TOPK=0` to stop the §3.3 crash; it was replaced by the
   IndexShare setting above. The prediction — written in advance — that
   `SGLANG_DSA_FUSE_TOPK=0` caused the garbling was tested and **disproved**:
   restoring the fused path left the output garbled. Simulated acceptance is the
   only variable aligned with coherent-vs-garbled across all deployments.

## Navigation

| path | what is there |
|---|---|
| `results/sweep_results.yihou.md` | THE numbers and the full running narrative, including the superseded run 1 |
| `results/results.csv`, `results/c080..c256/` | per-point `agentx_conc*.json` and `rails-*.txt` |
| `analysis/config_design.yihou.md` | how the P8D8 config was built |
| `analysis/max_running_cap.yihou.md` | why `max_running` was raised to 256 |
| `analysis/accept_length_is_input_driven.yihou.md`, `accept_length_config_diff.yihou.md` | acceptance-length behaviour and config deltas |
| `analysis/hardware_prep.yihou.md`, `image_transfer.yihou.md` | node prep and image staging |
| `spec/mission.md`, `PLAN.md`, `poll_log.md` | task of record, plan of record, and every wrong turn |
| `scripts/`, `scripts/bench-harness/` | launch/config/driver scripts and the vendored harness (`SHA256SUMS.txt`) |
| `logs/` | driver / orchestrator logs, `engine-argv-key-flags.txt`, gzipped server-log excerpts |
| `env/`, `patches/` | **empty placeholders — carry no artifacts.** The environment record is the image paragraph below plus `logs/engine-argv-key-flags.txt` and `logs/launch-summary.txt`. The applied patch diffs are **not** vendored here: sglang PR #37152 lives at the repo root as `pr37152.diff` (outside this kit), and the NextN shared-experts fusion fix is baked into the image only. |

Raw server logs (110 MB prefill / 90 MB decode) are **not** in the kit; they stay
on the nodes at
`/mnt/m2m_nobackup/yihou_p8p4/launch/sweep-20260919T113947Z/server-logs/`. Only
gzipped grep excerpts are packed under `logs/`.

Image `infera-sglang:v0519-yihou-0917-nextnfix-hicache`
(`sha256:fd7220a57b7d…`), base `lmsysorg/sglang-rocm:v0.5.19-rocm720-mi35x-20260917`,
in-image sglang `0.5.19.dev20260917+ga9fb1c3238`. Contains the NextN
shared-experts fusion fix and sglang PR #37152 (OPEN upstream, never merged, its
AMD ROCm CI is red).
