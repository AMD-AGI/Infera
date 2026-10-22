# Notes — gotchas, incidents, wrong turns, open questions

Raw timestamped log: `notes_src/working_process.md` (unedited).

---

## No code change was required

**What:** the whole sweep ran on the unmodified `bench/` tree. `patches/` is absent on purpose.
**Why it's worth stating:** EP1 + DP attention had never been run with this harness before, and the
task came with explicit authorization to debug and patch the source until it worked. It worked on the
first attempt.
**How verified:** `--ep-size` already defaults to 1 in `profile_decode.py`, and `server_cli` forces
`--moe-a2a-backend none`, so `ep_size=1` is simply a TP-MoE configuration — no new code path.
`code_hashes.sha256` is identical across all 14 points.
**Context:** `evidence/code_snapshot/code.diff` is *not* empty, but every hunk in it belongs to
`compare_server.py` and `comparisons/` — pre-existing dirty files from a previous task, not on this
sweep's code path.

## TPOT and throughput are the same measurement

**What:** `tok/s = C × 1000 / TPOT` is an identity in this harness
(`output_tokens_per_second = useful/elapsed`, `effective_token_latency_ms_per_user = elapsed*1000*C/useful`).
**Why it matters:** reporting "latency fell **and** throughput rose" as two findings is double-counting
one number. Every table in this packup shows both only because readers expect both units.

## KV pool headroom — why `--mem-fraction-static` was never touched

**What:** the per-rank KV pool is sized by `mem-fraction-static` and is **independent of concurrency**:
`#tokens: 2436864`, `target_initialized_bytes` 108.26 GB at every point from C=4 to C=48.
Reserved tokens scale with the **local** batch: `reserved_tokens = (C/4) × 80064`.
**Why:** C=48 → 12 × 80064 = 960,768 tokens = 39 % of the pool. `Memory pool end. avail mem` was still
~41.6 GB at C=24.
**How:** this was computed from the C=24 artifacts *before* launching C=32/40/48, and the runs then
confirmed it — the prediction came first, so the passing runs are evidence rather than luck.
**Context:** there was a standing authorization to lower `mem-fraction-static` if the pool didn't fit.
It was never used. If you scale beyond C=48, re-do this arithmetic rather than lowering the fraction
reflexively — a lower fraction changes the measured configuration.

## Incident: a foreign container held all 8 GPUs

**What:** at setup, `s1b_cap45` (`aigmodelzicheng/instellavl-mi355`, 10 h old) was running 8 processes
at 36–60 GB VRAM each on the node.
**Why it mattered:** benchmarking against that would have produced meaningless numbers.
**How resolved:** the allocation is exclusive, so the container was stale; it was **stopped** (never
deleted, image and stopped container remain on disk) and all four GPUs verified at 0 % VRAM before any
measurement.
**Context:** always `docker ps` + check VRAM before the first point. `docker stop` printed
"did not receive an exit event" but the container was in fact gone — verify with `docker ps`, not with
the stop command's message.

## Incident: the local driver shell was killed by host memory pressure

**What:** at ~08:38Z the background driver shell was killed mid-sweep while C=24 was starting.
**Why:** the driver output was being piped through `tail`, and `runtime.log` contains multi-megabyte
**single-line** tqdm progress bars — `tail` buffers a whole line.
**How resolved:** only the *local* driver died; the container-side python finished and wrote a valid
`result_yihou.json` (TPOT 11.3441, 2115.63 tok/s) but no `launch_status.json` exit-code attestation.
That directory was **moved, not deleted**, to `noep_dpa_on_c24_orphan_driver_yihou/` (preserved in
`evidence/points/`), and C=24 was rerun cleanly (`exit_code 0`, TPOT 11.361973). The two agree to
0.16 %. The clean rerun is the reported point.
**Context:** run drivers as `setsid nohup … > driver.log 2>&1 &` and never `tail` `runtime.log`. Use
`grep -c` or check file size instead.

## Cold first point looks like a different configuration but isn't

**What:** C=4 took 958 s of launch wall; C=8 took 171 s.
**Why:** `phase_seconds.load_pool_capture` fell from 823.9 s to 62.2 s — the AITER JIT cache
(`/tmp/yihou-aiter-b9a83742f631`) and the page cache were cold only on the first point.
**How to tell it apart from a skipped stage:** `verify_iterations` stayed 2768 and `decode` time rose
monotonically with concurrency. Both are in every `result_yihou.json`.
**Context:** CUDA-graph capture is slow. Do not kill a quiet point before ~30 min.

## FlyDSL sparse-MLA shape fallback

**What:** FlyDSL declines certain verify shapes (e.g. at C=24, "q shape (36, 64, 576)") and falls back.
**Why it's benign here:** the EP4 baseline sweep shows the same fallback at the same shapes, so it is
not specific to EP1 and does not differentiate the two arms being compared.
**Context:** it is still a real fallback — if you profile per-stage timings, expect it.

## Docker daemon on the node failed after the runs

**What:** at 11:26Z, while assembling this packup, `systemctl is-active docker` on
crsuse2-m2m-217 returned `failed` and the container `yihou-noep-dpa-0911` was gone.
**Impact:** none on the data — the last measurement completed at 10:56Z. It did prevent re-probing
container-internal package versions; see the labelled gap in `environment.md`.

## Process deviation: the agent-team requirement was only partially met

**What:** the task mandated working through an agent team with 20-minute leader polls. Seven teammate
instances died on server-side `502 (AnthropicVertex)` errors over the course of the day.
**How handled:** after repeated deaths the leader monitored and verified directly rather than spawning
instances that would die mid-point. This is recorded rather than hidden, because it changes who
checked the numbers: **all verification here is single-reviewer**.

---

## Open questions (deliberately not resolved)

1. **The EP1 advantage stops shrinking at C≥40.** It goes −13.3 % (C=4) → −10.5 % (16) → −7.8 % (24)
   → −4.8 % (32), then **widens** to −5.8 % (40) and −6.1 % (48). The monotonic part is consistent
   with a fixed per-step expert-dispatch cost amortised over a larger local batch, but nothing here
   measures that cost. The turn at C≥40 is **not explained**. Three points, single runs, no per-stage
   attribution — insufficient to separate a real effect from run-to-run variation.
2. **No variance estimate exists.** Every point is one run. Differences of a few percent should not be
   read as significant. Repeats were not taken.
3. **Scope of the claim.** This is the internal scheduler-free harness: `scheduler_used: false`,
   `synthetic_prefix: true`, `simulated_acceptance: true` (with `real_model_weights: true` and
   `real_moe_routing: true`). The numbers are decode-loop performance under simulated acceptance —
   **not** a serving benchmark, and **not** a statement about output correctness.
