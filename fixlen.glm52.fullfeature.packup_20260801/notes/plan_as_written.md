# Plan — bench the full-feature GLM-5.2 deployment

Status: `[ ]` todo · `[~]` running · `[x]` done · `[!]` finding/blocked

Workspace: this folder. Nothing is written outside it except `CLAUDE.md` (already done)
and the two packup folders at the repo root when Phases 1 and 2 finish.

---

## Phase 0 — bring-up and feature proof

Nothing measured runs until this passes. The deployment under test has **never** run in
this configuration: `ctx=262144` **and** MTP on, together, on `merged-e`, behind the
**Rust** router. Prior MTP validation was at `ctx=32768`. This is bring-up, not replay.

- [ ] **0.1 Reset both nodes** from `infera/engine-sglang:merged-e` via
      `reset_merged.sh` (`IMAGE=` override). Gate: `PORT_ACTIVE: 8`, `kvd socket OK`,
      etcd up on chi2879. Do not skip the GPU-idle wait.
- [ ] **0.2 Boot the legs** at the frozen sweep config (§Phase 1). prefill MTP=0,
      decode MTP=1. Poll HTTP (`wait_ready.sh`); never grep an appended log.
- [ ] **0.3 Rust router** — `--router-backend rust`, kv-aware, weights 20.0 / 2.0.
      Note `--kvd-socket-path` is **python-only** (`launch_rust.py` builds argv
      explicitly), so `/v1/cache/prewarm` is absent under Rust. Capability delta,
      recorded; the bench does not use prewarm.
- [ ] **0.4 Feature-proof matrix.** Six rows, each a signal that goes red if absent.

| feature | positive signal | source |
|---|---|---|
| PD | `/v1/workers`: prefill **and** decode, both `active` | router |
| DPA | 8 live `sglang::scheduler_DP*` per node | `ps` on node |
| kv-aware | **8 per-rank views, non-uniform block counts** | see 0.5 |
| kvd | `statctl` gets/hits climb, `sets` **flat**, after an engine restart | restart-replay |
| MTP | decode log `accept len` **2.1–2.6** (4.00 = repetition loop, *bad*) | `strings\|grep` |
| RDMA | no `MC_FORCE_TCP`; ionic devices present in the prefill log | `strings\|grep` |

- [ ] **0.5 Read kv-aware per rank, not per worker.** The discriminator is 8 distinct
      per-rank views. Under the **python** router: `GET /v1/admin/cache-view/<w>?dp_rank=N`
      for N=0..7. Under **Rust** that route does not exist (`handlers.rs:33-38` routes
      only `/health`, `/v1/workers`, `/v1/models`, `/metrics`, `/v1/{chat/,}completions`;
      `total_blocks()` is unrouted and `/metrics` emits only `active_workers`+`uptime`),
      so the signal must come from the policy log line — `policy.rs:314`, fields
      `picked=<worker>#dp<N> cache_hits=… request_blocks=…`, `tracing` at info.
      Capture `/tmp/router.log`; the proof is **which ranks get picked and with what
      `cache_hits`**, over real traffic.
      ⚠️ `feedback_control_leg_must_fail` — a live A/B of two router binaries already
      proved nothing once. Read this as a *presence* signal, not as an A/B.
- [ ] **0.6 Correctness, once**: `needle.py` 5/5 at a multi-chunk prompt with official
      sampling (temp 1.0 / top_p 0.95). Mission Rule 4 — do not spend more time here.

**Fallback rule (pre-agreed).** If a failure localises to the Rust router: record it in
`notes/` with evidence, switch to `--router-backend python`, continue. The Rust router
is not the deliverable.

---

## Phase 1 — fixlen sweep (sglang `bench_serving`)

**Paired percentiles, P99 dropped, one server for all 8 rounds.**

| pair | ISL | OSL | conc |
|---|---:|---:|---|
| p50 | 74,000 | 320 | 1 / 32 / 64 / 128 |
| p90 | 155,000 | 3,300 | 1 / 32 / 64 / 128 |

Server config, sized for the **largest** pair, then **frozen for all 8 rounds** — no
per-workload retuning, by instruction:

    --context-length       262144   # 155,000+3,300 needs 158,300; 262144 also covers Case A's 260K clamp
    --chunked-prefill-size  65536   # = 8192/rank at dp8, the measured sweet spot
    --cuda-graph-max-bs       128   # top conc in the sweep
    --max-running-requests   2048
    --mem-fraction-static    0.88 prefill / 0.85 decode
    --enable-cache-report           # else the cache-hit column reads 0
    --hicache-size             16   # absolute GB; never --hicache-ratio
    MTP decode only: EAGLE steps=3 topk=1 draft=4, --disable-custom-all-reduce

Per round:

    python3 -m sglang.bench_serving --backend sglang-oai-chat \
      --base-url http://<router>:8100 --model glm5.2-mxfp4 \
      --dataset-name random --random-input-len <ISL> --random-output-len <OSL> \
      --random-range-ratio 1.0 --max-concurrency <C> --num-prompts <N> \
      --cache-report --temperature 1.0 --top-p 0.95 \
      --output-file results/fixlen_<pair>_c<C>.jsonl --output-details

- [ ] 1.1 p50 pair × {1, 32, 64, 128}
- [ ] 1.2 p90 pair × {1, 32, 64, 128}
- [ ] 1.3 `statctl` + router per-rank picks captured **before and after each round**
- [ ] 1.4 **packup** (`experiment-result-packup`) — mission step 2 requires it here

**Budget, not prediction.** Wall-clock estimates from the spur Case A run (TTFT ~4.5 s
p50 / ~9 s p90, TPOT ~31 ms) are a first guess only — that run had MTP **off** on a
different cluster. Re-budget after round 1.

**`--dataset-name random` has no shared prefix**, so cache-hit ≈ 0 and kvd will show
sets ≫ gets. That is correct for a fixlen sweep and must be *stated*, not reported as a
cache failure. The 89 % hit rate lives in Case A.

**kvaware weight sweep (mission task 1.1).** The weight is genuinely exercised at 1P1D
because the scorer picks among 8 DP ranks. Plan: run one conc=64 round at
`--kv-prefill-overlap-weight` ∈ {1.0, 20.0} and compare per-rank pick distribution and
`cache_hits`. Cheap (one extra round), and it turns "set to a suitable value" from an
assertion into a measurement. Confirm the cost before spending it.

---

## Phase 2 — Case A (Optimus-AgenticBench)

**Authority: `Optimus-AgenticBench/CASE_AB_GUIDE.md`.** Run
`agent/workloads/glm52_crxx_caseA.fix.yaml` from the repo — not a copy of the spur
packup's derived YAML. Only two edits are legitimate:

| edit | from | to | why |
|---|---|---|---|
| `tokenizer` | `/path/to/GLM-5.2-MXFP4` | `/mnt/vast/xiaobo/models/GLM-5.2-MXFP4` | placeholder aborts the run |
| `initial_sessions` / `new_session_rate` | 32 / 0.10 | **re-solved from measured E2E** | the shipped pair assumes E2E≈15 s, explicitly a placeholder (guide Step 3) |

**`acc_len: 1.56` / `mtp_draft_tokens: 5` stay as shipped.** The spur run overrode them
to 1.0/1 *because MTP was off there*. **MTP is ON here**, which is the whole point of
this deployment — so the shipped values are correct, and the Case A spec's "56 %
acceptance, 5 draft tokens" becomes a directly checkable claim against the measured
`accept len`.

Guide facts that drive the plan:

- Closed-loop. `N = new_session_rate × E[turns] × (E2E + E[delay])`, with
  `E[turns]=9.50`, `E[delay]=18.0 s`, `E[input]=86,023`, `E[output]=1,433`.
- Offered rate `= new_session_rate × E[turns]`, **independent of E2E**.
- QPS knobs do nothing; `runner.py --auto-search` bisects on a variable with no effect.
- `ramp_duration` is a **warmup exclusion window**, not a ramp.
- Hitting `max_sessions` or `max_inflight` is a **failure signal**, not a cap to tune.
- `sustain_duration: 3600` is the honest window; shorter truncates the turn tail.
- The `sla:` block is documentation — to gate, pass `--slo-ttft-p90-ms` /
  `--slo-success-rate` to `runner.py`.

Steps:

- [ ] 2.1 Install the bench into a venv on the **jump host** (chi2866) — the driver is
      pure HTTP + tokenizer, and running it inside a container on a node under
      measurement puts the load generator on the host it is measuring.
- [ ] 2.2 `--mode preview` (no GPU). Must end `Plan looks self-consistent.` Record the
      Little's-law table and any warnings.
- [ ] 2.3 Probe run to measure E2E, then **re-solve** `new_session_rate` +
      `initial_sessions` per guide Step 3.
      ⚠️ The response is **superlinear** — the spur run assumed linear, pinned
      `max_inflight`, and had to abort a 20-minute window. Step up from a known-stable
      point.
- [ ] 2.4 Case A full: ramp 400 + sustain 3600 (~67 min). `--dashboard-mode` is
      **mandatory** or nothing structured is persisted (`summary.json`,
      `metrics.jsonl`, `metadata.json` are all written inside
      `if dashboard_mode and benchmark_name and data_dir:`).
      Abort live if in-flight pins at `max_inflight` or live sessions climb monotonically.
- [ ] 2.5 `statctl` before/after; both leg logs scanned for faults; per-rank pick
      distribution captured
- [ ] 2.6 **packup** — mission step 3

Guide's own sanity checks, to run before trusting the result: live session count near
target and flat (not climbing) · observed cache-hit near 0.89 · no `max_inflight`
throttling warning · mean E2E near the value used to solve the rate.

---

## Phase 3 — report

Format follows `../infera.glm5.2.experiment/agenticbench.glm52.spur.packup_20260731/analysis/`.

- [ ] 3.1 `sli_percentiles.md` — TTFT ladder p1…p99.9 recomputed from raw per-request
      samples, TPOT, supporting distributions; E2E labelled *derived* where no SLI exists
- [ ] 3.2 `yaml_vs_measured.md` — every knob vs measured, with the conversion rule;
      knobs with no measured counterpart carry their blast radius instead of a blank
- [ ] 3.3 `fixlen_analysis.md` — the 8 rounds: throughput / TTFT / TPOT vs conc, where
      the knee is, which phase sets it (prefill compute / KV transfer / decode)
- [ ] 3.4 Verdict vs **Case A original spec**: P50 E2E < 4.5 s, cache hit 88–90 %,
      **MTP 56 % acceptance @ 5 draft** (now measurable — MTP is on), turns/ISL/OSL
      distributions. Then: is the whole result *reasonable* for this hardware?

---

## Risks, ranked

1. **MTP + ctx 262144 + DPA has never run.** Long-context draft/verify under
   DP-attention is exactly where this stack has deadlocked before — branch patches 2
   and 3 exist for that. Phase 0 is the gate; a failure here is a real finding.
2. **Rust router is unproven end-to-end** and has no cache-view route, so the kvaware
   signal must come from its log. Pre-agreed fallback to python.
3. **Nodes are held by `yeandy-debug`.** Kill only our own sglang legs; never
   `scancel`. If yeandy starts real work mid-run, stop and re-plan.
4. **conc=128 at ISL 155K** ≈ 19.8 M prompt tokens against a pool that was 3.26 M
   tokens at ctx 32768. Read `max_total_num_tokens` off the boot log at 262144 before
   believing any conc=128 number — queueing, not OOM, is the expected outcome.
