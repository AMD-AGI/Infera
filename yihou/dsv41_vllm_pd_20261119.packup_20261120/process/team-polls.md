# Team polls (leader, every 20 min)

## 2026-11-19 08:13Z — poll #1 (workflow wf_7d3014f2-82b, phase Research, t+7 min)
- All 6 research agents active (transcripts updated 08:11-08:13Z); no research/*.md written yet (expected at t+7).
- model_arch: reading inference/convert.py, writing scripts/model_arch/kv_calc.py — on path.
- inferencex: cloned InferenceX into source/, reading inferencex-e2e metric code (tput_per_gpu) — on path.
- vllm: gh issues/PRs + AMD blog search — on path. sglang: reading sglang PRs + InferenceX PR 5034 — on path.
- colleagues: reading /perf_apps/yanfwang/InferenceX_CI/archive slurm_job-* dsv41 result json — on path (read-only).
- pd_method: found cc-traces dataset snapshot under /perf_apps/chaox/.../hf_home (read-only), writing analysis script in scripts/pd_method — on path.
- Issues: none. (Note only: agents write helper scripts under scripts/, inside workspace — OK.)

## 2026-11-19 08:33Z — poll #2
- wf_7d3014f2-82b (Phase 1 research) COMPLETED 08:31Z: 14 agents, 0 errors; SYNTHESIS.md + critic additions written. Closed.
- wf_5fe4b4da-c11 (Phase 2 kit, started 08:32Z, t+1 min): build:launcher reading chaox conc32 official_ci configs + server.log non-default args (right source for R4 verbatim); build:harness reading Muse ops rules/serve_vllm.py and InferenceX infx/bench/agentic — both on path.
- Issues: none.

## 2026-11-19 08:5xZ — security finding (leader note)
- serve.sh:264 `chmod a+rwx "$RUN" ...` on /perf_apps (shared 0777 NFS) while $RUN/entry.sh + config.env are executed/sourced as
  container root -> any cluster user could replace them. Valid. Fix AFTER wf_5fe4b4da-c11 finishes (reviewer may be editing serve.sh):
  executable files in an owner-only 0755 dir, files 0644, container-writable output in a separate sticky 1777 subdir; run_point.sh
  checks config.env owner/mode before sourcing.

## 2026-11-19 08:53Z — poll #3
- wf_5fe4b4da-c11: build:launcher DONE 08:42Z (configs r1/r2(c256,c384)/r4(+s8) + renders, serve/stop/preflight/pull_images). build:harness active
  (08:53 writing client_selftest.sh, watchdog.sh; venv-agentx + hf_home on /perf_apps/yihou) — on path. Reviewer not started yet.
- Open (recorded only): security fix serve.sh:264 a+rwx (apply after workflow ends); pyright type warnings summarize.py:109-130 (check after).

## 2026-11-19 09:13Z — poll #4
- wf_5fe4b4da-c11: both builders DONE (harness 08:59Z). review:kit running since 08:59Z (transcript updated 09:08Z) — on path.
- Recorded issues (security a+rwx serve.sh:264; pyright summarize.py:109-130) unchanged; not yet due (fix after workflow ends, by leader).

## 2026-11-19 09:20Z — leader fixes after wf_5fe4b4da-c11 (kit) completed
- SECURITY fixed: serve.sh no longer a+rwx; run/log dirs 1777 (sticky), executed/sourced files in runs/<tag>/kit (0755 dir,
  0644 files, mounted :ro, entrypoint /yihou/kit/entry.sh); run_point.sh + watchdog.sh refuse to source kit/config.env unless
  owned by us and not group/other-writable. bash -n OK; dry-run render OK. Backups debug/kit_bak/.
- summarize.py pyright warnings: heterogeneous dict typing only (runtime-correct by read); not changed.
- Kit review results: R4 = r4-tp2-c32 (CI exact, 16384 batched, no max-num-seqs); R4 image tag 404 on Docker Hub (risk);
  r1/r2 image present. Containers not tied to slurm job (requeue risk). UMBP 449 GB/server unmeasured -> cap 2 r2/node until E0.

## 2026-11-19T09:33Z — poll #5
- No active teammates (both workflows completed; kit ready). Waiting for openai grant (job 36733 PENDING Priority). Nothing to check.

## 2026-11-19T09:53Z — poll #6
- No active teammates. openai job 36733 still PENDING (Priority) since 08:06Z. Nothing to check.

## 2026-11-19T10:15Z — poll #7
- wf_6fe0cd91-f6a (G1/G3 code reads, started ~10:09Z): both readers active (10:14-10:15) — on path.
- wf_a8c03722-f52 (Dynamo+vLLM study, started ~10:13Z): 4 collectors active — just started, on path.
- Node n10-29 (b0bf, lent 10:1xZ): E0 pull/preflight/selftest chain running (logs /perf_apps/yihou/dsv41_vllm_pd_20261119/e0/).
- Issues: none.

## 2026-11-19 10:3xZ — leader notes
- E0 n10-29: 3,023 GiB DRAM, /dev/shm 1.6T, 8x ionic 400G (4 per NUMA), no foreign containers. selftest failed first: docker
  daemon cannot mount HOME NFS (root_squash) -> kit now stages kit + inferencex-e2e to /perf_apps/.../stage (stage.sh); ledger
  moved to /perf_apps/yihou/dsv41_vllm_pd_20261119/results/ledger.csv. selftest2 PASS.
- Launched 10:21Z (tags *-11191021): e1-r1-c128 (GPU0,1, :8000), e4-r1nosp-c128 (GPU2,3, :8010, only SPARSE_MLA=False),
  e2-r2-c384 (GPU4,5, :8020, UMBP 449 GB). Three concurrent points on one node = possible host interference (noted, CI ran 1/node).
- G1/G3 done. E4 meaning (code read, main 7a62ca06): with fp8_e4m3 the "AITER" path is _plain_fp8_sparse_mla (rocm.py:1540);
  SPARSE_MLA=False -> vLLM Triton sparse path. So E4 = plain_fp8 path vs Triton path; still the connector-forced difference.
- 10:4xZ FIRST-HAND: e4-r1nosp (fp8_e4m3 + SPARSE_MLA=False) refused at load: "--kv-cache-dtype fp8_e4m3 needs gfx950 and the
  aiter sparse MLA kernel ..., which is off with a KV connector" (vllm deepseek_v4/amd/rocm.py:170). => any PD connector
  (non-UMBP) forces fp8_ds_mla KV + Triton sparse path. Relaunched: e4b (fp8_ds_mla + SPARSE_MLA=False = PD-forced combo,
  GPU2,3) and e4c (fp8_ds_mla only, GPU6,7) vs e1-r1 (fp8_e4m3 + aiter) — e4c isolates dtype, e4b-e4c isolates kernel.
- 10:27Z watchdog ALERT valid: the 10:24Z e4b/e4c launch never reached the node (ssh hung, no 'launched'). Relaunched with ssh -n -o BatchMode=yes as e4b-r1dsnosp-c128-11191027 / e4c-r1ds-c128-11191027; runs dirs created. Lesson: always check 'launched' + runs dir after a remote launch.

## 2026-11-19T10:35Z — poll #8
- wf_a8c03722-f52 (Dynamo study): collectors done for vllm_side (verify running), others active 10:33 — on path.
- e1-r1 (10:21) DIED 10:32: RustFrontend "engine input registration timed out after 600s" while weights still loading from NFS
  (4 servers loading 476G model concurrently from /perf_apps). Fix: serve.sh sets VLLM_ENGINE_READY_TIMEOUT_S=3600 (setup only).
  Relaunched e1-r1-c128-11191033 via new scripts/kit/launch_remote.sh (verifies 'launched' + runs dir).
- RECORD (not yet acted): e2-r2 (vllm 10:24), e4b/e4c (10:27) were started WITHOUT the timeout env -> may die at ~10:34/10:38;
  waiter b3dr3ctx9 on e2. Next: relaunch any that die; later stage model to node-local /data to cut load time.
- 10:36Z e2-r2-c384-11191021 died (600 s frontend timeout); relaunched with timeout env.
- 10:40Z e4b/e4c -11191027 died (600 s frontend timeout, as predicted); relaunched as e4b-r1dsnosp-c128-11191039 / e4c-r1ds-c128-11191040 with timeout env. All 4 active points now carry VLLM_ENGINE_READY_TIMEOUT_S=3600.
- 10:5xZ all 4 servers READY (startup 616-842 s). All 4 clients failed at config: aiperf offline tokenizer lookup by HF id
  needs a COMPLETE cached snapshot (IncompleteSnapshotError, 86 files incl. weights). Fix: run_point.sh passes TOKENIZER=<local
  snapshot dir> (InferenceX replay.py:37 supports a checkpoint path). Servers had 0 prompt tokens (metrics) -> still fresh;
  re-ran run_point on the same servers (logs launch/<tag>.point2.log; old output kept as bench.fail-tokenizer). Dataset load OK.
  Lesson: selftest checked transformers tokenizer load, not aiperf's lookup path -> extend selftest later.

## 2026-11-19T10:55Z — poll #9
- wf_a8c03722-f52 (Dynamo study): all 4 collect+verify done; synthesize done; critic running — on path.
- Bench: 4 points (R1, R2-c384, E4b, E4c) in official run since 10:54Z on n10-29 after tokenizer fix; watchdog round sent.
- Open (recorded): router /v1/models forwards to 0.0.0.1 (harmless for bench); selftest does not cover aiperf tokenizer path.
- 11:2xZ ROOT CAUSE (first-hand) e2-r2-c384 failure: vllm-router process soft RLIMIT_NOFILE = 1024 in our container
  (/proc/<router>/limits; engine 65535). At c384 the warmup "cache pressure" phase bursts ~550 in-flight -> router fd exhaustion ->
  "error sending request" -> circuit breaker opens -> 503 -> root warmup request fails -> aiperf aborts. Repro on the idle
  server: 700 parallel tiny requests -> 507x200 / 2x500 / 191x503. Fix: serve.sh adds --ulimit nofile=1048576:1048576 (infra
  parity with CI). c128 points (2x128 fds) unaffected. Relaunching R2 c384.

## 2026-11-19T11:40Z — poll #10
- No research workflows active (Dynamo study done 11:0xZ). Teammate: watchdog agent (10-min rounds) — on path, its e2 ALERT led to the router fd-limit root cause.
- Bench: R1/E4b/E4c in official run; R2 relaunched 11:38Z with nofile fix.

## 2026-11-19T11:53Z — poll #11
- No research workflows active. Watchdog teammate on path (self-corrected its fd check: router cmdline is 'vllm::router', real pid shows nofile 1048576).
- Bench: R1/E4b/E4c profiling, ~15 min left, 0 errors; R2 c384 (11191138) starting, umbp+vllm+router up 11:42Z.

## 2026-11-19T12:10Z — RESULTS batch 1 (n10-29, 4 servers concurrent on one node, official 3600 s, single runs)
| point | KV dtype | sparse-MLA kernel | tok/s/GPU | qps | ITL p50 | p90i | KV pool tok/rank | prefix hit |
| e1-r1 c128 | fp8_e4m3 | aiter (plain fp8) | 255,797 (CI 271,329: -5.7%) | 4.166 | 14.7 ms | 41.6 | 29.67M | 0.960 |
| e4c c128 | fp8_ds_mla | aiter | 243,585 (-4.8% vs R1) | 4.015 | 16.9 ms | 36.9 | 25.85M | 0.960 |
| e4b c128 | fp8_ds_mla | Triton (PD-forced) | 221,199 (-9.2% vs e4c, -13.5% vs R1) | 3.741 | 20.7 ms | 29.8 | 25.61M | 0.958 |
- Caveats: single runs, noise band unknown; concurrent servers on one node (host interference possible); different GPU pairs.
- Reading (not concluded): the PD-forced kernel+dtype combo costs ~13.5% at c128 colocated; ITL grows 14.7->20.7 ms.
- 12:1xZ launched E1 replicate e1-r1rep-c128-11191210 (GPU0,1) + E3 sweep e3-r1-c192-11191211 (GPU2,3), e3-r1-c256 (GPU6,7), all r1 config, only concurrency varies.

## 2026-11-19T12:13Z — poll #12
- No research workflows. Watchdog teammate on path. Batch-1 results recorded. 4 points running (e1-r1rep c128, e3 c192/c256, e2-r2 c384 warmup).
- Open (recorded): R1 -5.7% vs CI (concurrent node load suspected, not shown); monitor seeding bug (replays old events, harmless).
- 12:21Z GPU mapping check: rocm-smi idx != HIP idx. VRAM pattern: smi4,smi7 = e2 (HIP 4,5, busy); smi5,smi6 at 67% = e3-c256 (HIP 6,7, loading); smi0-3 = e1rep+e3-c192. 8 distinct GPUs, no overlap. Watchdog item closed.

## 2026-11-19T12:33Z — poll #13
- No research workflows. Watchdog on path. e2-r2-c384 passed cache-pressure (nofile fix confirmed), profiling since ~12:19Z. e1rep/c192/c256 warmup, 0 errors.
- G2 answered (preflight in-image, image 60bd6dd2 vllm 0.34.0rc1.dev571): mooncake-transfer-engine-rocm 0.3.14 (import ok), nixl-rocm 1.5.0 (import 'nixl' fails -> module name differs?), amd_mori 1.2.3.post1 (umbp binaries present).

## 2026-11-19T12:53Z — poll #14
- wf_1d6f57ec-8f6 (PD kit build): build:pd-kit active — see file list; on path. Watchdog on path; 4 benches running.

## 2026-11-19T13:13Z — poll #15
- ISSUE (first sighting, record only): wf_1d6f57ec-8f6 build:pd-kit transcript not updated since 12:43Z (30 min); last tool = dry-run regression loop over r1/r2/r4 renders; no process / container of it running on a16-19; no PD files written yet. Intervene at next poll if still stalled.
- Watchdog on path; 4 benches finishing ~13:20-13:50Z.
- 13:21Z RESULT e2-r2-c384-11191138 (R2 = public bar recipe, DP2+UMBP c384, GPU4,5, 3 other servers concurrent on node):
  407,049 tok/s/GPU (CI 427,451: -4.8%), qps 7.18, ISL 112.6k, OSL 889, TTFT p50/p90 1.68/3.97 s, ITL p50 52 ms, p90i 14.8 (CI 15.8),
  GPU hit 0.898 + external(UMBP) 0.062, KV usage avg 0.70 / max 0.90, running avg 699 (both ranks), 0 preemptions, AL 3.51, contract ok.
  R1 -5.7% / R2 -4.8% vs CI: consistent offset; concurrent-node load is the untested suspect (isolated rerun needed to tell).

## 2026-11-19T13:33Z — poll #16
- build:pd-kit stall (poll #15) RESOLVED without intervention: transcript active again 13:31Z, configs/e7-pd-mooncake-1p1d-dp2.env written. (Probably a long model turn; no action.)
- e2-r2-c384 done (407,049). 3 benches finishing 13:32-13:42Z. Watchdog on path.
- 13:32Z NODE LOST: n10-29 (b0bf, job 36507) preempted+requeued; all containers gone. b0bf + bd2b pending.
- Salvaged from NFS raw profile_export.jsonl (method validated: recomputes e1-r1 = 255,797 exactly):
  e1-r1rep c128: 266,906 tok/s/GPU, full 3628 s window, 0 err (replicate of 255,797 -> spread 4.3%; CI 271,329 -1.6%)
  e3-r1 c192: 333,066 (PARTIAL window 3416 s, not official)   e3-r1 c256: 336,722 (PARTIAL 3093 s, not official)
  -> R1 (no UMBP) knee is >= c192..c256 (~+25% over c128); R2 c384 with UMBP 407,049. Partial points: directional only.

## 2026-11-19T13:53Z — poll #17
- build:pd-kit: transcript idle again since 13:31Z (22 min); serve_pd.sh written 13:30Z, no e7 renders yet. Recorded (2nd stall episode, same pattern as 12:43-13:31 which self-resolved). Intervene at next poll if still idle.
- No node (b0bf/bd2b pending, Priority). No active benches; watchdog paused.

## 2026-11-19T14:14Z — poll #18 (INTERVENED)
- build:pd-kit still idle (13:31Z -> 14:13Z, 42 min, 2nd consecutive poll): its last bash tool call (dry-run run_point on dev host) never returned; no process left on a16-19. Stopped wf whbguwb6d; launched wf_8337b5c2-ebb 'finish' agent (inspects partial files, regression-checks colocated renders, timeouts on every probe) + reviewer.

## 2026-11-19T14:33Z — poll #19
- wf_8337b5c2-ebb finish:pd-kit active (see mtimes) — on path. No node; no benches.
- ISSUE (first sighting): finish:pd-kit idle since 14:20Z (13 min) on a trivial bash call (sed + py_compile + python -c import). Hypothesis (memory 'workflow-workspace-inside-cwd'): hidden permission prompt in the subagent stalls the tool call — same likely for the 13:31 stall. Record only; at next poll, if still idle: stop wf and finish the kit inline (leader), avoiding subagent shell for the rest.

## 2026-11-19T14:53Z — poll #20 (INTERVENED)
- finish:pd-kit still idle since 14:20Z (33 min, 2nd poll). Stopped wf wwi45cm1b. Leader finishes/reviews the PD kit inline.
- 15:47Z Leader finished PD kit: static checks OK (bash -n, py_compile, colocated render regression identical, e7 P/D parity), wrote PD_README.md, staged. Runtime unverified.

## 2026-11-19T15:47Z — poll #21
- No active teammates/workflows; no node (both openai jobs pending); watchdog skipped. PD kit ready (leader-finished).

## 2026-11-19T16:13Z — poll #22
- No active teammates/workflows; no node (both openai jobs pending since 13:32Z). Waiting for grant or user decision on crusoe fallback.

## 2026-11-19T16:33Z — poll #23
- No active teammates/workflows; no node (both openai jobs still pending). Idle-waiting for grant.

## 2026-11-19T16:53Z — poll #24
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T17:13Z — poll #25
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T17:33Z — poll #26
- No active teammates/workflows; no node (openai 36733/36507 pending since 13:32Z). Idle-waiting.

## 2026-11-19T17:53Z — poll #27
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T18:13Z — poll #28
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T18:33Z — poll #29
- No active teammates/workflows; no node. bd2b extended to 2026-11-20 06:30Z (pool mgr). Idle-waiting.

## 2026-11-19T18:53Z — poll #30
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T19:13Z — poll #31
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T19:33Z — poll #32
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T19:53Z — poll #33
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T20:13Z — poll #34
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T20:33Z — poll #35
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T20:53Z — poll #36
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T21:13Z — poll #37
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T21:33Z — poll #38
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T21:53Z — poll #39
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T22:13Z — poll #40
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T22:33Z — poll #41
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T22:53Z — poll #42
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T23:13Z — poll #43
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T23:33Z — poll #44
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-19T23:53Z — poll #45
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T00:13Z — poll #46
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T00:33Z — poll #47
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T00:53Z — poll #48
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T01:14Z — poll #49
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T01:33Z — poll #50
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T01:53Z — poll #51
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T02:13Z — poll #52
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T02:33Z — poll #53
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T02:53Z — poll #54
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T03:13Z — poll #55
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T03:33Z — poll #56
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T03:53Z — poll #57
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T04:13Z — poll #58
- No active teammates/workflows; no node (openai 36733/36507 pending). bd2b extended to 2026-11-20 16:00Z. Idle-waiting.

## 2026-11-20T04:33Z — poll #59
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T04:53Z — poll #60
- No active teammates/workflows; no node (openai 36733/36507 pending). Idle-waiting.

## 2026-11-20T05:24Z — poll #61
- crusoe v3-186 (job 300344, oai, until 2026-11-20 17:11Z) granted 05:13Z; launched 05:21-22Z: c1-r1-c128 (GPU6,7 official), e7s PD 1P1D smoke (GPU0-3), e7r reference (GPU4,5). Kit ported (cru_launch.sh / cru_pd.sh, run_point KIT_VENV_CTR). Watchdog cron switched to crusoe. No workflows active.
