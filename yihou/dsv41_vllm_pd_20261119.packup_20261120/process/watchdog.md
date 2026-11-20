
### 2026-11-19T10:25:52Z watchdog
- e1-r1-c128-11191021: ctr Up 4 min, health 000 (startup, aiter GEMM/indexer init in progress), server.log age 24s, 0 error matches.
- e2-r2-c384-11191021: ctr Up 4 min, health 000 (startup), server.log age 34s, 0 error matches.
- e4b-r1dsnosp-c128-11191024 / e4c-r1ds-c128-11191024: NOT FOUND -- no launch/*.log and no runs/<tag>/ dir. Never launched (or launch failed before writing a log).
- e4-r1nosp-c128-11191021: known-dead, ignored. GPU VRAM 0:20% 3:20% 4:4% 7:4%, others 0%; util 0% all. watchdog.sh STATUS: OK.

### 2026-11-19T10:36:27Z watchdog
- e4b-r1dsnosp-c128-11191027: ctr Up 8 min, health 000, weight load ~98% (47/48 shards, prefetch done 10:36:15), 0 error matches.
- e4c-r1ds-c128-11191027: ctr Up 8 min, health 000, model loaded 10:36:27 (150.82 GiB, 439.9 s), entering KV/graph setup, 0 error matches. No "engine input registration timed out" seen on e4b/e4c (yet).
- e1-r1-c128-11191033: ctr Up 2 min, health 000, aiter GEMM init, server.log age 72s, 0 error matches. e2-r2-c384-11191035: ctr Up 30 s, no server.log yet (umbp/entry only), 0 errors.
- GPU VRAM 1/2/5/6 at 52-55%, 0/3 4%, 4/7 0%; util 0%. watchdog.sh STATUS: OK. Old 11191021 tags ignored.

### 2026-11-19T10:45:52Z watchdog -- ALERT
- e1-r1-c128-11191033: server READY (health 200, ready after 616 s), but official bench FAILED: CHAIN_DONE rc=1 at 10:45:28Z. aiperf client died at profile_configure (10:44:33): "TokenizerError: Failed to load tokenizer 'deepseek-ai/DeepSeek-V4.1-Flash' ... not present in the HuggingFace cache" -> no profile_export*.json. No requests were sent. Also run_point WARN: /v1/models does not list the model; bench/models.json = "Request failed ... http://0.0.0.1/v1/models". Server left running (not stopped).
- e2-r2-c384-11191035: Up 9 min, health 000, model loaded 10:45:52 (150.82 GiB, 382 s), 0 errors.
- e4b-r1dsnosp-c128-11191039 (Up 6 min) / e4c-r1ds-c128-11191040 (Up 5 min): health 000, aiter GEMM/indexer init, 0 errors, no registration-timeout. e4c server.log age 120s.
- GPU VRAM 0/3 89%, 1/2 53%, 4/7 55%, 5/6 21%. watchdog.sh STATUS: ALERT (e1 bench rc=1).

### 2026-11-19T10:56Z watchdog (point2 benches, started 10:53:48Z)
- All 4 clients are in the WARMUP phase (ramp fired ~10:55:46-49; 142 mandatory requests for c128, 409 for e2 c384). 0 client ERROR lines and 0 server Traceback/OOM/HSA/Killed/registration-timeout matches.
- Live engine metrics (~10:56, per engine DP0/DP1) as running/waiting, kv%, and prefix hits/queries:
  - e1 :6100: 1/61 + 2/36, kv 1.3/1.5%, prefix hits 23.9k/1.87M + 0/2.01M.
  - e4b :6110: 1/56 + 1/50, kv 1.2/1.7%, prefix hits 768/1.63M + 0/1.53M.
  - e4c :6130: 1/66 + 1/39, kv 1.2/1.4%, prefix hits 46.3k/1.46M + 20.5k/1.39M.
- e2 :6120 read all-zero at first. A re-read ~1 min later gave running+waiting=404, and server.log is serving: a Triton JIT-during-inference warning for _pa_mqa_logits_mxfp4_kernel at 10:57:05, plus untuned aiter bf16 lm_head GEMM M:3.

### 2026-11-19T11:05:47Z watchdog -- ALERT (e2 anomaly, no errors)
- Every tag has 0 client ERROR lines, client errors=0, and 0 server Traceback/OOM/HSA/Killed/timeout matches.
- Phase and live metrics per engine (DP0/DP1) as running/waiting, kv%, and local prefix hit = hits/queries:
  - e1: PROFILING (realtime 00:45 at 11:05:30). 10/0 + 17/0, kv 1.0/1.6%, hit 85.4/85.3%.
  - e4b: warmup returned=1020/1422 (570 s). 3/63 + 1/2, kv 1.9/1.2%, hit 78.1/84.2%.
  - e4c: warmup 923/1422 (570 s). 1/60 + 1/13, kv 2.0/3.0%, hit 82.1/84.3%.
- e2: warmup returned=241/4249 after 570 s, gaining about 10 per 30 s, with sent=409 and in_flight=168. Engines 1/65 + 1/87, kv 1.3/1.3%. Local prefix hit is 0.77/0.23% (171k/22.2M, 52k/22.1M). UMBP external hit is 0/22.06M on DP0 and 109k/22.07M = 0.50% on DP1; umbp store submitted 172k/171k, load 854/854 completed on DP1 only.

### 2026-11-19T11:40:34Z watchdog
- e1 / e4b / e4c are all PROFILING (official 3600 s), with 0 client ERROR and 0 server error matches. e1: realtime 35:46 (~24 min left). e4b: 33:59 (~26 min left). e4c: 34:00 (~26 min left).
- Live engine metrics (DP0/DP1) as running/waiting, kv%, prefix hit:
  - e1 :6100: 49/0 + 35/0, kv 21.7/16.4%, hit 93.4/93.6%.
  - e4b :6110: 41/0 + 45/3, kv 16.9/22.1%, hit 93.6/92.2%.
  - e4c :6130: 35/3 + 33/0, kv 19.0/14.6%, hit 93.1/93.6%.
- e2-r2-c384-11191138: ctr Up ~1 min, health :8020 000 (starting), no server.log yet. vllm-router pid 250 has Max open files 1048576/1048576, so the ulimit fix is applied.

### 2026-11-19T11:45:57Z watchdog
- e1 / e4b / e4c are PROFILING with 0 client ERROR and 0 server error matches. e1 40:46 (~19 min left); e4b 39:29 and e4c 39:30 (~20.5 min left).
- Live engine metrics (DP0/DP1) as running/waiting, kv%, prefix hit:
  - e1: 33/0 + 35/0, kv 16.1/12.6%, hit 93.8/94.0%.
  - e4b: 35/0 + 34/1, kv 14.1/14.4%, hit 93.8/92.8%.
  - e4c: 37/0 + 33/0, kv 18.7/15.2%, hit 93.4/94.0%.
- e2-r2-c384-11191138 is still STARTING, not in warmup. umbp became ready after 195 s (11:42:39); vllm (pid 340) and router (pid 344) started 11:42:40; server.log is in aiter GEMM/indexer init (11:45:38); health :8020 000; 0 errors. No bench client yet, so there are no returned/in_flight/cache-pressure numbers.
- CORRECTION to 11:40: that fd check hit pid 250 before the router existed. Also, `pgrep -f vllm-router` self-matches the bash running the check and does not match the router, whose cmdline is `vllm::router`. Re-verified on the real router (pid 344): Max open files 1048576/1048576.

### 2026-11-19T11:55:45Z watchdog
- e1 / e4b / e4c are PROFILING, not finished yet, with 0 client ERROR and 0 server error matches. e1 50:47 (~9 min left, ~12:05Z); e4b 49:00 and e4c 49:01 (~11 min left, ~12:06Z).
- Live engine metrics (DP0/DP1) as running/waiting, kv%, prefix hit:
  - e1: 35/0 + 33/3, kv 18.1/11.8%, hit 94.3/94.2%.
  - e4b: 48/0 + 35/0, kv 21.7/16.4%, hit 94.1/93.6%.
  - e4c: 51/1 + 36/1, kv 24.1/18.1%, hit 93.9/94.3%.
- e2-r2-c384-11191138: bench client started 11:53:34; WARMUP credit phase started 11:55:29 (409 primers, 9.9 s ramp). No progress line yet, so no returned/in_flight; cache-pressure phase not started. 0 client ERROR, 0 server errors.
- e2 engines at ~11:56: running/waiting 1/185 + 2/199, kv 1.3/1.3%, local hit 3.2/8.8% (30k/933k, 80k/913k), external(UMBP) 0/903k + 0/833k. Client WARNING: cached_tokens absent from usage.

### 2026-11-19T12:05:49Z watchdog
- e1-r1-c128-11191033: profiling ENDED 12:05:25 and the client is post-processing ("Waiting for result domains: server_metrics"); no score line in run_point.log yet. 15,095 ok / 0 err. Client final: tput_in 507,332/s, tput_out 4,261/s, ttft p50 578 ms / p99 7,096 ms, itl p50 15 ms, srv prefix_cache_hit 96.0% (trace theoretical 96.6%). One client ERROR at teardown: "Timeout waiting 10.0s for cancelled credits to return ... Forcing phase completion" (98 open slots released).
- e4b (59:01) / e4c (59:02) are still PROFILING at 12:05:23, 0 client ERROR, no score yet.
- e2-r2-c384-11191138: WARMUP returned=289/4249, sent=409, in_flight=120, errors=0, elapsed 600 s, gaining ~6-12 per 30 s. Cache pressure has NOT started. That is the same pace as dead run e2-11191035 (241/4249 at 570 s).
- e2 engines at ~12:06: running/waiting 1/40 + 1/69, kv 1.6/2.1%, local hit 0.41/0.39%, UMBP external hit 0.57/0.09% (151k/26.7M, 23k/26.6M). 0 server errors.

### 2026-11-19T12:15:56Z watchdog
- e2-r2-c384-11191138: WARMUP cache-pressure phase running since 12:12:07, with 0 client ERROR and 0 server errors so far. Progress returned=1369/4249, sent=1787, in_flight=418, errors=0 (1200 s). Server log at 12:15:35: running 8, waiting 396, prompt tput ~84-88k tok/s.
- e2 engines (DP0/DP1) as running/waiting, kv%, local hit, external hit:
  - DP0: 5/171, kv 5.3%, local hit 70.3%, UMBP external 1.9% (858k/44.6M).
  - DP1: 8/202, kv 6.6%, local hit 46.7%, UMBP external 25.8% (17.9M/69.4M).
- e1-r1rep-c128-11191210 (Up 5 min), e3-r1-c192-11191211 (Up 4 min), e3-r1-c256-11191212 (Up 3 min): all starting, health 000, aiter GEMM init, 0 errors.
- GPU use/VRAM: 4 and 7 at 100%/92%; 0,1,2,3,5,6 at 0-1%/20-21%. GPU7 is busy while e3-c256 (GPU6,7) is still starting, and GPU5 is idle while e2 (GPU4,5) runs; the watchdog GPU index vs the --device-ids mapping is unverified.
- watchdog.sh STATUS: OK.

### 2026-11-19T12:25:48Z watchdog
- e2-r2-c384-11191138: cache pressure PASSED with 0 errors (budget reached 12:19:01, handoff persisted 2175 streams 12:19:10). Now PROFILING at realtime 06:26, 0 client ERROR, 0 server errors.
- e2 engines (DP0/DP1): running/waiting 197/3 + 242/5, kv 61.2/83.9%, local hit 86.6/81.3%, UMBP external hit 7.0/32.8% (4.18M/59.7M, 32.6M/99.5M).
- New points (all ready, in initial WARMUP, 0 client ERROR, 0 server errors):
  - e1-r1rep-c128-11191210 (ready 586 s): returned 61/1422, in_flight 81 (180 s). Engines 1/24 + 1/44, kv 1.9/1.6%, hit 0/0.5%.
  - e3-r1-c192-11191211 (ready 556 s): 65/2128, in_flight 143 (152 s). Engines 1/70 + 1/60, kv 1.4/1.4%, hit 0.02/1.1%.
  - e3-r1-c256-11191212 (ready 586 s): 66/2845, in_flight 219 (90 s). Engines 1/130 + 1/76, kv 1.4/2.0%, hit 1.2/1.0%.

### 2026-11-19T12:35:48Z watchdog
- Every tag has 0 client ERROR and 0 server Traceback/OOM/HSA/Killed matches.
- Phase and engines (DP0/DP1) as running/waiting, kv%, local prefix hit:
  - e2-r2-c384-11191138: PROFILING 16:27. 163/1 + 232/8, kv 51.9/77.0%, hit 88.9/83.7%; UMBP ext hit 9.2/41.9% (6.66M/72.3M, 56.5M/134.7M).
  - e1-r1rep-c128-11191210: cache pressure passed (handoff 678 streams 12:32:25); PROFILING 03:12. 17/3 + 10/3, kv 2.4/1.5%, hit 87.3/85.4%.
  - e3-r1-c192-11191211: WARMUP cache pressure (since 12:32:40), returned 1543/2128, in_flight 127, errors 0. 1/93 + 1/0, kv 0.7/1.6%, hit 69.6/85.6%.
  - e3-r1-c256-11191212: primers took ~12 min to drain (285 sent, returned 279 at 690 s); cache pressure started 12:35:50. Engines nearly idle at the read (1/0 + 0/0), hit 1.2/1.3%. Server logged Triton JIT during inference (_mask_candidates_kernel) at 12:35:51.

### 2026-11-19T12:45:46Z watchdog
- All 4 are PROFILING, with 0 client ERROR and 0 server error matches; every cache-pressure warmup passed. Elapsed: e2 26:28, e1rep 13:12, e3-c192 08:53, e3-c256 03:31.
- Engines (DP0/DP1) as running/waiting, kv%, local hit:
  - e1rep: 47/4 + 61/1, kv 17.1/20.2%, hit 90.8/90.5%.
  - e3-c192: 85/4 + 98/2, kv 32.0/28.6%, hit 87.8/88.7%.
  - e3-c256: 43/0 + 27/0, kv 10.0/7.5%, hit 86.5/84.4%. Running total 70 is far below c256, observation only.
  - e2: 168/10 + 201/2, kv 56.7/76.6%, hit 90.0/84.6%, UMBP ext 13.0/48.6%.

### 2026-11-19T12:55:47Z watchdog
- All 4 are PROFILING, with 0 client ERROR, err=0 in the client counters, and 0 server error matches.
- Elapsed, requests done (all ok), and client averages:
  - e2: 36:29, done 15,256, avg rps 7.0, tput_in 773k/s, tput_out 5,870/s.
  - e1rep: 23:12, done 6,610, avg rps 4.7, tput_in 471k/s, tput_out 4,363/s.
  - e3-c192: 18:53, done 6,056, avg rps 5.3, tput_in 565k/s, tput_out 4,435/s.
  - e3-c256: 13:31, done 4,587, avg rps 5.7, tput_in 546k/s, tput_out 3,851/s.
- Engines (DP0/DP1) as running/waiting, kv%, local hit:
  - e1rep: 42/0 + 33/4, kv 15.6/10.9%, hit 92.7/92.5%.
  - e3-c192: 64/1 + 81/2, kv 26.8/26.2%, hit 91.1/91.9%.
  - e3-c256: 144/1 + 102/2, kv 56.7/35.6%, hit 89.9/88.6%. Running is now 246, near c256.
  - e2: 167/9 + 185/5, kv 53.2/71.7%, hit 91.1/85.0%, UMBP ext 13.0/53.8%.

### 2026-11-19T13:05:47Z watchdog
- All 4 are PROFILING, with 0 client ERROR, err=0, and 0 server error matches. e2 is not finished: 46:30, ~13.5 min left, ETA ~13:19Z; no score yet.
- Elapsed, requests done (all ok), avg rps, tput_out:
  - e2: 46:30, 19,723, 7.1, 6,124/s.
  - e1rep: 33:13, 9,219, 4.6, 4,438/s.
  - e3-c192: 28:54, 9,398, 5.4, 4,839/s.
  - e3-c256: 23:31, 7,997, 5.7, 4,562/s.
- Engines (DP0/DP1) as running/waiting, kv%, local hit:
  - e1rep: 22/0 + 40/1, kv 11.7/19.2%, hit 93.7/93.6%.
  - e3-c192: 80/0 + 49/0, kv 33.3/21.0%, hit 92.5/93.0%.
  - e3-c256: 151/9 + 96/5, kv 61.1/32.8%, hit 91.5/90.6%.
  - e2: 186/1 + 196/4, kv 54.4/72.1%, hit 91.6/85.2%, UMBP ext 14.5/57.3%.

### 2026-11-19T13:15:48Z watchdog
- All 4 are PROFILING, with 0 client ERROR, err=0, and 0 server error matches. e2 is not finished: 56:31, ETA ~13:19-13:20Z; no score yet.
- Elapsed, requests done (all ok), avg rps, tput_out:
  - e2: 56:31, 24,348, 7.2, 6,248/s.
  - e1rep: 43:13, 11,396, 4.4, 4,362/s.
  - e3-c192: 38:54, 12,548, 5.4, 4,911/s.
  - e3-c256: 33:32, 11,526, 5.7, 4,925/s.

### 2026-11-19T13:25:44Z watchdog
- e1rep / e3-c192 / e3-c256 are PROFILING, with 0 client ERROR, err=0, and 0 server error matches. e2 is finished and stopped, so it is ignored.
- Elapsed, requests done (all ok), avg rps, tput_out:
  - e1rep: 53:14, 13,922, 4.4, 4,378/s (ETA ~13:32Z).
  - e3-c192: 48:25, 15,161, 5.2, 4,934/s (ETA ~13:37Z).
  - e3-c256: 43:03, 14,304, 5.5, 4,874/s (ETA ~13:42Z).
