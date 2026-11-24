# Notes — gotchas, wrong turns, open questions, 2026-11-20 05:30Z – 11-24 (what / why / how / context)

Earlier gotchas (NFS root_squash, 600 s registration timeout, tokenizer snapshot, router nofile, ssh launch hang, rocm-smi index):
`../dsv41_vllm_pd_20261119.packup_20261120/notes.md`. Sources here: `process/team-polls.md`, `process/watchdog.md`, `research/*.md`.

## Gotchas / process lessons
1. **crusoe idle release (4 low-utilisation scans / ~30 min)** — what: v3-186 (job 300344) cut 05:48Z 11-20, v3-129 (300599) cut
   ~12:05Z 11-20, v3-200 (302960) got an idle warning 14:27Z 11-21. why: (a) client deps (venv, hf_home 3.5 G) were not yet on crusoe
   NFS, bench failed `dataset cache missing`; (b) the session did not answer the pool manager's util pings (busy / not running turns
   ~11:05-15:35Z); (c) servers loading in parallel showed 0% between spikes. how: stage venv + hf_home on crusoe NFS before requesting;
   answer pool pings first; have a bench start within minutes of "ready"; stagger starts. context: crusoe utilisation policy of the pool.
2. **crusoe node stalls** (`research/crusoe_r2_umbp_stall.md`, observations only, cause unknown) — what: "No available shared memory
   broadcast block found in 60 seconds" then EngineCore `shm_broadcast` dequeue TimeoutError or "RPC call to sample_tokens timed out";
   GPUs 0%; affected runs contract_ok=False or rc=1. 16 episodes: v3-200 14:47Z, ~18:30Z (11-21); v3-014 02:08, 06:30, 07:30, 08:16
   (recovered), 09:00, 09:49, 10:04, 17:44, 19:02, 19:52, 20:15, 23:50 (11-22), 02:22, 04:27 (11-23). Each later episode hit all
   servers of one PCIe domain (GPU0-3 = domain 0002, GPU4-7 = 0003) at the same second while the other domain kept running. Not
   required: R2/UMBP, the KV connector (R1 fails too), a concurrent server start (06:30Z single server). Persisted across 3 slurm
   jobs on v3-014. `journalctl -k` showed no amdgpu/xgmi/AER lines (readability not verified). how (handling used): stop the failed
   pair, relaunch sequentially; after a pair fails twice in one job, leave it idle (GPU4-7 idle on 305057 from 04:32Z 11-23);
   recommended to the user to exclude v3-014 (decision pending). Also the e4c c320 half-rate run (139,858) on GPU4,5 is void, cause unknown.
3. **Launch gap 480 s between server starts on crusoe** — what: 3 stalls on 11-21/22 began while several servers started/captured
   at once. how: `cru_seq_launch.sh` (wait for "ready after") and `cru_seq_swap.sh` (MIN_GAP=480). Sequential start alone did not
   prevent later stalls (#2). Minor cost: a LAST_START offset added a full extra 8-min gap once (watchdog 19:26Z 11-22).
4. **Chains must live on the node** — what: a swap chain started through a local ssh died when the ssh dropped (05:19Z 11-22,
   GPU0,1 idle ~5 min). how: `nohup … & disown` (or setsid) inside `spur exec`, track progress via `$R/launch/*.log` + ledger, not the
   local background output. Waiter caps: `cru_swap_after.sh` originally 540x10 s, too short for c>=320 armed early (now 1080x10 s).
5. **Kill by PID, never `pkill -f <script>` inside `spur exec bash -c "…"`** — what (14:1xZ 11-21): `pkill -f cru_pd_chain.sh`
   matched its own `bash -c` command line and killed itself; the relaunch did nothing; containers were then stopped by name. Same
   self-match class as `pgrep -f` waiters (memory "waiter-pid-not-pgrep"). how: record chain PIDs, `kill <pid>`; waiters use `kill -0 <pid>`.
6. **spur005 login node full** (98% memory, 09:44Z 11-23) — `spur`/ssh via spur005 unreliable; used spur006.
7. **openai pool daemon disk rule** — n02-25 job 36884 auto-cancelled 12:04:47Z 11-21 (docker_free 151 G < 200 G) while our
   e7h-c384b/e7p-c256 stacks were loading. Also `/data` was root-only on n02-25 (first launch 11211200 failed) -> node-local root
   `/scratch/yihou/...` for that node. openai login/squeue was failing from 11-22 ~10:03Z (rc=255 from 09:27Z 11-23).
8. **openai preemption** — n06-25 (36733) and n01-29 (36884) preempted ~08:08-08:13Z 11-21, killing e7h c512/c384 mid-warmup/measure.
9. **Release timing** — 305057 was released 12:35Z 11-23 after its final points, ~1 h before its ~13:37Z end, as agreed with the
   pool manager (GPU4-7 were idle by rule). For openai the standing rule is the opposite (memory "oai-cap-dont-release-early").
10. **Watchdog misreads / misses** — a finished point sat idle 16 min (16:50-17:06Z 11-21, one tag missed in a round -> watchdog now
    checks every tag each round); "No available shared memory broadcast block" also appears during long FULL graph capture and is
    benign there (do not count it as a stall unless GPUs go 0% / TimeoutError follows); GPU 0% at window end = export, not a stall.
11. **e7so "READY" was not a working PD** (correction 03:30Z 11-21): its D engine died on a port 6200 bind clash with e7sg; the
    serve_pd/serve.sh port guard did not catch it. Check both role engines + router health, not one READY line.
12. **vLLM MooncakeConnector does not send KV when prefill ends FINISHED_STOPPED** (second-hand from llying-001/vllm#3; the patch
    target matches once in vLLM main 7a62ca0) -> D waits until abort timeout. Kit: `MC_STOPPED_FIX=1`, refuses to start if unpatched.

## Findings with evidence level
- [FH] PD 1P1D c256 254,318 tok/s/GPU vs colocated R1 c256 346,601 (openai) / 358,296 (crusoe mean) at 2 GPUs: PD is below colocated
  per GPU at this shape. P queue 8.04 s/request dominates the server-side time; D KV avg 32% (research/e7_c256_stage_analysis.md).
- [FH] Connector-forced kernel path cost grows with concurrency (README crusoe table). Why: not measured.
- [FH] Same-node Mooncake HIP transport: `hipIpcOpenMemHandle` 201 "invalid device context" (code path read in hip_transport.cpp);
  why it is intermittent: unknown. MC_DISABLE_HIP=1 + RDMA: 0 failures before preemption (no completed scored run).
- [FH] crusoe + RDMA container setup (`--device /dev/infiniband` and/or IPC_LOCK+memlock) -> Triton Code 209 at kernel warmup in
  PD roles; without it the role comes up. Which of the two flags: split runs e7sg (devices only) and e7sh (lock only) on openai came
  up READY (no 209 on openai either way); not resolved on crusoe.
- [2H] llying 1P1D (P2D2) crusoe a4w4-prefill c640 486,737 tok/s/GPU; he reports P bound by MoE AllGather/ReduceScatter on one xGMI link.

## Wrong turns
- e7 at c512 without MC_DISABLE_HIP (warmup failures); first e7h retry on n02-25 with `/data` root (root-only).
- Mooncake TCP + MC_DISABLE_HIP on crusoe (e7th/e7tp) moved no KV; TCP with HIP (e7u) hit the same IPC 201.
- R2 next to other servers on crusoe (two node-wide stalls; stopped, then the stall reproduced without R2).

## Open
- Cause of the crusoe stalls (v3-014 exclusion pending with the user).
- A working PD on crusoe with this image; cross-node PD or Mooncake main (newer HipTransport, #4953/#3548) untested.
- 2P1D / prefill-side settings (e7p) never got a scored run (node cancelled). E8 go/no-go and E12/E13 rates still open.
- Whether llying's 486,737 reproduces on our kit/image.
