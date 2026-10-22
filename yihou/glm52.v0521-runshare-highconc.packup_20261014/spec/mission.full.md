# MISSION — GLM-5.2 v0.5.21 1P1D P4D4: more prefill + decode capacity, CONC 44 → 48, > 23k tok/s/chip

Re-injected every 10 minutes. This file is the task of record.

## CURRENT DIRECTIVE (user, 2026-10-12 ~06:25 UTC) — verbatim

> 1. 我们有关于大内存hicache的成功实验（450G）2. 我们有你刚整理的关于decode提升kvcache空间的报告。综合应用两者，
> 提高prefill和decode的容量，依次加大conc到44或48（先在44上调试实验保证>= 40）, 设法提升吞吐到23k以上
> (两台机器并行调试，加快工作，降低实验成本：1. 一台oai的机器。2. 137-139的任意一台)。
> 另外，同时详细调研配置decode侧的方法，保证running req永远有至少一半的kvcache使用空间（大概率是限制transfer
> req的数量，保证prefill不因为自身太慢，而过多霸占decode的预分配内存池）。有明确结论后，应用到前者的实验成果上。

English (working reading):

1. **Track A — capacity + concurrency → > 23k tok/s/chip.** Combine the proven prefill HiCache 450 GB result with the
   decode KV-capacity report (`/home/yihou/dev/git/infera.glm52.view/decode.kvcache.report.md`): raise prefill and
   decode capacity, then raise AgentX CONC to 44, then 48. **Debug at C44 first; C44 must score ≥ the C40 result**
   (22,114/chip, e2g A2). Goal: **> 23,000 tok/s/chip** (AgentX full, 3600 s, warmup 10/lane, same target config:
   triton DSA both legs, EAGLE MTP 5/1/6, index_share false unless proven).
2. **Two machines in parallel** to go faster and cut cost: (1) the oai node `crsuse2-m2m-102` (spur job 207495,
   `amd-oai`), (2) one of `crsuse2-m2m-137..139` (06:30 check: 137 idle — 8 GPUs 0 VRAM, only a `sleep` dev
   container; 138 busy with limou's vLLM; 139 has no dockerd and GPUs 4-7 partly used). Chosen: **137**.
3. **Track B — decode admission policy.** Research in detail how to configure the decode side so that **running
   requests always keep at least half of the decode KV pool** — most likely by limiting the number/tokens of
   requests in prealloc/transfer, so a slow prefill cannot hog decode's pre-allocated pool. Reach a clear,
   evidence-backed conclusion (code + logs + experiment), then **apply it to Track A's best result**.

## UPDATE (user, 2026-10-12 ~08:15 UTC) — supersedes the node choice above

- "别用13x洗了机器了，用oai的机器，申请两台用": do NOT use 137-139. Use **two amd-oai nodes**: 102 (job 207495) + a second
  hold, job 208970 (`yihou-oai-cap2`, -x 102, 24 h) — PENDING (QOSGrpNodeLimit, amd-oai-qos node=8) at 08:13. The 137
  image load was stopped (no image left tagged on 137, no containers).
- "尽快实验一些优化，比如把hi-cache开到360，decode的gmu调到0.95": round **k1** on 102 = e2g recipe with prefill
  --hicache-size 360 (decode mem 0.95 is already the setting), CONC 44, warm-stack protocol (A1 fresh -> flush -> A2).

- (user, 08:2x) "hi-cache开375是有额外配置的": searched all yihou packups — no 375/360 anywhere; the only extra config
  for HiCache > 256 GB is SGLANG_HICACHE_HOST_REGISTER_CHUNK_GB=512 (H1; k1 has it) + memlock/IPC_LOCK (harness default);
  optional unused PR #42553 (hipHostMalloc GTT). Asked the user whether 375 / another config is meant.
- (user, 08:2x) **LAST PRIORITY**: finally test index_share_for_mtp_iteration=true (user: false has a perf regression;
  v0.5.19 succeeded with true). Evidence: e2d (true, fresh, C40) 21,383 vs fresh false runs 19.3-20.4k; risk: C44 MAF
  history (H2). Run it after Track A/B, under the warm protocol, ideally at the best config.

- (user, ~09:00) max_running 64→96 is important. Track B patch per user rule: admit only if
  (candidate + held KV) − running KV < 50 % of the pool. Patch + debug on a separate node (burst or oai, job 209081 /
  208970) in parallel, until C40 runs with no perf regression.

- (user, ~10:50) "135-139挑一台能用的用": 10:50 survey — 135 GPU1 used by others; 136/138 yaoc stacks; 139 no dockerd;
  **137 idle** (only dev_xb sleep) -> chosen. /shared_nfs is read-only on 137 -> harness NODE_RW_ROOT
  /mnt/m2m_nobackup/yihou_v0521dbg. Image loaded there = ID sha256:e8e2838f (containerd store re-IDs the same content).
  Burst nodes abandoned (amd-burst-qos PreemptMode=cancel: both jobs cancelled at 20 min).

## Baseline to beat (first-hand)

- e2g A2 (warm-stack protocol, C40): **22,114 tok/s/chip**; warm replicates 21,921 / 21,696 / 21,681; first runs on a
  fresh stack 19.3-20.4k. Recipe + kit: `/home/yihou/dev/git/infera.glm52.view/glm52.oai-1p1d-v0521-agentx-c44_20261009.packup_20261012/`.
- C44 on v0.5.21 so far: e2c C44 164,194 tok/s = 20,524/chip (fresh stack, first run).
- **Score = the 2nd AgentX run on a stack after `/flush_cache` (warm-stack protocol)**; compare like with like.

## Rules (MUST, at all times)

- Re-inject this mission every 10 min. Leader polls the team every 20 min (first sign of a problem → record only in
  `debug/poll.md`; intervene if the next poll shows it unresolved). Watchdog every 10 min while a service/bench runs.
- Ultracode on: orchestrate substantive work with workflows; verify adversarially. Each experiment tests several
  hypotheses at once.
- Measure, don't assume; rank evidence (first-hand vs second-hand); suspend, don't conclude.
- Work in containers; host actions are docker / ssh / spur shim only. serena/LSP for code.
- **Never delete a file whose path lacks `yihou`. Never disturb other users' containers/GPUs.** 137 is shared: gate
  every launch on all 8 GPUs idle twice 60 s apart and no foreign GPU containers; never stop a foreign container.
- `/home/yihou/dev/git/infera.glm52.pd` is READ-ONLY.
- English for work; Chinese only for reports to the user. Ask the user before deviating from this directive.
- Never edit a running bash script in place (write new + mv). Drivers run detached (`setsid nohup … < /dev/null`).
- VRAM drain after a HiCache-450 stack stops: 28-35 min; gate on idle.

## UPDATE (user, 2026-10-13 ~05:15 UTC, new session on another machine) — verbatim + reading

> 可以，那你申请2-3台oai机器开始吧，如果机器不够的话，可以在135-139之间找一个空闲的来凑（limou占了一个机器一直在干啥，好几天了，没有真实利用率给他杀了）

- Got 3 amd-oai holds (05:06): 211169 crsuse2-m2m-085, 211170 crsuse2-m2m-026, 211171 crsuse2-m2m-274 (24 h, gpu:8 -c 160,
  8x MI355X verified via spur exec). 135-139 not needed; limou's node untouched. 085/274 carry guanchen's orphan container
  crusoe-sgl-q36 (pool_mtp.sh loop that can relaunch sglang on GPU0-5; his burst jobs were preempted) -> asked the user.
- Drivers: orchestrate from the new machine over ssh; long drivers detached ON the compute node via spur exec (user agreed).

> 并行展开一下实验：1. 大hi-cache 大conc的我我不理解为啥没有提升，因为之之前强行8 expert开大kv cache是可以到26k, 你去并行/串行做两个实验a.单独让decode使用8 expert腾出更多的kv-cache. b. 让prefill单独开expert, 获得更大的kv-cache观察下那边有提升，看看谁是瓶颈(conc就按昨天实验最高的我记得是52？8 expert怎么玩你可以去~/dev/git/infera.glm52.pd/yihou底下的packup找)。2. 关于decode限制transfer和prealloc cache那个事情。实验验证一下直接用最初conc=40那一版的hi-cache容量压conc=56/64, 期望是比~/dev/git/infera.glm52.pd/yihou之前sweep的性能回退少一点。

Reading:
1. Bottleneck isolation with the synthetic 8-expert model (yihou-expert8-pd packup): (a) decode leg only on 8 experts
   (frees weight memory -> bigger decode KV pool); (b) prefill leg only on 8 experts (bigger prefill KV / cache).
   Whichever gains shows the bottleneck side. CONC = the highest from yesterday (C52; best score was C48 -> run both on one stack).
2. Track B at high load: runshare v2 (decode in-flight transfer/prealloc KV cap) on the original C40 config HiCache
   capacity, pushed to C56 / C64; expectation: less regression than the earlier sweep in the yihou packups.
