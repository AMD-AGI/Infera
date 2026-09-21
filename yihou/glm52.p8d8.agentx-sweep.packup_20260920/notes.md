# Notes — gotchas, wrong turns, error analysis

This is the distillation of `spec/poll_log.md`. Every number here points at a file in
this kit. The recurring theme across the whole run: **a signal that cannot fail is not
a signal.** Five separate times a reading the leader was leaning on turned out to be
structurally incapable of reporting the problem it was being used to detect — the
acceptance gauge, `rocm-smi --showproductname`, `/metrics` answering while `/health`
hung, the progress-line grep, and `grep -c`. Read every gotcha below as an instance of
that.

---

## 1. The acceptance gauge is forced. It proves nothing about correctness.

**What.** `SGLANG_SIMULATE_ACC_LEN=3.61` was on for the entire sweep of record. The
per-rank `spec_accept_length` gauge read **3.48 / 3.55 / 3.63** at c080 warmup
(`spec/poll_log.md:491`) and **3.52–3.78** at the c144 stall (`spec/poll_log.md:733`,
`results/sweep_results.yihou.md:95`) — straddling the configured 3.61 the whole time.
Meanwhile three probe requests sent straight at the router returned
`'1!au!au!au!au!…'` degenerate text (`results/sweep_results.yihou.md:86-90`).
**A gauge reading ~3.6 and garbled output are both true at once.**

**Why.** Simulated acceptance forces the *count* of committed draft tokens, not *which*
tokens are right. `generate_simulated_accept_index` under `match-expected` /
`real-draft-token` constructs an accept set of a chosen size; it decides how many, not
whether they are correct (`results/sweep_results.yihou.md:173-175`,
`spec/poll_log.md:896-902`). Force-accepting a token the target would have rejected
sends EAGLE at topk=1 down the draft's own branch and the error compounds. So the gauge
reports the value it was told to report, and the model still emits nonsense.

**How it bit.** The leader cited "acceptance holding near 3.61" as a *health* signal in
poll after poll (`spec/poll_log.md:488-497`, `645`, `985-988`) before discovering, at
the c144 stall, that the reading carries zero correctness information. That reading was
withdrawn (`spec/poll_log.md:735-742`). Correctness is waived **by construction** — the
same waiver the reference c32/c40 kit makes about its own `3.61` numbers
(`results/sweep_results.yihou.md:169-181`, `spec/poll_log.md:904-909`). The sweep's
timings stand on that footing; the acceptance number is not evidence of anything.

**Context.** The one real cross-check that IS meaningful: with simulation OFF the
input-class experiment predicts ~2.5–2.6 for this agentic workload
(`analysis/accept_length_is_input_driven.yihou.md:28,42`). Seeing ~3.6 under forcing
confirms the forcing is *active* — nothing more. It also means simulation **flatters**
this stack by ~1.4x on committed tokens per verify step, not penalises it
(`analysis/accept_length_is_input_driven.yihou.md:65`).

---

## 2. The withdrawn `SGLANG_DSA_FUSE_TOPK=0` workaround, the pre-registered prediction, and its falsification

**What.** `SGLANG_DSA_FUSE_TOPK=0` was introduced on decode as the version-translated
mitigation for the issue.md 3.3 memory fault (`--no-enable-dsa-fused-indexer` does not
exist in this image; the equivalent env is read from `environ.py:166`,
`dsa_indexer_kpool.py:748,941` — `spec/poll_log.md:554-563`). It stopped the crash. It
was later **withdrawn** because the deployment was found to be emitting garbled text
(`spec/poll_log.md:811-832`, `results/sweep_results.yihou.md:141-146`).

**Why the leader suspected it.** The garbling was first seen on the run that carried it,
and code reading found the variable's only two effects were to stop supplying
`page_table` / `topk_offsets` to the top-k transform, with no compensating branch, on a
path issue.md ties to upstream PR #36714 "PD DSA fused-TopK seed remap on ROCm"
(`spec/poll_log.md:764-782`). Plausible mechanism: disabling fusion removes the index
remap the PD+spec path needs, indexer selects wrong KV, model degenerates.

**How it was falsified.** The prediction was written down **in the crash kit before the
test** — "relaunch with simulation OFF and the workaround kept will still be garbled; if
it comes back coherent the hypothesis is wrong" (`spec/poll_log.md:780-782`). The actual
test restored the fused path fully (`SGLANG_DSA_FUSE_TOPK` unset, verified absent from
the decode container) and set `index_share_for_mtp_iteration=false` instead. **Still
garbled, same `1!au!au!au!…` signature, 3/3 probes** (`spec/poll_log.md:872-881`). The
hypothesis is dead. The three-deployment table settles it
(`spec/poll_log.md:885-891`):

| deployment | sim acc | fused topk | custom AR | IndexShare | output |
|---|---|---|---|---|---|
| A (09-18) | **OFF** | ON | ON | ON | **coherent**, real accept 4.69 |
| B (09-18) | ON 3.61 | OFF | ON | ON | garbled |
| C (09-19) | ON 3.61 | ON | ON | **OFF** | garbled |

**Simulated acceptance is the only variable aligned with the outcome.** Writing the
prediction first is what makes this a clean falsification rather than a story adjusted
afterward. Note the persuasive-but-wrong argument that had to be retracted: "real 4.69 >
forced 3.61, so forcing is conservative and cannot corrupt" — wrong, because 4.69 is how
many tokens *agree with the target*, while forcing picks a *count*, not correct tokens
(`spec/poll_log.md:894-902`).

---

## 3. The pre-launch GPU-VRAM gate, and the incident that produced it

**What.** The single worst wrong turn of the session. The leader tore down the probe
deployment, relaunched the sweep deployment **two minutes later**, watched startup jam,
killed it with `docker rm -f`, could not reap it (D state), and concluded "driver wedge,
node needs an admin reset" — telling the user so. The node then released all 8 GPUs to
0 % on its own within 40 s (`spec/poll_log.md:227-264`). The user caught it with one
question: *"这难道不是在清理 hicache?"*

**Why it happened.** Relaunching 2 min after a teardown whose GPU-memory release was
still in flight meant several ranks could not get their allocation, so startup stalled
with three GPUs still near-empty — genuinely jammed, not merely slow
(`spec/poll_log.md:293-308`). Three compounding errors:
- **Verified the wrong resource.** Checked host RAM and container absence on 137, and
  GPU VRAM on 136 — but never GPU VRAM on 137, the one that mattered
  (`spec/poll_log.md:252-254`).
- **Called a hang at 8 minutes** when the previous healthy deployment on identical
  hardware took ~22 min to reach health, a number the leader had and did not use
  (`spec/poll_log.md:255-256`).
- **"Proved" node health with a test that cannot fail.** `rocm-smi --showproductname`
  only *enumerates* devices; it never allocates. It was used to assert the node was fine
  while the real allocation test hung — which was the actual answer
  (`spec/poll_log.md:257-260`). Also ignored its own evidence: GPU6 moved 86 %→3 %
  between two samples; a figure that moves is not a wedge (`spec/poll_log.md:262-264`).

The user's hicache warning (*hicache 的释放需要时间，不要误判为异常/内存泄漏*) was in
`mission.md`, re-read every 10 minutes, and the leader still walked into it. **Re-reading
a rule is not applying it** (`spec/poll_log.md:266-269`).

**How it was fixed.** A hard gate: *before any launch, `rocm-smi` VRAM must read 0 on
every GPU of every node — not host RAM, not "the containers are gone"* — plus **no
"is it stuck?" judgement inside the first 30 min of a launch**
(`spec/poll_log.md:272-279`). Every subsequent launch passed the gate explicitly.

**Context — release timing is the trap underneath.** HiCache release takes **33 min then
40 min** for the same shape (`spec/poll_log.md:382-388`, `607-619`) and is **NOT
monotonic**: one GPU at a time, then a **~20-minute plateau at an unchanged reading**,
then everything at once (`spec/poll_log.md:389-393`). Cross-checked that the plateau was
not a stale metric — `--showmeminfo vram` agreed in absolute bytes
(`spec/poll_log.md:391-392`). Because 33≠40, the budget for release must be a **ceiling
decided by the gate, not a timer** (`spec/poll_log.md:617-619`). The lesson connects
directly to gotcha 8: a plateaued reading and a dead process look identical.

---

## 4. issue.md 3.3 reproduced on v0.5.19, and the mitigation

**What.** A decode DP rank died mid-run at **16:06:58**, `Memory access fault by GPU
node-3 … Fatal Python error: Aborted`, followed 2 s later by a same-rail
`transport retry counter exceeded` (`spec/poll_log.md:527-532`). This is
`bench/glm5p2_pd/issue.md` §3.3 "Decode gfx950 fused DSA indexer memory access fault",
which explicitly says v0.5.19 uses a different implementation and *需要先确认问题是否仍
存在*. **It still exists** — reproduced on a base (20260917) newer than the one that
document suggested trying (`spec/poll_log.md:546-552`).

**Why the causality was first read backwards.** When the rail-fault counter ticked 0→1
the leader saw `local ionic_1, peer …@ionic_1`, concluded "same rail, transient, not the
phase-1 cross-rail wedge", and moved on (`spec/poll_log.md:534-540`). Same-rail was
correct; the conclusion was not. The RDMA retry failed **because the peer rank had
aborted two seconds earlier** — the fabric was fine, the process was gone. This is why
the new standing rule is: on any non-zero rail count, **check the peer's `rocm-smi` and
abort log BEFORE reading it as a fabric event** (`spec/poll_log.md:645-646`).

**How the timing was pinned.** The fault-repro deployment — call it **B-fault**
(decode-ready 14:40:26; IndexShare ON at the model default, fused ON — both verified as
appearing **0 times** in the launch argv) — ran 14:40:26 → fault 16:06:58 = **1 h 26
min**, about 40 min into sustained load (`spec/poll_log.md:941-946`). This is **not** the
fused-OFF `B (09-18)` of the gotcha-2 table above: that one garbled with fused top-k
OFF, this one faults with fused top-k ON. `spec/poll_log.md` reuses the bare label "B"
for both, but they are different runs. That 86-min fault lands inside the peer session's
TP4 fault range
(32/39/54/71 min), so "TP8 is immune" is off the table
(`spec/poll_log.md:945-947`).

**The mitigation, and why it changed.** First mitigation `SGLANG_DSA_FUSE_TOPK=0` was
withdrawn (gotcha 2 — it garbled output; trading a visible crash for silent corruption
is the worse deal, `spec/poll_log.md:828-830`). Replaced with
**`index_share_for_mtp_iteration=false`**, which keeps the fused indexer. Verified in
`dsa/utils.py::should_use_dsa_fused_topk` before trusting it: with IndexShare off,
`pd_index_share_seed` is False, the function returns `SGLANG_DSA_FUSE_TOPK` unchanged, so
#36714's PD seed remap stays in play (`spec/poll_log.md:834-840`). Peer evidence:
IndexShare ON faulted in 4/5 runs; OFF ran 4.5 h continuously with zero faults. The
sweep of record then ran **13 h 33 m fault-free** with IndexShare off
(`results/sweep_results.yihou.md:370-375`).

**Context — an overclaim caught in the crash kit.** "DP1 was the only rank absent from
the final per-rank lines" was written twice, then **disproved** during verification: in
the 30 lines before the fault all eight ranks appear, DP1 included; after it, none do.
Rank-index-absence is an unreliable method — the slice determines the answer
(`spec/poll_log.md:628-634`). What survives as evidence of which rank died is the VRAM
hole (GPU[1] at 4 % while seven held ~88 %); the fault message names `GPU node-3`, which
does **not** match device 1 — ROCm KFD node numbering and HIP device index are different
namespaces, and the mapping was not captured. Left as an open discrepancy, not
reconciled by assumption (`spec/poll_log.md:636-639`).

**RESOLVED 2026-09-20 — the two names are the same GPU.** Measured inside the decode
container on `crsuse2-m2m-136`: KFD enumerates **two CPU nodes first** (`simd_count=0`
at KFD nodes 0 and 1), so the eight GPUs sit at KFD nodes 2–9 and **HIP device *i* =
KFD node *i*+2**. KFD node 3 and `torch.cuda` device 1 both resolve to PCI
`0002:00:02.0`. So `GPU node-3` and the GPU[1] VRAM hole are two independent witnesses
to one GPU, not a contradiction, and the rank identification is firmer than the
paragraph above allows. **Caveat:** measured 2026-09-20, same host / image / shape, but
**not** captured at fault time — it reconstructs a hardware property rather than
recording one, and the `+2` offset shifts on a host with a different NUMA-node count.
Full table and provenance: `../glm52.dsa-indexer-gpu-fault.packup_20260919/README.md`,
§"Which rank died".

---

## 5. `--max-running-requests` is a GLOBAL cap divided by `attn_dp_size`

**What.** The decode engine captured CUDA graphs only up to `bs=16` despite
`--cuda-graph-max-bs-decode 128`. 16 = 128/8 was too neat
(`analysis/max_running_cap.yihou.md:10-16`, `spec/poll_log.md:116-121`).

**Why.** Read from the running image, not inferred:
`pool_configurator.py:847` sizes the request pool as
`max_running_requests // attn_dp_size`, and `base_cuda_graph_runner.py:64` clamps the
capture list to that pool (`analysis/max_running_cap.yihou.md:20-36`). So
`--max-running-requests` is a **global** budget split across DP ranks: P8D8 at 16 per
rank, P4D4 at 32 — both capping at **128 total in-flight**.

**How it would have bitten.** The mission accepts a top-point OOM as an end state, which
presumes the top point stresses the hardware. Past 128 the server simply **queues** — no
error, no OOM — and the benchmark would report a number describing the scheduler's cap
that *looks like* a saturation curve. A plausible wrong answer, not a crash
(`analysis/max_running_cap.yihou.md:60-70`). P8D8 hits it at its 3rd sweep point (144).
Fixed by raising `*_MAX_RUNNING` / `*_GRAPH_MAX_BS` to **256** so the top three points
are not silently queue-limited (`spec/poll_log.md:216-218`).

**Context.** The saturation the sweep actually found (peak at/below 80) is **NOT** this
ceiling — 112 sits well inside 256, and the cap was raised precisely to rule this out
(`results/sweep_results.yihou.md:75-77`, `spec/poll_log.md:700-701`).

---

## 6. Config traps: two overrides silently cleared, and one engine/config cross-check

**What / why — variables cleared by sourcing.** `config.full.sh:90` is a plain
assignment that overwrites whatever was set before it. **Both**
`JSON_MODEL_OVERRIDE_ARGS` (which carries `index_share_for_mtp_iteration=false`) **and**
`DSA_TOPK_BACKEND` must be assigned **AFTER** sourcing `config.full.sh`, or they reach
the engine empty (`spec/poll_log.md:844-848`). This trap was walked into twice: it was
already documented for `DSA_TOPK_BACKEND`, and the leader still hit it again on
`JSON_MODEL_OVERRIDE_ARGS`. Caught only by **sourcing the config and reading the value
back**, which showed empty — not by reading the file.

**What / why — the engine/config cross-check.** `tools/agentx_env.py` cross-checks the
**live engine's** `max_running` against what the CONFIG declares and aborts on mismatch:
`agentx_env: ERROR: prefill-0: live max_running=256, config expects 128`
(`spec/poll_log.md:403-411`). The leader had raised max_running only on the `launch.sh`
command line, so the engine ran 256 while the config still said 128. Without this check
the run would have produced a number whose recorded configuration did not describe the
server that produced it — a silently mislabelled result, worse than a failed run.

**How fixed.** At the **source of truth**, not the call site: put
`PREFILL_MAX_RUNNING` / `DECODE_MAX_RUNNING` / `*_GRAPH_MAX_BS = 256` into
`config.yihou.p8d8.sh` so launch and bench read the same value; a per-command-line
override would have worked once and drifted next time (`spec/poll_log.md:413-418`).
Every setting was verified by **sourcing** the config rather than reading it.

**Context.** Same discipline caught the orchestrator's step-5 check still demanding the
withdrawn `SGLANG_DSA_FUSE_TOPK=0` be present — it would have aborted the very launch it
exists to protect (`spec/poll_log.md:849-851`). The house rule earned its keep: a
setting written into a config is not a setting the engine received — verify it **in the
container** (`spec/poll_log.md:604-615`).

---

## 7. Warmup cost grows with saturation because requests burn the full 1,800 s timeout

**What.** Warmup was **27 min at c080** (`spec/poll_log.md:474`), **~2 h 11 m at c192**
(`spec/poll_log.md:1059`), **~3 h 10 m at c256** (`results/sweep_results.yihou.md:384-386`).

**Why.** `AGENTX_WARMUP_REQUESTS_PER_LANE=10` is per lane and lanes = concurrency, so
warmup request count itself scales (800+1120+1440+1920+2560 = 7,840 across the sweep,
`spec/poll_log.md:361-363`). But the dominant cost past saturation is different: a share
of requests each sit in `KVPoll.WaitingForInput` for the full
`SGLANG_DISAGGREGATION_WAITING_TIMEOUT` of **1,800 s** before failing and releasing
(`spec/poll_log.md:1033-1036`, `results/sweep_results.yihou.md:384-386`,
`243-246`). So every point past saturation pays roughly one extra timeout period in
warmup. Budget wall-clock as a ceiling, not from `DURATION`.

**Context — the warmup extrapolation wrong turn.** Early on the leader timed only the
**cold first wave** (84 requests against a completely cold prefix cache and unbuilt JIT
caches) and projected that rate across all 10.5 waves — producing a 1–2 h / ~9 h estimate
and a proposal to cut warmup (`spec/poll_log.md:432-448`). The user reversed it: warmup
**accelerates** as the cache builds. The measured c080 warmup ramped from ~7/min to
~115/min and finished in 27 min — the estimate was wrong by ~4x
(`spec/poll_log.md:474-485`). Same root as gotcha 8: projecting an early, unrepresentative
sample onto the steady state.

---

## 8. A slow process and a stopped one look identical at the wrong sampling interval

**What.** c144's warmup was declared a "reproducible PD queueing collapse, c144 does not
complete", and the leader recommended stopping the sweep on that basis
(`spec/poll_log.md:993-1015`). **Wrong.** c144 entered profiling at ~16:40 (1,595/1,598,
errors 0); the recommendation was withdrawn before it was acted on
(`spec/poll_log.md:1017-1021`).

**Why the misread.** The leader sampled the returned-counter **ten minutes apart**, saw
it unchanged, and called it wedged. Sampling **four minutes apart** showed 1,478→1,510 —
a crawl at **~8 requests/minute** as timed-out requests drained
(`spec/poll_log.md:1023-1028`, `results/sweep_results.yihou.md:210-213`). "No change
between two samples" was read as "no progress" — the exact same shape as the VRAM-release
misreading in gotcha 3, where a 20-minute plateau was nearly read as a wedge.

**How it propagated.** It also undermines the earlier kill of run 1's c144 at 1,180/1,598
— that one may have drained too, given time. The leader cannot show it would have, and
can no longer assert it would not (`spec/poll_log.md:1030-1032`,
`results/sweep_results.yihou.md:215-217`).

**Context — what actually survives.** The mechanism is real and holds:
prefill-busy / decode-idle, requests queuing at prefill longer than decode will wait for
their KV (`results/sweep_results.yihou.md:248-261`). And the c144 ITL/interactivity being
*better* than c112's is the starved-decode signature, not a contradiction
(`spec/poll_log.md:1047-1054`). The correct diagnostic on the anomaly — check the peer's
liveness **before** the client counter — paid off: no dead rank, 8/8 GPUs at 87-88 %,
rails 0, all endpoints 200, prefill busy / decode idle (`spec/poll_log.md:1004-1010`).

---

## 9. Checkers that silently match nothing

**What / why — `grep -c` fires the error fallback on success.** The orchestrator aborted
a **correctly configured, healthy** deployment. `grep -c` prints `0` **and exits
non-zero** when it finds nothing, so the `|| echo 0` fallback appended a second `0`; the
variable became `"0\n0"` and failed the `!= "0"` comparison. **The error fallback fired
on a successful check** (`spec/poll_log.md:911-918`). Fixed with `; true` and `| tail -1`,
and the text gate gained an `ALLOW_GARBLED` switch since aborting on garbling is the
wrong default under simulation.

**What / why — the progress-line grep matches the wrong phase.** The progress grep
matched the warmup format (`returned=N/TOTAL`) and returned **nothing during profiling**,
which has no total — reading as "no progress" while the run was healthy
(`spec/poll_log.md:517-519`). Use the phase lines and the periodic statistics block
instead.

**What / why — thousands separators.** Progress lines print counts with thousands
separators — `returned=1,180/1,598`, `1,478→1,510`, `1,595/1,598`
(`spec/poll_log.md:714`, `1024`, `1019`). A regex expecting bare `\d+/\d+` matches
nothing once the counter crosses 1,000, so a checker built against small numbers silently
goes blind exactly when the run is large.

**How to guard.** The session's standing lesson, stated three-for-three at
`spec/poll_log.md:192-194`: **test a checker against known input before trusting its
verdict.** The probe-script defects in gotcha-adjacent work were all found the same way —
by going to the raw data instead of trusting the verdict line
(`spec/poll_log.md:184-194`): a coherence check that read `message.content` alone while
GLM-5.2's `--reasoning-parser glm45` routed the whole budget to `reasoning_content`
(printed GARBLED against 16 coherent traces); a `{**post, **loaded}` metrics merge that
let an all-zero 3-second scrape overwrite a good post-run one; and a sampler that died on
a `SyntaxError` and shipped a header-only CSV.

---

## Two more, recorded but not chased

**5 `InvalidInferenceResultError` in both the fused-off and fused-on c080 runs, identical
count.** Not the DSA fault (the clean run never crashed), not the phase-1 cross-rail
wedge (0 rail faults, same-rail pinning verified in argv). A third, low-rate failure path
exists and is unexplained. Logged as open; not investigated mid-sweep because that meant
interrupting a deployment the user asked to run through
(`results/sweep_results.yihou.md:46-51`, `spec/poll_log.md:674-679`).

**The prefix cache persists across all five points on the one deployment**, so point 1
(c080) runs colder than point 5 (c256) — disclosed as a real confound of the
one-deployment choice (`spec/poll_log.md:367-368`). The one-deployment choice was kept
anyway: it means one HiCache flush at the end rather than five 33–40 min releases
(`spec/poll_log.md:364-366`).
