# Teammate poll log

Cadence: every 20 minutes. Rule: on the FIRST poll where a teammate looks
off-track or under-informed, only record it here. Intervene only if the same
problem is still present at the NEXT poll.

---

## 2026-09-18T04:27Z — Poll 1

**Mission step:** end of Step 1. Step 2 (preflight) is gated on the image
build finishing.

### image-patch — on track

- Step 1 (port `99fa0406`) complete and independently reviewed by the leader:
  `-e "INFERA_PD_DP_RANK_AFFINITY=$pd_dp_rank_affinity"` confirmed present at
  `launch.sh:136`, correctly placed after `--network host`. Without it the rust
  half would be dead code.
- Ran `cargo check --release --bin infera-router` *inside the real base image*
  rather than skipping the check for lack of a local toolchain. Exit 0. Good
  call — this caught nothing, but it was the right place to spend five minutes
  before a ~45-minute build.
- Image build running in background, teed to `notes/build.log`.

**Recorded, not yet escalated:** the slot-8 claim is currently supported by
*pre-patch base* evidence only. The failure mode that matters — a patch that
applies, compiles and passes the bytecode markers while reading
`prefill_cuda_graph_max_prefix_len` as a boolean — can only be excluded from
the POST-patch image. Asked for the underlying evidence and for the post-build
confirmation. Not an off-track finding yet; the teammate has not claimed the
post-build step is done. Re-check at Poll 2.

### cluster-prep — on track, one correction accepted

- Node prep complete. Leader verified first-hand: GPU 4-7 on both 135 and 138
  sit at ~297 MB (driver baseline). 138 fully clear across all eight.
- `check_nodes.sh` exits 1 solely on 135 GPU[1] (the root k8s pod).
  Thresholds were **not** relaxed to force a pass — correct behaviour.
- Base image digest matches on both nodes:
  `sha256:21c1cc9ab9b703cfe2bf4d62d5c2028654e5f54c810890d9d4b2019d1c31ca32`.
- Refused to kill the root-owned k8s `vllm serve Qwen3-32B` pod on 135
  (no privilege, live workload, kubelet would restart it). Correct refusal,
  and it checked the premise instead of executing on it.

**Correction the teammate raised, accepted:** the leader's HOLD on the 138
container arrived *after* the stop had already executed (finishedAt
2026-09-18T04:10:09Z). The teammate declined to record it as "held" because
that would have been false. That is the right call and is now reflected in the
user-facing report and in `evidence/NOTICE.md`.

Decision issued: do **not** cold-restart limou's container — it would not
recover the 23 h warm state and would re-grab all eight GPUs on 138. Container
left `Exited`, not removed.

### Blockers

None that stop the mission. The 135 GPU[1] k8s pod is permanently out of scope
by design (devices 4-7 avoid it); it is no longer treated as a blocker.

---

## 2026-09-18T05:28Z — Poll 2

**Mission step:** Step 3 (launch) in progress. etcd up on 135; `prefill-0`
(135) and `decode-0` (138) both started with `GPUs=2,3,4,5 TP=4 EP=1 DP=4
DPA=1 HiCache=0`; AITER JIT compiling for the first time on this image.

### Teammates

**None running.** Both `image-patch` and `cluster-prep` completed and stood
down before this poll. Nothing to intervene on.

Closeout confirmed for both:
- `image-patch` — port, build, slot-8 (two-sided evidence), provenance,
  distribute, verify all green. Its distribute/verify had in fact completed
  before my "No such image on 138" check; the two crossed in flight and the
  re-verify settled it. My check was stale, not its report.
- `cluster-prep` — node prep, evidence, NOTICE.md, READY.md.

### Poll-1 item, now closed

Poll 1 recorded that the slot-8 claim rested on pre-patch evidence only. That
is resolved: two independent post-patch confirmations (leader's and the
teammate's) agree that pack index 8 == read index 8, with slot 7 unchanged.
No escalation was needed.

### The significant event since Poll 1 was mine, not a teammate's

The user challenged the launch approach ("启动方式不对") after preflight failed
on GPUs 4-7. They were right, and the challenge was worth more than the
finding I was about to report:

- I had been about to report rails 4-7 as **broken hardware** and to work
  around it by shrinking `mem_fraction_static` or dropping to two GPUs.
- The prior fleet RDMA survey plus a pinned-NIC probe showed the rails are
  **fine**. Mooncake's auto-discovery was letting the two ends select
  different HCAs from the NUMA-local pool; rails are physically isolated, so a
  mismatch is unreachable rather than slow.
- Pinning both ends to one NIC: 16/16 verified on each of ionic_2/3/4/5.

Lesson recorded deliberately: a symmetric, cleanly-grouped failure looked like
a hardware fault and was a configuration default. The tell I initially
under-weighted was that the CPU-DRAM RDMA path passed on the same NICs — if
the rails were dead, that would have failed too. Check the cheap contradicting
evidence before naming hardware.

---

## 2026-09-18T05:56Z — Poll 3

**Mission step:** Step 5 — AgentX C32 running, AIPerf in PROFILING, warmup phase.

### Teammates

`image-patch` and `cluster-prep` both **idle**, work complete since ~1h ago.
Nothing to poll, nothing to intervene on. No teammate has been off-track at any
poll so far.

### Scope change since Poll 2 — user decision

The user waived correctness and directed straight to performance: *"不允许关
mtp，先测性能，这种边界情况不要管"*. The R02 experiment (MTP off) had already
launched; it was stopped and the performance profile relaunched instead. The
debug loop is halted, not abandoned — R01's evidence is preserved.

What was established before the waiver, and is now recorded in mission.md's
acceptance section so the periodic injection reflects reality:

- With simulated acceptance OFF, decode output is garbled on this P4D4 shape.
- First token correct, all later tokens degenerate → fault is on the decode
  side, and numerical rather than tokenizer/parser.
- Mooncake failure count 0 → the transport layer this run exists to fix is
  clean, and is ruled out rather than left as a suspect.
- Measured MTP accept rate 0.05 against a simulated stand-in of 3.61.

Recorded so the deliverable is not over-claimed: the benchmark runs with
`DECODE_SIMULATE_ACC_LEN=3.61`, which forces acceptance length and overrides the
real 0.05. The numbers are a valid timing comparison against other runs using
the same simulation; they are not evidence of correct decoding.

### Watch item (mine, not a teammate's)

Warmup showed `returned=0/66, in_flight=34, errors=0` at 90 s. Checked rather
than assumed: prefill is genuinely saturated — 8192-token chunks, ~1.5 M pending
tokens per rank, 6-7 queued requests — and decode has not emitted a batch line
yet because the first handoff is still in progress. Consistent with AgentX's
long agentic contexts, not a hang. Re-check at Poll 4; escalate if `returned`
is still 0 once warmup elapsed passes a few minutes with no decode activity.

---

## 2026-09-18T06:16Z — Poll 4

**Mission step:** Step 5 — AgentX C32 in the profiling phase.

### Teammates

Both **idle** and finished. No teammate has been off-track at any poll in this
run; nothing to record against either, nothing to intervene on.

### Poll-3 watch item — closed

Warmup's `returned=0/66` was not a hang. It drained cleanly:
`returned=59/66, sent=66, in_flight=7, errors=0` at 570 s, and profiling
started. Not escalating was the right call; the cause was simply AgentX's long
agentic contexts against a cold prefix cache.

Correction to my own Poll-3 method: I checked decode liveness with
`docker logs --tail 300 | grep -c "Decode batch"` and read 0. That count was
meaningless — those 300 lines were entirely `/metrics` polling. The tail window,
not decode, was the reason for the zero. Any real batch statistics must come
from the full log.

### Live health during profiling

| metric | value |
|---|---|
| errors | **0** (23 progress lines, all `errors=0`; the 24 "error" greps are field names, not failures) |
| server prefix-cache hit | 92.9-95.7% against a 96.1% theoretical ceiling |
| KV usage | 59-60% |
| throughput | in ~132-155 k/s, out ~463-524/s |

### New watch item — prefill HBM (issue.md §3.4)

Prefill GPUs 2-5 sit at **90-91% VRAM**. For context, §3.4's failures happened
at 99% and the allocator-GC treatment brought the peak down to 93%.

Deliberately **not** concluding from one sample: 90% may be the steady-state
working point, or it may still be climbing. Only repeated sampling separates
those, and §3.4 is explicit that this failure mode accumulates gradually and is
invisible to a pre-launch snapshot. Sampling across the rest of the run;
re-check at Poll 5.
