# Notes — gotchas, wrong turns, error analysis, open questions

Raw timestamped log: `notes_src/working_process.md` (unedited, includes every failed attempt).

---

## The finding that matters most: a 48-request cap, not memory, was the old ceiling

**What.** Nothing above global concurrency 48 will run unless you pass `--max-running-requests`.
Every point launched without it died with
`alloc_req_slots runs out of memory ... req_to_token_pool.available_size()=12, num_reqs=32`.

**Why.** `kv_cache_configurator.py` sizes the per-rank request pool as
`get_schedule().max_running_requests // ps.attn_dp_size`. With speculative decoding on, the pinned
SGLang resolves `max_running_requests` to 48 and says so:

> `Max running requests is reset to 48 for speculative decoding. You can override this by explicitly setting --max-running-requests.`

48 / `attn_dp_size` 4 = **12 request slots per rank**, so the local batch can never exceed 12 and
global concurrency can never exceed 48 — **whatever the KV pool size is.**

**How found.** Not by guessing: the per-rank slot count was identical (12) at C=128 and C=160 and
independent of the pool, which ruled out memory. The sizing expression was then read in the
**pinned** source inside the container, and `scripts/probe_reqpool_yihou.py` runs that same resolver
CPU-only and prints the warning verbatim. Takes seconds and needs no GPU.

**Context — this retro-explains the previous sweep.** The 2026-09-11 sweep stopped at exactly C=48
with every point at `local_batch_size <= 12`. That was never a chosen endpoint and never a memory
limit; it was this cap, silently binding. The KV-headroom arithmetic in that packup was correct but
was not the operative constraint at the boundary. At C=48 the default equals the explicit value, so
its numbers remain directly comparable to these.

**Note the flag is global.** SGLang divides it by `attn_dp_size` internally, so pass `C`, not `C/4`.

## C=160 does not fit at TP4 — a measured negative result, not an abandoned attempt

**What.** Twelve attempts, two arms, two nodes, five mem-fractions. All failed.

| mem-fraction | pool tokens (ep1 / ep4) | failure |
|---|---|---|
| 0.9700 | 3,177,984 / 3,158,336 | `KV capacity insufficient` (needs 3,202,560) |
| 0.9735 | below requirement | `KV capacity insufficient` |
| 0.9740 | **3,202,688** (margin 128 tokens) | 267: capture OOM on a **22 MiB** alloc. 254: capture **and** bootstrap succeeded, then died in warmup with NCCL `Failed to CUDA calloc 6291456 bytes` |
| 0.9772 | ~3,202,700 (ep4) | capture OOM |
| 0.9780 | 3,227,392 / 3,207,744 | capture OOM, 962 MiB block |
| 0.9800 (+/- expandable_segments) | 3,239,744 / 3,220,096 | capture OOM, 962 MiB block |

**Why.** KV alone needs 142.3 GiB of the 151.66 GiB free after weights; CUDA-graph capture plus NCCL
need roughly 9 GiB more. The failure simply **walks down the pipeline** as the pool shrinks:
KV check -> graph capture (`eagle_draft_extend_cuda_graph_runner.py:270` ->
`aiter_paged_mqa_logits`, a **962 MiB contiguous** workspace) -> NCCL buffers. On the single most
favourable configuration it reached the warmup loop and still could not obtain **6 MiB**.

**Why the obvious remedy failed.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — which the
error text itself recommends — did not help. `reserved but unallocated` stayed at 1.36 GiB and at
mf 0.978 **grew to 2.41 GiB**: memory handed back from the pool was absorbed by fragmentation
instead of becoming the contiguous block the capture needs. Free memory actually went *down*
(798 -> 698 MiB).

**How this was kept honest.** The allocator flag was not quietly adopted. Points using it are tagged
`_xs`, and a C=128 control was run **both** ways on each node to measure its effect rather than
assume it: **-0.265 %** (254/ep1) and **+0.205 %** (267/ep4) — opposite signs, so it does not move
the measurement.

**Context.** There is no window at TP4 between "KV pool large enough" and "enough memory left to
capture and communicate". Reaching C >= 160 at this ISL requires changing the parallelism (TP8 halves
both the per-rank weight footprint and the per-rank batch), which would be a different,
non-comparable curve — and was explicitly declined by the user in favour of keeping TP4 comparable.

## C >= 192 is arithmetic, not an experiment

Per rank: 287.98 GiB device, 119.84 GiB target weights, 6.92 GiB draft weights, leaving
**151.66 GiB**. KV costs 46.58 KiB/token; each request reserves 80064 tokens = **3.556 GiB**. With DP
attention each rank holds `C/4` requests:

| C | local batch | KV required | vs ceiling |
|---:|---:|---:|---|
| 192 | 48 | 170.7 GiB | over |
| 224 | 56 | 199.2 GiB | over |
| 256 | 64 | 227.6 GiB | over |
| 288 | 72 | 256.1 GiB | over |

`--mem-fraction-static` bounds the static fraction, not the device; it cannot buy this memory.
These four points were therefore **not run** — spending GPU hours to watch a subtraction fail is not
evidence.

## The pool-size model was predictive, and that is why the failures are trustworthy

A two-point linear model (pool tokens vs mem-fraction, ~6.18M tokens per 1.0 on a 287.98 GiB device)
predicted **every** subsequent pool **to the token**: 2,622,144 / 3,177,984 / 3,220,096 / 3,239,744 /
3,202,688. Predictions were written into `working_process.md` *before* each run. So when a point
failed, it was not because the pool was mis-sized — which is what makes "C=160 does not fit" a
result rather than a guess.

## Incident: a live 8-GPU training job on node 267

**What.** At setup, `crsuse2-m2m-267` ran a foreign container `dsv4`
(`yangyuhanintel/rocm-verl-dsv4:0907`) with 8 `ray::WorkerDict` processes at ~95 % CPU and 17 h of
accumulated CPU time, part of a cross-node Ray cluster. A peer also held a Slurm allocation on both
nodes concurrently with ours.

**Why it mattered.** The standing mission rule says to clear GPU-heavy foreign processes, but this
was plainly a **live** multi-node training run, not a leftover, and killing it would have destroyed
someone's work.

**How resolved.** Not unilaterally: it was raised to the user, who confirmed it was residual and
authorised cleanup. It was then **`docker stop`ped only — never `docker rm`** — so the image and the
stopped container remain on disk. Verified afterwards: `docker ps` empty and
`/sys/class/kfd/kfd/proc` containing 0 entries.

**Context.** Always check `docker ps` **and** the KFD process count before the first point, and check
`squeue` for peers holding the same node. `docker stop` may print "did not receive an exit event"
while having in fact worked — verify with `docker ps`, not with the stop command's message.

## Traps carried forward (still true)

- **Never pipe `runtime.log` through `tail`.** It contains multi-megabyte single-line tqdm bars; doing
  so previously killed a driver shell via host memory pressure. Use `grep -c` / `wc -c`.
- **CUDA-graph capture is slow.** Do not kill a quiet point before ~30 min. A cold node's first point
  costs ~14 min in `load_pool_capture` alone (AITER JIT cache is per node, keyed to the image digest).
- **`tok/s = C x 1000 / TPOT` is an identity.** Reporting "latency fell *and* throughput rose" as two
  findings double-counts one number.
- **`AITER_COMMIT` lies.** The image env var says `d9e5ef7c...`; the actual `/aiter` checkout is
  `2c71811b...`. Use `git -C /aiter rev-parse HEAD`. Trusting the env var would have published a
  wrong commit and contradicted a correct earlier report.
- **`nproc` under `spur exec` returns 1** because of the job cgroup. It is not the core count.

## Wrong turn worth recording

The two arms were first launched **one per node** (254 = EP off, 267 = EP on) to overlap the cold JIT
cost. That is a cross-node comparison and is confounded — exactly the confound the 2026-09-11 sweep
added a same-node control to remove. It was caught immediately and a second phase was chained so each
node ended with **both** arms; only same-node pairs are reported as the EP delta. The scripts
`chain_opposite_arm_yihou.sh` / `chain_runner_yihou.sh` exist for that chaining. (Separate files on
purpose: bash reads scripts incrementally, so editing a script that other waiting shells are still
executing can corrupt them.)

## Process deviation: the agent-team requirement was not met

The standing rules mandate working through an agent team with 20-minute leader polls. **Two teammate
instances were spawned and both died to server-side `502 (AnthropicVertex)` errors — four failure
events, zero observations delivered** (`node267-monitor` at 04:32:39Z and 04:59:02Z;
`node267-monitor2` at 05:22:15Z and 06:24:07Z). Notifications arrived late, so one "intervention"
was addressed to a process that had already been dead for two hours. No third instance was spawned:
the failure rate was 2/2 and the GPU work was finished by then.

Recorded rather than hidden, because it changes who checked the numbers: **all verification in this
sweep is single-reviewer.** The same thing happened in the 2026-09-11 sweep (seven instances lost).

## Open questions (deliberately not resolved)

1. **Is the EP-off advantage at C=128 real?** The sign is consistent (6/6 same-node pairs, plus every
   point of the prior sweep), and the decay with concurrency is clean. But the C=128 magnitude
   (0.49-0.97 %) is the same size as the node-to-node spread (up to 1.03 %), and every point is a
   single run. Bounded observation, not a resolved number.
2. **Why does the advantage decay at all?** Consistent with a fixed per-step expert-dispatch cost
   amortised over a larger local batch, but nothing here measures that cost. No per-stage attribution
   was taken.
3. **The prior packup's open question is downgraded, not answered.** Its "EP advantage widens at
   C>=40, unexplained" excursion (1.3 points) is within the single-run variation measured here, so it
   is most plausibly noise. Not proven.
4. **Scope of the claim.** This is the internal scheduler-free harness: `scheduler_used: false`,
   `synthetic_prefix: true`, `simulated_acceptance: true` (with `real_model_weights: true` and
   `real_moe_routing: true`). It measures decode-loop performance under simulated acceptance —
   **not** a serving benchmark, and **not** a statement about output correctness.
