# Notes — decisions, traps, and the things that went wrong

Written as what / why / how so the *reason* survives, not just the instruction.

---

## 1. Why the uniformity assertion was there, and what replaced it

**What.** `topology.py:69` used to read:

```python
valid = valid and len(set(accept_lens)) == 1 and len(set(previous_lens)) == 1
```

Only the `previous_lens` half was removed. The `accept_lens` half was kept.

**Why.** Those two clauses look symmetric and are not. The acceptance one is **true by upstream
construction** — `spec_utils.py:411-419` draws a single scalar `simulate_acc_len` per iteration and
broadcasts it with `sim_accept_index[:, :simulate_acc_len]` — and it is what makes
`realized_accept_length` bit-reproducible across runs, which is this harness's correctness gate.
Deleting it along with the other would have cost a real check for nothing.

**How you would notice if this were wrong.** Two runs at identical parameters would stop agreeing on
`realized_accept_length`.

---

## 2. The digest, and the bug the old check could not see

**What.** Signature fields 3 and 5 changed from `previous_lens[0]` / `new_lens[0]` to
`lens_digest(...)`, an order-sensitive rolling hash of the whole vector.

**Why.** The old check compared **element 0 only**. It could not detect a divergence in any other
position. Under uniform ISL that was harmless because every element was equal anyway; the moment
lengths vary it becomes a check that mostly does not check. Verified by hand that
`lens_digest([70,800,90])` differs from both `[90,800,70]` and `[70,801,90]`, while the old
element-0 field was identical for all three.

**Why order-sensitive is correct here.** Plan 1 guarantees every rank holds the same multiset *in
the same order*, so an ordering divergence is a genuine bug and we want it caught, not tolerated.

---

## 3. The float that would have silently produced the wrong batch

**What.** The bimodal ratio is parsed as an exact `fractions.Fraction`, not a float.

**Why.** `ceil(70 * 0.1)` is **8**, not 7, because IEEE gives `70 * 0.1 == 7.000000000000001`. The
realized batch would then have disagreed by one request with the spec string recorded next to it in
`config_yihou.json` — a result file that quietly misdescribes itself. Invisible to any test that
happens to use a "nice" ratio.

**How.** `Fraction("0.1") * 70 == 7` exactly. Side benefit: `bimodal:8192,70000,1/10` also parses.

---

## 4. Clamping saturates; it does not resample

**What.** `normal:` draws are rounded and clamped to `[lo, hi]` by saturation.

**Why.** Resampling would make the number of RNG draws depend on the values drawn, so any future
change to the bounds would silently reshuffle the whole sequence. Saturation is reproducible, and
its distortion is *visible*: `describe()` reports `clamped_low` / `clamped_high` counts, so a caller
can see when the clamp ate the tail rather than discovering it in a plot months later.

---

## 5. The generator must not touch a global RNG

**What.** `IslSpec.generate()` owns a private `random.Random(seed)` and there is a test asserting
`random.getstate()` is unchanged across a call.

**Why.** The acceptance coins come off the module-level `random` stream. Consuming a single value in
the ISL generator would shift that sequence and change `realized_accept_length` — i.e. it would
break the correctness gate in a way that looks like a real behavioural change rather than a bug in
instrumentation. This is a constraint, not a style preference, and the module docstring says so.

---

## 6. Resolve once in the parent, not once per rank

**What.** `resolve_input_lens()` runs before `mp.spawn`; the resulting list is pickled to all ranks.

**Why.** The original design had each rank recompute from the same seed. Same result, but this way
rank agreement is true *by construction* instead of true *by argument*. Anything that could ever
make one rank's RNG diverge — a library version, a lazily-imported module seeding itself — stops
being able to cause a subtle cross-rank mismatch.

**Bonus.** A malformed spec now fails before `mp.spawn`, so a typo costs a second rather than
~30 minutes of model load and CUDA-graph capture.

---

## 7. A mixed batch does not save KV. Say so out loud.

**What.** Allocation is uniform-max: `batch.prepare_for_extend()` runs *before* the per-request
prefix truncation, over rows sized for `max(ISL)`.

**Why it matters.** The intuitive expectation is that a `{8k, 70k}` mix is cheaper than all-70k.
It is not — it reserves exactly the same KV. If the all-70k batch does not fit, neither does the
mix. Anyone sizing an experiment on the opposite assumption will be confused by an OOM.

**Fixing it** means changing the allocation path, which was explicitly out of scope.

---

## 8. `input_len` is `null` for a mixed run, deliberately

**What.** `summary()["input_len"]` is the scalar only when every entry is equal; otherwise `None`,
with `input_lens` and `input_len_uniform` carrying the truth.

**Why.** Downstream tooling reads that field as "the ISL of this run". For a ragged batch, any
single number there — mean, max, first — is a false statement inside the result file itself. A
`None` that forces the reader to look at `input_lens` is better than a plausible number that is
wrong. `verify_point.py` gained `--expect-heterogeneous` to check the ragged case properly.

---

## 9. A shared host ignores the scheduler

**What.** On 2026-09-15 at 11:30 UTC the node was fully occupied by three workloads (MLPerf Flux
training via `torchrun --nnodes=4`, someone else's `profile_decode.py`, and inductor compile
workers) **while Slurm showed the whole node, all 8 GPUs, allocated to our job 30723.**

**Why it matters.** `squeue` saying the node is yours is not evidence the GPUs are free. Containers
started outside Slurm hold `/dev/kfd` regardless. Always check `rocm-smi --showpids` before
trusting a number; a run that contends is not a measurement.

**How it was resolved.** The user authorised preempting non-Slurm work on a node Slurm shows as
ours. The condition was verified first, then `docker stop -t 30` on two **named** containers —
not `docker rm`, not `kill -9`, and not a loop over "everything that is not ours". SIGTERM with a
grace period so a training job can checkpoint; both containers survive and can be resumed with
`docker start`. Full pre-action state, enough to notify and restart the owners, is in
`results/preempt_evidence_20260915-1140_yihou.txt`.

**Recorded, not acted on:** the Flux container was rank 3 of a four-node job, so stopping it also
ended work on three machines our allocation says nothing about. That was surfaced to the user
before they authorised.

---

## 10. Two mistakes made during this work

**The first smoke run died in zero seconds** because I chose `--batch-size 8` at dp=8, giving
`local_batch_size = 1`, and a bimodal spec cannot split one request. The error was correct and
useful:

```
bimodal:2048,8192,0.25 degenerates at count=1: 1 request(s) of 2048 and 0 of 8192.
A bimodal batch that holds only one length is a uniform batch reported under a false label;
raise count or move the ratio toward 0.5.
```

Worth keeping as a story, because it is the design working: the failure happened at parse time in
the parent, before any GPU work, rather than after a 30-minute startup. It also incidentally proved
that the comma-containing spec survives `run_decode.sh`'s shell quoting.

**`EXPECTED_ACCEPT` was a module constant** in `verify_point.py`, hardcoded to the 70000/10000
point. That made `--expect-heterogeneous` useless at any other output length, because
`realized_accept_length` depends on the iteration count and not on `--accept-length` alone. Caught
only when the new flag was first used in anger. Now `--expect-accept`, default unchanged.

---

## 11. The pre-existing traps that still apply

- `--max-running-requests <C>` is **mandatory**, not tuning. At dp=8 the default 48 sizes the
  per-worker `ReqToTokenPool` to 6 slots and `alloc_req_slots` fails with a message about KV bytes
  that is really about slot count.
- `run_decode.sh` refuses to run if the iteration directory already exists. Do not pre-create it.
- `pgrep -f <pattern>` self-matches the command containing the pattern. Use precise `ps` + grep.
- A `docker exec` killed by an outer timeout **leaves the in-container process running**.
- CUDA-graph capture is slow and prints nothing. Silence is not a hang.
- `runtime.log` holds multi-megabyte single-line tqdm bars — never pipe driver output through
  `tail`.
- A profiled run is **not** a performance measurement.
- **`num_attention_heads = 64` with DP attention means every rank carries all 64 q heads**
  (`attn_tp_size = 8 // 8 // 1 = 1`). That is why FlyDSL declines against its gate of 8 or 16 and
  TileLang runs instead — in this packup's runs exactly as in the published baseline.
