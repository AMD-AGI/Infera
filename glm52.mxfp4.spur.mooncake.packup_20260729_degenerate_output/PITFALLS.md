# Pitfalls — what went wrong, why, and what it cost

Written so the next person does not repeat any of it. Ordered by how much time
each one wasted.

---

## P1 — Never validated that the symptom was a bug (cost: the entire day)

**What:** treated "output collapsed into a repeating loop" as an engine defect
and spent hours bisecting MTP / PD / custom-all-reduce / DP-attention /
quantization.

**Why it was wrong:** two setup errors, either sufficient on its own —

1. **Chat template skipped.** GLM-5.2 is an instruct model shipping
   `chat_template.jinja` (`[gMASK]<sop>` + `<|system|>Reasoning Effort:`). All
   testing posted raw `text` to `/generate`, i.e. base-LM completion of the
   string `"Explain quantum computing in detail, part 31."` — continuing that
   with `1.2.3.4...` is ordinary behaviour.
2. **`temperature=0` forced.** `generation_config.json` recommends
   `temperature 1.0 / top_p 0.95`, and sglang already honours it via
   `--sampling-defaults model`. The harness overrode it. Greedy decoding
   degenerating into repetition is textbook (Holtzman et al. 2019).

**The missing control:** no baseline was ever measured. "1.5 % of outputs loop"
is uninterpretable without knowing what a *healthy* engine does on the same
prompts. A comparison against another config is not a baseline.

**How to avoid:** before treating an output-quality symptom as a bug, run the
model the way it is meant to be run — correct template, the sampling in
`generation_config.json` — and establish the rate there first.

---

## P2 — Comparison arms differed in more than the variable under test

`--disable-custom-all-reduce` sat **inside** the MTP block of
`pd_leg_spur.sh`, so `MTP=0` also flipped a known-deadlocking kernel back on.
The "MTP on vs off" result measured two changes at once.

Only caught by grepping the **live** `server_args` line rather than reading the
launch script. Do that every time:

```bash
strings $LOG | grep -oE 'disable_custom_all_reduce=[A-Za-z]+|speculative_algorithm=[^,]+|enable_dp_attention=[A-Za-z]+'
```

---

## P3 — A whole-string predicate hid half the failures

Degeneracy was scored by unique-character count and long single-character runs
over the **entire** output. A 512-token response that is 200 tokens of good
prose then 300 tokens of `1.1.1.` has plenty of unique characters and scored
**coherent**.

Re-checking stored tails: **11 of 501 (2.2 %) "coherent" outputs were looping at
the end.** Every rate reported before that check is a lower bound.

Fix: `probe_onset.py` stores full text and binary-searches the offset where the
tail turns periodic. Onsets then ranged 0 → 356 chars — one failure mode with a
variable onset, not two phenomena.

**Lesson:** a detector tuned on the worst examples will miss the median ones.
Store the raw artifact and re-judge offline.

---

## P4 — `hold_node.sh` ran `sleep 36000` under a 12 h wall

Ten hours is shorter than the wall, so the script exited on its own and the
jobs went **`COMPLETED`** — not preempted, not `TIMEOUT`. Two nodes vanished at
exactly `10:00:27` and `10:00:56`, killing an FP8 run mid-boot.

```bash
sacct -u $USER --format=JobID,State,Elapsed
9006  COMPLETED  10:00:27
9005  COMPLETED  10:00:56
```

Fix: `sleep infinity`, and let the scheduler's `-t` be the only thing that ends
a job.

---

## P5 — Patch scripts that back up an already-modified file

`fix_bug6_idle_qoffset.py` does `restore from backup first`. Its first run
failed (anchor absent because Bug 1 had not been applied) but **had already
written a pristine backup**. The next run restored that backup — silently
reverting the Bug 1 fix applied in between.

Symptom: the fix "would not apply" for reasons that made no sense.
Fix: delete the stale `.orig`/`.fix_bug6_orig` before re-running a chain, and
apply patches in dependency order (Bug 1 → Bug 6; Bug 6 *corrects* Bug 1).

---

## P6 — Patched a deployment that never executes the patched code

Applied Bug 1/5/6 to a **no-MTP** server. All three crash inside
`deepseek_nextn.py` — the draft model — which a no-MTP deployment never loads.
Beyond being useless it damaged the experiment: a control should sit as close
to pristine upstream as possible, and Bug 2 touches shared code that a no-MTP
server *does* execute.

Reverted; the control ran fully unpatched.

---

## P7 — Sent requests to a PD decode leg

The concurrency-sweep baseline was pointed at job 9006 — the **decode** half of
a PD pair. It cannot accept direct requests:

```
Error: Invalid request: Disaggregated request received without bootstrap room id
```

All 192 requests failed; the sweep reported `ok=0` three times and produced
nothing. Route through the router, or use a single-node mix.

---

## P8 — Backgrounding a docker client inside `spur exec`

```bash
spur exec $J bash -c 'nohup docker load -i big.tar &'   # silently killed
```

The exec namespace tears down and takes the client with it. Run `docker load`
in the foreground; for long-lived work use `docker run -d` / `docker exec -d`.

---

## P9 — `docker cp` to the host `/tmp` inside `spur exec`

Files landed in a namespace that disappeared. Copy into `/home/yihou` (NFS
bind-mount) so results survive the node.

---

## P10 — Stale `.pyc` silently reverts a patch (carried over, still true)

`shutil.copy2` preserves mtime, so a restored `.py` can match the cached
`__pycache__` entry and CPython runs the **unpatched** bytecode — source shows
the fix, runtime does not. This invalidated a full experiment on 2026-07-28.

Every patch script must `os.utime(path, None)` and delete the module's `.pyc`.
Verify with:

```bash
strings <module>.cpython-310.pyc | grep -c <YOUR_MARKER>
```

---

## P11 — Binary bytes in server logs defeat plain `grep`

`grep` reports "binary file matches" and `grep -c` returns 0. Use
`strings <log> | grep` or `grep -a`. A "0 matches" here is not evidence of
absence.

---

## P12 — Reporting an experiment as running when its flag never applied

Variant A (`--speculative-attention-mode decode`) was reported as under test
while `server_args` showed `speculative_attention_mode='prefill'` and the
process had been `Killed`. Always confirm the flag in the **live**
`server_args`, and confirm the process is alive, before reporting a result.

---

## Method notes that did work

- **Client-supplied `rid`.** `GenerateReqInput.rid` is honoured and echoed as
  `meta_info["id"]`, so a request can be correlated across client, prefill log
  and decode log. Assert the echo — a correlation on mismatched ids is worthless.
- **`meta_info` carries far more than expected**, per request, with no
  instrumentation and no restart: `dp_rank`, `num_retractions`, `cached_tokens`,
  `e2e_latency`, and the whole spec-decode telemetry
  (`spec_accept_length`, `spec_verify_ct`, `spec_accept_histogram`, …).
- **`SGLANG_DEBUG_DSA_ROWS=1`** (`dsa_indexer.py:63`) prints
  `q_fp8`/`q_offset`/`lengths`/`mqa_q` shapes at the exact site of the
  padded-vs-real row crashes. Enable it on any long run.
- **A replayed CUDA graph executes no Python.** Python-level probes are blind
  inside one; "0 collectives on the graph path" was a probe blind spot, not a
  measurement.
