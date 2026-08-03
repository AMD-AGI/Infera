# Notes — gotchas, wrong turns, and what this run does NOT establish

The four long-form investigations live in `notes/`; this file is the index plus
the traps that did not need one.

| file | the finding |
|---|---|
| `notes/feature_gate_summary.md` | every feature proven ON, with the check that would go red |
| `notes/kvd_serving_proof.md` | kvd **serves** L3 — and the false negative that had to be diagnosed first |
| `notes/needle_resolved.md` | the needle "failures" are sampling excursions, **not** KV corruption |
| `notes/bigram_not_exercised.md` | the bigram kv-event path is **not** exercised here — corrects a sanctioned kit |

---

## The three wrong turns, each with what / why / how / context

### 1. `pkill -f infera.engine.sglang` kills its own shell

**What.** The kvd restart-replay hung at step 1 for ~12 minutes with no output,
engine already dead.

**Why.** The pattern matches the `bash -c '...'` command string that *contains*
that text — i.e. the very shell running `pkill`. It killed itself. `router.sh`
already documented this trap for `infera.server`; I reintroduced it elsewhere.

**How fixed.** Bracket the pattern so it cannot match the literal command line:
`pkill -9 -f "[i]nfera.engine.sglang"`. Same for `[s]glang.launch_server`.

**Context.** The same latent bug sat in `boot.sh`, hidden by `|| true` — which
meant the *wait-for-teardown loop below it never ran*, and that loop is the whole
point of the step: without it the next leg can start while the old one still
holds the DP kv-event port block, and dies with
`port_base at N is not available in 30 seconds`, which reads like a port-allocation
bug rather than leftover state.

### 2. A plain engine reboot does not make a cold cache

**What.** The first kvd read-back proof came back `gets 0, hits 0, sets +15,781`
— L3 apparently write-only. It looked conclusive and was wrong.

**Why.** `boot.sh` restarts the *engine process* inside a container that keeps
running, so the replay re-populated the GPU tier before any prompt needed L3.
Nothing ever asked the store.

**How caught.** `misses_total: 0` **together with** `gets_total: 0`. A key
mismatch (hash seed, bigram view, page size) would show up as *misses*. Zero of
both means the query never left the engine — which ruled out every downstream
explanation before any of them was investigated.

**Context.** Settled by instrumenting all four early returns of
`prefetch_from_storage` in one round rather than guessing among them. Full write-up
in `notes/kvd_serving_proof.md`, including a plausible hypothesis (infera's own
`hicache_validate.py` comment about `prefetch_capacity_limit` collapsing to 0)
that the measurement **refuted** — this sglang build computes
`0.5 * mem_pool_host.size` = 356,128, not 0.

### 3. `temperature: 0` + MTP is indistinguishable from KV corruption

**What.** Inherited probes send `temperature: 0`, which under EAGLE/MTP walks a
reasoning model into a repetition loop that the draft model predicts perfectly.

**Why it misleads.** `accept len` pins at its maximum (4.00) and the response
runs to `max_tokens` — which reads as "MTP is healthy" and "the output is
corrupt" simultaneously. Both readings are wrong.

**How avoided.** Every probe here sends GLM-5.2's own
`generation_config.json` values (**temperature 1.0 / top_p 0.95**) and
`max_tokens 2048`. At 256 the reasoning is cut off mid-thought and the run-on tail
mimics corruption too.

**Context.** `accept len: 4.00` is a **symptom of a loop, not a health signal**.
Measured here, acceptance is bimodal: 1.75–2.80 healthy, 3.85–4.00 looping.

---

## Traps that did not need a whole document

* **Never probe a PD leg's own port** — `curl` to it hangs. Go through the router.
* **Grep engine logs through `strings`.** They contain binary bytes; a plain
  `grep -c` returns 0, which reads exactly like "the bad thing never happened".
* **Check `Errno 98` *after* the ready line.** A `--kv-snapshot-port` collision
  lets a leg log `ready to roll` and then die during etcd registration: healthy-
  looking, never registered.
* **`speculative_algorithm` is echoed quoted** (`='EAGLE'`). Matching a bare
  `=EAGLE` reports MTP absent on a leg that is running it — my gate did exactly
  that for one round.
* **Don't grep an appended log for readiness** — it matches the previous run
  within seconds. Poll the HTTP endpoint.
* **`docker exec -d $CTR bash -lc '...'` does not persist.** The detached login
  shell exits and takes the child with it: no process, no log, no error. Stage a
  script file.
* **Never background a `docker build` inside `spur exec`** — the exec namespace
  teardown kills it even under `nohup`/`setsid`. Keep it in the foreground and
  keep the login-side process alive.
* **`export DOCKER_CONFIG=/tmp/dockercfg` before every docker call.** docker 29's
  buildx plugin discovery fails on the node's root-owned default path.
* **VRAM release is asynchronous.** After `kill -9`, processes go `Z` and
  `rocm-smi` still reads ~90 % for a minute with no live holder. Poll to 0 %
  before rebooting or the next boot OOMs.
* **`--hicache-size` is absolute GB.** The default `--hicache-ratio 2.0` sizes
  off `max_total_num_tokens` and has computed to 355 GB *per DP rank* on this
  stack, which can wedge a spur node at kernel level.
* **`sbatch --output` is silently ignored on spur** — logs always go to
  `~/spur-<jobid>.out`.

---

## Two measurement artifacts that would be easy to misreport

### `accept_length` is `null` in all 8 bench JSONs — by construction

`bench_serving` reads `avg_spec_accept_length` from `<base_url>/server_info`
(`benchmark/serving.py:1525`). `--base-url` here is the infera **router**, which
has no such endpoint. Not a missing feature — a consequence of routing.

And the decode leg's own `/server_info` value is **not** a per-point number
either: it is `spec_total_num_accept_tokens / spec_total_num_forward_ct`
(`scheduler.py:3787`), a **cumulative** mean over the engine's lifetime, reported
**per DP rank** (8 entries that genuinely differ, 1.415–1.577 at the final point).
Snapshotting it after each point produces a monotone-looking decline that is
mostly dilution.

The real per-point distribution comes from binning the decode log's 12,654
timestamped `accept len:` samples into each point's measured window
(`scripts/accept_by_window.py`).

### The p90 point's low acceptance is the *benchmark's* prompt, not the model's

Acceptance at the p90 point (1.27–1.60) is much lower than at p50 (1.97–2.38).
Do **not** attribute this to prompt length. `--dataset-name random` builds a
prompt by sampling one ShareGPT conversation and **repeating its token ids** to
reach the target length:

    datasets/random.py:131-134
        ratio = (input_lens[i] + prompt_len - 1) // prompt_len
        input_ids = (prompt_token_ids * ratio)[: input_lens[i]]

At ISL 155,000 that is hundreds of repeats of the same text, and OSL 3,300 with
`ignore_eos` forces long generations off that degenerate context. It is a
property of the synthetic prompt, not of the deployment.

**Quote the p50-point value (~2.2–2.4) as this deployment's acceptance**, labelled
as measured on synthetic repeated-ShareGPT text.

---

## What this run does NOT establish

* **No kvd-off A/B.** kvd was ON (prefill) throughout, so **no performance claim
  is made for it**. Its read path is proven structurally, not by a latency win —
  and a latency win could never have attributed anything, because the in-GPU
  radix cache serves a repeated prefix without touching L3.
* **The sweep exercises kvd's write path only.** Every prompt is unique (distinct
  seed per point, `random` prompts do not nest), so there is no prefix for L3 to
  serve: `gets` stayed flat at 11,281 while `sets` grew +173,270 and evictions
  +172,857. Expected, not a defect — but it means the sweep says nothing about L3
  hit behaviour under reuse.
* **The bigram kv-event fix is not exercised.** Measured on the wire: the prefill
  leg emits plain ints. Had that fix been absent from this image, every number
  here would be unchanged. See `notes/bigram_not_exercised.md`.
* **These E2E numbers are not an SLA verdict.** `bench_serving` fires
  `--request-rate inf` — all `2×conc` requests offered at t=0, no think time — so
  queue depth is maximal by construction. Case A is closed-loop with a 4 s median
  inter-turn delay. The `e2e_p50_ms ≤ 4500` criterion is answered by the Case A
  run, not by this sweep.
* **One run per point, no repeats.** No confidence intervals on any percentile.
* **Case B not run** (needs a 520K-context engine).
* **MTP vs no-MTP is not a controlled A/B here.** The prior spur Case A run had
  MTP off, but it also used a different image; any comparison to it is a
  cross-run comparison and is labelled as such.
