# Notes — traps, wrong turns, and the things that cost hours

Ordered by how likely each is to bite a reproducer of **this** kit. The
predecessor kit's `notes.md` covers the patch-set rationale and two earlier probe
defects; this file covers what the branch-and-build run added.

---

## 1. `temperature: 0` + MTP is indistinguishable from KV corruption

**The most expensive mistake of the run**, and it was in our measurement tooling,
not the system under test.

**What.** G2 on the built image reported **3/5** needles:

| depth | needle | finish | ctok | `</think>` | tail |
|---|---|---|---:|---:|---|
| 0.0 | ✗ | length | 2048 | **1264** | `</think></think></think>…` |
| 0.25 | ✓ | length | 2048 | 1262 | |
| 0.5 | ✓ | length | 2048 | 0 | recites the filler corpus back |
| 0.75 | ✗ | length | 2048 | 0 | `0 the0 the0 the0 the…` |
| 1.0 | ✓ | **stop** | 86 | 1 | clean |

Depth 1.0 being the only clean one is *exactly* the mooncake early-send
signature — the final chunk goes through the sampling path, which already
synchronizes. Everything pointed at the early-send fix having failed.

**Why that reading was wrong.** The fix was in, and verifiably so:

- `_early_send_wait_event` present in the **bytecode** of all three files, on
  both nodes;
- the call sites wired — `prefill.py:721` records the event, `conn.py:1751` picks
  it up, `conn.py:1233` synchronizes before the RDMA read;
- the prompt genuinely split `8192+8192+1728`, confirmed from the prefill log.

The actual cause: all four probes sent `temperature: 0` with no `top_p`. The
checkpoint's own `generation_config.json` says:

```json
{ "temperature": 1.0, "top_p": 0.95 }
```

Greedy decoding sends a reasoning model into repetition on a long prompt, and
**EAGLE/MTP amplifies it**: the draft model predicts a loop perfectly, so
`accept len` pins at its maximum (4.00 with `--speculative-num-draft-tokens 4`)
and the response runs to `max_tokens`.

**`accept len: 4.00` is a symptom of the loop, not evidence MTP is healthy.**
That inversion is worth internalising — a metric that looks maximally good is the
tell.

**How it was settled — by control, not by argument.** Same image, same prefill
leg, same prompt, MTP off on the decode leg: **5/5, all `finish=stop`, 74–153
tokens**. One variable. Then MTP back on with the official sampling: **5/5**, and
`accept len` back to 2.17–2.60.

**How to tell it from real corruption:**

| | real chunk-boundary corruption | greedy+MTP degeneration |
|---|---|---|
| `</think>` | **cycles** with content between (`2183</think>2183</think>`) | absent (×0) or a solid run |
| `finish_reason` | can be `stop` | almost always `length` at the cap |
| at conc=1 replay | reproduces | **CLEAN** |
| depth sweep | spares depth 1.0 | hits every depth |

Note the depth-sweep row is *not* a discriminator on its own — it was the thing
that misled us here, because a loop that starts late spares the shortest answer
too.

**The same defect inflated the stress gate.** At OSL 1024, conc=128 showed 5/256
BAD — and `BAD all at cap: True`, every one of them. Raising OSL to 2048 alone
dropped it to **1/256**, matching the predecessor run, with nothing else changed.

**Fixed** in `scripts/needle.py` and `scripts/stress_capture.py`: both send
`temperature 1.0, top_p 0.95, top_k 40`, overridable by env for a deliberate A/B.
`probe.py` and `prefix_reuse.py` are left greedy on purpose — at 64 and 128 max
tokens they cannot reach the degenerate regime, and they passed 4/4 and 32/32 on
both images.

**The general lesson.** Before believing a failure that matches a known signature,
check the probe. A signature match is evidence *for* the hypothesis you already
have, which is exactly when it is least trustworthy.

---

## 2. Cold start is 3–9 minutes and is not a hang

Weight load (~35 s) plus CUDA-graph capture. A leg that has printed nothing for
five minutes is normal. Four cold starts sit on the critical path (G0 ×2, the G0
restart, G1 ×2), which is where most of the ~2 h goes.

Poll the HTTP endpoint, not the log — see §5.

---

## 3. Verify the IMAGE, not the build log

**What.** A build log saying each patch script printed success is not the same as
the running interpreter executing patched code.

**Why.** Python caches compiled modules keyed on source mtime. A patch script
that restores a backup with `shutil.copy2` preserves the original mtime, so an
edited `.py` can match a stale `.pyc` and CPython silently runs the **unpatched**
bytecode. **This has already invalidated a full experiment on this stack** — the
source showed the fix, the runtime did not have it.

**How.** `scripts/verify_built_image.sh` drops `__pycache__` for the directories
it checks, recompiles, and greps the fresh `.pyc` for identifiers the patches
introduce. It greps for **identifiers, never comment markers** — the compiler
discards comments, so a comment marker reads as a false negative.

Three things cannot be checked in bytecode and are source-checked, each with the
reason recorded at the check: the nextn prerequisite (an f-string split across
constants), patch 2a (changes an expression, introduces no new identifier), and
the msgspec field type (a lazily-evaluated annotation).

**One difference from the predecessor.** The patched base image carried **two**
infera copies — the pip-installed one and the `/opt/infera` source tree, which
shadows it for every `docker exec` because that is the WORKDIR. The built image
carries **one**, so that trap cannot arise here. The script reports what it found
rather than assuming either shape.

---

## 4. The reset ritual between rounds is mandatory

**What.** `reset_merged.sh` tears down the container and engine processes,
**waits for the GPUs to return to idle**, starts a fresh container, and verifies
8 `PORT_ACTIVE` inside it before anything else.

**Why.** Each skipped step maps to a concrete failure: skip the memory-release
wait → OOM mid-run that looks like a regression in your latest change; skip the
`PORT_ACTIVE` check → the libionic injection silently failed, mooncake drops to
TCP, and the run "works" while measuring nothing.

Step 5 of that script deliberately prints `NO PATCH STEP (that is the point)` —
anything applied after `docker run` would defeat the purpose of validating a
built image.

---

## 5. Two process/log traps that each cost a cycle

### 5a. `start_leg.sh` must kill the `sglang.launch_server` child

The stock script pkills `infera.engine.sglang` only. The wrapper exits, but the
`sglang.launch_server` child it spawned keeps the DP kv-event port block bound,
and the next leg dies with:

    ValueError: port_base at 30234 is not available in 30 seconds.
    port_base is used by a process already. process.name()='python3'

which reads as a port-allocation bug rather than as leftover state. This kit's
`start_leg.sh` and `restart_replay.sh` kill the tree and then **wait for it to be
gone** — the wait is the point, not the kill.

*(The argv in that error message was itself useful: it showed the old decode leg
carrying `--disaggregation-decode-enable-radix-cache` and no
`--enable-hierarchical-cache`, i.e. patches 6 and 7 gating correctly.)*

### 5b. Never grep an appended log for readiness

`restart_replay.sh` appends to a log that already contains `ready to roll` from
the pre-restart run, so the grep matches within seconds and the caller proceeds
against an engine still loading weights. Observed as `ready after 10s` for
something that takes minutes — and the kvd counters read afterwards would have
been meaningless.

Poll the HTTP endpoint instead (`scripts/wait_ready.sh`). It can only answer once
the engine is actually up. Both scripts in this kit are fixed.

---

## 6. The router KV view is per-process — measure it after driving traffic

`/v1/admin/cache-view/<worker>?dp_rank=N` reads a view held in the **router
process**. A freshly restarted router reports 0 blocks for every worker — which
looks exactly like the bigram fix having failed, since that count is its
discriminator.

Always: restart router → drive traffic → *then* read the view.
`scripts/cache_view.sh` carries this warning in its header.

---

## 7. The router module name

It is `python -m infera.server`. It is **not** `infera.router`:

    /opt/venv/bin/python3: No module named infera.router.__main__;
    'infera.router' is a package and cannot be directly executed

Hand-rolling the router command instead of using `scripts/start_router.sh` cost a
full G0 cycle. The kit's scripts are verified; use them.

---

## 8. Building on the nodes, not shipping an image

The two nodes' image ids **differ** (`1f7cf6964cee` vs `0d478433f1b3`) because
each built independently — Rust router objects and layer timestamps differ. That
is expected. **Do not check for equal digests**; check content equivalence with
`verify_built_image.sh`, which passed 18/18 identically on both.

Building on each node rather than `docker save`/`load`-ing a 28 GB tarball is
deliberate: the claim under test is that the *Dockerfile* reproduces the run. A
tarball would prove only that it survived the trip, and would carry whatever
local state the build machine had.

Related: `stage_source.sh` uses `git archive`, not a tar of the worktree, so an
uncommitted edit that changed the result cannot ride along invisibly.

---

## 9. What the branch deliberately does NOT carry

Four of PR #56's seven commits are absent, and two of the three taken were
reduced. The rule was **only code this experiment exercised enters the branch**,
because shipping unvalidated code under a validated branch name is worse than
shipping less.

The sharp edge among the omissions: **`rust/router/src/kv_event.rs` has the same
bigram bug, unfixed.** Every run here used `--router-backend python` (the
default), so a Rust-router deployment with MTP still silently degrades kv-aware
routing to round-robin. It is a real bug kept out of scope only by our backend
choice. PR #56 carries the fix.

Full table with per-item reasoning: `branch/MERGE_BRANCH.md`.

---

## 10. A claim the raw data never supported

The predecessor kit's summary table said conc=128 gave "256/256, 0 corrupt". Its
own `results/raw/stress_c128.json` has **1 `CORRUPT_REASONING`**. Both images show
exactly 1/256 — in both cases the single response that ran to the cap, and in both
cases a plain repetition loop (`</think>` × 0) rather than the chunk-boundary
signature. Corrected in both kits.

Worth stating because it is the failure mode of summary tables generally: the
number in the table drifted from the number in the JSON, and nothing caught it
until someone re-read the JSON.

---

## 11. Cluster hygiene

These are shared nodes. Other people's containers (`mlperf_gptoss2`,
`primus_train`, `mtt_pd`, the `robust-*` exporters) were running throughout and
were left alone. Nothing outside `merged_run` / `merged_run_etcd` / `vprobe` was
stopped, and no images were pruned. Before removing any container on `chi28xx`,
prove it is yours via `docker inspect` (Binds / Env / Created).

The jump host is heavily loaded (load ~44, ~23k zombie processes) and resets SSH
connections intermittently. `scripts/J.sh` retries; without it a transient reset
reads as a failed step on the node.
