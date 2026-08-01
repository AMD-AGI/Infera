# Notes — gotchas, wrong turns, and what is still open

## Refuted hypotheses (recorded so nobody re-tests them)

Four hypotheses were ranked before the micro-repro existed. Three were killed by a
single 30-second run, which is the entire argument for building the micro-repro first.

| # | Hypothesis | Verdict | Killed by |
|---|---|---|---|
| H1 | PR #58 patch 1 (a known hicache correctness fix) is missing | **refuted** | fault reproduces with the patch applied |
| H2 | `hicache_io_backend=kernel` is the problem; `direct` would work | **moot** | the defect is in *host allocation*, common to both backends |
| H3 | tail-page off-by-one in the index range | **refuted** | micro-repro faults on host pages **0-63**, the very first range |
| H4 | `write_through` policy races the kernel | **refuted** | same — a first-page fault has no race to lose |

H3 and H4 both predicted a fault at the *end* of a buffer. The fault was at the
beginning. One run, three hypotheses eliminated.

## Gotchas that cost real time

### `.pyc` staleness silently reverts a patch

CPython will run cached bytecode if the `.pyc` mtime still looks current. This has
invalidated a full experiment in this repo before. The patch deletes stale `.pyc`
itself, but **verify in bytecode anyway**:

    strings <module>.cpython-310.pyc | grep -c GLM52_ROCM_HOST_ALLOC   # want 1

A marker placed in a *comment* would always return 0 — comments do not survive
compilation. That is why the patch writes a real module-level string literal. My first
attempt used a comment and the verification silently "failed" against a correctly
patched file.

### The patch guard assumed an import that wasn't there

`common.py` imports only `alloc_mmap`; it does **not** import `is_hip`. The first
version of the patch asserted `is_hip` was already present and aborted. The shipped
version inserts the import alongside the dispatch change.

### VRAM release is asynchronous

After `kill -9` of the launcher and schedulers, the processes go to `Z` (defunct) and
`rocm-smi` still shows 90 % VRAM for a minute or two while KFD tears down. There is no
live holder to find. **Poll until all 8 GPUs read 0 %** before rebooting, or the new
engine OOMs during KV pool allocation and you will misread it as a config problem.

Corollary: a container recreate is *not* needed to reclaim VRAM — and would silently
drop in-container patches. Kill by explicit PID; a broad `pkill -f` can match your own
shell.

### `docker exec -d $CTR bash -lc '...'` does not persist

The detached login shell exits and takes the child with it: no process, no log, no
error, nothing to debug. Always stage a script file and `docker exec -d $CTR bash
/the_script.sh`.

### Server logs are binary

Plain `grep` says "binary file matches" and `grep -c` returns **0** — which reads
exactly like "the bad thing never happened". Use `strings <log> | grep` or `grep -a`.
An absent log line in a truncated log is not absence of the behaviour; that error
produced a standing false hypothesis about the KV-event plane on this stack once.

### L3 can be configured write-only

`cache_controller.py:467` computes

    self.prefetch_capacity_limit = int(0.5 * self.mem_pool_host.size)

If the host pool is sized via `--hicache-ratio` below ~1.5 this computes to ~0 and L3
is written and **never read** — the read-back proof would fail for a configuration
reason with no error message. Checked before trusting the replay result: with
`--hicache-size 32` (absolute GB) it is non-zero.

Related: never use the default `--hicache-ratio 2.0`. It sizes off
`max_total_num_tokens` and has computed to **355 GB per DP rank** on this stack. A
TB-scale pinned host allocation can wedge a spur node at kernel level (D-state,
unkillable); abandon a wedged node rather than fight it.

## Why vultr never saw this — and why that is not reassuring

The four sanctioned kits all ran kvd successfully on the **vultr** node pair, which is
why the initial framing was "kvd works on gfx950; spur is the new variable". That
framing was wrong in an instructive way.

`better/08`'s own limitations section records that its requests were **sequential and
≤ ~6,200 tokens** against a 16 GB host pool, 573 MB resident. The device→host write-back
path is only entered once hicache actually backs pages up. Spur's 120K-token prompts
entered it on the first request.

**So the bug is workload-specific, not cluster-specific.** vultr would fault too at this
scale. Do not read the vultr successes as evidence that ROCm host registration is fine.

## What is still open (deliberately not concluded)

**Needle retrieval depends on KV cache state, and the mechanism is unknown.**

The needle-at-depth suite scored 4/5 (kvd off), 3/5 (kvd on, cold), and finally **5/5**
with a warm L3 prefix (`cached=120000` at every depth) — all at the same forced
`temperature=0.0`. The failing depth *moved* between runs, so the score was never
attributable to kvd.

The leading hypothesis was sampling: `correctness.py` forces `temperature=0.0` while
`generation_config.json` recommends `1.0 / top_p 0.95`, and greedy decoding on a
reasoning model is a classic degenerate-repetition trigger — which is exactly what the
failing transcripts showed (`</think></think>...`, `5385227\n5385227\n...`). An A/B was
written (`needle_sampling.py`) and **never needed to run**: the 5/5 came at
`temperature=0.0`, refuting it.

Also ruled out along the way: the GPU fault itself (the kvd-off run had `faults=0` and
still failed), output truncation (re-ran at `max_tokens` 256 and 1024; depth 5 %
produced no 7-digit run at all at either), and a missing chat template (both legs log
the detected template).

What correlates is prefix-cache state. **Why** re-prefilling an identical prefix
degrades retrieval at some depths is not established, and is not asserted here.
Concluding a mechanism from three runs would repeat a mistake this repo has already
paid for.

## Evidence discipline that shaped this investigation

* A latency win is **not** evidence that kvd did anything — the in-GPU radix cache
  serves repeated prefixes without touching L3. Only restart-and-replay attributes.
* Prefer instrumentation over source-reading — but a **replayed CUDA graph executes no
  Python**, so Python-level probes are blind inside a graph. That is part of why the
  micro-repro calls the kernel directly.
* Check upstream before accepting a localization. `gh search` found no sglang PR or
  issue for this; it is genuinely unfixed, not something we re-derived.
