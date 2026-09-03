# The signature failure of this campaign: a check that returns the reassuring value in the failing state

**Include this in every packup.** It is cross-cutting — no single experiment owns
it — and it is the most transferable thing the campaign produced. Nine instances
in two days, each found the hard way, and they are not nine unrelated mistakes.

## The instances

| instrument | what it looked like | what it actually was |
|---|---|---|
| `rocm-smi --showmemuse` **`VRAM%`** | GPU occupancy | a high-water mark — does not fall when memory is released. Read 76 % on empty cards |
| **`base_gpu_id`** as a GPU-split check | which physical GPUs a leg holds | an index into the **visible** set. Reads `0` on both legs whether the split is right or wrong |
| **`docker ps`** showing none of ours | the node is free | absence of *our* containers. Eight GPUs were held by a colleague at 92 % |
| **`pgrep -f <string>`** | is the process running | matches **your own command line**. Fired twice, both times reporting "still running" when nothing was |
| **`/v1/models` → 200** | the engine is alive | the router answers from its own registry. The engine had been dead two hours |
| **`prompt_tokens_details: None`** | prefix caching is broken | correct for a probe shorter than `page_size`. Cache was at 40 % hit rate |
| **`MC_DISABLE_HIP_TRANSPORT`** as an A/B control | hip disabled | **not in the binary at all.** Guaranteed-zero differential reading as "hip does not matter" |
| **`HIP transport installed`** as the hip-off discriminator | hip is on | an **install-time** log; the knob gates **selection**. Reads 4/4 in both states forever |
| **`apply_chat_template=False`** in the resolved-args dump | the template was not applied | flag **parsed and never consumed**; the backend applies it server-side regardless |

## Why they get worse down the list

The first four corrupt a **diagnosis** — you look in the wrong place for a while.

The next three corrupt an **experiment** — a zero differential reads as "the
treatment does nothing" rather than "the control did nothing", and nothing
anywhere distinguishes them.

The last two are the worst, for different reasons:

- **`HIP transport installed`** produced a *structural* claim. From "the check did
  not flip" came "the A/B is impossible" — which was written into a deliverable
  and would have shipped. A correct hip-off deployment was discarded unmeasured
  on the strength of it. When it was finally run properly, **hip off was 23 %
  faster**: the opposite of the kit's stated premise.
- **`apply_chat_template=False`** appears in the **resolved-args dump**, the
  artifact we treat as ground truth and use for twelve-field config verification.
  A parsed-but-unread flag is indistinguishable there from a real setting. It
  made a hypothesis look leading that was in fact already eliminated, and that
  hypothesis reached a shipped file.

## A tenth, of a different shape: two honest sources that must not be compared

`requested` is what you set and what `/get_server_info` echoes. `resolved` is what
the engine log prints after dividing by `dp_size`:

| field | requested | resolved (dp8) |
|---|---:|---:|
| `max_running_requests` | 256 | **32** |
| `chunked_prefill_size` | 65536 | **8192** |

Both readings are correct. A verifier demanding 256 and fed the log's 32 **rejects
a correctly-configured deployment** — and invites someone to "fix" it upward.

## The countermeasures, in order of value

1. **Before trusting any check, make it return the failing value once.** Point the
   liveness probe at a dead port. Run the cache probe with a 50-token prompt. **A
   probe you have only ever seen succeed is not a probe.**
2. **When a check does not move, rule out that the check is blind before
   concluding the world did not move.** This is the one that would have saved the
   most here.
3. **Verify the control actually controls before trusting its null.** Env-present
   in `/proc/<pid>/environ` plus a source read showing the variable is consumed —
   not a log line, unless you have confirmed that log line can change.
4. **A gate you would hesitate to trip is not a gate.**
5. **A source tree shipped inside an image is not evidence about that image's
   binaries.** Recovery, no network needed: the build commit is usually still in
   the image's object store — `git cat-file -t <sha>`, then `git show <sha>:<path>`.
6. **Measure the noise floor before reading a differential.** The one differential
   in this campaign read against a *measured* floor (~5 %, from an arm whose only
   job was to quantify variance) is the only one that is safe. That arm looked
   like pure overhead when it was proposed.

## The corollary that matters for reporting

**"Open and unanswerable with the instruments we have", "an experiment we
skipped", and "a null result" are three different statements.** Usually only one
is true, and they read very differently to whoever decides what to trust.
