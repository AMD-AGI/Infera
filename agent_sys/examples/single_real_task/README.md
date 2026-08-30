# `single_real_task` — one real workload, end to end

The third example package, and the first whose subject is a real workload rather
than a synthetic one. `demo` and `demo2` exist to exercise graph shapes; this one
exists to answer a different question — **does the system hold up when the task
is something a person would actually be asked to do?**

The task: serve **`Qwen/Qwen3.6-27B`** in **mix** mode on **infera + sglang**,
prove it answers, and hand back a reproduction kit that somebody who was not
there can follow to the same result.

## Running it

Two variables have no default, because where a site keeps 52 GB of weights and
what its image is called are facts about that site:

```bash
agent-sys run --package agent_sys/examples/single_real_task \
  --var model_path=/srv/models/Qwen3.6-27B \
  --var image=<an image carrying both infera and a matching sglang>
```

Omit either and the load fails naming the file, the line and the variable. Two
more are optional: `--var etcd_image=` (defaults to a public tag) and
`--var work_root=` (defaults to `/var/tmp`).

`--var reproduce_timeout_seconds=` caps the AI validator. It defaults to 5400;
pass something small when you are testing the wiring rather than the workload.

## The shape

```
main                     non-leaf, no agent, one entry
  └── serve_qwen         leaf, agent: ai (Claude Code SDK), is_end
        outputs: runbook           content_type: code
          check_packup_shape       program, seconds,   completeness / strong
          check_reproduces         ai,      gpu_hours, usability    / weak
```

One leaf, one handoff kind, two validators. Every construct is there because the
task needs it, not because an example wanted to show it off.

## What this package shows that `demo` and `demo2` cannot

- **A validator with `kind: ai`.** Both other packages use AI for *agents* only;
  neither has an AI validator at all.
- **A validator that runs the artefact it is judging** rather than inspecting its
  shape. `check_reproduces` hands a fresh Claude Code session the packup and
  nothing else — not this readme, not the transcript, not the machine in the
  state the producer left it.
- **Two validators of deliberately different cost on one kind**, so the ordering
  rule in `validator` spec §5.3 has something to order: the cheap shape check can
  fail before the expensive reproduction is ever started.

## What it does not do

Stated because an example that overclaims is worse than none.

- **No `resources` block.** The leaf legitimately wants a GPU, but `cli/build.py`
  is the only reader of that field and the CLI composition root declares no
  pools, so writing one would name a pool that does not exist. Nothing here
  reserves a GPU; if two things run at once they will collide, and the agent's
  brief tells it to tear its server down for exactly that reason.
- **`repos: [infera, sglang]` is declared and read by nothing.** It is where the
  schema says such a fact goes; `env_mgr` does not act on it today.
- **No throughput, latency or accuracy claim.** The success criterion is that the
  model serves and that the kit reproduces it. Nothing more.

## Cost, and why it is not in the suite

This package needs real hardware — a ROCm GPU with room for a 27 B model — real
weights on local disk, and credentials for the agent backend. Main spec §4.3
rev. 13 admits that: an example may require what the workload it demonstrates
requires. What it may not do is make the repository's own tests depend on it, and
this one does not. Nothing imports it, nothing loads it during `pytest`, and it
is data in the tree like every other example.
