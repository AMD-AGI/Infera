# serve_qwen — bring Qwen3.6-27B up in mix mode, then hand back the kit

Serve **`Qwen/Qwen3.6-27B`** in **mix** mode using **infera + sglang** on the
host you are running on, prove it answers, and hand back a reproduction kit that someone
who was not here can follow to the same result.

Two deliverables and they are not the same thing. A run that works and is not
written down is worth nothing here; a kit that is written down and never ran is
worth less. **Do the work first, then pack up what you actually did.**

## What "mix" means, and how you will know you got it

`mix` is this repository's own word and it is **not** mixed precision. The repo
defines it — `examples/` and `docs/` are the reference and you should read them
rather than take my word for it. What you are aiming at is a deployment where
one worker does both phases, on a single node, with no RDMA.

The criterion is checkable and you should check it: **the worker and the router
must report the mixed disaggregation mode**, in the worker's own log line and in
the router's worker listing. Do not claim mix mode because you did not pass a
flag that would have turned it off. Read it back from the running system and put
the two lines in `results/`.

## Success, stated so it can fail

1. A server is up, `/health` answers.
2. A real chat completion comes back **through the infera router**, not only
   from the engine's own port.
3. The worker/router report the **mixed** disaggregation mode.
4. Every server, worker, router and container you started is **torn down** by
   the time you finish.
5. The kit reproduces it.

## What you are given

These are facts, not instructions. None of them is the recipe.

| variable | what it is |
|---|---|
| `$SRT_MODEL_PATH` | The weights, **already on local disk**. All shards are there. Do not download anything. Supplied per site with `--var model_path=` |
| `$SRT_IMAGE` | A docker image carrying **both** infera and a matching sglang. Nothing needs building or pulling. Supplied per site with `--var image=` |
| `$SRT_ETCD_IMAGE` | An etcd image, already present locally |
| `$SRT_WORK_ROOT` | Where a container workdir may go. **Local disk** |

Also true, and each one has already cost somebody a day:

- **The repository's own `README.md`, `docs/` and `examples/` are the reference.**
  There are worked launch scripts in `examples/` for other models on this class
  of hardware. Read them. Nothing in this readme is a substitute for them.
- **Pick your ports; do not assume them.** A development host is usually shared,
  and the ports the docs use are frequently already bound by somebody else.
  Check before you bind, and record the ports you actually used.
- **The container workdir must be on local disk.** A home directory is often on
  a network filesystem with `root_squash`, where a container's root maps to
  nobody and the engine fails to write its logs **silently** — no error, just no
  log. Use `$SRT_WORK_ROOT`, and check what you are actually writing to.
- **A cold start takes minutes, and the log repeats a health-check failure the
  whole time.** That is a JIT compile, not a hang. Killing it and
  retrying is the single most expensive mistake available here.
- **This is a reasoning model.** It emits a long thinking preamble before its
  answer, so a small `max_tokens` truncates it mid-thought and the response
  comes back with a length-based finish reason. That is the request being too
  small, not the server being broken — and a correctness check that greps the
  output for one word will report a working server as broken. Budget tokens, or
  parse the reasoning, and say in the kit which you did.
- **Tear everything down when you finish.** The kit is the deliverable, not a
  running process, and `check_reproduces` starts *another* bring-up from your
  kit. Two of these fighting over the same GPUs and ports is a fault somebody
  then has to diagnose.

## Where to write the kit, and in what shape

Write into the directory named by **`$AGENT_SYS_OUTPUT_RUNBOOK`**. That is your
output handoff's content directory; it already exists and you are granted write
on it. Do not create anything beside it — `claim/` and `manifest.yaml` are the
system's to write, and the manifest is what makes the version published.

```
README.md                                   with ## Purpose, ## Interface, ## Boundary
items/codes/<experiment-name>.packup_<YYYYMMDD>/
    README.md
    REPRODUCE.md
    environment.md
    notes.md
    scripts/
    results/
```

**`items/codes/` is not decoration.** This handoff's `content_type` is `code`,
and that type requires an item named `codes` and nothing else at the top level
(`handoff/content.py`). A file placed directly under `items/` is rejected before
anyone reads it.

**Exactly one packup directory**, named `<experiment-name>.packup_<YYYYMMDD>`
with a real eight-digit date. Two of them means a reproducer has to guess which
kit is the one that worked.

### The packup itself

Use the `experiment-result-packup` skill. It defines the layout and it is the
authority for it; what follows is only the part this task makes mandatory.

| entry | what it must carry |
|---|---|
| `README.md` | What this was, and **a `## Result` section** saying whether it worked and how you know |
| `REPRODUCE.md` | Ordered, copy-pasteable commands from zero to the same result, and an **`Expected output`** section naming what success looks like and where to see it |
| `environment.md` | The exact hardware and software: host, GPU, image tag, model path, versions. Pin things — this is the file whose absence breaks reproductions |
| `notes.md` | The gotchas and the wrong turns. Everything in the list above that bit you, plus everything that bit you and is not in it |
| `scripts/` | The scripts you actually ran, copied verbatim. Not paraphrases — byte-level flags have to survive |
| `results/` | The evidence: the completion you got back, the worker's mode line, the router's worker listing, timings |

`patches/` and `logs/` are optional. Ship them if the work produced them and
leave them out if it did not — an empty directory is worse than an absent one.

**`REPRODUCE.md`'s `Expected output` section is load-bearing.** It is what
`check_reproduces` hands its reproducer as the criterion; there is no other. If
it is vague, a correct reproduction can be judged a failure and an incorrect one
a success, and neither is recoverable afterwards.

### What is checked, and with what numbers

`check_packup_shape` runs first and is exact. Its rules, in full:

- exactly one `<name>.packup_<YYYYMMDD>/` under `items/codes/`;
- all four files and both directories above present;
- `scripts/` and `results/` each hold at least one file;
- **content lines** — non-blank, not a heading, not a fence marker — at least
  **5** in `README.md`, **8** in `REPRODUCE.md`, **8** in `environment.md`,
  **3** in `notes.md`;
- no `TODO`, `TBD`, `FIXME`, `XXX`, `to be filled in`, or a leftover `<…>`
  template placeholder, in any of the four;
- at least **5 command lines inside code blocks** in `REPRODUCE.md`;
- a `## Result` heading in `README.md`;
- at least one digit somewhere in `environment.md`.

`check_reproduces` runs second and is the expensive one: a fresh Claude Code
session gets a copy of your packup, follows `REPRODUCE.md` for real, and reports
whether it reproduced. It must be able to do that from your kit **alone** — it
does not get this readme, it does not get your transcript, and it does not get
the machine in the state you left it.

### `README.md`, the handoff's own

Three sections, all required by the `code` content type, all checked for being
non-empty and for not being a placeholder:

- **`## Purpose`** — what this handoff is: a reproduction kit for a
  Qwen3.6-27B mix-mode bring-up.
- **`## Interface`** — how it is consumed: point at the packup directory and at
  `REPRODUCE.md` as the entry point, and say what a reproducer needs to have
  before they start.
- **`## Boundary`** — what is **not** here. Be specific: what you did not
  measure (throughput, latency, accuracy are not in scope unless you measured
  them), what you did not try, what you left as an open question. An honest
  boundary is worth more than a confident one.

Do not claim in either README that anything was verified unless you ran it.
`check_reproduces` will disagree with you, in public, and its transcript is
kept.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" versus "a program task" means in
this system.
