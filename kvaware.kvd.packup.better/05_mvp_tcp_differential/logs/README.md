# Raw logs — GONE. Read this before looking for them.

**There are no raw log files in this directory, and none can be produced.**

The Qwen3-1.7B MVP rounds (r1–r5) all ran inside a single container named
`kvexp` on chi2879 on 2026-07-30. That container was **removed** at teardown,
before the GLM-5.2 two-node runs began. Its `/tmp/r4/` and `/tmp/r5/` trees —
`prefill.log`, `decode.log`, `kvd.log`, `router.log` for each arm — went with
it. They were never copied to the shared FS (`/mnt/vast`), which is the mistake:
the later GLM-5.2 runs wrote their logs to
`/mnt/vast/c_huggingface/glm52_kvexp` precisely so this could not happen again.

Fabricating stand-in log files here would be worse than having none, so this
directory is empty on purpose.

## What survived

Captured in-session while the rounds were running, and quoted verbatim in
`results/r4_r5_differential.txt`:

| Arm | Prompt | Output fragment |
|---|---|---|
| A (r4, ON) | capital of France | `v4 freddy\n\nAists. Log In andapace.a\n\nWho wouldin %%%%3...` |
| A (r4, ON) | 17*23 | `S情辣梯neig治\n\n杖\n\n及格...` |
| B (r5, OFF) | capital of France | `v4ই\n\n脐猫\n\nument=""&gt;&lt; t-tcan you-...` |

Three fragments. That is the entire surviving primary evidence for this
experiment, and the finding rests on the comparison between them — specifically
on both arms being garbled and both opening with `v4`.

## What is lost and cannot be recovered — and this one is material

- **The full completions.** The `...` in the quotes above is literal truncation
  from the capture, not elision. How the outputs continued is unknown.
- **Arm B's answer to the arithmetic prompt.** Only the France prompt's arm-B
  output survived, so the arm-to-arm comparison rests on **one** matched pair,
  not four.
- **Whether the two arms used byte-identical prompt sets.** Arm A's surviving
  record includes a `17*23` prompt; the standard 4-case probe uses `2+2=`. The
  prompts are quoted as captured. This is a genuine weakness in the original
  comparison's rigour, and it is why `scripts/probe.py` now pins the case list
  in code rather than leaving it to the operator.
- **kvd's counters in arm A** (`gets`/`hits`/`misses`/`sets`). Not read in this
  round, so there is no evidence kvd stored or served anything.
- **The router's per-request worker assignment** in either arm.
- Per-leg logs entirely, so the KV handoff can only be said to have "completed"
  on the strength of the absence of HTTP 500s, not from the transfer logs.

## Why the conclusion survives the gaps

The finding is a *negative* one — that flipping the switches changed nothing —
and negative findings are robust to missing detail in a way positive ones are
not. Three fragments showing garbling on both sides of the switch are enough to
say the switch is not the cause. They would be nowhere near enough to say the
switch works, which is exactly why `results/what_this_does_not_prove.md` exists.

## Regenerating logs

```bash
bash scripts/run.sh              # ARMS=both, ~18 min, runs A then B then compares
ARMS=A bash scripts/run.sh       # one arm only — proves nothing alone
```

Both arms must run in the same invocation for the comparison to be a control.
An arm measured on a different day, after a reboot, or beside a different
neighbouring job is not a control.

The driver writes per-arm evidence to `results/arm{A,B}_r{4,5}.observed.txt`,
structured results to `results/arm{A,B}.json`, and the comparison to
`results/differential_verdict.observed.txt`. To keep the whole log files, pull
them before the script's `trap cleanup EXIT` removes the container:

```bash
docker cp <ctr>:/tmp/r4/. ./logs/r4/
docker cp <ctr>:/tmp/r5/. ./logs/r5/
```
