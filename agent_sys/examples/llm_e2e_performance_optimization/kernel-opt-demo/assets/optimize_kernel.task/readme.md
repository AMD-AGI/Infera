# optimize_kernel — drive KernelForge over the workset, hand back a kit

You are the **wrapper** between two systems that do not know about each other.
On one side, `agent_sys` hands you a *workset* handoff and expects a
*kernel_optimization* handoff back, in a shape its validators check. On the
other, **KernelForge** is an autonomous optimization loop with its own CLI, its
own output tree, and five environment traps that fail quietly. Your job is the
conversion, and the judgement either side of it.

**You are not a shell script.** If driving KernelForge were one command, this
would be a `kind: program` leaf like `publish_workset`. It is not, because four
things here need a decision: whether the workset is coherent enough to optimize,
which invocation matches it, whether forge silently degraded, and whether the
speedup it reports is real. Do those four and the copying takes care of itself.

---

## 1. Success, stated so it can fail

1. A KernelForge campaign ran to completion over the workset's kernel, or
   `KFO_MOCK=1` and you said so in every place a reader could be misled.
2. The optimized kernel is **correct** — forge's own SNR gate passed, and you
   re-ran the workset's driver yourself and saw it pass.
3. The speedup is **measured by you**, not copied from forge's report, over at
   least 5 rounds per side, reported as a median.
4. The handoff is packup-shaped and carries the four evidence files
   `check_optimization_shape` requires.
5. Everything you started is torn down and every temporary file is under
   `$KFO_SCRATCH_ROOT`.

You will be checked by two validators. The cheap one looks at shape. The
expensive one **re-measures your speedup claim on this machine** and fails you
if it does not reproduce within tolerance. Do not round in your own favour.

---

## 2. What you are given

| variable | what it is |
|---|---|
| `$AGENT_SYS_INPUT_WORKSET` | the workset handoff's **version directory**. The files are under `content/items/codes/<name>/` |
| `$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION` | your output handoff's **content directory** — `content/` itself. Write into it; do not create a sibling |
| `$AGENT_SYS_TASK_PACKAGE` | the staged copy of this package. Few-shot examples are at `$AGENT_SYS_TASK_PACKAGE/assets/few_shot/` |
| `$KFO_KERNELFORGE_REPO` | a KernelForge checkout, already `pip install -e`'d |
| `$KFO_SCRATCH_ROOT` | **local disk**, writable, inside a `yihou/` directory. Everything temporary goes here |
| `$HIP_VISIBLE_DEVICES` | the one GPU you may use. **Do not change it** — this host is shared |
| `$KFO_MAX_HOURS` | forge's time budget. Default 3.0 |
| `$KFO_FORGE_MODEL` | the model for the **nested** forge loop. Default `Claude-Sonnet-5[1m]` |
| `$KFO_SNR_THRESHOLD` | forge's correctness gate, in dB. Default 30.0 |
| `$KFO_MOCK` | `1` = skip the real campaign. Default `0` |

Note the asymmetry in the first two, because it has bitten people: the **input**
variable points at the version directory (so `content/` is a hop below it) and
the **output** variable points at `content/` itself. They look like a pair. They
are one level apart.

`$TRITON_CACHE_DIR` and `$KNOWLEDGE_LOCAL_ROOT` are already set for you, and §5
explains why you must not undo them.

---

## 3. Read the workset before you do anything else

```
$AGENT_SYS_INPUT_WORKSET/content/items/codes/<name>/
├── README.md                  the operator, its provenance, the correctness bars
├── program.md                 the brief forge is given: objective, headroom, rules
├── integration.md             where it plugs back in, and what may not change
├── baseline_measurement.md    the baseline and its cross-check against the profile
├── environment.md             hardware, image, versions
└── kernel/
    ├── <operator>_kernel.py   the seed — this is what forge edits
    ├── driver.py              the oracle. THREE stdout modes
    ├── graph_harness.py       graph-replay timing used by driver.py
    └── measure_baseline.py    the 5-round protocol
```

**Then run the driver yourself, before spending a cent on a model.** Two
commands, about a minute:

```sh
cd <workset>/kernel
python3 driver.py                                     # correctness, all cases
python3 driver.py --bench-mode --warmup 10 --iters 30 # timing, all cases
```

Compare the `case_ms:` line for the production shape against the number in
`baseline_measurement.md` and against the profile figure in `README.md`. If they
disagree by more than a few percent, **stop and say so in your handoff** — it
means the standalone driver is not measuring the traced kernel, and every
speedup you could produce afterwards would be a speedup over nothing. This check
is the cheapest thing in the whole task and it is the one that makes the rest
mean anything.

---

## 4. The KernelForge invocation

Copy the workset's `kernel/` into a fresh directory under `$KFO_SCRATCH_ROOT`,
make it a git repository (forge commits its keeps), and run the loop there.
Never run forge inside `$AGENT_SYS_INPUT_WORKSET` — that is a published,
sealed artefact.

```sh
W="$KFO_SCRATCH_ROOT/forge/$(date +%Y%m%d-%H%M%S)-<operator>"
mkdir -p "$W"
cp <workset>/kernel/*.py "$W"/
cp <workset>/program.md  "$W"/
cd "$W" && git init -q && git add -A && git commit -qm "seed"

kernel-agents forge-loop \
  --kernel        "$W/<operator>_kernel.py" \
  --driver        "$W/driver.py" \
  --workspace     "$W" \
  --experiments-dir "$W/forge_experiments" \
  --result-json   "$W/forge_experiments/forge_result.json" \
  --program-md-file "$W/program.md" \
  --fellow        triton-fellow \
  --gpu-target    gfx942 \
  --gpu-type      mi300x \
  --framework     sglang \
  --operator-name <operator> \
  --snr-threshold "$KFO_SNR_THRESHOLD" \
  --max-hours     "$KFO_MAX_HOURS" \
  --git-branch    forge-optimize \
  --target-functions "<the public entry point>" \
  --model         "$KFO_FORGE_MODEL"
```

`--target-functions` is the public function the driver imports — read it out of
the workset's `program.md` "Modification rules", do not guess it.

**Before launching, run `kernel-agents status`** and read two lines: the GPU
target must say `gfx942`, and `rocprof-compute` must say `ready`. If it says
`dependencies are not ready`, see §5 trap 3.

The loop is **time-driven**: it runs until `--max-hours` is spent, reserving 30
minutes to finalize. It prints an event stream; a campaign at `--max-hours 3.0`
takes about three hours and is normal. Do not kill it because it looks idle
during a build or a profile.

### Smoke mode — `$KFO_MAX_HOURS` at or below 2.0

**The floor is 1.0 and it is enforced, so there is no such thing as a
five-minute forge run.** `kernel_agents/cli.py:46` sets `MIN_MAX_HOURS = 1.0`
and `_validate_max_hours` raises `click.BadParameter` below it — *"a run shorter
than this can't complete a productive campaign"*. If you are handed a smaller
`$KFO_MAX_HOURS`, **clamp it to 1.0, run, and say in `notes.md` that you did and
why.** Do not pass the smaller value and let the CLI reject it, and do not
silently pretend the budget you were given was honoured.

A budget at or below 2.0 is **not** a normal campaign. KernelForge silently
drops Analysis to static-only — you can see it in `events.jsonl` as
`analysis_result` with `available_tier: static` — and the implementer turn cap
falls from 500 to 100. With an hour of budget the loop will quite possibly
finish with no improvement at all. That is expected and it is not a failure.

It exists so the *plumbing* can be exercised against the real tool for a few
minutes instead of three hours. When you are in it:

- **Run the real campaign anyway.** This is not mock mode; `kernel-agents
  forge-loop` really runs.
- Set `"degraded": true` in `results/verification.json`, and record the budget
  you were given.
- Put **`SMOKE TEST — degraded budget`** in the first line of the packup
  `README.md`'s `## Result` section, and state plainly that no useful
  optimization was expected.
- If forge reports `improved: false`, that is a **successful smoke test**. Say
  so. Do not present a null result as a disappointment, and do not go looking
  for a speedup to report.

The rule is the same one mock mode follows: a run that could be mistaken for a
full campaign must be impossible to mistake for one.

### Mock mode

When `$KFO_MOCK` is `1`, do **not** run the campaign. Instead:

- run the driver's correctness and bench modes for real (they are cheap and they
  prove the wiring),
- write the handoff with every file the shape validator wants,
- set `"mock": true` in `results/forge_result.json` and `results/verification.json`,
- put **`MOCK RUN — no optimization was performed`** in the first line of the
  packup's `README.md` `## Result` section.

A mock that is not obviously a mock is worse than no mock at all. The shape
validator refuses a handoff that claims a speedup while `mock` is true.

---

## 5. The five traps. Four of them fail silently

Each of these has already cost a day. They are ordered by how quiet they are.

**1 — A non-clean `CLAUDE_CONFIG_DIR` crashes forge's backend probe.**
With a config directory that has plugins or MCP servers in it,
`claude --print --output-format json` returns a message **array** instead of a
single object, and KernelForge's probe does `payload.get("result")` on it
(`src/forge_llm/agent_backends/claude.py:333`) → `AttributeError: 'list' object
has no attribute 'get'`. This one at least fails loudly, immediately.
**Fix:** export a clean, empty `CLAUDE_CONFIG_DIR` under `$KFO_SCRATCH_ROOT`
*for the nested forge process only*. Do not touch your own session's.

**2 — `--max-hours <= 2.0` silently degrades the whole campaign.**
`kernel_agents/cli.py:1391` gates hardware profiling on
`max_hours > SHORT_FORGE_MAX_HOURS` (**strictly** greater, and the constant is
2.0 at `cli.py:47`); the implementer turn cap drops from 500 to 100 at the same
threshold. Nothing warns you. The campaign runs, produces a report, and the
analysis is static-only. **Use `$KFO_MAX_HOURS` and do not lower it below 3.0.**

**3 — `rocprof-compute` dependency conflict degrades profiling, silently.**
KernelForge pins `astunparse==1.6.3` / `kaleido==1.3.0`; ROCm 7.2's
rocprofiler-compute 3.4.0 needs `1.6.2` / `0.2.1`. Both cannot hold.
**Fix:** `pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt`.
pip prints a conflict warning that is expected and can be ignored. The **only**
symptom of not doing it is one line of `kernel-agents status`.

**4 — Writes to a NFS `$HOME` fail, and two of the three do it quietly.**
This container runs as root and `$HOME` is NFS with `root_squash`, so root maps
to nobody. `~/.triton` and the experience KB fail silently; a `~/.cache` write
once killed a whole sglang scheduler with an unremarkable `PermissionError`.
`$TRITON_CACHE_DIR` and `$KNOWLEDGE_LOCAL_ROOT` are already pointed at
`$KFO_SCRATCH_ROOT` for you. **Do not unset them and do not write anything under
`$HOME`.**

**5 — `kernel-agents list` and `show` look at the wrong directory.**
`config.experiments_dir` defaults to `<project_root>/experiments`, while
forge-loop writes `<workspace>/forge_experiments`. Without `--dir` you get
`No experiments found` and may conclude the campaign produced nothing.
**Fix:** `kernel-agents list --dir "$W/forge_experiments"`.

---

## 6. What KernelForge leaves behind

```
$W/forge_experiments/
├── forge_result.json          the summary. Also printed to stdout wrapped in
│                              __FORGE_RESULT__{...}__FORGE_RESULT__
├── optimization_report.md     the human-readable summary
├── best_result.json           the best iteration's numbers
├── campaign_config.json       exactly what was run
├── events.jsonl               the event stream
├── candidates/index.jsonl     one line per iteration: the decision and a lesson
└── best/iter_*/benchmark.json per-case timings, three independent measurements
$W/<operator>_kernel.py        the optimized kernel, on branch forge-optimize
```

`forge_result.json`'s useful fields: `baseline_ms`, `best_ms`,
`mean_case_speedup`, `improved`, `experiment_id`, and
`checkpoint.{decision,snr_db,validation_passed}`.

**Read `candidates/index.jsonl` even when the campaign succeeded.** Its `lesson`
field is where the loop records why an iteration was reverted, and those lines
are the most useful thing in the whole tree for your summary. A campaign that
kept 1 of 4 iterations is telling you something about the kernel, and the three
reverts are the part a reader learns from.

---

## 7. Do not trust the reported speedup. Measure it yourself

`mean_case_speedup` is forge's own number, measured by forge, during forge's
run. Your handoff is going to be checked by a validator that re-measures, so
find out now whether it holds.

```sh
# baseline: the workset's seed kernel, untouched
# optimized: the kernel forge produced
python3 measure_baseline.py --rounds 5 --iters 30 --json <side>.json
```

Five rounds per side, **each round a fresh process**, and compare **medians**.

Three facts about measurement on this machine, all learned the hard way:

- The **baseline is tight** (0.3–2% spread across rounds) and an **optimized
  kernel is loose** (~8% has been measured). You cannot compare a single sample
  of the loose side against anything.
- A single re-measurement once returned 21.67 µs where the median over four
  rounds was 18.9 µs, and that outlier was written up as a regression before a
  repeat showed it was noise. **One measurement is not a measurement.**
- Below about **1.05×** a "speedup" is not distinguishable from noise here. If
  that is where you land, say so plainly. Reporting 1.02× as an improvement is a
  false claim, and the validator's noise floor will catch it.

If your re-measurement and forge's disagree, **report both and say which you
believe and why.** That is a better handoff than a number that quietly matches.

---

## 8. Write the handoff

Everything goes under `$AGENT_SYS_OUTPUT_KERNEL_OPTIMIZATION`. Do not create
anything beside it — `claim/` and `manifest.yaml` are the system's to write.

```
README.md                                    ## Purpose / ## Interface / ## Boundary
items/codes/<operator>.packup_<YYYYMMDD>/
    README.md            what this was, and a ## Result section
    REPRODUCE.md         ordered copy-pasteable commands + an Expected output section
    environment.md       host, GPU, image, versions, what was installed on top
    notes.md             the gotchas and the wrong turns, including yours
    scripts/             every script you actually ran, verbatim
        kernel/          THE MEASUREMENT APPARATUS — see below. Required.
            driver.py            copied from the workset, unmodified
            graph_harness.py     copied from the workset, unmodified
            measure_baseline.py  copied from the workset, unmodified
            <operator>_kernel.py the SEED kernel, unmodified
    results/
        forge_result.json         forge's own summary, copied verbatim
        optimization_report.md    forge's report, copied verbatim
        optimized_kernel.py       the kernel forge produced
        verification.json         YOUR re-measurement (see below)
```

### `scripts/kernel/` is required, and it is not busywork

Copy the workset's `kernel/` directory into the packup **unmodified**, seed
kernel included. Two reasons, and the second is the one that bites:

1. **A kit that reports a speedup and does not carry the thing that measured it
   cannot be checked by anyone who does not already have the workset.** That is
   most readers.
2. **`check_speedup_substantiated` reads its apparatus from here.** It drops
   `results/optimized_kernel.py` over the seed in a scratch copy of this
   directory and runs `measure_baseline.py` against both. If `scripts/kernel/`
   is missing or holds anything other than exactly one seed module beside the
   three fixed files, it cannot run and you fail.

Do not edit any of the four on the way in. `driver.py` and `graph_harness.py`
are the oracle; a modified oracle makes every number here incomparable with the
baseline, and that is the one thing this whole task exists to avoid.

`items/codes/` is required by the `code` content type: a file placed directly
under `items/` is rejected before anyone reads it. **Exactly one** packup
directory, named `<name>.packup_<YYYYMMDD>` with a real eight-digit date.

Use the `experiment-result-packup` skill — it is available in this session and
it is the authority for the layout. The table above is only the part this task
makes mandatory.

### `results/verification.json`

Your own numbers, in a fixed shape, because a validator reads it:

```json
{
  "mock": false,
  "operator": "sampler_vocab_softmax",
  "rounds": 5,
  "iters": 30,
  "baseline_median_ms": {"B8_V151936": 0.05540},
  "optimized_median_ms": {"B8_V151936": 0.01895},
  "speedup_per_case": {"B8_V151936": 2.924},
  "mean_case_speedup": 2.83,
  "forge_reported_mean_case_speedup": 2.8328,
  "snr_db": 138.12,
  "correctness_passed": true,
  "noise_note": "optimized side spread 8%, baseline 1.7%; medians of 5 fresh processes"
}
```

### On the three README sections

- **`## Purpose`** — what this handoff is.
- **`## Interface`** — how a consumer uses it: which file is the kernel, what
  its signature is, what a re-integrator must not change.
- **`## Boundary`** — what is **not** here. Be specific and be honest: what you
  did not measure, what shapes you did not cover, what you could not explain.
  An honest boundary is worth more than a confident one, and the validator's
  transcript is kept.

**Do not claim anything you did not run.** The expensive validator re-measures,
in public, and it will disagree with you.

---

## 9. Few-shot examples

`$AGENT_SYS_TASK_PACKAGE/assets/few_shot/` holds worked examples — one handoff
that passes both validators, and two that fail, each with the verdict it drew
and why. Read them before you write your own. They are drawn from real
campaigns on this hardware, not invented.

---

## 10. Rules about this shared machine

- **Create only under `$KFO_SCRATCH_ROOT` and your output handoff.** Nowhere
  else.
- **Delete nothing you did not create.** No `rm -rf` on a path you were handed,
  no `docker rm` of a container you did not start. If a directory is in your
  way, fail and say so.
- **Never write a recursive delete whose target is a variable.** `rm -rf "$d"/*`
  with `$d` unset is `rm -rf /*`. That happened on this class of host on
  2026-08-31 and destroyed another engineer's git history. Delete named paths,
  or delete nothing.
- **Do not change `$HIP_VISIBLE_DEVICES`.** Other people are on the other cards.
- **Tear down** anything you started before you finish.
- **Never pass an explicit mode when you create a directory.** Let the default
  apply. Measured 2026-09-01: a run created `results/raw_measurements/` with
  mode `0644`, wrote seven files into it, and then could not read them back —
  a directory without its execute bit cannot be traversed, *by anyone,
  including its owner*. The task failed with `PermissionError` on a path it had
  just written, which reads like a sandbox problem and is not one. A file mode
  on a directory is always a mistake.

---

**This is a `readme.md` and there is no `entry.sh` beside it.** That one file's
difference is the whole of what "an agent task" rather than "a program task"
means in this system.
