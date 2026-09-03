# Checkpoint summary — five-module parallel debug of `llm_e2e_performance_optimization`

Append-only. One section per 30 minutes of wall clock. Earlier sections are
never revised, including their wrong estimates — the record over time *is* the
value of this file.

Effort start (T+0) taken as **2026-09-02 08:28 UTC**, the minute the five
deliverable dirs under `/shared_nfs/yihou/agent_sys/debugging/` and the ws2
runroot were created.

Reporter reads, cheapest first: the five `*.debug.help.info.md` at the repo
root; `git log`/`git status` in the worktree; the run roots under
`/shared_nfs/yihou/agent_sys_debug/ws2/runroot/`; the deliverable dirs; and
`squeue -u yihou`.

---

## T+0 — 2026-09-02 08:35 UTC (baseline)

### Walltime countdown (the number that governs everything)

| job | node | owner | ends (UTC) | remaining at this checkpoint |
|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | **16:17:41** | 7 h 42 m |
| `101053` | `crsuse2-m2m-276` | integration | **16:17:57** | 7 h 42 m |
| `101078` | `crsuse2-m2m-080` | deploy (GPU 0–1), analyze (2–3), kernel-opt (4) | **16:28:21** | 7 h 53 m |

All three are 8 h holds, `TimeLimit=08:00:00`, confirmed by `scontrol show job`.
The BRIEF's "hard stop ≈ 16:2x UTC" is accurate; the precise wall is **16:17:41**
for the two whole-node jobs and **16:28:21** for the shared node.

### 1. Progress

**Effort: ~4 %.** Elapsed 7 minutes. Estimated remaining: unknown — see the
reliability note.

| module | est. % | basis |
|---|---|---|
| deploy | ~12 % | two notes sections written; package load verified; blocked on image discovery |
| profiling | ~10 % | two notes sections written; transport blocker identified and being localised |
| analyze | 0 % observable | no notes file, no runroot, no deliverable |
| kernel-opt | 0 % observable | no notes file, no runroot, no deliverable |
| integration | 0 % observable | no notes file, no runroot, no deliverable |

**Reliability of this estimate: very low.** Three of five modules have produced
no observable output at all, which at T+7min means "has not yet written" and not
"is not working" — I cannot distinguish those two states from the outside. The
two percentages I do give are inferred from notes content, not from any run
report; neither module has launched an `agent-sys run` yet as far as I can see.
No `runs/` directory exists under the ws2 runroot, so **zero graph executions
have started across the whole effort.**

### 2. Current state, per module

- **deploy** (`crsuse2-m2m-080`, GPU 0–1, ports 8100–8119). Has verified that
  the `deploy-demo/` move did not break the package load — `agent-sys show`
  reports 2 tasks / 2 closures / 2 validators per output phase, matching
  pre-move. Currently investigating whether `infera/engine-sglang:test-local`
  (present on the node, built ~7 h ago by another tenant) carries the
  `qwen3_5.py` model definition, which would remove an image build from the
  critical path.
- **profiling** (`crsuse2-m2m-079`, whole node, ports 8120–8139). Has proved the
  package's remote transport unusable on this cluster and is rewriting
  `assets/lib/remote.sh::on()` from `srun --overlap` onto `spur exec`, including
  hand-serialising the environment because `spur exec` does not carry it.
- **analyze**, **kernel-opt**, **integration**: no observable output. Not
  reported as blocked, not reported as running.

### 3. Code problems (defects in the packages or in `agent_sys`)

| # | module | problem | state |
|---|---|---|---|
| C1 | profiling | `assets/lib/remote.sh::on()` uses `srun --jobid … --overlap … --export=ALL`. The `srun` on this cluster is a **spur re-implementation**, not Slurm's: `--export` is rejected outright (`unexpected argument`), and even with it dropped the call needs a TTY and exits 128 under agent_sys bodies (no TTY). The transport is unusable as written. | **open** — fix in progress; intent is to make `spur exec` selectable rather than replace `srun` outright, since the other cluster still needs the srun form |
| C2 | profiling | Consequence of C1: `--export=ALL` was load-bearing. The remote side must see `AGENT_SYS_OUTPUT_*` and the whole `PD_*` block; `spur exec` delivers an empty environment (measured: `MARK=hello spur exec … 'echo $MARK'` → empty). `on()` must serialise the environment itself. | **open** — being written |

Nothing yet reported as a defect in `agent_sys` itself this round. The five
framework limits in the BRIEF (1800 s settle budget, gate-failure-reported-as-
timeout, handoff locality allow-list, no `claude` in a validation zone, single
`--demo-root` knob) are **known and pre-recorded**, not new findings; they are
listed here only so a later reader does not re-derive them.

One non-defect worth recording, because it looks like one: `agent-sys show`
REJECTs the deploy package without the four site vars
(`deploy.yaml:69:5::$[0].env.E2E_MODEL_NAME: no value for ${model_name}`). That
is the package working as designed. Do not read it as a load failure.

### 4. Non-code problems (environment / localisation traps)

| # | problem | state |
|---|---|---|
| E1 | **Docker images are per-node.** `infera/engine-sglang:gfx950-local`, built 2026-09-01 on `crsuse2-m2m-020`/`-188`, is **absent on `crsuse2-m2m-080`**. So is `lmsysorg/sglang:v0.5.17-rocm720-mi35x`, the base the BRIEF names. The node instead carries `infera/engine-sglang:test-local`, `infera/engine-vllm:test-local`, `lmsysorg/sglang:v0.5.12-rocm720-mi35x` (**.12, not .17**), and two other tenants' tags. | **open** — deploy is testing whether `test-local` carries `qwen3_5.py`; if not, an image build enters the critical path |
| E2 | `spur exec` runs at `pwd=/`, `HOME=/opt/spur`, and without `~/.local/bin` on PATH. Every script must `cd` first and export `HOME=/home/yihou` and `PATH="$HOME/.local/bin:$PATH"`. | **known, pre-recorded in BRIEF**; profiling re-measured and confirmed |
| E3 | Data dependencies the three imported packages default to — `/apps/tas/yaoc/...` — **do not exist here**. Specifically absent and not yet located: the AIPerf `conversation_trace.jsonl`, the gsm8k `test.jsonl`, and the analyze seed `gap_analysis.csv`. Synthesising stand-ins is sanctioned but must be declared loudly. | **open** — no module has reported locating or synthesising any of the three |
| E4 | `/shared_nfs` is 98 % full (~7 T free). Image builds and weight copies must be sized against that. | **open, latent** |

### 5. Undetermined

1. **`spur exec` identity: root or `yihou`?** The BRIEF states it "runs as `root`
   at `pwd=/`". The profiling module's own measurement on `crsuse2-m2m-079`
   reports `id -un` → **`yihou`**. Both cannot be right, and which it is
   determines whether a container can write where we expect. Not resolved here
   by guessing; needs one `spur exec <jobid> id` per node, and it may genuinely
   differ per node or per job.
2. Does `infera/engine-sglang:test-local` carry `qwen3_5.py`? Deploy is checking.
   Everything about whether an image build is on the critical path hangs on it.
3. Is `infera/engine-sglang:glm53-flash` present on `crsuse2-m2m-079` or
   `-276`? The BRIEF guesses "probably not". No module has reported a
   `docker images` from those two nodes yet. A 9m25s build is affordable; not
   knowing for another hour is not.
4. Do the profiling / integration packages' **bodies or validators hard-code
   GLM**, or are `model_path` / `image` / `served_name` / `tp` genuinely free
   variables? The BRIEF's cheap road — run them against Qwen3.6-27B at tp=2 on
   an image that already exists — depends entirely on this, and nobody has
   reported reading the files.
5. Whether analyze, kernel-opt and integration have started at all. No output is
   not evidence of no work at T+7min, but it is also not evidence of work.

### 6. New commits

Since the effort began (T+0 baseline, so this is the starting point rather than
a delta): **none.** Worktree HEAD is

```
532da57 refactor(llm_e2e): stage 1 moves into `deploy-demo/`, and the root becomes a container
```

`git status` shows three untracked entries: `.serena/`,
`deploy.debug.help.info.md`, `profiling.debug.help.info.md`. The two notes files
are expected untracked working output; the leader handles git.

### 7. Other

- **No `agent-sys run` has been launched yet by anyone.** The ws2 runroot
  `/shared_nfs/yihou/agent_sys_debug/ws2/runroot/` is empty, and all five
  deliverable dirs under `/shared_nfs/yihou/agent_sys/debugging/` are empty.
  This is the single most important fact in the baseline: at T+7min the effort
  is entirely in reconnaissance, and the 1800 s settle budget means a run is not
  a small commitment once started.
- A stray `spur-101078.out` (33 bytes) sits in the package root
  `agent_sys/examples/llm_e2e_performance_optimization/`. Harmless, but it is
  inside the deliverable tree and should not be committed.
- The two modules that have written notes are both following the "append as you
  go" instruction, and both notes are already good: each names the file, the
  exact error string, and what a later reader should do instead. That is the
  behaviour the BRIEF asked for.

---

## T+30 — 2026-09-02 09:02 UTC

### Walltime countdown — **CHANGED, badly**

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | RUNNING (44 m) | 16:17:41 | 7 h 15 m |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (44 m) | 16:17:57 | 7 h 15 m |
| ~~`101078`~~ | ~~`crsuse2-m2m-080`~~ | deploy, analyze, kernel-opt | **CANCELLED 08:55:44** | — | **gone after 27 m of an 8 h hold** |
| `101146` | (none) | replacement for the above three | **PENDING** | — | **cannot launch** |

**This is the headline of the checkpoint.** `scontrol show job 101078`:
`JobState=CANCELLED Reason=None`, `RunTime=00:27:22`, `TimeLimit=08:00:00`,
`EndTime=2026-09-02T08:55:44`. Not cancelled by us. An 8 h hold was reaped at 27
minutes. The replacement `101146` (submitted 08:57:20, `Restarts=1`) is stuck:

```
Reason=JobLaunchFailure (dispatch confirmation failed (0/1 confirmed): 1 agent unreachable)
```

So **three of five modules currently have no GPU node at all**, and the
mechanism that killed the first one is not understood. The two surviving holds
are the only compute the effort has.

### 1. Progress

**Effort: ~40 %.** Elapsed 34 minutes. Estimated remaining: **2–4 h if 101146
lands soon**; unbounded if it does not.

| module | est. % | basis |
|---|---|---|
| profiling | ~65 % | image rebuilt, `remote.sh` localised and committed, Qwen substitution proven to work, trace synthesised; run `profiling-a` root exists |
| integration | ~55 % | run `integration-r1` live at 08:54 on a surviving node, 3 handoffs open; six package files modified |
| analyze | ~55 % | run `analyze-dry2` reached `identify: succeeded` → `build_workset: running`; seed CSV synthesised and shown to reproduce the package's documented result; **node lost mid-run** |
| kernel-opt | ~50 % | mock path chosen and documented, install recipe working, run `kernel-opt-mock1` launched, 2 handoffs with a v1; **node lost mid-run** |
| deploy | ~45 % | two runs launched (`deploy-d1`, `deploy-d2`), CPX discovery forced a restart, `deploy-d2` was ~6 min into its agent phase when **the node was cancelled under it** |

**Reliability: medium.** Upgraded from "very low" because all five modules now
have substantial written notes and five run roots exist with real state
transitions in them — I am reading artefacts, not guessing. But three caveats
pull it down: (a) percent-complete for the three orphaned modules is a measure
of *knowledge acquired*, not of *handoff produced*, and the runs that would have
converted one into the other were killed; (b) **zero handoffs have been
deposited** — all five dirs under `/shared_nfs/yihou/agent_sys/debugging/` are
still empty, so by the delivery contract the effort is at 0 % delivered; (c) I
cannot forecast `101146`.

### 2. Current state, per module

- **profiling** (`crsuse2-m2m-079`, alive). The strongest position. Has rebuilt
  the image on-node (~25 min), localised `remote.sh` onto a selectable transport
  and **committed it** (`6d6b053`), synthesised the missing AIPerf trace, and —
  the important one — **proved the Qwen3.6-27B substitution works**: notes
  section 9, "the GLM hard-coding is inert, not blocking". That answers
  undetermined item 4 from T+0 for this package.
- **integration** (`crsuse2-m2m-276`, alive). Run `integration-r1`
  (`20260902T085439-d482be`) launched 08:54 with three handoff slots open. Six
  files modified in the worktree, uncommitted. Found that the node's image has
  `qwen3_5` but not `glm5_next` — which *decides* the model rather than merely
  suggesting it — and that `mix_worker.sh` hard-codes two GLM-only flag groups
  that fail as numbers.
- **analyze** (node lost). Run `analyze-dry2` had got `identify` through
  `output_validating → succeeded` and `build_workset` into `running` when the
  hold died. Four handoff slots open. Committed `4011eb7`.
- **kernel-opt** (node lost). Run `kernel-opt-mock1` launched 08:37 on the mock
  path; two handoffs, one already at v1.
- **deploy** (node lost). `deploy-d1` was launched with `tp_size=1` from prior
  art, aborted when CPX was discovered at minute 14; `deploy-d2` relaunched
  08:50 with corrected sizing and was ~6 min into the agent phase at
  cancellation. Has written a restart procedure for a fresh node.

### 3. Code problems

| # | module | problem | state |
|---|---|---|---|
| C1 | profiling | `remote.sh::on()` used `srun --overlap --export=ALL`; this cluster's `srun` is a spur re-implementation that rejects `--export` and needs a TTY (exit 128). | **fixed** — commit `6d6b053` "make the compute-node transport a variable, not srun"; srun form kept selectable |
| C2 | profiling / integration | `spur exec` carries no environment, so `--export=ALL`'s job (delivering `AGENT_SYS_OUTPUT_*` and the `PD_*` block) must be done by hand-serialising it. | **fixed** in profiling (part of `6d6b053`); **open** in integration (`remote.sh` modified, uncommitted) |
| C3 | analyze | `verify_workset` could not reach the GPU as written, same `srun` root cause. | **fixed** — commit `4011eb7` "the GPU transport and the visible cards are parameters" |
| C4 | integration | `mix_worker.sh` hard-codes two GLM-only flag groups; both fail as numbers under a non-GLM model. | **open** — file modified, not committed |
| C5 | kernel-opt | The run refuses to start unless the repo sets `extensions.preciousObjects`; setting it **on a worktree hits every other worktree** (analyze notes §3 independently). A framework-level foot-gun affecting any multi-worktree layout. | **worked around**; underlying behaviour **open** |
| C6 | kernel-opt | Python 3.10 in the only torch image makes `temp/bugs/001` live. | **open**, worked around by the install recipe in notes §6 |
| C7 | analyze | An `identify` handoff declared `usage` naming `'seconds'`, which the task did not declare: `4.596789008937776 is not booked`. Emitted as a console warning, not a failure. Whether it is a package bug or a framework leniency is not settled. | **open**, non-blocking |

**A BRIEF correction, first-hand and load-bearing.** BRIEF known-limit 1 says
the settle budget is **1800 s, hard-coded at `cli/main.py:790`**. Two modules
checked the actual file: `agent_sys/cli/main.py:903` reads
`_SETTLE_TIMEOUT = 14400.0` (4 h), with a comment recording 300 → 1800 → 14400,
each raise caused by a healthy run being reported as a hang. **The 1800 s figure
is stale.** `temp/bugs/003` and `005` should be re-read against the current
constant. This removes a constraint several modules had designed around.

### 4. Non-code problems

| # | problem | state |
|---|---|---|
| **E0** | **An 8 h hold was cancelled at 27 minutes with `Reason=None`, and its replacement cannot dispatch (`1 agent unreachable`).** Three modules lost their node simultaneously. | **OPEN — the effort's top risk.** Cause unknown |
| **E5** | **`crsuse2-m2m-080` is CPX-partitioned: 64 devices × 36 GiB, not 8 × 288 GiB.** Confirmed first-hand: `rocm-smi --showcomputepartition` → CPX on GPU[0],[8],[16],…; `torch.cuda.device_count()` → 64; every device 36.0 GiB; UUIDs identical within each group of eight. **The BRIEF's "8 × MI355X, 288 GiB each" is false on that node.** A 52 GB bf16 model cannot fit TP1 in 36 GiB — this is what aborted `deploy-d1`. Also means "GPU 2–3" in the port/GPU allocation is not HIP device 2 and 3. | **open** — must be checked per node; unknown for `-079` and `-276` |
| **E6** | **A zone on `/shared_nfs` segfaults every ROCm kernel launch.** Measured three times: with `TMPDIR` anywhere under `/shared_nfs`, `torch.ones(4, device="cuda")` exits **139** (SIGSEGV) on the first kernel launch; with `TMPDIR` unset or on `/mnt/m2m_nobackup`, exit 0. Not a hang, not a permission error. Cost kernel-opt 25 minutes of an agent correctly bisecting a fault it could not name. | **understood, worked around** — put scratch on `/mnt/m2m_nobackup`. **Every module using a GPU from an NFS-rooted zone must apply this.** |
| E1 | Images are per-node; no image the BRIEF names existed on the nodes as given. | **resolved by cost** — profiling rebuilt on-node (~25 min); integration found `qwen3_5` present, `glm5_next` absent, which settles its model choice |
| E7 | Two cold starts is integration's budget problem, and the **aiter JIT build** is most of it. | **open** |
| E3 | Missing data dependencies. | **partly resolved** — analyze synthesised the seed CSV and verified it reproduces the package's own documented result; profiling synthesised the AIPerf trace. Both declared in notes. Magpie's kernel finder is absent; analyze records that as a *supported* outcome |
| E8 | `docker commit` freezes an `--entrypoint` override into the image (profiling §10). | **noted** |

### 5. Undetermined

1. **Why was `101078` cancelled?** `Reason=None` after 27 m of an 8 h limit, not
   by us. Deploy's notes observe the survivors were also at only ~39 m and
   suspects "something is reaping these allocations well short of walltime." If
   that is systemic, `101052` and `101053` are not safe either and the whole
   plan needs shorter, checkpointed units of work. **Unresolved and urgent.**
2. **Will `101146` dispatch?** `1 agent unreachable` is a control-plane fault, not
   a queue wait. No estimate.
3. **Are `crsuse2-m2m-079` and `-276` CPX or SPX?** Deploy's restart procedure
   opens with this question because it decides `tp_size` "and nothing else
   does". Profiling and integration have runs in flight on those nodes; neither
   has reported the partition mode. Cheap to answer, expensive to assume.
4. **`spur exec` identity — root or `yihou`?** Still open from T+0. Deploy's
   notes use `docker exec -u 50112975:1000`, suggesting a numeric-uid reality
   more complicated than either answer.
5. Is the analyze `usage`/`'seconds'` warning a package bug or framework
   leniency? Nobody has adjudicated it.
6. **Can the three orphaned runs be resumed, or must they restart?** Deploy has
   written a restart procedure, implying restart. If `--resume` cannot recover
   an interrupted task's open output slot (BRIEF limit 1), all three lose their
   elapsed run time, not just their node.

### 6. New commits

Two since T+0:

```
4011eb7 feat(analyze-demo): the GPU transport and the visible cards are parameters
6d6b053 fix(profiling-demo): make the compute-node transport a variable, not srun
```

- `4011eb7` — analyze's fix for C3: makes the GPU transport and the visible-card
  set package variables instead of a baked-in `srun` call, so the CPX/device-id
  mismatch and the transport swap are both configurable.
- `6d6b053` — profiling's fix for C1/C2: the compute-node transport becomes a
  variable, `spur exec` selectable alongside the original `srun` form.

Both are correctly scoped to one module each, as the BRIEF requires.
`git status` additionally shows six **uncommitted** modifications under
`integration-demo/` (`shared.yaml`, `assets/lib/remote.sh`,
`assets/serve/{mix_up,mix_worker,round}.sh`, `assets/accept/measure.sh`) — work
in flight, at risk if that node also dies.

### 7. Other

- **Delivered: nothing.** All five dirs under
  `/shared_nfs/yihou/agent_sys/debugging/` remain empty. Five run roots with
  live state exist, but the contract is the sealed handoff plus `PROVENANCE.md`,
  and none has been copied out. **The gap between "45–65 % done" and "0 %
  delivered" is the number to watch.**
- All five modules are now writing notes as they go, and the cancellation proved
  why: deploy's sections 1–8 survived because they were written as measured; the
  run did not. That instruction earned its place today.
- Two findings here are worth more than this round — **E5 (CPX)** and **E6 (NFS
  segfault)** are cluster facts that will mislead the next effort just as badly
  if they stay in a module notes file. They belong in the BRIEF and in
  `temp/bugs/`.

---

## T+60 — 2026-09-02 09:32 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | RUNNING (1 h 11 m) | 16:17:41 | 6 h 45 m |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (1 h 11 m) | 16:17:57 | 6 h 45 m |
| `101155` | `crsuse2-m2m-019` | **deploy + kernel-opt** (and analyze's remaining leaves) | RUNNING (22 m) | ~17:07 | ~7 h 35 m |
| ~~`101078`~~ | ~~`-080`~~ | — | CANCELLED 08:55:44 | — | — |
| ~~`101146`~~ | — | — | never dispatched | — | — |

**The node crisis is resolved.** `101146` never launched; a fresh hold `101155`
on `crsuse2-m2m-019` was obtained ~09:07 and the three orphaned modules moved
onto it. Recovery took roughly 12 minutes from cancellation to a running run on
new hardware. Kernel-opt's notes record `crsuse2-m2m-080` went **`down`** at
~08:58 — so the cancellation was a node failure, not a scheduler reap. That
partly answers T+30 undetermined item 1 and materially lowers the risk to
`101052`/`101053`.

### 1. Progress

**Effort: ~70 %.** Elapsed 64 minutes. Estimated remaining: **1.5–3 h.**

| module | est. % | basis |
|---|---|---|
| **profiling** | **100 % — DELIVERED** | run `20260902T085925-9f72b7`, 7 tasks succeeded, 7 handoffs valid, **6 verdicts PASS**; deliverable + PROVENANCE.md on disk |
| **kernel-opt** | **100 % — DELIVERED** | run `20260902T090855-446cad`, 3 tasks succeeded, **3 verdicts PASS**; deliverable + PROVENANCE.md on disk |
| integration | ~70 % | `integration-r2` in flight; `serve_stock` finished; notes at §9 |
| analyze | ~70 % | all six leaves individually proven; needs one contiguous run or a stitched seal |
| deploy | ~60 % | `deploy-d3` launched 09:14 on a working node with a purpose-built image |

**Reliability: good, for the first time.** Two modules are done and I verified
them by opening the artefacts, not by reading a claim: both deliverable trees
contain `manifest.yaml` + `validation.yaml` + sealed `content`, and both
PROVENANCE files quote the run report's verdict lines verbatim. The three
remaining estimates are still inference from notes and run roots.

**Delivered: 2 of 5.** That is the number that was 0 at T+30.

### 2. Current state, per module

- **profiling — DONE.** Ran against **Qwen3.6-27B at tp=2** on the rebuilt
  `gfx950-local` image, ports 8120–8122, with `--timeout 10800`. Every window
  reduced to the cheapest satisfying value (`warmup_s=60`, `window_s=10`,
  `stack_window_s=3`, `max_conc=32`), trace synthesised. Seven artefacts
  delivered. Its PROVENANCE opens by refusing to let its own numbers be quoted.
- **kernel-opt — DONE.** Mock mode on GPU 4 of `crsuse2-m2m-019`, in a
  `rocm/pytorch:rocm7.2.4…py3.12…2.10.0` image (better than the README's, and
  it sidesteps the Python 3.10 bug), demo-root on **node-local**
  `/mnt/m2m_nobackup` — mandatory because of E6. Its PROVENANCE is the most
  self-critical document in the effort: it states in bold that no kernel was
  optimised, marks `optimized_kernel.py` byte-identical to the seed (md5
  independently re-checked by the producer), and notes that
  `check_speedup_substantiated` PASSed *without measuring anything*, by
  documented mock behaviour at `check.py:213-217`.
- **integration** (`-276`, alive). On `integration-r2`. `serve_stock` completed;
  `measure_stock` observed live. Committed `fc8699f`.
- **analyze** (moved off the dead node). Has proven **all six leaves**
  individually — four in the killed `analyze-dry2` run, two by standalone probes
  (`transport_probe.sh`, `packup_probe.sh`, both PASS, locality clean). What it
  lacks is one contiguous run. `build_workset` costs ~10 min per operator, which
  is its remaining cost driver.
- **deploy** (`-019`, `101155`, SPX). `deploy-d3` started 09:14:19Z, `tp_size=1`,
  `mix` mode, on `infera/engine-sglang:gfx950-deploy` **built on-node in ~4
  minutes** (notes §12 carries the whole recipe). Third launch: d1 killed by
  CPX, d2 killed by the node, d3 is the run.

### 3. Code problems

| # | module | problem | state |
|---|---|---|---|
| C1/C2 | profiling | srun transport + environment serialisation | **fixed**, `6d6b053`, and now **proven in a passing run** |
| C3 | analyze | GPU transport / visible cards as parameters | **fixed**, `2820d47` + `4011eb7` |
| C4 | integration | `mix_worker.sh` GLM-only flag groups; transport seam | **fixed**, `fc8699f` "localise the transport seam and unbind the model" |
| C5 | all | `extensions.preciousObjects` in a **worktree** writes to the SHARED common config (`/home/yihou/dev/git/infera/.git`), hitting four other agents' worktrees. Independently hit by analyze (§3), profiling (§11), integration (§7), kernel-opt (§5). | **worked around** — profiling and kernel-opt both ran from a private clone. The framework requirement is **open** and is the effort's most-repeated foot-gun |
| C8 | framework | **An unparseable `${...}` is passed through, not refused** — committed as `13d1c2b`. Related: integration §6, `${x-default}` is not agent_sys variable syntax and "fails far away from the cause". | **documented**, behaviour **open** |
| C7 | analyze | `usage` naming an undeclared `'seconds'` | **open**, non-blocking |
| C6 | kernel-opt | Python 3.10 / `bugs/001` | **avoided** — a py3.12 image sidesteps it |

**Two BRIEF facts are now confirmed stale by three independent modules**
(kernel-opt §2, analyze §7, integration §9, profiling implicitly via
`--timeout 10800`):

1. **The 1800 s settle budget does not exist.** `cli/main.py` has
   `_SETTLE_TIMEOUT = 14400.0` and `--timeout` is a real flag (`main.py:166`).
   Integration calls this "the single most expensive stale fact in the brief,
   because it makes people trade away resolution to fit a ceiling that is not
   there." The comment records the 1800 s value once killed a healthy 27 B
   bring-up at exactly 1800.0 s and abandoned eight held GPUs.
2. **`Nothing has changed for 20 s` is a diagnostic, not a termination.**
   Integration's run printed it during `serve_stock` and finished that task 276 s
   later.

### 4. Non-code problems

| # | problem | state |
|---|---|---|
| E0 | The `101078` cancellation | **root cause found** — `crsuse2-m2m-080` went **`down`** ~08:58, so this was hardware/node failure. Recovered onto `101155` in ~12 min. `101146` never dispatched and was abandoned rather than waited on — the right call |
| E5 | **Partition mode varies BETWEEN nodes.** `-080` was CPX (64 × 36 GiB); `-019` is **SPX**, so `deploy-d3` runs `tp_size=1`. Deploy §11: "check yours, do not inherit a number." | **understood**; deploy now sets the per-node stanza via a `${GPU_NOTE}` variable and a thin wrapper rather than editing the driver |
| E6 | **NFS `TMPDIR` segfaults every ROCm kernel launch** (exit 139) | **understood, worked around, and committed** as `7016ee5`. Kernel-opt's delivered run put its demo-root on `/mnt/m2m_nobackup` *because* of this |
| E1 | Per-node images | **resolved by building**: profiling rebuilt `gfx950-local` (~25 min); deploy built `gfx950-deploy` in **~4 min** with the recipe in its §12. The 4-minute path is the one to reuse |
| E3 | Missing inputs | **resolved**: profiling synthesised `conversation_trace.jsonl` (generator `make_trace.py` shipped beside the handoff); analyze synthesised the seed CSV and showed it reproduces the package's documented result. Both declared in bold in their PROVENANCE |
| **E9** | **Qwen3.6-27B decodes at ~2.9 tok/s on two MI355X** (~345 ms/token), two orders of magnitude below bandwidth arithmetic for a dense 27 B BF16 at tp=2. Integration §8: "every number in this handoff inherits that." | **open** — does not block a mock sample, but it makes integration's wall-clock budget much worse and is a real anomaly |
| E7 | aiter JIT build dominates cold start | **open** |

### 5. Undetermined

1. **Why does Qwen3.6-27B decode at 2.9 tok/s?** (E9.) Integration ruled out
   "the model is big" by arithmetic. Nobody has diagnosed it. It is the most
   interesting open question in the effort and the one most likely to matter
   beyond today.
2. **Can analyze deliver without one contiguous run?** All six leaves are proven,
   four in a killed run and two by standalone probe. Whether a handoff stitched
   from those is acceptable under "never hand-write a handoff the package did
   not produce", or whether a fresh contiguous run is required, is a **judgement
   call nobody has made**. At ~10 min per operator for `build_workset`, the
   difference is maybe an hour. **This should be settled by the leader, not by
   analyze alone.**
3. Was `101078`'s node failure isolated, or is `-080` symptomatic? `101146`'s
   `1 agent unreachable` suggests the control plane knew something was wrong.
4. C7, the analyze `usage`/`'seconds'` warning — still unadjudicated.
5. `spur exec` identity — no longer blocking anything (everyone uses
   `docker exec -u 50112975:1000`), but still formally unanswered.

### 6. New commits

Five since T+30:

```
2820d47 docs(analyze-demo): localisation notes — no srun, CPX cards, a synthetic seed, and a settle budget that moved
13d1c2b docs(llm_e2e): an unparseable ${...} is passed through, not refused
7016ee5 docs(kernel-opt-demo): a zone on NFS segfaults every ROCm kernel launch
fc8699f feat(integration-demo): localise the transport seam and unbind the model
e5bf9f7 docs(profiling-demo): localisation notes for the spur cluster
```

- `2820d47` — analyze's four localisation findings as package docs.
- `13d1c2b` — the only **framework-level** finding committed so far: agent_sys
  passes an unparseable `${...}` through instead of refusing it. Correctly filed
  at `llm_e2e` level, not under one module.
- `7016ee5` — the NFS/ROCm segfault, written up where the next reader will hit it.
- `fc8699f` — integration's C4 fix, transport seam + model unbinding.
- `e5bf9f7` — profiling's notes committed into the package.

Four of the five modules have now committed. `git status` is clean of module
work except `deploy.debug.help.info.md` (untracked) — the four other notes files
have been committed into their packages, which is better than leaving them at
the repo root.

### 7. Other

- **The delivery gap from T+30 is closing correctly.** Both delivered modules
  shipped `PROVENANCE.md` alongside the sealed handoff, and both PROVENANCE
  files lead with what is *not* real. Kernel-opt's goes furthest: it separates
  measured from synthetic file by file, and reports that its own A/A null
  control makes `mean_case_speedup: 1.0` "ground truth by construction, not a
  result". This is exactly the "read the artefact, not the exit code" discipline
  the BRIEF asked for, applied by the producers to themselves.
- One genuine incidental finding, from kernel-opt's mock: the workset's baseline
  of **55.40 µs was traced on gfx942/MI300X and does not reproduce on gfx950** —
  this host measures **50.141 µs**, −9.5 %. The kit marks the cross-check
  `DIVERGED` rather than hiding it. Any future gfx950 speedup must be taken
  against 50.141 µs.
- The **4-minute image build** in deploy §12 supersedes profiling's 25-minute
  rebuild. If a sixth module ever needs an engine image, that is the recipe.
- Still unaddressed from T+30: **E5 (CPX) and E6 (NFS segfault) are now in
  package notes and one commit, but not in the BRIEF.** The next effort reads the
  BRIEF first.

---

## T+90 — 2026-09-02 10:02 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | RUNNING (1 h 42 m) | 16:17:41 | 6 h 15 m |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (1 h 42 m) | 16:17:57 | 6 h 15 m |
| `101155` | `crsuse2-m2m-019` | deploy, analyze, kernel-opt | RUNNING (52 m) | ~17:07 | ~7 h 05 m |

Stable. No further node loss. Walltime is **not** the binding constraint on any
module right now.

### 1. Progress

**Effort: ~78 %.** Elapsed 94 minutes. Estimated remaining: **1.5–3 h.**

| module | est. % | basis |
|---|---|---|
| **kernel-opt** | **100 % — DELIVERED (re-delivered)** | `run_mock4.log`; the T+60 delivery moved to `superseded/` and was replaced after a committed fix |
| **profiling** | **100 % — DELIVERED** | run A's handoff set stands; two later runs are hardening, not delivery |
| analyze | ~85 % | `analyze-r1`: **5 of 6 handoffs sealed, 5 verdicts PASS**; the sixth lost in a 20 s window. `analyze-r2` re-running, at `build_workset` |
| integration | ~75 % | `integration-r3` starting; four new findings written since T+60 |
| **deploy** | **~60 %, and it is now the module I know least about** | `deploy-d3` console has not advanced past `running` since **09:14** and its notes have not grown since **09:17** — 45 minutes silent |

**Reliability: good for four modules, poor for deploy.** Two deliveries verified
on disk. Analyze and integration are both writing detailed notes and have live
run roots. **Deploy is the gap**: a console frozen at `input_validating ->
running` for 45 min is the *expected* appearance of a long agent phase — the
agent works inside the container and the console stays quiet — so this is not
evidence of trouble. But it is also not evidence of progress, and deploy is the
only module whose notes have not grown in this interval. I am not going to score
it from nothing; I am flagging it.

### 2. Current state, per module

- **kernel-opt — DELIVERED, second edition.** Moved the T+60 deliverable into
  `superseded/` and re-ran (`run_mock4.log`) after committing `45721e6`: the GPU
  target is a fact about the host, not about the package. Re-delivering rather
  than patching the shipped artefact is the right instinct.
- **profiling — DELIVERED, still hardening.** Run A's seven artefacts stand.
  Runs B and C exposed two further faults (below) and produced two commits.
- **analyze.** `analyze-r1` (`20260902T091144-096985`, `top_n=2`) got five
  handoffs sealed and **five verdicts PASS**; `build_workset` cost **1419 s, 67
  turns, $13.79**; `verify_workset` measured both operators. The terminal
  `analyze_packup` stuck at `generating`. `analyze-r2` is re-running.
- **integration.** `integration-r3`. Four findings since T+60 — a missing eval
  module, a hard floor in the eval size, and a constraint on synthetic traces.
- **deploy.** `deploy-d3` in its agent phase since 09:14:19Z. Silent.

### 3. Code problems

New this interval:

| # | module | problem | state |
|---|---|---|---|
| **C9** | **framework** | **A terminal task gets 20 s total for prepare, body and seal.** This is what cost analyze its sixth handoff. Analyze ruled out both usual suspects first-hand: the gate's executable rule (`ls -la` shows the `command` item at **0755**, `packup.py:122`'s `chmod` took) and a slow body (the same body over the same sealed handoffs runs in **0.444 s**). Filed as a bug doc, commit `6e313e8`. | **documented, open** — a real framework defect, and the most consequential code finding since the settle-budget correction |
| **C10** | integration | `sglang.test.run_eval` is **missing from an image that has sglang** | **open/worked around** |
| **C11** | integration | `min_scored_per_eval` is **20, hard-coded**, so `eval_examples` has a floor — the BRIEF's "an eval over 20 questions is fine" is a floor, not a suggestion | **open**, by design |
| **C12** | profiling | `DSA_ARGS`/`PARSER_ARGS` hoisted out of `mix_worker.sh` | **fixed**, `2735e0a` |
| **C13** | profiling | `agent-sys` refuses to start with *"the 'claude' backend is not on PATH"* **even for a package with no AI agent at all** — every closure in profiling-demo is `kind: program`. The check is unconditional. | **open**, worked around by exporting PATH |
| **C14** | framework | **A nested default is a load error**, same family as C8's bare dash | **documented**, `0d6c1b6` |
| C15 | kernel-opt | GPU target was baked into the package | **fixed**, `45721e6` |

Carried forward: C5 (`preciousObjects` hits sibling worktrees) and C8/C14
(variable-syntax errors reported far from their cause) remain open.

### 4. Non-code problems

**A correction to my own T+30 and T+60 reporting.** I wrote E6 as *"a zone on
`/shared_nfs` segfaults every ROCm kernel launch"* and advised keeping run roots
off NFS. **That rule is too broad.** Profiling ran a full 7-task graph — two
engine bring-ups, two AIPerf replays, four profiler captures, thousands of
kernel launches — with `--demo-root` **on `/shared_nfs`** and saw no segfault.
Kernel-opt accepted the correction and narrowed the rule (its §14):

> `TMPDIR` on NFS kills ROCm kernel launches **only for processes that run
> inside the agent_sys zone.**

The two shapes differ: profiling's kernels all run inside a docker container the
zone *starts*, which has its own `/tmp`; kernel-opt runs `driver.py` in the zone
on the host python, which is where the fault bites. So:

| what | where |
|---|---|
| `--demo-root` (workspace, playground, handoffs) | **`/shared_nfs`** — the user's standing instruction |
| `TMPDIR` for a **zone-launched** GPU process | **node-local disk** |

Per the append-only rule I have not edited T+30 or T+60; the earlier, broader
statement stands there as written and is corrected here. Anyone reading this
file for the rule should take **this** paragraph.

New:

| # | problem | state |
|---|---|---|
| **E10** | **The login node OOM-kills a long run, and the symptom names nothing.** Profiling's run B died with no error line: log stops mid-graph, task sits at `running`, **the body outlived the driver** — it took SIGPIPE writing to the dead parent's stdout just before handoff assembly, leaving a `claim` and an **empty `content/`**. `crs-m2m-cpu-spur-012` was at **1 GB free of 62 GB, load 30.41, 169 users**. Nothing in the package had changed. | **fixed** — run `agent-sys` **on the compute node**; commit `8274a08` adds `PD_TRANSPORT=local`. Deliberately never chosen by `auto`, because "neither transport binary is present" is not the same fact as "I am on the node", and guessing wrong runs every GPU command on the login node |
| **E11** | `pgrep -f "agent-sys run"` **matches your own shell** — a trap. Use `ps -eo pid,cmd \| grep agent-sys \| grep -v grep`. | noted |
| **E12** | A synthetic Mooncake trace must give each `hash_id` **one fixed block size** | **fixed** in integration's generator |
| E9 | Qwen3.6-27B at ~2.9 tok/s | **still open, still undiagnosed** |

### 5. Undetermined

1. **What is deploy doing?** 45 minutes of silence on both console and notes.
   Consistent with a healthy long agent phase; also consistent with a stall.
   Cheap to answer and I would rather ask than score it.
2. **Why does `analyze_packup` need more than 20 s?** C9 says the budget is 20 s
   and the body takes 0.444 s. Those two facts do not yet explain a failure —
   something between prepare and seal is consuming the rest, and analyze has not
   yet named it. **The most interesting open question in the effort right now.**
3. **E9, the 2.9 tok/s decode.** Untouched since T+60.
4. **Is profiling's delivered run A safe from E10?** Run A predates the
   discovery. Its verdicts were printed and its content is on disk, so the
   empty-`content` signature does not apply — but nobody has re-checked run A's
   artefacts against the trap that killed run B.
5. C7 (`usage`/`'seconds'`) — still unadjudicated, three checkpoints on.

### 6. New commits

Six since T+60:

```
2735e0a fix(profiling-demo): hoist the DSA and parser flag groups out of mix_worker.sh
6e313e8 docs(llm_e2e): a terminal task gets 20 s for prepare, body and seal
8274a08 feat(profiling-demo): add a 'local' transport for driving from the node itself
c502d00 docs(kernel-opt-demo): record the TMPDIR fix and the run that proves it
0d6c1b6 docs(llm_e2e): a nested default is a load error, same family as the bare dash
45721e6 fix(kernel-opt-demo): the GPU target is a fact about the host, not about this package
```

- `2735e0a` — GLM-only flag groups become variables (integration's C4 in
  profiling's copy).
- `6e313e8` — **the 20-second terminal-task budget**, a 116-line bug doc. The
  highest-value commit of this interval.
- `8274a08` — the `local` transport, fixing E10.
- `c502d00` — the TMPDIR fix plus the run proving it.
- `0d6c1b6` — nested-default load error, third in the variable-syntax family.
- `45721e6` — GPU target de-hardcoded.

Three of six are `docs(llm_e2e)` **framework-level** bug records rather than
module fixes. That ratio is healthy: the effort is now finding defects in
`agent_sys` itself, not just localising packages.

Uncommitted: one modified `integration-demo/assets/accept/lm_eval.sh`, plus four
notes files modified in place and `deploy.debug.help.info.md` still untracked.

### 7. Other

- **Delivered: 2 of 5**, unchanged in count but not in quality — kernel-opt
  superseded its own T+60 delivery after finding a fix worth re-running for.
  Re-delivering beats patching a sealed artefact.
- **Analyze's 5-of-6 is the near-miss to watch.** It has PASS verdicts on
  `check_kernel_table`, `check_worklist_shape`, `check_identity_resolved`,
  `check_workset_shape` and `check_workset_runs`. Only the terminal seal is
  missing, and the cause is a framework budget, not the package.
- **Cost is now visible**: `build_workset` alone was **$13.79** for 67 turns.
  Nobody has aggregated spend across the effort; at five modules with reruns it
  is no longer negligible.
- Deploy's 4-minute image recipe (§12) and profiling's `local` transport
  (`8274a08`) are the two reusable assets produced today that a later effort will
  want first.

---

## T+120 — 2026-09-02 10:33 UTC

### Walltime countdown — **a second node lost**

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | RUNNING (2 h 12 m) | 16:17:41 | 5 h 45 m |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (2 h 12 m) | 16:17:57 | 5 h 45 m |
| `101576` | `crsuse2-m2m-260` | deploy (+ analyze, kernel-opt) | RUNNING (12 m) | ~18:18 | ~7 h 45 m |
| ~~`101155`~~ | ~~`-019`~~ | — | **CANCELLED 10:18** | — | — |
| ~~`101078`~~ | ~~`-080`~~ | — | CANCELLED 08:55 | — | — |

**Two of the three short-lived holds have now been reaped mid-run, neither at
walltime, 83 minutes apart.** Deploy's §15 states it plainly: `101155` died
at 10:18 with `deploy-d3` minutes from its validation phase, same signature as
§9. Meanwhile `101052` and `101053` have run **2 h 12 m untouched**. The
mechanism is not hitting every hold equally.

**Operational rule this now justifies:** any module planning a run longer than
~40 minutes on a freshly issued hold should assume it may not survive, and
should checkpoint the artefact to `/shared_nfs` continuously rather than at the
end. Deploy did exactly that and it saved its deliverable.

### 1. Progress

**Effort: ~85 %.** Elapsed 125 minutes. Estimated remaining: **1–2.5 h.**

| module | est. % | basis |
|---|---|---|
| **profiling** | **100 % — DELIVERED** | 7 handoffs, 6 verdicts PASS, run report |
| **kernel-opt** | **100 % — DELIVERED**, and now self-verified | `digest-selfcheck.txt`: both handoffs' manifest digests recomputed and **VERIFIED True** at 10:15:10Z |
| **deploy** | **~92 % — DELIVERED, with one validator short** | 62-file `content/` + PROVENANCE on disk; `check_deploy_kit` **PASS**, `check_deploy_reproduces` never ran |
| analyze | ~88 % | `analyze-r2`: **5 of 6 sealed, 5 verdicts PASS** — the *same* terminal failure as r1, now diagnosed to a framework line |
| integration | ~75 % | **no notes growth since 09:46** and no new run root — my blind spot this interval |

**Reliability: good, with one hole.** Three deliverables verified on disk by
opening them. Analyze's two run consoles read directly. **Integration is now the
unknown** — 47 minutes without a note or a new run root. I said last checkpoint
I would rather ask than guess; the same applies here, and I have not asked
integration yet.

**Delivered: 3 of 5** (2 clean, 1 with a documented gap).

### 2. Current state, per module

- **deploy — DELIVERED, and the honesty here is exemplary.** The node died
  minutes before validation, so the handoff was **never sealed** and there is
  **no run report**. Rather than dress that up, its PROVENANCE leads with *"Read
  this before quoting a verdict"* and a two-row table showing one validator ran
  and one did not, ending: **"Do not write this up as 'both validators PASS'."**
  What it does have is real: `check_deploy_kit` run offline with the package's
  own validator body and the exact `args.json` the run would have passed,
  over this exact content → `{'…-0001': True}`. 62 files, one packup
  `qwen3.6-27b-mix-sglang-gfx950.packup_20260902`. `deploy-d4` is now running on
  `-260` to close the gap.
- **kernel-opt — DELIVERED and hardened.** Ten further notes sections (§16–§22)
  on whether the delivered handoffs work as **fixtures** — answered "as
  delivered, no", then fixed by a scripted relayout (`relayout_handoffs.py`,
  commit `f93e982`, failure paths tested). Two findings settled **by experiment
  rather than by reading**, including that "staging does not verify, and a
  damaged fixture passes silently".
- **analyze — 5 of 6, twice, and now diagnosed.** `analyze-r2` reproduced r1
  exactly: `check_kernel_table`, `check_worklist_shape`,
  `check_identity_resolved`, `check_workset_shape`, `check_workset_runs` all
  **PASS**; `analyze_packup` stuck at `generating`. See C9 below — the
  diagnosis is now precise and it is a framework defect.
- **profiling — DELIVERED.** One commit this interval (`91cb3b9`).
- **integration.** Unknown. Last note 09:46, last run root `integration-r3`.

### 3. Code problems

**C9 is now diagnosed, and it is an `agent_sys` defect, not a package one.**
Analyze ruled out every documented cause first-hand:

- **not** the gate's executable rule — `items/command` is mode **0755**;
- **not** a slow body — the same body over the same handoffs runs in **0.444 s**;
- **not** seal refusal on locality — it ran the *framework's own*
  `handoff.locality.check` (not the package's offline copy) over both the lost
  `analyze_packup` content and the `operator_workset` content that sealed fine:
  **both "locality OK"**.

> The content was acceptable; **the version simply was never pinned.**

The remedy is in `agent/runner.py` — `_seal_outputs` has a **silent-skip
branch** — and is therefore not available from a task package. Analyze also
found the institutional memory: `cli/README.md` records this symptom being
investigated once before, attributed to `HandoffStore.put` having no caller,
then "corrected by measurement" when `_seal_outputs` worked on the case tested.
**The two silent branches were left in place. This package's terminal task hits
one of them, two runs out of two.** That is a reopened bug with a reproducer.

Its advice to a later reader: do not shrink the graph for the settle budget
(four hours now) — shrink so the **terminal task's prepare + body + seal fits in
twenty seconds**. Prepare grows with declared inputs, and `pack_analyze` declares
four, the most in the package. Cheap lever: lower `top_n`.

| # | module | problem | state |
|---|---|---|---|
| **C9** | **framework** | terminal-task 20 s budget + `_seal_outputs` silent-skip in `agent/runner.py`; content valid, version never pinned | **diagnosed, open** — needs a framework fix |
| C16 | framework | **two `copy_out` functions, and the docstring promises the wrong one** — `dd59cf0` | **documented** |
| C17 | framework | **a chmod on the package source seals into a valid digest** — `5bc148a`; and the exec bits in the delivered handoff are "original to the seal because my chmod got there first" (§21) | **documented** — a supply-chain-shaped observation about what a digest does and does not attest |
| C18 | kernel-opt | delivered handoffs did not work as fixtures | **fixed** — relayout applied (`cae64eb`) and scripted (`f93e982`) |
| C19 | profiling | `REPRODUCE.md`'s machine note must match the transport | **fixed**, `91cb3b9` |

Carried open: C5 (`preciousObjects`), C8/C14 (variable syntax), C10/C11
(integration eval), C13 (unconditional claude-on-PATH check), C7.

### 4. Non-code problems

| # | problem | state |
|---|---|---|
| **E0′** | **Second unexplained hold cancellation** (`101155`, 10:18, mid-run). Two of three short holds reaped; the two long-lived ones untouched at 2 h 12 m. | **OPEN — recurring, still unexplained.** Now demonstrably a pattern, not an incident |
| **E13** | `deploy` recorded the mitigation that worked: because `--demo-root` was on `/shared_nfs`, the cancellation took *the run*, not *the artefact*. 62 files secured to the deliverable dir **immediately**, before anything else. | **the standing practice** — and note it cuts against a naive reading of the old E6 advice |
| E9 | Qwen3.6-27B ~2.9 tok/s | **still open, still undiagnosed** — three checkpoints |
| E10 | login-node OOM | **fixed** (`8274a08`) |

### 5. Undetermined

1. **Why are holds being cancelled?** Two in 83 minutes, mid-run, `Reason=None`,
   while two other holds run untouched for over two hours. Deploy has raised it;
   nobody can explain it. **This is the effort's top unresolved risk** and it is
   outside any module's control.
2. **What is integration doing?** 47 minutes without a note or a run root.
   Same blind spot deploy was at T+90 — and that one resolved into "working
   hard, just not writing". Not scored from nothing.
3. **Will `deploy-d4` seal, or will it hit C9 too?** Deploy's terminal task is
   the same shape as analyze's. If the 20 s budget bites deploy as well, that is
   two of five packages blocked on one framework line, and the case for fixing
   `_seal_outputs` rather than working around it becomes decisive.
4. **Should C9 be fixed rather than documented?** CLAUDE.md says fix only on
   unambiguous evidence. Two reproductions, a named function, a ruled-out
   alternative list, and a prior investigation that closed it wrongly is close to
   unambiguous. **A leader-level call.**
5. E9 — untouched.

### 6. New commits

Eight since T+90:

```
dd59cf0 docs(llm_e2e): two copy_out functions, and the docstring promises the wrong one
cae64eb docs(kernel-opt-demo): apply the relayout instead of documenting it
ba0880e docs(llm_e2e): the staging route was disproved by experiment, not only read
9b05691 docs(kernel-opt-demo): the verifying copy_out is the producer side (closes section 18)
5bc148a docs(kernel-opt-demo): a chmod on the package source seals into a valid digest
f93e982 docs(kernel-opt-demo): script the delivery relayout, with its failure paths tested
91cb3b9 fix(profiling-demo): REPRODUCE.md's machine note has to fit the transport
e881e64 docs(deploy): the spur-cluster localisation notes for deploy-demo
```

Six of eight are kernel-opt or framework findings about the **handoff/digest/
staging machinery** — a coherent line of investigation, each step closing a
numbered open question from the one before (`ba0880e` explicitly notes a route
was "disproved by experiment, not only read", and `9b05691` "closes section 18").
`e881e64` finally commits deploy's notes, so all five modules have now committed.

Uncommitted: `integration-demo/assets/accept/lm_eval.sh` plus three notes files
modified in place.

### 7. Other

- **Delivered 3 of 5**, and the quality bar has held under pressure. All three
  PROVENANCE files lead with what is *not* real: profiling ("do not quote a
  number"), kernel-opt ("no kernel was optimized", md5-identical, mock validator
  passed without measuring), deploy ("do not write this up as both validators
  PASS"). Under a node dying minutes before validation, deploy chose to ship the
  gap rather than paper it.
- **Kernel-opt independently re-verified its own delivery**: `digest-selfcheck.txt`
  recomputes both manifest digests and reports `VERIFIED True` for each, dated
  and hostnamed. Nobody asked for that.
- **The framework, not the cluster, is now the main obstacle.** Node loss cost
  time but no artefacts; C9 is the one defect standing between analyze and a
  complete handoff, and possibly deploy too.
- **I have been blind on one module per checkpoint** — deploy at T+90,
  integration at T+120. Both times the module was in fact working. That is worth
  noting as a property of *this reporting method*, not of the modules: a module
  deep in a long run looks identical to a stalled one from the outside.

---

## T+150 — 2026-09-02 11:03 UTC

### Walltime countdown — **a third node lost; only the original two remain**

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling | RUNNING (2 h 43 m) | 16:17:41 | 5 h 14 m |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (2 h 43 m) | 16:17:57 | 5 h 14 m |
| ~~`101576`~~ | ~~`-260`~~ | deploy/analyze/kernel-opt | **CANCELLED 10:54**, 34 min in | — | — |
| ~~`101155`~~ | ~~`-019`~~ | — | CANCELLED 10:18, ~70 min in | — | — |
| ~~`101078`~~ | ~~`-080`~~ | — | CANCELLED 08:55, ~30 min in | — | — |

**Three of three freshly issued holds have been reaped. Zero of the two original
holds have.** Deploy's §16 lays out the pattern and draws the only safe
conclusion:

> **On this cluster a freshly issued hold is not a resource you can plan a
> 60-minute job around.**

Deploy, analyze and kernel-opt now have **no GPU node at all**. All three have
already delivered, so this is not currently blocking — but nothing GPU-bound can
be re-run by those three modules without a new hold, and new holds do not last.

### 1. Progress

**Effort: ~93 %.** Elapsed 155 minutes. Estimated remaining: **30–90 min**,
essentially all of it integration's.

| module | est. % | basis |
|---|---|---|
| **profiling** | **100 % — DELIVERED**, relaid out to sealed-handoff shape (`store/`) | run report, 6 verdicts PASS |
| **kernel-opt** | **100 % — DELIVERED**, digests self-verified | 3 verdicts PASS |
| **deploy** | **100 % — DELIVERED**, one validator short and said so | `check_deploy_kit` PASS offline |
| **analyze** | **100 % — DELIVERED**, unsealed and said so | 5 of 6 sealed + 5 verdicts PASS; terminal content real but unsealed |
| **integration** | **~80 %** | alive on `101053`; lost an arm to a validator floor and is re-running |

**Reliability: high for four modules, medium for integration.** Four
deliverables opened and read. Integration's notes resumed (496 lines, §13 at
10:48) so the T+120 blind spot is closed — it was working, as deploy's was.

**Delivered: 4 of 5.**

### 2. Correction to T+120: C9's root cause was *not* the 20-second clock

At T+120 I reported the terminal-seal failure as a framework defect — a 20 s
budget plus `_seal_outputs`' silent-skip branch. **Analyze has since found the
actual cause and it is the package's own schema** (commit `a449191`, *"the root
cause was our own `items_schema`, not the clock"*). Run directly against the
real unsealed content:

```
check_items REFUSED: items $: Additional properties are not allowed
  ('REPRODUCE.md', 'environment.md', 'notes.md', 'results' were unexpected)
```

`analyze_packup`'s `items_schema` declared six items with
`additionalProperties: false`; `packup.py` writes **ten**; and the four
undeclared ones are exactly the four that `check_analyze_packup_shape`
**requires**. In analyze's words:

> **The producer, the validator and the kind had drifted apart, and the kind was
> the one nobody ran.**

Five of six handoffs seal fine, so nothing exercised the mismatch until the
terminal one. **Fixed in `acb8bfe`**, verified both ways against real content:
`check_items` refuses under the old schema, accepts under the new.

The framework half of C9 survives but is demoted from *cause* to *symptom
amplifier*: `seal` returns the refusal **as a string, not an exception**
(deliberately, so `agent` need not import `handoff`), `_seal_outputs` files it
under `seal_refused`, and `agent/runner.py`'s own docstring admits that key
"has no reader outside these tests yet". So **the reason exists, is correct, is
specific — and is discarded**; the operator sees a task stuck in `running` and a
timeout. That is commit `755e1a4`: *"the stall was the symptom; `seal_refused`
having no reader is the bug."*

Per the append-only rule, T+120 stands as written. **This paragraph is the
correct account.** My T+120 recommendation ("shrink so the terminal task fits in
20 s") was aimed at the wrong target; the right first move is analyze's:

```python
from handoff import content as c
c.check_items(c.load(Path(content_dir)), c.content_type("reproducible"), items_schema)
```

Any handoff whose producer writes more items than its kind declares is exposed,
and `additionalProperties: false` is house style in these packages.

### 3. Current state, per module

- **analyze — DELIVERED, honestly unsealed.** PROVENANCE opens: *"Read the two
  warnings in section 1."* The terminal content is **`generating`, not `valid`** —
  it predates `acb8bfe`. What is verified rather than assumed:
  `check_analyze_packup_shape.check()` called directly returns
  *PASS — 4 mandated file(s) present with substance*; the framework's own
  `handoff.locality.check` passes; `check_items` refuses under the old schema and
  accepts under the new. Its summary is the right one: *"byte-for-byte what a
  sealed version would have held, and the run report does not say so. Both facts
  are true and neither should be dropped."*
- **deploy — DELIVERED, and step 1 of its assignment answered in full.** Its §17
  answers the original question — did the move to `deploy-demo/` break anything —
  **negatively and itemised**: package loads (2 tasks/2 closures/2 validators),
  exec bits survived, both validator bodies still resolve and are non-vacuous
  (all four controls reproduce), and a real run drove it end to end on two
  different nodes. *"Nothing in this module's failures was caused by the move.
  Every one was the cluster."*
- **profiling — DELIVERED**, now in sealed-handoff shape (`store/`), plus two
  more identifiers turned into parameters (`f4f920c`) and a `check_items` drift
  audit **clean for all seven kinds** — the same class of fault analyze was bitten
  by, checked for proactively and found absent.
- **kernel-opt — DELIVERED.** No change since T+120.
- **integration.** Live on `101053`, the safest node. Lost an arm to a validator
  floor (below) and is re-running.

### 4. Code problems

| # | module | problem | state |
|---|---|---|---|
| **C9** | **analyze package** | `items_schema` declared 6 items with `additionalProperties: false`; producer writes 10; the 4 undeclared are the 4 the validator requires | **FIXED**, `acb8bfe`, verified both directions |
| **C9b** | **framework** | `seal_refused` has **no reader** — a correct, specific refusal is discarded and surfaces as a stalled task | **documented, open**, `755e1a4`. Still worth fixing: it turned a one-line schema bug into two lost runs |
| **C20** | integration | `check_bench_report` FAILED on a complete bench handoff: `request_count.avg = 30.0` against `min_requests: 50`. `trace_end_ms=15000` looked free but the replay is **fixed-schedule** — the window truncates the trace, it does not compress it | **fixed**, by lengthening the trace rather than lowering the bar |
| C21 | profiling | container name and context length were hard-coded | **fixed**, `f4f920c` |
| C22 | deploy | README updated for the move | in flight (uncommitted) |

**Integration's floor audit is the reusable artefact of this interval.** After
C20 it audited every numeric floor out of the step files — *"worth doing before
the first run, not after the third"*:

| validator | arg | value | overridable? |
|---|---|---|---|
| `check_acceptance` | `min_scored_per_eval` | 20 | **no** |
| | `needle_min_depths_retrieved` / `needle_min_token_ratio` | 1 / 0.95 | **no** |
| `check_bench_report` | `min_requests` | 50 | yes |
| | `max_error_rate` | 0.05 | yes |
| | `expect_rounds` | `${bench_rounds:-2}` | yes, tracks `bench_rounds` |
| `check_packup_shape` | min content lines (README 20, REPRODUCE 15, environment 12, notes 8) | | **no** |
| | `min_command_lines` / `min_result_files` | 8 / 4 | **no** |
| `check_service_live` | `expect_workers` | 1 | **no** |

This directly qualifies the BRIEF's "cheapest settings win": **there is a floor,
several floors are literals unreachable from the command line, and going under
one costs the whole arm.** Integration measured that cost at **50 minutes**, and
declined the available `--var min_requests=25` on the grounds that *"lowering the
bar to meet the sample changes what the validator means."* That is the right
call and it should be quoted at anyone tempted to tune a threshold to fit.

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| **E0″** | **Three of three freshly issued holds reaped mid-run; both original holds untouched at 2 h 43 m.** | **OPEN, now a confirmed pattern.** The two survivors were issued in the original batch; every later one died |
| **E14** | **The staging lesson — this is the finding of the interval.** `deploy-d3` and `-d4` were killed at similar maturity; one kit survived and one did not, and the difference was *only* where the agent staged it. d3 wrote **directly into the handoff directory** (under `--demo-root`, on `/shared_nfs`) → **62 files survived and pass the shape check**. d4 staged on `$E2E_WORK_ROOT` (`/mnt/m2m_nobackup`, **node-local**) intending to copy at the end → node died first, **4 KB README is all that remains**, including a 19/19 verified deployment. | **open as a package-content decision.** Deploy recorded it rather than changing it, since it is not its call. Recommendation: *write into the handoff as you go; local scratch is for the container's logs, not the deliverable* |
| E9 | Qwen3.6-27B ~2.9 tok/s | **still open**, four checkpoints |

E14 and the narrowed E6 rule now interlock cleanly: `--demo-root` on
`/shared_nfs` (durable, and what survives a reap), `TMPDIR` on node-local disk
(only for zone-launched GPU processes).

### 6. Undetermined

1. **Why are freshly issued holds reaped?** Three for three. Unexplained, and
   nobody on the effort can resolve it. It now has a clear operational
   workaround, which is why it is no longer blocking.
2. **Should C9b be fixed?** The schema bug is fixed; the framework's discarding
   of a correct refusal is not. Two runs and ~90 minutes were spent finding by
   hand a reason the framework already had in a variable. **A leader-level call**
   — carried from T+120 with a sharper case.
3. **Should the unsealed deliverables be re-run now `acb8bfe` exists?** Analyze's
   content is byte-identical to what a sealed one would hold, and the fix is in.
   One clean run would convert "unsealed but verified" into "sealed with a run
   report" — but analyze has no node, and new holds die. **Weigh against the
   BRIEF's "mock samples, not measurements".**
4. E9 — untouched.

### 7. New commits

Seven since T+120:

```
acb8bfe fix(analyze-demo): declare the four packup items the seal was refusing
755e1a4 docs(llm_e2e): the stall was the symptom; seal_refused having no reader is the bug
aa502e0 docs(profiling-demo): the empty-content diagnosis, the login-node OOM, and the final run
d632b9c docs(deploy-demo): the move is clean; three cancelled holds and what survived them
a449191 docs(analyze-demo): the root cause was our own items_schema, not the clock
f4f920c fix(profiling-demo): the container name and the context length are parameters
4422962 docs(profiling-demo): the sealed-handoff delivery shape, and the last two hoists
```

- `acb8bfe` — the real fix for C9. One schema change, two lost runs behind it.
- `755e1a4` — the framework half, correctly separated from the package half.
- `a449191` — **a module publicly correcting its own earlier diagnosis** (232
  lines). This is the behaviour that makes the notes trustworthy.
- `d632b9c` — deploy's answer to step 1, plus the three-cancellation record.
- `f4f920c`, `4422962`, `aa502e0` — profiling's hardening.

Uncommitted: `deploy-demo/README.md`, `integration-demo/assets/accept/lm_eval.sh`,
`integration.debug.help.info.md`.

### 8. Other

- **4 of 5 delivered, and not one deliverable overclaims.** Profiling: don't
  quote a number. Kernel-opt: no kernel was optimised, and the validator passed
  without measuring. Deploy: not "both validators PASS". Analyze: `generating`,
  not `valid`. Every gap is stated by the module that produced it, in bold, at
  the top of its own PROVENANCE.
- **Two modules found the same class of bug from opposite ends**: analyze was
  bitten by `items_schema` drift; profiling then audited `check_items` across
  **all seven** of its kinds and found them clean. The second is only visible
  because the first was written down within the hour.
- The effort is now essentially **one module wide** — integration, on the safest
  node, with 5 h 14 m of walltime and one arm to re-run.

---

## T+180 — 2026-09-02 11:34 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling — **and now deploy's `deploy-d5`** | RUNNING (3 h 14 m) | 16:17:41 | **4 h 43 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (3 h 14 m) | 16:17:57 | **4 h 43 m** |

No new holds attempted since `101576` died. **The effort has consolidated onto
the two original holds** — deploy has moved `deploy-d5` onto `-079`, sharing
profiling's node, which is the correct response to "freshly issued holds do not
survive." Both holds are now past 3 h with no sign of the reaping that killed
three consecutive new ones.

### 1. Progress

**Effort: ~95 %.** Elapsed 186 minutes. Estimated remaining: **30–75 min**,
all of it integration's.

| module | est. % | basis |
|---|---|---|
| **profiling** | **100 % — DELIVERED** | sealed store, 6 verdicts PASS |
| **kernel-opt** | **100 % — DELIVERED** | 3 verdicts PASS, digests self-verified |
| **deploy** | **100 % — DELIVERED**; `deploy-d5` running to close the second validator | `check_deploy_kit` PASS |
| **analyze** | **100 % — DELIVERED**, now with a `store/` alongside the raw content | 5 verdicts PASS |
| **integration** | **~85 %** | `integration-r4` (`20260902T104817-8a2995`) launched 10:48, **5 handoff slots open**, still running at 46 min |

**Reliability: high.** Four deliverables read on disk. Integration's r4 has five
handoff directories and a live store — this is a real run in progress, not a
stall, which settles the T+120/T+150 uncertainty about that module. Its notes
last grew at 10:48, exactly when r4 launched, which is the expected pattern: a
module writes before and after a run, not during.

**Delivered: 4 of 5**, with the fifth in flight.

### 2. Current state, per module

- **integration.** `integration-r4` is the run of record, launched 10:48 right
  after the C20 fix (longer trace rather than a lowered bar). Four prior runs
  r1–r3 behind it. Five handoff slots open. This is the whole remaining critical
  path.
- **deploy.** Delivered, but not stopping there: `deploy-d5` is running on
  `-079` to obtain the `check_deploy_reproduces` verdict its delivered kit
  lacks. Also corrected a **stale control count** in its own notes (`8fe8c23`) —
  a module auditing its own earlier claim without being asked.
- **profiling.** Delivered; added `§22 Node-local paths do not exist on the login
  node` and a `per-node versus cluster-wide facts` split (`77a2c13`) — turning
  today's environment lessons into a reusable distinction rather than a list.
- **analyze, kernel-opt.** Delivered, no node, no further work possible.

### 3. The interval's main artefact: an `items_schema` audit of all five packages

Deploy ran this while `deploy-d5` was in its agent phase, prompted by analyze's
C9. **Result: no drift in any of the five.**

| package | evidence | finding |
|---|---|---|
| profiling | **all seven kinds sealed** in the delivered store | six `reproducible` carry exactly `[command, env, logs, result, watchout]`; `profile_packup` carries `[codes]` |
| kernel-opt | two sealed kinds | `[codes]` each; **neither declares an `items_schema`** |
| analyze | sealed `analyze_packup` | ten items, matching the widened schema |
| deploy | this module's kit | `[codes]`; `deploy_kit` **declares no `items_schema`** |
| integration | no content yet — producers read | ten kinds, **every one matching** |

Two things in it are worth more than the result:

**A method warning.** Grepping for `items/<name>` is **unsound** on these bodies
and produced two false positives before deploy caught them by reading:
`measure.sh` appeared to omit a required `logs`, `seed.py` a required
`watchout`. Neither is true — they build item paths through per-arm shell
variables (`A="$OUT_ACCEPT/items"`, `ITEMS="$OUT/items"`), so the literal string
never appears. **Check real content with `handoff.content.check_items`; fall back
to reading only when no content exists.** A static grep here would have produced
two confident, wrong bug reports.

**A structural conclusion**, which is the real lesson of analyze's bug:

> **A closed `items_schema` that merely restates its content type buys nothing
> and carries the whole risk.**

Four of five were never at risk *because their at-risk kinds declare no
`items_schema` at all* and fall back to the content type's rules, where producer
and type agree by construction. `deploy-demo/steps/deploy.yaml` already argues
this explicitly. `integration-demo`'s ten schemas are all of the risky shape:
correct today, **ten opportunities to drift tomorrow.**

### 4. Code problems

No new defects this interval. Standing:

| # | problem | state |
|---|---|---|
| C9 | analyze `items_schema` drift | **FIXED** `acb8bfe` |
| **C9b** | `seal_refused` has no reader — a correct refusal is discarded, surfacing as a stalled task | **open**, `755e1a4`. Unchanged and still worth a leader decision |
| **C23** | `integration-demo`'s ten closed `items_schema`s restate their content types — latent drift risk | **open, observation only** — nobody has proposed changing them, and the BRIEF's one-module-per-commit rule means it is integration's call |
| C20 | `min_requests` floor | **fixed** by lengthening the trace |
| C5, C8/C14, C10/C11, C13, C7 | carried | **open** |

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| E0″ | Three freshly issued holds reaped; both originals now past **3 h 14 m** untouched | **open, and now routed around** — deploy consolidated onto `-079` rather than requesting a fourth hold. That is the right response and it is working |
| E14 | The staging lesson (write into the handoff as you go; local scratch is not for the deliverable) | **open as a package-content decision** |
| **E15** | **Node-local paths do not exist on the login node** — profiling §22. Obvious once stated, and it invalidates any login-side check of a `/mnt/m2m_nobackup` artefact | noted, `77a2c13` |
| E9 | Qwen3.6-27B ~2.9 tok/s | **still open**, five checkpoints |

### 6. Undetermined

1. **Will `integration-r4` complete before it needs another arm?** 46 minutes in,
   five slots open, 4 h 43 m of walltime. Comfortable unless it loses an arm
   again — and a lost arm costs ~50 min with no resume.
2. **Will `deploy-d5` land the `check_deploy_reproduces` verdict?** It is an AI
   validator, so it needs `claude` reachable from the validation zone — the
   pre-recorded BRIEF limit 4. Nobody has reported passing `claude_cli` for d5.
3. **C9b** — carried, third checkpoint. Still a leader call.
4. **E9** — untouched, five checkpoints. It will end the effort undiagnosed
   unless someone picks it up, and that is a legitimate outcome to record rather
   than a gap to hide.
5. **Should `integration-demo`'s ten closed schemas be opened?** (C23.) Deploy
   found the risk; only integration can act on it, and it is mid-run.

### 7. New commits

Two since T+150:

```
8fe8c23 docs(deploy-demo): correct the stale control count, and audit every kind's items_schema
77a2c13 docs(profiling-demo): per-node versus cluster-wide facts
```

- `8fe8c23` — the five-package audit above, plus deploy correcting a stale count
  in its own earlier notes.
- `77a2c13` — profiling separating per-node facts from cluster-wide ones, so the
  next effort inherits a usable distinction rather than a flat list of traps.

Commit rate has fallen sharply (2 in 30 min, against 8 in the previous interval)
— consistent with four modules done and one mid-run, not with a stall.

Uncommitted: `integration-demo/assets/accept/lm_eval.sh` and
`integration.debug.help.info.md` — both integration's, both expected to land when
r4 finishes.

### 8. Other

- **The effort is one module wide and comfortably inside its walltime.** The
  binding risk is no longer time or hardware; it is whether integration's last
  arm passes its validators.
- **Deploy's audit is the best example today of a module doing work outside its
  own deliverable.** Analyze found a bug at 10:43; by 11:04 deploy had checked
  every other package for the same class, found none, and — more useful —
  identified *why* four were structurally immune and where the latent risk still
  sits. Neither module was asked to do this.
- Two of today's findings now have a general form worth carrying out of this
  effort: **"a closed `items_schema` that merely restates its content type buys
  nothing and carries the whole risk"**, and **"a static grep for item paths is
  unsound; check real content."**
- Still unaddressed across six checkpoints: the cluster facts (CPX per node, the
  narrowed NFS/`TMPDIR` rule, hold reaping, the numeric floors, the staging
  lesson) live in module notes and commits but **not in the BRIEF**, which is
  what the next effort reads first.

---

## T+210 — 2026-09-02 12:05 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy (`deploy-d5`) | RUNNING (3 h 45 m) | 16:17:41 | **4 h 12 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (3 h 45 m) | 16:17:57 | **4 h 12 m** |
| `101753` | `crsuse2-m2m-267` | **analyze** (new) | RUNNING (19 m) | ~16:03 | ~3 h 58 m |

A **fourth** hold was issued and — unlike the previous three — has survived 19
minutes so far. Analyze took it to re-run against the `acb8bfe` fix and get a
genuinely sealed terminal handoff. Too early to say the reaping has stopped; one
19-minute survival is not evidence against three deaths at 30/34/70 minutes.

### 1. Progress

**Effort: ~97 %.** Elapsed 217 minutes. Estimated remaining: **30–60 min.**

| module | est. % | basis |
|---|---|---|
| **deploy** | **100 % — DELIVERED, and now fully sealed with BOTH validators PASS** | see below |
| **profiling** | **100 % — DELIVERED** | 6 verdicts PASS |
| **kernel-opt** | **100 % — DELIVERED** | 3 verdicts PASS |
| **analyze** | **100 % delivered**, now **re-running for a sealed version** | `analyze-r5` at `verify_workset` |
| **integration** | **~90 %** | `integration-r4` alive, **8 handoff slots**, last write 11:52 |

**Reliability: high.** Every claim below was checked by opening a file.

**Delivered: 4 of 5**, one of them upgraded from "gap documented" to "complete".

### 2. Deploy closed its gap — verified in the sealed artefact

At T+120 and T+150 I recorded deploy as delivered with one validator short, on
its own insistence that it not be written up as "both validators PASS". `deploy-d5`
on `-079` has now closed that. From
`/shared_nfs/yihou/agent_sys/debugging/deploy/store/d54829ae-…/v1/validation.yaml`,
read directly:

```yaml
- validator: check_deploy_kit
  result: true      strength: strong   dimension: completeness
  at: '2026-09-02T11:44:05Z'
- validator: check_deploy_reproduces
  result: true      strength: weak     dimension: usability
  at: '2026-09-02T11:59:24Z'
```

**Both validators ran in the run, both returned true, and the handoff is
sealed** (`claim`, `content`, `manifest.yaml`, `validation.yaml` all present, v1).
`check_deploy_reproduces` is the AI validator — a fresh Claude Code session
following `REPRODUCE.md` and bringing the model up again — so T+180 undetermined
item 2 is answered: it did reach `claude` from the validation zone. The unsealed
d3 content is retained alongside as `unsealed-deploy-d3/`, which is the right
call: it is the kit that survived a node death and it documents a different
lesson.

**One discrepancy to flag rather than smooth over:** `PROVENANCE.md` in that
directory still has an **mtime of 10:24** and still carries the *"Read this
before quoting a verdict"* table saying `check_deploy_reproduces` did not run,
plus a section headed *"What the run did not get to"*. **The provenance file is
now stale with respect to its own deliverable, and understates it.** Almost
certainly deploy is mid-update — but as it stands on disk, a reader would
under-credit the artefact. Worth confirming it lands.

### 3. Current state, per module

- **integration** — the last module. `integration-r4` now shows **8 handoff
  slots** (up from 5 at T+180), with writes at 10:48, 10:54, 11:46 and 11:52.
  Alive and progressing. Notes have not grown since 10:48, which for this module
  has consistently meant "in a run", not "stopped".
- **analyze** — took hold `101753` and launched `analyze-r5`, currently at
  `verify_workset: input_validating -> running`. Also ran an `analyze-dryfix`
  dry run first (7 tasks resolved, 0 dispatched) to check the schema fix without
  spending GPU time. That is the cheap-first discipline the BRIEF asked for.
- **profiling, kernel-opt** — done, no further activity.

### 4. Code problems

No new defects. Standing set unchanged from T+180: **C9b** (`seal_refused` has
no reader) and **C23** (integration's ten closed `items_schema`s) are the two
open items anyone would act on; C5, C8/C14, C10/C11, C13, C7 carried.

Worth recording as **closed by evidence**: the `acb8bfe` schema fix is now being
exercised end-to-end by `analyze-r5`. If r5 seals its terminal handoff, C9 moves
from "fixed, verified offline" to "fixed, verified in a run".

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| E0″ | Hold reaping — three dead (30/34/70 min), two originals now at **3 h 45 m**, a fourth alive at 19 min | **open**; the workaround (consolidate onto surviving holds; treat a new hold as expendable) is holding |
| E14 | The staging lesson | **open as a package-content decision** |
| E9 | Qwen3.6-27B ~2.9 tok/s | **still open**, six checkpoints |
| E15 | Node-local paths absent on the login node | noted |

### 6. Undetermined

1. **Will `integration-r4` finish inside the walltime?** 4 h 12 m remain and it
   is 77 min in with 8 slots open. Comfortable. The risk is a failed output
   validation, which kills the arm with no resume (~50 min, measured).
2. **Will `analyze-r5` seal?** This is the test of `acb8bfe` in a live run.
3. **Deploy's stale PROVENANCE** — will it be refreshed before the effort ends?
   The artefact is better than its description right now.
4. **C9b** — fourth checkpoint carrying it. Leader call.
5. **E9** — sixth checkpoint untouched. I now expect this effort to end with it
   undiagnosed, and that should be stated as a finding rather than left implicit.

### 7. New commits

**None since T+180.** `git log` is unchanged at `77a2c13`. Uncommitted:
`integration-demo/assets/accept/lm_eval.sh` and
`integration.debug.help.info.md`.

Zero commits in 30 minutes is consistent with the observed state — four modules
finished, two runs in flight, nobody editing packages — and is not itself a
concern. It does mean deploy's d5 result and analyze's r5 are not yet reflected
in any commit or notes file.

### 8. Other

- **The deliverable set is now stronger than at any previous checkpoint**: three
  fully sealed with all validators PASS (profiling 6/6, kernel-opt 3/3, deploy
  2/2), one delivered-but-unsealed with the gap documented and a sealing run in
  flight (analyze), one in progress (integration).
- **Deploy went back for the gap rather than accepting it.** At T+120 it had a
  defensible deliverable and a written justification for the missing verdict.
  It spent another 90 minutes and two more runs to remove the caveat instead.
- The one thing I would want checked before this effort is called done: **the
  stale `PROVENANCE.md` in deploy's deliverable.** Every other provenance file
  today has erred toward understating its artefact deliberately; this one now
  understates it accidentally, which is a different thing and worth fixing.

---

## T+240 — 2026-09-02 12:35 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (4 h 15 m) | 16:17:41 | **3 h 42 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (4 h 15 m) | 16:17:57 | **3 h 42 m** |
| `101803` | `crsuse2-m2m-050` | **analyze** (fifth hold) | RUNNING (27 m) | ~16:33 | ~3 h 58 m |
| ~~`101753`~~ | ~~`-267`~~ | analyze | **gone** after ~30 min | — | — |

**Four of four freshly issued holds have now ended early** (`101078` ~30 min,
`101155` ~70, `101576` ~34, `101753` ~30), while the two originals are past
**4 h 15 m** untouched. The pattern from T+150 holds without exception. Analyze
is on its fifth hold and running `analyze-r6`.

### 1. Progress

**Effort: ~97 %.** Elapsed 247 minutes. Estimated remaining: **30–75 min.**

| module | est. % | basis |
|---|---|---|
| **deploy** | **100 % — DELIVERED, sealed, 2/2 PASS, provenance now current** | `e558f16` |
| **profiling** | **100 % — DELIVERED** | 6/6 PASS |
| **kernel-opt** | **100 % — DELIVERED** | 3/3 PASS |
| **analyze** | **100 % delivered**; two further attempts at a *sealed* version have both failed on a new cause | `analyze-r6` running |
| **integration** | **~90 %** | `integration-r4` alive — writing `measure.patched` artefacts at 12:01 |

Unchanged headline: **delivered 4 of 5**. No regression; the movement this
interval is quality, not count.

**Reliability: high.** Deploy's stale-provenance flag from T+210 is **resolved** —
`PROVENANCE.md` now has mtime 12:03 and deploy committed `e558f16`
("deploy-d5 is green — both validators PASS, handoff sealed and verified").

### 2. Analyze's sealing attempts hit a genuinely new failure

The `acb8bfe` schema fix **worked** — `analyze-r5` got past the seal problem
entirely. It then failed somewhere new, and the diagnosis is the most
operationally useful finding of this interval:

```
check_workset_shape:  PASS
workset_evidence slot v0: invalid
check_workset_runs:   FAIL
```

Both operators ran, both correct, `pass_ratio: 1.0`. The failure is **spread**:

```
moe1: per_group_ms [0.1344, 0.1437, 0.1315, 0.1424, 0.1272]  rsd 0.047  ok
moe2: per_group_ms [0.1772, 0.4358, 0.1876, 0.1842, 0.1965]  rsd 0.423  FAIL
```

`rocm-smi` on that node: **every card at 100 % use, VRAM 60–84 %, another
tenant**. So `max_rsd: 0.1` did exactly its job — the machine was not quiet.

Two properties of that rule are easy to get wrong, and analyze read the
validator rather than assuming:

- **It is a hard `return False`, not a per-operator note.** In
  `check_workset_runs/check.py`, `ran: false` and `correct: false` both
  `continue` into notes and are forgiven by `min_pass_ratio`; **an rsd breach
  returns immediately.** So `min_pass_ratio: 0.5` does *not* protect you from
  noise — one noisy operator fails the whole step no matter how many others were
  clean.
- **That inverts the `top_n` advice** analyze itself gave at T+150:

  | risk | `top_n=1` | `top_n=2` |
  |---|---|---|
  | agent writes one bad driver | fatal | survivable (`min_pass_ratio` 0.5) |
  | one operator hits node noise | one chance to be unlucky | **two** chances, either fatal |

  There is no universally right value: prefer 2 on a quiet node, and on a busy
  shared node neither is safe.

**This is a module correcting its own published advice within two hours**, for
the second time today (the first was C9's root cause). It is also the clearest
statement yet of a cost the BRIEF does not mention: **a shared, saturated node
can fail a validator on evidence that is entirely correct.**

### 3. Current state, per module

- **integration** — still the last module, and **alive**: files written at
  11:56 and 12:01 under `measure.patched/` (`accept/needle.json`, `steps.tsv`,
  `logs/probe.log`, `logs/needle.log`, `logs/smoke.log`). It is in the *patched*
  arm, i.e. past the stock arm. 8 handoff slots. Notes still last-written 10:48,
  now 107 minutes ago — for this module that has consistently meant "mid-run",
  and the artefacts confirm it.
- **analyze** — `analyze-r6` at `verify_workset`, fifth hold, third attempt at a
  sealed terminal handoff. Its delivered (unsealed) artefact remains valid and
  documented; everything since is upside.
- **deploy** — finished and tidy. Added `§21 Three rules this module earned,
  stated as rules` (`f1e495d`) — distilling its day into reusable form rather
  than leaving it as narrative.

### 4. Code problems

| # | problem | state |
|---|---|---|
| C9 | analyze `items_schema` drift | **FIXED and now proven in a live run** — r5 sealed past it |
| **C24** | `check_workset_runs` treats an rsd breach as a hard `return False` while forgiving `ran`/`correct` failures via `min_pass_ratio` — an inconsistency that makes `min_pass_ratio` misleading | **open, documented.** Arguably correct-as-designed; worth a decision, not a silent fix |
| C9b | `seal_refused` has no reader | **open** — fifth checkpoint |
| C23 | integration's ten closed `items_schema`s | **open** |
| C5, C8/C14, C10/C11, C13, C7, C20 | carried / fixed as previously recorded | — |

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| E0‴ | **Four of four new holds ended early; both originals past 4 h 15 m** | **open**, worked around |
| **E16** | **A saturated shared node fails `max_rsd` on correct evidence.** Analyze's r5 lost to another tenant's 100 %-utilised cards. Not a package fault and not fixable from inside the package | **open** — the real constraint on any timing-based validator here |
| E14 | Staging lesson | **open as a package decision** |
| E9 | Qwen3.6-27B ~2.9 tok/s | **still open**, seven checkpoints |
| E15 | Node-local paths absent on login node | noted |

### 6. Undetermined

1. **Will `integration-r4` complete?** It is in the patched arm at 107 min with
   3 h 42 m left. The measured cost of a lost arm is ~50 min with no resume.
2. **Will `analyze-r6` seal?** Third attempt, and the blocker is now node
   contention — outside analyze's control. It may simply not get a quiet node.
3. **Is C24 a bug or a design choice?** `min_pass_ratio` forgiving correctness
   failures but not noise is defensible, but it is surprising and undocumented.
4. **C9b** — fifth checkpoint.
5. **E9** — seventh checkpoint, untouched. **I now record it as a finding this
   effort will not resolve**, rather than as a pending item.

### 7. New commits

Two since T+210:

```
e558f16 docs(deploy-demo): deploy-d5 is green — both validators PASS, handoff sealed and verified
f1e495d docs(deploy-demo): three rules this module earned
```

- `e558f16` — records the green run and refreshes the provenance that T+210
  flagged as stale. The flag is cleared.
- `f1e495d` — deploy converting its findings into stated rules.

Uncommitted: `integration-demo/assets/accept/lm_eval.sh`,
`analyze.debug.help.info.md`, `integration.debug.help.info.md`.

### 8. Other

- **Deploy is the model finish**: closed its validator gap, refreshed its
  provenance, then wrote down the transferable rules. Three of its four notes
  sections in the last hour are for the *next* reader, not for its own delivery.
- **Two self-corrections in one day from two different modules** (analyze on C9's
  cause, analyze again on `top_n`), plus deploy correcting a stale control count
  and me correcting the E6 rule at T+90 and the C9 cause at T+150. The notes
  are trustworthy *because* of this, not despite it.
- The effort's remaining risk is entirely **integration-r4 completing** and,
  secondarily, whether analyze can find a quiet node. Neither is a code problem.

---

## T+270 — 2026-09-02 13:05 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (4 h 45 m) | 16:17:41 | **3 h 12 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (4 h 45 m) | 16:17:57 | **3 h 12 m** |

`101803` is gone — **five of five freshly issued holds ended early**; the two
originals are past 4 h 45 m. Analyze finished before losing it, so this cost
nothing. Only integration still needs compute.

### 1. Progress

**Effort: ~98 %.** Elapsed 277 minutes. Estimated remaining: **20–60 min.**

| module | est. % | basis |
|---|---|---|
| **analyze** | **100 % — DELIVERED SEALED. Six handoffs, six verdicts PASS** | verified below |
| **deploy** | **100 % — DELIVERED SEALED, 2/2 PASS** | |
| **profiling** | **100 % — DELIVERED SEALED, 6/6 PASS** | |
| **kernel-opt** | **100 % — DELIVERED SEALED, 3/3 PASS** | |
| **integration** | **~95 %** | r4 completed **both arms, 10 handoffs, 9 of 10 validators PASS**; r5 running with recalibrated bars; **r4 preserved as a fallback deliverable** |

**Delivered: 4 of 5 — and all four are now fully sealed with every validator
PASS.** Analyze's T+150 caveat ("unsealed, and it says so") is gone.

**Reliability: high.** I read all six of analyze's `validation.yaml` files.

### 2. Analyze went green — verified

`/shared_nfs/yihou/agent_sys/debugging/analyze/store/` holds **six sealed
handoffs from one run**, each with its verdict recorded `result: true`:

```
check_kernel_table        true      check_worklist_shape      true
check_identity_resolved   true      check_workset_shape       true
check_workset_runs        true      check_analyze_packup_shape true
```

Its PROVENANCE now opens *"Six sealed handoffs from one green run … each `valid`,
each with its validator's PASS recorded"* — and still leads its second paragraph
with the caveat that matters: *"the input profile is synthetic."* The earlier
unsealed delivery is kept as `store-run2-superseded/`.

That took **six runs** (r1, r2, r4, r5, r6 plus a dryfix), across **three
nodes**, through a schema bug it diagnosed and fixed itself and a node-contention
failure outside its control.

### 3. Integration: nine of ten, and the tenth is a mis-calibrated bar

`integration-r4` **completed both arms and produced all ten handoffs**. Nine
validators PASS. The tenth:

```
check_no_regression: FAIL   usability / strong
"output token throughput (avg): 59.16 -> 46.49, -21.4% against a bar of 5%"
"inter-token latency (avg):    427.23 -> 478.02, +11.9% against a bar of 10%"
```

**The validator and the report agreed**; both saw the arms differ, and that
stopped the graph before `packup`.

Integration's analysis is the strongest reasoning in the effort today. The patch
under test is the mock, whose entire per-call cost is **one boolean branch on a
module global** plus one `logger.warning` at import — next to a full decoder
layer of GPU work. That is not a 21 % effect. So **the 21 % is the arm-to-arm
spread of this deployment**, on a system decoding at 2–3 tok/s (E9) where a
60-request replay is queue-dominated.

And that is precisely the number the package says it lacks. From its own README,
under *Known gaps*:

> **The performance bars have no measured basis yet.** 5 % on throughput and
> 10 % on latency are placeholders; the number they want is the natural
> run-to-run spread of one arm, which the first full run will produce.

r4 **is** that first full run. So r5 sets `max_throughput_regression=0.35`,
`max_ttft_regression=0.30` — the measured spread plus margin — and integration
writes down exactly why, in `run.sh` and `PROVENANCE.md`:

> widening a bar after seeing the data is exactly the move that hides a real
> regression, and the only thing that separates the two is whether the reasoning
> is written down.

It then states what is **not** established, unprompted: one pair of arms is an
order-of-magnitude calibration, not a variance estimate; stock always runs first
so part of the gap may be systematic (thermal, page cache, cold allocator) and
this run cannot separate that from noise; the honest fix is `bench_rounds > 1`
or two stock arms back to back, neither of which fits the walltime; and **the
widened bars must not travel back to the GLM deployment** — the defaults stay 5 %
and 10 %.

**And r4 is preserved as a fallback**: nine handoffs including a complete
`integration_report`, at `ws2/integration/r4_fallback/handoffs/`. If r5 does not
finish, that is the deliverable with §14 as its explanation. Integration cannot
now finish empty-handed.

### 4. Code problems

| # | problem | state |
|---|---|---|
| **C25** | **The 5 %/10 % performance bars are placeholders with no measured basis** — the package's README says so, and r4 supplied the missing number (~21 % throughput, ~12 % latency spread on this deployment) | **open by design; now calibrated for this host only.** The defaults are deliberately unchanged |
| **C26** | `handoff`'s **locality check is not called** — the seal does not enforce locality; `redact.py` does. Settled first-hand by analyze and profiling (`23f3d6a`, `561813a`), and documented as **deliberate** | **documented.** Directly contradicts a natural reading of BRIEF limit 3 |
| C27 | analyze's locality helper was merging two different findings | **fixed**, `1a4e8f5` |
| C24 | `check_workset_runs` hard-fails on rsd while forgiving correctness | **open** |
| C9b | `seal_refused` has no reader | **open** — sixth checkpoint |
| C23 | integration's ten closed `items_schema`s | **open** |

**C26 is a correction to the shared BRIEF.** Known-limit 3 says handoff content
"may not name an absolute path… (`handoff/locality.py`)", which reads as a seal-
time enforcement. Two modules established first-hand that **the seal does not
call it**; `redact.py` is what actually acts, and the disconnection is
intentional. Anyone designing around "the seal will catch my paths" is wrong.

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| E0⁗ | **Five of five new holds ended early; both originals at 4 h 45 m** | **open**, fully worked around |
| E16 | A saturated shared node fails `max_rsd` on correct evidence | **open** |
| **E9** | **Qwen3.6-27B at ~2.9 tok/s** — now *load-bearing*: it is why the replay is queue-dominated and why the arm-to-arm spread is 21 % | **open, and it has propagated.** No longer a curiosity |
| E14 | Staging lesson | **open as a package decision** |

### 6. Undetermined

1. **Will `integration-r5` finish inside 3 h 12 m?** r4 took ~2 h for both arms.
   Tight but feasible; the fallback removes the downside.
2. **Is integration's 21 % noise or systematic?** Explicitly unresolved by the
   module, with the experiment that would settle it named and declined for
   walltime. **The right way to leave a question open.**
3. **C9b** — sixth checkpoint. Still unactioned.
4. **E9** — eighth checkpoint. It has now caused a validator failure and a bar
   recalibration. **It should be the first item of any follow-up.**
5. Should C25's widened bars be re-tested against GLM before reuse? Integration
   says no and gives its reason; nobody has disagreed.

### 7. New commits

Five since T+240:

```
d8a22fd docs(analyze-demo): the green run, and the max_rsd finding
1a4e8f5 fix(analyze-demo): the locality helper stops merging two different findings
23f3d6a docs(llm_e2e): the locality check is not called, and that is deliberate
81f2c21 docs(analyze-demo): the delivery is a store root and nothing else
561813a docs(profiling-demo): the seal does not enforce locality — redact.py does
```

`23f3d6a` and `561813a` are two modules landing the **same** framework finding
from different directions within minutes — the C26 correction above. `81f2c21`
standardises the delivery shape on a store root, which all four delivered
modules now use.

Uncommitted: `integration-demo/assets/accept/lm_eval.sh`,
`integration.debug.help.info.md`.

### 8. Other

- **Four modules delivered, every handoff sealed, every validator PASS.** Counts:
  profiling 7 handoffs/6 verdicts, analyze 6/6, kernel-opt 2/3, deploy 1/2.
- **Nobody widened a bar quietly.** Integration hit the only threshold failure of
  the day, established the threshold was a documented placeholder, produced the
  missing measurement, recalibrated with the reasoning written into two files,
  enumerated what its own measurement does not establish, and ring-fenced the new
  values from the deployment they were not measured on. That is the single best
  piece of work I have observed today.
- **The BRIEF now has three confirmed errors** found by this effort: the 1800 s
  settle budget (actually 14400 s), "8 × MI355X 288 GiB" (some nodes are CPX,
  64 × 36 GiB), and the locality check being enforced at seal (it is not called).
  All three are recorded in commits; **none is in the BRIEF**, which is what the
  next effort reads first.

---

## T+300 — 2026-09-02 13:36 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (5 h 16 m) | 16:17:41 | **2 h 41 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (5 h 16 m) | 16:17:57 | **2 h 41 m** |

### 1. Progress

**Effort: ~99 %.** Elapsed 308 minutes. Estimated remaining: **20–60 min**, all
contingent on `integration-r5`.

| module | est. % | basis |
|---|---|---|
| profiling / kernel-opt / deploy / analyze | **100 % — DELIVERED SEALED, all verdicts PASS** | unchanged |
| **integration** | **~95 % — deliverable now populated, but not by integration** | see below |

**All five deliverable directories are now non-empty**, which has not been true
before. But the fifth was assembled by other modules on the leader's
authorisation, and that distinction is the substance of this checkpoint.

**Reliability: high.** I read the leader's note and all ten `validation.yaml`
files in integration's store.

### 2. Integration's directory was populated out of band — and labelled as such

`/shared_nfs/yihou/agent_sys/debugging/integration/` now holds:

| path | what | produced by |
|---|---|---|
| `store/` | **nine sealed handoffs** from run `20260902T104817-8a2995` | integration's run; **copied here by `kernel-opt`** on the leader's authorisation |
| `store/COPIED-BY-KERNEL-OPT.md` | that copy's record + per-handoff verdicts | kernel-opt |
| `packup-out-of-band/` | `integration_packup` content, 47 files, **not sealed** | **produced by `deploy`**, using integration's own unmodified `packup.py` over the nine sealed handoffs |
| `packup-out-of-band/PRODUCED-BY-DEPLOY.md` | that production's record, with an unconditional offer to remove it | deploy |
| `DELIVERY-NOTE-FROM-LEADER.md` | why the directory is readable without integration's own provenance | the leader |

The leader's note opens: *"This is not `integration`'s own record. Their
`PROVENANCE.md` does not exist at the time of writing and only they can write
it."* Three separate hands touched this directory and **each labelled its own
contribution in a file named after itself.** Nothing here is passed off as
integration's work.

**Verified independently** — the ten verdicts across the store:

```
check_service_live   true    check_overlay_applies true    check_patch_shape true
check_bench_report   true    check_bench_report    true    check_patch_live  true
check_acceptance     true    check_acceptance      true    check_service_live true
check_no_regression  FALSE
```

**Nine true, one false.** The leader's note flags this in its own section headed
*"The one thing a reader must not miss"*:

> It is **a sound sample of a refused report and a misleading one if taken for
> a passing example.**

`check_no_regression` is `strength: strong`, so the false verdict invalidated
the `integration_report` handoff, the graph stopped, and **`integration_packup`
was never dispatched** — which is exactly why the packup here was produced out
of band. The note is careful about the causality: *"The packup step itself is
sound; it is downstream of a validator that correctly refused its input."*

The refusal is the C25 bar mis-calibration from T+270, unchanged: a declared
no-op patch measured 21 % apart across arms because the deployment decodes at
2–3 tok/s and the bars are documented placeholders.

**Assessment.** This is a defensible way to populate a directory under time
pressure and it is executed with unusual care — real artefacts from a real run,
the one refused verdict promoted rather than buried, every hand named, and a
standing offer to withdraw the out-of-band piece. What it is **not** is a
handoff of the terminal kind produced by integration's own run, which is what
the BRIEF asks for. `integration-r5` is still the thing that would satisfy that,
and it is still running.

### 3. Current state

- **integration** — `integration-r5` (`20260902T125156-4b968b`), 5 handoff slots,
  **last write 13:06, now 30 minutes ago.** Its notes have not grown since 12:52.
  Given r4 took ~2 h for both arms, a 30-minute quiet stretch mid-arm is
  unremarkable; but with 2 h 41 m left this is the number to watch.
- **profiling, kernel-opt, deploy, analyze** — done, and now spending their time
  on delivery hygiene and on *each other's* deliverables.

### 4. Code problems

No new package defects. The interval's commits are all about **how a delivered
handoff is verified**, which has become a small shared sub-project:

| # | finding | state |
|---|---|---|
| C28 | `relayout_handoffs.py` needed a **verdict gate and distinct exit codes** — an exit-status disagreement between two modules, resolved with a distinct code rather than a suppression flag, then `--allow-refused` restored with the round-trip cost named | **resolved**, `a9a6ab5` → `3e13a3e` → `bfe56e1`, `fdeb5d6` |
| C29 | **Verifying a delivered handoff takes three checks, not two** (`63ad06b`) | **documented** |
| C25 | placeholder performance bars | **open by design**, calibrated for this host only |
| C26 | locality check not called at seal | **documented** |
| C9b, C23, C24 | carried | **open** |

C28 is worth noting as process: two modules disagreed about an exit status, and
the resolution went **suppression flag → distinct exit code → flag restored with
its cost documented**, in three commits over ~20 minutes. The disagreement was
settled in the code and the reasoning left behind.

### 5. Non-code problems

Unchanged: E0 (five of five new holds died; both originals now 5 h 16 m), E9
(2.9 tok/s, **ninth checkpoint**, now the acknowledged cause of the only
validator failure in the effort), E14, E15, E16.

### 6. Undetermined

1. **Will `integration-r5` finish?** 2 h 41 m left, 30 minutes quiet. This is the
   last open question of the effort proper.
2. **Does the out-of-band directory satisfy the delivery contract?** The BRIEF
   asks for a handoff of the terminal kind that the package produced. Nine
   sealed handoffs qualify; the terminal `integration_packup` was produced out
   of band by another module. **A leader call, and the leader has already made
   it once by authorising the work** — but it should be stated explicitly in the
   final accounting rather than left to a reader of `PROVENANCE.md` files.
3. **Will integration write its own `PROVENANCE.md`?** The leader's note says
   only they can, and it is the one document the directory lacks.
4. **C9b** — seventh checkpoint, still unactioned.
5. **E9** — ninth checkpoint. Undiagnosed, and now demonstrably consequential.

### 7. New commits

Five since T+270:

```
63ad06b docs(profiling-demo): verifying a delivered handoff takes three checks, not two
a9a6ab5 docs(kernel-opt-demo): resolve the exit-status disagreement with a distinct code
fdeb5d6 docs(profiling-demo): record relayout_handoffs.py's verdict gate and exit codes
3e13a3e docs(kernel-opt-demo): drop the suppression flag, keep the distinct exit code
bfe56e1 docs(kernel-opt-demo): restore --allow-refused, and name what the round trip cost
```

All five are `docs` on the verification/delivery tooling — no package behaviour
changed. `bfe56e1` explicitly records the cost of the round trip it ends, which
is the honest way to close a reversed decision.

Uncommitted, unchanged for two hours: `integration-demo/assets/accept/lm_eval.sh`
and `integration.debug.help.info.md`.

### 8. Other

- **Four modules that had finished spent this interval on someone else's
  problem** — kernel-opt copying integration's store, deploy running
  integration's packup, profiling and kernel-opt jointly hardening the shared
  relayout tool. None of that was assigned.
- **The one refused verdict was promoted, not buried.** Three separate documents
  (integration's §14, the leader's note, kernel-opt's copy record) each state
  that `check_no_regression` returned false and why. The easiest thing to do
  with a 9-of-10 was to report "nine PASS"; nobody did.
- Carried from every checkpoint since T+120 and still true: **the three BRIEF
  errors** (settle budget 14400 s not 1800 s; CPX nodes are 64 × 36 GiB not
  8 × 288 GiB; the locality check is not called at seal) are in commits and
  module notes but **not in the BRIEF**.

---

## T+330 — 2026-09-02 14:05 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (5 h 46 m) | 16:17:41 | **2 h 11 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (5 h 46 m) | 16:17:57 | **2 h 11 m** |

Both originals still untouched. Under two hours remain.

### 1. Progress

**Effort: ~99 %.** Elapsed 338 minutes. Estimated remaining: **20–60 min.**

| module | est. % | basis |
|---|---|---|
| profiling / kernel-opt / deploy / analyze | **100 % — DELIVERED SEALED** | unchanged |
| **integration** | **~95 %** | `integration-r5` alive: **8 handoff slots** (up from 5 at T+300), last write **13:52** |

Unchanged in count. r5 has grown from 5 slots to 8 in half an hour, so it is
moving and has covered most of a graph that produced ten last time.

**Reliability: high**, with one caveat I want on the record: **integration has
not written a note since 12:52 (73 min) and has not committed since before
T+270.** Everything I know about that module this interval, I learned from
*other modules' notes about it*. That is a real gap in the record, and it is the
second time today I am reporting on a module through third parties.

### 2. A fact I reported wrongly by omission: integration's r4 was OOM-killed

At T+300 I described integration's directory as populated out of band and left
the reason as "under time pressure". Kernel-opt's §23 supplies what I did not
have: integration's **run was OOM-killed at 12:44 and the module was inactive**;
that is why the leader authorised two other modules to recover its artefacts.
This is E10 again — the login-node OOM that killed profiling's run B at T+90 —
now claiming a second victim, five hours later, in a module that had no reason
to expect it.

I am recording this here rather than editing T+300. **The out-of-band recovery
was a response to a crash, not to slowness**, which is a materially different
thing and reflects better on the module than what I wrote.

### 3. The interval's finding: "9/9 verified" was true and misleading

Kernel-opt did the copy of integration's nine handoffs and reported **9/9
verified** — every copy re-hashed against its own manifest *and* against the
untouched run-store original. Then it wrote §23, headed *"and the check I did
not do"*:

> I verified **integrity** and **shape** and reported "9/9 verified" — and never
> opened a single `validation.yaml`.

Deploy did open them, and found `integration_report` carrying
`check_no_regression: result=False, strength: strong`. Kernel-opt then read all
nine itself: **8 pass, 1 fails**, and the failure is on the terminal report of
the measurement chain — which is *why* `integration_packup` never ran. So:

> **A digest proves the bytes have not changed since sealing; it says nothing
> about whether what was sealed was acceptable.** Those are two independent
> questions and I answered one while sounding like I had answered both.

It also corrects the leader's expectation by measurement: there is **no tenth
directory** for the unsealed packup — the run's `handoffs/` holds exactly nine —
so the missing tenth is a **consequence of the refused verdict, not of the OOM
kill**.

And then deploy corrected *kernel-opt's own framing* in turn: the digest check
was not secondary, because deploy's diagnosis of the false verdict **depended**
on it — the bytes being provably the sealed bytes is what let them treat the
verdict as a property of the run rather than of the copy. **A check that rules
out an entire class of confusion is doing real work even when it is silent on
the question you care about.**

This is the cleanest instance today of the BRIEF's first rule. It produced the
delivery checklist that is the reusable output: **verify the digest, verify the
layout, and read the verdicts. Three checks, not two.**

### 4. Code problems

**No new defects, and no commits at all this interval** — `git log` is unchanged
at `bfe56e1`. Standing open items, all carried:

| # | problem | state |
|---|---|---|
| C9b | `seal_refused` has no reader | **open** — eighth checkpoint |
| C23 | integration's ten closed `items_schema`s | **open** |
| C24 | `check_workset_runs` hard-fails on rsd, forgives correctness | **open** |
| C25 | placeholder performance bars, now calibrated for this host only | **open by design** |
| C26 | locality not enforced at seal | **documented** |

### 5. Non-code problems

| # | problem | state |
|---|---|---|
| **E10′** | **The login-node OOM has now killed two runs** — profiling's run B (~09:34) and integration's r4 (12:44). Profiling diagnosed it at T+90 and fixed it *for itself* by moving `agent-sys` onto the compute node (`8274a08`); the fix did not propagate | **open, and the propagation failure is the lesson.** A fix written into one module's package at 11:00 did not reach another module at 12:44 |
| E0 | Five of five new holds died; both originals at 5 h 46 m | **open**, worked around |
| E9 | Qwen3.6-27B ~2.9 tok/s | **open**, tenth checkpoint |
| E14, E15, E16 | carried | **open** |

E10′ is worth stating plainly: **the single highest-value thing any module
learned today was known for three and a half hours before it claimed its second
victim.** The notes files are per-module by design, and there is no channel that
makes "this will kill your run too" arrive at another module unasked.

### 6. Undetermined

1. **Will `integration-r5` finish inside 2 h 11 m?** 8 of an expected ~10 slots,
   last write 13 minutes ago. The most likely outcome is yes; the fallback
   (r4's nine, already delivered) means the downside is bounded.
2. **Is integration still alive as a module?** 73 minutes without a note, after
   an OOM kill that another module reported on its behalf. Its run is
   progressing, which is the thing that matters, but I cannot tell whether
   anyone is watching it.
3. **Will integration's own `PROVENANCE.md` be written?** Still the one document
   its directory lacks.
4. **C9b** — eighth checkpoint. If it is not actioned it should be closed as
   "recorded, not fixed" rather than left implying someone will get to it.
5. **E9** — tenth checkpoint, undiagnosed.

### 7. New commits

**None.** `git log` unchanged since T+300 at `bfe56e1`. Uncommitted and
unchanged for two and a half hours: `integration-demo/assets/accept/lm_eval.sh`,
`integration.debug.help.info.md`.

The commit stream stopping is consistent with four modules finished and the
fifth mid-run. It does mean the last hour of work — kernel-opt's §23, the
three-check checklist, deploy's correction — exists in notes files that are
themselves uncommitted for two of the five modules.

### 8. Other

- **Four modules delivered and sealed; the fifth has nine sealed handoffs
  delivered on its behalf plus a live run that may supersede them.** No module
  will finish empty-handed.
- **Three modules corrected themselves or each other this interval alone** —
  kernel-opt on its own "9/9 verified", deploy on kernel-opt's framing, and
  kernel-opt on the leader's expectation of a tenth directory. Every correction
  went into a file rather than into a conversation.
- **My own two corrections today**, both recorded in the section following the
  error rather than by editing it: the NFS/`TMPDIR` rule (T+90, too broad) and
  C9's root cause (T+150, wrong cause). Add to those the omission corrected in
  §2 above. The append-only rule has cost nothing and made all three visible.
- Carried unchanged since T+120: **three confirmed BRIEF errors** (settle budget,
  CPX geometry, locality-at-seal) live in commits and notes but not in the BRIEF.
  With two hours left, this is the cheapest high-value thing anyone could still do.

---

## T+360 — 2026-09-02 14:35 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (6 h 16 m) | 16:17:41 | **1 h 42 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (6 h 16 m) | 16:17:57 | **1 h 42 m** |

Under two hours. Both originals have now run 6 h 16 m without interruption,
against five freshly issued holds that all died inside 70 minutes.

### 1. Progress

**Effort: ~99 %.** Elapsed 368 minutes. Estimated remaining: **20–50 min**, or
the effort ends at the walltime with what it already has.

| module | est. % | basis |
|---|---|---|
| profiling / kernel-opt / deploy / analyze | **100 % — DELIVERED SEALED** | unchanged for 90 minutes |
| **integration** | **~97 %** | `integration-r5`: 8 handoff slots, **7 verdicts, all `result: true`**, writing **as of 14:33** |

**Reliability: high.** I opened r5's `validation.yaml` files directly.

### 2. `integration-r5` is close, and so far it is clean

Seven verdicts recorded, **every one true**:

```
check_overlay_applies  true    check_service_live   true    check_patch_live   true
check_service_live     true    check_acceptance     true    check_patch_shape  true
check_bench_report     true
```

Compare with r4, which produced ten handoffs and failed exactly one —
`check_no_regression` on the terminal `integration_report`. **r5 has not yet
reached that validator.** The two outstanding pieces are precisely the two that
r4 could not deliver: the `integration_report` carrying `check_no_regression`,
and the `integration_packup` downstream of it that never dispatched.

So the whole question of whether integration delivers its own terminal handoff
comes down to the one validator that failed last time, with the recalibrated
bars (`max_throughput_regression=0.35`, `max_ttft_regression=0.30`) that
integration derived from r4's own measured spread and documented in two files.
The run was still writing two minutes before this checkpoint.

**I am not going to predict it.** The bars were set from a single pair of arms,
which integration itself called an order-of-magnitude calibration rather than a
variance estimate, and it noted the stock-arm-first ordering may make part of
the 21 % systematic. If the spread this time exceeds 35 %, it fails again.

### 3. Current state

- **integration** — r5 active, 7/7 clean, two handoffs short. Notes still last
  written 12:52 (**103 minutes**); no commit since before T+270. The run is
  healthy and the record is not being kept.
- **profiling, kernel-opt, deploy, analyze** — all quiet for 60+ minutes. Their
  notes last grew at 13:23, 13:27, 12:05 and 12:48. All four are finished and
  have stopped, which is the correct behaviour, not a stall.

### 4. Code problems

No new defects. No commits. The five open items are unchanged from T+330 —
**C9b** (`seal_refused` has no reader), **C23** (integration's ten closed
`items_schema`s), **C24** (`check_workset_runs` hard-fails on rsd),
**C25** (placeholder bars, now host-calibrated), **C26** (locality not enforced
at seal, documented).

With 1 h 42 m left and every module either finished or in a terminal run, **none
of these will be fixed in this effort.** They should be handed over as recorded
findings rather than left looking pending.

### 5. Non-code problems

Unchanged and all open: **E0** (5/5 new holds died, both originals at 6 h 16 m),
**E9** (2.9 tok/s — eleventh checkpoint, undiagnosed, and the acknowledged cause
of the only validator failure of the day), **E10′** (login-node OOM killed two
runs three and a half hours apart, the fix never propagating between modules),
**E14** (staging), **E15**, **E16** (a saturated node fails `max_rsd` on correct
evidence).

### 6. Undetermined

1. **Will `check_no_regression` pass in r5?** The single open question of the
   effort. Answered within the hour, one way or the other.
2. **Will integration write its own notes and `PROVENANCE.md`?** 103 minutes
   silent. If the walltime arrives first, the record of the fifth module will
   consist of its own notes up to 12:52 plus three other parties' accounts of
   what happened after — which is a worse outcome than the artefacts deserve.
3. **C9b** — ninth checkpoint. **Recommend closing it as "recorded, not fixed."**
4. **E9** — eleventh checkpoint. Same: it should be handed over as an open
   question with the evidence attached, not carried as if someone will get to it.
5. Whether the out-of-band packup stays or is withdrawn if r5 produces a real
   one. Deploy offered unconditionally to remove it; nobody has needed to decide.

### 7. New commits

**None**, for the second consecutive checkpoint. `git log` unchanged at
`bfe56e1` since ~13:30. Uncommitted for three hours:
`integration-demo/assets/accept/lm_eval.sh`, `integration.debug.help.info.md`.

An hour of no commits with four modules finished is expected. The thing worth
flagging is narrower: **kernel-opt's §23 and deploy's correction of it — the
best material of the last two hours — are in notes files, and two of the five
notes files are uncommitted.** If the session ends abruptly they are still on
disk in the worktree, so this is a tidiness risk rather than a loss risk.

### 8. Other

- **The effort is done except for one validator.** Four modules delivered sealed
  with every verdict PASS; the fifth has nine sealed handoffs already delivered
  on its behalf, a documented account of the one refused verdict among them, and
  a live run that may supersede all of it within the hour.
- **Nothing has regressed at any checkpoint today.** Every interval has been
  flat or forward: no deliverable withdrawn, no verdict revoked, no finding
  retracted — only narrowed, corrected in place, or superseded by a better run.
- The three BRIEF errors (settle budget 14400 s not 1800 s; CPX nodes 64 × 36 GiB
  not 8 × 288 GiB; locality not enforced at seal) remain in commits and notes and
  **not in the BRIEF**. This is the ninth checkpoint carrying that line. It is a
  ten-minute edit and it is the highest-leverage thing left undone.

---

## T+390 — 2026-09-02 15:05 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (6 h 46 m) | 16:17:41 | **1 h 12 m** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (6 h 46 m) | 16:17:57 | **1 h 12 m** |

### 1. Progress

**Effort: 100 % of what it can deliver.** Elapsed 398 minutes. **All five
modules have delivered.** Remaining work is optional hardening.

| module | state |
|---|---|
| profiling | DELIVERED SEALED, 6/6 PASS |
| kernel-opt | DELIVERED SEALED, 3/3 PASS |
| deploy | DELIVERED SEALED, 2/2 PASS |
| analyze | DELIVERED SEALED, 6/6 PASS |
| **integration** | **DELIVERED** — r5's nine sealed handoffs (**9 verdicts true, 1 false**), terminal packup validated out of band, own `PROVENANCE.md` written |

**Reliability: high.** All verdicts read from `validation.yaml` on disk.

**Integration returned and closed out fully**: two commits, 189 new lines of
notes, its own `PROVENANCE.md`, and — the substantial part — **it diagnosed the
anomaly that has been open since T+60.**

### 2. `integration-r5` failed the same validator, and the failure explains everything

r5 ran with the widened bars (35 %/30 %) and `check_no_regression` refused
anyway, with numbers of a **completely different order** from r4's:

```
output token throughput  r1:  193.59 ->  46.70    -75.9%  against a 35% bar
time to first token      r1:  178.14 -> 2060.77 +1056.8%  against a 30% bar
inter-token latency      r1:   40.95 ->  475.02 +1060.0%  against a 30% bar
```

**The stock arm was ten times faster than the patched arm**, visible in raw step
timings at byte-identical settings — `lm_eval` 23 s vs 428 s, `bench_r1` 44 s vs
161 s. Not the patch (one boolean branch on a module global). CUDA-graph config
**identical in both arms' logs**, so the obvious explanation was ruled out by
reading, not assumed.

### 3. E9 is diagnosed — the deployment is bistable, per bring-up

Integration had ~12 minutes and the engine container from the finished run was
still up, so it ran the experiment it had named. **Four identical replays against
one still-running instance:**

```
v1 out_tps=46.24 itl=480.45   v2 47.38 / 473.72
v3 47.40 / 471.31             v4 53.05 / 477.15
```

| metric | within-instance spread (n=4) |
|---|---|
| inter-token latency | **2 %** |
| output token throughput | 15 % |
| time to first token | 45 % |

The instance **never left its slow state**. So:

> **The flip is decided at bring-up and persists for the life of the container.**
> It is not noise within a measurement window; it is which of two states a fresh
> deployment instantiates into.

**This is E9** — the ~2.9 tok/s figure I have carried open for eleven
checkpoints. It was never a property of the model or the hardware; it is one of
two states a container draws at start-up, and the fast state (ITL ~41 ms against
~475 ms) is real and was observed in r5's stock arm.

**Three consequences, and integration got each right:**

1. **A per-arm bar cannot fix it, and integration retracts its own T+270
   recalibration**: within an instance ITL is stable to 2 %, so the package's
   original 10 % bar *"is not too tight — it is well chosen for the thing it can
   see. My widening to 30 % in r5 was calibrated against a cross-instance
   artefact and was, in hindsight, the wrong response to the wrong number. The
   defaults should stay 5 % and 10 %; the r5 bars should not be copied
   anywhere."*
2. **The fix belongs in the design.** The package's README says both arms run
   back to back in the same session so results are comparable. **That is not
   sufficient**: each arm gets a freshly created container and therefore an
   independent draw of the state. The design controls for session, node, trace,
   order and image — *and not for the one thing that dominates*. What is needed
   is a same-state gate; `check_service_live` proves a deployment is *live*, not
   that it is *comparable to the other arm's*.
3. **What the two states are is still unexplained.** Candidates named and
   untested: aiter kernel selection at first call, the GatedDeltaNet/linear-
   attention path, allocator/NUMA placement of a fresh container. *"Whoever picks
   this up starts here, and now knows to compare two bring-ups rather than two
   measurements."*

**And the conclusion for the stage**: `check_no_regression` and `compare` are
**not broken** — they recomputed from raw numbers, agreed, and refused, *"which
is exactly right, because a validator that certified a 10× difference as 'no
regression' would be the broken one."* The refusal reports that **this
deployment is not stable enough to support a two-arm comparison at any bar**.
The pipeline is sound; the measurement environment is not; those are different
findings, and the nine PASSing validators are the evidence for the first.

### 4. How integration's terminal handoff was obtained

`packup` never dispatched in either run. The terminal artefact was produced out
of band **using the package's own unmodified code** — `packup.py` driven
directly with `AGENT_SYS_INPUT_<KIND>` per input, then validated by the
package's own `check_packup_shape.validator` in a hand-built zone with
`args.json` copied verbatim from `steps/verdict.yaml:119-128`:

```
packup: 45 file(s), verdict REJECTED
check_packup_shape: oob-integration-packup PASS
```

With the caveat stated exactly right:

> It asserts that the content directory the terminal step would have produced
> passes the validator that step's handoff would have faced. It does **not**
> assert a seal. Say it that way round; *"validated out of band"* is easy to read
> as *"sealed"*.

Its delivery separates its own work from the leader-authorised r4 recovery,
leaving the latter untouched and explaining that r5 supersedes r4 *"as the better
sample, but reaches the same terminal verdict for the more informative reason."*

### 5. Code problems — final state

| # | problem | state |
|---|---|---|
| **C25** | performance bars | **resolved as a design finding**: defaults 5 %/10 % are correct; the r5 widening is retracted by its author; the real gap is a missing same-state gate |
| C9b | `seal_refused` has no reader | **open — recorded, not fixed** |
| C23 | integration's ten closed `items_schema`s | **open — recorded, not fixed** |
| C24 | `check_workset_runs` hard-fails on rsd | **open — recorded, not fixed** |
| C26 | locality not enforced at seal | **documented, deliberate** |

C9, C20, C28 and the transport/parameter fixes were all closed earlier.

### 6. Non-code problems — final state

| # | problem | state |
|---|---|---|
| **E9** | Qwen3.6-27B ~2.9 tok/s | **DIAGNOSED** — bistable per bring-up, fast state ~41 ms ITL, slow ~475 ms. Mechanism still unknown; next steps named |
| E0 | 5/5 new holds died; both originals at 6 h 46 m | **open, unexplained**, fully worked around |
| E10′ | login-node OOM killed two runs | **fixed for profiling; propagation failed** |
| E14 | staging: write into the handoff as you go | **open as a package decision** |
| E16 | saturated node fails `max_rsd` on correct evidence | **open** |

### 7. Undetermined

1. **What are the two deployment states?** The one genuinely open technical
   question, now sharply posed with three candidates and a stated method.
2. **Should the design gain a same-state gate?** Integration's recommendation;
   nobody has ruled on it.
3. **E0** — five holds, unexplained.
4. C9b / C23 / C24 — **recommend handing these over as recorded findings.** No
   time remains to act on them and they should not read as pending work.

### 8. New commits

```
50a1532 docs(integration-demo): record the spur localisation traps and the bistable deployment
b8a553a docs(integration-demo): the 10x swing is per-deployment, not per-measurement
```

**The working tree is now clean** apart from `.serena/` and this file — every
module's notes and fixes are committed.

### 9. Other

- **All five modules delivered.** Four sealed with every validator PASS;
  integration with nine sealed (one carrying a correctly-refused verdict) plus an
  out-of-band terminal artefact validated by the package's own validator.
- **The effort's best work was its last hour.** Integration came back from an OOM
  kill, ran a 12-minute experiment against a container that was still up, and
  turned "the only validator failure of the day" into a design finding — while
  **retracting its own earlier fix** as the wrong response to the wrong number.
- **E9 closed after eleven checkpoints.** It was the right thing to keep carrying
  as undetermined rather than guessing at, and it was solved by someone noticing
  a still-running container was the cheapest thing in the session to ask.
- Still not done, tenth checkpoint: **the three BRIEF errors** (settle budget
  14400 s not 1800 s; CPX nodes 64 × 36 GiB not 8 × 288 GiB; locality not
  enforced at seal) — plus, now, a fourth worth adding: **on this stack a fresh
  container draws one of two performance states, so back-to-back arms are not
  automatically comparable.**

---

## T+420 — 2026-09-02 15:35 UTC

### Walltime countdown

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (7 h 16 m) | 16:17:41 | **42 min** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (7 h 16 m) | 16:17:57 | **42 min** |

Under an hour. Both originals will reach walltime intact — the only two of seven
holds today to do so.

### 1. Progress

**Effort: 100 % delivered.** Elapsed 428 minutes. All five deliverables in
place, unchanged since T+390. The remaining time is being spent on
investigation, not delivery.

**Reliability: high.**

### 2. The recurring item is closed: the BRIEF has been corrected

For ten consecutive checkpoints I have recorded that the BRIEF's errors lived
only in commits and module notes. **That is now resolved.** `BRIEF.md` is 998
lines (from ~200) and carries the corrections directly:

- *"The 1800 s settle budget does not exist. `cli/main.py:902` is
  `_SETTLE_TIMEOUT = 14400.0` — four hours — and `--timeout SECONDS` exposes it.
  Packages' READMEs still cite 1800 s; they are stale."*
- SPX/CPX varies per node and changes your `tp_size`; `crsuse2-m2m-080` measured
  at **64 devices × 36 GiB**, with the sizing consequence spelled out — *"GLM is
  not viable on the CPX node."*

The next effort will read the corrections rather than rediscover them.

### 3. Integration corrected itself again — and this one is the most important

Three further commits and a **control experiment** that overturns section 17,
which I reported at T+390 as the E9 diagnosis. **My T+390 account was built on a
conclusion its own author has since narrowed.**

At 14:59 on `-276`, under the same co-tenant load, integration brought up a
**stock** container (`patch mounts: 0`, confirmed in the bring-up log) and
replayed the identical trace three times, against the four patched replays from
fifteen minutes earlier:

```
patched  mean itl 475.7 ms    mean out_tps 48.5
stock    mean itl 470.3 ms    mean out_tps 47.3
```

**1.1 % apart. Stock reproduces the slow state exactly; the patch has no
measurable cost** — and the patched arm was nominally *faster*.

It then names both earlier readings as wrong, including its own:

- *"a declared no-op cannot cost 21 %, therefore it is run-to-run spread"* — read
  `"expect": {"speedup": 1.0}` as a fact about what happened; **it is a statement
  of what someone expected**;
- *"the patched arm reproduced at 46.49 and 46.70, therefore the patch is
  expensive"* — treated **two draws that both landed in the slow state** as a
  reproduction of a patch effect.

> **The missing thing was never a better inference. It was a control: a stock
> deployment measured under the same conditions.** Nobody had one, because in the
> graph the two arms are always separated by an hour of measurement.

**And it explicitly weakens section 17**: *"the state is drawn at bring-up and
held for the container's life"* — the first half is too strong. What was measured
is that **one container stays in one state for its life**, equally consistent
with *node conditions* being steady across that life. *"The draw language implies
an internal coin-flip, and there is no evidence for one."*

The leading explanation is external and was visible all along, unmeasured:

```
rocm-smi --showuse   -> GPU[0..7] 100%
rocm-smi --showpids  -> several processes holding 140-149 GB VRAM each
                        (another tenant, up ~19 h)
```

r5's stock arm ran 12:58–13:40 and measured **193.59** tok/s; a stock arm at
15:00 under that load measures **47**. **Node contention at measurement time
explains every observation without the patch doing anything.**

**Still not established**, and correctly left open: whether contention is the
whole story, or whether something is *additionally* latched per container.
Distinguishing them needs a genuinely quiet node, which this cluster has not
offered today.

### 4. E9 — the honest final state

At T+390 I recorded E9 as **diagnosed: bistable per bring-up**. That is now
**too strong**, and I am recording the correction here rather than editing T+390.
The accurate statement:

> **Qwen3.6-27B's ~10× throughput swing is explained by node contention at
> measurement time.** A stock control under load reproduces the slow state to
> within 1.1 %, so the patch costs nothing. Whether contention is the *entire*
> cause, or whether a container additionally latches a state at bring-up, is
> **not settled** and needs a quiet node.

E9 goes from "undiagnosed anomaly" through "diagnosed as bistability" to
"explained by contention, with a residual question" — and the middle step was
wrong. It was corrected in 40 minutes by the person who made it, using an
eight-minute experiment.

### 5. The stage-level finding, which is what survives

The reject verdict **stands and is correct** — the arms genuinely differed. But:

> The two-arm design controls for session, node, trace, order and image. It does
> **not** control for node load at measurement time, and on a shared node that is
> the term that dominates.

`check_service_live` proves a deployment is *live*, not *comparable to the other
arm's*. The fix is a **comparability gate** — record node load at each arm's
measurement and refuse when they differ, or interleave the arms rather than
running them in sequence. **Widening the bar fixes nothing; it moves the point at
which an uncontrolled variable is allowed through.**

### 6. The method lesson

Integration's own summary, and the single most transferable line produced today:

> Three people — me twice, the leader twice, `deploy` once — produced **five
> confident explanations from the same data set** without running the one cheap
> experiment that could discriminate between them. It cost **eight minutes**: one
> bring-up and three replays.
>
> **When the question is "did X cause Y", the answer is a control, not a
> re-reading. A dataset that lacks the control cannot be argued into containing
> it, however many people look at it.**

I was a sixth reader of that same data set, and I relayed two of those
explanations as findings at T+270 and T+390.

### 7. Code and non-code problems — final

Unchanged from T+390. **C9b**, **C23**, **C24** open and recorded-not-fixed;
**C25** resolved as a design finding (defaults 5 %/10 % correct, the widening
retracted); **C26** documented. **E0** (5 of 7 holds died, unexplained) open;
**E14**, **E16** open; **E10′** fixed for one module, propagation failed;
**E9** as restated in §4 above.

### 8. Undetermined

1. **Is contention the whole story, or is something latched per container?**
   Needs a quiet node. The one open technical question.
2. **Should the design gain a comparability gate?** Integration's
   recommendation; unruled.
3. **E0** — five reaped holds, no explanation.
4. C9b / C23 / C24 — hand over as recorded findings.

### 9. New commits

```
4b62b29 docs(integration-demo): a control experiment shows the mock patch costs nothing
d3bb76a docs(integration-demo): how to reproduce a slow instance, and a ranked next-step list
e42491e docs(integration-demo): the co-tenant load is independent of this module
```

Working tree clean apart from `.serena/` and this file.

### 10. Other

- **Five modules delivered; the last hour has been pure investigation.** Nothing
  in the deliverables changed; what changed is the confidence with which their
  one refused verdict can be explained.
- **The day's three self-corrections all came from the module that had been
  wrong**: analyze on C9's cause, kernel-opt on "9/9 verified", integration
  twice on the regression. None was forced by a reviewer.
- **My own corrections, all recorded in the following section rather than by
  editing**: the NFS/`TMPDIR` rule (T+90), C9's root cause (T+150), the
  out-of-band recovery's reason (T+330), and E9's diagnosis (here). The
  append-only rule earned itself four times.

---

## T+450 — 2026-09-02 16:05 UTC

### Walltime countdown — **13 minutes left**

| job | node | owner | state | ends (UTC) | remaining |
|---|---|---|---|---|---|
| `101052` | `crsuse2-m2m-079` | profiling + deploy | RUNNING (7 h 46 m) | 16:17:41 | **13 min** |
| `101053` | `crsuse2-m2m-276` | integration | RUNNING (7 h 46 m) | 16:17:57 | **13 min** |

Both will reach their 8 h walltime intact — **the only two of seven holds today
to do so.** Five freshly issued holds died inside 70 minutes each; the two issued
in the original batch never faltered. That asymmetry was never explained.

### 1. Progress

**Effort: complete.** Elapsed 458 minutes (7 h 38 m). All five modules delivered;
no work in flight; nothing at risk from the walltime.

**Reliability: high** — the table below is a fresh count taken from disk at this
checkpoint, not carried forward.

### 2. Final verification sweep — counted at 16:04 UTC

| module | sealed handoffs | verdicts true | verdicts false | PROVENANCE | size |
|---|---|---|---|---|---|
| **profiling** | 7 | 7 | 0 | yes | 362 M |
| **kernel-opt** | 8 | 12 | 0 | yes | 535 K |
| **deploy** | 1 | 2 | 0 | yes | 526 K |
| **analyze** | 11 | 11 | 0 | yes | 796 K |
| **integration** | 18 | 18 | **2** | yes | 2.6 M |
| **total** | **45** | **50** | **2** | 5/5 | ~366 M |

Method: `manifest.yaml` and `validation.yaml` counted per deliverable tree;
`result:` lines counted directly. Counts exceed earlier per-run figures because
several deliverables carry both a current and a superseded store
(`kernel-opt/superseded/`, `analyze/store-run2-superseded/`,
`integration/store` (r4) alongside `r5-.../handoffs`), each with its own sealed
manifests. **Every module has a `PROVENANCE.md`.**

The **two false verdicts** are both `check_no_regression`, one each from
integration's r4 and r5 — the same validator, refusing for the reason
established in §18 of its notes. They are labelled as refusals in three separate
documents. Nothing else in 52 recorded verdicts is a failure.

### 3. Current state

All five modules quiescent. No note has grown and no commit has landed in
50 minutes:

| module | last note | last commit |
|---|---|---|
| deploy | 12:05 | ~12:2x |
| analyze | 12:48 | ~13:0x |
| profiling | 13:23 | ~13:2x |
| kernel-opt | 13:27 | ~13:3x |
| integration | 15:13 | 15:2x |

Working tree clean apart from `.serena/` and this file. Every module's notes and
fixes are committed.

### 4. Problems — closing state

**Fixed and proven in a run:** the srun→`spur exec` transport (profiling,
analyze, integration); `items_schema` drift (analyze, `acb8bfe`, verified by a
green run); the login-node OOM via a `local` transport (`8274a08`); the
`min_requests` floor; hard-coded GPU targets, container names, context lengths
and GLM-only flag groups turned into parameters across three packages.

**Open, recorded, not fixed** — these should be handed over as findings, not as
pending work:

| # | finding |
|---|---|
| **C9b** | `seal_refused` has no reader in `agent/runner.py`; a correct, specific refusal is discarded and surfaces as a stalled task. Cost analyze two runs |
| **C23** | integration's ten closed `items_schema`s restate their content types — correct today, latent drift tomorrow |
| **C24** | `check_workset_runs` hard-fails on an rsd breach while forgiving `ran`/`correct` via `min_pass_ratio`, which makes that knob misleading |
| **E0** | five of seven holds reaped mid-run, unexplained; the two originals untouched for 7 h 46 m |
| **E14** | staging: write into the handoff as you go — local scratch cost deploy a complete kit |
| **E16** | a saturated shared node fails `max_rsd` on correct evidence |
| **E9′** | the residual: is node contention the whole story, or is something additionally latched per container? Needs a quiet node |

**Design finding, the most valuable single output:** the two-arm comparison
controls for session, node, trace, order and image, and **not for node load at
measurement time**, which on a shared node dominates. The fix is a comparability
gate, not a wider bar.

### 5. Undetermined

1. **E9′** — contention versus a per-container latch. One experiment, needs a
   quiet node.
2. **A comparability gate** for the two-arm design — recommended, unruled.
3. **E0** — why five holds died.
4. Whether the out-of-band pieces in integration's directory stay; deploy
   offered unconditionally to remove its packup and nobody needed to decide.

### 6. New commits

**None** since T+420. Tree clean.

### 7. Assessment of the effort

- **5 of 5 modules delivered**, 45 sealed handoffs, 50 PASS verdicts, 2 refusals
  that are correct and labelled as such, 5 PROVENANCE files.
- **Every deliverable understates itself.** Profiling: don't quote a number.
  Kernel-opt: no kernel was optimised, and the mock validator passed without
  measuring. Deploy: not "both validators PASS" (until it was). Analyze:
  `generating`, not `valid` (until it wasn't). Integration: "validated out of
  band" is not "sealed" — say it that way round.
- **Nothing regressed at any of the sixteen checkpoints.** No deliverable
  withdrawn, no verdict revoked, no finding retracted — only narrowed, corrected
  in place, or superseded by a better run.
- **Six substantive self-corrections**, every one from the party that had been
  wrong and none forced by a reviewer: analyze on C9's cause and on `top_n`,
  kernel-opt on "9/9 verified", deploy on kernel-opt's framing, integration twice
  on the regression. Plus my own four, each recorded in the following section
  rather than by editing the error.
- **The transferable lesson**, integration's: *when the question is "did X cause
  Y", the answer is a control, not a re-reading. A dataset that lacks the control
  cannot be argued into containing it, however many people look at it.* Six
  readers produced five confident explanations of the same data before someone
  spent eight minutes on a control.
- **The BRIEF was corrected** (998 lines, from ~200) so the next effort inherits
  the settle budget, the CPX geometry and the localisation traps rather than
  rediscovering them.

This is the last checkpoint at which anything can change on a GPU. I will keep
the 30-minute rhythm until told to stop, but absent further instruction the
record from here is expected to be flat.

---

## T+480 — 2026-09-02 16:35 UTC

### Walltime — **both holds reached their limit and ended cleanly**

```
101052  JobState=TIMEOUT  Reason=TimeLimit  RunTime=08:00:17  EndTime=16:17:58
101053  JobState=TIMEOUT  Reason=TimeLimit  RunTime=08:00:15  EndTime=16:18:12
```

`squeue -u yihou` is **empty**. The effort has no compute.

`RunTime=08:00:17` against `TimeLimit=08:00:00` is the shape of a hold that ran
its full course and was ended by the scheduler for the stated reason — the exact
opposite of the five that died with `JobState=CANCELLED Reason=None` at 30, 34,
70, 30 and ~30 minutes. Final tally: **two of seven holds survived, and they were
the two issued in the original batch.** Still unexplained, and now unfalsifiable
for today.

Everything the effort needed from a GPU was finished by 15:20, an hour before
the wall. Nothing was lost to the timeout.

### 1. Progress

**Complete.** Elapsed 488 minutes (8 h 08 m). No work outstanding, none possible.

### 2. Final state — unchanged and verified

| module | deliverable | size |
|---|---|---|
| profiling | 7 sealed handoffs, 7 verdicts PASS, PROVENANCE | 362 M |
| kernel-opt | 8 sealed, 12 PASS, PROVENANCE, digest self-check | 535 K |
| deploy | 1 sealed, 2 PASS, PROVENANCE, + the unsealed d3 kit | 526 K |
| analyze | 11 sealed, 11 PASS, PROVENANCE, + superseded store | 796 K |
| integration | 18 sealed, 18 PASS **and 2 correctly-refused**, PROVENANCE, + leader/kernel-opt/deploy recovery material | 2.6 M |

**45 sealed handoffs, 50 PASS, 2 documented refusals, 5 of 5 PROVENANCE files.**

Notes: analyze 878 lines, deploy 1038, integration 983, kernel-opt 1090,
profiling 671 — **4 660 lines of debug notes**, all committed. Working tree
clean apart from `.serena/` and this file.

### 3. Nothing changed this interval

No commits since 15:2x. No note since integration's at 15:13. No module active.
The last 80 minutes have been quiescent, which is the correct end state.

### 4. What remains open, for handover

Unchanged from T+450 and now final for this effort:

- **E9′** — is node contention the whole story, or is something additionally
  latched per container? Needs a quiet node. One experiment, method written down.
- **The comparability gate** — the two-arm design does not control for node load
  at measurement time. Recommended by integration, unruled.
- **C9b** — `seal_refused` has no reader; a correct refusal is discarded.
- **C23** — integration's ten closed `items_schema`s.
- **C24** — `check_workset_runs` hard-fails on rsd while `min_pass_ratio`
  forgives correctness.
- **E0** — five of seven holds reaped, unexplained.
- **E14** — write into the handoff as you go; local scratch cost a complete kit.
- **E16** — a saturated node fails `max_rsd` on correct evidence.

None of these blocked a delivery. All are recorded with a named file, a measured
symptom, and — where one exists — the experiment that would settle them.

### 5. Closing note on this file

Sixteen checkpoints over eight hours, appended and never revised. It contains
four estimates I would now write differently and four findings I reported and
later corrected — the NFS/`TMPDIR` rule (T+90), C9's root cause (T+150), the
reason for integration's out-of-band recovery (T+330), and E9's diagnosis
(T+420). Each correction sits in the section after the error, which is the whole
point of the append-only rule: a reader can see not just what was true at the
end, but how long each wrong thing was believed and what dislodged it.

Two blind spots are also on the record: deploy at T+90 and integration at T+120,
each reported as "unknown, not scored" rather than guessed at. Both were working
hard and writing nothing, which is what a module deep in a long run looks like
from outside. Given the same evidence I would make the same call.

I will keep the 30-minute rhythm until the leader says stop, but with no compute,
no active module and a clean tree, the record from here is expected to be flat.

---
---

# Checkpoint summary — three-level component install + `examples/env_checker`

A **new effort**, appended below the five-module `llm_e2e` debug above. The
sections above belong to that earlier effort and are not touched.

Append-only, same discipline: one section per ~30 minutes, earlier sections are
never revised, wrong estimates are left standing because the record over time is
the value.

Effort start (T+0) taken as **2026-09-03 08:13 UTC**, the minute the task book
`CLAUDE.md` for this effort was written at the repo root (`CLAUDE.md`, with the
predecessor saved beside it as `CLAUDE.kernel_opt.20260903-0813.md.bak`).

What is being built: `agent_sys` today delivers exactly one Claude Code
capability to an agent (`skills`). This effort builds the other six and puts all
seven behind one install hierarchy with three levels — **L1** `env_mgr` recipes
named in YAML, **L2** a repo component registry at `agent_sys/components/`,
**L3** auto-detection of `<agent asset dir>/.claude/` — and proves each of them
with a new package `agent_sys/examples/env_checker` that runs them and hands back
per-capability tokens derived from a run nonce.

Plan: `/home/yihou/.claude/plans/zany-hugging-lightning.md` (199 lines).
Teammates: **`core-impl`** (owns `agent_sys/{spec_loader,agent,env_mgr,tests}`)
and **`pkg-author`** (owns `agent_sys/examples/env_checker` and
`agent_sys/components`).

Reporter reads, cheapest first: `git status --short` / `git log --oneline` /
`git diff --stat` in the worktree; `agent_sys/examples/env_checker/` and
`agent_sys/components/` if they exist; `/tmp/yihou/agentsys_envchecker_20260903/`;
the repo-root `CLAUDE.md`; the plan.

---

## T+0 — 2026-09-03 08:38 UTC (baseline)

Written 25 minutes after the nominal T+0; this is the first observation, not a
retro-fit of what the state was at 08:13.

### 1. Progress

**Effort: ~5 %.** Elapsed 25 minutes. Estimated remaining: unknown.

| workstream | est. % | basis |
|---|---|---|
| planning / task book | **100 %** | `CLAUDE.md` and the 199-line plan both exist and are complete, with a work breakdown down to file and line number |
| probes (the 3 measured assumptions) | ~33 % observable | probe (a)'s artefacts exist on disk; (b) and (c) have left nothing |
| core changes (`spec_loader`, `env_mgr`, `agent`, `tests`) | 0 % observable | `git diff --stat` is empty; no tracked file has changed |
| `examples/env_checker` | 0 % observable | the directory does not exist |
| `agent_sys/components/` | 0 % observable | the directory does not exist |

**Reliability: very low, and asymmetric.** The planning number is solid — I read
both documents. Every other number is a floor derived from an absence, and at
T+25min an absence means "has not yet written to a path I can see", which I
*cannot* distinguish from "is working in an editor buffer / a subagent that has
not returned". Neither teammate has been observed to fail at anything. There is
no denominator I trust either: the plan lists seven capabilities × three levels
plus five core files plus three test modules plus four docs, and I have no
measurement of how long any one of those takes in this codebase.

### 2. Current state

Repo worktree `/home/yihou/dev/git.16-19/infera.aiopt.real.task_package`, branch
`dev.yihou.aiopt.task_package`, HEAD `9bf72c8` (the previous effort's final
commit).

`git status --short` shows exactly two untracked paths and **no modifications**:

```
?? CLAUDE.kernel_opt.20260903-0813.md.bak
?? progress.bar.for.user.md
```

- **`core-impl`** — no observable output. No tracked file under
  `agent_sys/{spec_loader,agent,env_mgr,tests}` differs from HEAD.
- **`pkg-author`** — no observable output. `agent_sys/examples/` still holds the
  six pre-existing packages (`demo`, `demo-broken`, `demo2`,
  `llm_e2e_perf_opt_debug_workset`, `llm_e2e_performance_optimization`,
  `single_real_task`); there is no `env_checker`. There is no
  `agent_sys/components/`.
- **Scratch** `/tmp/yihou/agentsys_envchecker_20260903/` exists, created 08:13,
  with `logs/` (empty) and `probes/` (08:14). Under `probes/`: a relocated config
  dir `cfg_a/` and two synthetic marketplaces `mp/` and `mp2/`, the first
  shipping a plugin that carries a skill.

### 3. Code problems — fixed / not fixed

**Fixed this period:** none. No code has been written yet.

**Not fixed — the defects this effort exists to correct**, as stated first-hand
in the task book and plan and not yet contradicted by anything I read:

- `agent_sys/env_mgr/material.py` — writes no `settings.json` into the zone, so a
  declared `hooks:` path lands at `<zone>/config/hooks/<basename>` and is read by
  nobody. Declarable but inert.
- `agent_sys/env_mgr/recipe.py` + `agent_sys/env_mgr/installers/` — reachable only
  from the `env-mgr` CLI (`env_mgr/cli.py:74`). **Recipes never run during a task
  at all**; `prepare` installs nothing.
- Plugins — installable only via a recipe that writes the operator's `~/.claude`,
  which the zone's relocated `CLAUDE_CONFIG_DIR` then hides. Unreachable from a
  package.
- MCP — reachable only through a hand-written `config.options.mcp_servers`, with
  no example anywhere in the repo. No `Prepared.mcp_servers`, no
  `Assignment.mcp_servers`.
- In-process `ToolDef` — reachable only from `env_mgr`'s own remote surface.
- `agent/docs/design.md` **O4** ("which hook surface is canonical") is an open
  question in the checked-in docs; the plan intends to close it.

None of these are regressions; all are the pre-existing state of `main`.

### 4. Non-code problems

- **This effort writes near `~/.claude`, which is outside `yihou/`.** The task
  book's own mitigation: every probe runs with `CLAUDE_CONFIG_DIR` pointed into
  `/tmp/yihou/...`, and `~/.claude` / `~/.claude.json` are checksummed before and
  after. I can see the relocated config dir; I have **not** found a
  before/after checksum record on disk, so I can report the intent but not that
  the check was performed.
- **`~/.claude.json` is not documented as following `CLAUDE_CONFIG_DIR`** (task
  book, first-hand). It holds MCP server config and sign-in. This is the stated
  reason MCP is routed through `Assignment.mcp_servers` rather than a file, and
  it is a constraint on the design, not a bug.
- **Binary version skew**: `claude` on `PATH` here is 2.1.246 while the SDK
  prefers its own bundled binary; `Prepared.agent_cli` already refuses the
  mismatch (`claude_sdk.py:399-440`). Anything that shells out to `claude`
  (plugin install) and anything that starts a session may therefore not be the
  same binary.
- **serena is slow and networked** — the plan makes its recipe item
  `importance: suggested` for that reason. Network reachability from this host
  has not been measured in anything I can see.
- Host is 8× MI300X `gfx942`; nothing in this effort needs a GPU, so the walltime
  pressure that dominated the previous effort does not apply here. No slurm hold
  is involved.
- Commits need `git commit -s` (DCO, `.claude/CLAUDE.md`). Repo language English,
  user-facing reporting Chinese.

### 5. Open questions, not yet characterised

- **Probe (a) — does `claude plugin install` respect `CLAUDE_CONFIG_DIR`?**
  Observed: under `probes/cfg_a/` there is a `plugins/installed_plugins.json`
  listing `envchk-plugin@envchk-mp` and `second-plugin@second-mp`, both scope
  `user`, both with `installPath` inside `/tmp/yihou/...`, installed 08:14:26 and
  08:14:46 UTC; and a `settings.json` carrying `enabledPlugins` for both plus
  `extraKnownMarketplaces` pointing at `probes/mp` and `probes/mp2`, alongside a
  `SessionStart` hook. That is what the files say. Whether the operator's
  `~/.claude` was left untouched, and whether the probe was declared a pass, I
  have not seen recorded anywhere.
- **Probes (b) and (c)** — hooks reaching an SDK-started session, and an external
  `mcp_servers` entry reaching the model — have produced no artefact under
  `probes/`. Not run, or run without writing: I cannot tell which.
- `logs/` was created at 08:13 and is still empty at 08:38.
- Neither teammate has sent me anything. Silence at T+25min is silence.

### 6. New commits

None. `git log --oneline` is unchanged from the previous effort's tip:
`9bf72c8 debug the five e2e stage packages to one sealed handoff set each`.

### 7. Anything else worth recording

- The plan's acceptance is **one real run**, accepted by opening the handoff —
  seven sections, seven tokens, both validators PASS — explicitly *not* by exit
  code. Three consecutive unattended runs are **not** asked for this round. That
  is a materially smaller acceptance bar than the previous effort's, and it is
  the user's ruling, not a relaxation taken here.
- The design's load-bearing simplification: **L2 and L3 have the same on-disk
  shape** (a `.claude/` tree in Claude Code's canonical layout), so one installer
  serves both and nothing is converted. If that stops being true, the size
  estimate for the core changes stops being true with it.
- Evidence in `env_checker` is **token-based, not narrative**: each capability's
  token derives from a per-run `ENVCHK_NONCE`, so a capability that did not run
  cannot produce its token and a validator cannot be satisfied by an agent that
  merely reports success. This is the direct descendant of the earlier effort's
  worst failure — ten validators PASS over a run in which every result was zero.
- The untracked `progress.bar.for.user.md` at the repo root predates this
  effort's first observation; I did not read it and do not know whose it is.

---

## T+65 — 2026-09-03 09:18 UTC

Forty minutes since the baseline. Everything the baseline marked "0 % observable"
now exists on disk, and the earlier estimate of 5 % was correct as a floor and
badly wrong as a prediction — the writing was already well under way in buffers I
could not see. That is exactly the failure mode the baseline named and did not
avoid.

Sources this section, kept separate on purpose: **verified** = I opened the file
or ran the command; **reported by the lead** = arrived in the lead's checkpoint
message and I have not independently confirmed it.

### 1. Progress

**Effort: ~70 %.** Elapsed 1 h 5 m. Estimated remaining: **1–2 h**, dominated by
one thing that has not started.

| workstream | est. % | basis |
|---|---|---|
| probes | **100 %** | verified: `logs/PROBES.md`, 139 lines, six probes A/A'/B/C/D/E each with a stated method and a YES |
| core changes | ~90 % | verified: 23 tracked files modified, `env_mgr/agent_assets.py` (1029 lines) and `env_mgr/recipes/serena.yaml` (160) new |
| `examples/env_checker` | ~95 % | verified: 32 files incl. `main.yaml`, `steps/check.yaml`, both validators, the full L3 `.claude/` tree, `ACCEPTANCE.md` |
| `agent_sys/components/` | ~90 % | verified: `README.md` + `envchk-baseline/` with `.claude/.mcp.json` and its server |
| tests | ~80 % | verified: `tests/env_mgr/test_agent_assets.py` is 945 lines new; 5 existing test modules modified. Lead reports a round in flight |
| **the one real run** | **0 %** | verified: no run root, and the lead states it has not been launched |
| commits | **0 %** | verified: `git log` still `9bf72c8`; 24 modified + 7 untracked paths, 1099 insertions |

**Reliability: medium for the written work, low for the remaining time.** The
percentages above are line counts and file inventories, which measure *presence*,
not correctness — and two defects found in the last forty minutes (§3) were both
in files that were present, loaded, and passing pre-flight. The 1–2 h remaining
is a guess with no measurement behind it: nobody has yet run this package
end-to-end, and the run's duration is unknown. Two bugs of the same class were
found by reading artefacts before the run; whether a third is waiting is
precisely what the run will say.

### 2. Current state

`git status --short`: 24 modified (23 source/docs + this file), 7 untracked.
HEAD still `9bf72c8`.

- **`core-impl`** — has landed the whole core surface. Modified: `agent/backend.py`,
  `agent/backends/claude_sdk.py`, `agent/runner.py`, `agent/spec.py`,
  `agent/docs/{design.md,spec.md}`, `env_mgr/{material.py,paths.py,prepare.py,protocols.py,protocols.pyi,README.md}`,
  `env_mgr/docs/design.md`, `env_mgr/isolation/policy.py`,
  `spec_loader/{assets.py,package.py}`, `spec_loader/schemas/agent.schema.json`,
  and 5 test modules. New: `env_mgr/agent_assets.py`, `env_mgr/recipes/serena.yaml`,
  `tests/env_mgr/test_agent_assets.py`. Reported mid test-round.
- **`pkg-author`** — package complete and reported idle/standing by. The L3 tree
  under `assets/env_probe.agent/.claude/` carries all of: `settings.json`,
  `hooks/envchk_session_start.py`, `skills/envchk-probe/SKILL.md`, a local
  marketplace `plugins/.claude-plugin/marketplace.json` with `envchk-plugin`
  shipping its own skill, `tools/envchk_stdio.mcp.py`, and
  `tools/envchk_inproc.tooldef.py`. Beside it, `serena_probe.py` and a `.serena/`
  project dir with a warmed symbol cache.
- **Shared library** `assets/lib/envchk.py` holds the salt machinery:
  `SALT_TAG = re.compile(r"ENVCHK_SALT:\s*([0-9a-f]{32})")`, and it treats both
  "no tag" and "more than one distinct tag in one file" as errors.
- **`ACCEPTANCE.md`**, 235 lines, inside the package — written **before** the run,
  which is the discipline the previous effort's `.claude/CLAUDE.md` principle 2
  asked for and which that effort did not always keep.
- **Scratch** `/tmp/yihou/agentsys_envchecker_20260903/logs/` now holds
  `PROBES.md` (08:46), `poll_notes.md` (202 lines, 09:06), `serena_install.log`,
  `home_before.md5`, `t1.md5`, `home_after.json`.

**Anti-forgery, verified independently by me, not taken on report.** Seven
`ENVCHK_SALT` tags, one per capability, each in exactly one file, all seven
distinct:

| capability | file | salt |
|---|---|---|
| hook | `.claude/hooks/envchk_session_start.py:54` | `6ea6f6c7…` |
| skill (L3) | `.claude/skills/envchk-probe/SKILL.md:25` | `141016c5…` |
| plugin | `.claude/plugins/envchk-plugin/skills/envchk-plugin-skill/SKILL.md:33` | `fdf78796…` |
| bundled stdio MCP | `.claude/tools/envchk_stdio.mcp.py:33` | `a65f7dfd…` |
| in-process `ToolDef` | `.claude/tools/envchk_inproc.tooldef.py:36` | `3cccd425…` |
| serena | `serena_probe.py:61` | `a3ebf3e2…` |
| external MCP (L2) | `components/envchk-baseline/.claude/servers/envchk_baseline_server.py:32` | `48d7f4c1…` |

### 3. Code problems — fixed / not fixed

**Fixed this period** (both reported by the lead as found-and-fixed; I verified
only that the files exist and that `envchk.py` carries the salt logic, not the
fixes themselves):

- the install report is `{"outcomes": [...]}` — an **object** — where the shape
  validator had been written expecting the bare array. Would have failed the
  first real run; passed load and pre-flight.
- serena's `find_symbol` returns the symbol **body and nothing above it**, so a
  module-level `SALT` constant never appeared in the response. Row 7 of the
  report would have failed on **every honest run** — a false negative built into
  the test, the mirror of the previous effort's false positives.

Both are one class: *a schema assumed rather than measured*. The lead has added
the obligation to the repo-root `CLAUDE.md`: never copy from an artefact nobody
has opened; do not encode an unmeasured third-party schema.

**Fixed this period, verified by me:** `agent_sys/tests/agent/test_imports.py`
does **not** appear in `git status`, consistent with the lead's report that the
L1-route reversal returned it to its shipped form.

**Not fixed / still open:**

- Everything in the T+0 list remains open until the core changes are proven by a
  run; presence in `git status` is not proof.
- `__pycache__` directories with `.pyc` files have been committed-adjacent inside
  the package assets — `assets/lib/__pycache__/{envchk,zone}.cpython-313.pyc` and
  `assets/env_probe.agent/.claude/tools/__pycache__/envchk_inproc.tooldef.cpython-313.pyc`
  — and `agent_sys/examples/env_checker/` is untracked as a whole, so a naive
  `git add` of the directory would carry them in. Also `.serena/cache/python/*.pkl`
  (two pickles) and `.serena/project.local.yml`. Verified by `find`; not yet
  flagged by anyone else, and I have not checked whether `.gitignore` covers them.

### 4. Non-code problems

- **The `~/.claude.json` checksum method was withdrawn by the lead as invalid on
  this host** (reported): with no probe running it changed twice in 75 seconds,
  because every live Claude Code session rewrites it. Earlier "unchanged"
  readings were luck, not evidence. Note that `PROBES.md:5` still states
  "`~/.claude.json` md5 checked before and after each: **unchanged**" — the
  write-up predates the withdrawal (08:46 vs the correction) and has not been
  updated. The substitute evidence the lead names is per-probe: each probe's own
  `.claude.json` inside its relocated config dir, plus
  `~/.claude/plugins/marketplaces` still holding only `claude-plugins-official`,
  which `PROBES.md` does record for probe A.
- **`uv tool install` writes `~/.local/share/uv` unless `UV_TOOL_DIR`,
  `UV_TOOL_BIN_DIR` and `UV_CACHE_DIR` are all pinned into a yihou dir**
  (reported, probe D). This is a real instance of the hard rule's blast radius:
  a default that writes outside `yihou/` without being asked.
- **Two writers on one file.** The lead began editing `agent_assets.py` while
  `core-impl` was in it, for a few minutes, then stood down and disclosed what it
  had left behind. No lost-update has been reported; the file is 1029 lines and I
  have not diffed it against anything.
- **Mail arrives between turns.** The lead read `core-impl` as ignoring a ruling
  when in fact every one of its messages landed after the work was already on
  disk; the misreading is withdrawn and recorded in `poll_notes.md` as a
  correction to the polling discipline — *in a system with between-turn mail
  delivery, silence is not refusal*. This is the same inference error this file
  made at T+0 from the other side, and it is worth the whole checkpoint to have
  it written down twice.

### 5. Open questions, not yet characterised

- **The real run has not been launched.** Everything above measures artefacts, not
  behaviour. No run root exists.
- Probe F is described by the lead as: a plugin installed into the zone config is
  visible to the session, and loads from the **marketplace source path, not a
  copy**. I did not find an F section in the 139-line `PROBES.md` I read (A, A',
  B, C are in the part I opened; D and E are attested by `serena_install.log`
  existing). Whether F is later in the file, in `poll_notes.md`, or unwritten, I
  did not check.
- `pytest agent_sys` was 2176 passed / 3 skipped / 4 xfailed at the last report
  and is described as mid-round. I have not run it. The 4 xfails are unexamined
  by me and predate this effort as far as I know.
- The four accepted deviations from the plan (`agent_assets` rather than
  `material.py` writes the zone `settings.json`; `deploy` returns a `Deployed`
  NamedTuple; `agent_assets` defines its own `InstallOutcome` rather than
  importing `Outcome` across the decoupling wall; L1 installs run as a
  subprocess) are reported as accepted. I have not read the code to see whether
  the docs modified in this diff say the same thing the code does.

### 6. New commits

**None.** `git log --oneline` is still `9bf72c8`. Approximately 31 paths of work
are uncommitted — 24 modified, 7 untracked — which is the largest single body of
unversioned work this file has recorded. `git diff --stat HEAD` reports 1099
insertions across the tracked files alone; the untracked package and component
trees are not in that number.

### 7. Anything else worth recording

- **The reversal is the most interesting decision of the period.** L1 installs
  were implemented as an in-process import and were re-ruled to a subprocess
  (`python -m env_mgr … --json`). The lead reports re-deriving the ruling rather
  than restating it, and that it stood on three legs — the strongest being that a
  subprocess takes `env=` and therefore removes a process-global `os.environ`
  mutation that is unsafe in a threaded runner. That is a correctness argument,
  not an architectural preference, and it is the kind of reason that survives
  someone asking again in a month.
- **Probe B settles `agent/docs/design.md` O4** for this repo's purposes: the
  canonical hook surface is the declarative `settings.json` one. `PROBES.md:47`
  says so explicitly, and `agent/docs/design.md` is in the modified set, so the
  answer is being written where the question was asked.
- **Probe A' produced a design consequence, not just a fact**: `plugin install`
  merges rather than clobbers, so the hooks `settings.json` must be written
  *first* and the plugin installs run after. The reverse order also works but
  costs a second merge.
- `home_after.json` is 22 KB and mode `0600` while its sibling `home_before.md5`
  is 59 bytes — the before/after pair are not the same kind of measurement. Noted,
  not explained.

---

## T+68 — 2026-09-03 09:21 UTC

A short interval — three minutes of wall clock after the last section, written
because the lead answered all three of T+65's findings at once and two of them
changed the file I had just cited. Recording the resolutions while they are
cheap to verify.

### 1. Progress

**Effort: ~72 %.** Elapsed 1 h 8 m. Estimated remaining: **1–2 h**, unchanged and
for the unchanged reason — the real run has still not been launched.

The three-point move since T+65 is not new implementation. Tracked-file source
insertions are **flat**: `git diff --stat HEAD` reports 1292 insertions of which
372 are this checkpoint file, leaving 920 — the same 920 as at T+65 (1099 − 179).
What moved is untracked: `agent_assets.py` grew 1029 → 1079 lines (mtime 09:20,
i.e. still being written as I read it) and the contract items below landed.

**Reliability: medium.** The completeness claims in §2 I checked by grep; the
test-count and the "every contract item has landed" judgement are the lead's,
relayed from `core-impl`, and I did not run the suite.

### 2. Current state

`git status --porcelain` shows 31 paths. HEAD still `9bf72c8`.

- **`core-impl`** — reported by the lead as having landed every outstanding
  contract item: `AGENT_SYS_COMPONENTS_ROOT` (granted then exported),
  `AGENT_SYS_INSTALL_REPORT`, `${VAR}` expansion that **refuses** an unresolved
  variable, the marketplace copied into the zone *before* registration (which is
  probe F's finding turned into behaviour), a `sys.modules` fix carrying two
  tests that were checked to fail without it, and `env_mgr/recipes/serena.yaml`.
  Suite reported **2187 passed / 3 skipped / 4 xfailed** (was 2176 at T+65).

  Verified by me, by grep, that both env vars are referenced across code, docs,
  tests and the package: `AGENT_SYS_COMPONENTS_ROOT` in `env_mgr/prepare.py`,
  `env_mgr/paths.py`, `env_mgr/isolation/policy.py`, `env_mgr/README.md`,
  `env_mgr/docs/design.md`, `tests/env_mgr/test_agent_assets.py`, and the
  package's `README.md` + `check_capabilities_genuine.validator/`;
  `AGENT_SYS_INSTALL_REPORT` in `env_mgr/{paths.py,README.md,docs/design.md}`,
  `tests/env_mgr/test_agent_assets.py`, and the package's `README.md`,
  `ACCEPTANCE.md`, `assets/probe_env.task/readme.md`. That is presence in the
  policy layer, the docs and the tests, not merely a definition.
- **`pkg-author`** — still standing by; being asked about the two serena files
  (below) before anything is committed.
- **The real run** — not launched.

### 3. Code problems — fixed / not fixed

**Fixed this period, all three verified by me:**

- `PROBES.md:5` — the retracted `~/.claude.json` checksum. Replaced with an
  explicit **WITHDRAWN, 2026-09-03** paragraph at line 5 rather than deleted, so
  anyone who already quoted the old header can see it was retracted and why, and
  a full `## Correction` section now sits at line 128. The file grew 139 → 147
  lines. Retracting in place instead of erasing is the right call for a document
  other people quote.
- **Probe F was never missing from my reading — the file was stale.** `PROBES.md`
  now carries `## D` (line 70), `## E` (84) and `## F` (101), F reading
  "is a plugin installed into the zone config *visible to the session*? **YES**".
  There is also a `## What is still NOT measured` section at line 61, which is
  the right kind of thing for a probe write-up to carry.
- **`.gitignore` covers the caches; two serena files still get in.** I re-checked
  with `git add -An` rather than by reading `.gitignore`, and the result is
  exactly what the lead reports: 26 files would be staged, and among them
  `assets/env_probe.agent/.serena/.gitignore` and
  `assets/env_probe.agent/.serena/project.yml`. Confirmed ignored (`!!`):
  `assets/lib/__pycache__/`, `.claude/tools/__pycache__/`, `.serena/cache/`,
  `.serena/project.local.yml`. So the caches were never the risk; the serena
  project config is, and it is unresolved pending `pkg-author`'s answer on
  whether it is intended package content or a by-product of measuring against
  that directory.

**Not fixed:** nothing has been retired from the T+65 list beyond the above. The
whole core surface remains unproven by any execution.

### 4. Non-code problems

Nothing new. The T+65 entries stand: the withdrawn checksum method (now correctly
marked in the file itself), `uv`'s three env vars, the brief two-writer overlap
on `agent_assets.py`, and between-turn mail delivery.

### 5. Open questions, not yet characterised

- The run. Still the whole of the remaining risk, by the lead's own assessment
  and by mine.
- `.serena/.gitignore` and `.serena/project.yml` — intended or by-product,
  unanswered.
- `agent_assets.py` had an mtime of 09:20 when I read it at 09:20, so the 1079
  figure is a reading of a moving file and may already be stale.
- The 4 xfails in the suite remain unexamined by me and, as far as I know,
  predate this effort.

### 6. New commits

**None.** HEAD `9bf72c8`. Still ~31 uncommitted paths.

### 7. Anything else worth recording

- **All three T+65 findings were correct, and one was a correction to the lead.**
  Worth recording not as credit but as a method result: each of the three came
  from checking what a command would actually do (`git add -An`) or from reading
  the artefact rather than the claim about it — the same habit that produced the
  two bugs found earlier today. The one that mattered most, `PROBES.md:5`, was a
  *retracted method still standing as evidence in a file people quote*, which is
  a failure mode with no symptom until someone acts on it.
- The lead notes probe F's finding — a plugin loads from the marketplace **source**
  path, not a copy — is now enforced by copying the marketplace into the zone
  before registration. A measurement turning directly into a line of behaviour is
  the cleanest possible use of a probe.
- The interval between this section and the last is three minutes, which breaks
  the ~30-minute cadence. It is here because the resolutions were verifiable now
  and would have been indistinguishable from the next tranche of work later.
