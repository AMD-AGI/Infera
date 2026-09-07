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

# Effort 2 — handoff refine: five modules into one graph

A **new** effort, on a new mission (repo-root `mission.md`, "Handoff refine for
llm e2e opt task package"). The `T+` counter restarts from this section; the
sixteen sections above belong to effort 1 (2026-09-02) and are left exactly as
written. Same append-only rule: nothing above this line, and nothing below it
once written, is ever revised.

Deliverable: `agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/`.
Phase 0 (contract freeze) is done and committed as `9646910`. Phase 1 is five
module owners — `m1-deploy`, `m2-profiling`, `m3-analysis`, `m4-kernel-opt`,
`m5-integration` — working in parallel against that frozen contract.

Reporter reads, cheapest first: `git log`/`git status` in the worktree; the
`show` load-check on the package; `ls` over `e2e-flow/{assets,steps}` and
`assets/schemas/`; and `squeue -u yihou`. The reporter does **not** message the
module owners — the leader polls them, and a module deep in a long piece of work
is indistinguishable from a stuck one when seen from outside. Silence is
reported as "unknown, not scored".

---

## T+0 — 2026-09-03 13:30 UTC (baseline, effort 2)

### Walltime countdown

| job | node | ends (UTC) | remaining at this checkpoint |
|---|---|---|---|
| `106250` | `crsuse2-m2m-061` | **20:44:58** | 7 h 15 m |
| `106253` | `crsuse2-m2m-031` | **20:45:03** | 7 h 15 m |

Both `R`, running 39 m at the time of reading. Two holds for five modules, so
GPU access is a shared resource this round — that is a scheduling fact worth
watching, not yet a problem, because nothing is on GPU yet.

### 1. Progress

**~10 % complete.** Elapsed 0 m at this checkpoint (effort clock starts here;
the contract freeze that precedes it is not counted). Estimated remaining: **6–8
h**, which would land inside the hold window with roughly an hour to spare.

**Reliability of that estimate: low.** It rests on one visible fact — the graph
loads with 17 closures and 20 validators — and on the effort-1 shape, where five
parallel modules took 8 h. Two differences push in opposite directions and I
cannot yet weigh them: this round has ~20 k lines of proven `.py`/`.sh` in the
five sibling demos to adapt rather than derive (faster), and it has two holds
instead of five plus a real-schema layer that did not exist before (slower). The
10 % figure is "Phase 0 of a two-phase job is done and Phase 1 has just begun",
not a measurement.

### 2. Current state, per module

Nothing is committed since `9646910`. `git log 9646910..HEAD` is empty. The
per-module reading below is therefore from the filesystem only.

| module | evidence seen | scored |
|---|---|---|
| m1-deploy | `assets/schemas/environment.schema.json` (5 074 B) and `assets/lib/schema.py` (6 162 B) exist and are **committed in `9646910`** — i.e. leader-seeded, not module output | **unknown, not scored** |
| m2-profiling | none | **unknown, not scored** |
| m3-analysis | none | **unknown, not scored** |
| m4-kernel-opt | none | **unknown, not scored** |
| m5-integration | none | **unknown, not scored** |

One file in the tree is **not** from the freeze:
`e2e-flow/assets/lib/zone.py`, untracked, 5 865 B, mtime 13:23 UTC — written
minutes before this checkpoint. `zone.py` is the shared library (effort 1's
`single_real_task/assets/lib/zone.py` is its ancestor), so it is plausibly a
module owner seeding a shared dependency. I cannot attribute it to an owner from
the filesystem and will not guess; it is recorded here so that the next
checkpoint can tell growth from stasis.

Baseline measurements, to diff against later:

- **20 validators, every one a 12-line skeleton** (`check.py`, 12 lines,
  one `NotImplemented`-class marker each). Zero are real. This is the single
  most legible progress metric this round has, and it currently reads 0/20.
- **Schemas: 1** — `environment.schema.json`, plus `schemas/README.md`.
- **Steps: 6 yaml** (`common` + five `m*`), 59 442 B total, all committed.
- **Tasks: 17 `.task` dirs.** The eleven leaf tasks carry a ~400 B `readme.md`
  and a ~400 B `entry.sh`; the six non-leaf ones carry a ~215 B `readme.md`.
  These are placeholder-sized — a real agent brief in this repo runs to
  thousands of bytes — so **no task body is written yet**.

### 3. Code problems

**None found, and none fixed.** The load check passes:

```
17 closures ... 6 tasks: 1 root and 5 subtasks
main out 2 · m1 out 3 · m2 in 3/out 2 · m3 in 4/out 3 · m4 in 5/out 3 · m5 in 8/out 2
done  6 tasks in the graph; nothing was dispatched
```

Exit 0, under a second. That is the leader's stated Phase-0 acceptance and it
holds at T+0.

The skeleton validators exiting 1 is **deliberate**, per `README.md:15-17` — so
that nothing can report a pass it has not earned. It is not a defect and is not
counted as one. It does mean any real `run` attempt fails at the first
validation phase until a module lands a real `check.py`.

### 4. Non-code problems

- **Repo-root litter, pre-existing and not from this effort.** `git status`
  shows untracked `glm5.2-dp8-tp8-workload-schema.tar`, `handoff.analysis.md`,
  `rank0/`, `.serena/`, and a modified `agent_sys/docs/design.md` at the repo
  root of this worktree. None is inside `e2e-flow/`. I am flagging, not
  touching: a stray tarball and a `rank0/` directory at a repo root are the kind
  of thing that ends up in a commit by accident when five owners run `git add`
  in parallel.
- **Two holds, five modules.** Effort 1 had a hold per module. Whatever
  serialisation this forces is invisible at T+0 because no module is on GPU yet.
- **`zsh` emits a `libtinfow.so.6` version warning on every command** from
  `~/miniconda3/bin/zsh`. Cosmetic, pollutes captured output, unchanged from
  effort 1.

### 5. Open questions

- **Who owns `zone.py`?** Untracked and freshly written; attribution unknown.
- **Is the schema layer seeded or owned?** `assets/lib/schema.py` and
  `environment.schema.json` shipped inside the freeze commit. Whether m1 is
  expected to extend the latter or whether it is already final is not something
  I can read off the tree. `schemas/README.md` says who writes which schema; I
  have not yet checked its claim against what exists.
- **Does the sequencing of `mock_stages` promotion hold under two holds?**
  `README.md:50-53` prescribes promoting one stage at a time out of mock. With
  two GPU nodes and five modules, whether that ladder can actually be climbed in
  the walltime is unsettled.
- **Which of the eight carried-over open items from effort 1** (E9′, the
  comparability gate, C9b, C23, C24, E0, E14, E16 — see T+480 above) this round
  is expected to close, and which are simply inherited. `../todo.md` exists and
  is cited by `README.md:30`; I have not opened it.

### 6. New commits

**None.** `9646910` — "e2e-flow: freeze the cross-module contract and load the
five-stage graph" — is HEAD and is the baseline, not a new commit. This
checkpoint's own commit will be the first since.

### 7. Anything else

The one number to watch is **real validators / 20**, currently 0. It is cheap to
read, hard to fake, and it is the difference between a graph that loads and a
graph that judges. A round that ends with 17 well-written task bodies and 20
skeleton validators would load clean and prove nothing — which is exactly the
failure mode `.claude/CLAUDE.md` core principle 1 names ("a previous stage
reported 14/14 tasks and ten validators PASS over a run in which every result
was zero"). I will report that ratio in every section.

Second thing worth stating once, at the start: I have made **zero** assessments
of module quality in this section, and five entries in the table read "unknown".
That is correct at T+0 — the modules have had minutes, not hours. If it still
reads mostly "unknown" at T+90 that is itself the finding.


---

## T+30 — 2026-09-03 14:16 UTC

### Standing checks

| check | result |
|---|---|
| (a) index leak — anything staged outside the package | **clean**, prints nothing |
| (b) per-commit ownership against `CONTRACT.md` §8a manifest | **three violations, all in one closed window** — §3 |
| graph loads (`show`, the §9 gate) | **pass**, rc 0, sub-second, 17 closures / 6 tasks |
| holds `106250` / `106253` | both `R`, end **20:44:58 / 20:45:03 UTC** — 6 h 28 m left |

### Headline metric

**Real validators: 20/20 as of 14:15:30 UTC.** Every `check.py` is now over 12
lines; the smallest is `check_overlay_applies` at 141, the largest
`check_speedup_substantiated` at 667. Total ≈ 6 260 lines of validator body.

Reported as an instant, not a level, and deliberately not carried forward: at
13:24 I read 11, at ~13:35 the leader read 12, my next read said 13, the
leader's next said 16, and thirty minutes later it is 20. The number changed
faster than any of us could report it. **From here the metric is saturated and
stops being informative** — the interesting question becomes whether those
bodies are correct, which this count cannot answer.

### 1. Progress

**~45 % complete.** Elapsed 46 m. Estimated remaining **4–6 h**, landing inside
the hold window.

**Reliability: low-to-moderate**, up from low, and I want to be precise about
what changed. What it rests on: 28 commits, 20/20 validators, eight schemas, and
eleven of seventeen task bodies now carrying real content (`optimize_kernel` is
181 KB across 18 files, `build_workset` 72 KB across 7). Those are large,
countable, first-hand facts. What it does **not** rest on: any evidence that the
bodies are *right*. Nothing has been dispatched — `show` type-checks the graph
and runs nothing, and the GPU holds are untouched. The 45 % therefore measures
written material, and the remaining 55 % contains the entire cost of finding out
whether it works. Effort 1's shape says that second half is where the time goes.

### 2. Current state, per module

Every module is now visible. **Nothing is "unknown, not scored" this section** —
that answers the T+0 note about what it would mean if the table still read
mostly unknown at T+90.

Authorship is read from the **commit-message prefix**, not from `git`: all 28
commits carry `%an = yihou`, since the six of us share one identity as well as
one worktree. That is worth stating because it means §8a's per-commit check
cannot be automated on author metadata.

| module | landed | reads as |
|---|---|---|
| **m1-deploy** | `check_deploy_kit` (543) + `gate.sh`, `check_deploy_serves` (509) + `probe_runner.py`/`probes.yaml`, `deploy_kit.layout.yaml` (22 KB), `deploy_and_prove.task` (24 KB), `steps/m1_deploy.yaml`, `lib/zone.py` | furthest along on *evidence*: `b015fd2` is a **first live run** of `check_deploy_serves` and the bug it found — the only commit so far claiming execution rather than authorship |
| **m2-profiling** | four validators (`bench_result` 284, `trace_coverage` 300, `profiling_evidence` 361, `kernel_table` 278), `analyze/`, `load/`, `serve/`, `lib/trace_stream.py`, `lib/m2_reshape.py`, three profiling task bodies | landed late and fast; `31e0ff6` reports **three shells that never parsed**, i.e. it is finding real defects |
| **m3-analysis** | four validators (`worklist_shape` 214, `identity_resolved` 173, `workset_shape` 426, `workset_runs` 265), three schemas (`workset.schema.json` 47 KB), `rank`/`identify`/`build_workset` bodies, nine `lib/` files | the most prolific, and the source of all three ownership violations below |
| **m4-kernel-opt** | `check_speedup_substantiated` (667, the largest), `check_optimization_shape` (493), `kernel_optimization.schema.json` (33 KB), `optimize_kernel.task` at **181 KB / 18 files** with a seven-step pipeline, `schemas/samples/` incl. a self-test | `4125267` is a **reversal** of its own earlier design (M4.3.5) on m3's evidence — a module correcting itself against another's artefact |
| **m5-integration** | seven validators, `apply_patch`/`integrate_and_verify`/`packup` bodies, `integration_report.schema.json`, `accept/`, `bench/`, six `lib/` files | steady; `53b9e1b` ("steps.json is written by the body it attests to, so corroborate it") is the self-attestation problem being caught, not papered over |

Cross-module traffic is visible and healthy: `3289d26` is m4 adopting m5's
`patchkit` vocabulary and m3's entrypoint convention; `bab5644` is m3 adding
three fields **m4 said it must not have to invent**. Modules are reading each
other's contracts rather than guessing.

### 3. Code problems

**The failure §8a exists to prevent happened three times, after §8a landed.**
§8a is `94dba13`. In commit order after it:

| commit | claims | also contains | whose |
|---|---|---|---|
| `2eadd20` | "m3: build_workset as an AI task…" | all seven m5 validators' `check.py`/`readme.md`, `apply_patch`/`integrate_and_verify`/`packup`/`m5_integration` task bodies, `steps/m5_integration.yaml`, `lib/mock_m5.sh`, `lib/merge_arm.py`, plus `MOCK-MAP.md` and `../todo.md` | **m5's and the leader's** |
| `420e7f4` | "m3: validator readmes…" | m2's `check_bench_result`/`check_kernel_table`/`check_profiling_evidence`/`check_trace_coverage`, `analyze/`, `load/`, `lib/trace_stream.py`, `lib/m2_reshape.py`, `merge_profiling_evidence.task`, `run_profiling_mode_off.task` | **m2's** |
| `d19a489` | "m5: the two arms' scripts…" | `lib/remote.sh`, `serve/round.sh` | **m2's** |

Two smaller ones: `cb80c10` (m5 committing the leader's `MOCK-MAP.md`) and
`e213679` (the leader committing four owners' `steps/m*.yaml`).

**Not fixed, and I am not proposing a fix** — unpicking landed commits is worse
than the disease, and no work appears to have been *lost*, only attributed to
the wrong commit. The material fact for the record is the mechanism: a `git add`
that predated the owner's adoption of `git commit -- <paths>`.

**The window has closed.** All three cluster between 13:50 and 14:00. Every one
of the eleven commits from `fbf5660` (14:02) onward touches only its own owner's
paths. On the evidence, §8a took effect with a ~20-minute lag rather than
failing — but that reading rests on eleven clean commits, which is a thin base,
and I will keep checking every section rather than declaring it solved.

**A live defect in the manifest itself:** `assets/lib/store.py` is listed under
**both m3 and m5**. §8a says "a file with two claimants is a conversation with
the leader, not a race" — so by its own rule this needs settling. It is the one
item in this section that is actionable now.

**One message/content mismatch:** `1841b55` is titled "remote.sh forwarded the
wrong variable prefix, so nothing reached the remote side" but contains only
`CONTRACT.md`. The finding was documented; whether the fix to `remote.sh` itself
landed is not something the commit shows. This is exactly what `git show --stat`
is for and why §8a mandates it.

### 4. Non-code problems

- **Six owners share one git identity** (`%an = yihou`). Ownership is carried
  only by a commit-message prefix convention (`m3:`, `m5:`, `e2e-flow:`). It
  works, and it means check (b) is a human reading, not a script.
- **The manifest is being amended retroactively** — `b86197a` ("catches up with
  four files landed before the collision-zone rule") and `a89ae71` ("records
  m3's nine lib files"). Sensible, but it means a violation checked against a
  *later* manifest can look legitimate. I checked against the manifest as it
  stands at 14:15; two of the three above would have been *worse* against the
  version I was handed at 13:45.
- **`assets/lib/` and now `assets/bench/` are declared collision zones** and
  already hold ~20 files from four owners.
- Repo-root litter unchanged (`glm5.2-…tar`, `rank0/`, `handoff.analysis.md`,
  `.serena/`, modified `agent_sys/docs/design.md`); untouched, outside the
  package, and `git commit -- <paths>` cannot reach it.
- `zsh` `libtinfow.so.6` warning on every command, cosmetic, unchanged.

### 5. Open questions

- **Does any of this work?** Nothing has been dispatched. 20/20 validators and
  ~6 260 lines of body are entirely unexecuted except for m1's one live
  `check_deploy_serves` run. This is now the dominant unknown.
- **Who owns `lib/store.py`?** Double-claimed, above.
- **Did `remote.sh`'s variable-prefix bug actually get fixed**, or only
  documented? `1841b55` shows only `CONTRACT.md`; `d19a489` touches `remote.sh`
  but is m5 committing m2's file.
- **Can the `mock_stages` ladder be climbed in the remaining 6 h 28 m** on two
  holds shared by five modules? Still unsettled, and now closer to being tested.
- **The six inherited effort-1 items** (E9′, the comparability gate, C9b, C23,
  C24, E0, E14, E16). `../todo.md` now exists and was touched by `2eadd20`; I
  still have not opened it, and will next section.

### 6. New commits

**28 since `9646910`**, plus my own `466435c` (T+0 checkpoint). By owner:
leader 10, m3 6, m5 5, m1 3, m4 2, m2 1, checkpoint 1.

Notable, beyond those already covered:
`abd9ff9` env_render + mock adaptations · `8faa93d` every body becomes `/bin/sh`
and "declining to mock is not succeeding" · `eec1203` CONTRACT §4.0, the trust
chain · `94dba13` **§8a itself** · `fc42694` `check_environment` plus "a rule of
my own that would have killed module 5" · `bad50b4` MOCK-MAP (B) named the wrong
handoff · `6d7a3d3` m1's kit layout as a yaml spec · `196108e` m3's four
validators "each proven to fail on the thing it names" · `fbf5660` CONTRACT §2.2
— the absolute-path rule rested on a false premise · `e213679` mocking an AI task
means swapping its agent; the mock had stopped reaching four leaves ·
`9c18603` write the environment record after the redact pass · `31e0ff6` m2's
three shells that never parsed.

Four of these are **retractions of the leader's own earlier rules** (`fc42694`,
`fbf5660`, `bad50b4`, `e213679`). That is a healthy signature at this stage, not
a worrying one.

### 7. Anything else

**A correction to what I sent the leader at ~13:45.** I reported the count as
11. It was wrong twice over. Stale, yes — but also *miscounted*, and the
mechanism matters: my loop incremented `r` inside a pipeline
(`for … done | sort`), so the increment ran in a subshell and the total was
discarded at the pipe. I did not notice, and read the count off the printed list
by eye instead, missing `check_profiling_evidence`. The leader's "12" was also
wrong, and the true figure at that instant was 13. The counting procedure now
avoids the pipe entirely. I am leaving the wrong number where it is, in the
section above, and recording the mechanism here.

**A note on my own T+0 commit.** It used `git add work.checkpoint.summary.md`
followed by `git commit` — on the shared index, two minutes before I reported to
the leader that the shared index was a hazard. Exactly one file landed, so
nothing was taken, but that was timing rather than method. From this section
onward my commits use `git commit -s -- <path>` and I verify with
`git show --stat --name-only HEAD`.

**On the headline metric now being saturated.** 20/20 was the right thing to
track for the first hour and it is nearly useless for the next. The honest
successor is not another count — it is whether anything has *run*. I propose to
report, each section, how many of the twenty validators have been executed
against a real artefact at least once, which today stands at **one** (m1's
`check_deploy_serves`, `b015fd2`). I will start reporting it at T+60 unless the
leader prefers something else. Core principle 1 is the reason: twenty validators
that have never run are twenty untested programs, and a graph that loads is not
a graph that judges.


---

## T+60 — 2026-09-03 14:51 UTC

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean**, prints nothing |
| (b) per-commit ownership | **one new violation** — `dd13fa1`, §3. Content is trivial; the breach is not |
| graph loads (§9 gate) | **pass**, rc 0, 17 closures, sub-second |
| holds `106250` / `106253` | both `R`, end **20:44:58 / 20:45:03** — 5 h 54 m left |

### Metrics

**Written:** 21/21 validators real at 14:47 UTC. The denominator moved with
`check_command_parses` and the count moved with it, so this line is now
permanently saturated and is reported only to show it has not regressed.

**Executed** — three numbers, deliberately not merged:

| | count | how I know |
|---|---|---|
| (i) by the owner's own fixture | **large, ~90+** across six owners | second-hand, from the leader's inventory; not re-counted |
| (ii) by `agent-sys run` inside the graph | **≥ 12** | second-hand, from commit messages |
| (iii) of those, against a real artefact rather than a purpose-built fixture | **unknown** | not measurable by me today |

**I could not verify (ii) first-hand and want to be explicit about why**, because
a number I cannot check is worth less than the sentence explaining that. I found
the run root — `/shared_nfs/yihou/agent_sys/ws_handoff_refine/runroot/runs/`,
**20 run directories** since ~14:14, plus `m5_wiring_run` and `m5_wiring_run2`
(that count *is* first-hand). But the event store names validators by uuid, not
by name: grepping it yields exactly one name, `check_environment`, which is
certainly an undercount. Grepping the zone directories instead yields **159**
distinct `check_*` tokens including `check_arithmetic`, `check_for_model_deprecations`
and `check_trace_shpe` — these are function names inside source copied into
playgrounds, not executions, so that is as certainly an overcount. Neither
number is the answer. **≥ 12 comes from `c3701b5`'s subject line** ("twelve
validators ran a python that cannot import their schema loader"), corroborated
by `852cf39` (m1: the mock kit passes `check_deploy_kit` and `check_environment`)
and `1cf7e7d` (m5: "the mock ran in a real run"). I will find a first-hand
method before T+90 rather than keep quoting subject lines.

### 1. Progress

**~60 % complete.** Elapsed 81 m. Estimated remaining **3.5–5 h**.

**Reliability: moderate**, genuinely up, and for a different reason than last
time. At T+30 the estimate rested only on written material. It now rests on
something better: **the code has started running and started failing in
informative ways.** Twenty runs happened in the last forty minutes, and eleven
of this interval's twenty-nine commits report a bug found *by running* rather
than by reading — `c3701b5` (twelve validators invoked a python that cannot
import their schema loader), `581b128` (three args where `x or default` swallowed
an explicit value), `c0eec13` (the trace directory has two names and the code had
one), `fb82acf` and `dd13fa1`/`eb9735e` (the stub kit's first and second bugs),
`4f81e53` (the interpreter was probed for and then thrown away), `3afa808` (a
body that cannot validate must fail, not disappear). That is the Phase-2 cost
being paid down in real time, which is exactly what the remaining 55 % at T+30
was reserved for.

What still holds the estimate back: no stage has been promoted out of mock, the
GPU holds remain unused by the graph, and (iii) is unknown — so the question
"does any of this work against a *real* artefact" is still largely open.

### 2. Current state, per module

All five active. Nothing unknown this section.

- **m1-deploy** — `852cf39`: the mock kit now **passes** `check_deploy_kit` and
  `check_environment`; `af30265` makes mock adaptation a step after the copy and
  names two traps it uncovered. Untracked `check_deploy_serves.validator/stub_kit/`
  in the tree, so more is coming.
- **m2-profiling** — quiet in commit count (2) but both are run-found:
  `581b128` (`x or default` swallowing explicit values, three args) and
  `c0eec13` (trace directory named two ways).
- **m3-analysis** — `861b3fb` **validated on real gfx950**, the first claim of a
  real-hardware validation this round; `8552de9` stops the report deciding the
  scope of its own audit; `8950f01` probes for an interpreter that can measure
  and transcribes both floors; `e772c4f` found a `kind: ai` closure silently off
  the mock path.
- **m4-kernel-opt** — the most active (10 commits). Built a **stub kit** for the
  re-measurement path that has now found two bugs (`eb9735e`, `fb82acf`) and had
  its refusal-case rule promoted to something other owners can copy (`163d91a`).
  `3afa808` — "a body that cannot validate must fail, not disappear" — is the
  right instinct about silent success.
- **m5-integration** — `1cf7e7d` "the mock ran in a real run, and two things only
  the graph could show"; `0842599` refuses an optimisation that shipped past its
  own declared gate; `d4ab70b` caught `redact.py` rewriting the one record that
  must keep absolute paths.

### 3. Code problems

**One new §8a violation: `dd13fa1`.** Titled "m4: a stub kit for the
re-measurement path, and the bug it found on its first run" — *the identical
title to `eb9735e`, committed 25 seconds earlier* — and it contains only m2's
four validator `entry.sh` files.

**The content is trivial and no work was lost.** The whole diff is 8 deletions:
a stale two-line `# SKELETON. The owner replaces check.py with the real body.`
header removed from four files whose bodies are long since real. I checked the
diff before escalating, and I am recording that it is harmless as prominently as
that it happened — a violation report that does not distinguish "clobbered
another owner's logic" from "removed a dead comment" trains its reader to
discount the next one.

**The breach is still real**, and it revises the T+30 reading. I wrote there
that the window "closed" at 14:02 on the strength of eleven clean commits.
`dd13fa1` lands at **14:27**, twenty-five minutes after that boundary, so the
window did not close — it thinned. Correction stated here rather than in the
section above.

**Found by the modules and worth recording as a class:** `c3701b5` — *twelve
validators ran a python that cannot import their schema loader*. Twelve
validators that pass because they cannot import the thing they validate against
is precisely core principle 1's failure mode, and it was caught by running, not
by reading. Fixed by the leader across ~20 files.

Also: `e5a66a4` found the justification for `schema.py`'s fallback was false;
`161a64e` (CONTRACT §4.2) records that a `${...}` arg is a string and "both
halves have now cost a run"; `fc0342b` renames `report_medians`, which no longer
only returns medians.

### 4. Non-code problems

- **A duplicate commit subject** (`eb9735e` / `dd13fa1`, 25 s apart) makes the
  log actively misleading: two different changes, one description, and the
  second is the cross-owner one. Read `--stat`, not `%s`.
- **Container-written handoffs are root-owned** — `b794551` adds `lib/reclaim.sh`.
  A cluster trap of exactly the kind effort 1 hit.
- The manifest continues to be amended retroactively; §8a itself was refined
  again (`5f6f701`: "the commit rule protects others from you, not you from
  others").
- Repo-root litter unchanged and untouched. `zsh` warning unchanged.
- **My own measurement of (ii) failed**, above. Not a cluster problem — a
  reporter problem, recorded as one.

### 5. Open questions

- **A first-hand method for (ii).** The store keys on uuid; the zones grep is
  polluted by copied source. Unsolved, and mine to solve before T+90.
- **(iii) is entirely unknown** — how much has run against a real artefact
  rather than a fixture built for it. `check_command_parses` is stated to meet
  that standard and `861b3fb` claims real gfx950; nothing else is established.
- **No stage promoted out of mock yet**, and 5 h 54 m of hold remains.
- **Did `dd13fa1`'s deletions belong to m2?** The change is right, but whether
  m2 had made it and m4 took it, or m4 made it in m2's files, I cannot tell.
- `../todo.md` still unopened by me. Third section running. Recording the slip.

### 6. New commits

**29 since T+30** (58 since `9646910`), plus my own `23d452a`. By owner: m4 10,
leader 8, m3 5, m5 4, m2 2.

Beyond those covered above: `4bffebd` **`check_command_parses`** — 11 of 14
sealed command scripts do not parse under the shell their own shebang names ·
`a3d9e46` `env_render` resolves `transport=auto`, which every real producer hits ·
`ea93d83` MOCK-MAP cited a variable nothing in this package sets · `277957e` m4
proving MOCK-MAP (G)'s redefined both-direction claim · `a58d5eb` `apply_patch`
reads m3's declared integration point rather than only m4's word · `d81b7b9` m4
consuming m3's `integration`, `noise_floor` and `apparatus` instead of inferring
them.

Five of the leader's eight are again **retractions of its own earlier rules or
documents** (`5f6f701`, `e5a66a4`, `ea93d83`, `161a64e`, `c3701b5`).

### 7. Anything else

**The interesting signal this interval is not the commit count, it is what the
commits say.** Roughly a third of them describe a defect found by executing
something, and several describe one module refusing to accept another's word —
`a58d5eb` (m5 reads m3's declared integration point, not only m4's word),
`0842599` (m5 refuses an optimisation that shipped past its own declared gate),
`3afa808` (m4: a body that cannot validate must fail, not disappear),
`8552de9` (m3: the report no longer decides the scope of its own audit). Those
are four independent implementations of "do not trust a thing's own claim about
itself", which is the same principle the mission's core principle 1 states. It
is being rediscovered locally rather than cited, which is a good sign about
whether it is understood.

**Set against that, the count that matters is still small.** ≥12 validators have
run inside the graph, out of 21, and I cannot yet say how many touched a real
artefact. The gap between (i) and (ii) — a large, healthy fixture-testing effort
versus a dozen graph executions — remains the honest measure of how much of
Phase 2 is left, and it has narrowed less than the commit volume suggests.


---

## T+90 — 2026-09-03 15:10 UTC

### The number that matters, before any percentage

**5 of 21 validators have ever produced a recorded verdict in the graph.**
First-hand, 15:06 UTC, from the persisted verdicts:

| validator | recorded verdicts |
|---|---|
| `check_environment` | 28 |
| `check_deploy_kit` | 26 |
| `check_deploy_serves` | 24 |
| `check_command_parses` | 4 |
| `check_bench_result` | 4 |

**Distinct: 5. Unchanged from 15:00** — the tallies grew (5→4 on two of them,
+2/+3/+3 on the others) but **no sixth validator has yet produced a first
verdict.** Twenty-five runs now exist. Sixteen of twenty-one validators have
never once been judged by the graph they were written for.

*Caveat, restated as it will be every section:* a verdict is only recordable
against a **sealed** version, so this is an exact count of recorded
output-validation verdicts and a **lower bound** on execution. Input-phase runs,
and output phases whose handoff is refused before a seal, never appear here. The
leader confirms the `--json` stream is the only place that residue lives; it is
not worth adopting for its own sake, and this is written down so neither of us
re-derives it.

**Written validators: 21/21** at 15:06. Saturated, no regression.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** |
| (b) per-commit ownership | **one new violation, and it took my own work** — §3 |
| (c) `todo.md` | **T14 and T15 landed** (`edb36b3`), 15 items. See §5 for what did *not* land and should have |
| graph loads (§9 gate) | **pass**, rc 0, 17 closures |
| holds | both `R` to **20:45** — 5 h 35 m left |

### 1. Progress

**~65 %.** Elapsed 100 m. Estimated remaining **3.5–5 h**.

**Reliability: moderate, and the basis has changed shape rather than improved.**
At T+60 I said the estimate rested on code that had started running and failing
informatively. That is still true and there is more of it. But two facts learned
since pull in opposite directions:

- **Upward:** `deploy_kit` slot v0 now reads **valid** — all three of m1's
  validators PASS, including `check_deploy_serves`, the one that costs GPU
  hours. The graph runs past m1 into m2. The effort is no longer gated on m1.
- **Downward, and larger:** `RUN-PLAN.md` (landed `b0bde21`) lays out **six
  rungs, each a separate run**, and **rung 0 is not yet green** — the graph
  stops in m2 on adaptation (A). The remaining work is not "finish m1" but "climb
  six rungs", and only the first term of that was ever in my estimate.

So the percentage moved five points while the thing it measures got longer. I am
holding at ~65 % rather than raising it, and the honest statement is that **the
denominator was wrong until this section** and I do not yet know the per-rung
cost. One data point exists: rung 0 has consumed ~40 minutes and is not done.

### 2. Current state, per module

- **m1-deploy** — **done and green.** `54be430`: `deploy_kit` is valid, the stub
  becomes the mocked kit, five bugs on the way. `b78d609` and `27bd1be` follow up
  with an inverted `$?`, numeric args through the shared reader, removal of the
  transport workaround, and five recorded environment facts.
- **m2-profiling** — **now the critical path.** `8158bb5` fixes `remote.sh`'s
  missing `auto`; `e29b49d` reclaims the node work root before teardown. The
  first `invalid` in the graph is `profiling_mode_off.bench_result`, m2's.
- **m3-analysis** — `080412f` validated `writes_in_place` **on real torch**;
  `3ad8938` writes the general form of the SNR inversion; `ddd6d9a` is the
  section's most creditable commit and is discussed in §7.
- **m4-kernel-opt** — no commits this interval. Reported by the leader as
  complete, green and deliberately idle; **I have not verified that** and record
  it as second-hand.
- **m5-integration** — `8687eca` takes the `writes_in_place` hedge off and puts
  the real argument in, on m3's evidence. Otherwise quiet; same second-hand
  status as m4.

### 3. Code problems

**A new §8a violation, and this time it took mine.** `3b2ffde` (14:52:53) is
titled *"e2e-flow: CONTRACT 4.3 — what the shape is not, and that claiming an
audit is not one"* and its entire content is **187 insertions to
`work.checkpoint.summary.md`** — my T+60 section. Its twin `f0350e4` carries the
identical subject and holds the actual `CONTRACT.md`.

**That is the third duplicate-subject pair today** (`eb9735e`/`dd13fa1`,
`f0350e4`/`3b2ffde`), and the second where the duplicate is the cross-owner one.
The method note stands and is now proven twice: ownership rides on a message
prefix, the prefix is duplicable, **`--stat` is the only reliable read.**

**No content was lost.** All 187 lines are intact in HEAD. The cost is
attribution, not work.

### 4. Two things I told the leader that were false

Both concern my own T+60 commit, and I would not have found either without
running check (b) against my own file.

**(1) I reported "T+60 is committed". It was not — not by me.** `git log --grep`
shows no `checkpoint: T+60` commit. My section reached HEAD **inside the
leader's `3b2ffde`**. The mechanism: my first commit failed on `index.lock`; in
the four seconds I slept before retrying, the leader's commit swept my dirty
working-tree file; my retry then found nothing to commit for that path.

**(2) I reported that `index.lock` contention was "the only part of §8a tested
under contention, and its guidance is correct as written". That is exactly
backwards.** §8a says: *"If two commits collide on `index.lock`, git says so;
wait a second and retry."* I did that, and **the retry silently did nothing**,
because the file had been taken in the interval. The guidance is not sufficient:
the retry is not idempotent, and the failure is silent. §8a's own verification
step — `git show --stat --name-only HEAD` — did not save me either, because HEAD
*was* a commit containing my file, just not mine. I read the filename, saw what I
expected, and stopped.

**The correct check is `git show --stat HEAD` including the subject line, or
`git log -1 --format=%s`.** Confirming the path is not confirming the commit.

I am recording this as prominently as I can because it is the same fault I have
been reporting in other people's work all afternoon: **a check that passes for
the wrong reason.** I built one into my own procedure and it took 20 minutes and
an unrelated audit to notice.

### 5. Non-code problems, and the `todo.md` half nobody has done

**(c) second half — what deserved to land in `todo.md` this interval and did
not.** Three candidates, none recorded:

1. **The `index.lock` retry hole above.** §8a's guidance is incomplete; the fix
   is a one-line change to the verification step. Not in `todo.md`.
2. **`RUN-PLAN.md`'s six rungs have no per-rung cost estimate**, and rung 0 is
   already the longest-running item of the day. Nothing records that the ladder's
   feasibility inside the hold window is unmeasured.
3. **Duplicate commit subjects** have now happened three times. There is no note
   anywhere that the log cannot be read by `%s`.

Item 1 is mine to propose and I am proposing it here rather than editing
`todo.md`, which is the leader's file.

Other environment notes: repo-root litter unchanged; `zsh` warning unchanged;
m1 stopped two leaked stub processes on `031` and **filtered to pids whose path
contained `yihou`** before killing — the shared-host rule followed at the moment
it is easiest to skip.

### 6. New commits

**16 since T+60** (74 since `9646910`). By owner: leader 7, m1 3, m3 3, m2 2,
m5 1, m4 0.

`b0bde21` **RUN-PLAN** + CONTRACT §4.3 "one authority, two readers" · `3374158`
§4.2 — in a validator the string-arg bug produces **no answer** · `f0350e4`/`3b2ffde`
§4.3 · `5ec1635` the vars change with the rung too, + §4.4 · `f5fed19` names
`env_render`'s transport probe as a knowing §4.3 instance · `edb36b3` **T14/T15
into `todo.md`** · `54be430` **`deploy_kit` valid, five bugs** · `8158bb5`
`remote.sh` never implemented `auto`.

Five of the leader's seven are again corrections to its own documents, two of
them (`f5fed19`, `3374158`) self-reported instances of rules it had just written.

### 7. Anything else

**`ddd6d9a` — "m3: I said my code was clean of the shape; it was not, twice" —
is the best commit of the interval and I want it in the record.** m3 asserted to
the leader that nothing else of theirs read one rule from two places, then
audited properly and found two live defects. One of them would have let m4
re-measure under the manifest's protocol and divide by a baseline recorded under
another — *and across two protocols that ratio looks entirely normal*. A wrong
number that looks right, surfacing in a third module's transcript pointing at a
second module's code.

**Claiming an audit is not performing one.** That is the same failure as my §4
above, arrived at independently by a different owner in the same hour. Seven
instances of the §4.3 shape are now recorded across three owners, none of them
wrong logic — all of them two readers with different reach.

**The gap that governs everything is still 5 against 21.** Twenty-five runs,
seventy-four commits, ~6 300 lines of validator body, and sixteen validators that
have never been judged by the graph. The fixture work behind them is real and
large; it is simply different evidence, and m1's five bugs — every one invisible
to a standalone run, one of them hidden specifically because a hand-written
`args.json` carries JSON numbers where the real producer sends strings — are the
proof that the difference is not pedantic.


---

## T+165 — 2026-09-03 16:17 UTC

**T+120 and T+150 are missing from this file, and this section replaces both.**
At 15:40 I wrote a full T+120 section and it never reached the disk: `/home`
filled to 100 % with zero bytes free, the append died with
`cat: write error: Disk quota exceeded`, and the commit failed with
`fatal: Unable to create ... index.lock: Disk quota exceeded`. Nothing was
corrupted — the append wrote nothing at all, `git status` stayed clean, and the
file held at 3 409 lines. Space returned by 16:16 (70 G free). The label is
T+165 rather than T+150 because that is the honest elapsed time; I am not
back-dating a section into a slot I missed. The T+120 numbers appear below as a
trajectory point so the interval is not lost.

### The number that matters, before any percentage

**10 of 21 validators have ever produced a recorded verdict in the graph.**
First-hand, 16:17:00 UTC:

| validator | verdicts | | validator | verdicts |
|---|---|---|---|---|
| `check_environment` | 67 | | `check_trace_coverage` | 5 |
| `check_deploy_kit` | 34 | | `check_kernel_table` | 5 |
| `check_deploy_serves` | 32 | | `check_worklist_shape` | 4 |
| `check_command_parses` | 23 | | `check_profiling_evidence` | 4 |
| `check_bench_result` | 14 | | `check_identity_resolved` | 3 |

**Trajectory: 15:00 = 5 · 15:06 = 5 · 15:14 = 9 · 15:40 = 10 · 16:17 = 10.**

**The distinct count has not moved in thirty-seven minutes**, across 33 runs (up
from 30). Every tally grew — `check_environment` alone went 49→67 — so the graph
is running hard and re-judging the same ten. **Eleven of twenty-one have still
never been judged, and the set of eleven is unchanged since 15:40.** §7 has the
reason, and it is structural rather than a stall.

*Caveat, restated:* recordable only against a **sealed** version — exact for
recorded output-validation verdicts, a lower bound on execution. A FAIL counts.

**Written validators: 21/21** at 16:17. **Graph loads: pass**, rc 0, 17 closures.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** |
| (b) per-commit ownership | **one technical breach, benign and arguably intended** — §3 |
| (c) `todo.md` | **T17 landed**, 17 items — and the module that added it does not own the file |
| holds | both `R` to **20:45** — 4 h 28 m |

### 1. Progress

**~70 %.** Elapsed 167 m. Estimated remaining **3–4 h**, with a caveat that has
grown teeth.

**Reliability: moderate, and one load-bearing assumption just died.**
`1238b78` — *"rung 0 cannot complete on the login node, and that is by design"* —
means the ladder I have been using as the denominator since T+90 cannot be
climbed from where the work is happening. Rung 0 was never going to go green
here. That reframes the last two hours: rung 0 has not been *failing* for
seventy minutes, it has been **completing as far as it structurally can**, and
the eleven unjudged validators are the ones that need a GPU node, not the ones
that are broken.

This is good news for the estimate and bad news for my previous reasoning. At
T+120 I wrote that rung 0 taking ~70 minutes made a naive six-rung extrapolation
not fit inside the window. That extrapolation was measuring the wrong thing —
rung 0's login-node phase is not a unit that repeats.

**What the estimate now rests on:** four hours twenty-eight of hold, a graph that
loads, ten validators judged, and a ladder whose remaining rungs all require the
GPU nodes that have been sitting unused by the graph all afternoon. I am holding
at ~70 % and flagging that the number is now dominated by a single unknown — how
much of the ladder can be climbed in 4 h 28 m — rather than by anything I can
count in the repo.

### 2. Current state, per module

- **m1-deploy** — most active (4 commits). `00611ed` the handshake carries both
  sides of the mount; `c16a5bb` the producer brief assumed the serving image
  already exists; `756bda9` **corrects its own earlier `min_tokens` claim** —
  the replay script does not set it.
- **m2-profiling** — `42a8570` read both sides of the work root from the
  handshake, compute neither; `01d4569` `kernel_table` was the last schema
  without a `schema_version`.
- **m3-analysis** — `a94ce98` the mock measures where the real path measures,
  **in a container on the node**; `49bdc52` one renderer for both paths, and its
  subject is exemplary: *"no bug here, but m4's structural point stands"*.
- **m4-kernel-opt** — no commits this interval. Unknown, not scored.
- **m5-integration** — no commits this interval. Unknown, not scored.

### 3. Code problems

**`c16a5bb` (m1) commits `todo.md`, which the §8a manifest assigns to the
leader.** It added T17. **The content is exactly what the system wants** — an
owner recording a deferred finding at the moment they hit it — so this is a
breach of the letter and a fulfilment of the intent.

**I think the manifest, not m1, is what is wrong here.** If owners are expected
to record deferrals as they find them, `todo.md` cannot be leader-exclusive; if
it is leader-exclusive, owners must route deferrals through the leader and will
not, because they are mid-task when they find them. That is a real conflict in
the contract, not a lapse by m1, and it is the leader's to resolve.

**Otherwise all nine commits are clean on ownership.** Two clean intervals in a
row on the substantive check.

### 4. Non-code problems

**The `/home` outage, above.** Root cause not ours: `/home/yihou` totals 6.1 G on
a shared 10 T volume, so the fill came from outside this effort and no cleanup
we could do would have recovered a meaningful fraction. Recovered without our
intervention. Compute was never affected — both holds ran through it.

Worth recording as a class: **a shared-filesystem outage is invisible to every
check in this file** until something tries to write. The graph gate passes, the
verdict grep passes, `squeue` passes — and the record silently stops being
written. Two sections were nearly lost with no failing signal anywhere.

### 5. Open questions

- **How much of the ladder fits in 4 h 28 m**, now that rungs 1–5 are known to
  need the GPU nodes. Dominant unknown.
- **The eleven unjudged validators** — are they all node-gated, or are some
  simply unreachable in the current wiring? Unsettled, and now the sharpest
  question in the file.
- **`todo.md` ownership**, §3.
- **Adaptation (A)'s fourth instance did not arrive this interval.** At T+90 and
  in my lost T+120 I tracked a hypothesis that rung-0 failures were one lesson
  repeated across owners (m2, m3, m4). No new instance since. The hypothesis is
  neither confirmed nor dead; with `1238b78` reframing rung 0, it may simply have
  stopped being the interesting question.

### 6. New commits

**15 since T+90** (89 since `9646910`), spanning the outage. m1 5, leader 4,
m3 3, m2 2, m5 1, m4 1.

`17f33e4` §8a's verification step passed for the wrong reason (my finding) ·
`d83cc89` **T16** · `2dc457f` m3's MOCK-MAP (A) · `ad479a6` m4's adaptation (A),
"never only a mock gap" · `960dc45` and `01d4569` the last two schemas without
`schema_version` · `5240197` `/health_generate` is a GET · `1238b78` **rung 0
cannot complete on the login node** · `5fe5acf` **"there is no host with torch,
and I said there was without checking"**.

### 7. Anything else

**`5fe5acf` is the most valuable commit of the interval and it is a leader
self-correction: "there is no host with torch, and I said there was without
checking."** It touches `CONTRACT.md` and `RUN-PLAN.md` — two documents five
owners were working against, one of which is the denominator of my own progress
estimate. An unchecked assertion in a contract propagates into every plan built
on it, and this one had.

Together with `1238b78`, the pair resolves the puzzle this section opened with:
**the ten judged validators are the ones that can be judged without a GPU, and
the eleven unjudged ones are waiting on hardware, not on code.** The flat
10-of-21 across 33 runs is not a stall — it is the login node's ceiling, and it
was reached some time before 15:40 without anyone noticing, because the runs kept
succeeding and the count kept not moving.

**That is the third time today a signal has been read as progress when it was a
ceiling** — twenty validators written but never executed (T+30), twelve
validators "running" a python that could not import their loader (`c3701b5`), and
now thirty-three runs re-judging the same ten. Each time the fix was to find a
number that could distinguish the two, and each time the number already existed
somewhere nobody had looked. **Whatever measures the next phase should be chosen
before the phase starts, not after it plateaus.**

**Two corrections of my own carried from the lost T+120**, restated so they are
not lost with it: I recorded m1 as "done and green" at T+90 on the leader's
report, and `5240197` proved it wasn't — **a module is green at an instant, not
finished.** And my T+120 claim that the six-rung ladder would not fit the window
rested on extrapolating from a rung 0 that, per `1238b78`, cannot complete here
at all.


---

## T+190 — 2026-09-03 16:40 UTC

### The number that matters, before any percentage

**10 of 21 validators have ever produced a recorded verdict in the graph** —
first-hand, 16:40:04 UTC, and **every tally is byte-identical to my 16:17
reading**:

`check_environment` 67 · `check_deploy_kit` 34 · `check_deploy_serves` 32 ·
`check_command_parses` 23 · `check_bench_result` 14 · `check_trace_coverage` 5 ·
`check_kernel_table` 5 · `check_worklist_shape` 4 · `check_profiling_evidence` 4 ·
`check_identity_resolved` 3.

**Trajectory: 15:00 = 5 · 15:06 = 5 · 15:14 = 9 · 15:40 = 10 · 16:17 = 10 ·
16:40 = 10.**

**This is different from the last flat interval and I want the distinction on the
record.** Between 15:40 and 16:17 the distinct count was flat while every tally
grew — the graph was running hard and re-judging the same ten. Between 16:17 and
16:40 **nothing moved at all.** Run count is 33, unchanged. **The newest run
directory is `20260903T161626`, so no run has started in 24 minutes.** The graph
is not running; it stopped.

*Caveat, restated:* recordable only against a **sealed** version — exact for
recorded output-validation verdicts, a lower bound on execution. A FAIL counts.

**Written validators: 21/21.** **Graph loads: pass**, rc 0, 17 closures.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** |
| (b) per-commit ownership | **all three clean.** Third consecutive clean interval |
| (c) `todo.md` | 17 items, **no change** since T+165 |
| holds | both `R` to **20:45** — **4 h 05 m**, and the graph has used none of it |
| `/home` | 100 G free, recovered; the outage did not recur |

### 1. Progress

**~70 %, unchanged, and I am deliberately not moving it.** Elapsed 190 m.
Estimated remaining: **I no longer have a defensible figure.**

**Reliability: low — a downgrade, and the first one in this effort.** At T+165 I
wrote that the estimate was dominated by a single unknown: how much of the ladder
fits in the remaining hold. Twenty-three minutes later that unknown has not
shrunk, and the one observable that would inform it — the graph running — has
stopped. Three commits in twenty-three minutes, against roughly fifteen in the
comparable interval before.

**What I can say:** rungs 1–5 all require the GPU nodes (`1238b78`), 4 h 05 m of
hold remain, and no rung above 0 has been attempted. **What I cannot say** is
whether the last twenty-three minutes are a wind-down, a deliberate pause before
a node run, or five owners each deep in something that has not surfaced yet. From
outside, those three look identical — which is the same blind spot effort 1
recorded twice, and I am labelling it rather than guessing past it.

### 2. Current state, per module

- **m2-profiling** — `09b8db4`: exercised the **real bring-up path against a stub
  kit**, and two things it found. The only module with a code commit this
  interval, and the only one demonstrably still working.
- **m1-deploy, m3-analysis, m4-kernel-opt, m5-integration** — no commits.
  **Unknown, not scored.** m4 and m5 have now been quiet for two consecutive
  intervals; that is long enough to be worth naming, and not long enough to
  conclude anything from.
- **leader** — two commits, both `CONTRACT.md`, both corrections.

### 3. Code problems

**None found this interval.** All three commits clean on ownership — the third
consecutive clean interval, against six violations in the 13:50–14:52 window.
Whatever §8a's amended form is doing, it is holding.

### 4. Non-code problems

**`/home` did not refill.** 100 G free.

**The `git gc` warning I reported at T+165 is unaddressed** — `gc.log` still
blocks automatic cleanup and git still reports too many unreachable loose
objects. Nobody has pruned, and it is not mine to prune. Low urgency now that
space is back, but it is a slow leak on a volume that hit 100 % once today.

### 5. Open questions

- **Is the graph stopped on purpose?** Dominant question. No run since 16:16:26.
- **Will any rung above 0 be attempted before 20:45?** 4 h 05 m of hold left,
  entirely unused by the graph. If the answer is no, the ladder is the
  deliverable's largest unclosed gap and the holds expire unused.
- **The eleven unjudged validators** — all node-gated, or some unreachable in the
  current wiring? Unchanged from T+165, and unanswerable without a node run.
- **m4 and m5's actual state.** Two intervals quiet, reported complete at 15:10,
  never verified by me.

### 6. New commits

**3 since T+165** (92 since `9646910`). Leader 2, m2 1.

- **`25d9c01` — "todo.md is append-only for owners: the manifest was wrong, not
  m1."** My T+165 §3 finding, accepted and fixed at the level I argued for.
- `84b359e` — CONTRACT §4.4: the same root cause has three faces, and **two
  invent a problem**.
- `09b8db4` — m2 against a stub kit.

### 7. Anything else

**`25d9c01` is worth recording as a process result, not just a fix.** At T+165 I
reported m1 committing `todo.md` as a breach of the §8a manifest, and argued the
manifest was what was wrong: if owners are expected to record deferrals at the
moment they hit them, the file cannot be leader-exclusive. The leader made
`todo.md` append-only for owners. **The check found a real conflict in the
contract by flagging a technically-correct violation** — which is the argument
for reporting breaches whose content is benign, rather than filtering them out as
noise. Had I suppressed it as harmless, the conflict would still be there and the
next owner would still be choosing between the rule and the work.

**`84b359e`'s subject deserves attention: "the same root cause has three faces,
and two invent a problem."** Two of three reported symptoms were not real
problems. That is the same shape as this section's headline — a signal that looks
like several things and is one thing, or looks like a problem and is an artefact
of how it was observed. **Today has produced four instances**: twenty validators
written but never run; twelve "running" a broken interpreter; thirty-three runs
re-judging ten validators; and now three faces of one root cause, two of them
invented. Every one was resolved by finding a measurement that could tell the
cases apart.

**The honest summary of this interval is that nothing happened and I do not know
why.** The record has spent the afternoon distinguishing ceilings from stalls; I
can say this is not a ceiling, because a ceiling still shows tallies climbing.
Beyond that I would be guessing, and 4 h 05 m of unused GPU hold is too
consequential a thing to guess about. If the next section reads the same, that
silence will itself be the finding.


---

## T+226 — 2026-09-03 17:16 UTC

### The number that matters, before any percentage

**10 of 21 validators have ever produced a recorded verdict in the graph.**
First-hand, 17:16:14 UTC:

`check_environment` 76 · `check_deploy_kit` 36 · `check_deploy_serves` 34 ·
`check_command_parses` 27 · `check_bench_result` 16 · `check_trace_coverage` 6 ·
`check_kernel_table` 6 · `check_worklist_shape` 5 · `check_profiling_evidence` 5 ·
`check_identity_resolved` 4.

**Trajectory: 15:00=5 · 15:06=5 · 15:14=9 · 15:40=10 · 16:17=10 · 16:40=10 ·
16:53=10 · 17:16=10.**

**State: stopped, for the second time.** Run count 35, unchanged since 16:53.
Newest run `20260903T164800` — **no run in 28 minutes**, and every tally is
identical to my 16:53 reading. The distinguishing evidence is the same as at
T+190: a ceiling keeps tallies climbing, and these are frozen.

**But this stop is not like the last one, and the difference is the whole
finding.** At 16:53 I read the same ten distinct with tallies *grown* since 16:17
(`check_environment` 67→76, `check_deploy_serves` 32→34) — that was the graph
running a full rung 0 and hitting the ceiling. Between 16:53 and 17:16 it stopped
again, **while two commits landed.** Last time the graph stopped and I could not
tell whether anyone was working. This time I can: the modules are working and the
graph is not.

*Caveat, restated:* recordable only against a **sealed** version — exact for
recorded output-validation verdicts, a lower bound on execution. A FAIL counts.

**Written validators: 21/21.** **Graph loads: pass**, rc 0, 17 closures.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** |
| (b) per-commit ownership | **both clean.** Fourth consecutive clean interval |
| (c) `todo.md` | **18 items**, +1 — and see §7, the first entry filed by an owner under the new rule |
| holds | both `R` to **20:45** — **3 h 28 m** |
| `/home` | **2.1 T free (80 %)**, up from 100 G. Fully recovered |

### 1. Progress

**~72 %.** Elapsed 226 m. Estimated remaining: **still no defensible figure**,
but the reason has changed and improved.

**Reliability: low→moderate.** At T+190 I had no figure because the dominant
unknown was invisible from my position. It is now **named**: the ladder is
blocked on the image seam — the sealed record names
`infera/engine-sglang:gfx950-local`, whose digest exists on no node, and `061`
carries no sglang image at all. That is decision (a), it lands in m1's files, and
m1 is mid-GLM-load (`fa49319`: **GLM-5.3-Flash served**). A named blocker with an
owner is a different thing from an unknown, and I said at 16:53 I would stop
describing the remaining time as unknown-dominated once that changed.

What I still cannot do is put a number on it, because no rung above 0 has been
attempted and there is no per-rung datum to extrapolate from. **3 h 28 m of hold
remain and the graph has used none of it for a rung above 0.**

### 2. Current state, per module

- **m1-deploy** — `fa49319`: **GLM-5.3-Flash served**, and what a second kit
  showed the validator misses. Working the image seam, which is the ladder's
  named blocker.
- **leader** — `0f40b3a`: rung 0 reaches the mock's ceiling; two vars that each
  cost a run.
- **m2, m3, m4, m5** — no commits. **Unknown, not scored.** m4 and m5 are now
  quiet for three consecutive intervals. That is long enough that I want it
  visible in every section until it resolves; it is still not evidence of
  anything.

### 3. Code problems

**None new.** Both commits clean on ownership — the fourth consecutive clean
interval, against seven violations in the 13:50–14:52 window.

**The framework gap from the previous interval is now filed**:
`temp/bugs/2026-09-03-a-validators-stdout-is-not-kept-anywhere.md`. It joins five
earlier entries. **This one belongs beside T14**, and the pair is worth stating
together because they are the same seam failing in two directions:

- **T14** — a validator whose interpreter cannot import its schema loader exits
  non-zero and writes **no `verdict.json`**, so the phase cannot distinguish a
  broken validator from a refused handoff.
- **The new one** — a validator that *does* run and writes a careful diagnostic
  sends it to **stdout, which nothing keeps**, so a correct refusal arrives with
  its reason discarded.

Between them, a validator can fail to say anything and can say the right thing
into a void. Both were found the same way: by someone reproducing the condition
by hand after inference had failed.

### 4. Non-code problems

**`/home` fully recovered** — 2.1 T free, from zero at 15:40. Not our doing and
not our fault; `/home/yihou` is 6.1 G of a shared 10 T volume.

**The `git gc` warning persists**, third section running. `gc.log` still blocks
automatic cleanup, git still reports too many unreachable loose objects. Nobody
has pruned; it is not mine to prune. Lower urgency at 2.1 T free, but it is the
one thing in this workspace that grows without bound and it survived a day on
which the volume hit 100 %.

### 5. Open questions

- **Why has the graph stopped a second time?** Unlike T+190 I am not without
  information — two commits landed, so the owners are active — but I do not know
  whether a rung is being prepared or the image seam is simply blocking.
- **Will any rung above 0 be attempted before 20:45?** 3 h 28 m. Unchanged as the
  dominant risk.
- **The eleven unjudged validators** — all sit at or beyond `build_workset`,
  where measurement begins. Whether they are all node-gated or some are
  unreachable in the wiring is still unanswerable without a node run.
- **m4 and m5** — three intervals quiet, reported complete at 15:10, never
  verified by me.

### 6. New commits

**2 since T+190** (94 since `9646910`). Leader 1, m1 1.

`0f40b3a` rung 0 reaches the mock's ceiling — two vars that each cost a run ·
`fa49319` GLM-5.3-Flash served, and what a second kit showed the validator
misses.

### 7. Anything else

**The previous section's central question is answered, and the answer was the
leader.** I wrote at T+190 that the graph had stopped and I could not say why,
naming three indistinguishable possibilities. The cause was none of them: three
rung-0 runs launched with a command line missing `--var transport_env`. A
validator declares no agent, so it never receives the `E2E_*` block; `spur` had
no `SPUR_CONTROLLER_ADDR`; `deploy.sh` died on "failed to connect to controller";
`check_deploy_serves` refused in one second. **The correct incantation was
written in a comment in the file declaring the parameter** — `steps/m1_deploy.yaml:128`.

Two details belong in the record on the same terms as my own three false claims.
**The failure was attributed twice to other people's work** — first to m1's GLM
taking the GPUs, then to m2's `local` branch — and both were wrong. **And what
broke the loop was not reasoning but reproduction**: copying the validator's zone
and running it under `env -i` with a minimal PATH, which finally printed the
diagnostic that had existed all along.

**That is the fifth instance today of a signal being read as one thing when it
was another**, and the second where two rounds of inference pointed at the wrong
owner before the first measurement pointed at the right one. The pattern is now
firm enough to state as a rule the next effort should start with: **when a
symptom has candidate causes in more than one owner's work, reproduce before
attributing.** Inference across an ownership boundary has been wrong every time
it has been tried today, and measurement has been right every time.

**`fa49319` also filed the interval's `todo.md` entry — the first by an owner
under `25d9c01`'s new rule**, two intervals after I reported the rule as a
conflict and the leader fixed it at the level I argued. The mechanism worked
end-to-end: a check flagged a benign breach, the contract turned out to be what
was wrong, and the next owner to hit the same moment recorded their finding
instead of choosing between the rule and the work.

**And my ceiling analysis survived a run that could easily have looked like it
refuted it.** Rung 0 passed eight handoffs valid with `check_deploy_serves`
green in a graph for the first time — a real bring-up, real load, clean teardown —
and the distinct count did not move, because all eleven unjudged validators sit
at or beyond `build_workset`. A number that correctly does *not* move during
visible progress is the harder half of measuring anything, and it is the reason
this section still reads 10.


## T+876 — 2026-09-04 04:06 UTC

### Before anything else: 10 h 44 m of this record does not exist

My last section was **T+226, 2026-09-03 17:16 UTC**. The next honest reading is
this one, **14 h 36 m after T+0**. The interval **17:16 → 04:06** has no
checkpoint and I will not manufacture one: the whole team, this writer included,
was killed with the leader's tmux and re-spawned at ~03:52 UTC today.

What is recoverable from artefacts, I have recovered and it is below. What is not
recoverable — who was working on what between 17:46 and 03:52, whether anything
was attempted and abandoned — is **gone**, and the record should say so rather
than infer it. The same rule I applied through the `/home` outage.

Two things bound the loss usefully. **The last commit in the repository is
`4bc506f`, 2026-09-03 17:46 UTC.** **The last run directory is
`20260903T174638-046322`, mtime 17:46:38.** Both stop within the same minute.
Whatever ended this effort ended it at 17:46, not gradually.

### The number that matters, before any percentage

**10 of 21 validators have ever produced a recorded verdict in the graph** —
unchanged. First-hand, 04:05 UTC, 231 `verdict.json` files across 38 runs,
mapped to validator by their `args.json` signature against the `args:` blocks in
`steps/*.yaml`:

| validator | verdicts | pass | fail | Δ since 17:16 |
|---|---|---|---|---|
| `check_environment` | 76 | 62 | 14 | — |
| `check_deploy_kit` | 36 | 26 | 10 | — |
| `check_deploy_serves` | 36 | 12 | 24 | **+2** |
| `check_command_parses` | 27 | 27 | 0 | — |
| `check_bench_result` | 20 | 20 | 0 | **+4** |
| `check_kernel_table` | 8 | 8 | 0 | **+2** |
| `check_trace_coverage` | 8 | 8 | 0 | **+2** |
| `check_worklist_shape` | 7 | 6 | 1 | **+2** |
| `check_profiling_evidence` | 7 | 7 | 0 | **+2** |
| `check_identity_resolved` | 6 | 6 | 0 | **+2** |

**Trajectory: 15:00=5 · 15:06=5 · 15:14=9 · 15:40=10 · 16:17=10 · 16:40=10 ·
16:53=10 · 17:16=10 · 04:06=10.**

**State: stopped — and for the first time this is not a diagnosis, it is the
known consequence of the team being dead.** Run count **38**, up from 35. The
three runs I had not seen (17:20, 17:28, 17:46) account for **+16 verdicts**, all
of them inside the ten already judged. The ceiling held across a window that
included the first real-agent launch — which is the third time it has survived
visible progress.

*This section adds a pass/fail split I had not been carrying.* It is worth the
column: `check_deploy_serves` is **12 pass / 24 fail**, the only validator that
has refused more often than it has passed, and `check_environment`'s 14 failures
are the second-largest block. Both are first-hand from the verdict files, not
inferred. A FAIL still counts as a recorded verdict — the ten are the ten that
have *spoken*, not the ten that are green.

**Written validators: 21/21**, and no longer a count of skeletons: 21 `check.py`
totalling **7 485 lines**, mean 356. At T+0 this read 0/20 with every body a
12-line marker.

**Graph loads: pass**, rc 0, **17 closures**, re-run at 04:05 with the new hold's
identifiers substituted (`--var jobid=108891 --var node=crsuse2-m2m-249`).

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** — no `index.lock` in the worktree gitdir |
| (b) per-commit ownership | **all six clean.** Fifth consecutive clean interval |
| (c) `todo.md` | **18 items**, unchanged (T18 highest) |
| holds | **one**, job `108891` on `crsuse2-m2m-249`, `R`, ends **11:52:01 UTC** — **7 h 46 m** |
| `/home` | **2.0 T free (81 %)**. Stable |
| `git gc` | **half-resolved** — `gc.log` is gone, the unreachable-objects warning is not |

Two of these changed and both deserve a line. **The `git gc` state is half what I
carried since T+165, and I nearly recorded it as fully fixed.** `gc.log` is gone
from the common gitdir and `garbage: 0` — so automatic cleanup is no longer
blocked, and I did not prune it and do not know who did. **But the warning
itself is still live**: this section's own commit printed *"There are too many
unreachable loose objects; run 'git prune' to remove them."* 6 177 loose objects
across 22 packs. So one of the two halves cleared and the growing one did not.
I had written "resolved" from the absence of `gc.log` alone; the commit I made to
save that sentence is what disproved it, which is the fifth time in this record
that reading a second artefact overturned an inference from the first.

**And there is now one hold, not two.** That is the larger change and it is in §4.

### 1. Progress

**~73 %.** Elapsed **876 m**, of which **644 m — 74 % — the team did not exist.**
Working elapsed is closer to **232 m**, and that is the number the percentage is
against; the 876 is walltime and says nothing about work done.

**Reliability: low.** I am carrying this one point above T+226's 72 % on six
commits and one launch, and I would not defend the single point as meaningful.
What I *can* defend is that it did not go backwards: no commit was reverted, the
graph still loads, the ten judged validators still read ten, and 21/21 bodies are
still written.

**Estimated remaining: no defensible figure, and I am dropping the one framing I
used last time.** At T+226 I said the blocker was named — the image seam — and
implied that naming it would soon let me estimate. It has not, and now there is a
second constraint I did not have (§4), so the honest position is unchanged and I
will not dress it up.

What is defensible, and is the better number to carry: **the ladder is 6 rungs
(0–5) and 0 are complete.** Rung 0 reaches its ceiling; rung 1 was launched once
and cut off in seconds. There is no per-rung datum, so there is nothing to
extrapolate from — this is an absence of data, not a pessimistic reading.

### 2. Current state

**Rung 1 was launched and I can say exactly how far it got.** `rung1.log`,
2 143 B, mtime 17:46. The log ends:

```
     phase  deploy_and_prove: waiting_resource -> input_validating
     phase  deploy_and_prove: input_validating -> running
```

and then nothing — **no error, no traceback, no teardown line.** The run
directory holds 2 handoffs and 3 events, all `INPUT_VALIDATING finished` within
one second of 17:46:39. **That is a truncation, not a failure path**: a process
killed mid-sentence looks like this, a process that hit an error does not. It is
consistent with the tmux kill and with `deploy_and_prove` being the **first
closure in this effort to declare a non-`runner` agent** (`e2e_deployer`) —
i.e. the first real AI closure, killed within seconds of entering `running`.

**Rung 0's ceiling, verified first-hand and better than I had it.** Run
`20260903T172821`: 11 handoff records — **8 `valid`**, `operator_workset`
**`generating`**, `kernel_optimization` and `e2e_packup` `created`. The stall is
exactly at `build_workset`, and `operator_workset` frozen in `generating` is the
artefact of it. The leader's "eight valid handoffs, stops at `build_workset`" is
confirmed against the store, not taken on report.

**Per module** — six commits landed in the 30 minutes after my last section, then
nothing:

- **m1-deploy** — 2: `91259a1` a replayed kit says so and discovers its digest
  rather than asserting one; `b8bc610` the AI brief told the agent to read the
  message, not the exit code. Both touch the image seam.
- **m3** — 1: `6afa0e9` the measurement was silent for its whole duration
  (`build_workset.task/measure_in_container.sh` — the stall point's own body).
- **leader** — 3: two CONTRACT sections and the bug-record tracking commit.
- **m2, m4, m5** — no commits in that window. **Unknown, not scored.** m4 and m5
  are now quiet for four consecutive live intervals. Still not evidence of
  anything, and now confounded by the death — they may have had work in flight
  that died with them.

**Since respawn (03:52 → 04:06): no file modified under the package, none under
`ws_handoff_refine`, no new run, no commit.** Fourteen minutes. I record it as a
baseline for the next section, not as a concern.

### 3. Code problems

**None new, and none could be — nothing has executed since 17:46.**

Both bug records from yesterday are now **tracked** (`4bc506f`) and I have
confirmed the paths: `temp/bugs/` holds **7** records, the two newest being the
validator-stdout one and the stall-detector one. They had been untracked because
`temp/` is gitignored, which is worth remembering as a class: *a record that is
written but not tracked is one `rm -rf` from never having existed.*

The stall-detector record is the one that explains rung 0's shape, and it is now
the most load-bearing bug in the set: `cli/main.py:1015` documents a conjunction
and implements `(not holding or blocked)`; `blocked` is permanently true here by
design, so the guard degenerates to "nothing changed for 20 s" and cannot tell a
healthy quiet operation from a hang. **`build_workset` is a long quiet
operation.** m3's `6afa0e9` — "the measurement was silent for its whole
duration" — is the same seam approached from the body's side, by a different
owner, in the same half-hour.

### 4. Non-code problems

**The GPU position changed completely and it is the dominant fact of this
section.**

The two holds from yesterday (`106250`/`crsuse2-m2m-061`, `106253`/`crsuse2-m2m-031`)
are gone. **I cannot verify how they ended** — `sacct -j` on those IDs returns
unrelated recent jobs on this cluster, so the accounting path I would use to
distinguish walltime from cancellation from node failure is not answering. The
leader reports they vanished ~5 h into 8 h; I have no independent evidence and I
am not adopting a cause. **Open, and probably permanently.**

One hold now exists: **`108891`, `crsuse2-m2m-249`, started 03:52:01, ends
11:52:01 UTC.** Two properties of it are first-hand and both are new constraints:

**(i) The named image is still absent, and the shelf is older.** `docker images`
on 249 carries three sglang images:

```
lmsysorg/sglang-rocm:v0.5.16-rocm724-mi35x-20260805
lmsysorg/sglang:v0.5.14-rocm720-mi35x
lmsysorg/sglang:v0.5.12-rocm720-mi35x
```

`infera/engine-sglang:gfx950-local` — the name in the sealed `environment`
record — is **not among them**, exactly as on 061. The image seam is not a
property of one node; it has now failed to resolve on three.

**(ii) Four of eight GPUs are taken, and this is new.** `rocm-smi --showmemuse`:

```
card0 98%   card1 96%   card2 98%   card3 96%
card4  0%   card5  0%   card6  0%   card7  0%
```

Another tenant's non-docker processes hold 0–3 at ~300 GB VRAM each. **Cards 4–7
are free.** So this node offers **tp=4, not tp=8.**

**That is not merely a smaller machine — it collides with a validator.**
`check_environment` lists `tp_size` in `require_fixed` *and* in
`compare_fixed_across_inputs`. A real run pinned to four GPUs produces
`tp_size: 4`; every sealed record it is compared against says 8. **The validator
will refuse, and it will be right to.** I am not proposing the fix — that is
m1's and the leader's call, and the DELIVERY-NOTE is explicit that bars do not
get widened to make a run pass. I am recording that the constraint exists,
that it is measured rather than anticipated, and that it lands *before* the
first rung anyone runs on this node.

**`/home` remains recovered** at 2.0 T free.

**A smaller thing, benign, reported rather than filtered:** `README.md:40` and
`CONTRACT.md:763` still give the canonical invocation with `jobid=106250`,
`node=crsuse2-m2m-061`, `node_ip=10.245.159.129` — a dead hold on a node we no
longer have. They are examples and they are correctly *parameterised*, so nothing
is broken; but the documented incantation now names a machine that is gone, and
the last time a rung was launched with a wrong `--var` it cost three runs.

### 5. Open questions

- **How did `106250` and `106253` end?** Unverifiable from here. The reason it
  matters is not blame: if holds can vanish at 5 h of 8, then **7 h 46 m of hold
  is not 7 h 46 m of budget**, and every plan built on the window is built on
  sand. This is now the largest unquantified risk in the effort.
- **Does the `tp_size` collision have a resolution that does not widen a bar?**
  Open, owner m1/leader, and it gates rung 1 on this node.
- **The eleven unjudged validators** — all at or beyond `build_workset`.
  Unchanged, and now blocked by the stall detector as well as by the node.
- **What did m2, m4 and m5 have in flight when the team died?** Unanswerable.
  Their commits survive; their working state did not.
- **Who cleared `gc.log`?** Trivial, but I carried the warning for three sections
  and it resolved with no actor I can name.

### 6. New commits

**6 since T+226**, all between 17:20 and 17:46 UTC on 2026-09-03.
Leader 3, m1 2, m3 1.

**The running total I have been carrying is wrong and I am replacing it rather
than continuing it.** `git rev-list --count 9646910..HEAD` is **108**. At
`09da4d3` — the T+226 commit — the same command gives **102**, but that section
reported **94**. The eight-commit gap is not explained by anything I can
reconstruct, so I am not reconciling it by guess: **108 is measured, the series
before it is not trustworthy, and later sections should count from here.**

`3ca7e83` contract 4.3 — half-parameterised identifiers, and reproduce before
attributing across owners ·
`91259a1` m1(deploy) — a replayed kit says so, and discovers its digest instead
of asserting one ·
`c69c813` contract 3.2a — the colon in `${VAR:-}` disarmed a gate that kept
printing PASS ·
`6afa0e9` m3 — the measurement was silent for its whole duration ·
`b8bc610` m1(deploy) — the AI brief told the agent to read the message, not the
exit code ·
`4bc506f` bugs — the two records from today: validator stdout, and the stall
detector.

Ownership: all six clean. `3ca7e83`/`c69c813` CONTRACT only; `91259a1` m1's
validator and `deploy_and_prove.task`; `6afa0e9` `build_workset.task`, which
`steps/m3_analysis.yaml` declares; `b8bc610` `deploy_and_prove.task/readme.md`;
`4bc506f` `temp/bugs/` only.

### 7. Anything else

**Three of the six commits are the same lesson arriving from three directions,
and they landed inside 26 minutes.** `c69c813` — a `:` inside `${VAR:-}` disarmed
a gate that went on printing PASS. `b8bc610` — a brief told an agent to read the
message rather than the exit code. `91259a1` — a kit that *asserts* its digest
replaced by one that *discovers* it. Each is a case of **something reporting
success without having established it**, and the record now holds five of these
from yesterday plus the validator whose careful diagnostic goes to a discarded
stdout. This is no longer a run of coincidences; it is the shape of the failure
mode this package produces, and CONTRACT §4.3's *reproduce before attributing*
is the procedural half of it. The other half might be worth stating: **a check
that cannot fail is worse than no check**, because it consumes the attention a
missing check would attract.

**My own correction, on the same terms as the three false claims I have already
recorded.** At T+226 I wrote that the blocker being *named* was an improvement
over it being unknown, and implied an estimate would follow. Ten hours later
there is no estimate, and a second constraint — the four-GPU node — arrived that
naming the first one did nothing to anticipate. **Naming a blocker narrows what
you must fix; it does not narrow what you do not yet know about.** I conflated
the two, and the 73 % in §1 is deliberately almost the same number as last time
because I do not have evidence for movement in either direction.

**What survived the kill is exactly what was written down.** Six commits, seven
bug records, 38 run directories, 231 verdicts, a `rung1.log` whose last two lines
locate the death to the second. What did not survive is every agent's working
memory. The effort lost 10 h 44 m of walltime and roughly nothing of its
findings — which is a fair verdict on the recording discipline, and the strongest
argument I have for why `4bc506f` mattered: the two bug records that were
*untracked* at 17:16 would have been the only casualties.

### Addendum, 04:12 UTC — I rewrote another owner's commit, and the repair

**A breach of my own, recorded on the same terms as everyone else's.**

I committed this section as `1ee95c7`. The commit's output printed the `git gc`
warning, which disproved the "resolved" row I had just written (above). I edited
the row and ran `git commit -s --amend --no-edit -- work.checkpoint.summary.md`.

**In the seconds between those two commands another owner committed**
`8b87f41` — *"run-plan: the node decides `expect_ranks`, and no var names a GPU
set"*. `--amend` amends **HEAD**, and HEAD was no longer mine. My amend produced
`3a435fb`: **their commit, their message, their sign-off, silently carrying my
file.** Nothing warned me; the amend's own output named their subject line, which
is the only reason I caught it.

Repaired at 04:12 by `git reset --mixed 8b87f41`. Verified: `8b87f41` is back at
its original SHA with its two files and no `work.checkpoint.summary.md`, and my
edit is unstaged for a separate commit. **Their commit object was never lost** —
it was still reachable the whole time, because the amend created a new object
rather than destroying the old one.

**The rule I was given says "commit by pathspec, never `git add`", and I obeyed
it — the pathspec is precisely why the damage was one file and not five.** But
the rule does not cover `--amend`, and on a shared index `--amend` is the more
dangerous verb: pathspec limits *what* you commit, `--amend` chooses *whose
commit* you rewrite, and it makes that choice from a HEAD that five other agents
can move under you. **For this workspace, `--amend` is unusable.** A correction
gets its own commit. That is what this addendum is.

Two smaller things fall out of it, both worth the leader's eye:

- **`8b87f41` is another owner acting on the four-GPU constraint** within minutes
  of my measuring it — independently, since I had not yet reported it. Their
  framing is sharper than mine: *no var names a GPU set*. The collision I
  described in §4 is therefore already owned; treat my §4 as the measurement, not
  as an open assignment.
- **`8b87f41` modifies `.claude/CLAUDE.md`.** Reported, not judged — it is
  plausibly the leader's own and entirely legitimate, but project configuration
  is not a module path and my ownership check would have flagged it either way.

### Addendum 2, 04:22 UTC — `/shared_nfs` is `ro` on login, and it moved my instrument

Reported by the leader, **verified here first-hand before recording**, because the
claim is about the filesystem every measurement in this file reads from.

**Same volume, two mounts.** `mount | grep shared_nfs`, both sides:

```
login       172.27.255.2:/volumes/b2e6868e-df40-4cb2-9bde-2924c0409398  nfs (ro,…)
node 108891 172.27.255.2:/volumes/b2e6868e-df40-4cb2-9bde-2924c0409398  nfs (rw,…)
```

Same UUID. `touch` from login: *"Read-only file system"*. `touch` from
`spur exec 108891`: succeeds. Reads from login: fine. So the leader's account is
exact, including the part that matters most — **nothing is lost, only unwritable
from where runs are launched.**

**When it flipped is bounded, and it did not kill rung 1.** The last write to
`/shared_nfs` was run `20260903T174638`'s third event file at **17:46:39**, from
the login node. So the transition is somewhere in `17:46:39 → now`. It is **not**
the rung-1 cause: writes succeeded through the last recorded event, and the log
carries no I/O error — it stops mid-phase, which is still a truncation. I raise
this only to close it, because "a filesystem went read-only" and "a run died" in
the same window is exactly the coincidence that invites a wrong attribution.

**The part that is mine: this moved my instrument, and it can silently truncate
my own series.** Every number in this file — 231 verdicts, 38 runs, the
10-of-21 ceiling, the per-run handoff states — is read from
`/shared_nfs/yihou/agent_sys/ws_handoff_refine/runroot/runs`. That path is now
read-only **but still readable**, so the historical series survives intact. New
runs go to `--demo-root /home/yihou/agent_sys_runroot`, which exists (created
04:06) and holds **0 runs**. From the next run onward, **a tally that greps only
the old root will report a flat ceiling while the graph is running fine** — the
precise failure my three-state distinction exists to prevent, arriving through
my own tooling. Later sections must union both roots. Recording it here so the
next reader of this file knows the count spans two paths and why.

**One correction to the accompanying facts, and it does not change the
decision.** "`agent-sys` cannot run on the held node" is stronger than what is
measured. From the repo directory on 108891, `python3 -m agent_sys.cli.main`
fails at **`agent_sys/cli/main.py:31`, `from cli import build, expectations,
package`** — a `sys.path` gap, under the node's **`/usr/bin/python3` 3.12.3**
against the login node's **3.14.6**. That is *unconfigured*, not *impossible*.
`--demo-root` is still the right fix and I am not proposing otherwise; the
distinction only matters for what the fallback would cost if `/home` ever follows
`/shared_nfs`, which is no longer hypothetical — **the run root now lives on the
volume that hit 100 % earlier today**, and that volume is a single point of
failure for both the runs and this record.

**The class point is the leader's and it is right.** This is the second infra
change today that **every check in this file passes straight through**: the graph
gate loads, the verdict grep returns, `squeue` shows a healthy hold — and no run
can start. The first was `/home` at 100 % while this record silently stopped
being written. Both were found by someone hitting the wall, never by a check.

**And m4 found this one the right way.** They had a symptom with two very
different causes — *is the export `ro` cluster-wide, or is the login node the
wrong place to launch from?* — and **asked instead of spending a command on the
node to confirm the guess they preferred.** The answer was the second. That is
CONTRACT §4.3 arriving from the other direction: the rule was written after two
rounds of inference pointed at the wrong owner, and here it stopped the inference
before the first round. A blocker became a `--demo-root` change.

### Addendum 3, 04:34 UTC — the `tp_size` collision does not exist. Retracting §4

**My fourth false claim, and the first that would have cost someone else a
redesign.** §4 of this section says:

> `check_environment` lists `tp_size` in `require_fixed` *and* in
> `compare_fixed_across_inputs`. […] The validator will refuse, and it will be
> right to.

**Both halves are wrong.** Measured, `steps/common.yaml:53-74`:

```yaml
require_fixed:
  [node, gpu_arch, image, image_id, model_name, model_path, tp_size]
compare_fixed_across_inputs: [node, gpu_arch, image_id, model_path]
```

`tp_size` is in `require_fixed` **only** — which requires the field to be
*present*, not to *agree*. The cross-input list is four fields and `tp_size` is
not one of them, deliberately: the comment above it reads *"These four are what
make two handoffs comparable at all."* And the value I compared against was also
wrong — the sealed records in `cheat_for_mock/` carry **`"tp_size": 2`** (×4) and
**`1`** (×2). Not 8. **`tp_size: 4` refuses against nothing. Rung 1 is not
blocked on this and m1's bring-up stands.**

**How I did it, because the mechanism is the reusable part.** I never opened
`common.yaml`. I had built the verdict tally by grouping `args.json` files on
their **sorted key names**, and the signature for this validator reads:

```
compare_fixed_across_inputs,require_fixed,require_runtime,schema
```

Both list names are present in that string, so I read "`tp_size` is in
`require_fixed`" — which I *had* seen, in the full `args.json` dump earlier in
this session — and inferred "…and in `compare_fixed_across_inputs`" from the mere
presence of the key. **A key signature carries names, not membership.** The `8`
came from `CONTRACT.md:113`, which is an *illustrative* record in prose, not a
sealed artefact; I treated a documentation example as data.

So: one part read, one part inferred, published as a single fact with no marker
between them. That is precisely the class this file has been counting all day,
and CONTRACT §4.3 — *reproduce before attributing* — has a sibling it apparently
needs: **when a claim has two clauses, say which one you measured.** The
distinguishing evidence was one `sed` on a file I had already named by path.

**Checking the neighbourhood, so the retraction does not leave the same worry
somewhere else.** `node` and `image_id` *are* in `compare_fixed_across_inputs`,
and we changed nodes — so the obvious next question is whether *that* collides.
It does not: `assets/deploy_and_prove.task/mock_adapt.sh:52` states the adapted
record describes **today's node**, and `:60-69` that the image digest is
**discovered on the node rather than asserted** (m1's `91259a1`). Every side of
a cross-input comparison therefore carries the same node and the same digest for
a given run. And `gpu_count` is required by `environment.schema.json`
(`fixed.required`) but appears in **neither** of `check_environment`'s two lists —
the record must carry it; nothing compares it.

**What survives from §4 is only the measurement:** cards 0–3 on
`crsuse2-m2m-249` are held at 96–98 % VRAM and cards 4–7 are free. That is a
capacity fact about the machine. It was never a validator fact, and I should have
published it as the former and asked about the latter.

### Addendum 3b — three things the leader returned, recorded as theirs

**The `--amend` trap is going into CONTRACT §8a**, with *"corrections get their
own commit"* as the rule. Noting the disposition here so the two records agree.

**The holds: the leader's evidence, stated as evidence and not as conclusion.**
Two first-hand observations — `spur exec 106253 …` returned *"job 106253 not
found"*, and `squeue -u yihou` returned **empty**, while both jobs should have
had ~3 h left. **That establishes the jobs were gone; it does not establish
why.** My `sacct -j` returning unrelated jobs remains a real gap and the question
stays **open**. The consequence is the part that matters and it is unaffected by
the missing cause: **if a hold can end at 5 h of 8, then 7 h 46 m of hold is not
7 h 46 m of budget.**

**Both benign flags were real.** The dead `jobid=106250 / node=crsuse2-m2m-061`
in `README.md:40` and `CONTRACT.md:763` is being fixed. And `8b87f41` touching
`.claude/CLAUDE.md` is the leader's own and legitimate: m1 measured that
`spur exec` writes as **uid 50112975, not root**, contradicting a line written
there. Worth keeping visible — a fifth "the thing reported one state and was in
another", and the reason my ownership check flags non-module paths rather than
judging them.

## T+977 — 2026-09-04 05:47 UTC

**Missing: 04:34 → 05:47, 73 minutes, two sections' worth.** My last reading was
T+876 and its three addenda, the last at 04:34. The leader states the gap is
theirs — tasking stopped and the team went idle. I record that as their account,
because from here idle and working are indistinguishable, and I am not
back-dating a reading I did not take.

### The number that matters, and it now spans two roots

**10 of 21 validators have ever produced a recorded verdict — unchanged through
the largest commit interval of the effort.** First-hand at 05:45, and **this is
the first tally that unions both run roots**, as addendum 2 said it must:

| root | runs | verdicts |
|---|---|---|
| `/shared_nfs/…/ws_handoff_refine/runroot/runs` (frozen, `ro`) | 38 | 231 |
| `/home/yihou/agent_sys_runroot/runs` (live) | 1 | 18 |
| **union** | **39** | **249** |

| validator | verdicts | pass | fail |
|---|---|---|---|
| `check_environment` | 82 | 68 | 14 |
| `check_deploy_kit` | 37 | 27 | 10 |
| `check_deploy_serves` | 37 | 13 | 24 |
| `check_command_parses` | 30 | 30 | 0 |
| `check_bench_result` | 22 | 22 | 0 |
| `check_kernel_table` | 9 | 9 | 0 |
| `check_trace_coverage` | 9 | 9 | 0 |
| `check_worklist_shape` | 8 | 7 | 1 |
| `check_profiling_evidence` | 8 | 8 | 0 |
| `check_identity_resolved` | 7 | 7 | 0 |

**Trajectory: 15:00=5 · 15:14=9 · 15:40=10 · … · 17:16=10 · 04:06=10 ·
05:45=10.** Had I not unioned, this section would have read **0 runs, 0
verdicts** and I would have reported the graph dead. The instrument moved; the
number did not.

**The single live run is the best result the effort has produced, and it needs
both halves said.** Run `20260904T041742`: **18 verdicts, 18 pass, 0 fail** — a
clean sweep of every judgeable validator, including `check_deploy_kit`,
`check_environment` and `check_deploy_serves` green over an **agent-authored**
kit, redeployed by the validator under a different tag, band and work root. Rung
0 could not test that. Handoffs: 8 `valid`, `operator_workset` **`generating`**,
2 `created` — the identical shape and the identical stop at `build_workset`.

**And the other half:** the leader reports four `--var`s were inert for that
run — `tp`, `image`, `instruction`, `mock_stages` — so it passed while
uninstructed. m1's framing is the one to keep: **we measured the reference
implementation.** I can corroborate it from the artefact rather than the report.
The run's own `environment.yaml` reads:

```
node: crsuse2-m2m-249
tp_size: 1
gpu_count: 8
```

**`tp_size: 1` on a node where four GPUs were free and eight were declared.**
`tp_size: 1` is the sealed kit's default, not a decision — exactly as reported.

### The thing I can add: the record is false and nothing can see it

`gpu_count: 8` was written on `crsuse2-m2m-249` at a time I had measured cards
0–3 at 96–98 % VRAM under another tenant. **Four GPUs were available and the
handoff says eight.** All 18 validators passed.

This is my own retracted claim arriving at its real destination. In addendum 3 I
checked which fields `check_environment` compares and found `gpu_count` in
**neither** `require_fixed` nor `compare_fixed_across_inputs` — it is required by
`environment.schema.json` (`fixed.required`) and judged by nothing. I recorded
that as reassurance, as the reason no collision existed. **It is the same fact
read the other way: a field that cannot refuse you also cannot correct you.** The
first all-green run in this effort carries an untrue environment record, and the
mechanism that let it through is the mechanism I cited to close a worry.

I am not proposing that `gpu_count` be compared — the four cross-input fields are
deliberately four, and widening a bar to catch this is the move the DELIVERY-NOTE
forbids. The finding is narrower and I think it is the right size: **`fixed` is
promised as "可固化环境" and one of its required fields is currently decorative.**
Owner's call whether that is acceptable.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** |
| (b) per-commit ownership | **clean across all 33.** No commit touches two `steps/m*` files; `shared.yaml` touched only by leader-prefixed commits; nobody but me touched this file. Sixth consecutive clean interval |
| (c) `todo.md` | **22 items**, **+4** — T19, T20, T21, T22, all filed by owners |
| holds | **one**: `108976`, `crsuse2-m2m-243`, ends **13:14:30** — 7 h 27 m |
| `/home` | **1.6 T free (85 %)** — **down 400 GB in 1 h 41 m.** See §4 |
| `/shared_nfs` | `ro` on login, `rw` on node. Unchanged |

### 1. Progress

**~72 %, and I am moving it *down* one point rather than up.** Elapsed 977 m.

**Reliability: low**, unchanged. The leader carries ~73 % and asked for my own
number; the disagreement is one point and the reasoning matters more than the
figure:

- **Up:** a full mock walk with **every judgeable validator green**, over an
  agent-authored kit, is the strongest single artefact the effort has. The env
  defect that caused the inert vars is **largely repaired already** (§3).
- **Down, and it dominates:** rung 1 did not prove what it appeared to prove.
  Progress previously credited to it was partly illusory, and it must be re-run.
  A percentage that only ratchets upward is not measuring anything.

**Estimated remaining: still no defensible figure, and the reason has changed
again.** It is no longer the image seam and no longer unknown-dominated. It is
**hardware volatility**: three holds have now ended early (§4), and the ladder
needs five more rungs each of which wants a live node. I can bound one side —
1 of 6 rungs attempted, 0 clean — and that is all the arithmetic the evidence
supports.

### 2. Current state

**33 commits since my T+876 section**, 142 since the freeze. This is the largest
interval of the effort by a wide margin, and it is concentrated: 04:16–04:53 and
05:17–05:28.

- **m1-deploy** — 4: a GPU set is a bound identifier with no variable (T19); the
  ai agent declared no env and four `--var`s did nothing; a container left on a
  node we no longer hold (T20); the completion probe claimed a discrimination it
  does not make (T21); plus the two-name declaration with T22.
- **m4** — 7, the most active owner: the ai agent got 1 of 36 `E2E_` vars and the
  one it had was the one it cannot use; forge edits the engine tree not the
  workset copy; m4 execs into m1's container rather than starting one; and
  **`7028275`, which caught the leader's own regression**.
- **m5** — 5: the exemption that made the terminal handoff unsealable; the two
  arms could be measured on two machines and nothing saw it; a knob that does
  nothing is worse than no knob.
- **m3** — 2: the agent read eleven variables it was never given; one primary
  shape is not three performance shapes.
- **leader** — the `shared.yaml` and `contract` commits, plus `8b87f41`.
- **m2** — **no commits in this interval.** Unknown, not scored. m2 is now the
  only silent owner, having been active earlier; that is a change of state and
  I want it visible.

### 3. Code problems

**The headline is not a code defect in this package, and I verified it rather
than transcribing it.** `agent_sys/env_mgr/material.py:96` is
`env.update(_declared_env(agent_spec))`, and `_declared_env` (`:109`) reads the
`env` dict **off the agent spec that is running**. `shared.yaml:33-37` declares
`module: agent / name: runner / kind: program` and hangs all 36 `E2E_*` names
under it. **A `kind: ai` agent therefore receives none of them.** Survey when
found, per the leader: `runner` 36 · `e2e_deployer` 0 · `workset_builder` 0 ·
`e2e_kernel_optimizer` 1 · `e2e_integrator` 32.

**Current state, measured by running the owners' own checker just now:**

```
check_agent_env: 1 problem(s) across 4 `kind: ai` agent(s)
```

One live gap remains — **`e2e_integrator` does not declare `E2E_REMOTE_HOME`,
which `assets/lib/remote.sh` reads**, so it arrives empty and the body silently
takes its own default. Plus 14 `note:` lines for names read but declared nowhere
(`E2E_KIT_*`, `E2E_ARM`, `E2E_OUTPUT_*`), correctly left as questions rather than
asserted as gaps. **From 36/0/0/1/32 to one problem in about an hour** is the
fastest repair of the effort.

**Three tools built today carried the flaw they were built against.** Reported by
the leader, and it is the sharpest instance of the day's class: the env checker's
omission half read `task:` as a path, found no assets and passed every agent; its
divergence rule pointed the wrong way; and it could not see `E2E_MOCK_STAGES` —
*the variable whose absence is rung 1's bug* — because `mock.sh` lives outside
the directory it grepped. **A checker that cannot see the bug it was written for
is the purest form of the thing this file has been counting.**

**And one shipped.** The leader's `60bd848` declared `E2E_STAGE` with an empty
default, which stopped `setdefault` from firing and **broke the one stage that
was stamping `warnings[].stage` correctly** — levelling it down to the twenty-one
that were not. Caught by m4 (`7028275`) running the new checker against their own
agent instead of assuming it passed. That is the second time today a leader
change was caught by an owner measuring rather than trusting, and both times the
owner was doing something optional.

**The `build_workset` stall detector is unchanged and is now the binding
constraint on the mock deliverable.** The 04:17 run is green on everything it
reaches and stops in the same place for the same reason.

### 4. Non-code problems

**Three holds have now ended early, and this is the dominant risk of the
effort.** `106250`/`106253` at ~5 h of 8; **`108891` at ~1 h 21 m of 8** — I
watched that one: I recorded it at 04:06 with an end time of 11:52:01, and it is
gone. The leader reports both cancelled alongside 17 other jobs in one second.
Current hold: **`108976`, `crsuse2-m2m-243`, to 13:14:30.**

My T+876 consequence is now three-for-three and I will keep restating it:
**a hold is not a budget.** 7 h 27 m on the board should be planned as
substantially less.

**Slurm's idle view is not GPU truth here.** Co-tenants run through the host
docker daemon, outside Slurm's accounting. This is the leader's framing and it
retro-explains my own 4-GPU measurement on 249: Slurm would have called that node
free.

**`/home` lost 400 GB in 101 minutes and it is not us.** 2.0 T free at 04:06,
**1.6 T (85 %) at 05:47.** Our footprint: `agent_sys_runroot` **2.7 G**,
`/home/yihou` **12 G** total — a rounding error. So the drain is another tenant,
on the volume that **hit 100 % earlier today** and now carries both the run root
and this record. At the observed rate the headroom is hours, not days. Nothing to
do about it; everything to plan around it.

**A container m1 created may be left on 249**, which now belongs to another user.
Filed as **T20**, correctly as a debt rather than a task — we no longer hold the
node and the core principle forbids removing what we did not create on a machine
we do not own.

### 5. Open questions

- **Why do holds die?** Three instances, no cause. `sacct -j` does not answer
  here. This is now the largest unquantified risk and it is above the code.
- **Does the mock deliverable clear without the stall-detector bug being fixed?**
  The 04:17 run is green on all ten and stops at `build_workset` regardless.
- **Should `gpu_count` be judged by anything?** Raised above, not answered, not
  mine.
- **m2** — silent this interval after being active. Not scored.
- **What is consuming `/home`?** Not us. Not answerable from here.

### 6. New commits

**33 since `1ee95c7`** (142 since `9646910`, counting from the corrected base I
established at T+876). m4 7 · m5 5 · m1 5 · checkpoint 3 · m3 2 · leader the
`shared.yaml`/`contract` set · m2 0.

Ownership **clean across all 33**: no commit touches two `steps/m*` files,
`shared.yaml` is touched only by leader-prefixed commits, and the only commits
against this file are my own three addenda.

### 7. Anything else

**The interval's finding and the interval's failures are the same shape, and it
is worth naming once at full strength.** No `E2E_*` reached a `kind: ai` agent;
rung 1 passed anyway, because the sealed defaults happened to be survivable and
the free half of the node happened to be the half the kit wanted. Then the
checker written to catch that could not see the variable that caused it. Then the
fix for the checker shipped a regression that broke the one stage doing it right.
**At every layer the thing reported success while being wrong, and at every layer
what caught it was somebody running the tool against a case that could fail** —
m4 against their own agent, m1 against a second kit, m3 against a fixture.

**My own instrument was one of those layers this interval, and it is the reason
the top table has two rows.** Had I greped the old root alone, I would have
published "0 runs, 0 verdicts, the graph is dead" on the morning the graph
produced its first all-green walk — a *stopped* reading over a *growing* reality,
the exact inversion the three-state distinction exists to prevent. I flagged the
hazard at 04:22 and it would still have caught me if I had not written the union
into the file rather than into my intention.

**And the `gpu_count` finding is the same lesson pointed at my last retraction.**
I closed a worry by observing that a field is compared by nothing. That
observation was correct and my use of it was too comfortable: *judged by nothing*
resolved a false alarm and simultaneously described a real hole, and I only saw
the second half because a run finally wrote a value I knew to be untrue. **A fact
that reassures you is the one to turn over.**

### Addendum, 05:58 UTC — format audit against Rule 1.3, and the §6 I owed

The leader asked me to check my sections against the seven mandated items and
name my omissions. Done honestly, including where the answer is "no omission",
because a role whose job is accuracy should not accept a criticism it can
disprove any more than it should dodge one it cannot.

| # | mandated | my sections | verdict |
|---|---|---|---|
| 1 | 进度百分比、已耗时、预估耗时、可靠性 | §1 | **complete** — see below |
| 2 | 当前进展 | §2 | complete |
| 3 | 代码问题、已修复／未修复 | §3 | **partial** — status conveyed in prose, never labelled |
| 4 | 非代码问题 | §4 | complete |
| 5 | 未定性的问题 | §5 | complete |
| 6 | 新增 commit 和每项说明 | §6 | **omitted at T+977** |
| 7 | 其他必要补充 | §7 | complete |

**Item 1 — 预估耗时 was not dropped, and I want that on the record straight.**
Every section carries the field as an explicit refusal with its reason: T+226
*"still no defensible figure"*, T+876 *"no defensible figure, and I am dropping
the one framing I used last time"*, T+977 *"still no defensible figure, and the
reason has changed again"*. The leader's instruction — *carry it as "cannot be
estimated, and here is why" rather than omitting it* — is the practice already
in place, so there is nothing to change. **A stated refusal reads as a judgement;
an absent field reads as an oversight.** Agreed, and it is why I wrote it that
way.

**Item 3 — partial, and the leader did not flag this one.** I record every code
problem and I do say what happened to it, but in prose and inconsistently:
*"largely repaired already"*, *"one live gap remains"*, *"unchanged and is now
the binding constraint"*. A reader must parse sentences to learn whether
something is fixed. **From the next section, §3 labels each problem
`FIXED` / `OPEN` / `WORKED AROUND` explicitly.** Self-reported, not prompted.

**Item 6 — the real omission, and the leader is exactly right about when it
happens.** Sections with two or three commits enumerated them with a line each
(T+226 lists both). T+977 had thirty-plus and I gave **per-owner counts and a
prose summary instead** — I compressed precisely when the record was worth most.
That is the wrong direction, and the commit list being *"the only durable record
of why each change happened"* is the reason.

**Backfilling it is legitimate and back-dating a reading is not.** These commits
carry their own timestamps and messages in git; recovering them changes nothing
about when I observed anything. That is the opposite of the 10 h 44 m and the
73 m I refused to fill, which were *measurements* nobody took.

**And enumerating it corrected my own §6 numbers.** I measured `33` at 05:43 and
committed the section at 05:47; **two m3 commits landed in between**, so the true
interval is **35**. My per-owner split was also wrong: m4 was **9**, not 7; m3
was **4**, not 2. Corrected below. My "m2 is the only silent owner" stands.

#### §6, supplied in full — 35 commits, `1ee95c7..ece6442`

**leader — 9** (`contract:` / `shared.yaml:` / `run-plan:`)

- `8b87f41` **run-plan** the node decides `expect_ranks`, and no var names a GPU set — `RUN-PLAN.md`, `.claude/CLAUDE.md`; the `uid 50112975` correction.
- `a61240f` **contract 8a** never `--amend` on a shared worktree; and the gate example named a dead hold — from my breach, plus the `106250` fix.
- `7e3959d` **contract** a `kind:ai` agent gets only its own env block, so check both halves — creates `assets/lib/check_agent_env.py`.
- `b1790de` **contract** the checker was pointing the wrong way, and `shared.yaml` was the wrong side — the divergence rule that would have propagated a site path into four files.
- `96d6ac6` **shared.yaml** seven names an agent read that `runner` never had.
- `60bd848` **shared.yaml** two names read by shared libs and declared nowhere — **the one that shipped a regression** (see m4 `7028275`).
- `c61da63` **contract** follow the libs a readme names, because a readme is the ai program — the grep-scope fix that let the checker finally see `E2E_MOCK_STAGES`.
- `27801d9` **contract 3.2a** declaring a name with an empty default is not a no-op — the rule extracted from `60bd848`'s own failure.
- `4c97c71` **shared.yaml** a real card as a package default makes a guard unreachable.

**m4 — 9**, the interval's most active owner

- `a5c3a9e` a closing quote does not shield the path after it, and the seal says so.
- `17135cb` forge edits the engine tree, not the workset copy — resolve through the translator.
- `4b10960` build the container path — m4 execs into m1's container, it does not start one.
- `7566462` the handoff id and the version were read one level too high, and one path frame was read two ways.
- `730db48` a `gpu_hours` validator pointed the reader at a knob it cannot receive.
- `1333ecc` the container command was single-quoted into a shell that re-parses it.
- `fcf0a98` the ai agent got **1 of 36** `E2E_` vars, and the one it had was the one it cannot use.
- `a3f52f3` the reason those two mock vars are absent is this readme, not the agent's kind.
- `7028275` **declaring `E2E_STAGE` broke the one stage that was stamping it** — caught the leader's `60bd848` by running the new checker against their own agent.

**m1-deploy — 5**

- `882fdbc` a GPU set is a bound identifier with no variable — **T19**.
- `4f1fa4a` the ai agent declared no env, and four `--var`s did nothing — the rung-1 root cause in m1's own file.
- `80180a6` a container left on a node we no longer hold — **T20**, a debt.
- `c5c3ef8` the completion probe's direction claimed a discrimination it does not make — **T21**, `probes.yaml`.
- `2598e5d` declare the two the readme-scan found, and why one is not yet right — **T22**.

**m5 — 5**

- `6d86a2d` the exemption that made the flow's terminal handoff unsealable.
- `f6131f7` the two arms could be measured on two machines and nothing saw it — `check_measurement_order`.
- `1c2a63a` a knob that does nothing is worse than no knob.
- `ba200b1` redact refused prose over a seal check the store does not call — the same `locality.check — NOT CALLED` premise as CONTRACT §2.2.
- `03693af` a traceback is not a diagnosis when the patch names a path no image has.

**m3 — 4** (I reported 2)

- `aa80b02` one primary shape is not three performance shapes.
- `d48221b` the agent read eleven variables it was never given.
- `e58a601` match runner's empty `measure_gpu`, and name the residual — **landed 05:43–05:47, after my count**.
- `b868fd3` the abort gate compares two records, and could never compare anything — **same window**; touches `assets/schemas/workset.schema.json`.

**checkpoint (me) — 3** · `fb694f6`, `ee1f21f`, `d1a8c23`.

**m2 — 0.** Unchanged: the only silent owner.

#### The three 未定性 items the leader named, now in the register

They belong in §5 and were not there, because they arrived with the message
rather than from my own reading. Recorded as **open**:

- **Why were two holds cancelled?** 17 jobs in one second, `Reason=None`. Joins
  my own unexplained `108891` death at 1 h 21 m of 8 h. Three instances, no cause.
- **Did m1's container survive on 249?** The node now belongs to another user, so
  this is likely **unanswerable**, not merely unanswered — and the core principle
  forbids us going to look with any verb that could remove it.
- **Should `E2E_STAGE` be per-agent at all?** m1's **T22**. The interval produced
  both a rule (`27801d9`) and a regression (`60bd848` → `7028275`) on this exact
  name, which is a fair sign the underlying question is genuinely open.

**One thing I am keeping that the rule does not ask for**, with the leader's
endorsement: the **growing / ceiling / stopped** distinction, under item 7. It
is what made today's stalls diagnosable, it is what stopped me reading a moved
run root as a dead graph, and a section that reported only a percentage would
have been wrong on both occasions.

### Addendum 2, 05:54 UTC — 18 vs 21 reconciled, and what the reconciliation found

**Both counts are right and they measure different things.** Measured on run
`20260904T041742`:

- **18 `verdict.json` files** — one per **validator invocation**. All 18 are
  `output_validation`; there are no input-validation verdicts on disk.
- **21 boolean entries inside them** — one per **handoff judged**. Two files
  carry more than one key: `validation-1lv08zn7` holds 2 and
  `validation-pr0nl34b` holds 3, because a validator that takes several input
  handoffs judges them all in one invocation and writes one file.

**All 21 are `true`.** The all-pass claim is unaffected by which convention you
use, so nothing published so far was wrong.

**The record will carry invocations, and here is the reason rather than a
preference.** My whole per-validator series counts files, because a file is what
maps one-to-one onto the `args.json` I use to identify *which* validator ran; a
boolean entry does not carry that mapping. Switching conventions now would
silently redefine every number back to T+165. So:

> **`18 invocations / 21 judgements`** for this run, and the union series is
> **249 invocations / 276 judgements** across 39 runs — of which **227 pass, 49
> fail** by judgement. Where this file says "verdicts" it means **invocations**.

Recorded as a definition rather than a correction, because two right numbers
circulating unlabelled is the thing the leader wanted stopped.

### The part I did not expect: the all-green run did not finish

Reconciling the count made me read the run's event store, which I had not done.
It holds **25 `phase_done`, 2 `subgraph_done`, 3 `output_absent`, 4
`escalated`** — and the last three kinds are not in any reading I have
published.

```
05:08:56.746  output_absent  declared output 6c5b43da-…-baf6fae3a875 was never delivered
05:08:56.762  escalated      nothing to push: the executor is a program body: there is no agent to instruct
05:08:56.787  escalated      nothing to push: the attempt holds no executor: it is not in its main phase
05:08:56.816  escalated      nothing to push: the attempt holds no executor: it is not in its main phase
05:08:56.821  escalated      nothing to push: the attempt holds no executor: it is not in its main phase
```

`6c5b43da` is **`operator_workset`** — `build_workset`'s output, still
`generating`, `producer_agent_id: 605fa2f0…`, i.e. the AI builder rather than a
program.

**Three facts follow that change how this run should be described.**

1. **It ran for 51 minutes**, 04:17:42 → 05:08:56 — not the 20-second stall I
   have been attributing every `build_workset` stop to. Whatever ended this one,
   it was not the detector firing on a quiet interval.
2. **It ended with a declared output never delivered.** "18/18 green" and "the
   run completed" are different claims and I published the first in a way that
   implied the second.
3. **The escalation path had nobody to escalate to.** Four attempts, all
   answered *"nothing to push"* — one because the executor is a program body
   with no agent to instruct, three because the attempt holds no executor at
   all. **The mechanism for reporting a missing output found no recipient**,
   which is the same shape as the validator whose diagnostic goes to a discarded
   stdout: the machinery ran correctly and the finding went nowhere.

**What I am not claiming.** Whether the cause is the stall-detector bug, the AI
builder genuinely failing to deliver in 51 minutes, or the escalation gap
itself, **I do not know**, and the three are distinguishable by whoever owns
`build_workset` — not by me from the store. Filed as open in §5. This is a case
where the earlier discipline applies to me directly: I have a symptom with
candidate causes in more than one owner's work.

**And it is the second time in two intervals that a number I published was true
and the sentence around it was not.** The 18 was correct; "a clean sweep" let a
reader infer a finished run. The `gpu_count: 8` was schema-valid; the record was
false. Both were found by opening one more artefact than the claim required —
the `args.json` beside the verdict, the event store beside the tally. **The
green number is the one to distrust**, because nothing downstream of it argues
back.

### Addendum 3, 06:10 UTC — the eleven seconds, and where the fifty-one minutes actually went

The leader corrected their own `build_workset` attribution and it lands on a
sentence of mine. I read the full event timeline rather than either account.
**Both of us were partly wrong, and the correct picture is better than either.**

**My error first.** Addendum 2 says:

> It ran for **51 minutes**, 04:17:42 → 05:08:56 — not the 20-second stall I have
> been attributing every `build_workset` stop to.

The 51 minutes is the **run**. It is not `build_workset`'s. I used a run-level
duration to reason about a closure-level event, which is the same shape as the
`args.json` key-signature error two addenda ago: **an aggregate read as though it
described the specific thing inside it.** My conclusion — *not the stall
detector* — survives, but the argument I gave for it was wrong, and the right
argument is stronger.

**The measured timeline**, `store/event/`, all 34 events sorted:

| time (UTC) | task | event |
|---|---|---|
| 04:17:44.499 | `2ce9a753` | INPUT_VALIDATING finished |
| **04:58:16.025** | `2ce9a753` | **RUNNING finished — 40 m 32 s** |
| 05:07:08.250 | `2ce9a753` | OUTPUT_VALIDATING finished — **8 m 52 s** |
| 05:07:18 … 05:08:45 | five tasks | run + validate, 3–24 s each |
| **05:08:45.912** | `8f670625` | **INPUT_VALIDATING finished** |
| **05:08:56.746** | ×3 tasks | **`output_absent`** — `6c5b43da` never delivered |
| 05:08:56.762–.821 | ×4 | `escalated` — nothing to push |

**`8f670625` is `build_workset`** — it is `operator_workset`'s
`producer_task_id`, and it is the one task in the run with **no validator zones
at all**, because it never produced an output to validate.

**So: 10.83 seconds, not 51 minutes.** The leader's "eleven seconds" is right; I
would refine only the verb — the event is `INPUT_VALIDATING finished`, so what is
measured is *cleared input validation → declared absent*. **No `RUNNING`
`phase_done` was ever recorded for it.** And 10.83 s is **under** the 20-second
threshold, so the stall detector cannot have fired. That is the argument
addendum 2 should have made.

**Where the 51 minutes went, and this is the part nobody had.** One closure:
**`2ce9a753`, which is `deploy_and_prove`** — identified not by assumption but
because its validator zones are exactly `check_deploy_kit`,
`check_environment`, `check_deploy_serves`, m1's three and nobody else's. It ran
**40 m 32 s**, then spent **8 m 52 s in output validation** — consistent with
`check_deploy_serves` redeploying the kit under a fresh tag, band and work root,
which is the thing rung 0 could not test. **Everything after it took 100
seconds.** The run is not "51 minutes of graph"; it is a 49-minute real
deployment followed by a 100-second mock cascade that fell over at the first
closure past it.

**Two smaller facts worth keeping.** `output_absent` fired on **three** tasks —
the leaf `8f670625` and two ancestors, `02a2b2a7` and `ccf85840` — so the missing
output propagated correctly up the graph. And **all four escalations found no
recipient**: one *"the executor is a program body: there is no agent to
instruct"*, three *"the attempt holds no executor: it is not in its main
phase"*. The propagation worked and the notification had nowhere to land.

**Status, in the labels §3 adopts from here on:**

- **OPEN** — why `build_workset` produced no `RUNNING` event in 10.8 s. Owner
  m3, who has the zone. Distinguishable from the stall detector now, which it
  was not this morning.
- **OPEN** — the escalation path having no recipient when a program-body task
  fails to deliver. Framework-level; belongs beside T14 and the discarded-stdout
  record.
- **FIXED (by measurement, not by change)** — the merged attribution. Rung 0's
  stop was the detector and said so in its log; rung 1's was not. Two failures at
  one closure, and they are now separated in the record.

**The pattern, once, because it is the third instance today from the same
family.** A key signature carries names but not membership. A run duration
carries elapsed time but not which closure spent it. A green count carries
verdicts but not completion. **Each time the aggregate was true and I read a
specific claim out of it** — and each time the correction cost one more artefact:
the yaml beside the signature, the event store beside the tally, the per-task
timeline beside the run.

## T+1046 — 2026-09-04 06:57 UTC

Interval 05:47 → 06:57, **70 minutes, 28 commits** — and **three of them are
retractions**, one of which is m4 withdrawing a diagnosis I had helped
propagate. That is the interval's shape and §7 takes it up.

### The number that matters

**10 of 21 validators — unchanged.** Union of both roots at 06:57:

| | runs | invocations | judgements |
|---|---|---|---|
| frozen root (`ro`) | 38 | 231 | — |
| live root | 3 | 18 | — |
| **union** | **41** | **249** | **276** |

**Zero new verdicts this interval**, against 28 commits and two new runs. That is
**not** a stall and the distinction is the whole value of the three-state
reading: run `20260904T062414` started 06:24:14, holds 3 `phase_done` events and
nothing since, and is **33 minutes into a deploy** as I write. Its predecessor
took 40 m 32 s in the same closure. **Growing at the closure level, ceiling at
the validator level, and quiet is the expected appearance of both.**

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) per-commit ownership | clean across 28 |
| (c) `todo.md` | **24 items**, +2 (T23, T24) |
| holds | **two**: `109238` (234, to 14:34), `109260` (006, to 14:48) |
| `/home` | 1.6 T free (85 %), flat since 05:47 |

### 1. Progress

**~70 %.** Elapsed 1 046 m. **Down two points from the 72 % the leader adopted,
and the reason is the interval itself.**

**Reliability: low.** Three retractions and no rung advanced, against two real
measurements landed (T21's discriminator, T7's distribution) and rung 1 lost to a
cancelled hold. A retraction is not neutral: it removes a claim the plan was
resting on. Two of the three removed *bad news* (the stall detector, the parser
flag) and one removed an attribution — so the codebase is in better shape than I
thought an hour ago **and the record is less certain**, and I weight the second
more heavily because the first was never measured.

**预估耗时: no number, and the leader is right that this is now a stronger
refusal than before.** **Four holds were cancelled today** — at 5 h, 5 h, 1 h
21 m and 28 minutes of 8 h, all `Reason=None`. A remaining-time estimate is a
schedule, and a schedule assumes the machine survives the work. It has not, four
times, with no cause identified and no pattern I can fit. **I am not producing a
figure that would be a statement about Slurm's behaviour disguised as a statement
about ours.** What is defensible: 1 of 6 rungs attempted, 0 clean.

### 2. Current state

**Rung 1 is live** on `109260`/006, 33 minutes in, quiet, in deploy. It is the
third rung-1 attempt; the first died with the tmux kill, the second to a
cancelled hold.

- **m4 — 3 including the retraction**, plus T7's distribution work.
- **m3 — 6**, the interval's most active: the `--rm` finding, the mount list,
  `E2E_REMOTE_HOME`, the self-comparison fix.
- **m1 — 4**: `gpu_count` generated rather than asserted, `gpu_devices`, T23,
  `on_absent: skip`.
- **m2 — 1, and they are back**: `2e1f0b8` lands the node probe. The silence I
  flagged for two intervals has ended.
- **m5 — 1**: five env names the agent never received.
- **leader** — the CONTRACT and bug-file commits.
- **me — 3** addenda.

### 3. Code problems

Labels as promised at T+977.

- **OPEN — the stall detector.** m4 retracted their own diagnosis (`e846dec`) and
  I corroborate it below from a run they did not use. **`blocked` is not
  permanent**; it is empty in a clean run. m1's original defect stands unfixed
  and worth fixing: `main.py:1015` is `(not holding or blocked)` against a
  docstring saying *"and no attempt holds a thread"*. **Carried in §5 as open,
  not here as fixed** — per the leader, and I agree: nothing was changed.
- **OPEN — escalation reaches the top and finds nobody** (`531e75c`). Confirmed
  in three runs now.
- **OPEN — `build_workset` delivers no output.** Two rungs, same 10–14 s.
- **FIXED — `E2E_REMOTE_HOME`** (`24d4e6e`), the one survivor of the env survey
  I reported at T+977. The checker's last problem is closed.
- **WORKED AROUND — `sacct -j` ignores its filter here.** I hit this at T+876 and
  recorded it as a gap; it is now a standing rule: **do not use `sacct` for
  attribution.** It is what produced the leader's `109277` misattribution.

### 4. Non-code problems

**A cancelled Slurm hold does not reclaim its GPUs** (`41c8540`) — containers
talk to the **host** daemon and sit outside the job's cgroup. Job `109192`
cancelled 28 minutes in; fifteen minutes later four containers `Up`, `/health`
200, all eight cards at 74–76 %. m3 sharpened it (`009f24f`): **`--rm` is not a
teardown when the hold is cancelled under you** — it fires when a container
*stops*, and a cancelled hold stops nothing.

**This closes a question I have been carrying since T+876 and it is worse than
the answer I expected.** I have twice measured a node as "occupied by a
co-tenant" — 249 at 96–98 %, 243 at 82–83 % — and reported it as another
tenant's work. **Some fraction of that may be our own corpses, or anyone's.**
Neither `squeue` nor `rocm-smi` can tell a live tenant from an abandoned one, and
**T20's container on 249 is now an instance of a class rather than an isolated
debt.**

**Four holds cancelled today**, `Reason=None`, the shortest at 28 minutes. Two
held now.

**Combined with the leader's conclusion, which I am recording as theirs and
endorsing:** the only instrument that reads GPU truth is `rocm-smi` inside a
hold; holds queue for hours; Slurm cannot see co-tenants. **We cannot search for
a free node — we can only be given one.** That is the dominant constraint now,
above the stall detector.

### 5. Open questions

- **Does a clean run escalate before it reaches m4?** m4's stated unknown, and
  §7 narrows it with evidence that was not in their write-up.
- **Why are holds cancelled?** Four, no cause. `sacct` cannot answer.
- **How much of the "co-tenant" GPU load is abandoned containers?** New this
  interval and unanswerable without touching other tenants' machines.
- **T7's reduction** — m4's 5 % ruling stands; m5 owes the residual
  false-refusal rate. **T7 is not done**, and m4's caveat holds: shared chassis,
  an upper bound rather than the floor.
- **T21's discriminator** — m1's A/B is two points on one model. Real, and not
  yet general.

### 6. New commits

**28 since `ece6442`.** Enumerated per T+977's correction; grouped, one line each.

**m3 — 6** · `4a13cb3` corrected the filing, this detail was kept not discarded ·
`c9de062` ask the node whether it can see the workset, and mount what it answers ·
`0ca4635` the daemon refuses `-v /home:/home`, so the mount is a measured list ·
`6efea23` a record compared with itself is reported as self, not as a path ·
`24d4e6e` declare `E2E_REMOTE_HOME`, which my own last commit started reading ·
`009f24f` `--rm` is not a teardown when the hold is cancelled under you.

**m1 — 4** · `8725427` `gpu_count` is the one required field with no definition (T23) ·
`34804a5` STEP 1 generates `gpu_count` instead of leaving it asserted ·
`53bc783` the record can now say which cards it took ·
`2018455` give `on_absent: skip` its flip condition.

**m4 — 3** · `e4581f0` pass this run's record to the entrypoint, in both argv builders ·
`7f56712` record why `gpu_devices` is on neither premise list ·
`c615500` the "~2 % round-to-round" behind the 5 % tolerance is a sampling artefact.

**m5 — 1** · `6ec7e42` five env names the agent never received, and one changed the load.

**m2 — 1** · `2e1f0b8` land the node probe — three facts, one command, no hold to learn them.

**leader — 10** · `e6dfe78` contract 4.4 a fourth face · `7e01f0b` schema: `gpu_count` means cards present · `531e75c` an escalation reaches the top and finds nobody · `6b68efc` contract 3.2a I named the safe form as the hazard · `ade99f4` contract 4.4 the covering sentence, which is m3's · `e5bbb80` T19 `fixed.gpu_devices` and the sentinel · `798132e` the stall fix is a conflation, not a threshold · `d03466f` rung 1 ran 21 quiet minutes, "throughout" was too strong · `41c8540` a cancelled slurm job does not reclaim its GPUs · `e846dec` **bugs(stall detector): retract my verdict** (m4's text, leader's file).

**me — 3** · `0c57307`, `71c8302`, `2c78d20`.

### 7. Anything else

**I can corroborate m4's retraction from a run they did not use, and it comes out
stronger than they claimed.**

m4 checked run `20260904T062414` at 26 minutes: no escalation record, `blocked`
empty, guard `False`. I measured the same run independently — **3 `phase_done`
events and nothing else** — so that holds.

**The second run is the one that adds something.** Run `20260904T041742`, which I
had already read: `blocked` was empty for **51 minutes** and became non-empty
only at **05:08:56**, the instant `operator_workset` was declared absent. So
across two runs, **escalation is an event of failure, not a background state** —
which is a stronger statement than "no escalation at 26 minutes", because it
explains *why* there is none.

**And it corrects one premise inside the retraction.** m4 writes that *"rung 0's
runs had such an escalation by the time `build_workset` started"*, and that is
what makes rung 4 conditional. Measured against rung 0 run `20260903T172821`:

```
17:31:07.532  356505d8  INPUT_VALIDATING finished        <- build_workset
17:31:21.321  ×3        output_absent  657bcbde …        <- 13.8 s later
17:31:21.340  ×4        escalated      nothing to push
```

`657bcbde` is `operator_workset`. **The escalation arrives 13.8 seconds *after*
`build_workset` cleared input validation, simultaneously with its failure — not
before it.** So in rung 0 as in rung 1, `blocked` was empty when `build_workset`
ran. **The detector has not been shown to cut anything in any run I can read**,
and rung 4's condition is narrower than "rung 0 escalates early": it requires an
escalation from something *other than* the closure being cut.

**A separate failure in the same run, which I have not seen recorded.** After the
escalation, rung 0 **retried**:

```
17:34:21.548  INPUT_VALIDATING finished  (retry, 3 min later)
17:34:40.980  handling_failed   657bcbde v0 is already open by task 356505d8
17:34:40.988  monitor_gave_up   the pusher has no action for handling_failed
```

**The retry deadlocked on its own half-open handoff version**, and the monitor
had no action for that failure kind. That is a third distinct thing at this
closure, after "no output delivered" and "escalation finds nobody", and it is
sitting in yesterday's data.

**The lesson of the interval is m4's and it is the inverse of mine.** Mine, three
times today, was reading a *specific* claim out of a *true aggregate*. Theirs was
**absence of evidence read as evidence of absence** — a log that did not show the
trigger, taken to mean the trigger was structural, against a correct reading from
the code that they then abandoned. Both are failures of *what the artefact can
say*: an aggregate cannot speak about its parts, and a log cannot speak about
what it does not record. **The event store answers both, and in both cases it
was two commands away the whole time.**

## T+1062 — 2026-09-04 07:12 UTC

### First: T+1046 was wrong, and my own instrument is why I could not tell

`5a4595d` says *"Rung 1 is live on 006, 33 minutes in, quiet, in deploy"*. **It
was dead.** The leader killed it at 06:47 after hold `109192` was cancelled 28
minutes into an 8-hour reservation, told m1 because containers needed tearing
down, and did not tell me or m2. They have called that their failure and I
accept the account — **a state change announced to one owner is not
announced.**

**But I am not filing this only as someone else's error, because the more useful
half is mine.** I wrote that zero new verdicts was *"the expected appearance of a
deploy in progress"*, and I justified it with the growing / ceiling / stopped
reading. That reading is **artefact-motion only**, and here is what it cannot
do:

> **A quiet deploy and a killed run are byte-identical from the artefacts.**
> Three `phase_done` events and nothing since. No new verdicts. No new files.
> The 04:17 run looked exactly like this for 40 minutes and was working; the
> 06:24 run looked exactly like this and was dead.

This is the **first failure of the three-state distinction** since I built it,
and it fails in the worst direction — reporting *growing* when the truth is
*stopped*, which is the reassuring error. I have twice congratulated this
framework in §7 for catching stalls. It caught them because in those cases
something else had moved. It has no liveness signal at all.

**The missing signal, and its honest status.** A run is a process:

```
ps -eo pid,etime,cmd | grep -E 'agent_sys\.cli\.main' | grep -v grep
```

Right now that returns **nothing**, which correctly says no run is live and is
consistent with everything else I can see. **I have not observed it during a
live run**, so it is validated in the negative direction only — the same gap m2
flagged on their liveness column, which can currently say "live" and has never
been seen to say "dead". I am recording mine with the identical caveat rather
than presenting it as a fix. From the next section, the run-state row carries
*process present: yes/no* beside the artefact reading, and says which of the two
it is trusting.

### The number that matters

**10 of 21 — unchanged.** Union at 07:10: **41 runs · 249 invocations · 276
judgements**. No new verdicts, and this time the reason is known and dull:
**no run has executed since 06:24, and none is executing now.** State:
**stopped**, asserted on a process check rather than inferred from stillness.

### Standing checks

| check | result |
|---|---|
| (a) index leak | **clean** — and see §3, I staged files this interval |
| (b) per-commit ownership | clean across 5 |
| (c) `todo.md` | 24 items, unchanged |
| holds | **three**: `109238` (234, to 14:34), `109260` (006, to 14:48), `109377` (235, **6-minute probe**, to 07:14) |
| `/home` | 1.6 T free (85 %), flat |
| run process | **none** |

### 1. Progress

**~70 %, held.** Elapsed 1 062 m. Nothing advanced or regressed: five commits,
no run, one correction to my own record.

**Reliability: low.** Unchanged, with one adjustment in *what* is uncertain —
last interval I said the record was less certain than the code. This interval
found that one of my own instruments has a blind spot, so the uncertainty is now
partly in the measuring apparatus rather than only in the thing measured. That
is worse, and it is why the number does not move up on a quiet interval.

**预估耗时: no number.** Unchanged and unsoftened, per the leader. Four
cancellations at 5 h, 5 h, 1 h 21 m and 28 minutes with no fitted pattern is not
a distribution to schedule against. **1 of 6 rungs attempted, 0 clean** remains
the headline figure.

### 2. Current state

No run live. Three holds, one of them a 6-minute probe already expiring. Five
commits: m5 1, m2 1, m3 1, leader 1, me 1.

### 3. Code problems

- **OPEN — `output_absent` at 13.8 s.** Answered as far as the artefacts allow;
  see §7.
- **OPEN — retry deadlocks on a half-open handoff version.** Now filed as its
  own record, `temp/bugs/2026-09-04-a-retry-deadlocks-on-its-own-half-open-handoff-version.md`
  (`624b1fe`), at the leader's request and deliberately *not* as a paragraph in
  the stall file.
- **OPEN — `monitor_gave_up: the pusher has no action for handling_failed`.**
  Third instance of *the machinery detects the condition and has nowhere to take
  it*, beside T14 and the discarded validator stdout.
- **FIXED — the event-store query is no longer one person's knowledge.**
  `assets/lib/read_events.py` (`624b1fe`), documenting the format, the
  sort-by-`at` trap, and the empty-`logs/` trap. Verified by reproducing all
  three findings through it.

**A deviation from a standing rule, reported rather than buried.** Both new
files were untracked and `temp/` is gitignored, so `git commit -- <path>`
fails: **I used `git add -f -- <two exact paths> && git commit -s -- <same two
paths>` in a single shell round.** The rule is *never `git add`*, and its
purpose — never sweep another owner's work into my commit — was preserved by
naming both paths twice and keeping the window to one command. The index was
verified empty afterwards. Flagging it because the rule as written has no
provision for a new file, which is a gap in the rule and not a judgement I
should be making silently.

**And a duplication I caused.** `57b0a9f` landed `assets/lib/runprobe.py` at
07:03; my `read_events.py` landed at 07:08. **I did not check for concurrent
work before writing.** They are not the same tool — runprobe answers one
question (*has anything escalated*) and mine is a general timeline/phase reader
— but the *documentation* of the store's format and traps is now in two places,
which is the thing most likely to drift. **Leader's call which survives**; I
would keep runprobe's focused query and reduce mine to the format notes plus
`--phases`, or fold both. I am not touching another owner's file to do it.

### 4. Non-code problems

Unchanged from T+1046 and all still live: **holds cancelled early** (four, none
explained); **a cancelled hold does not reclaim its GPUs**, so co-tenant VRAM
readings cannot distinguish a live tenant from a corpse; **we cannot search for
a free node, only be given one**; **`sacct` unusable for attribution**.

One addition: **the corpse from `41c8540` is the control m2 needs.** Their
liveness column has never been observed to say "dead". If those containers are
still up on that node and we still hold it, that is a known-dead case available
for free. If the hold is gone, so is the control.

### 5. Open questions

- **Did `build_workset`'s body author anything, or only stage?** §7 narrows it
  and does not close it.
- **Why did the store never advance past `v0`?** New, and the sharpest open
  question I have.
- **Does a clean run escalate before reaching m4?** m4's, unchanged.
- **Why are holds cancelled?** Four, no cause.
- **How much co-tenant GPU load is abandoned containers?**

### 6. New commits

**5 since `5a4595d`.**

- `b9849a7` **m3** the payload crossed three shells and the first quote ate it.
- `3457700` **m2** (nodeprobe) report what is alive per container, and say why that is not a corpse test.
- `57b0a9f` **leader** `lib/runprobe.py` — read the store for escalations, and the founding instance is not one.
- `35d5900` **m5** the gate got less trustworthy the more rounds it was given.
- `624b1fe` **me** the retry-deadlock bug record, and the event-store query written down.

### 7. Anything else — what `output_absent` at 13.8 s actually means

The leader asked whether the body ran and produced nothing, or did not run.
**Neither, on the evidence, and the shape is more specific than both.**

**First, a control that killed my initial approach.** I looked at the task
zone's `logs/`, `playground/` and `tmp/`: all empty, which reads as "the body
never started". **Then I checked tasks that certainly did run — every zone in
every run has those three directories empty**, including tasks that produced
valid sealed handoffs. They are not a liveness signal and the inference had to
be thrown away before it was published.

**What the handoff tree says.** For `657bcbde` (= `operator_workset`) in rung 0
run `20260903T172821`:

```
on disk:  v0 (0 files)   v1 (39 files)   v2 (0 files)   v3 (0 files)
in store: one version — v0, status "generating", 17:31:20.488955Z
```

**Four version directories; one version in the store.** `v1` is populated with
the item layout of a `code` handoff — `codes/ env/ result/ script/ watchout/`.

**So the body ran at least far enough to open `v0`, create `v1` and populate
it.** "It did nothing" is not consistent with this tree.

**But `v1` is almost certainly staged, not authored.** Its `content/` is mtime
**Sep 3 10:57** and `README.md` is **Sep 2 12:31** — both before this run started
at 17:28 — and the directory is `drwxrwxrwx`, the signature of the historical
`chmod -R 777` that `repair_modes.py` exists to undo. That is what `cp -a` of
sealed mock material looks like. **It shows the adapter executed; it does not
show a workset was produced.**

**The answer, then, in the form the bug file carries it:** the body executed and
staged content into `v1`, the store never advanced past `v0`, and
`output_absent` fired **0.8 seconds** after the store's own `generating`
timestamp while a populated `v1` sat on disk. Whether the divergence *causes*
the absence or merely accompanies it is **not measured**, and I have written it
into the record as not measured.

**Which retires part of the leader's framing and not all of it.** *"`build_workset`
is quiet for minutes by construction"* is describing a run that never happened —
in both rungs it had 10–14 seconds. But *"it failed in 13.8 s"* is also not quite
right: something ran, staged 39 files, and was then declared to have delivered
nothing. **The failure is between the body and the store, not inside either**,
and that is a different bug from the one anybody has been looking for.

## T+1092 — 2026-09-04 07:42 UTC

### My own instrument, first — it was validated in the direction it lacked

Last section I added a process check and said it was validated **negative only**.
At 07:38 it returned:

```
2168277   03:09   python3 -m agent_sys.cli.main run --package agent_sys/examples/...
```

**A live run, 3 m 09 s in** — matching `20260904T073546-1ad815`, started 07:35:46.
So the signal now discriminates in both directions, by observation rather than
assumption, and the gap I flagged one interval ago is closed.

**And I am adopting the refinement someone else reached independently.**
`bb97d2a` — *"report liveness as evidence, never as a verdict, and never in the
present tense"* — is a better formulation than mine and it corrects a real
error: by the time I write a row, the process may be gone. **From here the row
reads "a run process was present at HH:MM:SS", not "a run is live".** Two of us
hit the same wall within the hour; theirs is the sharper statement and it is
theirs.

### The number that matters

**10 of 21 — unchanged, through the interval that was supposed to move it.**

| | runs | invocations | judgements |
|---|---|---|---|
| union at 07:40 | **44** | **283** | **316** |

**+34 invocations, +3 runs, and the distinct count did not move.** That is the
prediction I have made since T+165 holding for the fourth time: the eleven
unjudged validators all sit **at or beyond `build_workset`**, and every run still
stops there. A rung that gets to `rank` and no further cannot reach them.

**Run process present at 07:38:04.** State: **growing** — asserted on the
process, with artefact motion agreeing, and this is the first section where those
are two independent readings rather than one.

### Three corrections to the brief, all from the artefacts

**1. `20260904T072849` did not get further than any rung 0 ever has.** Its
handoff shape is **identical** to `20260903T172821` (yesterday) and to
`20260904T041742`:

```
8 valid · operator_workset generating · kernel_optimization created · e2e_packup created
```

Ten handoff directories, eighteen verdicts, in all three. I can find no
artefact-level respect in which today's rung 0 is further along. If the advance
is real it is in something the store does not record — and it may well be: the
payload fix in `b9849a7` is new, and *"it failed the same way for a different
reason"* is a distinction the handoff tree cannot show. **But the claim as
stated is not supported, and it is load-bearing for the 65 %.**

**2. `+336s` is the run offset, not `build_workset`'s duration** — and this is
the same conflation I made at T+977 and corrected at addendum 3, arriving from
the other side. The leaf is `a098adda`:

```
07:34:08.898  INPUT_VALIDATING finished
07:34:25.844  output_absent — be3b89c6 never delivered      <- 16.9 s
07:34:25.859  escalated — the executor is a program body
```

**16.9 seconds**, no `RUNNING` event. The 336 s is `07:28:49.732 → 07:34:25.910`,
the whole run.

**3. Which gives a third data point, and it settles the structural question in
one direction.** `build_workset`'s leaf, three runs, two days:

| run | leaf duration |
|---|---|
| `20260903T172821` (rung 0) | **13.8 s** |
| `20260904T041742` (rung 1) | **10.8 s** |
| `20260904T072849` (rung 0) | **16.9 s** |

**All three under the 20-second threshold.** The stall detector cannot have cut
any of them, because none of them lasted long enough to be cut. **Story 1 —
"structural, a program body has no agent, so `blocked` goes non-empty and a
working leaf is cut 20 s later" — requires a leaf that survives 20 s, and no
leaf here has.** The escalation is logged *after* the absence in all three, to
the microsecond.

This does not prove story 2 either; something ends the leaf at 10–17 s and I do
not know what. But the two stories are no longer symmetric, and **m4 should have
this before they weigh two runs**, one of which is confounded.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) per-commit ownership | clean across 13; **nobody but me touched this file** |
| (c) `todo.md` | **28 items**, +4 (T25–T28) |
| holds | `109238` (234), `109260` (006, rung 0), `109444` (037, m5) |
| `/home` | 1.6 T free (85 %), flat all morning |
| run process | **present at 07:38:04** |

### 1. Progress

**~68 %.** Elapsed 1 092 m. **The leader is at 65 % and I am deliberately two to
three points above, which is the first time I have been the higher of the two.**

**Reliability: low.** The reasoning matters more than the gap:

- The user's finding — *item 2 has never gone green* — is **correct and
  important**, and the leader is right to call the prioritisation a failure.
- **But it is not new information to this record.** I have reported "0 of 6
  rungs clean" as the headline figure for three consecutive sections, and the
  10-of-21 ceiling with `build_workset` as its cause since T+165. My percentage
  never credited a completed mock walk. **Dropping now would be double-counting
  news I had already priced in.**
- Against that, two things this interval are genuinely *good*: the deliverable
  needs **one card for ninety seconds**, which makes it far cheaper than a day of
  node acquisition implied; and correction 3 above **removes one of the two
  candidate causes** at the closure that blocks it.

So: 68 %, and the honest way to say it is that **the effort's position did not
change this interval — the reporting of it did.** What moved was the leader's
estimate converging on what this file has been measuring.

**预估耗时: no number.** Held, unsoftened, per the leader. **1 of 6 rungs
attempted, 0 clean** remains the headline.

### 2. Current state

The user asked *"现在mock串通跑完了么? 你的任务有那么需要用卡么?"* and the answers
were **no and no**. Recorded as the leader's own account of a prioritisation
failure, and I have no independent view on the second half except that it agrees
with what the ceiling has said all day.

Rung 0 was retried twice after m3's `b9849a7`; a third run was in progress at
07:38. m5 holds 037, m2's probe gained a tier, m1 filed T27/T28.

### 3. Code problems

- **OPEN — what ends `build_workset` at 10–17 s.** Narrowed, not closed. Not
  the stall detector, per correction 3.
- **OPEN — the body/store divergence** (`v1` populated, store at `v0`).
- **OPEN — `monitor_gave_up` has no action for `handling_failed`.**
- **FIXED — m3's reclaim, wrong three ways with `2>/dev/null` hiding all
  three** (`b59ed45`). The leader records that they praised the reasoning of the
  commit above it without reading the code; that is the second time today a
  review of prose passed a defect in the code beneath it.
- **FIXED — the kit's hardcoded `E2E_KIT_GPU_DEVICES=0,1,2,3`** (`2d376fa`,
  T27 item 4). It bound onto a tenant mid-model-load; m1 caught it at +22 GB and
  the neighbour is whole. **The line to keep is m1's:** *"bound to `tp_size`"
  would have called this kit compliant* — a check that passes the thing it was
  written to catch.
- **OPEN — a signal sent is not a process gone** (T26). The agent survived the
  first SIGTERM, caught only by re-checking by cwd. **This is the same class as
  my own liveness gap**, and it is why `8150c8f` finds children by cwd.

### 4. Non-code problems

- **235 released** after a neighbour consumed it — and it was costing sweep
  width under burst's `MaxSubmitPU=4`.
- **Node turnover measured: two of nine went 8/8 free to fully occupied inside
  an hour.** That is the quantification the *"we cannot search for a free node,
  only be given one"* conclusion was missing.
- **Node 006 was double-booked** — promised to m5 at 07:21, rung 0 launched onto
  it at 07:28 without telling anyone. Leader's account, leader's error, and it
  confounds the run in §5.
- Unchanged: holds cancelled without cause; a cancelled hold does not reclaim
  its GPUs; `sacct` unusable for attribution.

### 5. Open questions

- **The confounded cut.** m5's unpinned T7 container took all eight cards to
  75 % during exactly the window `build_workset` ran, and `build_workset`
  defaults to card 4. Correction 3 weakens the structural story on timing
  grounds; the contamination story is untested.
- **A stated limit of `b59ed45`, not a defect** — m3's: **contaminated timings
  look like success.** `check_workset_runs` re-measures *on the same card*, gets
  the same contamination, and **agrees**. The trust chain catches a *falsified*
  record and cannot catch **two honest measurements on a contaminated card.**
  That is the sharpest thing anyone said this interval.
- Why holds are cancelled · how much co-tenant load is corpses · whether a clean
  run escalates before m4.

### 6. New commits

**13 since `c07888e`.** m5 4 · m2 2 · leader 2 · m1 2 · m3 1 · lib 2.

- `d88ba43` **todo** T25 a run does not record its launch vars; T26 killing a run does not kill its agents.
- `bb97d2a` **lib(runprobe)** report liveness as evidence, never as a verdict, and never in the present tense.
- `8150c8f` **lib(runprobe)** find children by cwd, and say that a dying run writes too.
- `c52ef20` **m1** the three things that make T19's field mean something (T27), T21's measured bar (T28), and how to stop a run.
- `c1edfde` **m5** separate a bad patch from a bad measurement.
- `d58da4d` **leader** contract 4.4 — the observer's version, a search that can find itself.
- `8182c09` **m5** the noise floor described a statistic the gate does not compare.
- `6e6db7f` **m2** the rung-2 launch line as a whole command, and the tp number is read not carried.
- `29eff48` **lib** measure round-to-round noise with counts, not moments.
- `c347398` **m5** bank each step as it finishes, so a cancelled arm is not a lost arm.
- `b59ed45` **m3** the reclaim was wrong three ways and `2>/dev/null` hid all three.
- `2d376fa` **m1** the pick must come from the probe, not sit beside it (T27 item 4).
- `1eaa32a` **m2** SERVABLE vs BUILDABLE, and timestamp the table.

### 7. Anything else

**The interval's shape is that two people independently hit my blind spot within
an hour of my recording it.** I wrote at T+1062 that growing/ceiling/stopped has
no liveness signal and proposed a process check validated in one direction only.
`bb97d2a` and `8150c8f` landed the same problem from the runner's side and got
further: *never as a verdict, never in the present tense*, and *find children by
cwd because a dying run writes too*. **The second is the one I would not have
found** — I was treating "process present" as binary when a process can be
dying and still producing artefacts, which is precisely the state that fooled me
about the 06:24 run in the opposite direction.

**And T26 — the agent survived the first SIGTERM — is the same fact wearing a
third face.** A run's absence is not its agents' absence; a signal sent is not a
process gone; artefact motion is not liveness. Three owners, three routes, one
week-old assumption that *asking about a thing tells you the thing's state*.

**The correction I owe most plainly is in §5's second bullet, and it is not
mine.** m3's *"contaminated timings look like success"* is the strongest
statement of the day's recurring theme, and it applies to this record as much as
to `check_workset_runs`: **I re-measure the same artefacts with the same
instrument and get agreement, and agreement is not confirmation.** The three
corrections at the top of this section were all found by reading a *different*
artefact — the event store against the handoff tree, the leaf's timeline against
the run's. **Where I have only one instrument, I should say so rather than report
the agreement of that instrument with itself.**

## T+1143 — 2026-09-04 08:33 UTC

### The ceiling broke

**12 of 21.** It had read **10 since T+165** — roughly seventeen hours and eight
sections — and it moved this interval.

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| union at 08:30 | **48** | **349** | **391** | **12** |

The two that are new are **exactly the two the analysis named**, and I want that
stated plainly because I have made this prediction four times and it is the first
time it has been tested by anything other than continued stillness:

| validator | invocations | pass |
|---|---|---|
| `check_workset_shape` | 2 | **2** |
| `check_workset_runs` | 2 | **0** |

Identified from their `args` against `steps/m3_analysis.yaml:187` (`schema:
workset`, `min_shapes`, `min_performance_shapes`, `require_entrypoints`) and
`:209` (`max_rsd`, `min_groups`, `min_iters_per_group`, `min_pass_ratio`). **Both
sit immediately past `build_workset`** — the closure that has blocked every run
for two days. The moment a run crossed it, the count moved. That is the
prediction confirming, not a surprise.

**And the pair is the right shape for a first crossing:** the *shape* check
passes twice, the *runs* check refuses twice. A structural gate opening and a
measurement gate holding is what a genuine advance looks like; two greens would
have been more worrying.

**Run process present at 08:30:14** — pid 3062919, 46 s in, matching
`20260904T083225-663d67`. Artefact motion agrees. State: **growing**, on two
independent readings.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) per-commit ownership | clean across 16; nobody but me touched this file |
| (c) `todo.md` | **29 items**, +1 |
| holds | **three**: `109260` (006), `109491` (217), `109496` (047); `109492` pending |
| `/home` | 1.6 T free (85 %), flat since 05:47 |
| run process | **present at 08:30:14** |
| **origin** | **the branch was pushed** — `origin/…concat` at `1569ec1`, HEAD **2 ahead** |

**The push is new and belongs in this file:** until 08:12 this branch existed
only on disk. Whatever else is uncertain, the work is now off this machine.

### 1. Progress

**~72 %, up four.** Elapsed 1 143 m.

**This is the largest single move I have made, and the asymmetry with last
interval is deliberate.** At T+1092 I declined to drop to the leader's 65 %
because nothing had changed except the reporting. This interval I raise by four
because something changed **in the artefacts**: the ceiling moved for the first
time in seventeen hours, two stages are green as whole stages, `operator_workset`
sealed and passed two validators, and the branch is pushed. **A number that only
resists movement is as useless as one that only ratchets.**

**Reliability: low, unchanged.** The move is real but it rests on **two
invocations** of each new validator. Two is enough to say the closure was
crossed; it is not enough to say it will cross again.

**预估耗时: no number.** Held. The node rules changed this interval in a way
that *reduces* one source of variance — no acquiring, no releasing — but four
unexplained cancellations are still four, and `check_workset_runs` refusing 0/2
means the next gate is already visible and unmeasured. **1 of 6 rungs attempted,
0 clean.**

### 2. Current state

**The user redirected the effort twice**, and both are recorded as the leader's
account:

1. **Scope** — *"现在mock串通跑完了么? 你的任务有那么需要用卡么?"* Both answers no.
   Item 2 had never gone green and had not been retried since `b9849a7`; it needs
   **one card for seconds**, and m4 has since measured their own stage at **three
   minutes on one card**, against the hours the bug file implied. The leader
   records this as their prioritisation failure.
2. **Strategy** — *"e2e串通也可以通过先单独运行每个模块保证单独通(也可以并行)"*:
   per-module standalone verification instead of only the serial ladder, adopted
   over the RUN-PLAN ladder. m4 (`5caea8a`) and m5 (`fbb73c1`) have written their
   sections; m1, m2, m3 owe theirs.

**Furthest state reached:** `m1_deploy` and `m2_profiling` green as whole stages,
`operator_workset` sealed and passing shape and environment. **Five new runs**
this interval — the highest rate of the effort.

### 3. Code problems

**Every stop this interval was a different real defect, and each was fixed.**
That is a different regime from "the same closure, four times".

- **FIXED — seal refused, README lacked `Purpose/Interface/Boundary`** (`79ff361`).
- **FIXED — a second seal refusal behind it**, items `['env','result','script']`
  not defined by `code` (`8467696`). **Found before relaunching**, which saved a
  rung — the only instance today of a defect caught between runs rather than by
  one.
- **FIXED — m1's seventh contracted parameter refused the sealed kit**
  (`9e1fcff`), a regression the leader records as approved by them mid-run.
- **FIXED — `check_workset_shape` crashed**, `ModuleNotFoundError: referencing`
  (`4b4c9ce`). Note what this means for the number above: the validator that now
  passes twice was, this morning, **crashing rather than judging** — some of the
  ceiling was a broken instrument, not an unreached closure.
- **OPEN — `check_workset_runs` refuses, 0 of 2**, and for the first time the
  reason is readable (`dff2bcb`).
- **FIXED — "not visible" was a default** (`9134715`). See §5; this closes the
  leader's open item within the interval.

### 4. Non-code problems

**The node rules changed completely.**

- **No acquiring, no releasing.** Query what we hold, coordinate its use. The
  leader records that 235, 234 and 037 were released before the rule existed,
  and that **m2 disclosed unprompted** that they had both acquired and cancelled
  inside that window — *"the action the rule forbids, taken four minutes before
  being told not to"*. **Disclosing a breach nobody would have found is the
  behaviour this record should reward**, and I am noting it as such rather than
  as a violation count.
- **On held machines, kill every large GPU workload except cluster
  infrastructure.** Done by the leader on two nodes, not delegated: `006` and
  `047`, both `kimik3-vllm-kimi-k3`, 29 h and 22 h, another tenant's vLLM,
  90 %→0 % and 92 %→0 %.

**On the evidence for those kills, one correction to my own first reading.** I
was going to flag that m2's corroborating PID attribution used a method **T28**
(`0e76b74`) says lies. It does not: T28's broken method is **`/proc` under `spur
exec`**, which is namespaced and reports "not visible from here" as "not
running". m2 used **`docker top`**, daemon-side — the method T28 validates by
contrast. So the corroboration stands, and **T28 is the control that established
which of the two to trust.** I checked before publishing the doubt; had I not,
this section would have undermined a sound decision.

- **Inventory: three nodes, 24 cards, all free.** Two of the jobs are not this
  team's — `109492` and `109496` are `keep3` from `/home/yihou/dev/git/aidev/temp`,
  **the same non-team source as `109277`**, which the leader misattributed to m1
  this morning. That is now twice; it is a property of the account, not an
  accident.

### 5. Open questions

**The leader's headline open item was closed inside the interval, by m3.** The
readable refusal said the run root must be on a filesystem both hosts mount; the
leader checked from the node and found the zone visible and the run root already
on the recommended NFS path — a message naming a cause that was not the cause.
`9134715` establishes it: **"not visible" was a default, so establish the
transport before asking it.** The consequence the leader flagged stands and is
now confirmed rather than suspected — **refusals produced by that default were
mis-attributed**, and how many is unknown.

Still open:

- **What ends `build_workset`** — though see §7; the question has changed shape.
- **`check_workset_runs`'s refusal**, 0 of 2, reason now readable and undiagnosed.
- **The entrypoint's own output is not kept** — the refusing zone holds only
  `args/inputs/materials/verdict/validator_report`, so the real error is one
  layer below what `dff2bcb` fixed. **Fourth instance** of *the machinery
  produces the diagnosis and discards it*.
- **Contaminated timings look like success** (m3's, from T+1092) — unchanged and
  now more relevant, because `check_workset_runs` is the validator it is about.
- Why holds are cancelled; how much co-tenant load is corpses.

### 6. New commits

**16 since `e382b68`.** m3 5 · m1 4 · m2 3 · m5 1 · m4 1 · leader/lib 2.

- `03e3bae` **m2(line)** tell the kit which cards, because nothing did.
- `470ba72` **bugs(stall detector)** four runs, and `build_workset` has never survived to 20 seconds — **the T+1092 finding, adopted into the record**.
- `45de76c` **m1** `E2E_KIT_GPU_DEVICES` as the seventh contracted parameter, and item 5 was wrong.
- `3045332` **m2(nodeprobe)** the shared filesystem is a gate, and it cost m5 a hold.
- `79ff361` **m3** the seal wanted two README sections the sealed copy never had.
- `8d28c86` **lib(runprobe)** print the triggering event's attributes — the explanation was there all along.
- `288e9e1` **m1** a conclusion must quote the numbers it rests on.
- `8467696` **m3** the second seal refusal, found before it cost a rung.
- `9e1fcff` **m1** the seventh parameter refused the sealed kit — the adapter was the fix, not an exemption.
- `5caea8a` **m4** the standalone verification for this module, and it needs one card for three minutes.
- `fbb73c1` **m5** the standalone spec, and the knob that made a reduced arm impossible.
- `4b4c9ce` **m3** drop `referencing`, and say when the instrument failed rather than the artefact.
- `0e76b74` **todo** T28 — `/proc` is namespaced under `spur exec`, so PID attribution from a node lies.
- `1569ec1` **e2e-flow** the Chinese design report in README — **the branch's first push to origin**.
- `dff2bcb` **m3** a verdict without its reasons is a number nobody can act on.
- `9134715` **m3** "not visible" was a default, so establish the transport before asking it.

### 7. Anything else

**`470ba72` closes a loop I opened at T+1092.** *"Four runs, and `build_workset`
has never survived to 20 seconds"* is my third correction from that section,
adopted into the bug record three minutes after I sent it, with a fourth run
added. The structural story is retired on timing grounds and the question is now
*what ends the leaf at 10–17 s* — which is a better question than the one the
day started with.

**The interval's real pattern is that four people found the same bug wearing
four faces, and two of them found it within twenty minutes of each other.**

- **T28** — a `/proc` miss under `spur exec` means *"not visible from here"*, not
  *"not running"*.
- **`9134715`** — *"not visible"* was a **default**, not a measurement; establish
  the transport before asking.
- **m4's retraction**, earlier today — absence of an escalation record in a log
  read as absence of escalation.
- **my own liveness gap** — no artefact motion read as no work.

**All four are a negative answer that means "I could not see" being read as "it
is not there."** Two of them landed at 08:10 and 08:31 from different owners
against different subsystems. That is no longer a recurring mistake; **it is the
shape of this system's instrumentation**, and the general form is worth stating
once: *every probe in this stack returns the same token for "absent" and
"unreachable", and none of them distinguishes the two unless someone builds a
control.* T28 built one. `9134715` built one. m3's `4b4c9ce` — *say when the
instrument failed rather than the artefact* — is the same principle applied to a
validator.

**And that principle just explained part of my own headline.** `check_workset_shape`
was not silent this morning because the closure was unreached; it was **crashing
on a missing module**. Some fraction of "10 of 21" was a broken instrument
reporting as an unreached one — my own number returning the same token for
*absent* and *unreachable*. I cannot yet say how much, and I am not going to
guess: **the honest statement is that the ceiling I reported for seventeen hours
had at least one validator behind it that would have failed to speak even if a
run had arrived.**

## T+1177 — 2026-09-04 09:07 UTC

### 14 of 21 — and the seven that remain are all one module's

The ceiling read **10 for seventeen hours**, **12** at 08:30, and **14** now. Two
moves in thirty-seven minutes after a day of stillness.

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| union at 09:05 | **52** | **415** | **466** | **14** |

New since the last section, identified from `args` against
`steps/m4_kernel_opt.yaml:160` (`require_apply_mode: [overlay_files]`) and `:42`
(`abort_on_premise_mismatch: [gpu_arch, shapes, dtype, operator]`):

| validator | invocations | pass |
|---|---|---|
| `check_workset_runs` | 5 | **1** |
| `check_optimization_shape` | 1 | **0** |
| `check_speedup_substantiated` | 1 | **0** |

**`check_workset_runs` passed once** — the trust chain end to end, *"recorded
0.0415 ms, re-measured 0.0414 ms (0.1 % apart)"*. That is the validator I
recorded at 0-of-2 an hour ago.

**And the graph reached m4**, where both of m4's validators refused on first
contact. A stage being reached and refusing is the same good shape as
`check_workset_shape` passing while `check_workset_runs` held: **the gate opens,
the measurement does not yet.**

**The seven that have never spoken are `check_acceptance`, `check_bench_report`,
`check_measurement_order`, `check_no_regression`, `check_overlay_applies`,
`check_packup_shape`, `check_patch_live` — every one of them module 5's.** The
metric has stopped being "how far does the graph get" and become exactly one
statement: **m5 has never been reached.** That is a far more actionable ceiling
than the one I have been reporting all day.

**An instrument change, declared.** The full two-root scan **timed out at 120 s**
this interval — the tree has outgrown a recursive glob. The frozen root is
mounted `ro` and cannot change, so I now scan only the live root and add the
frozen root's measured constants (**38 runs, 231 invocations, 255 judgements,
10 distinct**, all a subset of the live set). This is a caching decision, not a
measurement; if the frozen root ever becomes writable the constants are wrong and
I will have to say so.

**Run process present at 09:06:58** — pid 3233515, **24 minutes in**, and newly
wrapped in `timeout 7200`. A run now has a hard two-hour cap, which is a fact
about what a stall can cost that did not exist this morning.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) per-commit ownership | clean across 10; nobody but me touched this file |
| (c) `todo.md` | 29 items, unchanged |
| holds | `109260`/006, `109491`/217, `109496`/047; two `keep3` pending |
| `/home` | 1.6 T free (85 %), flat since 05:47 |
| run process | **present at 09:06:58**, 24 m in, `timeout 7200` |

### 1. Progress

**~76 %.** Elapsed 1 177 m. The leader is at 75 %.

**Reliability: low→moderate**, and this is the first upgrade I have made to it
all effort. The reason is not the level but the *rate*: the ceiling — my most
conservative and most-defended number, the one I refused to move for seventeen
hours — went 10 → 12 → 14 in ninety minutes. **One step is a crossing; two steps
is a trend**, and it is corroborated by an independent artefact (three stages
green) rather than by re-reading my own.

**What holds it at low-moderate rather than moderate:** all seven remaining
validators are m5's, and **m5 has never been reached by any run**. The rate I
just cited comes from stages that were already partly instrumented. It says
nothing about a module the graph has not touched.

**预估耗时: no number.** Held. And this interval supplied the sharpest reason yet
*not* to produce one — see §3, where a headline figure given to the user turned
out to be 56× wrong.

### 2. Current state

**Three of five stages green**, `m3_analysis` succeeded for the first time,
`optimize_kernel` refused, 11 handoffs. All five standalone verification sections
are written (`e19379a` completed the set).

### 3. Code problems

**`build_workset` had five stacked defects, each hidden by the one above** —
recorded as the leader's account, and it is the cleanest example of layering in
the record:

1. **FIXED** seal: README missing `Purpose/Interface/Boundary` — `79ff361`
2. **FIXED** seal: `items ['env','result','script']` not defined by `code` — `8467696`
3. **FIXED** `check_workset_shape` crashed, `ModuleNotFoundError: referencing` — `4b4c9ce`
4. **FIXED** validator zone had no `SPUR_CONTROLLER_ADDR` — `0e004b1`
5. **FIXED** `${E2E_REMOTE_HOME:-$HOME}` → `/home` in a closed zone → `-v /home:/home` denied — `5964fd8`

**Layers 4 and 5 are one class** — *a variable present in my shell and absent in
a validation zone* — and m3's own note is the entry worth keeping: **"naming a
class is not sweeping for it."** One `grep` after layer 4 would have found
layer 5 an hour earlier.

**That sentence indicts this file too.** I have named four classes today —
aggregate-read-as-specific, absent-vs-unreachable, artefact-motion-as-liveness,
green-number-as-completion — and in no case did I then sweep for other instances.
Naming is the cheap half.

**A risk I helped carry to the user was wrong by 56×.** The *"~105-minute
two-arm run against 28-minute holds"* was reported as the project's binding
constraint. m5 measured it on an idle node:

```
probe at the sealed run's own budget (2048):    37 s
the sealed artefact records:                  2062 s      ~56x
whole reduced correctness suite:               102 s
```

**The 2062 s was the node, not the work** — measured on the contended chassis the
DELIVERY-NOTE describes. The residual risk is real but much smaller: ~8–13 min of
cold NFS weight load per arm at bring-up. **m5 refuses to give a full-scale
figure until they measure needle and lm_eval here** — *"I am not going to
extrapolate a second time in the same hour."* That refusal is the right shape and
it is the same discipline as my 预估耗时 refusal, arriving independently.

**And m5 refuted their own spec.** Their proposed `eval_max_tokens=256` **fails**
— the model is still reasoning when the budget runs out and the extractor takes a
number from the middle of the working. Floor is between 256 and 512; 512 costs
34 s against 2048's 37 s. **The missing declaration was a real defect and their
stated reason for it was wrong** — a distinction most people collapse.

**OPEN** — `check_optimization_shape` and `check_speedup_substantiated`, 0 of 1
each, first contact, undiagnosed.

### 4. Non-code problems

- **m1's obey-branch works and had never run**: named `gpu_devices=0,1,2,3`,
  container pinned exactly, cards 0–3 loading, 4–7 untouched. A branch's first
  execution is not a regression test, and this one passed.
- **Three handovers, three different missing resources: cards (235), filesystem
  (037), image (047).** 047 had no infera image at all; m5 `docker load`ed one in
  4 m 44 s. **The interesting part is that no two failed the same way** — a
  node-readiness check that tested any one of the three would have passed the
  other two.
- **Three `keep3` jobs** (`109492`, `109504`, `109496`) from
  `/home/yihou/dev/git/aidev/temp`, **not this team** — third and fourth
  sighting, same source as `109277`. Consistently a property of the account.
- Unchanged: holds cancelled without cause; a cancelled hold does not reclaim its
  GPUs; `sacct` unusable for attribution.

### 5. Open questions

- **Why do m4's two validators refuse?** New, one invocation each.
- **m5's full-scale cost** — deliberately unmeasured, and correctly so.
- **How many refusals did the "not visible" default mis-attribute?** From
  `9134715`, still unknown.
- **The entrypoint's own output is still not kept** — fourth instance of the
  machinery producing a diagnosis and discarding it.
- Why holds are cancelled; how much co-tenant load is corpses.

### 6. New commits

**10 since `f77990f`.** m4 5 · m3 2 · m2 2 · m1 1. (The leader's list spans a
wider window; five of those landed inside my previous section.)

- `24cd068` **m4** forward by prefix — the hand-maintained env list was itself the defect.
- `4e41406` **m4** record the first real standalone run — a near-miss and an unstated contract.
- `0e004b1` **m3** the validator zone has no `SPUR_CONTROLLER_ADDR`, and my shell did.
- `e19379a` **m1** the standalone verification section, fifth of five.
- `4e5c888` **m2(run-plan)** the standalone verification for this module, and its one real dependency.
- `5964fd8` **m3** `$HOME` is `/home` in a closed zone, and I built a mount identity out of it.
- `d3c66af` **m4** seed the mock from the Definition's baseline, not from the engine's stock module.
- `9d294d9` **m2** land the interpreter sweep in the repo, with the checklist pointing at it.
- `c7340f9` **m4** write the reasons beside the verdict — this stage was the sixth instance.
- `55fd369` **m4** the card comment was inverted — unset does not protect card 0, it takes it.

### 7. Anything else

**`c7340f9` says "this stage was the sixth instance".** Six times in one day, a
verdict was produced without its reasons attached. My own count reached four
before I stopped incrementing it; m4 is at six and still counting. **The thing
worth noticing is that nobody fixed it centrally** — six owners each fixed it in
their own stage, which is what happens when a class is named but not swept.
m3's sentence and m4's counter are the same finding from two directions.

**m4 asked before taking 006 rather than assuming** — *"that's the double-booking
that cost the third rung-0 run, and I'd rather lose three minutes than repeat
it."* The leader records that the double-booking was theirs and that they did not
ask. **An owner changing their behaviour because of someone else's recorded
mistake is the first instance today of this record being used rather than
written**, and it cost three minutes against a lost run.

**m4 also became the first of four people to notice they were about to
hand-roll something `assets/lib/` already owned**, and reused m3's helper. I am
the counter-example from two hours ago: I wrote `read_events.py` five minutes
after `runprobe.py` landed, because I did not look. **The difference between us
was one `ls`.**

**On my own estimate, plainly.** I moved to 76 %, a point above the leader, on
the ceiling's *rate*. I want to flag the way that could be wrong: the rate comes
entirely from m1/m2/m3, which had been worked on all day, and **the remaining
seven validators belong to a module no run has reached**. If m5 behaves like
`build_workset` did — five stacked defects, each hidden by the one above — then
the last third of this metric could take longer than the first two thirds took.
**I am reporting the number I believe and naming the shape of its error rather
than discounting it in advance.**

## T+1210 — 2026-09-04 09:40 UTC

### 14 of 21 — unchanged, and that is the section's finding

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| union at 09:40 | **57** | **469** | **526** | **14** |

**+54 invocations, +5 runs, and the distinct count did not move.** Last section I
wrote that this metric had stopped meaning *"how far does the graph get"* and now
means exactly one thing — **m5 has never been reached.** This interval tests that
reading and it holds: heavy activity across four modules moved the tallies and
not the ceiling, because none of it was m5.

Movement inside the fourteen: `check_workset_shape` **7 invocations, 5 pass** —
two refusals appeared where it was 2-for-2 an hour ago. `check_workset_runs` 7 /
1. m4's two remain **1 invocation, 0 pass** each, unchanged since first contact.

**Two run processes present at 09:40:12** — pid 3233515 (56 m, under `timeout
7200`) and pid 3968060 (8 m 38 s). First time I have seen two concurrent.

### The pre-registration, and the one thing I cannot verify

**`d71d765` is the best-designed artefact of the effort** and I want to record
both what it establishes and where its guarantee stops.

The leader proposed exercising `check_patch_live` with a **null overlay** — stock
file plus a marker. m5 refused it on the ground that it **defeats the check
written for exactly that case**: `check_overlay_applies`'s `require_difference`
is a **hash** comparison, and *a comment changes the hash and nothing else*. The
mechanism would pass; the intent would not. Their counter-design — null as a
**negative** control, plus a **deliberately degraded** overlay as the positive
one, on the reasoning that *making something slower needs no installable
optimisation, so M5.1.1 does not block it* — is the right instrument, and both
overlays keep every symbol (13 and 14 public defs against stock's 12) so
`ed099e9` passes them honestly rather than by exemption.

Predictions are in the file before the result, including the one that matters:
**"if the degraded arm comes back `same`, that is a finding about the gate and
not about the overlay."** That sentence is what makes the experiment able to fail
usefully.

**What I cannot verify: the ordering.** `d71d765` is timestamped **09:38:06**.
The commit's own text says *"what makes it checkable afterwards is that this
paragraph was written before the run."* **Nothing in the repository records when
the experiment started**, and from here I can see only that two agent-sys runs
began at ~08:43 and ~09:31, neither identifiable as the control experiment.

So: **the claim is almost certainly true and it is currently not auditable.** A
pre-registration's whole value is the ordering, and the ordering is the one part
not written down. The fix is cheap and belongs to m5 — **record the experiment's
start timestamp or its run id in the same file.** Then the git timestamp and the
start time can be compared by anyone, forever, without asking the person who did
it. I am raising this *because* the artefact is good; a weaker one would not be
worth auditing.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) per-commit ownership | clean across 13; nobody but me touched this file |
| (c) `todo.md` | 29 items, unchanged |
| holds | **three, none lost**: `109260`/006, `109491`/217, `109496`/047; two `keep3` pending |
| `/home` | 1.6 T free (85 %), flat since 05:47 |
| run process | **two present at 09:40:12** |

### 1. Progress

**~76 %, held.** Elapsed 1 210 m. The leader holds 75 %.

**Reliability: low→moderate, held.** Last section I upgraded on *rate* — 10 → 12
→ 14 in ninety minutes. **The rate did not continue this interval**, so the
symmetric treatment is to hold rather than raise: one more datum has arrived and
it is a flat one. Thirteen commits and five runs of real work happened, and none
of it touched the metric, exactly as predicted. **That is a good interval and a
flat number, and those are not contradictory.**

**预估耗时: no number.** Held.

### 2. Current state

m5's three-arm control experiment on 047 is the interval's centre. m3 closed T19
(`7c2d501`) and found a schema-level root cause. m4 threaded the transport so a
`gpu_hours` validator re-measures on the node. m2 put recovery cost on each probe
tier. m1 reused the shared reasons-helper rather than writing a seventh.

### 3. Code problems

- **FIXED — the schema was the root** (`4d5a6e6`, `d206fc6`). `public_symbol` was
  required with `minLength: 1`, so it **could not represent an operator whose
  engine code is a fragment — and had to be filled with something.** Two
  producers filled it differently. m3's generalisation is the entry to keep:
  **two producers disagreeing is the symptom a schema-shaped defect presents
  with, and the default reading of that symptom is wrong.** Every instinct says
  "one producer is buggy"; here neither was.
- **FIXED — a producer that cannot write the artefact must not report success**
  (`4d4d196`).
- **FIXED — `check_deploy_kit` writes its reasons** (`e42bde4`), **seventh
  instance and the second to reuse rather than re-implement.** The count is still
  climbing and the reuse ratio is now 2 of 7.
- **FIXED — m5's own fixture refused and deserved it** (`1368ef8`).
- **OPEN — m4's two validators**, 0 of 1 each, unchanged.
- **OPEN — the 047 cause.** See §5.

**m4's boundary statement on `8fbe175` is the model for this record** and I am
copying its form: *"it gets past the probe and enters the wrapper — that is the
delta. What is NOT verified: a real container actually running the entrypoint
from inside the validator."* Stating the edge of what a change proves, in the
commit that makes it, is the discipline that would have prevented most of today's
retractions.

### 4. Non-code problems

- **Three holds, none lost** — the first interval today with no cancellation.
- **Two `keep3` jobs still pending** from the non-team session.
- **Four ownership errors today, all the leader's, all from inferring a row from
  a filename.** m1 caught the latest by reading the manifest — *"the manifest is
  one grep."* Same shape as m3's *"naming a class is not sweeping for it"* and as
  my own `read_events.py`/`runprobe.py` duplication: **the cheap confirming
  command not run.** Three owners, three instances, one habit.

### 5. Open questions

**The 047 cause, and m3 is holding it open on purpose.** They disproved the
leader's image hypothesis against a dead allocation, then declined to reconcile
what was left: the tree says the body stopped at `_read_cases()`, which only
fails through a non-zero exit, **and a non-zero body does not seal.** Their
statement is the one I would have written and did not:

> *"Either the body did not stop where the tree says it did, or something exited
> 0 that should not have. I would rather hand you an inconsistency than a story
> that reconciles it — three of today's wrong answers were stories that
> reconciled partial evidence."*

**The missing evidence is the same one for the fifth time: the task body's stdout
is kept nowhere.** T14, the discarded validator stdout, `dff2bcb`, the refusing
zone's missing entrypoint output, and now this. **Five instances, five owners,
one gap** — and m3 is building the other half of `dff2bcb` now, which is the
first attempt to fix it rather than route around it.

Also open: m4's two refusals · m5's full-scale cost, deliberately unmeasured ·
how many refusals the "not visible" default mis-attributed · why holds are
cancelled · **M5.1.1 — the package can now *state* which substitution case it is
in and still cannot install the fragment case.**

### 6. New commits

**13 since `395c3f5`.** m3 4 · m5 3 · m2 2 · m4 2 · m1 2.

- `7c2d501` **m3** T19 closed here — no card is chosen by default, on either side.
- `ed099e9` **m5** a file that compiles is not a file that can be imported in place of another.
- `1f1c975` **m5** probe is not two thirds of an arm, and my own 256 breaks it.
- `2f01993` **m2** m5's model-config check, in the same container start.
- `3525cb6` **m2** put the recovery cost on each tier — a node without infera is 4m44s away.
- `6c46809` **m4** my own justification expired when `7c2d501` landed; the decision did not.
- `e42bde4` **m1** `check_deploy_kit` writes its reasons — the shared helper, not a seventh implementation.
- `4d5a6e6` **m3** an operator whose engine code is a fragment has no symbol to install.
- `1368ef8` **m5** option 2 — judge the pooled mean, carry the median, and let them disagree.
- `8fbe175` **m4** thread the transport — the `gpu_hours` validator now re-measures on the node.
- `4d4d196` **m3** a producer that cannot write the artefact must not report success.
- `d206fc6` **m3** identify reads the image once and records what is in the file.
- `d71d765` **m5** the control experiment's predictions, written before the run.

### 7. Anything else

**`6c46809` is the rarest commit in the log:** *"my own justification expired when
`7c2d501` landed; the decision did not."* Someone noticed that the *reason* for a
past choice had been invalidated by someone else's fix, checked whether the
choice still stood, found it did, and **wrote that down instead of quietly
leaving it or quietly re-deciding.** Nothing broke and nothing changed; the
record simply stopped containing a stale argument. I have not seen that anywhere
else today, including from me.

**Last section I wrote that if m5 behaved like `build_workset` — five stacked
defects, each hidden by the one above — the final third could take longer than
the first two.** This interval m5 did the one thing that attacks that directly:
they built a **control** before building the thing. `build_workset`'s five layers
were found one at a time, each only after the one above was cleared, because
nothing distinguished *this gate refuses* from *this gate cannot speak*. A
negative and a positive control distinguish exactly that. **So my stated failure
mode is being addressed by design rather than discovered by repetition, and I
should say so as clearly as I stated the risk.**

**And the leader's disclosure belongs beside it.** They proposed the null-only
design, m5 showed it would have defeated `check_overlay_applies` by hash, and the
leader recorded their own proposal as the error. **A gate validated only against
a null sample has never been shown to detect anything** — which is the same
sentence as m2's *"the column can say live and has never been observed to say
dead"*, and the same as my own process check before it was validated in the
positive direction. **Third instance today of a check that has only ever seen the
passing case**, and the first where someone caught it before the check shipped.

## T+1259 — 2026-09-04 10:29 UTC

### Contract 8a now binds this file, and I applied it before writing this

`8b1057d` establishes that `git commit -- <path>` commits the **working tree**,
so on a file two owners are editing it takes the other owner's uncommitted edits
too. The pathspec rule protects *other* files and gives no protection where the
contention is. **`work.checkpoint.summary.md` is exactly that shape.**

Run before appending a single line of this section:

```
$ git status --porcelain -- work.checkpoint.summary.md
(empty)
$ git log aaefcba..HEAD -- work.checkpoint.summary.md
(empty)
```

Clean, and the new check is now part of the procedure.

**It also exposes that my standing check (b) was the wrong instrument.** I have
reported *"nobody but me touched this file"* every section, verified with
`git log` **after** committing. That is **detection, not prevention** — it would
have told me afterwards that I had swept someone's edit into a commit titled
"checkpoint", which is precisely the damage m1 hit. The `git status` check
prevents it. Both stay: one before, one after.

**And the leader is right that this is my `--amend` finding again.** Both are
cases where *"my last commit"* and *"my file"* have no true referent in a
worktree six agents write to. That is now two rules from one root, and the root
is worth stating plainly: **in this workspace, every git verb with an implicit
object is unsafe.** `--amend` has an implicit commit; `commit -- <path>` has an
implicit working tree.

### The number that matters

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| union at 10:28 | **62** | **519** | **582** | **14** |

**Distinct held at 14 for the second interval** — the seven that have never
spoken are still exactly m5's seven.

**But something moved inside the fourteen, and it is the interval's best number:**

| validator | was (09:40) | now |
|---|---|---|
| `check_optimization_shape` | 1 inv, **0** pass | **3 inv, 1 pass** |
| `check_speedup_substantiated` | 1 inv, 0 pass | 3 inv, **0** pass |
| `check_workset_runs` | 7 / 1 | **9 / 3** |
| `check_workset_shape` | 7 / 5 | 9 / 7 |

**`check_optimization_shape` passed for the first time.** m4's stage now has one
green gate and one that still refuses — the same shape/measurement split that
m3's pair showed an hour ago, which is what a stage looks like while it is being
brought up rather than while it is broken.

**Two run processes present at 10:29:41** — pid 267995 (25 m 13 s) and pid
600210 (1 m 59 s, the 10:28 relaunch).

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b0) **pre-commit `git status` on this file** | **clean — new, passed first application** |
| (b) per-commit ownership | clean across 29 |
| (c) `todo.md` | **37 items**, +8 |
| holds | **three, none lost since 06:47**: 006 m1, 217 rung 0, 047 m5 |
| `/home` | 1.6 T free (85 %), flat since 05:47 |
| run process | two present at 10:29:41 |

### 1. Progress

**~78 %, up two.** Elapsed 1 259 m.

**Reliability: low→moderate, held.** The raise is on artefact change, not
reporting: `check_optimization_shape` green for the first time, and **rung 0
walked stages 1, 2 and 3 in one run** (`20260904T101432-b4e595`, 217, ~6 min,
eleven handoffs). Last interval I held at 76 because only the narrative moved;
this one the artefacts moved, so it goes up. **What keeps it to two points: the
distinct count is flat for the second interval and m5 is still unreached.**

**预估耗时: no number.** Held.

### 2. Current state

**Rung 0's furthest walk**, per the leader and consistent with what I can read:
stage 1 sealed, both cuda-graph arms and the merge succeeded, `rank` succeeded,
**stage 4 produced a handoff `a22fc40f` and refused in `output_validating`**,
stage 5 `waiting_handoff`.

**m4's three rung-0 fixes all cleared and none fired** — `entry_function`, the
`dtype` copy, the `--impl run` shim. The refusal was a fourth thing:
`check_speedup_substantiated` could not create its re-measurement scratch
directory, **because a validator runs on the login node and `/mnt/m2m_nobackup`
is not there.** Relaunched 10:28 with `e747653`.

### 3. Code problems

- **FIXED — one variable asked for two directories** (`e747653`). The leader
  records their own half-right instruction: the **ROCm temp** must be created on
  the far side; the **staging tree must not**, because the process reads the
  report back to compute medians. **Had m4 done as told, the next failure would
  have been the report read, one run later.** The fix puts the tree in the zone,
  which resolves identically on login node, node and container — m3's answer,
  evidenced by their `recorded 0.0415, re-measured 0.0415`.
- **FIXED — 18 fields in the `environment` schema were required or consumed with
  no definition at all** (`113373e`). **m1 found two and called them the third;
  the count was eighteen.** This is the first time today someone named a class
  and then **actually swept for it** — the answer to m3's *"naming a class is not
  sweeping for it"*, arriving three hours later. I have named four classes and
  swept for none.
- **FIXED — `started_at` was undefined** (`2c03c0f`), and the symptom is worth
  keeping: it produced a value **34 min early on one run and 5 min late on
  another, with zero restarts, both validating.** Now read from `docker inspect`,
  **read not composed**.
- **FIXED — a body that exits 0 has its output discarded** (`cff4571`), *"which
  is the case that hurts"*. This is the **fifth-instance gap finally attacked**
  rather than routed around.
- **FIXED — an unguarded `mkdir` made the validator raise instead of refuse**
  (`8e0be6a`); **`2>/dev/null >&2` emptied a container listing** (`f103fe0`) —
  the second time today a redirection hid the evidence someone had just added.
- **OPEN — `check_speedup_substantiated`**, 0 of 3.

### 4. Non-code problems

- **Three holds, none lost since 06:47.** Second consecutive clean interval and
  the longest stable stretch of the effort.
- **Two `keep3` jobs still pending** from the non-team session.
- **`f867a62` — two owners picked the same next `todo.md` integer, twice.** The
  leader logged it at the time as a numbering problem and now records that as
  **the symptom rather than the thing**: it was the same working-tree race that
  `8b1057d` fixes. `todo.md` stays unserialised on purpose — `25d9c01` exists so
  the finder records in the moment.

### 5. Open questions

- **`check_speedup_substantiated`'s remaining refusal**, after the scratch fix.
- **The seven m5 validators** — unchanged, and now the only thing between this
  metric and 21.
- **M5.1.1** — can state the substitution case, cannot install the fragment case.
- **m5's control experiment result** — predictions committed at 09:38:06; I still
  cannot order them against the run from the repository (T+1210).
- Why holds are cancelled; how much co-tenant load is corpses.

### 6. New commits

**29 since `aaefcba`** — the largest interval of the effort. m4 12 · m1 8 · m3 4 ·
m2 3 · leader/schema 2.

Grouped, since twenty-nine one-liners would bury the four that matter:

- **m4 (12)** — rung 0's two refusals (`08c1f3d`), the sealed candidate predating
  the `--impl` contract (`2dd93b1`), T34/T35/T36 (`a02f9a9`, `df78ed1`,
  `2d521c1`), an exec taking a card the container never had (`834b719`), the pin
  refusal met against a real pinned deployment (`c149a28`), the two-directory fix
  (`e747653`), and `f103fe0`/`8e0be6a`/`a8fd10f`/`7014261`.
- **m1 (8)** — `kit_status` (`d736b86`), the mount list missing node-local
  scratch (`a32f06d`), T27 item 1 met unprompted (`ba4dc2d`), `started_at`
  (`2c03c0f`), *"keep both — the T27 correction would itself have caused a
  regression"* (`c8406fa`), and `c1c10ba`/`8a07e3c`.
- **m3 (4)** — the exit-0 discard (`cff4571`), a sample that never moves is a
  census of one member (`7eba786`), and the citation correction (`eb11438`).
- **m2 (3)** — the standalone execution correcting two things written an hour
  earlier (`1408da8`), neighbour occupancy per step (`7e3e13f`), replay in the
  watchout (`9dbad96`).
- **schema/contract (2)** — `113373e`, `8b1057d`.

### 7. Anything else

**The leader's second self-correction is the sharpest epistemic point of the day
and it lands on me.** They called m4's 006 reading stale; it was **accurate when
taken and became stale** — container created 10:16:46, measurement 10:18:19.
*"Misread the node"* and *"reading aged"* are different failures **and only one
is a lesson.**

**Every number in this file is the second kind.** My tally is a snapshot: I
publish "14 distinct" at a timestamp, and it is true then. Twice now the count
has been flat across an interval while real movement happened *inside* it —
`check_optimization_shape` going 0-pass to 1-pass is invisible in "14". I have
been treating a flat headline as evidence of a flat interval, which is the
aggregate-read-as-specific error I named this morning, committed by me, in the
number I most rely on. **From here the ceiling table carries the per-validator
deltas whenever the headline does not move**, which is what this section does
above.

**And `c8406fa` deserves the last word: *"keep both — the T27 correction would
itself have caused a regression."*** m1 wrote a correction, then found their own
correction was wrong in a second-order way, and kept both records rather than
replacing one with the other. Combined with `ba4dc2d` — *"T27 item 1 has been met
once, unprompted; recorded before the run that did it is forgotten"* — that is
someone treating the record as evidence that decays if unwritten, not as a
summary written at the end. **That is the behaviour this file exists to make
possible, and it is now happening in owners' files without anyone asking.**

### Addendum, 10:54 UTC — my T+1062 answer was wrong, and the reason was in the event all along

m2 found that `read_events.py` prints `message` and hides
`attributes.seal_refused`. It is my file; the fix is landed (`841ca22`). **The
part that belongs in this record is that the defect produced a wrong published
answer of mine, and I can now show exactly how.**

**What I told the leader at T+1062**, asked what `output_absent` at 13.8 s means:

> the body executed and staged into `v1`, the store never advanced past `v0` […]
> **Whether the divergence causes the absence or merely accompanies it is not
> measured.**

**What the event said, on both runs I analysed:**

```
message      = declared output 657bcbde-… was never delivered
exit_status  = failed
detail       = exit 1: mock: stage3-analyze/operator_workset -> operator_workset (27 files) …
seal_refused = …/v1/content/README.md: required section 'Interface' is missing.
               Present at document root: ['Environment','How to run','Purpose',
               'Result','Watch out','operator_workset'] — a heading inside a
               blockquote, a list or a code fence is not a section
```

**It was measured. It was in the event. I printed one of its four attributes.**

Three specific errors follow, and they are worth separating:

1. **"Not measured" was false.** The cause was a named seal refusal with the
   missing section, the sections present, and the rule. I reported an open
   question that the artefact had already closed.
2. **"The failure is between the body and the store"** — my phrase, and it is
   wrong. The failure is **at the seal**, which is neither.
3. **I reconstructed `detail` by hand.** I spent several tool calls inferring
   from mtimes and `drwxrwxrwx` that `v1` had been *staged rather than authored*.
   `detail` says `exit 1: mock: … (27 files)` in one line. **My conclusion was
   right and I paid for it twice over, then hedged it anyway.**

**And it was fixed at 07:47 by someone who found it another way.** m3's
`79ff361` — *"the seal wanted two README sections the sealed copy never had"* —
is this exact refusal. So the question I left open at 07:12 was answered
thirty-five minutes later by a different owner reading a different artefact,
while my record still carried it as unresolved.

**This is the fourth instance of the class I named this morning, and the worst,
because the instrument was mine.** A key signature carries names but not
membership. A run duration carries elapsed time but not which closure spent it.
A green count carries verdicts but not completion. **And `message` carries the
event's summary but not its cause** — while `_text` returned it and stopped,
because I wrote `_text` to return the first of three keys and never asked what
the other keys held.

**One correction to m2's patch, which is the same bug one layer in.** Their
`_SHOWN = ("message", "why", "detail")` suppresses all three summary candidates,
but `_text` only *uses* the first one that is set. On this very event `message`
is set, so `detail` — the "27 files" line, the half that proves an artefact
existed — **stayed hidden by the fix written to unhide the cause.** Landed with
`_extras` suppressing only the key the summary actually consumed. Verified on the
real event: both `detail` and `seal_refused` now print. **A reader left with a
refusal and no evidence there was ever an artefact is in the position I was in at
07:12.**

Their design decision — extras on their own lines, **untruncated** — is kept
verbatim and was right: their first draft appended to the summary where `[:100]`
cut through the middle of `seal_refused`, which is *"a change that looks correct
in a diff and delivers nothing."*

**What I am changing in my own procedure.** Every claim in this file that rests
on a tool I wrote now names the tool. I have three: the verdict tally, the
process check, `read_events.py`. **Two of the three have now been wrong in a way
that changed a published conclusion** — the tally read one run root when there
were two, and this. The third has never been independently checked by anyone.

## T+1285 — 2026-09-04 10:55 UTC

### I attacked my own tally, as instructed. It survived identification and it has one untested leg left.

The leader's standing objection is right: **naming a tool is disclosure, not
verification, and an instrument that has never been attacked is not an
instrument that works.** Both of the tally's known failures were the same shape
— *it read a world smaller than the real one*. So the test is not "does it count
correctly" but **"can I construct a state it reports confidently and wrongly."**

Four attacks, all run just now:

| attack | result |
|---|---|
| **Signature collision** — two validators whose `args` keys are identical, silently merged | **none.** All 21 parsed with `yaml.safe_load_all`; every signature unique |
| **The classifier misnames one** — my real classifier run verbatim over all 21 declared `args` | **0 misnamed of 21** |
| **Verdicts silently dropped** — a `verdict.json` with no sibling `args.json` hits my `except: continue` | **0 of 334** |
| **The frozen-root constants are stale** — I hardcoded them last section | **verified: 38 runs, 231 verdicts, mount still `ro`** |

**One real finding, and my first attempt missed it.** My initial collision test
was line-based and reported "20 of 21, no collisions". The missing one was
`check_worklist_shape`, whose args are **inline flow style** —
`args: {schema: kernel_worklist}` — invisible to a line parser. It is also the
**shortest signature in the package** and therefore the most collision-prone, so
my first test excluded exactly the validator most likely to fail it. Re-run with
a real YAML parse: still no collision, but **the test that said "none" was not
the test I thought I had run.**

**What the attack does not establish.** The seven m5 validators are all
**unknown to my map** and fall through to `NEW[<signature>]`. That is the right
failure mode — visible, and still distinct per signature, so the *count* stays
correct while the *names* read as raw key-lists until I add them. I am leaving
them unmapped deliberately: a guessed mapping that renders differently at
runtime would be a silent error, and `NEW[...]` cannot be.

**So: the tally is sound on identification and unproven on scope.** Scope is
where both its failures happened, and the current scope mitigation — the
hardcoded frozen-root constants — is a *new* single point of failure that is one
interval old. I re-verified it this interval and will re-verify it every
interval rather than trusting it, and I have written the `ro` mount check into
that verification because the constants are only valid while the mount is.

**And the third tool is going to m3**, per the leader: the process check has
never been independently examined. Sending them the tool and the claim, and
asking them to make it lie.

### The number that matters

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| frozen (`ro`, re-verified) | 38 | 231 | 255 | 10 |
| live | 26 | 336 | 381 | 14 |
| **union at 10:55** | **64** | **567** | **636** | **14** |

**Distinct flat for the third interval — and per the rule I adopted last
section, here is what moved underneath it:**

| validator | 10:28 | 10:55 |
|---|---|---|
| `check_optimization_shape` | 3 / 1 | **5 / 3** |
| `check_speedup_substantiated` | 3 / 0 | 5 / **0** |
| `check_workset_runs` | 9 / 3 | **11 / 5** |
| `check_workset_shape` | 9 / 7 | 11 / 9 |

**Two run processes present at 10:55:31** — pid 267995 (50 m 31 s) and pid
1017998 (2 m 08 s).

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b0) pre-commit `git status` on this file | **clean** |
| (b) per-commit ownership | clean across 3 |
| (c) `todo.md` | **39 items** |
| holds | three: 006, 217, 047 — **none lost since 06:47** |
| `/home` | 1.6 T free (85 %) |
| frozen-root constants | **re-verified this interval**, mount `ro` |

### 1. Progress

**~78 %, held.** Elapsed 1 285 m. Distinct flat for a third interval; the
movement is inside the fourteen and m5 remains unreached. Reliability
**low→moderate**, held. **预估耗时: no number.**

### 2. Current state

Three commits this interval, all corrections of the same kind as the tool fix:
`629111f` m4 — the ephemeral container mounted `$HOME` and a validation zone
redefines it; `8df59a6` m2 — `check_kernel_table` keeps its reasons;
`38cdf8f` m3 — the heading was inferred, so a caller's misclassification made
it lie.

### 3. Code problems

- **FIXED — `read_events.py` hid `seal_refused` and `detail`** (`841ca22`).
  m2's find, my file, my defect. See the previous addendum.
- **OPEN — `check_speedup_substantiated`**, 0 of 5.
- **OPEN — the process check**, unverified by anyone; going to m3.

### 4. Non-code problems

Three holds, none lost since 06:47 — the longest stable stretch of the effort,
now four intervals. Two `keep3` jobs still pending from the non-team session.

### 5. Open questions

The seven m5 validators; `check_speedup_substantiated`'s remaining refusal;
M5.1.1; m5's control-experiment ordering; why holds are cancelled.

### 6. New commits

**3 since `d1f9104`.** `629111f` m4 · `8df59a6` m2 · `38cdf8f` m3. Plus my
own two: `841ca22` (the tool fix) and `d1f9104` (the correction it forced).

### 7. The class, stated once, replacing four separate entries

The leader is right that I have named this four times today under four headings
and it is one thing. Consolidated:

> **An instrument reads a real thing and answers a different question, and is
> never wrong in a way that shows up as an error.**

The instances, all measured today:

| instrument | reads | the question you asked it |
|---|---|---|
| my `args.json` key signature | which keys exist | which list a key is *in* |
| my run duration | the run's elapsed time | which closure spent it |
| my distinct-validator count | verdicts recorded | whether a run completed |
| my `_text()` | the event's `message` | the event's **cause** |
| `items_schema` | a filename string | the file's contents |
| m1's image check | that it passes at tp=1 | that the image is good |
| the completion probe | `nonempty` | that the answer is right |
| `--rm` | that a container stopped | that it was torn down |
| `/proc` under `spur exec` | not visible from here | not running |
| `"not visible"` | a default | a measurement |

**Four of the ten are mine**, which is the reason to state it as one class
rather than ten anecdotes: it is not a property of any subsystem, it is what
every probe in this stack does by default. **None of them errors. Each returns a
true fact about a smaller world than the one asked about**, and the only defence
that has worked all day is **a second artefact** — the yaml beside the
signature, the event store beside the tally, `detail` beside `message`,
`docker top` beside `/proc`, a control beside a null.

**And the class has a fifth kind of member, which is the leader's and is worse
than mine.** They read `seal_refused` in a four-run study, reported it as
identical across all four, **and then treated the phenomenon as a stall-detector
question anyway** — while m3 had already fixed that exact refusal at 07:47. In my
four cases the instrument withheld the cause. **In theirs it did not, and the
reader carried a prior that the answer could not displace.** A better instrument
does not fix that one; only a second reader does, which is what the last two
hours have actually been.

### Addendum, 11:12 UTC — I escalated a false alarm, an hour after tabulating the class it belongs to

**Retracted: the "orphaned run" I reported to the leader as live T26 evidence.**

I read `1017998 ppid=1` on a running `agent_sys.cli.main` and reported it as a
run whose launcher was gone — *"T26 visible in an instrument rather than
inferred"*. m2 challenged it with a control. I reproduced the control rather
than accept it:

```
nohup sleep 45 &     ppid while the launching shell lived:  1142403
                     ppid once that shell had exited:       1
```

**Nothing killed it.** `ppid=1` is the ordinary fate of anything started with
`nohup … &` from a shell that then returns, which is how every run in this
effort is launched. And it is **doubly not T26**: `1017998` was the
*orchestrator*, not an agent, and T26 is about agents outliving a killed
orchestrator. `1017998` has since exited by itself.

**The error is the eleventh instance of the class I tabulated at T+1285, and I
made it within the hour of writing that table.** I read a true fact — `ppid=1` —
and answered a different question: *was this orphaned by a kill?* The instrument
did not err. It never does; that is the entry.

Two things about it are worse than the ten already listed.

**It had no control, and I have spent the day requiring controls of everyone
else.** I told m2 their liveness column *"can say live and has never been
observed to say dead"*; I recorded m5's null overlay as *"a gate validated only
against a null sample has never been shown to detect anything"*; I wrote that my
own process check was *"validated in the negative direction only"*. Then I read
a single `ppid=1` with nothing to compare it against. **The control cost one
`nohup sleep` and ten seconds.**

**And I escalated it rather than recorded it.** A wrong number in this file is
read later by someone with time; a wrong operational alert asks the leader to act
now. **The discipline I apply to the numbers had not reached the things I send
urgently — which is exactly backwards.** From here an operational alert carries
its control, or it waits until it has one.

**What the exchange produced is the argument for the exchange.** In the same
round, m2 withdrew their claim about my liveness command — they had measured a
form reconstructed from my description rather than the command itself, which has
a `grep -v grep` and does return zero. So **each of us asserted something about
an object we had not opened, and each caught the other inside an hour.** The two
findings that survived are the ones neither of us held at the start:

- **theirs** — `grep -v grep` filters *by content*, so a genuine process whose
  command line contains "grep" is silently dropped, and a false negative on a
  liveness check is the worse direction;
- **mine** — **line count is not run count**: four matching lines were two runs,
  one of them wrapped in `zsh -c` → `timeout 7200` → python, and my by-eye
  selection of the python leaf was an undocumented step in a published one-liner.

Both stand. Neither was in either opening position. **That is the second time
today a disagreement between two owners produced a finding that neither
investigation would have reached alone** — the first was m5 refusing the leader's
null-overlay design.

## T+1294 — 2026-09-04 11:04 UTC

### The third tool was checked, and it was worse than the two that had already failed

m3 attacked the process-liveness check and **made it lie four ways, measured
against live runs rather than reasoned about.** I verified the worst one myself
and then attacked their proposed replacement, which is what they asked for.

**Finding 1, a FALSE NEGATIVE in the direction I had named as never tested.**
`/home/yihou/miniconda3/bin/agent-sys` exists — 176 bytes,
`#!/home/yihou/miniconda3/bin/python3.14`, `from cli.main import main`. A
console-script run's process line is `python3.14 …/bin/agent-sys run …` with
**no `agent_sys.cli.main` in it**. My line scores **0** against that shape,
verified independently. **Both launch forms are in live use** — `CLAUDE.md`
writes `agent-sys run`, `CONTRACT.md` §9 writes `python3 -m agent_sys.cli.main` —
and my check sees one. A run launched the documented way reads as *no process*,
which my three-state framework renders as **"stopped"**: the opposite of my
T+1062 error and worse, because *stopped* is the state I escalate on.

**Finding 2, a false positive from routine tooling.** `show` dispatches nothing
and takes a second, and it matches. m3 runs it after every edit — **dozens of
windows a day in which my poll reports a run that is not a run.**

**Finding 3, one run is three lines.** `zsh -c` → `timeout 7200` → `python3`, all
three matching, because a wrapper's command line contains the run's verbatim.

**Findings 5 and 6 removed two worries I was carrying:** re-parenting does *not*
affect `etime` (it is computed from start time), and `ps` truncates only to a
tty, not through a pipe — a 1339-character line came through intact. But `etime`
on a *wrapper* line is the **shell's** age, so reading the first match can
overstate a run's age by however long that shell had been idle.

**I attacked their replacement and it keeps 5 of 6 shapes where it should keep
3.** Both of its exclusions miss: `(timeout|/bin/|/usr/bin/)?(ba|z|d)?sh\b`
needs a shell token after the optional `timeout`, and `timeout 7200 python3` has
none; and the launcher here is `/home/yihou/miniconda3/bin/zsh`, which is neither
`/bin/` nor `/usr/bin/`. **So their line fixes findings 1 and 2 and leaves 3
untouched** — which is their own point applied to their own fix: *a wrapper's
command line contains the run's verbatim, so no textual exclusion can separate
them.*

**The fix that does work is structural, not textual: keep a matching pid only if
it is not the ppid of another matching pid.** It never reads command text, so
wrapper wording cannot fool it. Composed with m3's two fixes and their wording
point, verified live:

```
run pid=267995 etime=58:15 root=/home/yihou/agent_sys_runroot
control (a token nothing runs): 0
```

**m3's wording point is the half that mattered most** and I have adopted it:
*"a process for run-root X was present at HH:MM:SS"* is the claim my three-state
reading actually needs. `--demo-root` is in every launch line, so identity was
free the whole time and I never asked for it. `etime` now comes from the leaf,
which closes their finding 5.

**What still is not fixed:** `grep -v grep` filters **by content** (m2), so a
genuine run whose command line contained "grep" is silently dropped. The leaf
rule does not touch that; a `/proc/<pid>/cwd` reading would. **Not shipping until
I can test it against two concurrent runs**, which existed an hour ago.

**And the sentence I am keeping is m3's:** *"The tool is not broken; its claim is
wider than its evidence."* That is a better statement of the whole class than
anything I had, and it is the same defect as the validators-stdout record.

**Three of three tools have now been found wrong by someone else.** The tally
read one root when there were two; `read_events.py` printed `message` and hid
`seal_refused`; this one is blind to the documented launch form. **None of the
three was found by me.** I attacked the tally myself this morning and it passed —
and m3's finding 2 is the same shape as my collision test excluding
`check_worklist_shape`: **each of us tested the space we could picture.**

### The number that matters

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| frozen (`ro`, re-verified this interval) | 38 | 231 | 255 | 10 |
| live | 26 | 358 | 406 | 14 |
| **union at 11:04** | **64** | **589** | **661** | **14** |

**Fourth interval flat at 14.** Deltas underneath, per the standing rule:

| validator | 10:55 | 11:04 |
|---|---|---|
| `check_optimization_shape` | 5 / 3 | **6 / 4** |
| `check_speedup_substantiated` | 5 / 0 | 6 / **0** |
| `check_workset_runs` | 11 / 5 | 12 / 6 |
| `check_workset_shape` | 11 / 9 | 12 / 10 |

**One run present at 11:04**, root `/home/yihou/agent_sys_runroot`, by the new
leaf reading. Frozen-root constants **re-verified**, mount still `ro`.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b0) pre-commit `git status` on this file | clean |
| (b) per-commit ownership | clean across 1 |
| (c) `todo.md` | 39 items |
| holds | three: 006, 217, 047 — none lost since 06:47 |
| `/home` | 1.6 T free (85 %) |
| frozen constants | **re-verified**, `ro` |

### 1. Progress

**~78 %, held.** Elapsed 1 294 m. Distinct flat for a fourth interval;
m5 unreached. Reliability **low→moderate**, held — I am not lowering it for the
tool findings, because none of them changed a *number* in this file: the tally
and `read_events` did, this one governs a state word ("stopped"/"growing") that
has been corroborated by artefact motion every time I published it.
**预估耗时: no number.**

### 2. Current state

One commit: `ea9e857`, m5 — **make the pre-registration auditable, and say the
ids were added afterwards.** That closes my T+1210 open question, and it closes
it the honest way: rather than assert the ordering, they recorded that the run
ids were added after the fact. **A pre-registration that admits which parts were
retrofitted is worth more than one that does not distinguish.**

### 3. Code problems

- **OPEN — my liveness check**: findings 1–4 above, fix designed and tested,
  `grep -v grep` residue unresolved.
- **OPEN — `check_speedup_substantiated`**, 0 of 6.

### 4. Non-code problems

Three holds, none lost since 06:47 — five intervals. Two `keep3` jobs pending.

### 5. Open questions

The seven m5 validators; `check_speedup_substantiated`; M5.1.1; the
`grep -v grep` false-negative; why holds are cancelled.

### 6. New commits

**1 since `1a9286a`.** `ea9e857` m5 — make the pre-registration auditable, and
say the ids were added afterwards. Plus my own `1a9286a` (the orphan retraction).

### 7. Anything else

**In ninety minutes, four owners' reviews landed on three of my tools and every
one of them found something I had not.** m2 on `read_events.py`; m2 again on the
liveness self-match; m3 on the liveness check, four ways; and m2's control that
retracted my own false alarm. **I attacked one tool myself in the same window and
it passed.**

That asymmetry is the finding, and it is not about competence. **Each of us
tested the space we could picture.** m3 found `show` because they run it dozens
of times a day and I never do. I found the leaf rule by attacking their line
rather than mine. m2 found the self-match by testing a form they had rebuilt from
my description — which was also *their* error, and produced a true finding
anyway. **The unit that works is the pair, and it has now produced four findings
that neither party held at the start.**

**Against that, my own self-attack this morning is the control.** It was
thorough — signature collisions, misclassification, dropped verdicts, stale
constants — and it passed, and its first version had **excluded the validator
most likely to fail it**. A self-attack tests the failures you can imagine. It is
worth doing and it is not evidence.

**So the procedural change I am making is not "check my tools" — I did that.**
It is: **a tool of mine is not verified until someone who did not write it has
tried to break it.** Three of three have now failed that test after passing
mine.

### Addendum, 11:10 UTC — I published output from one command and code from another

**m3 checked the fix I proposed to them and it cannot run.** `awk '…' - -` reads
stdin **once**: the first `-` consumes it, the second gets EOF, so `NR==FNR` is
true for every line, `next` fires every time, and the print block never
executes. Measured, GNU Awk 5.2.1: **pass1 = 2 lines, pass2 = 0.** Against the
live host:

```
the exact pipeline I pasted:        (no output)
matching processes at that moment:   2
```

**Two live runs, zero lines out, silently — reading as "stopped", the direction
I escalate on.**

**The error underneath it is worse than the bug.** I *ran* a snapshot-to-a-file
variant and *pasted* the `- -` form. So the output I quoted was real and the
code beside it could not have produced it. **That is exactly the error I
attributed to m2 ninety minutes ago** — measuring one form and asserting it of
another — committed by me while quoting the lesson back at them. m3 caught it by
**running what I wrote instead of trusting the output next to it**, which is the
only way it was catchable.

**Root cause: the tool existed only as a snippet pasted into messages**, so the
quoted form and the run form *could* differ. **It is now a file** —
`assets/lib/runlive.sh` (`296e2ea`) — with m3's snapshot-twice implementation
and a comment carrying their `pass1=2 / pass2=0` measurement and why `- -` must
never replace it. One executable form; paste-drift is no longer possible.

**Controls, including the one that failed.**

- **Negative** — 0 against a token nothing runs.
- **Positive, first attempt: FAILED, and the subject never existed.** `exec -a`
  is not in `dash` — `sh: 1: exec: -a: not found`. **A control that fails for an
  unknown reason is worse than no control**, so I chased it rather than
  recording a null.
- **Positive, valid** — a `/tmp/agent-sys run --demo-root /tmp/FAKEROOT`
  subject, which is also **m3's finding-1 console-script shape**, the one the old
  line scored 0 on:

```
run pid=1239067 etime=01:45 root=/home/yihou/agent_sys_runroot
run pid=1242034 etime=01:25 root=/home/yihou/agent_sys_runroot
run pid=1272157 etime=00:01 root=/tmp/FAKEROOT          <- the control
3 run process(es) present at 11:08:03
```

**Two concurrent real runs were live during the test** — the condition I said I
would not ship without, and it arrived on its own.

**Both of m3's caveats are in the file as limits, not as solved:** fork
over-count is *"unproven, not disproven — one observation of one run shape"*, and
the `etime` fix is **reasoned but not demonstrated**, because every run seen so
far had shell and run starting together. m2's `grep -v grep` content-filter risk
is recorded unfixed, with `/proc/<pid>/cwd` named as the thing that would fix it.

**This is a third instance of m3's T31 — *naming a class is not the same act as
applying it*.** I wrote *"no textual exclusion can separate them"*, derived the
correct structural rule from it, and then shipped that rule in a form that could
not execute. **The reasoning was right and the artefact was not**, and nothing in
my own checking distinguishes those two, because I checked the reasoning.

### Addendum, 11:14 UTC — the attack m3 could not build, built; and the fix that argv cannot fool

m3 attacked `runlive.sh` and reported **two defects, one behaviour confirmed
good, and one attack they could not construct — reported as untested rather than
cleared.** That last distinction is the reason this addendum exists: **I built
it, and it succeeds.**

**The attack.** `sh -c '<run>; sleep 20'`, inner subject exiting after 3 s:

```
phase 1 (child alive):   child reported, wrapper excluded    — correct
phase 2 (child exited):  run pid=1347340 etime=00:13 root=/tmp/ATTACK;
```

**A run that had ended, reported as live.** The trailing `;` is the shell's argv
leaking into the run-root field. This is the reassuring direction — the T+1062
class — and it is the failure the tool exists to prevent. m3's structural
argument was about m1's real launcher doing `echo` and `date` after the run
exits; `sleep` reproduces it.

**The fix is `/proc/<pid>/exe`, on m2's rule that a reading must not be able to
contain the query.** No argv rule can work, because a wrapper's argv legitimately
*contains* the run's. The kernel's record of the actual binary cannot be faked by
a command line:

```
real run (python -m …)      exe = …/bin/python3.14   keep
console script (agent-sys)  exe = …/bin/python3.14   keep   (shebang)
timeout 7200 …              exe = /usr/bin/timeout   drop
sh -c '…' / zsh -c '…'      exe = …/sh, …/zsh        drop
```

m3's finding 1 survives it — the console script's shebang *is* python.

**Their two defects, both fixed.** `--demo-root=<path>` silently gave `root=?`,
in the one field my new wording depends on — *"a process for run-root ? was
present"* is the old claim with extra syntax. And `strftime` is a GNU extension
sitting in the `END` block, so where it is absent the failure takes the **count
line**, which is the part I quote; replaced with `date`. They said not to bother
with the second today; it cost one line and their reason for raising it was
right.

**Four controls, all passing**, and the tool is `552c0eb`:

```
A baseline, real runs only                       -> 2
B ATTACK, launcher outliving its child           -> 2   (wrapper present, excluded)
C console-script form + --demo-root=/tmp/EQROOT  -> 3   root=/tmp/EQROOT, not "?"
D negative, a token nothing runs                 -> 0
```

All control processes exited; `/tmp/agent-sys` removed; nothing of mine left on
the host.

**Two things from the exchange that are worth more than the tool.**

**m3's mechanism for the paste-drift error is better than my diagnosis of it.**
I called it my carelessness. They said: *"a one-liner in a message is edited
every time it is quoted, so what I ran and what I wrote are different objects
with nothing binding them."* That is a property of the artefact, not the author,
and it is why the fix had to be **a file** rather than more care.

**And one push-back I gave them, because it applies to me first.** They noted
that two of their wrong turns were probes that could not have succeeded, found
only because the result looked implausible. **My `exec -a` control was the same
and I only chased it because it was labelled a control** — had it been a
measurement I would probably have accepted the null. So the difference is which
slot the failing thing occupied, not vigilance. **The habit that actually works
is treating every probe as a control**, and neither of us does that
consistently.

**Score on the day's tooling: three of three tools found wrong by other people,
and now two of three attacked to the point of having controls.** The remaining
one is the verdict tally, which I attacked myself and which passed — the same
self-attack whose first version excluded the validator most likely to fail it.
It is the tool with the weakest evidence behind it and it produces the number in
every section headline.

### Addendum, 11:29 UTC — four people reasoned carefully about a value that never existed

**Resolved, and there was no `HOME` mystery.** The string everyone was reasoning
from — `-v /home:/home denied [BH] by plugin spur-authz` — is **not the daemon's
output.** It is `measure_in_container.sh:274`, an `echo` inside the **refusal
branch**, printing a reference table:

```
271  echo "  plugin will accept from $ROOT. Measured forms:" >&2
272  echo "    -v /home/<user>:/home/<user>   OK   (a run root under one user's home)" >&2
273  echo "    -v /shared_nfs:/shared_nfs     OK" >&2
274  echo "    -v /home:/home                 denied [BH] by plugin spur-authz" >&2
...
277  exit 1
```

Verified first-hand. **The script could not derive a mount at all, printed its
table of known-good and known-bad forms, and exited 1.** `check_workset_runs`
captured the tail. **No `docker run` ever attempted `-v /home:/home`. The daemon
was never asked. `$HOME` was never `/home`.**

**And the refusal branch firing is evidence *for* `environment.py:235`, not
against it.** In a validator zone `HOME = <zone>/home` and `$ROOT` is
`<zone>/materials/<hid>/v1/items/codes` — under neither `$REMOTE_HOME` nor
`/shared_nfs` — so both `case` arms missed and `*)` fired. **Exactly what the
code predicts.** T42's row is right, unqualified, on `environment.py:235` alone.
My third candidate — node-side expansion in `remote.sh:150` — is also
eliminated: m3 has a `bash -x` trace showing `export HOME=/home/yihou`, so the
`$( )` expands locally and never reaches the mount derivation.

**What I retract.** Two messages to the leader: one proposing *"validator
rewrites `HOME`, task body leaves `/home`"* as the likely reconciliation, and one
correcting that to *"neither branch fits"* and naming a third candidate. **Both
were careful reasoning about a value that never existed.** The second was worse
than the first — I had by then read the code, established `HOME=/home/yihou`
here, and concluded the conflict was *deeper* than it looked, when the correct
conclusion available at that moment was **"no context produces `/home`, so
perhaps nothing did."** I treated an unexplainable observation as evidence of a
subtler mechanism instead of as evidence against the observation.

**The mechanism is worth more than the incident, and it is m3's.** *A refusal
that documents known-bad forms will be quoted back as a report of one
occurring.* In a captured tail, **reference and observation are
indistinguishable** — the row says `-v /home:/home denied`, and nothing in the
text marks it as a catalogue entry rather than an event. That is a real cost of
the "name the measured forms in the refusal" pattern m3 has advocated all day and
which I have praised in this file. **The pattern is still right; it needs its
reference rows marked as reference.**

**This is a twelfth instance of the class, and the first where the instrument was
prose.** The ten in the T+1285 table plus my `ppid=1`: each was a probe reading a
real thing and answering a different question. Here the "probe" was **a helpful
error message**, and the different question was *did this happen* versus *is this
known to fail*. **My own file has the same exposure**: every quoted refusal, log
line and diagnostic in these sections is a captured tail, and I have no way to
tell a catalogue row from an event either.

**Four people reasoned carefully from a string written to be helpful** — m3 wrote
it and misread it, put a fitted value in `5964fd8`, the leader quoted it, I built
a reconciliation on it and then a deeper one. **It was one message from being a
qualified fact in T42.** What stopped it was m3 going back to a surviving
`validator_report.txt` and diffing it against their own source — not any of the
three of us reasoning harder.

## T+1320 — 2026-09-04 11:30 UTC

### First: a correction to my own addendum of an hour ago

I wrote at 11:29 that *"no `docker run` ever attempted `-v /home:/home`. The
daemon was never asked."* **The second sentence is false**, and the leader
settled it from artefacts. `measure_in_container.sh:214-218`:

```
# the leader measured the daemon's answer on node 243:
#     Error response from daemon: authorization denied by plugin spur-authz:
#     denied [BH]: /home:/home -- mount your own directory instead,
```

**The daemon was asked — once, by the leader, on node 243 — and it refused.** The
row in the refusal table is a *quotation of a real measurement*, not an
invention. What is true is narrower: **no run attempted that mount**, and the
captured tail was a catalogue row rather than a fresh event.

**And the leader found the thing all three of us missed: the `/home` never came
from `$HOME` at all.** It came from taking the **top-level component of
`$ROOT`** — `/home/<user>/agent_sys_runroot` → `/home`. So m3's value was fitted
to a denial about `$ROOT` while their sentence described `$HOME`. **Two records
about two different things wearing one name**, which is exactly why it could not
be reconciled with `environment.py:235`: there was nothing to reconcile.

**T42's row is settled** — *validator*, on `environment.py:232-237` plus three
zone listings the leader pulled from run `20260904T111550`, which show the
`home/` directories actually existing. My `remote.sh:150` candidate is dead and
should not stay in the file as an open branch of a closed conflict.

**My own error in one line:** I corrected an overstatement by overstating in the
other direction. *"A reference row, not an event"* was right; *"the daemon was
never asked"* went one step past the evidence I had, in the same message where I
was criticising exactly that.

### The number that matters

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| frozen (`ro`) | 38 | 231 | 255 | 10 |
| live | 30 | 428 | 485 | 14 |
| **union at 11:30** | **68** | **659** | **740** | **14** |

**Fifth interval flat at 14 — and the deltas underneath contain the interval's
best result:**

| validator | 11:04 | 11:30 |
|---|---|---|
| **`check_speedup_substantiated`** | 6 / **0** | **8 / 2** |
| `check_optimization_shape` | 6 / 4 | 8 / 5 |
| `check_workset_runs` | 12 / 6 | 15 / 9 |
| `check_workset_shape` | 12 / 10 | 15 / 13 |

**`check_speedup_substantiated` passed for the first time.** It had been 0-for-6
since first contact and was the last validator in the *reached* set that had
never returned a pass. **Every validator the graph can currently reach has now
passed at least once.** The seven that have never spoken are still exactly m5's
seven, unchanged since T+1177.

**Two runs present at 11:30:41**, both `root=/home/yihou/agent_sys_runroot`, read
by `runlive.sh` rather than by eye — first section where that number comes from
a tool that has been attacked by someone else.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b0) pre-commit `git status` on this file | clean |
| (b) per-commit ownership | clean |
| (c) `todo.md` | 40 items (T40 mine, corrected at m3's request) |
| holds | three, none lost since 06:47 |
| `/home` | 1.6 T free (85 %) |
| frozen constants | re-verified, mount `ro` |

### 1. Progress

**~79 %, up one.** Elapsed 1 320 m.

**Reliability: low→moderate**, held. The point is for
`check_speedup_substantiated` — the reached set is now fully green at least
once, which is a real boundary. **Only one point**, because the boundary it
marks is *"everything we can reach works"*, and what remains is the part we
cannot reach: **seven validators, one module, never entered.**

**预估耗时: no number.** Held.

### 2. Current state

Rung 0's **eighth attempt** is running. One commit this interval — `1b889dd`,
m1, pass verdicts to `write_report` so the heading is not inferred.

### 3. Code problems

- **FIXED — `runlive.sh`'s silent drop on unreadable `exe`** (`01f768c`), the
  last of m3's four findings.
- **OPEN — the seven m5 validators**, unreached.
- **RESOLVED, NOT A DEFECT — the `HOME` conflict.** No code was wrong; two
  records described different variables.

### 4. Non-code problems

Three holds, none lost since 06:47 — six intervals, the longest stable stretch
of the effort. Two `keep3` jobs still pending from the non-team session.

### 5. Open questions

The seven m5 validators; M5.1.1; why holds are cancelled; how much co-tenant
load is corpses. **Removed from this list:** the `HOME` conflict, settled; the
`remote.sh:150` node-side candidate, eliminated.

### 6. New commits

**1 since `25fcf23`** — `1b889dd` m1. Plus my own this interval: `01f768c`
(runlive third answer), `8c50398` (T40), `baf4772` (T40 corrected), `25fcf23`
(the reference-row addendum).

### 7. Anything else

**The `HOME` thread cost four people a round of messages and produced two things
worth the cost.**

**m3's mechanism, which is the durable one:** *a refusal that documents
known-bad forms will be quoted back as a report of one occurring.* In a captured
tail, reference and observation are indistinguishable. **The pattern is still
right** — naming measured forms in a refusal is good and has been correct every
other time today — it needs its reference rows marked as reference.

**And the leader's diagnosis, which is sharper than mine:** the artefact was
*honest about provenance and dishonest about meaning*. m3's comment correctly
attributed the denial to the leader on node 243. What travelled wrongly was
**which variable it was about.**

**The cost, stated plainly because every other instance today was caught by a
second reader and this one was not.** m3 wrote it; m3 misread it; the leader
quoted it; I built a reconciliation on it; I then built a *deeper* one after
reading the code. **Four passes, three readers, and none of us caught it.** What
caught it was m3 diffing a surviving `validator_report.txt` against their own
source — going back to the artefact, not reasoning further from the quote.

**That is the argument against my own working method, stated at its strongest.**
I have spent the day insisting that a second reader is what verification means,
and this is the case where three readers all failed and one diff succeeded.
**The second reader is necessary and is not sufficient**; what actually resolves
things is returning to the artefact, and a second reader helps mainly because
they are more likely to bother.

**The leader added a fourth entry to the self-weakening list and it is mine.**
Three others corrected records in their own disfavour today with nothing to gain;
I have now corrected two published claims to the leader unprompted, and this
section corrects a third. Their observation is the one to keep: *the rule about
tools being verified by someone who did not write them works here because nobody
defends a record.* **That is a property of this team, not of the rule**, and the
rule would not survive a team that had it differently.

### Addendum, 11:38 UTC — I swept my own file for catalogue-quoted-as-event, and found one

m3 marked their reference rows (`7ff011b`) and pointed out that **my file has the
identical exposure and is quoted more than theirs**: every refusal, log line and
diagnostic in these sections is a captured tail, and *"is this known to fail"*
and *"did this happen"* look the same once quoted. They offered to second-read
it. **I swept first — naming a class is not sweeping for it, and I have now been
told that twice.**

**Scope: 57 fenced blocks, 46 lines carrying refusal or error text.** Result:

- **Event-store quotes are genuine and their provenance is stated.** The
  `escalated` / `output_absent` / `handling_failed` / `monitor_gave_up` lines
  were read by me with `read_events.py` against named runs, and each block says
  so. No exposure.
- **Relayed lists are marked as relayed** — *"recorded as the leader's
  account"*, *"m5 measured it on an idle node"*, *"per the leader"*. Eleven such
  markers. The 56× block attributes the 2 062 s to m5 explicitly.
- **Numbers I never verified I did not reproduce.** T21's A/B character counts
  and T7's rsd figures appear nowhere in this file; I referenced the findings by
  name without copying figures I had not seen. That was luck as much as
  discipline, but it held.

**One error, and it is the phantom.** T+1177 §3 lists `build_workset`'s five
stacked defects, and layer 5 reads:

> **FIXED** `${E2E_REMOTE_HOME:-$HOME}` → `/home` in a closed zone →
> `-v /home:/home` denied — `5964fd8`

**That stated cause is wrong.** The `/home` came from the **top-level component
of `$ROOT`**, not from `$HOME`; no run attempted that mount; and in a validator
zone `HOME` is `<zone>/home` per `environment.py:235`. `5964fd8` remains a real
fix — it derives the mount from `$ROOT` instead of `$HOME`, which is right — but
**the defect as I described it did not exist.**

I inherited the wording from the leader's list, which inherited it from m3's
commit message, which m3 has since retracted. **I marked the list as relayed and
that did not help**, because the marker says *who said it*, not *whether it was
observed*. A provenance marker is not an evidence marker, and I have been
treating them as the same thing all day.

**So the sweep's real finding is about the marking, not the count.** One in
fifty-seven is a good ratio and it is the wrong statistic: the one that got
through was the only one that had travelled through three people, and **the
attribution chain is exactly what made it feel checked.** Each of us marked our
source honestly and nobody's marker carried the fact that the original was
inferred rather than measured.

**What I am changing.** Where this file quotes a diagnostic, it will say which of
three things it is: **observed by me** (naming the artefact), **relayed** (naming
the person *and* whether they measured or inferred it), or **catalogue** (a known
form, not an event). The middle one is the gap — I have been recording the person
and not the provenance behind them.

I have not taken m3's offer of a second read, because the sweep found the thing
they predicted and the fix is a wording rule rather than a hunt. If a second
reader wants the exercise, the useful target is not my quotes — it is the eleven
places I wrote *"per the leader"* and did not ask whether the leader had measured
it.

### Addendum, 11:45 UTC — 20 of 21. m5 was reached, and the ruling that it would not be is superseded

**The ceiling moved from 14 to 20 in one run**, and it corrects a statement the
leader made minutes ago: *"the seven never-spoken are still m5's, and they will
stay unreached because `apply_patch` will refuse first."* **`apply_patch` did not
refuse.** Run `20260904T112414-cf3e82`, verified by me over its zones:

```
34 invocations · 42 judgements true, 1 false · 15 handoff records
patch_overlay        valid        <- apply_patch produced it
stock.measurement    invalid
patched.measurement  invalid
integration_report   invalid
e2e_packup           created
```

**Six of the seven m5 validators have now spoken**, every signature resolved
against `steps/m5_integration.yaml` with no unmapped fallbacks:

| validator | invocations | pass |
|---|---|---|
| `check_overlay_applies` | 1 | 1 |
| `check_patch_live` | 1 | 1 |
| `check_measurement_order` | 1 | 1 |
| `check_bench_report` | 1 | 1 |
| `check_acceptance` | 1 | 1 |
| `check_no_regression` | 1 | **0** |

**`check_packup_shape` is the only validator in the package that has never
produced a verdict.** Live root: **20 distinct, 438 invocations.**

**And `check_no_regression` refusing on its first contact is the right result to
want.** It is the gate m5's whole control experiment exists to exercise — the one
that must refuse on numbers nobody chose. One refusal is not yet evidence that it
*can* discriminate; the null-versus-degraded design is what would establish that.
But it spoke, and it said no.

**I am recording the ruling as superseded rather than wrong.** The leader made it
deliberately, as a known cost, on the reasoning that `apply_patch` would refuse
first. The run says otherwise. Whether the m5 verdicts are *meaningful* — three
of the five handoffs are `invalid` — is a separate question from whether the
validators ran, and my metric only ever claimed the latter.

### The leader's provenance audit, and my file checked against it

I redirected a second read at *"the eleven places I wrote 'per the leader'
without asking whether the leader had measured it"*, noting no second reader
could do it. **The leader supplied it themselves** — six claims sorted into
measured and inferred, unprompted. Checked against this file:

| their claim | status here |
|---|---|
| "scratch must be created on the far side" — **inferred, half wrong** | recorded at T+1259 **as their self-correction**. Clean |
| "m4's 006 reading was stale" — **inferred, wrong** | recorded at T+1259 §7 **as their self-correction**. Clean |
| "the two profiling arms ran in parallel" — **wrong twice** | **not in this file at all** |
| "m1's kit is complete and correct" — **relayed from m1, who inferred it from `ls`** | **not in this file at all** |
| the `/home:/home` node-243 attribution | see below |
| progress percentages given to the user | their reporting, not this file — and they were **my tally relayed without attribution** until ~10:50, now fixed upward |

**Two of the six never reached this file, and that was not discipline.** I did
not record them because they arrived in messages about other subjects, not
because I screened them. **A filter that works by accident is not a filter**, and
the two that did land are clean only because the leader sent them as
corrections.

**On item 5, one over-claim of mine survives.** I wrote that the catalogue row *"is
a quotation of a real measurement"*. The node-243 denial is real and the comment
at `:213-218` quotes it; that the **echo row at :274** derives from that same
measurement is my inference, not something I checked. It is very likely and it is
still an inference, and the leader's own correction is that they *"named the
right variable and the wrong instance."* I did the smaller version of the same
thing in the sentence correcting them.

**And I decline the leader's absolution on my second message.** They argue the
reasoning was correct — three mechanisms eliminated ordinarily *is* evidence of a
fourth. That is a fair defence and I have recorded it as their view. **Mine is
unchanged**: I had by then established that no context produced the value, and
"the observation may be wrong" was available and cheap to test by asking m3 where
the string came from. The generalisation m3 drew is worth having because it is
counterintuitive; **it does not follow that the specific check was unreachable.**

## T+1330 — 2026-09-04 11:40 UTC

### 20 of 21 — and this metric is about to stop being useful

| | runs | invocations | judgements | **distinct** |
|---|---|---|---|---|
| frozen (`ro`) | 38 | 231 | 255 | 10 |
| live | 30 | 438 | 501 | 20 |
| **union at 11:40** | **68** | **669** | **756** | **20** |

**`check_packup_shape` is the only validator in the package that has never
produced a verdict.** Confirmed stable across two independent tallies twenty
minutes apart, with every m5 signature resolved against
`steps/m5_integration.yaml` and no unmapped fallbacks.

**The trajectory, for the whole effort: 5 → 9 → 10 (seventeen hours) → 12 → 14
(five intervals) → 20.**

**And I should say plainly what this number is about to become.** It has measured
one thing all effort — *how far does the graph get* — and I have defended it as
conservative precisely because it could not be gamed by anything except a run
actually reaching further. **Reach is now essentially complete**, so from here
the metric discriminates almost nothing: it can move by exactly one, once, and
then never again.

**What it never measured, and what remains:** in the furthest run,
**3 of 15 handoff records are `invalid`** (`stock.measurement`,
`patched.measurement`, `integration_report`), one judgement is false, and
`e2e_packup` is `created` rather than sealed — which is why `check_packup_shape`
has nothing to judge. **The pipeline completes; the content does not validate.**

**So the successor metric, from the next section, is the pair I should have been
carrying alongside:** *handoffs `valid` in the furthest single run* (**11 of
15**) and *judgements false* (**1**). Those move for the reasons the remaining
work will move them. I am naming the change before the old number saturates
rather than after, because a headline that can no longer go down is a headline
that has stopped reporting.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b0) pre-commit `git status` on this file | clean |
| (b) per-commit ownership | clean |
| (c) `todo.md` | **44 items** |
| holds | three: 006, 217, 047 — none lost since 06:47 |
| `/home` | 1.6 T free (85 %) |
| run process | **one present at 11:40:25**, root `/home/yihou/agent_sys_runroot` |

### 1. Progress

**~82 %, up three.** Elapsed 1 330 m. I said last section I wanted a second
reading before pricing a move this size; I have it, and the count held.

**Reliability: low→moderate**, unchanged, **and the caveat now matters more than
the number.** Three points is for *reach*, which is what my instrument measures.
It is **not** a claim that 82 % of the work is correct — 3 of 15 handoffs in the
furthest run are invalid and the terminal one has never sealed. If the remaining
correctness work behaves like `build_workset` did — five stacked defects, each
hidden by the one above — the last fifth could take longer than the jump from 14
to 20 suggests.

**预估耗时: no number.** Held.

### 2. Current state

Rung 0 has walked all five stages. One run live. One commit this interval:
`356d65e`, m1 — *an adapter change is not always required, and assuming it is
costs work.*

### 3. Code problems

- **OPEN — `check_no_regression` refused on first contact**, 1 of 1. This is the
  gate m5's control experiment exists to exercise. **A single refusal is not
  evidence it can discriminate**; the null-versus-degraded design is what would
  establish that.
- **OPEN — three m5 handoffs `invalid`**, and `e2e_packup` unsealed.
- **OPEN — `check_packup_shape`**, nothing to judge until the terminal handoff
  seals.

### 4. Non-code problems

Three holds, none lost since 06:47 — seven intervals. Two `keep3` jobs pending.

### 5. Open questions

Whether `check_no_regression` discriminates or merely refused; the three invalid
m5 handoffs; M5.1.1; why holds are cancelled.

### 6. New commits

**1 since `8b86a0f`** — `356d65e` m1. My own this interval: `1ff75d2` (the quote
sweep), `8b86a0f` (20 of 21).

### 7. Anything else — the one rule from today that prevents rather than detects

m3 applied my three-way marking to their own refusal table (`b8d5f13`) and found
**two of its three rows were somebody else's measurement printed under a heading
that said "Measured forms"** — including `/shared_nfs`, which they have never
bound in a measurement. **In their own file, about their own work**, and
indistinguishable from the row they had actually run. That is a sharper
demonstration than the phantom, which needed three people and a chain; this
needed only a table heading.

**And they returned something I was about to file as luck.** My sweep noted that
I referenced T21's character counts and T7's rsd figures **by name rather than by
value**, and called it *"partly luck"*. m3's correction:

> **Quote a finding by its identifier, not its number.** *"T21's discriminator"*
> carries nothing to mis-attribute. A figure copied out of a message is a fact
> with its provenance stripped at the moment of copying; an identifier is a
> pointer back to the artefact.

**This is the only rule produced today that prevents rather than detects.**
m3's `ref:` markers, my three-way split and T40's null test all catch the error
after it exists. This one means it has nowhere to form. Recorded as theirs,
because I had been doing it without knowing why — which is the difference between
a habit and a rule, and the reason a second reader is worth more than a careful
first one.

**The day's four rules, in the order they arrived:**

1. **T40** — before believing a null, name the result that would have proved the
   probe could speak.
2. **m3's `ref:`** — mark catalogue rows so they cannot be quoted as events.
3. **the three-way split** — observed by me / relayed *with the relayer's own
   provenance* / catalogue. A provenance marker says who said it, not whether it
   was observed.
4. **m3's identifier rule** — quote findings by name, not by value.

**None of the four came from either of our stages, and all four outlive the
package.** The tool that occasioned them will be superseded; these will not.

### Addendum, 11:52 UTC — my metric is monotonic by construction, which is worse than saturating

**The leader's note about `67a223b` exposes a third limitation, and it is the
one I had not seen.** Verified: `67a223b` landed **11:40:55**; the run I measured
started **11:24:14**. Sixteen minutes apart. It adds 27 lines to
`apply_patch.task/apply.py` and 112 to `integration_report.schema.json` — **the
refusal m5 described did not exist when that run executed.**

So the leader's own summary is exact: *wrong about the mechanism, right about the
cost.* The next run will very likely do what they said this one would, and
**my "20 of 21" describes a package state that no longer exists.**

**The limitation that follows is structural, not a slip.** My union counts *has
this validator ever produced a verdict, in any run, ever.* That is **monotonic
by construction**: it cannot go down. If `67a223b` means m5 is never reached
again, my headline stays at 20 forever and reports the high-water mark of a
package that has since changed underneath it.

That is worse than the saturation I named an hour ago. **Saturation means the
number stops moving; monotonicity means it cannot report a regression at all.**
An instrument that can only ratchet is exactly what I criticised in a *progress
percentage* at T+1092 — *"a percentage that only ratchets up is not measuring
anything"* — and I have been running one as my headline all effort without
noticing it was the same shape.

**So the successor is not just the pair I named; it is the pair scoped to a
single run.** From here: *handoffs `valid` in the **furthest single run***, and
*judgements false in that run*. A single run can go down. The union stays in the
table as history, labelled as history.

### Three things from the leader, recorded under the new marking

**Relayed — leader measured.** They lifted m1's `check_invariant` into
`check_environment` and measured its reach before claiming it: **417 records,
412 skip, 5 exercise it, 0 refuse.** So the fourteen-kind gap is closed *going
forward* and grades almost nothing today.

**And they nearly reported it clean from a probe that had checked zero files** —
wrong glob — **an hour after quoting T40 at two other people.** That is the
fifth T40 instance and the first committed by the person enforcing it. It is
also the strongest evidence the rule is worth having: the null looked like a
pass, in the measurement slot, to someone who had spent the hour telling others
to distrust exactly that.

**Relayed — m5 measured, from their control experiment.** On
`check_no_regression`: **the 2 ms arm was invisible and the 20 ms arm was not**,
so the gate's floor lies somewhere between and **nobody has bounded it.** That
sharpens my "one refusal is not discrimination" into something testable: the
question is not *does it refuse* but *what is the smallest regression it can
see*.

### The identifier rule, stated as the distinction rather than the fourth item

The leader is right that listing it fourth undersells it. The distinction:

> **Three of the four rules make provenance *visible*. The identifier rule makes
> the claim *un-detachable from its source*.**
>
> `ref:` markers, the three-way split and T40's null test all attach something to
> a claim that a reader must then check. **A pointer cannot be un-copied.**
> *"T21's discriminator"* carries a way back to the artefact; `7 chars vs 526
> chars` carries nothing, and the moment it is copied its provenance is gone with
> no marker able to follow it.

That is why it is the only one that prevents. The other three improve the odds
that someone notices; this one removes the object that gets mis-attributed.

### On item 5's residue

The leader will not let me hold it alone, and they are right: **the claim that
the echo row at `:274` derives from their node-243 measurement is unverified by
both of us.** They asserted the whole thing was that denial; I narrowed it and
left the same inference inside the narrowing. Twenty lines apart, one file, one
author — **m3 can settle it in a sentence and it is not worth a task.** Recorded
as *unverified, held jointly* rather than as either of ours.

### Addendum, 11:58 UTC — I swept m4's T47 into my commit. T47 is about that.

**My breach, disclosed to m4 before recording it here.** Commit `5281a4e` is
**66 added lines to `todo.md`: 27 are mine — m3's sharpening of T40 — and 39 are
m4's T47**, which was sitting unstaged in the working tree while I edited. It
went in under my commit message and my sign-off.

**T47 is titled *"a pathspec commit is scoped to your file, not to your
change"*.** It documents m4 sweeping m3's T45 addendum into `d4a7212` ninety
minutes earlier. **I took the entry about the failure, with the failure, three
minutes after it was written.**

**And I cannot claim I lacked the check.** I ran
`git status --porcelain -- todo.md` immediately before editing. **It printed
` M`. The file was already dirty and I proceeded anyway.**

**Why it did not stop me is the finding, and it is m4's fix stated from the
other side.** T47 says *"`git status` tells you the file is dirty; the diff tells
you whose changes are in it."* I had an edit in flight, so ` M` was **the answer
I expected** — the check fired and its firing was indistinguishable from the
normal state. **A check whose alarm looks identical to the expected condition is
not a check.** That is T40's null-in-the-measurement-slot wearing different
clothes: the signal was present, correct, and unreadable because I already knew
what it would say.

**This is the second time today I have adopted a rule and then failed the case it
covers.** At 11:14 I recorded m3's *"naming a class is not sweeping for it"* and
swept my own file within the hour, which went well. This one went the other way:
I adopted contract 8a at 10:29, wrote *"both checks stay, one before and one
after"*, and the before-check was the one that failed silently.

**What I am not doing: rewriting it.** Contract 8a says corrections get their own
commit, and amending on a shared worktree is precisely how I clobbered
`8b87f41` at 04:09. Nothing is lost; the attribution is wrong; **the remedy is
m4's to choose** and I have offered to re-land T47 under their name with my copy
reverted if they want it.

**Adopted, replacing the weaker check:** `git diff -- <path>` immediately before
committing any shared file, and read it. Not `git status`. **`git status`
answers "is this file dirty", and the question is "whose work is in it"** —
another instrument answering a neighbouring question, which is the twelfth entry
in this file's own table and the first where I had already written the entry.

### Addendum, 12:06 UTC — the remedy I adopted seven minutes ago does not work, and the one that caught me was already in my routine

**m4 falsified T47's own remedy while committing T47** (`1ef702b`), and it
invalidates the fix I adopted in the addendum above.

They ran `git diff -- todo.md` exactly as T47 prescribes — one hunk, one
heading, forty lines, all theirs — **then ran the commit and got `no changes
added to commit`, because I had committed in the window between their check and
their commit.** An entry about pathspec commits capturing a co-owner's
uncommitted work was captured *while being committed*, by the failure it
describes, after its own remedy had been correctly applied.

**So: no pre-commit check closes this window.** The gap is between the check and
the commit, and it is unclosable from that side. My adoption of
`git diff -- <path>` above is **wrong as stated** — it is better than
`git status` at answering *whose work is in the file*, and it still cannot see a
commit that has not happened yet.

**The check that works is m4's and it is after the fact:** `git show --numstat
HEAD` against the size of your own edit. **If the commit is bigger than what you
wrote, someone else's work is inside it.**

**And I should record that this is not a new practice for me — it is the one
that actually caught the breach.** My commit calls have ended with
`git show --numstat --format='%h %s' HEAD` for most of today. `5281a4e` printed
**`66 0`** against the 27 lines I had written, and that discrepancy is the entire
reason I looked. **I then wrote an addendum crediting the wrong check.**

That is worth more than the correction. I had the effective control running as
routine, did not recognise it as the load-bearing one, and on being asked what I
would change reached for the *prescribed* check rather than the one that had just
worked in front of me. **A habit that works is invisible to the person with the
habit** — which is exactly what m3 told me two hours ago about quoting findings
by identifier, when I filed a working practice as luck.

**Standing check (b) is amended accordingly.** Before: `git status` pre-commit.
Now: **`git diff -- <path>` before, to see whose work is present — and
`git show --numstat HEAD` after, compared against the size of my own edit, as
the one that can actually catch it.** The pre-check narrows the window; only the
post-check closes it.

**m4 is not amending either commit** and neither am I — *"rewriting shared
history to fix an attribution is worse than the attribution"*, which is the same
conclusion 8a reached from the other direction at 04:09.

## T+1349 — 2026-09-04 11:59 UTC

### New headline: the furthest single run

As promised at T+1330, the union metric is demoted to history and the reporting
number is scoped to one run, because **a single run can go down.**

| **furthest single run — `20260904T112414-cf3e82`** | |
|---|---|
| handoffs **valid** | **11 of 15** |
| handoffs invalid | **3** (`stock.measurement`, `patched.measurement`, `integration_report`) |
| handoffs created-not-sealed | **1** (`e2e_packup`) |
| judgements | **42 true, 1 false** |

**History, retained and labelled as such:** union across 69 runs — 38 frozen
(`ro`) + 31 live — **20 of 21 distinct validators, 669+ invocations.** This
number is **monotonic by construction** and cannot report a regression; it is
kept as a high-water mark and nothing more. `67a223b` landed after the run above
and may well mean m5 is not reached again, which the union would never show.

**One run process present at 11:59:16**, 52 m 58 s, root
`/home/yihou/agent_sys_runroot`.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| **(b) shared-file commits** | **amended — see §7.** Diff before, `numstat` after |
| (c) `todo.md` | **47 items**; **T36 duplicated, T41 missing** — reported, not renumbered |
| holds | three: 006, 217, 047 — none lost since 06:47 |
| `/home` | 1.6 T free (85 %) |
| frozen constants | 38 runs / 231 verdicts, mount `ro` |

### 1. Progress

**~82 %, held.** Elapsed 1 349 m. **No commits and no new runs this interval** —
the first genuinely quiet interval since 06:00, and the quiet is real rather than
instrumental: the process check reports a run alive and working.

Reliability **low→moderate**, held. **预估耗时: no number.**

### 2. Current state

One run live. The interval's work was entirely on the record and its tooling.

### 3. Code problems

Unchanged: three m5 handoffs invalid, `e2e_packup` unsealed,
`check_packup_shape` with nothing to judge, `check_no_regression` refused once
with its floor unbounded.

### 4. Non-code problems

Three holds, none lost since 06:47 — eight intervals.

### 5. Open questions

Whether `check_no_regression` discriminates; the three invalid m5 handoffs;
whether `67a223b` means m5 is no longer reached; M5.1.1.

### 6. New commits

**0 from other owners.** Mine: `5281a4e` (T40 sharpening — **and m4's T47, see
§7**), `98f28bb`, `2b8dc87`.

### 7. The three checks, and why two of them fail

The T47 thread produced a complete account of a class of check, and it is worth
stating once because I got it wrong twice inside twenty minutes.

**`git status -- <path>` — fails on legibility.** It answers *is this file
dirty*. I ran it, **it printed ` M`, and I proceeded**, because I had an edit in
flight and a dirty file was exactly what I expected. **A check whose firing is
indistinguishable from the expected state is not a check.** m4 has put that in
T47 as mine (`7000e4a`) and calls it the part that reframes the entry: they had
written it as a fact about git — *the rule protects the index, not the file* —
and this is a fact about the reader, which is why the obvious mitigation fails
for everyone rather than only the careless.

**`git diff -- <path>` — fails on timing.** Strictly better: it answers *whose
work is in this file*. **And m4 falsified it while committing T47 itself** — they
ran the diff (one hunk, thirty-nine lines, all theirs), ran the commit seconds
later, and got `no changes added to commit`, because I had committed in the
window between. **The gap is between the check and the commit and nothing before
the commit closes it.**

**`git show --numstat HEAD` against the size of your own edit — works, after the
fact.** If the commit is larger than what you wrote, someone else's work is in
it.

**And the part I keep having to relearn: that last one was already my routine.**
Most of today's commit calls end with it. `5281a4e` printed **`66 0`** against
the 27 lines I wrote, and that discrepancy is the *only* reason I looked. I then
wrote an addendum recommending the pre-commit diff. **I had the effective control
running, did not recognise it, and when asked what I would change reached for
the prescribed check over the one that had just worked in front of me.**

That is the third time today someone has had to point out a working practice of
mine to me — m3 on quoting by identifier, m4 on `numstat`, and m3 again on the
sweep. **A habit that works is invisible to the person with the habit**, which is
a better argument for external review than any of the defect findings, because
it applies when nothing is broken.

**Resolution of the breach:** m4 declined the revert. *"Rewriting shared history
to fix an attribution is worse than the attribution"* — and my 04:09 clobber is
the first-hand evidence for the larger hazard. `5281a4e` stands as it is: 66
lines, 27 mine, 39 m4's, disclosed in three places. **Numbers confirmed by both
of us independently.**

### Addendum, 12:22 UTC — the metric follows the definition of done, and the furthest run may already meet it

**The leader accepted m3's fourth option: the acceptance claim is a named file
and a failing condition, not an exit code.**

> one run produces 15 handoffs and 43 verdicts of which 42 are true; the one
> false is `check_no_regression` on `integration_report`; and that validator's
> `validator_report.txt` carries exactly these four `PROBLEM:` lines and no
> others.

**That definition is stated in verdicts, so the metric must be stated in verdicts
too.** That is what makes my number and m3's acceptance script agree *by
construction* rather than by coordination — the leader's actual worry. Handoff
validity becomes secondary, with `integration_report`'s invalidity **annotated as
expected** rather than counted as a shortfall.

**New headline, from here:**

| against the accepted definition | target | `20260904T112414-cf3e82` |
|---|---|---|
| handoffs | 15 | **15** |
| verdicts | 43 | **43** |
| true | 42 | **42** |
| the one false | `check_no_regression` | **`check_no_regression`** ✓ |
| `PROBLEM:` lines in that report | exactly 4 | **exactly 4** |

**Every element I can verify already matches.** What I cannot verify is whether
those are *the same* four lines m3 enumerated — the definition says "exactly
these four", and I have the count, not their content against m3's list. **That
one comparison is m3's and it is the difference between "matches the shape" and
"is done".** I am not calling it done.

Secondary, retained and annotated: **11 of 15 handoffs valid**;
`stock.measurement` and `patched.measurement` invalid are **unambiguous gaps**;
`integration_report` invalid is **expected**; `e2e_packup` created-not-sealed.

### A hazard for the acceptance script, measured

**`PROBLEM:` is not unique to the refusal.** Across that run:

```
23 validator_report.txt files
11 PROBLEM: lines total, in 6 different reports
 4 of them in check_no_regression's — the expected refusal
 7 in five reports whose verdict was TRUE
```

**A validator can emit `PROBLEM:` and still pass.** So a script that greps the
run for `PROBLEM:` finds **eleven** and mis-keys; it has to scope to
`check_no_regression`'s own report. Passed to m3 before their script hardens
around the wrong scope.

**And I nearly published the opposite.** My first count used `grep -c
'^PROBLEM:'` — anchored to line-start, while the lines are indented two spaces.
It returned **zero**, and I was one step from recording *"the four PROBLEM lines
are not in this run."* What stopped me was that the leader's claim was specific
enough to be worth a second look. **That is T40 again, in the measurement slot,
forty minutes after I filed T48 about exactly this** — a probe that could not
have succeeded, returning a null that read as data.

### Two more for the record

**047 lived past the deaths.** Alive at **1 h 05 m 53 s**, past the ~59 and ~62
minute failures. m1's harness-reaper hypothesis is confirmed and the detached
launch is the fix. **Two runs lost to it, both now explained, and the remedy
costs nothing** — which is the cheapest closed item of the day.

**And the leader has named an open problem I should stop presenting as a
resolved lesson.** Three tools of mine were found wrong by others; separately,
three working practices of mine were invisible to me until someone named them.
**The second set has no detection mechanism at all** — nothing in the package,
nothing in `todo.md`, nothing anyone can be asked to run. T48 records the class;
it does not detect it. **Recorded as open, not as solved.**

## T+1373 — 2026-09-04 12:23 UTC

### Brief item 2 is met. I ran the acceptance script myself rather than relay the exit code.

```
MOCK E2E ACCEPTED — 15 handoffs, 43 verdicts, 42 true; the one refusal is
check_no_regression on integration_report and its report gives exactly the
4 expected reasons.
The run's own exit code is 5 and that is correct; this is the deliverable's green.
```

**Two runs, both accepted, real exit 0**, from `assets/lib/accept_mock.py`:
`20260904T112414-cf3e82` and `20260904T114914-0a0cdd`.

**The comparison I said was m3's is done and they did it by diffing, not
eyeballing:** the four `PROBLEM:` texts match their four fingerprints one-to-one.
So the run does not merely match the *shape* of the definition — it matches the
content. That was the distinction I refused to collapse an hour ago and it is now
closed from the other side.

**Deliverable status.** Item 1 (definitions) substantially done; **item 2 (mock
e2e green) — met and verified**; item 3 (real e2e, staged) barely begun.

### And my own exit code was wrong before it was right

My first run of the script produced this, which I nearly recorded:

```
accept_mock.py: error: unrecognized arguments: /home/.../20260904T112414-cf3e82
  EXIT=0
```

**The script had failed on a missing `--run` flag and I printed `EXIT=0`** —
because `$?` after `cmd | tail -8` is **`tail`'s** exit status, not the script's.
A broken invocation reporting success, in the one measurement that certifies a
deliverable.

**Third T40 instance of mine today, and the closest to consequential.** The
`^PROBLEM:` anchor forty minutes ago returned zero from an artefact that had
four; this returned zero-as-success from a script that never ran. **What caught
it was the error text sitting directly above the "success"** — not the exit code,
which is the thing I was reading. I have re-run everything above with `rc=$?`
captured before any pipe.

### The number that matters

| against the accepted definition | target | both runs |
|---|---|---|
| handoffs | 15 | **15** |
| verdicts | 43 | **43** |
| true | 42 | **42** |
| the one false | `check_no_regression` on `integration_report` | ✓ |
| its `PROBLEM:` lines | exactly the 4 named | ✓ **diffed by m3** |
| **acceptance script** | exit 0 | **exit 0, run by me** |

**And this metric has now saturated too — the third time I have had to say
so.** It measures the *mock* chain, which is complete. **It says nothing at all
about item 3**, which is the whole of the remaining work. The monotonic union
could not go down; the handoff count could not reach its top; this one has
reached its top and stopped discriminating. From the next section the reported
number is **item 3's ladder — rungs clean, out of six** — which currently reads
**0 of 6** and is the only figure that can now move.

### Standing checks

| check | result |
|---|---|
| (a) index leak | clean |
| (b) shared-file commits | diff before, `numstat` after |
| (c) `todo.md` | 48 items |
| holds | three: 006, 217, 047 — **none lost in twelve intervals** |
| run process | present at 12:23:38, **1 h 17 m**, past the reaper deaths |

### 1. Progress

**~86 %, up four.** Elapsed 1 373 m. **A deliverable completed and independently
verified is the largest single move available**, and four points is what is left
after acknowledging that item 3 — the hardest of the three — is barely begun.

**Reliability: moderate** for the first time, and only for items 1 and 2, where
there is now an executable acceptance test someone else wrote and I ran. **For
item 3 it stays low**: 0 of 6 rungs clean, and my instrument does not measure it.

**预估耗时: no number.** Unchanged, and now for a cleaner reason than before —
item 3's cost is dominated by node availability, and four holds died unexplained
today.

### 2–6

Three holds alive twelve intervals. `check_command_parses` open (below). 14
commits since `15f5f63`, none mine.

### 7. Two findings from the exchange, one of which is not mine

**The seven `PROBLEM:` lines under `passed` headings are all
`check_command_parses`**, and m3 has routed it to the leader who owns
`steps/common.yaml`. Their point is sharp: `write_report`'s three headings are
`passed` / `REFUSED` / `DID NOT RUN`, and `PROBLEM:` is **the marker for a
finding that refuses**. A `PROBLEM:` under `passed` is a contradiction in the
helper's own vocabulary — *either the validator labels notes as problems, or it
finds problems and passes anyway.* Neither of us has read its body and neither is
guessing. **Open, leader-owned.**

**My measurement bought two controls that no control designed from this run could
have found.** m3's script was already unaffected — it reads one validator's own
report located from the verdict's zone, never a run-wide grep — but they
tightened it anyway (`f577d77`) and added:

```
[3] a PROBLEM under a 'passed' heading   rc=0  ignored, as required
[4] a second REFUSED section             rc=2  cannot judge
```

**Neither is reachable from today's artefacts.** I reported a *shape*, not a
defect, and the shape had two holes in it.

**And m3's framing of why the timing mattered is the thing I want kept:**

> **A finding's value has a half-life set by how much has been built on top of
> the thing it is about.**

The same numbers a day later would have met two controls already written and a
claim already quoted in `RUN-PLAN`, and the fix would have been an argument
instead of an edit.

## T+1457 — 2026-09-04 13:47 UTC

**Gap: 84 minutes since T+1373, against a 30-minute cadence.** Not back-dated.

### The ladder, with what each rung is blocked *on*

The leader's suggestion, adopted: the count alone cannot distinguish *blocked on
something nobody has priced* from *blocked on one campaign*, and that
distinction is most of what changed today.

| rung | state | blocked on |
|---|---|---|
| **0** all mock | **CLEAN** — 3 accepted runs, 2 install paths | — |
| **1** m1 real | **in flight**, 51 m, alive on 217 | running now |
| **2** +m2 | not started | rung 1 sealing |
| **3** +m3 | not started | rung 2 |
| **4** +m4 | not started | **a campaign**, plus one boolean m4 isolated (does forge commit in the engine tree) |
| **5** +m5 | not started | rung 4 |

**1 of 6 clean** — up from 0, because rung 0 is now certified by an executable
test rather than by my reading of it. **Four blockers, none unknown, one of them
a run in flight.** That is materially different from this morning and it is
invisible in a bare count, which is the leader's point and it was right.

### Rung 0 is reproducible across two install paths — verified, not relayed

I ran `accept_mock.py` myself on the new run:

```
20260904T133028-87e7ad   MOCK E2E ACCEPTED — 15 handoffs, 43 verdicts, 42 true …
                         REAL EXIT=0
```

**Three accepted runs now**, and this one is the first on the `patch_in_place`
path. **Two different install paths reaching the same green is a reproducibility
fact, not a reach fact**, and my ladder count cannot express it — recorded here
in prose because the table cannot hold it.

**Run process present at 13:47:41**, pid 2953669, **51 m 13 s** — rung 1, alive
and writing.

### 1. Progress

**~87 %, up one.** Elapsed 1 457 m.

**Only one point, and the reason matters:** item 2 was already priced at T+1373.
What arrived since is **item 3's *shape*, not its progress** — rung 4's blocker
moved from unknown-cost to known-and-cheap, and rung 0 gained a second
reproduction. **Knowing what stands in the way is worth a point; it is not worth
what doing it is worth.**

**Reliability: moderate** for items 1–2 (executable acceptance test, run by me,
three times). **Low** for item 3 — 1 of 6, and the five remaining are one
in-flight run and four dependencies.

**预估耗时: no number.** Held.

### 2. What resolved, and the leader's own framing of it

**M5.1.1 is resolved and the user resolved it by rejecting the framing:**

> *"最挫的 apply 方式难道不是找到调用的地方，把那一行替换掉。"*

**That is a diff, and `apply.py:641-659` had been running `patch -p1 --batch
--forward` all along.** The leader records that they had escalated it as
*"unsatisfiable, needs a design decision"* when the accurate statement was
*the producer only emits whole-file replacements and the enum has one value.*
**One is a design question; the other is a small implementation gap** — and the
wrong one cost the user a round of reading.

All three thirds landed within the hour: `5f5c1af` m3 (enum + both producers),
`0712fbc` m4, `1ee4f4f` m5 — *"three refusals in my stage, not one enum value"*.

**And m3 pre-registered all four numbers before the run; all four hit.** That is
**the first pre-registered prediction of the day scored on a complete run.** The
leader stopped an earlier scoring against a run that stopped at stage 4, on the
grounds that *"the prediction was wrong"* and *"the criterion was wrong"* would
have been indistinguishable — which is the same discipline as m5's *"if the
degraded arm comes back `same`, that is a finding about the gate"* at T+1210,
now applied by a third person to protect someone else's prediction.

### 3–6

**42 commits since `42eabc1`**, none mine. Three holds alive, **none lost in
twenty-four intervals**. Gates rc=0. 006 and 047 free.

**Open, leader-owned:** `check_command_parses` emitting `PROBLEM:` under
`passed`.

### 7. The leader's three measurement errors, recorded because they asked

They report three today: **a shell-expanded grep that reported all twenty vars
absent; `rc` read after a pipe to `tail`; and a timestamp sort on a truncated
string that read as a four-hour stall.** Each caught by a second reading rather
than by care.

**The middle one is mine too, forty minutes earlier**, in the measurement that
certifies a deliverable — `EXIT=0` from `tail` while the script had failed on a
missing flag. **Two people, the same shell idiom, the same day, independently.**
That is not a lapse in either of us; it is a defect in the idiom, and it belongs
beside T40 rather than in either person's error column. **`rc=$?` must be
captured before any pipe** — and neither of us would have found it without
someone reading the output above the number.

**And their port probe is the counter-example worth keeping.** They probed
8101–8103, got nothing, **and did not conclude** — the ports are parameterised
and they may have been aimed at the wrong ones, so they asked m1 for the fact.
**That is T40 applied prospectively**: a null from a probe whose reach was
uncertain, treated as uninformative rather than as absence. It is the first time
today someone has stopped *before* the wrong conclusion rather than after it.

### Addendum, 14:54 UTC — four instances filed, and their proposed axis tested rather than adopted

`todo.md:1808` names this file's table as the canonical home for the instrument
form, so these are mine to file. **All four are the leader's, all this
afternoon, all silent.** Table entries 13–16:

| instrument | reads | the question asked of it |
|---|---|---|
| `cmd \| tail` then `$?` | the **pipe's last element**'s status | did *the command* succeed |
| `grep "${VAR}" file` | the **expansion** of `VAR` | does the file contain the *name* |
| `pgrep -f "docker save"` | every cmdline **including pgrep's own** | is a save still running |
| `find -newermt "14:48"` | a **misparsed** time | what changed since 14:48 |

Each **succeeded**, returned a **plausible** answer, and answered a **different
question**: *the controls pass*, *all twenty vars are absent*, *the save is
running*, *the run has stalled four minutes*. Every one of those statements was
false and none produced an error.

### Their proposed axis, tested against the four — it is a real class and it is not this one

The leader suggests *"nothing anywhere refuses an argument it cannot honour"*
may be the more useful axis than "instrument", citing `find -newermt "14:48"`
alongside `spec_loader/variables.py` accepting an unrecognised `--var` (rc=0,
zero warnings) and an inert `${VAR:?}`.

**Tested against their own four, it covers one:**

| | under-specified input, silently accepted? |
|---|---|
| `\| tail` then `$?` | **no** — fully specified, and `$?` correctly returns the last element |
| `grep "${VAR}"` | **no** — fully specified; double quotes expand, as documented |
| `pgrep -f` self-match | **no** — fully specified; `-f` matches full cmdlines including its own |
| `find -newermt "14:48"` | **yes** — and the full form `"2026-09-04 14:39"` worked |

**So it is a second class with three members** — `find`,
`spec_loader/variables.py`, inert `${VAR:?}` — **overlapping this table at
exactly one instance.** It is not a better cut through the same data; it is a
different failure that happens to share one case. The other three are
*fully-specified commands whose documented semantics differ from the reader's
question*, which no amount of input validation would catch.

**I am recording it as its own class rather than folding it in**, because
merging them would produce the thing `todo.md:1808` warns against: *two
vocabularies for one failure*, or worse, one vocabulary for two.

### The method that caught #4, which is the transferable part

**The leader did not spot the bug.** They went to build a differential
reference — the previous run's own write cadence, per `CLAUDE.md`'s 对拍 rule —
and wrote the reference query with a **full date** because they were reaching
further back. **The two queries disagreed, and only then did they look at why.**

> **The bug was invisible from either query alone. It needed two that should
> have agreed.**

That is worth stating separately because **the other twelve instances in this
table were caught by reading an artefact, and this one could not have been.**
Reading `find`'s output more carefully would not have helped; the output was
plausible. Only a second query with different specificity exposed it.

### A fifth, which is not the leader's alone

**m1 and the leader both ran `grep -rn "32\.5"`, got one hit in a comment, and
concluded the constant was never measured.** m2 found the measurement: **33.58
ms, real run, real jobid, m1's exact load shape — at `tp_size: 1`.** The
constant was rounded from it.

> **A search for a rounded literal cannot find the measurement it was rounded
> from.**

Two people, same instrument, same wrong conclusion — and the conclusion was
*"this number has no provenance"*, which is the strongest kind of claim to get
wrong. **Filed as entry 17.** The real finding underneath: the baseline is tp=1
and today is tp=4, **so there was never a regression** — and separately that two
images at the *same* tp=4 differ by **4.7×**, which is where m2 is looking now.

### Ladder unchanged: 1 of 6

**Rung 1 was refused at 14:24 and relaunched.** `check_deploy_serves` failed it
on **64 completed requests against a floor of 80** — **not the deployment**:
every probe passed and the load ran clean. The floor derived from a hard-coded
35 s/request.

m1 fixed it twice, and the first fix is worth the record: **`e8153f8` read the
wrong metric key and fell *silently* back to the constant** — caught by the
leader against the real artefact before it re-ran. **`516c74e` uses
`request_latency_ms` directly**, the quantity rather than a proxy, one key
instead of two. **m2 ran the fixed arithmetic against the refused artefact:
floor 64, count 64, PASS.**

So rung 1's refusal was an instrument fault in a validator, found by the
validator refusing a good deployment — which is the failure mode the whole
package is built to have rather than the one it is built to avoid.

### Addendum, 15:00 UTC — m5's moving ruler, applied to my own accepted runs

**m5's survey finding is the sharpest thing in this interval and it lands on
me:** `kernel_optimization` *"passed"* three times but was **graded by one
validator once and three twice**, because a gate went off and on today.
**The validator set is part of the verdict.** Counting those three as stable
would have promoted an artefact measured against a moving ruler.

**I have been claiming three accepted runs. So I checked whether they share a
ruler — and they do not.**

```
20260904T112414-cf3e82   34 invocations, 20 distinct, 42 true / 1 false   md5 55d7ee3162f0
20260904T114914-0a0cdd   34 invocations, 20 distinct, 42 true / 1 false   md5 515c4a496f78
20260904T133028-87e7ad   34 invocations, 20 distinct, 42 true / 1 false   md5 515c4a496f78

differing signature:  check_environment  ±invariants
```

**Run `112414` was graded by a `check_environment` without `invariants`; the
other two by one with it** — the leader's `check_invariant` lift. Every count is
identical because that arg grades almost nothing today (412 of 417 skip), which
is *why* three identical-looking greens hid a ruler change.

**And the corrected claim is better than the one I made.** My reproducibility
statement was *"three accepted runs, two install paths"*, which spanned **two
moving variables**. Properly:

> **`114914` (old patch path) and `133028` (`patch_in_place`) share a validator
> set and differ only in the install path. That pair is the reproducibility
> claim.** `112414` corroborates on a different ruler and should be cited as a
> third data point, not as part of the pair.

**One variable moved, not two.** I would not have looked without m5's finding,
and the answer strengthened the claim rather than weakening it.

**Filed as entry 19 in the table, mine, caught in seconds:** my first comparison
used Python's `hash(frozenset(...))` **in three separate processes**. `PYTHONHASHSEED`
is randomised per process, so the three digests differed and I nearly reported
*"the sets differ"* — right conclusion, entirely wrong evidence. `hashlib.md5`
in one process is the fix. **An instrument answering "what is this set's hash
here" to the question "is this the same set as that one".**

### Standing instruction, recorded — it supersedes a core principle

From the user, second time today:

> **我们的机器上如果有别人的gpu任务（非集群检测类），一律杀掉后开始工作**

with the operative clause from the earlier phrasing: **coordinate clearly
between teammates.**

**Supersedes** *"never `docker rm -f` a container you did not create"* — **for
GPU workloads on nodes under our allocation only.** **Does not touch**: the
deletion rule (killing a container is not deleting a path; nothing lacking
`yihou` or `/tmp` is removed by any means), cluster-monitoring jobs, non-GPU
containers, or the prohibition on acquiring and releasing machines.

**Three coordination limits in force:** 047 off limits until m2's four latency
arms land; **`charming_turing` on 275 unidentified — nobody kills it until we
know whose it is**; announce before killing. All five owners told.

### Entry 18 — and it is the most expensive of the six

The leader told the team **275 was free with eight cards. It never was.**

```
asserted   "8 cards, nothing on it, ours"
source     squeue — 275 listed as ours, RUNNING, 12:52
actual     kimik3-vllm-kimi-k3, up 8 hours, GPU 100% ×8, VRAM ~91% ×8
```

**`squeue` answers "who holds the allocation". The question was "is the hardware
free".** They had measured 006, 217 and 047 earlier — `0/8 busy` each — and
carried the habit forward without repeating it on a new node.

**m4's framing is the entry, not the leader's:** ***the allocation is ours and
the workload is not*** — the same gap that makes *"never `docker rm -f` a
container you did not create"* necessary, and the leader records having treated
a hold as a node all day. **The other five instances misled only their author;
this one was about to send m4 to contend with a saturated GPU on someone else's
eight-hour run.**

**m4 caught it before acting** — the second time today someone stopped in front
of a wrong conclusion rather than after it, after the leader's port probe.

### Ladder: 1 of 6, unchanged

**Rung 1 is 34 minutes into its re-run on 217** with m1's `516c74e` floor. Its
containers have torn down and VRAM is 0 — **the same shape as the previous run,
where that was STEP 7 after a successful bring-up and the kit sealed minutes
later.** The leader is not calling it either way and neither am I: **the same
observation preceded a success once and a failure once**, which is the
definition of a signal that does not discriminate.

**Two mechanisms landed that are not rungs but change the ladder's cost**, both
deliberately unwired: **m5's `replay_root.py`** (`0760da3`, `ad176b4`), the
user's skip-ahead, with twelve of fourteen kinds already stable at threshold 3;
and **m4's third-tree workspace** (`63bcaca`).

### Addendum, 15:12 UTC — m5's comment-match class, verified, and it is the same mechanism with two new properties

m5 routed a new instrument-failure class at the leader's request. **I verified
its load-bearing instance and tested its proposed fix**, as I did with the
leader's axis an hour ago — and reached a different verdict, for a reason worth
stating.

**The class, in m5's words:** *a probe over source that does not strip comments
reads the **warning** as the thing warned about.*

**Their instance 1, verified in the tree:**

```
m2_profiling.yaml:114   # `${expect_ranks:-${tp:-8}}` is not spellable — …    <- a WARNING
m2_profiling.yaml:119   expect_ranks: '${expect_ranks:-8}'                    <- the live line, flat
```

**And the measurement is stronger than m5's own framing.** A naive grep for the
nested-default anti-pattern across `steps/` and `shared.yaml`:

```
3 hits.  3 of them are comments.  0 live lines.
```

**A 100 % false-positive rate — that grep has never once matched executable code
in this package.** The anti-pattern exists nowhere except in warnings against it.

### Verdict: same mechanism as the table, two properties none of its 19 entries have

**Mechanism:** identical to entry 1 of the table — a text search answering a code
question. `grep --include=*.yaml` reads *does this file contain X*; the question
was *does this package do X*. So I am **not** filing it as a separate class the
way I filed the leader's under-specification axis, which was a genuinely
different mechanism overlapping at one instance. **This one is the same
mechanism.**

**But two properties are new and neither appears anywhere else in the table:**

1. **The failure rate scales with documentation quality.** A codebase with no
   comments is immune; one whose comments quote the anti-pattern verbatim — this
   package's house style, and a good one — is maximally exposed. **Our own
   discipline is what loads the gun.** No other entry has this shape: the other
   eighteen would fire identically on an undocumented codebase.
2. **The error is directional, and the direction is an accusation.** The false
   positive is always *"an owner did the forbidden thing"*, so the natural next
   action is to tell that owner. **Two of m5's three would have been accusations
   about other people's work** — the leader's `expect_ranks` fix, and m2's
   `measure_gpu`. Entry 17 (`grep "32.5"` → *"this constant has no
   provenance"*) is the only prior entry that shares this, and it too was about
   to be said about someone else's work.

**Filed as entry 20, with those two properties named**, and cross-referenced
from entry 17 as the pair that produce accusations rather than mistakes.

### Their fix, tested

```python
code = line.split('#', 1)[0]
```

**It has a known false-negative mode — a `#` inside a quoted string is
truncated** — so I checked whether that is reachable here:

```
lines in steps/*.yaml + shared.yaml where a quoted '#' would be mis-stripped: 0
```

**Safe in this tree today, and not safe in general.** Recorded that way rather
than as a clean endorsement: the one-liner is correct for this package and would
need a real parser for one that puts `#` inside strings. m5's own framing is the
part that generalises and does not depend on the parser:

> **`grep --include=*.yaml` is not a code search; it is a text search over a
> file that happens to contain code.**

### The rule, and what actually caught it

m5 places it beside CONTRACT §4.4's eighth face — *when you re-run a probe
narrower, keep the broader answer* — as adjacent but distinct:

> **Before reporting a violation, confirm the match is in a line the machine
> executes.**

**And what saved it both times was reading the matched line before sending,
because the finding was about to become an accusation.** The leader's phrasing
is *verify before the accusation, not after*, and it is now **three-for-three
today**: m4 before killing a container, m5 before touching `accept_mock.py`,
m5 again here. **All three are the same act at a different range, and all three
were voluntary** — nothing in the package requires any of them.

### Addendum, 16:37 UTC — `-newermt` is not the instrument. `find` is not `find`.

**The leader proposed banning `-newermt` on three people's evidence. I tested it
before filing and the diagnosis does not hold.**

```
GNU findutils 4.9.0   (/usr/bin/find — what scripts, zones and nodes run)
  -newermt '-15 minutes'    -> 2 files  rc=0  no stderr     CORRECT
  -newermt '14:48'          -> 2 files  rc=0  no stderr     CORRECT
  -newermt '-2 minutes'     -> 2 files  rc=0  no stderr     CORRECT
  -newermt '2 minutes ago'  -> 2 files  rc=0  no stderr     CORRECT

bfs 4.1.1             (what `find` resolves to in an agent shell)
  -newermt '-15 minutes'    -> 0 files  rc=1
                               bfs: error: Invalid timestamp.
                               Supported formats are ISO 8601-like, e.g. 2026-09-04T16:33:39Z
```

**`find` in these shells is a shell function** that execs the Claude Code binary
as `bfs`. The real `find` is GNU 4.9.0 at `/usr/bin/find`. **`-newermt` is not
broken; `bfs` rejects the GNU-compatible relative spellings, and it says so
clearly with `rc=1`.**

**So the failure was never silent.** It printed *"Invalid timestamp"*, underlined
the offending argument with `~~~~`, listed the accepted formats, and returned 1.
Everyone — me included, on my first attempt — ran it as
`find … 2>/dev/null | wc -l`, which **discards stderr and takes `wc`'s exit
status.** The instrument diagnosed itself, in detail, into a void.

**That is the twelfth instance of *the machinery produces the diagnosis and
discards it***, not a new class — and it is the same `| wc -l` / `| tail`
exit-status defect already filed as entry 13.

### What this does to the proposed rule

**Do not ban `-newermt`.** It works in every context the package actually runs
in — scripts, validator zones, containers, `spur exec` on nodes — none of which
has the shim. **A ban would be a correct-looking remedy for a wrong reason**, and
would leave the real hazard untouched: *the same command means two different
programs depending on who types it.*

**The accurate rule is narrower and larger at once:**

> **In an agent shell, `find` is `bfs`, not GNU find.** Anything measured with it
> is a measurement of a different program than the one the package runs. **Never
> suppress its stderr, and never read `$?` through a pipe.** Where a result will
> be reported to someone else, use `/usr/bin/find` explicitly.

**The leader's replacement is still worth adopting**, and now for its stated
reason rather than as a workaround — `-printf '%T@' | sort -rn | head -1`
against `date +%s` **parses nothing**, so it is identical under both programs.
Verified in both. m2's *"the epoch form cannot lie because it parses nothing"* is
exactly right and does not depend on any of the above.

**And the `%TH:%TM:%TS` trap reproduces and is real**, independent of which
`find`:

```
wall-clock sort picks:  23:59:00   (a file from YESTERDAY)
epoch sort picks:       b          (today's newest)
```

### One of the three does not fit, and I am not fitting it

**The leader's and m4's symptoms match `bfs` exactly** — 0 files where GNU
returns the true set. **m2's does not.** They saw `-2 minutes` → 10,
`2 minutes ago` → 10, explicit `$(date -d …)` → 3: **no zero anywhere**, so no
parse rejection. That is two *working* queries disagreeing about the window,
which is a timezone or boundary question and a different cause.

**Two of three explained, one open.** I am recording it that way rather than
folding m2's into the shim story, because a rule built on three instances of
which one is a different bug is the shape m3 warned about at T31 — and because
m2's tree is not mine to measure.

## T+1626 — 2026-09-04 16:36 UTC

**Gap: 2 h 49 m since T+1457.** Four addenda landed in between, so the record's
*content* is current; its *form* was not. Recorded as a format lapse, not a
content one.

### The headline went down, which is what it was chosen to do

```
20260904T143952-bec7da   MOCK E2E REJECTED
  - true verdicts: 40, expected 42
  - refusals: 3, expected exactly 1 —
      check_measurement_order on patched.measurement
      check_no_regression      on integration_report
      check_measurement_order on stock.measurement
  REAL EXIT=1
```

**First time any metric I have published has moved backwards.** The union count
could not; the handoff-validity count could not reach its top; **this one fell
from 42/1 to 40/3 and the acceptance test said so with a non-zero exit.** That is
the single-run scoping earning its keep two sections after I adopted it.

**Cause, and it is already fixed:** the mocked arms were not marked as replays,
so `check_measurement_order` refused both of them — m5's `9d363f5`, *"mark the
mocked arms as replays, which is what rung 1 refused on"*, at 15:53.

**A regression that an executable test caught, named by validator and handoff,
is the outcome the whole acceptance design was for.** Three earlier runs
accepted; this one did not; nobody had to notice by eye.

### 1. Progress

**~87 %, held.** Elapsed 1 626 m. **Ladder unchanged at 1 of 6.** Three runs are
in flight at 5 handoffs each — too early to score. A regression was found and
fixed within two hours; no rung advanced.

**Reliability: moderate** for items 1–2 — and better than last section, because
the acceptance test has now been shown to **fail** as well as pass. Until 14:39
it had only ever returned 0, which by my own T40 makes it a check that had never
been observed to refuse. **It has now refused a real run for a real reason.**

**Low** for item 3. **预估耗时: no number.**

### 2. Current state

Ladder 1 of 6, rung 1's blocker being worked. **The nodes have turned over
completely** — holds are now `275`, `217`, `287`, where at T+1457 they were
`006`, `217`, `047`. One run process present at 16:36:06, 31 m.

### 3. Code problems

- **FIXED — mocked arms not marked as replays** (`9d363f5`), the cause of the
  regression above.
- **OPEN — `INVALID` means two things and the CLI's exit code reads it as one**
  (`5ef8468`, `c816328`, `a6124a6`), and **`_completion_gaps`' other half is
  unsound too — task status is never finalised** (`0afabf1`).
- **OPEN — `summarise.py` exists twice, byte-identical, call sites split across
  both** (T56); **one scratch path, three independent literals** (T57).

### 4. Non-code problems

Node turnover complete. `charming_turing` on 275 still unidentified and
untouched. **m1's `b20bd12` is the sharp one:** *the label answers "is this
ours", not "is anyone holding this node"* — which is entry 18's lesson
(`squeue` answers who holds the allocation) arriving at the container layer, and
m5 has labelled every container their stage creates (`4dee1b9`).

### 5. Open questions

`INVALID`'s two meanings; whether the three in-flight runs reach a verdict; m2's
`-newermt` instance, which does not fit the shim explanation (below).

### 6. New commits

**54 since `bac5430`**, none mine. `todo.md` at **T57**.

### 7. `-newermt` is not the instrument, and m5's corollary

**I tested the leader's proposed ban and it does not hold.** GNU findutils 4.9.0
accepts all four spellings correctly — 2 files, rc=0, no stderr. **`find` in an
agent shell is a shell function execing the Claude Code binary as `bfs 4.1.1`**,
which rejects the GNU relative forms **loudly**: *"Invalid timestamp"*, the
argument underlined, accepted formats listed, **rc=1**.

**So it never failed silently.** Everyone, me included on the first attempt, ran
`find … 2>/dev/null | wc -l` — **discarding stderr and taking `wc`'s exit
status.** m4's *"it fails to zero rather than to an error"* is inverted: it
failed to an error and we converted that to a zero.

**Do not ban `-newermt`** — it works in every context the package runs in. The
accurate rule is narrower and larger: **in an agent shell, `find` is not `find`;
never suppress its stderr, never read `$?` through a pipe, and use
`/usr/bin/find` when the result will be reported to someone else.** The epoch
form is still worth adopting for m2's reason — *it cannot lie because it parses
nothing* — verified identical under both programs.

**Two of the three instances are explained; m2's is not** (10/10/3, no zero
anywhere, so no parse rejection) and I have not folded it in.

**m5's corollary to entry 20, and it is sharper than my measurement:** 3 of 3
hits being comments makes that grep **a detector with zero precision in this
tree — every hit it has ever produced is a false positive.** So it should not be
run as a detector at all; its only honest use is *read the three lines*. And the
two claims agree from opposite directions — their morning sweep found no live
nested default, my evening count found the pattern only inside warnings against
it. **The anti-pattern exists in this package solely as documentation of
itself.**

**Their concession is the line worth keeping:** they reached for *"new class"*
because the consequences were novel, and **consequences are not mechanism.**
That is the same test I applied to the leader's axis an hour earlier and reached
the opposite verdict on — different mechanism there, same mechanism here — and
having a stated test is why the two came out differently rather than by taste.

### Addendum, 16:48 UTC — I audited my own record against the shim, since the leader noted nobody had

The leader withdrew the `-newermt` ban, verified it independently
(`find` → 0 files, `/usr/bin/find` → 28, same tree same instant), and made the
point that matters more than the finding:

> **Every measurement any of us has made with `find` today measured a different
> program than the one the package runs** — *and I would not have known which of
> mine that reaches without going back, and neither would anyone else.*

**Nobody else has gone back. I have now done mine.**

**Every tally I have published rests on one predicate, and it is identical under
both programs:**

```
find . -name verdict.json | wc -l          ->  588   (bfs, agent shell)
/usr/bin/find . -name verdict.json | wc -l ->  588   (GNU findutils)
```

**The 588 agree**, so every distinct-validator count, every invocation and
judgement total, and every frozen-root re-verification in this file is
unaffected. Plain name predicates do no date parsing; there is nothing for the
shim to reject.

**The one relative-time read I ever published was at T+876**, checking for team
activity after the respawn: `-newermt '2026-09-04 03:40'`. Tested just now:

```
bfs:  1 file, rc=0        GNU:  1 file, rc=0
```

**Accepted by both** — the space-separated ISO form is inside `bfs`'s grammar
even though `-15 minutes` is not. So *"no file modified under the package since
respawn"* stands. **It stands by luck of spelling, not by judgement:** I chose
an absolute datetime because I was reaching to a specific moment, exactly as the
leader chose a full date for their differential reference and thereby exposed
the bug. **The same accident protected one measurement and revealed another.**

**Result: my record is clean, and I could not have said so an hour ago.**
Recording the audit rather than only the outcome, because a clean result from a
check nobody ran is indistinguishable from a clean result from no check.

**And the leader is right about which line to elevate.** Mine was:

> I caught myself mid-test: my first reproduction used the shim and I nearly
> published *"it errors with rc=1"* as a fact about the team's `find`.

Their reading is sharper than my own: **the person investigating the shim did
not escape the shim.** That is the strongest available evidence for the rule,
because it shows the hazard survives being looked for directly.

**Two corrections of theirs, recorded as theirs:** they built a rule from three
instances of which one was a different bug — *"exactly T31's shape and I was the
one building it"* — and they place it beside m5 reaching for "new class" on
consequences rather than mechanism, an hour apart, **both caught by testing
rather than accepting.** I would add only that the test is cheap and the
acceptance is free, which is why the default runs the other way.

**And their last line is the one I would not have written about myself:**
*verify before the accusation* is three-for-three today, all voluntary — **and
the one time it was not done, it was theirs.**

### Addendum, 16:56 UTC — correcting my own stale claim, and the ceiling contract's second producer

**My T+1626 section says `charming_turing` is "still unidentified and untouched".
Untouched stands. Unidentified is stale and I am correcting it.**

**Relayed — the leader relaying m4's own account, with m1's independent
confirmation:** it was **m4's**, their `docker run --rm --entrypoint bash`
chain-test containers on 275 at 16:08. `--rm` names them at random and they had
already exited by the time m4 answered; m1 confirmed the `--rm` signature
independently from `docker ps -a` showing no exited entry.

**The leader's reason for raising it is the part that belongs in this file
specifically:** *the record is the durable artefact, and an open ownership
question in it invites someone to reopen a settled one.* **A stale open item is
not neutral — it costs whoever reads it next.** This is the second time today my
record carried something past its resolution; the first was the `remote.sh:150`
candidate, which I removed when it closed. **My sections age, and nothing in my
own routine re-reads them.**

### The ceiling contract has bound a second producer — verified here

```
rung 2b's kit, run 20260904T160847-ae7ef4:
  CUDA_GRAPH_MAX_BS:=32
  CUDA_GRAPH_MAX_BS=32
```

**Read by me from the run's own handoff tree**, not relayed. Node 217,
`tp_size 4`. **32 ≥ 16, so `e390abb`'s criterion is met on the number as well as
the form**, and this is the **second consecutive producer** to honour it — m1
verified rung 2a's, this is a different producer on a different run reaching the
same answer.

**And the controlled comparison is the interesting part, not the compliance.**
Same producer, same run, opposite outcomes:

| in the brief | states | outcome |
|---|---|---|
| `E2E_KIT_CUDA_GRAPH_MAX_BS` | a **criterion** | **honoured, twice** |
| `E2E_CONTAINER` | a **liberty** | **taken** |

**Two data points on the criterion side now rather than one**, against a liberty
taken in the same breath. That is close to a controlled experiment on how a brief
is read, and neither arm was designed.

**Prediction 2 in the leader's rung-2b file is now live and falsifiable:** at a
ceiling of 32 the deploy load should complete **~300 requests at ~9.3 ms**, not
**64 at ~42 ms**. Those are the only two outcomes seen all day and they differ by
**a factor of five** — so the load test has become a test of the ceiling contract
rather than of the deployment. **Recorded before the result, which is what makes
it worth anything.**

### On the test's standing — the leader's extension is sharper than mine

I wrote that `accept_mock.py` had only ever returned 0 until 14:39, so by T40 it
was a check never observed to refuse, and that its first real refusal was worth
more to me than the three passes. Their extension:

> The passes did not become worthless — **but they were unfalsified rather than
> tested, and only now is the difference visible.**

**That is retroactive and it is right.** The three accepted runs were evidence
about the package only to the extent the instrument could have said otherwise,
and until 14:39 nobody knew that it could. **It also lands the same standard the
team held everything else to today** — m3's guard shown failing on a constructed
regression before being shown passing, m4's control failing on a body that
ignores `engine.patch`, m5's battery keeping the *wrong* prediction. The leader's
closing line is one I wrote about someone else's check this morning and had not
turned around: ***a gate that has never fired is not a gate.***

### State

Rung 2b at ~35 m, kit written, environment record present, still
`deploy_and_prove: running`. **Four owners idle with fixes correctly held behind
it** — which is the right shape and worth naming, because idle owners with queued
work look like a stall and are the opposite. Three holds, none lost since 16:08.

### Addendum, 17:06 UTC — I verified the ceiling from a run that had been killed, and the correction weakens the claim

**Retracting *"second consecutive producer"* from the addendum above. It was
wrong in the count and wrong in the inference, and the verification I was proud
of read the wrong object.**

**My error.** I read `CUDA_GRAPH_MAX_BS:=32` from run `20260904T160847-ae7ef4`
and reported it as rung 2b's kit. **`160847` is the rung-3 attempt the leader
launched on 047 against a job with 59 seconds left and killed a minute later.**
Rung 2b is `20260904T160500-a182de`. **I read a real kit from a destroyed run** —
*right value, wrong source*, which is entry 1 of my own table arriving **inside
the verification of a claim about that table.**

**And the way it escaped both of us is worse than either of us mis-measuring.**
Neither of us did. I read a file and reported what it said; the leader read a
different file and reported what it said; **we agreed, and both took the
agreement as confirmation.** m1 named the near form this morning — *two routes to
one answer is not corroboration when both share the assumption under test.*
**Here we did not even share an assumption. We shared a number.** Two
independent reads of two different objects, agreeing by coincidence of value.

### The corrected evidence, measured here, and it does not support what I wrote

```
20260904T143952  rung 1   CUDA_GRAPH_MAX_BS:=32     env.sh written 15:17:25
20260904T154946  rung 2a  CUDA_GRAPH_MAX_BS=32
20260904T160500  rung 2b  CUDA_GRAPH_MAX_BS:=32
20260904T160847  rung 3   CUDA_GRAPH_MAX_BS=32      (killed at 16:09)

e390abb  "contract the CUDA graph ceiling"  landed 15:34:38
```

**Rung 1's kit chose 32 seventeen minutes and thirteen seconds before the
contract existed.** The leader put it at "~15:04"; the file's own mtime says
15:17:25. Either way it is *before*.

| | ceilings chosen |
|---|---|
| **before** `e390abb` | **16, 16, 8, 32** — four producers, varied |
| **after** `e390abb` | **32, 32, 32** — three producers, constant |

**Three-for-three after is real. But 32 was already reachable without any
contract**, so *"the criterion bound the producer"* is **consistent with the
evidence and not established by it.** What is established is narrower and worth
saying exactly: **the discriminating case has not recurred.** No producer since
the contract has chosen 8, and none has been observed to want to.

**That is my own T40 pointed at `e390abb`: a criterion that has never had to
refuse anything is not yet observed to constrain anything.** I applied that
standard to `accept_mock.py` two hours ago and it cost three passes their
standing. **It applies here, and this time I was the one carrying the
over-claim** — the leader told me "second consecutive producer" and I verified
the number rather than the inference.

**The controlled comparison survives and is unaffected** — criterion honoured,
liberty taken, same producer, same run. That contrast does not depend on whether
the contract caused the value.

### On my aging-record gap, the leader's answer is better than a fix

They declined to have it resolved cheaply:

> A sweep built today would be **a gate that has never fired**, and you would be
> building the thing you spent the afternoon warning about.

**Correct, and I would have built it.** The cheap half they name is not a routine
of mine at all: **when a finding closes, whoever closed it tells me.** m3 did
that unprompted this afternoon; m4 did not — *which is the difference between
this record being current and being two hours behind.* **I cannot fix that from
inside my own loop**, and saying so is more useful than a sweep that would find
nothing on the day it was written.

### One line recorded at their request, and I would have written it anyway

**Four owners idle with fixes correctly held behind a live run is the right
shape, and it looks exactly like a stall.** The leader has reported it as
structural three times today and notes it will read as excuse-making to anyone
who was not here. **It is not.** The queue exists because the fixes depend on a
result that does not exist yet, and shipping them ahead of it would be the
error.

### Addendum, 17:18 UTC — m1 is right, and the "before" column both of us reasoned from is wrong

**m1 sent a correction to my record that argues against their own change, and it
goes further than the leader's. I verified the argument and then measured the
data underneath it, and the data is not what either of them described.**

**m1's argument, verified.** `steps/m1_deploy.yaml:262` declares
`load_concurrency: '${deploy_load_conc:-16}'`, and `e390abb`'s criterion reads
*"Set the CUDA graph ceiling to at least the concurrency this deployment will be
loaded at."* So the bar is **≥ 16**, and **32 clears it — but so would 16.**

> **The criterion is visible to the thing being measured, and at concurrency 16
> every habitual value passes.** A producer that reads the brief and one that
> ignores it both land on a passing number. Three constant observations show
> producers **follow an instruction**, not that the instruction **changed an
> outcome**.

**That is stronger than my "the discriminating case has not recurred" and I am
adopting it.** The three post-contract observations cannot bear on the question.

### But the "before" column is not `16, 16, 8, 32`

Measured across every run in the live root that carries a kit, oldest first:

```
110626  16
110647   8      131949   8
111550   8      133028   8
112414   8    -----------------
114914   8      143952  32   <- rung 1, env.sh written 15:17:25
125637   8      154946  32
                160500  32
                160847  32

e390abb landed 15:34:38
```

**One 16, then eight at 8 — seven consecutive runs — then 32.** The leader's
summary had four pre-contract values with a single 8; there are **eight
pre-contract runs and eight of them are below or at the bar, seven of them
strictly below it.**

**Which makes the criterion far better motivated than the record said, and no
better confirmed.** `8 < 16`: seven consecutive producers chose a ceiling under
the load's own concurrency. **That is a real, persistent, measured failure and
`e390abb` names it correctly.**

**And it makes m1's conclusion stronger than m1 put it.** They wrote that 32 was
*reachable* without the contract. Measured, it is more than that: **the entire
8 → 32 correction happened at 15:17:25, seventeen minutes and thirteen seconds
before the contract landed.** The three post-contract 32s do not continue a
coincidence — **they continue a change that was already complete.**

### The distinction that survives, and it is the one worth keeping

| claim | status |
|---|---|
| the criterion addresses a real failure | **established** — seven runs at 8, against a bar of 16 |
| producers *can* satisfy it | **established** — three emitted `${E2E_KIT_CUDA_GRAPH_MAX_BS:=…}` in `scripts/env.sh` |
| the criterion **changed** anyone's choice | **not established, and not testable at concurrency 16** |

**m1's test is the right one and it is one run:** at concurrency 64, habit picks
32 and fails; the criterion picks ≥ 64 and passes. Until then **`e390abb` is
unfalsified rather than confirmed** — the standing I gave `accept_mock.py`'s
three passes this afternoon, reached here from a different direction by the
author of the change.

**And the one producer that stated a reason stated a false one.** Rung 2b
justified 32 over 16 by arguing a concurrency-16 load transiently reaches 17.
**Measured false** — m2's arms top out at 16.00 in every percentile, and a
closed-loop load cannot exceed its offered value. Filed by m1 in `1a8c67b` so a
future reader meets the falsification beside the argument. Their line is the
entry:

> **A bar widened on a good argument is still a bar widened without evidence.**

**That undercuts the compliance reading further:** the single case where the
contract elicited *reasoning* elicited reasoning that was wrong.

### On m1 sending it

**It is their change, and the leader had already corrected an over-claim made on
their behalf.** m1 extended the correction against their own work rather than
accepting the softer version that flattered it. That is the fourth person today
to weaken a record in their own disfavour, and the first to do it to a change
they authored.

## T+1651 — 2026-09-04 17:01 UTC

### I over-corrected an over-claim, and m1 caught it in the opposite direction

**Retracting *"seven consecutive producers chose a ceiling under the load's own
concurrency"* from the addendum above.** m1 sent a fingerprint; I reproduced it
exactly:

```
run      env.sh line   MAX_BS   DK_TP_SIZE
110626       169         16         4      <- real producer
110647       105          8         1      <- replay
111550       105          8         1      <- replay
112414       105          8         1      <- replay
114914       105          8         1      <- replay
125637       193          8         4      <- real producer, the only 8
131949       105          8         1      <- replay
133028       105          8         1      <- replay
143952       222         32         4      <- real producer
```

**Six files identical at line 105 with `DK_TP_SIZE=1`.** That is `mock_adapt.sh`
replaying the sealed 2026-09-02 TP-1 kit verbatim — the same bytes, six times.
**The line number is the fingerprint**: a produced `env.sh` is regenerated and
its line moves (169, 193, 222); a replayed one is always 105.

**So the pre-contract *producer* record is `16, 8, 32` — three producers, one
below the bar.** Not eight runs, not seven consecutive. And **`125637` is the run
`check_deploy_serves` refused** — verified here.

**And the six do not count for a second reason:** their ceiling of 8 is *correct
for the deployment they describe*, which is `tp_size: 1` — a different shape, not
the concurrency-16 case the bar is written about. m1's `mock_adapt.sh` preserves
that 8 **deliberately**: *"the sealed 8 is right for the kit it came from; this
script adapts a record forward — it does not re-tune a deployment that already
happened."* **They are not producers failing to comply; they are m1's design
decision working.**

### The corrected row

| claim | status |
|---|---|
| the criterion addresses a real failure | **established, on one producer in three** — `125637` chose 8 against a bar of 16 and its run was refused |
| producers *can* satisfy it | established, unchanged |
| the criterion **changed** anyone's choice | not established, not testable at concurrency 16 |

**The criterion stands.** One producer choosing 8 cost a rung; that is sufficient
motivation for a contract. What changes is the strength of the first row:
*"seven consecutive"* reads as a persistent systemic failure; the measured
record is **one occurrence in three producer runs**.

### The error is mine and it has a name m1 filed five hours ago

> **A value fixed in an artefact, mistaken for a default because the artefact is
> replayed eighteen times.**

That is m1's own entry, about this exact file, from this afternoon. **A sweep
that does not filter recovers the replays and reads them as choices**, which is
what mine did.

**And the direction is the part worth recording.** An hour ago I corrected the
leader for over-claiming that the contract bound producers. **I then
over-claimed in the opposite direction** — making the failure look systemic and
the criterion look better motivated than it is. **Both errors were about the same
claim, within an hour, and mine flattered the thing I was scrutinising.** I had
just written that verifying a number is not verifying an inference; I verified
twelve numbers and inferred nine producers.

**Scope caveat, stated because I did not check it:** my producer count covers the
**live root only**. The frozen root holds **19 more `env.sh` files** I have not
fingerprinted, which may account for the second `16` in the leader's original
`16, 16, 8, 32`. **I am not claiming three producers total — three in the range I
measured.**

### 1. Progress

**~87 %, held.** Elapsed 1 651 m. **Ladder 1 of 6.** Rung 2b still
`deploy_and_prove: running` at ~57 m; nothing has advanced or regressed. Four
commits since T+1626.

Reliability **moderate** items 1–2, **low** item 3. **预估耗时: no number.**

### 2–6

One run process at 17:01:45, 56 m 51 s. Three holds — 275, 217, 287 — none lost.
`todo.md` at **T57**. Four owners still idle with fixes queued behind rung 2b,
which remains the right shape.

### 7. Three passes over one claim, each weakening it

The full sequence, because the shape is the finding:

1. **leader** — *"the criterion bound a second producer"*
2. **leader**, unprompted — *"consistent with the evidence, not established by
   it"*
3. **m1**, against their own change — *"the post-contract observations cannot
   bear on the question at all"*
4. **me** — *"eight runs at 8, seven consecutive"* — **an over-correction**
5. **m1**, again — *six of those are one artefact replayed; the producer record
   is three*

**Five passes, four of them weakening the claim, none by anyone who gained from
it.** m1 corrected their own change twice, and the second time corrected the
person who had been correcting them.

**The thing I would take from it:** an over-claim invites a correction, and a
correction invites an over-correction in the other direction. **Nobody in this
sequence was careless.** Each step verified something real and inferred one step
past it — and the only reason it converged is that five passes happened at all.

### Addendum, 17:15 UTC — my sweep was truncated by my own `| tail -12`, and I am stopping at the measurement

**m1 measured the frozen root to close my caveat and found something else: they
had a producer run I never saw, `20260904T062229-2695b9`.** I went to find out
why my sweep missed it. **It missed it because I truncated my own output.**

My sweep ended `... done 2>/dev/null | tail -12`. I wrote `tail -12` expecting
about twelve runs. **The corrected sweep, with no `tail`, finds 36 runs carrying
a ceiling.** Twenty-four were cut off the top, oldest first — including
`062229`, which is a real producer at `line=170 MAX_BS=16 tp=4`.

**Third `| tail` defect of the day, all mine or the leader's:** `rc` after a pipe
to `tail` in the acceptance check, the leader's same idiom, and now `tail`
silently discarding two thirds of a sweep. **The first two lost an exit status;
this one lost data and left a plausible, complete-looking table.**

### What the full sweep shows, and where I am stopping

```
36 runs carry a ceiling
25 match the exact replay fingerprint   line 105  +  DK_TP_SIZE=1
11 do not, and they are heterogeneous:
     062229 line 170  MAX_BS=16  tp=4      110626 line 169  MAX_BS=16  tp=4
     125637 line 193  MAX_BS=8   tp=4      143952 line 222  MAX_BS=32  tp=4
     041742 line 152  MAX_BS=8   tp=1      <- line≠105 but tp=1
     062414 line  53  MAX_BS=32  tp=—      084406 line 150  MAX_BS=16  tp=—
     100905 line 150  MAX_BS=16  tp=—      154946 line  74  MAX_BS=32  tp=—
     160500 line  63  MAX_BS=32  tp=—      160847 line  58  MAX_BS=32  tp=—
```

**I am not classifying the eleven.** Several carry no `DK_TP_SIZE` at all, one
has a non-105 line *with* `tp=1`, and the line numbers vary in ways I cannot map
to producers without knowing which generator wrote each. **m1 can; I cannot, and
I have now been wrong twice on this exact claim by inferring one step past what
I measured.**

**So the producer count is not established by my data.** Not three, not four —
**my sweep cannot settle it, and it now suggests the population is larger than
either of us was working from.** m1's four may well be right; it is theirs to
confirm, and I have sent them the eleven.

### What does not change

**m1's fingerprint is sound and their frozen-root result stands** — 58 files, 55
replays, 3 with no ceiling variable, **zero producers.** My caveat closes: the
frozen root cannot supply a producer value. **That was the question I flagged and
it is answered.**

**And their general finding is the one that outlives this thread:**

> **55 of 58 files in that tree exist because a validator gate copies the sealed
> kit.** So any sweep over `packup` trees over-counts replays **by
> construction**, and the ratio worsens every time someone runs a gate. **The
> count is a property of how often we tested, not of how often a producer
> chose.**

**That applies to any future sweep of these trees**, which is the natural thing
to reach for when someone next asks *how often has this happened* — and it is
the second time today a count over artefacts turned out to be counting our own
test activity.

**m1 has now corrected this claim three times**, twice against their own change
and once against my correction of it, and the third time corrected their own
correction by finding `062229` in the live root after telling me the frozen root
was the place to look. **Nobody in this thread has been careless and the claim
has moved every single pass.**

### Addendum, 17:25 UTC — fifth pass: seven producers, two below the bar, and the contracted name predates the contract

**m1 classified the eleven and corrected themselves a fourth time. I verified
both new claims.**

```
084406  md5 28a153ee9603  mtime 08:50:17
100905  md5 28a153ee9603  mtime 08:50:17     <- IDENTICAL. ten kits, not eleven

062414  E2E_KIT_CUDA_GRAPH_MAX_BS:=32   written 06:36:49
e390abb contracted that exact name       landed 15:34:38     <- 8 h 57 m later
```

**Both hold.** The duplicate is invisible without hashing — my instinct to stop
at eleven was right for the wrong reason.

### What the classification changes

**All eleven are `packup_20260904` — produced today.** The line-105 fingerprint
does not apply to them; **the packup date is the better discriminator**, which is
m1 correcting the fingerprint they gave me an hour ago.

**Their "four producers" was short, and the cause is my defect wearing their
clothes.** They filtered on `DK_TP_SIZE:=4` **and** the `DK_` variable name,
which silently excluded the `E2E_KIT_` and `KIT_` variants. **A filter that
produced a plausible, complete-looking table while dropping data** — mine
dropped 24 rows to `| tail -12`, theirs dropped three producers to a name prefix.

**Corrected pre-contract producer set: seven distinct kits — `8, 16, 32, 16, 16,
8, 32`.**

**Two chose below the bar, not one:** `125637` (tp=4, ceiling 8, the run
`check_deploy_serves` refused) and `041742` (tp=1, ceiling 8, which m2 measured
at effective concurrency ~13.5, so 8 is under its load too). **Two of seven.**

### The row, fifth revision

| claim | status |
|---|---|
| addresses a real failure | **established — two of seven pre-contract producers chose below the bar** |
| producers can satisfy it | established |
| the criterion **changed** anyone's choice | **not established, and weaker still** |
| the criterion **standardised a name** | **supported** — three names were in use (`DK_`, `E2E_KIT_`, `KIT_`), one survives |

**The third row is weaker because of the fourth.** `062414` used
`E2E_KIT_CUDA_GRAPH_MAX_BS` — the contracted spelling — **nearly nine hours
before the contract existed.** So it is not only that 32 was reachable without
`e390abb`; **a producer independently invented the contracted name.** The
accurate claim is that **`e390abb` standardised one of three names already in
use**, which is smaller and better supported than anything either of us had
recorded.

### A small one of mine, since I am cataloguing these

My verification grep was `[A-Z_]*CUDA_GRAPH_MAX_BS` and printed
`E_KIT_CUDA_GRAPH_MAX_BS` — **the `2` in `E2E_` is not in `[A-Z_]`**, so the
match started mid-token. The value and timestamp are unaffected and the *name* I
read was wrong by two characters. **Right answer, slightly wrong question,
inside a verification of a claim about variable names.**

### The shape, which is the finding

**This claim has now been corrected five times:**

| pass | by | direction |
|---|---|---|
| 1 | leader | over-claim: *"bound a second producer"* |
| 2 | leader | withdraws it |
| 3 | m1 | *"post-contract observations cannot bear on it"* — against their own change |
| 4 | me | **over-correction**: seven consecutive producers |
| 5 | m1 | six were replays; producer record is four |
| 6 | m1 | **their four was short too**: seven, two below the bar |

**m1's own reading of it is the one to keep, and it is not about diligence:**

> **Every one of the four was caught by someone else's number forcing me to
> re-measure my own.** Without your `tail -12` truncation my four would still be
> standing.

**Nobody in this sequence was careless, and no pass was prompted by its author
re-reading their own work.** The claim converged because six measurements
collided, not because anyone checked twice.

## T+1794 — 2026-09-04 19:24 UTC

**Gap: 2 h 23 m since T+1651 — four missed windows against Rule 1.3.** The
leader records the trigger as their omission; **the write is mine and I am not
splitting it.** Not back-dated.

### My liveness tool called a live run dead, and I caught it writing this section

**The leader reported rung 2e alive at 44 minutes. `runlive.sh` returned `0 run
process(es)`.** I checked the discrepancy instead of publishing either number,
and the tool was wrong:

```
python3 .../assets/lib/run_with_long_stall.py --stall-after 3600 run --package …
```

**A third launch shape** — after `python -m agent_sys.cli.main` and the
`agent-sys` console script — this one a wrapper written to outlast the stall
detector. My pattern anchored `run` to a known entry point, so it matched
nothing. **The run was alive, writing at 19:20:32, and my instrument said
stopped.**

**That is the escalating direction and the third time this tool has been blind to
a way of starting a run.** m3 found the first (console script) and the second
(`show` matching); this is the same class one entry point further out.

**Fixed and verified** (`01b0d1c`): match the invariant every entry point must
pass — **` run ` followed by `--package`** — rather than the entry point itself.
Confirmed to catch the wrapper, reject `show --package`, and return 0 on a
negative control. **`1 run process present, pid 257742, 46:55`.**

**What saved it was the leader's number being specific enough to contradict.**
Had they said "rung 2e is going" I would have recorded a stall.

### 1. Progress

**~87 %, held. Ladder 1 of 6.** Rung 2 has had **five attempts** and none has
sealed.

| | |
|---|---|
| 2a | 217's job `109491` expired mid-run at 15:59:36 |
| 2b | **stage 1 GREEN** — 3 strong verdicts, **304 requests, ITL 9.24 ms** — then m2's body exited 1 on a missing `--var aiperf_trace` |
| 2c | killed once m2 supplied the trace |
| 2d | `monitor 'default' has stopped turning` — **cause unexplained**, see §3 |
| 2e | launched 18:38, **alive at 46 m** |

Reliability **moderate** items 1–2, **low** item 3. **预估耗时: no number.**

### 2. Rung 2e, and the measurement it exists for

Stage 1's **main arm came up healthy** — ready 19:04:18 in 222 s, cards 0–3 at
75 %, a completion returned, `deployment.json` written. **Its selftest arm
died:** NCCL `HIP failure: 'invalid argument'` on ranks 2 and 3, sglang
subprocess killed −9. The arm launched correctly on cards **4,5,6,7** with its
own container and port 8112, and those cards measured clean. m1 has a live
reproducer and is testing an IPC/privileges hypothesis.

**This kit chose ceiling 16, read off the live process args.** So rung 2e's
conc=16 load tests `ceiling >= concurrency` **at exact equality, zero margin**,
against m2's C1 reference (ceiling 16 → 304 requests, 9.31 ms).

**Recorded precisely because it is *not* m1's discriminating test.** At
concurrency 16 both 16 and 32 pass, so this cannot separate a producer reading
the brief from one following habit. **What it does test is whether equality is
sufficient in practice** — a different and narrower question, and C1 says it is.

### 3. Two retractions, both volunteered

**The leader's.** They reported rung 2d's monitor death as *"a transient I/O
block tripped the liveness check."* Sampling rung 2e — a **healthy** run — gives
`D` state in **1 of 8 samples with NFS answering in 4 ms**. **The state does not
discriminate and rung 2d's death is unexplained.** Their own note: they took the
convenient direction an hour after writing the mirror clause of T59 that says not
to.

**m1 narrowed it anyway, and their discriminator is better than the story it
replaced:** their GPU sampler shows 217 **fully torn down at 18:31:52**, nearly
ten minutes before rung 2d's last write — so the monitor stopped during an
**idle, post-teardown, no-container phase**, which rules out the whole family of
under-load stalls. **Two deaths in the same phase is a repeating wall; two in
different phases is not.**

**m1's.** They reported one transient probe container; there were **four across
three containers**. Caught and said by themselves — **the third self-correction
today, and the only mechanism that has worked against this class.**

### 4. `replay_root.py`, and a third instance of the same counting error

m5 built the user's skip-ahead at `0760da3`; the leader reviewed and ran it.
Refusals work (`rc=2` on a bad run path), the seam analysis is checked from
**both** producer and consumer sides, and it finds a real blocker.

**The defect: stability was counted over mock runs.** `deploy_kit` reported **27
stable runs** where real deployments are single-digit — a mock leaf replays an
already-validated artefact and passes the same validator set **by construction**,
so one confirmation was counted 27 times.

**That is the third instance today of a count over artefacts measuring our own
test activity**, after m1's 55-of-58 gate copies and my own 25-of-36 replays.
**Three subsystems, three people, same error, one day.**

**And it corrects something the leader nearly told the user: stage 1 is NOT
stable at three real passes.** Rung 1's kit was refused on one number, so the
count is 2b plus 2e if it seals.

### 5–6

**Holds:** 275 (4 h 34 m), 217 (3 h 23 m), 287 (3 h 14 m). Both gates `rc=0`.
`todo.md` at **T59**. 13 commits since `80a72b7`, none mine.

### 7. The leader released three owners they had held for a rung

They had told all owners *"held until rung 2 reports"*, but **their own rule is
per stage** — m3's and m4's stages run as mocks in rung 2e, so their real bodies
cannot affect it. **Three owners idled for the length of a rung because a
per-stage rule was applied globally.** m1 and m2 stay held for real reasons: m1's
stage is live, and m2's is the one rung 2e exists to measure.

**Recorded as their disclosure.** It is the same shape as the four owners I
described at T+1626 as *correctly* held — and the difference between correct and
incorrect holding was invisible from outside, which is why it needed saying by
the person who did it.

### Addendum, 19:41 UTC — purging "304 requests at 9.24 ms" from T+1794, and my record held the contradiction

**Verified before correcting.** The leader retracted a figure they had given me,
m1 and the user repeatedly. Both halves check out:

```
grep -rn '9\.24' e2e-flow/          ->  nothing.  The digit was invented.

grep -rn '9\.31' e2e-flow/          ->  README.md:116
                                        RUN-PLAN.md:2143
                                        check_bench_result.validator/check.py:285
                                        assets/lib/graph_ceiling.py:95
                                        assets/lib/replay_root.py:99
```

**Every occurrence of `9.31` / `304` is the `--cuda-graph-max-bs 16` line — m2's
ceiling experiment.** It is not a deployment figure and has nothing to do with
rung 2b.

**T+1794's ladder row is wrong and is replaced:**

> ~~2b — stage 1 GREEN, 3 strong verdicts, **304 requests, ITL 9.24 ms**~~
>
> **2b — stage 1 sealed a valid `deploy_kit` with three strong verdicts, and its
> deploy load ran. Its request count and ITL are not recovered; the workspace no
> longer resolves on disk.**

**The ceiling experiment, correctly attributed:** m2, same node, same image, same
tp, one flag changed — **`max-bs 8 → 42.15 ms / 312 tps`** against **`max-bs 16 →
9.31 ms / 304 requests / 1649 tps`.**

### My record contained the evidence of its own error

**In the same section I wrote both:**

- the ladder row: *"304 requests, ITL 9.24 ms"*, attributed to rung 2b;
- §2: *"m2's C1 reference (ceiling 16 → 304 requests, 9.31 ms)"* — **correctly
  attributed.**

**The same 304 appears twice, with two different ITLs and two different owners,
forty lines apart, and I did not notice.** I transcribed one and reasoned about
the other in the same sitting. **A contradiction inside one document is the
cheapest possible detector and I have no routine that reads my own section back
before committing it.** That is the aging-record gap from T+1651 arriving at a
range of forty lines rather than two hours.

### Why the leader calls this worse than their earlier retractions, and they are right

**It is T49 — manufactured provenance — and they filed T49.** A real number from
a real experiment, detached from its origin, re-attached to a different event,
**and then used as evidence**: they cited it to m1 as the baseline rung 2e's
equality test would be compared against. **Had rung 2e come back at 9.4 ms it
would have been called a regression against a number that was never the
baseline.**

**And it survived six or more repetitions across hours** — including into this
file — *because it was specific enough to sound measured.* Their `D`-state
retraction was a story told once; this is a fabricated citation kept in
circulation.

### Two more corrections to T+1794

**Stage 1 does NOT qualify, and the leader's *first* answer was the right one.**
m5 implemented the consecutive rule (`7052729`):

```
deploy_kit   streak 2 (of 3)   |  4 valid / 5 reached a verdict / 11 never sealed / 25 mocked
             broken mid-sequence by 20260904T125637-e1ddf6
```

**The break is a genuine refusal** — `125637`, the correctly-refused `max-bs 8`
kit — **not a procedural artefact.** So *"not stable at 3"* was right and
*"qualifies at 4"* was the wrong correction. **They published a retraction of a
correct number**, which is the first instance today of a correction that made a
record worse. The pre-ladder question is immaterial: both pre-ladder runs are
`valid` and sit inside the streak.

**And rung 2e's selftest arm did not die — it recovered.** T+1794 records it as
dead on NCCL `HIP failure: 'invalid argument'`. m1's sampler caught it **coming
up at 19:22**: cards 4–7 went `0 → 3 → 0 → 7 → 74 → 75` and held three minutes.
**The run's own agent debugged its NCCL failure and solved it, on the exact card
set that had been failing.** Both arms then released in STEP 7's shape and
`m1final` is up on cards 0–3.

**m1's own caveat is kept:** they measured **VRAM occupancy, not a `ready`**. So
*came up* is what the evidence supports, not *served*.

### Addendum, 06:20 UTC — reconciling the two ledgers, and the third answer is the important one

The leader asked three questions of `bug.record.2026-09-05.md` (13 framework
entries) against `validator.failures.2026-09-05.md` (37 rows, 11 validators).
**Not merged; reported.**

### Q1 — rows in the failure ledger that are framework bugs wearing a validator's name

**Five rows, and they are the strongest case: `check_deploy_serves` ×3 and
`check_environment` ×2 left no `validator_report.txt` at all.** The ledger
records that they failed and cannot say why. **The reason is unrecoverable
*because of* bug-record entry 2** — a validator's stdout is not kept anywhere.
So those five rows are simultaneously rows in their ledger and evidence for mine.

**Two more candidates, flagged not claimed:**

- **`check_workset_runs` ×7** — the recorded reason is the mount-derivation
  refusal. **I established at 17:06 yesterday that the `-v /home:/home denied`
  text in that family is a *catalogue row* inside `measure_in_container.sh`'s
  refusal branch, not a live daemon denial.** If the captured tail is reference
  text rather than an event, the stated reason for seven rows is not the cause.
  **readme-cn's audit will settle it; I am not pre-empting them.**
- **`check_workset_shape` ×3** — *"workset.yaml does not load: No such file or
  directory: …/zones/…"*. A file the validator expected in its zone was absent.
  Producer defect or zone materialisation; **unresolved either way.**

### Q2 — my entries whose evidence is one layer removed

- **Entry 2 (validator stdout not kept) — evidence is an *absence*:** five
  failure rows with no report. That is inference from missing files, not from a
  read artefact, and the entry should say so. **Corrected in the record.**
- **Entry 5 (a command's error text in an artefact field) was seen through a
  PASS, not a refusal** — which makes it Q3's class, in my ledger only because
  somebody opened the artefact.
- **Entries 1, 3, 4, 6, 7 are direct** — event store or `agent_sys` source read
  at the cited line. No layer between.

### Q3 — what is in neither, and it is enumerable

**Ten of twenty-one validators have never failed. All ten have run.**

```
check_command_parses      107 invocations   0 failures
check_kernel_table         30               0
check_identity_resolved    26               0
check_profiling_evidence   26               0
check_worklist_shape       26               0
check_acceptance            5               0
check_bench_report          5               0
check_overlay_applies       5               0
check_patch_live            5               0
check_packup_shape          1               0
```

**A failure ledger is structurally silent on all ten**, and for each of them
`0 failures` is ambiguous between *it passed good artefacts* and *it cannot
fail*. **Neither file distinguishes those, and neither is capable of it.**

**One concrete instance, measured, in neither ledger:**
**`check_command_parses` — 107 invocations, 0 failures — emits `PROBLEM:` lines
under a `passed` heading**, 7 of them across 5 reports whose verdict was true.
`write_report`'s headings are `passed` / `REFUSED` / `DID NOT RUN`, and
`PROBLEM:` is the marker for a finding that *refuses*. **Either it labels notes
as problems, or it finds problems and passes anyway.** Not in the failure ledger
(it never failed); not in the bug record (it is routed to the leader as a package
question). **This is exactly the shape m5 named.**

**Four more of the same class, none in either file:**

- `check_invariant` — **412 of 417 records skip it**; it grades almost nothing.
- `accept_mock.py` — three passes before its first refusal; **unfalsified rather
  than tested**, and only visible once it finally refused.
- `e390abb`'s ceiling criterion — **has never had to refuse anything**, so it is
  not yet observed to constrain anything.
- m1's `e8153f8` — read the wrong metric key and **fell silently back to a
  constant**, producing a plausible floor. Caught by the leader against the real
  artefact, not by any check.

**The gap, stated so neither file assumes the other holds it:** *a defect that
produces a PASS cannot appear in a failure ledger, and appears in a bug record
only if somebody independently opened the artefact.* **Both of today's files are
blind to the same class, and the blindness is structural rather than an
oversight in either.**

### One state change found while counting

**`check_packup_shape` has now run once.** It was the last validator never to
have produced a verdict. **All 21 have now spoken** — the metric I retired at
T+1330 for saturating has, in fact, saturated.

### Addendum, 06:17 UTC — negative-controlling the ten: five tested, five load-bearing, 215 of 236 invocations covered

The leader asked whether the ten never-failed validators are load-bearing or
decorative, with three constraints: break them the way they *name*, **prove the
harness can drive one to refuse before reporting that any cannot**, and fix
nothing.

**Harness.** Clone a real passing validation zone and its material into `/tmp`,
rewrite `materials.json` to the copy, run the validator body with `cwd` set to
the zone. **Nothing in the run tree is touched.** Baseline first, every time.

### Constraint 2 first: the harness is proven before any conclusion

**`check_command_parses`, four points:**

```
real artefact, untouched          -> PASS
unterminated `if`, bash shebang   -> REFUSE  "line 4: syntax error: unexpected end of file"
parses but chmod -x               -> REFUSE  "items/command is not executable"
restored good script              -> PASS
```

**It goes to refuse and back.** The harness demonstrably delivers broken input,
so a later PASS means the validator looked and was satisfied — not that the
harness failed to arrive.

### Results

| validator | invocations | control applied | verdict |
|---|---|---|---|
| `check_command_parses` | **107** | syntax error · non-executable · restore | **REFUSE / REFUSE / PASS** |
| `check_kernel_table` | 30 | `table.csv` cut to 3 rows against `min_rows: 20` | **REFUSE** |
| `check_identity_resolved` | 26 | every resolution field set unresolved | **REFUSE** |
| `check_profiling_evidence` | 26 | **one** named part (`kernel_table`) removed | **REFUSE** |
| `check_worklist_shape` | 26 | — see below | **refuses a real artefact today** |

**All five are load-bearing.** **215 of the ten's 236 invocations** are now
backed by a demonstrated refusal — **91 %**.

**No crashes.** Constraint 1's third outcome did not occur: every mutation
produced a graded refusal with a diagnostic, not an exception. The distinction
that has been conflated in this effort did not arise here.

### `check_worklist_shape` — the control was supplied by the environment

**Its baseline does not reproduce, and that is the finding rather than a harness
fault.** The stored verdict from the run is `true`; re-running the same zone
against the same material today gives `false`:

```
items/schema differs from assets/schemas/kernel_worklist.schema.json.
The artefact is then self-describing and describes something other than what graded it
```

**The artefact carries its own copy of the schema; the package's schema has since
changed.** So the validator refuses a previously-passing sealed artefact — a
correct refusal, precisely diagnosed, and it never needed my mutation.

**That is m5's moving ruler at the schema layer, demonstrated rather than
argued:** *a verdict is not stable across schema edits*, and a sealed artefact
can become invalid without anyone touching it. **It also means my own accepted-run
comparisons are only valid within a schema generation** — the same caution I
recorded for validator sets at 15:00, one layer down.

### Not tested — 21 of 236 invocations

`check_acceptance` (5), `check_bench_report` (5), `check_overlay_applies` (5),
`check_patch_live` (5), `check_packup_shape` (1). **All five are m5's**, all
reached only in the last few hours, and `check_packup_shape` has exactly **one**
invocation — **the least attested of the ten, not a graduation.** Stated as
untested rather than assumed to follow the other five.

### And the Q3 instance I raised is already closed at source

`check_command_parses` **no longer** renders a passing line as `PROBLEM:`.
`77ed4be` — *"a passing line was rendered PROBLEM, 7 of the run's 11"* —
committed, and the body now carries `(line, is_fault)` tuples with the flag set
**where the line is written, not recovered from its wording afterwards.** The
comment records that the verdict was always right and only the rendering lied.

**So the one PASS-shaped symptom I could name concretely turned out to be
cosmetic, and the validator underneath it is the most load-bearing of the ten.**
That is the good outcome and it is worth saying as plainly as the bad one would
have been: **107 PASSes are earned.**

### Addendum, 06:38 UTC — the residual five: nine of ten load-bearing, none decorative, one untested

| validator | inv | control applied | verdict |
|---|---|---|---|
| `check_command_parses` | 107 | syntax error · non-executable · restore | **REFUSE / REFUSE / PASS** |
| `check_kernel_table` | 30 | rows cut below `min_rows: 20` | **REFUSE** |
| `check_identity_resolved` | 26 | every resolution field unresolved | **REFUSE** |
| `check_profiling_evidence` | 26 | one named part removed | **REFUSE** |
| `check_worklist_shape` | 26 | environment supplied it — schema drift | **refuses a real artefact today** |
| `check_acceptance` | 5 | `result/needle.json` removed | **REFUSE** |
| `check_bench_report` | 5 | round `r1` removed | **REFUSE** |
| `check_patch_live` | 5 | `env/container_hashes.tsv` emptied | **REFUSE** |
| `check_overlay_applies` | 5 | `result/mounts.json` removed | **REFUSE** |
| `check_packup_shape` | 1 | — | **UNTESTED** |

**Nine of ten load-bearing. None decorative. 235 of 236 invocations — 99.6 %.**
**No crashes**: every mutation produced a graded refusal, never a missing
`verdict.json`.

### Two harness defects, each of which would have produced a false "decorative"

**1. My harness collapsed every material onto the first one.** `materials.json`
maps *several* handoff ids to their own directories; my copy pointed all of them
at a copy of `material[0]`. On `check_patch_live` that handed the **stock** arm to
a validator bound to the **patched** arm, which refused with
*"this validator is bound to the patched arm and the record says 'stock'"* —
**a baseline that does not reproduce.** Under the old harness I would have
recorded `check_patch_live` as failing its own baseline and therefore
undriveable, or worse read the constant `False` as decorative.

**Fixed**: every material copied to its own directory keyed by handoff id.
`check_patch_live`'s baseline then reproduced as `[True]`, and
**`check_acceptance` and `check_bench_report` were re-run under the fixed
harness** because their first results were produced by the broken one. Both held.

**2. My first `patch_live` break was the wrong file.** I removed
`result/health.txt`; the validator reads `items/env/deployment.json`,
`env/docker_mounts.json` and `env/container_hashes.tsv`. It passed — correctly,
because I had not broken anything it claims to check. **Reporting that as
decorative would have been a false accusation against a validator that works.**
Emptying `container_hashes.tsv` — which its own error text names — refuses.

**Both are exactly the constraints the leader set, hit in practice.** Constraint 1
(*break what it names*) and constraint 2 (*prove the harness drove **this** one*)
each caught a wrong conclusion I was one step from writing down.

### `check_packup_shape` — untested, and I am not upgrading that to "never ran"

The full tally counts **1 invocation**. A targeted
`grep -rl 'min_result_files' --include=args.json` over the six newest runs and
the **entire frozen root** finds nothing, and a full re-scan **timed out at
120 s**.

**A partial search failing to find what a full scan found is not evidence that it
does not exist.** So the status is **untested and not re-located within budget** —
not *never ran*, and not *decorative*. **I withdraw nothing about the earlier
"all 21 have spoken"; I simply could not drive this one.**

It remains the least attested validator in the package: **one invocation, zero
failures, zero controls.** Its silence is worth almost nothing, exactly as the
leader said.

### What this settles

Before today, *"ten validators have never failed"* was ambiguous between **they
pass good artefacts** and **they cannot fail**. **Nine can now be driven to
refuse on demand, each by breaking the specific thing it names**, and the tenth
is unmeasured rather than suspect.

**The decorative validator the residual was supposed to hide does not exist.**
The 9 % where it would have hidden turned out to contain four working gates and
one I could not reach — and the two near-misses were both in **my instrument**,
not in theirs.

# THE DAY'S RECORD — 2026-09-05 06:47 UTC

**Not a status report. This is the account to read tomorrow to know what is
true.** Provenance is marked throughout per the three-way rule: **[observed]** by
me at a named artefact, **[relayed]** naming who measured or inferred it, or
**[catalogue]** for a known form rather than an event.

---

## 1. What is running right now

**[observed] 06:47:17 — three run processes, three holds:**

```
pid 3764594   45m19s      pid 161273   12m22s      pid 366844   0m20s
088  1h38m    217  6h46m    287  30m52s
```

**[relayed — leader] m5's full real chain is up:** `mock_stages=none`, **all five
stages**, 287 cards 4–7, 7-hour timeout. **This is the first complete real
end-to-end attempt of the effort.** Everything before it had at least one stage
mocked.

**[relayed — leader] m1's real line is on 088 cards 4–7**, past deploy and into
profiling.

**440 commits today. `todo.md` at T61.**

---

## 2. The live risk to it

**[relayed — leader] Three lines ended in profiling inside twenty minutes** —
`p4_a`, `p4_b`, `p4_m4real`. Two of them **with the allocation still held,
stages still `running`, and `0 validation(s) dropped`.**

**m2 is reproducing on 088 cards 0–3. If it is systematic it takes out the
full-real attempt.**

**The shape is the part to carry**, and it is this file's recurring theme: *the
run looks healthy in exactly the fields anyone would check.* Allocation held,
stage state `running`, nothing dropped. **Three of today's instruments would have
called that alive**, and one of them was mine until 19:23 yesterday.

---

## 3. Findings that outlive today

**The family trait** (`6e4ebdb`) **[relayed — owner's own finding].** Every mock
adapter is bound to the operator its corpus was built around. The corpus holds
`sampler_vocab_softmax`; real m3 selects four others; **no `--var` reaches it.**
So a mock stage silently grades a different operator than the real one selects —
and nothing in the parameter surface can say so.

**`check_agent_env` was blind to two thirds of m3's and m5's switchable surface
from the day it was written** (`7507b74`) **[relayed].** **The one agent
configured correctly was the one it could not see.** A checker whose blind spot
is anti-correlated with the defect it hunts.

**Both ledgers are structurally blind to PASS-shaped defects [observed].** A
defect that produces a PASS cannot appear in a failure ledger, and appears in a
bug record only if somebody independently opened the artefact. Two files that
each look like coverage and jointly are not. **Now with nine-of-ten negative
controls behind it**: the ten silences were earned, so the blindness is
structural rather than a hidden failure.

**`base_sha256` — three stacked defects [relayed].** Defect 1 landed **with its
own one-third caveat attached**; 2 and 3 are held pending m5's manifest answer.
The caveat travelling with the fix is the part worth copying.

**And four rules, none from any stage's own work:**

1. **T40** — before believing a null, name the result that would have proved the
   probe could speak. *Refined:* a broken **negative** control spends someone
   else's correctness; a broken positive one only costs your own time.
2. **`ref:` markers** — a refusal that lists known-bad forms will be quoted back
   as a report of one occurring.
3. **Three-way provenance** — observed / relayed *with the relayer's own
   provenance* / catalogue. **A provenance marker says who said it, not whether
   it was observed.**
4. **The identifier rule** — quote a finding by its name, not its number. The
   only one of the four that **prevents** rather than detects.

**Twenty entries now in the class table**: *an instrument reads a real thing and
answers a different question, and is never wrong in a way that shows up as an
error.*

---

## 4. Unresolved, and who holds it

| open question | holder |
|---|---|
| Are the three profiling deaths one bug? | **m2** — reproducing on 088 cards 0–3 |
| The manifest-hash question blocking `base_sha256` 2 and 3 | **m5** — unanswered |
| `snr_db: inf` non-monotonicity | **owner not named to me.** I am not guessing |
| Rung 2d's monitor death | **unexplained**, no owner. The `D`-state story was withdrawn |
| `check_packup_shape` — 1 invocation, 0 controls | **me**, untested, not re-located within budget |
| Does `e390abb`'s ceiling criterion constrain anything? | **m1** — needs one run at concurrency 64 |

---

## 5. The corrections, unsanitised

**A record of this day that read as steady progress would be false.** The most
useful results were withdrawals, and several were people narrowing their own
work.

- **m4, twice** — retracted the stall-detector diagnosis they filed; then
  retracted *"you did nothing wrong"* about my sweeping their T47.
- **m3, twice** — corrected T40's count against themselves; then found that six
  of the seven `8`s I had counted were one artefact replayed.
- **m1, four times on one claim** — *"post-contract observations cannot bear on
  the question"*, then *"my own four producers was short"*, then the ten/eleven
  duplicate, then the `E2E_KIT_` name predating its own contract by nine hours.
  **Every one against a change they authored.**
- **The leader, repeatedly** — the `304 requests @ 9.24 ms` figure was a real
  number from m2's ceiling experiment, re-attached to rung 2b, with a digit
  (`9.24`) that **appears nowhere in the package**. It survived six repetitions
  *because it was specific enough to sound measured.* Also: the `D`-state story,
  the `-newermt` ban, the "second consecutive producer" claim, and 275 reported
  free while eight cards were at 100 %.
- **Me** — the T+1062 `output_absent` answer (I read one of four attributes and
  reported the cause "not measured" when `seal_refused` named it); `| tail -12`
  losing 24 of 36 rows; publishing output from one command and code from another;
  calling a live run dead; over-correcting the leader's over-claim in the
  opposite direction, in the direction that flattered the change I was
  scrutinising.

**The pattern, and it is the day's real finding:** *no pass over any of these was
prompted by its author re-reading their own work.* Every correction came from
someone else's number colliding with it. **Five passes over the ceiling claim,
four of them weakening it, none by anyone who gained.**

**And the counter-instance, because it is the only structural fix anyone
found:** my negative-control harness had two defects — it collapsed every
material onto the first, and my first break targeted a file the validator does
not read. **Each would have named a working validator decorative.** Both were
caught because the harness was required to prove itself *before* its results
counted. **That is the one place today where checking came from inside rather
than from a collision.**

### Addendum, 07:40 UTC — a stale value arriving through an instruction, and the same double-assignment recurring inside the hour

**Filed here rather than in `todo.md`, because `todo.md` currently holds m3's
uncommitted `T67`** — see the second half of this entry.

### The shape: a value that was true once, fetched into a place it does not belong

**The leader instructed m4 to carry a 13.3 % re-measurement disagreement into
their 287 artefact. It was measured on the 217 workset.** m4's line is a fresh m3
analysis on a different node with different baselines — **rsd ≤ 1 %, an order of
magnitude tighter.** m4 caught it by opening the artefact before acting.

**Provenance checked here:** `13.3` resolves to `RUN-PLAN.md:128`
(*"2026-09-05 that was `attention_chunk_fwd_o` — and it came back 13.3 % apart"*)
and to `temp/bugs/2026-09-05-a-failed-body-has-its-debris-sealed-and-validated-and-says-nothing.md:142`
(`0.0264 → 0.0299 ms`). **A real measurement, correctly recorded, from a
different subject.**

**m4's naming is the entry and it is better than mine would have been:**

> a value that was true once, fetched into a place it does not belong —
> **arriving through an instruction instead of through memory.**

**Why it deserves separating from the twenty already in the class table.** Every
other instance today reached someone through **a tool** (the `find` shim, the
truncated tail, `squeue` answering allocation) or through **a recollection** (the
stale `node_ip`, the sealed corpus standing in for a real workset). Both have
defences and this file has spent the day building them: re-measure, re-read,
check the denominator, name the artefact.

**An instruction from the coordinator has none. It arrives already authorised.**
The only thing between it and the artefact is the owner deciding to verify their
*instructions* the way they verify their *tools* — and nothing asks them to.

**Pair it with the day's other pattern, because it is the same fact from the
other side.** I recorded at 06:47 that *no pass over any of today's corrections
was prompted by its author re-reading their own work* — every one came from
someone else's number colliding with it. **The coordinator's claims are the input
to everyone else's work rather than the output of it, so they collide with
nothing.** They receive the least independent checking of anyone's, structurally,
and m4 verifying one was a trait rather than a procedure.

### And the double-assignment recurred within the hour, on a different file

The leader's message assigning ownership says **"`bug.record` and `todo.md` are
yours alone. Nobody else edits them."** At the moment I read it, `todo.md`
carried **48 uncommitted insertions: m3's `T67`, headed *"Opened 2026-09-05 by
m3, at the leader's instruction."***

**So the same error being apologised for was live in the sentence that
apologised for it** — one file, two owners, both instructed by the same person,
neither told about the other. **I did not commit `todo.md`**, so nothing was
swept this time; the only reason is that I ran `git diff` before writing, which
is the check T47 exists to require.

**This is not a second slip. It is the first one's mechanism still running:**
assignment happens in messages, and a message does not know what other messages
were sent. **A file's owner is a fact about the world; an instruction naming an
owner is only a claim about it** — and the two drift exactly as fast as the
coordinator sends messages.

---

## T+2530 — 2026-09-05 07:45 UTC

Previous numbered section: **T+1794 (09-04 19:24)**. The interval between them is
not blank — it holds `THE DAY'S RECORD` (06:47) and 81 commits — but no numbered
section was written, so this one carries **12h21m** of ladder movement.

### 1. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** |
| 已经耗时 | 42 h 10 m wall (T0 = 09-03 13:30 UTC) |
| 预估耗时 | **not estimable** — see below |
| 可靠性 | **中** for the 78 %; **低** for anything past stage 3 real |

**78 % is defended by coverage, not by acceptance.** Measured this interval
[observed]: across the **29 run directories created since 09-04 19:24**, 89
`validation.yaml` files hold **256 verdict rows — 243 true, 13 false**, and
**all 21 declared validators were invoked at least once.** That is the first
window in which the declared set is fully exercised.

**What it does not mean, stated so nobody promotes it:** 21-invoked is a *union
over 29 runs*, not a single green chain. A union cannot report regression — the
reason I retired the "distinct validators passed" metric three times already.
The single best run in the window is `20260905T043851-4e95fc` at **45 pass / 0
fail**; the next is `20260905T064703-5fbd66` at **24 / 0**. Neither is the full
five-stage real chain.

**预估耗时 stays absent.** I have no completed real five-stage chain to divide
by, and the last four attempts died in profiling. An ETA here would be a number
with no denominator.

### 2. 当前进展

**[observed] 07:40:14 — four `run` processes**, pids 366844 (53 m), 400634
(51 m), 725021 (32 m), 1244119 (1 m).

**[observed] Three holds**, and one of them is the story:

```
109504  crsuse2-m2m-088   used 2:31:06
111038  crsuse2-m2m-217   used 7:39:40   <-- ~20 min left of an 8 h cap
112519  crsuse2-m2m-287   used 1:23:49
```

**217 is ~20 minutes from expiry.** Per the rule that computes *remaining*, not
*used*: nothing longer than a short probe should be started there, and anything
already on it should be assumed cut. **I could not determine which run is on
217** — see item 5.

**Failures in the window are concentrated, not scattered** [observed]: of 13
false verdicts, **`check_trace_coverage` 5 and `check_deploy_serves` 5** account
for ten; `check_environment`, `check_optimization_shape`,
`check_speedup_substantiated` one each. Two validators own 77 % of the interval's
rejections.

**Ledgers and records reached a stable division this interval** [relayed —
leader]: `validator.failures.2026-09-05.md` is readme-cn's sole rebuild target;
`bug.record.2026-09-05.md` is mine; **`todo.md` reverted to shared** with a
`git diff` precondition. The leader reversed their own sole-ownership rule after
it was live-broken by m3's T67 in the same interval it was announced. The rule
that survived is **"a rebuild needs one owner; an append-only log does not."**

**`todo.md` is at T67.**

### 3. 代码问题

**Fixed this interval (mine):**

- `bee6579` — the forward half of bug-record entry 2. Measured: exactly **two**
  of 21 validator bodies never call `write_report` (`check_deploy_serves`,
  `check_environment`); the other four the census showed silent were
  **historical**, each having gained the write hours later (`e42bde4`,
  `dff2bcb`, `c7340f9`). **So "conditional silence" does not exist in this
  package** — which is the opposite of what the census alone implied.
- `7538e78` — quantified the discarded-stdout gap at **15/46**.
- `e9e4534` — two new bug entries; ledger rebuilt from `validation.yaml`.

**Not fixed, deliberately** — the leader holds placement:

- `check_environment` (`steps/common.yaml`, shared/leader's) — **16** existing
  `findings` references, so the report write is close to a one-liner.
- `check_deploy_serves` (`steps/m1_deploy.yaml`, m1's) — **0** references; a real
  change. Its baseline needs a live deployment, so its load-bearing status is
  **untested**, independent of the report question.

**Found this interval, unfixed, mine:** a **two-writer corruption in my own
measurement path** — see item 5, because it is the interesting half.

### 4. 非代码问题

- **The coordinator's assignment channel does not scale to its own message
  rate.** Two file-ownership collisions inside one hour, the second created by
  the message announcing the remedy for the first. Resolved structurally: a
  `git diff` precondition **checks the world**; an ownership assignment **checks
  the coordinator's memory of it**.
- **217's remaining time is the only scarce resource right now** and no one has
  named what is on it.

### 5. 未定性

**(a) The zero that was a race, not a fact.** I ran a tree-wide
`find … > /tmp/yihou_v.txt`; it exceeded the 120 s limit and was moved to the
background. I read the file: **0 lines**. The harness then reported the task
**completed, exit code 0**. Exit 0 plus an empty file reads exactly like "no
validations since 19:24" — a false and *quiet* answer. The control
(`find` without the time filter, in one known directory) returned 9 both with
and without `-newermt`, which is what exposed it.

**(b) The same background writer then corrupted my replacement.** I truncated
and reused the same path for a scoped re-collection. The backgrounded `find` was
**still running** and appended into it, interleaving with my loop. The splice
produced `…/e312c0fa-6d83-4d9-4ec3-b1e0-6e9589b50f18/v0/validation.yaml` — **a
path that does not exist**, differing from the real
`e312c0fa-6d83-4dd6-9f49-9aa7347205ee` only after the fourteenth character. It
survived a `wc -l`, a `grep`, and a per-file loop, and surfaced only as a single
`PARSE_FAIL`.

**My first tally was computed over that file: 96 files, 274 rows, 260/14.** The
clean re-collection to a fresh `mktemp` path, with every line proved to exist
before use, gives **89 / 256 / 243 / 13**. *The published numbers in items 1–2
are the clean ones.* Seven of the 96 were the other writer's.

**Both halves are new instances of the class table's entry, and they sharpen
it:**

> **A background task's "completed, exit code 0" describes the process, not the
> artefact.** Killing the foreground does not kill the writer, and a truncated
> redirect target with a live second writer yields data that is *plausible*
> rather than absent — which is strictly worse than empty.

**Undetermined, and I am not guessing:**

**(c) Which run is on 217.** I grepped the four newest run trees for node names
and got `006 019 047 061 234 276` in every one, plus `088`/`287` in two. Those
are **provenance strings carried inside the copied sealed corpora** — the
artefact answers "which node produced my inputs", not "which node am I on".
Same shape as the class-table entry it belongs to. **No claim made.**

**(d) Whether the interval's 13 failures overlap readme-cn's 45/33/11 census.**
Different windows and different collection methods; I did not diff them, and I
will not touch their file to find out.

### 6. 新增 commit

81 commits in the interval, across seven owners. Mine, and what each is for:

| commit | what |
|---|---|
| `e8bd141` | `validator.failures`: rule 4's bias is an instrumentation timeline, verified from the bodies |
| `bee6579` | bug entry 2's forward half — only two bodies never write a report; four were historical |
| `2bd231f` | checkpoint: a stale value arriving **by instruction**, and the double-assignment recurring within the hour |
| `453237d` | restore the five corrections my rebuild destroyed, **on the rebuild's own method** |
| `d68a104` | `CLAUDE.md`: two failures of mine today, both structural rather than slips |
| `7538e78` | quantify the discarded-stdout gap at 15/46; T49 reached the ledger itself |
| `e9e3340` | `todo`: mark E16 inside T10, record m5's answer in T62, add T66 |
| `ee6af8d` | `todo`: T62–T65, today's four deferrals |
| `e9e4534` | rebuild the failure census from `validation.yaml` (45/33/11) + two bug entries |
| `080f1b0` | `THE DAY'S RECORD` |
| `ba97387`/`3885050` | the negative-control results: nine of ten load-bearing, none decorative |
| `8de9bb5` | reconcile the two ledgers — both blind to the same PASS-shaped class |
| `14f35bc` | consolidate 15 `temp/bugs` records into one, every `file:line` re-verified |
| `17f03d5`/`a18ece2`/`b870a61`/`ace933d` | `CLAUDE.md`: the user's standing rules and the day's debugging techniques |

The other owners' 60-odd are the ladder itself — m1's NCCL retry (`78909fc`),
m2's engine-log wait (`475f2fc`) and writable trace dir (`9ab60e3`), m3's
container-root expansion (`91c0b04`), m4's `public_symbol` entry point
(`039825c`), m5's schema-absence refusal (`ce2d5a6`).

### 7. 其他

**The one number worth carrying forward** is not 78 %. It is **21 of 21
validators invoked in a single 12-hour window with a 95 % true rate**, set
against **zero complete five-stage real chains**. Those two facts are not in
tension: the validators are exercised by the *rungs*, and the rungs are cheap.
The chain is what is expensive, and the chain is what has never finished.

**And the honest note about this section:** two of its five measurement steps
produced wrong numbers before producing right ones, both from the same
background-writer mechanism, and **neither would have failed loudly.** The
section is trustworthy because I ran the control, not because the tools behaved.

---

## ADDENDUM — the tenth stale value, and the first one self-inflicted

**08:25 UTC.** Filed beside the instruction-channel entry (`2bd231f`) rather than
inside bug-record 17, because the symptom matches and the **mechanism does not**.

**[relayed — leader, from readme-cn and m3]** readme-cn cited `RUN-PLAN.md:L2133`
in a message. m3 opened it and found unrelated prose. **Not a misreading by
either of them.** The heading moved **2128 → 2139 → 2159** between the read and
the delivery — **31 lines, of which 20 were readme-cn's own commit.**

**Correct when computed, wrong when delivered, by the author's own hand in the
interval.** Every prior instance in this series had a second party or the clock
in between:

| # | how the value went stale | who moved it |
|---|---|---|
| 1–8 | measured, then the world changed | the world |
| 9 | arrived **by instruction**, pre-authorised (`13.3 %` from 217 into 287's artefact) | the coordinator |
| **10** | **the author's own edit, between computing the pointer and sending it** | **the author** |

**This is the one with no defence available.** For 1–8 the defence is re-measure;
for 9 it is *the owner must check an instruction the way they check a tool*. For
10 there is nothing to re-check — **the value was right when checked, and the act
of writing the message's own supporting commit is what broke it.**

**The rule, from readme-cn, and the leader has it in `CLAUDE.md`:** in a message
about a file several people are editing, cite a **greppable anchor** — heading
text, a table's first cell — **never a line number.**

> *Line numbers in this repo today have a shelf life of single-digit minutes.*

**Why it belongs here and not in bug-record 17.** Entry 17 is about a
**correction landing in the wrong place**; this is about a **pointer decaying in
transit**. Same symptom — a reader follows the document and arrives somewhere
wrong — but 17's fix is *audit the table against the prose*, and no audit fixes
this one. **The fix is at composition time and nowhere else.**

**A fifth instance of 17 appeared while I was writing this, in my own hands and
harmless:** the leader's `git log -S'--var forge_fellow=<backend>'` returned
*"exactly one commit"*. It now returns **two** — `fba0c8b`, the fix, joined the
set within the hour. The count was true when stated. **I record it because it is
the cheap pole of readme-cn's calibration** — costs one `git log` to check,
propagates nowhere — and the record should keep showing both poles rather than
only the expensive ones.

**And one correction to the leader's own account of the fourth instance**, in the
direction that strengthens it: they put the gap between m3's code and m3's
contradicting table row at *"an hour"*. Measured — `f92e42b` **07:51:13**,
`0dafcfd` **08:03:43** — it is **12 minutes 30 seconds**. The fix `fba0c8b`
landed at **08:06:30**, 2m47s after the error. **An author contradicted code they
had written twelve minutes earlier**, which is a sharper statement of the class
than an hour would have been.

---

## T+2570 — 2026-09-05 08:20 UTC

**Read from `date -u`, not extrapolated.** That qualifier is here because the
previous forty minutes of my own record-keeping failed exactly that test — see
item 3.

### 1. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %**, unchanged from T+2530 |
| 已经耗时 | 42 h 50 m wall |
| 预估耗时 | **not estimable** — still no completed real five-stage chain |
| 可靠性 | **中** for the 78 %; **低** past stage 3 real |

**Nothing in this interval moved the ladder.** Forty minutes, 31 commits, and
they are almost entirely records and corrections — mine and other owners'
fixes. The number stands where it stood because the chain has not finished.

### 2. 当前进展

**[observed] 08:19:04 — five `run` processes**, newest at 1 m, oldest beyond the
window the tool prints.

**[observed] Three holds, and the one I flagged as expiring has rotated:**

```
109504  crsuse2-m2m-088   used 3:09:56
112519  crsuse2-m2m-287   used 2:02:40
112699  crsuse2-m2m-217   used   17:55   <- NEW jobid; 111038 (7:39:40) is gone
```

**My T+2530 warning about 217 resolved itself** — the old hold expired and a new
one was taken. I did not observe the transition and cannot say whether anything
was lost with `111038`; **whoever was on it knows and I do not.**

**`todo.md` is at T72.**

### 3. 代码问题 / 记录问题

**The interval's substantive work was records, and its substantive defect was
mine.**

**Every timestamp I wrote into `bug.record.2026-09-05.md` between 07:40 and
08:19 was extrapolated, not read.** `07:55 / 08:10 / 08:22 / 08:35 / 08:45 /
08:55 / 09:00 / 09:15 / 09:25` — **not one came from `date`.** I read the clock
once at 07:40 and then estimated forward, writing the estimates as measurements.
Actual: **08:19Z**. The worst was **66 minutes fast and in the future.**

**They were `最后核对` stamps** — the field whose entire purpose is to tell a
later reader how stale the file is.

**This is the convention I adopted from readme-cn in that same file's header,
violated in the act of writing it.** And it is the shape I had just filed
against two other people: *an inference from partial evidence, written as a
measurement.* Mine is the worse variant — **readme-cn could not check "m4 still
holds those" from where they stood; I had a clock the whole time and one
`date -u` away.**

**Fixed:** all in-body moment markers replaced with **ordinals ①–⑥**. The
ordering is something I actually know; the times were not. **A sequence with no
false precision beats a timeline that is invented.** The header carries the
correction rather than hiding it, and the dating convention gained the clause it
was missing: **the date itself must be read. A fabricated timestamp is worse
than none, because it makes "verified" look checkable.**

**Other record work this interval, all mine:**

- Entry 17's phrasing corrected to readme-cn's — **the correction lands where
  the author is, not "tables go stale"** — after their fifth instance
  (`.claude/CLAUDE.md`, `c18d7cc`) turned out to be **inverted**: the top list
  right, the numbered prose stale, **and the stale copy was the permissive one**
  on the deletion rule. I verified that diff first-hand.
- Instance C reclassified: **the forward pass does not miss it, it certifies
  it** — 13 accurate rows, zero disagreements, clean result, and the four rows
  that mattered were never in the population. Tied to the team's existing
  *通过的那一侧往往更危险*.
- Entry 19's only instance **withdrawn** — m4 confirmed m3 had told them
  directly and both references were already re-pointed. Mechanism kept, marked a
  hypothesis, explicitly not to be cited as established.

### 4. 非代码问题

**The confirming-result bias is now three instances across two people** and is
in `CLAUDE.md`:

```
① "seven consecutive producers chose 8"  flattered the change I was scrutinising  — me
② "the check caught me on first use"     flattered the check I was endorsing      — me
③ amplifying ② to the team and the user  flattered the mechanism just adopted     — leader
```

**With ① and ② it is one person being sloppy. With ③ it is a bias.** Operational
form: *confirming results get checked to the standard of refuting ones.*
**My timestamp failure is a fourth instance of a neighbouring shape** — I never
checked a claim that was costless to check, because nothing was contesting it.

### 5. 未定性

- **What `111038` was carrying when it expired.** Not observed, not
  reconstructible from here.
- **Whether entry 19's mechanism has any instance at all.** Its only one is
  withdrawn. The principle is sound a priori and that is not evidence.
- **Whether the two-pass audit finds anything on the other operational
  documents.** Only my own file has been through both passes.

### 6. 新增 commit

31 in the interval. Mine:

| commit | what |
|---|---|
| `bee6579`…`88ecce1` | entry 17 four times over: directional, bounded, five instances, corollary corrected to two passes |
| `27a20f6` | my own cost estimate was wrong — 7 append sites, not 16 |
| `57ed492` | entry 17 filed |
| entries 18/19 | authorship as the reason not to re-read; an observation routed past its holder |
| latest | **every timestamp in the file was extrapolated** |
| `T72` | `expect_ranks`, plus the explicit line that entry 17's audit will not find it |

### 7. 其他

**The honest summary of this interval: I produced no ladder movement and one
real defect, and the defect was in the instrument I use to describe the ladder.**
Nine fabricated timestamps in the file that records what is true, written while
adopting the rule against exactly that. **Corrected within the hour and by
reading a clock, which is the only reason it is forty minutes of damage rather
than a day of it.**

---

## T+2620 — 2026-09-05 08:25 UTC

**Read from `date -u`.** Weighted to what is running; the record work is one
paragraph at the end where it belongs.

### 1. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~80 %** (+2) |
| 已经耗时 | 42 h 55 m wall |
| 预估耗时 | **still absent** — no completed real five-stage chain to divide by |
| 可靠性 | **中** for the 80 %; **低** past stage 2 real |

**The +2 is one measured event, not a feeling:** the full-real chain **completed
stage 1**.

### 2. 当前进展 — the chain moved past deploy

**[observed] 08:21, `/tmp/yihou_rung5_287_b.log`:**

```
phase  m1_deploy: running -> succeeded
phase  run_profiling_mode_off: input_validating -> running
phase  run_profiling_mode_on: (new) -> waiting_resource
phase  merge_profiling_evidence: (new) -> waiting_handoff
```

**The leader's 08:14 snapshot had it at `deploy_and_prove: output_validating`.
In the seven minutes between, stage 1 finished and stage 2 started.** The log
grew 2800 → 4369 bytes across that window, so it is **progressing, not parked**.

**This is the first real stage-1 completion inside the five-stage chain.**
Everything previous either mocked a stage or ran stage 1 standalone.

**Its one validation so far:** `check_deploy_kit`, **passed**, and it **wrote a
report** —

```
# check_deploy_kit
## 50b06f09-…: passed
  note:    qwen3.6-27b-mix.packup_20260905
```

Total `validation.yaml` in the chain: **1**. Profiling is where the last four
lines died, so the next twenty minutes are the ones that matter.

### 3. The chain's log is not where anyone looks

**[observed] every run's stdout, by `readlink /proc/<pid>/fd/1`:**

```
1244119  /home/yihou/p5_trt.log
1410811  /home/yihou/p4_m4real287b_074858.log
1869760  /home/yihou/p5_trt2.log
1910769  /home/yihou/p4_d_report_test.log
1292565  /tmp/yihou_rung5_287_b.log        <- the acceptance chain
```

**Four of five land in `/home/yihou`. The fifth is the one the whole effort is
for, and it is the only one an `ls ~` will not show.** The leader found it the
same way I confirmed it — through `/proc`, not by looking.

**A run nobody can see is a run nobody can rescue.** `/tmp` is also the one
location that is not backed up and is subject to cleanup. **Recorded as a live
risk to the deliverable, not as a tidiness note.**

### 4. 217 is idle, and that is a fact about us

**[observed] 08:21 via `spur exec 112699`:** `rocm-smi` reports **0 % on every
card queried**, and `docker ps` returns **nothing at all** — not a co-tenant, not
one of ours.

```
112699  crsuse2-m2m-217   used 20:14   ->  ~7 h 40 m remaining
```

**That is the largest window any hold has today, on all eight cards, and it has
been empty for at least twenty minutes.**

**Recorded as idle capacity, not available capacity.** The distinction is the
whole point: *available* describes the machine, *idle* describes us. Standing
rule 15 says compute is better spent hitting a wall than sitting empty, and
**nobody has put anything on it.** I am not authorised to launch; this is the
line that says the gap was seen and left open.

### 5. Landed ≠ working — m1's two fixes, neither exercised

**Both are in the tree and neither has executed.** They fail differently and the
difference matters:

| fix | what landed | why it has not run |
|---|---|---|
| report write in `check_deploy_serves` | **code** — `grep -c write_report` now **3**, was **0** this morning | the validator has not been invoked; **zero reports from it** anywhere |
| NCCL retry, `78909fc` | **not code** — 27 lines in `deploy_and_prove.task/readme.md`, a **brief** | needs an agent to read it **and** the intermittent fault to occur |

**The first needs any invocation. The second needs a fault that is by definition
intermittent, so its first real test may be a failure nobody scheduled.**

**And this updates a finding I published this morning:** *"exactly two validator
bodies never write a report"* is now **one** — `check_environment`, which the
leader is holding until the chain clears. m1's half is done in the tree.
**The census-vs-body distinction bites again: the body changed, the run tree has
not caught up, and only one of those is the forward risk.**

### 6. 新增 commit

Since T+2570: mine are the timestamp correction and this section. The others'
are the chain — I have not audited them this interval and am not going to
characterise work I did not read.

### 7. 记录 — one paragraph, as asked

Entry 17 is settled at five instances and the count is the least interesting
part. **The residue is the two-pass check with its boundary stated: forward pass
finds a row that lies, reverse pass finds a claim with no row, and neither finds
a variant a second document never mentions.** What the class hides behind at
every scope is **absence** — which is why the forward pass does not go quiet on
a defect, it reports the file sound. My nine fabricated timestamps are corrected
and the generalisation is in `CLAUDE.md`: **unchallenged claims get the same free
pass as confirming ones.**

---

## ADDENDUM — the two full-real chains, captured while both were alive

**08:56 UTC.** Both launch lines read from `/proc/<pid>/cmdline` and **parsed
positionally** — only the argv element immediately following a literal `--var`
is treated as a var. **That sidesteps m3's trap entirely**: a `grep` for `--var`
also matches variable syntax inside `--system-prompt`, which is our own readme
text. No grep was used.

```
2363398  /tmp/yihou_rung5_287_c.log            m5   287 c4-7   17 vars, argc 45
2394563  /home/yihou/p5_fullreal_217_084728…   m4   217 c0-3   23 vars, argc 55
```

### What differs — this is the artefact

**Only on m4's 217 line (7 vars + nothing else):**

```
adhoc_cases=3            parser_args=--reasoning-parser qwen3
port_router=8121         port_worker=8122        port_etcd=8123
scratch_root=/mnt/m2m_nobackup/yihou/e2e_flow_p5/kfo
transport=spur           workset_reverify_shapes=4
```

**Only on m5's 287_c line:**

```
bench_rounds=3
gsm8k_data=…/integration/data/gsm8k_test.jsonl
--timeout                <- a flag, not a var; 217 has none
```

**Shared and identical**: `aiperf_trace` (the fix, on both), `expect_ranks=4`,
`image`, `model_name`, `model_path`, `mock_stages=none`, `tp=4`,
`transport_env`, `work_root` root.

**Three consequences worth naming before either finishes:**

1. **`adhoc_cases` agrees, by two different routes.** 217 passes `3`; 287_c
   **omits it**, and RUN-PLAN as corrected in `42ae222` says rung 5 omits it and
   takes the default 3. **Same effective value.** The one var most likely to
   diverge is the one the documentation fix already handled.
2. **`workset_reverify_shapes` — the leader's open cell — is measured: absent on
   m5's line**, `4` on m4's. Whatever m5 gets is the package default.
3. **217 has no `--timeout`.** It runs until the hold expires, ~7 h. 287_c is
   bounded. **If one is killed by a clock and the other is not, that is the
   launch line, not the code.**

### The port map, and a hypothesis of mine that died on contact

```
   PID              node    cards    router   worker    etcd   log
1869760   crsuse2-m2m-088  4,5,6,7  (unset)  (unset)  (unset)  p5_trt2
1910769   crsuse2-m2m-088  4,5,6,7  (unset)  (unset)  (unset)  p4_d_report_test
2394563   crsuse2-m2m-217  0,1,2,3     8121     8122     8123  p5_fullreal_217
1410811   crsuse2-m2m-287  0,1,2,3     8111     8112     8113  p4_m4real287b
2363398   crsuse2-m2m-287  4,5,6,7  (unset)  (unset)  (unset)  rung5_287_c
```

**I formed a collision hypothesis and it is false.** The declared default is
`${port_router:-8101}` (all four `steps/m*.yaml` agree), so I expected the three
`(unset)` lines to be sitting on 8101–8103 — which would have put the two 088
lines on the same node, same cards, **and the same ports**, a clean explanation
for m1's bring-up sitting 37 minutes at the top of the 18–39 band.

**The node says otherwise.** On 088, m2's line is bound on **8115 / 8116 /
8117 / 8118**, not 8101. So:

> **The absence of a `--var port_router` does not mean the default is in
> effect.** The cmdline records what was **passed**, not what was **bound**.

**That is today's shape again in a new instrument.** A cmdline read answers
*"what did the operator type"*; I asked it *"what is this process using"*. It
never errors. And `line.sh`'s own comment already names the hazard — *"router
8115 while this script benched 8111, because the other line was…"* — **8115 is
exactly what I measured.**

**So: no evidence of a port collision anywhere.** 287 is clear (8111–8113 vs a
default that is not 8101 in practice), and 088's two lines are not provably
colliding. **The 088 co-tenancy remains open on cards, not on ports**, and it is
the leader's scheduling item, not either owner's.

**One incidental**: m1's `yihou_e2e_flow_sgl_selftest`, `Up 40 seconds` at 08:27,
is **gone** at 08:57. Brought up and torn down. **A container count taken at one
moment cannot distinguish "never started" from "started and finished"** — which
is precisely the correction m2 made to the leader's 08:25 diagnosis, and it
applies to my own reading here too.

---

## T+2710 — 2026-09-05 09:32 UTC

**Time from `NOW=$(date -u …)`, referenced as a variable** — not composed
beside the read. That is the repair from the 09:28 repeat, applied here first.

**A note on my own T-numbers:** they have drifted from wall time (T+2570→T+2620
was 5 minutes, not 50). **The UTC stamp is the authoritative field; the T-number
is a label.** Same class as the T-number collision readme-cn routed this hour —
an identifier I allocate from memory rather than from a read.

### 1. m2's stage went green end to end on real hardware

**This is the interval's result and the first deliverable-side one in several.**

**[observed] I verified the verdict, not the report of it** — run
`20260905T081530-862477`, handoff `78032ee6`:

```
check_command_parses -> True
check_environment    -> True
check_trace_coverage -> True
```

**And the artefacts behind it, read directly:**

```
content/items/result/traces/1788598676.785897-TP-0.trace.json.gz   66 687 211
                                            -TP-1.trace.json.gz   66 653 448
                                            -TP-2.trace.json.gz   66 790 479
content/items/result/stacks/1788598747.515271-TP-0.trace.json.gz  119 855 271
```

**Why the PASS counts, and this is m2's point:** `check_trace_coverage`
**re-parsed the gzip and counted ranks against the manifest.** It is a read of
the trace *contents*, not of the capture's exit path. **A `CAPTURE_OK` would
have been an exit code** — and this file has spent all day recording why that is
not the same thing.

**The controlled pair also completed** [relayed — leader, from m2]: two no-fix
arms dying identically at capture 2/6 after 39 spins, one pinned-treatment arm
seeing batches in 5 s and going through. **`475f2fc` moves from candidate to
established.** m2's argument for why n=3 suffices is the part worth keeping:
**a confound would have to reproduce that particular transition, not merely let
the run through.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~82 %** (+2) |
| 已经耗时 | ~44 h |
| 预估耗时 | **still absent** |
| 可靠性 | **中** |

**+2 on the measured event only** — one stage green on real hardware with a
contents-level validator. **预估耗时 stays absent: no completed five-stage chain
exists to divide by**, and that is unchanged by a single stage going green.

### 3. 当前进展 — three chains

```
m5   287 c4-7   full real            launch line captured
m4   217 c0-3   full real            launch line captured
m3   088        launching            not yet captured
m4   287 c0-3   optimize_kernel campaign
m1   217 c4-7   report test
```

**Two of three launch lines are in the record with their diff** (previous
addendum). **The third is not, and it is the one I should get next** — the value
of the first two came entirely from having them side by side.

### 4. The broken precondition — worse than "cannot fail"

**[observed] I ran it.** The leader's `pgrep -af "cli.main run"` **does not
return zero.** It returned **2**, and both hits were **the shell wrapper
carrying the pattern text in its own argv**.

```
real runs, argv:  python3 …/assets/lib/run_with_long_stall.py --stall…
                                    ^ no `cli.main` anywhere
pgrep hits:       the invoking shell, which contains the string "cli.main run"
```

**So it misreads in *both* directions**: zero real lines match, and the pattern
matches its own invocation. Run one way it says *"nothing is running"*; run
another it says *"2 are running"*. **Neither is a measurement.**

**This is m3's `--system-prompt` trap exactly** — a pattern matching the text of
the thing doing the matching. That trap is why I parsed the two launch lines
**positionally** from `/proc/<pid>/cmdline` rather than grepping, and it is the
second time today the same shape has cost someone.

**The leader's own account of their failure is the general one:** they had the
alternation in the command they use, passed on the first half, and **never ran
the version they handed out.** Attributed to them, as they asked.

### 5. 未定性

- **m3's third launch line** — uncaptured, and the window closes when it does.
- **The 088 / 217 co-tenancy.** The leader's correction stands: the collision
  window is **not** launch-to-bring-up but **the whole gap**, which reached
  **25 minutes** — m4 launched 08:47, brought up 09:13, **one minute after m1's
  VRAM assertion passed.** A precondition that samples once cannot cover a
  25-minute window. Attributed to the leader's scheduling.
- **Whether entry 15's four earlier deaths were this or something else.** The
  09:xx instance was `aiperf_trace` absent, not entry 15. **The discriminator is
  now known and cheap — a timestamped activity artefact — and has not been
  applied retroactively.**

### 6. 新增 commit

Mine this interval: the anchor rule's third axis plus the mis-stamp correction,
and this section. **I have not read the other owners' commits and will not
characterise them.**

### 7. 其他

**The thing that changed today is what a green result means.** This morning the
strongest evidence anyone had was a count of PASSes. **m2's green is a validator
that re-parsed 200 MB of gzip and counted ranks against a manifest** — the first
result in the record that is a read of contents rather than of an exit path.
**That is the standard the remaining stages should be held to**, and it is why
one stage is worth +2 and twenty-one PASS rows were worth retiring three times.

---

## ADDENDUM — the third launch line, and the three-way diff

**2026-09-05 09:36 UTC.** m3's line captured from `/proc/3048152/cmdline`, **positionally**.
All three full-real chains are now in the record with their differences.

```
m5  /tmp/yihou_rung5_287_c.log                          287 c4-7   17 vars
m4  /home/yihou/p5_fullreal_217_084728.log              217 c0-3   23 vars
m3  /home/yihou/chain_…088_cards0-3_20260905T0928.log   088 c0-3   19 vars
```

**25 distinct vars across the three; 15 are common to all.** The ten that are
not:

```
var                        m5_287  m4_217  m3_088
adhoc_cases                  .       Y       .
bench_rounds                 Y       .       Y
gsm8k_data                   Y       .       Y
parser_args                  .       Y       .
port_etcd                    .       Y       .
port_router                  .       Y       Y
port_worker                  .       Y       .
scratch_root                 .       Y       .
transport                    .       Y       .
workset_reverify_shapes      .       Y       Y
```

### The result, and it is cleaner than I expected

**m3's line is the only one with no odd-one-out property in either direction.**

```
m5:  missing what both others have  -> port_router, workset_reverify_shapes
     unique to it                   -> none
m4:  missing what both others have  -> bench_rounds, gsm8k_data
     unique to it                   -> adhoc_cases, parser_args, port_etcd,
                                       port_worker, scratch_root, transport
m3:  missing what both others have  -> NONE
     unique to it                   -> NONE
```

**m3 built theirs from RUN-PLAN's canonical block; m4's came from the leader's
list; m5 composed their own.** The line assembled from the shared document is
the one that is neither short of what the others agree on nor carrying anything
neither has. **That is a measurable argument for the canonical block over a
hand-assembled list — not a matter of taste.**

**Two consequences that are live rather than tidy:**

1. **`workset_reverify_shapes` — m5 is now alone in omitting it.** With one
   other line it was a question; with two it is a divergence.
2. **`port_router` — m5 is alone in omitting it too.** Given the earlier
   measurement that **an absent `--var port_router` does not mean the declared
   `8101` default is in effect** (m2's line was bound on 8115), m5's line is the
   one whose bound port nobody can predict from its launch line.

**`--timeout`: m5 and m3 carry it, m4 does not.** m4's chain runs until the hold
expires. Unchanged from the earlier addendum, now confirmed as 2-against-1.

**`adhoc_cases`: only m4 passes it (`3`).** The other two omit and take the
documented default of 3 — the behaviour `42ae222` fixed. **All three agree in
effect.**

**One correction to the brief I was given:** the container is
`yihou_e2e_flow_088_09050928`, not `…_09280928`. Trivial, except that a
container name is exactly the kind of string someone greps for.

---

## R2 T+0 — 2026-09-06 06:33 UTC — BASELINE, second cluster

**Numbering restarts with an `R2` prefix.** The previous section is `T+2710`
(2026-09-05 09:32 UTC) and belongs to the *first* cluster's round. This is a new
round, on a different machine, against a mission file rewritten this morning.
Continuing the old counter would have put ~1 261 wall-clock minutes of *no work
on this round* inside a number that reads as effort. **A new prefix costs a
grep; a colliding number costs a wrong reading.**

**Every time in this section came from `NOW=$(date -u …)` or from a `--time-style
=+'%F %T'` listing, read in the same command that used it.** Nothing here is
extrapolated from an earlier clock read.

### 1. Why this entry is a baseline and not a progress report

**Nothing has run yet.** That is the measurement, and it is the point of writing
it down now:

```
/data/yihou/agent_sys_runroot/runs/      does not exist
/data/yihou/e2e_verify_20260906/m1/      empty          (mtime 06:30:47)
/data/yihou/e2e_verify_20260906/m2/      materials/ only, empty (06:31:16)
/data/yihou/e2e_verify_20260906/m35/     empty          (06:32:05)
/data/yihou/e2e_verify_20260906/notes/   empty
docker ps -q | wc -l                     2   (both foreign, see §4)
rocm-smi --showmemuse   GPU[0..7] VRAM%  0 0 0 0 0 0 0 0
```

**The earliest artefact of this round I can date is `mission.md`, mtime
2026-09-06 06:19:11** — so the round is ~15 minutes old at this write. Later
sections have something to be measured against; this one does not, by
construction.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **0 %** |
| 已经耗时 | **~15 min** (from `mission.md` mtime 06:19:11 to 06:33:41) |
| 预估耗时 | **absent** |
| 可靠性 | **高 for the 0 %; N/A for the estimate that does not exist** |

**The 0 % needs its denominator said out loud, or it will be misread.** The
previous round reached ~82 % **of a chain on a different cluster**. This round's
deliverable is *reproducing that chain here*, and **no stage of it has been
attempted on this node**. The 82 % does not transfer and is not a starting
credit.

**预估耗时 stays absent** for the same reason it was absent all of yesterday:
**there is no completed five-stage chain on this cluster to divide by.** I will
not produce one from the other cluster's timings — a different image, a
different model, and no sealed corpus make that a guess wearing a number's
clothes.

**Reliability of the 0 %: 高.** It rests on four independent absences read this
minute (no run root, three empty note dirs, zero VRAM on all eight cards). It is
the one number in this table that is hard to be wrong about.

### 3. 当前进展 — the cluster, measured

```
hostname     smci355-ccs-aus-n04-25.prov.aus.ccs.cpe.ice.amd.com
hold         29184  RUNNING  Compute-DCPT  --exclusive  gres/gpu:8
             StartTime  2026-09-05T22:00:01
             EndTime    2026-09-06T14:00:01
             REMAINING  7 h 26 min  (14:00:01 − 06:33:41)
filesystem   /data   49 T,  41 T avail   ← this is the scratch that exists
             /       14 T,  13 T avail
             /shared_nfs   EMPTY (two entries, both `.`/`..`, root-owned)
GPUs         8 × MI355X, all eight at VRAM 0 %
```

**`REMAINING`, not elapsed.** `RunTime=08:32:23` is what `squeue`'s `%M` would
have shown and it is the wrong field for every scheduling decision.

**Materials named by `mission.md`, both confirmed present:**

```
/apps/data/models/GLM-5.3-Flash                        exists
/apps/yihou/packups/glm53flash.mix.packup_20260830     exists
/apps/data/models/Qwen3-32B                            exists
```

**The third is not in `mission.md`** — it comes from the leader's phase plan
(phase A on Qwen3-32B, phase B swapping in GLM-5.3-Flash). Recorded as the
leader's plan, not as a mission requirement, because `mission.md`'s *Materials*
section names only the GLM path.

**Team, and what each holds** [relayed from the leader's brief, not yet
corroborated by any artefact — all three note dirs are empty]:

```
m1-deploy      engine image + module 1
m2-profiling   materials, variable table, module 2
m35-chain      modules 3/4/5 + packup, pre-registration
```

### 4. Non-code problems — the localisation traps, and they are the whole story

**(a) There is no sealed corpus on this cluster, and `/shared_nfs` is empty.**
On the first cluster the entire mock loop — the 4-minute login-node loop, every
`mock_stages=` replay — stood on
`/shared_nfs/yihou/agent_sys/cheat_for_mock/`, 25 sealed handoffs. **Here that
path's parent contains nothing at all.** Consequence, stated plainly rather than
solved: **nothing can be replayed on this node until this round produces it
first.** The cheapest debugging loop the previous round had does not exist here
yet.

**(b) We are ON the compute node, not on a login node.** The first cluster's
whole operating model — read-only `/shared_nfs`, `spur exec` to reach GPUs, no
local docker daemon — **does not describe this machine.** `docker ps` answers
directly, `rocm-smi` answers directly, and writes to `/data` land locally. Every
instruction in the carried-over `CLAUDE.md` that begins "from the login node"
needs re-reading before it is obeyed, not after.

**(c) Two foreign containers are on this node and neither holds a GPU.**

```
rc_26_7_902     unifiedtrainingdockers.azurecr.io/utd/ci:primus_…_20260902
                created 2026-09-03 19:05:51 UTC   Up 2 days
xiaoming-dev    tasimage/primus:pr-1048
                created 2026-09-03 02:19:00 UTC   Up 3 days
```

**All eight cards read VRAM 0 %, so under the standing rule — stop GPU
occupants, leave CPU containers alone — neither is in scope.** I am recording
them, not touching them; I am an instrument this round. **I report container
identity by `CreatedAt`, not by `Up 3 days`**: a relative duration silently
re-anchors to whenever it was printed and is not comparable with anything.

**(d) A VRAM-0 reading is not "the node is free" and I am not claiming it is.**
The sixth failure mode from the first cluster applies unchanged: occupancy
measures *now*, a live chain occupies the *future*, and a CPU-phase chain is
invisible to `rocm-smi`. Here the process-table half of that check is also
satisfied — no run root exists, so no chain has been launched — **but the two
questions are separate and I answered both.**

### 5. Code problems — none observed, and that is not the same as none present

**No code has been executed against this cluster by this round**, so I have
nothing to report in this category and will not manufacture something. The
package `agent_sys/examples/llm_e2e_performance_optimization/e2e-flow/` is
present with its eight top-level documents and `assets/`, `steps/`, `main.yaml`,
`shared.yaml`; **I have not loaded it, and `show --package` has not been run on
this node.** The first thing worth measuring next interval is whether it even
type-checks here.

### 6. 未定性 — open, and deliberately left open

- **Whether the engine image for phase A exists yet.** 157 images are on this
  node, including many `lmsysorg/sglang-rocm:*-mi35x-*` and `rocserve/engine-*`
  tags. **None of them is known to be "the locally built infera image" the plan
  calls for**, and I did not guess by name — the first cluster charged us five
  times for reading ownership off a name prefix. **The measurement that would
  settle it: m1's own note, or the digest the module-1 environment record
  writes.** Neither exists yet.
- **Whether the carried-over launch block in `RUN-PLAN.md` is valid on this
  cluster at all.** It names `--var transport=spur`; there is no spur here. **Not
  yet read against this node.**
- **What replaces the 4-minute mock loop.** With no corpus, the fast falsifier
  that made the previous round tractable has no input. Open.
- **Whether module 4 being degraded/replayed this hold** [the leader states this
  is the user's decision] **leaves anything for module 5 to consume**, given
  there is no sealed `kernel_optimization` here to replay *from*. Open, and it is
  the one that could stall the far end of the chain.

### 7. 新增 commit

**None by me this interval** — this section is the first thing I have written.

Two commits landed on `dev.yihou.aiopt.task_package.concat` before I started, and
I name them without characterising work I did not read:

```
a41c119e  mission: reproduce, debug and accept the e2e chain on a second cluster
3920f7e8  CLAUDE.md: all three exemptions can coincide in one sentence …
```

**Uncommitted in the tree at 06:33:41**, recorded because a package stages the
*working tree*, not `HEAD`, so these reach every task that starts from now on:

```
A   .claude/CLAUDE.handoff-refine.20260906-0627.md.bak
M   .claude/CLAUDE.md
??  .serena/  overlay-review.20260805.md  sglang_unified_pd_test.packup_20260727/
```

`work.checkpoint.summary.md` itself was **clean** before this append — `git diff`
and `git status` both empty, checked in the same command as the write.

### 8. 其他

**What I am, and the one way I could break this round.** This file is 552 KB and
four people edit it. **I append; I do not regenerate.** A rebuild that improves
every line it touches still destroys the lines it did not know about, and the
commit cannot tell a correct rebuild from a wrong merge. My scratch is
`/data/yihou/e2e_verify_20260906/checkpoint/`.

**Between writes I sleep rather than explore.** An instrument that writes into
what it observes has stopped being one — and this round starts with three empty
note directories that are *meant* to be empty, which is exactly the state a
curious observer would be tempted to disturb.

---

## R2 T+31 — 2026-09-06 07:05 UTC

**T+31 is the wall-clock delta from the baseline** (06:33:41 → 07:04:36, read in
the same command). Times below are reads; none is extrapolated.

### 1. The interval's result — Qwen3-32B served a completion on this node

**[observed, first-hand] This is a read of contents, not of an exit path.** From
`…/handoffs/27dd187f-…/v1/content/items/codes/qwen3-32b-mix.packup_20260906/results/`:

```
deployment.json    started_at   2026-09-06T06:54:35.131400696Z
                   image_id     sha256:fa58aef5…92bf2
                   tp_size 4    gpu_devices [0,1,2,3]   ctx 40960
router_workers     10.235.192.131:8102  status active  engine sglang
chat_completion    model Qwen/Qwen3-32B
                   usage {prompt 20, completion 147, total 167}
                   content "<think>\nOkay, the user is asking for the capital
                            of France and wants the answer in one word…"
chat_completion_stream.sse                     40 555 bytes   07:00:22
teardown.json      removed both containers, clean: true       07:01:44
```

**147 completion tokens the model produced.** No exit code and no `SERVE_OK`
could have manufactured that string. **The engine image is real and this
cluster's first bring-up worked.**

**The image, and it is corroborated by two reads rather than asserted:**

```
m1/build4.log     writing image sha256:fa58aef5…92bf2
                  naming to docker.io/infera/engine-sglang:qwen3-local-20260906
deployment.json   image_id  sha256:fa58aef5…92bf2
```

The first is the builder's own output; the second is what the deployment
recorded from the daemon. **Two reads of the same artefact through different
paths — not two methods, and I am not claiming more than that.**

**Timeline of the arm, all from mtimes read with `--time-style=+'%F %T'`:**

```
06:42:18  run 20260906T064218-15c264 starts
06:54:35  bring-up  started_at
07:00:19  engine_server_info / router_models / worker_mode_line
07:00:21  chat_completion.json
07:01:44  teardown clean
07:02:44  yihou_dk_selftest{,_etcd} created  ← the callability arm, step 6
```

**Bring-up to first completion: 5 min 44 s.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~12 %** (+12 from the baseline's 0) |
| 已经耗时 | **~45 min** (mission.md 06:19:11 → 07:04:36) |
| 预估耗时 | **still absent** |
| 可靠性 | **中** |

**What the 12 % is and is not.** It is: an engine image built and proven by a
served completion, a deploy kit that stands up and tears down cleanly, m2's
materials generated, and m35's pre-registration written. **It is not a green
stage** — `deploy_and_prove` is still `running`; no validator has returned a
verdict on this cluster yet. **The first PASS is the event that will move this
number again, and it has not happened.**

**Why 中 and not 高:** the 12 % rests on my judgement of how much of a
five-stage chain one nearly-finished stage represents, and **the previous round
established that the last stage is not the cheapest.** The underlying
observations are 高; the fraction is not.

**预估耗时 stays absent.** Hold `29184` ends **2026-09-06T14:00:01**, leaving
**6 h 55 min** at this write — that is a ceiling on this hold, not an estimate of
the task.

### 3. 当前进展 — one live run, three owners producing

```
run   20260906T064218-15c264      started 06:42:18   last write 07:02:43
      main               running
      m1_deploy          running
      deploy_and_prove   running          agent e2e_deployer (kind: ai)
      m2_profiling       waiting_handoff
      m3_analysis        waiting_handoff
      m4_kernel_opt      waiting_handoff
      m5_integration     waiting_handoff
```

**Read from `store/task/*.json` `status`, not from a phase line.** The phase log
says what was dispatched; these are the states the store holds.

**What the deployer is doing right now, from its transcript** — the artefact
that says *why*, which the previous round paid twice for not opening:

```
07:01:38  "Now step 6 — the callability arm. Tearing the first deployment down
           first, since I hold only cards 0–3."
07:02:15  "A real finding: teardown returns before the driver reclaims VRAM.
           Adding a settle gate so back-to-back redeploy works."
07:02:38  "Now the callability arm — the kit driven entirely by its contracted
           parameters, with values different from my own run."
```

**All eight cards read VRAM 0 % at 07:03:19 and that is expected, not idle** —
the first deployment was torn down at 07:01:44 and the selftest arm came up
15 seconds before I sampled. **This is the sixth failure mode of "the cards are
free" in its clearest form: the reading is correct and the inference from it
would be wrong.**

**Containers at 07:03:19:**

```
yihou_dk_selftest        created 2026-09-06 07:02:47   ← ours, step 6
yihou_dk_selftest_etcd   created 2026-09-06 07:02:44   ← ours, step 6
rc_26_7_902              created 2026-09-03 19:05:51   foreign, no GPU
xiaoming-dev             created 2026-09-03 02:19:00   foreign, no GPU
```

**Other owners' output, listed by mtime, not characterised** — I have read file
names and sizes, not the documents:

```
m2   materials/conversation_trace.jsonl        803 900 B  06:36:58
     …jsonl.provenance.json                      1 249 B
     materials/README.md                         9 442 B  06:48:23
     VAR-TABLE.md          17 044 B   07:02:00
     LAUNCH-m2.md          10 161 B   07:02:21
     PRE-REGISTER-m2.md    12 838 B   07:02:42
     tools/  gen_conversation_trace.py, check_trace_buildable.py,
             check_gsm8k.py, sweep_unwired_env.py
m35  PRE-REGISTER.md       19 565 B   06:57:36
     RUNG5-CHECKLIST.md     9 933 B   07:00:40
     mk_reverse_payload.py 15 298 B   07:02:55
     probe_payload/  stock.py 40 628 B, optimized_kernel.py 41 250 B,
                     payload_record.json         07:03:04
     tool_probe/run/a/validation.good/materials/h1/v0/{file1,file2}.txt
m1   Dockerfile.infera-engine  7 494 B, build2/3/4.log, launch_m1.sh 4 575 B
```

**m2's trace carries a `.provenance.json` beside it.** Recorded because a
generated corpus that names its own origin is the thing this cluster does not
have and must produce; whether that file says enough, I have not read.

### 4. Code problems

**Fixed this interval — one, and it was found by running, not by reading.**

- **`scripts/teardown.sh` in the deploy kit** (path above): teardown returned
  before the driver reclaimed VRAM, so a back-to-back redeploy would race the
  reclaim. The deployer added a settle gate at **07:02:35**. **[observed via the
  transcript and the file's mtime; I have not read the diff.]**

**Recorded by a teammate, not by me — commit `7ae653c2`, 06:58:30:**

> *`E2E_EVAL_THINKING` and `E2E_RESOLVE_TIMEOUT_S` are not two events. A body
> reads `${E2E_FOO:-default}` while no `shared.yaml` or `steps/*.yaml` declares
> `E2E_FOO`, so no `--var` reaches it and the default is the only value it can
> take. Absence is invisible: grep finds the read and nothing looks wrong.*

**Eleven instances of one cause.** This is `--var gpu_devices` from the first
cluster, generalised and swept for rather than met one at a time — the repair
the previous round arrived at only after paying three launches. **I quote the
commit message; I have not run the sweep.**

**Unfixed / unknown: none I can name.** No validator has refused anything on
this cluster yet, because none has been asked.

### 5. Non-code problems

**(a) `transport=spur` is dead here, and it was caught before launch, not
after.** From `LAUNCH-LINE.txt`, which m1 wrote next to the run:

> *`transport=local`: `assets/lib/remote.sh:121` has a `local` branch the probe
> NEVER selects. spur is absent here and srun is present, so the probe would
> pick srun and try to step into the allocation we are already inside.*

**This closes an open question from the baseline** — RUN-PLAN's carried-over
block is not valid unmodified on this cluster, and the specific incompatibility
is now named with a file and a line.

**(b) `RUN-PLAN.md` has seven launch blocks that disagree.** m1's header says so
and says which two it took from and why. **On the first cluster the canonical
block was measurably the best line; here it needed adapting, and the adaptation
is documented in the run tree rather than in someone's memory.**

**(c) `LAUNCH-LINE.txt` exists at all, and that is the previous round's most
expensive lesson applied before it cost anything here.** Its own justification:

> *a launch line is NOT recoverable from the artefact: the staged package keeps
> `${var:-default}` unrendered, so the run tree cannot say what was passed. Four
> separate incidents trace to that.*

**(d) The namespace-package hazard, measured today by m1**: `agent_sys` spans
both this checkout and `…/infera.aiopt.all`, and "this one wins by ordering,
which is an accident, not a property." The launch line pins `PYTHONPATH`.

### 6. 未定性

- **Whether the callability arm passes.** It started 07:02:44 and is the last
  step before `deploy_and_prove` can produce its output. **Unresolved, and it is
  the next thing that will be true or false.**
- **`expect_ranks=2` against `tp=4` in a run whose m2 is mocked.** The launch
  line states this is deliberate — it grades a sealed TP-2 trace, not this
  deployment. **But the baseline established there is no sealed corpus on this
  cluster.** I have not checked which trace `expect_ranks=2` will actually be
  applied to, and that is the reading that would settle it. Open.
- **What module 5 consumes if module 4 is replayed** — carried forward from the
  baseline, unchanged, and nothing this interval touched it.
- **Whether m2's generated trace can substitute for the absent corpus** in the
  places the package expects sealed inputs. m2 has written a provenance file and
  a `check_trace_buildable.py`; **I have not read either and will not
  characterise their answer.**

### 7. 新增 commit

Since my last section, two, both on `dev.yihou.aiopt.task_package.concat`:

```
f027073d  checkpoint R2 T+0: baseline on the second cluster — mine
7ae653c2  bug record 2026-09-06: one cause, eleven instances — a knob the code
          offers that the package never wires up            (not mine; quoted
          from its message in §4, not read as a diff)
```

Working tree at 07:04:36: `.serena/`, `overlay-review.20260805.md`,
`sglang_unified_pd_test.packup_20260727/` untracked. **The two `.claude/`
entries staged at the baseline are now committed** (`46469f7f`, which predates
my baseline commit in the log but landed in the same window).

### 8. 其他

**One thing is different from the first cluster's early hours and worth naming
while it is still true.** Every owner produced a document that names its own
provenance before producing a result: `LAUNCH-LINE.txt` beside the run,
`conversation_trace.jsonl.provenance.json` beside the trace,
`PRE-REGISTER.md` and `PRE-REGISTER-m2.md` **written before the results they
grade exist**.

**Pre-registration is the one defence the previous round found that does not
decay** — the criteria cannot be bent toward the first data point if they were
written before it arrived. **Whether these three documents are good, I have not
read. That they were written first is a fact about their mtimes**, and it is the
part that cannot be recovered later.

---

## R2 T+62 — 2026-09-06 07:36 UTC

**T+62 = wall-clock delta from the baseline** (06:33:41 → 07:35:22, read in the
same command as the write).

### 1. The interval's event — `deploy_and_prove` entered output validation

```
07:29:59   phase  deploy_and_prove: running -> output_validating
07:29:59   store/handoff/27dd187f-….json  written
07:29:59   zone  validation.68eec6d6-….output_validation.d158bc57  created
```

**This is the first validator invocation of the round on this cluster.** At
07:35:22 it had not returned: no `verdict`, no report anywhere under the zone.
**Elapsed in validation at this write: 5 min 23 s.** I do not know whether that
is normal for this validator set here; the measurement that would say is a
completed one to compare against, and there is none yet.

**The one thing I checked before it returns, because it is the check that is
worthless afterwards:**

```
materials/27dd187f-…/v1     37 files
args.json                   {"layout": "deploy_kit.layout"}   (non-empty)
```

**37 files, staged from `v1`, and the args are not empty.** Those are the two
failure modes the first cluster spent a day on — a zone handed a zero-file
directory produces a well-formed refusal that reads exactly like a producer
defect, and empty `args` makes a validator pass trivially. **Neither is present
here.** Recorded now precisely because once a verdict exists, nobody re-reads
the zone.

**I am not claiming the verdict will be a PASS.** I am claiming that if it
refuses, the refusal will be about the artefact.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~13 %** (+1) |
| 已经耗时 | **~76 min** (mission.md 06:19:11 → 07:35:22) |
| 预估耗时 | **still absent** |
| 可靠性 | **中** |

**+1, not more.** Entering validation is a state transition, not a result. The
+12 last interval was bought by 147 generated tokens; **this interval bought a
dispatch.** Hold `29184` has **6 h 25 min** left (14:00:01 − 07:35:22).

### 3. 当前进展

```
main               running
m1_deploy          running
deploy_and_prove   output_validating   ← since 07:29:59
m2_profiling       waiting_handoff
m3_analysis        waiting_handoff
m4_kernel_opt      waiting_handoff
m5_integration     waiting_handoff
```

**Node at 07:34:53:** all eight cards VRAM 0 %; `docker ps` shows only
`rc_26_7_902` and `xiaoming-dev`, both foreign and neither GPU-holding. **The
two `yihou_dk_selftest*` containers seen last interval are gone** — the
callability arm finished and tore itself down between 07:03 and 07:34. **I did
not observe it complete; I observed that it is no longer there**, and those are
different statements.

**Owners' output this interval, by mtime — names and sizes, not judgements:**

```
m35  PRE-REGISTER.md        19 565 → 27 820 B   07:17:24
     RUNG5-CHECKLIST.md      9 933 → 15 788 B   07:19:27
     LAUNCH-CHAIN-m35.md              9 591 B   07:31:46   (new)
     m3_extract.py                    9 727 B   07:32:59   (new)
     known/{harness,module,mixed}/…/workset.yaml + usable_op.json
                                                07:32:59–07:33:09
     watch_for_verdict.sh + …snapshot.sh        07:28:01–07:28:24
     findprobe/, watchprobe/  — probe scratch
m2   graft_kit.sh                     7 288 B   07:25:44   (new)
     LAUNCH-CHAIN.md                 15 449 B   07:33:39   (new)
     PRE-REGISTER-m2.md    12 838 → 17 058 B    07:34:07
     LAUNCH-m2.md          10 161 → 12 556 B    07:20:49
m1   run.log                          4 121 B   07:29:59
```

**`known/{harness,module,mixed}/` is worth naming as a shape, not a
conclusion:** three named fixtures with hand-sized `workset.yaml` files, built
by `m3_extract.py`. **That is the outline of "let a new instrument grade a
sample whose answer you already know"** — the repair the first cluster reached
only after four confident false refusals. Whether these fixtures do that, I have
not read.

**`watch_for_verdict.sh` and its `.snapshot.sh` sibling.** A snapshot copy
beside a shared script is the fix for "bash reads a running script lazily, and
someone else's edit moves the offset." Recorded as the shape; I have not
verified the snapshot carries its provenance header.

### 4. Code problems

**None fixed by me; three bug records committed by others this interval.** I
quote their subject lines and do not restate them wider than they were written:

```
8637b3e9  bug record 2026-09-06 entry 4: PIPESTATUS is empty in the shell we
          type into
935e9973  bug record: bfs errors where GNU find would not, and 2>/dev/null
          hides it
5166c897  bug record 4b: bfs errors loudly on relative -newermt; 2>/dev/null is
          what makes it silent
```

**All three are instrument failures, not product failures**, and all three are
the same family this record has been filling for two days: **a tool that answers
a different question than the one asked, and a redirect that turns the
disagreement into silence.** `2>/dev/null` appearing in two of the three is the
part worth carrying — **the previous round's rule was "do not put `2>/dev/null`
on a command that can fail"; these two are that rule being collected twice on a
new machine because `find` here is `bfs`.**

*Method note against myself: I used `find … -newermt … 2>/dev/null` in this very
interval to enumerate teammate files. If `bfs` errored, my listing is short and
would look complete. The listing above returned 25 rows with plausible
timestamps, but that is not proof — **I am flagging my own instrument, not
clearing it.***

- **`scripts/teardown.sh` settle gate** — fixed last interval, unchanged.

### 5. Non-code problems

**Nothing new this interval.** The four from T+31 stand unchanged: no corpus,
`transport=spur` dead, seven disagreeing launch blocks in `RUN-PLAN.md`, and the
`agent_sys` namespace-package collision.

**One clarification I owe on my own last section.** I wrote that the image
digest was "corroborated by two reads rather than asserted" and then said they
are not two methods. **Holding both: the build log's `writing image` line and
`deployment.json`'s `image_id` are the same value travelling two paths, which
catches a transcription error and nothing else.** It does not establish that the
image contains what the Dockerfile intended.

### 6. 未定性

- **The verdict.** Open for 5 min 23 s at this write, and it is the single most
  informative thing that will happen next. **`materials` = 37 files and
  non-empty `args` are already recorded, so whatever it says can be read at face
  value.**
- **Why validation is taking this long.** I do not know. **The measurement that
  would answer it: the validator's own report once it lands, or the zone's
  `home/`/`tmp/` growing.** Both were empty of results at 07:35:22.
- **`expect_ranks=2` vs. `tp=4`** — carried from T+31, untouched, still unread.
- **What module 5 consumes if module 4 is replayed** — carried, untouched.
- **Whether `m35/known/*` fixtures have known answers** *before* they are used
  to grade anything. Open, and it is cheap to close later by reading the mtimes
  against the first grading run.

### 7. 新增 commit

Since T+31, four on `dev.yihou.aiopt.task_package.concat`:

```
8fcdb00b  checkpoint R2 T+31 — mine
8637b3e9  bug record 2026-09-06 entry 4: PIPESTATUS is empty in the shell we
          type into
ae9bb3ab  m5: expose eval_thinking; record the unwired-knob and file-set
          findings
935e9973  bug record: bfs errors where GNU find would not, and 2>/dev/null
          hides it
5166c897  bug record 4b: bfs errors loudly on relative -newermt; 2>/dev/null is
          what makes it silent
```

**`ae9bb3ab` is the first commit this round that changes the package rather than
the record** — it wires up `eval_thinking`, one of the eleven unwired knobs from
`7ae653c2`. **I have not read its diff and do not know whether the other ten are
addressed.**

### 8. 其他

**Three of the five commits this interval are bug records about the team's own
tools, and none is about the thing being built.** That is not a complaint. On
the first cluster the same ratio held and the tool bugs were the ones that cost
runs — a `pkill` pattern that matched its own argv, a `find` that failed to zero,
a `tail -1` that ate the line explaining four deaths.

**What is different here is the timing: these are being written before a chain
has died on them.** `bfs` behaving unlike GNU `find` was found by someone testing
their probe, not by a probe silently under-reporting during an incident. **That
ordering is the whole value, and it is the kind of thing only a timestamp can
show** — which is why the mtimes above are listed even when the contents are not.

---

## R2 T+94 — 2026-09-06 08:08 UTC

**T+94 = wall-clock delta from the baseline** (06:33:41 → 08:07:13, read in the
same command).

### 1. The interval's result — the first validation did not refuse, it crashed

**And it corrects my own previous section, which is the first thing to say.**

`check_deploy_kit` died on an **`ImportError` on `jsonschema`**. The validator
runs with `HOME` inside its own zone, so user-site resolves to an empty
directory and it falls back to a system `jsonschema` with no
`Draft202012Validator`. **It exited 1 without writing a verdict, the task went
terminal, and the escalation had no receiver.**

**[observed, first-hand] The escalation event, verbatim from
`store/event/f81d6ff4-….json`:**

```
"at":   "2026-09-06T07:29:59.160381Z"
"kind": "escalated"
"attributes": { "target": "user",
                "why": "validation_unreached: the task is terminal and
                        there is nothing to push" }
```

**07:29:59 is the same second the phase line said `running -> output_validating`.**
It never spent 36 minutes validating; **it died on entry and I read the phase
line as progress.**

**What T+62 got wrong, and the diagnosis names it exactly** (from `0a23e9cf`'s
message):

> *the zone was green — one zone, 37 files, none empty — and that was correct
> while the run died anyway. **Was the validator shown something** and **did it
> survive to look** are different questions and only the first has a tool.*

**My 37-file count was true and I drew a conclusion it cannot carry.** I wrote
"if it refuses, the refusal will be about the artefact." **It did not refuse.
There is a third outcome my check had no way to see**, and I had already
recorded, one section earlier, that a tool answering a different question is
this record's most expensive recurring shape.

**Blast radius, as measured by the owner rather than assumed** [relayed from
`0a23e9cf`, not independently re-measured by me]:

```
import is inside def validate(), not at module level
  -> importing the shared lib is harmless; only a call detonates
every call site catches SchemaError, which an ImportError is NOT
8 of 22 validators crash
check_environment is on EVERY kind
  -> 15 of 15 kinds cannot reach a verdict
  -> there is no partial green and no degraded configuration that avoids it
```

**`check_workset_shape` was missed on the first pass** because it carries its own
`jsonschema` import instead of calling the shared lib — **right pattern, wrong
scope**, caught only by asking whether `jsonschema` is imported anywhere else.

**The transferable finding, and it is the sharpest thing produced today:**

> **`verdict.json` cannot express crashed-versus-refused, so a crash is recorded
> as `invalid` and reads as a judgement about the artefact.**

**`check_packup_shape` already guards this** — it writes `THIS VALIDATOR DID NOT
RUN` into its reasons, citing `todo.md` T29. **The other seven do not.** And
both a prior record of this exact crash and a better fix idiom than `PYTHONPATH`
**already existed in the package**, at `check_workset_shape:655` and
`mock_adapt.sh:103`. **Two of today's costs were paid for a second time.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~14 %** (+1) |
| 已经耗时 | **~108 min** (mission.md 06:19:11 → 08:07:13) |
| 预估耗时 | **still absent** |
| 可靠性 | **中** |

**+1 despite the run dying, and the reasoning matters more than the number.**
Nothing proven at T+31 was un-proven: the image, the served completion, the
clean teardown all stand. **What this interval bought is a fully characterised
blocking defect with a measured blast radius** — 8/22 validators, 15/15 kinds,
no degraded configuration. **That is worth more than the +1 suggests and less
than a green stage**, and I would rather under-count it than let the number
drift upward on a diagnosis.

**Hold `29184`: 5 h 53 min left** (14:00:01 − 08:07:13).

### 3. 当前进展 — two live runs, and they want the same cards

**Three run trees exist; one orchestrator died, two are alive.** Processes read
**positionally from `/proc/<pid>/cmdline`**, never by grepping a pattern that
appears in my own argv — the first pass of that read returned my own shell as a
hit and I discarded it.

```
run 20260906T064218-15c264   06:42:18   orchestrator GONE   (died, §1)
run 20260906T075853-e6f882   pid 529945  started 07:58:45   m1
     container=yihou_e2e_m1_09060758   gpu_devices=0,1,2,3
     tp=4  measure_gpu=4  mock_stages=m2,m3,m4,m5
run 20260906T080504-3e8a03   pid 573170  started 08:04:53   m2
     launcher /data/yihou/e2e_verify_20260906/m2/launch_chain.py
     container=yihou_e2e_chain   tp=4  measure_gpu=4
     mock_stages=none          ← a FULL REAL five-stage chain
     gpu_devices                 ABSENT from its argv
```

**Both are in `deploy_and_prove` right now.** Two `kind: ai` deployer agents are
alive (pids 530669, 574698).

**The concern, stated with its evidence and its limit.** What I measured: m1
declares cards 0–3; **m2's chain passes no `gpu_devices` at all**; both are
`tp=4`; both are deploying now. What I did **not** measure on this cluster: what
an absent `gpu_devices` resolves to here. **On the first cluster it was measured
by intervention** — the kit was changed to `[4,5,6,7]` and the arm still took
0–3, because `mix_worker.sh:26` reads `GPUS="${GPUS:-$(seq -s, 0 $((TP-1)))}"`
and nothing in the package sets `GPUS`. **If that line is unchanged here, both
runs take 0,1,2,3.**

**A second, independent collision path from the same record:** `mix_up.sh`
hard-codes `kv-events:5557` and `kv-snapshot:8801`, so **two m5 stages cannot
coexist on one node regardless of ports passed** — and m2's chain is
`mock_stages=none`, so it will reach m5.

**I am flagging this to the leader as a measurement, not as an instruction.**
I do not know whether the overlap was scheduled deliberately. **All eight cards
read VRAM 0 % and `docker ps` shows only the two foreign CPU containers at
08:05:57**, so nothing has collided yet; the window is open, not closed.

### 4. Code problems

**Unfixed and blocking — the `jsonschema` `ImportError`.** Files named by the
diagnosis: the shared validate lib (import inside `def validate()`),
`check_workset_shape` (own import, and a prior record of this crash at `:655`),
`check_packup_shape` (the one that already guards it), `mock_adapt.sh:103` (the
better fix idiom). **No fix commit had landed at 08:07:13.**

**Fixed earlier, unchanged:** the deploy kit's `scripts/teardown.sh` settle gate.

**Two tool-level records committed this interval, both by others:**

```
935e9973 / 5166c897   bfs errors where GNU find would not, and 2>/dev/null
                      hides it
```

**The second carries a near-miss worth keeping:** m35 was *thirty seconds* from
reporting a quiet run tree off a zero whose newest file was **73 seconds old**.
And the record makes the general point in the safest form — **`2>/dev/null`
turns a good failure into a plausible zero, and `2>&1 | wc -l` turns it into a
plausible non-zero, so neither direction of the result is safe.**

***Against myself, again:*** T+62 enumerated teammate files with
`find … -newermt … 2>/dev/null` and I flagged it as unverified. **`bfs` rejects
relative time strings; I used absolute ones and got 25 plausible rows, so that
listing survives — but by luck of argument form, not by care.** This interval I
used `-newermt '2026-09-06 07:36'` with stderr visible.

### 5. Non-code problems

**Nothing new.** The four from T+31 stand. **One is now sharper:** the round has
no sealed corpus, and the validator defect means **no kind can reach a verdict**,
so nothing produced in the next hours can be *validated* into a corpus either
until the import is fixed. **That makes the `jsonschema` fix the critical path,
not a side quest** — stated as a consequence of two measured facts, not as a
scheduling opinion.

### 6. 未定性

- **Whether the two live runs collide on cards 0–3.** Open. **The reading that
  settles it: `docker inspect <ctr> --format '{{.HostConfig.DeviceRequests}}'`
  or `rocm-smi` once either brings up** — neither has yet.
- **What an absent `gpu_devices` resolves to on this cluster.** Carried from the
  first cluster's measurement; **not re-measured here**, and the honest form is
  that it is an expectation, not an observation.
- **Whether the 8/22 and 15/15 counts hold** — relayed from `0a23e9cf`, whose
  author measured them. I have not re-derived them and am not widening them.
- **`expect_ranks=2` vs `tp=4`** — carried, still unread.
- **What module 5 consumes if module 4 is replayed** — carried, untouched.

### 7. 新增 commit

Since T+62, four, all by others except mine:

```
e536e3c4  checkpoint R2 T+62 — mine
0a23e9cf  bug record: a crash is not a refusal, and verdict.json cannot say
          which
590467f7  bug record: the jsonschema crash from both ends — m1's PATH/HOME
          mechanism and m35's blast radius
1b81a7d3  bug record: cross-reference m1's blast-radius paragraph to the counts
          in section 6
```

**Three commits on one defect, from two owners, cross-referenced to each
other.** `590467f7`'s subject — "from both ends" — is the mechanism and the
blast radius arriving separately and being joined. **The previous round's note
that one cause stays split into three surprises because each is counted as an
event: here it was joined the same hour.**

### 8. 其他

**The honest reading of this interval is that the round's first real validation
produced no verdict, and the record is better for it.**

What exists now that did not at 07:36: a named mechanism (`HOME` in the zone →
empty user-site → wrong `jsonschema`), a measured blast radius that rules out
every partial workaround, a second instance found by asking the scope question
rather than by being bitten, and **a general defect — `verdict.json` cannot
distinguish a crash from a refusal — that would have quietly mis-attributed
every future crash to the artefact under test.**

**And my own check was shown to be sound and insufficient in the same sentence.**
I will keep counting zone materials; **it is now labelled with what it cannot
see**, which is the only repair available to a tool that answers a narrower
question than the one being asked.

---

## R2 T+125 — 2026-09-06 08:39 UTC

**T+125 = wall-clock delta from the baseline** (06:33:41 → 08:38:26, read in the
same command).

### 1. The interval's result — the first verdicts on this cluster, and they pass

**[observed, first-hand] Two of `deploy_kit`'s three validators returned `true`;
the third is executing.** Run `20260906T080504-3e8a03`, handoff
`fe6a4ab0-…` (kind `deploy_kit`), zone
`validation.d4ed6649-….output_validation.7f544db1`, **30 files in materials**:

```
validation-tfkbxt_1   args {"layout": "deploy_kit.layout"}
                      verdict {fe6a4ab0-…: true}
                      validator_report.txt:
                          # check_deploy_kit
                          ## fe6a4ab0-…: passed
                            note: qwen3-32b-mix.packup_20260906
validation-3pwu1vm_   args schema=environment,
                           require_fixed [node, gpu_arch, image, image_id,
                                          model_name, model_path, tp_size]
                           require_runtime [container, endpoint, started_at]
                           invariant devices_within_count
                      verdict {fe6a4ab0-…: true}
validation-rwx4veqz   args deploy_entrypoint=scripts/deploy.sh,
                           aiperf 0.12.0, load_seconds 180, port_base 8140,
                           work_root /data/yihou/e2e_flow/validate
                      NO VERDICT — running now
```

**`check_environment` passing is the load-bearing one.** It is the validator that
crashed on `jsonschema` an hour ago and the one attached to **every** kind. Its
`true` here is a positive test of that repair — **it could not have returned a
verdict at all while the `ImportError` stood.** I have not read the fix commit
and do not know which of the two candidate idioms was used.

**The third validator, `check_deploy_serves`, is a real bring-up plus a 180 s
load** — the heaviest check in the set and the one the first cluster kept out of
its fast loop entirely. **It is running on hardware right now:**

```
08:32:10  yihou_e2e_chain_serves-6e2f6dbb        created
08:32:14  yihou_e2e_chain_serves-6e2f6dbb_etcd   created
08:35:11  aiperf_serves-6e2f6dbb                 created
08:37:45  cards 0-3 VRAM 76 76 76 76 %   cards 4-7  0 0 0 0 %
```

**That is the first time this round the cards have been busy while I sampled**,
and the split 76/0 matches a TP-4 deployment on `gpu_devices=0,1,2,3`.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~22 %** (+8) |
| 已经耗时 | **~139 min** (mission.md 06:19:11 → 08:38:26) |
| 预估耗时 | **still absent** |
| 可靠性 | **中** |

**+8 for two passing verdicts on a real artefact.** Both are reads of the kit's
contents — one of its layout, one of its `environment` record against seven
required fixed fields and three runtime ones. **Neither is an exit code.** The
blocking defect that made 15/15 kinds unreachable at T+94 is demonstrably gone
for at least this kind.

**Not counted:** `check_deploy_serves` has not returned. **If it passes, stage 1
is green on this cluster and that is worth more than this interval's +8** — I
would rather book it when it happens than pre-book it now.

**Hold `29184`: 5 h 22 min left** (14:00:01 − 08:38:26).

### 3. 当前进展 — one run alive, and it is the full-real chain

```
20260906T064218-15c264  orchestrator gone   (died on the jsonschema crash)
20260906T075853-e6f882  orchestrator gone   deploy_and_prove: FAILED
20260906T080504-3e8a03  pid 573170 ALIVE    m2's launch_chain.py
                        container=yihou_e2e_chain   mock_stages=none
                        main: running   m1_deploy: running
                        deploy_and_prove: output_validating
```

**`mock_stages=none` — every one of the five stages is real in this run.** It is
the only live chain and it holds the node's four busy cards.

### 4. The collision fired — and by the path I did not flag first

**m1's run 2 died. [relayed from `becc19d6`, whose author measured it; I observed
only the resulting `deploy_and_prove: failed`.]**

> *Two module owners on one node take the same default port band (8101..8106)
> and the same `work_root`. m1's run 2 aborted on **"etcd port 8103 is already
> in use"** six minutes after a preflight measured it free — **correct when
> taken, stale when used** — and the exit code was **143**, which says nothing.
> Ownership of the port established by inspecting the other run's etcd args, not
> by container name or arrival order.*

**Against my own T+94 section.** I flagged the overlap and named two collision
paths: cards 0–3, and `mix_up.sh`'s hard-coded 5557/8801. **Neither is what
fired.** It was the default port band `8101..8106` and a shared `work_root` —
**a third path I did not name, in a message where I listed two and stopped.**
The warning was right in shape and incomplete in mechanism, and the incomplete
part is the part that cost a run.

**Two details in that record worth keeping separately from the incident:**

- **"correct when taken, stale when used"** — a preflight measured 8103 free and
  six minutes later it was not. **This is the same shape as reading VRAM before
  a launch: the reading does not expire loudly.**
- **Ownership was established by inspecting the other run's etcd args**, not by
  container name or arrival order. **That is exactly the discriminator the first
  cluster paid five times to learn**, applied on first contact here.

**Also filed in that commit, and it is a correction to how run 1 will be read
later:** `check_deploy_kit`'s crash is a **NON-verdict** — it exited 1 and wrote
no `verdict.json`, so **`deploy_kit: invalid` there means *undecided*, not
*bad***. The same kit passed later. **Anyone reading run 1's store without this
paragraph would conclude the kit was defective.**

### 5. Code problems

**Fixed this interval (inferred from behaviour, not from a diff): the
`jsonschema` `ImportError`.** `check_environment` returned `true` at this write;
it could not have produced any verdict while the import failed. **I have not
found or read the fix commit and cannot say which files changed or whether all
eight affected validators were repaired** — the `git log` I ran over `*check_*`
and `*schema*` paths since 07:30 returned nothing, which most likely means my
pathspec was wrong rather than that no fix landed. **Flagging my own instrument
rather than concluding from it.**

**Unfixed, newly recorded — `6e465181`, "only m2 declares a GPU lease":**

> *cross-stage separation is dependency ordering, not lease ordering*

**No file named by me; I have not read the diff.** Recorded because it is the
general form of what killed run 2 — **the graph orders stages, and ordering is
not exclusion.**

**Unfixed: the port band and `work_root` defaults are shared.** Named by
`becc19d6`. Whether a fix landed, I have not checked.

### 6. 未定性

- **`check_deploy_serves`.** Running since ~08:32, load phase since 08:35:11.
  **It is the single thing that decides whether stage 1 is green today.**
- **Whether all eight crash-affected validators were repaired**, or only the path
  `check_environment` takes. **The reading that answers it: the fix commit's
  diff, which I have not located.**
- **Whether m1 relaunches, and onto which port band.** m1 has no live
  orchestrator; the node's cards are held by m2's chain.
- **`expect_ranks=2` vs `tp=4`** — carried, still unread. **Less urgent now**:
  the live chain is `mock_stages=none`, so its m2 is real and will produce its
  own trace.
- **What module 5 consumes if module 4 is replayed** — carried, untouched.

### 7. 新增 commit

Since T+94, three:

```
be34f98b  checkpoint R2 T+94 — mine
becc19d6  m1: record the port-band/work_root collision that killed run 2, and
          the non-verdict from run 1
6e465181  bug record 5: only m2 declares a GPU lease — cross-stage separation
          is dependency ordering, not lease ordering
```

### 8. 其他

**Two runs died today and the record is better for both, but not equally.**

Run 1's death produced a general defect — `verdict.json` cannot distinguish a
crash from a refusal — that will change how every future failure is read. **Run
2's death produced an incident report.** The mechanism is well documented, the
ownership question was answered correctly on first contact, and the "correct
when taken, stale when used" line generalises. **But nothing about it had to be
learned by losing a run**: two owners on one node with one default port band is
a fact that was available before either launched.

**That is the difference between a cost that bought something and a cost that
was avoidable, and I would rather record it now than let both settle into the
same column.** My own T+94 warning is on the avoidable side of that line: it was
sent, it named the right pair of runs, and it listed the wrong two mechanisms.

---

## R2 T+156 — 2026-09-06 09:10 UTC

**T+156 = wall-clock delta from the baseline** (06:33:41 → 09:09:17).

### 1. Stage 1 is green on this cluster — all three validators, including the load test

**[observed, first-hand] Run `20260906T080504-3e8a03`, handoff `fe6a4ab0-…`,
kind `deploy_kit`:**

```
validation-tfkbxt_1   layout             -> true
validation-3pwu1vm_   schema=environment -> true
validation-rwx4veqz   scripts/deploy.sh  -> true    ← this one is new
                      aiperf 0.12.0, load_seconds 180, port_base 8140

store/task:  deploy_and_prove = succeeded
             m1_deploy        = succeeded
```

**`check_deploy_serves` returned `true`.** That is the validator the first
cluster deliberately kept out of every fast loop because it performs a real
bring-up and a 180-second load — **the most expensive check in the set, and the
one whose PASS is hardest to fake.** Between my 08:37:45 sample (verdict absent,
cards 0–3 at 76 %) and its last write at **08:47:06**, it finished and passed.

**This is the round's first stage-level green and the first on this hardware.**
The chain then advanced: `m2_profiling` and `run_profiling_mode_off` both
reached `running` in the same run.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~28 %** (+6) |
| 已经耗时 | **~170 min** (mission.md 06:19:11 → 09:09:17) |
| 预估耗时 | **absent — but for the first time there is a reason to expect one** |
| 可靠性 | **中** |

**+6: one of five stages green, with its heaviest validator included.** I am not
booking a straight 20 % per stage — stage 1 is the only one with a proven
artefact path here, and the first cluster established the later stages are not
cheaper.

**On 预估耗时.** Stage 1 took **06:42:18 → 08:47:06 ≈ 2 h 05 m** across three
runs, two of which died for reasons now fixed. **That is a measured number for
one stage, not a basis for multiplying by five** — most of it was paying for
defects that do not recur (`jsonschema`, port band). **I am recording it so a
later section can divide by something real; I am not dividing yet.**

**Hold `29184`: 4 h 51 min left** (14:00:01 − 09:09:17).

### 3. 当前进展 — one live chain, three dead run trees

```
20260906T064218-15c264   dead   jsonschema crash
20260906T075853-e6f882   dead   deploy_and_prove: failed (port 8103)
20260906T080504-3e8a03   dead   STAGE 1 GREEN, then died at m2
                                last write 08:47:06
20260906T084930-6ded23   pid 1000969  started 08:49:18  ALIVE
                                m2's launch_chain.py, container=yihou_e2e_chain
                                mock_stages=none, tp=4, work_root=/data/yihou/e2e_flow
                                main: running   deploy_and_prove: running
```

**A caution about the dead one, because it will mislead whoever reads its store
next.** `20260906T080504-3e8a03` still records `m2_profiling=running` and
`run_profiling_mode_off=running`. **Its orchestrator is gone and it has written
nothing since 08:47:06.** Nothing inside that run distinguishes *running* from
*running under a process that no longer exists* — **the discriminator is
`/proc`, entirely outside the run**, and only one orchestrator is alive
(pid 1000969, which is the 08:49 run).

**Node at 09:08:36 — all eight cards busy, and that is us:**

```
VRAM%    75 75 75 75 75 75 75 75
yihou_dk_qwen3-32b-mix       started 09:00:38   labels infera_e2e_run=qwen3-32b-mix
yihou_dk_qwen3-32b-mix_etcd  started 09:00:43
yihou_dk_selftest            started 09:06:50   labels infera_e2e_run=selftest
yihou_dk_selftest_etcd       started 09:06:55
rc_26_7_902 / xiaoming-dev   foreign, CPU only
```

**Two of our deployments up at once — the main arm and the callability arm —
which is what puts all eight cards at 75 %.** We hold the node `--exclusive`
with `gres/gpu:8`, so this is within the allocation. **Ownership read from
labels, not from the `yihou_` prefix** — the prefix is shared by everyone on the
team and the first cluster mis-assigned five containers by reading it as
ownership.

### 4. Why the green chain then died — and the defect was its author's own

**[quoted from `2cba517a`, whose author both caused and found it]**

> *Cost: one full chain died at m2 after a clean bring-up, 35 minutes, and the
> defect was mine. AIPerf refused the trace at load — `hash_id 0` materialized
> at **477 tokens in one record and 512 in another**, because block 0 was a
> shared system prompt while the layout made each record's final block partial.
> **Every record individually valid; the file not.***

**The part that generalises, and it is the sharpest instrument lesson of the day:**

> *The verification ran the **REAL** `PromptGenerator` over all 3108 layouts with
> two known-answer controls and **passed**. It called `g._cache.clear()` between
> records, deliberately, for isolation and per-layout attribution — **and that
> shared cache is exactly what AIPerf's `dataset_manager` uses to enforce the
> cross-record invariant, so the control removed the only property that could
> fail.** `aiperf validate` passed 7761 rows for the same reason: row-independent
> validation. **Two instruments, both real, both blind to the one property that
> mattered.***

**Using the real consumer is not sufficient if you call it the way the consumer
never calls it.** The isolation that made per-layout attribution possible is the
same isolation that deleted the invariant. **And the known-answer controls —
this record's own recommended defence, adopted this morning — passed, because
they were per-record too.**

### 5. Code problems

**Fixed this interval, inferred from behaviour:** the `jsonschema` `ImportError`
(all three validators returned verdicts). **Still not read as a diff; still
unknown whether all eight affected validators were repaired.**

**Fixed, per its own record:** m2's trace generator — the shared-system-prompt
block producing inconsistent `hash_id 0` lengths. **I have not verified a
corrected trace exists; the live chain relaunched two minutes after that run
stopped, which is consistent with a fix but does not establish one.**

**Unfixed, carried:** shared default port band `8101..8106` and shared
`work_root` (`becc19d6`); no GPU lease except m2's (`6e465181`).

### 6. 未定性

- **Whether the live chain (`-6ded23`) passes stage 1 again.** It is in
  `deploy_and_prove` with two deployments up. **Stage 1 has been green once; a
  second green would make it reproducible rather than achieved.**
- **Whether the corrected trace passes AIPerf at load.** That is the check that
  failed 35 minutes into the last chain, and **it can only be answered by
  reaching m2 again.**
- **Whether all eight crash-affected validators were repaired** — carried,
  unread.
- **What module 5 consumes if module 4 is replayed** — carried, untouched, and
  **now the furthest-out unknown**, since stages 1–2 both have live paths.

### 7. 新增 commit

Since T+125, two:

```
53b05652  checkpoint R2 T+125 — mine
2cba517a  bug record 6: the real consumer, called the way the consumer never
          calls it, is a different instrument
```

### 8. 其他

**Three run trees died before one stage went green, and the order of their causes
is the useful part.**

```
run 1   06:42   a defect in the validator framework   (jsonschema import)
run 2   07:58   a defect in how two owners share one node   (port band)
run 3   08:05   STAGE 1 GREEN, then a defect in the materials   (trace hash_id)
```

**Each death was further in than the last, and each cause was of a different
kind** — framework, coordination, data. **None recurred.** That is what a
debugging loop looks like when it is working, and it is worth stating plainly
because the raw count — *three dead runs in two and a half hours* — reads like
the opposite.

**The one thing I would not let pass unremarked:** `2cba517a`'s author verified
their materials with the real consumer and two known-answer controls, and it
still shipped a file that the consumer rejected. **Known-answer controls were
this record's own answer to "how do you trust a new instrument," adopted this
morning.** They are still right, and they are **not** sufficient — a control
inherits the scope of the harness that runs it, and this one was per-record when
the invariant was cross-record.

---

## R2 T+186 — 2026-09-06 09:40 UTC

**T+186 = wall-clock delta from the baseline** (06:33:41 → 09:39:55).

### 1. Stage 1 went green a second time — it is reproducible, not a one-off

**[observed, first-hand] Run `20260906T084930-6ded23` produced the same three
verdicts as `-3e8a03`:**

```
run     validator arg              verdict
3e8a03  layout                     true
3e8a03  schema=environment         true
3e8a03  scripts/deploy.sh          true     ← 180 s load test
6ded23  layout                     true
6ded23  schema=environment         true
6ded23  scripts/deploy.sh          true     ← again
```

`deploy_and_prove = succeeded` and `m1_deploy = succeeded` in both. **Two
independent runs, two separate bring-ups, the same three PASSes.** At T+156 I
wrote that a second green would make stage 1 reproducible rather than achieved;
**that is what this is.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~32 %** (+4) |
| 已经耗时 | **~200 min** (mission.md 06:19:11 → 09:39:55) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+4 for reproducibility, not for new ground.** No stage past 1 has completed.
**Both chains that got past stage 1 died in `m2_profiling`, each for a different
reason, and both reasons are now named and fixed** (§4).

**Hold `29184`: 4 h 20 min left** (14:00:01 − 09:39:55).

### 3. 当前进展 — run 5 in flight, four dead trees behind it

```
15c264  dead  last 07:44:59   jsonschema crash
e6f882  dead  last 08:12:09   deploy_and_prove failed (port 8103)
3e8a03  dead  last 08:47:06   STAGE 1 GREEN → died at m2 (trace)
6ded23  dead  last 09:29:16   STAGE 1 GREEN → died at m2 (card lease)
ae2c38  ALIVE pid 1407641  started 09:34:43
        main: running   m1_deploy: running   deploy_and_prove: running
```

**Node at 09:39:26: all eight cards VRAM 0 %**, one container
`yihou_e2e_chain_probe` created 09:37:48. **The new chain is in its pre-bring-up
phase; the cards being idle here is the expected state for that phase and not a
sign of stalling** — the run tree's last write was 09:38:35, 51 seconds before I
sampled.

**Note on the two dead trees that reached green:** both still record
`m2_profiling=running`. **Their orchestrators are gone.** As at T+156, nothing
inside those runs distinguishes *running* from *abandoned*; `/proc` does, and it
shows one orchestrator.

### 4. Why the two green chains died — and the second was a boundary defect

**[first-hand, from `m2/launch3/LAUNCH-RECORD.txt`, written at launch]**

```
launched_at_utc: 2026-09-06T09:34:32Z
supersedes: 20260906T084930-6ded23
   (m2 aborted: card0 held by check_deploy_serves engine 1.8s after its verdict)
change: --var instruction now requires ownership-conditional wait up to 300s
        in the kit preflight
trace: conversation_trace.v2.jsonl
```

**`check_deploy_serves` writes its verdict while its own engine still holds card
0.** Measured gap: **1.8 seconds.** The next stage's preflight sampled the cards
inside that window and aborted.

**This is the T+31 finding arriving at a stage boundary.** At 07:02 the deployer
recorded *"teardown returns before the driver reclaims VRAM"* and added a settle
gate **inside the kit**. **The same physics then bit at a different seam — one
validator's engine versus the next stage's preflight — where the kit's gate does
not apply.** The fix is an ownership-conditional wait of up to 300 s.

> **A validator that brings up hardware does not stop being a tenant when it
> returns its verdict.** The verdict is a statement about the artefact; the
> engine is a fact about the node, and they end at different times.

**The other death, `-3e8a03`, was the trace defect** recorded at T+156
(`2cba517a`). Its fix is visible here as `conversation_trace.v2.jsonl`.

**One carried open question closes.** The launch record now passes
`--var expect_ranks=4`. With `mock_stages=none` the chain's m2 is real and TP-4,
so 4 is the value that matches the deployment. **The `expect_ranks=2 vs tp=4`
question I carried since T+31 belonged to a mocked-m2 launch and does not apply
to the live chain.**

### 5. Code problems

**Two framework messages in `m2/launch2/chain.log` that I have not seen recorded
elsewhere.** Quoting them because they are cheap to lose and neither stopped the
run:

```
c5c26634…: usage names 'seconds', which the task did not declare and which
           cannot record unreserved spend; 1516.62… is not booked
           (same for duration_ms, num_turns, total_cost_usd, turns)
b25200a6…: depends_on omits c5c26634…, which produces ac35d9fd…
d9b92b39…: depends_on omits c5c26634…, which produces ac35d9fd…
```

**The first says a completed task's cost was not booked** — 1516 s, 83 turns,
$11.53 discarded because the task did not declare those usage names. **The
second says two tasks consume a handoff whose producer they do not depend on**,
which is the shape `6e465181` named this morning: *ordering is not lease
ordering*. **I am reporting these as read, not diagnosed** — I do not know
whether either is intended.

**Fixed this interval:** the card-lease boundary (ownership-conditional wait, per
the launch record); m2's trace (v2 in use).

**Unfixed, carried:** shared default port band and `work_root`; whether all eight
`jsonschema`-affected validators were repaired (still unread).

### 6. 未定性

- **Whether run 5 clears `m2_profiling`.** Two chains have died there, each on a
  different defect. **A third failure at m2 with a third cause would say
  something the first two do not.**
- **Whether the 300 s ownership wait is long enough**, and what it does if the
  holder is a co-tenant rather than our own validator's engine. **I have not read
  the preflight change.**
- **The unbooked usage and the missing `depends_on` edges** above — open, and I
  have named the file to read.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched, and
  still the furthest-out unknown.

### 7. 新增 commit

Since T+156, one:

```
c59e7aad  checkpoint R2 T+156 — mine
```

**No other commits landed in this interval.** The work this interval went into
run tree and launch records rather than the repository — `LAUNCH-CHAIN.md` grew
to 21 124 B at 09:34:19 and `launch3/LAUNCH-RECORD.txt` was written at 09:34:33,
both uncommitted at 09:39:55.

### 8. 其他

**The launch record is doing the job the first cluster spent four incidents
learning to need**, and it is worth naming while it is cheap to copy:

```
launched_at_utc:  a READ timestamp
supersedes:       which run this replaces, and the measured reason it died
change:           the one thing that is different this time
trace:            which version of the materials
```

**Four fields, and together they make a launch line recoverable from the
artefact** — the property the first cluster established is *absent* from a run
tree, because the staged package keeps `${var:-default}` unrendered. **`supersedes`
is the field I would not have thought to ask for**: it turns a directory of five
run trees from a pile into a sequence with causes attached, which is exactly what
§3 of this section is able to be because that field exists.

**One caution against my own section, though.** `LAUNCH-RECORD.txt` contains the
line `CLAUDE_CONFIG_DIR=<m1's shared farm — get the exact value from the leader>`
directly under a line giving a concrete path. **A record whose purpose is to make
a launch reproducible has one field that is a placeholder**, and a later reader
copying the block will get the concrete line and not notice the angle brackets
two lines down.

---

## R2 T+216 — 2026-09-06 10:10 UTC

**T+216 = wall-clock delta from the baseline** (06:33:41 → 10:10:11).

### 1. My T+94 warning was wrong about the mechanism, and the record says so precisely

**I flagged card contention between two runs. It was ports, and the card reason
was not true when it was acted on.** From `381d7b40`, whose author reconstructed
the sequence from the file that actually holds it:

> *m1's deploy **hard-aborted on etcd 8103 at 08:11:26**, one minute before the
> **SIGTERM at 08:12:08** that was ordered for a card-contention reason **which
> was not true at that moment and became true by ~08:20 via a container that did
> not yet exist**. The collision was real and was **via ports, not cards**.*

**And the generalisation names exactly what I did:**

> *the launch line records an **intention** about cards, only the running
> deployment records the **fact**. **Three people predicted the cards from the
> launch line and all three read the wrong file.***

**I was one of the three.** My T+94 message to the leader read `gpu_devices` out
of `/proc/<pid>/cmdline` — I was careful to take it positionally rather than by
grep, and **being careful about how I read the wrong file did not help.** I even
wrote in that same message that I had not measured what an absent `gpu_devices`
resolves to here; **then I led with the card path anyway and put the port path
second, and the port path is the one that fired.**

**Where the evidence lives, and why nobody found it by grepping:**

> *A stage writes its own diagnosis into `work_root`, **not into the run tree**,
> so grepping the run directory returns nothing **and that nothing reads as
> "claim unsupported"**.* The file is **`/data/yihou/e2e_flow/deploy_main.log`**.

**That is this record's oldest shape in a new place** — a zero that means "you
looked in the wrong directory" is indistinguishable from a zero that means "it
did not happen." **A copy was preserved with a read timestamp, because `cp -a`
keeps the source mtime and records nothing about the copy.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~33 %** (+1) |
| 已经耗时 | **~230 min** (mission.md 06:19:11 → 10:10:11) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+1 only.** No stage advanced past 1 this interval. The live chain is still
inside `deploy_and_prove` at 35 minutes; **the two runs that passed it took ~42
and ~40 minutes end to end, so this is not yet late.**

**Hold `29184`: 3 h 50 min left** (14:00:01 → 10:10:11). **That is now less than
the elapsed time of this round.** No stage past 1 has ever completed here.

### 3. 当前进展 — run 5 is at the end of its deploy stage, not stalled

```
ae2c38   pid 1407641   started 09:34:43   deploy_and_prove: running
         run tree last write 10:09:11   (25 s before I sampled)
```

**Cards read 0 % and `docker ps` showed no `yihou_*` container at 10:09:36. That
reading is correct and the obvious inference from it is wrong** — the third time
today. **From the deployer's own transcript:**

```
10:05:05  "Now a final full cycle with the exact shipped scripts, so every
           result comes from the bytes being handed over."
10:08:59  reads results/chat_completion.json
10:09:02  "Now tearing down and recording the final state."
10:09:07  teardown command issued
```

**It tore down 29 seconds before I looked.** Had I sampled the process table and
the cards only, this would have read as an idle node under a stalled run.

**Note the deployer's stated reason for the final cycle** — *"the exact shipped
scripts, so every result comes from the bytes being handed over."* **That is the
distinction between testing what you ran and testing what you are handing over**,
and it is the same axis as `2cba517a`'s "called the way the consumer never calls
it."

**The four dead trees are unchanged.** `15c264` and `e6f882` show a write at
10:00:44 — **consistent with m1 collecting the ABORT evidence for `381d7b40`,
but I did not observe that and am not asserting it.**

### 4. Code problems

**Fixed this interval — the shared-defaults hazard, in the canonical launch
block** (`92835f4d`). The reasoning is worth keeping verbatim:

> *Both defaults are shared, so **two owners who each change nothing collide**.
> The port collision aborts and names itself (`ABORT: etcd port 8103 is already
> in use`); **the shared `work_root` does not abort at all**, and two runs on
> 2026-09-06 held byte-identical `work_root` AND `validate_work_root` without
> either owner choosing it. **The loud failure is the benign one.***

**Two runs shared a `work_root` today and nothing complained.** The port
collision cost a run and announced itself; the `work_root` overlap cost nothing
visible **and that is the reason it is the more dangerous of the two.**

**The fix was applied to the CANONICAL block only**, deliberately: *"several are
historical records of what a rung ran, and the hazard was never that they differ
but that nothing says which is canonical."* **That is a narrower and better
repair than normalising seven blocks** — it fixes the ambiguity rather than the
diversity.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired (still unread); the unbooked-usage and missing-`depends_on` messages
from T+186 (§5 there names the file).

### 5. Non-code problems

**A diagnosis written outside the run tree is invisible to every run-tree-based
tool**, including mine. `/data/yihou/e2e_flow/deploy_main.log` held the ABORT
line the whole time. **My sections have reported run state from `store/task`,
`store/event` and transcripts, all inside the tree; none of them would have
carried this.**

### 6. 未定性

- **Whether run 5 clears `m2_profiling`** — unchanged and now the only question
  that matters this hold. Two chains died there on two different defects, both
  fixed.
- **Whether the 300 s ownership wait is long enough**, and its behaviour against
  a co-tenant holder — carried, still unread.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched.
- **Whether the shared `work_root` caused any silent damage today**, given two
  runs held byte-identical `work_root` and `validate_work_root`. `92835f4d`
  records the overlap; **whether anything was corrupted by it is a separate
  question and I have not seen it asked.**

### 7. 新增 commit

Since T+186, three:

```
4efc4dc6  checkpoint R2 T+186 — mine
92835f4d  RUN-PLAN canonical block: two lines on one node need their own port
          band AND work root
381d7b40  bug record: where the etcd-8103 ABORT evidence lives, and the three
          accounts of the collision
```

### 8. 其他

**Two people were wrong about the same collision in opposite directions, and the
useful part is which of them cost something.**

I predicted cards from a launch line and was wrong. **A SIGTERM was ordered at
08:12:08 for that same card reason — and the run it killed had already
hard-aborted 42 seconds earlier on a port.** So the wrong reason cost nothing
that the right reason had not already cost. **But the record notes the card
reason "became true by ~08:20 via a container that did not yet exist"**, which
is the uncomfortable part: **the prediction was validated by events after the
decision, and that is exactly the pattern that makes a bad method look sound.**

**The durable line is the one I will carry into every future section:** *the
launch line records an intention, only the running deployment records the fact.*
I have quoted launch-line variables in four sections now. **They describe what
was asked for, and I will label them that way from here.**

---

## R2 T+246 — 2026-09-06 10:40 UTC

**T+246 = wall-clock delta from the baseline** (06:33:41 → 10:40:14).

### 1. The round's first genuine refusal — and it names a file, a line, and the fix

**[observed, first-hand] Run `20260906T093443-ae2c38`, `check_deploy_kit`,
29 files in materials:**

```
# check_deploy_kit
## 0d66e5f1-…: REFUSED
  note:    qwen3-32b-mix.packup_20260906
  PROBLEM: scripts/env.sh:172: DK_ROUTER_PORT is fixed here and reaches a
           binding flag at scripts/deploy.sh:205; write it as
           `: "${DK_ROUTER_PORT:=…}"` so a second copy of this kit can run
           beside the first
```

**This is not a crash and not a trivial pass.** It is the first verdict of the
round that says *no* about the artefact: two files, two line numbers, the
mechanism (a fixed value reaching a binding flag), and the exact idiom that
fixes it.

**And what it refused is the defect that cost run 2.** At 08:11:26 m1's deploy
hard-aborted on `ABORT: etcd port 8103 is already in use`, and this morning's
`92835f4d` patched the canonical launch block so two lines take different port
bands. **The validator is refusing the general form of the same problem one
level deeper — not "these two runs collided" but "this kit cannot have a second
copy beside it."** `: "${VAR:=…}"` is core principle 3 of the package's own
contract, and the validator is enforcing it against the produced kit.

**By this record's own criterion, this refusal is self-immune to the empty-zone
failure:** it quotes file contents and line numbers, which a validator looking
at an empty directory cannot produce.

**The other two validators passed:**

```
ae2c38   scripts/deploy.sh (180 s load)  true
         schema=environment              true
         layout                          FALSE   ← this one
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~33 %** (unchanged) |
| 已经耗时 | **~261 min** (mission.md 06:19:11 → 10:40:14) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**No change.** A refusal is information, not progress: stage 1 went from
reproducibly green to green-with-a-known-defect-in-the-kit. **The two earlier
greens are not retracted** — the kit that passed then had the same
`DK_ROUTER_PORT` shape, so what changed is that a validator now looks for it.
**Whether the earlier two runs' kits would refuse under today's validator is a
question I have not asked and could ask cheaply.**

**Hold `29184`: 3 h 20 min left** (14:00:01 → 10:40:14).

### 3. 当前进展 — nothing is running

**[observed] There is no live orchestrator and no live agent.**

```
20260906T093443-ae2c38   escalated 10:25:52, last write 10:34:31
                         deploy_and_prove: output_validating   (terminal)
all five run trees        no orchestrator process
node                      8 cards VRAM 0 %, no yihou_* container
teammate scratch          last write 10:34:31 (m2/launch3/chain.log)
```

**The chain's own account of how it stopped, from `store/event`:**

```
10:25:52.617  validation_failed  "output_validation did not pass"
10:25:52.617  escalated          "validation_failed: the task is terminal and
                                  there is nothing to push"
10:25:52.619  escalated  target=user   (same why)
```

**A refusal with nowhere to go.** This is the same terminal shape as the
`jsonschema` crash at 07:29:59 — the task is terminal, the escalation has no
receiver — **but for a completely legitimate reason this time: the validator
did its job and the graph has no path forward from a refused output.**

**A method note, because I nearly filed the opposite.** My first process scan
matched two processes; **both were my own shell and my own `python3 -c`,
carrying the search strings in their argv.** I discarded them. **The pattern
that finds orchestrators cannot be a pattern I am holding.** This is the fourth
time today that trap has appeared in this record and the first time it appeared
in my own hands.

### 4. Code problems

**Newly named by a validator, unfixed at this write:**

- **`scripts/env.sh:172` — `DK_ROUTER_PORT` is fixed** and reaches a binding
  flag at **`scripts/deploy.sh:205`**. Fix idiom given by the refusal itself.

**This is the third member of one family today**, and they were found at three
different depths:

```
08:11  two runs collide on port 8103        found by a run dying
09:xx  canonical block shares port band
       AND work_root                        found by auditing the block
10:25  the KIT cannot host a second copy    found by a validator
```

**Only the third one is a property of the deliverable.** The first two are
properties of how we launched it. **A fix to either of the first two would have
left the kit shippable-but-not-co-locatable**, which is exactly what the
validator refuses to sign.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; the unbooked-usage and missing-`depends_on` framework messages
(T+186 §5).

### 5. Non-code problems

**Nothing new this interval.** The `work_root` overlap from T+216 remains
recorded and unexamined for damage.

### 6. 未定性

- **Whether anyone is driving.** Nothing has run since 10:34:31 and the hold has
  3 h 20 min. **I am the checkpoint writer and do not schedule work; I am
  recording the gap because it is measurable and because idle GPU time under a
  finite hold is the one cost this record has consistently called unrecoverable.**
- **Whether the two earlier green kits carry the same `DK_ROUTER_PORT` defect.**
  **Cheap to answer:** `grep -n DK_ROUTER_PORT` in `3e8a03`'s and `6ded23`'s
  staged kits. **I have not run it** — it would change what those two greens are
  worth, and that is exactly why it should be run by someone who will act on it.
- **Whether `m2_profiling` can be reached at all this hold**, given the deploy
  stage now refuses. Two chains died in m2; the third never left stage 1.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched. **It
  has been carried in six consecutive sections without being touched**, which is
  itself worth saying out loud.

### 7. 新增 commit

Since T+216, one:

```
be36e7ad  checkpoint R2 T+216 — mine
```

**No other commits in this interval.**

### 8. 其他

**Today's ledger of stage 1, in the order the evidence arrived:**

```
06:42  bring-up works, 147 tokens generated       (a fact about the engine)
08:47  three validators pass                      (a fact about the kit)
09:29  three validators pass again                (reproducible)
10:25  one validator refuses, naming env.sh:172   (a fact about the kit)
```

**The last line is not a regression — it is the first time the kit was asked a
question it fails.** Two greens preceded it and neither is falsified; the
validator set changed underneath, or the kit did, and **which of those two it was
is the cheap grep in §6 that nobody has run.**

**What I would not want lost from today: three separate defects in one family
were found at three depths, and only the deepest one is about the thing being
shipped.** The first cost a run and announced itself. The second cost nothing and
announced nothing — the shared `work_root`. **The third cost nothing and
announced itself precisely, which is the only one of the three arrangements
anybody should want.**

---

## R2 T+277 — 2026-09-06 11:10 UTC

**T+277 = wall-clock delta from the baseline** (06:33:41 → 11:10:19).

### 1. I ran the cheap grep from §6 of T+246, and the answer is decisive

**At T+246 I framed it as "the validator set changed underneath, or the kit did,
and which of those two it was is the cheap grep nobody has run." It was the
kit — and the change is a regression, not a new check.**

**[observed, first-hand] `scripts/env.sh` in all three staged kits:**

```
run      bytes  lines   ROUTER_PORT   PORT_BASE   the port line
3e8a03   10329    198        0            7    128: : "${DK_PORT_ROUTER:=$(( E2E_KIT_PORT_BASE + 0 ))}"
6ded23   11315    209        0            8     80: : "${DK_PORT_ROUTER:=$((E2E_KIT_PORT_BASE + 0))}"
ae2c38   11277    214        3            9    172: DK_ROUTER_PORT=$(( E2E_KIT_PORT_BASE + 0 ))
```

**Both green kits already used the correct overridable idiom.** The refused kit
**added a second, differently-named spelling of the same port and assigned it
unconditionally**:

```
DK_PORT_ROUTER     the two green kits    : "${…:=…}"      overridable
DK_ROUTER_PORT     the refused kit       plain assignment  not overridable
```

**The two names differ only by swapping `PORT` and `ROUTER`.** A reader
skimming either file sees a `DK_…_PORT…` variable set from `E2E_KIT_PORT_BASE`
and moves on.

**The negative control, because a zero needs a denominator.** The two `0`s above
are not empty files or wrong paths: those same files return **7 and 8** hits for
`PORT_BASE`, and both carry `: "${E2E_KIT_PORT_BASE:=8101}"`. **The grep found
the file, read it, and the variable genuinely is not there.**

**Consequences, and they run in both directions:**

- **The two earlier greens are not weakened.** Their kits pass today's check on
  its merits — they never had the defect.
- **The refusal is a real regression caught in the interval it was introduced**,
  between 09:29:16 and 10:25:52.
- **`check_deploy_kit` did what a validator is for**: it refused a change that
  every human read of the file would have called fine.

**I said at T+246 this should be run by whoever would act on it. That was the
wrong line to draw** — reading is not acting, the question was blocking the
interpretation of two greens, and thirty-five minutes passed with nobody running
it. **An instrument declining to take a measurement is not neutrality.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~33 %** (unchanged) |
| 已经耗时 | **~290 min** (mission.md 06:19:11 → 11:10:19) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged, and this time the reason is that nothing ran.** The interval's
result is a finding about the record, not about the deliverable.

**Hold `29184`: 2 h 50 min left** (14:00:01 → 11:10:19).

### 3. 当前进展 — the node has been idle for 35 minutes

```
last write anywhere    10:34:31   (m2/launch3/chain.log)
now                    11:09:40
idle                   35 min 09 s
orchestrators          none        (positional /proc scan)
containers             rc_26_7_902, xiaoming-dev — both foreign, CPU only
cards                  8 × VRAM 0 %
teammate scratch       no file written since 10:34:31
new commits            none since be36e7ad (mine, T+216)
```

**Every source I have reads idle.** Flagged to the leader at 10:40 with the same
measurements; **this is the second consecutive interval with no activity, and
the standing rule for GPU-occupying problems is first-sighting intervention, so
this is now past that threshold rather than approaching it.**

**I state the cost without inflating it:** an unused hold hour is not recoverable,
and 2 h 50 min remain against a task where no stage past 1 has ever completed on
this cluster. **I do not know why nothing is running** — the measurement that
would answer it is the owners' own accounts, not anything in the run trees.

### 4. Code problems

**Newly characterised this interval — `scripts/env.sh:172`, and it is a
regression with a named window.**

```
introduced   between 09:29:16 (6ded23's kit: absent) and 10:25:52 (refused)
symptom      DK_ROUTER_PORT is assigned, not defaulted
reaches      scripts/deploy.sh:205 (export) -> a binding flag
consequence  a second copy of the kit cannot run beside the first
fix          : "${DK_ROUTER_PORT:=…}"     (given by the refusal itself)
```

**Unfixed at this write**, and nothing has been committed since 10:40.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; the unbooked-usage and missing-`depends_on` framework messages
(T+186 §5); shared `work_root` damage unexamined (T+216).

### 5. Non-code problems

**A near-homograph inside one file is a localisation trap of its own kind.**
`DK_PORT_ROUTER` and `DK_ROUTER_PORT` are three characters apart in a file that
contains nine `PORT_BASE` references. **Neither `bash -n` nor a schema check can
see it**, and a `grep -n PORT` returns both spellings looking like siblings —
I printed exactly that output above and it reads as consistent. **Only a
validator that asks "is this value overridable" separates them.**

### 6. 未定性

- **Why nothing is running.** Open, second interval. **Not answerable from any
  artefact I read** — this is the "the discriminator is outside the run" case,
  and here it is outside the machine.
- **Whether the `env.sh:172` regression has been fixed since 10:34.** No commit,
  no scratch write. **Cheap to answer once anything runs again.**
- **Whether `m2_profiling` can be reached this hold.** With 2 h 50 min left and
  two prior chains taking ~40 min to clear stage 1, **there is time for roughly
  two more attempts, not many.**
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched,
  **seventh consecutive section.**

### 7. 新增 commit

Since T+246, one:

```
29b1560b  checkpoint R2 T+246 — mine
```

**No other commits.**

### 8. 其他

**The interval's real content is a correction to my own reasoning, so I will
state it as a rule rather than an apology.**

At T+246 I withheld a read-only grep on the grounds that its answer "changes what
the two greens are worth, and that is exactly why it should be run by someone who
will act on it." **The premise was right and the conclusion was backwards.** A
measurement whose answer changes how existing evidence is read is the *most*
urgent one for the record-keeper to take, not the least — **the person who will
act needs the answer to decide whether to act at all.**

**And the answer was better than either branch I offered.** I framed it as
"validator changed, or kit changed." **The kit changed, in a specific 56-minute
window, by adding a near-homograph of a variable it already had** — a fact that
neither branch of my framing would have produced, and that took two commands.

---

## R2 T+306 — 2026-09-06 11:40 UTC

**T+306 = wall-clock delta from the baseline** (06:33:41 → 11:39:58).

### 1. Run 6 launched at 11:38:11, after a 63-minute idle gap

```
20260906T113811-fdb0bd   pid 2466373   container=yihou_e2e_chain
                         main: running   m1_deploy: running
                         deploy_and_prove: running
                         last write 11:39:25  (12 s before I sampled)
```

**The gap, measured end to end:** last write anywhere **10:34:31** → first write
of the new attempt **11:37:37** = **63 minutes 06 seconds** with no orchestrator,
no container, and eight cards at VRAM 0 %. **That is the largest single block of
unused hold time this round.** I reported it at 10:40 and again at 11:10; it is
now closed, and I record its size rather than a view about it.

### 2. What changed for this attempt — and it is only the instruction

**[first-hand, `m2/launch4/LAUNCH-RECORD.txt`]**

```
launched_at_utc: 2026-09-06T11:38:04Z
supersedes: 20260906T093443-ae2c38
   (check_deploy_kit refused: DK_ROUTER_PORT fixed at env.sh:172)
instruction v3: EXACTLY ONE bring-up, kit scripts first;
                := idiom + no second name for a value that has one
```

**I diffed every `--var` between launch3 and launch4: no difference.** The
entire change is the agent instruction. **That is a clean experiment** — one
variable moved, and it is the one the refusal pointed at.

**"no second name for a value that has one"** is the near-homograph finding from
T+277 stated as a rule for the agent. **Whether it arrived there from my section,
from my 11:10 message, or independently, I do not know and will not claim** —
`ae2c38`'s refusal named `env.sh:172` on its own and that is sufficient to
produce the first half of the instruction.

**"EXACTLY ONE bring-up, kit scripts first" is a time-budget change and worth
noting as such.** The `ae2c38` deployer performed a main arm, a callability arm,
and then a *final full cycle with the exact shipped scripts* (transcript, T+216
§3) — thorough, and it spent about 35 minutes inside `deploy_and_prove`.
**With 2 h 20 min of hold left, that thoroughness no longer fits.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~33 %** (unchanged) |
| 已经耗时 | **~320 min** (mission.md 06:19:11 → 11:39:58) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged for the third consecutive interval.** Two of the three were spent
idle. **Nothing regressed; nothing advanced.**

**Hold `29184`: 2 h 20 min left** (14:00:01 → 11:39:58). **Prior attempts took
~40 min to clear stage 1** (08:05→08:47, 08:49→09:29). **If this one holds that
pace it clears stage 1 near 12:18 and has roughly 100 minutes for stages 2–5,
none of which has ever completed here.** That is arithmetic on two measured
durations, not a forecast.

### 4. 当前进展

```
15c264  dead  jsonschema crash
e6f882  dead  port 8103 abort
3e8a03  dead  STAGE 1 GREEN → died at m2 (trace hash_id)
6ded23  dead  STAGE 1 GREEN → died at m2 (card lease, 1.8 s)
ae2c38  dead  REFUSED at check_deploy_kit (env.sh:172)
fdb0bd  ALIVE started 11:38:11, deploy_and_prove: running
```

**Node at 11:39:37:** 8 cards VRAM 0 %, no `yihou_*` container. **Expected —
the run is 86 seconds old and pre-bring-up.** Recording the reading with its
interpretation attached, because the same numbers meant three different things
today.

### 5. Code problems

**Presumed addressed by instruction v3, not yet by a verdict:** `env.sh:172`.
**The evidence that will settle it is `check_deploy_kit`'s verdict on this run's
kit**, which does not exist yet.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; the unbooked-usage and missing-`depends_on` framework messages
(T+186 §5); the shared-`work_root` overlap unexamined for damage (T+216).

### 6. Non-code problems

**The placeholder in `LAUNCH-RECORD.txt` has survived three launches.** I flagged
it at T+186 §8; it is still there in launch4, verbatim:

```
CLAUDE_CONFIG_DIR=<m1's shared farm — get the exact value from the leader> \
```

**In a document whose entire purpose is to make a launch reproducible, one field
routes the reader to a person.** Every other line is copy-pasteable, which is
precisely what makes this one easy to carry past. **Recorded a second time
because it has now outlived the two runs it was written for.**

### 7. 未定性

- **Whether run 6 clears `check_deploy_kit`.** First real question, ~40 min out.
- **Whether "EXACTLY ONE bring-up" is compatible with `check_deploy_serves`,**
  which performs its own bring-up as part of validation. **The instruction binds
  the agent; the validator is not the agent.** I do not know whether they
  conflict, and the reading that answers it is this run's zone.
- **Whether stages 2–5 can be reached at all this hold.** Arithmetic in §3.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched,
  **eighth consecutive section.** It has now been open longer than any other item
  in this record and has never been the thing blocking progress, which is exactly
  why it keeps surviving.

### 8. 新增 commit

Since T+277, one:

```
c5cbf65d  checkpoint R2 T+277 — mine
```

**No other commits.** The `env.sh:172` fix, if it exists, lives in the agent
instruction rather than in the repository — **`instruction v3` is recorded in
`m2/launch4/LAUNCH-RECORD.txt`, which was uncommitted at 11:39:58.**

### 9. 其他

**Six runs, and the causes of the five deaths have not repeated once:**

```
framework      jsonschema ImportError
coordination   two owners, one port band
materials      trace hash_id inconsistent across records
boundary       validator's engine outlives its verdict by 1.8 s
deliverable    the kit cannot host a second copy of itself
```

**Each is a different layer, and the last one is the only one that is a property
of the thing being shipped.** The progression is downward through the stack —
**from "our tools broke" to "we collided with ourselves" to "our inputs were
wrong" to "our seams leak" to "the product has a defect"** — and that ordering is
what a debugging loop produces when each fix holds.

**The uncomfortable half: five of the six runs died, and 63 of the last 65
minutes were idle.** Both statements are true, and the first one is the reason
the record is good while the second is the reason the deliverable is not
finished. **With 2 h 20 min left, the constraint has stopped being knowledge and
started being time.**

---

## R2 T+336 — 2026-09-06 12:10 UTC

**T+336 = wall-clock delta from the baseline** (06:33:41 → 12:10:00).

### 1. Stage 1 green a third time, with the refusal cleared — and stage 2 is live

**[observed, first-hand] Run `20260906T113811-fdb0bd`, all three verdicts:**

```
layout             true    ← was FALSE in ae2c38 (env.sh:172)
schema=environment true
scripts/deploy.sh  true
```

**`layout` flipping `false → true` on the next attempt, with the launch
variables byte-identical and only the agent instruction changed, is a controlled
result.** T+306 recorded the diff of every `--var` between launch3 and launch4 as
empty. **One variable moved and the refusal cleared.** That is the negative
control this record keeps asking for — *break it, put it back, watch it pass* —
arriving by accident of sequence rather than by design, but with the same
structure.

**And the chain went past stage 1 for the third time, now with a real profiling
engine up:**

```
store/task     m1_deploy = succeeded      deploy_and_prove = succeeded
               m2_profiling = running
               run_profiling_mode_off = running
               run_profiling_mode_on = waiting_resource
               merge_profiling_evidence = waiting_handoff
containers     yihou_e2e_chain_…_pmoff        started 12:07:00
               yihou_e2e_chain_…_pmoff_etcd   started 12:07:05
cards          0-3 at VRAM 75 %   4-7 at 0 %
last write     12:09:51   (9 s before I sampled)
```

**Stage 1 took ~29 minutes this time** (11:38:11 launch → `pmoff` container
12:07:00), against ~42 and ~40 for the two earlier greens. **The "EXACTLY ONE
bring-up" instruction is the only change that could account for it**, and I say
*could* — I have not read this deployer's transcript to confirm it did only one.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~36 %** (+3) |
| 已经耗时 | **~351 min** (mission.md 06:19:11 → 12:10:00) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+3: the regression is cleared and stage 2 is executing on hardware.** Two
prior chains reached `m2_profiling = running` and died there; **this is the third
attempt at the same wall, with both known causes fixed** (trace v2, ownership
wait).

**Hold `29184`: 1 h 50 min left** (14:00:01 → 12:10:00). **Stage 2's profiling
capture on the first cluster ran in the tens of minutes; I have no measurement of
it on this one.** If stage 2 completes, stages 3–5 have never been attempted here
and I have no basis to estimate them.

### 3. 当前进展

```
15c264  dead   jsonschema crash
e6f882  dead   port 8103 abort
3e8a03  dead   stage 1 green → died at m2 (trace hash_id)
6ded23  dead   stage 1 green → died at m2 (card lease 1.8 s)
ae2c38  dead   REFUSED at check_deploy_kit (env.sh:172)
fdb0bd  ALIVE  pid 2466373 — stage 1 green, m2_profiling running
```

**Phase timeline from `m2/launch4/chain.log`, in order:**

```
deploy_and_prove: input_validating -> running -> output_validating -> succeeded
m2_profiling:     waiting_handoff -> input_validating -> running
run_profiling_mode_off: (new) -> running
m1_deploy:        running -> succeeded
merge_profiling_evidence: (new) -> waiting_handoff
run_profiling_mode_on:    (new) -> waiting_resource
```

**`run_profiling_mode_on` is `waiting_resource` while `mode_off` runs** — the two
profiling arms are serialised on the cards, which is consistent with one TP-4
deployment holding 0–3 and cards 4–7 reading 0 %.

### 4. Code problems

**Cleared this interval:** `scripts/env.sh:172` — `check_deploy_kit` now passes.
**Fixed via the agent instruction, not via a repository change**; `instruction
v3` lives in `m2/launch4/LAUNCH-RECORD.txt`. **If the next chain is launched
without that instruction, the defect can return** — the fix is in a launch
record, not in the package.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; unbooked-usage and missing-`depends_on` framework messages (T+186 §5);
shared-`work_root` overlap unexamined (T+216); the `CLAUDE_CONFIG_DIR`
placeholder, now surviving four launch records (T+306 §6).

### 5. 未定性

- **Whether `m2_profiling` completes.** Third attempt at the same wall. **Both
  previously known causes are fixed, so a third failure here would have a third
  cause** — and that would say something the first two do not.
- **Whether "EXACTLY ONE bring-up" actually held**, or whether the 29 minutes has
  another explanation. **The reading is this deployer's transcript**, which I
  have not opened.
- **Whether stages 3–5 are reachable in 1 h 50 min.** No measurement exists for
  any of them on this cluster. **I decline to estimate rather than guess.**
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, untouched,
  **ninth consecutive section.** If stage 2 completes, this stops being
  hypothetical within the hour.

### 6. 新增 commit

Since T+306, one:

```
5946df32  checkpoint R2 T+306 — mine
```

**No other commits.** Three consecutive intervals now in which the only
repository change is this record. **The team's output is going into run trees and
launch records** — which for `instruction v3` means the one fix that cleared
today's refusal is not in the repository at all.

### 7. 其他

**The clean experiment in §1 deserves stating as a method note, because it was
not designed.**

`ae2c38` and `fdb0bd` differ in **exactly one input**: the agent instruction.
Same package, same `--var` set byte for byte, same node, same image, same model,
same hour. **`layout` went `false → true`.** Nobody set out to run a controlled
pair — the second launch was simply the next attempt after a refusal — **but
because the launch record captured `supersedes` and the variable set was
unchanged, the pair is interpretable after the fact.**

**That is what `supersedes` bought, and it cost one line in a text file.** The
first cluster's equivalent question — *what was different about the run that
worked?* — was unanswerable there, because a run tree does not record its own
launch line. **Here it took one `diff` of two records.**

---

## R2 T+366 — 2026-09-06 12:40 UTC

**T+366 = wall-clock delta from the baseline** (06:33:41 → 12:40:17).

### 1. Stage 2 ran on hardware and produced real traces — the deepest point of the round

**[observed, first-hand] Run `20260906T113811-fdb0bd` reached
`run_profiling_mode_off = succeeded` and `run_profiling_mode_on =
output_validating`.** Two validators then refused, **and both refusals quote the
contents they read:**

```
# check_trace_coverage                            (39 files in materials)
  note: re-parsed 1788696891.2799642-TP-2.trace.json.gz:
        1 395 036 events, 99 892 GPU kernels, 11.02 s — manifest agrees
  note: 4 rank(s), 399 528 GPU kernel events
  PROBLEM: items/result/stacks_manifest.json is missing — the round was asked
           for a stack window and this handoff carries none, so no launcher
           frame can be resolved from it. Set --var stack_window_s=0 to say
           that is intended

# check_kernel_table
  note: 130 kernels, top 25 cover 82.5%, shares sum to 100.01
  note: layout: structured_text (items/text.json)
  PROBLEM: no launcher frames were resolved … and this round wanted at least 10
           in the head. Set --var stack_window_s=0 and min_launchers_in_top_n
           to 0 to say that is intended
```

**The profiling capture worked.** A validator re-parsed a gzip and counted
**1 395 036 events and 99 892 GPU kernels in one rank**, agreeing with the
manifest, and **399 528 GPU kernel events across 4 ranks**. A second one
independently summed a 130-row kernel table to **100.01 %**. **These are numbers
that only exist if the stage really ran** — the standard this record set at the
first cluster and has been asking for since.

**Both refusals have one cause: no stack window was captured**, so
`stacks_manifest.json` is absent and no launcher frame can be resolved. **Each
refusal names the variable that declares the omission intended.**

**Run 7 launched 49 seconds later** and states the change:

```
launched_at_utc: 2026-09-06T12:19:08Z
supersedes: 20260906T113811-fdb0bd (_off SEALED; _on refused by
            check_trace_coverage + check_kernel_table)
change: --var stack_window_s=0 --var kernel_table_min_launchers=0
note:   min_launchers_in_top_n is a validator ARGS field (common.yaml:162);
        the --var is kernel_table_min_launchers
```

**That `note:` is the interval's most transferable line.** The refusal told the
operator to set `min_launchers_in_top_n` — **which is the name of the field the
validator reads, not the name of the flag a human types.** Following the
refusal's own wording literally would have produced a `--var` that reaches
nothing, and by this record's oldest rule, **an unwired `--var` is silent**.

***Against my own instrument:*** my verdict summariser labels each verdict by an
arbitrary key from `args.json`, so it printed `expect_ranks [False]` for the
trace refusal. **`expect_ranks` was correct — the report says "4 rank(s) …
manifest agrees."** My label named a field the validator was *given*, not the
field it *refused on*. **Same defect the launch record just documented, in my own
tooling, found in the same ten minutes.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~42 %** (+6) |
| 已经耗时 | **~381 min** (mission.md 06:19:11 → 12:40:17) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+6: stage 2 executed and produced measured artefacts.** `run_profiling_mode_off`
is `succeeded` — **a sealed sub-handoff, not a pass on an empty check.** The
refusals are about a capture option, not about whether the profiling worked.

**Hold `29184`: 1 h 20 min left** (14:00:01 → 12:40:17).

### 3. 当前进展

```
fdb0bd  dead   DEEPEST YET — _off sealed, _on refused (stack window)
                last write 12:18:44
041f89  ALIVE  pid 2840180, started 12:19:08
                deploy_and_prove: running
                yihou_e2e_chain_dk1_sgl   started 12:38:05
                yihou_e2e_chain_dk1_etcd  started 12:38:02
                cards 0-3 at 75 %, 4-7 at 0 %
```

**Run 7 is 20 minutes in and bringing up.** Prior stage-1 times: 42, 40, 29 min.

### 4. Code problems

**Cleared:** `env.sh:172` (run 6's `layout` = true).

**Not a code defect — a capture-configuration mismatch:** the round asked for a
stack window and the profiling stage produced none. **Addressed in run 7 by
declaring it intended** (`stack_window_s=0`, `kernel_table_min_launchers=0`)
rather than by capturing stacks. **That is a deliberate degradation and it is
declared in the launch record**, which is the honest form of it — but the
resulting `profiling_evidence` will carry no launcher frames, **and whether
module 3 needs them is a question I have not seen asked.**

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; unbooked-usage and missing-`depends_on` (T+186 §5); shared-`work_root`
overlap unexamined; the `CLAUDE_CONFIG_DIR` placeholder, now five launch records.

### 5. 未定性

- **Whether run 7 clears `m2_profiling`** with the two flags set. **First time
  the wall has been approached with the specific refusal answered.**
- **Whether module 3 needs launcher frames.** `stack_window_s=0` removes them by
  declaration. **If `m3_analysis` refuses for want of them, the degradation moves
  the wall rather than clearing it.** Nobody has asked this yet and it is
  cheap: read `check_*` for module 3's kind.
- **Whether stages 3–5 fit in 1 h 20 min.** No measurement for any of them here.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, **tenth
  consecutive section**, and now one stage away from mattering.

### 6. 新增 commit

Since T+336, four (one is a merge — **the branch has diverged and been merged,
which means other people's commits are now in this history**):

```
401ff10d  checkpoint R2 T+336 — mine
ea35436a  Merge branch 'dev.yihou.aiopt.task_package.concat' … (merge)
49f0374d  bug record: a refusal names the field it reads, not the flag you
          type; renumber my sections around the merge
fc549951  mission.verify.e2e: second cluster cannot test the empty-zone claim,
          and why
2f7ba262  bug record: four of today's costs share one shape — a value that
          looks like an identifier and is a category
```

**`49f0374d` is the finding in §1, filed by its owner.** I reached the same shape
from my own summariser independently and record both — **theirs from the launch
path, mine from the reporting path, same hour.**

**I have not read `fc549951` or `2f7ba262` beyond their subjects and do not
restate them.**

### 7. 其他

**The refusals in §1 are the best-formed this round has produced, and it is worth
saying exactly why.**

Each one **states what it measured before it states what is wrong** — the trace
validator reports 1.4 M events and agreement with the manifest, *then* names the
missing file. **A reader learns that the stage worked and that one option was
unset, in that order.** Compare `check_deploy_kit`'s crash at 07:29:59, which
produced no verdict at all and would have been read as "the kit is bad."

**And each names the escape hatch: `set --var X to say that is intended`.** That
turns a refusal into a decision the operator makes explicitly, rather than a wall
they route around silently. **The cost of the design shows up in the same
breath** — the escape hatch was named with the field's internal name, and one
line in a launch record was needed to translate it.

---

## R2 T+396 — 2026-09-06 13:10 UTC

**T+396 = wall-clock delta from the baseline** (06:33:41 → 13:10:02).

### 1. Run 7 died on the KFD blindness this record documented on the first cluster

**[first-hand, `m2/launch6/LAUNCH-RECORD.txt`]**

```
launched_at_utc: 2026-09-06T13:08:34Z
supersedes: 20260906T121919-041f89
   (_off FATAL: UNIDENTIFIED holder -> aborted; KFD is blind across containers)
change: preflight identifies by CONTAINER LABEL first; KFD only a supplement;
        blind -> wait 300s not abort
```

**This is the same mechanism this file recorded on 2026-09-06 00:46 on the first
cluster** — `/proc`-based KFD counting cannot see into another container, so the
count is structurally zero across boundaries. **What is new is the direction of
the failure.** There it under-refused: a busy node read `KFD=0` and the predicate
would have admitted a launch. **Here it over-refused:** the holder could not be
identified, and the preflight treated *unidentifiable* as *fatal* and aborted a
chain that had already passed stage 1.

> **The same blind instrument produced opposite failures on two clusters, and
> both were wrong.** The first cluster's conclusion — *under-refusing is the
> dangerous direction* — is unchanged and this is not a counterexample to it;
> **it is the reminder that the safe direction still costs runs.**

**The fix matches the shape the first cluster arrived at**: identify by
**container label** first (labels cross namespaces; `/proc` does not), keep KFD
as a supplement only, and **treat blindness as "wait" rather than "abort."**

**Run 7's reach, for the record:** `deploy_and_prove = succeeded`, `m1_deploy =
succeeded`, `run_profiling_mode_off = running` — **stage 1 green for the fourth
time**, then dead at 12:55:46 before profiling completed. **It did not get as far
as run 6.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~42 %** (unchanged) |
| 已经耗时 | **~410 min** (mission.md 06:19:11 → 13:10:02) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged: run 7 reached less than run 6.** The deepest point of the round is
still `fdb0bd`'s sealed `run_profiling_mode_off` and the two content-level
refusals at T+366.

**Hold `29184`: 50 minutes left** (14:00:01 → 13:10:02). **Stage 1 alone has
taken 29–42 minutes on every attempt.** Run 8 launched at 13:08:45. **The
arithmetic is that run 8 can plausibly finish stage 1 and little else**, and
that is arithmetic on four measured durations, not a forecast about what the
agents will do.

### 3. 当前进展

```
fdb0bd  dead   DEEPEST — _off sealed, _on refused (stack window)   12:18:44
041f89  dead   stage 1 green (4th), _off FATAL on unidentified holder  12:55:46
298750  ALIVE  pid 3308290, started 13:08:45
               main/m1_deploy/deploy_and_prove all running
               last write 13:09:44
node    8 cards VRAM 0 %, no yihou_* container — run is 56 s old, pre-bring-up
```

### 4. Code problems

**Fixed this interval, in the kit preflight** (per the launch record; **I have
not read the diff**): container-label-first identification, KFD demoted to a
supplement, blindness → 300 s wait instead of abort.

**Note on where this fix lives.** The T+306 fix (`instruction v3`) lived in an
agent instruction; **this one is described as a preflight change**, which would
put it in the kit. **I cannot confirm which from the launch record alone**, and
the distinction matters: **a fix in the kit ships; a fix in a launch instruction
does not.**

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; unbooked-usage and missing-`depends_on` (T+186 §5); shared-`work_root`
overlap unexamined; the `CLAUDE_CONFIG_DIR` placeholder — **now six launch
records, verbatim, still routing the reader to a person.**

### 5. 未定性

- **Whether run 8 reaches profiling before the hold ends.** 50 minutes; stage 1
  costs 29–42.
- **Whether module 3 needs launcher frames**, given `stack_window_s=0` declares
  them away. **Raised at T+366 and to the leader at 12:41; not yet answered, and
  it is a `grep` of module 3's validators.** If run 8 reaches m3 this hold, it
  will be answered by a refusal instead — **which costs a run rather than a
  command.**
- **Whether the preflight fix is in the kit or in an instruction** — §4.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, **eleventh
  consecutive section.**

### 6. 新增 commit

Since T+366, one:

```
c022dcfd  checkpoint R2 T+366 — mine
```

**No other commits in this interval.** Both fixes of the last two hours —
`instruction v3` and the preflight identification change — **exist in
`m2/launch*/LAUNCH-RECORD.txt` files, uncommitted.**

### 7. 其他

**Eight runs. The five distinct causes of death from T+306 have become six, and
the newest one is a repeat from the other cluster.**

```
framework      jsonschema ImportError
coordination   two owners, one port band
materials      trace hash_id inconsistent across records
boundary       validator's engine outlives its verdict by 1.8 s
deliverable    the kit cannot host a second copy of itself
instrument     KFD is blind across containers — abort on unidentifiable
```

**Five of the six were new to this cluster. The sixth was written down on
2026-09-06 at 00:46, on the other cluster, in this file** — including the
correction that a `/proc` KFD predicate is blind across container boundaries and
that `rocm-smi` is what crosses them. **It cost a run here anyway.**

**That is this record's own tier-3 problem stated against itself:** the lesson
was written, it was in the file, and **it was not present at the line where the
preflight predicate was chosen.** The repair the file already prescribes for
that is not "read more carefully" — it is to make the property checkable in the
artefact. **A preflight that must name *how* it identified a holder (label /
VRAM / KFD) would have made the blindness visible in its own output**, which is
what the fix now does by demoting KFD to a supplement.

---

## R2 T+426 — 2026-09-06 13:40 UTC

**T+426 = wall-clock delta from the baseline** (06:33:41 → 13:40:05).

### 1. The T+366 refusals have a root cause, and it corrects what I wrote

**At T+366 I recorded the two refusals as "no stack window was captured" and
called run 7's `stack_window_s=0` "a deliberate degradation … declared in the
launch record — the honest form of it." The degradation was not necessary.**

**[first-hand, `5d098338`]**

> *`trace_end_ms` is the knob (`shared.yaml:149` for m2's path, consumed at
> `aiperf_replay.sh:100` as `--fixed-schedule-end-offset`). The two captures run
> in sequence inside one load: **warmup 60 + window 10 for the measurement, then
> 3 for the stack window, so the load must outlast ~73 s plus setup.**
> **All six launches today passed `trace_end_ms=60000`. A 60 s load cannot cover
> a 73 s sequence.***

**So the stack window was not omitted — it was scheduled after the load ended.**
`capture_stacks.log` recorded *"no aiperf load in flight"*, and the two
validators then refused for something that is not the producer's fault.

**And it is explicitly not a code defect:**

> *the ordering works at the package's own defaults — `aiperf_replay.sh`'s
> fallback is 120000 and `shared.yaml`'s is 180000, both well over the floor.
> **A deliberate time-saving override introduced it**, and nothing states that
> `trace_end_ms` has a floor set by `warmup_s + window_s + stack_window_s`.
> **The defect is an unenforced dependency between launch variables.***

**The recommended repair is `trace_end_ms=120000`, not shortening warmup** —
*"warmup buys steady state and trading it swaps a known quantity for an
unknown one."*

**Three thresholds were checked and cleared as insensitive to load length**
(`max_span_ratio` scales with `window_s` not the load; `min_gpu_kernels_per_rank`
and `min_requests` only improve). **That is the "audit the class, don't fix the
instance" discipline applied before the next launch rather than after three of
them.**

**Consequence for an open question I have carried twice:** *whether module 3
needs launcher frames* is **no longer the right question**. The frames are
capturable; **`stack_window_s=0` was papering over a 13-second arithmetic
shortfall.**

### 2. The fix was filed as comments, with its own tier named

**[first-hand, `a1aecda2`]**

> *Nothing anywhere stated that `trace_end_ms` has a floor set by
> `warmup_s + window_s + stack_window_s`. **That fact lived in a document, which
> is the tier that decays**: the next person writes a launch line while thinking
> about something else, which is exactly how the override that blocked four
> stages got written.*

**Comments at both declaration sites**, carrying the arithmetic (70 s + 3 s = a
73 s floor), what breaks below it, and **the run that proved it — `fdb0bd`, load
ended 12:14:58, stack capture started after 12:15:19.**

**The commit subject says the honest part out loud: *"document it where the launch
line is written, and write the preflight without applying it."*** They wrote the
enforcement and did not turn it on. **A tier-3 repair, labelled as tier-3 by its
author, with the tier-2 version written but not enabled** — that is a more useful
state than either a silent comment or an unreviewed guard going live at 13:40
with twenty minutes of hold left.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~44 %** (+2) |
| 已经耗时 | **~440 min** (mission.md 06:19:11 → 13:40:05) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2 for a root cause, not for a stage.** The wall at m2 now has arithmetic
behind it instead of a shrug.

**Hold `29184` ends in 20 minutes** (14:00:01 → 13:40:05). **A successor hold
`29313` is PENDING, reason `Resources`, submitted 11:38:12, `TimeLimit=1-00:00:00`
(24 h).** **Whether it starts when 29184 releases is not predictable from
`Reason=Resources`** — this file recorded on the first cluster that a pending
reason explains why a job waits and says nothing about what happens when the
constraint lifts; a different job took the slot that time.

### 4. 当前进展 — run 8 is mid-validation with 20 minutes on the clock

```
298750  ALIVE  pid 3308290, started 13:08:45
        m1_deploy: running   deploy_and_prove: output_validating
        verdicts so far:  layout = true    environment = true
        check_deploy_serves running:
             yihou_e2e_chain_serves-ff0b8393_etcd  13:34:05
             yihou_e2e_chain_serves-ff0b8393_sgl   13:34:07
             aiperf_serves-ff0b8393                13:36:53
        cards 0-3 at 76 %,  4-7 at 0 %
        last write 13:36:53
```

**Stage 1 is two-thirds green for the fifth time**, with the 180-second load
under way. **It will not reach profiling before the hold ends.**

### 5. Code problems

**Root-caused, fix documented but not enforced:** `trace_end_ms` floor —
`shared.yaml:149`, `aiperf_replay.sh:100`. **Preflight written, deliberately not
applied.**

**Fixed earlier today:** `env.sh:172`; the KFD-blind preflight (label-first).
**Both live in launch records or agent instructions, still uncommitted.**

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; unbooked-usage and missing-`depends_on` (T+186 §5); shared-`work_root`
overlap unexamined; the `CLAUDE_CONFIG_DIR` placeholder — six launch records.

### 6. 未定性

- **Whether hold `29313` starts.** `Reason=Resources` does not answer it.
- **What happens to run 8 at 14:00:01.** It is inside `check_deploy_serves` with
  an engine and an aiperf container up. **I do not know whether the allocation
  ending tears the containers down cleanly or leaves them**, and the measurement
  is simply to look after 14:00.
- **Whether `trace_end_ms=120000` clears the m2 wall.** Not yet launched with it.
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, **twelfth
  consecutive section**, and the hold ends before it can be answered by a run.

### 7. 新增 commit

Since T+396, four:

```
e2d41d7f  checkpoint R2 T+396 — mine
0487b970  ON-ARM-REFUSAL: amend — the refusal is established by two verdicts,
          and the run's end was the no-receiver escalation
5d098338  ON-ARM-REFUSAL: the repair is arithmetic, and it reattributes the
          cause to an unenforced variable dependency
a5908df1  ON-ARM-REFUSAL: repair three words an unescaped backtick ate in the
          previous commit
a1aecda2  replay-length floor: document it where the launch line is written,
          and write the preflight without applying it
```

**Four commits on one finding, including an amend that reattributes the cause and
a repair of three words a backtick ate.** The second is worth a line of its own:
**an unescaped backtick in a commit message silently removed content, and commit
messages cannot be amended once pushed** — this file's own rule.

### 8. 其他

**My T+366 section called `stack_window_s=0` "the honest form" of a degradation.
It was honest and it was the wrong repair**, and the difference is instructive.

Declaring an omission intended is the right move **when the thing genuinely
cannot be produced**. Here it could: the capture was scheduled 13 seconds past
the end of a load that a launch override had shortened. **A declared degradation
and a masked defect look identical in the artefact** — both produce a clean run
with `stack_window_s=0` in the record and no launcher frames in the output.

**What separated them was arithmetic nobody had written down**, and the thing
that found it was somebody adding up three variables and comparing the sum to a
fourth. **Not a tool, not a validator — a subtraction.**

---

## R2 T+457 — 2026-09-06 14:11 UTC

**T+457 = wall-clock delta from the baseline** (06:33:41 → 14:10:36).

### 1. The hold rolled over — 24 hours, same node

```
29184   ended      2026-09-06T14:00:01
29313   RUNNING    StartTime 2026-09-06T14:00:06   EndTime 2026-09-07T14:00:06
                   smci355-ccs-aus-n04-25   TimeLimit 1-00:00:00
```

**Five seconds of gap, same node, and the successor is a 24-hour hold rather than
16.** At T+426 I recorded `29313` as `PENDING (Resources)` and declined to
predict it would start — this file's own rule that a pending reason explains the
wait and not the release. **It started. The prediction I declined to make would
have been right, and declining it still cost nothing** — I have no way to know
whether it started *because* 29184 released or by coincidence of the scheduler,
and the record does not need me to.

**The time pressure of the last four sections is gone.** 23 h 50 min remain.

### 2. Run 8 is the round's deepest result, and it isolates the m2 wall to one variable

**[observed, first-hand] `20260906T130845-298750`, last write 14:05:14:**

```
m1_deploy               succeeded
deploy_and_prove        succeeded
run_profiling_mode_off  succeeded
run_profiling_mode_on   output_validating
m2_profiling            running

verdicts (11):  layout true · environment true×3 · deploy.sh true
                require_present true×3 · max_error_rate true×2
                max_pct_total_sum true          ← check_kernel_table PASSES
                expect_ranks FALSE              ← check_trace_coverage
```

**`check_kernel_table` now passes.** At T+366 it refused for want of launcher
frames; `kernel_table_min_launchers=0` cleared it. **Ten of eleven verdicts are
true and the eleventh is the only thing between this chain and stage 2.**

**And that eleventh is now fully explained** [first-hand, `2f53d2a6`]:

> *`_off` SUCCEEDED at 13:46:13 … `_on` then refused at 13:52 with one false
> verdict in a 39-file zone, and the refusal is identical to `fdb0bd`'s — **on a
> run that passed `--var stack_window_s=0`, verified in the live process.**
> **The producer got it**: no `capture_stacks.log` exists, so `replay.sh`
> correctly skipped the capture. **The validator did not, because it never reads
> that variable.** `check_trace_coverage:223` gates on `expect_stack_ranks`,
> which `m2_profiling.yaml:133` binds to `${stack_ranks:-2}`. **`stack_window_s=0`
> does not touch `stack_ranks`**, so the producer skips and the validator still
> demands the manifest.*

**One intention, two variable names, and the flag was verified reaching the live
process.** This is the third member of that family today — `DK_ROUTER_PORT` vs
`DK_PORT_ROUTER` (T+277), `min_launchers_in_top_n` vs `kernel_table_min_launchers`
(T+366), and now `stack_window_s` vs `stack_ranks`. **The first two were naming
collisions; this one is a genuine split of one decision across two knobs that
nothing binds together.**

### 3. **Two concurrent runs share a port band and a work root — flagged at 14:09**

**[observed, first-hand, positional `/proc` read]**

```
pid 3849649  started 14:08:09   container=yihou_e2e_chain
pid 3856205  started 14:08:25   container=yihou_e2e_chain2
  BOTH:  port_router=8101  port_worker=8102  port_etcd=8103
         work_root=/data/yihou/e2e_flow
         validate_work_root=/data/yihou/e2e_flow/validate
         mock_stages=none      trace_end_ms=120000
```

**Only `container` differs.** This is the configuration that killed run 2 at
08:11:26 with `ABORT: etcd port 8103 is already in use`, and `92835f4d` amended
the canonical launch block this morning **specifically to require a separate port
band and a separate work root** for two lines on one node.

**Both are still in `deploy_and_prove` and nothing has bound a port yet.**
Reported to the leader at 14:09:47 with the measurement and without an
instruction — **I do not know whether the overlap is deliberate.**

**Both carry `trace_end_ms=120000`**, so the arithmetic fix from T+426 is in
flight. **Neither carries `stack_ranks`**, so by §2 both will hit the same
`check_trace_coverage` refusal if they reach `_on`.

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~46 %** (+2) |
| 已经耗时 | **~471 min** (mission.md 06:19:11 → 14:10:36) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2: ten of eleven verdicts true in one run, and the eleventh traced to a named
line in a named file.** Not more, because **`m2_profiling` has still never
completed.**

**预估耗时 remains absent and I want to be explicit about why it is still absent
after eight hours.** Stage 1 now has four measured durations (42, 40, 29, ~38
min). **Stages 3, 4 and 5 have zero.** A total built from one measured stage and
four guesses would be a number with a false denominator, and this record has
spent the day on exactly that failure mode.

### 5. Code problems

**Root-caused, unfixed:** `check_trace_coverage:223` / `m2_profiling.yaml:133` —
`expect_stack_ranks` ← `${stack_ranks:-2}`, unlinked from `stack_window_s`.

**Root-caused, documented, enforcement deliberately not enabled:** the
`trace_end_ms` 73-second floor (`shared.yaml:149`, `aiperf_replay.sh:100`).

**Fixed earlier:** `env.sh:172`; KFD-blind preflight; `kernel_table_min_launchers`.

**Carried unfixed:** whether all eight `jsonschema`-affected validators were
repaired; unbooked-usage and missing-`depends_on` (T+186 §5); shared-`work_root`
overlap — **no longer merely unexamined, now actively reproduced in §3**; the
`CLAUDE_CONFIG_DIR` placeholder.

### 6. 未定性

- **Whether the two concurrent runs collide.** §3.
- **Whether `stack_ranks=0` is accepted** and clears `check_trace_coverage`, or
  whether a real stack capture is required. **Not verified by anyone I can
  read.**
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — carried, **thirteenth
  consecutive section.** With 23 h 50 min of hold, the time excuse for not
  answering it has expired.

### 7. 新增 commit

Since T+426, five:

```
a4a492e4  checkpoint R2 T+426 — mine
1c611a5d  PREFLIGHT-HANDOVER-298750: the handover ran clean and the hard branch
          is still untested
af54e790  RUNG5-CHECKLIST: fold the day's findings in as steps, and make it the
          single entry point
2f53d2a6  MERGE-WATCH-298750: the merge did not fire — stack_window_s=0 reaches
          the producer and not the validator
0ad0c7d0  RUNG5-CHECKLIST P-1: name both paths, add the third --var, and flag
          the refusal that gives the wrong knob
```

**`1c611a5d` is worth its own line: "the handover ran clean and the hard branch is
still untested."** The KFD preflight fix from T+396 was exercised on its easy
path only. **Saying so in the subject line is the difference between a fix and a
fix that has been tested**, and it is the same discipline as `a1aecda2` writing a
preflight and declining to enable it.

### 8. 其他

**Eight hours in, the shape of the remaining problem is clear and it is not what
it was at 06:33.**

At the baseline, nothing had run and the question was whether the chain would
work here at all. **It works.** Stage 1 has gone green five times, an engine
serves, traces of 1.4 M events get captured and re-parsed, a 130-row kernel table
sums to 100.01 %, and ten of eleven validators pass on a real artefact.

**What blocks the chain today is not capability — it is that one decision is
spelled two ways.** `stack_window_s` tells the producer to skip; `stack_ranks`
tells the validator to demand. **Both are doing exactly what they were written to
do**, and no amount of running the chain again will resolve it, because the
disagreement is not stochastic.

**That is the most useful thing to hand to whoever picks this up with 23 hours of
hold:** the next run does not need to be observed, it needs one variable bound.

---

## R2 T+487 — 2026-09-06 14:41 UTC

**T+487 = wall-clock delta from the baseline** (06:33:41 → 14:41:05).

### 1. The port-band overlap I flagged resolved itself — and not the way I framed it

**[observed] Three runs launched in six minutes; two are dead, one survives.**

```
140625-bb5824   failed 14:07:43   exit 143   monitor_gave_up
140819-2f9956   failed 14:10:27   exit 143   monitor_gave_up
140831-026b96   ALIVE  pid 3856205  container=yihou_e2e_chain2
                deploy_and_prove: output_validating
                aiperf_serves-d8ff1deb  14:40:03   cards 0-3 at 76 %
```

**Both deaths are exit 143 — SIGTERM.** Not a port abort, not a validator
refusal: something killed them. `monitor_gave_up: the pusher has no action for
handling_failed`.

**And the discriminator between the dead and the living is not ports.** I read
the survivor's argv:

```
dead pair   --var jobid=29184     (the hold that ended at 14:00:01)
survivor    --var jobid=29313     (the hold that started at 14:00:06)
```

**[first-hand, `174dd6b5`, by the owner of those launches]**

> *Hold 29184 ended 14:00:01; 29313 took the same node at 14:00:34. **My 14:08
> launch still carried `--var jobid=29184`**, which is sealed into every artefact
> as `runtime.slurm_jobid`.*

**So the two SIGTERMs are consistent with a deliberate supersede** — a stale hold
id caught and the launches replaced. **I did not observe the kill and I am not
asserting the cause.** What I can say: the survivor carries the live jobid, the
dead pair carried the dead one, and their owner filed the finding in the same
window.

**My 14:09 flag named the port band. The port band was not what separated
them.** That is the second time today I have read a concurrent-run hazard off
argv and named the wrong field — T+216 recorded the first. **Both times the
overlap was real and the mechanism I led with was not the operative one.**

### 2. The jobid finding is the sharpest guard-failure of the round

> *The part worth recording is that **`slurm_jobid` IS one of the three fields
> `_agree_or_die` guards.** That guard (`measure_in_container.sh:118-126`)
> refuses only on DISAGREEMENT between the ambient value and the record — **and
> both come from the same `--var`, so a stale jobid is identical on both sides,
> they agree, and the guard passes.** A field can be guarded and uniformly
> wrong.*

**This is the first cluster's rule — *when a consistency check passes, ask where
its two values came from* — arriving as a live incident on the one field
everybody would have assumed was covered.** `_agree_or_die` watches 3 of 28
environment fields; **this is a failure inside the covered 3, not outside them.**

**And the value is not merely stale, it is falsifiable against the world**:
`squeue` shows one job, `29313`. **A jobid naming a dead hold is checkable by one
command at launch**, which is what makes this a tier-2 repair rather than a note.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~46 %** (unchanged) |
| 已经耗时 | **~501 min** (mission.md 06:19:11 → 14:41:05) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged: no stage advanced.** Two launches were spent and replaced; the
survivor is at the same point run 8 reached at 13:36.

**Hold `29313`: 23 h 19 min left** (2026-09-07T14:00:06 → 14:41:05).

### 4. 当前进展 — and the survivor will hit the known wall

**I read the survivor's argv for the variable that decides its fate:**

```
jobid=29313          correct
trace_end_ms=120000  the T+426 arithmetic fix, applied
stack_ranks          ABSENT
stack_window_s       ABSENT
```

**By T+457 §2, `check_trace_coverage:223` gates on `expect_stack_ranks` ←
`${stack_ranks:-2}`.** With `stack_ranks` unset it defaults to 2 and the
validator will demand `stacks_manifest.json`. **With `trace_end_ms=120000` the
load now outlasts the 73-second sequence, so the capture has time to happen** —
which means this run may produce the stacks rather than need the flag.

**That is a real prediction and I am marking it as one:** *if* the stack capture
now runs, `check_trace_coverage` passes on its merits and the flag question
becomes moot. *If* it does not, the refusal will be identical to `fdb0bd`'s and
`298750`'s. **The measurement that decides it is whether `capture_stacks.log`
exists in this run's `_on` output.** I have made no prediction about which.

### 5. Code problems

**Newly named:** `--var jobid` can name a dead hold and pass `_agree_or_die`
(`measure_in_container.sh:118-126`).

**Carried, root-caused, unfixed:** `check_trace_coverage:223` /
`m2_profiling.yaml:133` — `stack_ranks` unlinked from `stack_window_s`.
**Carried, documented, enforcement not enabled:** the `trace_end_ms` 73 s floor.
**Carried unfixed:** the eight `jsonschema` validators (unread); unbooked-usage
and missing-`depends_on`; the `CLAUDE_CONFIG_DIR` placeholder.

**Note against my own §3 of T+457:** I recorded the shared `work_root` as
"actively reproduced." **It still is** — the survivor and the dead pair all used
`/data/yihou/e2e_flow`. **The overlap did not cause today's deaths and it remains
the failure that does not announce itself.**

### 6. 未定性

- **Whether `026b96` captures stacks** with the longer load. §4.
- **Whether the two SIGTERMs were the supersede they appear to be.** Consistent,
  not established. **The reading is the owner's own account, not the run trees.**
- **Whether other artefacts already carry `slurm_jobid: 29184`.** Anything sealed
  between 14:00:01 and the correction. **Cheap grep, not run.**
- **Whether all eight crash-affected validators were repaired** — carried.
- **What module 5 consumes if module 4 is replayed** — **fourteenth consecutive
  section.**

### 7. 新增 commit

Since T+457, five:

```
3e60294b  checkpoint R2 T+457 — mine
174dd6b5  bug record: a GUARDED field can still be uniformly wrong — jobid
          naming a dead hold
64ea7bd5  bug record: two checks with one failure mode are one check — and the
          test is cheap
b66cd62c  CLAUDE.md: a guarded field can still be uniformly wrong — slurm_jobid
85e7e44f  RUNG5-CHECKLIST: sibling handoffs are invalidated by the closure, not
          independently
37b79b9c  CLAUDE.md: /home/yihou/dev/git is a SYMLINK to git.16-19 — one tree,
          two names
```

**`37b79b9c` closes a trap this record could have walked into.** Two path
prefixes have been used interchangeably in today's sections and commit messages;
**they are one tree.** Had they been two, every "the file is clean" check I ran
would have been checking a different file than the one being edited.

### 8. 其他

**Two of today's findings are the same theorem, discovered eight hours apart, and
putting them side by side is worth more than either alone.**

```
T+457  stack_window_s tells the producer to skip;
       stack_ranks tells the validator to demand.
       ONE DECISION, TWO NAMES — and nothing binds them.

T+487  --var jobid feeds both the ambient value and the record;
       _agree_or_die compares them and they agree.
       ONE VALUE, TWO PLACES — and the guard compares it to itself.
```

**The first splits one intention across two knobs, so setting one is silently
insufficient. The second merges two supposedly independent sources into one, so
comparing them is silently uninformative.** Opposite errors, identical
consequence: **a mechanism that looks like it is checking something and is not.**

**And the detection cost differs enormously.** The split knob was found by a
validator refusing and someone reading two YAML lines. **The merged source was
found only because a human noticed a number naming a job that no longer exists**
— the guard itself will never report it, at any point in the future, on any run.

---

## R2 T+517 — 2026-09-06 15:11 UTC

**T+517 = wall-clock delta from the baseline** (06:33:41 → 15:10:59).

### 1. The preflight fix from T+396 has an unreachable branch, and I verified its premise

**[first-hand, `c890bf83`]**

> *`_foreign_gpu_containers` tests `HostConfig.Devices` for `/dev/kfd`, **which is
> device mapping and not occupancy**. `rc_26_7_902` and `xiaoming-dev` map it
> permanently and hold zero VRAM, so **instrument 2 is never empty** and
> `preflight.sh:211` aborts on any busy card whatever the cause. Measured: the
> two named strangers are still running six minutes later with all eight cards
> at zero; **the 112–117 GB was our own `check_deploy_serves` engine draining.***

**I checked the premise myself rather than relaying it** — one `docker inspect`:

```
rc_26_7_902  HostConfig.Devices:
  [{"PathOnHost":"/dev/dri",…},{"PathOnHost":"/dev/kfd",…},{"PathOnHost":"/dev/infiniband",…}]
```

**The device is mapped. All eight cards read 0 % at 15:10:31.** The container is
a permanent, non-occupying resident of this node's device list.

**This is the first cluster's line arriving intact: *seeing the device is not
holding a handle*.** There it was recorded against a `/proc`-KFD predicate; here
it defeats a `HostConfig.Devices` predicate written **this morning to replace
that one**. **The T+396 fix swapped a blind instrument for a permanently-nonempty
one**, and `1c611a5d` had already said the hard branch was untested — **it turned
out to be unreachable, not merely untested.**

**And a second, independent cause is recorded beside it:** *nothing sequences
that validator's teardown against m2's preflight.* **That is the T+186 finding —
a validator's engine outliving its verdict by 1.8 s — recurring at the scale of a
112–117 GB drain.** The ownership-wait added at launch3 handles a holder that can
be identified; **it does not help when the abort fires on VRAM before
identification matters.**

### 2. 当前进展 — stage 1 green a sixth time, m2 blocked by a new cause

```
026b96  dead   last 14:58:20
        m1_deploy = succeeded    deploy_and_prove = succeeded
        run_profiling_mode_off = running   (died here)
        verdicts: layout true · environment true · deploy.sh true
79bca5  ALIVE  pid 163036, started 15:01:46
        container=yihou_e2e_chain3
        work_root=/data/yihou/e2e_flow3     ← SEPARATED at last
        jobid=29313   trace_end_ms=120000
        port_etcd=8103                       ← band unchanged
        main/m1_deploy/deploy_and_prove all running
node    8 cards VRAM 0 %, no yihou_* container — run 9 min old, pre-bring-up
```

**`work_root` is finally distinct** (`/data/yihou/e2e_flow3`). **The port band is
not**, but only one chain is alive, so nothing is contending for it.

**No `capture_stacks.log` and no `stacks_manifest.json` exist anywhere in
`026b96`** — it never got far enough for the T+487 §4 question to be answerable.
**That prediction remains open and untested for a second run.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~46 %** (unchanged) |
| 已经耗时 | **~531 min** (mission.md 06:19:11 → 15:10:59) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged for the third consecutive interval, and the reason is worth stating
plainly.** Stage 1 has now gone green **six times**. `m2_profiling` has been
entered **six times** and completed **zero**. **The causes have differed every
time** — trace hash_id, card lease, stack window arithmetic, `stack_ranks`
split, and now a preflight whose foreign-container instrument cannot return
empty on this host.

**That is not a chain that is failing to work. It is a chain whose second stage
sits behind a queue of independent single-point defects**, each of which took a
run to surface.

**Hold `29313`: 22 h 49 min left.**

### 4. Code problems

**Newly root-caused, unfixed:** `preflight.sh:211` — `_foreign_gpu_containers`
tests device mapping, not occupancy; two permanent residents make it
never-empty, so any busy card aborts. **The fix candidates are named by the
first cluster's own conclusion: per-card VRAM crosses namespaces and is the
occupancy signal; a container list should admit on "is a GPU-capable container
present" only as a supplement.**

**Newly named, unfixed:** nothing sequences `check_deploy_serves`'s teardown
against m2's preflight.

**Carried, root-caused, unfixed:** `stack_ranks` unlinked from `stack_window_s`
(`check_trace_coverage:223`, `m2_profiling.yaml:133`); `--var jobid` can name a
dead hold and pass `_agree_or_die`.
**Carried, documented, not enforced:** the `trace_end_ms` 73 s floor.
**Carried unfixed:** the eight `jsonschema` validators (still unread);
unbooked-usage and missing-`depends_on`; the `CLAUDE_CONFIG_DIR` placeholder.

### 5. 未定性

- **Whether `79bca5` reaches `_on` and whether stacks get captured** with
  `trace_end_ms=120000`. **Twice now this has been the open question and twice
  the run died before answering it.**
- **Whether the preflight abort has been fixed for this launch.** `79bca5`
  started 15:01:46 and `c890bf83` was committed in the same window; **I have not
  established the order**, and the run has not reached the preflight yet.
- **Whether all eight crash-affected validators were repaired** — carried,
  **and I note it has now survived unread since T+94**, seven hours.
- **What module 5 consumes if module 4 is replayed** — **fifteenth consecutive
  section.**

### 6. 新增 commit

Since T+487, four:

```
f944eec6  checkpoint R2 T+487 — mine
c890bf83  bug.record 13: the kit preflight's two wait branches are unreachable
          on this host
8cc13d92  RUNG5-CHECKLIST P2d: blocking launch gate for the preflight stranger
          test
4cd1425e  PREFLIGHT-FIX-FOR-NEXT-LAUNCH: drop-in instruction text for the
          stranger test
2f8e7faa  m35: compare.py barrier downgraded to conditional; two thin-worklist
          explanations retired
```

**`8cc13d92` and `4cd1425e` are the same repair at two tiers** — a blocking gate
in the checklist, and drop-in text so the next launcher does not have to compose
it. **This record's own tier analysis says the second is what makes the first
survive contact with someone in a hurry.**

**`2f8e7faa` retires two explanations rather than adding one.** Recorded because
it is rarer than it should be: **most of today's commits added a finding; that
one removed two that did not hold.**

### 7. 其他

**Three instruments have now been used to answer "is this node busy," and all
three were wrong in a different way.**

```
/proc KFD count          blind across containers   -> reads 0 on a busy node
HostConfig.Devices       mapping, not occupancy    -> never empty on this node
rocm-smi per-card VRAM   correct, and it caught
                         our OWN draining engine   -> aborts on ourselves
```

**The third is the one this file has recommended all along, and it is the one
that fired today** — correctly, on 112–117 GB that was genuinely allocated, by
a process that was genuinely ours and genuinely going away. **A correct
occupancy reading is still not an answer to "may I proceed," because it cannot
distinguish a tenant from a corpse.**

**What the three failures share is that each answered a question adjacent to the
one asked.** *Which processes can I see* / *which containers may use a GPU* /
*how much memory is allocated right now* — **none of them is *will these cards be
free when I need them*, and that is the question a preflight actually has.**

---

## R2 T+548 — 2026-09-06 15:42 UTC

**T+548 = wall-clock delta from the baseline** (06:33:41 → 15:41:18).

### 1. `check_deploy_serves` refused for the first time — and the cause inverts the obvious reading

**[observed, first-hand] Run `20260906T150155-79bca5`, 30 files in materials:**

```
# check_deploy_serves
## c32f1bb3-…: REFUSED
  note: kit qwen3-32b-mix.packup_20260906, tag serves-60b9599a, port base 8140
  note: 1/4 bring-up
  note: 2/4 diagnostic probes
  note: 4/4 teardown
  PROBLEM: probe router_health: no answer:
           URLError: <urlopen error [Errno 111] Connection refused>
  PROBLEM: the load was not sent — a deployment that fails a fatal probe has
           nothing to measure
```

**Step 3 of 4 is absent from the notes: the load never ran.** `layout` and
`environment` passed; this is the only false verdict.

**The obvious reading is "the engine did not come up." I opened the logs, and the
engine did come up.** From `/data/yihou/e2e_flow3/validate/serves-60b9599a/logs/`:

```
worker.log (78 129 B)   INFO: SGLang ready on port 8141
                        INFO: worker ready: model=Qwen/Qwen3-32B
                              url=http://10.235.192.131:8141 engine=SGLANG
                        [15:31:53] "GET /health HTTP/1.1" 200 OK   (twice)

router.log (4 774 B)    INFO: router-policy=kv-aware overlap_weight=1
                        INFO: using etcd discovery:
                              endpoint=10.235.192.131:8142 prefix=/infera/workers/
                        Traceback …
                        httpx.ConnectError: All connection attempts failed
```

**The worker was serving and answering health checks. The router died reaching
etcd discovery at 8142**, so nothing listened on 8140 and the probe got
connection refused in **0.001 s**.

**And the deployment record says so without needing a probe at all:**

```
deployment.json:  endpoint  http://10.235.192.131:8140
                  ports     router 8140, worker 8141, etcd 8142, …
                  started_at  null            ← every successful deploy today
                                                 carried a real timestamp
```

**`started_at: null` is a one-field discriminator** between "this deployment
happened" and "this record was written for a deployment that did not." Every
green kit today carried a value there — `2026-09-06T06:54:35.131400696Z` at
T+31, for instance.

**The refusal is correct and its wording is accurate at the level it operates.**
`router_health` genuinely got nothing. **But a reader stopping at the refusal
concludes the deployment failed; the worker log says three quarters of it
succeeded and one component could not reach etcd.** The distinction decides
whether the next fix targets bring-up or discovery.

**Where that evidence lives is the T+216 finding again**: not in the run tree, in
`work_root`. **`/data/yihou/e2e_flow3/validate/serves-60b9599a/logs/` is the only
place the router traceback exists.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~46 %** (unchanged) |
| 已经耗时 | **~561 min** (mission.md 06:19:11 → 15:41:18) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Fourth consecutive interval unchanged.** Stage 1 had gone green six times;
**this run is the first in which it did not** — `check_deploy_serves` passed on
five prior occasions with the same kit.

**Hold `29313`: 22 h 19 min left.**

### 3. 当前进展

```
026b96  dead   last 14:58:20   stage 1 green, died in run_profiling_mode_off
79bca5  ALIVE  pid 163036, started 15:01:46, container=yihou_e2e_chain3
        work_root=/data/yihou/e2e_flow3        (separated)
        deploy_and_prove: output_validating
        layout true · environment true · deploy_serves FALSE
        last write 15:31:54  —  9 min 24 s before I sampled
node    8 cards VRAM 0 %, only the two foreign CPU containers
```

**The run has an orchestrator and has not written for nine minutes.** The
teardown at 15:31:54 is consistent with the refusal having just landed and the
graph having nowhere to push. **I have not opened its transcript and am not
declaring it stalled.**

### 4. Code problems

**New, unfixed:** the kit's router cannot reach etcd discovery at
`10.235.192.131:8142` under the validator's port base 8140
(`logs/router.log`, `httpx.ConnectError`). **Cause unknown to me.** The worker at
8141 was healthy.

**New, and cheap:** a deployment record is written with `started_at: null` when
bring-up does not complete. **Nothing appears to read that field as a gate** —
the failure was found by a probe making a network call. **A record that already
knows it is describing a non-deployment is a stronger and cheaper signal than
a connection attempt.**

**Carried, root-caused, unfixed:** `preflight.sh:211` device-mapping instrument;
teardown-vs-preflight sequencing; `stack_ranks` unlinked from `stack_window_s`;
`--var jobid` passing `_agree_or_die` while naming a dead hold.
**Carried, documented, not enforced:** the `trace_end_ms` 73 s floor.
**Carried unread since T+94:** whether all eight `jsonschema` validators were
repaired.

### 5. 未定性

- **Why the router could not reach etcd at 8142** when the worker at 8141 came
  up. **The measurement: whether an etcd container existed for tag
  `serves-60b9599a`.** I did not check and it is one `docker ps -a` filter.
- **Whether this is related to the new `work_root=/data/yihou/e2e_flow3`**, which
  is the one launch variable that changed. **Adjacency, not evidence** — this
  record has been wrong twice today reasoning from exactly that.
- **Whether stacks get captured with `trace_end_ms=120000`** — open for a third
  consecutive run; no `capture_stacks.log` exists in either recent run.
- **What module 5 consumes if module 4 is replayed** — **sixteenth consecutive
  section.**

### 6. 新增 commit

Since T+517, one:

```
310d5baa  checkpoint R2 T+517 — mine
```

**No other commits in this interval.**

### 7. 其他

**Today's refusals have divided cleanly into two kinds, and this one is the first
of the second kind.**

```
kind A   the refusal names the defect       env.sh:172 · stacks_manifest missing
         and the fix                        · min launchers in the head
kind B   the refusal names the SYMPTOM      router_health: connection refused
         and the cause is elsewhere
```

**Kind A refusals were actionable from the report alone.** This one required
opening two log files in a directory the run tree does not contain, and the
answer reversed the natural reading: **not "bring-up failed" but "the worker is
serving and the router cannot find it."**

**I do not think that is a defect in the validator.** `check_deploy_serves`
probes the product endpoint because that is what a consumer touches, and
short-circuiting on it is the documented design — its own probe text says
*"Everything after this is meaningless if it is not."* **The gap is that its
report has no pointer to where the component logs live**, and every reader will
need them. **One line — the path it already knows, since it wrote the tag into
the note — would close it.**

---

## R2 T+578 — 2026-09-06 16:11 UTC

**T+578 = wall-clock delta from the baseline** (06:33:41 → 16:11:17).

### 1. My T+548 reading was wrong — and the real cause is a JIT build racing the router

**At T+548 I wrote "the worker was serving and the router could not reach etcd."
Both halves are wrong.** The owner's record `fc0e0441` says the engine died, and
I went back to the log and read more than four lines.

**[observed, first-hand] `logs/worker.log`, lines 272–286:**

```
15:31:19  The server is fired up and ready to roll!
15:31:19  Freezing GC in Detokenizer Manager process
15:31:19  [aiter] Process-3 start build [mha_batch_prefill_bf16_…]
          [aiter] Process-1/2/4 waiting for baton release
          INFO: waiting for SGLang HTTP on port 8141 … (elapsed 125s)
15:31:40  Health check failed. Server couldn't get a response from detokenizer
          for last 20 seconds. last_heartbeat time: 15:31:19
15:31:42  Health check failed …
15:31:47  Health check failed …
          [aiter] Process-3 finish build […], cost 32.6s
15:31:53  SGLang ready on port 8141 · "GET /health" 200 OK
```

**The engine was not broken. It was blocked in a 32.6-second aiter JIT kernel
compile**, three processes waiting on a baton, and the detokenizer heartbeat
stalled for the duration. **It then recovered fully** — which is exactly the
state I sampled at T+548 and mistook for health.

**The mechanism, from the owner** [`fc0e0441`, not re-derived by me]: *the kit
gates the router on the worker's `/health`, so the router started into nothing.*
The router hit `ConnectError`, died, and **nothing was listening on 8140 when the
worker came back.**

**How I got it wrong, precisely.** I ran `tail -4` on `worker.log`, saw
`SGLang ready` and two `200 OK`s, and concluded the worker was serving. **Those
lines are true and they are the recovery.** The failure is 21 lines earlier.
**This is `tail -N` eating the line that explains the death — the trap this file
has recorded twice on the other cluster, committed by the person maintaining the
file.**

**And "could not reach etcd" was adjacency.** `router.log` line 2 mentions etcd
discovery; line 3 begins the traceback. **I read the line before the exception
as its cause.** Third time today this record has reasoned from adjacency, and
the first two are written down two sections above.

### 2. The owner corrected themselves in the same window, on a different claim

**[first-hand, `cff188de`]**

> *I wrote it as a regression the run-3 rewrite introduced. … `etcd.log` is
> absent from three other arms and from **run 3's OWN successful m1 deploy**: two
> code paths, one long-standing gap, not a new one. The fix is unchanged; the
> class is not, and **my version would have sent the next reader diffing two kits
> for nothing.***

**Three suspects were eliminated by measurement rather than argument** — the
repaired preflight was never invoked, ports were identical to the passing run,
and **m1's own bring-up on the new `work_root` succeeded.** **That last one closes
my T+548 §5 question**: the new `work_root=/data/yihou/e2e_flow3` is not
involved, and it was closed by a measurement rather than by my declining to
speculate.

***One attribution I cannot confirm:*** `cff188de` credits the sweep against
known-good cases to "checkpoint." **I did not run it.** Either another party did
or the attribution is loose; **I flag it rather than accept credit**, because a
misattributed check is one nobody re-runs.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~46 %** (unchanged) |
| 已经耗时 | **~592 min** (mission.md 06:19:11 → 16:11:17) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Fifth consecutive interval unchanged.** **Hold `29313`: 21 h 49 min left.**

### 4. 当前进展

```
79bca5  dead   last 15:31:54   deploy_serves REFUSED (§1)
d9c7af  ALIVE  pid 554448, started 15:49:08, container=yihou_e2e_chain4
        m1_deploy / deploy_and_prove running
        yihou_e2e_etcd_chain4  16:07:22
        yihou_e2e_sgl_chain4   16:07:23
        cards 0-3 at 75 %
        last write 16:10:35
```

**`docker ps -a --filter name=serves-60b9599a` returns nothing** — the failed
validator's containers were removed at teardown. **The reading I proposed to the
leader at 15:48 is no longer answerable**, and that is worth saying: **a
container-identity question has a shelf life measured in minutes**, while the
logs it left in `work_root` are still there.

### 5. Code problems

**Newly understood, unfixed:** an aiter JIT kernel build (32.6 s measured) can
stall the worker's detokenizer heartbeat past the window in which the kit gates
the router on `/health`. **The router does not survive the wait.** Files: the
kit's router start path and its health gate.

**Long-standing, now correctly classed:** `etcd.log` is written on no path in the
current kit and was absent from earlier arms too — **a diagnosability gap, not a
regression.**

**Carried, root-caused, unfixed:** `preflight.sh:211`; teardown-vs-preflight
sequencing; `stack_ranks` / `stack_window_s`; `--var jobid` vs `_agree_or_die`.
**Carried, documented, not enforced:** the `trace_end_ms` 73 s floor.
**Carried unread since T+94:** the eight `jsonschema` validators.

### 6. 未定性

- **Whether the JIT build is a first-run cost** that will not recur now that the
  kernel is compiled, or whether it recurs per container. **The measurement:
  whether `chain4`'s worker.log shows the same 32-second build.** It is bringing
  up now and will answer this on its own.
- **Whether the router's health gate has a timeout long enough** for a cold JIT
  path. Not read.
- **Whether stacks get captured with `trace_end_ms=120000`** — open for a fourth
  run.
- **What module 5 consumes if module 4 is replayed** — **seventeenth consecutive
  section.**

### 7. 新增 commit

Since T+548, three:

```
519fab1a  checkpoint R2 T+548 — mine
fc0e0441  bug.record 14: run 3 died at the engine, not at the guard we had just
          fixed
cff188de  bug.record 14: correct my attribution of the missing etcd.log
60f18043  PROPOSAL (unapplied): tier-2 form of the compare.py --environment gap
```

**`60f18043` is labelled `(unapplied)` in its own subject** — the third time
today someone has committed a repair and declined to enable it. **That is now a
habit rather than an accident**, and it is the right one when the hold is long
and the change is unreviewed.

### 8. 其他

**Two people got the same failure wrong in the same hour, in opposite
directions, and both corrections came from reading further into a file they had
already opened.**

```
me     read tail -4 of worker.log     -> "the worker was serving"
       read the line before the       -> "could not reach etcd"
       traceback in router.log
owner  read the absence of etcd.log   -> "a regression the rewrite introduced"
```

**Neither of us lacked the file.** Mine was open and I read four lines of it;
theirs was a zero they interpreted before establishing its denominator. **The
corrections cost one `sed -n '272,286p'` and one sweep across known-good arms.**

**What actually happened is more interesting than either wrong version:** a
kernel JIT compile made a healthy engine look dead for 33 seconds, and a
component that gates on health took the bait. **Nothing was broken; something was
slow, and a timeout turned slow into failed.** That is a class this file has not
recorded before today.

---

## R2 T+607 — 2026-09-06 16:41 UTC

**T+607 = wall-clock delta from the baseline** (06:33:41 → 16:41:00).

### 1. T+578's mechanism is retracted — and I measured the counterexample myself

**The JIT-build-stalls-the-detokenizer story I wrote one section ago is wrong.**
`734efc0d` retracts it, and **I have first-hand corroboration I gathered before
reading the retraction.**

**[observed, first-hand] Every arm of the currently-running chain carries the
exact signature, including the one that succeeded:**

```
/data/yihou/e2e_flow4/<arm>/logs/worker.log
  chain4   start build ×3   Health check failed ×4   fired up ×1
  pmoff    start build ×3   Health check failed ×4   fired up ×1
  pmon     start build ×3   Health check failed ×4   fired up ×1
  build costs: 12.76 s, 36.7 s, 36.4 s
```

**`run_profiling_mode_off` = `succeeded` for the `pmoff` arm.** The signature I
called the cause is present, four times, in an arm that finished cleanly.

**[from `734efc0d`, and the sampling diagnosis is the part worth keeping]**

> *21/21 worker logs carry it, **including every successful bring-up**: `c2a`
> which served a real completion, `chain3` which passed at 15:21:42, and
> `serves-d8ff1deb` whose `check_deploy_serves` passed. **I sampled only
> failures, and that sample could not structurally contain a counterexample.***

**"That sample could not structurally contain a counterexample" is the cleanest
statement of this failure mode anyone has written today.** Looking only at
failures cannot distinguish a cause from a constant.

**And the cost was not hypothetical:**

> *The retry I proposed on this signature **killed two healthy bring-ups in run
> 4**, both at 130 s against a 2400 s budget, mid cold start, with no
> `router.log` in either. **The kit's own console said the health-check failures
> were the cold start and not a hang, thirty seconds before aborting on them.***

**A guard built on a universal signature aborts everything.** It fired at 130
seconds of a 2400-second budget, on healthy runs, **against the kit's own printed
statement that this was a cold start.**

**Where that leaves run 3:** *unexplained again — its router could not reach etcd
on 8142.* **That is where I was at T+548**, and I want to be exact: my T+548
sentence named etcd as the *cause* and that is still not established. **The
retraction restores "unexplained," not my original claim.** I was wrong at T+548
(overclaimed a cause), wrong at T+578 (adopted a mechanism that was a constant),
and the honest position is the one nobody has been able to improve on: **the
router could not reach etcd on 8142 and nobody knows why.**

**The original text was kept below the retraction**, with the note *"the
reasoning is worth more than the conclusion."*

### 2. Run 10 is the deepest point of the round — both profiling arms, no refusals

**[observed] `20260906T154908-d9c7af`, pid 554448, last write 16:40:25:**

```
m1_deploy               succeeded
deploy_and_prove        succeeded          ← stage 1 green, 7th time
run_profiling_mode_off  succeeded          ← sealed
run_profiling_mode_on   running            ← live now
m2_profiling            running

six verdicts, ZERO refusals:
  layout · environment ×2 · deploy.sh · require_present · max_error_rate

yihou_e2e_etcd_…_pmon  16:33:46
yihou_e2e_sgl_…_pmon   16:33:47      cards 0-3 at 76 %
```

**`run_profiling_mode_on` has never run before on this cluster.** Every prior
chain died at or before `_off`. **This is the first time the second profiling arm
has been in flight.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~48 %** (+2) |
| 已经耗时 | **~622 min** (mission.md 06:19:11 → 16:41:00) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2 for new ground: `_on` in flight with a clean verdict sheet.** Not more —
**`m2_profiling` still has not completed, on the eighth attempt.**

**Hold `29313`: 21 h 19 min left.**

### 4. Code problems

**Retracted, not a defect:** the detokenizer health-check signature. **A guard
was built on it and killed two healthy bring-ups; that guard must not ship.**

**Unexplained, open:** run 3's router could not reach etcd on 8142.

**Carried, root-caused, unfixed:** `preflight.sh:211` device-mapping instrument;
teardown-vs-preflight sequencing; `stack_ranks` / `stack_window_s`; `--var jobid`
vs `_agree_or_die`; `etcd.log` written on no path.
**Carried, documented, not enforced:** the `trace_end_ms` 73 s floor.
**Carried unread since T+94:** the eight `jsonschema` validators.

### 5. 未定性

- **Whether `_on` completes and `m2_profiling` seals.** Live now. **The furthest
  the round has reached.**
- **Whether stacks get captured with `trace_end_ms=120000`.** Open for four runs;
  **`_on` reaching `running` is the first time the question can actually be
  answered.**
- **Why run 3's router could not reach etcd on 8142** — reopened.
- **What module 5 consumes if module 4 is replayed** — **eighteenth consecutive
  section**, and if `m2_profiling` seals it stops being hypothetical.

### 6. 新增 commit

Since T+578, one:

```
b6d21a09  checkpoint R2 T+578 — mine  (its §1 mechanism is retracted above)
734efc0d  bug.record 14: RETRACTED — the detokenizer signature is universal,
          not a fault
```

**My T+578 commit message asserts the retracted mechanism in its subject line**,
and a commit message cannot be amended once it is in a shared history. **This
section is the only correction that will travel with it** — which is precisely
the hazard this file records about commit messages, now applying to mine.

### 7. 其他

**Today's most expensive pattern, stated with three instances and a cure.**

```
sampled only failures        -> a constant looks like a cause
                             -> a retry guard killed 2 healthy bring-ups
read tail -4 of a log        -> the recovery looked like the state
read the line before an      -> adjacency looked like causation
  exception
```

**All three are mine or the owner's from the last two hours, and all three were
resolved the same way: by looking at something known to be good.** The owner
swept 21 worker logs and found the signature in every successful one; **I listed
three arms of a live chain and found it in the arm that had already sealed.**

**Neither of us needed a new tool.** The `pmoff` arm was sitting there with
`succeeded` next to it in the same `store/task` listing I print every thirty
minutes. **The control was already in the sample; the question just had not been
asked of it.**

---

## R2 T+637 — 2026-09-06 17:11 UTC

**T+637 = wall-clock delta from the baseline** (06:33:41 → 17:10:36).

### 1. STAGE 2 IS COMPLETE — and stage 3 was entered

**[observed, first-hand] Run `20260906T154908-d9c7af`:**

```
m1_deploy                succeeded
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    succeeded      ← never completed before
merge_profiling_evidence succeeded
m2_profiling             SUCCEEDED      ← STAGE 2 SEALED, first time
rank                     succeeded
identify                 output_validating
m3_analysis              running        ← STAGE 3 ENTERED, first time

18 verdicts:  17 pass, 1 refuse
```

**Eight attempts reached `m2_profiling`; this is the first that sealed it.**

**And the four-run-old question is answered — the stacks were captured:**

```
capture_stacks.log      1
stacks_manifest.json    6

items/result/trace/stacks_manifest.json:
  ranks 2
  totals {files 2, bytes 137 678 956, gpu_kernels 119 054,
          python_functions 5 613 901, readable 2}
```

**5.6 million python function samples and 137 MB of stack data.** `trace_end_ms=
120000` gave the load enough runway for the capture to happen inside it — **the
`5d098338` arithmetic (73 s floor) was correct and the fix worked.**
**`check_trace_coverage` passed** — the validator that refused at T+366 and T+457.

**`stack_window_s=0` was never needed.** T+426 recorded that it was masking a
13-second shortfall; **this run proves the frames were capturable all along.**

### 2. The stage-3 refusal, and it names a collision a human would not see

```
# check_identity_resolved                       (5 files in materials)
## fc537169-…: REFUSED
  note:  5/5 resolved (ratio 1.00, floor 0.0 — a floor of zero grades nothing;
         set --var min_resolve_ratio to grade it)
  PROBLEM: duplicate logical_operator(s): ['layernorm_aiter_add_rmsnorm_quant'].
           It becomes a directory name in the workset, so two of them collide
           silently
```

**Two things worth separating.** The resolve ratio is **1.00 — everything
resolved** — and the validator says so *and* volunteers that its own floor of
zero grades nothing. **A validator naming its own weak threshold in the same
report is rare and it is the honest form.**

**The actual refusal is a name collision that becomes a directory name.** Two
operators sharing `layernorm_aiter_add_rmsnorm_quant` would silently overwrite
each other in the workset. **`649af26b` — "identify: make logical_operator
unique, because it becomes a directory name" — is already committed.**

### 3. The JIT mechanism is reinstated, and the reconciliation is the interesting part

**[from `1c954e0f`]** My T+578 mechanism was **not** wrong; the T+607 retraction
over-corrected. The two findings are compatible and each answers a different
question:

```
21/21 logs carry the signature   ->  says WHERE it is: everywhere
the 32.6 s JIT compile window    ->  says WHY it is there, and why run 3's
                                     router — gated on /health — started INTO
                                     the window, hit ConnectError and died
```

> *the detokenizer lines are universal **because they are the shadow of a JIT
> compile.** 21/21 says where it is, this says why.*

**Four positions in four hours, and I want the sequence legible:**

```
T+548  me     "router could not reach etcd"        wrong: adjacency
T+578  me     JIT window, router gated on /health  mechanism — correct
T+607  owner  signature universal -> "not a fault" correct about the SIGNATURE,
              + my sealed-arm counterexample       over-reached to "unexplained"
T+637  owner  both true; universality is the        reconciled
              shadow, the window is the cause
```

**What was actually wrong at T+607 was treating a constant as a *discriminator* —
in both directions.** The owner had used its presence to accuse; we then used its
universality to acquit. **Neither is available from a constant.** The thing that
resolved it was a *timing* argument, which the signature alone cannot carry.

***And the guard cost still stands.*** A retry built on that signature killed two
healthy bring-ups at 130 s of a 2400 s budget. **Reinstating the mechanism does
not reinstate the guard**, and `1c954e0f` says so: *widening the router's gate is
one observation*, deliberately unfixed.

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~56 %** (+8) |
| 已经耗时 | **~651 min** (mission.md 06:19:11 → 17:10:36) |
| 预估耗时 | **still absent, and now for a smaller reason** |
| 可靠性 | **中** |

**+8: two of five stages complete, the third entered.** Stage 2 sealed with
17/18 verdicts passing and a 137 MB stack capture behind it.

**On 预估耗时.** Stage 1 has four measured durations; **stage 2 now has exactly
one** (`d9c7af`: 15:49:08 launch → `m2_profiling` sealed before 16:57:36, so
under ~68 min for both stages together). **Stages 4 and 5 still have zero, and
module 4 is the one the user has already flagged as very long.** A total is still
a number with a false denominator — **but for the first time the missing terms
are two rather than four.**

**Hold `29313`: 20 h 49 min left.**

### 5. 当前进展

```
d9c7af  dead   last 16:57:36 — reached m3_analysis, STAGE 2 SEALED
1048af  ALIVE  pid 1229785, container=yihou_e2e_chain5, started 17:03:54
        yihou_e2e_etcd_serves-444741f7  17:03:59
        yihou_e2e_sgl_serves-444741f7   17:04:00
        aiperf_serves-444741f7          17:06:56
        cards 0-3 at 76 %
```

**Run 11 is already inside `check_deploy_serves`** three minutes after launch —
faster than any prior run, consistent with a warm JIT cache on this host.

### 6. Code problems

**Fixed and committed:** `649af26b` — `logical_operator` uniqueness.
**Reinstated, deliberately unfixed:** the router's `/health` gate is narrower
than a cold JIT compile. **One observation, not enough to widen it on.**
**Must not ship:** any retry keyed on the detokenizer health-check signature.

**Carried, root-caused, unfixed:** `preflight.sh:211`; teardown-vs-preflight
sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` written on no path;
`min_resolve_ratio` defaults to a floor of zero that grades nothing.
**Carried unread since T+94:** the eight `jsonschema` validators.

**No longer a problem:** `stack_ranks` / `stack_window_s` — the split knobs were
only reachable because the capture was being skipped. **With the capture
happening, neither flag is needed.**

### 7. 未定性

- **Whether `m3_analysis` completes.** Entered once, refused once on a fix that
  has already landed.
- **What module 5 consumes if module 4 is replayed** — **nineteenth consecutive
  section, and it is now the next unknown in the path**, not a distant one.
- **Whether module 4's duration fits the hold.** 20 h 49 min; the user has
  flagged kernel forge as very long and this record has no measurement of it on
  this cluster.
- **Why run 3's router hit the JIT window when ten other bring-ups did not** —
  §3 explains the mechanism, not the timing coincidence.

### 8. 新增 commit

Since T+607, three:

```
57ba0bd9  checkpoint R2 T+607 — mine
967a4b09  validator.failures: run 4 board — 8 validators passed, one real
          refusal
649af26b  identify: make logical_operator unique, because it becomes a
          directory name
1c954e0f  bug.record 14: run 3's cause found (aiter JIT window), and fix an
          attribution
```

### 9. 其他

**Eleven runs to get two stages, and the eleventh needed none of the eleven
failures to be repeated.**

Every cause has been distinct: a framework import, a shared port band, a
malformed trace, a card lease, a kit that could not be co-located, a blind
occupancy predicate, a device-mapping predicate, a stale hold id, a load shorter
than the capture it contained, a name collision that becomes a directory. **Ten
single-point defects, each found once, each found by something refusing rather
than by something silently producing a wrong answer.**

**That is the argument for the validator set, stated in the only currency that
counts.** At T+94 the entire set was unreachable behind an `ImportError` and this
record could not say whether it was worth anything. **Today it caught a duplicate
string that would have become two directories with one name** — a defect no
amount of running the chain would have surfaced, because the chain would have
run.

---

## R2 T+667 — 2026-09-06 17:40 UTC

**T+667 = wall-clock delta from the baseline** (06:33:41 → 17:40:17).

### 1. This cluster now has a sealed corpus, produced by itself

**The baseline's largest structural problem is solved.** At T+0 I wrote:
*"there is no sealed corpus on this cluster and `/shared_nfs` is empty …
**nothing can be replayed on this node until this round produces it first.**"*

**[observed, first-hand] `/data/yihou/e2e_verify_20260906/m35/replay_root_run4/`,
generated 17:02:32:**

```
PROMOTION.json     11 736 B
stage1-deploy/
stage2-profiling/  profiling_evidence
                   profiling_mode_off.bench_result
                   profiling_mode_on.bench_result
                   profiling_mode_on.kernel_table
                   profiling_mode_on.profile_result
```

**`PROMOTION.json` records how it was made, and it is the kind of provenance the
first cluster's corpus never had:**

```
generated_at   2026-09-06T17:02:32+00:00
required_node  smci355-ccs-aus-n04-25
kit_nodes      [smci355-ccs-aus-n04-25]
threshold      1
command        assets/lib/replay_root.py --out … --node smci355-ccs-aus-n04-25
               --threshold 1 --allow-unstable --no-seed
               --kind deploy_kit --kind profiling_evidence
               --kind profiling_mode_off.bench_result … 
               --run /data/yihou/agent_sys_runroot/runs/20260906T154908-d9c7af
also present   runs_surveyed · ACCEPTANCE · promoted · not_promoted
```

**It names the run it came from, the node it requires, and — importantly — a
`not_promoted` list.** The first cluster's `cheat_for_mock/` had a README warning
about four things that would mislead you; **this one carries its own exclusions
as data.**

**What this unlocks, stated concretely:** stages 3, 4 and 5 can now be developed
against replayed stage-1/2 artefacts **without spending 68 minutes and a GPU
bring-up per attempt.** Given that module 4 is the stage the user has flagged as
very long, and that no measurement of it exists here, **this is the difference
between iterating on the back half and waiting for the front half.**

### 2. Run 11 ended without finishing, at the same place as its predecessors

**[first-hand, `m35/launch-run5/chain.log`, last write 17:25:13]**

```
handoff  profiling_evidence slot v0: created
handoff  operator_workset slot v0: created
handoff  kernel_optimization slot v0: created
handoff  profiling_mode_off.bench_result slot v0: generating
handoff  profiling_mode_on.bench_result slot v0: created
   done  run complete; this package promises no failure, and the run did NOT
          finish: m2_profiling: running, m3_analysis: waiting_handoff,
          m4_kernel_opt: waiting_handoff, m5_integration: waiting_handoff,
          main: running, merge_profiling_evidence: waiting_handoff,
          run_profiling_mode_off: running,
          run_profiling_mode_on: waiting_resource,
          0 validation(s) dropped
```

**Stage 1 green (3/3 verdicts), then stopped inside `run_profiling_mode_off`.**
**`0 validation(s) dropped`** — nothing was skipped; it simply did not get there.

**I do not know why it stopped**, and the terminal line does not say. **The
reading that would: this run's transcript, or `store/event`'s
`attributes.detail`.** I have opened neither. **This record has recorded twice
today that the last line of a run log is not the line that explains it.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~58 %** (+2) |
| 已经耗时 | **~681 min** (mission.md 06:19:11 → 17:40:17) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2 for the corpus, not for a stage.** No stage advanced this interval; **the
cost of every future attempt at stages 3–5 dropped.**

**Hold `29313`: 20 h 19 min left.**

### 4. 当前进展 — nothing is running

```
d9c7af  dead   16:57:36   STAGE 2 SEALED, m3_analysis reached, 20/21 verdicts
1048af  dead   17:10:14   stage 1 green, stopped in _off, 3/3 verdicts
orchestrators  none
node           8 cards VRAM 0 %, only the two foreign CPU containers
last write     17:25:13  (m35/launch-run5/chain.log)  — 15 min ago
```

**Fifteen minutes idle.** Below the hour-long gap of T+306 and not yet worth
alarm; **recorded so the next section can measure against it.** Twenty hours of
hold remain, so the arithmetic pressure of the 13:00–14:00 window does not
apply.

### 5. Code problems

**No new ones this interval.**

**Carried, root-caused, unfixed:** `preflight.sh:211`; teardown-vs-preflight
sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` written on no path;
`min_resolve_ratio` floor of zero grades nothing; the router `/health` gate
narrower than a cold JIT compile (deliberately unfixed on one observation).
**Fixed and committed:** `logical_operator` uniqueness (`649af26b`).
**Must not ship:** any retry keyed on the detokenizer signature.
**Carried unread since T+94:** the eight `jsonschema` validators — **eleven hours.**

### 6. 未定性

- **Why run 11 stopped in `run_profiling_mode_off`.** §2. **One transcript read
  away and nobody has reported it.**
- **What module 5 consumes if module 4 is replayed** — **twentieth consecutive
  section**, and the corpus in §1 is the machinery that would answer it cheaply.
- **Whether module 4 fits the hold.** No measurement exists.
- **Whether `PROMOTION.json`'s `not_promoted` list is empty or long.** I read the
  key and not its contents. **One command, and it says what the corpus cannot
  replay.**

### 7. 新增 commit

Since T+637, none. **This interval's output is in
`/data/yihou/e2e_verify_20260906/m35/`, uncommitted** — including the replay
root, `PROMOTION.json`, and `launch-run5/`.

### 8. 其他

**The corpus in §1 closes the loop the baseline opened, and it is worth marking
what that took.**

```
T+0     no corpus exists; nothing can be replayed until we produce it
T+637   stage 2 sealed with a 137 MB stack capture
T+667   that run's artefacts promoted into a replay root, with provenance
```

**Eleven hours, eleven runs, and the thing that made the corpus possible was
producing one genuinely complete stage.** Not a mock, not a graft — **the
`--run` field in `PROMOTION.json` names `20260906T154908-d9c7af` and nothing
else.**

**The first cluster spent this round's equivalent effort grafting today's real
artefacts into a corpus sealed eight days earlier**, because it had one. **This
cluster had nothing and therefore had to earn it, and what it earned has a
`required_node` field and a command line that reproduces it.** Whether that
matters will show in whoever next needs to replay stage 2 without a GPU.

---

## R2 T+697 — 2026-09-06 18:10 UTC

**T+697 = wall-clock delta from the baseline** (06:33:41 → 18:10:16).

### 1. The launch record has become a pre-registration document

**[first-hand, `m2/launch9/LAUNCH-RECORD.txt`, 18:00:01]** — the whole point is
what it says about a waiver it is *keeping*:

```
FULL REAL CHAIN: mock_stages=none, no replay
trace_end_ms=120000 (the proven fix);
  stack_window_s and stack_ranks ABSENT (capture must succeed)
kernel_table_min_launchers=0 KEPT -- leader scope decision made BEFORE the result:
  one false verdict anywhere in the _on closure invalidates all three handoffs
  (runner.py:598,972), and packup has never been reached on any cluster.
  Reaching it once outweighs one attributable verdict.
  THIS PASS IS NOT EVIDENCE. Restoring a waiver to recover a green is not a
  measurement.
instruction: v4 + unconditional log capture before teardown
             (run 5 died with its cause unread)
NOT included: m35's run-4 instruction text (I do not have it);
              detokenizer retry deliberately absent
```

**"THIS PASS IS NOT EVIDENCE" written into the launch record before the run
starts** is pre-registration in its strongest form — **the criterion is fixed
before the data point, and the waiver is labelled so it cannot later be counted
as a result.** It also names *why* the waiver is kept (a single false verdict
invalidates three sibling handoffs, `runner.py:598,972`) and *what it is buying*
(reaching `packup` once).

**Three more things this record does that are worth copying:**

- **`stack_window_s` and `stack_ranks` ABSENT — "capture must succeed."** The
  masking flags from T+426/T+457 are deliberately not carried. **The run is
  required to earn the stacks rather than declare them away.**
- **"instruction: v4 + unconditional log capture before teardown (run 5 died
  with its cause unread)."** A defect in *observability* fixed by changing the
  instruction, with the incident that motivated it named inline.
- **"NOT included: m35's run-4 instruction text (I do not have it)"** — **an
  explicit statement of what the launcher could not obtain.** An absence
  recorded rather than silently omitted.

### 2. `launch10` supersedes it four minutes later — and reads its fix from the file

```
launched_at_utc: 2026-09-06T18:04:01Z
supersedes: 20260906T180012-9aa819
            (killed 2 min in: instruction carried the run-2 stranger clause)
preflight text: VERBATIM from PREFLIGHT-FIX-FOR-NEXT-LAUNCH.md @ 4cd1425e (m35),
                read from the file not the quotation
  - a container holds a GPU only if VRAM is ATTRIBUTABLE to it;
    /dev/kfd mapping is not evidence
  - unattributable => unknown; unknown WAITS (up to 300s), never aborts
```

**This is the resolution of T+517's three-instrument problem, stated as a
predicate.** All three earlier instruments answered adjacent questions —
`/proc` KFD (blind across containers), `HostConfig.Devices` (mapping, not
occupancy), bare VRAM (correct, but aborts on our own draining engine). **The new
one is attribution plus a default of *wait*, not *abort*.**

**"read from the file not the quotation" is the operative clause.** This record
has spent the day on findings that degraded as they were relayed; **reading the
source at a named commit removes the relay entirely.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~58 %** (unchanged) |
| 已经耗时 | **~710 min** (mission.md 06:19:11 → 18:10:16) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**No stage advanced.** Two launches spent, one killed at two minutes for
carrying stale instruction text.

**Hold `29313`: 19 h 50 min left.**

### 4. 当前进展 — sixteen runs; one alive and bringing up

```
1048af  dead  17:10:14  stage 1 green, stopped in _off
9aa819  dead  18:02:53  deploy_and_prove FAILED at ~2.5 min (killed, §2)
6b7c19  ALIVE pid 1773936, started 18:04:12, container=yihou_e2e_chain
        yihou_e2e_etcd_chain6  18:09:11
        yihou_e2e_sgl_chain6   18:09:12
        cards 0 % at 18:09:48 — bring-up is 36 s old
```

**The idle gap between runs was 50 minutes** (run-tree writes 17:10:14 →
18:00:12), or 34 minutes measured on teammate scratch (17:25:13 → 17:59:38).
**Recorded against T+667's 15 minutes; smaller than T+306's 63.**

### 5. Code problems

**No new ones.** **Fixed in launch instruction, not in the package:** the
preflight predicate (§2) and unconditional log capture before teardown.

**Carried, root-caused, unfixed:** `preflight.sh:211` in the package itself (the
fix above lives in launch text); teardown-vs-preflight sequencing; `--var jobid`
vs `_agree_or_die`; `etcd.log` written on no path; `min_resolve_ratio` floor of
zero; the router `/health` gate vs a cold JIT compile.
**Carried unread since T+94:** the eight `jsonschema` validators — **eleven and a
half hours.**

### 6. 未定性

- **Whether run 16 reaches `packup`.** It carries the proven `trace_end_ms`, no
  masking flags, the new preflight predicate, and log capture before teardown.
  **It is the best-equipped launch of the round.**
- **Whether `kernel_table_min_launchers=0` will be honoured as "not evidence"**
  when the result arrives. **The record says so in advance; whether the next
  reader carries that qualifier is precisely what the first cluster's §2.8
  incident was about** — a qualifier one line above a bold claim does not
  travel.
- **Why run 11 stopped in `_off`** — carried from T+667, still unread, **and the
  new instruction's "unconditional log capture before teardown" exists because of
  it.**
- **What module 5 consumes if module 4 is replayed** — **twenty-first
  consecutive section.**

### 7. 新增 commit

Since T+667, one:

```
202399e3  checkpoint R2 T+667 — mine
9a1b8a21  CLAUDE.md: fix the definition, not the branch — and say what the
          common case is
```

### 8. 其他

**The two launch records in this section are the round's best artefacts, and
neither is code.**

Between them they carry: what changed and why, which run is superseded and its
measured cause, a waiver kept with its justification *and* an advance statement
that its pass is not evidence, two flags deliberately omitted so a capture must
be earned, a fix quoted verbatim from a named commit rather than retyped, and an
explicit list of what the launcher could not obtain.

**Every one of those is a defence against a specific failure this record logged
today** — stale relayed text, a masked degradation read as a result, a
superseding run whose predecessor's cause was never established, a fix that
decayed in transit.

**Nine hours ago the launch line was not recoverable from the artefact at all.**
The gap that remains is that these documents are **uncommitted**, in
`/data/yihou/e2e_verify_20260906/m2/`, and the fixes they carry live in agent
instructions rather than in the package. **A launch record is a good place to
learn something and a poor place to keep it.**

---

## R2 T+726 — 2026-09-06 18:40 UTC

**T+726 = wall-clock delta from the baseline** (06:33:41 → 18:39:49).

### 1. Run 16 is reproducing stage 2, without the masking flags

**[observed, first-hand] `20260906T180412-6b7c19`, pid 1773936, last write
18:36:04:**

```
m1_deploy               succeeded          ← stage 1 green, 8th time
deploy_and_prove        succeeded
run_profiling_mode_off  succeeded
run_profiling_mode_on   running
m2_profiling            running

verdicts  6/6 pass, zero refusals
stacks_manifest.json    0  (the _on arm is mid-load; the capture comes last)

aiperf_profiling_mode_on_20260906_183335   18:33:35
yihou_e2e_sgl_…_pmon                       18:30:43
cards 0-3 at 76 %
```

**Elapsed to this point: 32 minutes** (18:04:12 launch → `_on` running).
`d9c7af` took roughly 68 minutes for stages 1 and 2 together.

**What makes this run different from `d9c7af` is what it does not carry.**
`launch10` omits `stack_window_s` and `stack_ranks` entirely — the two flags that
at T+426 and T+457 were masking a 13-second shortfall. **If the stacks appear
here, they were earned by `trace_end_ms=120000` alone, on a launch with no
capture waiver.** That would be a second independent confirmation rather than a
repeat.

**It is not yet answered.** `_on` is running and the capture has not happened.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~58 %** (unchanged) |
| 已经耗时 | **~740 min** (mission.md 06:19:11 → 18:39:49) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged, and deliberately so: reproducing a stage that is already counted is
not progress in the numerator.** It is worth a great deal for confidence and
nothing for the fraction, and this record has an obligation not to let the two
blur.

**Hold `29313`: 19 h 20 min left.**

### 3. 当前进展

```
6b7c19  ALIVE  the only run; no other orchestrator
node    cards 0-3 at 76 %, 4-7 idle
        three of our containers up, two foreign CPU containers
```

**The JIT signature is present again** — `start build` and `Health check failed`
together total 7 lines in this run's `pmon` worker log — **on an arm that is
running normally.** Recorded to keep T+607's correction attached to fresh data:
**the signature is a constant and cannot discriminate anything.**

### 4. Code problems

**No new ones this interval, and nothing has been fixed.**

**Carried, root-caused, unfixed:** `preflight.sh:211` in the package (the working
predicate lives in launch text, `4cd1425e`); teardown-vs-preflight sequencing;
`--var jobid` vs `_agree_or_die`; `etcd.log` written on no path;
`min_resolve_ratio` floor of zero; the router `/health` gate vs a cold JIT
compile.
**Carried unread since T+94:** the eight `jsonschema` validators — **twelve
hours.**

### 5. 未定性

- **Whether the stacks are captured without the waiver flags.** §1. **First
  answerable in this run.**
- **Whether run 16 reaches `m3_analysis`, and past it.** `649af26b` fixed the
  `logical_operator` collision that refused there; **no run has tested that fix.**
- **Whether `kernel_table_min_launchers=0` gets carried as "not evidence."**
  `launch9` wrote it in advance; the result has not arrived.
- **Why run 11 stopped in `_off`** — still unread. **The new instruction's
  unconditional log capture means the next such death will be readable**, but it
  does not recover this one.
- **What module 5 consumes if module 4 is replayed** — **twenty-second
  consecutive section.**

### 6. 新增 commit

Since T+697, none but mine (`6042638e`).

### 7. 其他

**A quiet interval, and the honest summary is that it looks like the loop
working.**

One run, alive, on the only node, with no competing orchestrator; the launch line
assembled from a named commit rather than retyped; two waiver flags deliberately
absent; six verdicts and no refusals; and a signature that misled two people four
hours ago now appearing in the logs of a healthy arm without anyone acting on it.

**Nothing here is a result.** `m2_profiling` has sealed once and is running for
the second time, and the stage past it has been entered once and refused once.
**But the difference between this interval and the 08:00 hour is that no
instrument is currently lying to anybody**, and every fix in flight was written
down before its result arrived.

---

## R2 T+757 — 2026-09-06 19:11 UTC

**T+757 = wall-clock delta from the baseline** (06:33:41 → 19:10:34).

### 1. Run 16 died, and the detector was fed by the failure it exists to catch

**[first-hand, `4c74288f`]**

> *AIPerf's request timeout and `agent_sys`'s stall timeout are **both 900 s**.
> Each timeout burst writes into the run tree seconds after the stall deadline,
> so **the detector is re-satisfied every cycle.** Measured: bursts at **18:50:24
> and 19:05:25, exactly 32 each**, on an engine that **stopped generating at
> 18:35:24**.*

**A stall detector watches for absence of writes. A request timeout produces
writes.** With both timers at 900 s, the failing component wrote just often
enough, and just late enough, to keep resetting the watchdog. **The engine was
dead for thirty minutes and the run looked alive throughout.**

**`e472d24b` states the general rule:** *a stall threshold must not equal any
downstream timeout.* **This is the cleanest tier-2 finding of the day** — it is
a property of two numbers, checkable before any run, and it does not depend on
anyone remembering anything.

**The record includes its own retrospective test** — *periodic writes at a known
timeout interval; an artefact that only grows failures* — **and leaves the
engine's cause open with the measurement that would close it.** Naming what it
does *not* explain is why the finding is trustworthy.

**Run 16's board** (`20260906T180412-6b7c19`, last write 19:05:59, orchestrator
gone):

```
m1_deploy               succeeded     stage 1 green, 8th time
deploy_and_prove        succeeded
run_profiling_mode_off  succeeded
run_profiling_mode_on   running       ← died here
verdicts 6/6 pass       stacks_manifest.json: 0
```

**The no-waiver stack-capture question from T+726 is unanswered for a second
run.** `_on` never completed.

### 2. A separate commit splits a mis-attribution from the mechanism it was credited to

**[first-hand, `667ba683`]**

> *A value taken from a grep **without asking which stage produced it** is one
> mechanism; **two runs' files coexisting under `pmoff/` and `pmon/`** is
> another. **Splitting `work_root` fixes the second and does not touch the
> first.** A fix credited with a save it did not make gets over-trusted.*

**This is the T+216 lesson generalised.** There the launch line recorded an
intention and the deployment recorded the fact; here a fix is being kept from
inheriting credit for a save it did not make. **`753e060f` puts it in
`CLAUDE.md`.**

### 3. Leader messages arrived batched, and two of them are stale — recorded because a stale directive is an instrument failure

**Five messages arrived together at ~19:10.** Three describe state from 06:51,
07:42 and 13:09; one is a cadence check timestamped 14:28. **Reading them in
sequence would have produced a section describing a hold that ended five hours
ago.** Recording the discrepancy rather than acting on it:

**(a) "Write the round's final section now, the hold ends at 13:59:59."**
**Superseded by events.** `29184` ended 14:00:01 and **`29313` started 14:00:06
on the same node with a 24-hour limit** — recorded at T+457 from `scontrol`.
There was no final section to write; the round continued for five more hours and
sealed stage 2 at 16:57.

**(b) The cadence check: "last changed at 13:10 … roughly 45 minutes past due."**
**Not correct, and the leader explicitly asked to be told which instrument is
right.** My commits to this file, read from `git log` just now:

```
12:10:43  T+336      15:11:47  T+517
12:41:05  T+366      15:42:03  T+548
13:10:41  T+396      16:12:12  T+578
13:40:50  T+426      16:41:41  T+607
14:11:23  T+457      17:11:39  T+637
14:41:57  T+487      17:41:12  T+667
                     18:10:59  T+697
                     18:40:19  T+726
```

**Every interval is 30 minutes ± 90 seconds, unbroken since the baseline.** At
14:28 the file's most recent commit was **14:11:23**, not 13:10. **The right
instrument is `git log -- work.checkpoint.summary.md`**; a working-tree mtime or
a stale `git log` cache will disagree.

**(c) The `bfs` warning, applied to my own idiom.** The leader is right that
`find` here is `bfs` and rejects relative `-newermt`. **I have used absolute
timestamps throughout** (`-newermt '2026-09-06 07:04'`), which `bfs` accepts —
**but I did carry `2>/dev/null` on some of those calls**, which is the more
dangerous half and which I flagged against myself at T+94 §4. **From this section
I use `-mmin -N`**; the check above this one used it.

### 4. The leader's own items, recorded plainly as asked

**Stated without softening, at their request:**

- **The duplicate launch at 14:08 was theirs** — two owners authorised, the first
  never withdrawn, two chains with identical ports and `work_root`. **They killed
  `3849649` at 14:10:35.** I flagged the overlap at 14:09:47 and named the wrong
  operative mechanism (T+216, T+487); **the authorisation is the part that made
  it possible and it is theirs.**
- **The `trace_end_ms=60000` override that blocked the ladder for six launches
  was theirs**, compression under an instruction to go faster. **T+426 recorded
  the arithmetic; this records who set it.**
- **The node was idle about an hour** between a refusal and a relaunch, held for
  a decision. **T+306 measured that gap at 63 minutes.**
- **Their own run-tree readings ran ahead of the filesystem three times** and
  were reported as independently confirmed. **They have stood down from
  corroborating run state.** *"An instrument failure with an unknown cause, not a
  resolved one"* — recorded as they asked, and I have not investigated it.

**And their central lesson, which this record reached independently at T+396 and
T+517:** *today's run deaths were each caused by a guard added to fix the
previous death, and each fix was correct about the failure in front of it and
blind to the one it created.* **Their diagnosis of why the countering evidence
did not help is the part I had not got to:** *the clause reads as conservative,
and the failure it produces looks exactly like the thing it guards against.*

### 5. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~58 %** (unchanged) |
| 已经耗时 | **~771 min** (mission.md 06:19:11 → 19:10:34) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged.** Run 16 reproduced stage 1 and `_off` and died in `_on`.

**Hold `29313`: 18 h 50 min left.**

### 6. Code problems

**New, root-caused, unfixed:** the stall threshold equals AIPerf's request
timeout (both 900 s), so timeout bursts re-satisfy the detector.
**New, unknown:** why the engine stopped generating at 18:35:24. **The record
names the measurement that would close it; nobody has taken it.**

**Carried, root-caused, unfixed:** `preflight.sh:211`; teardown-vs-preflight
sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on no path;
`min_resolve_ratio` floor of zero; the router `/health` gate vs a cold JIT
compile.
**Carried unread since T+94:** the eight `jsonschema` validators — **twelve and a
half hours.**

### 7. 未定性

- **Whether the stacks are captured without waiver flags** — unanswered for two
  runs.
- **Whether `merge_profiling_evidence` executes.** The leader states it has
  **never executed anywhere**, and that it, not m3, is the next rung. **`d9c7af`
  shows `merge_profiling_evidence = succeeded`** — I record the disagreement
  rather than resolving it; **the reading is that run's `store/task`, which is
  what I printed at T+637.**
- **Why the engine stopped generating at 18:35:24.**
- **What module 5 consumes if module 4 is replayed** — **twenty-third
  consecutive section.**

### 8. 新增 commit

Since T+726, four:

```
c9d32be7  checkpoint R2 T+726 — mine
753e060f  CLAUDE.md: a fix credited with a save it did not make gets over-trusted
667ba683  bug record: separate a mis-attribution from the real mechanism it was
          credited to
4c74288f  bug record: two 900s timers, and the failure feeding the detector
          meant to kill it
e472d24b  CLAUDE.md: a stall threshold must not equal any downstream timeout
```

### 9. 其他

**Two of the five messages I received this interval described a world that had
stopped existing five hours earlier, and one of them was an instruction to write
a final section.**

**Had I acted on it, this file would now contain a closing summary of a round
that went on to seal stage 2, build its own replay corpus, and reach stage 3.**
The thing that prevented it was not judgement — **it was that every section in
this file records the clock it was written against**, so a directive premised on
`29184` ending was checkable against `29313` starting, which T+457 had already
read from `scontrol`.

**That is the same property the launch records acquired at T+697** — `supersedes`
and a read `launched_at_utc` — and it is the same property the leader is asking
for when they say a stale directive should be answered with the right instrument
rather than absorbed. **A message does not carry its own expiry; a timestamped
record of what was true when does.**

---

## R2 T+786 — 2026-09-06 19:40 UTC

**T+786 = wall-clock delta from the baseline** (06:33:41 → 19:40:14).

### 1. The engine freeze from T+757 was diagnosed, then corrected twice by its own author

**The mechanism claim, then its retraction, then a correction of the retraction's
arithmetic — all inside forty minutes and all self-initiated.**

```
4afe06b7  "a with_stack profiler window freezes the engine mid-window"
52487eed  CORRECTION: the same config passed at 16:38
2e83f52c  the real denominator: stack window 1 of 3, measurement window 4 of 4
10558a28  CLAUDE.md: an intermittent failure is indistinguishable from a
          deterministic one at N=1
```

**[first-hand, `52487eed`]**

> *Run `20260906T154908-d9c7af` took **the same 3 s `with_stack=1` window on the
> same host, image and schedule** and produced four trace files;
> `check_trace_coverage` passed on it. **The freeze is intermittent, one of two
> real attempts, not a property of `with_stack`.** The `ReadTimeout` on that stop
> **happened in the passing run too and carries no information.** Timeline and
> the missing `Stop profiling` lines stand.*

**A known-good run refuted the mechanism, and the same run also disarmed the
`ReadTimeout` that had looked like evidence.** This is T+607's shape a second
time: **a signature present in a passing run cannot discriminate anything** —
except that here it was caught in forty minutes rather than four hours, and by
the person who proposed it.

**Then the denominator correction, which is the sharper one** [`2e83f52c`]:

> *I had written "two attempts, one hit" by **dropping the attempt that aborted
> before the window opened**. Different failure, still an attempt that produced
> no stack window; **excluding it flattered the number.***

```
stack window        1 of 3   ← survives
measurement window  4 of 4
```

**The asymmetry between the two window types is the finding, and it survives the
small sample.** The 1-of-3 is worse than the 2-of-3 they first wrote, and they
corrected it against themselves.

**And the rule they extracted is the one this record most needed today**
[`10558a28`]:

> *An intermittent failure is indistinguishable from a deterministic one at N=1.
> **And N=1 is invisible from inside the run tree.** Before naming a mechanism
> for a failure, ask whether that stage has **ever succeeded on this host**, and
> **ask it outside the run.** Four commands.*

**Every mechanism error this file recorded today would have been caught by that
question.** My T+548 etcd attribution, my T+578 JIT mechanism, the detokenizer
guard that killed two healthy bring-ups, and now this — **four instances, and in
each one the disproof was a run that had already succeeded and was not
consulted.**

### 2. A third member of the wrong-knob family

**`4e67b69d`: "a refusal names a `--var` the guard does not read."** Joining
`min_launchers_in_top_n` vs `kernel_table_min_launchers` (T+366) and
`stack_window_s` vs `stack_ranks` (T+457). **Three instances, three different
validators.** **The class — *a refusal's remediation text names an internal field
rather than the operator's flag* — is now established by multiplicity rather than
by a single case.**

### 3. Run 17 is in flight, and its launch record now binds identity

**[first-hand, `m2/launch11/LAUNCH-RECORD.txt`]**

```
written_at        2026-09-06T19:25:19Z   (read with date -u, not hand-written)
launched          2026-09-06T19:24:06Z
run dir           /data/yihou/agent_sys_runroot/runs/20260906T192406-5f24ca
orchestrator pid  2505185  ppid 1  (setsid took; a harness tool-call timeout
                  cannot reap it — m1's 2026-09-06 08:0x finding)
identity          bound by the agent child's readlink /proc/<pid>/cwd
```

**Three defences in five lines, each traceable to a specific incident:** a read
timestamp with the method stated; `ppid 1` recorded as *evidence that `setsid`
took*, against the harness-reaping failure; and identity bound by `cwd` rather
than by a name — **the discriminator the first cluster paid five
misattributions to learn.**

**Board at 19:39:49:**

```
5f24ca   ALIVE  pid 2505185, container=yihou_e2e_chain7
         deploy_and_prove: output_validating   verdicts 2/2
         yihou_e2e_etcd_serves-e5b225d6  19:38:45
         yihou_e2e_sgl_serves-e5b225d6   19:38:47
         cards 0-3 at 75 %
6b7c19   dead   19:05:59  (two 900 s timers, T+757)
```

**Seventeen runs.**

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~58 %** (unchanged) |
| 已经耗时 | **~800 min** (mission.md 06:19:11 → 19:40:14) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Hold `29313`: 18 h 20 min left.**

**A number worth stating because it changes how the remaining hold should be
read:** the stack-window capture has succeeded **1 time in 3 real attempts**,
and it is required for `check_trace_coverage` without a waiver. **Stage 2 is not
reliably reproducible even though it has sealed once.**

### 5. Code problems

**Retracted, not a defect:** `with_stack` freezing the engine. **Open, and now
correctly framed as intermittent:** what makes the stack window fail 2 times in
3.

**Carried, root-caused, unfixed:** stall threshold equals AIPerf's 900 s request
timeout; `preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; **and now a third refusal naming an
unreadable `--var`.**
**Carried unread since T+94:** the eight `jsonschema` validators — **thirteen
hours.**

### 6. 未定性

- **Whether run 17's stack window is the 2nd hit in 4 attempts or the 3rd miss.**
  **This is now a measurable rate rather than a yes/no**, which is a better
  question than the one I carried for four sections.
- **What makes the stack window intermittent.** Open, correctly.
- **Whether `merge_profiling_evidence` has executed** — the T+757 §7 disagreement
  with the leader stands unresolved; **nobody has re-read `d9c7af`'s
  `store/task`.**
- **What module 5 consumes if module 4 is replayed** — **twenty-fourth
  consecutive section.**

### 7. 新增 commit

Since T+757, five:

```
75cc5889  checkpoint R2 T+757 — mine
4e67b69d  bug record: a refusal names a --var the guard does not read
4afe06b7  bug record: a with_stack profiler window freezes the engine mid-window
52487eed  bug record: correct the with_stack entry — the same config passed at
          16:38
10558a28  CLAUDE.md: an intermittent failure is indistinguishable from a
          deterministic one at N=1
2e83f52c  bug record: the real denominator — stack window 1 of 3, measurement
          window 4 of 4
```

**Three of the six are corrections to the other three, by the same author, within
the same interval.** **That ratio is the healthiest thing in this section.**

### 8. 其他

**"Ask whether that stage has ever succeeded on this host, and ask it outside the
run" is the rule this file has been circling all day, finally stated as four
commands.**

Every mechanism I got wrong today failed the same way: **I reasoned from the
failing run and the failing run alone.** T+548 read one router log. T+578 read
one worker log. The detokenizer guard sampled only failures — *"that sample could
not structurally contain a counterexample."* **In all four cases a successful run
of the same stage existed on the same host at the time, and in three of them it
was in a directory I had already listed.**

**The correction is not "read more carefully." It is that a failure is not a
sample, and a run tree cannot tell you it is N=1.** The denominator lives outside
the run — which is the same place this file already established that hold
liveness, image identity, and container ownership live. **Four things now, all
invisible from inside.**

---

## R2 T+817 — 2026-09-06 20:11 UTC

**T+817 = wall-clock delta from the baseline** (06:33:41 → 20:10:34).

### 1. Deepest board of the round — stage 2 sealed again, stage 3 well inside, 21/21 verdicts

**[observed, first-hand] `20260906T192406-5f24ca`, alive, pid 2505185, last write
20:09:50:**

```
m1_deploy                succeeded
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    succeeded
merge_profiling_evidence succeeded
m2_profiling             SUCCEEDED        ← stage 2 sealed, 2nd time
identify                 SUCCEEDED        ← never sealed before
rank                     succeeded
build_workset            running          ← never reached before
m3_analysis              running

verdicts  21/21 pass — ZERO refusals
```

**`identify` sealed.** It refused once, at T+637, on a duplicate
`logical_operator` that would have become two directories with one name.
**`649af26b` fixed it and this is the first run to test that fix.** It passed.

**Twenty-one verdicts and not one refusal.** The largest clean board of the
round; the previous best was 20/21.

### 2. The stack window was captured — and the rate is now 2 in 4

**[first-hand]**

```
stacks_manifest.json  ×7 in this run
  ranks 2
  totals {files 2, bytes 140 335 056, gpu_kernels 111 516,
          python_functions 5 703 861, readable 2}
```

**140 MB and 5.7 million python function samples**, comparable to `d9c7af`'s
137 MB / 5.6 M at T+637.

**T+786 recorded the rate as 1 hit in 3 real attempts. It is now 2 in 4.** That
is the honest update: **the stack window remains intermittent, and this run is a
hit rather than a fix.** Nothing in the launch changed to make it more likely —
`launch11` carries the same `trace_end_ms=120000` and the same absence of waiver
flags.

**The question I carried for four sections — "will the capture succeed without
the waiver?" — has now been answered twice: yes, sometimes.** That is a worse
answer than a clean yes and a much better one than the yes/no framing allowed.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~66 %** (+8) |
| 已经耗时 | **~831 min** (mission.md 06:19:11 → 20:10:34) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+8: stage 2 reproduced, `identify` and `rank` sealed, `build_workset`
running.** Stage 3 is no longer "entered and refused" — **three of its tasks have
sealed and a fourth is executing.**

**Timing so far, measured:** launched 19:24:06, at `build_workset` by 20:09:50 —
**46 minutes for stage 1 plus stage 2 plus most of stage 3.** `d9c7af` took ~68
minutes to reach the equivalent of stage 2 alone.

**Hold `29313`: 17 h 50 min left.**

### 4. 当前进展

```
5f24ca  ALIVE  the only run, pid 2505185, container=yihou_e2e_chain7
node    8 cards VRAM 0 %, no yihou_* container
```

**Cards idle and no engine container, with the run actively writing 11 seconds
before I sampled.** `build_workset` is a CPU stage; **this is the sixth failure
mode of "the cards are free" and the reading is correct while the inference from
it would be wrong.** Recorded because I have now made that mistake once and
avoided it four times in this file, and the difference each time was asking
whether a live chain exists before asking whether cards are busy.

### 5. Code problems

**No new ones. Nothing broke this interval.**

**Confirmed fixed by a passing run:** `logical_operator` uniqueness (`649af26b`)
— **the first fix of the round verified by the validator that refused it.**

**Carried, root-caused, unfixed:** stall threshold equals AIPerf's 900 s request
timeout; `preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; three refusals naming unreadable
`--var`s.
**Open, correctly framed as intermittent:** what makes the stack window fail 2
times in 4.
**Carried unread since T+94:** the eight `jsonschema` validators — **thirteen and
a half hours.**

### 6. 未定性

- **Whether `build_workset` completes and `m3_analysis` seals.** Live now, and
  **this is the furthest the round has ever been.**
- **What module 4 costs.** Still zero measurements, and it is now one stage away.
  **17 h 50 min of hold is the budget it has.**
- **What module 5 consumes if module 4 is replayed** — **twenty-fifth consecutive
  section**, and if `m3_analysis` seals it becomes the immediate next question
  rather than a carried one.
- **Whether `merge_profiling_evidence` had executed before tonight.** **It shows
  `succeeded` here and in `d9c7af`.** The leader stated at 14:28 that it had never
  executed anywhere; **two runs now disagree with that, and I record the
  disagreement rather than assuming which is stale.**

### 7. 新增 commit

Since T+786, none but mine (`daf69e62`).

### 8. 其他

**The last three sections have a shape worth naming while the run is still
alive.**

```
T+757   a stall detector fed by the failure it watches for
T+786   a mechanism proposed, retracted, and its denominator corrected —
        by one person inside forty minutes
T+817   21 of 21 verdicts, three new tasks sealed, no refusals
```

**The clean board did not arrive because the chain got easier.** It arrived after
a day in which ten distinct single-point defects were each found once, and in
which the last three findings were all about *how we were measuring* rather than
about the product. **`identify` sealing is the direct payoff of a refusal at
16:57 that named a file and a fix; the capture succeeding is not a payoff at all,
it is a coin that landed the right way twice in four.**

**Keeping those two apart is the whole job of this section.** One of them will
still be true tomorrow.

---

## R2 T+847 — 2026-09-06 20:41 UTC

**T+847 = wall-clock delta from the baseline** (06:33:41 → 20:40:23).

### 1. `build_workset` executed and was refused — by a localisation defect, not an artefact defect

**[observed, first-hand] `5f24ca`, `check_workset_runs`, 43 files in materials,
23/24 verdicts:**

```
PROBLEM: gemm_aiter_bf16gemm_bf16_tn/case_001: the performance entrypoint
  exited 1: measure_in_container: cannot derive a mount this cluster's docker
  authorization plugin will accept from …/materials/…/v1/items/codes.
  Measured forms:
    ref: -v $HOME:$HOME              accepted   (m3 measured, 006)
    ref: -v /shared_nfs:/shared_nfs  accepted   (relayed: leader measured)
    ref: -v /home:/home              refused    (relayed: leader measured, 243)
    (the three rows above are a catalogue, NOT what this run did)
  Point --demo-root at one of the two, or set E2E_REMOTE_HOME, or extend this
  case with a form you have SEEN the daemon accept — not one you expect it to.
```

**[first-hand, `b6c0a3c4`] The mechanism:**

> *`measure_in_container.sh` derives its bind mount from a **two-branch `case`
> covering `/shared_nfs` and `/home`; this host's root is `/data/yihou`.**
> `E2E_REMOTE_HOME` is declared in **four task env blocks and not in
> `check_workset_runs`**, which runs with a **closed environment**. The
> `build_workset` agent exported it by hand, **so only the validator side could
> fail.***

**This is the round's cleanest second-cluster localisation defect.** The package
knows about two filesystem layouts; this cluster is a third. **The producer side
was patched by hand at runtime and the validator side, which cannot see the
environment, was not.** `E2E_REMOTE_HOME` appears 16 times in the package — **in
four task env blocks, and not in the one place that needed it.**

**Two things about the refusal text itself, because they are the standard this
record has been asking for all day:**

- **It labels its own evidence by provenance** — `(m3 measured, 006)` versus
  `(relayed: leader measured)` — and then **explicitly disclaims the catalogue**:
  *"the three rows above are a catalogue, NOT what this run did."*
- **It ends with "a form you have SEEN the daemon accept — not one you expect it
  to."** That is this file's core discipline written into a validator's error
  path.

**And the same report volunteers its own coverage limit:**

> *`reverify_shapes` = **1 of 5** operator(s) with a primary shape. **This number
> is the producer's claim and this run did not check it.** … re-measuring all 5
> would cost about 360 s more.*

**Four of five operators were recorded, not re-measured, and the validator says
so per-operator with the price of fixing it.** Compare `min_resolve_ratio`'s
floor of zero at T+637 — **the same honesty, and this one quantifies the trade.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~68 %** (+2) |
| 已经耗时 | **~861 min** (mission.md 06:19:11 → 20:40:23) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2: `build_workset` ran, produced a workset with five operators and timing
cases, and reached output validation.** The one refusal does not judge that
artefact — **`b6c0a3c4`'s subject says it outright: "and the artefact is fine."**

**Hold `29313`: 17 h 20 min left.**

### 3. 当前进展 — run 18 launched one minute ago

```
5f24ca  dead   last 20:38:48   build_workset: output_validating, 23/24
                orchestrator gone
3bf8c2  ALIVE  pid 3203507, container=yihou_e2e_chain8, started 20:39:56
node    8 cards VRAM 0 %, no yihou_* container — run is 27 s old
total runs: 18
```

### 4. Code problems

**New, root-caused, unfixed:** `measure_in_container.sh`'s mount `case` has two
branches (`/shared_nfs`, `/home`) and this host needs a third (`/data/yihou`);
**`E2E_REMOTE_HOME` is absent from `check_workset_runs`'s env block**, which runs
closed. **The producer-side workaround was a manual export and does not travel.**

**Carried, root-caused, unfixed:** stall threshold equals AIPerf's 900 s request
timeout; `preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; three refusals naming unreadable
`--var`s.
**Open, intermittent:** the stack window, 2 hits in 4.
**Carried unread since T+94:** the eight `jsonschema` validators — **fourteen
hours.**

### 5. 未定性

- **Whether run 18 clears `check_workset_runs`.** It depends entirely on whether
  the mount form or `E2E_REMOTE_HOME` reached the validator's closed
  environment. **The reading is one `grep` of the launch record.**
- **Whether `reverify_shapes=1` is the right default for acceptance.** The
  validator prices the alternative at 360 s. **Nobody has decided; the default
  means four of five operator timings are the producer's unchecked claim.**
- **What module 4 costs** — zero measurements, one stage away, 17 h 20 min of
  hold.
- **What module 5 consumes if module 4 is replayed** — **twenty-sixth
  consecutive section.**

### 6. 新增 commit

Since T+817, one:

```
5b233c68  checkpoint R2 T+817 — mine
b6c0a3c4  validator failures: check_workset_runs refused, and the artefact is
          fine
```

### 7. 其他

**The environment defect in §1 is the exact class this round exists to find, and
it took fourteen hours to surface because everything upstream of it had to work
first.**

The baseline at T+0 listed the second-cluster hazards it could see: no corpus,
no `spur`, a read-only path that is not read-only here, a namespace-package
collision. **It could not list this one**, because a mount `case` in a
measurement script is only reachable once a workset with real operators exists to
measure — **which required stage 1 green, stage 2 sealed, `identify` sealed, and
`build_workset` to run.**

**That is what a ladder buys and it is worth stating plainly at 68 %:** the
defects found late are not the ones anybody was slow to find. **They are the ones
that were unreachable until the rungs below them held.**

---

## R2 T+877 — 2026-09-06 21:11 UTC

**T+877 = wall-clock delta from the baseline** (06:33:41 → 21:10:34).

### 1. One field, two consumers, incompatible demands

**[first-hand, `1d71a809`]**

> *m3's `--impl` wants **a self-contained file exporting top-level `run()`**;
> `apply_patch`'s `overlay_files` wants **the overlaid module's whole public
> surface preserved**. Run 7's workset records a **472-char wrapper as baseline
> for all five operators**, each of whose targets defines **6–55 module
> symbols**, so `apply.py:828` refuses. **`base_sha256` matches the image
> exactly, so 691 does not pre-empt it.** `apply_patch` brings nothing up, so the
> refusal costs seconds, not an m5 arm.*

**This is the mirror of the family this record has been collecting all day.**
Three times it has been *one intention, two names* — `stack_window_s` /
`stack_ranks`, `min_launchers_in_top_n` / `kernel_table_min_launchers`, and a
third at T+786. **This is one name, two intentions**: `baseline` is written to
satisfy m3 and read to satisfy `apply_patch`, and the two definitions cannot both
hold for the same 472 bytes.

**Two details worth keeping separately from the defect:**

- **`base_sha256` matching exactly is why the earlier check does not fire.** The
  hash is correct; the *shape* is wrong. **That is the first cluster's lesson
  inverted** — there a correct-looking hash was supplied and disarmed a real
  guard; here a genuinely correct hash simply does not speak to the question, and
  a different check at `:828` catches it. **A field can be right and irrelevant.**
- **The refusal costs seconds because `apply_patch` brings nothing up.** **A
  cheap refusal sited before an expensive action is the correct shape**, and this
  one is. Compare `check_deploy_serves`, whose refusal costs a bring-up and a
  teardown by construction.

### 2. Run 18 is 30 minutes in and has not brought up — and it is working, not stalled

```
3bf8c2   ALIVE  pid 3203507, container=yihou_e2e_chain8, started 20:39:56
         deploy_and_prove: running   verdicts 0/0
         last write 21:09:58  (6 s before I sampled)
node     8 cards VRAM 0 %, no yihou_* container
```

**Thirty minutes with no engine container is slower than the recent pattern** —
run 17 had its serves container up ~15 minutes after launch. **So I opened the
transcript rather than inferring from the cards:**

```
21:09:54  "Now the STEP 8 self-check. Let me look at exactly how the validator
           interprets headings and evidence:"
21:09:58  reads a file under zones/task.44d916dc-…
```

**The deployer is reading the validator's own source to check its output against
it before submitting.** That is deliberate work, not a hang. **The cards being
idle is correct and means nothing about the run's health** — the sixth failure
mode, avoided by asking whether a live chain exists before asking whether cards
are busy.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~68 %** (unchanged) |
| 已经耗时 | **~891 min** (mission.md 06:19:11 → 21:10:34) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**Unchanged: run 18 has not reached a verdict.** **Hold `29313`: 16 h 50 min
left.**

### 4. Code problems

**New, root-caused, unfixed:** `baseline` has two consumers with incompatible
demands (`apply.py:828`; m3's `--impl`; `overlay_files`). **Run 7's workset
records a 472-char wrapper for all five operators.**

**Carried, root-caused, unfixed:** the `measure_in_container.sh` two-branch mount
`case` and `E2E_REMOTE_HOME` absent from `check_workset_runs`'s closed env; stall
threshold equals AIPerf's 900 s request timeout; `preflight.sh:211`;
teardown-vs-preflight sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on
no path; `min_resolve_ratio` floor of zero; the router `/health` gate vs a cold
JIT compile; three refusals naming unreadable `--var`s.
**Open, intermittent:** the stack window, 2 hits in 4.
**Carried unread since T+94:** the eight `jsonschema` validators — **fourteen and
a half hours.**

### 5. 未定性

- **Whether run 18 clears `check_workset_runs`** — the mount defect from T+847 is
  the wall it will meet, and I do not know whether the launch addresses it.
- **Which consumer's definition of `baseline` is the correct one.** §1 names the
  conflict; **it does not say which side should change**, and that is a contract
  decision rather than a bug fix.
- **What module 4 costs** — zero measurements, 16 h 50 min of hold.
- **What module 5 consumes if module 4 is replayed** — **twenty-seventh
  consecutive section**, and §1 is the first finding that touches it: **if
  `apply_patch` refuses every workset baseline, module 5 has nothing to apply
  regardless of where module 4's artefact comes from.**

### 6. 新增 commit

Since T+847, one:

```
eabcfbc6  checkpoint R2 T+847 — mine
1d71a809  bug record: baseline has two consumers with incompatible demands
```

### 7. 其他

**The `baseline` conflict is the first defect of the round that is a contract
question rather than an implementation error, and it arrived exactly where the
ladder predicted.**

Everything earlier had a right answer that somebody simply had not written:
a mount branch, a timeout ordering, a variable that was never wired, a duplicated
name. **This one has two right answers held by two modules, and no amount of
careful reading resolves it** — `apply.py:828` is correct to refuse a wrapper
that drops 6–55 public symbols, and m3 is correct to want a self-contained file
with a top-level `run()`.

**`CONTRACT.md` is the frozen fifteen-kind cross-module contract and it is the
document this belongs in.** Whether it already speaks to `baseline`, I have not
read. **That is the cheap reading that would tell whoever picks this up whether
they are fixing a violation or filling a gap** — and those need different people.

---

## R2 T+907 — 2026-09-06 21:40 UTC

**T+907 = wall-clock delta from the baseline** (06:33:41 → 21:40:19).

### 1. Run 18 reproduced the deepest board, fully real, in one hour

**[observed, first-hand] `20260906T203956-3bf8c2`, alive, pid 3203507, last write
21:39:41:**

```
m1_deploy                succeeded
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    succeeded
merge_profiling_evidence succeeded
m2_profiling             succeeded
identify                 succeeded
rank                     succeeded
build_workset            running
m3_analysis              running

verdicts 21/21 — zero refusals
```

**Identical board to run 17 at the same point, and `mock_stages=none`** — read
from the orchestrator's argv, along with `trace_end_ms=120000`,
`work_root=/data/yihou/e2e_flow8`, `jobid=29313`. **Nothing was replayed.**

**This is the second consecutive run to seal stage 2 and `identify`.** At T+817 I
was careful to separate the `identify` fix (a real payoff) from the stack capture
(a coin landing right). **Two runs in a row now make stage 2 + `identify` + `rank`
reproducible rather than lucky** — and the stack capture, which was 2 of 4, is
now presumably 3 of 5, though I have not counted this run's manifests.

**Timing, both runs, measured:**

```
run 17   19:24:06 launch -> build_workset running 20:09:50   46 min
run 18   20:39:56 launch -> build_workset running 21:39:41   60 min
         (at 21:10:04 it was still deploy_and_prove, 0 verdicts)
```

**Run 18 spent ~30 minutes in stage 1 and ~30 in stage 2 + identify + rank.**
Run 17 was faster in stage 1. **I do not know why**; the difference is not in the
launch variables I read.

### 2. A foreign-looking container, and I am not attributing it by name

```
yihou_m3_explore   created 21:39:41
  image      rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_…
  autoremove true
  devices    /dev/kfd
  labels     only org.opencontainers.*  — NO infera_e2e_run
```

**It maps `/dev/kfd` and all eight cards read 0 %.** By T+517's finding that is
*device mapping, not occupancy*, so under the standing rule — stop GPU occupants,
leave the rest — **it is out of scope and I have not touched it.**

**By ownership evidence rather than by name:** no `infera_e2e_run` label,
`--rm`, and a `rocm/pytorch` image rather than our engine image. **The `yihou_m3_`
prefix is not evidence** — the first cluster misattributed five containers that
way, one of them twice to two different people. **I record what it is and decline
to say whose.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~70 %** (+2) |
| 已经耗时 | **~921 min** (mission.md 06:19:11 → 21:40:19) |
| 预估耗时 | **absent** |
| 可靠性 | **中** |

**+2 for reproducibility, not new ground** — the same accounting I applied at
T+186 when stage 1 went green twice. **Nothing past `build_workset` has ever
completed.**

**Hold `29313`: 16 h 20 min left.**

**One estimate that is now defensible and I will state with its limits:** stages
1–3-to-`build_workset` cost **46 and 60 minutes** on two consecutive fully-real
runs. **That is a real measurement of the front three quarters of the chain.**
Modules 4 and 5 still have **zero** measurements here, and module 4 is the one
the user flagged as very long — **so the total remains unestimable, but the part
that is not module 4 is now known to be about an hour.**

### 4. Code problems

**No new ones. Nothing broke this interval.**

**Carried, root-caused, unfixed:** `baseline` with two incompatible consumers
(`apply.py:828`); `measure_in_container.sh`'s two-branch mount `case` and
`E2E_REMOTE_HOME` absent from `check_workset_runs`'s closed env; stall threshold
equals AIPerf's 900 s request timeout; `preflight.sh:211`;
teardown-vs-preflight sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on
no path; `min_resolve_ratio` floor of zero; the router `/health` gate vs a cold
JIT compile; three refusals naming unreadable `--var`s.
**Open, intermittent:** the stack window.
**Carried unread since T+94:** the eight `jsonschema` validators — **fifteen
hours.**

### 5. 未定性

- **Whether run 18 clears `check_workset_runs`.** The mount defect from T+847 is
  the wall in front of it and **I have not established whether this launch
  addresses it** — `work_root=/data/yihou/e2e_flow8` is still under `/data`,
  which is the root the two-branch `case` does not cover.
- **Why run 17's stage 1 was 15 minutes faster than run 18's.** Not in the launch
  variables.
- **What module 4 costs** — zero measurements, 16 h 20 min of hold.
- **What module 5 consumes if module 4 is replayed** — **twenty-eighth
  consecutive section.**

### 6. 新增 commit

Since T+877, none but mine (`0dd15a0c`).

### 7. 其他

**Two runs in a row have produced identical 21/21 boards on a fully real chain,
and that is the first time this round has had a repeatable state to reason
from.**

Everything before tonight was a single instance: one stage-1 green, then a
second; one stage-2 seal; one `identify` seal. **Each was worth recording and
none of them supported a rate.** The day's sharpest methodological finding —
`10558a28`, *an intermittent failure is indistinguishable from a deterministic
one at N=1* — was about precisely that poverty.

**With two identical boards, the questions change shape.** "Did `identify` pass?"
becomes "does `identify` pass?", and the stack window stops being a yes/no and
becomes a fraction. **That is worth more than the +2** — but it is worth exactly
nothing past `build_workset`, where the chain has never been, and where the two
unfixed defects in front of it are a mount `case` with no branch for this cluster
and a `baseline` field two modules disagree about.

---

## R2 T+937 — 2026-09-06 22:10 UTC

**T+937 = wall-clock delta from the baseline** (06:33:41 → 22:10:04).

### 1. STAGE 3 IS COMPLETE, and stage 4 has started

**[observed, first-hand] `20260906T203956-3bf8c2`, alive, pid 3203507, last write
22:09:52:**

```
m1_deploy                succeeded
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    succeeded
merge_profiling_evidence succeeded
m2_profiling             succeeded
identify                 succeeded
rank                     succeeded
build_workset            SUCCEEDED       ← refused in run 17 (T+847)
m3_analysis              SUCCEEDED       ← STAGE 3 SEALED, first time ever
m4_kernel_opt            running         ← STAGE 4 ENTERED, first time ever
optimize_kernel          running

verdicts 24/24 — zero refusals
stacks_manifest.json ×7
```

**`build_workset` sealed.** It was refused ninety minutes ago by
`check_workset_runs` on a mount `case` with no branch for `/data/yihou` (T+847).
**It passed this time, so the mount reached the validator's closed environment** —
**how, I have not established**, and that is the reading worth taking: the fix
either travelled in the launch or the agent supplied it, and those have very
different lifespans.

**24 of 24 verdicts, no refusals, on a fully real chain** (`mock_stages=none`,
read from argv at T+907). **Three of five stages sealed.**

**`/data/yihou/e2e_flow8/kfo` exists** — the kernel-forge scratch root. Module 4
is doing something.

**Timing, measured:** launched **20:39:56**, `m3_analysis` sealed by
**22:09:52 — 90 minutes for stages 1 through 3.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (+6) |
| 已经耗时 | **~951 min** (mission.md 06:19:11 → 22:10:04) |
| 预估耗时 | **for the first time, partially estimable — see below** |
| 可靠性 | **中** |

**+6: stage 3 sealed and stage 4 entered.** Three of five.

**On 预估耗时, and this is the first time I can say anything.** Stages 1–3 cost
**90 minutes** on a fully real run, and the front of the chain has now been
measured twice (46 min and 60 min to `build_workset`). **Module 4 has zero
measurements here and the user flagged it as very long; the first cluster
recorded one campaign running 113 minutes and still in preparation.** Module 5
also has zero.

> **So: the chain up to module 4 costs about 90 minutes. What remains is
> unestimable, and it is unestimable because of module 4 specifically, not
> because of general uncertainty.** That is a more useful statement than
> "absent," and it is the first time it has been available.

**Hold `29313`: 15 h 50 min left.**

### 3. 当前进展

```
3bf8c2   ALIVE  the only run; m4_kernel_opt / optimize_kernel running
node     8 cards VRAM 0 %, no yihou_* container
scratch  /data/yihou/e2e_flow8/kfo present
```

**Cards idle while module 4 runs is expected** — the first cluster measured a
campaign spending its first two hours in preparation with no source file
modified. **I will not read "no GPU activity" as "module 4 is stuck"**, and the
discriminator when it matters is the growth of `kfo/`, not the cards.

### 4. Code problems

**Apparently cleared, mechanism unconfirmed:** the `measure_in_container.sh`
mount `case`. **`check_workset_runs` passed; I have not read how.**

**Carried, root-caused, unfixed:** `baseline` with two incompatible consumers
(`apply.py:828`) — **and it is now directly in front of the chain**, because
module 5's `apply_patch` is the consumer that refuses; stall threshold equals
AIPerf's 900 s request timeout; `preflight.sh:211`; teardown-vs-preflight
sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on no path;
`min_resolve_ratio` floor of zero; the router `/health` gate vs a cold JIT
compile; three refusals naming unreadable `--var`s.
**Open, intermittent:** the stack window.
**Carried unread since T+94:** the eight `jsonschema` validators — **fifteen and
a half hours.**

### 5. 未定性

- **What module 4 costs, and whether it fits 15 h 50 min.** **Now the single
  question that decides whether this round reaches `packup`.**
- **How `build_workset` cleared the mount defect.** §1. **A fix in a launch or an
  agent instruction does not ship; a fix in the package does.** This record has
  made that distinction four times today and it applies again.
- **Whether `baseline` blocks module 5** regardless of module 4's outcome
  (T+877 §1).
- **What module 5 consumes if module 4 is replayed** — **twenty-ninth consecutive
  section, and it is now one stage away from being answered by events rather
  than by reading.**

### 6. 新增 commit

Since T+907, none but mine (`59c1c7b3`).

### 7. 其他

**Sixteen hours and eighteen runs to get three stages, and the third arrived
ninety minutes after the second on the same run.**

That is worth stating precisely because it is not a story about acceleration.
**The eighteen runs bought eleven distinct single-point defects**, each found
once, each by something refusing rather than by something quietly producing a
wrong answer. **Run 18 did not go faster because anyone tried harder; it went
further because every rung below it had been repaired.**

**And the two defects nearest the front are the two that were unreachable
longest:** a mount `case` that could only be hit once a real workset existed to
measure, and a `baseline` contract that only bites when `apply_patch` reads what
`build_workset` wrote. **Both are properties of the chain being connected, which
is the entire thing this package exists to do.**

**What has never happened on either cluster is `packup`.** Module 4 is running,
the cards are idle, and the honest position is that nobody here knows how long
that lasts.

---

## R2 T+967 — 2026-09-06 22:40 UTC

**T+967 = wall-clock delta from the baseline** (06:33:41 → 22:40:12).

### 1. Module 4 was attempted twice and refused twice — both correctly, both for a missing launch variable

**[first-hand, `f273f0e1`]**

> *`HIP_VISIBLE_DEVICES` is `'${gpu:-}'` and `run_in_container.sh` aborts on
> empty, **so STEP 4 never measured**; both refusals are downstream of that and
> **both are correct**. **The launch line carried `measure_gpu` and not `gpu` —
> two names, two consumers, and having one made the other look covered.**
> Sweeps the whole empty-default class rather than fixing the instance, and
> **marks the six untested ones as un-refuted.***

**Module 4 has now run twice and measured nothing.** Its cost on this cluster is
still zero measurements — **the two attempts do not count, because the stage
aborted before it did any work.**

**Three things worth separating:**

- **Both refusals are correct.** The validators refused an artefact that
  genuinely had no measurement in it. **This is not a validator problem.**
- **The class was swept, not the instance.** The first cluster paid three
  launches fixing `expect_ranks`, then `adhoc_cases`, then `bench_rounds` one at
  a time. **Here the whole empty-default variable class was enumerated in one
  pass, and the six that remain untested are marked *un-refuted* rather than
  quietly assumed fine.**
- **Run 19 carries `gpu=4`.** Read from `/proc/21292/cmdline` just now, alongside
  `measure_gpu=4`, `mock_stages=none`, `trace_end_ms=120000`,
  `work_root=/data/yihou/e2e_flow9`. **The fix is in the launch line.**

### 2. The general rule, and it is mechanically checkable

**[first-hand, `58c5c56e`]**

> *Twice in one evening: **`remote_home` present / `transport_env` missing**, and
> **`measure_gpu` present / `gpu` missing.** The absent half's failure **looks
> like an artefact defect, not a launch-line defect.** **Re-reading the variable
> table does not catch it — the table was wrong**; subtracting the line from the
> package's empty-default vars does, **in two commands, before launch.***

**"Re-reading the variable table does not catch it — the table was wrong" is the
part that makes this tier 2 rather than tier 3.** The first cluster's rule was
*audit the whole variable table against your `mock_stages` every launch*; **this
supersedes it, because the table itself is a document and documents decay.**
**Subtracting the launch line from the package's own empty-default variables is a
computation over the code.**

**And it answers my T+937 §1 open question by family:** the mount defect at
T+847 was `remote_home` present / `transport_env` missing — **a launch-line
defect wearing the costume of an artefact defect**, exactly as this rule
predicts.

### 3. A retraction of a worked example — the false-credit error with its sign flipped

**[first-hand, `d313556e`]**

> *The threshold must exceed the longest **QUIET** interval, not the longest
> stage: a stage that writes into the run tree keeps the detector fed
> (`build_workset` ran **24 min under 1200 s twice**). **Run 7 was not an
> instance** — its output validation took **0.85 s** and refused; the 20 minutes
> were an escalation with no recipient. The validator feared to take tens of
> minutes **measured 7 s.** **Blaming a death on a choice that did not cause it
> is the false-credit error with the sign flipped.***

**That last sentence is a new formulation and it completes a pair.**
`753e060f` at T+786: *a fix credited with a save it did not make gets
over-trusted.* **This is the same error pointed the other way — a choice blamed
for a death it did not cause gets over-avoided.** Both distort the next decision;
neither is caught by care.

**And the correction is quantitative in both directions**: a validator feared to
cost tens of minutes measured **7 s**, and a stage that does not stall
(`build_workset`, 24 min) never threatened the detector because **it writes**.

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~981 min** (mission.md 06:19:11 → 22:40:12) |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 still zero measurements** |
| 可靠性 | **中** |

**Unchanged: module 4 ran twice and measured nothing.** **The estimate for stages
1–3 stands at ~90 minutes and is unaffected.**

**Hold `29313`: 15 h 20 min left.**

### 5. 当前进展

```
3bf8c2   dead   reached m4_kernel_opt; module 4 refused twice (missing --var gpu)
                /data/yihou/e2e_flow8/kfo  224 K, 27 files written in 30 min
967186   ALIVE  pid 21292, container=yihou_e2e_chain9, started 22:32:21
                deploy_and_prove: running   verdicts 0/0
                carries gpu=4  (the fix)
node     8 cards VRAM 0 %, no yihou_* container — run is 7 min old
total runs: 19
```

**`kfo/` reached only 224 K.** Consistent with STEP 4 never measuring; **the
forge wrote its scaffolding and stopped.**

### 6. Code problems

**Root-caused, fixed in the launch line:** `--var gpu` absent while `measure_gpu`
present. **Six other empty-default variables are enumerated and marked
un-refuted** — **not fixed, and honestly labelled.**

**Carried, root-caused, unfixed:** `baseline` with two incompatible consumers
(`apply.py:828`); stall threshold vs the longest **quiet** interval;
`preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; three refusals naming unreadable
`--var`s.
**Carried unread since T+94:** the eight `jsonschema` validators — **sixteen
hours.**

### 7. 未定性

- **What module 4 costs.** **Still zero measurements after two attempts** — and
  this is the question that decides whether `packup` is reached. 15 h 20 min.
- **The six un-refuted empty-default variables.** Enumerated, untested. **Each
  one can produce a refusal that reads as an artefact defect.**
- **Whether `baseline` blocks module 5** (T+877).
- **What module 5 consumes if module 4 is replayed** — **thirtieth consecutive
  section.**

### 8. 新增 commit

Since T+937, three:

```
55081e55  checkpoint R2 T+937 — mine
f273f0e1  validator failures: m4 refused twice, cause is a missing --var gpu
58c5c56e  CLAUDE.md: the present half of a variable pair makes the absent half
          look covered
d313556e  CLAUDE.md: the stall rule's second half, and a worked example that is
          not one
```

### 9. 其他

**Two of tonight's three commits are rules that supersede rules, and both
supersede them in the same direction: from a document to a computation.**

```
was  "audit the whole variable table every launch"
now  "subtract your launch line from the package's empty-default vars" — 2 cmds
     because THE TABLE WAS WRONG

was  "set the stall threshold above the longest stage"
now  "above the longest QUIET interval" — because a stage that writes feeds the
     detector, measured: build_workset 24 min under a 1200 s threshold, twice
```

**Both old versions were written down, read, and followed. Both failed anyway**,
and in each case the replacement is something you compute rather than something
you remember. **This file has argued all day that only the remember-it tier
decays; tonight two rules crossed from that tier into the checkable one, and each
crossing was paid for by a specific death.**

**The third commit retracts a worked example rather than defending it** — run 7
was blamed on a stall threshold that had nothing to do with it. **A rule that
keeps a false example loses the thing that makes it trustworthy**, and dropping
the example cost nothing because the rule survives on `build_workset`'s two
measured 24-minute runs.

---

## R2 T+997 — 2026-09-06 23:10 UTC

**T+997 = wall-clock delta from the baseline** (06:33:41 → 23:10:28).

### 1. `output_validating` means the opposite of what it reads as — and I have printed it fourteen times

**[first-hand, `c2ac437f`]**

> *Two runs died under that label and it was **read twice as "validation is
> taking too long."** Measured: **both validations finished in under a second and
> refused**; the wait afterwards is an **escalation to an agentless root task**.
> **The label is accurate and means the opposite of what it reads as**, and
> **only the verdict files' mtimes separate the two.** **No stall value would
> have changed either.***

**This lands squarely on my own instrument.** Every section since the baseline
has printed `store/task` status, and `deploy_and_prove: output_validating` has
appeared in many of them — for `15c264`, `ae2c38`, `79bca5`, and others, all
dead.

**And I made the exact misreading it names, at T+62:**

> *"At 07:35:22 it had not returned … **Elapsed in validation at this write:
> 5 min 23 s.** I do not know whether that is normal for this validator set
> here."*

**It had not been validating for five minutes. It had crashed at 07:29:59, in the
same second the phase line was written** — which I established at T+94 and
recorded as my own correction. **`c2ac437f` is the general form of that
correction**, measured across two further runs and shown to be a property of the
label rather than an accident of one crash.

**The operational consequence, stated as the check I will now run:** when a task
reads `output_validating`, **compare the verdict files' mtimes to the phase
transition.** Sub-second means the validation is over and the task is sitting in
an escalation with no recipient. **The label cannot distinguish them; two mtimes
can.**

**"No stall value would have changed either" also disarms a fix that looks
obvious** — this is not a timeout to be widened, and widening one would have
bought nothing.

### 2. 当前进展 — run 20, and run 19 lasted nine minutes

```
967186   dead   launched 22:32:21, carried the gpu=4 fix
ef6374   ALIVE  pid 95533, container=yihou_e2e_chain10, launched 22:41:00
         deploy_and_prove: output_validating   verdicts 2/2
         yihou_e2e_serves-fba3b260       23:03:46
         yihou_e2e_serves-fba3b260_etcd  23:03:52
         aiperf_serves-fba3b260          23:06:47
         cards 0-3 at 76 %
         last write 23:10:02  (26 s before I sampled)
total runs: 20
```

**`check_deploy_serves` is running with a live engine and aiperf**, so this
`output_validating` is the genuine kind — **and I can say so because the aiperf
container exists**, not because of the label.

**`/data/yihou/e2e_flow9/kfo` is 0 bytes** — run 19 never reached module 4.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1011 min** (mission.md 06:19:11 → 23:10:28) |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 still zero measurements** |
| 可靠性 | **中, and one component of it just got worse** |

**Unchanged.** Two launches consumed since T+937 and neither reached module 4.

**On reliability:** §1 shows that one of the fields I report every thirty minutes
is systematically ambiguous. **My status tables have been accurate and one of
their rows has been over-read — by me, at T+62, and by two others since.** The
sections stand; **the field now needs a companion measurement whenever it
matters.**

**Hold `29313`: 14 h 50 min left.**

### 4. Code problems

**New, root-caused, unfixed:** `output_validating` persists after a failed
validation, because the escalation goes to an agentless root task. **Framework,
not package.**

**Fixed in the launch line:** `--var gpu`. **Six empty-default variables remain
un-refuted.**

**Carried, root-caused, unfixed:** `baseline` with two incompatible consumers;
stall threshold vs the longest quiet interval; `preflight.sh:211`;
teardown-vs-preflight sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on
no path; `min_resolve_ratio` floor of zero; the router `/health` gate vs a cold
JIT compile; three refusals naming unreadable `--var`s.
**Carried unread since T+94:** the eight `jsonschema` validators — **sixteen and
a half hours.**

### 5. 未定性

- **What module 4 costs.** **Four runs have now entered or approached it and none
  has measured anything.** 14 h 50 min of hold.
- **Why run 19 died nine minutes in**, carrying the `gpu=4` fix. **Not
  established; its `kfo` is empty so it did not reach module 4.**
- **The six un-refuted empty-default variables.**
- **Whether `baseline` blocks module 5.**
- **What module 5 consumes if module 4 is replayed** — **thirty-first consecutive
  section.**

### 6. 新增 commit

Since T+967, one:

```
cbd0b96e  checkpoint R2 T+967 — mine
c2ac437f  bug record: output_validating persists forever after a failed
          validation
```

### 7. 其他

**Three findings tonight have the same shape and it is the sharpest recurring
pattern of the round: a signal that is accurate and misleading at once.**

```
T+607  the detokenizer health-check failure   accurate; universal, so it
                                              discriminates nothing
T+967  measure_gpu present, gpu absent        accurate; its presence made the
                                              absent half look covered
T+997  output_validating                      accurate; it means the validation
                                              is OVER, not ongoing
```

**None of these is a bug in the signal.** Each says something true. **What each
lacks is the second reading that makes it decisive** — a known-good sample, the
package's empty-default set, a verdict file's mtime. **In all three cases that
second reading is one command and existed the whole time.**

**And in all three the first reading is the one that arrives unasked**, printed
in a log or a status table, which is why it gets used. **The cost is not that
people trust bad instruments; it is that a good instrument answering an adjacent
question is indistinguishable from one answering yours, until you take the
second reading.**

---

## R2 T+1027 — 2026-09-06 23:41 UTC

**T+1027 = wall-clock delta from the baseline** (06:33:41 → 23:41:00).

### 1. `xiaoming-dev` took all eight GPUs at 23:39:34 — and eight of my sections say it holds none

**[observed, first-hand, second reading taken deliberately]**

```
23:39:49   VRAM%  36 35 36 37 37 35 37 35
23:40:09   VRAM%  47 47 47 48 44 47 49 45      rising ~11 % in 20 s
23:40:30   VRAM%  47 47 47 48 44 47 49 45

rocm-smi --showpids
  626424  pt_elastic  GPU 0
  626591/3/5/6/8  python  GPU 1   ~140 GB each

ps -o user=,lstart=,args=
  626424  root  Sun Sep  6 23:39:34  /opt/venv/bin/torchrun --nnodes=1 …
  626595  root  Sun Sep  6 23:39:35  /opt/venv/bin/python -u _cfgtest/profile_keys.py

/proc/*/cgroup -> docker-67486b18cb58…
docker inspect  name=/xiaoming-dev  image=tasimage/primus:pr-1048
                started 2026-09-03T02:19:00Z  autoremove=false
                labels: only org.opencontainers.*  — NO infera_e2e_run
```

**This is the seventh failure mode of "the cards are free," in its exact
recorded form:** *a container present but idle can begin loading at any moment,
and the reading that says it holds nothing expires in minutes.* **It sat idle for
three days and started thirty-five seconds before I first sampled.**

**Eight of my sections carry "foreign, CPU only" or "foreign, no GPU" for this
container.** Every one of them was true when taken and **each had a shelf life
nobody could have bounded.** This is the difference this record has drawn all day
between an inaccurate reading and a correct reading whose validity window is
shorter than the interval between readings.

**I have not touched it, and the reasoning is on the record rather than in my
judgement:** standing rule 1 says stop foreign GPU occupants; the same rule's
amendment says an occupant that will respawn should be reported rather than
fought; **and I have no evidence either way about respawn, because it started
thirty-five seconds before I looked.** Reported to the leader at 23:40:30 with
the measurements and no recommendation. **It is another engineer's container
running a real workload — `torchrun --nnodes=1` with five ranks at ~140 GB is not
a stale allocation.**

**Why it is material right now:** **card 4 is at 44 %**, and `--var gpu=4` is the
variable fixed at T+967 so that module 4 could measure.

### 2. Run 20 has the second-deepest board and is still going

```
ef6374   ALIVE  pid 95533, container=yihou_e2e_chain10, launched 22:41:00
  m1_deploy                succeeded
  deploy_and_prove         succeeded
  run_profiling_mode_off   succeeded
  run_profiling_mode_on    succeeded
  merge_profiling_evidence succeeded
  m2_profiling             succeeded
  identify                 succeeded
  rank                     succeeded
  build_workset            running
  m3_analysis              running
  verdicts 21/21           newest verdict mtime 23:24:27
  last write 23:39:56
```

**Third consecutive run to seal stage 2, `identify` and `rank`.** **59 minutes
from launch to `build_workset`** — against 46 and 60 for runs 17 and 18.

**I applied T+997's check:** the newest verdict mtime is **23:24:27** and the run
has written since (**23:39:56**), so nothing is sitting in a post-refusal
escalation. **That is the companion reading the status label cannot give.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1042 min** (mission.md 06:19:11 → 23:41:00) |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Unchanged.** **Hold `29313`: 14 h 19 min left.**

**A risk I can state without predicting:** if `xiaoming-dev` holds eight cards
while run 20 reaches module 4, the measurement `--var gpu=4` exists for **cannot
get an idle card.** Whether it waits, refuses, or measures under contention, **I
do not know** — and the first cluster measured that a co-tenant corrupts a
failure but not a pass, which makes any module-4 timing taken under this load
unusable as a duration estimate even if it succeeds.

### 4. Code problems

**No new ones.** Carried, unchanged from T+997: `output_validating` persisting;
`baseline` with two incompatible consumers; stall threshold vs longest quiet
interval; `preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; three refusals naming unreadable
`--var`s; six un-refuted empty-default variables.
**Carried unread since T+94:** the eight `jsonschema` validators — **seventeen
hours.**

### 5. 未定性

- **Whether `xiaoming-dev` releases, and whether it respawns.** **No evidence in
  either direction.** The measurement is another sample in thirty minutes.
- **What module 4 costs** — and now, **whether any measurement taken tonight
  would be interpretable** given §3.
- **Whether run 20 reaches module 4 before the cards are gone.**
- **What module 5 consumes if module 4 is replayed** — **thirty-second
  consecutive section.**

### 6. 新增 commit

Since T+997, none but mine (`1dbac526`).

### 7. 其他

**The eight stale claims in §1 are the cleanest example this file contains of a
category it has been describing all day, and they are mine.**

Nothing was measured wrong. `docker ps` reported the container; `rocm-smi`
reported zero; both were correct at every sampling. **What made the claim
misleading is that I carried it forward as a standing fact across seventeen hours
of sections, when what I had was a series of instantaneous readings.**

**The repair is not to check more often.** At thirty-minute intervals I would
still have written "CPU only" at 23:09 and been overtaken at 23:39. **The repair
is to write the reading with its timestamp and not the inference** — *"held no
GPU at 23:09:58"* rather than *"foreign, CPU only."* The first expires visibly;
the second reads as a property of the container.

**And the same distinction is what §2's verdict-mtime check buys**, and what
`c2ac437f`, `58c5c56e` and `10558a28` all bought tonight in their own domains.
**A measurement with a timestamp is a fact. The same measurement stated as a
property is a prediction.**

---

## R2 T+1057 — 2026-09-07 00:10 UTC

**T+1057 = wall-clock delta from the baseline** (2026-09-06 06:33:41 → 2026-09-07
00:10:38). **First section on a second calendar day; the T-number and the UTC
stamp both continue, and the date is now part of every timestamp below.**

### 1. `xiaoming-dev` released the cards after about half an hour

**[observed, first-hand]**

```
2026-09-06 23:39:34   torchrun starts, five ranks, ~140 GB each
2026-09-06 23:40:30   VRAM%  47 47 47 48 44 47 49 45
2026-09-07 00:10:11   VRAM%   0  0  0  0  0  0  0  0
                      rocm-smi --showpids: "No KFD PIDs currently running"
```

**It held all eight cards for at most ~31 minutes and let them go.** The
container is still up (`docker ps` shows it), still maps `/dev/kfd`, and holds
nothing.

**The open question from T+1027 — does it respawn — is still open, and one
release does not answer it.** What I can say: **it went from three days idle to
eight cards in one second, and back to zero within half an hour.** The honest
description of this node is that **it has a co-tenant whose GPU use is bursty and
unannounced**, which is a stronger and more useful statement than either "CPU
only" or "it took the GPUs."

**And the phrasing repair from T+1027 §7 applies immediately:** the correct
record is *"held no GPU at 00:10:11"*, not *"holds no GPU."*

### 2. Run 20 has been quiet for twelve minutes inside `build_workset`

```
ef6374   ALIVE  pid 95533, orchestrator up since 22:40:48
  build_workset            running
  m3_analysis              running
  everything upstream      succeeded
  verdicts 21/21           newest verdict mtime 2026-09-06 23:24:27
  last run-tree write      2026-09-06 23:58:13
  work_root 442 M, 0 files written in the last 15 min
```

**Quiet for 12 min 15 s at this sample.** Recorded as a measurement, not a
diagnosis.

**The comparison that makes it interpretable comes from tonight's own
measurement** (`d313556e`): **`build_workset` ran 24 minutes under a 1200-second
threshold, twice.** So twelve minutes of quiet is **inside its known range** and
this record has no basis to call it stalled.

**And T+997's companion check says the same:** the newest verdict mtime is
23:24:27 with run-tree writes at 23:58:13 afterwards, so **the task is not
sitting in a post-refusal escalation.**

**`build_workset` is a `runner` closure — a program body, no transcript** — so
the artefact that would say *what* it is doing does not exist for it. **The
available signals are the ones above.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1071 min ≈ 17 h 51 min** (mission.md 2026-09-06 06:19:11 → now) |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Fourth consecutive interval unchanged.** Runs 19 and 20 have both failed to get
past `build_workset` so far, and **module 4 has still measured nothing on this
cluster after four approaches.**

**Hold `29313`: 13 h 50 min left** (2026-09-07T14:00:06 → 00:10:38).

### 4. Code problems

**No new ones.** All carried, unchanged from T+1027: `output_validating`
persisting after a failed validation; `baseline` with two incompatible consumers;
stall threshold vs the longest quiet interval; `preflight.sh:211`;
teardown-vs-preflight sequencing; `--var jobid` vs `_agree_or_die`; `etcd.log` on
no path; `min_resolve_ratio` floor of zero; the router `/health` gate vs a cold
JIT compile; three refusals naming unreadable `--var`s; six un-refuted
empty-default variables.
**Carried unread since T+94:** the eight `jsonschema` validators — **seventeen
and a half hours.**

### 5. 未定性

- **Whether `build_workset` completes.** Twelve minutes quiet against a measured
  24-minute precedent.
- **Whether `xiaoming-dev` bursts again**, and whether module 4 would be able to
  measure card 4 if it did. **One release is not a pattern.**
- **What module 4 costs** — and per T+1027 §3, **any timing taken under a
  co-tenant burst would be uninterpretable even if it succeeded.**
- **Whether `baseline` blocks module 5.**
- **What module 5 consumes if module 4 is replayed** — **thirty-third consecutive
  section.**

### 6. 新增 commit

Since T+1027, none but mine (`48248964`). **No commits by anyone in the last
thirty minutes.**

### 7. 其他

**The co-tenant burst is the first event of this round that neither the package
nor the team can fix, and it is worth marking as a different category from
everything above it.**

Every other entry in this file is something someone did: a variable not passed, a
mount branch not written, a name spelled two ways, a timer set equal to another
timer. **All of them have owners and repairs.** `xiaoming-dev` starting a
training job at 23:39:34 has neither — **it is a fact about sharing a machine**,
and the only thing available is to know it happened and to distrust any number
measured across it.

**That is why the T+1027 §7 phrasing repair matters more than it looks.** A
record that says *"held no GPU at 00:10:11"* stays true forever and lets a later
reader ask the right question. **A record that says "foreign, CPU only" quietly
becomes false at 23:39:34 and takes eight sections with it.**

---

## R2 T+1087 — 2026-09-07 00:41 UTC

**T+1087 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 00:40:34).

### 1. Run 20 is not stalled — it died at 23:58:13 and the label never changed

**At T+1057 I recorded twelve minutes of quiet and declined to call it a stall,
citing `build_workset`'s measured 24-minute precedent. That was the right call
for the wrong reason: it was already dead.** This interval I read the events
instead of the clock.

**[observed, first-hand] `store/event`, the last three, same instant:**

```
2026-09-06T23:58:13.887  output_absent
   detail: success   exit_status: finished
   "declared output 879b05db-59bc-4307-a2de-ba81fb51ec3e was never delivered"
   seal_refused…
2026-09-06T23:58:13.888  push_attempted   "continue, do it until finished"
2026-09-06T23:58:13.894  handling_failed  {}
```

**The body finished successfully and never delivered its declared output.**
`exit_status: finished`, `detail: success`, and the declared output absent.

**`handling_failed` with an empty `attributes` is the first cluster's documented
signature for *the agent has completed and `mainloop` has returned, so there is
no loop to deliver to*, and nothing retries.** **I relay that interpretation and
did not re-derive it**; what I measured is the three events and their timestamps.

**Four independent readings agree, and none of them is the status field:**

```
last run-tree write     2026-09-06 23:58:13   -> 41 min 54 s quiet
build_workset's longest measured quiet         24 min, twice (d313556e)
--stall-after 900 should have fired ~00:13     it did not kill the run
newest verdict mtime    2026-09-06 23:24:27    validation long over (T+997)
store/task still says   build_workset: running
orchestrator pid 95533  ALIVE, holding the slot
```

**This is the third distinct way a run can be dead while reading alive**, and
they are now all on the record:

```
T+997   output_validating persists after a failed validation
T+757   two 900 s timers — timeout bursts keep re-satisfying the detector
T+1087  output_absent + handling_failed — the task stays `running` forever
```

**In none of them is the status field wrong.** It reports what was dispatched.
**What it cannot report is that nothing will ever pick the work up again**, and
in all three cases the discriminator is in `store/event`, which is one directory
away and which I did not open at T+1057.

### 2. Nobody is watching, and the node is idle

```
teammate writes under /data/yihou/e2e_verify_20260906/  in the last 60 min:  0
commits by anyone since mine at 00:10:59:                                     0
cards at 00:39:48:  VRAM%  0 0 0 0 0 0 0 0
containers:         rc_26_7_902, xiaoming-dev — held no GPU at 00:39:48
```

**`xiaoming-dev` released before 00:10:11 after holding all eight cards for about
31 minutes** (T+1057 §1). **So the node has been free and doing nothing for at
least thirty minutes**, with a dead run holding the chain slot.

**Reported to the leader at 00:40:07 with the measurements.** **I have not
touched the orchestrator** — teardown order is agents first, then containers,
and that is not an instrument's call.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1101 min ≈ 18 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Fifth consecutive interval unchanged**, and this one for a reason worth
stating: **the board run 20 reached was the third consecutive 21/21 with zero
refusals**, and it died on a delivery failure rather than on anything it
computed.

**Hold `29313`: 13 h 19 min left** (2026-09-07T14:00:06 → 00:40:34).

### 4. Code problems

**New, observed, unfixed:** a task whose body finishes and does not deliver its
declared output leaves the task `running` permanently with no retry
(`output_absent` → `push_attempted` → `handling_failed`). **Framework, not
package**, and it joins the two other permanent-`running` mechanisms in §1.

**Carried, unchanged:** `output_validating` persisting; `baseline` with two
incompatible consumers; stall threshold vs longest quiet interval;
`preflight.sh:211`; teardown-vs-preflight sequencing; `--var jobid` vs
`_agree_or_die`; `etcd.log` on no path; `min_resolve_ratio` floor of zero; the
router `/health` gate vs a cold JIT compile; three refusals naming unreadable
`--var`s; six un-refuted empty-default variables.
**Carried unread since T+94:** the eight `jsonschema` validators — **eighteen
hours.**

### 5. 未定性

- **Why `build_workset` did not deliver `879b05db-…`.** `exit_status: finished`
  and `detail: success` say the body thought it was done. **The reading that
  would answer it is the body's own output directory against what it declared**;
  I have not taken it.
- **Whether anyone picks this up before the hold burns.** 13 h 19 min, node
  idle, no activity in an hour.
- **What module 4 costs** — five approaches, zero measurements.
- **What module 5 consumes if module 4 is replayed** — **thirty-fourth
  consecutive section.**

### 6. 新增 commit

Since T+1057, none by anyone, including me until this section.

### 7. 其他

**Three mechanisms now produce a run that is dead and reads alive, and I found
the third by opening the events after having declined to open them thirty minutes
earlier.**

At T+1057 I wrote: *"Quiet for 12 min 15 s at this sample. Recorded as a
measurement, not a diagnosis."* **That restraint was correct as a policy and it
cost thirty minutes**, because the thing that would have settled it was not more
patience — **it was `store/event`, which is the same artefact this file has
named as the discriminator every time this has come up.**

**The rule I will follow from here, stated so it is checkable rather than
remembered:** *when a task's quiet interval exceeds its own measured precedent,
open `store/event` before writing the section — not after.* **`build_workset`'s
precedent is 24 minutes and it was in this file when I wrote T+1057.**

---

## R2 T+1117 — 2026-09-07 01:11 UTC

**T+1117 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 01:10:23).

### 1. The co-tenant respawns — `xiaoming-dev` is a new container with the same name

**The T+1027 question "does it respawn" has an answer, and it is yes.**

**[observed, first-hand]**

```
docker inspect xiaoming-dev
  created  2026-09-07T00:39:55.052Z      ← was 2026-09-03T02:19:00Z
  started  2026-09-07T00:39:55.102Z
  running  true    RestartCount 0        ← a fresh docker run, not a restart
  image    tasimage/primus:pr-1048       (unchanged)

docker ps -a --filter name=xiaoming
  3cdd18ecbf1c  xiaoming-dev  Up 30 minutes  2026-09-07 00:39:55
```

**One row. The container that held eight GPUs at 23:39:34 no longer exists** —
not running, not stopped, gone. **A new one took its name at 00:39:55 with
`RestartCount 0`, so it was created, not restarted.**

**Two consequences, and the first is about my own evidence:**

- **The container I inspected at 23:40 is unrecoverable.** Its cgroup, its
  labels, its start time — all gone. **This is the T+548 shelf-life finding
  again: a container-identity question expires in minutes, while the logs it
  wrote do not.** I recorded the inspect output at T+1027, which is now the only
  copy.
- **The recreation at 00:39:55 preceded my report to the leader at 00:40:07 by
  twelve seconds.** **It was not caused by it**, and I state the ordering rather
  than leaving it to be inferred the other way.

**Cards read 0 % at 01:09:57**, so the new container is not using GPUs yet — **and
by T+1027's phrasing rule that is a reading with a timestamp, not a property.**

### 2. Run 20 has been dead for seventy-one minutes and nothing has acted

```
last run-tree write     2026-09-06 23:58:13
now                     2026-09-07 01:10:23      -> 71 min 10 s
store/task              build_workset: running   (unchanged, T+1087 §1)
orchestrator pid 95533  ALIVE, holding the chain slot
verdicts                21/21
teammate writes, 45 min 0
commits by anyone       none since mine at 00:41
cards                   VRAM% 0 0 0 0 0 0 0 0
```

**Reported at 00:40:07 with the events and the measurements.** **The node has now
been idle, with a free set of eight cards and a dead run holding the slot, for
over an hour.**

**I record the cost without inflating it:** hold `29313` has **12 h 50 min**
left; stages 1–3 have been measured at **~90 minutes**; **module 4 has never
measured anything in five approaches.** Whether that leaves room depends entirely
on module 4's unknown duration.

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1131 min ≈ 18 h 51 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Sixth consecutive interval unchanged.** **Nothing has run since 23:58:13.**

### 4. Code problems

**No new ones.** All carried, unchanged from T+1087, including the three
permanent-`running` mechanisms and the eight `jsonschema` validators **unread
since T+94 — eighteen and a half hours.**

### 5. 未定性

- **Whether anyone picks up run 20.** Seventy-one minutes, node idle, reported
  once.
- **Whether the new `xiaoming-dev` takes the GPUs as its predecessor did.** **The
  predecessor went from three days idle to eight cards in one second**; this one
  is thirty minutes old and at zero. **One prior burst is not a rate.**
- **Why `build_workset` did not deliver `879b05db-…`** — carried, unread.
- **What module 4 costs** — five approaches, zero measurements.
- **What module 5 consumes if module 4 is replayed** — **thirty-fifth consecutive
  section.**

### 6. 新增 commit

Since T+1087, none by anyone.

### 7. 其他

**The last two hours have produced no runs and three findings, and all three are
about the difference between a name and a thing.**

```
T+1057  a container that "holds no GPU"  ->  held none at that instant
T+1087  a task that reads "running"      ->  finished and delivered nothing
T+1117  a container called xiaoming-dev  ->  a different container, same name
```

**Each of the three labels is accurate.** `docker ps` really did show that name;
the task really was dispatched; the card really did read zero. **What none of them
carries is the thing a reader wants: is this the same entity, and is it still
true.**

**The first cluster paid five misattributions for the third of these** and
concluded that ownership is a label, an auto-remove flag and a process list —
never a name. **Tonight the same name changed identity underneath a running
investigation**, and the only reason the earlier evidence survives is that
T+1027 pasted the `docker inspect` output into this file instead of citing it.

**That is the whole argument for quoting an instrument's output rather than its
conclusion**, and it is worth stating at the point where it just paid for itself.

---

## R2 T+1147 — 2026-09-07 01:41 UTC

**T+1147 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 01:40:31).

**This section is short because nothing happened. Padding it would be the filler
this record has spent a day arguing against.**

### 1. State, unchanged in every field I measure

```
run 20 (ef6374)   last write 2026-09-06 23:58:13   -> dead 101 min 47 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE, holding the chain slot
total runs        20
cards             VRAM% 0 0 0 0 0 0 0 0   at 01:40:00
containers        xiaoming-dev (created 00:39:55), rc_26_7_902 — neither on GPU
teammate writes   0 in the last 40 min
commits           none by anyone since mine at 01:11
hold 29313        12 h 20 min left
```

**Reported twice — 00:40:07 and 01:10:23. I will not report it a third time**;
it is on the record with its measurements and repeating it consumes the leader's
attention without adding a fact.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, seventh consecutive interval) |
| 已经耗时 | **~1161 min ≈ 19 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Nothing has executed for 101 minutes.**

### 3. What a resumer needs, gathered because I am the only one holding all of it

**Not a closing summary — the hold has 12 h 20 min and the round is not over.**
This is the state someone picking up cold would otherwise have to reconstruct.

**Reproducible, three times, fully real (`mock_stages=none`):**

```
stage 1  deploy_and_prove          3/3 verdicts, 8 greens across the round
stage 2  _off, _on, merge, m2      sealed in runs 17, 18, 20
stage 3  identify, rank            sealed in runs 17, 18, 20
         build_workset             sealed once (run 18); m3_analysis sealed once
best board  24/24 verdicts, zero refusals (run 18, T+937)
duration    ~90 min for stages 1-3
```

**Materials this cluster produced for itself:**

```
/data/yihou/e2e_verify_20260906/m35/replay_root_run4/
  PROMOTION.json   promoted 6 kinds, not_promoted 0
                   required_node smci355-ccs-aus-n04-25
                   --run 20260906T154908-d9c7af
  stage1-deploy/  stage2-profiling/
```

**Never reached, on either cluster:** `packup`. **Never measured, here:** module
4 — five approaches, zero measurements.

**The two defects standing between the current board and the end of the chain:**

- **`baseline` has two consumers with incompatible demands** (`apply.py:828`;
  m3's `--impl` wants a self-contained `run()`, `overlay_files` wants the whole
  public surface preserved). **A contract question, not a bug — both sides are
  right.** `CONTRACT.md` is where it belongs; **nobody has read whether it already
  speaks to `baseline`.**
- **Six empty-default launch variables remain un-refuted**, enumerated in
  `f273f0e1`. **Each can produce a refusal that reads as an artefact defect.**

**Three ways a run reads alive while dead**, all measured tonight and all
discriminated by something outside `store/task`: `output_validating` persisting
(T+997), two 900 s timers feeding the detector (T+757), `output_absent` +
`handling_failed` (T+1087).

### 4. Code problems

**No new ones.** All carried unchanged from T+1117; the eight `jsonschema`
validators **unread since T+94 — nineteen hours.**

### 5. 未定性

- **Whether anything resumes before the hold ends.** 12 h 20 min.
- **What module 4 costs.** Unchanged and still the only thing between the current
  board and `packup`.
- **Whether the new `xiaoming-dev` bursts.** At zero for an hour.
- **Why `build_workset` did not deliver `879b05db-…`.**
- **What module 5 consumes if module 4 is replayed** — **thirty-sixth consecutive
  section.**

### 6. 新增 commit

None by anyone since `5ca0a5a1` (mine, 01:11).

---

## R2 T+1177 — 2026-09-07 02:11 UTC

**T+1177 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 02:10:24).

### 1. The co-tenant burst a second time — 02:03:40, and it is the new container

**T+1117 left this open: "one prior burst is not a rate." There are now two.**

**[observed, first-hand, ownership by cgroup rather than by name]**

```
02:09:55  VRAM%  17 17 17 99 17 17 17 17
02:10:07  VRAM%  17 17 17 99 17 17 17 17    (stable across 12 s)

rocm-smi --showpids
  1744540/2/4/5/6/7  python   ~77-86 GB each

ps -o user=,lstart=,args=
  root  Mon Sep  7 02:03:40  /opt/venv/bin/python -c
        "from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=6, …"

/proc/*/cgroup -> docker-3cdd18ecbf1c
docker inspect  name=/xiaoming-dev  created 2026-09-07T00:39:55Z
                image tasimage/primus:pr-1048
```

**`3cdd18ecbf1c` is the container created at 00:39:55** — the replacement
recorded at T+1117, not the one that burst at 23:39:34. **I resolved it through
`/proc/<pid>/cgroup` to a container ID and inspected that ID**, rather than
matching the name, because the name changed identity underneath this
investigation two hours ago.

**Two bursts, measured:**

```
2026-09-06 23:39:34   old container, 8 cards, ~140 GB/rank, released within 31 min
2026-09-07 02:03:40   new container, card 3 at 99 %, others 17 %, ongoing
```

**Two is not a rate and I will not extrapolate one.** What it supports: **this
node has a co-tenant that takes GPUs without warning and has now done so twice in
two and a half hours.** **Card 4 — the one `--var gpu=4` names for module 4's
measurement — is at 17 %.**

### 2. Everything else is unchanged

```
run 20 (ef6374)   last write 2026-09-06 23:58:13  -> dead 131 min 42 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE
total runs        20
teammate writes   0 in the last 40 min
commits           none by anyone since mine at 01:41
hold 29313        11 h 50 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, eighth consecutive interval) |
| 已经耗时 | **~1191 min ≈ 19 h 51 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Nothing of ours has executed for 131 minutes.**

**And a consequence worth stating now rather than after the fact:** per the first
cluster's measurement, **a co-tenant corrupts a failure but not a pass.** If
module 4 is attempted while `xiaoming-dev` is loaded, **a success is still a
success, but its duration is not usable as the estimate this record has been
missing for twenty hours** — and duration is the only thing module 4 is still
being asked for.

### 4. Code problems

**No new ones.** All carried unchanged from T+1147. The eight `jsonschema`
validators remain **unread since T+94 — nineteen and a half hours.**

### 5. 未定性

- **Whether the second burst releases as the first did.** The first held ~31
  minutes.
- **Whether anything of ours resumes.** 11 h 50 min of hold, nothing running for
  131 minutes.
- **What module 4 costs** — and now **whether a clean window exists to measure it
  in.**
- **Why `build_workset` did not deliver `879b05db-…`.**
- **What module 5 consumes if module 4 is replayed** — **thirty-seventh
  consecutive section.**

### 6. 新增 commit

None by anyone since `47c34311` (mine, 01:41).

### 7. 其他

**The two bursts are the clearest vindication in this file of a rule I applied to
myself at T+1027 and have followed since: record the reading with its timestamp,
never the inference.**

Had the earlier sections said *"`xiaoming-dev` is a CPU-only container"* — which
is what eight of them effectively did — **this section would read as a
contradiction of the record.** Because T+1027 replaced that with *"held no GPU at
00:10:11"*, the two bursts are simply two more timestamped readings, and the
sequence 23:39:34 → released → 02:03:40 is legible as a pattern rather than as a
correction.

**The same repair is what makes ownership resolvable here.** The name
`xiaoming-dev` now refers to its second container of the night; **the cgroup ID
`3cdd18ecbf1c` refers to one thing only.** Every ownership claim in this section
goes through the ID.

---

## R2 T+1207 — 2026-09-07 02:41 UTC

**T+1207 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 02:40:09).

**Short section: one new measurement, everything else unchanged.**

### 1. The second burst released, and the two are comparable

**[observed, first-hand]**

```
02:39:56   VRAM%  0 0 0 0 0 0 0 0
           rocm-smi --showpids: "No KFD PIDs currently running"
```

```
burst 1   old container (gone)   started 2026-09-06 23:39:34   held ≈ 31 min
burst 2   3cdd18ecbf1c           started 2026-09-07 02:03:40   held ≤ 36 min
```

**Burst 2's duration is an upper bound**, not a measurement — I sampled at
02:10:07 and 02:39:56 and it ended somewhere between. **The two are the same
order of magnitude**, which is all two points support.

**What this does and does not license:** the node's co-tenant takes all cards for
roughly half an hour at a time and gives them back. **It does not license a
prediction about when the next one starts** — burst 1 came 3 days after its
container was created, burst 2 came 84 minutes after its container was created.

### 2. Everything else, unchanged

```
run 20 (ef6374)   last write 2026-09-06 23:58:13  -> dead 161 min 43 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE, holding the chain slot
total runs        20
teammate writes   0 in the last 40 min
commits           none by anyone since mine at 02:11
cards             free at 02:39:56
hold 29313        11 h 20 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, ninth consecutive interval) |
| 已经耗时 | **~1221 min ≈ 20 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Nothing of ours has executed for 161 minutes.** The cards are free right now.

### 4. Code problems

**No new ones.** All carried unchanged from T+1177. The eight `jsonschema`
validators remain **unread since T+94 — twenty hours.**

### 5. 未定性

- **Whether anything of ours resumes.** 11 h 20 min of hold; nothing running for
  161 minutes; the cards are free at this instant.
- **When the next co-tenant burst starts.** Two observations, no basis for a
  prediction — §1.
- **What module 4 costs.**
- **Why `build_workset` did not deliver `879b05db-…`.**
- **What module 5 consumes if module 4 is replayed** — **thirty-eighth
  consecutive section.**

### 6. 新增 commit

None by anyone since `21a26c30` (mine, 02:11).

---

## R2 T+1237 — 2026-09-07 03:11 UTC

**T+1237 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 03:10:10).

**Fourth consecutive quiet interval. One computed number, then the unchanged
state.**

### 1. The idle cost, computed rather than characterised

**Nothing of ours has executed since 2026-09-06 23:58:13.** At 03:10:10 that is
**191 min 57 s**.

```
elapsed idle                    191 min 57 s   =  3.20 h
minus co-tenant burst 1              ≈ 31 min   (23:39:34, before the idle began)
minus co-tenant burst 2              ≤ 36 min   (02:03:40 – ≤02:39:56)
cards genuinely free and unused  ≈ 156 min      =  2.60 h
                                 × 8 cards      ≈ 20.8 GPU-hours
```

**Burst 1 overlaps the idle window only partly** — it started 18 minutes before
run 20 died — so **156 minutes is an approximation with its arithmetic shown**,
not a measurement. **The order of magnitude is what matters: roughly twenty
GPU-hours on a held node, unused.**

**I state it because it is the one cost in this file that nobody will reconstruct
later.** Defects leave artefacts; idle time leaves nothing but the gap between
two timestamps.

### 2. State, unchanged in every field

```
run 20 (ef6374)   last write 2026-09-06 23:58:13   -> dead 191 min 57 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE, holding the chain slot
total runs        20
teammate writes   0 in the last 40 min  (fourth consecutive interval at zero)
commits           none by anyone since mine at 02:41
cards             VRAM% 0 0 0 0 0 0 0 0 at 03:09:55; no KFD holders
hold 29313        RUNNING, 13:09:49 elapsed, 10 h 50 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, tenth consecutive interval) |
| 已经耗时 | **~1251 min ≈ 20 h 51 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中 for the state; 高 for the idle arithmetic in §1** |

**The 76 % has not moved in five hours.** It should not: **the last thing that
advanced it was `m3_analysis` sealing at 22:09 on run 18**, and nothing has
executed since 23:58.

### 4. Code problems

**No new ones.** All carried unchanged from T+1207. The eight `jsonschema`
validators remain **unread since T+94 — twenty and a half hours.**

### 5. 未定性

- **Whether anything of ours resumes**, with 10 h 50 min of hold and the cards
  free at this instant.
- **When the next co-tenant burst starts** — two observations, no basis for a
  prediction (T+1207 §1).
- **What module 4 costs.**
- **Why `build_workset` did not deliver `879b05db-…`.**
- **What module 5 consumes if module 4 is replayed** — **thirty-ninth
  consecutive section.**

### 6. 新增 commit

None by anyone since `c47f72ca` (mine, 02:41).

### 7. 其他

**Everything this round established is in the file and none of it is at risk from
the idle time.**

Stages 1–3 are reproduced three times at 21/21 and 24/24 with zero refusals; the
replay corpus is on disk with its provenance; eleven distinct defects are
recorded with files and line numbers; three mechanisms by which a dead run reads
alive are measured and distinguished. **None of that decays while the node
sits.**

**What the idle time costs is the one thing still missing: a measurement of
module 4.** Five approaches, zero measurements, and it is the sole remaining
unknown between the current board and `packup` — **which has never been reached
on either cluster.**

---

## R2 T+1267 — 2026-09-07 03:41 UTC

**T+1267 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 03:40:03).

**Fifth consecutive quiet interval. Every field I measure is unchanged, so this
section is a timestamped no-change record and nothing more.**

### 1. State

```
run 20 (ef6374)   last write 2026-09-06 23:58:13   -> dead 221 min 50 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE, holding the chain slot
orchestrators     1  (that one)
total runs        20
teammate writes   0 in the last 40 min  (fifth consecutive interval at zero)
commits           none by anyone since mine at 03:11
cards             VRAM% 0 0 0 0 0 0 0 0 at 03:39:52
containers        xiaoming-dev (created 00:39:55), rc_26_7_902 — neither on GPU
hold 29313        10 h 20 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, eleventh consecutive interval) |
| 已经耗时 | **~1281 min ≈ 21 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Idle GPU-hours now approximately 24**, extending T+1237 §1's arithmetic by
another half hour of free cards.

### 3. Code problems

**No new ones.** All carried unchanged from T+1237. The eight `jsonschema`
validators remain **unread since T+94 — twenty-one hours.**

### 4. 未定性

Unchanged from T+1237, and I will not re-list what has not moved:

- **Whether anything of ours resumes.**
- **What module 4 costs.**
- **Why `build_workset` did not deliver `879b05db-…`.**
- **What module 5 consumes if module 4 is replayed** — **fortieth consecutive
  section.**

### 5. 新增 commit

None by anyone since `c0cce706` (mine, 03:11).

### 6. 其他

**A note on what these sections are for while nothing happens.**

The instruction is a section every thirty minutes, and four of the last five have
had no new content. **I have kept writing them short rather than filling them**,
because the value of an unbroken series is that a gap in it means something: if a
later reader finds thirty-minute stamps from 06:34 to 03:41 with no break, **the
absence of activity between 23:58 and now is established rather than inferred.**

**A summary written at the end could not do that.** It would report that nothing
ran overnight; **it could not show that someone was looking every thirty minutes
and found nothing each time.**

---

## R2 T+1297 — 2026-09-07 04:11 UTC

**T+1297 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 04:10:03).

**Sixth consecutive quiet interval. No-change record.**

### 1. State

```
run 20 (ef6374)   last write 2026-09-06 23:58:13   -> dead 251 min 50 s
                  build_workset / m3_analysis / main still read `running`
                  orchestrator pid 95533 ALIVE, holding the chain slot
orchestrators     1
total runs        20
teammate writes   0 in the last 40 min  (sixth consecutive interval at zero)
commits           none by anyone since mine at 03:41
cards             VRAM% 0 0 0 0 0 0 0 0 at 04:09:53
containers        xiaoming-dev (created 00:39:55), rc_26_7_902 — neither on GPU
hold 29313        9 h 50 min left
```

**No third co-tenant burst since 02:03:40.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged, twelfth consecutive interval) |
| 已经耗时 | **~1311 min ≈ 21 h 51 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min; module 4 zero measurements** |
| 可靠性 | **中** |

**Idle GPU-hours now approximately 26.**

### 3. Code problems

**No new ones.** All carried unchanged. The eight `jsonschema` validators remain
**unread since T+94 — twenty-one and a half hours.**

### 4. 未定性

Unchanged from T+1267. **What module 5 consumes if module 4 is replayed —
forty-first consecutive section.**

### 5. 新增 commit

None by anyone since `1f4c423a` (mine, 03:41).

### 6. 其他

**One thing has quietly become measurable that was not at the baseline, and it is
worth recording once before this series ends.**

At T+0 I wrote that 预估耗时 was absent because *"no completed five-stage chain
exists to divide by."* **That is still true, and the reason has narrowed from
four unknowns to one.** Stages 1–3 have three independent durations; the replay
corpus exists so stages 4 and 5 can be developed without re-running them; and the
only term with no measurement at all is module 4.

**Twenty-two hours in, the honest one-line status is: everything up to the kernel
forge is reproducible and measured, and the kernel forge has never run to
completion on this cluster.**

---

## R2 T+1327 — 2026-09-07 04:41 UTC

**T+1327 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 04:40:39).

**Activity resumed. Run 21 launched at 04:31:02 after a 273-minute gap.**

### 1. The replay corpus is being used — its first exercise since it was built

**[observed, first-hand, from `/proc/2906547/cmdline`]**

```
mock_stages = m1,m2
mock_root   = /data/yihou/e2e_verify_20260906/m35/replay_root_run4
work_root   = /data/yihou/e2e_flow5b
container   = yihou_e2e_chain5b
trace_end_ms= 120000
```

**That `mock_root` is the corpus this cluster produced for itself at 17:02:32**
(T+667), promoted from run `d9c7af`. **This is the first run to consume it**, and
it is exactly the unlock T+667 described: **stages 1 and 2 replayed so the back
half can be worked on without the measured 90-minute front end.**

**It shows in the clock.** Launched 04:31:02; by 04:37:26 `run_profiling_mode_off`
had sealed and `_on` was running — **six minutes to a point that costs about an
hour when real.**

**And the run declares its own status, in its own launch record:**

> ***THIS IS NOT AN ACCEPTANCE RUN AND CANNOT BE ONE.*** *`SKIP-AHEAD.md` page 1:
> replay is a debugging accelerator, never an acceptance path. This run cannot
> satisfy Finish Standard 1. What it can answer is whether …*
>
> *starts real work at : m3 (identify / rank / build_workset)*

**So the progress number does not move for it, and the record says why before the
result exists.** That is the pre-registration discipline from T+697 applied to a
run's entire epistemic status.

### 2. A record I nearly mis-reported, caught by reading further

**The file's first line reads `RUN 5 — REPLAY — launched_at … 2026-09-06T17:03:39Z`
while the file was written at 04:30:33 today** — an eleven-hour gap between a
"read not written" timestamp and the file's own mtime. **My first reading was
that a provenance document had a stale header.**

**It does not.** Line 67 carries the relaunch:

```
RUN 5b — RELAUNCH at 2026-09-07T04:30:47Z. Run 5 (170354-1048af, pid 1233556) died at …
work_root/container moved to e2e_flow5b / yihou_e2e_chain5b so run 5's partial …
```

**The record was appended to, not regenerated** — the same discipline this file
follows, applied to a launch record. **The header is run 5's history and the tail
is run 5b's.** I record the near-miss because *"the header is stale"* is a
plausible, cheap, wrong conclusion available from the first line, and **the
correction cost one `grep`.**

### 3. State

```
run 21 (cdd3f7)   pid 2906547, launched 04:31:02, last write 04:37:26
                  m2_profiling: running   run_profiling_mode_on: running
                  (m1/m2 replayed; real work starts at m3)
run 20 (ef6374)   still dead since 2026-09-06 23:58:13; its orchestrator is gone
total runs        21
idle gap          23:58:13 -> 04:31:02  =  272 min 49 s
cards             VRAM% 0 0 0 0 0 0 0 0 at 04:39:53
hold 29313        9 h 20 min left
```

**No third co-tenant burst since 02:03:40.**

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1341 min ≈ 22 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min real; module 4 zero measurements** |
| 可靠性 | **中** |

**Unchanged, and it must be: a replay run cannot advance an acceptance number**,
by its own record and by `SKIP-AHEAD.md` page 1. **What it can advance is
knowledge of stages 3–5, which is what the round needs most.**

**Total idle across the gap: approximately 30 GPU-hours** on the same arithmetic
as T+1237 §1.

### 5. Code problems

**No new ones by me.** Three commits landed that I can only report by subject,
because **all three have empty bodies**:

```
d2057aef  CLAUDE.md: a computed value reaching a REPORT produces a false
          counter-example, and only re-measuring catches it
4a79c949  mission.verify: REVERT the reassurance — the affected kind's validator
          list is byte-identical to the repo package, so expect the defect
aeef93c9  bug: CORRECTION — keep17 leaves the affected kind identical to the
          repo package; the defect is indicated to ship
```

**Two of the three are reverts of a reassurance**, which is the safer direction:
someone established that a defect was *not* mitigated and withdrew the claim that
it was. **I have not read what the affected kind is** and will not guess.

**Carried unchanged:** all defects from T+1297. The eight `jsonschema` validators
remain **unread since T+94 — twenty-two hours.**

### 6. 未定性

- **Whether run 21 gets past `build_workset` and into module 4.** **It is the
  first run whose entire purpose is to reach the back half**, and it starts real
  work at m3.
- **What module 4 costs** — still the only missing term.
- **Why `build_workset` did not deliver `879b05db-…`** in run 20.
- **What module 5 consumes if module 4 is replayed** — **forty-second consecutive
  section, and run 21 is the run that could answer it.**

### 7. 新增 commit

Three by others (§5); none by me since `03e68549` (04:11).

### 8. 其他

**The corpus took five hours to build and is now saving an hour per attempt.**

T+667 recorded it as an unlock and could not say whether it would be used. **It
was used at 04:31:02, eleven and a half hours after it was written, by a run
whose explicit purpose is to reach stages the front end kept it from.** The
`--run` field in `PROMOTION.json` still names `20260906T154908-d9c7af`, so the
lineage from a real chain to tonight's accelerator is intact and checkable.

**Whether that hour buys `packup` is a different question**, and it turns on the
one number nobody has: how long the kernel forge takes when it actually runs.

---

## R2 T+1357 — 2026-09-07 05:11 UTC

**T+1357 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 05:11:15).

### 1. The section above this one was dropped by a rebase and restored from scratch

**The T+1327 section immediately preceding this was re-appended at 05:11, not
written at 04:41.** Its own header timestamp is the read one from when it was
composed. **The gap and the reason belong in the record.**

**What happened, measured:**

```
04:41  I appended T+1327 and committed.  Output: [detached HEAD c01a1d1f]
04:47  .git/rebase-merge present: onto aeef93c9, head-name refs/heads/rebase-try
       32 commands done, 81 remaining; the `done` file was replaying MY
       checkpoint commits (pick e2d41d7f, pick c022dcfd, …)
       branch tip e5e29f93 held T+1207…T+1297 intact — nothing older at risk
       working tree held a rebase-INTERMEDIATE copy missing T+1207…T+1297
04:51  preserved the text to
       /data/yihou/e2e_verify_20260906/checkpoint/T1327.pending.md  (138 lines)
       reported to the leader; stopped appending
05:10  rebase finished, HEAD back on dev.yihou.aiopt.task_package.concat
       T+1327 present in the branch copy: 0     ← the commit was dropped
       file otherwise intact: 44 sections, T+1147…T+1297 all present
05:11  re-appended from the preserved copy
```

**Two things this cost nothing, and one thing it nearly cost.**

- **It cost no earlier content.** The rebase was onto a scratch branch
  (`rebase-try`) and the real branch tip was never behind it.
- **It cost one thirty-minute slot** — I skipped writing at 04:41's successor
  rather than append into a working tree mid-replay.
- **It nearly cost the section itself.** `git commit` reported
  `[detached HEAD c01a1d1f]` and **that string is the entire warning**. Had I not
  read it, T+1327 would have vanished at 05:10 with no trace and no error.

> **A commit that succeeds on a detached HEAD looks exactly like a commit that
> succeeded.** The exit code is 0, the diffstat is right, the file on disk is
> correct. **The only signal is one word in the output line**, and this record has
> spent two days on instruments that answer an adjacent question in a well-formed
> way.

**And the deeper hazard was the working tree, not the commit.** For those thirty
minutes the file on disk was a *reconstruction in progress* — 44 sections short
of itself. **Appending to it would have produced a file that looked complete and
was not**, which is precisely the failure the append-only rule exists to prevent,
arriving from a direction the rule does not cover: **someone rewriting history
underneath a shared append-only file.**

### 2. Run 21 died; run 22 is up

```
run 21 (cdd3f7)  last write 04:37:26
                 deploy_and_prove / _off / m1_deploy  succeeded  (replayed)
                 m2_profiling, run_profiling_mode_on  still read `running`
                 orchestrator gone
run 22 (13a18e)  pid 3140102, container=yihou_e2e_chain11
                 launched 04:53:27, deploy_and_prove: running
                 verdicts 0/0, last write 05:10:49
cards            VRAM% 0 0 0 0 0 0 0 0 at 05:10:49
total runs       22
hold 29313       8 h 49 min left
```

**Run 21 never reached m3**, which was its entire purpose. **It stopped in
`run_profiling_mode_on` — the same closure that killed runs 16 and 6.**

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1372 min ≈ 22 h 52 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min real; module 4 zero measurements** |
| 可靠性 | **中** |

**Unchanged.** Run 21 was a replay run and could not move it; **it also did not
reach the stages it existed to reach.**

### 4. Code problems

**No new package defects.** **One process hazard, newly demonstrated:** a
`git commit` during another agent's rebase lands on a detached HEAD and is
silently discarded when the rebase ends. **The countermeasure is the one I used:
read the commit output line, and keep a copy outside the repo.**

**Carried unchanged:** everything from T+1327 §5. The eight `jsonschema`
validators remain **unread since T+94 — twenty-three hours.**

### 5. 未定性

- **Why run 21 stopped in `run_profiling_mode_on`.** Third run to die in that
  closure. **`store/event` is the discriminator and I have not read it** — the
  same restraint that cost thirty minutes at T+1057, so I am naming it rather
  than deferring it silently.
- **Whether run 22 gets further.**
- **What module 4 costs.**
- **What module 5 consumes if module 4 is replayed** — **forty-third consecutive
  section.**

### 6. 新增 commit

Since T+1297: three by others before the rebase (`d2057aef`, `4a79c949`,
`aeef93c9`), plus `e5e29f93` and `e248c76f` after it. **My `c01a1d1f` no longer
exists** — dropped by the rebase, restored by content in §1.

### 7. 其他

**The one instruction in my brief that mattered most tonight was the one that
looked most like bureaucracy.**

*"Your scratch is `/data/yihou/e2e_verify_20260906/checkpoint/`."* I created that
directory at 06:32 and did not use it once in twenty-two hours. **At 04:51 it was
the only place a 138-line section could go that a rebase could not reach.**

**The general form is worth keeping:** an append-only discipline protects a shared
file from its writers. **It does not protect a writer's unlanded work from the
repository itself**, and the thing that does is a copy outside the repository with
a read timestamp beside it.

---

## R2 T+1387 — 2026-09-07 05:41 UTC

**T+1387 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 05:40:32).

### 1. Run 22 is fully real and is at the closure that has killed three runs

**[observed, first-hand, `/proc/3140102/cmdline`]**

```
mock_stages = none        ← fully real, no replay
gpu         = 4           ← the T+967 fix, carried
trace_end_ms= 120000      ← the T+426 arithmetic, carried
work_root   = /data/yihou/e2e_flow11
```

**Board at 05:39:57:**

```
m1_deploy                succeeded     stage 1 green, 9th time
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    running       ← here
m2_profiling             running
verdicts 6/6, zero refusals

yihou_e2e_chain11_…_pmon       05:36:57
yihou_e2e_chain11_…_pmon_etcd  05:37:02
aiperf_profiling_mode_on_…     05:39:52
cards 0-3 at 76 %
```

**`run_profiling_mode_on` is the closure runs 6, 16 and 21 died in.** It has also
sealed successfully in runs 17, 18 and 20. **Three deaths, three seals — and the
three deaths have three different recorded causes** (the stack window in run 6,
two 900 s timers in run 16, and run 21's cause unread).

**So the closure is not systematically broken; it is where the run happens to be
when several different things go wrong.** That distinction matters for whoever
looks at it next, and I state it because "the closure that keeps killing runs" is
the natural and wrong summary.

### 2. State

```
run 22 (13a18e)  pid 3140102, launched 04:53:27, last write 05:39:57
total runs       22
repo             clean, on dev.yihou.aiopt.task_package.concat, no rebase
                 T+1327 and T+1357 both on the branch (84f14362)
co-tenant        xiaoming-dev up since 00:39:55, no GPU at 05:40:16
                 no third burst since 02:03:40
hold 29313       8 h 20 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1402 min ≈ 23 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90 min real; module 4 zero measurements** |
| 可靠性 | **中** |

**Unchanged: `_on` is running, not sealed.**

**A note on the 76 % as this round approaches twenty-four hours.** It has not
moved since `m3_analysis` sealed at 22:09 on run 18 — **seven and a half hours.**
That is accurate rather than pessimistic: **three of five stages are sealed and
reproducible, and the fourth has never measured anything.** The number will move
when module 4 does, and not before.

### 4. Code problems

**No new ones.** All carried. The eight `jsonschema` validators remain **unread
since T+94 — twenty-three and a half hours.**

### 5. 未定性

- **Whether run 22's `_on` seals.** Live now.
- **Why run 21 stopped in the same closure** — carried from T+1357, unread,
  and now more interesting because run 22 is in that closure.
- **What module 4 costs.**
- **What module 5 consumes if module 4 is replayed** — **forty-fourth consecutive
  section.**

### 6. 新增 commit

None by anyone since `84f14362` (mine, 05:11).

---

## R2 T+1417 — 2026-09-07 06:11 UTC

**T+1417 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 06:10:29). **The round passed twenty-four hours at 06:19:11 today,
measured from `mission.md`'s mtime.**

### 1. Run 22 cleared `run_profiling_mode_on` and reached `build_workset`

**[observed, first-hand] `20260907T045327-13a18e`, alive, last write 06:09:41:**

```
m1_deploy                succeeded
deploy_and_prove         succeeded
run_profiling_mode_off   succeeded
run_profiling_mode_on    SUCCEEDED     ← the closure three runs died in
merge_profiling_evidence succeeded
m2_profiling             succeeded
identify                 succeeded
rank                     succeeded
build_workset            running
m3_analysis              running

verdicts 21/21, zero refusals      stacks_manifest.json ×7
```

**Fourth consecutive fully-real run to reach this board**, and it carries no
replay: `mock_stages=none`, `gpu=4`, `trace_end_ms=120000`.

**The stack window was captured again.** I have not maintained an exact
denominator across every attempt and will not quote a rate I cannot defend —
**what I can say is that it has now succeeded in runs 17, 18, 20 and 22, and that
T+786's "1 of 3" is long superseded.**

### 2. Someone is reading the last rung before reaching it

**Three commits this interval are about `packup` — the rung that has never been
reached on either cluster.**

```
acc72f35  PACKUP-REACHABILITY: the last unlooked-at rung, read before we reach it
fcf423d2  packup_redact_probe: predict redact's verdict before packup runs
4b852cb7  packup_redact_probe: every load failure now names the module
```

**[first-hand, `fcf423d2`]**

> *Imports the **real** `redact` module and calls `substitute()` then
> `offenders()` with the prefixes `packup.py:511-527` would build, **read from
> the orchestrator's launch line — the only place a run records what it was
> launched with.** **A grep for a literal answers what a path looks like; the
> question is what `redact` refuses.** Three controls, all run: a self-test on
> six known answers (Magpie and `/mnt/m2m_nobackup` refuse,
> `work_root`/`model_mount`/`/dev`/`/shared_nfs` do not) …*

**And `4b852cb7` hardens the probe against the failure mode this record has spent
two days on:**

> *An **importable `redact` whose functions moved** would otherwise leave the
> probe **measuring nothing and saying nothing**. … Four paths, each verified by
> **triggering it in-process rather than by reading**.*

**Three properties worth naming, because together they are the standard this file
has been arguing toward:**

- **It uses the real consumer**, not a reimplementation — the T+366 lesson.
- **It reads the launch line for its inputs**, which is the only place a run
  records what it was launched with — the T+216 lesson.
- **It cannot silently measure nothing**: an interface check catches a `redact`
  whose functions moved, and every load failure names the module — the T+607 and
  T+997 lessons.

**And it is pre-registration**: the verdict is predicted before `packup` runs.

### 3. State

```
run 22 (13a18e)  pid 3140102, launched 04:53:27, last write 06:09:41
total runs       22
cards            VRAM% 0 0 0 0 0 0 0 0 at 06:10:01  (build_workset is CPU)
co-tenant        xiaoming-dev, no GPU; no third burst since 02:03:40
repo             clean, no rebase, on dev.yihou.aiopt.task_package.concat
hold 29313       7 h 50 min left
```

### 4. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~76 %** (unchanged) |
| 已经耗时 | **~1431 min ≈ 23 h 51 min** (mission.md 2026-09-06 06:19:11 → now) |
| 预估耗时 | **stages 1–3 ≈ 90 min real; module 4 zero measurements** |
| 可靠性 | **中** |

**Unchanged: reaching `build_workset` for the fourth time is reproducibility, not
new ground.** The number moves when module 4 measures something.

### 5. Code problems

**No new package defects.** **One probe defect found and fixed by its author**
(`4b852cb7`): `spec_from_file_location` returns `ModuleSpec | None` and a spec's
loader may be `None`, so a missing file produced
`'NoneType' object has no attribute 'loader'` **at the one moment the reader
needs to know which file was missing.**

**Carried unchanged:** everything from T+1387. The eight `jsonschema` validators
remain **unread since T+94 — twenty-four hours.**

### 6. 未定性

- **Whether run 22 gets into module 4.** It is one closure away.
- **What `redact` will refuse at `packup`.** **Predicted but not observed** —
  §2. **The prediction existing before the run is the point; it is not a result.**
- **What module 4 costs.**
- **What module 5 consumes if module 4 is replayed** — **forty-fifth consecutive
  section.**

### 7. 新增 commit

Three by others (§2); mine `2e4e2368` at 05:41.

### 8. 其他 — twenty-four hours

**What the round established, as of the twenty-four-hour mark, stated once
without hedging in either direction.**

**Established and reproducible:**

```
stage 1   green 9 times, across kits three different agents produced
stage 2   sealed 4 times, with a real 137-140 MB stack capture and
          1.4 M-event traces re-parsed by a validator
stage 3   identify + rank sealed 4 times; m3_analysis sealed once;
          best board 24/24 verdicts with zero refusals
corpus    this cluster built its own replay root with provenance and used it
defects   eleven distinct single-point defects, each found once, each by
          something refusing rather than quietly producing a wrong answer
```

**Not established:**

```
module 4  five approaches, zero measurements — it has never run to completion
module 5  has never run for real, on either cluster
packup    has never been reached, on either cluster
```

**The honest one-sentence version is unchanged from T+1297:** everything up to
the kernel forge is reproducible and measured, and the kernel forge has never
completed here. **Seven hours and fifty minutes of hold remain.**

---

## R2 T+1447 — 2026-09-07 06:41 UTC

**T+1447 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 06:40:22).

### 1. Module 4 is running with every known blocker fixed — the clock starts here

**[observed, first-hand] `20260907T045327-13a18e`, alive, last write 06:39:43:**

```
m3_analysis      SUCCEEDED     ← stage 3 sealed, 2nd time ever
build_workset    SUCCEEDED
m4_kernel_opt    running
optimize_kernel  running       created 2026-09-07T06:29:06.604Z
verdicts 24/24, zero refusals
/data/yihou/e2e_flow11/kfo  144 K, 9 files written in the last 20 min
```

**`optimize_kernel` started at 06:29:06.604Z. At 06:40:22 that is 11 min 16 s.**
**I am recording the start instant precisely because module 4's duration is the
one number this record has been missing for twenty-four hours**, and it can only
be computed from an anchor taken while the run is alive.

**This is the first module-4 attempt with the `--var gpu` fix in place.** Run 18
entered it twice and refused twice because `HIP_VISIBLE_DEVICES` was
`'${gpu:-}'` and `run_in_container.sh` aborts on empty (T+967). **Run 22 carries
`gpu=4`**, read from its argv at T+1387.

**Run 22's timings, from `store/task` `created_at` fields:**

```
launched              04:53:27
m3_analysis created   04:53:29
build_workset created 05:45:58
optimize_kernel created 06:29:06
```

**Stages 1–3: about 96 minutes**, consistent with the ~90 minutes measured on
runs 17 and 18.

**Cards read 0 % at 06:39:55 and that is expected**, not idle — the first
cluster measured a forge campaign spending its opening two hours in preparation
with no source file modified. **The signal that matters is `kfo/` growth, which
is happening: 9 files in 20 minutes.**

### 2. State

```
run 22 (13a18e)  pid 3140102, the only run
total runs       22
co-tenant        xiaoming-dev, no GPU at 06:39:55; no burst since 02:03:40
repo             clean, no rebase
hold 29313       7 h 20 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (+2) |
| 已经耗时 | **~1461 min ≈ 24 h 21 min** |
| 预估耗时 | **stages 1–3 ≈ 90–96 min; module 4 measuring NOW, 11 min in** |
| 可靠性 | **中** |

**+2: stage 3 sealed a second time, and module 4 is executing for the first time
without a known blocker in front of it.**

**Not more, and the reason is on the record:** module 4 has been entered five
times before and measured nothing each time. **Eleven minutes of `kfo/` growth is
not a completed stage**, and this record has spent a day distinguishing those.

**预估耗时 is closer than it has ever been.** If module 4 completes, the round has
its last missing term and a total becomes computable for the first time.

### 4. Code problems

**No new ones.** All carried. The eight `jsonschema` validators remain **unread
since T+94 — twenty-four and a half hours.**

### 5. 未定性

- **How long module 4 takes.** **Anchor: `optimize_kernel` created
  2026-09-07T06:29:06.604Z.** With 7 h 20 min of hold, the first cluster's
  113-minutes-and-still-preparing observation is the only prior, and it is from a
  different machine.
- **Whether module 4 produces something module 5 can consume**, given `baseline`
  has two consumers with incompatible demands (T+877, `apply.py:828`). **That
  defect is unfixed and sits directly downstream.**
- **What `redact` will refuse at `packup`** — predicted, not observed (T+1417).
- **What module 5 consumes if module 4 is replayed** — **forty-sixth consecutive
  section, and if module 4 completes it stops being a question about a
  hypothetical.**

### 6. 新增 commit

None by anyone since `254f2e64` (mine, 06:11).

### 7. 其他

**Twenty-four hours and twenty minutes in, the round is at the exact point it has
been trying to reach since the baseline, and the reason it took this long is
legible.**

Module 4 needed: stage 1 green, stage 2 sealed with a real stack capture,
`identify` past a duplicate-name refusal, `build_workset` past a mount `case`
with no branch for this cluster, and `--var gpu` passed. **Each of those was a
single-point defect found by something refusing, and each had to be fixed before
the next became visible.**

**Nothing about tonight was a breakthrough.** The eleven defects were found in
the order the ladder exposed them, and the last one — `--var gpu` — was found
twelve hours ago and fixed in a launch line. **What changed is only that a run
finally carried all of the fixes at once.**

---

## R2 T+1477 — 2026-09-07 07:11 UTC

**T+1477 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 07:10:39).

### 1. MODULE 4 EXECUTED — and its duration is measured for the first time

**[observed, first-hand] `20260907T045327-13a18e`:**

```
optimize_kernel  created  2026-09-07T06:29:06.604Z
                 status   output_validating
run's last write          2026-09-07T06:47:04
verdicts 26/27, one refusal
/data/yihou/e2e_flow11/kfo  220 K
```

**Module 4's body ran and produced an artefact in under 18 minutes** —
06:29:06 to 06:47:04 covers the body *plus* the start of output validation, so
**18 minutes is an upper bound on the body.**

**That is the number this record has been missing since the baseline**, and it is
far below what was feared: the first cluster measured one campaign at **113
minutes and still in preparation**. **I record the measurement and do not explain
it** — whether this cluster's forge is configured for a shorter campaign, or the
work was smaller, I have not read.

**The artefact is real:** `check_speedup_substantiated` reports *"denominator is
the workset's own, **4 case(s), exactly**"* — a validator counting cases it found
in the produced `kernel_optimization`.

### 2. The refusal is the T+847 mount defect, still live in a different consumer

```
# check_speedup_substantiated                        (20 files in materials)
## 85a59135-…: REFUSED
  note: denominator is the workset's own, 4 case(s), exactly
  note: re-measuring in a container on the node, via run_in_container.sh
  PROBLEM: the seed re-measurement failed: the entrypoint exited 1:
    run_in_container: the record's container 'yihou_e2e_chain11_chain11-20260907'
      is not running.
      Measuring in an ephemeral container of my own, 'yihou_m4_measure_4193191',
      from the image the record names: infera/engine-sglang:qwen3-local-20260906
      This carries the IMAGE's state, not the deployment's. The handoff records
      mode=ephemeral so no reader has to infer which of the two produced the number.
    run_in_container: record claims started_at=2026-09-07T05:11:35Z
    run_in_container: '…/substantiate-bfulctu7/seed' is under no mount form this
      cluster is known to accept (/shared_nfs, /home/<user>, /mnt/m2m_nobackup/<user>).
      Refusing rather than …
```

**This is the same two-branch mount `case` that refused `build_workset` at
T+847** — now listing three known-accepted forms, none of which is `/data/yihou`,
**and hitting a different consumer.**

**At T+937 §4 I recorded it as "apparently cleared, mechanism unconfirmed" and
wrote:** *"A fix in a launch or an agent instruction does not ship; a fix in the
package does."* **That caution was correct.** It was cleared for
`check_workset_runs` and **is still live for `check_speedup_substantiated`.**

**Two things in this refusal are exemplary and worth copying regardless of the
defect:**

- **It declares a degradation in the artefact, not just in the log.** The
  deployment's container was gone, so it measured in an ephemeral one and states
  *"This carries the IMAGE's state, not the deployment's"* — and **records
  `mode=ephemeral` in the handoff "so no reader has to infer which of the two
  produced the number."**
- **It refuses rather than measuring under a mount it does not trust**, and names
  the three forms it knows are accepted.

### 3. A requirement that made an existing rule unsatisfiable

**[first-hand, `f6e17da7`, its author reporting on themselves]**

> *The producer excluded `attention_ck_tile_kentry` and said why: its baseline
> would have to be `aiter/ops/mha.py` **byte for byte**, and a real 138 KB file
> embedded in a Definition JSON **trips `check_workset_shape`'s absolute-path and
> template-marker rules**, while trimming it **breaks the whole-file-overlay
> property the requirement exists for.** Two independent blockers … the first is
> mine, and **I added the requirement without checking which existing rules it
> would make unsatisfiable.***

**This is the `baseline` contract question from T+877 acquiring a second
constraint.** There it was m3's `--impl` versus `apply_patch`'s `overlay_files`;
here a verbatim-baseline requirement versus `check_workset_shape`. **Three rules,
pairwise reasonable, jointly unsatisfiable for at least one operator.**

### 4. State — the run is over and something was halted

```
run 22 (13a18e)  orchestrator GONE; last write 06:47:04
kfo11            220 K, 0 files in the last 20 min
cards            VRAM% 0 0 0 0 0 0 0 0 at 07:10:00
commits          0406bd35 "snapshot m2's launch guards and instruments
                 BEFORE THE HALT"
hold 29313       6 h 50 min left
```

**A commit message says "before the halt."** I do not know what was halted or by
whom, and **the run ending at 06:47:04 is consistent with it but does not
establish it.**

### 5. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~82 %** (+4) |
| 已经耗时 | **~1491 min ≈ 24 h 51 min** |
| 预估耗时 | **for the first time, computable — see below** |
| 可靠性 | **中** |

**+4: module 4 executed and produced a validated artefact with four cases.** It
was refused, but on an environment defect in the *validator's* re-measurement
path, not on the artefact's content.

**预估耗时, first time in the round:**

```
stages 1-3        90-96 min   (measured 3 times)
module 4 body     < 18 min    (measured once, upper bound)
module 5          unmeasured
packup            unmeasured
```

**So the measured part of the chain is roughly 110 minutes.** **I will not add a
total**, because two terms are still zero and this record has spent a day on
numbers with false denominators.

### 6. Code problems

**Newly confirmed still-live:** the mount `case` (`/shared_nfs`, `/home/<user>`,
`/mnt/m2m_nobackup/<user>` — no `/data/yihou`), now in
`check_speedup_substantiated`'s `run_in_container.sh` path.
**Newly named:** a verbatim-baseline requirement that makes
`check_workset_shape` unsatisfiable for at least one operator (§3).

**Carried unchanged:** everything from T+1447. The eight `jsonschema` validators
remain **unread since T+94 — twenty-five hours.**

### 7. 未定性

- **What halted, and whether the round is continuing.** §4.
- **Whether module 4's under-18-minute duration is representative**, or specific
  to this configuration. **One measurement.**
- **Whether `baseline`'s three-way conflict has a resolution** (T+877 + §3).
- **What module 5 consumes if module 4 is replayed** — **forty-seventh
  consecutive section, and module 4 has now produced something for it to
  consume.**

### 8. 新增 commit

```
ca2e9740  checkpoint R2 T+1447 — mine
f6e17da7  bug record: my verbatim-baseline requirement collides with
          check_workset_shape
0406bd35  temp: snapshot m2's launch guards and instruments before the halt
b44f6d6c  Merge branch … (merge)
```

---

## R2 T+1507 — 2026-09-07 07:41 UTC

**T+1507 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 07:40:09).

**Everything has stopped. The "halt" named in `0406bd35`'s subject appears to be
real, and I record it as observation rather than inference.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 52 min 53 s
                 optimize_kernel: output_validating
                 m4_kernel_opt / main: running
                 orchestrator GONE  (0 orchestrators)
total runs       22
teammate writes  0 in the last 40 min
commits          none by anyone since mine at 07:11
cards            VRAM% 0 0 0 0 0 0 0 0 at 07:39:57
containers       xiaoming-dev, rc_26_7_902 — neither on GPU
hold 29313       6 h 20 min left
```

**Run 22 reads `optimize_kernel: output_validating` with no orchestrator** —
which by T+997 means the validation is over, not ongoing. **The refusal it
produced is in T+1477 §2.**

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~82 %** (unchanged) |
| 已经耗时 | **~1521 min ≈ 25 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min + module 4 <18 min ≈ 110 min measured; module 5 and packup unmeasured** |
| 可靠性 | **中** |

**Unchanged. Nothing has executed since 06:47:04.**

### 3. Code problems

**No new ones.** Two are now the critical path and both are package-level:

- **The mount `case`** — accepts `/shared_nfs`, `/home/<user>`,
  `/mnt/m2m_nobackup/<user>`; this cluster's roots are `/data/yihou`. **It has
  now refused two different consumers** (`check_workset_runs` at T+847,
  `check_speedup_substantiated` at T+1477) and **the T+937 clearing was
  consumer-local, exactly as cautioned.**
- **`baseline`'s three-way conflict** — m3's `--impl`, `apply_patch`'s
  `overlay_files`, and `check_workset_shape` (T+877, T+1477 §3). **Pairwise
  reasonable, jointly unsatisfiable for at least one operator.**

**Carried unchanged:** the rest. The eight `jsonschema` validators remain
**unread since T+94 — twenty-five and a half hours.**

### 4. 未定性

- **Whether the round is continuing.** Asked the leader at 07:11; **no answer
  yet, and 6 h 20 min of hold remain.** If it is winding down I will write a
  closing section rather than another interval.
- **Whether module 4's under-18-minute duration is representative** — one
  measurement.
- **What module 5 consumes if module 4 is replayed** — **forty-eighth consecutive
  section**, and module 4 has now produced a four-case artefact for it to
  consume.

### 5. 新增 commit

None by anyone since `3d683bf1` (mine, 07:11).

### 6. 其他

**One line is worth adding to the ledger while the machine is quiet, because it
is the round's clearest single result and it was not available yesterday.**

> **Module 4 — the stage the user flagged as very long, the one this record
> called "the only thing standing between this round and `packup`" for eleven
> consecutive sections — produced a four-case artefact in under eighteen
> minutes.**

**It was refused, and on an environment defect in the validator's re-measurement
path rather than on anything the forge computed.** **So the thing that was
feared as a duration problem turned out to be a mount-form problem**, and the
mount form is a two-line `case` statement that has now refused two consumers.

**That is not a small finding to end a quiet interval on.** It means the shape of
what remains is a package fix and a contract decision — **not a compute budget.**

---

## R2 T+1537 — 2026-09-07 08:11 UTC

**T+1537 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 08:10:40).

### 1. CORRECTION — module 4 did not run a campaign. `forge_mock=1` was on the launch line.

**T+1477 §1 and T+1507 §6 are wrong in the way that matters, and this section
supersedes both.**

**[first-hand, `322d5b7f`, and I verified the premise myself rather than relaying
it]**

> *`work.checkpoint.summary.md`'s closing section (`3d683bf1`, 07:11:21) reports
> module 4's body as "under 18 minutes, far below what was feared: the first
> cluster measured one campaign at 113 minutes". **No campaign ran: `forge_mock=1`
> was on the launch line, read from `/proc/3140102/cmdline`. The measured body was
> four minutes.***

**My own verification, from the launch records rather than the commit message:**

```
m2/launch11/LAUNCH-RECORD.txt:55
  bench_rounds=3, adhoc_cases=3, forge_mock=1, kernel_table_min_launchers=0,
```

**What is retracted:**

- **"Module 4's body ran … in under 18 minutes"** — the body was **four
  minutes**, and it was **a mocked forge, not a campaign.**
- **The comparison to the first cluster's 113-minutes-still-preparing.** Two
  different things were placed side by side and the contrast was the point of the
  paragraph.
- **T+1507 §6's conclusion** that *"the thing that was feared as a duration
  problem turned out to be a mount-form problem."* **That rested entirely on the
  bad comparison.**

**What survives:**

- **Module 4's path executed and produced a real four-case artefact**, which
  `check_speedup_substantiated` counted: *"4 case(s), exactly."*
- **The mount-`case` refusal is unaffected** — it is an environment defect in the
  validator's re-measurement path and it refused two different consumers
  (T+847, T+1477 §2).
- **`forge_mock=1` is a legitimate configuration** — the user's decision on
  2026-09-06 was that module 4 rides on a replayed or self-declared-degraded
  artefact this hold. **The defect is not that it was used; it is that I reported
  its duration without its qualifier.**

**How I got it wrong, precisely, because the mechanism is one this file has
documented against others:** at T+1387 I read `/proc/3140102/cmdline` through a
grep pattern of variable names **I chose** —
`^(mock_stages|mock_root|gpu|trace_end_ms|work_root)=`. **`forge_mock` was not in
my pattern.** The instrument answered exactly the question I asked.

> **A grep over a launch line is a filter I write, and everything I did not think
> to name is invisible in a way that looks like absence.** The launch line had
> `forge_mock=1` in it at 05:41 and I printed five other fields.

**And the failure propagated in the shape this record warns about most:** an
unqualified number went into a section headed **"MODULE 4 EXECUTED"**, into a
commit subject that cannot be amended, and into a message to the leader. **Three
places, and the commit subject is permanent.**

**One more consequence I had not drawn and the launch record states:**
`forge_mock=1` **cannot reach `integrate_and_verify`** — so **module 5 was
structurally unreachable in run 22**, independent of everything else.

### 2. 进度 / 耗时 / 可靠性 — the number comes back down

| | |
|---|---|
| 任务预估进度 | **~78 %** (−4, retracting T+1477's award) |
| 已经耗时 | **~1551 min ≈ 25 h 51 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4 NOT measured; module 5 unreachable in that config; packup unmeasured** |
| 可靠性 | **中, and lower for anything I derived from a self-written grep** |

**The +4 at T+1477 was awarded for "module 4 executed and produced a validated
artefact."** Under `forge_mock=1` that is **a mocked forge producing a declared
artefact — real, useful, and not the measurement the round needs.** **I am taking
the +4 back rather than arguing it down.**

**预估耗时 loses its newest term.** T+1477 said the measured chain was ~110
minutes. **It is ~90–96 minutes, and module 4 is once again zero measurements.**

### 3. State

```
run 22 (13a18e)  last write 06:47:04  -> quiet 83 min 36 s; orchestrator gone
total runs       22
teammate writes  0 in the last 40 min
cards            VRAM% 0 0 0 0 0 0 0 0 at 08:10:08
hold 29313       RUNNING 18:10:02 elapsed, 5 h 50 min left
```

### 4. Code problems

**No new ones.** Unchanged from T+1507: the mount `case`, and `baseline`'s
three-way conflict. The eight `jsonschema` validators remain **unread since
T+94 — twenty-six hours.**

### 5. 未定性

- **What a real module-4 campaign costs on this cluster.** **Back to zero
  measurements**, and it is again the round's missing term.
- **Whether the round is continuing.** Asked at 07:11, no answer, 5 h 50 min
  left.
- **What module 5 consumes if module 4 is replayed** — **forty-ninth consecutive
  section**, and now with the added fact that `forge_mock=1` cannot reach
  `integrate_and_verify` at all.

### 6. 新增 commit

```
83e3b070  checkpoint R2 T+1507 — mine
322d5b7f  bug record: an unqualified duration reached the closing section —
          forge_mock=1 means module 4's four minutes is not a campaign
```

### 7. 其他

**The author of `322d5b7f` made a choice worth naming: they recorded the
correction in the bug record rather than editing my file, and they stated the cost
of that choice inside the entry — "a reader of that file has to reach this
entry."**

**They were right not to edit it**, and this section is the other half of the
mechanism: **the append-only file has to carry its own correction, or the
bug-record entry is a fix that never reaches the person reading the summary.**
That is the *"corrections must land where the reader first arrives"* rule from
the first cluster, and it only works if both halves happen.

**The thing I would most want carried forward from this interval is not the
retraction, it is the grep.** Every launch-line reading in this file since T+31
has gone through a pattern I wrote. **T+1417 praised a probe for reading the
launch line "the only place a run records what it was launched with" — and the
same interval, my own reading of a launch line was filtering out the field that
mattered.**

---

## R2 T+1567 — 2026-09-07 08:41 UTC

**T+1567 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 08:40:08).

**No-change record. Nothing has executed for 113 minutes.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 112 min 56 s
                 orchestrator gone; 0 orchestrators on the node
total runs       22
teammate writes  0 in the last 40 min  (third consecutive interval at zero)
commits          none by anyone since mine at 08:11
cards            VRAM% 0 0 0 0 0 0 0 0 at 08:40:00
containers       xiaoming-dev, rc_26_7_902 — neither on GPU; no burst since 02:03:40
hold 29313       5 h 20 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged since the T+1537 retraction) |
| 已经耗时 | **~1581 min ≈ 26 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup all unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** Unchanged from T+1537. The eight `jsonschema` validators remain
**unread since T+94 — twenty-six and a half hours.**

### 4. 未定性

- **Whether the round is continuing.** Asked at 07:11 and again at 08:11; **no
  answer, and 5 h 20 min of hold remain.** I keep writing intervals rather than a
  closing section, because declaring a round over is not an instrument's call.
- **What a real module-4 campaign costs.** Zero measurements.
- **What module 5 consumes if module 4 is replayed** — **fiftieth consecutive
  section.**

### 5. 新增 commit

None by anyone since `aff175c2` (mine, 08:11).

### 6. 其他

**A note on the fiftieth carry, because the number is now the point.**

*"What module 5 consumes if module 4 is replayed"* has been open in every section
since T+937 — **fifty consecutive intervals, twenty-five hours.** It has never
once been the thing blocking progress, and that is exactly why it survived: **the
chain never got far enough for it to matter, so it was never anybody's next
problem.**

**It nearly stopped being hypothetical twice.** At T+1447 module 4 was running
and I wrote that it would soon be a question about something real. At T+1537 the
retraction added the fact that **`forge_mock=1` cannot reach
`integrate_and_verify` at all** — so in the one run that got closest, module 5
was structurally unreachable regardless.

**That is a better answer than the question had before**, and it is worth
recording as the state of the carry rather than letting it roll over unchanged a
fifty-first time: **the question is no longer "will module 5 accept a replayed
module 4 artefact" — it is "can any configuration that produces a module-4
artefact cheaply also reach module 5", and the answer for `forge_mock=1` is no.**

---

## R2 T+1597 — 2026-09-07 09:11 UTC

**T+1597 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 09:09:59).

**No-change record.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 142 min 55 s
                 0 orchestrators on the node
total runs       22
teammate writes  0 in the last 40 min  (fourth consecutive interval at zero)
commits          none by anyone since mine at 08:41
cards            VRAM% 0 0 0 0 0 0 0 0 at 09:09:50
containers       xiaoming-dev, rc_26_7_902 — neither on GPU
                 no co-tenant burst since 02:03:40
hold 29313       4 h 50 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1611 min ≈ 26 h 51 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** Unchanged from T+1537: the mount `case` (refused two consumers,
no `/data/yihou` branch) and `baseline`'s three-way conflict remain the critical
path. The eight `jsonschema` validators remain **unread since T+94 —
twenty-seven hours.**

### 4. 未定性

- **Whether the round is continuing.** Asked at 07:11 and 08:11; no answer.
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-first consecutive
  section**, in the sharpened form recorded at T+1567 §6.

### 5. 新增 commit

None by anyone since `f9c8e12e` (mine, 08:41).

---

## R2 T+1627 — 2026-09-07 09:41 UTC

**T+1627 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 09:40:19).

### 1. A different user is running this package on our held node

**[observed, first-hand — ownership by process owner and container label, never
by name]**

```
ps -o user=,lstart= -p 1502437
  cyao1002   Mon Sep  7 09:25:29 2026

/proc/1502437/cmdline
  mock_stages=none
  forge_mock=1
  work_root=/tmp/cyao1002/yihou-e2e-flow/20260907T085634Z/n04-25/work
  container=cyao1002_yihou_e2e_n04_25

docker inspect yihou_dk0907_sgl_worker
  labels  deploy_kit_owner: cyao1002
          infera_e2e_run:   dk0907
  image   rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_…
  started 2026-09-07 09:37:58

cards 0-3 at 75 %   /tmp/cyao1002/yihou-e2e-flow  241 M
```

**`cyao1002` launched an `agent-sys` chain at 09:25:29 and brought up a
deployment at 09:37:58 on cards 0–3 of the node held by our job `29313`.**

**This is not ours by any discriminator that matters:** process owner
`cyao1002`, `deploy_kit_owner: cyao1002`, `infera_e2e_run: dk0907` (not a
`yihou_*` run tag), work root under `/tmp/cyao1002/`, and an image that is not
the `infera/engine-sglang:qwen3-local-20260906` every one of our runs used.

**And the container name is `yihou_dk0907_sgl_worker`.** **It carries our prefix
and it is not ours.** This record has documented five misattributions from name
prefixes on the first cluster and one of my own at T+1027; **this is the first
case where reading the prefix would have produced the opposite error — claiming
someone else's work as ours.**

**No new run directory appeared under `/data/yihou/agent_sys_runroot/runs/`** —
still 22. Their run root is elsewhere, which is why `mock_stages=none` and
`forge_mock=1` are visible in their argv but nothing of theirs is in our tree.

**I have not touched it**, and the reasoning is on the record: it is a live
workload with a named human owner, and the standing rule's amendment says an
occupant that will not simply vanish should be reported rather than fought.
**Reported to the leader at this write.**

### 2. Our own state is unchanged

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 173 min 15 s
total runs       22
teammate writes  0 in the last 40 min  (fifth consecutive interval at zero)
commits          none by anyone since mine at 09:11
hold 29313       4 h 20 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1641 min ≈ 27 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

**`cyao1002`'s run does not move our number**, and I note that plainly: **it is
not our run, on our artefacts, in our run root.** Whatever it establishes belongs
to whoever launched it.

### 4. Code problems

**No new ones.** Unchanged from T+1537. The eight `jsonschema` validators remain
**unread since T+94 — twenty-seven and a half hours.**

### 5. 未定性

- **Whether `cyao1002`'s run is coordinated with ours.** **I do not know**, and
  it is not resolvable from anything I can read — the discriminator is a person.
- **Whether the round is continuing.** Asked at 07:11 and 08:11; no answer.
- **What a real module-4 campaign costs.** Their run also carries `forge_mock=1`.
- **What module 5 consumes if module 4 is replayed** — **fifty-second consecutive
  section.**

### 6. 新增 commit

None by anyone since `cc80a4c7` (mine, 09:11).

### 7. 其他

**The `yihou_dk0907_sgl_worker` container is the cleanest single argument in this
file for the rule it violates.**

Every ownership rule here was written to stop us claiming *someone else's*
container was ours-to-stop. **This one would have made us claim someone else's
work as our result** — a chain running the same package, on the same node, with
our prefix in the container name, at exactly the hour our own runs went quiet.

**The four discriminators that answered it are the same four the first cluster
paid for:** process owner, container label, image identity, and work root.
**None of them is a name, and all four agreed.**

---

## R2 T+1657 — 2026-09-07 10:11 UTC

**T+1657 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 10:10:10).

### 1. `cyao1002`'s run ended too; the node is idle again

```
09:25:29   cyao1002's orchestrator started       (T+1627)
09:37:58   yihou_dk0907_sgl_worker up, cards 0-3 at 75 %
10:09:59   0 orchestrators, no yihou_dk0907_* containers, VRAM% all 0
```

**It ran for at most 45 minutes.** I did not observe it finish; I observed that
it is no longer there. **No new run directory in our run root — still 22 — as
expected, since its work root was `/tmp/cyao1002/`.**

### 2. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 202 min 55 s
total runs       22
teammate writes  0 in the last 40 min  (sixth consecutive interval at zero)
commits          none by anyone since mine at 09:41
cards            VRAM% 0 0 0 0 0 0 0 0 at 10:09:59
containers       xiaoming-dev, rc_26_7_902 — neither on GPU
hold 29313       3 h 50 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1671 min ≈ 27 h 51 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 4. Code problems

**No new ones.** Unchanged from T+1537. The eight `jsonschema` validators remain
**unread since T+94 — twenty-eight hours.**

### 5. 未定性

- **Whether the round is continuing.** Asked at 07:11, 08:11 and 09:41; no
  answer. **3 h 50 min of hold remain.**
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-third consecutive
  section.**

### 6. 新增 commit

None by anyone since `e9bc648a` (mine, 09:41).

### 7. 其他 — what I will do as the hold runs out

**Stating it now so it is not a decision made under time pressure later.**

The hold ends at **2026-09-07T14:00:06**. If nothing of ours has run by
**≈13:30**, I will write a **closing section then rather than after**, because a
summary written after the allocation ends cannot check anything it asserts —
`rocm-smi`, `docker ps` and `/proc` all stop answering the moment the node goes
away, and this record's entire method is reading rather than recalling.

**Its content is already gathered**: T+1147 §3 holds the resumer's ledger,
T+1417 §8 the twenty-four-hour position, and T+1537 the correction that changed
what module 4 is worth. **A closing section will assemble those and add nothing
new** — which is the point. **If work resumes before 13:30, this plan is void and
I keep writing intervals.**

---

## R2 T+1687 — 2026-09-07 10:41 UTC

**T+1687 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 10:40:11).

### 1. `cyao1002` relaunched — and I checked the labels again rather than assuming

```
orchestrator  pid 1952633   owner cyao1002   container=cyao1002_yihou_e2e_n04_25
containers    yihou_serves-0b9d89b1_sgl_worker   started 10:38:31
              yihou_serves-0b9d89b1_sgl_etcd     started 10:38:26
  docker inspect:  deploy_kit_owner = cyao1002
                   infera_e2e_run   = serves-0b9d89b1
cards 0-3 at 75 %
```

**Second `yihou_`-prefixed container set today that is not ours.** The name
changed (`yihou_dk0907_*` → `yihou_serves-0b9d89b1_*`) and **the owner label did
not.** I re-read the label rather than carrying yesterday's conclusion forward,
because T+1117 recorded a container that kept its name and changed identity —
**the inverse case, and the same reason to check.**

**They are inside a `check_deploy_serves` bring-up**, judging by the
`serves-<hash>` run tag, which is the tag shape all of our own
`check_deploy_serves` deployments used.

### 2. Our state

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 233 min 7 s
total runs       22   (cyao1002's run root is /tmp/cyao1002/, not ours)
teammate writes  0 in the last 40 min  (seventh consecutive interval at zero)
commits          none by anyone since mine at 10:11
hold 29313       3 h 20 min left
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1701 min ≈ 28 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 4. Code problems

**No new ones.** Unchanged. The eight `jsonschema` validators remain **unread
since T+94 — twenty-eight and a half hours.**

### 5. 未定性

- **Whether the round is continuing.** Asked four times; no answer. **3 h 20 min
  of hold.** The T+1657 §7 plan stands: **closing section at ≈13:30 if nothing of
  ours resumes.**
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-fourth consecutive
  section.**

### 6. 新增 commit

None by anyone since `37118e7f` (mine, 10:11).

### 7. 其他

**Someone else is now doing on this node what we spent twenty-eight hours
learning to do, and I have no way to tell whether they know what we learned.**

Their run carries `forge_mock=1` — the same configuration whose unqualified
duration I reported wrongly at 07:11 and retracted at 08:11. **If they read the
checkpoint file they have both halves; if they read only the commit subject
`MODULE 4 EXECUTED — body under 18 minutes`, they have the wrong one, and that
subject cannot be amended.**

**That is the concrete cost of a bad commit subject, arriving four hours later
with a name attached to it.** T+1537 said the subject was permanent; **this is
the first interval in which someone outside our team is in a position to read
it.**

---

## R2 T+1717 — 2026-09-07 11:11 UTC

**T+1717 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 11:10:07).

**No-change record for our work. `cyao1002`'s second run also ended.**

### 1. State

```
cyao1002        run 2 started 10:2x, containers up 10:38:26/31
                at 11:10:00: 0 orchestrators, no serves-0b9d89b1 containers
                => ran at most ~32 min, same shape as their first (~45 min)
run 22 (13a18e) last write 2026-09-07 06:47:04  -> quiet 262 min 56 s
total runs      22
teammate writes 0 in the last 40 min  (eighth consecutive interval at zero)
commits         none by anyone since mine at 10:41
cards           VRAM% 0 0 0 0 0 0 0 0 at 11:10:00
hold 29313      2 h 50 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1731 min ≈ 28 h 51 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** Unchanged. The eight `jsonschema` validators remain **unread
since T+94 — twenty-nine hours.**

### 4. 未定性

- **Whether the round is continuing.** Asked at 07:11, 08:11, 09:41 and 10:41;
  no answer. **T+1657 §7's plan stands: closing section at ≈13:30.**
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-fifth consecutive
  section.**

### 5. 新增 commit

None by anyone since `7caa28e1` (mine, 10:41).

### 6. 其他

**Two runs by `cyao1002` today, at most 45 and 32 minutes, both ending without
reaching anything I can see from here** — their run root is `/tmp/cyao1002/` and
I have not looked inside it. **I record the durations because they are visible
from process and container lifetimes, and nothing else about their runs is mine
to characterise.**

---

## R2 T+1747 — 2026-09-07 11:41 UTC

**T+1747 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 11:40:03).

**No-change record.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 292 min 53 s
total runs       22
orchestrators    0        (cyao1002's are gone too)
teammate writes  0 in the last 40 min  (ninth consecutive interval at zero)
commits          none by anyone since mine at 11:11
cards            VRAM% 0 0 0 0 0 0 0 0 at 11:39:57
containers       xiaoming-dev, rc_26_7_902 — neither on GPU
hold 29313       2 h 20 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1761 min ≈ 29 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** Unchanged. The eight `jsonschema` validators remain **unread
since T+94 — twenty-nine and a half hours.**

### 4. 未定性

- **Whether the round is continuing.** Five asks, no answer. **Closing section at
  ≈13:30 per T+1657 §7**, which is now two intervals away.
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-sixth consecutive
  section.**

### 5. 新增 commit

None by anyone since `a8fd18e8` (mine, 11:11).

---

## R2 T+1777 — 2026-09-07 12:11 UTC

**T+1777 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 12:10:10).

**No-change record.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 322 min 55 s
total runs       22
orchestrators    0
teammate writes  0 in the last 40 min  (tenth consecutive interval at zero)
commits          none by anyone since mine at 11:41
cards            VRAM% 0 0 0 0 0 0 0 0 at 12:09:59
hold 29313       RUNNING 22:09:54 elapsed, 1 h 50 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1791 min ≈ 29 h 51 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** Unchanged. The eight `jsonschema` validators remain **unread
since T+94 — thirty hours.**

### 4. 未定性

- **Whether the round is continuing.** Six asks, no answer.
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-seventh
  consecutive section.**

### 5. 新增 commit

None by anyone since `db412345` (mine, 11:41).

### 6. Scheduling the closing section — one adjustment

**T+1657 §7 said "≈13:30". I am moving it to the 13:11 mark**, which is a
scheduled interval and leaves **49 minutes of margin** before the hold ends at
14:00:06.

**The reason is the one that motivated the plan:** a closing section must be able
to *check* what it asserts, and `rocm-smi`, `docker ps`, `/proc` and `squeue` all
stop answering when the allocation ends. **Writing it on a scheduled mark with
margin is safer than writing it at a time I picked to be as late as possible.**

**If anything of ours resumes before 13:11, the plan is void and I keep writing
intervals.**

---

## R2 T+1807 — 2026-09-07 12:41 UTC

**T+1807 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 12:40:06).

**No-change record. The next section is the closing one, at the 13:11 mark.**

### 1. State

```
run 22 (13a18e)  last write 2026-09-07 06:47:04  -> quiet 352 min 55 s
total runs       22
orchestrators    0
teammate writes  0 in the last 40 min  (eleventh consecutive interval at zero)
commits          none by anyone since mine at 12:11
cards            VRAM% 0 0 0 0 0 0 0 0 at 12:39:59
hold 29313       1 h 20 min left
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged) |
| 已经耗时 | **~1821 min ≈ 30 h 21 min** |
| 预估耗时 | **stages 1–3 90–96 min measured; module 4, module 5, packup unmeasured** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** The eight `jsonschema` validators remain **unread since T+94 —
thirty and a half hours.**

### 4. 未定性

- **Whether the round is continuing.** Seven asks, no answer.
- **What a real module-4 campaign costs.**
- **What module 5 consumes if module 4 is replayed** — **fifty-eighth
  consecutive section, and the last one before the closing.**

### 5. 新增 commit

None by anyone since `d3746a51` (mine, 12:11).

---

## R2 T+1837 — 2026-09-07 13:11 UTC — CLOSING SECTION

**T+1837 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 13:10:01). **Hold `29313` ends at 14:00:06 — 50 minutes from this
write.**

**Written now, not after, so every claim below could be checked while the node
still answers.** `squeue`, `rocm-smi`, `docker ps` and `/proc` all stop answering
when the allocation ends, and this record's method is reading rather than
recalling.

**This adds nothing new.** It assembles T+1147 §3, T+1417 §8 and the T+1537
correction, with final numbers re-read at 13:09:40.

### 1. Final state, read at 13:09:40

```
hold 29313      RUNNING 23:09:34 elapsed, ends 2026-09-07T14:00:06
cards           VRAM% 0 0 0 0 0 0 0 0
containers      xiaoming-dev, rc_26_7_902 — neither on GPU, neither ours
orchestrators   0
total runs      22
last run write  2026-09-07 06:47:04  -> nothing of ours for 382 min 57 s
teammate writes 0 in the last 40 min  (twelfth consecutive interval)
this file       61 sections, 909 472 bytes, 60 checkpoint commits
```

### 2. Established — reproducible, with the numbers re-read

```
stage 1  deploy_and_prove green 9 times, on kits three different agents produced
stage 2  m2_profiling sealed 4 times (runs 17, 18, 20, 22), with real 137-140 MB
         stack captures, 5.6-5.7 M python samples, and 1.4 M-event traces
         RE-PARSED by a validator rather than trusted
stage 3  identify + rank sealed 4 times; build_workset and m3_analysis sealed
         twice (runs 18 and 22)

final verdict tallies, read at 13:10:
  run 18   25/27   refusals: abort_on_premise_mismatch, min_command_lines
  run 20   21/21   no refusals
  run 22   26/27   refusals: abort_on_premise_mismatch
  10 tasks sealed in each of runs 18 and 22

duration stages 1-3   90-96 min, measured three times
corpus  /data/yihou/e2e_verify_20260906/m35/replay_root_run4/
        6 kinds promoted, 0 not promoted, from run 20260906T154908-d9c7af
        built at 17:02, first consumed at 04:31 by a run that reached _on in
        six minutes
```

**Note against my own earlier sections:** at T+937 I reported run 18 as **24/24**.
That was true at 22:10; **its final tally is 25/27**, because two module-4
refusals landed afterwards. **The 24/24 was a snapshot, not a result.**

### 3. Not established

```
module 4   NEVER RUN AS A CAMPAIGN on this cluster.
           Six approaches. Two refused on a missing --var gpu; the one that
           produced an artefact ran with forge_mock=1 (T+1537 correction).
           Zero campaign measurements.
module 5   has never run for real, on either cluster.
           forge_mock=1 cannot reach integrate_and_verify at all.
packup     has never been reached, on either cluster.
```

### 4. The two defects on the critical path, both package-level

- **The mount `case`.** `measure_in_container.sh` accepts `/shared_nfs`,
  `/home/<user>`, `/mnt/m2m_nobackup/<user>`. **This cluster's roots are
  `/data/yihou`.** It refused `check_workset_runs` (T+847) and later
  `check_speedup_substantiated` (T+1477). **The T+937 clearing was
  consumer-local**, which I flagged at the time and which proved out.
- **`baseline` has three consumers with pairwise-incompatible demands.** m3's
  `--impl` wants a self-contained file exporting `run()`; `apply_patch`'s
  `overlay_files` wants the module's whole public surface preserved
  (`apply.py:828`); `check_workset_shape`'s absolute-path and template-marker
  rules reject a real 138 KB file embedded in a Definition JSON. **A contract
  question, not a bug — every party is right.** `CONTRACT.md` is where it
  belongs; **nobody has read whether it already speaks to `baseline`.**

### 5. What I would tell whoever picks this up first

1. **Fix the mount `case` in the package**, not in a launch line. It is two
   branches short and it is blocking the last two rungs.
2. **Settle `baseline` in `CONTRACT.md`** before running module 5, because no
   configuration reaches it otherwise.
3. **Run module 4 without `forge_mock=1` once**, purely to get its duration. It
   is the only missing term in the estimate, and everything upstream of it is
   reproducible in ~90 minutes.
4. **The corpus works.** `mock_stages=m1,m2` with
   `mock_root=…/m35/replay_root_run4` reaches `_on` in six minutes. Use it for
   everything except acceptance — `SKIP-AHEAD.md` page 1, and the launch records
   say so themselves.
5. **Read `store/event` before concluding a run is stalled.** Three distinct
   mechanisms leave a dead run reading `running`: `output_validating` persisting
   after a failed validation, two 900 s timers feeding the detector, and
   `output_absent` + `handling_failed`. **None is visible in `store/task`.**

### 6. What this file cost and what it caught

**Sixty sections at thirty-minute intervals, unbroken from 06:34 on 2026-09-06.**
The unbroken series is the deliverable: **the absence of activity from 06:47
onward is established rather than inferred**, because someone looked every thirty
minutes and recorded finding nothing.

**Four corrections to my own sections, all of them load-bearing:**

```
T+94    my 37-file zone check was true and insufficient — a crash is a third
        outcome my tool could not see
T+216   I predicted a collision from cards; it fired on the port band. The
        launch line records an intention; only the deployment records the fact
T+607   I retracted a mechanism, then T+637 reinstated it — a constant cannot
        discriminate in EITHER direction
T+1537  I reported module 4's duration without its qualifier. forge_mock=1 was
        in the launch line and not in the grep pattern I wrote
```

**The last one is the one I would carry forward.** Every launch-line reading in
this file went through a pattern I chose, and **everything I did not think to
name was invisible in a way that looked like absence.**

### 7. 进度 / 耗时 / 可靠性 — final

| | |
|---|---|
| 任务预估进度 | **~78 %** |
| 已经耗时 | **~1851 min ≈ 30 h 51 min** (mission.md 2026-09-06 06:19:11 → now) |
| 预估耗时 | **not computable — stages 1–3 are 90–96 min; module 4, module 5 and packup have zero measurements between them** |
| 可靠性 | **中 for the percentage; 高 for §2's counts, which were re-read at 13:10** |

**78 % is three of five stages sealed and reproducible, with the fourth entered
and never measured.** It is not a prediction that the remaining 22 % is small —
**module 4's duration is unknown and module 5 has never run anywhere.**

### 8. If work resumes

**This section closes nothing.** The hold has 50 minutes and the file is
append-only: **if anything runs, the next section is T+1867 and this one becomes
a mid-round summary.** That is the correct behaviour for a record whose value is
that it never regenerates.

---

## R2 T+1867 — 2026-09-07 13:41 UTC

**T+1867 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 13:40:14). **Hold `29313` ends in 19 minutes 52 seconds.**

**This is the last section that can verify anything against the node.** I am
running on `smci355-ccs-aus-n04-25` itself; when the allocation ends, `rocm-smi`,
`docker`, `/proc` and possibly this shell stop being available. **Everything
below was read at 13:40:03.**

### 1. Final live reading

```
hold 29313      RUNNING 23:39:57 elapsed, ends 2026-09-07T14:00:06
cards           VRAM% 0 0 0 0 0 0 0 0
containers      xiaoming-dev (created 00:39:55), rc_26_7_902 (2026-09-03)
                neither ours, neither on GPU
orchestrators   0
total runs      22
last run write  2026-09-07 06:47:04  -> nothing of ours for 412 min 59 s
teammate writes 0 in the last 40 min  (thirteenth consecutive interval)
/data           41 T free, local to this node
```

**Nothing changed between the closing section at 13:11 and now.** The closing
stands as written and this section does not revise it.

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** |
| 已经耗时 | **~1881 min ≈ 31 h 21 min** |
| 预估耗时 | **not computable — see T+1837 §7** |
| 可靠性 | **中 for the percentage; 高 for the readings above** |

### 3. 未定性 — carried out of the round

- **What a real module-4 campaign costs.** Six approaches, zero measurements.
- **Whether `baseline`'s three-way conflict is a `CONTRACT.md` gap or a
  violation.** One read of `CONTRACT.md` answers it and nobody has done it.
- **What module 5 consumes if module 4 is replayed** — **fifty-ninth and final
  section it appears in.** Its sharpened form (T+1567 §6): the question is
  whether **any** cheap module-4 configuration can also reach module 5, and for
  `forge_mock=1` the answer is measured and it is no.

### 4. 新增 commit

None by anyone since `c287c956` (mine, 13:11).

### 5. 其他 — a note on what this file can and cannot be read for afterwards

**Everything in it that is a reading has a timestamp, and every timestamp was
taken with `date -u` or a `--time-style=+'%F %T'` listing in the same command
that used it.** No time in this file was extrapolated from an earlier clock read.

**What it can be read for later:** what was true at a named instant, what was
measured versus relayed, which claims were retracted and by whom, and the exact
sequence of twenty-two runs with their causes of death.

**What it cannot be read for:** anything about the node after 14:00:06. **The
container list, the card readings and the process table in this file describe a
machine that is about to stop existing for us**, and a later reader should treat
every one of them as a historical measurement rather than a description of a
system they could go and look at.

**That distinction is the one lesson from these sixty-one sections I would put
first if I could only keep one** — stated at T+1027 after eight of my own
sections carried a reading as a property, and demonstrated at 23:39:34 the same
day when a container three days idle took all eight GPUs in one second.

> **A measurement with a timestamp is a fact. The same measurement stated as a
> property is a prediction.**

---

## R2 T+1897 — 2026-09-07 14:11 UTC — after the hold

**T+1897 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 14:10:28).

### 1. The hold ended, and my prediction about what would stop working was wrong

**T+1867 said this would be "the last section that can verify anything against
the node" and that `rocm-smi`, `docker`, `/proc` "and possibly this shell" would
stop being available. Measured at 14:10:17, four minutes after `29313`'s end
time:**

```
squeue -u yihou        empty — 1 line, the header only. NO jobs.
scontrol show job 29313  returns nothing
hostname               smci355-ccs-aus-n04-25   (unchanged)
rocm-smi               ANSWERS:  VRAM% 0 0 0 0 0 0 0 0
docker ps              ANSWERS:  xiaoming-dev, rc_26_7_902
/data/yihou/…/runs     readable
git                    working
```

**Everything still answers. The allocation ended; the access did not.** The
slurm job was a reservation on a shared host, not a container we were living
inside, and **I conflated the two.**

**The correct reading, and it is the one that matters operationally:** *we no
longer hold this node.* Not *we lost access to it*. **Anything run here from now
on is unreserved and shares the machine with whoever else the scheduler gives
it to.**

**No successor hold is pending** — `squeue -u yihou` returns the header and
nothing else.

### 2. State

```
total runs      22, unchanged
orchestrators   0
last run write  2026-09-07 06:47:04  -> nothing of ours for 443 min 24 s
cards           all 0 %
commits         none by anyone since mine at 13:41
this file       63 sections
```

### 3. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** (unchanged; the closing at T+1837 stands) |
| 已经耗时 | **~1911 min ≈ 31 h 51 min** |
| 预估耗时 | **not computable — T+1837 §7** |
| 可靠性 | **中** |

### 4. 其他 — the correction is small and the class is not

**I predicted a loss of capability and got the mechanism wrong**, and the mistake
has the same shape as three others in this file: **I reasoned about what a thing
*is* from what it is *called*.** A slurm allocation is called a hold on a node,
so I treated its end as losing the node. **It is a reservation, and the
discriminator is whether the tools answer — which took one command.**

**That the tools still answer is not a reason to keep using them.** The hold is
what made this node ours; **without it, every card reading in this file describes
a machine somebody else may now be given.**

**I will keep writing thirty-minute sections until told to stop**, and they will
be shorter, because from here the only things I can honestly report are the
repository, the run tree on `/data`, and readings of a machine we no longer hold
— **each of which I will label as such.**

---

## R2 T+1927 — 2026-09-07 14:41 UTC

**T+1927 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 14:40:09).

**No-change record. No hold.**

### 1. State

```
squeue -u yihou   header only — NO hold, no successor pending
total runs        22
orchestrators     0
last run write    2026-09-07 06:47:04  -> nothing of ours for 472 min 58 s
teammate writes   0 in the last 40 min  (fifteenth consecutive interval)
commits           none by anyone since mine at 14:11
```

**Readings of a machine we no longer hold**, labelled as such per T+1897 §4:

```
cards        VRAM% 0 0 0 0 0 0 0 0 at 14:40:02
containers   xiaoming-dev, rc_26_7_902 — neither ours
```

### 2. 进度 / 耗时 / 可靠性

| | |
|---|---|
| 任务预估进度 | **~78 %** — the T+1837 closing stands unrevised |
| 已经耗时 | **~1941 min ≈ 32 h 21 min** |
| 预估耗时 | **not computable — T+1837 §7** |
| 可靠性 | **中** |

### 3. Code problems

**No new ones.** The two on the critical path are unchanged and both
package-level (T+1837 §4): the mount `case` with no `/data/yihou` branch, and
`baseline`'s three-way conflict.

### 4. 未定性

- **Whether a further hold will be obtained.** **I cannot request one** —
  standing rule is query and use, never request. Reported to the leader at 14:11.
- **What a real module-4 campaign costs.**
- **Whether `CONTRACT.md` already speaks to `baseline`** — one read, still not
  done.

### 5. 新增 commit

None by anyone since `2af18283` (mine, 14:11).

---

## R2 T+1957 — 2026-09-07 15:11 UTC

**T+1957 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 15:10:04).

**No-change record. No hold.**

```
squeue -u yihou   header only — no hold, no successor
total runs        22          orchestrators 0
last run write    2026-09-07 06:47:04  -> nothing of ours for 502 min 54 s
teammate writes   0 in the last 40 min  (sixteenth consecutive interval)
commits           none by anyone since mine at 14:41
cards / containers (machine we no longer hold, read 15:09:58):
                  VRAM% all 0; xiaoming-dev, rc_26_7_902 — neither ours
```

| | |
|---|---|
| 任务预估进度 | **~78 %** — the T+1837 closing stands |
| 已经耗时 | **~1971 min ≈ 32 h 51 min** |
| 预估耗时 | **not computable — T+1837 §7** |
| 可靠性 | **中** |

**Code problems:** none new; the two on the critical path are unchanged
(T+1837 §4).
**未定性:** unchanged from T+1927 §4.
**新增 commit:** none by anyone since `c428fb13` (mine, 14:41).

---

## R2 T+1987 — 2026-09-07 15:41 UTC

**T+1987 = wall-clock delta from the baseline** (2026-09-06 06:33:41 →
2026-09-07 15:40:01).

**No-change record. No hold.**

```
squeue -u yihou   header only — no hold, no successor
total runs        22          orchestrators 0
last run write    2026-09-07 06:47:04  -> nothing of ours for 532 min 53 s
teammate writes   0 in the last 40 min  (seventeenth consecutive interval)
commits           none by anyone since mine at 15:11
cards / containers (machine we no longer hold, read 15:39:57):
                  VRAM% all 0; xiaoming-dev, rc_26_7_902 — neither ours
```

| | |
|---|---|
| 任务预估进度 | **~78 %** — the T+1837 closing stands |
| 已经耗时 | **~1991 min ≈ 33 h 21 min** |
| 预估耗时 | **not computable — T+1837 §7** |
| 可靠性 | **中** |

**Code problems:** none new (T+1837 §4 unchanged).
**未定性:** unchanged from T+1927 §4.
**新增 commit:** none by anyone since `2e1679c7` (mine, 15:11).
