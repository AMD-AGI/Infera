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

---

## T+94 — 2026-09-03 09:47 UTC

Twenty-six minutes. The most consequential period of the effort so far, and none
of it was new features: a fourth teammate was added to review the diff
adversarially, and it found a defect that would have made the run's headline
result meaningless rather than merely failed.

### 1. Progress

**Effort: ~80 %.** Elapsed 1 h 34 m. Estimated remaining: **1–2 h**, unchanged
for the fourth section running, and I am now stating plainly that this number
has not moved because I have no way to size the one thing left. The run has not
been launched.

| workstream | est. % | basis |
|---|---|---|
| probes | **100 %**, now including the re-measurement | verified: `PROBES.md` 147 → 188 lines, `B'/C'/F'` section at line 157 |
| core changes | ~95 % | verified: committed in `45d2a12`, plus 10 files re-modified since |
| `examples/env_checker` | ~95 % | verified: committed; 3 files re-modified since |
| `agent_sys/components/` | **100 %** committed | verified in `45d2a12` |
| tests | ~95 % | reported 2198 passed / 3 skipped / 4 xfailed, 43 module tests; verified only that `test_agent_assets.py` is +541 lines since the commit |
| review round | ~90 % | reported: 4 findings, all fixed; verified only the blocker's fix shape (`_NOT_PLACED`) |
| **the one real run** | **0 %** | verified: no run root anywhere under the scratch dir |
| commits | ~60 % | verified: 2 commits landed, 6804 insertions; the CLI/PATH and review fixes are still uncommitted |

**Reliability: medium-high on the artefacts, and I want to flag an asymmetry.**
Everything I can verify says the work got *better* in this period while the
percentage moved only 8 points, because what changed was correctness of code
already counted as present. If the reviewer had not run the installer, the
percentage would look the same and the effort would have been in a worse state.
A completion percentage cannot see this class of change, which is an argument
for reading §3 rather than §1.

### 2. Current state

**Two commits have landed.** `git log`:

- `45d2a12 feat(agent_sys): per-agent components at three install levels, and examples/env_checker` — **54 files, 6804 insertions, 29 deletions**
- `5834cc0 docs: update the running work log for the per-agent components round`

Working tree since those: 10 modified, 3 untracked, **961 insertions / 135
deletions** — the review fixes and the CLI/PATH fix, going in as a follow-up.
Modified: `env_mgr/{agent_assets.py,docs/design.md,isolation/policy.py,material.py,paths.py,prepare.py}`,
`tests/env_mgr/test_agent_assets.py` (+541), and three package files
(`ACCEPTANCE.md`, `README.md`, the genuine validator's `readme.md`).

- **Four teammates now**: `core-impl`, `pkg-author`, **`reviewer`** (new,
  read-only), and this reporter.
- **`core-impl` declined to review its own rewritten code**, citing the rule
  against verifying one's own work; the lead spawned `reviewer` instead. Recording
  the refusal because a teammate declining a task *on principle* and being backed
  is the mechanism working, not friction.
- **T+68's serena question is resolved in the artefact.** `git ls-files` on
  `assets/env_probe.agent/.serena/` returns nothing — the two files were kept out
  of `45d2a12`, and the directory now shows as a single untracked `??` entry. The
  finding was actioned before the commit, which is the only time it is cheap.
- **The real run**: no `runs/` or `runroot` exists under
  `/tmp/yihou/agentsys_envchecker_20260903/`. Not launched.
- Pre-flight reported **9/9 green** (it grew to nine plus a `6b`). I did not find
  a pre-flight artefact in `logs/`, which holds only the six files it held at
  T+65 plus the updated `PROBES.md`.

### 3. Code problems — fixed / not fixed

**Fixed this period. Six defects, and the class matters more than the count.**

Found by `reviewer`, **by running the real installer rather than reading it**:

1. **BLOCKER — `_install_tree` copied `skills/` and `plugins/` and nothing else.**
   `hooks/` and `servers/` never reached the zone. Measured: the hook script
   absent while `settings.json` named it; for the L2 component the config tree
   was **entirely empty** while the install report said `ok`. Capability 5 worked
   **only by luck**. Three docstrings and a README asserted the copy that did not
   happen. — This is the effort's own headline failure mode reproduced inside its
   own installer: *a report saying `ok` over a directory containing nothing.*
2. **HIGH — path escape.** The marketplace `contained()` check ran *after*
   `copy_out`, with an author-controlled `manifest["name"]` joined into the path.
   Measured with `"name": "../../../ESCAPED"`: the tree **was written outside the
   zone**, and only then did the refusal fire. A guard that fires after the write
   is not a guard.
3. **MEDIUM** — `recipes:` and hand-bound `assets:` could escape the staged package.
4. **MEDIUM** — **three tests could not fail**, proven by mutation: deleting the
   settings-ordering behaviour left 34 tests passing; removing the `PYTHONPATH`
   pin also passed.

Found before the review, and they **masked each other**:

5. `agent_assets` called `claude` by **bare name** under the policy-derived
   `PATH`, resolving to `/usr/local/bin/claude` **2.1.197** while the session pins
   **2.1.246**.
6. Underneath it, `material.deploy` never passed a `PATH` at all, so the child had
   none. Fixing only 5 leaves `uv: not found`; fixing only 6 **activates** 5.
   Both fixed by pinning `Prepared.agent_cli` and threading `base_env`.

**The blocker's fix is the one worth remembering: the default was inverted.**
Verified by me in `agent_sys/env_mgr/agent_assets.py:225` —
`_NOT_PLACED = frozenset({SETTINGS_FILENAME, MCP_FILENAME, "plugins"})`, applied
at line 701 as `if name in _NOT_PLACED: …`, with the docstring at 644 stating
each member is a **read** or a **relocation**, not a skip. Everything under
`.claude/` is now placed *unless* it is in that closed set, because a copy-list
is an allow-list and an allow-list fails **silently** on anything nobody thought
of. `plugins/` is in the set as a relocation, not an omission — the comment at
252 says so.

Reported after the fixes: `pytest agent_sys` **2198 passed / 3 skipped / 4
xfailed**, 43 module tests, and **five mutations now fail that previously
passed**. I did not run the suite. The mutation count is the only one of these
numbers that measures whether the tests can detect anything.

**Not fixed:** nothing above is retired by an execution. Six defects were found in
code that had already passed load, pre-flight and a green suite.

### 4. Non-code problems

- **`git gc` reports too many unreachable loose objects in the shared repo across
  eleven worktrees.** Unactioned, and correctly so: `prune` deletes, the repo is
  shared, and it is the user's call. Recording it here so it is not lost — this is
  exactly the shape of the operation the hard rule exists for.
- The T+65/T+68 entries stand unchanged.

### 5. Open questions, not yet characterised

- **The run.** Fourth section in a row. Still the whole of the remaining risk.
- I did not find the nine-item pre-flight recorded as a file; `logs/` is unchanged
  apart from `PROBES.md`. Whether it lives in a teammate's transcript, in
  `ACCEPTANCE.md`, or nowhere, I did not check.
- `poll_notes.md` has not been written since 09:06 (10004 bytes, unchanged across
  three of my readings).
- The 4 xfails: still unexamined, still assumed to predate this effort.

### 6. New commits

- **`45d2a12` `feat(agent_sys): per-agent components at three install levels, and examples/env_checker`** —
  54 files, +6804/−29. The whole first body of work: the three-level installer
  (`env_mgr/agent_assets.py`, `env_mgr/recipes/serena.yaml`), the core wiring
  across `spec_loader`/`agent`/`env_mgr`, `agent_sys/components/envchk-baseline/`,
  the complete `examples/env_checker` package, and the new/updated tests. The two
  `.serena/` files were **not** included.
- **`5834cc0` `docs: update the running work log for the per-agent components round`** —
  the work log.

### 7. Anything else worth recording

- **The two corrections the lead recorded against itself are the most useful
  entries of the period**, and both are the same error:
  - *"`uv` is fine, miniconda's bin is on the derived `PATH`"* — true of the
    **policy**, false of **what the code received**. Checked the policy, not the
    plumbing.
  - **Probes B, C and F were evidence about the wrong binary.** They ran without
    `cli_path`, and the SDK's `_find_cli` prefers its **bundled** CLI (2.1.251)
    over `PATH`. Three of six conclusions were about a build the run will not use,
    and **no artefact said so** — the write-up's header named 2.1.246 because that
    is what was on `PATH`, which was a true sentence about the wrong thing.

  Re-measured on the pinned 2.1.246: all three still positive, recorded as
  `B'/C'/F'` at `PROBES.md:157`. Verified by me: the file is now 188 lines, the
  header carries an explicit note that "2.1.246 is on `PATH`" says nothing about
  which binary ran, the section tabulates
  `claude_agent_sdk/_bundled/claude -> 2.1.251` against
  `shutil.which("claude") -> 2.1.246`, and the header now **requires every section
  to state which build it was measured against**. A probe write-up that cannot be
  misread the same way twice is worth more than the three re-runs.
- Both corrections and the blocker share one shape: **the artefact was consistent
  with the claim and the claim was about something else.** `ok` over an empty
  tree; a `PATH` version that no session used; a guard after the write; tests that
  pass with the behaviour deleted. Only running the thing distinguishes them, and
  in every case here it was running it that did.

---

## T+124 — 2026-09-03 10:17 UTC

**Run 1 happened.** Thirty minutes after the last section, and after four
sections of "the run has not been launched" it launched, failed, and failed
*correctly* — the one outcome that was worth waiting for and could not be
predicted from any artefact.

This is the first section in which I could open the run's own output. Most of
what follows is verified rather than reported.

### 1. Progress

**Effort: ~88 %.** Elapsed 2 h 4 m. Estimated remaining: **45 min – 1 h 30 m**,
and for the first time the estimate is grounded in something: one capability out
of seven is unwired, its fix is on disk, and run 2 is the only unknown left. The
first run's duration is now known — launched 09:57:22, its zones and handoff
complete, so the run itself is not an hours-long object.

| workstream | est. % | basis |
|---|---|---|
| probes | **100 %** | unchanged |
| core changes | ~97 % | verified: 5 fix commits since the feature commit; 685 insertions still uncommitted |
| `examples/env_checker` | ~95 % | verified: package committed, 7 files re-modified since |
| `agent_sys/components/` | ~95 % | verified: `components/serena/` now exists (untracked), `.claude/.mcp.json` + `README.md` |
| tests | ~97 % | reported 2213 passed, fourteen mutations failing that previously passed |
| **run 1** | **done** | verified: run root, both verdicts, the handoff, six of seven tokens |
| **run 2** | **0 %** | verified: only one run directory exists |
| commits | ~80 % | verified: HEAD `c92063a`, six commits ahead of `9bf72c8` |

**Reliability: high for run 1, medium for the remainder.** I read the run's own
files. What I cannot judge is whether the serena wiring is *sufficient* — that is
precisely what run 2 measures, and `pkg-author`'s static check (below) is
evidence about the gap, not about the fix's completeness in a live session.

### 2. Current state

**Run 1**, verified from `/tmp/yihou/agentsys_envchecker_20260903/runroot/runs/20260903T095722-53312a/`:

- Two verdicts under the same validation zone, on handoff
  `7a53f3dc-02d1-48b6-a461-83d8cb6b9278`: one `true`, one `false`. Matches the
  reported `check_env_report_shape` **PASS** / `check_capabilities_genuine`
  **FAIL**.
- The handoff has **two versions**, `v0` (empty `claim`/`content`) and `v1` with
  `manifest.yaml`, `validation.yaml`, `content/README.md` and
  `content/items/text.json`.
- I parsed `text.json` myself. `nonce_digest 4254090a6627`, 13 install-report
  entries, and the seven rows:

  | capability | level | status | token |
  |---|---|---|---|
  | `skill` | L3 | ok | `ENVCHK-SKILL-f5668adf55df` |
  | `hook` | L3 | ok | `ENVCHK-HOOK-1a041c200f74` |
  | `plugin` | L3 | ok | `ENVCHK-PLUGIN-89e6b7d27ee1` |
  | `mcp_external` | L2 | ok | `ENVCHK-MCP_EXTERNAL-9090b9…` |
  | `mcp_stdio` | L3 | ok | `ENVCHK-MCP_STDIO-ab586bb49…` |
  | `tooldef` | L3 | ok | `ENVCHK-TOOLDEF-190b154f069…` |
  | **`serena`** | **L1** | **unavailable** | **`null`** |

  Six distinct tokens, one per capability, and the failing row carries `null`
  rather than a value — the scheme behaved as designed.

- Reported by the lead and **not** independently recomputed by me: the six tokens
  recompute from the salts in their own artefacts, the hook payload carried
  `session_id` and `hook_event_name: SessionStart`, and the three-level install
  was confirmed by opening the zone — `config/hooks/envchk_session_start.py`,
  `config/servers/envchk_baseline_server.py`, `config/marketplaces/envchk-mp`,
  `config/skills/`, `config/tools/` all present. That last one matters: it is
  `reviewer`'s BLOCKER fixed **in reality**, not only in tests.

**Repo**: HEAD `c92063a`, six commits ahead of `9bf72c8`. Working tree 12
modified / 4 untracked, **685 insertions / 41 deletions** uncommitted. New
untracked `agent_sys/components/serena/` — `.claude/.mcp.json` and `README.md`,
i.e. **declaration only**, no installer, which is the correct shape given the
diagnosis below.

Scratch has grown to 14 directories including `judge/`, `review/`, `selftest/`,
`redtest_a/`, `scale/` and a `serena.recipe.draft.yaml`.

### 3. Code problems — fixed / not fixed

**The defect run 1 was for, verified by me by reading the agent's own report:**

**Nothing in the package declared serena as an MCP server.** The recipe installed
the binary; the only `.mcp.json` declared `envchk_baseline`. The agent's
`README.md` in the handoff records the investigation in full — it called
`mcp__serena__find_symbol` as briefed, got `No such tool available`, then checked
rather than stopping: `$UV_TOOL_BIN_DIR` holds `serena`, `serena-agent`,
`serena-hooks`, the install report says `recipe serena.yaml: OK` with
`serena-agent 1.7.1.dev0` from git, and
`grep -rl serena "$CLAUDE_CONFIG_DIR" --include='*.json'` found nothing. Its
conclusion: **binary installed; MCP registration never happened.**

Reported by the lead, and the sentence worth keeping: the serena entry existed
**as a comment in the recipe**, correct in every detail. *A comment is not a
declaration.*

**The agent's own `## Limits` section is the best artefact this effort has
produced.** Verified, quoting the structure: it scores its own serena result as a
FAIL and hands it over anyway; it notes the brief permits `"unavailable"` only
when `install_report` carries a non-`ok` outcome naming serena, and that it does
not — every serena entry is `info`; and it explicitly records that the salt *was*
available to it by `Read`ing `serena_probe.py` and that it did not use it,
because reading a file and reaching a capability are different things and that
difference is the whole scheme. An agent that could have forged a passing row,
declined, and said so in writing.

**Fixed since the run** (reported; I verified the commits exist):

- `205ff8f` marketplace name must be a single directory name, + a PATH test
- `e1b9f54` tooldef module keyed on its source, + cross-component overwrite warning
- `c92063a` symlinks resolved at every depth

Two more precede the run and were not in my last section: `04e8f97` (pin the CLI,
give the child a PATH, place every member of a `.claude/` tree — the CLI/PATH and
BLOCKER fixes) and `99d3aea` (import a tooldef from the zone copy, not the
component source), which is the HEAD run 1 was pinned at.

**Fixed, and the fix is a second detector, not just a patch**: `pkg-author` built
a **static** check that catches run 1's gap **with no run at all** — green on the
real tree, red against a copy with serena's component removed, and it produces
run 1's message verbatim. serena is now wired as L2 `components/serena/`
(declaration) while the agent keeps `recipes: [serena]` (install). Run 1 proved
**neither half implies the other**, and the fix keeps them as two halves rather
than merging them.

**Not fixed:** run 2 has not run. Uncommitted: the three-route naming fix and
`reviewer`'s 15-member closure fixture.

### 4. Non-code problems

- A **scope line is in force** from here to run 2: *blocking* means it would make
  run 2 fail, make its result unattributable, or write outside the zone; anything
  latent for a package that does not exist yet is recorded for a follow-up.
  `reviewer` **checked that classification rather than accepting it** and found no
  reachable path to any of the three latent findings. A scope rule that is itself
  audited is a different object from a scope rule that is asserted.
- The `git gc` unreachable-loose-objects item stands, unactioned, the user's call.

### 5. Open questions, not yet characterised

- **Run 2.** Whether declaring serena at L2 is sufficient for
  `mcp__serena__find_symbol` to exist in the session is unmeasured. Probe E
  showed the installed serena answers an MCP handshake and serves 21 tools; that
  is not the same statement as the tool appearing in this package's session.
- The two validation zones contain `args.json` / `inputs.json` /
  `materials.json` / `verdict.json` and no stdout or reason file that I could
  find, so the *reason* the genuine validator failed is not recoverable from the
  zone alone — I inferred it from the handoff. Whether a validator is expected to
  leave a reason behind, I do not know.
- Handoff `v0` exists with empty `claim` and `content`. Normal versioning or a
  first attempt, I did not determine.
- The 4 xfails: still unexamined.

### 6. New commits

Five since T+94, four of them fixes:

- **`04e8f97`** `fix(env_mgr): pin the CLI, give the child a PATH, and place every member of a .claude/ tree` — the two masked CLI/PATH defects and `reviewer`'s BLOCKER, in one commit.
- **`99d3aea`** `fix(env_mgr): import a tooldef from the zone copy, not the component source` — **the HEAD run 1 was pinned at.**
- **`205ff8f`** `fix(env_mgr): a marketplace name must be a single directory name, and test the PATH fix`
- **`e1b9f54`** `fix(env_mgr): key a tooldef module on its source, and report a cross-component overwrite`
- **`c92063a`** `fix(env_mgr): resolve symlinks when placing a .claude/ tree, at every depth`

### 7. Anything else worth recording

- **The lead's `judge.py` scored the run 11/11 and was wrong; the package's
  validator was right.** It read `level: "info"` as non-`ok`, so a benign
  `info recipe serena.yaml: OK` counted as a failed install and let serena's
  `unavailable` through against what was actually a clean report. Fixed, and it
  now independently reaches 10/11 with serena FAIL. Two things follow. First, the
  independent judge was written **before** the run and deliberately did not reuse
  the package's validator — which is why the disagreement was informative rather
  than invisible. Second, **the disagreement was resolved by finding the judge
  wrong**, which is the outcome an independent check is *for* and the one that
  feels least like progress at the time.
- **The lead retracted a one-line spec it had given `core-impl`** after
  `pkg-author` opened the real install report and found the three MCP routes
  record three *different* shapes: the fix is three lines, not one. Third
  correction of the day with the same root — a claim checked against the design
  rather than against the artefact.
- Run 1's headline, stated the way the effort should state it: **six of seven
  capabilities verified independently, the seventh failed for a real reason, and
  the failure was reported honestly by the agent that could have forged it.** A
  green run 1 would have been a weaker result than this one.

---

## T+222 — 2026-09-03 11:56 UTC

**Run 2 happened, and the design caught something no test could have.** Ninety-nine
minutes since the last section — the longest gap in this file, and the interval in
which the effort's most interesting result landed.

I was able to do a differential comparison across both runs' handoffs myself, so
the central finding below is **verified first-hand**, not relayed.

### 1. Progress

**Effort: ~92 %.** Elapsed 3 h 43 m. Estimated remaining: **unknown, and larger
than it looked at T+124.** I am revising the direction of the estimate for the
first time. At T+124 I said 45 min – 1 h 30 m with one capability left to wire;
that capability is now wired and passing, and a *different* problem appeared
underneath it which is explicitly **beyond the approved plan and open with the
user**. Remaining time is now gated on a decision, not on work.

| workstream | est. % | basis |
|---|---|---|
| the seven capabilities | **7/7 reached** | verified: all seven rows `ok` in run 2's handoff |
| serena (L1 install + L2 declaration) | **done** | verified: `ENVCHK-SERENA-582b29cd2876`, status `ok` |
| **capability 6 (`tooldef`) correctness** | **open** | verified failing: token identical across two runs with different nonces |
| core changes | ~97 % | verified: HEAD `9a9fdff`; nothing committed since |
| tests | ~97 % | reported 2216 passed |
| run 2 | **done** | verified: `runroot2/runs/20260903T103441-53acfd`, both verdicts |
| the contract decision | **0 %, user's** | reported open with the user |

**Reliability: high on run 2's facts, and I decline to estimate the remainder.**
The percentage above counts capabilities reached, which is now a poor summary:
seven of seven arrived and one of them arrived carrying a false value. That is a
better state than run 1 and a *less finished* one than "7/7" suggests.

### 2. Current state

**Run 2**, verified from
`/tmp/yihou/agentsys_envchecker_20260903/runroot2/runs/20260903T103441-53acfd/`:

- A **fresh `runroot2`** — verified, two run roots now exist side by side.
  Reported reason: reusing run 1's root would have manufactured the ambiguity that
  abort condition 3 exists to detect. Not reusing a root you are about to reason
  about is the same discipline as pinning a run id at launch.
- Both verdicts on handoff `9ccfd8af…`: one `true`, one `false`. Shape **PASS**,
  `check_capabilities_genuine` **FAIL**.
- `nonce_digest 41dc185e5731`, **14** install-report entries (run 1 had 13), and
  **all seven rows `ok`**:

  | capability | level | token |
  |---|---|---|
  | `skill` | L3 | `ENVCHK-SKILL-9fede1294ce5` |
  | `hook` | L3 | `ENVCHK-HOOK-d950b4f53084` |
  | `plugin` | L3 | `ENVCHK-PLUGIN-b3b4c6c9de79` |
  | `mcp_external` | L2 | `ENVCHK-MCP_EXTERNAL-15112e08e7a5` |
  | `mcp_stdio` | L3 | `ENVCHK-MCP_STDIO-acb5fa842482` |
  | `tooldef` | L3 | `ENVCHK-TOOLDEF-190b154f0697` |
  | `serena` | **L1** | `ENVCHK-SERENA-582b29cd2876` |

- serena passed **end to end** — install *and* declaration — with `find_symbol`
  returning `body_location {'start_line': 45, 'end_line': 66}`, reported as the
  same numbers `pkg-author` measured before writing the static check. The install
  report now names what it produced, `{"names": ["serena"]}`, where run 1 had
  `details: {}`.

**Repo**: HEAD `9a9fdff` (`feat(env_mgr): the install report names every MCP
server and tool it produced`). The lead reports the tree clean; `git status`
shows **9 modified package/component files plus untracked `components/serena/`**
— `check_capabilities_genuine/{check.py,readme.md}`, `assets/lib/envchk.py`,
`steps/check.yaml`, `probe_env.task/readme.md`, both `README.md`s under
`components/`, and `ACCEPTANCE.md`. Nothing under `env_mgr/` or `tests/`. So
"clean" holds for the **core**, and run 2 ran with the serena wiring
**uncommitted**. Stating the discrepancy, not resolving it.

### 3. Code problems — fixed / not fixed

**Fixed:** run 1's gap. serena declared at L2, installed at L1, reached in a real
session. `pkg-author`'s static check caught it before run 2 and would have caught
it with no run at all.

**FOUND, not fixed — and this is the finding of the effort:**

**An in-process `ToolDef` cannot see the run's environment.** Capability 6's
token was computed from an **empty** `$ENVCHK_NONCE`.

I verified this myself by differencing the two handoffs. Six of seven tokens
change between run 1 and run 2, as they must — different nonces. One does not:

```
skill         diff   f5668adf55df -> 9fede1294ce5
hook          diff   1a041c200f74 -> d950b4f53084
plugin        diff   89e6b7d27ee1 -> b3b4c6c9de79
mcp_external  diff   9090b978deb0 -> 15112e08e7a5
mcp_stdio     diff   ab586bb4903e -> acb5fa842482
tooldef       SAME   190b154f0697 -> 190b154f0697     <- two runs, two nonces, one token
serena        diff   (null)       -> 582b29cd2876
```

`ENVCHK-TOOLDEF-190b154f0697` is byte-identical across two independent runs with
different nonces. The lead's arithmetic — run 2's nonce yields `043e2bc708f0`,
run 1's `70877fcb8947`, the **empty** string yields `190b154f0697` — I did not
recompute, but the invariance across runs is sufficient on its own: a token that
does not move when the nonce moves is not derived from the nonce.

Mechanism, reported and independently corroborated twice: an in-process `ToolDef`
runs in the **supervisor's** process, while `Prepared.environment` goes to the
**CLI child**. `core-impl` probed through `_adapt_tool` — same pid as the
supervisor, variable absent. `reviewer` reproduced the **byte-identical** failing
token **with no model call at all**, and supplied a differential control drawn
from run 2's own pids: `mcp_external` 672056 and `mcp_stdio` 672058, CLI-spawned
seconds apart, against `tooldef` **669295**, a process that existed long before.
Three independent routes to one conclusion, one of them a process-identity
argument that does not depend on the code at all.

**The agent did nothing wrong.** Correct call, correct namespace, faithful
quotation of what it received. *The tool lied to it.* This is the exact failure
the token scheme was built to detect and the exact failure a narrative report
would have hidden — the row said `ok`, the `how` field would have described a
real call, and only the value betrayed it.

`reviewer` also widened the blast radius past the lead's statement of it: there
is **no channel at all** by which an in-process tool can learn the zone
environment. And it found the working precedent — `env_mgr`'s own remote
`ToolDef`s take their per-run context as **constructor parameters** and close over
it, so they are immune. A fix shape exists in this codebase already.

**Ruled out, with a reason:** (B) setting `os.environ` around the handler is
**dead** — threaded runner plus `asyncio.to_thread` means concurrent attempts take
each other's values. This is the same argument that moved L1 installs to a
subprocess at T+65; the second time today that process-global mutation has been
refused on concurrency grounds.

**Ruled in independently:** the **bundled** MCP entry gets an explicit `env`,
because it currently works **only by inheritance through the CLI child** and
nothing states why. A capability that works for an unstated reason is a capability
that breaks silently later.

**`core-impl` measured, proposed shapes, and implemented nothing.** Recorded
approvingly: the contract question is the user's, and building the fix first
would have made the decision for them.

### 4. Non-code problems

- **Open with the user, beyond the approved plan**: whether `agent_sys`'s
  in-process `ToolDef` contract should change so the route can be run-aware at
  all, or be **documented as the supervisor's environment by construction**. Both
  are defensible and the choice is not the team's.
- The `git gc` item stands, unactioned, the user's call.

### 5. Open questions, not yet characterised

- **The second failure the lead reported from run 2 may not exist.** Recorded
  here as the lead asked, as a correction rather than background: it reported
  *two* failures, the second being `$AGENT_SYS_COMPONENTS_ROOT` unset so the L2
  row could not be re-derived. Because a validator's reason is **not recoverable
  from the artefacts** (the same gap I recorded at T+124), it re-ran the validator
  by hand — and its own shell did not have that variable set. `reviewer`'s A/B
  measurement shows exactly that shape: set → 1 FAIL, unset → 2 FAILs. **So run 2
  may have had only one failure.** `reviewer` also refuted its own earlier
  prediction as the cause. Both are suspending rather than offering a mechanism,
  and so am I.

  The general form is worth more than the instance: **the recovery method
  contaminated the measurement**, and it was only detectable because someone
  measured the recovery method too.
- Whether the constructor-parameter precedent transfers to package-declared
  tooldefs, which are imported from a file rather than constructed by `env_mgr`,
  is not something I can judge.
- The 4 xfails: still unexamined.

### 6. New commits

One since T+124:

- **`9a9fdff`** `feat(env_mgr): the install report names every MCP server and tool it produced` — the HEAD run 2 was pinned at. Visible in the artefact: run 2's report carries `{"names": ["serena"]}` where run 1 carried `details: {}`, and the entry count went 13 → 14.

### 7. Anything else worth recording

- **This is what the token scheme was built for, and it is the only thing that
  could have caught it.** A capability that ran, in the right namespace, returning
  a well-formed string, from an agent behaving correctly — and the string was
  computed from an empty variable. No exit code, no validator shape check, no
  narrative `how` field, and no unit test of the tool in isolation would have
  seen it. Two runs with different nonces and one unchanged token did. The
  decision at T+65 to make evidence token-based rather than narrative is now paid
  for.
- **Run 2 is a better result than a green run 2 would have been**, for the second
  time in this effort. Run 1 proved install and declaration are independent; run 2
  proved the in-process route cannot see the environment. A clean sweep would have
  shipped the second defect unnoticed, because six of seven capabilities do get
  their environment and the seventh's failure is invisible in a single run.
- Three corrections have now been recorded against the lead's own reports in one
  day (the `PATH` policy-vs-plumbing claim, the probes on the wrong binary, the
  `judge.py` scoring, and now this one — four). Every one was found by someone
  measuring the instrument rather than the subject.

---

## T+243 — 2026-09-03 12:16 UTC

Twenty minutes. No new run of the package's own accord, but **two process
incidents**, four commits, and a user exchange that added an option nobody on the
team had proposed. This is the first section whose content is mostly about how
the team works rather than what the code does, and both incidents are worth the
space because both were **reported by their author before anything was touched**.

### 1. Progress

**Effort: ~93 %.** Elapsed 4 h 3 m. Estimated remaining: **still gated on a
decision, not on work**, and I continue to decline a number. Nothing is being
implemented by design.

| workstream | est. % | basis |
|---|---|---|
| the seven capabilities | 7/7 reached | unchanged from run 2 |
| capability 6's *meaning* | **open, three options now** | reported: the user added a third |
| core changes | ~98 % | verified: HEAD `b299bf2`, package tree **clean** |
| tests | ~98 % | reported **2217 passed / 3 skipped / 4 xfailed**, and the lead states it ran the suite itself |
| commits | ~95 % | verified: four new, tree carries nothing but my own file |
| run 3 (the real one) | not launched | verified: `runroot3` does not exist |

**Reliability: high on state, and the state is genuinely quiet.** `git status`
now shows only `work.checkpoint.summary.md` (mine) plus the two long-standing
untracked files at the repo root. The nine modified package files I flagged at
T+222 are committed. The discrepancy I raised is resolved in the artefact.

### 2. Current state

**Repo**, verified: HEAD `b299bf2`, ten commits ahead of `9bf72c8`. Working tree
clean apart from this file.

**Runs**, verified — there are now **three** run directories:

```
runroot/runs/20260903T095722-53312a    run 1
runroot2/runs/20260903T103441-53acfd   run 2
runroot2/runs/20260903T120338-a3fe5d   the stray, 12:03:38
```

The stray landed **inside `runroot2`**, which is how it reached run 2's launch
records. It is **kept, not deleted** — the correct call, and consistent with the
hard rule this file has carried since T+0.

**Run 2's judged artefact is intact.** I verified the exact numbers the lead
gave, independently: `items/text.json`, **13804 bytes, mtime 10:37**. The lead
additionally reports re-running `judge.py` against it with identical output,
11/11; I did not re-run it. Three log files did not survive — `LAUNCH2.md`, the
console log, `stream.json`. Pins were reconstructed into a file **explicitly
marked as reconstructed**, which is the only honest way to hold a pin you did not
capture.

### 3. Code problems — fixed / not fixed

**Fixed and committed** — `13154c6` is the substantive one and I read its stat:
13 files, +560/−42, spanning `env_mgr/agent_assets.py` (+65),
`components/serena/` (new, +113), both component READMEs, `ACCEPTANCE.md` (+127),
`assets/lib/envchk.py` (+72), the genuine validator, `steps/check.yaml`, and
`tests/env_mgr/test_agent_assets.py` (+47). Its message states both halves: **a
bundled MCP entry states the run environment** (the "works only by inheritance,
and nothing says why" item ruled in at T+222), and **record what an in-process
tool cannot see**.

**Not fixed, deliberately, and now with three options rather than two** — see §4.

### 4. Non-code problems

**Incident 1 — one git index, four agents.** `pkg-author` staged eleven package
files; `core-impl` staged two and ran a bare `git commit`, which committed **the
whole index**. Eleven of `pkg-author`'s files landed in `13154c6` under a message
about `env_mgr`. Nothing lost, unpushed, DCO fine — both commit as the same
identity.

**The ruling was: do not reset.** `git reset --soft` mutates a **shared** index
while another agent may be mid-`add`, in order to repair an attribution error in
which **every byte is intact**. The lead's formulation: *the cure touches shared
state and the disease does not.* The run-to-tree binding was restored by an
**empty commit** (`90b34fe`, `docs(env_checker): record which tree run 2
executed`) instead. New rule in force: `git commit -- <explicit paths>`, with the
staged set read back before committing.

This is the same shape as the `git gc --prune` item that has sat unactioned since
T+94, and it is now the second time today the answer to "shared state, cosmetic
gain" has been *don't*. Recording that the rule generalised rather than being
re-derived.

Attribution, as the lead stated it: the protocol failure is the lead's, the
command is `core-impl`'s — and `core-impl` **insisted on owning the second half**
when the lead tried to take the whole thing.

**Incident 2 — a gate tested by firing it.** `pkg-author` added a gate refusing
to launch on a dirty package tree, then ran `launch.sh` to check the gate passes.
The gate passed, **so the script launched**, and a third run started and
overwrote run 2's `LAUNCH2.md`, console log and `stream.json`.

Its own diagnosis: **a gate whose only self-test is a live launch will be tested
by launching.** Fixed by adding `--check`, verified by confirming it creates
nothing. Reported clean by the lead: `launch.sh --check` green, `runroot3`
reserved and non-existent.

Both incidents were disclosed by their author before remediation. That is twice
today (three times counting the two-writers-on-one-file overlap at T+65) and it
is the reason both are recoverable entries in a log rather than mysteries.

**The user exchange, which added a third option.** The lead had framed the
in-process `ToolDef` question as *document it* vs *change `agent_sys`'s
contract*. The user asked whether the **verification method** could be redesigned
instead — and that is a third route the team had not proposed: **have the model
pass the nonce as a tool argument**, since `defn.call(**args)` already carries
the model's arguments. No `agent_sys` change, freshness preserved, the salt still
only in the artefact; and arguably the correct design for an in-process tool —
inputs through arguments, not ambient environment.

The cost, which the lead insists goes **into the acceptance table rather than
being slipped in**: it changes what capability 6 *proves*, from *the tool can see
the run's environment* to *the tool is callable and computes correctly*. The
latter is what the route can offer; the former it cannot. Its recommendation is
that **plus** documenting the limitation, because **the limitation does not
disappear because we changed how we test around it**.

**Still not ruled. Nothing is being implemented.**

### 5. Open questions, not yet characterised

- **Whether run 2 had one failure or two — still open, and deliberately so.** The
  lead's second finding may be an artefact of its own re-run, whose shell lacked
  `AGENT_SYS_COMPONENTS_ROOT`; `reviewer` refuted its own predicted cause; and the
  lead could not close it from the artefacts because **the stream records no
  environment at all**. Instrumentation is in place so the next occurrence is
  evidence. It stays unresolved rather than being written up as a finding — which
  is the right disposal and the harder one.
- Whether the stray run at `20260903T120338-a3fe5d` produced a handoff of its own,
  and whether anything downstream could mistake it for run 2, I did not check
  beyond confirming the directory exists with the usual six subdirectories.
- The 4 xfails: still unexamined, seven sections running.

### 6. New commits

Four since T+222, all verified present:

- **`13154c6`** `fix(env_mgr): a bundled MCP entry states the run environment; record what an in-process tool cannot see` — 13 files, +560/−42. **This is the commit that carries eleven of `pkg-author`'s files under an `env_mgr` message**, per incident 1. The content is correct; the message describes two of the thirteen files' worth of intent.
- **`90b34fe`** `docs(env_checker): record which tree run 2 executed` — the **empty** commit that restored the run-to-tree binding without touching the shared index.
- **`0a5840b`** `docs(env_checker): the instrument lessons, and ignore serena's by-product` — which also finally disposes of the `.serena/` question first raised at T+65: it is now ignored rather than left untracked.
- **`b299bf2`** `docs(env_checker): record the launch gates' operating rule in the repo` — incident 2's lesson written where the next person will hit it.

### 7. Anything else worth recording

- **Three of the four commits in this period are `docs:`**, and none of them are
  documentation of features. They record: which tree a run executed, what an
  instrument taught, and how a gate must be operated. An effort that commits its
  process findings at the same cadence as its code is unusual and is the reason
  this checkpoint file has anything to say.
- **The user found an option four agents had missed.** Worth recording plainly:
  the team had converged on a binary — document the limitation, or change the
  contract — and the user's question was neither, it was *change the instrument*.
  The binary was not wrong, it was **narrow**, and it was narrow because everyone
  in it had spent two hours inside the same mechanism. The lead's response is the
  right one: adopt it, and refuse to let it quietly redefine what capability 6
  proves.
- `0a5840b`'s title pairs "the instrument lessons" with "ignore serena's
  by-product". The second half is a two-line `.gitignore` change I flagged at
  T+65 and the lead's own T+94 check found half-real. It took four hours and a
  commit to close, which is the normal cost of a small correct thing and worth
  noting against the temptation to skip them.

---

## T+273 — 2026-09-03 12:46 UTC

**Run 3 reached seven of seven** and one measurement closed three questions that
had been open since T+222. And while writing this section I found **run 4 already
in flight** — launched 12:43:21, its handoff written 12:45:07, validators not yet
finished. This section therefore reports a state the lead's own briefing predates.

### 1. Progress

**Effort: ~96 %.** Elapsed 4 h 33 m. Estimated remaining: **30–60 min**, and the
estimate is grounded again — run 4 is the last planned action and its handoff was
already on disk 1 m 46 s after launch.

| workstream | est. % | basis |
|---|---|---|
| the seven capabilities | **7/7 `ok` in two independent runs** | verified: run 3 and run 4 handoffs both show all seven `ok` |
| capability 6's correctness | **fixed and proven** | verified: the frozen token is gone — see §3 |
| `check_capabilities_genuine` | **PASS** at run 3 | reported; the two verdicts (one `true`, one `false`) are verified but not attributable from the zone |
| `check_env_report_shape` | fixed at `65754ba` | verified: `check.py` +51/−6, `README.md` +39 |
| run 4 | **in flight** | verified: handoff written 12:45:07, no verdicts yet at 12:46 |
| the contract decision | resolved by redesigning the instrument | reported |

**Reliability: high, with one attribution gap I could not close.** Run 3 has two
verdicts on handoff `96c08fea…`, one `true` and one `false`, and the validation
zones carry only `args/inputs/materials/verdict.json` — no name, no reason. Which
of the two is the shape check is the lead's report, not my reading. This is the
third section in which that same gap has appeared, and it is now the single
thing most limiting what I can verify.

### 2. Current state

**Runs — there are now five run directories:**

```
runroot /runs/20260903T095722-53312a   run 1
runroot2/runs/20260903T103441-53acfd   run 2
runroot2/runs/20260903T120338-a3fe5d   the stray (T+243, kept)
runroot3/runs/20260903T123051-f53888   run 3
runroot4/runs/20260903T124321-2d27d9   run 4, IN FLIGHT
```

**Run 3**, verified: `nonce_digest baeb5ad02e11`, 14 install-report entries, all
seven `ok` across all three levels.

**Run 4**, verified at 12:46: handoff `0d3c8e1c…` written at 12:45:07,
`nonce_digest a5a86ca7d43f`, 14 entries, all seven `ok`. Its single task zone was
still being written at 12:46:35 and **no `verdict.json` exists yet**. So: the
agent's half of run 4 is complete and passing; the validators' half is unfinished
at the time of writing. I am recording that split rather than a result.

**Repo**, verified: HEAD is **`65754ba`**, not the `47de7ed` the lead named — the
placeholder fix has already landed and run 4 launched behind it. Tree clean apart
from this file. Fifteen commits ahead of `9bf72c8`.

### 3. Code problems — fixed / not fixed

**Capability 6 is fixed, and I can prove it from the artefacts.** The token that
was frozen across runs 1 and 2 now moves:

```
run 1  ENVCHK-TOOLDEF-190b154f0697
run 2  ENVCHK-TOOLDEF-190b154f0697   <- identical: computed from an empty nonce
run 3  ENVCHK-TOOLDEF-1ede902d53cf
run 4  ENVCHK-TOOLDEF-f3bed5dfbcd7   <- three distinct values, three nonces
```

Two further independent values after the redesign. The route by which the model
passes the nonce as a **tool argument** — the user's third option, recorded at
T+243 — is measured working, twice. All other six tokens also differ between run
3 and run 4, as they must.

**Fixed:** `1f3fef0` `feat(env_checker): row 6 takes no input; add 6b, the
placed-copy guard`, and `47de7ed` which tells the agent what row 6's tool now
returns. Also `7d25011`, the positive rule for an in-process tool's per-run
context, and `50105f4`, which **corrects a false strength claim in
`ACCEPTANCE.md` section 3** — an effort correcting its own acceptance document
downward is worth a line.

**Fixed — the eighth instrument defect, and a new species.** `65754ba`
`fix(env_checker): the placeholder rule failed on correct input`, verified as
`check_env_report_shape/check.py` +51/−6 plus 39 lines of `README.md`. The rule
flagged the agent's Schema line ``` `ENVCHK-<LABEL>-<12 hex>` ``` as an unfilled
placeholder: **it cannot tell documenting a placeholder from leaving one.** The
agent did the right thing and the check was wrong.

The first seven instrument defects today were all *checks that could not fail*.
This one **fails on correct input** — the other face of the same coin, and the
repair points the opposite way. Recording the pair because a team that has spent
a day hardening checks against false negatives is exactly the team that will
introduce a false positive.

**And it is in five files.** `pkg-author` grepped: the same regex sits in five
places, lifted from `single_real_task`, and it verified **the original copy flags
the exact line run 3 died on**. The ruling — *fix ours only; record the other
four; do not build the shared helper* — is on the same principle as "do not
reset" at T+243: changing a validator in a package nobody is running would alter
an acceptance criterion for **already-accepted work without re-running it**.
Third time today that touching something shared for a non-blocking gain has been
refused.

`pkg-author`'s own diagnosis is the finding: *"I introduced it by lifting a regex
that already had it, without re-reading what it matched."* **A defect that
spreads by copying gets more entrenched with every reuse, and each copy arrives
carrying the authority of the file it came from.**

### 4. Non-code problems

- The four other copies of the placeholder regex are **recorded, not fixed**, by
  ruling. A follow-up exists and is not this round's work.
- The `git gc` item stands, unactioned, the user's call. Unchanged since T+94.

### 5. Open questions, not yet characterised

- **Run 4's verdicts.** Not written at the time of this section.
- **Which verdict belongs to which validator is not recoverable from a run's
  zones.** Third section running. Every attribution of a PASS or FAIL to a named
  validator in this file has come from a teammate's report, never from my reading
  of the run.
- The 4 xfails: unexamined, eight sections running. I am going to stop repeating
  this unless someone acts on it.

### 6. New commits

Five since T+243, all verified present:

- **`50105f4`** `docs(env_checker): correct a false strength claim in ACCEPTANCE.md section 3`
- **`7d25011`** `docs(env_mgr): the positive rule for an in-process tool's per-run context`
- **`1f3fef0`** `feat(env_checker): row 6 takes no input; add 6b, the placed-copy guard`
- **`47de7ed`** `docs(env_checker): tell the agent what row 6's tool now returns` — the HEAD run 3 was pinned at
- **`65754ba`** `fix(env_checker): the placeholder rule failed on correct input` — 2 files, +84/−6; the HEAD run 4 is running behind

### 7. Anything else worth recording

- **Three open questions closed by one measurement, and the closure is a chain
  worth preserving.** `check_capabilities_genuine` passed ⇒ the L2 row was
  re-derived ⇒ `components_root()` returned a path. It has two sources; the lead
  measured that the staged package has **no ancestor containing
  `agent_sys/components`** (printed the list, empty), and `pkg-author` had already
  established the upward search can never fire in a real run. Therefore the
  variable was present. Which yields: a real validation zone **does** carry
  `AGENT_SYS_COMPONENTS_ROOT`, §4's validation-zone third is **refuted**, and
  **run 2 had ONE failure, not two** — the lead's second finding was its own hand
  re-run's missing shell variable. **Recorded as a correction**, as asked, and it
  closes the item this file has carried open since T+222. Note the form: nobody
  measured the variable directly; a passing validator was used as an instrument to
  measure it, which is only sound because the alternative source was independently
  excluded first.
- **`core-impl` corrected the lead back, and its correction is sharper than the
  measurement.** The result shows the lead's unification was wrong **on the axis**,
  not in one of three parts: the discriminator is not *did this context receive
  the environment* but **spawned as a child versus sharing the supervisor's
  process**. A validation zone *is* spawned and does get one; an in-process tool
  is not spawned at all. Which is why option (D) has to be a **factory** rather
  than an environment fix. A junior correcting the axis rather than the answer is
  the most valuable kind of correction and the easiest to wave off.
- **The lead authorised both fixes without waiting on the user**, on the reasoning
  that a validator failing correct input is a **correction inside the plan's
  scope**, and that a package whose method is *prove it by running it* does not
  ship a repaired check unrun. Recording the reasoning, not just the decision,
  because the same two sentences would justify a scope creep if the first clause
  were dropped.
- Run 3's `judge.py` agreement at 11/11 is reported as meaningful **because the
  instrument was annotated before the run** rather than found to agree afterwards.
  That distinction is the whole difference between a corroboration and a
  coincidence, and it is the fourth time today the team has paid attention to it.

---

# Checkpoint summary — PR 155 review response: fold L1/L2/L3 into the recipe layer system

A **third effort**, appended below the five-module `llm_e2e` debug and the
three-level component install above. Neither earlier effort's sections are
touched.

Append-only, same discipline: one section per ~30 minutes, earlier sections are
never revised, wrong estimates are left standing because the record over time is
the value.

Effort start (T+0) taken as **2026-09-04 06:14 UTC**, the minute this round's
task book was backed up (`CLAUDE.envchecker.20260904-0614.md.bak`).

What is being done: the previous effort shipped the three-level per-agent
component install (L1 recipes / L2 `agent_sys/components/` / L3 auto-detected
`.claude/`) plus `examples/env_checker`, and opened **PR 155** (`AMD-AGI/Infera`,
head `dev.yihou.aiopt.task_with_agent_config` → base
`dev.yihou.aiopt.task_package`). The repo owner left review comments whose
central objection is that **L1/L2/L3 is a parallel concept for a job the existing
recipe *layer* system should already own** — *"more concept make more complexity
to maintain"*, *"L1 L3 is just a user friendly declare way, not a new system for
the core code"*, *"actually I think the L2 should not exist"*.

Owner rulings carried into this round (relayed by the lead, not read by me in
this form on the PR): four layers `system` → `workspace` (the `default`
mechanism) → `task_package` (the `main` mechanism) → `agent`; only the `agent`
layer may carry `.claude/` in assets; conflict handling **detection-only**, no
override; **`layer` is a system mechanism invisible to declarations and users**
(loader-assigned, removed from recipe YAML).

Teammates: **`researcher-core`** (the `layer` change set inside `env_mgr`) and
**`researcher-pkg`** (the package-facing items). Both **research-only** at T+0.

Reporter reads, cheapest first: `git status --short` / `git log --oneline` /
`git diff --stat` in the worktree; `/tmp/yihou/agentsys_pr155_20260904/research/`;
`agent_sys/docs/TODO.md` and `agent_sys/engineer_principle.md`;
`gh api repos/AMD-AGI/Infera/pulls/155/comments`.

---

## T+2 — 2026-09-04 06:16 UTC (baseline)

Written 2 minutes after the nominal T+0. This is a genuine baseline, not a
retro-fit: everything below was observed at 06:16, not reconstructed.

### 1. Progress

**Effort: ~3 %.** Elapsed 2 minutes. Estimated remaining: unknown.

| workstream | est. % | basis |
|---|---|---|
| review comments gathered | **100 %** | I fetched all 16 from the GitHub API and read every body |
| task book (`CLAUDE.md`) rewritten for this round | **0 %** | measured: `diff -q CLAUDE.md CLAUDE.envchecker.20260904-0614.md.bak` reports the files **identical**. The backup was taken; the replacement has not been written |
| `engineer_principle.md` — the principle the objection turns on | **100 %** | 15 added lines, read in full (see §6) |
| `docs/TODO.md` — the `assets/` item the owner opened | **100 %** | one new row `4g`, read in full |
| `researcher-core` output | 0 % observable | `/tmp/yihou/agentsys_pr155_20260904/research/` is **empty** |
| `researcher-pkg` output | 0 % observable | same directory, same emptiness |
| production code for this round | **0 %, measured** | `git diff --stat` names exactly two files, both documentation; no `env_mgr`, `spec_loader` or `examples/` file has changed |

**Reliability: very low, and I want to be precise about which way.** The four
100 % figures are solid — I opened those artefacts and diffed them. The two
zeroes for the researchers are *floors derived from an absence*, and at T+2min an
absence means **"has not yet written to a path I can see"**. I cannot distinguish
that from "is reading code right now and has nothing to write yet", which is the
overwhelmingly likely state two minutes in. Neither researcher has been observed
to fail at anything, and neither has been observed to do anything.

The **denominator is worse than usual this round.** 16 review comments is a count
of comments, not of work: three of them are questions the owner wants *answered*
(`nonce`, per-call MCP process model, `agent_assets.py` vs `material.py`), one is
a request for a written explanation of an existing overlay/fallback system, and
one — *"just reuse the recipe level system"* — is a core refactor whose size
nobody has estimated in my hearing. **A 3 % figure against that denominator is a
gesture.**

### 2. Current state

Worktree `/home/yihou/dev/git.16-19/infera.aiopt.real.task_package`, branch
**`dev.yihou.aiopt.task_with_agent_config`**, HEAD **`8cac792`**
(`docs(env_checker): the follow-up list, as items with triggers`, committed
2026-09-03 12:53 UTC — **17 hours before this checkpoint**, and the previous
effort's final commit).

Note a discrepancy I am recording rather than resolving: the still-current
`CLAUDE.md` states *"Branch is `dev.yihou.aiopt.task_package`"*, but that is the
**base** of PR 155; the checkout is on the **head**. This is consistent with
`CLAUDE.md` being the previous round's book, not with a stale checkout.

`git status --short`:

```
 M agent_sys/docs/TODO.md
 M agent_sys/engineer_principle.md
M  work.checkpoint.summary.md
?? CLAUDE.envchecker.20260904-0614.md.bak
?? CLAUDE.kernel_opt.20260903-0813.md.bak
?? progress.bar.for.user.md
```

Two tracked-and-modified files, both documentation, both edited this round
(mtimes 06:04 and 06:07 UTC). `work.checkpoint.summary.md` is staged with the
previous effort's 889 lines. The two `.bak` files and `progress.bar.for.user.md`
are untracked; `progress.bar.for.user.md` is stale from 2026-09-02 and belongs to
the kernel-opt effort, not this one.

**On disk in the scratch workspace** `/tmp/yihou/agentsys_pr155_20260904/`:
three directories — `logs/`, `research/`, `scratch/` — all created at 06:14 and
**all three empty**. So the workspace has been laid out and nothing has been put
in it.

**PR 155**: OPEN, 18 commits, `reviewDecision` empty (the owner's 16 comments
were submitted as 16 separate `COMMENTED` reviews, each with an empty top-level
body — so there is no summary review to read; the argument exists only in the
inline comments).

**What each teammate is doing**: I have no first-hand signal. The lead's brief
says both are research-only. I can confirm only that neither has written to
`research/`.

### 3. Code problems — fixed / not fixed

**Fixed this round: none.** No production code has been touched (measured, §1).

**Not fixed — the owner's 16 comments, verified by reading the API response.**
Grouped by what they demand, with the file each names:

*The central architectural objection (4 comments, all on
`agent_sys/examples/env_checker/steps/check.yaml:110`)*

- *"actually I think the L2 should not exist / and L1/L3 should also not exist in
  the core code. just reuse the recipe level system to implement it. make L3 as a
  new top layer or a repo layer"*
- *"more concept make more complexity to maintain"*
- *"L1 L3 is just a user friendly declare way, not a new system for the core code"*
- *"components is a tag identify that this item is from inside the agent_sys. the
  there L1/L2/L3 classify way is a virtual concept not a user aware thing"*
- (a fifth on the same line: *"components should also declared in the recipes."*)

*Declaration surface — `check.yaml:89`, the longest comment*

- rename `recipe[s]` → `env_recipe[s]`;
  `env_recipe.${agent_name}.yaml` auto-detected across the whole package;
  a package-level `main.env_recipe.yaml` at the workspace kind;
  **embedded recipe definitions not allowed** (reference an existing one only);
  and *"I don't think list/arr is needed here"*. It also **asks a question**:
  *"If i am not wrong, there is a overlay/fallback system of different kind
  recipe like workspace/repo/project..., tell me details about that and we make
  sure how next step do."*

*A new default recipe — `agent_sys/env_mgr/recipes/serena.yaml:1` (2 comments)*

- add a `default.yaml` used when a task package declares nothing for an agent:
  bottom-most kind, the owner's **own currently-installed Claude plugins**, `cli`,
  **`gh` cli** (added in a follow-up comment), and *"install something else, make
  it good enough for claude code to works for a develop and experiment job working
  in repos like infera/sglang"*, plus **print a warning when it is used**.

*Placement and implementation — `agent_sys/components/envchk-baseline/.claude/servers/envchk_baseline_server.py`*

- `:2` — *"the whole components folder should lives in the env_mgr folder as a
  sub-folder, as it a something about env"*
- `:129` — *"do not implement MCP from scratch / use standard lib"*, with a
  sketch importing `mcp.server.MCPServer`. **The comment's sketch also contains
  `import httpx2`**, which I am recording verbatim and not interpreting.
- `agent_sys/components/serena/.claude/.mcp.json:5` — *"I think there is an
  official claude code plugin can install serena, so why this file exists?"*

*Questions the owner wants answered, not code changes*

- `check.yaml:75` — *"what is nonce"*
- `check.yaml:76` — *"So is this a way to declare shell variable for a agent? It
  nice"*
- `...envchk_baseline_server.py:148` — *"So every time the mcp call is invoked by
  claude code, a new python process is start to response here?"*
- `agent_sys/env_mgr/agent_assets.py:35` — *"what's the relationship of this file
  and material.py"*
- `agent_sys/spec_loader/assets.py:341` — *"really? not sure ... so how did
  validators like `check_env_report_shape.validator` find their folder?"*

The last of these has already produced a written answer in the repo: `TODO.md`
row **4g** (§6) records the owner's ruling that the `assets/` mechanism *"resolves
entry points and pretends to be a resource mechanism"* and wants a small-scope
refactor. That is the only one of the five questions with an artefact behind it
at T+2.

### 4. Non-code problems

- **The task book for this round does not exist yet.** `CLAUDE.md` is byte-identical
  to the backup of the *previous* round's book. Anyone — human or agent — reading
  `CLAUDE.md` right now gets the L1/L2/L3 task that PR 155 is objecting to,
  described as the thing being built. This is the single highest-risk state on
  disk at T+2, and it is a *stale-instruction* risk, not a lost-work risk: the
  backup was taken correctly.
- **Two stale `.bak` files and a stale progress bar sit untracked at the repo
  root.** `CLAUDE.kernel_opt.20260903-0813.md.bak` is now two rounds old. Not a
  problem yet; noted so that a future section can say whether it grew.
- **The owner's argument is distributed across 16 zero-body reviews.** There is no
  single review body summarising the position, so any tool or teammate that reads
  "the review" rather than "the review *comments*" gets nothing. I hit this: the
  `reviews` endpoint returns 16 rows of `body_len=0`.
- **The brief says 15 comments; the API returns 16.** Recorded, not reconciled.
- Two research directories were created and left empty for 2 minutes. That is not
  a problem, it is the baseline; it is here so the T+30 section has something to
  compare against.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- The owner asks for *details about* an *"overlay/fallback system of different
  kind recipe like workspace/repo/project"*, prefixed with *"If i am not wrong"*.
  **I have not read `env_mgr/recipe.py` and do not know whether such a system
  exists, whether it has those kind names, or whether the four-layer ruling the
  lead relayed is a description of it or a replacement for it.** The measurement
  that would answer it: read the kind/layer resolution in `env_mgr/recipe.py` and
  `env_mgr/cli.py` and name what is actually there.
- The lead's brief states the four layers as *rulings already given*. I have not
  seen them in any of the 16 PR comments I read; the closest is *"make L3 as a new
  top layer or a repo layer"*, which names neither four layers nor `system` /
  `workspace` / `task_package` / `agent`. **They may have been given in a channel
  I cannot see.** I am recording that the provenance is the lead's brief, not the
  PR — per the rule that a teammate report and a read artefact are different
  grades of evidence.
- `import httpx2` in the owner's suggested MCP sketch. I do not know whether that
  is a package that exists, a typo, or something local. Not investigated.
- Whether the previous round's `examples/env_checker` and its two validators
  survive the refactor at all, or are rewritten against the new declaration
  surface. Nothing on disk speaks to this.

### 6. New commits

**None.** HEAD is unchanged at `8cac792` from the previous effort, 17 hours old.

Two **uncommitted** documentation changes exist and I read both in full:

- `agent_sys/engineer_principle.md` (+15). Adds a MUST row — ***"Never add a
  responsibility an existing mechanism can already cover"*** — plus three
  paragraphs arguing that a second concept for an existing job is *"the same
  failure wearing a friendlier face, and it is harder to see because nothing is
  obviously in the wrong place"*, and setting the order **find the owner, change
  the owner, and only then consider a new component**. This is the owner's PR 155
  objection generalised into the repo's standing principle, written **before** any
  code was changed to comply with it.
- `agent_sys/docs/TODO.md` (+2). Adds row **4g**, recording that `spec_loader/assets.py`
  finds *one file per role* (`body.readme`, `body.entry`) and that every other file
  an object needs arrives only because `layout.stage_package(include=None)` copies
  the whole package — so `body.entry` is *"a pointer, not a manifest"*, each body
  re-derives the same path string by hand, and a layout change *"breaks them one by
  one at run time rather than at load"*.

### 7. Anything else worth recording

- **The principle was written before the code.** `engineer_principle.md`'s new MUST
  row is the general form of the very objection that opened this round, and it
  landed at 06:04 — ten minutes before the effort's nominal T+0 and before any
  refactor exists. Whether the refactor then actually obeys it is the thing to
  check later; recording now that the standard was set first, so that a T+N section
  can hold the work against a rule that was not written to fit it.
- **Row 4g is an answer to a review question, filed as a TODO.** The owner's
  `assets.py:341` comment was a *question* (*"Am I missing something?"*), and the
  response on disk is not a reply but a recorded defect with a scope ruling. Worth
  watching whether the question also gets answered on the PR — a TODO row is not
  visible to the reviewer.
- **Nothing has been pushed and nothing has been replied to on GitHub.** All 16
  comments are, as of 06:16, unanswered.
- The last of the owner's comments landed at 05:40:51 UTC, 33 minutes before this
  checkpoint. So the review was still arriving very recently; a 17th comment is
  possible.

---

## T+39 — 2026-09-04 06:53 UTC

**A checkpoint firing at 06:38 produced no section.** The previous section is the
06:16 baseline; between it and this one there is a 37-minute hole in a 30-minute
cadence. `CronCreate`'s documentation says jobs fire only while the REPL is idle,
and long working turns covered that window — **that is a candidate, not a
finding**, and I am not recording it as the cause. What is recorded: the cadence
missed a window, and this section covers 37 minutes rather than 30.

### 1. Progress

**Effort: ~7 %.** Elapsed 39 minutes. Estimated remaining: unknown.

| workstream | est. % | basis |
|---|---|---|
| review comments gathered and read | **100 %** | unchanged from T+2 |
| research — `researcher-core` | **~100 % of what was asked** | `core.md` (312 lines, 06:19) and `core-changeset.md` (668 lines, 06:34) on disk; I read their heading structure, not every line |
| research — `researcher-pkg` | **~100 % of what was asked** | `pkg.md` (508 lines, 06:27, incl. a `SERENA_HOME` addendum at 06:36) |
| lead's own hypothesis | **100 %** | `lead-analysis.md` (3.5 kB, 06:20), read in full |
| docs / spec written this round | **~4 items** | 3 commits landed: `engineer_principle.md` §2 MUST, `TODO.md` 4g and 4h, `env_mgr/docs/spec.md` §9.1 + principle 8 |
| task book (`CLAUDE.md`) for this round | **0 %** | still byte-identical to the backup — deliberate sequencing, see §4 |
| **production code for this round** | **~0 %, measured** | `git diff HEAD --stat` names **one** file, `work.checkpoint.summary.md`. No `env_mgr`, `spec_loader` or `examples/` source file has changed. The only non-doc commit is a **test** fix |
| PR 155 comments answered on GitHub | **0 % observable** | I have not checked the PR for replies this window; last measured at T+2, all 16 unanswered |

**Reliability: low, and the shape of the doubt has changed since T+2.** The
research numbers are now solid in one direction only — I can see 1,488 lines of
research artefact and I read their structure, so "the research phase produced
output" is artefact-verified. Whether that output *answers* the owner is not
something I measured; I read headings, not arguments.

The implementation figure is the one to trust: **near-zero, and I am stating it
as such deliberately.** Three commits landed this window and **not one of them is
the change set.** Two are documentation and the third is a test assertion. The
round's actual work — folding L1/L2/L3 into the recipe layer system — has not
begun, and §5 says why.

The denominator got *better* this window and the news is not good: `core-changeset.md`
now costs the core edit at 668 lines including **36 fixture sites**, so there is
for the first time a real size estimate, and 7 % against it is if anything
generous.

### 2. Current state

Branch `dev.yihou.aiopt.task_with_agent_config`, HEAD **`33b9f56`** (was `8cac792`).
Three commits this window, all by `yihou`, all signed off, all made with explicit
pathspecs.

`git status --short` is now **four** entries, down from six:

```
MM work.checkpoint.summary.md
?? CLAUDE.envchecker.20260904-0614.md.bak
?? CLAUDE.kernel_opt.20260903-0813.md.bak
?? progress.bar.for.user.md
```

`agent_sys/docs/TODO.md` and `engineer_principle.md` have gone from modified to
committed. **The `MM` on my own file is a hazard and it is mine** — see §3.

**Scratch workspace** `/tmp/yihou/agentsys_pr155_20260904/` — no longer empty:

| path | size | mtime |
|---|---|---|
| `research/core.md` | 23 kB / 312 lines | 06:19 |
| `research/lead-analysis.md` | 3.5 kB | 06:20 |
| `research/pkg.md` | 29 kB / 508 lines | 06:27 |
| `research/core-changeset.md` | 33 kB / 668 lines | 06:34 |
| `research/serena.broken.yaml` | 8.3 kB | 06:36 |
| `logs/poll_notes.md` | 11 kB | 06:53 |
| `research/{cfg_fresh,probe_cfg,uvbin,uvcache,uvtool}/` | — | 06:15–06:17 |
| `scratch/` | **empty** | 06:14 |

`serena.broken.yaml` in `research/` is the deliberately-broken **copy** used to
prove `bc1a284`'s test red; the shipped `env_mgr/recipes/serena.yaml` was never
touched (reported by the lead; I confirmed only that the copy exists at that path
and that `git status` shows no modification to the shipped file).

**Teammates.** `researcher-core` was dispatched at 06:52 — one minute before this
checkpoint — to measure whether `installers/claude.py::_present_names` can ever
match, against captured verbatim bytes of real `claude plugin` output, using a
**copy** of the function in scratch. That task is deliberately not gated by
decision (A) because the shipped file is not touched. `researcher-pkg` is idle
with nothing outstanding. Both states are **lead-reported**; what I verified is
that `scratch/` is empty, which is consistent with a task dispatched 60 seconds
ago and proves nothing either way.

**The four-layer ruling's provenance, resolved.** Not a PR artefact. The source is
the **user, in session, 2026-09-04**, in Chinese: the four layers `system` /
`workspace` (*"就是我说的 default 机制"*) / `task_package` (*"就是我说的 main
机制"*) / `agent`, with *"强调下只有第四层允许有 asset 中的 .claude 文件格式的声明"*;
then *"layer 字段是系统机制，对声明和用户不可见"*; then **"我选(甲)"** for
detection-only conflict handling. The PR's closest text is *"make L3 as a new top
layer or a repo layer"*, which is the origin but not the model. **A later reader
searching PR 155 for the four layers will not find them** — that is why the
provenance is written out here.

**Two new owner rulings this window**, both lead-relayed, both verbatim:

- *"不要和154做重复冲突的事情哈，保证两者遵循同样的原则即可，如果有什么要等他合入
  以后要做的事情，记入 TODO"* — do not duplicate or conflict with PR 154; same
  principle is enough; anything waiting on 154 goes to `TODO.md`. Already
  discharged as row **4h**, which I read.
- *"所有声明要安装的项目，除了 ai 相关的，比如需要放进 .claude 里的，是安装到具体
  agent workspace 的，其余都是安装到公共目录大家一起使用的… 把这条补到 env_mgr 的
  spec 里"* — already discharged as `spec.md` §9.1, which I read.

### 3. Code problems — fixed / not fixed

**Fixed this window — one, and it is an instrument, not product code.**

- `agent_sys/tests/env_mgr/test_agent_assets.py:745` (`bc1a284`, +20/-1). **The
  ninth cannot-fail check of this effort.** The assertion was truthy on
  `document["outcomes"]`, which is non-empty on success *and* on `RecipeError`
  alike — so the test was green whether the serena recipe parsed or not. Now
  asserts `document["status"] == "OK"`. Proven red in both directions against a
  broken **copy**. Suite unchanged at 473 pass / 2 / 1. **Lead-reported for the
  red-proof and the suite numbers; I verified only the commit, its diffstat and
  its message.**

**Not fixed — the 16 review comments.** Unchanged from T+2 in substance. One
correction to how T+2 recorded them, below.

**Not fixed — three defects named this window, none touched:**

- `agent_sys/env_mgr/installers/claude.py::_present_names` — suspected never able
  to match. Under measurement as of 06:52; blocked from repair by decision (A).
- `agent_sys/env_mgr/layer.py` — a docstring the lead calls false. Not read by me.
- `env_mgr` **criterion 22's byte-identity pin** covers the whole
  `installers/` directory, so a source-byte assertion blocks the `_present_names`
  repair even though none of the 65 tests the criterion cites would fail.

### 4. Non-code problems

- **My own file is `MM` in a shared index and the staged blob is stale.** Measured:
  index vs HEAD `+889`, working tree vs HEAD `+1160` — so **271 lines of this
  round's T+0/T+2 exist only in the working tree**. Four agents share this index;
  last round a bare `git commit` swept eleven of another agent's staged files into
  someone else's commit. The team rule *"explicit paths, read the staged set
  back"* catches a **wrong path** and does **not** catch a **stale blob at the
  right path** — the lead's poll notes name this distinction, and the repair they
  state is to read the **staged diff**, not the staged file list. I am committing
  this section with `git commit -s -m … -- work.checkpoint.summary.md` and reading
  `git diff HEAD -- <path>` back first. I will not `git reset` and will not
  unstage anything of anyone else's.
- **`CLAUDE.md` is still the previous round's book — reclassified.** At T+2 I
  called this "the single highest-risk state on disk". That was the wrong
  classification and I am correcting it here rather than editing T+2: the user's
  rule for a new task is *research → plan → sub-workspace → write `CLAUDE.md`
  (backing up the old) → work*, so the replacement is **due after the plan, not
  before**. It is a **known open window with a stated closing condition: closed
  when the plan lands.** The hazard text stands unchanged — anyone reading
  `CLAUDE.md` right now gets the L1/L2/L3 task book as the thing being built, and
  that is exactly what this round is dismantling.
- **A corrected premise about PR 154, worth recording because the wrong names are
  searchable.** The shared root is **`AGENT_SYS_HOME`**, default
  **`~/.infera_agent_sys`**, `~/.local`-shaped (`bin/ share/ state/ run/`), read
  by the lead first-hand from `gh pr diff 154`. The owner's recollection was
  `${agent_sys_root}` / `~/.agent_sysxxx` — **both wrong**, and the owner had
  flagged their own uncertainty (*"忘了是不是"*, *"好像"*). Anyone searching the
  old names finds nothing.
- **The poll cadence had never once sent a message.** Three ticks, three
  deliberate refusals, each locally reasonable (everyone idle). Stacked, the
  channel was never exercised while the reports described a running cadence. The
  lead records the stated cause as **their own**: A–H gate the downstream work, so
  nothing was assigned, and then *"nobody is working"* was used as the reason not
  to poll — a state they created offered as the excuse for the state. Fixed this
  tick by sending a poll carrying real work.
- **`poll_notes.md`'s own two defects, found by its author.** Ticks are numbered
  **1, 2, 4** — there is no 3, so the numbering is not evidence of firing count;
  and the hand-written labels (~06:30, ~06:40) never matched the cron's actual
  minutes (`:04 :14 :24 :34 :44 :54`). I verified both against the file.
- **Both cadences are registered.** `CronList`: `faf6468a` poll at
  `4,14,24,34,44,54`; `f247a8d6` checkpoint at `8,38`. Lead-reported.
- Three stale untracked files still sit at the repo root, unchanged from T+2.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- **Eight decisions A–H remain with the owner and gate nearly everything**: (A) the
  byte-identity pin, (B) `sglang.repo.yaml` has no honest level, (C) `system` would
  have no possible member, (D) the level would have no runtime reader, (E)
  `env_recipe.<agent>.yaml` discovery scope, (F) "L2 should not exist" vs keeping
  `components/`, (G) six routes not seven in `env_checker`'s acceptance, (H)
  serena's `.mcp.json` vs the official plugin. This is the **second consecutive
  poll with the same block and zero implementation progress.** I have not read the
  eight as written; the list is the lead's. What I can say from disk is that no
  source file has changed, which is consistent with the block and does not prove
  it.
- The overlay/fallback question from T+2 is **partly closed and I am not closing
  it.** `core.md` §2 is titled *"Where the loader learns the level"* and §5
  *"The nine items, and what level each honestly has"*, so someone has measured
  it — but I read the headings, not the findings. The user's *"If i am not wrong"*
  still has no answer in this record.
- Whether the research *answers* the owner, as opposed to existing. 1,488 lines
  across three files is a volume measurement, not a quality one.
- `import httpx2` from the owner's MCP sketch — still uninvestigated. `pkg.md` has
  a section Q4a on `from mcp.server import MCPServer`; whether it addresses the
  `httpx2` line, I did not read.
- Whether PR 155 has acquired a 17th comment, or any reply. Not checked this
  window.

### 6. New commits

Three, all `2026-09-04`, all signed off, all with explicit pathspecs.

- **`bc1a284`** 06:37 — `test(env_mgr): the serena-recipe test was green whether the
  recipe parsed or not`. `tests/env_mgr/test_agent_assets.py` +20/-1. The
  cannot-fail assertion described in §3.
- **`562be04`** 06:37 — `docs(agent_sys): forbid a second mechanism for a job that
  has an owner`. `engineer_principle.md` +15, `docs/TODO.md` +2. This is the pair
  of uncommitted changes T+2 reported as *"read in full, uncommitted"* — now
  landed unmodified.
- **`33b9f56`** 06:44 — `docs(env_mgr): one rule for where a declared install
  lands`. `env_mgr/docs/spec.md` +34, `docs/TODO.md` +6. Adds §9.1 — two
  destinations, shared root by default, the agent's zone only for a `.claude/`
  tree — with the deciding question written as a blockquote: ***"would two agents
  in the same run need different copies of it?"*** Plus principle 8 in §2, and
  `TODO.md` row **4h** under a **new "Blocked on another change landing"
  subsection**. I read the whole diff.

### 7. Anything else worth recording

- **A new instrument species, named this window: *a pin that is stronger than the
  criterion it cites*.** `env_mgr` criterion 22 says *"its 65 tests keep passing
  untouched"*; the test actually asserts that **source bytes do not change**.
  Repairing `_present_names` would fail none of those 65 tests yet trips the byte
  assertion. This is neither a check that cannot fail nor a wrong check — it is a
  check **enforcing more than the rule it claims to enforce**, so a reader who
  trusts the citation draws a false conclusion about what is permitted. Distinct
  from the nine cannot-fail checks and worth keeping separate from them.
- **Two corrections the lead issued against their own earlier statements**, recorded
  as corrections and not softened. (1) They reported **15** PR comments; there are
  **16**; the 16th (`id 3931202300`, *"sorry, add gh cli"*) is a **correction to
  comment 10's "add cli"**, so **15 requirements + 1 correction, no requirement
  missed** — which also corrects T+2, where I listed all 16 as if each were a
  requirement. (2) They called the `_present_names` fix *"decision-free"*; it is
  not, because criterion 22's pin covers the whole `installers/` directory — they
  had checked the policy and not the pin.
- **`33b9f56` is the shared-install rule written *without* a constant to back it.**
  §9.1 states the rule and names PR 154's `AGENT_SYS_HOME` as its single owner,
  while 4h records that the module is not in this branch and today's installs pin
  four variables one at a time (`UV_TOOL_DIR`, `UV_TOOL_BIN_DIR`, `UV_CACHE_DIR`,
  `SERENA_HOME`). The commit message states the reasoning: a second root *"would
  be exactly the parallel mechanism `engineer_principle.md` §2 forbids"*. **This
  is the principle committed 7 minutes earlier being applied to a decision 7
  minutes later** — the standard was set first and then obeyed under cost, which
  is the only evidence that a principle is load-bearing rather than decorative.
  Recording it because T+2 flagged the write-the-principle-first ordering and said
  the thing to check later was whether the work then obeys it. First instance:
  it did.
- **`lead-analysis.md` marks itself as a hypothesis, in its own second line:**
  *"Written before the researchers reported, so it is a hypothesis to test, not a
  finding. Anything here that a researcher measures differently wins."* Its
  load-bearing half is a *negative*: `Installer` returns `list[Outcome]` and
  nothing else, so `mcp_servers` and `tools` **cannot** become installer outputs —
  the L1 route is a subprocess and a live Python object does not survive it. The
  note records that `Outcome.details` was considered as a channel and rejected.
  A hypothesis that names where it does not reach is a different object from one
  that does not.
- The owner has checked out a new branch and pushed round 1 themselves; round 1
  needs nothing further from us. Lead-reported; not verified by me.

---

## T+66 — 2026-09-04 07:20 UTC

**The block broke.** Five of the eight decisions A–H are settled, one of them by
being withdrawn as the lead's own error. This is the first window of the round in
which the constraint on progress was not "waiting on the owner".

### 1. Progress

**Effort: ~12 %.** Elapsed 66 minutes. Estimated remaining: unknown, but for the
first time this round the *uncertainty* has moved rather than the *number*.

| workstream | est. % | basis |
|---|---|---|
| review comments gathered and read | **100 %** | unchanged |
| research (both researchers) | **100 % of what was asked** | unchanged; 1,488 lines on disk |
| decisions A–H | **5 of 8** | (C)(E)(F) ruled, (G) withdrawn, (H) returned to the lead and made. **(A)(B)(D) open** |
| task book (`CLAUDE.md`) for this round | **100 %** | measured: `diff -q` against the backup now reports **DIFFERENT**; new file 8,646 bytes, mtime 07:04. I read its first 30 lines |
| plan | **100 % exists** | `PLAN.md`, 6.4 kB, 06:29, at the workspace root. I confirmed the file; I did not read it |
| docs / spec / bug records | **5 items** | 4 commits now: `engineer_principle.md`, `TODO.md` 4g+4h, `spec.md` §9.1, and the 134-line bug record |
| probe: `_present_names` | **complete, with an open tail** | `probe_claude_plugin_list/README.md` read in full — verdict reached, three candidate fixes still tied |
| **production code for this round** | **~0 %, measured** | `git status --short` names **three untracked files and nothing else**. No shipped source file has changed. `bc1a284`'s test fix remains the only repo change from a researcher |
| PR 155 comments answered on GitHub | **0 % observable** | not checked this window either |

**Reliability: still low on the number, but the *kind* of doubt has changed and
that is the reportable event.** At T+39 I could not size the work because five
decisions were unmade. Now three of the five turned out to cost little — (F) in
particular resolves to *use a mechanism that already ships* (§2) — and only (A)
gates the shipped tree. **That is a change in confidence, not progress**, and I
am stating it as the lead asked rather than converting it into percentage points.
7 % → 12 % reflects four commits and five decisions, not a line of the change set.

The honest summary of 66 minutes: **the round has finished deciding what to build
and has not started building it.**

### 2. Current state

Branch `dev.yihou.aiopt.task_with_agent_config`, HEAD **`30e958b`** (was `c136d1c`).

`git status --short` — **three untracked entries, nothing modified, nothing
staged**:

```
?? CLAUDE.envchecker.20260904-0614.md.bak
?? CLAUDE.kernel_opt.20260903-0813.md.bak
?? progress.bar.for.user.md
```

The `MM` hazard on my own file is gone and has not returned.

**`CLAUDE.md` has been replaced — the T+2/T+39 open window is closed.** Verified
by me: `diff -q CLAUDE.md CLAUDE.envchecker.20260904-0614.md.bak` now reports the
files **differ**; the new book is 8,646 bytes at 07:04 and opens
*"# Task — act on PR 155's review: remove the parallel concept, keep one
mechanism"*. It states the branch, **16 review comments** with the parenthetical
*"count verified against the API, not remembered — an earlier claim of 15 was
wrong"*, and names the gap as the task: *"the review and the principle now agree
and the code does not."* The closing condition I recorded at T+39 — *closed when
the plan lands* — was met; `PLAN.md` is dated 06:29 and the book 07:04, in that
order.

**Scratch workspace** — one new top-level directory since T+39:

| path | size | mtime |
|---|---|---|
| `PLAN.md` | 6.4 kB | 06:29 |
| `probe_claude_plugin_list/` | — | 06:58 |
| `probe_claude_plugin_list/README.md` | 6.6 kB | 06:59 |
| `research/*` (4 md + broken yaml) | 1,488 lines | unchanged, 06:19–06:36 |
| `logs/poll_notes.md` | 11 kB | 06:53 |
| `scratch/` | **still empty** | 06:14 |

**Teammates — all three stood down or idle, none blocked on each other.**
`researcher-core` was **stood down cleanly at ~07:01** with a handoff:
`bc1a284` its only repo change, `core.md` + `core-changeset.md` apply-ready, the
probe README carrying the verdict. *Nothing speculative was started in the
shipped tree* — which `git status` corroborates, since it is clean. `researcher-pkg`
and I are idle.

**The `tags` measurement — I verified all four sites myself**, because it decides
whether (F) is a change or a declaration:

| site | what I read |
|---|---|
| `agent_sys/env_mgr/recipe.py:18` | `_CLI_KEYS = {"installer", "importance", "layer", "tags", "version"}` — so `tags` is excluded from `Item.spec` and cannot reach an installer |
| `agent_sys/env_mgr/recipe.py:38` | `tags: list[str] = field(default_factory=list)` on `Item` |
| `agent_sys/env_mgr/cli.py:29` | `s.add_argument("--tag", action="append", default=[], dest="tags")` |
| `agent_sys/env_mgr/runner.py:34` | `if filters.tags and not (set(filters.tags) & set(it.tags)): continue` |

**Selection by tag intersection already runs.** The owner's *"作为一个 tag 或者
独立的 key"* resolves to **tag, at zero schema cost**, and `env-mgr install --tag
internal` works today. This is the one thing in this section I did not take on
report — the lead measured it first and I re-read the same four lines.

### 3. Code problems — fixed / not fixed

**Fixed this window: none.** No source file changed.

**Documented this window — one, and it is a defect record, not a repair:**

- `agent_sys/env_mgr/installers/claude.py::_present_names` — `30e958b` adds a
  134-line bug record. Root cause as recorded: **two green tests on a format the
  CLI does not produce**, plus a second wrongness around disabled plugins. The
  three candidate fixes are recorded as an **open tie**, not a resolution. Repair
  still blocked by decision (A).

**Not fixed — carried unchanged:** the 16 review comments; `env_mgr/layer.py`'s
false docstring; criterion 22's byte-identity pin over the whole `installers/`
directory.

### 4. Non-code problems

- **A convention the lead checked rather than assumed, and it is the kind that
  bites silently.** `temp/` is gitignored — I confirmed it at `.gitignore:50`,
  alongside `CLAUDE.md` — **yet all five existing bug records are tracked.** So
  bug records are force-added, and anyone who writes one and trusts `git status`
  will believe it is committed when it is not. `30e958b` did force-add.
- **Five consecutive empty poll ticks were recorded before the block broke.** The
  lead put three options to the owner at 07:01 — decide, pause the cadences, or
  keep recording empties — and **did not act unilaterally**, on the grounds that
  the cadences are a mechanism the owner asked for. Recording the restraint as
  well as the emptiness: the alternative, silently pausing an instrument because
  it keeps reporting nothing, is how an instrument stops being one.
- **The `system` layer's first real member is PR 154's `agentsview`**, which
  deepens a dependency `TODO.md` 4h already carries. Two of this round's items now
  wait on 154 rather than one.
- The `CLAUDE.md` window from T+2/T+39 is **closed** (§2). The two stale `.bak`
  files and the stale progress bar remain untracked at the repo root, unchanged.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- **(A), (B), (D) remain open and are now the only block.** The owner asked for
  all three to be *explained concretely* rather than ruled on, and the lead reports
  having done so. (A) is the byte-identity pin; (B) is `sglang.repo.yaml` having no
  honest level; (D) is the level having no runtime reader. I have not read the
  three explanations.
- **The three candidate `_present_names` fixes tie on all four real inputs**, and
  the probe README says so in its own words: the case that would separate `fix_b`
  from `fix_c` is a metadata line containing an `@`, *"no such line was observed in
  any capture, and I did not construct one"*. **Recorded as open by its author** —
  which is the right disposition and worth noting as such, because constructing
  the input would have produced a tidier verdict resting on a fabricated sample.
- Whether the research artefacts *answer* the owner. Unchanged from T+39: I have
  read structure, not arguments.
- `import httpx2` — still uninvestigated, three windows running.
- Whether PR 155 has a 17th comment or any reply. Still not checked.
- `PLAN.md` exists and I have not read it. Its 6.4 kB is the closest thing to a
  denominator this round has, and I am not quoting a number out of a file I have
  only stat'd.

### 6. New commits

One since T+39.

- **`30e958b`** 07:00 — `docs(bugs): the claude plugin check cannot pass, and its
  tests keep it that way`. One file, **+134**, under
  `examples/.../temp/bugs/`, force-added past `.gitignore:50`. Records the root
  cause (two green tests on a format the CLI never produces), the disabled-plugin
  second wrongness, the measurement provenance, and the three-candidate tie as
  **open**.

`c136d1c` (my T+39 checkpoint) is now one commit behind HEAD and was verified by
the lead: 1425 insertions / 0 deletions, one file, signed off.

### 7. Anything else worth recording

- **Five owner rulings, quoted, because this is the round's turning point.**
  - **(C) RULED** — *"system 层就是个概念层，保留，可以为空，也可以像 pr 154 那样，
    声明一个 agent_sys 的依赖，在 env_mgr 启动时，检查不存在就安装。"* So `system`
    is agent_sys's own dependencies, checked at env_mgr startup, installed if
    absent.
  - **(E) RULED** — *"可以放在 asset 里."* Discovery scoped to `assets/`.
  - **(F) RULED, and it reshapes the round** — *"component 文件夹放的是 agent_sys
    自己定义的一些环境依赖，主要是 ai agent 的 plugins。比如 tools、hook。L2 不该
    存在，而是在 recipe 里声明来源是内部，作为一个 tag 或者独立的 key. components
    名字也换一下，叫这个和语义不太一致。"* No L2 level; internal origin declared as
    a tag; the directory renamed.
  - **(G) DISSOLVED by (F)** — see the correction below.
  - **(H) RETURNED** — *"为什么要问我。"*
- **Three corrections the lead issued against themselves, recorded as corrections
  and attributed to them.**
  1. **(G) was their error and the owner caught it.** They had written that
     removing L2 leaves `env_checker` proving *six* routes not seven. The owner:
     *"啊？根据 F 的解答，不矛盾吧."* Correct — removing the **level** does not
     remove the **route**; the material still exists and is still installed, only
     the declaration moves from a `components:` key to a tag. **Seven capabilities
     remain seven.** (G) is **withdrawn, not resolved** — the distinction matters,
     because a withdrawn item leaves no residue and a resolved one does.
  2. **(H) should never have reached the owner.** *"为什么要问我"* is recorded by
     the lead as a fair rebuke: a design choice with measured evidence on both
     sides and no user-facing consequence is exactly the routine judgement call
     they are meant to make. Now made — keep our own `.mcp.json`, because the
     official plugin has no `--project` and `--project` has no env-var equivalent
     (`grep -c "envvar=" serena/cli.py` → **0**) — and swap `HOME` → `SERENA_HOME`.
     **Escalating it cost the owner a decision slot for nothing.**
  3. **Their (C) finding was true and its sentence was too broad.** They had
     reported that *under system = OS-provided, we do not declare it*, no recipe
     item can be `system`. That holds **under that definition** — and they had
     treated their own definition as the only one available. The owner supplied a
     different one under which the level has members. **The measurement was not
     wrong; the sentence built on it was broader than what had been checked** —
     which is a species already in this round's record, now recurring in a new
     place.
- **The probe's own discipline is worth keeping.** `probe_claude_plugin_list/README.md`,
  which I read in full, labels `populated2.tty.stdout` **`DECOY — do not build on
  it`** and states why: `base.py:31` uses `capture_output=True`, so the CLI never
  sees a tty in production, and the tty form differs in ANSI escapes, line endings
  and a trailing blob. It also states *how* the real captures were taken —
  `claude plugin list > <file>` — with the reason spelled out: *"a reader should
  not have to infer this from the absence of escape codes."* Provenance is given
  as `claude` **2.1.246** at `/home/yihou/.local/bin/claude`, resolved via `PATH`
  because `base.py:31` uses `shell=True`, **explicitly not the SDK bundle** — which
  is last round's probe-provenance lesson applied without being asked. The README
  further claims `~/.claude/plugins` hashed identically before and after every
  probe; I read that claim, I did not re-run the hash.
- **`populated.stdout` is marked NON-DISCRIMINATING by its own author** — one
  entry cannot distinguish a per-entry bullet from a selection cursor — and round 2
  went and got a three-plugin sample to settle it. A sample retired for being
  unable to discriminate, rather than kept because it produced an answer, is the
  same instinct this file has been tracking all round.
- The owner has pushed round 1 onto a new branch themselves; the new `CLAUDE.md`
  states *"nothing further is owed to it."*

---

## T+70 — 2026-09-04 07:24 UTC (out-of-cadence correction)

**Not a checkpoint section, and deliberately not carrying the seven parts** — it
is a single correction that lands on a finding already in this file, and it
should not wait for the `:38` tick while the durable record carries something
that is not true. The next numbered section resumes the cadence.

### Withdrawn: "a pin that is stronger than the criterion it cites"

At **T+39 §7** I recorded a new instrument species under that name, and I recorded
it as *distinct from the nine cannot-fail checks*. **It has zero instances. It is
withdrawn.** The lead named it, I wrote it down, and the writing-down is what made
it durable — so the withdrawal belongs at the same weight.

I verified every citation myself before writing this, rather than accepting the
retraction on report:

| claim | what I read |
|---|---|
| criterion 22, verbatim | `agent_sys/env_mgr/docs/spec.md:561-562` — ***"The shipped recipe and installer machinery is untouched**: `pytest agent_sys/tests/env_mgr` passes unchanged."* |
| the "65 tests … keep passing untouched" phrasing | `agent_sys/env_mgr/docs/design.md:1409` — **a different document**, in §12.1 *"Unchanged, and a test says so"* |
| the test's own docstring | `agent_sys/tests/env_mgr/test_cli_subcommands.py:118`, `test_the_shipped_modules_are_byte_identical` — *"Criterion 22's **first clause**, asserted against the git index rather than against a memory of what was changed."* |

So the byte-identity assertion is a **faithful** implementation of the word
***untouched***, and its author said which clause it implements. **There is no
overreach, so there is no species.** The decision option built on top of it — the
one put to the owner as part of (A) — is withdrawn with it.

**One nuance I add from reading the files, which does not excuse the error.**
`design.md:1409` says *"65 tests pass today and criterion 22 requires them to keep
passing untouched"* — so **`design.md` itself paraphrases criterion 22 in terms of
the 65 tests.** The conflation had a real source in an artefact; it was not
invented. That makes it a *more* dangerous shape, not a lesser one: the misreading
was available to anyone who read `design.md` and stopped there.

### The failure this actually is, recorded under the name this round already has

**Stating something about an artefact nobody had opened.** `spec.md` §10 was one
`sed` away throughout, and the criterion was quoted from memory across at least
four messages before anyone looked. This is the repo-root principle *"Never state
anything about an artefact nobody has opened"* — a **standing obligation, not a
closed count** — recurring at a moment when the round had already logged the
species twice and named it in the task book.

And the "65" was **two facts stale at once**: it is a 2026-08-30 snapshot, and it
is not current. I ran the collection myself — `pytest tests/env_mgr --collect-only -q`
reports **476 tests collected**. A number repeated as current was wrong about the
count and wrong about being a count of now.

### Second correction: there is no `--layer` flag

The lead has said *"`--layer` optional on the CLI"*, and it is written that way in
`PLAN.md` decision B. **No such flag exists.** Verified two ways:

- `grep -rn -- '--layer' agent_sys/env_mgr/` returns **nothing**.
- `agent_sys/env_mgr/recipe.py:58` — `layer = raw["layer"]`, then
  `if layer not in LAYER_ORDER: raise RecipeError(...)`. It is a **required key in
  the recipe YAML**, and a missing or unknown value raises.

So **decision (B) concretely means *make the YAML key optional*, not *make a CLI
flag optional*.** Those are different edits touching different files, and `PLAN.md`
currently describes the wrong one.

### What is not withdrawn, kept separate on purpose

- **The `_present_names` defect stands.** Real, measured, and `30e958b` records it
  correctly. Nothing in this correction touches it.
- **The other species from that same window stands**: *a green test encoding an
  unmeasured third-party format*, which manufactures a **passing** check rather
  than a failing one. `agent_sys/tests/env_mgr/test_installers.py:151` is the
  confirmed instance. It is the inverse of a cannot-fail check and it survives the
  withdrawal of the pin species intact.

A withdrawal is not a licence to sweep the neighbouring findings out with it; two
of the three things named that window were sound.

### One open item closes

**PR 155 has no 17th comment.** Measured by the lead across all three sources —
16 inline review comments, 0 review submissions with a body, 0 issue-level PR
comments. The count of **16** moves from *lead-reported* to *artefact-verified*,
and my §5 item asking whether a 17th had arrived is closed. It has not.

`import httpx2` from the owner's MCP sketch remains genuinely unread by anyone,
including the lead. It stays named and open.

---

## T+102 — 2026-09-04 07:56 UTC

**开工.** The first window in which implementation is running. All eight decisions
are closed, one of them by being revised away after being ruled — and the round's
target changed shape in the process: **there is no layer vocabulary left to fold
L1/L2/L3 into, because the layer model itself is being deleted.**

### 1. Progress

**Effort: ~22 %.** Elapsed 102 minutes. Estimated remaining: unknown, and the
denominator moved again — this time upward in scope and downward in ambiguity.

| workstream | est. % | basis |
|---|---|---|
| decisions A–H | **8 of 8 closed** | (A)(B)(D) ruled this window; (C) then **revised away entirely**; (G) withdrawn, (H) returned earlier |
| spec / docs / fence | **4 commits landed** | `5d334a3`, `74c331f`, `fc200a2`, `fcf325e` — all read by me |
| `_present_names` + its format tests | **landed** | `93bf0ac`, +131/−16 across 2 files. **See §7 — this is change-set work and the lead's brief did not list it** |
| layer removal (`core-impl`) | **in the worktree, uncommitted** | measured: `env_mgr/layer.py` and `tests/env_mgr/test_layer.py` both show `D`; 11 further source/test files show `M`; `Item` no longer carries a `layer` field |
| `components/` → `agent_plugins/`, `env_recipe.<agent>.yaml` (`pkg-impl`) | **0 % observable** | no such path exists; `git status` shows no rename |
| **full suite** | **red, mid-edit** | 10 failed / 465 passed / 2 skipped / 1 xfailed at 07:56. §4 — and by the lead's own ruling this is **not evidence about any one agent's change** |

**Reliability: the estimate is more honest than any previous one and still
unreliable.** For the first time I can point at deleted files and a changed
dataclass rather than at absences. But 22 % is a judgement about a denominator
the lead sizes as ~7 modules, 2 YAMLs and 4+ test files for the layer removal
alone, plus `pkg-impl`'s three items, none of which has left a trace. **Nothing
of the layer removal is committed**, so all of it could still change shape.

I am deliberately not converting "all decisions made and two agents running" into
a large number, as instructed. The move from 12 % to 22 % is: four doc commits,
one real fix commit, and a worktree that visibly contains the layer deletion.

### 2. Current state

Branch `dev.yihou.aiopt.task_with_agent_config`, HEAD **`fcf325e`** (was `88dba47`).
**Five** commits this window, not four.

`git status --short` — 20 tracked paths in flight, the first substantial working
tree of the round:

```
 M agent_sys/env_mgr/{README,__init__,cli,protocols,recipe,runner}...
 M agent_sys/env_mgr/docs/{design,spec}.md
D  agent_sys/env_mgr/layer.py
 M agent_sys/env_mgr/recipes/{serena,sglang.repo}.yaml
 M agent_sys/spec_loader/{assets,package}.py
D  agent_sys/tests/env_mgr/test_layer.py
 M agent_sys/tests/env_mgr/{test_agent_assets,test_cli,test_imports,test_recipe,test_runner}.py
 M agent_sys/tests/spec_loader/test_assets.py
```

plus the three long-standing untracked files at the repo root.

**The layer model is measurably going.** I read `env_mgr/recipe.py` in the working
tree: `Item` is now `installer, importance, tags, version, spec` — **no `layer`
field**. `env_mgr/layer.py` is deleted and staged as such.

**Teammates.** Two implementers dispatched ~07:44, both lead-reported as to their
assignments:

- **`core-impl`** — remove the layer model across `layer.py`, six sites in
  `recipe.py`, `runner.py:75-76`, `cli.py:39`, `__init__.py:3`, `protocols.py:8`,
  two recipe YAMLs, and every test constructing `Item(..., layer=...)`. The `M`/`D`
  set above is consistent with this being underway; I have not read the diffs.
- **`pkg-impl`** — `_present_names` + its two format tests (**landed as `93bf0ac`**),
  then `components/` → `agent_plugins/` with `tags: [internal]`, and
  `env_recipe.<agent>.yaml` under `assets/`. The latter two have left no trace.

**Two file seams the lead sequenced rather than let collide**: `test_installers.py`
(`core-impl` strips the kwarg first as a mechanical commit, then `pkg-impl`
rewrites the two format tests) and `env_mgr/recipes/serena.yaml` (`core-impl`
removes `layer:`, then `pkg-impl` adds `tags:`). Neither agent may hold either
file while the other does.

### 3. Code problems — fixed / not fixed

**Fixed this window — one, and it is the round's first real repair.**

- `agent_sys/env_mgr/installers/claude.py::_present_names` (`93bf0ac`, +44/−0 in
  `claude.py`, +103/−16 in `test_installers.py`). I read the commit body. It took
  `line.split()[0]` of `claude plugin list`, but **every entry is bulleted with
  U+276F**, so it returned `{'Installed','Scope:','Status:','Version:','❯'}` and
  **never a plugin name** — check reported every declared plugin missing on every
  run and install re-ran unconditionally. Now anchored on the `name@marketplace`
  shape. The body states the root cause was **the tests**: they fed `"superpowers
  1.0"`, *a format the CLI does not produce*, and were green on the broken parser.
  Rewritten against bytes captured from `claude` 2.1.246 **on a pipe**; **three of
  the four go red on the shipped function, the fourth is a negative assertion and
  cannot, which its comment says.** The two glyph-free variants still tie and the
  choice is recorded as resting on *"a failure-mode asymmetry, not on evidence
  against the others"*. The disabled-plugin wrongness is **recorded in a test and a
  docstring, not fixed** — an undecided design question.

**Not fixed — an observation about the working tree, recorded as an observation.**

All **10** current failures are in `tests/env_mgr/test_installers.py`, all with the
same message: `TypeError: Item.__init__() got an unexpected keyword argument
'layer'`. What I can state factually: `Item` in the working tree no longer has the
field (read above); `test_installers.py` is **unmodified since `93bf0ac`** (`git
status` names it not at all) and contains **10** occurrences of `layer=` — matching
the 10 failures exactly. **That is the sequenced seam in its mid-state: the
kwarg-strip half has not happened in that file yet.** Whether that is normal
in-flight ordering or a missed handoff I cannot tell from outside, and I am not
guessing — §5.

**Carried unchanged:** `env_mgr/layer.py`'s false docstring is now moot (the file
is deleted). The 16 review comments remain unanswered on GitHub.

### 4. Non-code problems

- **A full-suite run in a two-agent worktree is not evidence about anyone's
  change** — a ruling the lead gave both implementers, with `docs/TODO.md` 4d
  recording the same lesson from 2026-08-29. So the 10 failed / 465 passed I
  measured at 07:56 is **a fact about the worktree at that instant and about
  nothing else**. Each agent reports counts for its own files; the single
  authoritative run is taken after both land. I record the number because it is
  observable, and I record the ruling beside it so the number is not later read as
  a verdict on either agent.
- **`TODO.md` 4i — a product nobody chose.** User-level AI material **outlives the
  run that declared it**, because §9.1 sends package-declared material to the
  agent_sys root and PR 154 puts that root deliberately outside any run root. Both
  decisions are correct alone; **their conjunction was never decided by anyone.**
  Recorded *before* implementation rather than discovered after — which is the
  distinction that makes it a design note instead of an incident.
- **The lead put a stale number in both implementer briefs.** Both say the pytest
  baseline is **476 collected**; it is **475**. A test was retired in `fc200a2`
  *after* the briefs were written and the pre-removal count was quoted. For
  `pkg-impl`, whose job both deletes and adds tests, this would have corrupted its
  own before/after arithmetic. **Second time this round a repeated number was
  stale** — the other was `65` against a live 476. (My own collection at 07:56
  reads **478**, which is a *third* number and reflects the uncommitted work in
  flight, not a baseline.)
- **`fc200a2` left the suite red and both implementers found it before the lead
  did**, because `env_mgr/README.md`'s criterion-22 row still cited the deleted
  test and `test_every_test_the_readme_cites_exists` failed. The lead's own words,
  recorded as they asked and not softened: they ran only the one file they edited,
  saw "7 passed", and stopped — **"while the brief I had just written for both of
  them requires running the whole `tests/env_mgr` before every commit. I required
  of them what I did not do myself."**
- **The repair then failed the same check a second time.** The replacement text
  named the deleted test **in backticks**, and the check scans backticked `test_*`
  names — it **cannot tell citing a test as cover from naming one that was
  removed**. That is the *check that fails on correct input* species, the same
  shape as run 3's placeholder validator. Repaired by **narrowing, not deleting**:
  the name stays for a human reader, unbackticked, with the reason inline so the
  next editor does not re-break it. `fcf325e` is that repair.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- **Whether the 10 `test_installers.py` failures are the seam mid-flight or a
  handoff that did not happen.** The facts are in §3; the distinguishing
  measurement is whether `core-impl` still has the mechanical kwarg-strip commit
  ahead of it. I cannot see an agent's queue and I am not inferring it from a
  timestamp.
- Whether `research/core-changeset.md` retains any value. It is **invalidated
  wholesale on the layer axis** (lead-reported, from `core-impl`): it implemented
  loader-*assigned* layers, which is the opposite of removal. Its 36 fixture-site
  survey may or may not still apply; nobody has said.
- The two glyph-free `_present_names` variants still tie, and the separating case
  — a metadata line containing `@` — was never observed and not constructed. Open
  by its author's choice, now in a shipped commit body.
- `import httpx2` — **fourth window**, still unread by anyone.
- Whether `PLAN.md` still describes the round. It was written 06:29, before (D)
  deleted the layer field and (C) was revised away; its decision B is already known
  wrong on the `--layer` point (T+70). Not read by me.

### 6. New commits

Five. Four are the lead's; **the fifth is not in the brief's list** (§7).

- **`5d334a3`** 07:30 — `docs(env_mgr): design.md stops paraphrasing criterion 22`.
  +11/−2. Removes the root of the T+70 misreading — the artefact that invited it.
- **`74c331f`** 07:45 — `docs(env_mgr): no layers at all; adopt Claude Code's
  user/project split`. `spec.md` +33/−8, `TODO.md` +2. §9.1 rewritten as a
  **derivation rather than a table of levels**; §2 principle 8 replaced. I read the
  new principle 8 in the file: *"Adopt Claude Code's user/project split; invent no
  levels of our own … There is no layer field and no layer vocabulary."* Plus
  `TODO.md` **4i**.
- **`fc200a2`** 07:48 — `test(env_mgr): retire the byte-identity fence; criterion 22
  keeps its second clause`. `spec.md` +21, `test_cli_subcommands.py` **−28**. The
  fence withdrawn at T+70 is now actually gone; old wording and the reason kept.
- **`93bf0ac`** 07:52 — `fix(env_mgr): claude plugin check could not pass; retest on
  captured bytes`. §3.
- **`fcf325e`** 07:53 — `docs(env_mgr): README criterion 22 row follows the spec; my
  fc200a2 left it red`. +1/−1. The narrowing repair. **The commit subject names the
  author's own commit as the cause** — worth noting as a habit, not just an event.

### 7. Anything else worth recording

- **A commit in the log that the brief's own summary omits.** The lead's message
  lists four commits as "my commits this window" and says *"nothing of the change
  set has landed yet — my four commits are spec, docs and the fence."* That is true
  of their four. But **`93bf0ac` sits between `fc200a2` and `fcf325e`**, is
  `pkg-impl`'s, and is **change-set work**: the `_present_names` repair with its two
  format tests rewritten and proven red. So the accurate statement at T+102 is
  *one* change-set commit has landed, not zero. I am recording this as a gap in a
  summary rather than as an error — the lead described their own commits correctly
  and the sentence's scope was wider than what it enumerated. **Which is, once
  again, this round's most-repeated species: a true sentence broader than what was
  checked.** Fourth instance in the record.
- **The round's target changed shape, and this is the entry to reread later.**
  It began as *fold L1/L2/L3 into the existing recipe layer system*. (D) deleted the
  `layer` field from every item; (C) was then revised away — *"哦，那 system 层不该
  存在，只是一个完全的虚拟概念"* — so **there is no layer system left to fold into.**
  The four-level scheme collapsed into Claude Code's own two scopes. The owner's
  destination rule: non-AI → system-wide, preferring the agent_sys root; AI from
  `main.yaml`/`default.yaml` → agent_sys level; AI declared by the agent → the
  agent's workspace root — *"简单来说就是遵循 claude code 自己对于 user level 和
  project level 的管理方式."*
- **The argument that made the field indefensible is worth preserving.** Asked what
  distinguished `workspace` from `system`, the answer was **only who declared it** —
  and under the destination rule both land in the same place, so the field would
  **restate a fact the file path already carries**. That is the *second writer for
  one fact* defect, one level up, and it is the same shape as `design.md`
  paraphrasing criterion 22. The repo's own `CLAUDE.md` principle — *one writer per
  fact* — retired a field this window.
- **`core-impl`'s own finding, and the ruling on it.** `_CLI_KEYS` is an
  **exclusion** set, so simply dropping `"layer"` would make a stale author-written
  `layer:` **silently become `Item.spec["layer"]` and ride into every installer**.
  Ruled: **reject it explicitly with a dated migration message** — *"silent
  pass-through is how this round's earlier defects shipped."* A deletion that
  creates a silent accept is a deletion that leaves a trap.
- **A non-intervention, recorded because it is the entry.** A diagnostic caught
  `pkg-impl` mid-edit with `re` used before `import re` in `claude.py`. It was **not
  raised** — first-sighting rule — and was gone from the next pass, fixed by its
  author inside a minute. The restraint is the record; the transient error is not.
  An observer who reports every mid-edit state trains the team to edit defensively
  rather than quickly.
- **(A)'s ruling in the owner's words**: *"我们这次说的应该是设计级别的改动吧，测试
  当然是要改的，没用的测试去掉，该补的测试补上。"* criterion 22's *untouched* does not
  apply to this round — which is what let `fc200a2` retire the fence at all.

---

## T+129 — 2026-09-04 08:23 UTC

**Implementation is complete and the round is one owner decision from finished.**
This section also folds in the corrections and closures from the 07:59 exchange,
which were deliberately not given a section of their own: T+102 was accurate at
07:56 and was overtaken by events, which is the cadence working rather than the
record misleading anyone. **T+70 broke cadence because it carried a finding that
was never true; being overtaken is a different thing and is left to the next
tick.**

### 1. Progress

**Effort: ~85 %.** Elapsed 129 minutes. Remaining: **one owner decision**, plus
whatever it implies.

| workstream | est. % | basis |
|---|---|---|
| decisions A–H | **8 of 8 closed** | unchanged |
| layer removal | **done** | verified: `env_mgr/layer.py` absent; `LAYER_ORDER` survives **only** as a historical note in `spec.md:454`; no live definition anywhere |
| `_present_names` | **done** | `93bf0ac` |
| `components/` → `agent_plugins/` | **done** | verified: `agent_sys/components/` **does not exist**; `agent_sys/agent_plugins/{README.md, envchk-baseline, serena}` does |
| `env_recipe.<agent>.yaml` under `assets/` | **done** | `b83eb2a` |
| **F's declaration half — `tags: [internal]`** | **0 %, by design** | verified: no recipe carries it. §5 — pending the owner, **not** an omission |
| two packaging defects | **recorded, unfixed** | `adf319d` |
| **full suite** | **green** | **I ran it myself**: `2232 passed, 3 skipped, 4 xfailed` in 67 s. Working tree clean, index empty, three untracked root files |

**Reliability: the highest of the round, and 85 % is still not 100 % for a
reason.** I verified the tree state, the suite, the deleted and created
directories, the migration guard and the packaging comment with my own hands.
What I cannot size is the remaining decision: under one reading `internal` gains
a witness *and* the packaging defect dissolves; under the other, both persist.
**The last 15 % is one question whose two answers imply different amounts of
work**, so a single number for it would be false precision.

Do not read this as "complete". F's declaration half is undelivered by design and
two packaging defects are recorded and unfixed.

### 2. Current state

Branch `dev.yihou.aiopt.task_with_agent_config`, HEAD **`adf319d`**.
**13 commits** in `fc200a2~1..HEAD` — I counted them with `git rev-list --count`.

`git status --short` at 08:24:05Z — **three untracked files and nothing else**:
the two `.bak` task books and the stale `progress.bar.for.user.md`. No modified
file, no staged entry. The 31-file staged set and the six `R`/`RM` rename entries
I recorded at 07:59:32Z are gone into commits.

**What I verified directly, rather than taking on report:**

| claim | what I ran |
|---|---|
| suite | `pytest tests -q` → `2232 passed, 3 skipped, 4 xfailed` |
| layer model gone | no `env_mgr/layer.py`; `LAYER_ORDER` matches only `spec.md:454`, which records it as **deleted** and says neither `LAYER_ORDER` nor `layer_index` was load-bearing |
| the migration guard is real | `recipe.py:73-74` — `if "layer" in raw: raise RecipeError(_LAYER_REMOVED...)`, message dated `2026-09-04` |
| rename residue swept | `grep -rn 'repositorys_components'` → **no matches** |
| `tags: [internal]` absent | the only `internal` in `env_mgr/recipes/` is `serena.yaml`'s **decision comment** |

**A precision the summary should not lose:** "layer vocabulary at zero in
`env_mgr`" is right about the *model* and would be wrong as a literal count. Eleven
lines still contain the word — the migration guard and its rationale, a
`runner.py:73` comment explaining the removal, and four uses of "layer" as an
ordinary English word (*"one layer down"*, *"the SDK's own approval layer"*). The
field, the ordering and the module are gone; the word survives where it means
something else.

**Teammates.** Both implementers are **holding** on the A/B question with the
owner. That is the only blocker.

### 3. Code problems — fixed / not fixed

**Fixed this window** — delivered against the eight rulings, lead-reported except
where §2 says I checked it:

- **A** — fence retired, criterion 22 revised with its old wording kept.
- **B** — resolved by D.
- **C** — layer vocabulary gone (with §2's precision).
- **D** — `layer.py` deleted; a stale `layer:` **explicitly rejected** with a dated
  migration message rather than silently swallowed.
- **E** — one `_ROLES` entry under `assets/`.
- **F** — renamed to `agent_plugins/`, plus **seven stale references, three of which
  were paths that no longer existed**.
- **G** withdrawn (the lead's). **H** decided by the lead — keep our `.mcp.json`,
  `HOME` → `SERENA_HOME`.

**Not fixed — two packaging defects, both recorded in `adf319d`, both pre-existing
and nothing to do with the rename.**

- `env_mgr/recipes/*.yaml` **do not ship in the wheel.** The lead opened it: 149
  members, **zero `.yaml`**, zero `env_mgr/recipes` members, while
  `spec_loader/schemas/*.json` — the previously-fixed instance — does ship. So
  `recipes: [serena]`, which `examples/env_checker` **actually declares**, cannot
  resolve from a wheel.
- `agent_plugins/` has **no packaging route at all**, and its prognosis differs:
  `env_mgr = ["recipes/*.yaml"]` is a one-line candidate, but `package-data` needs
  an owning package and `agent_plugins/` sits **beside** `env_mgr/`, not inside it.

**Not fixed — the tenth cannot-fail check of the effort**, and the first a teammate
found **in their own work while writing it**: a helper reading `result.documents`,
an attribute `LoadReport` does not have, returning `[]` unconditionally.

### 4. Non-code problems

- **The meta-defect is the round's own theme sitting in the repo's build config,
  and I read it myself.** `pyproject.toml:83-88` already carries a comment
  describing this exact failure — *"a bare `agent_sys/schemas/` is not a package, so
  `find_packages` cannot see it… Reading it by relative path works from a git
  checkout and dies from a wheel."* The stanza beneath it lists exactly one line,
  `spec_loader = ["schemas/*.json"]`. **The author saw the class, wrote it down,
  fixed the one instance, and never swept it** — and two more instances now sit
  beside the comment that predicts them. A written diagnosis of a class is the
  strongest possible evidence its author understood it, which is precisely what
  makes the unswept remainder worse rather than better.
- **A rename whose search is narrower than its target leaves residue in the names
  most specific to the thing being renamed.**
  `test_l2_resolves_a_bare_name_under_the_repositorys_components_directory`
  survived a rename searching for `components`, because **`grep -w` finds no word
  boundary inside `repositorys_components_directory`**; eighteen fixture
  directories rode along with it. The surviving names are the ones a reader most
  trusts to be current. (I verified the sweep finished: the string now matches
  nothing.)
- **The green suite carries the same caution the red one did.** At T+102 I recorded
  10 failures and wrote that a full-suite run in a two-agent worktree is not
  evidence about anyone's change. **The symmetry has to hold in the other
  direction**: the lead's 08:2x green run was taken while both agents were still
  editing, and a `recipe.py` diagnostic (`_LAYER_REMOVED` not defined) fired
  seconds after it. My own run at 08:24 is green on a **clean tree with an empty
  index**, which is a stronger claim than theirs was — but the temptation to accept
  green uncritically after refusing red is exactly the asymmetry worth naming.
- **A twelve-minute exposure window, with the accusation withdrawn.** 07:57 →
  08:09:40 was a window in which a growing staged set was reachable by anyone's
  bare `git commit` in a four-agent index. The **risk stands**; the lead's
  inference that it showed a bad habit does not — see §7.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- **The one open question, and it is one question wearing two sets of clothes.**
  `internal` has **no witness in the tree**: `agent_plugins/<name>/` is reached by a
  **directory key on `AgentSpec`**, so there is no recipe item for an origin tag to
  sit on. Under the reading where that key goes away and those components are
  declared through `recipes:`, the witness appears **by construction** *and* the
  packaging defect dissolves, because they would then be recipe files under a
  package that can carry `package-data`. Under the reading where the key stays, the
  tag may have nothing to do and `agent_plugins/` still has no packaging route.
  **With the owner.**
- The two glyph-free `_present_names` variants still tie; the separating case was
  never observed. Unchanged, open by its author's choice.
- `import httpx2` — **fifth window**, still unread by anyone.
- Whether `PLAN.md` still describes the round. Not read by me; known wrong on
  `--layer` since T+70 and superseded twice since.
- Whether the 16 review comments have been answered on GitHub. Not checked since
  T+2; the count 16 is settled but the replies are not observed.

### 6. New commits

Eight since T+102's `ac05ec9`, thirteen in the round's range.

- **`3db1ed5`** — `refactor(env_mgr): remove the layer model`.
- **`b83eb2a`** — `feat(spec_loader): find an agent's env_recipe by convention under assets/`.
- **`0295391`** — `test(env_mgr): drop the removed layer= kwarg from test_installers`. The
  seam's second half; §7.
- **`5454663`** — `fix(env_mgr): reject a stale layer: instead of letting it fall through`.
  The migration guard I read at `recipe.py:73`.
- **`cb5b870`** — `docs(env_mgr): principle 7 named one reason the fence went; there were two`.
- **`ee3ad0a`** — `refactor(agent_sys): components/ becomes agent_plugins/; the levels become origins`.
- **`75edb7c`** — `docs(agent_sys): the agent_plugins rename left seven stale references`.
- **`5477c37`** — `docs(agent_sys): finish retiring the level vocabulary outside env_checker's data`.
- **`adf319d`** — `docs(bugs): two declaration routes resolve to paths a wheel does not ship`.
  The second bug record of the round, after `30e958b`.

### 7. Anything else worth recording

**Folded in from 07:59, closing two items this file left open.**

- **T+102 §5's open question is closed: crossed mail, resolved.** The
  `test_installers.py` seam was not a failed handoff — the lead's ruling and
  `core-impl`'s question asking for it passed each other in flight. The lead
  re-sent rather than diagnosed, and `0295391` landed. **Not an ordering fault by
  either agent.**
- **And the sequencing ruling was overtaken in the other direction — by luck.** The
  rule was *`core-impl` strips first, then `pkg-impl` edits the file*. `pkg-impl`
  had in fact committed its half at **07:52, one minute before the ruling was
  sent**. It worked out. **Recorded as luck, not as a plan that held**, because a
  sequencing rule that is satisfied by accident has not been tested.
- **The fourth-instance entry, with the lead's own addition.** T+102 §7 recorded
  *"a true sentence broader than what was checked"* as the fourth instance this
  round. The lead confirms it and asks that the record note **the previous three
  were also theirs**. The count was also understated in my favour: five change-set
  commits had landed, not one.

**Four corrections of the lead's this window. Two are worse than any earlier one.**

1. **An inference withdrawn.** Seeing 31 files staged at **08:09:08**, they wrote
   *"do not leave a large set staged while you work on the next thing"*. `ee3ad0a`
   committed at **08:09:40** — they had caught a commit **mid-assembly** and called
   it a habit. **Withdrawn.** The transferable lesson is the good part: **a staged
   set has two causes that look identical from outside — assembly and neglect — and
   the written condition could not tell them apart.**
2. **An instruction issued without opening the file.** They told `pkg-impl` to add
   `tags: [internal]` to `env_mgr/recipes/serena.yaml`. Every item in it installs
   third-party code, so serena is the canonical example of the origin the tag
   exists to distinguish **from**. `pkg-impl` refused, reasoned it out, and wrote
   the refusal **as a decision at the top of the file** rather than leaving a
   silence — I read it: *"This recipe carries no `tags: [internal]`, and that is the
   decision, not an omission."* **Fourth instance of the species from them, and the
   first that was an instruction rather than a claim — strictly worse, because a
   claim gets checked and an instruction gets executed.**
3. **Measuring, and then not believing the measurement.** They told `pkg-impl` to
   fix `serena.yaml:4`'s "L1 industry component" comment. `75edb7c` had removed it
   at 08:12:32, three minutes earlier — **and the repo-wide grep they ran
   immediately before writing that message did not list `serena.yaml` either.** They
   had the current answer in their own output and quoted an older `head -8` over
   it. **That is not failing to measure; it is measuring and then not believing
   it**, which is a step further down than the four instances before it. If one
   entry from this window survives, it should be this one.
4. **A list narrower than the thing it enumerated.** They gave four sites in
   `test_agent_assets.py` (there were **12**, including three *test function
   names*) and two in `check.yaml` (there were **9**), and missed
   `tests/spec_loader/test_assets.py:414` entirely. `pkg-impl` re-ran the grep
   instead of working the list, **which is the only reason the sweep finished**.
   Its framing, verbatim and worth keeping: *"the second time it came wearing the
   authority of a measurement someone else had taken."* **A list from the lead
   reads as a spec, and a spec is what people stop looking behind.**

**Other entries.**

- **`pkg-impl` corrected the lead twice, both times from measuring.** Neither
  packaging failure is silent — both raise a named `PrepareRefused`, and the
  `agent_plugins` one is **misleading rather than quiet**; and the two have
  **different prognoses**, with the one-line fix available to only one of them. It
  correctly **did not run** the one-line fix it had identified.
- **A deviation ruled correct against the lead's own instruction.** The lead said
  keep the baseline server's `Capability.level` data and reword the prose;
  `pkg-impl` saw that the `#:` comment, `TOOL_DESCRIPTION` and `LEVEL = "L2"` are
  **one coherent unit**, and that rewording two-thirds yields a file whose comment
  contradicts the constant it documents. Ruled: **coherent-and-dated beats
  half-renamed.**
- **Both packaging defects were found by building the wheel rather than reading
  `pyproject.toml`.** The config *reads* as though it handles this — its comment
  even explains the failure mode. Only the artefact showed that two of three
  instances were unfixed. **Read the artefact, not the config that describes it**
  is the same rule this file opened with, arriving in a new place.

---

## T+191 — 2026-09-04 09:25 UTC

**The owner halted the round mid-way, redirected the design twice, and the scope
grew.** T+129's 85 % is not carried forward and must not be read as regression:
the work behind it still exists and still passes. **The denominator changed.**

### 1. Progress

**Effort: ~55 %. This number is lower than T+129's 85 % because the job got
bigger, not because work was lost.** Everything counted at T+129 is still done and
still committed. What has been added since: a `run_server` installer, a server
registry with a lifecycle owner **that does not exist today**, a port policy, the
in-process `ToolDef` removal across `env_mgr`/`agent`/`claude_sdk`, two further
recipe layers, and an `env_checker` acceptance rewrite.

| workstream | est. % | basis |
|---|---|---|
| everything counted at T+129 | **still done** | HEAD is unchanged at `1c75de1`; nothing was reverted |
| `run_server` installer + registry + tests | **in the worktree, uncommitted** | measured: `env_mgr/servers.py` (462 lines), `env_mgr/installers/run_server.py` (225), `tests/env_mgr/test_run_server.py` (408) — **all untracked**; `registry.py` and `test_imports.py` modified |
| in-process `ToolDef` removal | **0 %** | verified: `ToolDef`/`tooldef` still referenced in **8 files** incl. `agent/backend.py`, `agent/backends/claude_sdk.py`, `env_mgr/{agent_assets,prepare,material,protocols,remote/tools}.py` and `env_checker`'s own `envchk_inproc.tooldef.py` |
| recipe layers 2 and 3 | **in progress** | lead-reported; no artefact I can point at |
| the default recipe | **0 %** | verified: **no `default.yaml` exists anywhere in the tree**; `env_mgr/recipes/` holds exactly `serena.yaml` and `sglang.repo.yaml`, which are **demos, not the default** |
| `env_checker` acceptance rewrite | **0 % observable** | no change on disk |
| **commits this window** | **zero** | `git rev-list --count 1c75de1..HEAD` → **0** |

**Reliability: moderate, and the honest caveat is that 55 % is a judgement, not
arithmetic.** I can measure what exists (1,095 lines of new uncommitted code, and
eight files still carrying the concept that is to be deleted). I cannot measure
the size of a registry lifecycle owner, a port policy or an acceptance rewrite
against a design that was settled less than an hour ago. **A checkpoint whose
number rises while the job gets bigger is worse than no number**, so this one
falls and says why.

**Zero commits landed this window.** All of the window's code is in the working
tree.

### 2. Current state

Branch unchanged, HEAD **still `1c75de1`** — my own T+129 checkpoint commit.

`git status --short` at 09:23:37Z:

```
 M agent_sys/env_mgr/registry.py
 M agent_sys/tests/env_mgr/test_imports.py
?? agent_sys/env_mgr/servers.py
?? agent_sys/env_mgr/installers/run_server.py
?? agent_sys/tests/env_mgr/test_run_server.py
?? (the three long-standing root files)
```

**The design turned twice this window, and both turns are rulings.**

1. **MCP servers can go through an installer.** The lead's objection — recorded in
   this file at T+39 from `lead-analysis.md`, that `Installer` returns
   `list[Outcome]` so an MCP server cannot come back as data — is **withdrawn**. The
   argument that closed it, in the owner's words: *"我专门写一个 claude plugin 的
   install 是想吃屎么"* — `installers/claude.py` already exists and its entire job is
   configuring Claude Code. **A stdio server is launched by the harness itself, so
   nothing has to travel back to the supervisor.**
2. **The in-process `ToolDef` should not exist and is being deleted** — *"根本就不
   应该存在"*. Add-ons ship **standalone servers**: separate process, started by
   `env_mgr` at init. If `agent_sys` must ever serve MCP itself, that is a
   **roadmap** item — a separate thread loaded at init, declaration installed as a
   plugin.

Plus: a `run_server` installer, a registry, shutdown at `agent_sys` exit, and a
**port policy** — same binary → `warn`, otherwise → `error`.

**And `recipe` now has three layers** (owner's ruling 4): the agent's own
(`env_recipe.<agent>.yaml`, already built as `b83eb2a`), the task package's
(`main`), and **default, living in `env_mgr/`**. At agent level **both systems
exist** — its own recipe *and* the `assets/.claude/` copy route — both may be
absent, under the settled absence logic (declared-and-absent is an error;
undeclared-and-absent is simply absent).

**Teammates**: both working, neither blocked. `core-impl` on `servers.py` +
`run_server.py` + tests, registry at `<layout.run>/servers.json` through one
exported variable. `pkg-impl` off hold, building recipe layers 2 and 3, and
**told to decide three implementation questions itself** rather than route them to
the lead — after two escalations this round that were the lead's to settle.

### 3. Code problems — fixed / not fixed

**Found before it was built on — the find of the window, and I confirmed both
halves.**

`subprocess.run(capture_output=True)` **waits for pipe EOF, not for the child to
exit.** I read `env_mgr/agent_assets.py:1498` — `subprocess.run(list(argv),
capture_output=True, text=True, env=..., timeout=timeout)` — and
`agent_assets.py:364`, `RECIPE_TIMEOUT_SECONDS = 20 * 60`. A detached server
inheriting those pipes holds `_run_recipe` open **for its whole life**, so a
**perfectly started server reports `fail` twenty minutes later**. Measured by
`core-impl` at 25 s versus 0 s. **Found by measuring the call it was about to
build on, before writing the code** — the opposite of this round's usual
discovery order.

**Recorded, deliberately not changed:** the recipe child **runs unconfined**. I
verified it: `preexec_fn` appears in `env_mgr/prepare.py:319` (the confined agent
path) and **nowhere on the recipe path**. Installs are a documented hole in §4's
confinement, necessarily so.

**Not fixed, carried:** the two packaging defects (`adf319d`); the tenth
cannot-fail check; the two tied `_present_names` variants.

### 4. Non-code problems

- **A denied deletion, denied twice, and a lead who did not launder it.** The lead
  **withdrew their own authorisation to delete `agent_sys/build/`**: they reasoned
  from the rule's wording to a conclusion about what the machine would allow, and
  `guard_rm` denied it, then the permission system denied it. **They checked the
  rule and not the enforcement.** They also **did not execute it themselves** after
  `pkg-impl` was denied — *"a lead running a peer's denied operation is the same
  operation with the authorisation laundered."* `pkg-impl` had already refused to
  retry it, including against a direct instruction: **"a teammate's authorisation
  is not the operator's."** Affirmed by the lead. This is the global rule holding
  under pressure from inside the team rather than from outside it.
- **A sample blind to the case that matters — the lead's, one hour after they
  diagnosed the identical shape in someone else's work.** Their
  `/proc/<pid>/cmdline` sample had all four lines `yihou`-owned, so every one had a
  pid. `core-impl` had measured a **foreign-uid** port: empty Process column in
  `ss -ltnp`, `lsof` silent, `/proc/<pid>/fd` denied. **A sample that cannot
  contain the hard case makes the easy answer look total** — which is the exact
  sentence the lead had used on `pkg-impl`'s one-plugin capture an hour earlier.
- **An over-engineered brief.** `core-impl` was told to *"measure what is available
  and reliable on this host"* for something one command away. **Presenting a solved
  thing as a decision costs a teammate a detour and lends the detour the lead's
  authority.**
- **A second escalation of a question that was the lead's.** *"取代还是互补"*; the
  owner: *"你自己拿着 registry 自己不知道怎么搞？"* The registry answers *start once,
  warn on duplicate*; which file the declaration lives in is irrelevant. **The tell
  is the same as with (H): the question was about shape and had no measurement in
  it.** Second time this round.
- **PR 154's shared root is live on this host, not merely specified.** I verified
  it: `/home/yihou/.infera_agent_sys/bin/agentsview` exists (133 MB, Sep 1) and pid
  **2353356** is running `agentsview serve --no-browser --host 127.0.0.1 --port
  18888`. `TODO.md` 4h's dependency is on a thing that is already running.

### 5. Open questions, not yet characterised

**Suspend, don't conclude.**

- The registry's lifecycle owner does not exist yet, and **a recipe runs in a
  short-lived child**, so the registry cannot live in memory. `core-impl` measured
  that `PR_SET_PDEATHSIG` **does** close the SIGKILL case but is the wrong tool at
  that site; the guarantee is stated as *"stopped on normal and handled-error
  exit"*. What happens on an unhandled exit is **named, not solved**.
- Whether `internal` gains a witness — T+129's open question — is now entangled
  with the new three-layer recipe design. Nobody has said whether the new design
  answers it, moots it, or leaves it.
- Whether `env_checker`'s `envchk_inproc.tooldef.py` is rewritten as a standalone
  server or deleted. The owner's ruling deletes the *mechanism*; the package
  currently proves a capability through it.
- `import httpx2` — **sixth window**, still unread by anyone.
- Whether the 16 review comments have been answered on GitHub. Unchecked since T+2.

### 6. New commits

**None.** `git rev-list --count 1c75de1..HEAD` → **0**. HEAD is still my own T+129
checkpoint commit, and the window's 1,095 lines of new code are untracked in the
working tree. This is the second window of the round to end with the change set
uncommitted; the first was T+102, and that one resolved within the hour.

### 7. Anything else worth recording

- **Seven corrections from the lead this window. Three are severe, and the pattern
  across them is one thing.** Corrections 1, 3 and 7 are all *reasoning from a
  model instead of from the artefact*: (1) "MCP cannot go through an installer",
  reasoned from a **return type**, never asking *why the data must come back at
  all*; (3) "`addons/` empties out", which rested on the **old** model where
  `.mcp.json` was the only content; (7) telling the owner `main.yaml` is a
  task-graph root and therefore not a recipe — **true of the file and beside the
  point**, because the owner was describing the *layer structure*, and the
  confusion came from reading `env_mgr/recipes/`'s contents as shipped recipes when
  they are demos. Correction 2 is the same shape aimed at a **permission system**
  rather than a design.
- **Correction 3 is the one with teeth, and the owner closed it in a single
  question:** *"你不做独立进程，你那个 envchecker 的 checksum 验证机制怎么跑通？"*
  Under the new design **the server is the content** — `envchk-baseline`'s carries
  `SALT` and `token(nonce)`, and **that token is only producible by running it**.
  The lead's proposed scope split would have left `env_checker` **unable to prove
  itself**. A design change that silently disarms the thing that proves the design
  is the failure mode this whole package exists to catch, and it was caught by a
  question about the artefact, not by an argument.
- **`core-impl`'s port-check inversion, which is the best reframing of the
  window.** The lead and `pkg-impl` were both stuck on *"what is this process?"* —
  **unanswerable**, since `comm`/`exe` read `python3` for serena and for any
  stranger alike. `core-impl` asked instead **"is it the thing I was about to
  start?"** and keyed on the declared program token from the item's **own
  `command`**. The holder's `argv[0]` never enters the comparison. **Two people
  failed to answer a question that had no answer; the fix was to ask a different
  one.**
- **`pkg-impl`'s serena result is stronger than the obvious version of it.**
  Headless is a no-op with one INFO line, and **serena detects headless
  explicitly** — which is a claim about the program's logic rather than *"it did
  not hang once"*, i.e. about the mechanism instead of the sample. It also
  demonstrated the free-port search **live at four concurrent instances** and
  measured `SERENA_HOME` isolation on a running process.
- **A fifth instance from the lead of *a claim about a model stated as a count*,
  logged as mild and deliberately not inflated**: *"layer vocabulary at zero in
  `env_mgr`"*. The field, the ordering and the module are gone; the word survives
  where it means something else. **The count difference was a scope difference and
  not a disagreement** — I pinned it: my `\blayer\b` over `env_mgr/*.py` gives 11,
  their substring over `env_mgr/*.py` plus `installers/*.py` gives 12, `installers/`
  contributes **zero**, and the single extra line is `prepare.py:315`
  `# nothing: layers intersect.` — the plural, which a word-boundary pattern
  excludes. Both figures correct for their own pattern.
- **The defect class named at T+129 gets its own entry, because it is the one most
  likely to bite next.** *Skepticism that only fires on bad news is not skepticism;
  it is a preference dressed as rigour.* The evidence is already in this file:
  every counts-are-not-evidence caution this round — `TODO.md` 4d, the lead's
  ruling to both implementers, my own T+102 note — **was invoked while the suite
  was red**, and not one was invoked at 2232 green on the same tree with the same
  two agents editing. **The discipline had never been tested in the direction where
  it costs something.**

## T+246 — 2026-09-04 10:20 UTC — written by the lead, because `scribe` was stopped

**Not a `scribe` section.** The owner stopped all five agents at ~09:50. A
`SendMessage` would resurrect one, so the cadence was recorded and not executed,
and this section is written by hand to keep the window's findings out of a
context window. Format follows the seven parts loosely; provenance is marked.

### 1. Progress

**Implementation is stopped, not finished, and the tree is broken.** HEAD
`502a3cda`; **13 tracked files modified, 24 tests failing** in `tests/env_mgr`.
`pkg-impl` was stopped mid-commit: it had **deleted**
`env_mgr/addons/{serena,envchk-baseline}/.claude/.mcp.json` and had not yet moved
them into the agent's own assets. Two routes back — restore the 13 to `ea5253db`
(green) or finish the half — **put to the owner four times, neither taken.**
Reason for inaction stated rather than implied: restoring discards a teammate's
work, and finishing commits the owner's ruling by the lead's own hand while the
owner is still asking questions about it.

Effort is not comparable to T+191's 55 %: the denominator moved again and then
the team was stopped.

### 2. Commits this window

| | |
|---|---|
| `f754a6e4` | the two pid guards proven with a sacrificial child, plus a **positive control**; the `cli/main.py` entry covered and red-checked |
| `0ea5b41` | duplicate server declaration warns **from the registry alone** — the owner's *"全局启动一次，重复启动就warning"*, with no scope mechanism |
| `ea5253db` | `agent_plugins/` → **`env_mgr/addons/`**, `package-data` added and **proven by building a wheel: 6 of 6 files ship**, where the bug record's build had 0 |
| `502a3cda` | `spec.md` §9.2 — two justification rules and the named exception; `docs/TODO.md` 4l |

### 3. Owner rulings

- **The remote tools become a named exception**, not a deletion. `spec.md` §9.2
  carries it **with its closing condition**: reprovide the three as a standalone
  server started by `run_server`, after which the section has no exception left.
- **Two justification rules**, both aimed at one temptation — *reaching for Python
  because it is nearer than a recipe*. Adding an MCP server or tool from code
  needs a justification that **no declarative route works**; running one inside
  the `agent_sys` process needs one that **no separate process works**.
- **A correction of their own**: *"这是我对代码库了解不够，remote机制是work的，我
  手动测试过。"*

### 4. Corrections issued by the lead

1. **A recommendation withdrawn before it caused a decision.** The lead had told
   the owner to delete the component-supplied in-process route and keep
   `remote/tools.py`. `core-impl` read the code: **`ToolDef` is defined at
   `remote/tools.py:31` and nowhere else**, and the in-process route is the
   **only** delivery path for the remote surface. So the option bought nothing it
   claimed — the mechanism would stand while a commit message said it was gone.
   **First correction of the day to arrive before the thing it was about**, and
   recorded as *luck plus someone else's measurement*, not as improvement.
2. **An argument weakened by the owner's disclosure.** *"No remote task in the
   tree"* was offered as evidence the capability was unused. It is a true fact
   about the repository and **misleading as evidence**: the capability has a live,
   manually-verified user, who is the owner.

### 5. The finding that reopens a load-bearing claim

`env_mgr/agent_assets.py:287` — the comment the *read-do-not-place* design rests
on — says placing `.mcp.json` *"would put a file in the zone that **nothing
reads**"*. The installed SDK's `strict_mcp_config` docstring says the CLI would
otherwise load *"project `.mcp.json`"*. **They disagree.**

**Not resolved.** The two "project"s may not be the same location, and a trust
prompt may stand between. **One experiment, not one more inference** — this exact
question has already been answered wrongly three times today by reasoning.

### 6. Architecture answered from code, so it is not re-derived

- **How an agent knows it is remote**: not from prose. `prepare.py:645` returns
  three tools if the zone has a far side and `()` if not — **the toolbox is the
  answer** — plus `AGENT_SYS_*_REMOTE` mirrors.
- **In-process, not a separate process**: `claude_sdk.py:95-98`
  `create_sdk_mcp_server`.
- **Injected, not installed**: `claude_sdk.py:393` puts a live object into
  `ClaudeAgentOptions["mcp_servers"]`. **No file is written anywhere** — which is
  exactly why an installer cannot carry it and a `.mcp.json` can.
- **`.claude/` vs SDK overlap is real, known to the SDK, and resolved by switches
  rather than precedence**: additive by default; `setting_sources=[]` and
  `strict_mcp_config=True` are the exclusivity knobs; **same-name collisions have
  no SDK-level arbitration**, which is why whoever merges owns the collision.

### 7. Open with the owner

The broken tree; `serena.yaml` classed a demo while `examples/env_checker`
depends on it; whether to measure the `:287` claim.

## T+281 — 2026-09-04 10:55 UTC — written by the lead; `scribe` is still stopped

Second hand-written section, same reason as T+246: `scribe` was stopped with the
other four and a `SendMessage` would resurrect it.

### 1. Progress

**The round turned from stalled to converging.** At T+246 the tree was broken at
24 failures with a half-finished refactor and no team. It is now **green at 1086
passed, 2 skipped, 3 xfailed** across `tests/cli spec_loader env_mgr agent`, with
two agents working and nothing blocked on the owner.

Landed since: `88a4b5dc`, `6e72e27a`, `3dca5f04`, `29a39ae6`. In flight: the
in-process-tool narrowing (`core2`) and `env_checker`'s acceptance rewrite
(`pkg2`).

### 2. The owner's ruling that changed how this is run

> *"麻烦你给本次任务单独开个spec.md，不要啥事都来问我… 遇到问题自己推导。不要一会儿
> 整一个耦合的胶水怪给我，一会又说这也不敢定，那也不敢定。"*

**Both halves are one failure.** Deciding too much when I should have asked, then
asking everything when I should have decided. The corrective is not *ask less*: it
is `agent_sys/docs/spec.provisioning.md` (`88a4b5dc`, 208 lines), so the answer
has a place to live and neither side carries it in a conversation. **Both new
briefs say: bring me only what contradicts the spec or a ruling.**

Also ruled: `serena.yaml` needs no decision — *"你让agent recipe refer它…不就行了"*.
`pkg2` then established the mechanism **already exists**: `check.yaml` carries
`recipes: [serena]` and `_recipe_paths` resolves a bare name to it. Nothing built.

### 3. Corrections issued by the lead — three, and the third is a new shape

1. **The `.mcp.json` "contradiction" was not one.** I had recorded `agent_assets.py:287` (*"nothing reads it"*) as contradicted by the SDK's *"the CLI would otherwise load project `.mcp.json`"*. **Derived, no experiment needed:** `.mcp.json` is a **project-scope** filename; a zone's `$CLAUDE_CONFIG_DIR` is **user** scope. Different locations, both true. **Two documents can disagree in wording and agree in fact** — the test is to ask what location each names, which is a smaller question than the experiment I was about to run.
2. **A list of mine was stale, not narrow.** I told `core2` to edit `protocols.pyi:120` and `agent/runner.py:763`. Neither needs it: `Prepared.tools` **stays**, as the carrier for the exception. **I wrote that list against the superseded *delete everything* plan and handed it over as current.**
3. **I stated a decision as if it were the artefact.** My brief said serena's entry *"keeps `SERENA_HOME`"*. **That key has never existed in that file** — an earlier round *ruled* the swap and it never happened. **Worse in a brief than in a claim, because a brief gets executed rather than checked.**

### 4. Teammate findings, two of which generalise past this round

- **`core2`, and it is better than the framing I asked for:** *"any file left modified by a stopped agent needs a read before anyone commits the path, for its **content** and not only its diff."* It caught `ROADMAP.md` asserting a still-running mechanism had been deleted — **true when written, overtaken while nobody owned the file, and nothing on disk marked it stale.** **The habit that caught it was the index-safety habit, not a documentation habit.**
- **`core2` again:** it backed out a second file so *"a red **I** caused"* would not sit in a file the lead would read as someone else's. **Reasoning about how the observer will misattribute** — the mirror of the fifteen-minute misattribution earlier today. Corollary it drew: a doc change true only after a code change **is part of that code change**.
- **`pkg2`:** `recipes/serena.yaml` carried a **copy of the real `.mcp.json` in a comment, already drifted** (`AGENT_SYS_MY_PLAYGROUND` vs `${TMPDIR}`), and `check.yaml` had picked up the same wrong name. **Removed, not corrected** — one fact, one writer. **A drifted copy propagates by being copied.**
- **`pkg2`:** removed a cross-tree MCP-collision warn that with one tree has an **empty intersection by construction** — the eleventh check-that-cannot-fail this round, and the second found by its own author while editing around it.
- **`pkg2`:** wheel counted, not assumed — 3 addon members on disk, 3 in the wheel, and the stale *"4 of 6"* corrected.

### 5. Measurements I made myself

`1086 passed` verified twice; `prepare.py`/`agent_assets.py` clean before release;
the ROADMAP orphan's authorship and its stale clause; `.mcp.json`'s scope from the
CLI's `--mcp-config` and the SDK's own wording.

### 6. Non-code

The cadences fired into a stopped team for four ticks and were recorded, not
executed — **a SendMessage would have overridden a human decision with a timer.**

### 7. Open

`env_mgr/recipes/*.yaml` still do not ship in a wheel (`temp/bugs/2026-09-04-*`,
one-line candidate unrun); cross-layer version conflicts; the registry sweep;
installs run unconfined. All four are named in `spec.provisioning.md` §8 as
deliberately unsettled, which is the point of having written it.

## T+311 — 2026-09-04 11:25 UTC — written by the lead; `scribe` still stopped

Third hand-written section, same reason. Format loose; provenance marked.

### 1. Progress — the round's implementation is complete

**Full suite 2258 passed, 3 skipped, 4 xfailed, 0 failed. Working tree clean.**
Every one of the owner's rulings is in: the layer model gone, `addons/` inside
`env_mgr` and shipping in a wheel, three recipe layers, `run_server` + registry +
stop, the in-process route narrowed to its one named exception, `env_checker`
rebuilt on six capabilities and two install routes.

Landed since T+281: `8be9204e`, `67cbdcd8`, `fc05b80d`, `60435123`, `1d6a37ff`,
`f2c4321a`.

### 2. Corrections issued by the lead — three, and the third cost a teammate's time

1. **I told `core2` three modified files were "three of yours". Only one was.**
   The other two were `core-impl`'s orphans. **Second time today I put its name on
   a diff it did not write** — the same shape as the morning's misattribution,
   opposite direction: *attribute the unowned thing to whoever is standing
   nearest.* I committed both myself (`60435123`), read for content first,
   authorship named.
2. **I flagged `agent.schema.json` as an uncommitted orphan. It was committed** —
   I read the tree in the window before `1d6a37ff` landed. A stale read presented
   as a state.
3. **I dispatched `core2` to build a guard it had already built.** `pkg2` reported
   *"nothing fails if a future in-process route arrives"*; **I acted on the report
   without checking the tree.** The test had been in HEAD since `8be9204e`. *A
   claim in a message is not a fact about the repository* — I had said exactly
   that to both agents twice each, then did it myself, **with the cost landing on
   someone else's time rather than my own.**

### 3. What the team did that is worth keeping

- **`core2`'s guard is stricter than the brief I would have written.** It had
  already written the objection I later declared mandatory: *"a negative assertion
  about an absent mechanism passes for free, and an empty result is exactly what a
  fixture that placed nothing would also produce."* Both halves built — subject
  proven present, subject proven fatal if touched — and the fixture deliberately
  keeps the deleted route's own suffix, *"because a suffix nobody ever recognised
  could not go red."*
- **`pkg2` verified `SERENA_HOME` twice**, reading the source *and then running
  it*, because *"a source read alone would have been the same species as the
  brief."* The run found what a grep could not: `language_servers/static` lands
  under `SERENA_HOME`, `~/.solidlsp` never appears, `~/.serena` byte-identical
  across 210 entries.
- **`pkg2` grepped for its own overstatement before editing**, found it had never
  reached a file, and **recorded the correct fact instead of deleting a wrong
  one** — the real defect being that the files said nothing about the trade at all.
- **Both volunteered which half was measured and which inferred**, unprompted, and
  `pkg2` declined to drive `core2`'s test red because it was not its file. **A
  claim labelled *inferred* is worth more than the same claim labelled *proved* by
  someone who did not run it.**

### 4. The species this round should carry out

**`pkg2`, on a wrong belief that never reached a file:**

> *"a claim that is wrong and unwritten still costs something. Mine was heading
> for a file — I would have written the trade into `serena.yaml` next time
> somebody asked why `HOME` was dropped, and by then the reasoning would have
> looked settled."*

**A wrong belief held privately is a file that has not been written yet.** It gets
written at the moment someone asks, when it reads as considered rather than
improvised. It was caught because it put the weak half in the report — **the
mechanism ran both directions today**, catching the lead's `SERENA_HOME` claim the
same way.

### 5. Non-code

Three crossed-mail misreadings today; the third was the costly direction —
**a teammate concluded its own finding had been weighed and rejected when it had
been adopted.** A dropped question gets re-asked; a finding believed rejected
stops being pursued.

### 6. Open with the owner

`env_mgr/recipes/*.yaml` do not ship in a wheel, so `recipes: [serena]` — which
`examples/env_checker` itself declares — cannot resolve from a wheel install. The
one-line `package-data` candidate is **unrun**, deliberately: it is `env_mgr`
packaging and it breaks a live path, so it wants its own change and review.
Recorded as known gap 6 in `env_checker/README.md`, on the `recipes:` line, and in
`spec.provisioning.md` §8.
