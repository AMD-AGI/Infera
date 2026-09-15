# `llm_e2e_performance_optimization` — deferred work

Every item here was **named and deferred by the mission**, or carried over from
an earlier effort with a measured symptom. Nothing is here because it was
forgotten.

Format: what · why it is not done · what would settle it.

## Index

| # | Item | Status | Lines |
|---|---|---|---|
| T1 | `check_trace_coverage` against the sglang source and the model structure | — | 12 |
| T2 | the `vendor_tuned` bucket | — | 12 |
| T3 | one handoff per operator | — | 12 |
| T4 | the analysis programs are hand-written and may be narrow | — | 10 |
| T5 | the patch mechanism should hack the registry, not bind-mount files | — | 18 |
| T6 | permission and visibility management for the shared container | — | 14 |
| T7 | the comparability gate at bring-up (was E9′) | — | 18 |
| T8 | `seal_refused` has no reader (was C9b) | — | 7 |
| T9 | `env_mgr.fs.layout` has two `copy_out` functions (was E14/C-two-copy_out) | — | 6 |
| T10 | `check_workset_runs` hard-fails on rsd while `min_pass_ratio` forgives correctness (was C24 and  | — | 6 |
| T11 | ten closed `items_schema`s in the integration stage (was C23) | CLOSED | 6 |
| T12 | the mock cannot exercise M5.4's ad-hoc correctness rules | — | 14 |
| T13 | `compare.py` finds the kernel's profile share by substring | — | 18 |
| T14 | a task body cannot name the interpreter the run is using | — | 27 |
| T15 | `E2E_KIT_ENGINE_EXTRA_ENV` is a seam with no consumer | — | 19 |
| T16 | `git`'s `index.lock` retry is not idempotent, and its no-op is silent | — | 19 |
| T17 | the model-specific engine flag groups are free-form strings and cannot be checked | — | 34 |
| T18 | `shared_identifiers` sees flags, and a shared host is bound by more than flags | — | 41 |
| T19 | a GPU *set* is a bound identifier with no variable, and prose is not a validator | — | 49 |
| T20 | a borrowed resource's cleanup must be idempotent and unowned | — | 49 |
| T21 | the completion probe grades shape, not answer, and the bar that would fix it needs a measurement | — | 59 |
| T22 | `E2E_STAGE` names a per-stage fact and `--var` carries one value per run | — | 45 |
| T23 | `fixed.gpu_count` is the one required field with no definition | — | 61 |
| T24 | a fallback nobody can reach is still a second reader of the number | CLOSED | 36 |
| T25 | a run records the environment it minted, but not the vars it was started with | — | 43 |
| T26 | killing a run does not kill its agents | — | 52 |
| T27 | a default that takes everything free is not a default | — | 137 |
| T28 | T21's bar, measured: 7 characters against 526 | — | 40 |
| T29 | a crashed instrument and a refused artefact are the same value in `verdict.json` | — | 30 |
| T30 | a finding recorded against one stage is a question for every stage | — | 18 |
| T31 | naming a class is not sweeping for it | CLOSED | 36 |
| T32 | the evidence records which node, and not which card | — | 43 |
| T33 | a mechanical reformat makes a diff unreviewable, so the semantic check moves before the commit | — | 61 |
| T34 | the environment record can outlive the container it names, and a downstream field repeats it as  | — | 105 |
| T35 | a sealed handoff from the old layout is not consumable by the definition that replaced it | — | 25 |
| T36 | a claim about *who owns this* never looks like a claim, so nobody tests it | — | 42 |
| T37 | two producers disagreeing is what a schema-shaped defect looks like | — | 35 |
| T38 | nothing in the graph orders m2's two lines; only the GPU count does | — | 55 |
| T39 | `read_events.py` prints `message` and hides the attribute that holds the cause | — | 31 |
| T40 | treat every probe as a control, because the slot decides whether you check the null | — | 86 |
| T41 | `/proc` is namespaced under `spur exec`, so PID-based attribution from a node lies | CLOSED | 71 |
| T42 | a validation zone is not a normal process environment, and anything read from it that names a ho | — | 32 |
| T43 | the artefact is honest about provenance and dishonest about meaning | — | 121 |
| T44 | `rebuild` is reachable in one schema and emittable by none | — | 63 |
| T45 | a comment that asserts a wiring gap sends the next reader to change code, when only a number was | — | 44 |
| T46 | a fix verified correct is not a fix verified reached | — | 48 |
| T47 | the pathspec window is irreducible, so the check has to be after the commit | — | 45 |
| T48 | the control that caught it is invisible because it worked, so remedy-selection reaches for the p | — | 52 |
| T49 | the verdict is right and the message says why it is right, wrongly | — | 92 |
| T50 | an instruction verified correct against a world that does not exist yet | — | 41 |
| T51 | at promotion only the brief travels, and the brief drifts in both directions | — | 50 |
| T52 | no `df` reachable from `spur exec` predicts whether a `docker load` fits | — | 44 |
| T53 | the var table's rung-2 advice for `expect_ranks` is right only at `tp=8` | — | 23 |
| T54 | declaring `runtime.replayed_from` would not close the hole; `additionalProperties` is the hole | — | 76 |
| T55 | the CUDA graph ceiling belongs in the environment record and in `check_environment`'s compared s | — | 59 |
| T56 | `summarise.py` exists twice, byte-identical, with call sites split across both copies | — | 47 |
| T57 | one scratch path, three independent literals, agreeing by coincidence | — | 64 |
| T58 | counting kits on disk counts how often we tested, not how often a producer chose | — | 55 |
| T59 | the instruments that failed, and not one failed toward "I cannot tell" | — | 145 |
| T60 | the graph ceiling is chosen by stage 1's load and spent by stage 2's | — | 58 |
| T60a | addendum — the override channel already exists and is one line from working | — | 123 |
| T60b | closed by rung 2f — and 25.42 was a symptom, not a requirement | CLOSED | 47 |
| T61 | a defect the instrument hides by functioning correctly | — | 67 |
| T62 | `base_sha256` defects 2 and 3, held pending an answer that may not exist | CLOSED | 41 |
| T63 | `snr_db: inf` is non-monotonic on `layernorm`, and both proposed mechanisms are disproved | — | 14 |
| T64 | `rank.task` and `identify.task` are 15-line skeletons, not briefs | — | 26 |
| T65 | four profiling lines died in one morning, cause open | — | 22 |
| T66 | the agent's system prompt points at a path that is not in the repository | — | 34 |
| T67 | `reverify_shapes` counts operators, and its name says shapes | — | 48 |
| T68 | an attested number and a claimed one are indistinguishable in the handoff | — | 46 |
| T69 | `identify` leaves `fellow` empty for most operators, and two producers write the tag | CLOSED | 72 |
| T70 | `kernel_taxonomy.yaml` offers a fellow KernelForge does not have | — | 30 |
| T71 | three separate defects share one cause: the mock corpus has a shape the real path does not | — | 46 |
| T72 | `expect_ranks` 在两份文档之间有缝,两份都没错 | — | 54 |
| T73 | the Triton fellow pattern needs a leading underscore and a terminal `_kernel`; real kernels have | — | 167 |
| T74 | a killed validator leaves its own deployment holding the cards | — | 33 |
| T75 | constrain the `optimized_kernel.py` slot by KIND, not by filename — deferred, and the deferral i | — | 41 |
| T76 | emitting `environment.yaml` is a producer's duty with no central enforcement, and three producer | — | 55 |
| T77 | `redact` refuses by file suffix, and an evidence record is not a script — deferred, and it is a  | — | 64 |
| T79 | a run does not record the `--var` values it was launched with, and at least five of today's inci | — | 107 |

**80 entries.** `CLOSED` and `HELD` are set only where the entry says so
explicitly; **`—` means the status was never recorded**, which is true of
73 of them and is a gap in this file rather than a claim that they are open.
Setting those is the owner's call, entry by entry.

Numbering is stable and never reused: entries are cited by number from the
assets and from each other, so a gap in the sequence is a retired entry rather
than a mistake.

---

### T1 — `check_trace_coverage` against the sglang source and the model structure
*M2.8.2, which says 先不做.*

Today the validator counts kernels and compares against a floor. It cannot tell
a trace that captured every layer from one that captured the first two and
stopped, because it has no model of what "every layer" means.

**Would settle it:** read the model's config for its layer count and the
engine's own module list, derive the expected kernel families per layer, and
check the trace covers them. It makes the validator model-aware, which is a real
cost and the reason it is deferred.

### T2 — the `vendor_tuned` bucket
*M3.3.*

`assets/lib/kernel_taxonomy.yaml` sorts kernels first-match-wins into
`collective` / `vendor_tuned` / `framework_native` / `routable`, and only
`routable` is a candidate. `vendor_tuned` — Tensile, rocBLAS — is excluded
wholesale, which is right for a first pass and wrong in general: a vendor kernel
can be beaten on a shape the vendor did not tune for.

**Would settle it:** a per-shape comparison against the vendor kernel's own
timing, so the bucket stops being a category and becomes a measurement.

### T3 — one handoff per operator
*M3.7.7, which says 目前可合成一个.*

`operator_workset` carries every candidate operator today. One handoff per
operator would let m4 fan out across operators and let one operator's failure
not invalidate the rest.

**Would settle it:** agent_sys has no "one output slot per element of a runtime
list". A kind naming two output slots is **exported for neither**
(`env_mgr/grants.py`, which names this as a hole and declines to close
it), so this needs framework work, not package work.

### T4 — the analysis programs are hand-written and may be narrow
*M3.8.*

`rank.py` and `identify.py` are rule tables over symbol names. They are
deterministic and cheap, and they can only recognise what somebody wrote a rule
for. The mission suggests AI-led with the program part frozen underneath.

**Would settle it:** run both on a second model's profile and count what the
rule tables miss. Until that number exists this is a preference, not a finding.

### T5 — the patch mechanism should hack the registry, not bind-mount files
*M5.3, which says 但现在就这样吧.*

> 这里的 patch 机制我都不是很认同，本身就应该是 hack sglang 的 registry 或者
> python 的运行。

The current mechanism bind-mounts replacement files over the image. It works, it
is provable (`check_patch_live` re-hashes inside the running container), and it
cannot express a change that is not a whole-file replacement. A registry hook or
an import hook would be finer-grained and would not need a container restart.

**Precedent if this is picked up:**
the integration stage's original `bench/pythonpath/sitecustomize.py` already injects code
into the engine's interpreter by `PYTHONPATH`.

**`overlay_files` stays until then.** Two mechanisms would mean two proofs that
the patch was live, and the proof is the expensive half.

### T6 — permission and visibility management for the shared container
*mission rule 7.*

All tasks sharing one runtime container saves the bring-up cost and unifies the
experiment, and it also means every task can see and change every other task's
state. agent_sys's zone model stops at the filesystem; it has nothing to say
about a container.

**Would settle it:** decide whether the shared container is a resource with an
owner, or an environment with no owner, and then say what a task may do to it.

---


### T7 — the comparability gate at bring-up (was E9′)

The two-arm design controls for session, node, trace, order and image, and
**not for node load at measurement time**. `check_service_live` proves a
deployment is *live*, not that it is *comparable* to the other arm's.

Measured: a patched arm at 475.7 ms mean ITL against a stock control at
470.3 ms — 1.1% apart — while the recorded run showed the same two arms 12%
apart, because they were measured fifteen minutes and one co-tenant apart.

**Do not widen the bars.** 5% / 10% are measured to be right: the within-arm
round-to-round spread on a steady node is ~2%. Widening them to 35% / 30% is the
wrong response to the gap above — it hides the comparability problem instead of
measuring it.

**Would settle it:** a quiet-node baseline, or interleaving the two arms'
measurements round for round.

### T8 — `seal_refused` has no reader (was C9b)

A correct refusal is computed and then discarded. The refused
`integration_report` is the worked example: the verdict was right, it stopped
the graph as designed, and the artefact that records *why* is not read by
anything downstream.

### T9 — `env_mgr.fs.layout` has two `copy_out` functions (was E14/C-two-copy_out)

One verifies the tree it copied and one is a plain `shutil.copytree`. The
consuming path uses the plain one, so **nothing on the way out of a zone
verifies a digest**.

### T10 — `check_workset_runs` hard-fails on rsd while `min_pass_ratio` forgives correctness (was C24 and E16)

A saturated node fails `max_rsd` on evidence that is otherwise correct, while a
kernel that is wrong on a minority of shapes can still pass. The two knobs
express opposite philosophies in one validator.

### T11 — ten closed `items_schema`s in the integration stage (was C23)

`additionalProperties: false` on an items schema rejects `logs` and `watchout`,
which the content type itself lists as optional. Harmless until somebody adds a
log.

### T12 — the mock cannot exercise M5.4's ad-hoc correctness rules


`check_acceptance` requires `min_adhoc_cases` per-run correctness cases with
their generator prompt recorded, none repeating a frozen case or each other, and
the same set on both arms (M5.4 — 免得作弊). **No sealed handoff carries an
`adhoc.json`**, because the requirement post-dates every run under
`cheat_for_mock/`, and synthesising one would be exactly what `MOCK-MAP.md`
forbids. So a mock run passes `--var adhoc_cases=0` and four of the validator's
rules are untested until the first real m5 run.

**Would settle it:** the first real run. Nothing to build; this is a note so that
a green mock is not read as coverage it does not have.

### T13 — `compare.py` finds the kernel's profile share by substring


M5.1.3.2 needs the optimised kernel's fraction of m2's profile. `compare.py`
finds it by matching `operator_id` case-insensitively against the `Name` column
of m2's kernel table. That is a rule table over symbol names — the same shape as
`rank.py` and `identify.py`, and the same objection as T4: it can only recognise
what somebody wrote a rule for, and an operator whose workset name differs from
its kernel symbol silently yields "share unknown".

It fails **safe**: an unmatched operator produces
`kernel_reconciliation.unavailable_because` rather than a wrong number, and the
block is a warning rather than a blocker anyway.

**Would settle it:** m3's `operator_identity` already resolves a logical operator
to its kernel symbols — carry that mapping into the workset and have `compare`
read it instead of guessing.

### T14 — a task body cannot name the interpreter the run is using
*Reproduced in three stages.*

`cli/main.py` exports `AGENT_SYS_DEMO_PYTHON` into **`validation_env` only**,
and the comment above it says a task body never reaches it. So a task body's
policy `PATH` resolves `python3` to the system interpreter, which may have
`yaml` and `jsonschema` and **not `referencing`**, and may have no `torch` at
all — which is what m3's `build_workset` entrypoints need.

**Worked around in the package, four different ways**, which is itself the
argument for fixing it upstream: `schema.py` stopped needing `referencing` by
inlining cross-file `$ref`s; `mock_adapt.sh` probes for an interpreter that can
import what it needs; `build_workset` probes for `torch` and refuses up front
with the reason; m4 carries the chosen interpreter through `KFO_PYTHON` **and**
on `PATH`, because the workset's entrypoint is a shell script and an interpreter
can only reach it through the environment.

**Would settle it:** export the variable to task bodies too, or give a body a
declared way to ask for an interpreter with named imports. This is framework
work (`agent_sys/cli/`), not package work, which is why it is here.

**The failure mode is why it is worth fixing rather than working around.** In a
validator the missing import produces a non-zero exit and **no `verdict.json`**,
so the phase reads a broken validator rather than a refused handoff — the same
signature as `check_deploy_serves`'s crash. Twelve of twenty-one validators are
exposed to it.

### T15 — `E2E_KIT_ENGINE_EXTRA_ENV` is a seam with no consumer


`deploy_kit.layout.yaml`'s `runtime_contract` requires a kit to honour both
`E2E_KIT_ENGINE_EXTRA_ARGS` and `E2E_KIT_ENGINE_EXTRA_ENV`. The first is used —
it is how m2's two lines differ by CUDA graph on/off. **The second has no
consumer**: both m2 lines leave it empty, and the profiler-attached line needs a
*router* flag (`E2E_KIT_ROUTER_EXTRA_ARGS`) that no engine seam can reach.

The example that motivated it was `SGLANG_TORCH_PROFILER_DIR`, which **nothing
in this package sets** — the engine is told where to write per capture, in
`/start_profile`'s `output_dir`. The requirement reached two contract documents
before that was checked.

**Kept and labelled rather than removed**, because a required parameter is
cheaper to keep than to re-negotiate, and because the argv/environment
distinction is real even though this instance of it was not. **Would settle it:**
the first real consumer, or a decision to drop the requirement.

### T16 — `git`'s `index.lock` retry is not idempotent, and its no-op is silent

`CONTRACT §8a` tells a writer whose commit hits `index.lock` to wait a second and
retry. Measured: **the retry can silently do nothing.** In the seconds between
the failure and the retry, a concurrent commit may name a tree and take the first
writer's dirty file; the retry then finds nothing to commit for that path and
**exits quietly**, so the writer believes their work is committed when another
commit carries it.

**Verifying the path does not catch it**, which is what lets it survive:
`git show --stat --name-only HEAD` prints exactly the expected path, because HEAD
is the other commit **holding that path**. **Confirming the path is not
confirming the commit.** §8a's check therefore leads with
`git log -1 --format='%h %s'`.

**Would settle it properly:** one worktree per writer. The reason to decline that
is work already in flight in one tree; it does not hold once the remaining work
is runs rather than edits.

### T17 — the model-specific engine flag groups are free-form strings and cannot be checked
*From comparing a second model's sealed recipe against what `shared.yaml` can
express.*

`E2E_DSA_ARGS` and `E2E_PARSER_ARGS` exist because their contents are **traps
that do not error when wrong**: a model with no DSA attention path does not take
`--dsa-*-backend`, and the wrong `--reasoning-parser` yields an **empty
`content` for every request while the request still succeeds**. That is why they
are parameters rather than something the agent derives.

Both are free-form strings. GLM needs *two* flags in one of them —
`--reasoning-parser glm45` and `--tool-call-parser glm47` — and a single string
carries both happily. **That is also the limit: it carries a typo in either one
just as happily.** `--reasoning-parser glm54` is accepted by the package, passed
to the engine, and produces exactly the empty-`content` failure the variable
exists to prevent.

So the variable makes the fact *sayable* and does nothing to make it *right* —
the same shape as `items_schema` validating a filename instead of a file
(CONTRACT §3.1). Nothing downstream can catch it either: no schema sees the
string, and the one probe that would notice — `completion_nonstreaming`'s
non-empty `content` — fires only after a full bring-up.

**This is a real limit on what a second model proves.** GLM running does not
show the package can express GLM's flags *correctly*; it shows one hand-checked
spelling worked.

**Would settle it:** a per-model manifest under `assets/schemas/` listing the
parser and attention-backend names an engine build actually accepts, validated
at load time — the engine already knows the set, so this is extraction rather
than invention. Cheaper interim: have the producer read the chosen parser back
out of `/get_server_info` and record it in `environment.yaml`, which turns a
silent wrong answer into a recorded one.

### T18 — `shared_identifiers` sees flags, and a shared host is bound by more than flags
*From running a **second** model's kit; the first model's kit could not have
shown it.*

`check_deploy_kit`'s `shared_identifiers` scan is a deliberate list of flags —
`--name`, `--publish`, `--volume`, `--mount`, `--port`, `-p`, `-v` — on the
stated grounds that guessing "this string looks like a container name" fails
honest kits. That reasoning still holds. What a second kit showed is that the
list is **narrower than the property it stands for**, and in four measured ways:

| in the GLM recipe | why the scan misses it |
|---|---|
| `docker rm -f … glm53_standalone` | an argument to a **command**, not a flag — and it *destroys* rather than binds |
| `--listen-client-urls http://0.0.0.0:2379` | a port literal **inside a URL**, and the flag is not in the list |
| `--etcd-endpoint $MY_IP:2379` | the same, and it is the **consumer** of the port above |
| `reset_gpus.sh` doing `kill -9` on every KFD pid | not an identifier at all — a node-wide destructive act with no flag to see |

The third is the instructive one. **Parameterising the producer of a shared
identifier and not its consumer leaves the deployment broken in a way that looks
like something else** — the router fails with `ConnectError: All connection
attempts failed`, naming neither the port nor the mismatch. It is an easy mistake
to make while fixing the other three.

**Would settle it, and it is a design question rather than a patch:**

1. a list of flags whose *value* is a `host:port` or a URL, scanned for a literal
   port — `--etcd-endpoint`, `--listen-client-urls`, `--advertise-*`, `--host`.
   Cheap, and it generalises the existing rule rather than replacing it;
2. a rule that a port literal appearing **more than once** in `scripts/` is
   almost certainly a producer/consumer pair, so parameterising one occurrence
   and not the others is a *detectable* half-fix;
3. destructive verbs — `docker rm -f`, `docker kill`, `kill -9`, `pkill` — with a
   bare literal or an unbounded match are a different category from binding and
   probably want their own rule. A kit that kills every GPU process on the node
   passes every check this package has today.

Until then the layout's `shared_identifiers` comment should say what it does
**not** cover, because a check that reads as "no identifier is frozen" and means
"no identifier reaches one of seven flags" is the gap between a claim and a
measurement that `todo.md` exists to record.

### T19 — a GPU *set* is a bound identifier with no variable, and prose is not a validator
*From composing a bring-up against a shared node.*

Every other identifier this package binds on a shared host has a `--var`:
container name, three ports, the container workdir. **The GPU set does not.**
`E2E_TP` is a *count*; the *index* is left entirely to the agent's `rocm-smi`
read at `deploy_and_prove.task/readme.md` STEP 1, whose criterion is only that
*"you can say which device index you are taking"*.

The producer brief's own trap list already names the GPU index alongside the
others — *"container names, host ports, the container workdir and the GPU index
sit in one namespace with everybody else"* — so the property is agreed and the
parameter is simply missing.

**Measured on a node where this matters.** GPUs 0–3 hold ~300 GB each with no
`docker ps` entry behind them — a co-tenant no kit may `kill -9` — while 4–7 are
free. An agent that takes the default devices takes 0–3 and OOMs against
another tenant's work — and the failure surfaces as a bring-up problem, not as a
placement problem, which is the expensive kind.

**The available workaround, and why it is not a fix:** a sentence of the form
*"GPUs 0-3 on this node are held by another tenant … take only 4-7"* goes through
`--var instruction=`, which is the declared channel for a plain-words site fact
and the right way to avoid changing the package mid-run.
But **a site fact carried in prose is a site fact nothing validates.** No
validator can tell that the agent read it, no schema records what was asked, and
`environment.yaml` has no field that would let `check_deploy_kit` compare the
devices requested against the devices used.

**Would settle it, cheapest first:**

1. `E2E_GPU_DEVICES` in `shared.yaml`, default empty meaning *"choose freely"*,
   carrying a `HIP_VISIBLE_DEVICES`-shaped list when the operator knows the
   answer. This is the same shape as `E2E_DSA_ARGS`'s `none` sentinel: the
   distinction that must survive is *"the operator said 4-7"* versus *"the
   operator said nothing"*;
2. `fixed.gpu_devices` in `environment.schema.json` beside the existing
   `gpu_count`, written by `env_render.py` from what the bring-up actually used.
   `gpu_count: 4` today records *how many* and cannot record *which*, so two runs
   on disjoint halves of one node produce identical records;
3. with both, `check_deploy_kit` can compare requested against recorded — which
   is the step that turns this from an instruction into a check.

Note this is the **same shape as T17**: a variable that makes a fact *sayable*
does nothing to make it *right*, and here there is not even a variable to say it
with. The mechanism itself is verified — `HIP_VISIBLE_DEVICES=4,5,6,7` inside the
built image yields `torch.cuda.device_count() == 4` — so it is only
unparameterised and unchecked.

### T20 — a borrowed resource's cleanup must be idempotent and unowned
*From a probe container left behind on a node the run no longer holds.*

**The shape.** A body stands up a container on a node it does not own, so that
another stage can close an unverified line — a `docker exec` — without acquiring
a container lifetime that `CONTRACT` §5 puts elsewhere. It carries a label naming
the run, holds one GPU visible and no GPU memory, publishes no ports, and its
teardown is explicitly owned by the creator: *"teardown is mine and I am holding
it; no timer, because the only thing that should end it is somebody saying the
verification is done."*

**That reasoning assumes the node outlives the decision.** It need not. A
scheduler allocation can be cancelled between the work finishing and the
`docker rm`, after which no `exec` into that node is available and the container
is stranded — possibly still running, on hardware now allocated to somebody else.
Whether such a container survives cancellation is **unmeasured**: nobody has
checked whether the scheduler reaps containers at job teardown.

**Cleaning it afterwards is the wrong response.** Taking a hold on a node
specifically to reach into it while another tenant is working there is worse than
the thing being cleaned up, and the rule that says *never `docker rm -f` what you
did not create* protects that tenant exactly as it protects this run. The
container is reclaimed when a hold on that node next exists, and not before.

**A timer is also the wrong fix** — it would have cut the verification that was
still in progress. The general form:

> **A borrowed resource's cleanup has to be idempotent and unowned**, so that
> anyone with access can do it and nobody has to be alive to decide.

Concretely, for anything this package stands up on a node it does not own:

1. the container carries a label naming the run — which is the only reason a
   stranded one is identifiable at all;
2. **a reclaim pass keyed on that label is runnable by anyone**, at any time,
   with no knowledge of who created what. `assets/lib/reclaim.sh` is the existing
   place for it, and `CONTRACT` §5.0 already requires bodies to call it in a
   `finally`. What is missing is the case where the *creator* never gets to run
   its `finally`;
3. so a run's **first** act on a node should be to reclaim the labels of runs
   that are provably over, and a hold's **last** act should not be the only
   chance. Cleanup that depends on a specific process still being alive is the
   same class as a rule that depends on someone remembering.

Pairs with the reclaim finding in `check_deploy_serves`: a teardown that crashes
warns that ports may be held, and the ports it names may belong to somebody
else's live run. Both are about cleanup needing to be safe for a stranger to run.


### T21 — the completion probe grades shape, not answer, and the bar that would fix it needs a measurement
*From reading a bring-up's completion output. The probe's `direction` text is
corrected; the bar is what needs a measurement.*

**What passes.** A reasoning model with no `--reasoning-parser`:

```
finish_reason : stop
usage         : {prompt_tokens: 23, completion_tokens: 157, reasoning_tokens: 0}
content       : "Here's a thinking process:\n\n1.  **Analyze User Input:** …"
```

157 tokens of chain-of-thought in `content`, `reasoning_tokens: 0`, to the prompt
*"What is the capital of France? Answer with one word."*

**Why it passes.** `probes.yaml`'s `completion_nonstreaming` asserts
`status: 200`, `finish_reason equals stop`, `content nonempty: true`, and
`model not_matches ^/`. All four hold. **`nonempty: true` catches a parser that
removes too much and is structurally blind to one that removes nothing.** So the
probe discriminates one direction of the reasoning-parser fault, not both.

It is the same rotation as *"had the parser been wrong, `content` would have been
empty on a request that still returned 200"*, with the sign flipped.

**Three fixes considered and rejected, each for a failure this package has
already paid for elsewhere:**

| candidate | why not |
|---|---|
| match `Paris` in `content` | **does not discriminate.** A reasoning preamble ends with the right answer, so it passes both ways |
| bound `content` length | discriminates, and the bound would be a number invented from **one observation** — T7's 35 % / 30 % widening, pointed the other way |
| require `usage.reasoning_tokens > 0` | precise about the property, and **refuses a legitimate non-reasoning model.** It asserts a fact about the model while claiming to test the deployment |

**Proposed shape, needing one measurement before it is written:**

> `usage.reasoning_tokens > 0` **OR** `content` is short —
> *either the reasoning was accounted separately, or there was none to account.*

A statement about **the parser** rather than about the model, and it fails in the
loud direction. "Short" is the number nobody has.

**The measurement that would set it**, small enough to ride along with a future
bring-up rather than needing its own: send this exact one-word prompt to (a) a
reasoning model **with** a correct `--reasoning-parser`, and (b) a non-reasoning
model, and record `len(content)` for each. The bar goes between them, nearer (a).
Two requests against a deployment that exists for another purpose. **Until that
is taken, do not invent the number** — a validator bar chosen on the login node is
the artefact-tuned-to-the-instrument mistake this package refuses elsewhere.

**Related, and separable:** `E2E_PARSER_ARGS` defaults to `none`, which is correct
only for a non-reasoning model. A reasoning model whose chat template carries
`<think>` / `</think>` is not one, and an image's
`ReasoningParser.DetectorMap` may offer two candidates — `qwen3` and
`qwen3-thinking`. **Which of the two is right is untested**, and T17 is why that
matters: a free-form string accepts `qwen3-thnking` as happily as `qwen3` and the
wrong one produces no error.
Setting the default is not this entry's fix — a correct default would have
*hidden* the probe's blindness rather than removed it.

### T22 — `E2E_STAGE` names a per-stage fact and `--var` carries one value per run
*From declaring the variable `check_agent_env.py` flags. Declaring it is correct
and does not make it right.*

`env_render.py` stamps every tolerated difference with
`{"stage": os.environ.get("E2E_STAGE", "")}` so that a reader of
`warnings[].stage` can tell **who** tolerated it. The name is now declared on
`runner` and on the `kind: ai` agents that reach `env_render.py`, spelled
`'${stage:-}'` on every one of them — byte-identity is what
`check_agent_env.py` requires, and diverging locally would be exactly the drift
it exists to catch.

**But the value is a property of the stage, and `--var stage=m1` is a property of
the run.** One command line drives all five stages, so a single `--var` can
stamp at most one of them truthfully; the other four get a label naming somebody
else's stage, which is **worse than the empty string it replaces**. Empty says
"nobody recorded who"; `m1` on m3's warning says something false.

So the variable is currently in the one state where it cannot be used: correct
when unset, wrong when set.

**What would settle it — a value bound per agent rather than per run.** Three
shapes, cheapest first:

1. **A constant in each agent's `env` block** — `E2E_STAGE: m1` on
   `e2e_deployer`, `m3` on `workset_builder`, and so on. Correct by
   construction, no `--var`, nothing to pass. **It requires
   `check_agent_env.py`'s byte-identity rule to make room**, and the `DELIBERATE`
   map is already the mechanism for that — an entry per agent, each carrying the
   reason. That is five entries whose reason is identical, which is a hint the
   rule wants a third category rather than five exceptions:
   *per-agent-by-design*, checked for **presence** but not for agreement.
2. **Derive it in `env_render.py`** from something the body already knows.
   `AGENT_SYS_MY_ZONE` and `AGENT_SYS_OUTPUT_<KIND>` are both exported and both
   name the closure; a mapping from output kind to stage would need no variable
   at all. Cheaper to run, harder to read.
3. **Leave it empty and delete the field.** A field that is empty in four cases
   out of five is not carrying information, and `warnings[]` already lives
   inside a record that names its producer. Worth considering rather than
   dismissing: the least code is the field nobody has to keep true.

Not urgent — no run is blocked, and the empty string is the safe state. It is on
this list because **the next person to notice the empty stamp will "fix" it by
passing `--var stage=`**, which is the one action that makes the record lie.

### T23 — `fixed.gpu_count` is the one required field with no definition
*Third direction on T19.*

A bring-up record can say `gpu_count: 8` on a node where four cards are held by a
co-tenant at 96–98 % VRAM, and all 21 verdicts pass.

**`gpu_count` is not a claim among two measurements, and the difference changes
the fix.** `env_render.py` says:

> `gpu_arch`, `gpu_count` and `image_id` are absent on purpose: they are
> discovered during bring-up, and a variable holding them would be a claim

**All three arrive as `--set` and `env_render.py` measures none of them.** The
real difference is upstream, in the producer brief: STEP 1 tells the agent to run
`docker image inspect --format '{{.Id}}'` and `rocm-smi`, so `image_id` and
`gpu_arch` are *transcriptions of a named command's output*. **No step produces a
count.** STEP 1's criterion is *"free VRAM per device exceeds the checkpoint size
… and you can say which device **index** you are taking"* — an index, never a
total. So `gpu_count` is not a claim where the others are measurements; it is the
one of the three that no instruction generates.

**And the deeper reason nothing refused it: the field has no definition.**
`environment.schema.json` gives `gpu_count` exactly

```json
{"type": "integer", "minimum": 1}
```

**no `description`** — alone among the eight required `fixed` fields, every one
of which otherwise explains what it means and why (`gpu_arch` says why an
architecture and not a product name; `image_id` says why a digest and not a tag).

So `gpu_count` has two defensible readings — **cards present on the node** and
**cards this deployment could use** — and `8` is *true* under the first, so an
agent writing it is not wrong. **A field that cannot be wrong cannot be a
measurement**,
and `fixed` is promised as 可固化环境, which is the second reading.

**Deliberately not proposed: adding `gpu_count` to `check_environment`'s
`compare_fixed_across_inputs`.** Those four fields are four on purpose and the
rule is explicit that bars are not widened. A cross-input comparison
would also not catch this — every stage agrees on the same undefined value.

**What would settle it, and it is one decision, not three:** say which reading
`fixed.gpu_count` requires, in the schema, in a `description` like every
neighbouring field has. Then the producer brief gets a STEP 1 criterion that
generates it, and `check_deploy_kit` can check the record against something.

**T19's `fixed.gpu_devices` makes the decision cheap rather than forced**,
which is why these are one problem: with a device list, `gpu_count` keeps the
node fact and `len(gpu_devices)` carries what the deployment used, and neither
reading has to lose. Without it, whichever meaning is chosen makes the other
unrecordable — and the case above needs both to be honest: *eight present, four
usable, one taken.*

Not blocking. Recorded rather than fixed **because the fix is a definition and
the definition is entangled with T19** — writing a criterion first would bake in
whichever meaning the producer happened to pick.

---

### T24 — a fallback nobody can reach is still a second reader of the number

**Stage m5. Not blocking; the live half is fixed and this is what is left.**

`assets/accept/measure.sh` reads its load shape as
`${E2E_MAX_CONC:-256}`, `${E2E_WORKERS:-16}`, `${E2E_BLOCK_SIZE:-512}`,
`${E2E_REQ_TIMEOUT:-900}` and `${E2E_TRACE_END_MS:-120000}`. `shared.yaml`'s
`runner` declares 32, 8, 512 and 900; `e2e_integrator` now declares the same
four, plus 60000 for the trace window, deliberately.

**The live defect is fixed.** Until those declarations landed, none of the four
reached this stage at all — a name only `runner` declares does not reach a
`kind: ai` agent (`env_mgr/material.py`) — so the script's fallbacks win and m5
replays at **concurrency 256 against m2's 32**, with `--var max_conc=` inert on
one side of a comparison M5.1.3.1 requires to hold within
`stock_vs_m2_tolerance`. An omission check over m5's manifest only reports this
once the names are on `runner`, which is why it has to be re-run when
`shared.yaml` grows.

**What is left is three numbers for one knob** and no way to tell which is
intended: `E2E_TRACE_END_MS` is 180000 on `runner`, 60000 here, 120000 in the
script. The declaration wins wherever the graph runs the script, so the
fallbacks are unreachable *there* — but `measure.sh` is also meant to be run by
hand, which is the case the fallbacks exist for, and by hand it measures
something the graph never would.

**Not reconciled here, because the right value is a measurement and not an
edit.** 256/16 and 32/8 are both defensible shapes, nothing says which the
two-arm comparison should use, and picking one in a comment is how a number
acquires a third reader. What settles it: the offered load the m2-vs-stock
comparison assumes is stated once, and both the declaration and the fallback
cite it.


---

### T25 — a run records the environment it minted, but not the vars it was started with

**Not blocking. Cheapest item on this list.**

Every `--var` a run is launched with shapes what the graph does, and **none of
them survive into the run tree.** `handoffs/<hid>/v<N>/content/items/*/
environment.yaml` records the environment m1 *minted* — node, image, image_id,
tp_size — which is a fact about the deployment, not about the request. The
store keeps tasks, events and handoffs. Nothing keeps the command line.

So *"what was this run asked to do"* is unanswerable from the artefact, and it is
asked constantly, by people who were not the one who typed it.

**Three questions the record would answer**, each of which otherwise costs a
message or a run:

1. **Was `--var expect_ranks=2` passed?** `check_trace_coverage` is `strong` and
   declares `${expect_ranks:-8}`; a mocked TP-2 trace against a TP-4 deployment
   is three numbers, and `expect_ranks` is deliberately not derived from `${tp}`
   (`steps/m2_profiling.yaml`). Whether a refusal is expected cannot be checked
   from the tree either way.
2. **Was the run still alive?** A killed run leaves a tree that does not say so.
3. **Was its agent still alive?** Also not recorded — see T26.

The same gap in the other direction is the more expensive half: a
`check_deploy_kit: FAIL` on a previously green stage can be caused by `--var
image=` naming a tag present on the node instead of the one the sealed kit
renders. The validator refuses correctly, and **believing that failure sends
somebody auditing commits for a defect that is in a command line**
(`CONTRACT` §4.4, face 2) — with no reading of the run tree able to distinguish
the two.

**What would settle it:** the run writes its resolved variables — every `--var`
plus every default that was taken — into the run root at launch, once. Resolved
rather than raw, because a default that was *taken* is exactly the case nobody
can reconstruct afterwards. `spec_loader/variables.py` already computes it; the
value is thrown away after substitution.

**Not fixed here**: `agent_sys/cli/` and `agent_sys/spec_loader/` are outside
this package.

---

### T26 — killing a run does not kill its agents

**Not blocking. It changes what "the run was killed" can be relied on to mean.**

**One level up from the cgroup problem.** A cancelled scheduler job does not
reclaim its GPUs, because the containers talk to the **host** docker daemon and
are therefore not in the job's cgroup. The same shape holds one layer further
out: **an AI agent a run dispatched is not in the orchestrator's process tree**,
so killing the orchestrator leaves it running.

Measured. Twenty minutes after a run was ended, its tree was still being written:

```
files written since the kill   29 -> 30      (two readings, two minutes apart)
newest write                   moving forward
```

under `zones/task.…/task.…/task.…/config/projects/…/<uuid>.jsonl`, an agent's own
transcript, with its last entry an `Edit` tool call. All three task records in
that chain still read `status=running`, and the process is findable only by
matching its `--system-prompt` against the task's brief.

**What an orphan does while unattended.** It creates containers and replaces
others on the shared host, ten to twenty minutes after the kill. Those containers
are attributable to nobody: each person asked correctly answers that they are not
theirs, and the object belongs to a run everyone believes is dead.

**Why this is not merely untidy.** The rule is that the first real run of a
`kind: ai` closure happens supervised, *because* an AI agent with a live node and
a docker daemon is the one thing here that can change state nobody asked for. An
orphan is unsupervised **precisely because everyone believes the run is dead** —
the belief that removes the supervision is caused by the same event that creates
the orphan.

And nothing it produces can land: the orchestrator is gone, so no handoff can be
sealed and no task can progress. It holds a node and mutates a shared host for an
outcome with no consumer.

**What would settle it:** a run that is ending terminates the agents it
dispatched, and says how many, before it reports that it stopped. Failing that,
the run root records agent PIDs at dispatch so a person can check — which is
T25's record in a second use.

**A note on the method, because it is the transferable part.** A probe reporting
*"zero files modified in the run tree in the last N minutes"* is **not a
measurement** if `find` on that host is `bfs`, which does not support `-newermt`
and errors to stderr while stdout is piped to `wc -l` — the empty result reads as
zero. That is a check that cannot fail. What works instead is opening a file and
counting its entries: **a file you have opened cannot lie to you about whether it
exists.** `CONTRACT` §4.4, in the instrument built to catch §4.4.


### T27 — a default that takes everything free is not a default
*T19's third direction. Five items, one cause; they belong together because
fixing any four leaves the fifth failure available.*

**The shape.** A kit's `env.sh` fills its device list from a picker when the
caller passes none:

```sh
: "${E2E_KIT_GPU_DEVICES:=$(_pick_gpus)}"
```

`_pick_gpus` returns **every free card**, so a worker started without the safety
variables takes the whole node and collides with a deployment that had already
named its own four.

**The victim is the well-behaved container.** The only deployment on the node
that declared `HIP_VISIBLE_DEVICES` is the one that gets stepped on. That is the
strongest argument for pinning there is: **declaring your cards protects you from
nothing if the next process declares nothing**, because "nothing" means "all of
them".

#### The five items

**1. Pin the container, not the worker process.** `HIP_VISIBLE_DEVICES="$GPUS"`
set inline on an `exec` binds the processes that carry it and nothing else, so
`docker inspect` on such a container shows no device restriction at all — which
is how a container whose *worker* is pinned reads as "unpinned and greedy". Pin
at `docker run` and a new process inside inherits the right default instead of
seeing every card.

**2. Bound `_pick_gpus` to `tp_size`.** A picker asked for a deployment of width
N should return N cards. Returning everything free is not a conservative default,
it is a land grab that happens to be quiet on an empty node and hostile on a
shared one. Note the shape: **it is `E2E_DSA_ARGS`'s trap inverted** — there the
danger is a value that is wrong and does not error; here it is an *absent* value
filled in maximally, which also does not error.

**3. Record what was picked.** `fixed.gpu_count` records **how many** and cannot
record **which**, so two runs on disjoint halves of one node produce identical
records, and attributing a card afterwards costs `rocm-smi --showpids` and PID
matching. Whatever `_pick_gpus` decides must land in `fixed.gpu_devices`.

**4. The pick must come from the probe, not sit beside it.** Items 1–3 assume the
device set is *chosen*; it need not be. A kit whose `env.sh` is a literal
`: "${E2E_KIT_GPU_DEVICES:=0,1,2,3}"` — written by an agent that ran `rocm-smi`
first and then hardcoded the first four cards — satisfies item 2 exactly,
matching `E2E_TP=4`, and still binds four cards a co-tenant is mid-load on. **A
probe reading that does not reach the parameter is decoration.**

  And nothing checks it: a `deploy.sh` that prints `preflight ok: PORTS free,
  NAMES free` preflights the port band and the container names and **never looks
  at a card**. The port check is the model — it exists, and it aborts rather than
  waiting or stealing. The cards are simply not treated as a namespace.

**5. Nothing compares a document's conclusion to its own numbers.** A kit's
`results/preflight.json` can carry, under one `measured_at`, a structured
`gpu_cards[]` reading of **198–199 GiB used, 89–90 GiB free** and a prose
`gpu_devices_rationale` claiming *"all eight were free (≤300 MB used each, no
co-tenant)"* — wrong by a factor of three against the numbers two keys above it.
It is not two readings with nothing to tell them apart; it is **one honest
reading and a conclusion that ignores it**, and the rationale can be perfectly
considered while resting on a stale premise.

  `check_deploy_kit` grades the layout and validates `environment.yaml` against
  its schema; it never opens `preflight.json`, and the `results/` floor asks for
  *two non-empty `.json` files* — which such a file satisfies while contradicting
  itself.

  **It is not expressible as an evidence rule today**, which is the obstacle
  rather than an excuse: the existing rules are regexes over a directory
  (`forbid`, `require_each`, `require_together`), and *"this sentence disagrees
  with that array"* is not a regex. A directory-level *"some file carries a
  `*_at`"* rule discriminates nothing — it passes a well-formed kit and a
  self-contradicting one alike.

  What catches it is narrower and belongs to the producer: **STEP 1 requires the
  rationale to cite the numbers it rests on**, free VRAM per card quoted from the
  same reading, so a stale premise is visible in the sentence rather than only in
  the array. A conclusion that restates its evidence cannot silently outlive it.
  Written into `deploy_and_prove.task/readme.md` STEP 1.

**Why all five.** Pin without bounding and a caller who passes no list still
grabs the node — pinned, but pinned to everything. Bound without pinning and the
container still sees cards the worker was told to avoid. Do both without
recording and attribution is back to PID matching. Do all three and the pick can
still be a literal the probe never reached. **T19 is the field; these are the
things that have to be true for the field to mean anything.**

#### The env pin is intent, not enforcement — and both are wanted

`start_container.sh` exposes every card's device node and narrows with a
variable:

```
--device /dev/kfd
--device /dev/dri                                  <- EVERY card on the host
--env "HIP_VISIBLE_DEVICES=${E2E_KIT_GPU_DEVICES}" <- the pin
```

So `docker exec -e HIP_VISIBLE_DEVICES=4` overrides it and runs on a card the
deployment was never allocated. It does not fail; it returns a number. **A
convention a later process can override is not construction.**

**What construction would look like:** `--device /dev/dri/renderD<N>` per card
rather than the whole `/dev/dri`, so the cards the deployment did not take are
not present in the container at all. Unmeasured — nobody has checked whether the
cluster's authorisation layer accepts per-card device flags, and **that is the
question to answer before it is written into a brief.**

**Keep both. Do not replace the env pin with the whitelist.**
`run_in_container.sh` refuses an exec whose requested card is outside the
container's own `HIP_VISIBLE_DEVICES`. A container pinned only by device
whitelist has no such variable, so that check finds nothing to compare, falls
into its *unpinned, constrains nothing* branch, and **waves the exec through**.
The work still would not run on an absent card — but it would fail deep inside
HIP instead of being refused by name. *Safe by absence rather than safe by
refusal*, which is the wrong direction.

| | what it gives |
|---|---|
| `--env HIP_VISIBLE_DEVICES` | a **readable statement of intent** a consumer can check against and refuse by name |
| per-card `--device` | **enforcement** a later process cannot override |

Until the second exists, the division is: **the kit states the intent, and the
consumer refuses to violate it.** `run_in_container.sh`'s check is the mirror of
`start_container.sh`'s inside-out check, and is where enforcement lives today.

#### The class behind three of the five

`captured_at` searched for where the file says `measured_at`; transcript greps
matching *file reads* and reported as *decisions*; a comment about **ports** read
as a comment about **cards**. All three are the *search* half of the pattern
`CONTRACT` §4.4's observer section describes: **an anticipated string standing in
for the data.** The remedy is the same and cheap in every instance — *list what
is there instead of searching for what you expect.*


### T28 — T21's bar, measured: 7 characters against 526
*T21 said "do not invent the number". The number exists now.*

Two engines alive at the same moment on one node — one launched with
`--reasoning-parser qwen3`, one without — so the A/B is one model, one node, one
image, two `curl`s.

| | parsed (`:8101`) | unparsed (`:8118`) |
|---|---|---|
| `finish_reason` | `stop` | `stop` |
| **`len(content)`** | **7** — `'\n\nParis'` | **526** |
| `reasoning_content` | 554 chars | `null` |
| `reasoning_tokens` | **157** | **0** |
| `completion_tokens` | 160 | 151 |

**Both pass today's probe.** T21 demonstrated rather than argued.

**`completion_tokens` is not the discriminating axis** — 160 vs 151, the wrong
way round and inside noise. A rule written against it would be a check that
cannot fail, in the validator whose blindness it exists to fix: `CONTRACT` §4.4's
third face.

**The rule, now with numbers behind it:** `usage.reasoning_tokens > 0` **OR**
`len(content) <= 200`. Parsed passes on the first clause (157); unparsed fails
both (0, and 526). The separation is a factor of 75, so the bound is chosen with
room rather than fitted to one observation.

**The caveat, and it ships with the bar rather than after it:** *two points, one
model. A verbose non-reasoning model answering a one-word question in three
sentences fails a 200-character bound, and `reasoning_tokens > 0` does not
protect it, because such a model reports 0 too.* The bound is defensible and not
proven general, and this entry is where it says which.

**Not implemented yet.** It needs two new things in
`probes.yaml`/`probe_runner.py` — a length assertion and a disjunction — and
**editing a validator whose body is being copied into a zone mid-run is how a run
gets a fault nobody can attribute.** It goes in between runs, not during one.

---

### T29 — a crashed instrument and a refused artefact are the same value in `verdict.json`

`zone.py` writes `verdict.json` as `dict[str, bool]`. A validator that
**refused** and a validator that **could not run** therefore produce the same
value, and the graph reads no difference between them.

Measured: `check_workset_shape` crashing on a `ModuleNotFoundError` writes no
verdict at all, and `operator_workset` comes out `invalid` — *a missing
dependency reported as a judgement about the artefact*. That is the false
attribution `check_workset_runs` exists to prevent, arriving one layer up, in the
validator itself.

**Worked around, not fixed.** Both workset validators now catch, say `THIS
VALIDATOR DID NOT RUN`, keep the traceback, and write **False** — because a
check that did not execute has established nothing, and passing on that basis
is the one option that is actually wrong. But False is still a judgement in the
only field a consumer reads, and the text beside it is the sole thing carrying
the distinction.

**What the framework owes:** a third verdict state — `undecided` / `errored` —
so a phase can tell "this artefact is bad" from "this instrument is broken"
without parsing prose. The consumer difference is real: the first should stop
the graph, and the second should stop *and name the instrument*, because
re-running it after fixing the artefact will fail identically.

**Related and separate:** a validator's stdout is not kept anywhere, which is
why the prose has nowhere to live either. `workset_io.write_report` puts it
beside `verdict.json` for m3's two workset validators; every other validator in
this package discards its own, and a general fix belongs in `zone.py`.

### T30 — a finding recorded against one stage is a question for every stage

A record of the form *"a validator declares no agent, so the package's `env`
block never reaches it, and with `transport_env` unset the scheduler client has
no controller address"* reads as history about the stage it is filed under. The
same hole is in every other stage's validators, and reading the paragraph does
not surface that — the construct is not named, so nobody searches for it.

Two more of the same shape: `require_visible_on_node`'s misattributing message is
shared by six call sites across four stages, and `_pick_gpus` taking every free
card is the same property as a `:=4` default in a different file (T27).

**The remedy is not diligence, it is a grep.** When a defect is recorded against
any stage, the record names the *construct* — a variable, a helper, a default, a
message — and every other stage greps its own files for it before the next run.
**A finding filed against one stage is unowned by everyone else, and unowned is
where it stays.**

### T31 — naming a class is not sweeping for it

**The operational half of T30, and it fires on your own findings rather than
other people's.**

Finding a scheduler-client variable absent from a validation zone names the
class — *a variable present in an interactive shell and absent in a closed zone*
— and does not by itself find the next instance. In the same file, `$HOME`
resolving to `/home` inside that zone produces `-v /home:/home` and a denial from
the authorisation layer. One `grep -nE '\$(HOME|USER|PWD|PATH)'` at the first
finding reaches the second.

The same shape appears in fixtures: building a replacement fixture **by
subtracting the variables you suspect from your own contaminated shell**, rather
than taking the environment from the zone, keeps whichever one you did not
suspect.

**Naming a class produces the search term. Doing the search is a separate act,**
and the gap between them is where the next instance lives. When a class is
written down, the same commit carries the sweep, or says why it does not.

**The sharpest instance is not about sweeping — it is about what a comment cannot
do.** A task-body output capture written as
`{ /bin/sh "$0" "$@" 2>&1; echo $? > "$_st"; }` turns a body exiting 7 into a
wrapper exiting 2: `set -e` kills the subshell before `echo $?` runs. **A comment
twenty lines below in that same file says exactly that** — *"under `set -e` a
simple command exiting non-zero kills the script before the assignment runs"*.
The trap is documented, in the file, and walked into anyway while adding a
mechanism whose entire purpose is to preserve exit information.

**A comment warning about a trap does not prevent the trap. The test does.** And
the transferable half is what the test has to be: *a capture mechanism verified
only on the success path is worse than none, because it looks like evidence.*
The same holds for a null overlay in any other subsystem — **an instrument that
cannot be observed failing is not an instrument.**

### T32 — the evidence records which node, and not which card

`evidence.measured_on` carries `node`, `gpu_arch`, `container` and `at`. It
does **not** carry the GPU index the measurement ran on, and neither does
`environment.fixed`, which records `gpu_count` — how many, not which.

**Why it matters now rather than as tidiness.** `check_workset_runs`
re-measures and compares against the recorded number. If the producer measured
on a card that was quietly shared, the recorded number is inflated; the
validator re-measures on **whatever card it is given**, and either reproduces
the inflation (agrees, both wrong) or does not (refuses, and the reason looks
like the artefact rather than the neighbour). **Neither outcome names the
cause, and the card index is the one fact that would.**

Not fixed here because the producer and the validator now both refuse to
choose a card at all (`T19`), so the index is at least *deliberate* on both
sides. Recording it would make it *checkable*, which is a different property
and the one that closes this.

**STILL OPEN, and said so deliberately.** It is the one thing in this stage a
fixture cannot reach: it needs two measurements on two cards to show itself, and
a control on a login node has no card at all. Closing it on the strength of
*"the guards are deliberate now"* would be its own instance of the class below.

**It is the third member of a class with three known instances:**

| | the configuration | what it decides | what grades it |
|---|---|---|---|
| m2 | `--cuda-graph-max-bs 8` against concurrency 16 | decode exceeds the captured graph on essentially every step — a 4.7× latency spread | nothing; it was read as an image difference |
| m1 | `32.5` | a floor, recorded as prose and read back as a measurement | nothing; the real number was 33.58 at `tp_size: 1` |
| **m3** | **`E2E_MEASURE_GPU`** | **which card every number in `evidence/` came from** | **nothing; `measured_on` names the node** |

**A configuration that determines the number and is invisible to everything that
grades it**, and this is the purest of the three: `check_workset_runs`
re-measures **on the same card**, so when the card is the fault **it agrees for
exactly the reason the original was wrong**. The agreement is not evidence of
correctness; it is evidence that both readings share a premise nobody recorded.

That is *"a relative check cannot detect a fault both sides share"*, reached in
this stage from a third direction. The fix is unbuilt: **record the index in
`evidence.measured_on`**, so the two readings can be compared on their premise
and not only on their result.

### T33 — a mechanical reformat makes a diff unreviewable, so the semantic check moves before the commit

`json.dumps(indent=2)` re-serialises a whole schema, so a five-key change lands
as **1074 insertions / 240 deletions** and no reader can see what changed — 164
lines for one sentence is an ordinary instance. Verifying and disclosing
*afterwards* is the right check in the wrong order: an
after-the-fact semantic diff reassures the author and does nothing for the
reviewer, who has already been handed a diff they cannot read.

**Do not restore hand-formatting.** Canonical `json.dumps` form is
stable and reproducible, and the next programmatic edit would reformat it
again — restoring buys one reviewable diff at the price of the next one.

**The rule is about order and disclosure.** When an edit is programmatic:

1. run a **semantic** diff against `HEAD` *before* committing — walk both
   parsed documents and report added / removed / changed keys;
2. if the textual diff is reformat-heavy, **put that output in the commit
   message**. `1074 insertions` beside *"semantic diff: exactly five
   differences, all intended, listed below"* is reviewable; `1074 insertions`
   alone requires the reader to reconstruct it or to trust the author.

**Related:** `T31` — this is the same family. Naming the hazard after the fact
is not the same act as checking for it before.


**Extended — the same check, a failure it did not cover.**
T33 was about a diff being *unreadable*. This is a diff being *wider than its
author*: in a shared worktree, `git commit -- <pathspec>` names a **file**, and
the file contains whatever any teammate has uncommitted in it.

A commit whose message says *"four fixes to one section"* can carry a
concurrent rewrite of another section in the same file — a hundred insertions
where thirty were written. Nothing is lost and the content may be correctly
placed, but the message does not describe the commit, the other writer's commit
describes changes it does not contain, and each has signed off on lines they did
not write.

**The standing rule — *commit by pathspec, never `git add`* — does not defend
against this and was never meant to.** It stops you sweeping up *other files*. It
says nothing about *other lines inside the same file*, which is the hazard in any
document several writers share.

**The mitigation is T33's own check, one step later:** run
`git diff --stat <file>` *immediately before* committing and ask whether the
number matches what you wrote. A rewrite that replaces a section reporting
`65 insertions, 0 deletions` refutes itself, because a replacement cannot have
zero deletions.

**The check is *compare to what you did*, not *look at the stat*, and the
difference is not pedantry.** An **impossible** number only turns up in
replacement-shaped work. **Sweep an additive edit and the number is merely
bigger** — `+43` where you expected `+30` — and *plausible-but-larger* is not
impossible. **Nobody rejects a number that is only larger than they
remembered.**

So the rule needs the second operand. Stated as *"check the stat"* it invites the
substitution of `git status --short`, which prints `M` and not counts — and **a
check that reports presence where you needed magnitude is not the check you
thought you ran.**

### T34 — the environment record can outlive the container it names, and a downstream field repeats it as observation
*From two ends of the same artefact. Neither side's defect individually; the join
is unowned.*

**The producer half.** A bring-up record can say:

```
runtime.container   <container>
runtime.started_at  T+00:00
```

while `docker inspect` on the container of that name says:

```
Created       T+34:00
StartedAt     T+34:00
RestartCount  0
```

**The gap is the finding**: the record names a container that had not been
created when the record claims the run started, so the two cannot describe the
same process.

**`Created` equals `StartedAt` and restarts are zero**, so it was not restarted —
it is **a different container carrying the same name**, brought up after the
first was torn down. The record describes an instance that no longer exists, and
**every field in it still validates.** `check_environment` and `check_deploy_kit`
both pass it, correctly: nothing they check is wrong.

**The consumer half, and it is the sharper one.** `optimize_kernel`'s
`10_read_inputs.py` fills `premise.run_environment` from
`lib.load_environment()` — **m1's record, verbatim, with no observation
anywhere.** So a field named for *the environment m4 ran in* is m4 repeating m1's
claim, and a re-created container makes that claim wrong **while the handoff that
carries it validates.**

**What is missing is a join, not a field.** Each side is internally consistent:
m1's record is true about the container it was written about, and m4's premise
faithfully carries what it was given. **Nothing anywhere asks whether the
container the record names is the container that did the work** — and after
teardown the same lookup resolves to nothing at all, which is at least a loud
failure rather than a quiet one.

Same class as T27's `preflight.json` prose: **internally consistent, externally
stale, and the join unchecked.** Different in one way that matters — that one was
a conclusion contradicting data in the same file, and this one is two files that
each tell the truth.

**Not fixed, and the field is unchanged on both sides.**
`premise.run_environment`'s reader is m4's premise gate, so redefining it is a
contract decision rather than one stage's; and `runtime.started_at` belongs to
the shared schema. What exists today is a stopgap: the same `docker inspect` that
checks liveness also logs the observed `Id / Created / StartedAt / RestartCount`
beside the record's claim — **the only place in the flow where the container that
actually did the work identifies itself.** That is a log line, not a check.

**What would settle it, cheapest first:**

1. **`runtime.container_id`** in the record — the container's `Id`, not its name.
   A name is rebindable and an id is not, so a consumer can ask *"is the thing I
   am about to exec into the thing this record describes?"* and get an answer.
   One field, and it makes the join checkable for the first time.
2. **A consumer-side assertion** once (1) exists: `run_in_container.sh` already
   does the `docker inspect`; comparing the observed `Id` against the recorded
   one is one line and turns a log into a gate.
3. **Or drop the pretence** — rename `premise.run_environment` to something that
   says it is *the environment the producer declared*, not the one m4 ran in. A
   field whose name misdescribes its provenance is the thing that made this hard
   to see, and if (1) is not wanted then the honest fix is the name.

#### Sharpened by a second bring-up: `started_at` has no definition, so two runs measured two different things

The entry above blames container **re-creation**. A second real bring-up shows
that is one cause of the gap and not the cause, because the gap appears with no
re-creation at all:

| run | record `started_at` | container `Created` | `RestartCount` | gap |
|---|---|---|---|---|
| A | 09:03:51 | **09:37:18** | 0 | record is **34 min early** — it describes a container that no longer existed |
| B | **10:21:54** | 10:16:46 | 0 | record is **5 min late** — same container throughout, still up |

**Neither is wrong, because nothing says what the field means.**
`environment.schema.json`'s `runtime.started_at` is `{"type": "string"}` —
**required, and with no `description`.** So is `runtime.endpoint`. One agent
recorded a moment before the container it eventually used; the other recorded
something like *when the service became ready*, five minutes after its container
started. Both validate.

**This is T23's shape, in a second section of the same document.** T23 calls
`fixed.gpu_count` *"the one required field with no description"* — true of
`fixed`, and `runtime` has two more. A field that cannot be wrong cannot be a
measurement, and here it produces two incompatible readings without either being
a defect.

**So the fix is cheaper and more definite than T34's three options suggest:
define it.** One sentence in the schema, deciding between *when the container
started* and *when the deployment became ready* — and if it is the first, it
should be **read from `docker inspect` rather than written by the agent**, which
makes it a fact rather than a claim and removes case A entirely.
`runtime.endpoint` wants the same sentence: one run recording loopback
`http://127.0.0.1:8101` and another recording the routable
`http://<node ip>:8101` are not interchangeable, and a consumer on another host
can only use one of them.


### T35 — a sealed handoff from the old layout is not consumable by the definition that replaced it

`check_speedup_substantiated` looks for the measurement apparatus at
`scripts/workset` (`check.py`, `_APPARATUS`). The sealed stage-4
`kernel_optimization` in `cheat_for_mock/` carries its apparatus at
`scripts/kernel/` — `driver.py`, `graph_harness.py`, `measure_baseline.py`,
`sampler_softmax_kernel.py`. **So the one real stage-4 artefact this effort
owns cannot be fed to the validator that grades stage-4 artefacts.**

Not a regression. The sealed handoff is output from the five-separate-packages
layout, and `e2e-flow`'s definitions are written against the chained ones;
`mock_adapt.py` is the bridge and exists for exactly this. **Left as is.**

**Recorded because the assumption it breaks is easy to make and expensive to
discover.** Reaching for the sealed artefact as a ready-made fixture is the
obvious move when a validator needs a real input — it is the only stage-4
artefact that ever came off a cluster — and it fails at a path lookup rather
than at anything that names the layout change. It cost a wrong turn on
while looking for an input to exercise the container path.

**The open question, if anyone ever wants it:** whether sealed output of a
stage should stay *directly* consumable by that stage's next definition, or
whether an adapter is the intended and permanent shape. Nothing today depends
on the answer.

### T36 — a claim about *who owns this* never looks like a claim, so nobody tests it

**Six instances, all the same move: an assertion about *who* or *what* — an
owner, a boundary, a blocker, a row — stated before reading the thing that would
have answered it.**

- **Twice in one stage.** *"The mount question belongs to m1, m4 or the
  contract; I am not picking"* — while m1's sealed kits already mounted the
  answer, and m4's own `scratch_root` default already pointed at the third form.
  And *"blocked on m3's `--impl` contract"* — while the Definition's `baseline`,
  in a file already open for other reasons, carried `def sampler_softmax` beside
  `def run(*args, **kwargs)`: **one file satisfying both consumers,
  demonstrated, in the artefact.** There was no contract to arbitrate, only a
  shim not copied.
- **the package owner, four times.** Ownership inferred from a filename rather than
  read out of the manifest.

**Why the existing rule does not cover it.** *"Read the artefact, not the exit
code"* is about distrusting a **result** — and a result announces itself as
something to check. *This* class never produces a result. A boundary, an owner,
a blocker is a **framing**, asserted on the way to the work rather than
returned by it, and so it is the one claim in the room that nothing is pointed
at. **In all six the untested claim was the speaker's own.**

**It is the exact inverse of the falsification items beside it** (T-items on
gates validated only against null samples, captures verified only on the
success path, and the `stubkit` mode whose two halves agreed on every case).
There the discipline is *distrust a passing result*. Here it is **distrust your
own statement of the problem** — and the second is harder, because a passing
result at least arrives as evidence, while a framing arrives as context.

**The operational form, which is cheap:** before writing *"this belongs to X"*
or *"I am blocked on Y"*, open the artefact that would settle it. **It is one
read, and in all six cases here the answer was already on disk.** Naming an
owner is not research, and *"I am not guessing across the boundary"* is only
discipline when you have first checked whether the boundary exists.

**No code change. This is a habit item**, recorded because — unlike every other
entry here — **there is nothing to detect it with.**

---

### T37 — two producers disagreeing is what a schema-shaped defect looks like

*Renumbered from **T34**, which collided with an earlier item of the same number — six owners append to this file and two picked the same next integer. Commits and messages citing T34 for *this* item still resolve: the other T34 is a different subject and the two are not confusable by title.*

**From m4's finding that `public_symbol: sampler_softmax`
is defined nowhere in the file it names.**

`integration.public_symbol` was `required`, `type: string`, `minLength: 1`. An
operator whose engine code is a fragment inside a method **has no such symbol**,
and the schema had nowhere to say so — so both producers filled the field, and
filled it differently. `mock_adapt` wrote the Definition's own function name
six lines below an `entry_function` that names a method; `scaffold` wrote the
method qualname itself. **Neither producer was careless. The schema left them
nowhere to put the truth.**

**The generalisable part is the symptom, and its default reading is wrong.** A
required field with no representable *not applicable* does not announce itself
as a schema defect. It announces as **two producers disagreeing about one
field** — and that reads as carelessness, or as one producer being wrong, until
somebody checks the value against ground truth. Here that took reading the
file out of the image; m4 did it, and the premise became visible only because
one value was *provably* wrong.

**So: when two producers of one field disagree, ask what the field cannot
express before asking which producer is wrong.** The answer is sometimes that
both are, in the only way the schema allowed.

**Remedy, applied:** `substitution: module_symbol | call_site_fragment` with
`public_symbol` nullable and `if/then` binding them, plus `module_symbols`
recorded from the image so the claim is *checked* rather than asserted.
**What it does not do** is say how a
fragment-inside-a-method optimisation reaches the engine — that is M5.1.1, a
design question for the user, and the package can now state which case it is in
without being able to install the second.

### T38 — nothing in the graph orders m2's two lines; only the GPU count does

**Not blocking — today's configuration is correct. It is a
constraint on changing that configuration, and the failure mode is a wrong
number rather than an error.**

`profiling_mode_off` is the only throughput in this flow worth quoting, and
`profiling_mode_on` runs the same model on the same cards with a profiler
attached and CUDA graphs off. **They must not overlap.** Everyone, including me,
has been calling that *"sequential by construction (M2.5)"*. Measured — it is
not construction:

```
run_profiling_mode_off   froms: []   resources: {gpu: 8}
run_profiling_mode_on    froms: []   resources: {gpu: 8}
merge_profiling_evidence froms: [run_profiling_mode_off, run_profiling_mode_on]
```

**Both leaves depend on nothing.** The graph offers them together — the package owner
read exactly that in rung 0's log at 10:28, where both entered
`waiting_resource` within three lines of each other and only *then* ran one
after the other. **What serialises them is arithmetic: two tasks each asking for
8 GPUs do not fit in an 8-GPU node.**

**M2.5 says something different and true**: *a task that needs a service brings
it up itself* — that is why each line deploys and tears down its own engine. It
says nothing about two such tasks not running at once.

**Why this is a correctness constraint and not tidiness.** The two lines use
different port bands (`PORT_OFFSET=10`) and different run tags, so if they ever
did overlap **they would both come up cleanly**. Nothing would fail. The clean
line's throughput would simply be measured beside a profiler-attached load on
the same cards, and the number that m5's stock arm must reproduce (M5.1.3.1)
would be quietly wrong. **A guarantee whose only enforcement is a resource count
fails silently when the count changes.**

**And the count is already inconsistent with practice.** The declaration is
`gpu: 8`; the runs that have actually happened pass `--var tp=4` with
`gpu_devices=0,1,2,3`. So the declaration over-states what a line uses, and it
is that over-statement — not a rule — that is currently protecting the
measurement. Anyone who "corrects" `gpu: 8` to `gpu: 4` to match reality, or
runs on a node with more cards, removes the protection **and gets no error.**

**What would settle it:** declare the dependency where the guarantee lives —
`run_profiling_mode_on` with `froms: [run_profiling_mode_off]`. It costs
nothing today (they already run in that order) and it survives a change to the
resource pool. Not done here because the ordering is m2's stage but the subgraph
shape is the package owner's to approve, and because a change that alters graph
topology deserves its own rung rather than riding on a fix.

*Numbered T38 against a max of T37; `todo.md` currently has duplicate `T36`s and
the package owner is reconciling numbering, so treat this number as provisional.*

---

### T39 — `read_events.py` prints `message` and hides the attribute that holds the cause

**Not blocking. One line, and it is in checkpoint's file, so it
is routed rather than done here.**

`assets/lib/read_events.py` renders each event as its `message`. For
`output_absent` that message is
*"declared output … was never delivered"* — **which can be false** — while the
true reason sits in `attributes.seal_refused` and is not printed. So the tool
built for reading the event store reproduces the misdirection the store already
has — `output_absent` states a cause that is false.

**Two separate investigations have paid for this** — a stall study and a
replayed-kit investigation — and **both ended up running `cat` on a raw event
JSON** to find the same attribute.

**The precedent already exists.** `runprobe` prints **every** attribute of a
triggering event, which turns a lost reason into a one-command answer.
`read_events.py` has no equivalent.

**What would settle it:** print `seal_refused` and `detail` beside `message`
when present — or all non-empty attributes, which is what `runprobe` does and
needs no per-kind knowledge.

**Routed to checkpoint as the file's author**, with the diff, on the package owner's
rule that the author of a file cares most about its output being right and a
change landed by someone else and merely attributed is worse than one landed by
its owner.

---

### T40 — treat every probe as a control, because the slot decides whether you check the null

**Sharpening of `T31`.**

**The observation.** Several wrong turns in this package were probes that *could
not have succeeded* — a null that looked like an answer: an `abc` payload that
decoded to a non-command, a `-p $W` probe that dumped the whole process table,
and a positive control for `runlive.sh` using `exec -a`, which **`dash` does not
have** (`sh: 1: exec: -a: not found`), so the subject never existed.

**It only gets chased because it is labelled a control.** The same failing probe
sitting in the *measurement* slot gets its null written down and moved past —
which is what happens to empty `logs/`, `playground/` and `tmp/` in a task zone
read as "the body never ran", an inference that has to be thrown away once tasks
that certainly did run show the same empty directories.

**So the variable is not care, it is the slot.** Same person, same hour, same
diligence: a null in a control is *by definition* suspicious, and a null in a
measurement reads as data — and the split holds across stages: wrong turns in
the measurement slot, catches in the control slot.

**Two counts are easy to merge here, and the distinction matters to the entry's
own argument.** *"Four wrong turns, all in the measurement slot"* is about the
**slot**; it does not say four were dead probes. The others are **reasoning
errors from artefacts nobody opened** — endorsing a duration signature while
`evidence/performance.json` sits on disk saying otherwise.

**That is a different cause with a different cure, and conflating them
overstated this entry's evidence.** A dead probe is cured by *name the result
that would have proved the probe could speak*. A reasoning error from an
unopened artefact is not — it is cured by **opening the artefact**, which is
`T31`'s territory and `CONTRACT` §4.3's. **Four instances of one cause would be
stronger evidence than the record can support; two is what it can.** m3 asked
for the entry to be right rather than flattering, having been credited with a
tidier failure than they had.

**The operational form, and it is a test rather than an exhortation:**

> **Before believing a null, name the result that would have proved the probe
> could speak at all. If you cannot name one, you have not measured anything —
> you have observed that your instrument is quiet.**

m3 reports using it twice today by accident (the `abc` payload; the `-p $W` that
dumped the process table) and not at all on the other four.

**Instances already in the record:**

- empty zone `logs/`/`playground/` read as "the body did not run" — control:
  those dirs are empty for *every* task, including ones that sealed valid
  handoffs.
- `ppid=1` read as "orphaned by a kill" — control: `nohup sleep &` reaches
  `ppid=1` with nothing killed. Escalated before the control existed.
- `exec -a` positive control — the subject never existed.
- `readlink /proc/<pid>/exe || continue` — an unreadable `exe` dropped silently,
  so "cannot decide" and "nothing there" produce identical output.

**Sharpening — which control failure to spend the extra minute
on when you cannot afford both.**

> **A broken positive control wastes your own time. A broken negative control
> spends someone else's correctness.**

The rule above is about whether a probe *could have succeeded*. This is about
**direction, and who pays for the error**, which the rule does not say.

**A broken positive control refuses, and a refusal makes you look at it.** Three
schema fixtures of this shape — an unresolvable `$ref`, a bad `gpu_arch`, a
`kernel_id` pattern miss — each look exactly like a working probe, and each costs
the time of whoever built them.

**A broken negative control passes, and its output is a claim about somebody
else's artefact: *"your binding is toothless."*** A dict merge that re-adds the
key it was meant to strip prints `NOT CAUGHT` against a binding that is in fact
sound. Unless the probe is rebuilt before anything is concluded, that claim
travels as evidence — and **the action it invites is weakening a correct
contract.**

So the two failures are not the same size. One is self-limiting; the other
propagates, propagates *as evidence*, and reaches someone with no access to the
instrument that produced it. **When you can only afford to verify one control,
verify the negative one** — its false output is the one that leaves your hands.

**Not blocking.** It is a habit, not a defect, and the instances above are
already fixed or recorded. Filed because `T31` says naming a class is not
sweeping for it, and this is the sweep condition for that class.

### T41 — `/proc` is namespaced under `spur exec`, so PID-based attribution from a node lies

*Renumbered twice. Several writers append here and pick the same next integer,
and a renumber can collide the same way the original allocation did.*

*The rule is worth stating because it will recur:* **the entry with external
citations keeps its number.** A commit message and `CONTRACT.md` both cite T36
meaning *"a claim about who owns this never looks like a claim"*; nothing outside
this file cites T36 meaning `/proc`. Renumbering the cited one would break two
references to save one.

*T41 was free — the numbers otherwise run 1–47 — so this consumes the gap rather than
extending the range. Citations of **T28** or **T36** for the `/proc` subject still
resolve by title; the three are not confusable, which is the only reason a renumber is
survivable at all.*

*Reported independently twice, and renumbered by neither — correctly, since a
renumber touches other entries. That is the second numbering collision and the
mechanism is unchanged: **`todo.md` has no allocator**, and
`git status` cannot show a number someone else is about to use in an editor.*

**Not blocking. Recorded because ownership misattributions here are name-based,
and the obvious fix — attribute by PID instead — is broken in the one place
people will reach for it.**

**The control is the finding.** `docker top` reports a PID running inside a
container; from a `spur exec` shell on the same node, `/proc/<pid>` **does not
exist**:

```
docker top <container> -eo pid   ->  … <pid> …     (via the daemon)
[ -d /proc/<pid> ]               ->  NO            (via /proc)
```

So `spur exec` puts you in a PID namespace that cannot see the host's
processes. **A `/proc` miss there means "not visible from here", not "not
running"** — and the two are indistinguishable without a control.

The opposite conclusion is one sentence away: `rocm-smi --showpids` lists a PID,
it is absent from `/proc`, and the obvious reading is *"stale, does not exist"*.
What refutes that is checking whether a **known-live** PID is visible either — it
is not.

**What is reliable from `spur exec`, and what is not:**

| reading | reliable? | why |
|---|---|---|
| `docker ps` / `docker inspect` / `docker top` | **yes** | goes to the host daemon, which is outside the namespace |
| `rocm-smi --showmemuse` | **yes** | reads the devices |
| `/proc/<pid>/*` | **no** | namespaced; host PIDs are absent |
| `rocm-smi --showpids` | **not as an inventory** | listed a single row holding **0 bytes** while all eight cards read 90 % |

**Why it matters beyond tidiness.** A kill decision made from a PID list taken
this way would be operating on a table that is both incomplete and
unfalsifiable. The kill actually performed on 006 was decided through
`docker inspect` and its mounts and verified by cards going 90 % → 0 % and the
container list emptying — both daemon-side, so sound by this rule; that was not
luck, but it was not checked against this rule either, because the rule did not
exist yet.

**One consumer to check:** any liveness probe reading `/proc/<pid>/cwd` is
correct **on the login node** — same namespace as the run — and would be
silently wrong if moved onto a node. That is a real move somebody will make,
because the node is where the containers are.

**Not fixed, because there is nothing to fix** — this is a property of the
transport. It is a rule about which instrument answers which question, and the
generalisation is the one this package keeps relearning: **when a reading can
only come back one way, it is not a reading.** Establish that the instrument can
see a positive before believing a negative.

### T42 — a validation zone is not a normal process environment, and anything read from it that names a host path is suspect

**A validator does not run in the environment you think it does.** The zone
rewrites variables a body would take for granted, and every one of them named a
host path that turned out to be somewhere else. Four instances,
each found only by something failing:

| variable | what the zone makes it | what it cost |
|---|---|---|
| `HOME` | `<zone>/home` | m4's ephemeral container mounted a *subdirectory of the zone* instead of the run root, and rung 0 died on `./run_performance.sh: No such file or directory` — a container that came up perfectly and could not see its own script |
| `PATH` | `/usr/bin:/bin` | `spur` lives in `/usr/local/bin`, so the transport is simply absent |
| `SPUR_CONTROLLER_ADDR` | **unset** | m3 lost three non-reproductions to it, because their own login shell had it |
| `TMPDIR` | `<zone>/tmp` | forced by the zone as an invariant (`validator/environment.py`), so a producer's `env` cannot reach it — and on this cluster's NFS a `TMPDIR` there SIGSEGVs every HIP kernel launch |

**The general rule, which is the point of the entry:** *anything read from a
validation zone's environment that names a host path is suspect.* Not "`HOME` is
redefined" — that is one instance and covers nothing else. The four above were
found one at a time, by four different people, each paying separately, because
each was treated as a fact about that variable rather than as a fact about the
zone.

**What follows operationally.** A validator that needs a host path must take it
from an **argument** (`--var`), from the **artefact**, or by **asking the node** —
never from its own environment. Where a variable is unavoidable, it is passed in
explicitly: `transport_path` and `transport_env` exist for exactly this, and are
the shape to copy.

**And the inverse, which bit hardest:** a path that *is* correct here may be
correct nowhere else. `$HOME` was a perfectly good value — it just did not name
what the reader assumed. **A wrong value announces itself; a right value that
answers a different question does not.**

### T43 — the artefact is honest about provenance and dishonest about meaning

**The baseline first, because it is the reason the class exists rather than an
example of it.** Swept all eight schemas under `assets/schemas/`:
**four express no cross-field constraint at all** — `bench_result`,
`environment`, `kernel_table`, and `integration_report`, the last at 786 lines
with zero. The other four carry thirteen `if/then` between them. So the package
has **eight schemas that check fields and four that check meaning**, and the
instances below are what that ratio produces.

**Three instances, one shape.** A field carries something
faithfully — the copy is exact, the producer did nothing wrong — and the field's
*name* says it is something else. Every field validates; the document is false.

- **`premise.run_environment`** — `10_read_inputs.py` fills it with
  `lib.load_environment()`: m1's `deploy_kit` record, verbatim, no observation
  anywhere. A field named for *the environment m4 ran in* is m4 repeating m1's
  claim. m1 then measured a record whose `started_at` was 09:03:51 against a
  container reporting `Created == StartedAt == 09:37:18` with `RestartCount: 0`
  — **a different container wearing the same name, every field valid** (T34).
- **m1's *"by construction"*** in T27 item 1 — the kit was said to make record
  and runtime agree by construction; the pin is an **environment variable**, and
  `docker exec -e HIP_VISIBLE_DEVICES=…` overrides it. Convention described as
  enforcement. m1 corrected their own entry.
- **`protocol.timing: event`** — declared in `workset.yaml`, read by nothing;
  `run_performance.sh` is `perf_counter` around a
  `torch.cuda.synchronize()`. Copied into `kernel_optimization` by
  `check_optimization_shape`'s field-for-field comparison.

**Why one entry and not three.** The *fix* differs every time — observe beside
the claim, change the mechanism, change the word — and the *failure* does not.
That is exactly when a class beats its instances.

**Why it evades the existing rules.** *Read the artefact, not the exit code*
governs results, and these are not results. `items_schema` cannot see it: the
value is well-formed and of the right type. And a reviewer reading the producer
sees a faithful copy and correctly approves it — **the defect is not visible
from either end, only in the join between the name and the source.**

**The diagnostic question,** cheap enough to ask of any field: *what would have
to be true for this name to be accurate, and did anything check it?* For
`run_environment` the answer was "somebody observed the container", and nothing
had.

**The fourth instance is a *pair*, and it generalises past the other three.**
`integration.substitution` and `apply_mode` are two fields with legal and
illegal combinations — `call_site_fragment` says the edit lives inside an
existing function, `overlay_files` says replace the whole file — and **no code
anywhere knows they are a pair.** Each validates alone: both are strings from
their enum. So m4 can emit an impossible `apply` block in silence, and the first thing to
notice is m5, two stages later.

**A constraint between two fields is invisible to a schema that validates them
separately.** That is the sharper form and it is not covered by the three
instances above, where a single field's name misdescribes a single field's
source. Here every field is individually honest and the *combination* is the
lie. **Some of these are expressible in JSON Schema and some are not, and the
difference decides where the rule belongs.** `if/then` relates a value to a
*constant*: `substitution: module_symbol` implies `public_symbol` is a
non-empty string, which m3 bound exactly that way. It cannot relate a value to a
*sibling's value* — `len(gpu_devices) <= gpu_count` needs `$data`, an Ajv
extension absent from draft 2020-12, and `spec_loader/validate.py` runs a stock
`Draft202012Validator`. **So "nobody looked" is the wrong diagnosis for half of them**: the constraint
is inexpressible in the schema and lives instead in
`deploy_kit.layout.yaml`'s invariants, with a gate fault behind it.

**The rule that generalises:** reading an unexpressed constraint as an oversight
is **cheap to say and expensive to be wrong about.** It converts a deliberate
placement into a defect, and the fix it implies — *move it into the schema* — is
one the schema cannot execute. **The evidence that somebody looked is not in the
schema, which is exactly why the schema reads as if nobody did.** An absent
constraint is silent about whether it was considered. So: check before claiming,
or claim only about the case you checked.

**And the component that could have caught all of them is the one nobody
instruments.** When a constraint lives *between* two artefacts, the only code
that can see it is the comparison — and **comparisons are written to decide,
not to explain.** `_same` reported that two values differ and stopped, which is
exactly enough to know something is wrong and not enough to know what: for
`entry_function` it said `''` differed from `Sampler.forward` without saying the
producer had read a different field, and for `protocol.timing` it said `'event'`
differed from `'wall_clock_sync'` without saying one end permitted five values
and the other one. Neither end's narrowing is wrong in isolation and no review
of either would catch it; **the defect exists only in the relation, and the
relation is what the comparison refuses to print.**

Fixed rather than only recorded: `_same` now names the differing *keys* instead
of dumping two dicts, and prints both ends' declared vocabularies when it can
resolve them, refusing when the leaf name is ambiguous rather than guessing.
Verdict unchanged; a report change, not a gate change.

**The sharpest sub-case: a constraint enforced for one carrier of a document
that fifteen carry.** CONTRACT §2 puts the *same* `environment.yaml` in all
fifteen kinds. `count_of: fixed.gpu_devices / at_most: fixed.gpu_count` is
enforced in `deploy_kit`'s layout — **for `deploy_kit` only**. Meanwhile
`check_environment` is wired at fourteen sites across all five modules, loads
that same record, and mentions `gpu_devices` **zero times**. So a
`profiling_evidence` or an `operator_workset` may carry `gpu_count: 4` beside
eight devices and validate cleanly.

**That is worse than an unexpressed constraint and reads better.** The rule
exists, is written down, has a fault number, and is *demonstrably enforced* — so
a reader who finds it reasonably concludes the document is checked. It is
checked in one of fifteen places it travels. **A rule enforced at one carrier of
a shared document is indistinguishable, from the artefact, from a rule enforced
everywhere.**

**Written and switched off** at
`check_optimization_shape.validator/check.py:_substitution_matches_apply_mode`,
behind `_ENFORCE_SUBSTITUTION_PAIR`. Left inert deliberately so rung 0 can reach
m5 and exercise seven validators that have never seen a graph-produced artefact;
a gate that has never fired is not a gate, and that argument applies to theirs
before it applies to this one. Demonstrated firing against the real artefact
before being switched off.

**Cross-reference: checkpoint's table is this disease from the other end** — *an
instrument reads a real thing and answers a different question*, eleven
instances. Theirs is the instrument form, this is the artefact form. Neither is
a superset; they should point at each other rather than become two vocabularies
for one failure.

### T44 — `rebuild` is reachable in one schema and emittable by none

`integration_report.patch.apply_mode` accepts `['overlay_files', 'rebuild']`.
Upstream, `workset`'s `integration.apply_mode` is `['overlay_files']` and
`kernel_optimization`'s `apply.apply_mode` is `const: "overlay_files"`. **So no
producer in the package can put `rebuild` into an artefact m5 would read**, and
the branch is dead.

Not a bug and nothing is broken by it — recorded because **a reader has no way
to tell a deferred option from a live one.** `kernel_optimization`'s own
`const` says the alternative is *"deferred rather than unimagined"* and cites
M5.3; m5's enum says nothing, so the same decision reads as available there.

**One line, not three owners' attention** (leader's call). The fix,
whenever the mechanism question is settled, is for the three to agree — either
`rebuild` becomes emittable or it stops being acceptable.

**The same defect facing the other way: `must_preserve`, emitted and read by
none.** `60_write_handoff.py` populates it from the workset's
`integration` — signature, invariants, `requires_restart`, `build_step` —
`kernel_optimization.schema.json` documents it, two samples carry it, and
**no consumer anywhere reads it.** Found when the mock's empty
`must_preserve` looked like the `dtypes` omission that had cost a rung-0
attempt; m5 grepped their whole stage and there is nothing to feed.

**Deliberately not populated in the mock** (leader's call): consistency with a
producer nobody reads is not worth a rung-0 attempt.

**The pair is the point.** `rebuild` is *accepted by a reader no producer can
satisfy*; `must_preserve` is *produced for a reader that does not exist*. Both
validate, both are documented, and **neither can fail** — so a reader finding
either one reasonably concludes it is load-bearing. The tell is the same in both
directions: **trace a field to a consumer before believing it does anything**,
and a field with a description but no consumer is exactly as inert as one with a
consumer and no producer.


**Third instance, and it is the one that cost something: `KFO_KERNELFORGE_REPO`.**
Declared in this package (`m4_kernel_opt.yaml`) **and** in the kernel-optimisation stage,
and read by no body in either. The old readme calls it *"a KernelForge checkout,
already `pip install -e`'d"* — so the environment was always prepared **out of
band**, and the variable only ever told a *reader* where it was.

It cost something because it looked like the answer. When rung 4 turned out to be
blocked on `forge-loop` being installed nowhere, `KFO_KERNELFORGE_REPO` is the
first thing anyone finds, and it reads as a mechanism that has come unwired. It
never was one. **There is no prior art to restore** — whatever gets built is new
work, and the demo that produced this effort's proven assets ran on a machine
somebody had prepared by hand, which nothing in the package records.

**The class, which covers all three:** *a variable that was never a mechanism,
only a note to a reader spelled as configuration.* And the
tell is the same in both directions — `rebuild` has a reader and no producer,
`must_preserve` and this one have producers or declarations and no reader:

> **You cannot tell from a declaration whether anything reads it, and
> `grep -rn "$NAME"` across the bodies is a two-second check that nobody runs
> because a declaration looks like plumbing.**

m3 ran it against both of their own `E2E_MEASURE_*` declarations while they were
in there; both have consumers. That is the whole remedy and it is cheaper than
the entry describing it.

### T45 — a comment that asserts a wiring gap sends the next reader to change code, when only a number was missing

**Found by measuring instead of reading.**

`check_identity_resolved.check.py` said `min_resolve_ratio` *"is not passed
by `steps/m3_analysis.yaml`, so the arm above cannot refuse in this package as
configured."* The yaml passes it —
`min_resolve_ratio: '${min_resolve_ratio:-0.0}'` — so the arm is fully
parameterised and `--var min_resolve_ratio=0.8` reaches it with no edit at all.
Measured in both directions on one operator with an honest unresolved entry:
`0.0` passes, `0.8` refuses with `resolve_ratio 0.500 is below the floor 0.8`.

**Two distinct costs, and the second is the reason this is a todo and not a
typo.**

1. It is wrong about the artefact. A reader who wants the bar enforced concludes
   they must edit yaml and code, when they need only pass a var.
2. **It made a defect look bigger than it was, in the direction of a finding.**
   Both of us had it filed as an instance of the package owner's "validators that grade
   nothing" sweep. It is not one: `0.0` is a defended default with a written
   argument, and the arm is live. The true defect was one stale sentence. A
   stale comment does not merely fail to inform — it *manufactures* the finding
   the sweep is looking for, and a sweep that trusts comments will report it.

**And the obvious first fix is itself unreachable.** An "unset — this arm did
not grade" wording behind `if raw is None or raw == ""` cannot execute here,
because the yaml *does* pass the arg and `raw` is never absent: the default path
still prints `floor 0.0`. Checking that the new wording is correct is not
checking that it is reachable — the same omission in a different costume, since a
comment claiming the arg is unpassed is exactly what makes an `is None` branch
look sufficient. Verified by driving `_check` over a real `operator_identity`
with all three arg values and reading the note each produces.

**The rule.** A comment that describes *wiring* — what is passed, what is
reachable, what cannot fire — is a claim about a file other than the one it sits
in, and it goes stale silently because nothing loads it. Either grep the file
you are describing at the moment you write the sentence, or describe the
behaviour of the code in front of you and let the reader look up the wiring.
The second is usually the better sentence anyway: *"a floor of zero grades
nothing"* is true wherever the value comes from.

**The general form of the second half is `T46`** — *a fix verified correct is
not a fix verified reached*. Kept as a pointer rather than a second copy.

### T46 — a fix verified correct is not a fix verified reached

**Two instances in the same stretch of work**, both a change checked for being
*right* and shipped without being checked for being *executed*. Those are
different questions, and only the second needs the surrounding wiring in view.

**The silent one.** A validator note that printed `floor 0.0`, replaced by one
saying *"this arm did not grade"* and guarded with `if raw is None or raw == ""`.
**The yaml always passes the argument, so `raw` is never absent** — the branch
cannot execute, the default path goes on printing the exact string the fix
removed, and the file's comment now claims the fix was needed.

**The loud one.** A stubkit entrypoint referencing the shell's `$IMPL` from
inside the Python emitter — `NameError` on the first candidate run, immediate.

**The asymmetry is the entry.** An entry carrying only the second instance would
teach that this class announces itself. **It does not; that was luck.** The
same defect is a crash when the unreached code is malformed and a permanent
silent wrong answer when it is well-formed — and *well-formed* is the normal
case, because the code was written carefully and only the reachability was
assumed. **The better the fix, the quieter the failure.**

**What makes it actionable rather than cautionary: the step is already
performed intermittently, without being named.** Two examples of it passing for
ordinary care —

- the `substitution` × `apply_mode` gate **armed, run against a real artefact to
  watch it refuse, then switched off**. That step catches a `NameError` waiting
  in the disabled path: the call site has neither `packup` nor `notes` in scope,
  so the gate **could not have run when enabled** — a gate that cannot fire when
  switched on, shipped as *written and ready*;
- `_same`'s vocabulary reporting exercised by **monkeypatching the older enums**,
  because a fix elsewhere had made the divergence branch unreachable on current
  schemas. Shipping it unexercised is the alternative.

**So the rule is not new behaviour, it is a name for something already done half
the time.** The half where it is skipped is the half where the change looks too
small to need it — a guard, a message, a default.

**The check, and it is one question:** *what would have to be true for this new
line to run, and has that happened?* If the answer is "nothing, it runs on every
call", say so. If it is "a variable would have to be absent" or "an enum would
have to differ", go and make that true once, on purpose, before shipping.

**Distinct from `T40`**, which is about probes and whether a null in a control
slot gets questioned. This is about **fixes**: the code is not an instrument, the
result is not being read, and nothing about the slot prompts suspicion.

### T47 — the pathspec window is irreducible, so the check has to be after the commit

**The mechanism is `CONTRACT` §8a, not this entry.** `git commit -- <path>`
takes the *working tree*, so it also takes a concurrent writer's uncommitted
edits to that same path. Found independently three times, which is stronger
evidence than any one report.

**What is new here is that no pre-commit check can close it, and that was
established by accident.** Both earlier reports assumed a pre-commit check was
sufficient and argued about *which one*.

**Measured.** The obvious prescription is *read `git diff -- <path>`
immediately before committing*. Run against a one-hunk addition to this file, the
check passes, the commit is issued seconds later, and it returns **`no changes
added to commit`** — a concurrent writer committed `todo.md` in the interval and
swept the addition into their own commit. **The remedy is falsified by the act of
committing it.**

**The window is between the check and the commit.** `git commit -- <path>` reads
the tree at commit time and there is no atomic verify-then-commit for a path.
**Checking earlier moves the window; checking harder does not shrink it.** So
every pre-commit check — `git status`, `git diff`, any of them — is a mitigation
that narrows the odds, and none is a fix. That is a property of git, not of our
discipline.

**The post-check, and the framing matters.** §8a records `git show --numstat
HEAD` justified as *proving nothing was removed*. Zero deletions does not catch
this: a swept-up entry is an **addition**. The check is **the commit's size
against the size of what you wrote** — if the commit is bigger, someone else's
work is inside it and its message is now wrong about its own contents.

**And a pre-commit check can fail by firing.** The checkpoint writer ran
`git status --porcelain -- todo.md`, saw ` M`, and proceeded — correctly reading
a dirty file as *their own* edit, because they had one in flight. **A check
whose alarm is indistinguishable from the expected condition is not a check**,
which is why the pre-commit half cannot be made load-bearing by choosing a
better command.

**Do not amend.** Every instance so far was harmless and each was resolved by
disclosure. Rewriting shared history to fix an attribution is a larger hazard
than the attribution — the standing rule against `--amend`, reached
independently from the other direction at 04:09.

---

### T48 — the control that caught it is invisible because it worked, so remedy-selection reaches for the prescribed check instead

**Deliberately separate from `T47`: that entry is about a git mechanism, this is
about how a remedy gets chosen after an incident.**

**The shape.** A commit sweeps up 39 lines of somebody else's entry. What catches
it is a **post-commit** check that had been running all along without being
thought of as a control: `git show --numstat HEAD` printing **`66 0`** against
the 27 lines actually written, and that discrepancy is the only reason anyone
looks.

**The near-regression.** The obvious remedy to write down is **`git diff --
<path>` before committing** — a check already falsified in T47, by being run
correctly and losing the race anyway. That is one step from replacing a control
that *demonstrably just caught the bug* with one already known not to close the
window.

**Why the working control was not in reach.** It had never produced a story. A
check that quietly succeeds every time generates no incident, no message, no
entry — **so when you go looking for "what should I do differently", the
effective control is the one thing not in the search space.** The prescribed
check was in reach because it had just been written down; the working check was
invisible *because* it worked.

**This is distinct from `T40`.** T40 is about whether a probe *could have
succeeded* — the null in the measurement slot. This is about **which control you
credit afterwards**, and it bites even when every probe was sound: the failure
is in remedy-selection, not in measurement.

**The rule:**

> **After an incident, before adopting a remedy, name what actually caught it —
> and check whether that is already in the routine.** If it is something already
> being done, the remedy is to make it explicit and load-bearing, not to add a
> new check beside it.

**Two more instances of the same shape:**

- referencing findings **by identifier rather than by value**, recorded as
  *"partly luck"* when it is a habit — and structurally immune to
  mis-attribution, which makes it the one rule that **prevents rather than
  detects**;
- `numstat` as the load-bearing check above, run for a long time and never once
  described as a control.

**The tell: a practice that works is filed as luck.** It produces no evidence of
itself, so its owner is the last person able to see it. **That makes this one of
the few classes here an outside reader finds more easily than the author** — and
unlike the tool defects, it fires when nothing is broken.

**Not blocking.** No defect; a reasoning habit with one near-miss recorded.

### T49 — the verdict is right and the message says why it is right, wrongly

**Named as a class by m5 after the third instance in one afternoon.
Six now, across five owners.** The sixth is the one that matters most for this
entry's own reach: **T49 occurred inside the failure ledger itself**, where the
recorded *reason* for four failures was a `note:` line the log had relabelled
`PROBLEM:`. **Twice what the first summary of it said** — checkpoint reported two
rows, another owner's audit found four. Recorded here
rather than in `T43` because `T43` is about a *field* copied faithfully and
understood by nothing; this is about a *message* that describes a narrower check
than the sentence it prints.

| where | the message claims | what was actually checked |
|---|---|---|
| `apply.py` (m5) | `workset declares None in [...]` | a `call_site_fragment` operator — `public_symbol: null` through `!r` |
| `check_packup_shape` (m5) | `holds 0 non-empty file(s)` | one file, in a subdirectory `iterdir` did not descend into |
| `check_workset_shape` (m3) | `not defined at module level` | not a `def`/`class` at module level — assignments are invisible to `identify.py` |
| `check_command_parses` (leader) | `PROBLEM: items/command bash -n clean` | a *passing* line, routed into `write_report`'s problems slot unconditionally |
| `validator.failures` ledger | a quoted `PROBLEM:` line as the failure reason | **four rows** were `note:` lines the log had relabelled — `check_bench_result`, `check_measurement_order`, `check_trace_coverage`, `check_no_regression` |
| `run_in_container.sh` (m4) | *"m1's bring-up has been torn down"* | the node was never reached — **a transport failure wearing a teardown's clothes, and it named a specific person** |
| `run_in_container.sh` (m4) | a blank where the container list goes | *none* — an empty diagnostic is indistinguishable from one that failed to look |
| `check_speedup_substantiated` (m4) | `exited 127: l so no reader has to infer…` | `stderr.strip()[-400:]` — a slice by character, splicing a comment fragment into the error |
| `m3_analysis.yaml` (m3) | *"the body falls back to card 4 via `${E2E_MEASURE_GPU:=4}`"* | `measure_in_container.sh` **refuses** — prints the node's live `rocm-smi` and exits 1. There is no fallback |

**Every one gives the correct verdict.** That is what makes it the hardest kind
to notice: nothing downstream is wrong, no run fails that should pass, and the
only defect is in the sentence a human reads. A reader who checks the claim and
finds it false stops believing the *next* message from the same validator, which
is the actual cost and it is unbounded.

**Three of the four are found by someone reading the message rather than the
code**, and none by its author — one of them only because somebody counting
`PROBLEM:` lines for an unrelated reason noticed seven of them under a `passed`
heading. **The author knows what the check does, so the author reads the message
as a summary of it** — the same blindness as T48's invisible working practice,
pointed at prose.

**The tell, and it is cheap:** read the message *as if you did not write the
check*, then ask what would have to be true for it to be exactly right. In all
four the answer was a strictly broader condition than the code tests.
`check_workset_shape`'s is the sharpest — *"not defined at module level"* is
true of a module-level assignment, and the check cannot see one, so the message
is a correct English sentence about a case the code would get wrong.

**Eight, and the eighth is the one that changes the entry.** One instance is
written by an author who had **filed the class themselves** — T45, about a
comment describing another file that goes stale silently. Every other instance is
someone who did not know the class existed. **Knowing it did not help**, which is
worth more than the count going from seven to eight, and it is why this entry
ends with a detector rather than with advice.

**And the count only reaches eight once the class is named**, which is itself the
point: **each author can see the others' instances and not their own, until
somebody else reads it.**

**The asymmetry that decides the fix, and it is not obvious:** *narrowing the
message cannot break a consumer; widening the field can.* `check_workset_shape`
presence-checks against `module_symbols`, so adding assignments to it makes some
previously-refused worksets pass — correct if they were correct, a **silent
loosening** if not. Both of m5's instances were resolved by narrowing the
message, which is evidence about the base rate and not about any particular case.

**Not blocking, and not a code sweep.** Fixing the four is done or routed;
the entry exists because the fifth will be written by whoever writes the next
refusal, and the cost lands on a reader rather than on a run.

**The sibling class, recorded because one afternoon produced four of
it: manufactured provenance.** T49 is a *true* verdict with a false reason. This
is a *false record* that reads as a true one — and it is worse, because the check
you would run against it passes:

| | what was manufactured |
|---|---|
| `entry_function_line: 183` | a mock constant, **corroborated inside its own `resolution_evidence` prose** — field and evidence agreed, both wrong |
| `32.5 ms` | an estimate written in prose, read back later as a measurement, then the calibration constant of a `strong` validator |
| `runtime.replayed_from` | **overwritten by the tool that writes it**, collapsing `corpus → run X → here` into `run X → here` |
| `report["impl_path"]` | `args.impl` echoed at parse time; reads identically whether the file was exec'd, imported, shadowed, or never opened |

**The rule that falls out**, m3's, from declining to backfill the sealed corpus:
**never write a record nobody made.** A backfilled `from_identity` or
`environment.yaml` would be manufactured today and *indistinguishable from a real
one* — which is CONTRACT §5.3 pointed at provenance rather than at facts: *"A
mock may obtain a real fact by a route the producer does not use. It may not
assert a fact the producer does not have."*

**Why it is a different entry and not a fifth instance of T49.** T49's damage is
that a reader is sent the wrong way and can recover by checking. Here the
artefact *survives* checking: *"was this measured?"* returns yes, and only
*"measured on **what**?"* catches it. Two of the four are only findable because
the author volunteered the history of their own instrument, which no reader could
have demanded.

### T50 — an instruction verified correct against a world that does not exist yet

**Three owners wrote one on the same afternoon; two were unexecutable the moment
they were committed, and the third survives by where the ladder happens to be.**
Routed to the package owner by m4 rather than written by any of them, because T49 had
just been three people one message from three copies of one class.

| section | reads from | state when committed |
|---|---|---|
| m5, rung 5 | *"rung 4's run"* | **no rung-4 run exists** — and cannot, `forge-loop` is installed on neither the node host nor the image |
| m4, rung 4 | *"rung 3's run"* | **no rung-3 run exists** — identical defect, written an hour later |
| m2, rung 2 | *"rung 1's run"* | **executable — because rung 1 is the rung in flight.** Safe by position, not by construction |

**It is not T46.** T46 is *a fix verified correct is not a fix verified reached* —
code, correct, unreached. This is **prose**: an instruction that reads perfectly,
survives every review, and **cannot be executed at all**, because the artefact it
names has never been produced. T46's instances fail when something runs; these
fail when someone tries to start.

**The tell is that it reads better than a correct instruction would.** *"Read the
image and `tp_size` from rung 3's run before typing anything"* is more specific,
more careful and more actionable than *"read them from the nearest run that
recorded them"* — and specificity is what made it wrong. **A cross-reference to a
future artefact is indistinguishable from a cross-reference to a present one, in
the only place anyone looks: the sentence.**

**The fix is a generalisation rather than a correction:** the
values are m1's to mint, so **every rung from 1 upward records them in the same
`items/codes/environment.yaml`** — name the *nearest run that has them*, not a
rung. Both m5 and m4 repeated the caveat rather than cross-referencing it, and it
belongs here too: **this is not a licence to skip the ladder.** Rung 5 still waits
on rung 4, which waits on rung 3. Without that sentence the amendment reads as a
way around the blocker, and that misreading is available in every section.

**What makes it worth an entry rather than three fixes:** m4 only saw their own
after m5 described theirs, and neither would have looked at m2's. The base rate
is three of three, and **the one that works does so for a reason its author did
not arrange.**

**Not blocking.** All three sections are corrected or correct.

### T51 — at promotion only the brief travels, and the brief drifts in both directions

**Promotion swaps a program for a conversation.** A `kind: ai` task never runs
`entry.sh`, so everything the program knew is lost unless the brief says it —
and `agent/runner.py` closes the cheap escape: *"an env var cannot instruct
an agent. A conversation is not a process reading `os.environ`."* An `env:` block
makes a value **reachable**; only the brief makes it **used**.

So the brief is the whole interface at promotion, and it drifts two ways.

**Direction 1 — the brief omits what the program knew (m3).** `workset_builder`'s
readme is 330 lines and ten STEPS with **zero** occurrences of
`measure_in_container.sh`, `E2E_MEASURE_GPU`, or `rocm-smi`. STEP 7/8 say
`cd "$WS" && ./run_correctness.sh` **directly, on whatever host the agent is on**,
and those hosts have no torch. Everything that knows better is on the branch the
promotion turns off — `entry.sh`. **The mock is not a weaker version of the
real path here; it is the only version carrying the knowledge**, so promoting m3
removes the container step and the card check together, silently. First symptom:
a torch import error, or worse, a plausible number measured on a card nobody
chose.

**Direction 2 — the brief contradicts what the program does, and this is the one
that survives review.** A brief saying the wrapper *"execs into the recorded
container and never starts or removes one — CONTRACT §5.2 is absolute about
that"* goes false the moment the wrapper gains an ephemeral-container path: with
no running container it starts one, `--rm`, trap-removed.

**A gap reads as something to fill; a false prohibition reads as a constraint to
respect.** An agent handed a mock-chain record whose container nobody brought up
would have concluded the wrapper could not help **and stopped — correctly,
according to its brief.** And a confident sentence about a constraint is exactly
what a reviewer nods at, which is why direction 2 outlives direction 1.

**Why one entry.** Same seam, opposite signs, one cause: **the code moves and the
prose does not, in the single artefact that is the entire interface at
promotion.** Splitting it would file the symptom twice.

**The constructive half is m4's and it is the only preventive thing here:** they
wrote the mirroring requirement into the **arming instruction** rather than into
a note beside it. m3's reading of why that is different — *a note says what
someone should have done; an instruction that cannot be followed without doing it
first is a different object.* Where a brief must stay in step with code, put the
dependency in the step the reader has to execute, not in prose next to it.

**Neither is found by reading the brief.** One comes from listing what a later
run reads; the other from being asked the question the first raised. **Both
authors had read their own briefs many times.**

**Blocking** until `workset_builder`'s brief is written.

### T52 — no `df` reachable from `spur exec` predicts whether a `docker load` fits

**Measured, and it overturns the refusal it looks like it justifies.**

Declining to `docker load` a 28.5 GB engine-image backup because every node
looked too small: 006 70 G free, 047 87 G, 217 57 G, against an image whose
`docker images` SIZE reads **110GB**. m2 ran the load anyway, on 047, and it
succeeded in 2m16s.

**047's disk did not move.** Before the load and after it, by two independent
routes:

```
exec namespace          df -h /               123G  37G used  87G free
through the daemon      docker run -v /var/lib/docker:/hd:ro … df -h /hd
                                              123G  37G used  87G free
docker system df        Images 47   1.19TB
```

1.19 TB of images cannot sit in 37 G, and **loading 28.5 GB changed nothing** —
so whatever `/var/lib/docker` resolves to from inside `spur exec`, by *either*
route, is not where images live. `docker info` says `Docker Root Dir:
/var/lib/docker`, `Storage Driver: overlayfs`, and that is a dead end; mounting
host `/` to look for the real one is refused by `spur-authz` (`denied [B1]`).

**Both instruments available in there are unusable for this question**, in
opposite directions: `df` reads a filesystem the operation does not touch, and
`docker system df` reports **logical** sizes that double-count layers shared
between images (which is also why one image reads 110GB when its restored cost
is nearer the 28.5 GB tar).

**Until someone finds the real path: do not gate a `docker load` on free space.
Try it.** A load that runs out of space fails and leaves partial blobs; that is
recoverable-by-retry, whereas *not loading* on a number that measures the wrong
device is how a verification gets skipped for no reason.

**The reusable part is how the wrong answer survives a check.** Checking whether
the instrument is pointed at the right device — mounting the host's docker root
through the daemon specifically to rule out the exec namespace — returns the same
number, which reads as confirmation. **Two routes to one wrong answer is not
corroboration when both routes share the assumption under test.** The only
measurement that settles it is the before/after across the load itself, which
requires doing the thing the number was being used to avoid.

### T53 — the var table's rung-2 advice for `expect_ranks` is right only at `tp=8`

**In the shared var table rather than in this package.**

```
| `expect_ranks` | **2** | **omit it** (defaults to 8), or track `--var tp` |
```

The second half is correct and the first half is a trap at any `tp` but 8.
`m2_profiling.yaml` is `expect_ranks: '${expect_ranks:-8}'` and it
deliberately does **not** track `tp` (a `${}` default may contain no `}`, so
`${expect_ranks:-${tp:-8}}` is not spellable — the yaml says so at :114-116).
So at rung 2 with the `tp=4` we are actually running, omitting it yields 8 and
`check_trace_coverage.validator/check.py` fails with `expected 8 rank(s), the
manifest lists 4` — a real deployment graded against a rank count nobody chose.

Suggested: drop "omit it (defaults to 8)" and leave "**must equal the
deployment's `tp`**", since the two are only the same sentence at `tp=8`.

The yaml's own comment has the same soft spot — *"a real run leaves it alone and
sets `--var tp=` if the deployment is not eight-way"* reads as though setting
`tp` moves `expect_ranks`. It does not, by the design two lines above it.

### T54 — declaring `runtime.replayed_from` would not close the hole; `additionalProperties` is the hole

`runtime.replayed_from` is **consumed in five places and declared in none** —
`check_deploy_kit/check.py`, `kit_status.py`,
`check_measurement_order/check.py`, `load/line.sh` — validating only
because `environment.schema.json`'s `runtime` has `additionalProperties: true`.
Confirmed: it is absent from `runtime.properties`, and `fixed` is open too.

**The obvious fix does not work.** Declaring the field constrains its *type* when
present and changes nothing about a misspelling, because the object stays open.
Demonstrated against the real rung-1 record with a stock `Draft202012Validator`:

```
key=replayed_from    schema_errors=0   consumers read '20260904T110647-fbaba0'
key=replayed_form    schema_errors=0   consumers read None      <- one character
key=zzz_not_a_field  schema_errors=0   consumers read None
```

**And it fails in the unsafe direction.** `None` means *"not a replay"*, so a
typo turns a replayed kit into one every consumer treats as a real bring-up —
the precise thing the field exists to prevent.

**This is the second instance of one shape at the schema layer, and naming it is
the point of this entry.** The first is `items_schema`, measured to validate the
*filename string* and never the contents (`handoff/content.py`, mission
rule G2). Both are **a check that is present and does not check the property** —
and in both the presence of the check is what stops anyone looking. Declaring
`replayed_from` would have *added* to that: the schema would then name the field
and still not defend it, so the next reader would have one more reason not to
check. **The next person to find an undeclared field will reach for the same
obvious fix; the fix is the object, not the field.**

**Only `additionalProperties: false` closes it, and the cost is now a number
rather than a worry.** Swept every environment record in the run root — 241
across the store's handoffs — for keys the schema does not declare:

```
runtime.replayed_from   217
fixed.sglang             10
runtime.work_root         9
```

**Three keys, and that is all.** So the change is: declare those three, then
close both objects — bounded, and verifiable by re-running the sweep to zero.
Not a contract-wide migration, which is what it looks like before counting.

**Not done, deliberately.** `environment.schema.json` is shared by all fifteen
kinds (CONTRACT §2) and closing an object turns a tolerated field into a hard
failure mid-run. It wants a decision and a green run behind it, not a quiet edit
while a run is in flight.

**Mitigated at the one producer that is new**, and it needs no schema change:
`replay_root.py` writes the record, then **reads `replayed_from` back through the
accessor a consumer uses** and refuses the whole run if it does not name this
hop. Proved by injecting the exact typo into a copy of the tool — `rc=1`, with
the refusal saying *"shipping this root would hand the flow a replayed kit
wearing a real one's face."*

That does not close T54; it closes the hole where a **new** producer could open
it. Every existing occurrence and the other two undeclared keys are still
governed only by `additionalProperties: true`.

**And running that test found a second defect, in m5's own tool.** The refusal
printed the value it read back, and it was **not `None`** — rung 1's `deploy_kit`
already carried a `replayed_from` pointing into the mock set's
`stage1-deploy/deploy_kit`,
because that run mocked stage 1. The rewrite was **overwriting** it, collapsing
`sealed corpus → run X → here` into `run X → here` and telling a reader the
numbers came from a real bring-up one hop back when they never came from one at
all. Now chained: `<run> <- <prior>`.

**The general form, which is the part worth keeping:** *the tool that writes a
provenance field is the one best placed to destroy provenance*, and a test aimed
at a typo is what surfaced it. On a kit with no prior `replayed_from` the
behaviour is identical and every check written for it passes.

### T55 — the CUDA graph ceiling belongs in the environment record and in `check_environment`'s compared set

**The half of the graph-ceiling defect that item 1 does not fix.
Approved by the package owner; deferred for the same reason as T54.**

Measured: **4.7x** in decode latency between two runs whose every *recorded*
variable is equal — node, image id, model, `tp_size`, cards,
`mem-fraction-static`, load shape. The difference was the CUDA graph ceiling,
against a load at concurrency 16 (M1.2.3.4): below the ceiling decode runs
captured, above it the engine falls to eager.

**There was no default to be wrong**, which is the part that makes this
structural. `DK_CUDA_GRAPH_MAX_BS` appears **nowhere** in the package; the
producing agent invents a value each bring-up. Measured across every kit in the
run root:

```
kit        DK_TP_SIZE   MAX_BS   note
sealed     1            8        replayed x18, byte-identical every time
real (a)   4            16
real (b)   4            16
real (c)   4            8
real (d)   4            32       in flight
```

**16, 16, 8, 32 over four real bring-ups**, and the setting sits at a different
place in the file each time because `env.sh` is regenerated rather than edited.
That is **strictly worse than a wrong default, because a wrong default is at
least reproducible.** Two of the four ship a ceiling below the concurrency the
mission grades at, and any run that is right is right **by luck, not by
construction**. The `:=8` that looks
like a default is a `tp_size: 1` record replayed eighteen times — *a value fixed
in an artefact, mistaken for a default because the artefact is replayed*.

**Landed already (item 1):** the eighth contracted parameter
`E2E_KIT_CUDA_GRAPH_MAX_BS`, the brief's `>= concurrency` criterion with *say
what you chose and why*, the adapter line that keeps the sealed kit passing, and
gate fault 14. That binds the producer.

**Still open, and it is the half that catches a violation rather than asking for
compliance:** the ceiling is absent from `environment.schema.json` and from
`check_environment`'s `compare_fixed_across_inputs`. **Two engines differing
4.6x in decode speed validate as the same one.** Same argument as `image_id`
over `image` — the compared set exists precisely to catch two things that claim
to be one.

**Deferred, not dropped.** `environment.schema.json` and `check_environment` are
shared by all fifteen kinds; a change that can hard-fail mid-run has no green
rung behind it, so a failure would be unattributable between the schema and the
stage. Land after rung 1 seals green, as a proposal first.

**One caveat for whoever lands it:** `gpu_count` carries a warning against
adding it to `compare_fixed_across_inputs` — every stage inherits m1's record,
so all five agree by construction and cross-input agreement detects nothing. The
graph ceiling has the same inheritance, so comparing it **across inputs** is
worth no more than comparing `gpu_count`. What is needed is a comparison against
**the concurrency the load actually ran at**, which lives in m2's artefacts, not
a cross-input equality.

### T56 — `summarise.py` exists twice, byte-identical, with call sites split across both copies

**Found while verifying an account of four empty summary files; recorded rather
than acted on, because whether the duplication is deliberate is not
establishable from the tree.**

```
9246d23165e72b5cdb359689b7892dc0  assets/load/summarise.py
9246d23165e72b5cdb359689b7892dc0  assets/bench/summarise.py     <- same md5
```

Three call sites, split between them:

```
assets/accept/measure.sh              -> $BENCH/summarise.py
assets/load/replay.sh                      -> $LOAD/summarise.py
check_deploy_serves/check.py               -> assets/bench/summarise.py
```

**Every one passes both arguments and is correct.** The script refuses on
`len(argv) != 2` with a usage message and rc 2, so a one-argument call produces
an empty file rather than a wrong one — which is the right failure, and is what
an experiment harness reaching for `$BENCH/../load/summarise.py` produces: zero-byte
summaries, from the navigation rather than from the flow.

**Why it is worth an entry anyway.** The mistake is *navigating between two
copies of one file*, and that is the seam this package exists to remove — each of
the handoff analysis's three seams is one name over two things. Two identical
copies with owners on both sides is the same shape one level down: harmless while
they agree, and the day they stop agreeing nothing will say so, because nothing
compares them.

**Not acted on, and the uncertainty is real.** `assets/load/` is m2's stage
directory and `assets/bench/` is where the validator reaches; the duplication may
be a deliberate ownership boundary rather than an accident, and deleting either
copy breaks live call sites. **This wants its two owners to agree on one home,
not a unilateral edit.**

**Unrelated but measured on the way past, and it corrects a plausible reading:**
an empty summary in `check_deploy_serves` does not fall back with a warning, it
**fails** — `check.py` catches the `ValueError` from `json.loads` and
returns *"the load ran but produced no readable summary … A load with no numbers
has not shown the deployment serves under load"*. The announcing-`WARNING` path
is the different case of a summary that parses but carries neither
`request_latency_ms` nor `output_sequence_length x inter_token_latency_ms`.
Two failure modes, two behaviours, and they are easy to conflate.

### T57 — one scratch path, three independent literals, agreeing by coincidence

**From dry-running a late stage's launch line against an early stage's tree.
Spans m1, m4 and m5, and the same variable is reachable from either end.**

Three variables name one directory tree and **none derives from another**:

```
work_root           '${work_root:-<work root>}'                    m5, m1, shared.yaml
scratch_root        '${scratch_root:-<work root>}'             m4, five sites
validate_work_root  '${validate_work_root:-<work root>}'  m1
```

**They agree today because the rung-5 command passes `work_root` equal to its own
default.** Change it and the other two silently keep pointing at the shared
default — a run whose work root is elsewhere and whose scratch and validate
trees are not.

**And the guidance invites exactly that.** A reader is told to pass a
**run-unique `container=`** because the default is a fixed name on a shared host.
Anybody applying that reasoning one variable over — a run-unique
`work_root`, which is the same argument for the same reason — splits the three
without a word from anything.

#### The part that closes a loop: they *cannot* be derived

The natural spelling is:

```yaml
scratch_dir: '${scratch_root:-${work_root:-/mnt/…/e2e_flow}/kfo}'
```

**That is the nested default**, which expands to a literal string with **zero
problems reported** — an unparseable variable reference is passed through
silently.
So the derivation is not merely unwritten — **it is unspellable**, and three
owners each wrote the constant out because the loader gives them no other
option.

**A framework defect found in the morning produced a cross-owner hazard by the
afternoon**, not by breaking anything, but by making the correct expression
unavailable and the incorrect one silent.

#### The other end, m4's

`work_root` is on m4's launch line and **never read by their stage** — they
carry a variable whose value does not reach them, while `scratch_root`, which
does, is a separate name nobody passes. Two halves of one thing: a var passed
and not read, and a var read and not passed.

#### Shape

`min_requests` for paths (T-series, the `integration_min_requests` split): one
name that should be one thing, spelled independently by owners who each read a
correct value in their own file. **The difference is that `min_requests` could
be split and this one cannot be joined.**

#### Not fixed, deliberately

Three owners, and the only correct fix — deriving two from one — is the
unspellable form. The options are: pass all three explicitly on every launch
line (verbose, and a launch line is where this class of defect lives),
or resolve the nesting fault upstream. **Neither is m5's alone.**

### T58 — counting kits on disk counts how often we tested, not how often a producer chose

**The fingerprint is the only part anyone can act on.**

A census of `packup` trees is the natural way to answer *"how often has a producer
done X?"*, and it is wrong by construction. Measured over the frozen root:

```
find <shared path> -path '*packup*' -name env.sh    ->  58 files

55   line 105, DK_TP_SIZE:=1     the sealed kit, byte-identical replays
 3   no ceiling variable at all  two 13-line stubs, one a stripped copy
--
 0   produced by an agent
```

**Most of the 55 are `ws_handoff_refine/m1/gate*/{good,bad}` — copies `gate.sh`
makes of the sealed kit on every invocation, two per run.** So the corpus is
mostly a record of testing, and **the ratio gets worse every time anyone
runs a gate.** Our diligence tilts the evidence, monotonically, in the direction
of "this happened constantly".

It already misled a real sweep: eight pre-contract runs were read as eight
producers choosing a ceiling of 8, when six were one artefact replayed. The
corrected producer record is **four runs — `16, 16, 8, 32`, one below the bar**.

#### The fingerprint

**A produced `env.sh` is regenerated each run, so the same setting lands in a
different place each time. A replayed one is byte-identical, so it does not
move — and every replay therefore carries `DK_TP_SIZE:=1`, while every real
bring-up carries `DK_TP_SIZE:=4`.**

So: **before counting a kit as evidence of a choice, read the `tp_size`.** A kit
whose `tp_size` is 1 is the sealed artefact, whatever
directory it is sitting in. Anyone can apply that; nobody would infer it.

And the second reason those 55 do not count even as failures: their ceiling of 8
is **correct for the deployment they describe**, which is `tp_size: 1` — not the
concurrency-16 shape the bar is about. `mock_adapt.sh` preserves it deliberately
(*"adapts a record forward; it does not re-tune a deployment that already
happened"*), so they are a design decision working, not a producer failing.

#### Why it is worth an entry rather than a correction

**The discipline itself generates the false signal, and being more careful makes
it worse.** It is the same shape as the grep trap in T57 — there, good comments
make a naive detector fire; here, thorough testing makes a naive census
over-count. **Neither is fixable by care.** Both need the artefact's
provenance established *before* it is counted, which is what the fingerprint is
for.

Applies to any future sweep of `packup` trees, which is exactly what someone will
reach for the next time the question is *"how often has this happened?"*

### T59 — the instruments that failed, and not one failed toward "I cannot tell"

**An observation about the set. Deliberately untitled by count: the count grows,
and a number in a heading is the thing this file has already had go stale
twice.**

#### What to do about it, which is not "be careful"

**Re-ask any question whose answer you liked.** It is the only thing that works,
and it is not diligence — every catch here comes from distrusting a *convenient*
result rather than from being more thorough: the `df` differential, a second
spelling, an argv print, testing an axis instead of adopting it.

**Knowing this list does not protect you.** Three of its members can turn up
inside one four-minute check of whether a run has stalled — a `pgrep` that
matches its own shell, a `ps` pattern too narrow to see the process, and a
`git ls-files` run from the wrong directory. The catalogue is for **diagnosing
afterwards**; it does not help you **avoid**. What stops all three is widening
the query a third time, and the third query is not more careful than the first
two. **It is more sceptical of an answer you wanted.**

**The mirror: an inconvenient answer is waved through because accepting it feels
like rigour.**

*"Re-ask any question whose answer you liked"* catches the flattering case. The
unflattering one has no such trigger, and the reason is that **accepting a cost
feels like the check rather than a substitute for it.** When someone tells you
something that means more work for you, taking it on the chin reads as honesty —
you are visibly not defending your own position — and that feeling occupies the
place where the verification would have gone.

Measured: a report that `run_with_long_stall.py` is untracked. It is plausible,
it is about somebody else's file, and it means work.
**Nothing about it felt wrong**, which is the point. It was checked only because
the day had made checking reflexive:

```
git ls-files --error-unmatch <path>   ->  TRACKED, and identical to HEAD
```

The claim came from a repo-root-relative path used from inside `e2e-flow/`.

**And it is worse than the convenient case in one specific way.** A flattering
answer has a natural sceptic — anybody who does not benefit from it. **An
unflattering one has none.** The person it costs has already accepted it, and
nobody else has a reason to look. So the convenient error gets caught by the
room, and the inconvenient one only by the person paying for it, who is the least
motivated to try.

**So the rule is not "distrust convenient answers". It is: re-ask any question
whose answer you had a reaction to.** The reaction is the signal, not its
direction.

Each of these was reached for as a measurement, returned a confident answer, and
the answer was wrong. **None returned an error, an empty-with-reason, or
anything a reader would treat as "unknown".**

```
find -newermt '-6 minutes'   returned NOTHING one minute after a file was written.
                             Reporting it would have called a working run stalled.
                             The shell here is `bfs`, which rejects GNU relative
                             time spellings -- and that complaint is routinely
                             discarded with `2>/dev/null`.

df /var/lib/docker           reported 123G/70G on a filesystem a 28.5 GB `docker
                             load` does not move, and the load was refused on it.
                             **Two independent routes agreed** -- the exec
                             namespace and a host bind-mount through the daemon --
                             and both were wrong, because both shared the
                             assumption under test.

pgrep -f "docker save"       matched its own command line, because the pattern was
                             inside the `bash -c` string being run. Reported a
                             finished save as still running. (leader's)

docker inspect --format
  '{{if hasPrefix ...}}'     `hasPrefix` is not a docker template function. It
                             printed EMPTY rather than erroring, so "the container
                             has no GPU pin" and "the template is broken" were the
                             same output. The pins were there: 0,1,2,3 and 4,5,6,7.
```

```
grep -oE '[A-Z_]*CUDA_GRAPH_MAX_BS'          returned `E_KIT_CUDA_GRAPH_MAX_BS`.
                             **`2` is not in `[A-Z_]`**, so the match began mid-token
                             and the variable NAME came back wrong by two characters
                             -- inside a verification of a claim ABOUT variable names.
                             (checkpoint's. The value and timestamp it was
                             checking were unaffected, which is why it survived: the
                             answer was right and the label on it was not.)
```

```
git ls-files --error-unmatch <repo-root-relative-path>   run from a SUBDIRECTORY
                             returned `Did you forget to 'git add'?` for a file
                             that IS tracked. The path did not exist *from that
                             cwd* -- so git answered a question about TRACKING
                             when the true answer was about LOCATION, and named
                             the wrong cause with total confidence.
                             **`ls` on the identical path said "No such file or
                             directory".** Same input, same instant: one tool
                             right, one wrong, and the wrong one was the one
                             whose answer sounded like a finding.
```

**The sixth is a different failure from the other five, and worth separating.**
The others returned *silence* — empty output, or a match against themselves —
and silence is at least ambiguous on its face. This one returned a **specific,
actionable, confident misattribution**: a real diagnosis, of the wrong thing.
**It is the one shape where re-running the same query more carefully cannot
help, because the query was fine** — only a *different* tool exposes it, and
here the tool with the right answer was already installed and one word away.

**The fifth is the sharpest of the set.** A character class that silently starts
matching mid-token, used to check what a name *is*. It fails in the same
direction as the other four -- confident, wrong, no error -- and it does so
**inside the act of verifying**, which is the one place that reads as safe.

**The property is the direction, not the count.** A tool that degrades toward
*"I cannot tell"* costs a second measurement. These four cost a **finding** —
each one has a false conclusion already drafted behind it, and three of those
conclusions would land on somebody else's work: a stalled run, a corrupt backup,
a container with no pin.

**What actually caught all four was a second, independent measurement**, never
care or suspicion. `-newermt` fell to `-printf '%T@' | sort -rn`; the `df` fell
to a before/after across the operation itself; `hasPrefix` fell to a plain dump.
**The `df` is the instructive one: a second route was already sought and gave
the same wrong answer, because agreement between two routes is not corroboration
when both rest on the assumption being tested.**

**So the rule is not "be careful with tools".** It is:

1. **prefer instruments that can say "I do not know"** — and when one cannot,
   assume the confident answer is a hypothesis rather than a reading;
2. **make the second measurement structurally different**, not merely a second
   attempt — a different mechanism, ideally a before/after across the very
   operation in question;
3. **an empty result is a hypothesis about the world OR a hypothesis about the
   query, and nothing in the output distinguishes them.** Three of the four were
   empty output read as a fact.

Related in kind, different in layer: see **T52** for the `df` case in full, and CONTRACT §4.4's sixth face for the launch-var version of the same
shape (a name that resolves while the value it names is false).

### T60 — the graph ceiling is chosen by stage 1's load and spent by stage 2's

**The first cross-stage refusal.** `check_bench_result` refuses
`profiling_mode_off.bench_result`, `strong`:

```
kit ceiling (--cuda-graph-max-bs)      16
decode concurrency the load achieved   25.42
-> decode exceeded the captured graph on essentially every step, engine fell
   back to eager decode; measured 4.6x decode difference from this cause alone,
   same image, same node
```

**The refusal is correct and must not be widened.** The same run's
`check_deploy_serves` passed `strong` on the *same kit*, because stage 1's own
load runs at conc=16 and 16 >= 16. **Stage 1 chose a ceiling adequate for the
load stage 1 runs, and stage 2 then ran a heavier one against the same kit.**

#### Why five separate packages could never have found it

Stage 2 has never consumed stage 1's kit before this run. The seam only exists
once the stages are one graph, which is the mission's whole premise — **the first
real chained run produced a defect that no amount of work on the five demos
would have surfaced.**

#### What the code says, against the package owner's first instinct

The lead's initial lean was *"stage 1 should choose a higher ceiling"*. **Reading
the code says the other path is the designed one:**

* `deploy_and_prove/mock_adapt.sh` — `: "${E2E_KIT_CUDA_GRAPH_MAX_BS:=${DK_CUDA_GRAPH_MAX_BS:-8}}"`, i.e. `:=`, so a consumer that exports it first **wins**;
* `check_deploy_kit.validator/gate.sh` — planted fault **14** is *"the graph ceiling bound with no parameter"*, so the gate already treats an unparameterised ceiling as a defect. **The parameter exists so someone downstream can bind it.**
* and `E2E_KIT_CUDA_GRAPH_MAX_BS` is **not** a `--var` in `shared.yaml` or any step yaml — it is bound only inside the kit.

**Nothing in stage 2 sets it.** No reference in the profiling assets or
`assets/lib/line.sh`. So stage 2 inherits stage 1's choice silently, which is
exactly the condition CONTRACT §4.6 says no comparison can detect, because both
arms would inherit the same wrong value.

#### The open decision, for m1 and m2 jointly

1. **stage 2 binds the ceiling before running the kit's `deploy.sh`** — matches
   `:=`, keeps the kit a recipe, and puts the choice with the module that knows
   its own load. Costs capture time and memory, which is why a blanket high value
   is not obviously right.
2. **stage 1 chooses a ceiling covering the heaviest downstream load** — simpler,
   but stage 1 cannot know what stage 2 will demand, so it is a guess dressed as
   a default.

**Not decided here**, and recorded because the next run hits it again.

#### What must not happen

**Do not raise the bar to make this pass.** The 5 % / 10 % precedent applies:
widening a bar that refused correctly was already tried once and was the wrong
response. A ceiling of 16
against concurrency 25.42 is a real 4.6x, not a threshold artefact.

### T60a addendum — the override channel already exists and is one line from working

Found while reading `assets/load/line.sh` after rung 2e died.

**`E2E_KIT_ENGINE_EXTRA_ARGS` is the designed override channel**, and the file
says so itself at `line.sh`:

> *"`EXTRA_ARGS` is appended to the worker's argv **last**, so it overrides an
> earlier occurrence of the same flag rather than racing it."*

So `--cuda-graph-max-bs 32` passed through that channel **beats the kit's own 16**,
by design, with no change to the kit and no decision about what stage 1 should
choose. It is already plumbed end to end — `line.sh` passes three
`E2E_KIT_*` overrides into the kit's `deploy.sh`, and this is one of them.

**The only obstacle is `line.sh`:**

```bash
export E2E_KIT_ENGINE_EXTRA_ARGS=""        # hardcoded; an inherited value is discarded
```

A value-preserving form — `"${E2E_KIT_ENGINE_EXTRA_ARGS:-}"` — changes nothing
when unset and makes the hook usable. **One line, behaviour-identical by default.**

**Two things that are NOT solved by it**, so nobody reads this as the whole fix:

* the `CAPTURE=1` branch at `line.sh` **overwrites** the variable with
  `--disable-cuda-graph`, so an inherited value is still lost on the
  `profiling_mode_on` line. Only `profiling_mode_off` — which is the line that
  refused — is covered.
* it makes the ceiling settable, it does not decide **what it should be**. 32
  covers the 25.42 observed once; nothing says that generalises to another trace.

**Recorded rather than applied.** It is m2's file and the decision spans m1 and
m2, so the exact fix is written down instead — landing it is a minute's work
rather than a rediscovery.

#### Correction to the addendum — the direct parameter is the better lever, and the real gap is `KIT_ENV_PREFIX`

The addendum above recommended unpinning `E2E_KIT_ENGINE_EXTRA_ARGS` at
`line.sh`. **Reading the sealed kit itself shows a cleaner lever and a
different gap.** From rung 2e's own `deploy_kit`, not from a comment:

```
scripts/env.sh            : "${E2E_KIT_CUDA_GRAPH_MAX_BS:=16}"
scripts/env.sh           : "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
scripts/start_worker.sh       --cuda-graph-max-bs '${E2E_KIT_CUDA_GRAPH_MAX_BS}' \
scripts/start_worker.sh       ${E2E_KIT_ENGINE_EXTRA_ARGS}
```

**Both are `:=`**, so an inherited value wins for either. And `EXTRA_ARGS` really
is last on the argv — line 95, four lines after the ceiling flag — so the comment
was accurate. **Two working levers, not one.**

**But the direct parameter is better**: `E2E_KIT_CUDA_GRAPH_MAX_BS=32` sets the
value the kit is built around, where the EXTRA_ARGS route sets the same flag a
second time and relies on last-wins. One is configuration; the other is an
override of an override.

**And the gap is not `line.sh`. It is `line.sh`:**

```bash
KIT_ENV_PREFIX="E2E_KIT_RUN_TAG='$E2E_KIT_RUN_TAG' \
  E2E_KIT_PORT_BASE='$E2E_KIT_PORT_BASE' \
  E2E_KIT_WORK_ROOT='$E2E_KIT_WORK_ROOT'"
```

Three variables. **The ceiling is not among them, so it never reaches the kit at
all** — which is why stage 2 inherits stage 1's choice silently. Adding a fourth
line in the identical pattern is the whole change, and **passing it empty is safe
precisely because `env.sh` uses `:=`** — a null value takes the default 16, so
the behaviour is identical until someone sets it.

**Still m2's file and still not applied by the lead.** The correction is recorded
because the addendum's advice would have worked by the weaker route while leaving
the actual gap in place.

#### Root cause — the producer was briefed to choose 16, and did

Both entries above look for a **lever** to override the ceiling. Reading
`assets/deploy_and_prove.task/readme.md` shows there is nothing to
override: **the kit shipped 16 because the brief asks for 16.**

> *"Set the CUDA graph ceiling to at least the concurrency **this deployment will
> be loaded at**, and write down why you chose the number. The load is
> **concurrency 16** (mission M1.2.3.4)."*
>
> *"**Criterion:** the ceiling your kit ships is `>= 16`…"*

**The producer complied exactly.** A kit at 16 satisfies the stated criterion, and
`check_deploy_kit` passed it `strong` — correctly. Three of the four historical
kits chose 16 or 8 for the same reason.

**The defect is the scope of one phrase.** *"the concurrency **this deployment**
will be loaded at"* is right for the five separate demos, where the deployment
was loaded only by the task that made it. In the chained flow the kit is loaded
again by **stage 2**, whose Mooncake trace replay reached **25.42** — and nothing
tells stage 1 that consumer exists.

So the same sentence is correct in the old world and wrong in the new one, which
is exactly the class this mission was created to find, and it took a real chained
run to surface it.

**Why this is the fix and the levers are not.** A `--var` or an `EXTRA_ARGS`
override lets an operator paper over a briefed value that is wrong; changing the
brief makes the *next* producer choose correctly with no operator involved. The
levers stay useful for a one-off experiment; they are not the repair.

**Shape of the repair** (m1's readme, one paragraph — still not applied by the
lead, still m1's file):

* scope the instruction to **the heaviest load any consumer will apply**, not to
  this deployment's own;
* name the measured datum — stage 2's trace replay reached **25.42** on
  so `>= 16` is not sufficient for the flow;
* keep the *"write down why"* requirement unchanged. It is the best part of the
  brief and it is what makes a wrong number auditable: *"a number with no reason
  fails this even when the number is right."*

**Do not simply raise the criterion to `>= 32`.** That repeats the original
mistake with a bigger constant — a number that happens to cover one observed
trace, with no statement of what it must cover.

### T60b closed by rung 2f — and 25.42 was a symptom, not a requirement

**The fix worked end to end.** rung 2f (`20260904T225556-55e566`), node 088:

```
m1_deploy                succeeded  (59 min; kit/environment/deploy_serves all true)
run_profiling_mode_off   succeeded  <- FIRST TIME STAGE 2 HAS EVER COMPLETED
  check_bench_result     true
  721 request records, 0 errored
  decode graph ceiling 32 >= decode concurrency 4.909
```

Three links, all measured: the corrected brief made the producer choose **32**
instead of 16; the producer used the non-deprecated `--cuda-graph-max-bs-decode`
and verified from `/get_server_info` that capture buckets reached 32; and
`graph_ceiling` was taught that spelling so it read the kit instead of refusing it.

#### The part that corrects the brief

**Two runs of one trace measure decode concurrency 25.42 and 4.909 — same trace
(721 records both times), same tp, same model.**

The ratio is ~5.2x, and the eager-fallback penalty measured on this cluster is
**4.6x**. So the most likely reading is that **the high concurrency was caused by
the low ceiling**, not demanded by the trace: a ceiling below the batch forces
eager decode, eager decode is ~4.6x slower, slower decode leaves more requests
in flight, and the in-flight count is the concurrency. **A positive feedback loop
in which the symptom looks like the requirement.**

**This makes the number in `deploy_and_prove.task/readme.md` misleading.** It
says *"stage 2's Mooncake trace replay reached 25.42, so `>= 16` is not
sufficient"*. The first half is a true measurement of a *degraded* run; a future
producer sizing to cover 25.42 would over-provision on a rationale that does not
hold once the ceiling is right.

**Confound, stated rather than buried:** the two runs were on **different nodes**
(217 and 088). Node speed also moves in-flight count. Nothing here separates the
two causes, and one clean test would: run the same trace on **one** node at
ceiling 16 and at ceiling 32 and compare the achieved concurrency. That is m2's
C1/C2 shape and it has not been done for concurrency, only for ITL.

**Not corrected in the brief yet**, because the honest replacement is not obvious:
the instruction *"cover the heaviest consumer's load"* is still right, but the
load it must cover is the one measured **at a correct ceiling**, which is
circular unless you measure twice. Recorded so the next person does not read
25.42 as a fixed property of the trace.

### T61 — a defect the instrument hides by functioning correctly

**From a warning of m2's that was about something else.** Named
here because both the package owner and m2 independently said it is a shape the day's
collection did not already contain.

`check_no_regression/check.py`, before the fix:

```python
name = args.get("schema")
if name:
    schema_lib.validate(str(name), report)
```

**With `schema` absent, the validator's first and strongest check does not run
and nothing says so.** The other five args are thresholds that degrade to safe
defaults and still bite (`0.05`, `0.10`, `0.10`). This one degrades to nothing.
That much is ordinary — `items_schema`'s shape, present and checking nothing.

#### What makes it its own entry

**It is invisible exactly when the tooling is correct.**

It surfaced only because `probe_validators.py` was passing `args={}` to every
validator — a *defect* in the probe. Under that defect, `check_no_regression`
returns **True** with all six args discarded, and the passing row is the tell:
*a validator stripped of its thresholds passes trivially.*

Once the probe passes the real args, **the condition that reveals this can no
longer occur through the tool.** Every other instrument failure recorded here is
visible *because* something is broken:

| shape | how it becomes visible |
|---|---|
| a check that cannot fail (§4.4) | by asking what it would report if the subject were broken |
| a bar neutralised by a default | by reading the default |
| a probe that reads a warning as the thing warned about | by reading the matched line |
| **this** | **only while a second tool is malfunctioning** |

A working probe supplies `schema`, the branch is taken, and the validator does
its job. There is no observation, from inside the package's own tooling, that
distinguishes *"this check ran"* from *"this check would silently not run if
asked differently"*.

#### The general form

**A guard that is conditional on its own configuration is only as present as the
caller's discipline, and no test that supplies the configuration can see its
absence.** The absent-arg path has to be tested *deliberately* — the way a
refusal path has to be tested deliberately — because nothing produces it by
accident once the callers are correct.

#### What was done

Absent `schema` is now a refusal that names the missing arg. Verified both
directions — with the step's args it refuses on the documented 35 %/30 % bars,
with `schema` removed it refuses on the arg.

#### Where else to look

Not swept. The candidate shape is `if args.get(X):` guarding a check rather than
selecting between behaviours — as distinct from `args.get(X, <safe default>)`,
which is the correct pattern and the one the other five use. **Worth a sweep by
whoever next has a quiet hour; it is not urgent and it is not one owner's.**

---

### T62 — `base_sha256` defects 2 and 3, held pending an answer that may not exist

*From the day's traffic. Defect 1 is landed.*

**Defect 1 shipped with its own one-third caveat attached** — the fix states what
it does *not* cover rather than leaving the reader to find out. Defects **2 and
3 are held pending m5's manifest answer.**

**The part that makes this a deferral rather than a queue item:** if m5's answer
is *"the manifest has never run against an `sgl_kernel` operator"*, then 2 and 3
are **not a choice we are postponing — they are an unknown we have not measured.**
Those are different states and the record should not let them read alike.

**m5 answered: (a)** — and for a stronger reason than m3 had.
`SGL_KERNEL_ROOT` names **build sources**, so the site-packages module is a file
object no manifest ever references. **m5 also measured on 287 that
`/sgl-workspace/sglang/sgl-kernel` does not exist in the image at all** — the
root points at nothing.

**Record both halves, because they are different facts.** The decision is (a).
*And* the manifest has **never** run against an `sgl_kernel` or an `aiter`
operator:

```
@SGLANG_ROOT@        38 across every manifest that has ever existed (leader's count,
                     history-wide; 31 in the current tree excluding runs/ — checkpoint)
@AITER_ROOT@          0 in any manifest.  The single occurrence in the tree is a
                     `description` string in workset.schema.json, not a use
@SGL_KERNEL_ROOT@     1, and it is the guard constant at
                     optimize_kernel.task/steps/60_write_handoff.py
```

**So both the `_NOT_OVERLAYABLE` refusal and the `rebuild` branch are unexercised
code.** The original framing survives the answer: **we chose between two
branches, neither of which has ever run.** An answered question and a measured
one are still different things.

**Holder: m5.** Not blocking today.

---

### T63 — `snr_db: inf` is non-monotonic on `layernorm`, and both proposed mechanisms are disproved

*From the day's traffic.*

**Two mechanisms were proposed and both were disproved. One candidate remains
untested.** The entry exists so the two dead ones are not re-proposed: an
eliminated mechanism is worth as much as a confirmed one and costs the same to
lose.

**Not blocking.** Holder not assigned as of writing — **do not infer one from
whoever touched `layernorm` last.**

---

### T64 — `rank.task` and `identify.task` are 15-line skeletons, not briefs

*Checkpoint. Measured.*

```
rank.task/readme.md            15 lines
identify.task/readme.md        15 lines
build_workset.task/readme.md  377 lines
optimize_kernel.task/readme.md 386 lines
```

**Promoting either to `kind: ai` today hands an agent the mission rule and
nothing else** — 15 lines against the 377 and 386 that the two promoted stages
carry. The gap is not a matter of polish; it is the difference between a brief
and a placeholder.

**Consequence if promoted as-is:** the agent has no measured context, no prior
findings, no statement of what has already been tried — so it will re-derive,
and re-derive differently each run. `.claude/CLAUDE.md` rule 9 makes this
explicit: an agent can only absorb what the markdown says.

**Not blocking while both stages run as programs.** Blocking the moment either
is promoted.

---

### T65 — four profiling lines died in one morning, cause open

*Placeholder so m2's answer has somewhere to land.*

`p4_a`, `p4_b`, `p4_m4real`, and **`p4_b_m1real`**.

**The fourth is the informative one: it ran with m1 real, which rules out a bad
upstream artefact as the common cause.** Three lines with a shared mock upstream
could have shared its defect; the fourth could not.

**Why it is hard to see:** `0 validation(s) dropped`, no refusal, allocation
still held, stage still `running`. **The run is healthy in every field a reader
would check.**

**Holder: m2**, where the right instrument for it exists — a two-arm control,
one arm with the fix and one without, on disjoint cards of one node.

**Do not merge with the capture step waiting on another package's log.** Their
proximity in time is not evidence that they are one fault.

---

### T66 — the agent's system prompt points at a path that is not in the repository



`build_workset.task/readme.md` — **the live system prompt for a `kind: ai`
agent** — tells it that `../../../../../rank0/definitions/` *"holds two worked
examples and is the thing to imitate."* Three references, all off by one level.

**The level is not the defect:**

```
git ls-files rank0     ->  0 files
ls rank0/definitions   ->  gemm  moe      (present in THIS working tree)
```

**`rank0/` is untracked.** It exists only in working trees that happen to have
it. **On a fresh checkout there is nothing to point at, at any depth.**

**Why this is a third shape, distinct from a hang and from stale sibling data:**
the reference is **silently unreachable, and it lives in an instruction rather
than in code.** The agent gets no error, finds nothing, and **improvises the
Definition.** Nothing fails; the output is merely unanchored.

**Why nobody noticed:** damage is bounded because `check_workset_shape` enforces
the Definition keys downstream. **A validator caught the consequence and so the
cause never surfaced** — which is the same relationship as
a defect where good comments hid a bad grep.

**Holder: m3**, who is holding the fix until the full-real chain clears their
stage, because that readme is the live system prompt and editing it mid-run
changes what a running agent reads.

---

### T67 — `reverify_shapes` counts operators, and its name says shapes

*M3, at the package owner's instruction, while confirming the cost
of raising it.*

`assets/check_workset_runs.validator/check.py`:

- **`:279-282`** builds `picked` as **one primary shape per operator** — it
  iterates `document["operators"]` and takes the single shape with
  `is_primary`.
- **`:308-309`** `verified, unverified = picked[:wanted], picked[wanted:]` then
  `picked = verified`, where `wanted = W.arg_num(args, "reverify_shapes", 1, int)`
  (`:269`).

So the argument spelled `reverify_shapes` selects **operators**. The two units
coincide today only because the ranker emits four operators and each declares
exactly one primary shape.

**Why this is the day's recurring class and not a naming nit:** *an instrument
reads a real thing and answers a different question, and is never wrong in a way
that shows up as an error.* Setting `reverify_shapes=4` on a five-operator
workset re-measures four of five and reports success. Nothing fails.

**It is not silent, but the disclosure arrives late.** `:310-315` names every
unverified operator — `recorded, NOT re-measured` — and `:316-320` prints what
raising it would cost. Both land in the validator's **notes**, which a reader
meets after they have already read the verdict.

**Not renamed today, deliberately.** `workset_reverify_shapes` is a launch
`--var` on lines that are running; renaming an argument mid-run breaks the
launch command rather than the code, which is the worse failure to introduce
while a chain is live. The package owner agreed and asked for the entry instead of the
rename.

**What to do when the runs are quiet:** rename to `reverify_operators`, or make
the selection genuinely per-shape. **Do not just raise the default** — the
comment at `:296-304` argues against that, and it is right: each re-verify is a
container start and a torch import, ~90 s against ~3 s of timing, and the cost
scales with operator count.

**Blocking the moment the ranker's operator count changes**, because from then
on the argument's name actively misleads whoever sets it. Not blocking before
that.

**Holder: m3.**

---

### T68 — an attested number and a claimed one are indistinguishable in the handoff

*M3, from m4's question. m4 ranks it above both forge
defects and the package owner agrees; recording that because it is a consumer's
ranking of a producer's defect.*

**Re-verification is the validator's act and never writes back.**
`check_workset_runs` re-measures shapes through the workset's own selector and
compares against the recorded figure — but the result lands in
`<zone>/validation-<id>/validator_report.txt`, which is **not** part of the
handoff. The workset's `evidence/performance.json` has no field for it, under
any spelling, because there is nothing that would write one.

**Why this is not T67.** T67 is a name that lies about a unit. This is an
attestation that **exists, is correct, and is unreachable from the artefact it
attests**. m4 grepped for it and could not have found it.

**Why it is worse than it sounds.** M4.3.5 makes the workset m4's ground truth
*strictly* — they are told not to re-measure — and that instruction is only
safe because this validator ran. So the consumer is required to trust numbers
they cannot tell apart:

```
attention_chunk_fwd_o/case_001    recorded 0.1353 ms, re-measured 0.1352 ms
attention_chunk_gated_delta_rule  recorded, NOT re-measured
elementwise_…act_and_mul          recorded, NOT re-measured
layernorm_layer_norm_fwd_1pass    recorded, NOT re-measured
```

All four look identical in the handoff. Run `20260905T064703-5fbd66`, workset
`91ea967b`.

**And precision does not substitute for it.** All four rsd values are ≤ 1 %.
m4 declined to infer attestation from that and was right to; the attested one
happens to be the top-ranked operator, which is **luck, not a property of the
process** — `reverify_shapes=1` samples the first in ranker order.

**The fix is a field, not a bigger sample**, and it is small: the validator
already computes exactly what would go in it. `reverify_shapes=4` raises
coverage but leaves a five-operator workset in the same state.

**Holder: m3.** Not blocking; the package owner has ruled the entry itself is the
deliverable today.

---

### T69 — `identify` leaves `fellow` empty for most operators, and two producers write the tag

*At the package owner's instruction. The unblock is landed;
this is the root, filed separately so the workaround does not close it.*

**This entry's original diagnosis was wrong and is corrected here.** It said
*"the tags `identify` writes are bare language names"*, inferred from workset
`91ea967b`'s Definitions carrying
`['attention', 'linear-attention', 'triton', 'gated-delta-rule']`. Reading a
real `operator_identity` — run `20260905T074905-9ec798`, 287 — shows the
opposite: **`identify` writes a suffixed `fellow`** (`identify.py`), and
`scaffold.py` copies it straight into the Definition's tags. The original
`_fellow`, matching a tag ending in `-fellow`, was **correct by design**.

**What actually happens, measured on that run's five operators:**

```
k004  fellow=''            lang='unknown'   @SGLANG_ROOT@
k014  fellow=''            lang='unknown'   @SGLANG_ROOT@
k015  fellow=''            lang='unknown'   @SGL_KERNEL_ROOT@
k018  fellow='ck-fellow'   lang='ck'        @AITER_ROOT@
k024  fellow='triton-fellow' lang='triton'  @SGLANG_ROOT@
```

**`identify` resolves the language for 2 of 5 and leaves the rest empty**, and
`scaffold` then writes no fellow tag at all for those three — `[t for t in (...)
if t]` drops it. So the fallback fired **because the value was absent, not
because the spelling disagreed**.

**And there are two producers of the Definition, with different vocabularies.**
`91ea967b`'s bare-language tags are not `scaffold`'s shape
(`op_type, precision, fellow`) — `build_workset` is `kind: ai`, so **the agent
wrote those Definitions**, in its own vocabulary. `scaffold` writes suffixed,
the agent writes bare, and nothing reconciles them. That part of the original
entry stands: **two producers disagreed and nothing compared them** — just not
the two producers it named.

**The mock is a third vocabulary and the worst of them.**
`mock_adapt.py` hard-codes `"tags": ["softmax", "sglang", "generic-fellow"]`
— a literal name KernelForge does not register, and unguarded it flows through
untouched. So the mock does not merely fail to show the defect; **the mock corpus
is where `generic-fellow` comes from in the first place.**

**Landed, and it is better than the wrong diagnosis deserved.** The generator
accepts **both** vocabularies — suffixed from
`scaffold`, bare from the agent — validates against `fellows/constants.py`, and
refuses instead of substituting. It would have been correct under either
diagnosis, which is why the wrong one survived a landing. **The defect is
that two producers disagreed about a spelling and nothing compared them.**
`identify`'s taxonomy also offers `tilelang-fellow`
(`assets/lib/kernel_taxonomy.yaml`), which is not a KernelForge backend
either — that row can never be honoured and now refuses.

**What would actually close it, revised:** (a) find out why `identify` reports
`lang='unknown'` for three operators including two Triton files it resolved to
a source path — that is the real gap, and it is upstream of every vocabulary
question; (b) one declared vocabulary shared by `identify`, `scaffold`, the
agent's brief and `mock_adapt`, validated against KernelForge's constants at
build time rather than at campaign time; (c) delete `generic-fellow` from
`mock_adapt.py`.

**How the wrong diagnosis got here, since it is the day's shape once more:** it
was inferred from one artefact — a workset whose Definitions happened to be
agent-written — without opening the `operator_identity` that fed it. One
artefact, one producer assumed, a cause named. The correction costs one
`json.load` of a file that was already sealed.

**Holder: m3.** Not blocking — the wrapper refuses rather than substitutes, so
the dangerous outcome is gone even while the disagreement stands.

---

### T70 — `kernel_taxonomy.yaml` offers a fellow KernelForge does not have

*Split out of the T69 fix rather than folded into it.*

`assets/lib/kernel_taxonomy.yaml` maps a symbol shape to
**`tilelang-fellow`**. KernelForge registers seven backends
(`kernel_agents/fellows/constants.py`): `ck`, `flydsl`, `triton`,
`aiter`, `hip`, `hipblaslt`, `intellikit`. **`tilelang` is not among them**, so
that row can never be honoured.

**Unguarded it would substitute**; it now refuses. That is the
safe direction and it is also why this needs an entry: **the refusal reads like
a configuration error at the point of use**, hours away from the row that
caused it, to somebody who did not write either. The wrapper prints the seven
valid names, which is a hint and not an explanation.

**Note the asymmetry with the other four rows.** `triton-fellow`, `ck-fellow`
and `hip-fellow` all name real backends. One row out of four is wrong, which is
exactly the ratio that survives review — a table where every row is wrong gets
noticed.

**What would close it:** validate the taxonomy's fellow column against
KernelForge's constants **when the taxonomy is loaded**, not when a campaign
tries to start. Same shape as T69's fix and it should probably be the same
change.

**Holder: m3.** Not blocking — nothing dispatches a tilelang operator today.

---

### T71 — three separate defects share one cause: the mock corpus has a shape the real path does not

*M3 at the package owner's instruction. **This entry exists
because the third instance made it a pattern rather than a coincidence**, and
until now each was filed only where it was found.*

The mock corpus is 25 real sealed handoffs. They are real, and they were
produced by five *separate* packages under the old design. **Where those
packages differed from `e2e-flow`, the corpus encodes the old shape** — and a
mock run then exercises a branch the real run cannot reach. The failure is
always the same: **mock green, real refuses, and the difference is in the
material rather than the code.**

**Instance 1 — operator names.** *m3, measured.* Contract-diffing the real m3
output against the mock-injected m3 material gave **zero intersection on
operator names**, and three of the files m4 required existed only because the
mock adapter wrote them. m4's stage refused on the real path for that reason.

**Instance 2 — the four-versus-one operator count.** *Relayed rather than
measured here; recorded so the set is complete, and marked so nobody reads it as
first-hand.*

**Instance 3 — the fellow tag's spelling.** *Measured.* T69: the corpus
carries `<lang>-fellow`, real `identify` writes bare `<lang>`. The generator
matched the corpus, so mock took the matching branch on every operator and real
took the fallback on every operator — **100 % divergence, and both sides
looked healthy.**

**Why it keeps producing PASS-shaped defects.** A mock run that succeeds is the
evidence that the wiring works. When the corpus supplies the shape, the mock
proves the wiring against material the real producer will never emit — so the
green is real and tells you nothing about the path you are about to run.

**This is not an argument for dropping the corpus.** It is the only thing that
made stage-parallel debugging possible. **It is an argument for one specific
control:** for any field a mock consumer reads, diff the corpus's value against
what the real producer emits, **before** the real run — the same
contract-diff that found instance 1, applied ahead of the failure instead of
after it. Cheap, no node, and it is the only one of the three that was caught
before it cost a run.

**Holder: unassigned.** Instances are m3's; **the control is nobody's yet** and
should not be inferred to be m3's because m3 filed the entry.

---

### T72 — `expect_ranks` 在两份文档之间有缝,两份都没错

**来源:the launch notes 的审计;两处引用均已回源核对。**

**没有哪一份文件写错了,缝在它们中间——所以两边都没有可改的东西。**

- **`the launch notes:32`** 把 `expect_ranks` 按 **rung 号**取值(rung-5 表 `L2139`
  重复一次):rung 0–1 → `2`,rung 2 onward → 部署实际的 `tp`。
- **同一份文件的正文 `L200–222` 写的原则是对的**:*「it is a fact about **the
  artefact being graded**, not about the run」*。
- **在平直阶梯上,rung 号恰好等价于「m2 是否为真」**,所以今天它是对的。
- **在 skip-ahead 下不等价。** `replay_root.py`(docstring:*"Materialise a
  completed run's sealed handoffs into a `mock_root`"*,用法行直接给出
  `--var mock_stages=m1,m2`)可以把一次**真实 rung-2 capture** 塞进一个
  **m2 被 mock** 的调试 run。此时 capture 是 `tp=4`,而表格按 rung 0–1 判它
  该是 `2`。**`check_trace_coverage` 会在一次完整 bring-up 和加载之后,拒绝一份
  完全正确的 capture。**
- **the replay manual 里 `expect_ranks` 出现 0 次**(实测 `grep -c`)。
  another owner 认这个遗漏是自己的。

**形状**:与 `adhoc_cases` 同族,高一层——
**一个键在常见情形下是正确的代理,在一个已写进文档的变体下静默失效。**
`adhoc_cases` 是同一文件内两节打架;这一条是**两份文件之间**,所以连
「读全文就能发现」都不成立。

#### ⚠ 针对 `adhoc_cases` 的那条检查**找不到这一条**——不要跑完表格审计就以为覆盖了

第 17 条的可执行检查是**「拿速查表逐格对正文核,单方向」**。它对
`adhoc_cases` 那一族有效,**对 T72 无效**:

| | `adhoc_cases`(第 17 条) | `expect_ranks`(本条) |
|---|---|---|
| 速查表 | **错** | **对**,在它自己的语境里 |
| 另一处 | 正文,对,同一文件 | the replay manual,**只字未提** |
| 检查在比什么 | 格 vs 正文 —— 抓得到 | 格 vs **无** —— **没有可比对象** |

**`the launch notes:32` 不是陈的,那份文件里没有一个字是错的。** 缺陷是**缺席**,
而缺席对一条「比对两个陈述」的检查不可见。

**所以:跑完表格审计得到「干净」,不等于这一类被覆盖了。** 记在这里是因为
下一个人最可能犯的错就是这个推论。本族只有纪律,没有检查(another owner 原话):

> **当一份文档引入既有机制的变体时,义务在新文档一侧:把该变体作废掉的每一个
> 键重新申明一遍。没有任何东西能检测到这个遗漏。**

**为什么记成 todo 而不是 bug**:没有代码缺陷,也没有哪一行文档是错的。要补的是
**the replay manual 里缺的那一段**,以及把 `the launch notes:32` 的键从 rung 号换成
「被评的 artefact 的 tp」——后者是行为不变的措辞改动,但**要在链空闲时做**。

**持有人:未分配。** 实例来自 another owner 的审计,the replay manual 是 another owner 写的,
但 `the launch notes:32` 那一行是共享的,**不要因为是 another owner 报的就默认归他们。**

---

### T73 — the Triton fellow pattern needs a leading underscore and a terminal `_kernel`; real kernels have neither

*M3. This answers T69's revised close (a) — why `identify`
reports `lang='unknown'` for Triton files it resolved to a source path.*

**Measured** by running `taxonomy.fellow_of` against the five device symbols in
`operator_identity`, run `20260905T074905-9ec798` (287):

```
k004  chunk_fwd_kernel_o                              -> {'fellow': '', 'language': ''}
k014  chunk_gated_delta_rule_fwd_kernel_h_blockdim64  -> {'fellow': '', 'language': ''}
k015  _ZN7sgl_hip10activation18act_and_mul_kernelI…   -> {'fellow': '', 'language': ''}
k018  _ZN7ck_tile6kentryILi1ENS_38FmhaBatchPrefill…   -> {'fellow': 'ck-fellow',     'language': 'ck'}
k024  _layer_norm_fwd_1pass_kernel                    -> {'fellow': 'triton-fellow', 'language': 'triton'}
```

**Defect A — the Triton rule is anchored at both ends.**
`kernel_taxonomy.yaml`, `fellows:` → `triton-fellow`, pattern
`^_[a-z0-9_]+_kernel$`. It requires a **leading underscore** *and* `_kernel` as
the **final** token. Real flash-linear-attention kernels are
`<op>_kernel_<suffix>` with no leading underscore, so they miss on **two
independent counts**. `k024` matches only because it happens to satisfy both —
**one of three Triton kernels, and the one that passed is why the rule looked
like it worked.**

**Defect B — no rule matches `sgl_hip`.** The `hip-fellow` patterns are
`^_ZN5aiter`, `^void aiter::`, `topk_transform`, `anonymous namespace` — all
aiter. `_ZN7sgl_hip…` is matched by nothing, so a compiled HIP kernel comes back
with no language at all.

**The obvious fix is a trap, and the population says how big a one.** Relaxing
the pattern to the substring `_kernel` looked like it cost one misclassification
in the five-operator sample. Run against the **whole 124-row real
`kernel_table`** from the same run, it would additionally capture **58 symbols**
the anchored candidate correctly leaves alone:

```
void at::native::elementwise_kernel_manual_unroll<128, 8, …>
void at::native::tensor_kernel_scan_innermost_dim<float, std::plus<float> >
_ZN7sgl_hip10activation18act_and_mul_kernelI14__hip_bfloat16…
void rocprim::…::detail::trampoline_kernel<…>
void at::native::index_elementwise_kernel<128, 4, …>
```

PyTorch ATen, rocprim, sgl_hip. `fellows:` says *"first match wins"* and
`triton-fellow` is **first in the list**, so every one of those would be handed
to the Triton fellow **before `hip-fellow` is ever consulted** — **58 of 124,
47 % of the table.** That is the wrong-fellow hazard T69's fix exists to stop,
reintroduced at scale by relaxing the pattern. **In the five-operator sample it
looked like a 20 % edge case.**

Full anchoring is what does the work — a symbol containing a space, `::` or `<`
cannot match `^[A-Za-z_][A-Za-z0-9_]*_kernel…$` at all. `(?!_Z)` covers the
remaining case: mangled names that *are* otherwise valid identifiers.

**Validated candidate**, tested against all five:

```
^(?!_Z)[A-Za-z_][A-Za-z0-9_]*_kernel([_A-Za-z0-9]*)$
    k004 True   k014 True   k024 True      (the three Triton kernels)
    k015 False  k018 False                 (the two mangled C++ symbols)
```

The `(?!_Z)` is load-bearing: it is what keeps an Itanium-mangled name out of
the Triton bucket regardless of rule order.

**Not landed, deliberately.** m1's full real chain is on 217 and m3 is
downstream of it. The change turns two refusals into two campaigns — it makes
the pipeline do *more* work, on a matcher validated against **five symbols**.
Landing a widened matcher into a live chain's downstream stage is the same call
as `build_workset.task/readme.md`, and the same answer. **Ready to land the
moment 217 clears m3.**

**Run against the population, and the result is the table above.** On all 124
rows the candidate moves **11 kernels** from unclassified to `triton-fellow`
(9 → 20) and changes nothing else:

```
                current                with candidate
hip-fellow           23                     23
ck-fellow             2                      2
triton-fellow         9                     20
(none)               90                     79
```

**And the number that matters more than the fix: 79 of 124 still have no
fellow — 64 % of the real kernel table.** The taxonomy classifies about a third
of what the profiler actually sees. That is not a pattern bug and it is not
fixed by this change; it is the size of the gap between the taxonomy and the
workload, and it is otherwise unmeasured. **Worth its own entry once
somebody decides whether 64 % unclassified is acceptable** — most of those
kernels are never promoted by `rank`, so the practical exposure is much smaller
than the raw ratio, and *how much* smaller is unmeasured.

**Renumbered from T72 to T73**, because a concurrent `expect_ranks` entry took
T72 first and this one is the duplicate. **A number read from a heading listing
taken before the other entry existed** is the stale-pointer class again, this
time on an identifier rather than a line number, which is exactly what
*quote a finding by its name, not its number* rule is for. References in
`assets/lib/kernel_taxonomy.yaml` updated in the same commit.

---

**`buckets:` measured, and it is the more consequential copy.** The
same old pattern sits at `kernel_taxonomy.yaml`, under `buckets:` rather than
`fellows:`, and the consequence is not symmetrical:

- in `fellows:`, a miss means **no label, then the wrapper refuses** — visible;
- in `buckets:`, `taxonomy.py` puts an unmatched symbol in `unknown` with
  **`routable: False`**, so a miss means **excluded from the candidate pool**.
  `rank` never sees it. **A defect that mislabels is visible; a defect that
  excludes is not.**

**Measured on the same 124-row table** (one replacement, in the `routable`
bucket). Seven kernels move `unknown/routable=False` -> `routable/routable=True`;
nothing moves between named buckets and nothing loses routability:

```
  2.350%   fused_recurrent_gated_delta_rule_packed_decode_kernel
  0.540%   chunk_gated_delta_rule_fwd_kkt_solve_kernel
  0.120%   fused_qkv_split_gdn_prefill_kernel
  0.100%   chunk_local_cumsum_scalar_kernel
  0.100%   fused_gdn_gating_kernel
  0.010%   track_mamba_states_all_layers_kernel
  0.000%   compute_position_kernel

routable today   18 kernels   10.740 % of profiled GPU time
gained            7 kernels    3.220 %          pool +30 %
```

**The ranking consequence is narrower than the pool growth.** `rank` promotes by
`% Total`; the pool's current top is `chunk_fwd_kernel_o` at 2.860 %, so the
largest newcomer at **2.350 % would land at rank 2** — ahead of everything
currently selected except k004. The other six are <= 0.54 % and would not be
promoted. **So: one kernel that should be a top-two candidate is invisible to
`rank`**, and six pool members that change nothing.

**It does not invalidate existing worksets.** The four operators m4 optimises are
selected correctly; none of the seven displaces them below rank 5. The distortion is an **omission at rank 2**, not a wrong ordering of what was
chosen.

**Scope limit, and the number is only as general as the trace:** this is one
profile, one workload, one node. `2.350 %` for a *decode* kernel is a property
of this trace's prefill/decode mix. On a decode-heavier trace that kernel's
share would rise and the omission would matter more. **That is unmeasured, and
nothing here extrapolates to it.**

**HELD, for the round's goal rather than the
measurement.** This round is 跑通即可 — get the chain through, explicitly not real
performance improvement. Expanding the pool changes *which operator gets
optimised*: a 2.35 % newcomer at rank 2 means the next m3 run may hand m4 a
different operator than `attention_chunk_fwd_o`, discarding everything
characterised today — the attested baseline (0.1353 recorded vs 0.1352
re-measured), `noise_floor: 1.1348`, the premise gate, the `--kernel` frame — on
a node budget that has already lost most of a day. And there is no green chain
yet whose input this would change.

**One cost of holding, stated so it is a decision:** the omission is in `rank`'s
*input*, so the first green chain will be green over a distorted pool. **Getting
the chain through will not have validated the selection**, and nobody should
later read that green as evidence that it was.

**The right time is after the first green chain.** One line, with this
measurement already behind it.

**Holder: m3.**

### T74 — a killed validator leaves its own deployment holding the cards

**Recorded as a todo rather than fixed**, because the observed cause is already
fixed and this is the residual.

`check_deploy_serves` brings a deployment up itself. A run torn down while it is
in flight delivers **SIGTERM mid-`finally`**, so `teardown.sh` never runs, and
the serving container plus its etcd sidecar keep four cards at 75 % for as long
as anyone leaves them — measured at thirty-three minutes after the orchestrator
was gone. Nothing in the run reclaims them: the containers outlive the job by
design, and the run's own exit path does not know the validator brought anything
up.

**What is already done.** Every `on()` call is bounded at 600 s, which removes
the hang that gets this validator killed in the first place. That closes the
instance, not the class.

**What is left, and why it is not obviously worth doing.** A `SIGTERM` handler
in the validator that runs the teardown would close it. Against that: a handler
racing the framework's own kill sequence is a new failure mode in a body whose
job is to be trustworthy, and the leak is detectable in one command
(`docker ps` on the node) by anyone who notices a run died. **The cheap
mitigation is operational** — after any run that ends in `ValidatorInvalid`,
check the node for `*_serves-*` containers before launching on those cards.

**And it is not one validator's problem.** Any validator that brings something
up on a node has this shape; `check_patch_live` and m5's arms are the same
construction. So if it is worth a handler, it is worth one in a shared helper
rather than four.

**Holder: m1**, pending someone wanting the class closed rather than the
instance.

### T75 — constrain the `optimized_kernel.py` slot by KIND, not by filename — deferred, and the deferral is the decision

**Handed up rather than taken**, because it changes a handoff contract, which is
not a module owner's call.

**The defect** (29's thesis, and it has two instances today): 「一个优化后的 kernel」
和「一个内嵌了优化后 kernel 的测量候选」**同名、同后缀、同位置**。
`60_write_handoff.py` publishes whatever is there as `results/optimized_kernel.py`,
and `apply_patch` overlays it into a real image.

- **Instance A:** the slot holds the workset's `reference.md` -- prose.
  `ast.parse` fails outright. Guarded only by `base_sha256`, **and that guard is
  removed by anyone who measures the true hash on a node and fills it in.**
- **Instance B:** the slot holds a valid, self-documenting, correct
  three-part composite -- an `--impl` for the workset harness whose Part 2
  monkey-patches sglang at import and whose Part 3 defines `run`.
  **Part 1 verifiably IS iteration 2; the file does not lie.**
  **No syntactic check separates B from a real engine module.**

**The direction m4 recorded** (as a record, not a recommendation): have the artefact
self-report `engine_module` vs `impl_candidate`, and have the consumer constrain on
that rather than on the filename.

**Decision: not this round.** Reasons, in order:

1. **It is a contract change.** Phase-0's whole discipline was that the contract is
   frozen before the owners work in parallel; five chains have run against the
   current shape today. Risk 1 in the plan is exactly this churn.
2. **Phase (5) has never completed.** Nothing changes the kind contract while the
   only thing that has never worked end-to-end is still being attempted.
3. **The hazard is currently contained, by accident and knowingly.** `apply_patch`
   refuses B -- `_module_surface` is `ast`-based and Part 1's defs sit inside a
   string literal, so every stock name reads as dropped. **Correct outcome,
   unrelated mechanism**, and `apply.py`'s docstring already says *"this is a
   one-artefact rule"*. **Recording that we are relying on an accident is the
   point of deferring in writing rather than silently.**

**What would reopen it:** anyone proposing to install a `results/optimized_kernel.py`
that was not produced by `30_run_forge.sh` on the same run. **Under that condition
the accident stops covering us, and this stops being deferrable.**

### T76 — emitting `environment.yaml` is a producer's duty with no central enforcement, and three producers do not discharge it

**Found offline against a recorded run**, after
`check_environment` refused three kinds and the shape suggested one cause rather than three.

**The failing condition is identical in all three: step (1) *present*.** No schema
failure, no cross-input disagreement. `check_environment.validator/check.py`'s
`find_record` looks in exactly three places:

```
items/env/environment.yaml        reproducible, structured_text
items/codes/environment.yaml      code
items/codes/*/environment.yaml    code wrapped in one named dir
```

**Twelve of fifteen handoffs in that run carry a record at one of those paths;
three carry none, and they are exactly the three that refuse.**
**Positive control in the same run:** `deploy_kit` and `operator_workset` both hold
`items/codes/environment.yaml` and both pass — **so the paths are reachable and the
glob works.** (The evidence here is the miss, not a grep hit; see the audit rule.)

**The one cause:** `assets/lib/env_render.py` is the only thing in the package that
writes `environment.yaml` (its `_REL` map at `:58-60` is the source of those three
contract paths). **It is invoked per producer and nothing enforces the invocation.**
Seven call sites exist — `apply_patch`, `build_workset`, `deploy_and_prove`,
`merge_arm.py`, `merge_profiling_evidence`, `optimize_kernel/steps/_lib.py`,
`run_profiling_mode_off`. **`packup.py` is not among them, and neither is
`integrate_and_verify`.**

| kind | what sits where the record belongs | site |
|---|---|---|
| `integration_report` | nothing; `items/` is flat | `integrate_and_verify` never calls `env_render` |
| `e2e_packup` | `items/codes/environment.md`, prose + a markdown table, no YAML | `packup.task/packup.py` writes only the `.md` |
| `kernel_optimization` | `items/codes/<packup>/environment.md`, first line *"Copied verbatim from the workset's `environment.yaml`"* | same `packup.py` writer; **this artefact is the replayed one** |

**Routing, and two corrections to the obvious assignment:**

- **m5 is two sites, not one.** `merge_arm.py` *does* call `env_render`, which is
  why `stock.measurement` and `patched.measurement` pass in the same run. m5 emits
  for the measurements and not for the report or the packup.
- **`kernel_optimization` is not m4's live producer.** `_lib.py` calls
  `env_render`; the refusing artefact is the **replayed** one (its embedded YAML names
  `node: node-088`). **m4's action is "confirm the live path emits it",
  not "fix the producer".**

**A fix already exists for one of the three:** a machine-generated patch that is
`git apply --check` clean. **It closes `integration_report` only.** `e2e_packup` needs `packup.py`
to emit the YAML beside the `.md` it already writes — it has the data in hand there.

**The item is the layer above, not the three sites.** Closing all three leaves the
next producer free to forget. **Options not decided here:** have the seal require the
record for kinds whose content type implies it, or have `env_render` be called by the
framework rather than by each body. **Both change a contract, so neither belongs in
this round** — see T75 for the same reasoning.

### T77 — `redact` refuses by file suffix, and an evidence record is not a script — deferred, and it is a contract question

**m4 answers a question the package owner routed out of their own hands** (their words: three
wrong calls on the same artefact today, so a fourth is the worst-quality opinion
available). **The answer is: do not edit the builder. This is the same shape as T75.**

**The failure.** `packup.py` shells to `redact.py` with `check=True`; two paths in
`results/kernel_optimization.json` cannot be named, so it exits 1 and `entry.sh`'s
`set -eu` kills `packup.py` before `exec env_render`:

```
:32  "model_path": "<shared path>"
:61  "work_root":  "<work root>"
```

**`redact` is behaving correctly and this is not a `redact` bug.** Its prefix list is
built at `packup.py` from **this** run's environment — `E2E_WORK_ROOT`,
`E2E_MODEL_PATH`, `E2E_MOCK_ROOT`. The embedded record carries **another run's** root
(`e2e_flow_088a`). A prefix list assembled from the live environment can never cover a
foreign root, and the codebase already knows this: **`MOCK_ROOT` exists precisely
because a replayed kit records a root from the run that sealed it.** That was one
known foreign root; verbatim inheritance produces an unbounded set of them.

**So "pass another prefix" is a treadmill, and each new lap is discovered by an
`exit 1` after a real deployment has already run.** That is the expensive place to
find out.

#### The real question, and why it is not about paths

`redact.py` — `REFUSE_SUFFIXES = frozenset({".py", ".sh", ".json", ".jsonl"})`.
**The refusal is scoped by file suffix.** Its own docstring scopes it by *merit*:

> *a script carrying one host's directory layout does not run on the next host*

and it deliberately spares prose, because *"prose saying `/v1/models` bakes in no
host's directory layout."* **`premise.workset_environment` / `premise.run_environment`
sit on the prose side of that argument, not the script side.** They are not consumed
to locate anything — the record's own note says a consumer must read the handshake,
not this field. **They exist to say which host established the premise, so naming the
host is the entire content.** A `.json` suffix put them in the executable bucket.

**And today gave the cost of the alternative.** `e2e_flow_088a` in an artefact
circulating on 093 is exactly the provenance signal that took a `store/task` read to
settle. **A field whose job is to say "this came from somewhere else" is the last
field to launder into a placeholder.**

#### Options, none decided here

- **Scope the refusal by what the content is, not by its suffix** — the same
  correction T75 asks for on the `optimized_kernel.py` slot (kind, not filename).
  A record could self-declare `content_role: evidence | executable`.
- **Do not embed a host-specific record verbatim in generated content** — carry a
  reference (handoff id + version + digest) and let a reader resolve it.

**Both change a contract, so neither belongs in this round.** T75's reasoning applies
unchanged.

#### Unblocking the package owner's artefact today, and its cost

Passing `WORKSET_ROOT=<work root>` as a fourth prefix makes
`packup.py` complete. **It is a workaround, it is one lap of the treadmill, and it
should be labelled as such where it is added** — the next corpus with a different root
fails the same way, after another deployment.

### T79 — a run does not record the `--var` values it was launched with, and at least five of today's incidents are that one fact

**T79 and not T78 is deliberate: the number is the package owner's to assign.**

**The measurement.** A run stages its own copy of the package under
`zones/<task>/package/`. **That copy is NOT rendered.** On `p9`
(`20260905T163424-bdb4d8`), `zones/.../package/steps/m2_profiling.yaml` still reads:

```yaml
    expect_ranks: '${expect_ranks:-8}'
```

**So the launch line is not recoverable from the run.** Not from the staged package,
not from `store/task`, not from any artefact. It exists in the operator's shell
history and nowhere else, and the process table loses it the moment the run ends.

#### The worked example: a verdict that is a function of an unrecorded argument

Grading two runs' `profiling_mode_on.profile_result` offline
(`assets/lib/grade_offline.py`) with `check_trace_coverage`:

| `expect_ranks` | verdict |
|---|---|
| `4` | PASS |
| default `8` | REFUSED — *"expected 8 rank(s), the manifest lists 4"* |

Both runs are `tp_size: 4` — **read from each run's own `environment.yaml`, not
assumed** — and the validator independently re-parsed the trace and found 4 ranks
carrying 823736 GPU kernel events. **So `4` is certainly right about the deployment.
What no one can establish is whether the launch said so.** If either line ran without
the `--var`, `check_trace_coverage` **would have refused during the real run too** —
which would be a defect in a launch block, not a grading detail. m2 and m3 have been
asked; the point of the entry is that **the question had to be asked of a person.**

**m2 answered from the file, not from memory** — `<home>:18` reads
`--var expect_ranks=4`. **So `p9`'s PASS is a grading of the run**, and the tidy answer
and the true one coincided here; m2 said so explicitly rather than letting it be taken
on trust. **That does not weaken the item — the answer came from a shell script in
somebody's home directory, which is exactly the storage this entry is about.**

**And m2 named the case where the same variable is genuinely different**, which the
worked example above would otherwise hide: the launch notes precondition 2c gives
`expect_ranks` **2 for a mocked stage 2** (the 09-02 corpus is a TP-2 capture) and
**`tp` for a real one**. Grading mocked and real artefacts in one sweep with a single
value therefore manufactures refusals **that read exactly like producer defects**. A
grading table needs a mocked/real column, not one value for the sweep. *(This sweep
graded only real runs, so it is not affected — but nothing in the harness would have
noticed if it had not been.)*

**The launch value for one such run is recoverable only by asking**:
`grep -rl expect_ranks` across that run tree returns only unrendered package sources,
and `E2E_EXPECT_RANKS` appears in no staged environment anywhere in the run — which
is the entry's central claim, confirmed twice while answering a question about
something else.

**And the careful reading cuts against the tidy answer:** two reports of `4` are
**not independent** — one comes from `tp_size: 4`, the other from a canonical block
whose rule is literally `--var expect_ranks=<the kit's tp_size>`.
**Both terminate at the same field**, so the agreement establishes that the run was
launched with 4, **not that 4 is correct**. The one genuinely independent leg comes
from the artefact: `items/env/trace_manifest.json` enumerates ranks 0–3, four entries,
each with its own bytes/sha256/event count — **and the four per-rank `gpu_kernels`
sum to exactly the 823736 the validator reaches by re-parsing the traces from
scratch.**

**And the sharper framing:** both runs were `-noval`, so `check_trace_coverage`
**never executed during the run at all**. The counterfactual *"would it have refused"* is therefore **unreachable from the
run in either direction** — there is no in-run verdict to compare against, only
`check_nothing`. **Knowing the launch value does not make it answerable**; offline
grading establishes what the validator says now, and nothing about what it would have
said then.

*The counterfactual is not answerable merely because the launch value is known.*

#### Why this is one item and not five

The package owner's pairing, and it is the reason this is filed rather than mentioned:

| incident | the same fact |
|---|---|
| `produced_by.commit = 'unknown'` in every circulating workset | the value was knowable, nobody supplied it, **and nothing recorded that nobody had** |
| `expect_ranks` wrong in the 78-minute `-noval` baseline | invisible for an afternoon; found only by turning a validator on |
| the 2c class — a var with a mocked value and a real one | carrying the wrong one produces refusals that read as producer defects |
| `--var gpu_devices` inert across five launches | undetectable **because the default matched what everyone wanted** |
| this one | a PASS and a REFUSAL differ by an argument no artefact carries |

**Every one is "the launch line is not recoverable from the artefact."** Counting them
as five incidents is what kept them separate; **the fifth was found by grading, the
fourth by a collision, the first by an audit — three different accidents for one cause.**

#### Options, none decided here

- **Render the staged copy**, so the package a run actually used says what it used.
- **Write the resolved variables into the run record**, e.g. a `store/run.json`, and
  let every consumer — validator, grader, auditor — read them from there.

**Both change a contract, so both defer like T75/T76/T77. The item does not:** it was
unfiled, not deferred, and the difference matters because nothing was tracking it.

#### One consequence already visible

Offline grading must report a verdict that depends on a supplied `--var` as
**conditional**, naming the value and where it came from. `grade_offline.py`'s
docstring says so and prefers values read from the run's own artefacts
(`environment.yaml`'s `tp_size`) over remembered ones. **That is a mitigation, not a
fix — it narrows who guesses, not whether anyone does.**
