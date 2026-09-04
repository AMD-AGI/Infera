# `llm_e2e_performance_optimization` — deferred work

Opened 2026-09-03 alongside the `e2e-flow/` refine. Every item here was **named
and deferred by the mission**, or carried over from the 2026-09-02 effort with a
measured symptom. Nothing is here because it was forgotten.

Format: what · why it is not done · what would settle it.

---

## From `mission.md`, deferred by the mission itself

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
(`env_mgr/grants.py:340-355`, which names this as a hole and declines to close
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
`integration-demo/assets/bench/pythonpath/sitecustomize.py` already injects code
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

## Carried over from the 2026-09-02 effort

### T7 — the comparability gate at bring-up (was E9′)

The two-arm design controls for session, node, trace, order and image, and
**not for node load at measurement time**. `check_service_live` proves a
deployment is *live*, not that it is *comparable* to the other arm's.

Measured: a patched arm at 475.7 ms mean ITL against a stock control at
470.3 ms — 1.1% apart — while the recorded run showed the same two arms 12%
apart, because they were measured fifteen minutes and one co-tenant apart.

**Do not widen the bars.** 5% / 10% were measured to be right: the within-arm
round-to-round spread on a steady node is ~2%. A previous round widened them to
35% / 30% in response to this and that was the wrong response.

**Would settle it:** a quiet-node baseline, or interleaving the two arms'
measurements round for round.

### T8 — `seal_refused` has no reader (was C9b)

A correct refusal is computed and then discarded. The 2026-09-02 run's refused
`integration_report` is the worked example: the verdict was right, it stopped
the graph as designed, and the artefact that records *why* is not read by
anything downstream.

### T9 — `env_mgr.fs.layout` has two `copy_out` functions (was E14/C-two-copy_out)

One verifies the tree it copied and one is a plain `shutil.copytree`. The
consuming path uses the plain one, so **nothing on the way out of a zone
verifies a digest**. Recorded in `temp/bugs/`.

### T10 — `check_workset_runs` hard-fails on rsd while `min_pass_ratio` forgives correctness (was C24)

A saturated node fails `max_rsd` on evidence that is otherwise correct, while a
kernel that is wrong on a minority of shapes can still pass. The two knobs
express opposite philosophies in one validator.

### T12 — the mock cannot exercise M5.4's ad-hoc correctness rules
*Opened 2026-09-03 by m5.*

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
*Opened 2026-09-03 by m5.*

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

### T11 — ten closed `items_schema`s in `integration-demo` (was C23)

`additionalProperties: false` on an items schema rejects `logs` and `watchout`,
which the content type itself lists as optional. Harmless until somebody adds a
log.

### T14 — a task body cannot name the interpreter the run is using
*Opened 2026-09-03 by m1, confirmed by m2, m3 and m4 independently.*

`cli/main.py:668` exports `AGENT_SYS_DEMO_PYTHON` into **`validation_env` only**,
and the comment above it says a task body never reaches it. So a task body's
policy `PATH` resolves `python3` to `/usr/bin/python3`, which on this host has
`yaml` and `jsonschema` and **not `referencing`** — and had no `torch` at all,
which is what m3's `build_workset` entrypoints actually need.

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
so the phase reads a broken validator rather than a refused handoff — measured
by m2, and it is the same signature as `check_deploy_serves`'s crash. Twelve of
twenty-one validators had it.

### T15 — `E2E_KIT_ENGINE_EXTRA_ENV` is a seam with no consumer
*Opened 2026-09-03.*

`deploy_kit.layout.yaml`'s `runtime_contract` requires a kit to honour both
`E2E_KIT_ENGINE_EXTRA_ARGS` and `E2E_KIT_ENGINE_EXTRA_ENV`. The first is used —
it is how m2's two lines differ by CUDA graph on/off. **The second has no
consumer**: both m2 lines leave it empty, and the profiler-attached line needs a
*router* flag (`E2E_KIT_ROUTER_EXTRA_ARGS`) that no engine seam can reach.

It was asked for on the strength of `SGLANG_TORCH_PROFILER_DIR`, which **nothing
in this package sets** — the engine is told where to write per capture, in
`/start_profile`'s `output_dir`. m2 gave the example and retracted it; by then it
had propagated into two contract documents.

**Kept and labelled rather than removed**, because a required parameter is
cheaper to keep than to re-negotiate, and because the argv/environment
distinction is real even though this instance of it was not. **Would settle it:**
the first real consumer, or a decision to drop the requirement.

### T16 — `git`'s `index.lock` retry is not idempotent, and its no-op is silent
*Opened 2026-09-03 by checkpoint, against a collision with the leader.*

`CONTRACT §8a` told an owner whose commit hit `index.lock` to wait a second and
retry. Measured: **the retry can silently do nothing.** In the seconds between
the failure and the retry, another owner's commit named a tree and took the
first owner's dirty file; the retry then found nothing to commit for that path
and **said so by exiting quietly**.

The owner reported *"T+60 is committed"* and it was not — by them. `3b2ffde` is
the artefact: one owner's subject over 187 lines of another's file.

**§8a's own verification step did not catch it**, which is the part that made it
survive twenty minutes: `git show --stat --name-only HEAD` printed exactly the
expected path, because HEAD was somebody else's commit **holding that path**.
**Confirming the path is not confirming the commit.** Fixed in §8a — the check
now leads with `git log -1 --format='%h %s'`.

**Would settle it properly:** one worktree per owner, which is the structurally
clean answer §8a declined in the morning because work was already in flight in
one tree. That reason no longer holds as strongly — every module is complete and
the remaining work is runs rather than edits. **Worth doing before the next
effort, not during this one.**

### T17 — the model-specific engine flag groups are free-form strings and cannot be checked
*Found by m1 on 2026-09-03, comparing the sealed GLM-5.3-Flash recipe against
what `shared.yaml` can express.*

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
*Found by m1 on 2026-09-03, running a **second** model's kit. Qwen's kit could
not have shown it.*

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
like something else** — the router failed with `ConnectError: All connection
attempts failed`, naming neither the port nor the mismatch. m1 committed exactly
this mistake while writing the fix for the other three.

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
*Found by m1 on 2026-09-04, composing rung 1 against `crsuse2-m2m-249`.*

Every other identifier this package binds on a shared host has a `--var`:
container name, three ports, the container workdir. **The GPU set does not.**
`E2E_TP` is a *count*; the *index* is left entirely to the agent's `rocm-smi`
read at `deploy_and_prove.task/readme.md` STEP 1, whose criterion is only that
*"you can say which device index you are taking"*.

The producer brief's own trap list already names the GPU index alongside the
others — *"container names, host ports, the container workdir and the GPU index
sit in one namespace with everybody else"* — so the property is agreed and the
parameter is simply missing.

**Measured on the node this matters on.** GPUs 0–3 hold ~300 GB each with no
`docker ps` entry behind them (the co-tenant class m1 refused to `kill -9`);
4–7 are free. An agent that takes the default devices takes 0–3 and OOMs against
another tenant's work — and the failure surfaces as a bring-up problem, not as a
placement problem, which is the expensive kind.

**What was done instead, and why it is a workaround rather than a fix:** the
sentence *"GPUs 0-3 on this node are held by another tenant … take only 4-7"*
was routed through `--var instruction=`, which is the declared channel for a
plain-words site fact and the right refusal to change the package mid-rung.
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
with. Verified green on the node before rung 1: `HIP_VISIBLE_DEVICES=4,5,6,7`
inside the built image yields `torch.cuda.device_count() == 4`, so the mechanism
works — it is only unparameterised and unchecked.

### T20 — a container left on a node we no longer hold, and the cleanup design that would have prevented it
*Left by m1 on 2026-09-04. Recorded as a debt rather than a task, per the leader.*

**The artefact.** On `crsuse2-m2m-249`, possibly still running:

```
name    yihou_e2e_sgl_m1real-20260904
image   infera/engine-sglang:m1-rung1-20260904
cmd     sleep infinity
labels  infera_e2e_run=m1real-20260904 · m1_probe_container=true
holds   GPU 4 visible (HIP_VISIBLE_DEVICES=4), zero GPU memory, no published ports
```

It was stood up so m4 could close the one unverified line in `run_in_container.sh`
— the `docker exec` itself — without m4 acquiring a container lifetime that
CONTRACT §5 puts with m1. It did that job: m4's exec returned `EXIT=0` with
`torch 2.11.0+rocm7.2` and the sampler hash matching m1's reference run.

**Why it is still there.** Job `108891` was **CANCELLED at 05:13:17**, 1 h 21 m
into an ~8 h hold and **38 seconds before** the `docker rm`. `spur exec` then
refused: *"job 108891 is not running (state: CANCELLED)"*. The node is now
allocated to job `108943`, a different user.

**It is deliberately not being cleaned, and that is the right call.** Taking a
hold on a node specifically to reach into it while another tenant is working
there is worse than the thing being cleaned up, and the rule that says *never
`docker rm -f` what you did not create* protects that tenant from us in exactly
the same way it protects us. If a hold on 249 returns, clear it then.
Whether the container survived the cancellation is **unmeasured** — nobody has
checked whether this scheduler reaps containers at job teardown.

**The design error, which is the part worth keeping.** The container was created
with an explicit *"teardown is mine and I am holding it; no timer, because the
only thing that should end it is somebody saying the verification is done."*
That reasoning **assumed the node would outlive the decision.** It did not.

The fix is not a timer — a timer would have been wrong too, and would have cut
m4's verification. The leader's framing is the general one:

> **A borrowed resource's cleanup has to be idempotent and unowned**, so that
> anyone with access can do it and nobody has to be alive to decide.

Concretely, for anything this package stands up on a node it does not own:

1. the container carries a label naming the run (this one did — that is why it
   is identifiable at all, and it is the only reason this entry can be precise);
2. **a reclaim pass keyed on that label is runnable by anyone**, at any time,
   with no knowledge of who created what — `assets/lib/reclaim.sh` is the
   existing place for it, and CONTRACT §5.0 already requires bodies to call it
   in a `finally`. What is missing is the case where the *creator* never gets to
   run its `finally`;
3. so a run's **first** act on a node should be to reclaim the labels of runs
   that are provably over, and a hold's **last** act should not be the only
   chance. Cleanup that depends on a specific process still being alive is the
   same class as a rule that depends on someone remembering.

Pairs with the reclaim finding already in `check_deploy_serves`'s history: a
teardown that crashes warns that ports may be held, and the ports it names may
belong to somebody else's live run. Both are about cleanup needing to be safe
for a stranger to run.
