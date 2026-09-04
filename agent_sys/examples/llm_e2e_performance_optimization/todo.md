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

### T21 — the completion probe grades shape, not answer, and the bar that would fix it needs a measurement
*Found by the leader on 2026-09-04 reading rung 1's completion output; localised by
m1, who owns the probe. The `direction` text is corrected in the same commit as
this entry — that half needed no measurement. The bar does.*

**What passed.** Rung 1, Qwen3.6-27B, no `--reasoning-parser`:

```
finish_reason : stop
usage         : {prompt_tokens: 23, completion_tokens: 157, reasoning_tokens: 0}
content       : "Here's a thinking process:\n\n1.  **Analyze User Input:** …"
```

157 tokens of chain-of-thought in `content`, `reasoning_tokens: 0`, to the prompt
*"What is the capital of France? Answer with one word."*

**Why it passed.** `probes.yaml`'s `completion_nonstreaming` asserts
`status: 200`, `finish_reason equals stop`, `content nonempty: true`, and
`model not_matches ^/`. All four hold. **`nonempty: true` was written against a
parser that removes too much and is structurally blind to one that removes
nothing.** The probe's `direction` claimed it discriminated the reasoning-parser
fault; it discriminates one direction of it.

Same rotation as the GLM finding in `fa49319` — *"had the parser been wrong,
`content` would have been empty on a request that still returned 200"* — with the
sign flipped and nobody having thought to flip it.

**Three fixes considered and rejected, each for a failure this effort has already
paid for:**

| candidate | why not |
|---|---|
| match `Paris` in `content` | **does not discriminate.** A reasoning preamble ends with the right answer, so it passes both ways |
| bound `content` length | discriminates, and the bound would be a number invented from **one observation** — the 35 % / 30 % widening of the previous round, pointed the other way |
| require `usage.reasoning_tokens > 0` | precise about the property, and **refuses a legitimate non-reasoning model.** It asserts a fact about the model while claiming to test the deployment |

**Proposed shape, needing one measurement before it is written:**

> `usage.reasoning_tokens > 0` **OR** `content` is short —
> *either the reasoning was accounted separately, or there was none to account.*

A statement about **the parser** rather than about the model, and it fails in the
loud direction. "Short" is the number nobody has.

**The measurement that would set it**, small enough to ride along with a future
rung rather than needing its own: send this exact one-word prompt to (a) a
reasoning model **with** a correct `--reasoning-parser`, and (b) a non-reasoning
model, and record `len(content)` for each. The bar goes between them, nearer (a).
Two requests against a deployment that exists for another purpose. **Until that
is taken, do not invent the number** — a validator bar chosen on the login node is
the artefact-tuned-to-the-instrument mistake this package refuses elsewhere.

**Related, and separable:** `E2E_PARSER_ARGS` defaults to `none`, which is correct
only for a non-reasoning model. Qwen3.6-27B is not one — its chat template carries
`<think>` ×4 and `</think>` ×5, and the built image's `ReasoningParser.DetectorMap`
offers both `qwen3` and `qwen3-thinking`. **Which of the two is right is
untested**, and T17 is why that matters: a free-form string accepts
`qwen3-thnking` as happily as `qwen3` and the wrong one produces no error.
Setting the default is not this entry's fix — a correct default would have
*hidden* the probe's blindness rather than removed it.

### T22 — `E2E_STAGE` names a per-stage fact and `--var` carries one value per run
*Found by m1 on 2026-09-04 while declaring the variable `check_agent_env.py`
flagged. Declaring it was correct and does not make it right.*

`env_render.py:173` stamps every tolerated difference with
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

### T23 — `fixed.gpu_count` is the one required field with no definition, and 8 was defensible
*Found by checkpoint in rung 1's own handoff, verified by the leader, localised
by m1 who owns the producer. Third direction on T19.*

Rung 1's record says `gpu_count: 8` on a node where four cards were held by a
co-tenant at 96–98 % VRAM. All 21 verdicts passed.

**The framing this arrived with is not how the code works, and the correction
matters because it changes the fix.** It was reported as *`gpu_arch` and
`image_id` are discovered at bring-up; `gpu_count` is a claim*. Measured —
`env_render.py:64-66`:

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
**cards this deployment could use** — and `8` is *true* under the first. The
agent was not wrong. **A field that cannot be wrong cannot be a measurement**,
and `fixed` is promised as 可固化环境, which is the second reading.

**Deliberately not proposed: adding `gpu_count` to `check_environment`'s
`compare_fixed_across_inputs`.** Those four fields are four on purpose and the
DELIVERY-NOTE is explicit that bars are not widened. A cross-input comparison
would also not have caught this — every stage would have agreed on the same
undefined 8.

**What would settle it, and it is one decision, not three:** say which reading
`fixed.gpu_count` requires, in the schema, in a `description` like every
neighbouring field has. Then the producer brief gets a STEP 1 criterion that
generates it, and `check_deploy_kit` can check the record against something.

**T19's `fixed.gpu_devices` makes the decision cheap rather than forced**,
which is why these are one problem: with a device list, `gpu_count` keeps the
node fact and `len(gpu_devices)` carries what the deployment used, and neither
reading has to lose. Without it, whichever meaning is chosen makes the other
unrecordable — and the run above needed both to be honest: *eight present, four
usable, one taken.*

Not blocking. Recorded rather than fixed **because the fix is a definition and
the definition is entangled with T19** — writing a criterion first would bake in
whichever meaning m1 happened to pick.

---

### T24 — a fallback nobody can reach is still a second reader of the number

**m5. Not blocking; the live half is fixed and this is what is left of it.**

`assets/accept/measure.sh:120-122,184-185,315-316` reads its load shape as
`${E2E_MAX_CONC:-256}`, `${E2E_WORKERS:-16}`, `${E2E_BLOCK_SIZE:-512}`,
`${E2E_REQ_TIMEOUT:-900}` and `${E2E_TRACE_END_MS:-120000}`. `shared.yaml`'s
`runner` declares 32, 8, 512 and 900; `e2e_integrator` now declares the same
four, plus 60000 for the trace window, deliberately.

**The live defect is fixed.** Until those declarations landed, none of the four
reached this stage at all — a name only `runner` declares does not reach a
`kind: ai` agent (`env_mgr/material.py:96`) — so the script's fallbacks won and
m5 replayed at **concurrency 256 against m2's 32**, with `--var max_conc=` inert
on one side of a comparison M5.1.3.1 requires to hold within
`stock_vs_m2_tolerance`. Found by re-running the leader's omission check over
m5's manifest after `shared.yaml` grew; the checker's first pass named only
`E2E_REMOTE_HOME` because these four were not yet on `runner`.

**What is left is three numbers for one knob** and no way to tell which is
intended: `E2E_TRACE_END_MS` is 180000 on `runner`, 60000 here, 120000 in the
script. The declaration wins wherever the graph runs the script, so the
fallbacks are unreachable *there* — but `measure.sh` is also meant to be run by
hand, which is the case the fallbacks exist for, and by hand it measures
something the graph never would.

**Not reconciled here, because the right value is a measurement and not an
edit.** 256/16 was the shape some earlier run wanted; 32/8 is what `runner`
carries now; nobody has said which the two-arm comparison should use, and
picking one in a comment is how a number acquires a third reader. What settles
it: one owner states the offered load the m2-vs-stock comparison assumes, once,
and both the declaration and the fallback cite it.


---

### T25 — a run records the environment it minted, but not the vars it was started with

**m2. Not blocking. Cheapest item on this list, and it would have prevented three
incidents on 2026-09-04 alone.**

Every `--var` a run is launched with shapes what the graph does, and **none of
them survive into the run tree.** `handoffs/<hid>/v<N>/content/items/*/
environment.yaml` records the environment m1 *minted* — node, image, image_id,
tp_size — which is a fact about the deployment, not about the request. The
store keeps tasks, events and handoffs. Nothing keeps the command line.

So *"what was this run asked to do"* is unanswerable from the artefact, and it
is asked constantly, by people who were not the one who typed it.

**Three questions on one day that this record would have answered**, each of
which instead cost a message or a run:

1. **Was `--var expect_ranks=2` passed?** `check_trace_coverage` is `strong` and
   declares `${expect_ranks:-8}`; the mocked trace is TP-2 while the real
   deployment was TP-4 — three numbers, and `expect_ranks` is deliberately not
   derived from `${tp}` (`steps/m2_profiling.yaml:93-100`). I predicted a
   refusal I could not check; the leader had passed it. **The prediction was
   unverifiable, not wrong** — and the cost of asking was the same either way.
2. **Was the run still alive?** It had been killed 20 minutes earlier and the
   tree does not say so. Two owners reported it as live and one committed that
   into a checkpoint.
3. **Was its agent still alive?** Also not recorded, and the answer was yes —
   see T26.

The same gap in the other direction is the more expensive half: rung 0 returned
`check_deploy_kit: FAIL` on a stage that had been green, because `--var image=`
named a tag present on the node instead of the one the sealed kit renders. The
validator refused correctly. **Believing that failure would have sent two owners
auditing their commits for a defect that was in a command line** (CONTRACT §4.4,
face 2) — and no reading of the run tree could have distinguished the two.

**What would settle it:** the run writes its resolved variables — every `--var`
plus every default that was taken — into the run root at launch, once. Resolved
rather than raw, because a default that was *taken* is exactly the case nobody
can reconstruct afterwards. `spec_loader/variables.py` already computes it; the
value is thrown away after substitution.

**Not fixed here**: `agent_sys/cli/` and `agent_sys/spec_loader/` are outside
this effort's activity scope. Recorded with the three instances so whoever owns
the runner has the argument as well as the request.

---

### T26 — killing a run does not kill its agents

**m2, from the leader's incident of 2026-09-04. Not blocking; the instance is
stopped. It changes what "I killed the run" can be relied on to mean.**

**One level up from `41c8540`.** That record says a cancelled Slurm job does not
reclaim its GPUs, because the containers talk to the **host** docker daemon and
are therefore not in the job's cgroup. The same shape holds one layer further
out: **an AI agent a run dispatched is not in the orchestrator's process tree**,
so killing the orchestrator leaves it running.

Measured. The leader ended run `20260904T062414-be315b` at 06:47. At 07:06:50 —
twenty minutes later — the run tree was still being written:

```
files written since 06:47   29 -> 30      (two readings, two minutes apart)
newest write               07:04:29 -> 07:06:40
```

under `zones/task.…/task.…/task.…/config/projects/…/<uuid>.jsonl`, an agent's
own transcript: **490 entries, the last an `Edit` tool call**. All three task
records in that chain still read `status=running`. The leader then found it
directly — `PID 1097159`, 44:42 elapsed, `claude … --system-prompt "#
deploy_and_prove — deploy the …"`, cwd inside that zone — and sent SIGTERM.

**What it did while unattended.** It created container
`infera_e2e_sgl_m1-20260904` at **06:57:38, ten minutes after the kill**, and
replaced the `selftest` pair with `infera_e2e_sgl_kitselftest-m1` at 07:06:24.
Both were attributed to a person; both were the orphaned agent. **Four owners
were each asked whether the containers were theirs and each correctly answered
no** — the object was ours the whole time, and no one could see it.

**Why this is not merely untidy.** RUN-PLAN's own rule is that the first real
run of a `kind: ai` closure happens with the leader watching, *because* an AI
agent with a live node and a docker daemon is the one thing here that can change
state nobody asked for. It was unwatched for twenty minutes **precisely because
everyone believed the run was dead** — the belief that made it unsupervised was
caused by the same event that orphaned it.

And nothing it produced could ever land: the orchestrator was gone, so no
handoff could be sealed and no task could progress. It held a node and mutated a
shared host for an outcome with no consumer.

**What would settle it:** a run that is ending terminates the agents it
dispatched, and says how many, before it reports that it stopped. Failing that,
the run root records agent PIDs at dispatch so a person can check — which is
T25's record in a second use.

**A note on how it was found, because the method is the transferable part.** The
first probe reported *"zero files modified in the run tree in the last eight
minutes"* and it was **not a measurement**: `find` on the login node is `bfs`,
which does not support `-newermt` and errors to stderr, while stdout was piped
to `wc -l` and the empty result read as zero. A check that could not fail.
What worked instead was opening a file and counting its entries — **a file you
have opened cannot lie to you about whether it exists.** CONTRACT §4.4, in the
instrument built to catch §4.4.

### T27 — a default that takes everything free is not a default
*m1, 2026-09-04. **T19's third direction, and the one that stopped being an
argument and became an incident.** Three items, one cause; they belong together
because fixing any two leaves the third failure available.*

**What happened.** The orphaned `deploy_and_prove` agent (`T26`) re-ran its own
kit's `deploy.sh` at 06:57:38 **without the two safety variables its earlier
invocation had passed** — no `E2E_KIT_NAME_PREFIX`, no `E2E_KIT_GPU_DEVICES`.
`env.sh:230`:

```sh
: "${E2E_KIT_GPU_DEVICES:=$(_pick_gpus)}"
```

`_pick_gpus` returned **every free card**, the worker took all eight, and it
collided with a container that had *already named its own four*.

**The victim was the well-behaved container.** Of the three deployments on that
node, the selftest pair was the only one that had declared `HIP_VISIBLE_DEVICES`
— and it is the one that got stepped on. That is the strongest argument for
pinning there is: **declaring your cards protects you from nothing if the next
process declares nothing**, because "nothing" means "all of them".

#### The three items

**1. Pin the container, not the worker process.** `mix_worker.sh:89` sets
`HIP_VISIBLE_DEVICES="$GPUS"` inline on the `exec`, so it binds the processes
that carry it and nothing else. `docker inspect` on such a container shows no
device restriction at all — which is exactly how this one was first read as
"unpinned and greedy" when its *worker* was in fact pinned. Pin at `docker run`
and **the record and the runtime agree by construction rather than by the worker
remembering.**

**2. Bound `_pick_gpus` to `tp_size`.** A picker asked for a deployment of width
N should return N cards. Returning everything free is not a conservative default,
it is a land grab that happens to be quiet on an empty node and hostile on a
shared one. Note the shape: **it is `E2E_DSA_ARGS`'s trap inverted** — there the
danger was a value that is wrong and does not error; here it is an *absent* value
that is filled in maximally and does not error.

**3. Record what was picked** — *m3's item, and the one that makes the other two
checkable.* `fixed.gpu_count` records **how many** and cannot record **which**,
so two runs on disjoint halves of one node produce identical records. The cost is
not hypothetical: reconstructing who owned which card today required
`rocm-smi --showpids` and PID matching, twice, by two different people. Whatever
`_pick_gpus` decides must land in `fixed.gpu_devices`.

**Why all three.** Pin without bounding and a caller who passes no list still
grabs the node — pinned, but pinned to everything. Bound without pinning and the
container still sees cards the worker was told to avoid. Do both without
recording and the next person attributing a card is back to PID matching. **T19
is the field; this is the three things that have to be true for the field to mean
anything.**

#### Item 1 has been met once, unprompted — recorded because the run that did it produced no verdict

m1's 2026-09-04 standalone on 217 was killed before any validator ran, so it has
no verdict and is not a pass. **Three facts survive it, read from artefacts, and
this is the one that belongs here:**

```
docker inspect …_sgl_e2e-main-20260904
  Config.Env   HIP_VISIBLE_DEVICES=0,1,2,3     <- the CONTAINER, not just the worker
  a new process inside: torch.cuda.device_count() -> 4
```

**The first kit whose container is pinned rather than only its worker process**,
and the operator's `--var gpu_devices=0,1,2,3` was obeyed exactly — the
*obey-or-stop* branch running for the first time, with the value landing in
`fixed.gpu_devices`. The two kits before it pinned the worker only, which is why
`docker inspect` showed no restriction on the container that took eight cards.

**It was immediately load-bearing rather than cosmetic.** m4 was about to `docker
exec` in, assumed they would land on the free 4–7 because those read 0%, and
would have landed on **0–3 beside a live engine** — the pin is what made that
knowable in advance instead of a confusing measurement afterwards.

*No producer instruction asked for this; the agent did it on its own. So item 1
is demonstrated possible and still unspecified — a brief that required it would
make it reliable rather than fortunate.*

> **⚠ "by construction" is wrong, and m4 found it within the hour. Corrected
> here rather than edited away.**
>
> The pin is an **environment default, not a device whitelist**.
> `start_container.sh:37-44`:
>
> ```
> --device /dev/kfd
> --device /dev/dri                                  <- EVERY card on the host
> --env "HIP_VISIBLE_DEVICES=${E2E_KIT_GPU_DEVICES}" <- the pin
> ```
>
> **Every card's device node is exposed and only an env var says which to use**,
> so `docker exec -e HIP_VISIBLE_DEVICES=4` overrides it and runs on a card the
> deployment was never allocated. It does not fail; it returns a number.
>
> **So item 1 as met is strictly better than pinning the worker and is not
> enforcement.** A new process inside inherits the right default instead of
> seeing all eight — that is the real gain, and it is what made m4's landing
> spot knowable in advance. But *"record and runtime agree by construction"*
> overstates it: a convention a later process can override is not construction.
>
> **What construction would look like:** `--device /dev/dri/renderD<N>` per card
> rather than the whole `/dev/dri`, so the cards the deployment did not take are
> not present in the container at all. Unmeasured — nobody has checked whether
> this cluster's `spur-authz` accepts per-card device flags, and **that is the
> question to answer before anyone writes it into a brief.**
>
> **KEEP BOTH. Do not replace the env pin with the whitelist** — m4's coupling,
> and the paragraph above would have caused a regression without it.
>
> m4's `run_in_container.sh` refuses an exec whose requested card is outside the
> container's own `HIP_VISIBLE_DEVICES`. **A container pinned only by device
> whitelist has no such variable**, so that check finds nothing to compare, falls
> into its *unpinned, constrains nothing* branch, and **waves the exec through**.
> The work would still not run on a card that is absent — but it would fail
> **deep inside HIP** instead of being refused by name. *Safe by absence rather
> than safe by refusal*, which is the exact direction this effort has spent the
> day converting failures **out** of.
>
> So the two are not alternatives:
>
> | | what it gives |
> |---|---|
> | `--env HIP_VISIBLE_DEVICES` | a **readable statement of intent** a consumer can check against and refuse by name |
> | per-card `--device` | **enforcement** a later process cannot override |
>
> **Verified on a live container, 2026-09-04** — not read off the script:
> `HostConfig.Devices` is `/dev/kfd` and `/dev/dri` whole, with
> `Config.Env HIP_VISIBLE_DEVICES=0,1,2,3`. Every card's node present, one
> variable narrowing it.
>
> *Recorded because a correction that would itself cause a regression is worth
> more than the correction: I wrote "whitelist is the real answer" and m4, who
> reads the field I would have removed, is the only person positioned to notice.
> Their wrapper now carries the same coupling as a note beside the check, so the
> two point at each other and neither can be changed in ignorance of the other.*
>
> Until then the honest division is m4's: **the kit states the intent, and the
> consumer refuses to violate it.** Their `run_in_container.sh` check reads the
> container's own `HIP_VISIBLE_DEVICES` and refuses a request outside it —
> which is the mirror of `start_container.sh:67`'s inside-out check, and is
> where the enforcement actually lives today.

#### Item 4, added 2026-09-04 after the entry above was demonstrated on a live node

**The pick must come from the probe, not sit beside it.**

Items 1–3 assume the device set is *chosen*. The 07:16 run showed it need not be.
That agent's kit contained **no `_pick_gpus` at all** — `env.sh:62` is a literal

```sh
: "${E2E_KIT_GPU_DEVICES:=0,1,2,3}"
```

**and the agent had probed the node at transcript record 92 before writing it.**
It ran `rocm-smi`, read the output, and then hardcoded the first four cards. The
probe informed its *narrative* and not its *parameter*.

**This is strictly worse than the land grab it replaced**, and worth stating
plainly because "bound to `tp_size`" would have called this kit compliant: it
takes exactly four cards, matching `E2E_TP=4`. Item 2 is *satisfied* here. The
deployment still bound four cards a co-tenant was mid-load on, and rose them from
198 GB to 220 GB before it was stopped.

**Nothing checked.** `deploy.sh:61` prints `preflight ok: PORTS free, NAMES free`
— it preflights the port band and the container names and **never looks at a
card**. A `rocm-smi` reading that does not reach the variable is decoration.

**Two agents, two kits, two device policies, neither validated.** That is the
finding above the incident: **the device policy is not a property of this package
at all.** It is whatever the day's agent writes, and no validator, schema or
brief constrains it. Items 1–3 fix the kit; item 4 says the kit is not the unit
of repair.

**What would settle it:** STEP 1's criterion must make the probe's output the
*source* of `fixed.gpu_devices` and of the kit's default — a named command whose
result is the value, not a reading the agent is invited to consider. And
`deploy.sh`'s preflight must refuse a card it cannot verify free, on the same
footing as the port band it already refuses. **The port check is the model: it
exists, it aborts rather than waiting or stealing, and it was written by the same
agent on the same day.** The cards simply were not thought of as a namespace.

*Corrected while writing this: `env.sh:99`'s comment — "checks each is free and
aborts rather than waiting or stealing" — is **about the ports**, and it sits two
lines from where a grep for card handling lands. It was nearly reported as a card
check that exists. Opening `deploy.sh` gave the opposite answer.*

**The documentary proof, found after the above was written, and it is sharper
than the claim it supports.** The kit's own `results/preflight.json` contains
**the true reading and a false claim derived from a stale one, in the same
file**:

| | |
|---|---|
| structured, `gpu_cards[]` | all eight cards **198–199 GiB used, 89–90 GiB free** |
| prose, `gpu_devices_rationale` | *"All eight were free (**<=300 MB used each, no co-tenant**)"* |
| prose, `vram_headroom_note` | *"**~288 GiB free per card** … mem-fraction-static 0.85 leaves ~232 GiB"* |
| `captured_at` | **absent** |

The prose is wrong by a factor of three against the numbers two keys above it.

**So the agent did re-probe, and the late reading reached the record's structured
half and neither the prose nor the parameter.** That is worse than "the probe
informed its narrative": the narrative was written from the *early* reading and
never revised, the numbers were refreshed and never reconciled, and **with no
timestamp on either half a reader cannot tell they describe different moments.**

The decision was not careless — `gpu_devices_rationale` reasons that *"0-3 was
taken so that the low half is this deployment's and 4-7 stays available for the
step-6 self-test"*. **A considered rationale resting on a stale premise, and the
evidence refuting it is in the same document.**

**Nothing catches this.** `check_deploy_kit` grades the layout and validates
`environment.yaml` against its schema; it does not read `preflight.json`, and no
rule anywhere compares a document's prose against its own numbers. The layout's
`results/` floor asks for *two non-empty `.json` files* — this file satisfies it
while contradicting itself.

**A fifth item follows and it is cheap:** any evidence file that records a
measurement must carry **when it was taken**. A reading without a timestamp
cannot be known to be stale, which on this cluster — where a card reading is true
for seconds — makes it indistinguishable from a guess.

> **⚠ The paragraph above is wrong, and the table above it has one wrong row.
> Corrected by m1 within the hour, before anything was built on it.**
>
> **`captured_at` is not absent. The file carries `measured_at`, at the top
> level, and its value is `2026-09-04T07:26:58Z`.** I searched the document for
> the key name I expected instead of listing the keys it has, which is the third
> time today a search for an anticipated string returned the wrong answer where
> opening the data returned the right one.
>
> **This makes the finding simpler and worse, not weaker.** 07:26:58 is *after*
> the co-tenant arrived at 07:19:39 — so the `gpu_cards[]` array is **correct for
> the moment the file declares**, and the prose contradicts numbers captured at
> that same declared moment. It is not two readings in one file with nothing to
> tell them apart; it is **one honest reading and a conclusion that ignores it.**
>
> **So item 5 as scoped is void — the timestamp was already there and did not
> help.** The defect it was aimed at is real and needs a different fix:
>
> **Item 5, restated: nothing compares a document's conclusion to its own
> numbers.** `check_deploy_kit` validates `environment.yaml` against a schema and
> counts files in `results/`; it never opens `preflight.json`, and no rule
> anywhere reads a prose field against a structured one beside it. The `results/`
> floor asks for two non-empty `.json` files, **which this file satisfies while
> contradicting itself.**
>
> **And it is not expressible as an evidence rule today**, which is the honest
> obstacle rather than an excuse. The existing rules are regexes over a
> directory (`forbid`, `require_each`, `require_together`); *"this sentence
> disagrees with that array"* is not a regex. Tested before claiming it: a
> directory-level "some file carries a `*_at`" rule passes **both** the sealed
> kit (5 of 14 files) and this one (1 of 1), so it discriminates nothing.
>
> What would actually catch it is narrower and belongs to the producer: **STEP 1
> should require the rationale to cite the numbers it rests on** — free VRAM per
> card, quoted from the same reading — so that a stale premise is visible in the
> sentence rather than only in the array two keys above it. A conclusion that
> restates its evidence cannot silently outlive it.
>
> **Written, `deploy_and_prove.task/readme.md` STEP 1** — approved on its own
> merits after item 5 was withdrawn.
>
> **Where the class belongs.** The three instances behind this entry —
> `captured_at` searched for where the file says `measured_at`; transcript greps
> matching *file reads* and reported as *decisions*; `env.sh:99`'s comment about
> **ports** read as a comment about **cards** — are the *search* half of the
> pattern CONTRACT §4.4's observer section and m2's fourth face already describe.
> **One class, six people, one day:** m3's `ImportError` "pass", m2's verdict
> reader, the leader's `bfs` predicate, m4's `/proc` scan, and these three.
> The shared shape is **an anticipated string standing in for the data** — and
> in every instance the remedy was the same and cheap: *list what is there
> instead of searching for what you expect.*

### T28 — T21's bar, measured: 7 characters against 526
*m1, 2026-09-04. T21 said "do not invent the number". The number exists now.*

Both engines were alive on `006` at the same moment — the kit-launched one with
`--reasoning-parser qwen3`, the agent's earlier ad-hoc one without — so the A/B
is one model, one node, one image, two `curl`s.

| | parsed (`:8101`) | unparsed (`:8118`) |
|---|---|---|
| `finish_reason` | `stop` | `stop` |
| **`len(content)`** | **7** — `'\n\nParis'` | **526** |
| `reasoning_content` | 554 chars | `null` |
| `reasoning_tokens` | **157** | **0** |
| `completion_tokens` | 160 | 151 |

**Both pass today's probe.** T21 demonstrated rather than argued.

**`completion_tokens` is not the discriminating axis** — 160 vs 151, the wrong
way round and inside noise. Had the rule been written against it, it would have
been a check that cannot fail, in the validator whose blindness it was written to
fix. That is CONTRACT §4.4's third face, caught before it was written instead of
after.

**The rule, now with numbers behind it:** `usage.reasoning_tokens > 0` **OR**
`len(content) <= 200`. Parsed passes on the first clause (157); unparsed fails
both (0, and 526). The separation is a factor of 75, so the bound is chosen with
room rather than fitted to one observation.

**The caveat, and it ships with the bar rather than after it:** *two points, one
model. A verbose non-reasoning model answering a one-word question in three
sentences fails a 200-character bound, and `reasoning_tokens > 0` does not
protect it, because such a model reports 0 too.* The bound is defensible and not
proven general, and this entry is where it says which.

**Not implemented yet, deliberately.** It needs two new things in
`probes.yaml`/`probe_runner.py` — a length assertion and a disjunction — and
**rung 1 was live in `check_deploy_serves`'s path when the measurement landed.**
Editing a validator whose body is being copied into a zone mid-run is how a run
gets a fault nobody can attribute. It goes in after the rung, not during it.

---

### T36 — `/proc` is namespaced under `spur exec`, so PID-based attribution from a node lies

*Renumbered from **T28**, which collided with an earlier item of the same number — six owners append to this file and two picked the same next integer. Commits and messages citing T28 for *this* item still resolve: the other T28 is a different subject and the two are not confusable by title.*

**m2, 2026-09-04. Not blocking. Recorded because three ownership misattributions
today were name-based, and the obvious fix — attribute by PID instead — is
broken in the one place people will reach for it.**

**The control is the finding.** On node 006, `docker top` reported PID
`3260888` running inside `kimik3-vllm-kimi-k3`. From a `spur exec 109260`
shell, `/proc/3260888` **does not exist**:

```
docker top kimik3-vllm-kimi-k3 -eo pid   ->  … 3260888 …     (via the daemon)
[ -d /proc/3260888 ]                     ->  NO              (via /proc)
```

So `spur exec` puts you in a PID namespace that cannot see the host's
processes. **A `/proc` miss there means "not visible from here", not "not
running"** — and the two are indistinguishable without a control.

I nearly reported the opposite. `rocm-smi --showpids` listed PID `90546`; it was
absent from `/proc`, and I was one sentence from *"stale, does not exist"*. The
only thing that stopped it was checking whether a **known-live** PID was visible
either — it was not.

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

### T29 — a crashed instrument and a refused artefact are the same value in `verdict.json`

**Owner m3, 2026-09-04, found by fixing the wrong half of it.**

`zone.py:132` writes `verdict.json` as `dict[str, bool]`. A validator that
**refused** and a validator that **could not run** therefore produce the same
value, and the graph reads no difference between them.

**Measured, twice in one hour.** `check_workset_shape` crashed on a
`ModuleNotFoundError`, wrote no verdict at all, and `operator_workset` came out
`invalid` — *a missing dependency reported as a judgement about the artefact*.
That is the false attribution `check_workset_runs` exists to prevent, arriving
one layer up, in the validator itself.

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

**Related and separate:** `temp/bugs/2026-09-03-a-validators-stdout-is-not-kept-anywhere.md`
is why the prose had nowhere to live either. `workset_io.write_report` now puts
it beside `verdict.json` for m3's two workset validators; the other validators
in this package still discard theirs, and a general fix belongs in `zone.py`,
which is not m3's file.

### T30 — a finding recorded against one stage is a question for every stage

**Owner m3, 2026-09-04. Three instances in one session, none of them noticed
by the person who had already read the record.**

`RUN-PLAN.md`'s var table said, about m1's stage: *"A validator declares no
agent, so the package's `env` block never reaches it… With `transport_env`
unset, `spur` has no `SPUR_CONTROLLER_ADDR`… It cost three rung-0 runs and two
wrong attributions."* **I read that paragraph the same day**, while checking
whether rung 0's stop was a stale claim, took it as history about deploy, and
never asked whether the same hole was in `check_workset_runs`. It was. It then
cost a fourth run, and m4 called it *"the third run this shape has cost"*.

Two more of the same: `require_visible_on_node`'s misattributing message is
shared by **six call sites across four owners**, and `_pick_gpus` taking every
free card was reported against m1 while m3's `:=4` default was the same
property in a different file.

**The remedy is not diligence, it is a grep.** When a defect is recorded
against any stage, the recording owner names the *construct* — a variable, a
helper, a default, a message — and every other owner greps their own files for
it before the next run. **A finding filed against one stage is unowned by
everyone else, and unowned is where it stays.**

### T31 — naming a class is not sweeping for it

**Owner m3, 2026-09-04. The operational half of `T30`, and it fires on your
own findings rather than other people's.**

At layer 4 of the `build_workset` stack I found `SPUR_CONTROLLER_ADDR` absent
from a validation zone, wrote the class down — *a variable present in my shell
and absent in a closed zone* — and **did not then grep my own file for other
ambient reads.** Layer 5, an hour later, was `$HOME` being `/home` in that same
closed zone, in the same file, producing `-v /home:/home` and a denial from
`spur-authz`. One `grep -nE '\$(HOME|USER|PWD|PATH)'` at layer 4 would have
found it.

The same session produced the same shape twice more: after fixing a fixture
that was more convenient than production, I built the next fixture **by
subtracting the variables I suspected from my own contaminated shell** rather
than taking the environment from the zone — and kept the one that mattered.

**Naming a class produces the search term. Doing the search is a separate
act,** and the gap between them is where the next instance lives. When you
write down a class, the same commit should carry the sweep, or say why it did
not.

**The sharpest instance, and it is not about sweeping — it is about what a
comment cannot do.** Building the task-body output capture (`cff4571`), m3
wrote `{ /bin/sh "$0" "$@" 2>&1; echo $? > "$_st"; }` and a body exiting 7
produced a wrapper exiting 2: `set -e` killed the subshell before `echo $?`
ran. **The comment twenty lines below in that same file says exactly that** —
*"under `set -e` a simple command exiting non-zero kills the script before the
assignment runs"*. The trap was documented, in the file, by the same author,
and was walked into anyway while adding a mechanism whose entire purpose is to
preserve exit information.

**A comment warning about a trap does not prevent the trap. The test did.** And
the transferable half is what the test had to be: *a capture mechanism verified
only on the success path would have been worse than none, because it would have
looked like evidence.* m5 reached the same conclusion about a null overlay in a
different subsystem within the hour — **an instrument that cannot be observed
failing is not an instrument.**

### T32 — the evidence records which node, and not which card

**Owner m3, 2026-09-04, found while closing `T19` on this stage.**

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

### T33 — a mechanical reformat makes a diff unreviewable, so the semantic check moves before the commit

**Owners m3 and the leader, 2026-09-04, the same error twice in one morning.**

`json.dumps(indent=2)` re-serialises a whole schema, so a five-key change lands
as **1074 insertions / 240 deletions** and no reader can see what changed. The
leader hit it first — 164 lines for one sentence in
`environment.schema.json` — and m3 hit it hours later in
`workset.schema.json`. **Both of us verified afterwards and disclosed
afterwards**, which is the right check in the wrong order: an
after-the-fact semantic diff reassures the author and does nothing for the
reviewer, who has already been handed a diff they cannot read.

**Ruled: do not restore hand-formatting.** Canonical `json.dumps` form is
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

### T34 — the environment record can outlive the container it names, and a downstream field repeats it as observation
*m1 and m4, 2026-09-04, from two ends of the same artefact. Nobody's defect
individually; the join is unowned.*

**The producer half.** m1's rung-1 record on 217 said:

```
runtime.container   yihou_e2e_flow_sgl_e2e-main-20260904
runtime.started_at  2026-09-04T09:03:51Z
```

and `docker inspect` on the container of that name said:

```
Created       2026-09-04T09:37:18Z
StartedAt     2026-09-04T09:37:18Z
RestartCount  0
```

**`Created` equals `StartedAt` and restarts are zero**, so it was not restarted —
it is **a different container carrying the same name**, brought up after the
first was torn down. The record describes an instance that no longer exists, and
**every field in it still validates.** `check_environment` and `check_deploy_kit`
both pass it, correctly: nothing they check is wrong.

**The consumer half, m4's, and it is the sharper one.** `optimize_kernel`'s
`10_read_inputs.py:137` fills `premise.run_environment` from
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

**Not fixed, and neither of us changed the field.** `premise.run_environment`'s
reader is m4's premise gate, so redefining it is a contract decision rather than
an owner's; and `runtime.started_at` is the leader's schema. What m4 *did* do is
make the same `docker inspect` that checks liveness also log the observed
`Id / Created / StartedAt / RestartCount` beside the record's claim — **the only
place in the flow where the container that actually did the work identifies
itself.** That is a good stopgap and it is a log line, not a check.

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
| 217 | 09:03:51 | **09:37:18** | 0 | record is **34 min early** — it describes a container that no longer existed |
| 006 | **10:21:54** | 10:16:46 | 0 | record is **5 min late** — same container throughout, still up |

**Neither is wrong, because nothing says what the field means.**
`environment.schema.json`'s `runtime.started_at` is `{"type": "string"}` —
**required, and with no `description`.** So is `runtime.endpoint`. One agent
recorded a moment before the container it eventually used; the other recorded
something like *when the service became ready*, five minutes after its container
started. Both validate.

**This is T23's shape, in a second section of the same document.** T23 called
`fixed.gpu_count` *"the one required field with no description"* — that was true
of `fixed` and `runtime` has two more. A field that cannot be wrong cannot be a
measurement, and here it has produced two incompatible readings in one afternoon
without either being a defect.

**So the fix is cheaper and more definite than T34's three options suggest:
define it.** One sentence in the schema, the leader's file, deciding between *when
the container started* and *when the deployment became ready* — and if it is the
first, it should be **read from `docker inspect` rather than written by the
agent**, which makes it a fact rather than a claim and removes the 217 case
entirely. `runtime.endpoint` wants the same sentence: 217 recorded loopback
`http://127.0.0.1:8101` and 006 recorded the routable
`http://10.245.151.128:8101`, and a consumer on another host can only use one of
them.


### T37 — two producers disagreeing is what a schema-shaped defect looks like

*Renumbered from **T34**, which collided with an earlier item of the same number — six owners append to this file and two picked the same next integer. Commits and messages citing T34 for *this* item still resolve: the other T34 is a different subject and the two are not confusable by title.*

**Owner m3, 2026-09-04, from m4's finding that `public_symbol: sampler_softmax`
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
recorded from the image so the claim is *checked* rather than asserted
(`4d5a6e6`, `d206fc6`). **What it does not do** is say how a
fragment-inside-a-method optimisation reaches the engine — that is M5.1.1, a
design question for the user, and the package can now state which case it is in
without being able to install the second.

### T35 — a sealed handoff from the old layout is not consumable by the definition that replaced it

`check_speedup_substantiated` looks for the measurement apparatus at
`scripts/workset` (`check.py:81`, `_APPARATUS`). The sealed stage-4
`kernel_optimization` in `cheat_for_mock/` carries its apparatus at
`scripts/kernel/` — `driver.py`, `graph_harness.py`, `measure_baseline.py`,
`sampler_softmax_kernel.py`. **So the one real stage-4 artefact this effort
owns cannot be fed to the validator that grades stage-4 artefacts.**

Not a regression. The sealed handoff is 2026-09-02 output from the
five-separate-packages layout, and `e2e-flow`'s definitions were rewritten
against `mission.md` afterwards; `mock_adapt.py` is the bridge and exists for
exactly this. **Leader's ruling, 2026-09-04: leave it.**

**Recorded because the assumption it breaks is easy to make and expensive to
discover.** Reaching for the sealed artefact as a ready-made fixture is the
obvious move when a validator needs a real input — it is the only stage-4
artefact that ever came off a cluster — and it fails at a path lookup rather
than at anything that names the layout change. It cost a wrong turn on
2026-09-04 while looking for an input to exercise the container path.

**The open question, if anyone ever wants it:** whether sealed output of a
stage should stay *directly* consumable by that stage's next definition, or
whether an adapter is the intended and permanent shape. Nothing today depends
on the answer.

### T36 — a claim about *who owns this* never looks like a claim, so nobody tests it

**Six instances between the leader and m4 on 2026-09-04, all the same move: an
assertion about *who* or *what* — an owner, a boundary, a blocker, a row —
stated before reading the thing that would have answered it.**

- **m4, twice.** *"The mount question belongs to m1, m4 or the contract; I am
  not picking"* — while m1's sealed kits already mounted the answer, and m4's
  own `scratch_root` default already pointed at the third form. And *"blocked on
  m3's `--impl` contract"* — while the Definition's `baseline`, in a file m4 was
  reading for other reasons, already carried `def sampler_softmax` beside
  `def run(*args, **kwargs)`: **one file satisfying both consumers,
  demonstrated, in the artefact.** There was no contract to arbitrate, only a
  shim not copied.
- **the leader, four times.** Ownership inferred from a filename rather than
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

**No code change. This is a habit item**, recorded because it cost real time
today and because — unlike every other entry here — **there is nothing to
detect it with.**

---

### T38 — nothing in the graph orders m2's two lines; only the GPU count does

**m2, 2026-09-04. Not blocking — today's configuration is correct. It is a
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

**Both leaves depend on nothing.** The graph offers them together — the leader
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
shape is the leader's to approve, and because a change that alters graph
topology deserves its own rung rather than riding on a fix.

*Numbered T38 against a max of T37; `todo.md` currently has duplicate `T36`s and
the leader is reconciling numbering, so treat this number as provisional.*
