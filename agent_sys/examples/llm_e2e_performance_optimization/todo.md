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

### T10 — `check_workset_runs` hard-fails on rsd while `min_pass_ratio` forgives correctness (was C24 and E16)

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

### T41 — `/proc` is namespaced under `spur exec`, so PID-based attribution from a node lies

*Renumbered twice: **T28 → T36 → T41**. Six owners append here and two keep picking
the same next integer; my own T28→T36 renumber (`f867a62`) landed on a number a
second owner had already taken, so the fix collided the same way the fault did.*

***This*** *item moved rather than the other T36, and the rule is worth stating because
it will recur:* **the entry with external citations keeps its number.** `2d521c1`'s
commit message and `CONTRACT.md:927` both cite T36 meaning *"a claim about who owns
this never looks like a claim"*; nothing outside this file cites T36 meaning `/proc`.
Renumbering the cited one would have broken two references to save one.

*T41 was free — the numbers otherwise run 1–47 — so this consumes the gap rather than
extending the range. Citations of **T28** or **T36** for the `/proc` subject still
resolve by title; the three are not confusable, which is the only reason a renumber is
survivable at all.*

*Reported independently by m3 and by `checkpoint`, neither of whom renumbered it —
correctly, since it touches other owners' entries. That is the second numbering
collision today and the mechanism is unchanged: **`todo.md` has no allocator**, and
`git status` cannot show you a number someone else is about to use in an editor.*

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

**STILL OPEN, 2026-09-04 end of day, and said so deliberately.** The stage has
been exercised standalone all afternoon and this is the one thing in it that a
fixture cannot reach: it needs two measurements on two cards to show itself, and
every control here runs on a login node with no card at all. Closing it on the
strength of *"the guards are deliberate now"* would be its own instance of the
class below.

**It is the third member of a class the afternoon produced three of**, and the
leader had counted two before this one:

| | the configuration | what it decides | what grades it |
|---|---|---|---|
| m2 | `--cuda-graph-max-bs 8` against concurrency 16 | decode exceeds the captured graph on essentially every step — a 4.7× latency spread | nothing; it was read as an image difference |
| m1 | `32.5` | a floor, recorded as prose and read back as a measurement | nothing; the real number was 33.58 at `tp_size: 1` |
| **m3** | **`E2E_MEASURE_GPU`** | **which card every number in `evidence/` came from** | **nothing; `measured_on` names the node** |

**A configuration that determines the number and is invisible to everything that
grades it.** The leader's sharpening is the part to keep, and it makes this the
purest of the three: `check_workset_runs` re-measures **on the same card**, so
when the card is the fault **it agrees for exactly the reason the original was
wrong**. The agreement is not evidence of correctness; it is evidence that both
readings share a premise nobody recorded.

That is m5's *"a relative check cannot detect a fault both sides share"*,
arriving in this stage from a third direction and hours before they proved it in
theirs. The fix is unchanged and unbuilt: **record the index in
`evidence.measured_on`**, so the two readings can be compared on their premise
and not only on their result.

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


**Extended 2026-09-04, evening — the same check, a failure it did not cover.**
T33 was about a diff being *unreadable*. This is a diff being *wider than its
author*: in a shared worktree, `git commit -- <pathspec>` names a **file**, and
the file contains whatever any teammate has uncommitted in it.

`3b4d390` says *"four fixes to rung 3's launch section"* and also carries
**m4's rewrite of RUN-PLAN §2a** — roughly 57 of its 100 insertions and 42 of
its 44 deletions. Nothing was lost and the content is correctly placed, but my
message does not describe my commit, m4's `b838fae` describes changes it does
not contain, and I signed off on sixty lines I did not write.

**The standing rule — *commit by pathspec, never `git add`* — does not defend
against this and was never meant to.** It stops you sweeping up *other files*.
It says nothing about *other people's lines inside the same file*, and
`RUN-PLAN.md` is the one file all six owners write to.

**The mitigation is T33's own check, one step later:** run
`git diff --stat <file>` *immediately before* committing and ask whether the
number matches what you wrote. m4 caught this because their commit reported
`65 insertions, 0 deletions` and a rewrite that replaces a section cannot have
zero deletions.

**The check is *compare to what you did*, not *look at the stat*, and the
difference is not pedantry.** m4's limit, and it is the one that keeps this from
hardening into a bad rule: an **impossible** number only turns up in
replacement-shaped work. Their rewrite could not have had zero deletions, so the
stat refuted itself. **Sweep an additive edit and the number is merely bigger** —
`+43` where you expected `+30` — and *plausible-but-larger* is not impossible.
**Nobody rejects a number that is only larger than they remembered**, which is
precisely the failure this entry records: `100 / 44` did not look absurd to me,
it looked like a commit.

So the rule needs the second operand. Stated as *"check the stat"* it invites
the substitution I actually made — `git status --short`, which answers a
different question — and **a check that reports presence where you needed
magnitude is not the check you thought you ran.**

**And I had the baseline and did not compare it**, which is the part worth the
entry. My own stat after the first edit was **43 insertions, 2 deletions**; at
commit time it was **100 / 44**. I ran `git status --short` in between — which
prints `M` and not counts — and committed on that. *A check that reports
presence where you needed magnitude is not the check you thought you ran*, and
the correct number was in my own scrollback.

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

---

### T39 — `read_events.py` prints `message` and hides the attribute that holds the cause

**m2, 2026-09-04. Not blocking. One line, and it is in checkpoint's file, so it
is routed rather than done here.**

`assets/lib/read_events.py` renders each event as its `message`. For
`output_absent` that message is
*"declared output … was never delivered"* — **which can be false** — while the
true reason sits in `attributes.seal_refused` and is not printed. So the tool
built for reading the event store reproduces the misdirection the store already
has; see
`temp/bugs/2026-09-04-output_absent-states-a-cause-that-is-false.md`.

**Two owners have now paid for this in the same day** — the leader's four-run
stall study and m2's replayed-kit investigation — and **both ended up running
`cat` on a raw event JSON** to find the same attribute.

**The precedent already exists.** m4 hit it from the other side and fixed *their*
reader: `runprobe` prints **every** attribute of a triggering event, and the
leader credits that change with turning a lost reason into a one-command answer
three times on 2026-09-04. `read_events.py` has not had the equivalent.

**What would settle it:** print `seal_refused` and `detail` beside `message`
when present — or all non-empty attributes, which is what `runprobe` does and
needs no per-kind knowledge.

**Routed to checkpoint as the file's author**, with the diff, on the leader's
rule that the author of a file cares most about its output being right and a
change landed by someone else and merely attributed is worse than one landed by
its owner.

---

### T40 — treat every probe as a control, because the slot decides whether you check the null

**Owner: checkpoint writer, 2026-09-04. Sharpening of `T31`, and m3 asked for it
to carry my name; the pattern it names is theirs as much as mine.**

**The observation.** m3 reported that **two** of their wrong turns today were
probes that *could not have succeeded* — a null that looked like an answer: the
`abc` payload that decoded to a non-command, and the `-p $W` probe that dumped
the whole process table. I had just done the same: my positive control for
`runlive.sh` used `exec -a`, which **`dash` does not have** (`sh: 1: exec: -a:
not found`), so the subject never existed. I chased it.

**But I only chased it because it was labelled a control.** Had that same failing
probe been sitting in the *measurement* slot, I would very likely have written
down the null and moved on — which is exactly what happened to me at 07:12, when
I read empty `logs/`, `playground/` and `tmp/` in a task zone as "the body never
ran" and had to throw the inference away after checking tasks that certainly did
run and finding those directories empty for every one of them.

**So the variable is not care, it is the slot.** Same person, same hour, same
diligence: a null in a control is *by definition* suspicious, and a null in a
measurement reads as data. m3 confirmed the same split — their wrong turns in
the measurement slot, their catches in the control slot.

**Corrected 2026-09-04 at m3's request, and the correction matters to the
entry's own argument.** This first said *four* of m3's wrong turns were dead
probes. **It was two.** I merged two different counts of two different things:
they had said *"my four wrong turns were all in the measurement slot"*, which is
about the **slot**, not about dead probes. Their other two were **reasoning
errors from artefacts they had not opened** — endorsing a duration signature
while `evidence/performance.json` sat on disk saying otherwise, and the
killed-mid-work hypothesis.

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

**Instances already in the record**, all 2026-09-04:

- empty zone `logs/`/`playground/` read as "the body did not run" — control:
  those dirs are empty for *every* task, including ones that sealed valid
  handoffs.
- `ppid=1` read as "orphaned by a kill" — control: `nohup sleep &` reaches
  `ppid=1` with nothing killed. Escalated before the control existed.
- `exec -a` positive control — the subject never existed.
- `readlink /proc/<pid>/exe || continue` — an unreadable `exe` dropped silently,
  so "cannot decide" and "nothing there" produced identical output (`01f768c`).

**Sharpening, m3, 2026-09-04 — which control failure to spend the extra minute
on when you cannot afford both.**

> **A broken positive control wastes your own time. A broken negative control
> spends someone else's correctness.**

The rule above is about whether a probe *could have succeeded*. This is about
**direction, and who pays for the error**, which the rule does not say.

**A broken positive control refuses, and a refusal makes you look at it.** m3's
three schema fixtures were this shape — an unresolvable `$ref`, a bad
`gpu_arch`, a `kernel_id` pattern miss — each looked exactly like a working
probe, and each cost an hour of **their own**.

**A broken negative control passes, and its output is a claim about somebody
else's artefact: *"your binding is toothless."*** m4 had one within the hour: a
dict merge that re-added the key it was meant to strip, printing `NOT CAUGHT`
for the `impl`/`impl_path` binding m3 had just landed. **They rebuilt the probe
before concluding anything.** Had they not, the claim would have travelled to m3
as evidence, and **the action it invites is weakening a correct contract.**

So the two failures are not the same size. One is self-limiting; the other
propagates, propagates *as evidence*, and reaches someone with no access to the
instrument that produced it. **When you can only afford to verify one control,
verify the negative one** — its false output is the one that leaves your hands.

**Not blocking.** It is a habit, not a defect, and the four instances above are
already fixed or recorded. Filed because `T31` says naming a class is not
sweeping for it, and this is the sweep condition for a class that has produced
at least four instances across two owners in one day.

### T42 — a validation zone is not a normal process environment, and anything read from it that names a host path is suspect

**A validator does not run in the environment you think it does.** The zone
rewrites variables a body would take for granted, and every one of them named a
host path that turned out to be somewhere else. Four instances on 2026-09-04,
each found only by something failing:

| variable | what the zone makes it | what it cost |
|---|---|---|
| `HOME` | `<zone>/home` | m4's ephemeral container mounted a *subdirectory of the zone* instead of the run root, and rung 0 died on `./run_performance.sh: No such file or directory` — a container that came up perfectly and could not see its own script |
| `PATH` | `/usr/bin:/bin` | `spur` lives in `/usr/local/bin`, so the transport is simply absent |
| `SPUR_CONTROLLER_ADDR` | **unset** | m3 lost three non-reproductions to it, because their own login shell had it |
| `TMPDIR` | `<zone>/tmp` | forced by the zone as an invariant (`validator/environment.py:233,86`), so a producer's `env` cannot reach it — and on this cluster's NFS a `TMPDIR` there SIGSEGVs every HIP kernel launch |

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
example of it.** Swept all eight schemas under `assets/schemas/` on 2026-09-04:
**four express no cross-field constraint at all** — `bench_result`,
`environment`, `kernel_table`, and `integration_report`, the last at 786 lines
with zero. The other four carry thirteen `if/then` between them. So the package
has **eight schemas that check fields and four that check meaning**, and the
instances below are what that ratio produces.

**Three instances on 2026-09-04, one shape.** A field carries something
faithfully — the copy is exact, the producer did nothing wrong — and the field's
*name* says it is something else. Every field validates; the document is false.

- **`premise.run_environment`** — `10_read_inputs.py:137` fills it with
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
  `run_performance.sh:44-55` is `perf_counter` around a
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
their enum. So m4 emits an impossible `apply` block in silence and the first
thing to notice is m5, two stages later.

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
`Draft202012Validator`. **So "nobody looked" is the wrong diagnosis for half of
them**; m1 looked, found it inexpressible in the schema, and put it in
`deploy_kit.layout.yaml`'s invariants with a gate fault behind it (`53bc783`).
Corrected here after I asserted the general form and was wrong — see the
carrier case below, which is what actually survived.

**And the rule that generalises out of that mistake, m3's wording:** reading an
unexpressed constraint as an oversight is **cheap to say and expensive to be
wrong about.** It converts someone's deliberate placement into a defect, and the
fix it implies — *move it into the schema* — is one the schema cannot execute.
m1 had looked, found it inexpressible there, and put it in a layout invariant
with a gate fault behind it; **the evidence that they looked was not in the
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
and the other one. Neither `d08047b` nor m3's narrowing was wrong in isolation
and no review of either would have caught it; **the defect existed only in the
relation, and the relation is what the comparison had refused to print.**

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

**One line, not three owners' attention** (leader's call, 2026-09-04). The fix,
whenever the mechanism question is settled, is for the three to agree — either
`rebuild` becomes emittable or it stops being acceptable.

**The same defect facing the other way: `must_preserve`, emitted and read by
none.** `60_write_handoff.py:272-281` populates it from the workset's
`integration` — signature, invariants, `requires_restart`, `build_step` —
`kernel_optimization.schema.json:320` documents it, two samples carry it, and
**no consumer anywhere reads it.** Found 2026-09-04 when the mock's empty
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
Declared in this package (`m4_kernel_opt.yaml:358`) **and** in `kernel-opt-demo`,
and read by no body in either. The old readme calls it *"a KernelForge checkout,
already `pip install -e`'d"* — so the environment was always prepared **out of
band**, and the variable only ever told a *reader* where it was.

It cost something because it looked like the answer. When rung 4 turned out to be
blocked on `forge-loop` being installed nowhere, `KFO_KERNELFORGE_REPO` is the
first thing anyone finds, and it reads as a mechanism that has come unwired. It
never was one. **There is no prior art to restore** — whatever gets built is new
work, and the demo that produced this effort's proven assets ran on a machine
somebody had prepared by hand, which nothing in the package records.

**m3's statement of the class, which covers all three:** *a variable that was
never a mechanism, only a note to a reader spelled as configuration.* And the
tell is the same in both directions — `rebuild` has a reader and no producer,
`must_preserve` and this one have producers or declarations and no reader:

> **You cannot tell from a declaration whether anything reads it, and
> `grep -rn "$NAME"` across the bodies is a two-second check that nobody runs
> because a declaration looks like plumbing.**

m3 ran it against both of their own `E2E_MEASURE_*` declarations while they were
in there; both have consumers. That is the whole remedy and it is cheaper than
the entry describing it.

### T45 — a comment that asserts a wiring gap sends the next reader to change code, when only a number was missing

**m3, 2026-09-04, found by m2 measuring instead of reading.**

`check_identity_resolved.check.py:143` said `min_resolve_ratio` *"is not passed
by `steps/m3_analysis.yaml`, so the arm above cannot refuse in this package as
configured."* The yaml passes it —
`min_resolve_ratio: '${min_resolve_ratio:-0.0}'` — so the arm is fully
parameterised and `--var min_resolve_ratio=0.8` reaches it with no edit at all.
m2 measured both directions on one operator with an honest unresolved entry:
`0.0` passes, `0.8` refuses with `resolve_ratio 0.500 is below the floor 0.8`.

**Two distinct costs, and the second is the reason this is a todo and not a
typo.**

1. It is wrong about the artefact. A reader who wants the bar enforced concludes
   they must edit yaml and code, when they need only pass a var.
2. **It made a defect look bigger than it was, in the direction of a finding.**
   Both of us had it filed as an instance of the leader's "validators that grade
   nothing" sweep. It is not one: `0.0` is a defended default with a written
   argument, and the arm is live. The true defect was one stale sentence. A
   stale comment does not merely fail to inform — it *manufactures* the finding
   the sweep is looking for, and a sweep that trusts comments will report it.

**And the fix I shipped first was itself unreachable.** `212773d` added an
"unset — this arm did not grade" wording behind `if raw is None or raw == ""`.
Because the yaml *does* pass the arg, `raw` is never absent and that branch
could never execute in this package: the default path still printed
`floor 0.0`. I checked that the new wording was correct and not that it was
reachable — the same omission in a different costume, since a comment claiming
the arg is unpassed is exactly what makes an `is None` branch look sufficient.
Verified after correcting, by driving `_check` over a real `operator_identity`
with all three arg values and reading the note each produced.

**The rule.** A comment that describes *wiring* — what is passed, what is
reachable, what cannot fire — is a claim about a file other than the one it sits
in, and it goes stale silently because nothing loads it. Either grep the file
you are describing at the moment you write the sentence, or describe the
behaviour of the code in front of you and let the reader look up the wiring.
The second is usually the better sentence anyway: *"a floor of zero grades
nothing"* is true wherever the value comes from.

**The general form of the second half is `T46`** — *a fix verified correct is
not a fix verified reached*. m4 filed it with both instances while this
paragraph was still in my working tree, and theirs is the better entry: it has
the asymmetry (their `NameError` announced itself, mine would have been silent
forever) and it has the two cases where the check was already being applied
without a name. Cut to a pointer rather than kept as a second copy.

### T46 — a fix verified correct is not a fix verified reached

**Two instances on 2026-09-04, one from m3 and one from m4, in the same stretch
of work.** Both of us changed code, checked the change was *right*, and shipped
without checking it was *executed*. Those are different questions, and only the
second needs the surrounding wiring in view.

**m3's would have been silent forever.** They replaced a validator note that
printed `floor 0.0` with one saying *"this arm did not grade"*, and guarded it
with `if raw is None or raw == ""`. **The yaml always passes the argument, so
`raw` is never absent** — the branch could not execute, and the default path
went on printing the exact string the fix had removed, in a file whose comment
now claimed the fix was needed. Caught by m2 (`5ca132e`).

**m4's announced itself.** The stubkit's entrypoint referenced the shell's
`$IMPL` from inside the Python emitter — `NameError` on the first candidate run,
loud and immediate.

**The asymmetry is the entry.** An entry carrying only the second instance would
teach that this class announces itself. **It does not; that was luck.** The
same defect is a crash when the unreached code is malformed and a permanent
silent wrong answer when it is well-formed — and *well-formed* is the normal
case, because the code was written carefully and only the reachability was
assumed. **The better the fix, the quieter the failure.**

**What makes it actionable rather than cautionary: people already do this
intermittently, without naming it.** Both of m4's other changes that day
included a reachability step that felt like ordinary care at the time —

- the `substitution` × `apply_mode` gate was **armed, run against a real
  artefact to watch it refuse, then switched off** (`fc0784c`). That step caught
  a `NameError` waiting in the disabled path: the call site had neither `packup`
  nor `notes` in scope, so the gate **could not have run when enabled** — a gate
  that cannot fire when switched on, shipped as *written and ready*;
- `_same`'s new vocabulary reporting was exercised by **monkeypatching the
  pre-`d08047b` enums**, because the leader's own fix had made the divergence
  branch unreachable on current schemas (`28c177d`). Shipping it unexercised was
  the alternative.

**So the rule is not new behaviour, it is a name for something already done half
the time.** The half where it is skipped is the half where the change looks too
small to need it — a guard, a message, a default.

**The check, and it is one question:** *what would I have to do to make this new
line run, and has that happened?* If the answer is "nothing, it runs on every
call", say so. If it is "a variable would have to be absent" or "an enum would
have to differ", go and make that true once, on purpose, before shipping.

**Distinct from `T40`**, which is about probes and whether a null in a control
slot gets questioned. This is about **fixes**: the code is not an instrument, the
result is not being read, and nothing about the slot prompts suspicion.

### T47 — the pathspec window is irreducible, so the check has to be after the commit

**The mechanism is `CONTRACT` §8a, not this entry** (`8b1057d`, and `5e61480`
for the post-check). `git commit -- <path>` takes the *working tree*, so it also
takes a co-owner's uncommitted edits to that same path. Three owners found it
independently — m1 first, then the checkpoint writer, then m4 — which is
stronger evidence than any one report.

**What is new here is that no pre-commit check can close it, and that was
established by accident.** Both earlier reports assumed a pre-commit check was
sufficient and argued about *which one*.

**Measured 2026-09-04.** This entry originally prescribed *read
`git diff -- <path>` immediately before committing*. That check was run against
this very entry — one hunk at EOF, one heading, thirty-nine lines, all mine —
the commit was issued seconds later, and it returned **`no changes added to
commit`**: the checkpoint writer had committed `todo.md` in the interval,
sweeping this entry into `5281a4e`, a commit about T40. **The remedy was
falsified by the act of committing it.**

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

**Owner: checkpoint writer, 2026-09-04. Found by m4 in my own correction, and
deliberately kept out of `T47` by them because that entry is about a git
mechanism and this is about how a remedy gets chosen after an incident.**

**The incident.** I swept 39 lines of m4's `T47` into `5281a4e`. What caught it
was the **post-commit** check I had been running all day without thinking about
it: `git show --numstat HEAD` printed **`66 0`** against the 27 lines I had
written, and that discrepancy is the only reason I looked.

**The near-regression.** Asked what I would change, I wrote an addendum adopting
**`git diff -- <path>` before committing** — a check m4 had **already falsified**,
by running it correctly on `T47` itself and losing the race anyway. So I was one
step from replacing a control that had *demonstrably just caught the bug* with
one already known not to close the window.

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
> and check whether that is already in your routine.** If the answer is "a thing
> I was already doing", the remedy is to make it explicit and load-bearing, not
> to add a new check beside it.

**Two more instances, both mine, both named by someone else on the same day:**

- I recorded that I had referenced findings **by identifier rather than by
  value** and called it *"partly luck"*. m3 corrected it: a habit, not an
  accident, and structurally immune to mis-attribution — **the only rule that
  day which prevented rather than detected.**
- m4 named `numstat` as the load-bearing check above, which I had been running
  since morning and had never once described as a control.

**m4's summary is the tell and it is worth keeping verbatim: *twice today you
have filed something that worked as luck.*** A practice that works produces no
evidence of itself, so its owner is the last person able to see it. **That makes
this one of the few classes here that an outside reader finds more easily than
the author** — and unlike the tool defects, it fires when nothing is broken.

**Not blocking.** No defect; a reasoning habit with one near-miss recorded.

### T49 — the verdict is right and the message says why it is right, wrongly

**Named as a class by m5 on 2026-09-04 after the third instance in one afternoon.
Six now, across five owners.** The sixth is the one that matters most for this
entry's own reach: **T49 occurred inside the failure ledger itself**, where the
recorded *reason* for four failures was a `note:` line the log had relabelled
`PROBLEM:`. **Twice what the first summary of it said** — checkpoint reported two
rows, readme-cn's audit found four. Recorded here
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
| `m3_analysis.yaml` (m3) | *"the body falls back to card 4 via `${E2E_MEASURE_GPU:=4}`"* | `measure_in_container.sh:142` **refuses** — prints the node's live `rocm-smi` and exits 1. There is no fallback |

**Every one gives the correct verdict.** That is what makes it the hardest kind
to notice: nothing downstream is wrong, no run fails that should pass, and the
only defect is in the sentence a human reads. A reader who checks the claim and
finds it false stops believing the *next* message from the same validator, which
is the actual cost and it is unbounded.

**Three of the four were found by someone reading the message rather than the
code**, and none by its author. m5's two were found by m2 and m4; the leader's
was found by `checkpoint` counting `PROBLEM:` lines for an unrelated reason and
m3 noticing that seven of them sat under a `passed` heading. **The author knows
what the check does, so the author reads the message as a summary of it** — the
same blindness as T48's invisible working practice, pointed at prose.

**The tell, and it is cheap:** read the message *as if you did not write the
check*, then ask what would have to be true for it to be exactly right. In all
four the answer was a strictly broader condition than the code tests.
`check_workset_shape`'s is the sharpest — *"not defined at module level"* is
true of a module-level assignment, and the check cannot see one, so the message
is a correct English sentence about a case the code would get wrong.

**Eight, and the eighth is the one that changes the entry.** m3's is the only
instance written by someone who had **filed the class themselves that morning** —
T45 is theirs, eight hours earlier, about a comment describing another file that
goes stale silently. Every other instance here is someone who did not know the
class existed. **Theirs says that knowing it does not cure it**, which is worth
more than the count going from seven to eight, and it is why this entry ends with
a detector rather than with advice.

m4 declined to hold it when m3 handed it to them *"for the collection"* — they do
not own this entry and letting it sit with them is the gap the entry describes,
which nearly happened once already when three people each assumed someone else
would file it.

**Seven, not four.** m4 added three of their own after m5 named the class and I
had already filed it — which is itself the point: **each of us could see only our
own, and none of us could see our own until someone else read it.** m4 offered
the entry to m5 rather than taking it; m5 had offered it to nobody; I filed it
without either of them knowing. That is three people declining to claim a class
none of them could have assembled alone, and one person assembling it badly.

**m4's asymmetry, which decides the fix and is not obvious:** *narrowing the
message cannot break a consumer; widening the field can.* `check_workset_shape`
presence-checks against `module_symbols`, so adding assignments to it makes some
previously-refused worksets pass — correct if they were correct, a **silent
loosening** if not. Both of m5's instances were resolved by narrowing the
message, which is evidence about the base rate and not about any particular case.

**Not blocking, and not a code sweep.** Fixing the four is done or routed;
the entry exists because the fifth will be written by whoever writes the next
refusal, and the cost lands on a reader rather than on a run.

**The sibling class, recorded 2026-09-04 because the afternoon produced four of
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
*"measured on **what**?"* catches it. Two of the four were found by someone
volunteering the history of their own instrument, which no reader could have
demanded.

### T50 — an instruction verified correct against a world that does not exist yet

**Three owners wrote one on the same afternoon; two were unexecutable the moment
they were committed, and the third survives by where the ladder happens to be.**
Routed to the leader by m4 rather than written by any of them, because T49 had
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

**The fix is m5's and it is a generalisation rather than a correction:** the
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
and `agent/runner.py:801` closes the cheap escape: *"an env var cannot instruct
an agent. A conversation is not a process reading `os.environ`."* An `env:` block
makes a value **reachable**; only the brief makes it **used**.

So the brief is the whole interface at promotion, and it drifts two ways.

**Direction 1 — the brief omits what the program knew (m3).** `workset_builder`'s
readme is 330 lines and ten STEPS with **zero** occurrences of
`measure_in_container.sh`, `E2E_MEASURE_GPU`, or `rocm-smi`. STEP 7/8 say
`cd "$WS" && ./run_correctness.sh` **directly, on whatever host the agent is on**,
and those hosts have no torch. Everything that knows better is on the branch the
promotion turns off — `entry.sh:168`. **The mock is not a weaker version of the
real path here; it is the only version carrying the knowledge**, so promoting m3
removes the container step and the card check together, silently. First symptom:
a torch import error, or worse, a plausible number measured on a card nobody
chose.

**Direction 2 — the brief contradicts what the program does (m4), and this is
the one that survives review.** The brief said the wrapper *"execs into the
recorded container and never starts or removes one — CONTRACT §5.2 is absolute
about that."* `48c3337` — the leader's own ruling — made that false: with no
running container the wrapper now starts an ephemeral one, `--rm`, trap-removed.

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

**Neither was found by reading the brief.** m3's came from listing what rung 3
reads; m4's came from being asked the question m3's finding raised. **Both authors
had read their own briefs many times.**

**Not blocking for m4** — theirs is corrected. **Blocking for rung 3** until
`workset_builder`'s brief is written.

### T52 — no `df` reachable from `spur exec` predicts whether a `docker load` fits

**m1, 2026-09-04. Measured, and it overturned a refusal I had already made.**

I declined to `docker load` the 28.5 GB engine-image backup because every node
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
device is how a verification gets skipped for no reason. That is exactly what I
did.

**How I got it wrong is the reusable part.** I checked whether my instrument was
pointed at the right device — I mounted the host's docker root through the
daemon specifically to rule out the exec namespace, and got the same number,
which I read as confirmation. **Two routes to one wrong answer is not
corroboration when both routes share the assumption under test.** The only
measurement that would have settled it was the before/after across the load
itself, and that requires doing the thing I was using the number to avoid.

### T53 — the var table's rung-2 advice for `expect_ranks` is right only at `tp=8`

**m1, 2026-09-04. Not mine to edit — `RUN-PLAN.md:31` is the shared var table.**

```
| `expect_ranks` | **2** | **omit it** (defaults to 8), or track `--var tp` |
```

The second half is correct and the first half is a trap at any `tp` but 8.
`m2_profiling.yaml:119` is `expect_ranks: '${expect_ranks:-8}'` and it
deliberately does **not** track `tp` (a `${}` default may contain no `}`, so
`${expect_ranks:-${tp:-8}}` is not spellable — the yaml says so at :114-116).
So at rung 2 with the `tp=4` we are actually running, omitting it yields 8 and
`check_trace_coverage.validator/check.py:72` fails with `expected 8 rank(s), the
manifest lists 4` — a real deployment graded against a rank count nobody chose.

Suggested: drop "omit it (defaults to 8)" and leave "**must equal the
deployment's `tp`**", since the two are only the same sentence at `tp=8`.

The yaml's own comment has the same soft spot — *"a real run leaves it alone and
sets `--var tp=` if the deployment is not eight-way"* reads as though setting
`tp` moves `expect_ranks`. It does not, by the design two lines above it.

### T54 — declaring `runtime.replayed_from` would not close the hole m5 found; `additionalProperties` is the hole

**m1, 2026-09-04. m5's observation, measured and then corrected in the direction
that matters.**

m5 found that `runtime.replayed_from` is **consumed in five places and declared
in none** — `check_deploy_kit/check.py:404-425`, `kit_status.py:66,158`,
`check_measurement_order/check.py:280,330`, `load/line.sh:149` — validating only
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
*filename string* and never the contents (`handoff/content.py:184-197`, mission
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
Not "a contract-wide migration", which is what I assumed before counting.

**Not done, deliberately.** `environment.schema.json` is shared by all fifteen
kinds (CONTRACT §2) and closing an object is the kind of change that turns a
tolerated field into a hard failure mid-run. It wants the leader's call and a
green rung before it lands, not a quiet edit while rung 1 is in flight.

**Mitigated at the one producer that is new, m5, `4e7a18d`.** m1's suggestion,
and it needs no schema change: `replay_root.py` writes the record, then **reads
`replayed_from` back through the accessor a consumer uses** and refuses the whole
run if it does not name this hop. Proved by injecting m1's exact typo into a copy
of the tool — `rc=1`, with the refusal saying *"shipping this root would hand the
flow a replayed kit wearing a real one's face."*

That does not close T54; it closes the hole where a **new** producer could open
it. The 217 existing occurrences and the other two undeclared keys are still
governed only by `additionalProperties: true`.

**And running that test found a second defect, in m5's own tool.** The refusal
printed the value it read back, and it was **not `None`** — rung 1's `deploy_kit`
already carried `replayed_from: /shared_nfs/…/cheat_for_mock/stage1-deploy/deploy_kit`,
because that run mocked stage 1. The rewrite was **overwriting** it, collapsing
`sealed corpus → run X → here` into `run X → here` and telling a reader the
numbers came from a real bring-up one hop back when they never came from one at
all. Now chained: `<run> <- <prior>`.

**The general form, which is the part worth keeping:** *the tool that writes a
provenance field is the one best placed to destroy provenance*, and a test aimed
at a typo is what surfaced it. On a kit with no prior `replayed_from` the
behaviour is identical and every check written for it passes.

### T55 — the CUDA graph ceiling belongs in the environment record and in `check_environment`'s compared set

**m1, 2026-09-04. The half of the graph-ceiling defect that item 1 does not fix.
Approved by the leader; deferred for the same reason as T54.**

m2 measured **4.7x** in decode latency between two runs whose every *recorded*
variable was equal — node, image id, model, `tp_size`, cards,
`mem-fraction-static`, load shape. The difference was the CUDA graph ceiling,
against a load at concurrency 16 (M1.2.3.4): below the ceiling decode runs
captured, above it the engine falls to eager.

**There was no default to be wrong**, which is the part that makes this
structural. `DK_CUDA_GRAPH_MAX_BS` appears **nowhere** in the package; the
producing agent invents a value each bring-up. Measured across every kit in the
run root:

```
env.sh:105  DK_TP_SIZE:=1  MAX_BS:=8    x18   the sealed 2026-09-02 kit, replayed
env.sh:170  DK_TP_SIZE:=4  MAX_BS:=16         062229-2695b9   real
env.sh:169  DK_TP_SIZE:=4  MAX_BS:=16         110626-43f0de   real
env.sh:193  DK_TP_SIZE:=4  MAX_BS:=8          125637-e1ddf6   real
env.sh:222  DK_TP_SIZE:=4  MAX_BS:=32         143952-bec7da   real, in flight
```

**16, 16, 8, 32 over four real bring-ups**, the line number moving each time
because `env.sh` is regenerated rather than edited. That is **strictly worse
than a wrong default, because a wrong default is at least reproducible.** Two of
the four shipped a ceiling below the concurrency the mission grades at, and the
run in flight is right **by luck, not by construction**. The `:=8` that looked
like a default is a `tp_size: 1` record replayed eighteen times — *a value fixed
in an artefact, mistaken for a default because the artefact is replayed*.

**Landed already (item 1, this commit):** the eighth contracted parameter
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

**m1, 2026-09-04. Found while verifying m2's account of four empty summary files;
recorded rather than acted on, and I could not establish whether it is
deliberate.**

```
9246d23165e72b5cdb359689b7892dc0  assets/load/summarise.py
9246d23165e72b5cdb359689b7892dc0  assets/bench/summarise.py     <- same md5
```

Three call sites, split between them:

```
assets/accept/measure.sh:218,:408              -> $BENCH/summarise.py
assets/load/replay.sh:147                      -> $LOAD/summarise.py
check_deploy_serves/check.py:592               -> assets/bench/summarise.py
```

**Every one passes both arguments and is correct.** The script refuses on
`len(argv) != 2` with a usage message and rc 2, so a one-argument call produces
an empty file rather than a wrong one — which is the right failure and is what
m2's own experiment harness hit, reaching for `$BENCH/../load/summarise.py`.
Their four zero-byte summaries came from that, not from the flow.

**Why it is worth an entry anyway.** The mistake was *navigating between two
copies of one file*, and that is the seam this whole effort exists to remove —
`handoff.analysis.md`'s three seams are each one name over two things. Two
identical copies with owners on both sides is the same shape one level down: it
is currently harmless because they agree, and the day they stop agreeing nothing
will say so, because nothing compares them.

**Not acted on, and the uncertainty is real.** `assets/load/` is m2's stage
directory and `assets/bench/` is where my validator reaches; the duplication may
be a deliberate ownership boundary rather than an accident, and deleting either
copy breaks live call sites. **This wants its two owners to agree on one home,
not a unilateral edit** — and m2 asked me not to chase the empty-file question
further, which I have not.

**Unrelated but measured on the way past, and it corrects something I told m2:**
an empty summary in `check_deploy_serves` does not fall back with a warning, it
**fails** — `check.py:592-601` catches the `ValueError` from `json.loads` and
returns *"the load ran but produced no readable summary … A load with no numbers
has not shown the deployment serves under load"*. The announcing-`WARNING` path
is the different case of a summary that parses but carries neither
`request_latency_ms` nor `output_sequence_length x inter_token_latency_ms`.
Two failure modes, two behaviours, and I conflated them in a message.

### T57 — one scratch path, three independent literals, agreeing by coincidence

**m5, 2026-09-04, from dry-running rung 5's launch line against rung 1's tree.
Spans m1, m4 and m5, and m4 found the same variable from the other end.**

Three variables name one directory tree and **none derives from another**:

```
work_root           '${work_root:-/mnt/m2m_nobackup/yihou/e2e_flow}'                    m5, m1, shared.yaml
scratch_root        '${scratch_root:-/mnt/m2m_nobackup/yihou/e2e_flow/kfo}'             m4, five sites
validate_work_root  '${validate_work_root:-/mnt/m2m_nobackup/yihou/e2e_flow/validate}'  m1
```

**They agree today because the rung-5 command passes `work_root` equal to its own
default.** Change it and the other two silently keep pointing at the shared
default — a run whose work root is elsewhere and whose scratch and validate
trees are not.

**And the section invites exactly that.** RUN-PLAN's rung-5 §5 tells the reader
to pass a **run-unique `container=`** because the default is a fixed name on a
shared host. Anybody applying that reasoning one variable over — a run-unique
`work_root`, which is the same argument for the same reason — splits the three
without a word from anything.

## The part that closes a loop: they *cannot* be derived

The natural spelling is:

```yaml
scratch_dir: '${scratch_root:-${work_root:-/mnt/…/e2e_flow}/kfo}'
```

**That is the nested default**, which expands to a literal string with **zero
problems reported** (`temp/bugs/2026-09-02-an-unparseable-variable-reference-is-passed-through-silently.md`).
So the derivation is not merely unwritten — **it is unspellable**, and three
owners each wrote the constant out because the loader gives them no other
option.

**A framework defect found in the morning produced a cross-owner hazard by the
afternoon**, not by breaking anything, but by making the correct expression
unavailable and the incorrect one silent.

## The other end, m4's

`work_root` is on m4's launch line and **never read by their stage** — they
carry a variable whose value does not reach them, while `scratch_root`, which
does, is a separate name nobody passes. Two halves of one thing: a var passed
and not read, and a var read and not passed.

## Shape

`min_requests` for paths (T-series, the `integration_min_requests` split): one
name that should be one thing, spelled independently by owners who each read a
correct value in their own file. **The difference is that `min_requests` could
be split and this one cannot be joined.**

## Not fixed, deliberately

Three owners, and the only correct fix — deriving two from one — is the
unspellable form. The options are: pass all three explicitly on every launch
line (verbose, and the launch lines are where every defect this week has been),
or resolve the nesting fault upstream. **Neither is m5's alone.**

### T59 — the instruments that failed today, and not one failed toward "I cannot tell"

**m1, 2026-09-04. An observation about the set, filed at the leader's request.
Deliberately untitled by count: it was four when written, six by evening, and a
number in a heading is the thing this file has already had go stale twice.**

## What to do about it, which is not "be careful"

**Re-ask any question whose answer you liked.** That is the only thing that has
worked all day, and it is not diligence — every catch came from someone
distrusting a *convenient* result, not from someone being more thorough:
the `df` differential, m2's second spelling, m3's argv print, checkpoint testing
the leader's axis instead of adopting it.

**Knowing this list does not protect you.** I wrote it, named the mechanism, and
then hit **three** of its members inside one four-minute check of whether a run
had stalled — a `pgrep` that matched its own shell, a `ps` pattern too narrow to
see the process, and a `git ls-files` run from the wrong directory. The
catalogue is for **diagnosing afterwards**; it does not help you **avoid**.
What stopped all three reaching the leader was widening the query a third time,
and the third query was not more careful than the first two. **It was more
sceptical of an answer I wanted.**

**The mirror: an inconvenient answer is waved through because accepting it feels
like rigour.** (leader's, worded by them, landed verbatim.)

*"Re-ask any question whose answer you liked"* catches the flattering case. The
unflattering one has no such trigger, and the reason is that **accepting a cost
feels like the check rather than a substitute for it.** When someone tells you
something that means more work for you, taking it on the chin reads as honesty —
you are visibly not defending your own position — and that feeling occupies the
place where the verification would have gone.

Measured, 2026-09-04: m1 reported that `run_with_long_stall.py` was untracked. It
was plausible, it was about a file the leader had written, and it meant work.
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
                             time spellings -- and we had all been discarding that
                             complaint with `2>/dev/null`.

df /var/lib/docker           reported 123G/70G on a filesystem a 28.5 GB `docker
                             load` does not move. I refused the load on it.
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
                             has no GPU pin" and "my template is broken" were the
                             same output. The pins were there: 0,1,2,3 and 4,5,6,7.
```

```
grep -oE '[A-Z_]*CUDA_GRAPH_MAX_BS'          returned `E_KIT_CUDA_GRAPH_MAX_BS`.
                             **`2` is not in `[A-Z_]`**, so the match began mid-token
                             and the variable NAME came back wrong by two characters
                             -- inside a verification of a claim ABOUT variable names.
                             (checkpoint's, 2026-09-04. The value and timestamp it was
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

**The fifth is the sharpest of the set** and it arrived after this entry was
filed. A character class that silently starts matching mid-token, used to check
what a name *is*. It failed in the same direction as the other four -- confident,
wrong, no error -- and it did so **inside the act of verifying**, which is the one
place we had been treating as safe.

**The property is the direction, not the count.** A tool that degrades toward
*"I cannot tell"* costs a second measurement. These four cost a **finding** —
each one had a false conclusion already drafted behind it, and three of those
conclusions would have landed on somebody else's work: a stalled run, a corrupt
backup, a container with no pin.

**What actually caught all four was a second, independent measurement**, never
care or suspicion. `-newermt` fell to `-printf '%T@' | sort -rn`; the `df` fell
to a before/after across the operation itself; `hasPrefix` fell to a plain dump.
**The `df` is the instructive one: I had already sought a second route and got
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

### T58 — counting kits on disk counts how often we tested, not how often a producer chose

**m1, 2026-09-04. Filed at the leader's request, with the fingerprint, because the
fingerprint is the only part anyone can act on.**

A census of `packup` trees is the natural way to answer *"how often has a producer
done X?"*, and it is wrong by construction. Measured over the frozen root:

```
find /shared_nfs/yihou/agent_sys -path '*packup*' -name env.sh    ->  58 files

55   line 105, DK_TP_SIZE:=1     the 2026-09-02 sealed kit, byte-identical replays
 3   no ceiling variable at all  two are my own 13-line stubs, one a stripped copy
--
 0   produced by an agent
```

**Most of the 55 are `ws_handoff_refine/m1/gate*/{good,bad}` — copies `gate.sh`
makes of the sealed kit on every invocation, two per run.** So the corpus is
mostly a record of my own testing, and **the ratio gets worse every time anyone
runs a gate.** Our diligence tilts the evidence, monotonically, in the direction
of "this happened constantly".

It already misled a real sweep: eight pre-contract runs were read as eight
producers choosing a ceiling of 8, when six were one artefact replayed. The
corrected producer record is **four runs — `16, 16, 8, 32`, one below the bar**.

## The fingerprint

**A produced `env.sh` is regenerated each run, so its line numbers move. A
replayed one is byte-identical, so they do not.**

```
produced   env.sh:169  env.sh:193  env.sh:222     DK_TP_SIZE:=4
replayed   env.sh:105  every time                 DK_TP_SIZE:=1
```

So: **before counting a kit as evidence of a choice, check the line number and
the `tp_size`.** Line 105 with `tp_size: 1` is the 2026-09-02 artefact, whatever
directory it is sitting in. Anyone can apply that; nobody would infer it.

And the second reason those 55 do not count even as failures: their ceiling of 8
is **correct for the deployment they describe**, which is `tp_size: 1` — not the
concurrency-16 shape the bar is about. `mock_adapt.sh` preserves it deliberately
(*"adapts a record forward; it does not re-tune a deployment that already
happened"*), so they are a design decision working, not a producer failing.

## Why it is worth an entry rather than a correction

**Our own discipline generates the false signal, and being more careful makes it
worse.** That is the same shape as m5's grep trap the same afternoon — there,
good comments made a naive detector fire; here, thorough testing makes a naive
census over-count. **Neither is fixable by care.** Both need the artefact's
provenance established *before* it is counted, which is what the fingerprint is
for.

Applies to any future sweep of `packup` trees, which is exactly what someone will
reach for the next time the question is *"how often has this happened?"*

# T60 — the graph ceiling is chosen by stage 1's load and spent by stage 2's

**First real cross-stage refusal, rung 2e, 2026-09-04 20:45.** `check_bench_result`
refused `profiling_mode_off.bench_result`, `strong`:

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

## Why five separate packages could never have found it

Stage 2 has never consumed stage 1's kit before this run. The seam only exists
once the stages are one graph, which is the mission's whole premise — **the first
real chained run produced a defect that no amount of work on the five demos
would have surfaced.**

## What the code says, against the leader's first instinct

The lead's initial lean was *"stage 1 should choose a higher ceiling"*. **Reading
the code says the other path is the designed one:**

* `deploy_and_prove/mock_adapt.sh:180` — `: "${E2E_KIT_CUDA_GRAPH_MAX_BS:=${DK_CUDA_GRAPH_MAX_BS:-8}}"`, i.e. `:=`, so a consumer that exports it first **wins**;
* `check_deploy_kit.validator/gate.sh:163` — planted fault **14** is *"the graph ceiling bound with no parameter"*, so the gate already treats an unparameterised ceiling as a defect. **The parameter exists so someone downstream can bind it.**
* and `E2E_KIT_CUDA_GRAPH_MAX_BS` is **not** a `--var` in `shared.yaml` or any step yaml — it is bound only inside the kit.

**Nothing in stage 2 sets it.** No reference in the profiling assets or
`assets/lib/line.sh`. So stage 2 inherits stage 1's choice silently, which is
exactly the condition CONTRACT §4.6 says no comparison can detect, because both
arms would inherit the same wrong value.

## The open decision, for m1 and m2 jointly

1. **stage 2 binds the ceiling before running the kit's `deploy.sh`** — matches
   `:=`, keeps the kit a recipe, and puts the choice with the module that knows
   its own load. Costs capture time and memory, which is why a blanket high value
   is not obviously right.
2. **stage 1 chooses a ceiling covering the heaviest downstream load** — simpler,
   but stage 1 cannot know what stage 2 will demand, so it is a guess dressed as
   a default.

**Not decided here.** Recorded because the team was unreachable for 75+ minutes
when it was found and the next rung will hit it again within the hour.

## What must not happen

**Do not raise the bar to make this pass.** The 5 % / 10 % precedent from
2026-09-02 is in `DELIVERY-NOTE-FROM-LEADER.md`: widening a bar that refused
correctly was already tried once and was the wrong response. A ceiling of 16
against concurrency 25.42 is a real 4.6x, not a threshold artefact.

## T60 addendum — the override channel already exists and is one line from working

Found 2026-09-04 21:50 by the lead, reading `assets/load/line.sh` after rung 2e died.

**`E2E_KIT_ENGINE_EXTRA_ARGS` is the designed override channel**, and the file
says so itself at `line.sh:74-76`:

> *"`EXTRA_ARGS` is appended to the worker's argv **last**, so it overrides an
> earlier occurrence of the same flag rather than racing it."*

So `--cuda-graph-max-bs 32` passed through that channel **beats the kit's own 16**,
by design, with no change to the kit and no decision about what stage 1 should
choose. It is already plumbed end to end — `line.sh:258-262` passes three
`E2E_KIT_*` overrides into the kit's `deploy.sh`, and this is one of them.

**The only obstacle is `line.sh:78`:**

```bash
export E2E_KIT_ENGINE_EXTRA_ARGS=""        # hardcoded; an inherited value is discarded
```

A value-preserving form — `"${E2E_KIT_ENGINE_EXTRA_ARGS:-}"` — changes nothing
when unset and makes the hook usable. **One line, behaviour-identical by default.**

**Two things that are NOT solved by it**, so nobody reads this as the whole fix:

* the `CAPTURE=1` branch at `line.sh:85` **overwrites** the variable with
  `--disable-cuda-graph`, so an inherited value is still lost on the
  `profiling_mode_on` line. Only `profiling_mode_off` — which is the line that
  refused — is covered.
* it makes the ceiling settable, it does not decide **what it should be**. 32
  covers the 25.42 observed once; nothing says that generalises to another trace.

**Deliberately not done by the lead.** It is m2's file and T60's decision is m1's
and m2's jointly; the lead recorded the exact fix instead of applying it, so that
landing it is a minute's work rather than a rediscovery. The team was unreachable
for 2 h 20 min when this was written, and 217 sat idle rather than being spent on
a run that would refuse identically.

### Correction to the addendum — the direct parameter is the better lever, and the real gap is `KIT_ENV_PREFIX`

The addendum above recommended unpinning `E2E_KIT_ENGINE_EXTRA_ARGS` at
`line.sh:78`. **Reading the sealed kit itself shows a cleaner lever and a
different gap.** From rung 2e's own `deploy_kit`, not from a comment:

```
scripts/env.sh:95            : "${E2E_KIT_CUDA_GRAPH_MAX_BS:=16}"
scripts/env.sh:101           : "${E2E_KIT_ENGINE_EXTRA_ARGS:=}"
scripts/start_worker.sh:91       --cuda-graph-max-bs '${E2E_KIT_CUDA_GRAPH_MAX_BS}' \
scripts/start_worker.sh:95       ${E2E_KIT_ENGINE_EXTRA_ARGS}
```

**Both are `:=`**, so an inherited value wins for either. And `EXTRA_ARGS` really
is last on the argv — line 95, four lines after the ceiling flag — so the comment
was accurate. **Two working levers, not one.**

**But the direct parameter is better**: `E2E_KIT_CUDA_GRAPH_MAX_BS=32` sets the
value the kit is built around, where the EXTRA_ARGS route sets the same flag a
second time and relies on last-wins. One is configuration; the other is an
override of an override.

**And the gap is not `line.sh:78`. It is `line.sh:223`:**

```bash
KIT_ENV_PREFIX="E2E_KIT_RUN_TAG='$E2E_KIT_RUN_TAG' \
  E2E_KIT_PORT_BASE='$E2E_KIT_PORT_BASE' \
  E2E_KIT_WORK_ROOT='$E2E_KIT_WORK_ROOT'"
```

Three variables. **The ceiling is not among them, so it never reaches the kit at
all** — which is why stage 2 inherits stage 1's choice silently. Adding a fourth
line in the identical pattern is the whole change, and **passing it empty is safe
precisely because `env.sh:95` uses `:=`** — a null value takes the default 16, so
the behaviour is identical until someone sets it.

**Still m2's file and still not applied by the lead.** The correction is recorded
because the addendum's advice would have worked by the weaker route while leaving
the actual gap in place.

### Root cause — the producer was briefed to choose 16, and did

Both entries above look for a **lever** to override the ceiling. Reading
`assets/deploy_and_prove.task/readme.md:239-249` shows there is nothing to
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
  2026-09-04, so `>= 16` is not sufficient for the flow;
* keep the *"write down why"* requirement unchanged. It is the best part of the
  brief and it is what makes a wrong number auditable: *"a number with no reason
  fails this even when the number is right."*

**Do not simply raise the criterion to `>= 32`.** That repeats the original
mistake with a bigger constant — a number that happens to cover one observed
trace, with no statement of what it must cover.

### T60 closed by rung 2f — and 25.42 was a symptom, not a requirement

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

## The part that corrects the brief I wrote

**rung 2e measured decode concurrency 25.42. rung 2f measured 4.909 — same trace
(721 records both times), same tp, same model.**

The ratio is ~5.2x, and the eager-fallback penalty measured on this cluster is
**4.6x**. So the most likely reading is that **the high concurrency was caused by
the low ceiling**, not demanded by the trace: a ceiling below the batch forces
eager decode, eager decode is ~4.6x slower, slower decode leaves more requests
in flight, and the in-flight count is the concurrency. **A positive feedback loop
in which the symptom looks like the requirement.**

**This makes the number I put in `deploy_and_prove.task/readme.md` misleading.**
I wrote *"stage 2's Mooncake trace replay reached 25.42, so `>= 16` is not
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

**m5, 2026-09-05, from a warning of m2's that was about something else.** Named
here because both the leader and m2 independently said it is a shape the day's
collection did not already contain.

`check_no_regression/check.py:567`, before `ce2d5a6`:

```python
name = args.get("schema")
if name:
    schema_lib.validate(str(name), report)
```

**With `schema` absent, the validator's first and strongest check does not run
and nothing says so.** The other five args are thresholds that degrade to safe
defaults and still bite (`0.05`, `0.10`, `0.10`). This one degrades to nothing.
That much is ordinary — `items_schema`'s shape, present and checking nothing.

## What makes it its own entry

**It is invisible exactly when the tooling is correct.**

It surfaced only because `probe_validators.py` was passing `args={}` to every
validator — a *defect* in the probe. Under that defect, `check_no_regression`
returned **True** with all six args discarded, and the passing row is what m2
flagged: *a validator stripped of its thresholds passes trivially.*

The probe has since been fixed to pass the real args. **So the condition that
reveals this can no longer occur through the tool.** Every other instrument
failure recorded today was visible *because* something was broken:

| shape | how it becomes visible |
|---|---|
| a check that cannot fail (§4.4) | by asking what it would report if the subject were broken |
| a bar neutralised by a default | by reading the default |
| a probe that reads a warning as the thing warned about (entry 20) | by reading the matched line |
| **this** | **only while a second tool is malfunctioning** |

A working probe supplies `schema`, the branch is taken, and the validator does
its job. There is no observation, from inside the package's own tooling, that
distinguishes *"this check ran"* from *"this check would silently not run if
asked differently"*.

## The general form

**A guard that is conditional on its own configuration is only as present as the
caller's discipline, and no test that supplies the configuration can see its
absence.** The absent-arg path has to be tested *deliberately* — the way a
refusal path has to be tested deliberately — because nothing produces it by
accident once the callers are correct.

## What was done

`ce2d5a6`: absent `schema` is now a refusal that names the missing arg. Verified
both directions — with the step's args it refuses on the documented 35 %/30 %
bars, with `schema` removed it refuses on the arg.

## Where else to look

Not swept. The candidate shape is `if args.get(X):` guarding a check rather than
selecting between behaviours — as distinct from `args.get(X, <safe default>)`,
which is the correct pattern and the one the other five use. **Worth a sweep by
whoever next has a quiet hour; it is not urgent and it is not one owner's.**

---

### T62 — `base_sha256` defects 2 and 3, held pending an answer that may not exist

*Opened 2026-09-05 by checkpoint, from the day's traffic. Defect 1 is landed.*

**Defect 1 shipped with its own one-third caveat attached** — the fix states what
it does *not* cover rather than leaving the reader to find out. Defects **2 and
3 are held pending m5's manifest answer.**

**The part that makes this a deferral rather than a queue item:** if m5's answer
is *"the manifest has never run against an `sgl_kernel` operator"*, then 2 and 3
are **not a choice we are postponing — they are an unknown we have not measured.**
Those are different states and the record should not let them read alike.

**m5 answered 2026-09-05: (a)** — and for a stronger reason than m3 had.
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
                     `description` string in workset.schema.json:495, not a use
@SGL_KERNEL_ROOT@     1, and it is the guard constant at
                     optimize_kernel.task/steps/60_write_handoff.py:115
```

**So both the `_NOT_OVERLAYABLE` refusal and the `rebuild` branch are unexercised
code.** The original framing survives the answer: **we chose between two
branches, neither of which has ever run.** An answered question and a measured
one are still different things.

**Holder: m5.** Not blocking today.

---

### T63 — `snr_db: inf` is non-monotonic on `layernorm`, and both proposed mechanisms are disproved

*Opened 2026-09-05 by checkpoint, from the day's traffic.*

**Two mechanisms were proposed and both were disproved. One candidate remains
untested.** The entry exists so the two dead ones are not re-proposed: an
eliminated mechanism is worth as much as a confirmed one and costs the same to
lose.

**Not blocking.** Holder not assigned as of writing — **do not infer one from
whoever touched `layernorm` last.**

---

### T64 — `rank.task` and `identify.task` are 15-line skeletons, not briefs

*Opened 2026-09-05 by checkpoint. Measured.*

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

*Opened 2026-09-05 by checkpoint. Placeholder so m2's answer has somewhere to
land — see `bug.record.2026-09-05.md` entry 15.*

`p4_a`, `p4_b`, `p4_m4real`, and **`p4_b_m1real`**.

**The fourth is the informative one: it ran with m1 real, which rules out a bad
upstream artefact as the common cause.** Three lines with a shared mock upstream
could have shared its defect; the fourth could not.

**Why it is hard to see:** `0 validation(s) dropped`, no refusal, allocation
still held, stage still `running`. **The run is healthy in every field a reader
would check.**

**Holder: m2**, who has built the right instrument for it — a two-arm control on
088, `p5_m2cap` on cards 0–3 with the fix and `p5_ctl` on cards 4–7 without.

**Do not merge with m2's `475f2fc`** (the capture step waiting on another
package's log) until m2 rules. **Their proximity in time is not evidence.**

---

### T66 — the agent's system prompt points at a path that is not in the repository

*Opened 2026-09-05 by m3's sweep; verified by checkpoint.*

`build_workset.task/readme.md:98` — **the live system prompt for a `kind: ai`
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
`bug.record.2026-09-05.md` entry 20, where good comments hid a bad grep.

**Holder: m3**, who is holding the fix until the full-real chain clears their
stage, because that readme is the live system prompt and editing it mid-run
changes what a running agent reads.

---

### T67 — `reverify_shapes` counts operators, and its name says shapes

*Opened 2026-09-05 by m3, at the leader's instruction, while confirming the cost
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
while a chain is live. The leader agreed and asked for the entry instead of the
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

*Opened 2026-09-05 by m3, from m4's question. m4 ranks it above both forge
defects and the leader agrees; recording that because it is a consumer's
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

**Holder: m3.** Not blocking; the leader has ruled the entry itself is the
deliverable today.

---

### T69 — `identify` writes no `-fellow` tag, and the generator read one

*Opened 2026-09-05 by m3 at the leader's instruction. The unblock is landed;
this is the root, filed separately so the workaround does not close it.*

**`forge_export.py:_fellow` looked for a Definition tag ending in `-fellow`.**
The tags `identify` writes are bare language names — measured on real workset
`91ea967b`: `['attention', 'linear-attention', 'triton', 'gated-delta-rule']`.
So nothing ever matched, the `generic-fellow` fallback was **the only branch
for every operator in every real workset**, and `generic` is not a backend
KernelForge registers. `cli.py`'s `--fellow` help says in as many words that
unsupported fellows fall back to `flydsl-fellow` — so a Triton attention kernel
was being optimised by the FlyDSL fellow, with a warning as the only trace.

**Mock never showed it**: the injected material carried the suffixed spelling,
so the mock path took the branch the real path cannot reach. Another instance
of the mock exercising code the real run does not.

**Landed (`62032fc`, `f92e42b`), and that is the workaround, not the fix.** The
generator now reads the bare language names, validates against
`fellows/constants.py`, and refuses instead of substituting. **The defect is
that two producers disagreed about a spelling and nothing compared them.**
`identify`'s taxonomy also offers `tilelang-fellow`
(`assets/lib/kernel_taxonomy.yaml:129`), which is not a KernelForge backend
either — that row can never be honoured and now refuses.

**What would actually close it:** one declared vocabulary for the language tag,
read by `identify` when it writes and by `forge_export` when it reads, with the
backend set validated against KernelForge's own constants at build time rather
than at campaign time.

**Holder: m3.** Not blocking — the wrapper refuses rather than substitutes, so
the dangerous outcome is gone even while the disagreement stands.
