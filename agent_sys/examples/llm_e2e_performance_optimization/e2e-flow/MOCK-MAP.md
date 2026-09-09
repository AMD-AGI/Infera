# Mock map — which sealed handoff stands in for which kind

Mission Brief item 2: *"通过 mock ai agent，结合上轮单独运行的真实 handoff，构造
符合 validator 的。快速跑通 e2e 流程"*. The point of the exercise is that **the
validators are unchanged**. A mock that writes something a validator accepts but
a real producer would never emit proves nothing, so nothing here synthesises —
`assets/lib/mock.sh` copies bytes a real run on this cluster produced.

Source: `/shared_nfs/yihou/agent_sys/cheat_for_mock/`. **Read its `README.md`
first**; it documents four things that mislead.

## Mocking an AI task means swapping its agent, not its body

**A `kind: ai` task does not run `entry.sh`.** The backend runs the readme
through a model, so the moment m1, m3, m4 and m5 promoted their leaves from the
skeleton's `agent: runner` to a real AI agent, those four went off the mock path
entirely — `assets/lib/mock.sh` was still there and was never reached.

Measured, and it is why this section exists: the first full mock run sat at
`deploy_and_prove: running` while an AI deployment agent prepared to bring a
model up **for real**, on the node passed in `--var`. Stopped by hand; the node
was checked and nothing had been created.

So each of the four declares `agent: '${m<N>_agent:-<the real agent>}'`, and a
mock run swaps them:

```sh
--var mock_stages=all \
--var m1_agent=runner --var m3_agent=runner \
--var m4_agent=runner --var m5_agent=runner
```

`runner` is the shared program agent, so it runs `entry.sh`, which mocks. **The
default is the real agent** — a normal run is unchanged and the mock is the
thing you have to ask for, which is the right way round for a switch that
decides whether a model gets called.

Two more vars a mock run wants, both facts about the *artefact being graded*
rather than about this run: `--var expect_ranks=2` (the sealed capture is TP-2
while the flow's default `tp` is 8) and `--var adhoc_cases=0` (no sealed handoff
carries an `adhoc.json` — `../todo.md` T12).

## The map

| kind | mock source | adaptation needed |
|---|---|---|
| `deploy_kit` | `stage1-deploy/deploy_kit` | **environment.yaml** (A) · **runtime contract shim** (I) |
| `profiling_mode_off.bench_result` | `stage2-profiling/aiperf_baseline` | (A) |
| `profiling_mode_on.bench_result` | `stage2-profiling/aiperf_profiled` | (A) |
| `profiling_mode_on.profile_result` | `stage2-profiling/torch_trace` | (A) · **360 MB** |
| `profiling_mode_on.kernel_table` | `stage2-profiling/kernel_table` | (A) · the **real** 124-kernel one · **content type differs** (B) |
| `profiling_evidence` | — none, and none wanted — | **(H)** |
| `kernel_worklist` | `stage3-analyze/kernel_worklist` | (A) |
| `operator_identity` | `stage3-analyze/operator_identity` | (A) |
| `operator_workset` | `stage3-analyze/operator_workset` | (A) · **merge with stage4/workset** (C) |
| `kernel_optimization` | `stage4-kernel-opt/kernel_optimization` | (A) · **the sealed one has no `apply` and no `premise`** (G) |
| `patch_overlay` | `stage5-integration/patch_overlay` | (A) |
| `stock.measurement` | `stage5-integration/{deployment,acceptance,bench}_stock` | (A) · **three into one** (D) |
| `patched.measurement` | `stage5-integration/{deployment,acceptance,bench}_patched` | (A) · (D) |
| `integration_report` | `stage5-integration/integration_report` | (A) · **carries a refused verdict** (E) |
| `e2e_packup` | — none sealed — | **(F)** |

## The six adaptations, and why each is real work rather than a copy

### (A) `environment.yaml` does not exist in any sealed handoff

The fields do — as `items/env/deployment.json` and `context.json` — but the
document `assets/schemas/environment.schema.json` describes does not, and it
**deliberately requires three fields those records lack**: `gpu_arch`,
`gpu_count` and `image_id`. Verified: feeding the raw `context.json` to the
schema returns eight problems.

That is the schema doing its job, not an oversight. A mock therefore **renders**
an `environment.yaml` from the sealed record plus the run's own `--var`s. One
renderer, `assets/lib/env_render.py`, used by every mock and by every real
producer — **owner: leader**, since it is the one piece all fifteen kinds share.

### (B) the sealed `kernel_table` is `reproducible`; the kind is `structured_text`

**Corrected 2026-09-03 — the original row here was wrong**, and it named the
wrong handoff. Found by m2, verified: the sealed
`stage2-profiling/kernel_table` carries `items/{command,env,logs,result,
watchout}`, which is `reproducible`. CONTRACT §1 declares the kind
`structured_text`, whose items are `text.json`/`text.yaml`/`text.xml`/`schema`
plus whatever the kind declares.

So `check_items` would refuse it outright — `unknown = items - known -
declared` — and its `README.md` lacks the `Schema` section the content type
requires. A (B)-style reshape, not merely (A): `assets/lib/m2_reshape.py` does
it.

The reshape is the right direction, not a workaround. The table *is* structured
text; it was sealed as `reproducible` because the stage-2 package of the day had
one shape for everything it produced, which is the kind of thing this refine
exists to fix.

### (H) `profiling_evidence` has no mock, and should not have one

m2's call, and a better one than mine. The merge's four inputs are already
mocked upstream, so `merge_profiling_evidence` can perform **the real merge** in
every mode — mock, staged and real alike.

Two things that buys, and both matter more than the convenience:

- one fewer synthesised artefact in a directory whose whole claim is that
  nothing in it was synthesised;
- **`check_profiling_evidence`'s cross-part rules are genuinely exercised in the
  mock run.** Graded against a hand-shaped stand-in they would be checking that
  the stand-in was shaped correctly, which is a test of the mock and not of the
  validator. `require_same_environment` in particular can only mean anything
  over parts that arrived separately.

The former row pointed at `stage2-profiling/profile_packup`, a `code`-typed
packup being reshaped into a `reproducible` kind. Deleted.

### (C) `operator_workset` is the merged kind; neither source is it yet

`stage3-analyze/operator_workset` (27 files, two operators) and
`stage4-kernel-opt/workset` (10 files, one operator, `sampler_vocab_softmax`)
are the two halves M3.7 merges. Neither alone satisfies
`check_workset_shape`'s `min_shapes: 3` **and** `require_entrypoints:
[correctness, performance]`.

**This is m3's first deliverable and m4 is blocked on it** — the two owners
agree the merged shape before either writes a body.

### (D′) `integration_report` must carry m2's number, because its validator cannot fetch one

`check_no_regression`'s only input is `integration_report`, so M5.1.3.1 — the
stock arm must reproduce m2's `profiling_mode_off` bench — is only checkable if
the **producer carries that number into the report**. It is a required block in
`integration_report.schema.json`, not an optional one: a producer that skips it
must be visibly incomplete rather than silently absent.

Same for M5.1.3.2's kernel-share reconciliation. `integrate_and_verify` already
has `profiling_evidence` and `kernel_optimization` as inputs, so no new edge is
needed — only the discipline of writing both numbers down.

### (D) Six evidence kinds became two — **done**, `assets/lib/mock_m5.sh arms`

`stock.measurement` folds `deployment_stock` + `acceptance_stock` +
`bench_stock`; likewise patched. `assets/lib/merge_arm.py` does it, and it is the
same module a real run uses at step 9 of `integrate_and_verify`'s STEPS — the
mock does not get its own merge.

**The union is flatter than expected and that is measured.** `result/` gains no
subdirectories: `smoke.json`, `needle.json`, `probe.json`, `lm_eval/` and `r1/`
sit side by side and nothing collides, so `check_acceptance` and
`check_bench_report` read exactly the paths they always did. 39 files per arm,
zero collisions.

Exactly three files exist in more than one source — `README.md`, `items/command`
and `items/watchout`, because each script wrote a whole handoff of its own. They
are **concatenated with a header naming each half**, not overwritten, and a
genuine collision (same path, different bytes) is a refusal rather than a
last-writer-wins.

**`env/steps.json` needed one thing written.** The measurement script's record
already carries the five measured steps with timestamps — so the merged order is
not invented — but no sealed artefact records the **bring-up window**, and
without it the disjointness check cannot express the guarantee it exists for
("the patched arm's bring-up did not start until the stock arm had finished").
`merge_arm.py` therefore requires `--serve-started` and `--serve-seconds` with no
default: a guessed bring-up window would make the check pass by construction. The
mock derives windows from the sealed step timestamps — stock measured
12:58:12–13:41:32 and patched 13:45:47–14:35:51, so a stock bring-up ending just
before 12:58 and a patched one beginning after 13:41:32 is the order the real run
had — and **writes a `note` into the step saying it is derived**.

### (D″) `adhoc.json` has no sealed source, and the mock does not invent one

M5.4's per-run correctness cases post-date every sealed handoff. Synthesising a
set would be exactly what this document forbids, so **a mock run passes
`--var adhoc_cases=0`** and `check_acceptance`'s ad-hoc rules are not exercised
until the first real run. Recorded here rather than left as a surprise: it is the
one m5 rule the mock cannot test.

### (E) `integration_report`'s sealed verdict is `false`

```
check_no_regression   result: false   strength: strong
```

It is a **sound sample of a refused report and a misleading one if taken for a
passing example.** Because `check_no_regression` is `strong`, mocking it
verbatim will correctly stop the graph — which is *also* a useful test, once,
of the refusal path.

Two mock modes, and the run should exercise both. **Done**, as
`assets/lib/mock_m5.sh report` with `$E2E_MOCK_REPORT`:
- `refused` (default) — verbatim. Expect the graph to stop at m5. This is the
  only cheap end-to-end test of a strong refusal we have. Verified:
  `check_no_regression` FAILs, its recomputation agrees with the report's stated
  verdict and with its stated reasons, and the sole complaint is the real one.
- `accepted` — the same report with the two arms' numbers taken from the **stock
  control** measured under matched load (patched 475.7 ms vs stock 470.3 ms mean
  ITL, 1.1% apart), which is what a non-regressing run on this cluster actually
  looks like. Verified: PASS.

**Two blocks have to be added in either mode, and they are the interesting
part.** `integration_report.schema.json` requires `stock_vs_m2` (M5.1.3.1) and
`kernel_reconciliation` (M5.1.3.2), and the 2026-09-02 run predates both — fed
the sealed `text.json`, the schema returns exactly two problems and nothing else,
which is the schema being written against the real artefact rather than around
it. The mock fills both in as *not measured, and here is why*, which is the case
the schema forces a producer to state rather than omit.

**Also mocked verbatim: the declared bars.** The sealed report says 0.35/0.30,
and `check_no_regression` refuses a report whose bars are looser than its own
args — so the `refused` mock produces that complaint too, on top of the
regression. That is correct and it is worth seeing once: a producer may not pick
its own threshold.

### (G) the sealed `kernel_optimization` has no `apply` and no `premise`

Found by m4 on 2026-09-03, and it is structural rather than a detail to patch.

That run was `KFO_MOCK=1`: no campaign, no optimised kernel, `mean_case_speedup:
1.0` by construction. `operator` and the evidence half are there; `workset_ref`
survives only as a UUID in prose in `README.md`; **`apply` and `premise` do not
exist in any form**, because there was nothing to apply. And a reconstructed
premise would read gfx942 — the workset's architecture — against a gfx950 host,
which is **exactly the mismatch M4.3.5 says must abort**.

So the mock **renders** the three missing fields, the way (A) renders
`environment.yaml` and (D) writes `env/steps.json`: `workset_ref` and `premise`
come from the merged workset produced on *this* node, so the premise matches and
m4 proceeds. `apply` is written against the workset's declared integration point
(M5.1.1), which is what lets m5's `apply_patch` stay a program.

**Two modes, and the second is worth running once**, parallel to (E):

- `--var mock_premise=matched` (default) — rendered as above, m4 proceeds.
- `--var mock_premise=mismatched` — premise left at the sealed gfx942. Expect
  m4 to **abort**. It is the only cheap end-to-end test of the abort path, and
  it cannot be the default or nothing downstream of m4 ever runs.

**What this means for the schema's both-direction proof:** "the real document
validates" cannot mean the sealed bytes validate unchanged. It means the
*rendered* document validates and the sealed one is rejected with `apply` and
`premise` named — which is the more useful direction anyway, since it proves the
required-list is doing work.

### (I) the sealed `deploy_kit` needs a five-line `env.sh` shim and a `deployment.json`

`deploy_kit.layout.yaml` adds a **runtime contract**: `scripts/deploy.sh` and
`scripts/teardown.sh` must honour `E2E_KIT_RUN_TAG`, `E2E_KIT_PORT_BASE`,
`E2E_KIT_WORK_ROOT`, `E2E_KIT_ENGINE_EXTRA_ARGS` and `E2E_KIT_ENGINE_EXTRA_ENV`,
and write `$E2E_KIT_WORK_ROOT/deployment.json` carrying `endpoint`, `container`
and `run_tag`.

**This is not decoration, it is what makes M2.3 true.** Without a
machine-readable entrypoint m2 cannot deploy from this handoff, and `serve_baseline`
could not have been deleted. The two engine parameters are m2's seam
specifically: `EXTRA_ARGS` is appended **last** so an override beats the kit's own
flag, and `EXTRA_ENV` reaches the **worker process only** rather than the
container, so a variable meant for the engine does not also reach the router.

**Corrected: the motivating example for the second seam was wrong.**
`SGLANG_TORCH_PROFILER_DIR` was cited here and in `deploy_kit.layout.yaml`, and
**nothing in this package sets it** — the engine is told where to write per
capture, in the `/start_profile` body's `output_dir`. m2 gave it when asking for
the seams and retracted it after checking. So **`EXTRA_ENV` has no consumer
today**: m2's two lines both leave it empty, and the profiler-attached line uses
`E2E_KIT_ROUTER_EXTRA_ARGS` instead, because `--enable-profiling` is a *router*
flag that no engine seam can reach.

The argv/environment distinction is still real — they are not interchangeable,
and a parameter added later would need one or the other. But a seam with no
consumer, justified by an example that does not exist, is a thing a reader will
build on. Kept and labelled rather than removed, so the next person asking for
it finds the history.

All five concepts already exist in the sealed kit under `DK_RUN_TAG`,
`DK_PORT_BAND_LO` and `DK_WORK_ROOT`, so the adaptation is five
`: "${E2E_KIT_X:=${DK_Y:-…}}"` lines appended to `scripts/env.sh` plus the
`deployment.json`. m1 verified that exact shim makes the untouched sealed kit
pass, and it is carried in
`assets/check_deploy_kit.validator/gate.sh` — which builds the positive fixture
from `$E2E_MOCK_ROOT`, builds a negative one with eleven planted faults, and
asserts each is reported by name. **Run it; it is the worked example of "gated
both directions" for this package.**

### (J) 11 of the 14 sealed `command` scripts do not parse — repaired by `mock.sh`

Found by m2, confirmed here first-hand. One cause in every case: an apostrophe
inside a `${VAR:?word}` message opens a single-quoted string that runs to end of
file.

```
: "${SCRIPTS:?export SCRIPTS=<the package's assets/load directory>}"
                                          ^ opens a quote that never closes
```

**Judged by the shell the file's own shebang names**, which is the part that
decides it. On `stage2-profiling/aiperf_baseline`: shebang `#!/usr/bin/env
bash`; `sh -n` clean **and the script runs**, aborting correctly on the first
unset variable; `bash -n` fails at line 9. dash tolerates the unterminated
quote, bash does not — so the script is broken under the shell it names and
works under the shell it does not, and a reproducer typing `./command` gets the
shebang and therefore the error.

`assets/lib/mock.sh` rewrites `the package's assets` to `the package assets` in
any copied `items/command` or `items/script` that carries the pattern, and
**says so per file** in the run log. Nothing else in the script is touched.

**Why repairing is legitimate here and not a violation of "nothing is
synthesised":** the sealed bytes predate `check_command_parses`, exactly as they
predate `environment.yaml` — (A) renders a record the sealed set never carried,
and this repairs a defect nothing ever checked for. Both are "the sealed set
predates the contract". The alternative is a rule that can never go green in a
mock run, which would mean not having the rule.

**The unrepaired bytes remain the negative fixture.** `check_command_parses` was
proven in both directions against them: the sealed `aiperf_baseline` FAILs
naming the shell, the line and the reason; `stage5-integration/bench_stock`,
one of the three that were always fine, PASSes.

### (F) `e2e_packup` has no sealed source

`integration_packup` was never sealed — the graph stopped at (E) before it was
dispatched. An unsealed 47-file packup exists at
`/shared_nfs/yihou/agent_sys/debugging/integration/packup-out-of-band/`,
produced by `integration`'s own unmodified `packup.py` over the nine sealed
handoffs and graded PASS against the real `check_packup_shape` body.

Use it as the mock source, and note in the run's record that it is **not
sealed** and its provenance is `PRODUCED-BY-DEPLOY.md` in that directory.

**Done**, as `assets/lib/mock_m5.sh packup`. Its `content/` is copied wholesale
rather than its parent: the kit is already handoff-shaped —
`content/items/codes/{README.md,REPRODUCE.md,results,logs,scripts,handoffs}` —
because a real `packup.py` wrote it into a real output slot. Only
`PRODUCED-BY-DEPLOY.md` sits outside, and that is the provenance record rather
than part of the kit. The mock writes the not-sealed fact into the handoff's own
`items/watchout`, so a reader meets it in the artefact and not only here.
Verified PASS against `check_packup_shape` with the step yaml's verbatim `args`.

## Digests

Every digest under `cheat_for_mock/` is invalid — a past `chmod -R 777` changed
every file's executable bit, and the tree digest is git-shaped so it records
exactly that. **Content is intact**;
`/shared_nfs/yihou/agent_sys/temp/leader/repair_modes.py` restores the modes by
searching candidate executable-sets for the one that reproduces the manifest's
own digest, which is proof rather than a guess.

Nothing on the consuming path verifies a digest anyway
(`env_mgr.fs.layout.copy_out` is a plain `copytree` — `todo.md` T9), so this
does not block the mock run. **It does mean the mock run cannot be used to test
digest verification.**
